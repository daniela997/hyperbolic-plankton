"""Unseen-species evaluation (Piece 6) — Planktonzilla-faithful protocol.

Re-implements the paper's CLIP-style zero-shot eval (notebooks/metrics_paper.ipynb:
`clip_preds_and_features` + `evaluate_taxonomic_metrics`), swapping cosine similarity for
Lorentzian distance so the hyperbolic model is compared apples-to-apples.

Paper protocol (verified against the planktonzilla repo):
  - Class label = the space-joined cumulative lineage `" ".join([Kingdom..Species])` over
    non-empty ranks (gen_datasets.py::build_tax_string). This is exactly our
    `data.build_taxonomy(row)["full"]`. Eval is restricted to plankton rows with Kingdom
    present (full != "unknown").
  - Unseen classes = `full` strings that appear in the held-out datasets but NOT in the
    seen training pool (Table 3: classes absent from training).
  - Prediction: encode each class's `full` string as a text prototype; for each image pick
    the nearest prototype (paper: argmax cosine; ours: argmin Lorentzian distance), with
    the label space masked to the unseen class set (`allowed_indices`).
  - Per-rank macro-F1: truncate predicted + true `full` strings to the first `k` tokens
    (k = rank depth) and run sklearn `f1_score(average="macro")` — paper's
    `evaluate_taxonomic_metrics`. Plus an overall full-string macro-F1 ("Original F1").

Secondary (our method's selling point, not in the paper): `predict_per_rank` lets a sample
fall back to the rank it is most confident about (entailment-style), instead of always
committing to the deepest class then truncating.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from . import lorentz as L
from .data import RANKS, build_taxonomy

PROMPT = "a photo of a {label}"


# --------------------------------------------------------------------------------
# class set: unseen `full` strings present in held-out but absent from the seen pool
# --------------------------------------------------------------------------------

def _full_strings(hf_dataset) -> list[str]:
    """The paper class string per row: `build_taxonomy(row)["full"]` (Kingdom..Species)."""
    cols = {c: hf_dataset[c] for c in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]}
    n = len(hf_dataset)
    out = []
    for i in range(n):
        row = {c: cols[c][i] for c in cols}
        out.append(build_taxonomy(row)["full"])
    return out


def build_unseen_classes(unseen_full: list[str], seen_full: set[str]) -> list[str]:
    """Sorted unseen class strings: present in held-out, absent from the seen pool.

    `unseen_full` / `seen_full` are the per-row `full` strings (use `_full_strings`).
    Drops "unknown" (no Kingdom) — the paper requires Kingdom present.
    """
    unseen = {f for f in unseen_full if f != "unknown"} - set(seen_full)
    return sorted(unseen)


# --------------------------------------------------------------------------------
# prototypes + prediction (hyperbolic analogue of the paper's cosine argmax)
# --------------------------------------------------------------------------------

@torch.no_grad()
def encode_prototypes(model, class_strings: list[str], prompt: str = PROMPT, batch_size: int = 256) -> torch.Tensor:
    """Encode each class's name into a hyperbolic text prototype `[C, D]`."""
    model.eval()
    prompts = [prompt.format(label=c) for c in class_strings]
    chunks = []
    for i in range(0, len(prompts), batch_size):
        chunks.append(model.encode_text(prompts[i : i + batch_size]))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict(img_embs: torch.Tensor, proto_embs: torch.Tensor, curv) -> torch.Tensor:
    """Nearest-prototype index per image: argmin Lorentzian distance `[B] -> [0..C-1]`.

    Hyperbolic analogue of the paper's `argmax(100 * img @ text.T)` over normalized
    cosine; smaller hyperbolic distance = better, so we argmin.
    """
    dist = L.pairwise_dist(img_embs, proto_embs, curv)  # [B, C]
    return dist.argmin(dim=1)


# --------------------------------------------------------------------------------
# metrics — paper-faithful per-rank macro-F1 via token truncation
# --------------------------------------------------------------------------------

def _truncate(full: str, k: int) -> str:
    """First `k` lineage tokens of a `full` string (paper: `" ".join(tokens[:k])`)."""
    return " ".join(full.split()[:k])


def taxonomic_macro_f1(
    true_full: list[str], pred_full: list[str], ranks: list[str] = RANKS
) -> dict:
    """Per-rank macro precision/recall/F1 + overall full-string F1.

    Re-implements `evaluate_taxonomic_metrics`: for rank depth k (1..len(ranks)) truncate
    both true and predicted `full` strings to k tokens, then sklearn macro metrics. A
    sample with fewer than k tokens contributes its whole (shorter) string — the paper's
    slice semantics, so a sample is effectively scored only down to the depth it has.
    """
    results: dict = {}
    for k, rank in enumerate(ranks, start=1):
        yt = [_truncate(f, k) for f in true_full]
        yp = [_truncate(f, k) for f in pred_full]
        results[rank] = {
            "precision": precision_score(yt, yp, average="macro", zero_division=0),
            "recall": recall_score(yt, yp, average="macro", zero_division=0),
            "f1": f1_score(yt, yp, average="macro", zero_division=0),
        }
    results["full"] = {"f1": f1_score(true_full, pred_full, average="macro", zero_division=0)}
    return results


# --------------------------------------------------------------------------------
# secondary: per-rank (entailment-fallback) prediction — our method's advantage
# --------------------------------------------------------------------------------

@torch.no_grad()
def run_unseen_eval_cosine(
    model,
    unseen_ds,
    classes: list[str],
    prompt: str = PROMPT,
    batch_size: int = 128,
    num_workers: int = 8,
    limit: int | None = None,
) -> dict:
    """Paper-faithful EUCLIDEAN baseline (notebooks/metrics_paper.ipynb).

    Bypasses the projector/hyperbolic lift entirely: uses the raw open_clip backbone
    (`model.clip.encode_image/encode_text`), L2-normalizes, and predicts
    `argmax(img @ text.T)` — exactly the paper's off-the-shelf CLIP zero-shot. Running
    this on our split/class-set/metric and matching the paper's reported BioCLIP numbers
    validates the data pipeline independently of any hyperbolic code.
    """
    from torch.utils.data import DataLoader

    from .train import TaxonomyCollator

    backbone = model.clip
    backbone.eval()
    device = model.device

    prompts = [prompt.format(label=c) for c in classes]
    tok = model.tokenizer(prompts).to(device)
    text = backbone.encode_text(tok, normalize=True)  # [C, D], L2-normalized

    collate = TaxonomyCollator(model.preprocess)
    loader = DataLoader(
        unseen_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    true_full: list[str] = []
    pred_idx: list[int] = []
    for pixel_values, taxonomy_batch, _ in loader:
        img = backbone.encode_image(pixel_values.to(device), normalize=True)  # [B, D]
        logits = img @ text.T  # cosine (both normalized)
        pred_idx.extend(logits.argmax(dim=1).tolist())
        true_full.extend(taxonomy_batch["full"])
        if limit is not None and len(true_full) >= limit:
            break
    if limit is not None:
        true_full, pred_idx = true_full[:limit], pred_idx[:limit]

    pred_full = [classes[i] for i in pred_idx]
    return {
        "metrics": taxonomic_macro_f1(true_full, pred_full),
        "n": len(true_full),
        "n_classes": len(classes),
    }


@torch.no_grad()
def run_unseen_eval(
    model,
    unseen_ds,
    classes: list[str],
    prompt: str = PROMPT,
    batch_size: int = 128,
    num_workers: int = 8,
    limit: int | None = None,
    ranks: list[str] = RANKS,
) -> dict:
    """End-to-end Table-3-style unseen eval: encode prototypes, predict, score.

    `unseen_ds` is an `HFTaxonomyDataset` over held-out rows whose `full` string is in
    `classes` (filter first with `build_unseen_classes` + a `full in classes` select).
    Returns `{metrics: taxonomic_macro_f1(...), n: int, n_classes: int}`. `limit` caps the
    number of images (for quick smoke runs).
    """
    from torch.utils.data import DataLoader

    from .train import TaxonomyCollator

    model.eval()
    protos = encode_prototypes(model, classes, prompt=prompt)
    curv = model.curvature

    collate = TaxonomyCollator(model.preprocess, ranks=ranks)
    loader = DataLoader(
        unseen_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )

    true_full: list[str] = []
    pred_idx: list[int] = []
    for pixel_values, taxonomy_batch, _ in loader:
        img = model.encode_image(pixel_values.to(model.device))
        pred_idx.extend(predict(img, protos, curv).tolist())
        true_full.extend(taxonomy_batch["full"])
        if limit is not None and len(true_full) >= limit:
            break
    if limit is not None:
        true_full, pred_idx = true_full[:limit], pred_idx[:limit]

    pred_full = [classes[i] for i in pred_idx]
    return {
        "metrics": taxonomic_macro_f1(true_full, pred_full, ranks),
        "n": len(true_full),
        "n_classes": len(classes),
    }


def class_set_from_dataset(ds) -> list[str]:
    """Sorted unique `full` strings present in a dataset (drops 'unknown') — the class
    label space for a seen-val eval (model predicts among classes seen in this split)."""
    return sorted({f for f in _full_strings(ds) if f != "unknown"})


@torch.no_grad()
def geometry_stats(model, taxonomy_batch, ranks=RANKS) -> dict:
    """Per-rank geometric diagnostics for understanding training dynamics.

    Encodes the batch's per-rank taxonomy text into the hyperboloid and reports, for each
    rank with valid entries:
      - `radius`: mean geodesic distance-from-origin ‖x‖_hyp. Thesis prediction is a
        MONOTONIC ordering kingdom(small) → species(large); inversion/flattening = trouble.
      - `aperture`: mean cone half-aperture. Coarser ranks should have WIDER cones. This is
        also the quantity SEL shrinks via curvature, so a uniform widening across ranks is
        the signature of curvature-collapse "cheating".
      - `entail_ok`: for each consecutive (parent, child) rank edge present in the batch,
        the fraction of valid pairs where the child lies INSIDE the parent's cone
        (oxy_angle ≤ half_aperture) — the direct measure of entailment being learned,
        independent of the loss value.
    Returns a flat dict keyed `geom/{rank}/radius`, `geom/{rank}/aperture`,
    `geom/{parent}->{child}/entail_ok`, plus `geom/curv`.
    """
    was_training = model.training
    model.eval()
    embs = model.encode_taxonomy(taxonomy_batch)  # {rank: [B,D], rank_valid: [B]}
    curv = model.curvature
    out: dict = {"geom/curv": float(curv)}

    for r in ranks:
        if r not in embs:
            continue
        valid = embs[f"{r}_valid"]
        if not bool(valid.any()):
            continue
        x = embs[r][valid]
        out[f"geom/{r}/radius"] = float(L.distance_from_origin(x, curv).mean())
        out[f"geom/{r}/aperture"] = float(L.half_aperture(x, curv).mean())

    present = [r for r in ranks if r in embs]
    for parent, child in zip(present[:-1], present[1:]):
        pv, cv = embs[f"{parent}_valid"], embs[f"{child}_valid"]
        both = pv & cv
        if not bool(both.any()):
            continue
        p, c = embs[parent][both], embs[child][both]
        angle = L.oxy_angle(p, c, curv)
        aperture = L.half_aperture(p, curv)
        out[f"geom/{parent}->{child}/entail_ok"] = float((angle <= aperture).float().mean())

    if was_training:
        model.train()
    return out


def flatten_metrics(metrics: dict, prefix: str) -> dict:
    """taxonomic_macro_f1 output -> flat `{prefix}/{rank}_f1` dict for wandb logging."""
    out = {}
    for rank, m in metrics.items():
        out[f"{prefix}/{rank}_f1"] = m["f1"]
    return out


@torch.no_grad()
def predict_per_rank(
    img_embs: torch.Tensor,
    proto_embs: torch.Tensor,
    class_strings: list[str],
    curv,
    ranks: list[str] = RANKS,
) -> dict[str, list[str]]:
    """Independent nearest-prototype prediction at each rank (not paper-faithful).

    Instead of committing to the deepest class then truncating, we collapse the class
    prototypes to each rank (mean of prototypes sharing the same k-token prefix) and pick
    the nearest at that rank. This lets a sample land on a higher rank it is confident
    about — the entailment-fallback behaviour the hyperbolic geometry is meant to give.
    Returns `{rank: [B] predicted truncated strings}`.
    """
    out: dict[str, list[str]] = {}
    for k, rank in enumerate(ranks, start=1):
        # group class prototypes by their k-token prefix
        prefixes = [_truncate(c, k) for c in class_strings]
        uniq = sorted(set(prefixes))
        idx_of = {p: i for i, p in enumerate(uniq)}
        groups = [[] for _ in uniq]
        for ci, p in enumerate(prefixes):
            groups[idx_of[p]].append(ci)
        rank_protos = torch.stack(
            [proto_embs[g].mean(dim=0) for g in groups], dim=0
        )  # [U_k, D] (Euclidean mean of space components; a rank "centroid")
        dist = L.pairwise_dist(img_embs, rank_protos, curv)
        pred_idx = dist.argmin(dim=1).tolist()
        out[rank] = [uniq[i] for i in pred_idx]
    return out
