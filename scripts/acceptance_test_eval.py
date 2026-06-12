"""ACCEPTANCE TEST: does the Planktonzilla CLIP reproduce the paper's Table 2/3 numbers
through OUR split + OUR eval?

If this passes, our seen/unseen split and our per-rank macro-F1 eval are validated as
faithful to the paper, and every LoRA-vs-paper comparison downstream is trustworthy.

Setup (matches the planktonzilla repo's metrics_paper.ipynb exactly):
  - their weights (project-oceania CLIP), off-the-shelf zero-shot
  - FULL seen (test_idx) + FULL unseen (unseen_idx, the 220-class benchmark) splits
  - rows with NO taxonomy (Kingdom empty -> full=='unknown') EXCLUDED, matching their
    dataset (their features['label'].names has no 'unknown'; "evaluated only at ranks for
    which a valid annotation is available"). Ragged rows (valid to some depth) are KEPT.
  - prompt "a photo of a {label}", predict argmax cosine over present classes
  - per-rank truncate-deepest macro-F1 (== their evaluate_taxonomic_metrics, our
    taxonomic_macro_f1 is byte-equivalent)

  PYTHONPATH=src python scripts/acceptance_test_eval.py [--limit N]
"""

from __future__ import annotations

import argparse

import numpy as np
import open_clip
import torch
from datasets import load_from_disk

from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.eval import flatten_metrics, run_unseen_eval_cosine

import json
from eval_planktonzilla_clip import _OpenClipShim, HF_ID

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
SPLIT = "/scratch/daniela/hyperbolic_plankton_splits"

# Paper Table 2 (Standard Classification, ViT-B/16 OpenAI+PZ) — SEEN
PAPER_SEEN = dict(kingdom=0.961, phylum=0.905, class_=0.865, order=0.853,
                  family=0.822, genus=0.797, species=0.786)
# Paper Table 3 (unseen, ViT-B/16 OpenAI+PZ) — UNSEEN
PAPER_UNSEEN = dict(kingdom=0.408, phylum=0.204, class_=0.118, order=0.088,
                    family=0.072, genus=0.065, species=0.055)


def _present_no_unknown(ds):
    return sorted({ds[i]["taxonomy"]["full"] for i in range(len(ds))} - {"unknown"})


def _full_split(idx_name, cache):
    """Select idx rows, DROP rows with empty Kingdom (full=='unknown'), wrap.

    SHUFFLE the kept rows (seeded) so a --limit run is a RANDOM representative subset, not
    a contiguous slice (the cache is ordered by source dataset → first-N is one clade and
    badly non-representative). For the full run (no limit) shuffle is a harmless reorder.
    """
    idx = np.load(f"{SPLIT}/{idx_name}.npy").tolist()
    sub = cache.select(idx)
    # Kingdom empty/None -> the row's full would be 'unknown' -> exclude (their dataset has none)
    keep = [i for i, k in enumerate(sub["Kingdom"]) if k not in (None, "", "nan")]
    rng = np.random.default_rng(0)
    rng.shuffle(keep)
    sub = sub.select(keep)
    return HFTaxonomyDataset(sub), len(idx), len(keep)


def _report(shim, name, ds, classes, paper, num_workers, limit):
    res = run_unseen_eval_cosine(shim, ds, classes, num_workers=num_workers, limit=limit)
    m = flatten_metrics(res["metrics"], prefix=name)
    print(f"\n=== {name}: {res['n']:,} images, {len(classes)} classes ===")
    print(f"  {'rank':8s} {'ours':>7s} {'paper':>7s} {'Δ':>7s}")
    worst = 0.0
    for r in RANKS:
        ours = m.get(f"{name}/{r}_f1")
        pap = paper[r if r != "class" else "class_"]
        if ours is None:
            continue
        d = ours - pap
        worst = max(worst, abs(d))
        print(f"  {r:8s} {ours:7.3f} {pap:7.3f} {d:+7.3f}")
    print(f"  -> max |Δ| = {worst:.3f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per split (debug; full split is the real test)")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {HF_ID} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(HF_ID)
    tokenizer = open_clip.get_tokenizer(HF_ID)
    model.to(device).eval()
    shim = _OpenClipShim(model, preprocess, tokenizer, device)

    cache = load_from_disk(CACHE)
    seen_ds, n_seen0, n_seen = _full_split("test_idx", cache)
    unseen_ds, n_uns0, n_uns = _full_split("unseen_idx", cache)
    print(f"seen:   {n_seen:,}/{n_seen0:,} kept (dropped {n_seen0-n_seen:,} unknown)")
    print(f"unseen: {n_uns:,}/{n_uns0:,} kept (dropped {n_uns0-n_uns:,} unknown)")

    seen_classes = _present_no_unknown(seen_ds)
    with open(f"{SPLIT}/unseen_classes.json") as f:
        unseen_classes = json.load(f)

    w1 = _report(shim, "seen", seen_ds, seen_classes, PAPER_SEEN, args.num_workers, args.limit)
    w2 = _report(shim, "unseen", unseen_ds, unseen_classes, PAPER_UNSEEN, args.num_workers, args.limit)
    print(f"\n{'='*50}\nACCEPTANCE: max|Δ| seen={w1:.3f} unseen={w2:.3f}  "
          f"-> {'PASS' if max(w1,w2) < 0.03 else 'INVESTIGATE'} (tol 0.03)")


if __name__ == "__main__":
    main()
