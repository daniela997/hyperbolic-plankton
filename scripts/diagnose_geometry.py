"""Full geometry diagnostics for a trained hyperbolic checkpoint — one pass, all probes.

Loads a checkpoint, AUTO-rebuilds the architecture from its saved args (rank/blocks/sel_text),
encodes the FULL test_seen split (images + per-rank text on the config's trained SEL text form),
and reports every probe we ran ad-hoc this session:

  geometry (text):  per-rank radius, aperture; entail_ok per edge; nesting slack per edge
  image:            transitivity (image in each ancestor cone); radius mean/std; angular spread
                    (mean pairwise cos of image directions); image<->species distance
  classification:   species top-1 acc + bootstrap 95% CI; inter-species prototype separability
  scalar:           learned curvature

Self-configuring: reads lora_r/blocks/sel_text/geometry from the checkpoint's `args`, so it runs
on ANY variant (B0..C10, euclidean, --sel-margin) without per-config flags.

  PYTHONPATH=src python scripts/diagnose_geometry.py --ckpt <ckpt.pt> [--n 4878] [--dataset bioscan]
  # euclidean ckpts: geometry probes (cones/entailment) are n/a; only the classification rows print.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from hyperbolic_plankton import lorentz as L
from hyperbolic_plankton.bioscan import BioscanHDF5Dataset, BIOSCAN_RANKS
from hyperbolic_plankton.data import RANKS as PZ_RANKS
from hyperbolic_plankton.lora import apply_lora
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.train import TaxonomyCollator


def _build_from_ckpt(path, device):
    """Rebuild the model with the architecture saved in the checkpoint's args."""
    sd = torch.load(path, map_location="cpu")
    a = sd.get("args", {})
    a = a if isinstance(a, dict) else vars(a)
    geom = a.get("geometry", "hyperbolic")
    model = HyperbolicCLIP(backbone=a.get("backbone", "clip"),
                           use_proj=not a.get("no_proj", False))
    if not a.get("no_lora", False):
        r = a.get("lora_r", 64)
        model = apply_lora(
            model, r=r, alpha=a.get("lora_alpha") or r,
            adapt_visual_blocks=a.get("lora_visual_blocks", 12),
            adapt_text_blocks=a.get("lora_text_blocks", 12),
            include_mlp=a.get("lora_mlp", False),
        )
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(device).eval()
    # independent per-rank text is only TRAINED when SEL is on with sel_text=independent. For CL-only
    # configs (lambda_sel=0: LRCL, RINCE-clonly, C4) the independent text is untrained junk — use the
    # cumulative text those configs actually trained, so the per-rank rows reflect a real embedding.
    sel_on = float(a.get("lambda_sel", 1.0)) > 0.0
    indep = sel_on and a.get("sel_text", "independent") == "independent"
    return model, geom, indep, a


@torch.no_grad()
def diagnose(path, dataset, n, device):
    model, geom, indep, a = _build_from_ckpt(path, device)
    curv = model.curvature
    ranks = BIOSCAN_RANKS if dataset == "bioscan" else PZ_RANKS
    if dataset != "bioscan":
        raise SystemExit("only --dataset bioscan wired in this diagnostic for now")
    import os
    hdf5 = os.environ.get(
        "BIOSCAN_HDF5",
        "/scratch/daniela/bioscan1m/data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5")
    ds = BioscanHDF5Dataset(hdf5, "test_seen")
    n = min(n, len(ds))
    items = [ds[i] for i in range(n)]
    pv, tb, _ = TaxonomyCollator(model.preprocess, ranks=ranks)(items)
    img = torch.cat([model.encode_image(pv[i:i + 256].to(device))
                     for i in range(0, pv.shape[0], 256)])  # batch to fit 24GB
    txt = model.encode_taxonomy(tb, indep=indep)  # config's trained SEL text form

    tag = path.split("/")[-1].replace("_final.pt", "").replace("_best.pt", "")
    print(f"\n===== {tag} =====")
    print(f"geometry={geom}  sel_text={'indep' if indep else 'cumul'}  "
          f"curv={curv.item():.4f}  n={n}  ranks={ranks}")

    # ---- classification (works for euclidean too: euclidean uses cosine-argmax) ----
    sp = tb["species"]
    uniq = sorted({s for s in sp if s})
    idx = {s: i for i, s in enumerate(uniq)}
    true = torch.tensor([idx[s] for s in sp], device=device)
    cum = model.encode_taxonomy(tb)  # cumulative prototypes = classifier's space
    P = torch.zeros(len(uniq), cum["species"].shape[1], device=device)
    cnt = torch.zeros(len(uniq), device=device)
    for i, s in enumerate(sp):
        P[idx[s]] += cum["species"][i]; cnt[idx[s]] += 1
    P = P / cnt[:, None].clamp(min=1)
    if geom == "euclidean":
        d = -(torch.nn.functional.normalize(img, dim=-1) @ torch.nn.functional.normalize(P, dim=-1).T)
    else:
        d = L.pairwise_dist(img, P, curv)
    pred = d.argmin(1)
    correct = (pred == true).float().cpu().numpy()
    acc = correct.mean()
    rng = np.random.default_rng(0)
    boot = [correct[rng.integers(0, n, n)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  species top-1 acc (DISTANCE) = {acc:.4f}  (95% CI [{lo:.4f}, {hi:.4f}], {len(uniq)} species)")

    # ---- ATMG angle-based classification (paper §6.1: predict = argmin exterior angle α) ----
    # α = oxy_angle(class_TEXT, image) — text is the apex (NOT symmetric: oxy_angle(img,text) is the
    # OTHER vertex). pairwise_oxy_angle(P, img) -> [C, B]; per image (column) pick the class (row) with
    # MINIMUM angle. (ATMG's released code does abs()+argmax, which CONTRADICTS its paper's "minimum
    # average α"; verified on the toy that min-angle = correct match, so we follow the PAPER.)
    if geom != "euclidean":
        ang = L.pairwise_oxy_angle(P, img, curv)        # [C, B] angle at each class-text apex
        pred_a = ang.argmin(0)                          # per image, class with smallest angle
        correct_a = (pred_a == true).float().cpu().numpy()
        acc_a = correct_a.mean()
        boot_a = [correct_a[rng.integers(0, n, n)].mean() for _ in range(2000)]
        lo_a, hi_a = np.percentile(boot_a, [2.5, 97.5])
        print(f"  species top-1 acc (ANGLE/ATMG) = {acc_a:.4f}  (95% CI [{lo_a:.4f}, {hi_a:.4f}])")

        # ---- CONE-ENERGY classification (Dhall 2020, learning_embeddings E_operator) ----
        # predict = the species whose cone the image LEAST violates (min order-violation energy).
        # E(class_text, image) = relu(oxy_angle(class_text, image) - half_aperture(class_text)) — our
        # Lorentz form of their `clamp(theta_between_x_y - psi_x, min=0)` (order_embeddings_h.py:1097).
        # This makes the CONE the classifier (their method), vs distance-to-prototype (ours). Tests
        # whether SEL's cones are useful-but-unused, or genuinely useless, for classification.
        cone_E = torch.clamp(L.pairwise_oxy_angle(P, img, curv)
                             - L.half_aperture(P, curv).unsqueeze(1), min=0.0)  # [C, B]
        pred_e = cone_E.argmin(0)                        # per image, class with least cone violation
        correct_e = (pred_e == true).float().cpu().numpy()
        acc_e = correct_e.mean()
        boot_e = [correct_e[rng.integers(0, n, n)].mean() for _ in range(2000)]
        lo_e, hi_e = np.percentile(boot_e, [2.5, 97.5])
        frac_contained = (cone_E.gather(0, true.unsqueeze(0)).squeeze(0) == 0).float().mean().item()
        print(f"  species top-1 acc (CONE-ENERGY/Dhall) = {acc_e:.4f}  (95% CI [{lo_e:.4f}, {hi_e:.4f}]; "
              f"img inside own-species cone {frac_contained:.2f})")
    offdiag = (L.pairwise_dist(P, P, curv) if geom != "euclidean"
               else 1 - torch.nn.functional.normalize(P, dim=-1) @ torch.nn.functional.normalize(P, dim=-1).T)
    od = offdiag[~torch.eye(len(uniq), dtype=bool, device=device)]
    print(f"  inter-species proto dist = {od.mean():.4f}  (MEAN separability — misleads; see NN below)")

    # ---- classifier subspace (cumulative-species protos = what actually classifies) ----
    # NN-separability + per-image margin are what predict F1 (mean separability does not):
    # classification is argmin distance, decided by the NEAREST confusable proto, not the average.
    PP = offdiag.clone()
    PP[torch.eye(len(uniq), dtype=bool, device=device)] = float("inf")
    nn = PP.min(1).values  # each proto's distance to its nearest neighbour
    proto_r = L.distance_from_origin(P, curv) if geom != "euclidean" else None
    d_correct = d.gather(1, true[:, None]).squeeze(1)
    dd = d.clone(); dd.scatter_(1, true[:, None], float("inf"))
    margin = dd.min(1).values - d_correct  # >0 => correctly classified; magnitude = confidence
    print(f"  CLASSIFIER protos (cumulative species): "
          f"radius {proto_r.mean():.3f}  NN-sep {nn.mean():.3f} (median {nn.median():.3f})")
    print(f"  per-image margin = {margin.mean():+.3f} (median {margin.median():+.3f}, "
          f"frac>0 {(margin > 0).float().mean():.3f})")

    if geom == "euclidean":
        print("  (euclidean: cone/entailment geometry n/a)")
        return

    # ---- per-rank text geometry (the SEL subspace) ----
    # txt = encode_taxonomy(indep=indep) = the text form SEL was TRAINED on. For indep runs this is
    # the independent per-rank text — a DIFFERENT embedding from the cumulative-species protos that
    # classify (reported above). So these rows describe SEL's geometry, NOT the classifier's.
    sub = "independent (SEL); classifier uses CUMULATIVE species above" if indep else "cumulative (= classifier space)"
    print(f"  per-rank text [{sub}]  radius   aperture")
    for r in ranks:
        if r not in txt:
            continue
        v = txt[f"{r}_valid"]
        if not bool(v.any()):
            continue
        x = txt[r][v]
        print(f"    {r:8s}  {L.distance_from_origin(x, curv).mean():.3f}    "
              f"{L.half_aperture(x, curv).mean():.3f}")

    # ---- per-edge: entail_ok, nesting slack ----
    present = [r for r in ranks if r in txt]
    print("  edge            entail_ok  slack(ψp-ψc-angle)  fits%")
    for p, c in zip(present[:-1], present[1:]):
        both = txt[f"{p}_valid"] & txt[f"{c}_valid"]
        if not bool(both.any()):
            continue
        pe, ce = txt[p][both], txt[c][both]
        ang = L.oxy_angle(pe, ce, curv)
        app, apc = L.half_aperture(pe, curv), L.half_aperture(ce, curv)
        eo = (ang <= app).float().mean()
        slack = (app - apc - ang)
        print(f"    {p}->{c:8s}  {eo:.3f}      {slack.mean():+.3f}             {(slack > 0).float().mean():.2f}")

    # ---- image: transitivity (in each ancestor's cone), radius, angular spread ----
    print("  image in cone of:  ", end="")
    for r in ranks:
        if r not in txt:
            continue
        v = txt[f"{r}_valid"]
        ang = L.oxy_angle(txt[r][v], img[v], curv)
        ap = L.half_aperture(txt[r][v], curv)
        print(f"{r}={(ang <= ap).float().mean():.2f} ", end="")
    print()
    ir = L.distance_from_origin(img, curv)
    dn = torch.nn.functional.normalize(img, dim=-1)
    cos = (dn @ dn.T)
    offc = cos[~torch.eye(n, dtype=bool, device=device)]
    print(f"  image radius = {ir.mean():.3f} ± {ir.std():.3f}   "
          f"image-dir mean pairwise cos = {offc.mean():.3f} (1=collapsed, 0=spread)")
    # image <-> its own species text distance. For indep runs txt["species"] is the SEL (independent)
    # species — NOT what the image is CL-aligned to. Report both: image<->SEL-species and the
    # image<->cumulative-species (cum["species"], the CL/classifier target).
    spv = txt["species_valid"]
    dii = torch.stack([L.pairwise_dist(img[i:i + 1], txt["species"][i:i + 1], curv)[0, 0]
                       for i in range(n) if spv[i]]).mean()
    cspv = cum["species_valid"]
    dic = torch.stack([L.pairwise_dist(img[i:i + 1], cum["species"][i:i + 1], curv)[0, 0]
                       for i in range(n) if cspv[i]]).mean()
    sel_lbl = "indep-SEL" if indep else "cumul"
    print(f"  image<->own-species-text dist: {sel_lbl} {dii:.3f}  |  cumulative(CL target) {dic:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="bioscan", choices=["bioscan", "planktonzilla"])
    ap.add_argument("--n", type=int, default=4878, help="#test_seen images (default = full)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diagnose(args.ckpt, args.dataset, args.n, device)


if __name__ == "__main__":
    main()
