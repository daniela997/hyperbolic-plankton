"""Fig-3a-style HoroPCA 2D projection of our hyperbolic embeddings (Poincaré disk).

Encodes a sample of per-rank taxonomy texts (kingdom..species) + image embeddings from a
trained checkpoint, converts Lorentz -> Poincaré ball, centers on the Fréchet mean, runs
HoroPCA (Chami et al. 2021) to 2D, and scatters in the disk colored by rank. Visualizes
the radial kingdom->species hierarchy the geom/* metrics measure numerically.

HoroPCA is imported from a local clone (no pip package); set HOROPCA_DIR if elsewhere.

Usage:
  PYTHONPATH=src python scripts/visualize_horopca.py \
      --ckpt /scratch/daniela/hyperbolic_plankton_ckpts/<tag>_it18000.pt \
      --backbone bioclip --lora --n 250 --out /scratch/daniela/horopca.png

HoroPCA clone needs a one-line fix for modern torch (removed `torch.solve`): in
geom/minkowski.py and geom/horo.py replace `torch.solve(B, A)` (returns (X, LU)) with
`torch.linalg.solve(A, B)` (args reversed). Runs on GPU; free the encoder first so the
N×N×D pairwise tensors fit (the script does this).
"""

import argparse
import os
import sys

import numpy as np
import torch

HOROPCA_DIR = os.environ.get("HOROPCA_DIR", "/home/daniela/other/HoroPCA")
sys.path.insert(0, HOROPCA_DIR)

from datasets import load_from_disk  # noqa: E402

from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset  # noqa: E402
from hyperbolic_plankton.lora import apply_lora  # noqa: E402
from hyperbolic_plankton.model import HyperbolicCLIP  # noqa: E402
from hyperbolic_plankton.train import TaxonomyCollator  # noqa: E402

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
SPLIT_DIR = "/scratch/daniela/hyperbolic_plankton_splits"


def lorentz_to_poincare(space: torch.Tensor, curv: float) -> torch.Tensor:
    """Lorentz (space components, our convention) -> Poincaré ball.

    Lorentz point: (x_time, x_space), x_time = sqrt(1/c + ||x_space||^2). Poincaré map:
        p = x_space / (x_time + 1/sqrt(c)).
    HoroPCA works in the curvature-1 Poincaré ball, so we rescale by sqrt(c) (the ball of
    curvature -c is the unit ball under x -> sqrt(c) x).
    """
    c = float(curv)
    x_time = torch.sqrt(1.0 / c + (space**2).sum(-1, keepdim=True))
    p = space / (x_time + 1.0 / (c**0.5))
    return p * (c**0.5)  # -> curvature-1 ball


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="planktonzilla", choices=["planktonzilla", "bioscan"])
    ap.add_argument("--backbone", default="bioclip", choices=["clip", "bioclip"])
    ap.add_argument("--lora", action="store_true", help="ckpt was trained with LoRA")
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--n", type=int, default=400, help="#samples (images) to plot")
    ap.add_argument("--out", default="/scratch/daniela/horopca.png")
    ap.add_argument("--sel-text", default="independent", choices=["independent", "cumulative"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HyperbolicCLIP(backbone=args.backbone)
    if args.lora:
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_r)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(device).eval()
    curv = model.curvature.item()
    print(f"loaded {args.ckpt}  curv={curv:.4f}")

    # sample seen-val rows (planktonzilla) or test_seen rows (bioscan)
    if args.dataset == "bioscan":
        from hyperbolic_plankton.bioscan import BIOSCAN_RANKS, BioscanHDF5Dataset

        ranks = BIOSCAN_RANKS
        BIOSCAN_HDF5 = "/scratch/daniela/bioscan1m/data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5"
        full = BioscanHDF5Dataset(BIOSCAN_HDF5, "test_seen")
        sel = np.random.default_rng(0).choice(len(full), size=args.n, replace=False)
        items = [full[int(i)] for i in sel]
    else:
        ranks = RANKS
        cache = load_from_disk(CACHE)
        val_idx = np.load(f"{SPLIT_DIR}/val_idx.npy")
        sel = np.random.default_rng(0).choice(val_idx, size=args.n, replace=False)
        ds = HFTaxonomyDataset(cache.select(sorted(sel.tolist())))
        items = [ds[i] for i in range(len(ds))]
    pix, tax, _ = TaxonomyCollator(model.preprocess, ranks=ranks)(items)

    # encode: images + each rank's text (cumulative/independent), all on the hyperboloid.
    # no_grad here only — HoroPCA.fit below needs autograd for its projection optimisation.
    with torch.no_grad():
        img = model.encode_image(pix.to(device))                   # [N, D]
        txt = model.encode_taxonomy(tax, indep=(args.sel_text == "independent"))                           # {rank: [N,D], rank_valid}
        pts, labels = [lorentz_to_poincare(img, curv)], ["image"] * img.shape[0]
        for r in ranks:
            valid = txt[f"{r}_valid"]
            if valid.any():
                pts.append(lorentz_to_poincare(txt[r][valid], curv))
                labels += [r] * int(valid.sum())
        P = torch.cat(pts, dim=0).float()
    print(f"points: {P.shape[0]}  dim={P.shape[1]}")

    # free the encoder (frees ~19GB) so the HoroPCA N×N×D pairwise tensors fit on GPU.
    del model, img, txt
    if device == "cuda":
        torch.cuda.empty_cache()

    # HoroPCA pipeline (Fréchet-center -> fit -> map to disk), as in HoroPCA/main.py — on GPU.
    import geom.poincare as poincare
    from learning.frechet import Frechet
    from learning.pca import HoroPCA

    P = P.to(device)
    mu, conv = Frechet(lr=1e-2, eps=1e-5, max_steps=5000).mean(P, return_converged=True)
    print(f"Fréchet mean converged: {conv}")
    Pc = poincare.reflect_at_zero(P, mu)
    hp = HoroPCA(dim=P.shape[1], n_components=2, lr=5e-2, max_steps=500).to(device)
    hp.fit(Pc, iterative=False, optim=True)
    emb2d = hp.map_to_ball(Pc).detach().cpu().numpy()

    _plot(emb2d, labels, args.out, curv, ranks)


def _plot(emb2d, labels, out, curv, ranks):
    import matplotlib.pyplot as plt

    order = ["image"] + ranks
    colors = {  # rank -> color (coarse->fine = dark->bright), images grey
        "image": "0.6", "kingdom": "#440154", "phylum": "#414487", "class": "#2a788e",
        "order": "#22a884", "family": "#7ad151", "genus": "#fde725", "species": "#d62728",
    }
    labels = np.array(labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1.0, color="k", fill=False, lw=0.8)
    ax.add_patch(circle)
    for name in order:
        m = labels == name
        if m.any():
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=8, c=colors[name], label=name, alpha=0.7)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(markerscale=2, fontsize=8, loc="upper right")
    ax.set_title(f"HoroPCA 2D (Poincaré disk)  curv={curv:.3f}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
