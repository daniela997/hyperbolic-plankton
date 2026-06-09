"""Final evaluation — the paper numbers, NOT the periodic monitor.

Runs the FULL seen-test and FULL unseen splits against their full frozen class sets and
reports per-rank Macro-F1 (sklearn average='macro', zero_division=0 — matching
Planktonzilla's `f1_score(..., average='macro', zero_division=0)` over the fixed
`features['label'].names`). Use this for any reported result; the in-training periodic eval
is a seeded subsample for monitoring only.

  PYTHONPATH=src python scripts/final_eval.py --ckpt <ckpt.pt> --dataset planktonzilla --lora
  PYTHONPATH=src python scripts/final_eval.py --ckpt <ckpt.pt> --dataset bioscan --lora
"""

import argparse
import json

import numpy as np
import torch
from datasets import load_from_disk

from hyperbolic_plankton.bioscan import BIOSCAN_RANKS, BioscanHDF5Dataset
from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.eval import flatten_metrics, run_unseen_eval
from hyperbolic_plankton.lora import apply_lora
from hyperbolic_plankton.model import HyperbolicCLIP

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
SPLIT_DIR = "/scratch/daniela/hyperbolic_plankton_splits"
BIOSCAN_HDF5 = "/scratch/daniela/bioscan1m/data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5"


def _present_classes(ds) -> list[str]:
    """Distinct `full` lineages present in a dataset — Planktonzilla's CLIP protocol
    (`torch.unique` over the eval texts). Computed over the FULL split (not a subsample),
    so it is unbiased; the bias only arises when `unique` is taken over a subsample."""
    return sorted({ds[i]["taxonomy"]["full"] for i in range(len(ds))} - {"unknown"})


def _planktonzilla_sets():
    """(seen_ds, seen_classes), (unseen_ds, unseen_classes) over the FULL splits.

    Class sets = classes PRESENT in each full eval split (matches Planktonzilla's CLIP
    eval). unseen uses the prebuilt 220-class set (the paper's exact unseen benchmark)."""
    cache = load_from_disk(CACHE)
    test_idx = np.load(f"{SPLIT_DIR}/test_idx.npy")
    unseen_idx = np.load(f"{SPLIT_DIR}/unseen_idx.npy")
    seen_ds = HFTaxonomyDataset(cache.select(test_idx.tolist()))
    unseen_ds = HFTaxonomyDataset(cache.select(unseen_idx.tolist()))
    seen_classes = _present_classes(seen_ds)
    with open(f"{SPLIT_DIR}/unseen_classes.json") as f:
        unseen_classes = json.load(f)  # paper's fixed 220-class unseen benchmark
    return (seen_ds, seen_classes), (unseen_ds, unseen_classes), RANKS


def _bioscan_sets():
    """BIOSCAN: class set = classes present in each full test split (Planktonzilla CLIP
    protocol). No curated unseen benchmark here, so unseen also uses present-classes."""
    seen_ds = BioscanHDF5Dataset(BIOSCAN_HDF5, "test_seen")
    unseen_ds = BioscanHDF5Dataset(BIOSCAN_HDF5, "test_unseen")
    return (seen_ds, _present_classes(seen_ds)), (unseen_ds, _present_classes(unseen_ds)), BIOSCAN_RANKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="planktonzilla", choices=["planktonzilla", "bioscan"])
    ap.add_argument("--backbone", default="bioclip", choices=["clip", "bioclip"])
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HyperbolicCLIP(backbone=args.backbone)
    if args.lora:
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_r)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(device).eval()
    print(f"loaded {args.ckpt}  curv={model.curvature.item():.4f}")

    sets = _bioscan_sets() if args.dataset == "bioscan" else _planktonzilla_sets()
    (seen_ds, seen_classes), (unseen_ds, unseen_classes), ranks = sets

    for name, (ds, classes) in [("seen", (seen_ds, seen_classes)), ("unseen", (unseen_ds, unseen_classes))]:
        print(f"\n=== {name}: {len(ds):,} images, {len(classes)} classes ===")
        res = run_unseen_eval(model, ds, classes, num_workers=args.num_workers, ranks=ranks)
        m = flatten_metrics(res["metrics"], prefix=name)
        for r in ranks:
            k = f"{name}/{r}_f1"
            if k in m:
                print(f"  {r:8s} macro-F1 = {m[k]:.4f}")


if __name__ == "__main__":
    main()
