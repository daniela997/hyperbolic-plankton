"""Run the Planktonzilla-faithful unseen-species eval on the cached plankton subset.

Builds the unseen class set (held-out `full` strings absent from the seen pool), restricts
the held-out rows to those classes, and reports per-rank macro-F1 (paper Table 3 protocol,
hyperbolic-distance prediction).

Usage:
  PYTHONPATH=src python scripts/run_unseen_eval.py --backbone clip --limit 5000
  PYTHONPATH=src python scripts/run_unseen_eval.py --backbone bioclip --ckpt path/to.pt

By default the model is an untrained HyperbolicCLIP (random projector) — a sanity floor.
Pass --ckpt to load a trained state_dict (Piece 5 output).
"""

import argparse

import torch
from datasets import load_from_disk

from hyperbolic_plankton.data import HFTaxonomyDataset, split_seen_unseen
from hyperbolic_plankton.eval import _full_strings, build_unseen_classes, run_unseen_eval
from hyperbolic_plankton.model import HyperbolicCLIP

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip"])
    ap.add_argument("--ckpt", default=None, help="optional trained state_dict")
    ap.add_argument("--limit", type=int, default=None, help="cap #eval images (smoke)")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    print(f"loading cache: {args.cache}")
    ds = load_from_disk(args.cache)
    seen_ds, unseen_ds = split_seen_unseen(ds)
    print(f"seen rows: {len(seen_ds)}  held-out rows: {len(unseen_ds)}")

    print("building unseen class set (held-out `full` minus seen pool)...")
    seen_full = set(_full_strings(seen_ds))
    unseen_full = _full_strings(unseen_ds)
    classes = build_unseen_classes(unseen_full, seen_full)
    print(f"unseen classes: {len(classes)}")

    # restrict held-out rows to the unseen classes
    in_class = set(classes)
    keep = [i for i, f in enumerate(unseen_full) if f in in_class]
    print(f"unseen eval rows (in-class): {len(keep)}")
    unseen_eval = HFTaxonomyDataset(unseen_ds.select(keep))

    model = HyperbolicCLIP(backbone=args.backbone)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
        print(f"loaded ckpt: {args.ckpt}")
    if torch.cuda.is_available():
        model = model.cuda()

    out = run_unseen_eval(model, unseen_eval, classes, limit=args.limit)
    print(f"\n=== unseen macro-F1 ({args.backbone}, n={out['n']}, "
          f"{out['n_classes']} classes) ===")
    for rank, m in out["metrics"].items():
        if rank == "full":
            print(f"{'OVERALL':<10} F1={m['f1']:.4f}")
        else:
            print(f"{rank:<10} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")


if __name__ == "__main__":
    main()
