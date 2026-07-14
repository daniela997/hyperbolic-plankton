"""Build the Planktonzilla-faithful seen split once; save row-index files to scratch.

Saves `{train,val,test}_idx.npy` (indices into the full cache) + `unseen_idx.npy` and
`unseen_classes.json` under SPLIT_DIR. The training/eval code loads these (cheap) instead
of re-running the multi-minute stratified split every launch.

Run once:  PYTHONPATH=src python scripts/build_splits.py
"""

import json
import os

import numpy as np
from datasets import load_from_disk

from hyperbolic_plankton.data import HELD_OUT_DATASETS
from hyperbolic_plankton.eval import build_unseen_classes
from hyperbolic_plankton.split import _RANK_COLS, stratified_split_seen
from hyperbolic_plankton.data import _clean

CACHE = os.environ.get("HP_CACHE", "/scratch/daniela/planktonzilla_cache/plankton")
SPLIT_DIR = os.environ.get("HP_SPLIT_DIR", "/scratch/daniela/hyperbolic_plankton_splits")


def _full_strings_fast(ds):
    cols = {c: ds[c] for c in _RANK_COLS}
    n = len(ds[_RANK_COLS[0]])
    out = []
    for i in range(n):
        toks = [t for t in (_clean(cols[c][i]) for c in _RANK_COLS) if t is not None]
        out.append(" ".join(toks) if toks else "unknown")
    return out


def main():
    global SPLIT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--split-dir", default=SPLIT_DIR)
    ap.add_argument("--num-proc", type=int, default=16,
                    help="parallel workers for the stratified split. Each worker copies a "
                         "large Arrow slice, so lower this (e.g. 2-4) on a low-RAM pod to "
                         "avoid OOM (exit 137).")
    args = ap.parse_args()
    SPLIT_DIR = args.split_dir
    os.makedirs(SPLIT_DIR, exist_ok=True)
    full = load_from_disk(args.cache)
    n = len(full)
    print(f"cache rows: {n:,}")

    # carry an explicit global row id through the split so we can map splits back to
    # exact cache indices (original_path is not globally unique).
    full = full.add_column("_cache_idx", list(range(n)))

    dsets = np.array(full["dataset"])
    held = set(HELD_OUT_DATASETS)
    is_unseen = np.array([d in held for d in dsets])
    seen_global_idx = np.where(~is_unseen)[0]
    unseen_global_idx = np.where(is_unseen)[0]
    print(f"seen rows: {len(seen_global_idx):,}  held-out rows: {len(unseen_global_idx):,}")

    # ---- unseen class set + in-class eval indices (paper: 220 / 113,089) ----
    fulls = np.array(_full_strings_fast(full), dtype=object)
    seen_full = set(fulls[seen_global_idx].tolist())
    unseen_full = fulls[unseen_global_idx]
    classes = build_unseen_classes(unseen_full.tolist(), seen_full)
    in_class = set(classes)
    unseen_eval_idx = np.array(
        [int(unseen_global_idx[i]) for i, f in enumerate(unseen_full.tolist()) if f in in_class]
    )
    print(f"unseen classes: {len(classes)}  unseen eval rows: {len(unseen_eval_idx):,}")
    np.save(f"{SPLIT_DIR}/unseen_idx.npy", unseen_eval_idx)
    with open(f"{SPLIT_DIR}/unseen_classes.json", "w") as f:
        json.dump(classes, f)

    # ---- seen 60/20/20 stratified split (paper recipe) ----
    seen_ds = full.select(seen_global_idx.tolist())
    tr, va, te = stratified_split_seen(seen_ds, seed=42, num_proc=args.num_proc)
    print(f"seen split: train={len(tr):,} val={len(va):,} test={len(te):,}")

    for name, split in [("train", tr), ("val", va), ("test", te)]:
        idx = np.array(split["_cache_idx"])
        np.save(f"{SPLIT_DIR}/{name}_idx.npy", idx)
        print(f"  saved {name}_idx.npy  ({len(idx):,})")

    # ---- SEEN class set = `full` lineages present in the seen-VAL split ----
    # Matches Planktonzilla's CLIP eval protocol (`torch.unique` over the eval texts),
    # computed over the FULL val split so it is unbiased — the periodic monitor scores an
    # IMAGE subsample against THIS full class set (the bias only arises if `unique` is taken
    # over a subsample). Final eval (scripts/final_eval.py) recomputes present-classes over
    # the full test split directly.
    val_fulls = fulls[np.array(va["_cache_idx"])]
    seen_classes = sorted({f for f in val_fulls.tolist() if f != "unknown"})
    print(f"seen classes (present in val): {len(seen_classes)}")
    with open(f"{SPLIT_DIR}/seen_classes.json", "w") as f:
        json.dump(seen_classes, f)

    print(f"DONE -> {SPLIT_DIR}")


if __name__ == "__main__":
    main()
