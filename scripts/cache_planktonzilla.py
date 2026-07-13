"""Download Planktonzilla-17M and cache the plankton subset to local disk.

The HF dataset is ~91GB / 17.4M rows across 187 parquet shards. We train only on the
~3.74M `plankton==True` subset, so we:
  1. load the full dataset (downloads all shards to the HF hub cache),
  2. filter to plankton rows,
  3. save_to_disk the filtered subset for fast repeated training reads.

The raw shards stay in the HF hub cache; the filtered subset is the artifact we keep.
Run with the dino_plankton env (needs pyarrow >= 24 to read these parquet files).

Usage:
    python scripts/cache_planktonzilla.py [--out DIR] [--num-proc N]
"""

from __future__ import annotations

import argparse
import os
import time

from datasets import load_dataset

REPO = "project-oceania/planktonzilla-17M"
DEFAULT_OUT = os.environ.get("HP_CACHE", "/scratch/daniela/planktonzilla_cache/plankton")


def _load_with_retry(num_proc: int, retries: int = 8, base_delay: float = 10.0):
    """`load_dataset` with retry+backoff. The ~91GB download over 187 shards regularly
    hits transient HTTP drops (`RemoteProtocolError: peer closed connection ...`), and a
    single dropped file aborts the whole load. Completed shards stay in the HF cache, so
    each retry RESUMES from where it broke rather than restarting — the loop just has to
    outlast the flaky connection."""
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(REPO, split="train", num_proc=num_proc)
        except Exception as e:  # noqa: BLE001 — network layer raises many exc types
            if attempt == retries:
                raise
            delay = base_delay * attempt  # linear backoff: 10, 20, 30, ... seconds
            print(f"  download attempt {attempt}/{retries} failed ({type(e).__name__}: {e}); "
                  f"retrying in {delay:.0f}s (cached shards are kept)...", flush=True)
            time.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--retries", type=int, default=8,
                    help="max load_dataset attempts on transient network failure")
    args = ap.parse_args()

    print(f"Loading {REPO} (downloads ~91GB to HF cache on first run)...", flush=True)
    ds = _load_with_retry(args.num_proc, retries=args.retries)
    print(f"Full dataset: {len(ds):,} rows, columns: {ds.column_names}", flush=True)

    print("Filtering to plankton==True ...", flush=True)
    plankton = ds.filter(lambda b: b["plankton"], batched=True, num_proc=args.num_proc)
    print(f"Plankton subset: {len(plankton):,} rows", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Saving to {args.out} ...", flush=True)
    plankton.save_to_disk(args.out, num_proc=args.num_proc)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
