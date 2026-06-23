"""Filter the cached plankton dataset to Kingdom != "" (drops the ~921 non-taxonomic junk
rows: empty-Kingdom 'mix'/'spore'/'unknown' labels). Reads the existing plankton==True cache,
applies the extra filter, saves to a NEW dir. Does not touch the original.

  PYTHONPATH=src python scripts/filter_cache_kingdom.py
"""

from datasets import load_from_disk

SRC = "/scratch/daniela/planktonzilla_cache/plankton"
OUT = "/scratch/daniela/planktonzilla_cache/plankton_kingdom"


def main():
    ds = load_from_disk(SRC)
    print(f"source: {len(ds):,} rows", flush=True)
    kept = ds.filter(lambda b: [k not in (None, "", "nan") for k in b["Kingdom"]],
                     batched=True, num_proc=8)
    print(f"after Kingdom != '': {len(kept):,} rows  (dropped {len(ds) - len(kept):,})", flush=True)
    kept.save_to_disk(OUT, num_proc=8)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
