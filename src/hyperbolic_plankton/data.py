"""Data bridge — Planktonzilla cache → taxonomy items (Piece 3).

Reads the cached plankton subset (`scripts/cache_planktonzilla.py` output) and yields
items shaped for the model: `{image, taxonomy, folder}` where `taxonomy` is the per-rank
dict the model's `encode_taxonomy` consumes.

Spec source: HF schema (docs/planktonzilla.md) + scratchpad
`mine/hyperbolic/dataset.py::build_taxonomy_texts`. v1 scope: cumulative-lineage rank
strings + ragged `_valid` tracking. No hard negatives / independent-ranks / transforms
(the model + collator handle preprocessing).
"""

from __future__ import annotations

from torch.utils.data import Dataset

# HF columns (coarse -> fine) and the lowercase rank keys we emit. `Folder` (the class
# identity) is appended as the deepest rank so image->folder entailment has a leaf.
_HF_RANK_COLUMNS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
RANKS = [c.lower() for c in _HF_RANK_COLUMNS] + ["folder"]

# The 4 source datasets held out for unseen-species evaluation (lowercase `dataset`
# column values; see docs/planktonzilla.md).
HELD_OUT_DATASETS = ["global_uvp5", "planktoscope", "planktonset1.0", "syke_ifcb_2022"]


def _clean(v) -> str | None:
    """Normalise a cell to a stripped non-empty string, else None."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def build_taxonomy(row: dict) -> dict:
    """Build the per-rank taxonomy dict for one row.

    Each rank's string is the **cumulative** lineage through that rank (e.g. family =
    "kingdom phylum class order family"); missing ranks are None. `folder` is
    `proposed_label`. `full` is the deepest cumulative string (or "unknown").
    Includes `_valid_ranks` (list of populated rank keys).
    """
    taxonomy: dict = {}
    cumulative: list[str] = []
    valid_ranks: list[str] = []

    for col in _HF_RANK_COLUMNS:
        val = _clean(row.get(col))
        key = col.lower()
        if val is not None:
            cumulative.append(val)
            taxonomy[key] = " ".join(cumulative)
            valid_ranks.append(key)
        else:
            taxonomy[key] = None

    folder = _clean(row.get("proposed_label"))
    if folder is not None:
        cumulative.append(folder)
        taxonomy["folder"] = " ".join(cumulative)
        valid_ranks.append("folder")
    else:
        taxonomy["folder"] = None

    taxonomy["full"] = " ".join(cumulative) if cumulative else "unknown"
    taxonomy["_valid_ranks"] = valid_ranks
    return taxonomy


class HFTaxonomyDataset(Dataset):
    """Wraps the cached HF plankton dataset, emitting `{image, taxonomy, folder}`."""

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        taxonomy = build_taxonomy(row)
        return {
            "image": row["image"],  # PIL.Image (HF decodes)
            "taxonomy": taxonomy,
            "folder": _clean(row["proposed_label"]) or "unknown",
        }


def split_seen_unseen(hf_dataset):
    """Split the cached dataset into (in-domain pool, held-out) by source dataset.

    Returns `(seen_ds, unseen_ds)`: `unseen_ds` is the 4 held-out source datasets
    (for unseen-species evaluation); `seen_ds` is the other 11 (the in-domain training
    pool, before the stratified train/val/test split).
    """
    held = set(HELD_OUT_DATASETS)
    unseen = hf_dataset.filter(lambda b: [d in held for d in b["dataset"]], batched=True)
    seen = hf_dataset.filter(lambda b: [d not in held for d in b["dataset"]], batched=True)
    return seen, unseen
