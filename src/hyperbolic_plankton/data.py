"""Data bridge — Planktonzilla cache → taxonomy items (Piece 3).

Reads the cached plankton subset (`scripts/cache_planktonzilla.py` output) and yields
items shaped for the model: `{image, taxonomy, proposed_label}` where `taxonomy` is the
per-rank dict the model's `encode_taxonomy` consumes and `proposed_label` is the class id.

Spec source: HF schema (docs/planktonzilla.md) + scratchpad
`mine/hyperbolic/dataset.py::build_taxonomy_texts`. v1 scope: cumulative-lineage rank
strings + ragged `_valid` tracking. No hard negatives / independent-ranks / transforms
(the model + collator handle preprocessing).
"""

from __future__ import annotations

import io

import datasets
from PIL import Image
from torch.utils.data import Dataset

_BLANK_IMAGE = Image.new("RGB", (224, 224))

# HF taxonomic-rank columns (coarse -> fine) and the lowercase rank keys we emit.
# `proposed_label` is the WoRMS-harmonised class identity (often the full binomial, e.g.
# "aegina citrea", while Species holds only "citrea"); it is used as the CLASS LABEL, not
# as an extra taxonomy rank — so it is NOT in RANKS. SEL-inter entails the image into the
# deepest valid REAL rank (species/genus/...), and proposed_label defines positives/macro-F1.
_HF_RANK_COLUMNS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
RANKS = [c.lower() for c in _HF_RANK_COLUMNS]

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
    """Build the per-rank taxonomy dict for one row (real ranks only).

    Each rank's string is the **cumulative** lineage through that rank (e.g. family =
    "kingdom phylum class order family"); missing ranks are None. `full` is the deepest
    cumulative string (or "unknown"). `_valid_ranks` lists the populated rank keys.

    `proposed_label` is intentionally NOT a rank here — it is the class identity, carried
    separately by `HFTaxonomyDataset` (it is often a binomial overlapping Genus+Species).
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

    taxonomy["full"] = " ".join(cumulative) if cumulative else "unknown"
    taxonomy["_valid_ranks"] = valid_ranks
    return taxonomy


class HFTaxonomyDataset(Dataset):
    """Wraps the cached HF plankton dataset, emitting `{image, taxonomy, proposed_label}`.

    `proposed_label` is the class identity (WoRMS-harmonised label) used for contrastive
    positives and macro-F1; `taxonomy` holds the per-rank cumulative lineage strings.
    """

    def __init__(self, hf_dataset):
        # Disable HF's eager image decode: with decode=True, `ds[idx]` would raise on a
        # corrupt cell before we can catch it. With decode=False we get raw {bytes,path}
        # and decode ourselves inside a try (the dataset has a few undecodable images).
        self.ds = hf_dataset.cast_column("image", datasets.Image(decode=False))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        try:
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        except Exception:
            image = _BLANK_IMAGE.copy()
        return {
            "image": image,
            "taxonomy": build_taxonomy(row),
            "proposed_label": _clean(row["proposed_label"]) or "unknown",
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
