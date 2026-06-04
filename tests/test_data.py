"""Verification for src/hyperbolic_plankton/data.py (piece 3).

Uses the real cached plankton subset (small .select() slices to stay fast). Checks:
  1. build_taxonomy: cumulative strings (real ranks only), ragged None handling,
     full + _valid_ranks — by exact hand-computed values on synthetic + real rows.
  2. HFTaxonomyDataset emits {image, taxonomy, proposed_label} with a PIL image.
  3. split_seen_unseen routes the 4 held-out sources to unseen, others to seen.
"""

import os

import pytest

from hyperbolic_plankton.data import (
    HELD_OUT_DATASETS,
    RANKS,
    HFTaxonomyDataset,
    build_taxonomy,
    split_seen_unseen,
)

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
pytestmark = pytest.mark.skipif(not os.path.exists(CACHE), reason="plankton cache not present")


@pytest.fixture(scope="module")
def cache():
    from datasets import load_from_disk

    return load_from_disk(CACHE)


# --------------------------------------------------------------------------------
# build_taxonomy — exact values
# --------------------------------------------------------------------------------

def test_build_taxonomy_cumulative_full_lineage():
    row = {
        "Kingdom": "chromista", "Phylum": "heterokontophyta", "Class": "bacillariophyceae",
        "Order": "fragilariales", "Family": "fragilariaceae", "Genus": "diatoma",
        "Species": None, "proposed_label": "diatoma",
    }
    t = build_taxonomy(row)
    assert t["kingdom"] == "chromista"
    assert t["phylum"] == "chromista heterokontophyta"
    assert t["genus"] == "chromista heterokontophyta bacillariophyceae fragilariales fragilariaceae diatoma"
    assert t["species"] is None  # missing -> None
    # proposed_label is NOT a taxonomy rank; full = deepest real-rank cumulative string
    assert "folder" not in t and "proposed_label" not in t
    assert t["full"] == t["genus"]
    assert t["_valid_ranks"] == ["kingdom", "phylum", "class", "order", "family", "genus"]


def test_build_taxonomy_shallow_ragged():
    """A coarse-only row (only Kingdom present among ranks)."""
    row = {
        "Kingdom": "chromista", "Phylum": None, "Class": None, "Order": None,
        "Family": None, "Genus": None, "Species": None, "proposed_label": "acantharia",
    }
    t = build_taxonomy(row)
    assert t["kingdom"] == "chromista"
    assert t["phylum"] is None and t["species"] is None
    assert t["full"] == "chromista"
    assert t["_valid_ranks"] == ["kingdom"]


def test_build_taxonomy_all_missing():
    row = {c: None for c in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]}
    row["proposed_label"] = None
    t = build_taxonomy(row)
    assert t["full"] == "unknown"
    assert t["_valid_ranks"] == []
    assert all(t[r] is None for r in RANKS)


def test_build_taxonomy_strips_and_nulls():
    row = {"Kingdom": "  animalia  ", "Phylum": "nan", "Class": "", "Order": None,
           "Family": None, "Genus": None, "Species": None, "proposed_label": "copepoda"}
    t = build_taxonomy(row)
    assert t["kingdom"] == "animalia"  # stripped
    assert t["phylum"] is None  # "nan" -> None
    assert t["class"] is None  # "" -> None
    assert t["full"] == "animalia"  # only Kingdom valid among ranks


# --------------------------------------------------------------------------------
# HFTaxonomyDataset — on real cached rows
# --------------------------------------------------------------------------------

def test_dataset_item_shape(cache):
    from PIL.Image import Image as PILImage

    ds = HFTaxonomyDataset(cache.select(range(4)))
    assert len(ds) == 4
    item = ds[0]
    assert set(item.keys()) == {"image", "taxonomy", "proposed_label"}
    assert isinstance(item["image"], PILImage)
    assert isinstance(item["proposed_label"], str) and item["proposed_label"]
    # taxonomy has every rank key + full + _valid_ranks
    for r in RANKS:
        assert r in item["taxonomy"]
    assert "full" in item["taxonomy"] and "_valid_ranks" in item["taxonomy"]


def test_dataset_handles_corrupt_image(cache):
    """A row whose image bytes are unreadable must yield a blank RGB, not raise."""
    import datasets
    from PIL.Image import Image as PILImage

    # craft a 1-row dataset with deliberately invalid image bytes
    base = cache.select(range(1)).cast_column("image", datasets.Image(decode=False))
    bad = base.map(lambda r: {"image": {"bytes": b"not-an-image", "path": None}})
    ds = HFTaxonomyDataset(bad)
    item = ds[0]
    assert isinstance(item["image"], PILImage)
    assert item["image"].size == (224, 224)  # the blank fallback


def test_dataset_real_taxonomy_consistency(cache):
    """each rank's cumulative string is a prefix-extension of the previous rank's."""
    ds = HFTaxonomyDataset(cache.select(range(0, 300000, 50000)))
    for i in range(len(ds)):
        t = ds[i]["taxonomy"]
        vr = t["_valid_ranks"]
        # each present rank's cumulative string extends the previous one (prefix nesting)
        for j in range(1, len(vr)):
            assert t[vr[j]].startswith(t[vr[j - 1]]), (vr, t[vr[j]])


# --------------------------------------------------------------------------------
# split_seen_unseen
# --------------------------------------------------------------------------------

def test_split_routes_by_source(cache):
    # take a slice that contains at least one held-out and one in-domain source
    sub = cache.select(range(0, len(cache), max(1, len(cache) // 2000)))  # ~2000 rows
    seen, unseen = split_seen_unseen(sub)
    assert len(seen) + len(unseen) == len(sub)
    held = set(HELD_OUT_DATASETS)
    assert all(d in held for d in unseen["dataset"])
    assert all(d not in held for d in seen["dataset"])
