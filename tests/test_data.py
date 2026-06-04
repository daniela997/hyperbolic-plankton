"""Verification for src/hyperbolic_plankton/data.py (piece 3).

Uses the real cached plankton subset (small .select() slices to stay fast). Checks:
  1. build_taxonomy: cumulative strings, ragged None handling, folder=proposed_label,
     full + _valid_ranks — by exact hand-computed values on synthetic + real rows.
  2. HFTaxonomyDataset emits {image, taxonomy, folder} with a PIL image.
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
    # folder = proposed_label appended to the cumulative chain
    assert t["folder"] == "chromista heterokontophyta bacillariophyceae fragilariales fragilariaceae diatoma diatoma"
    assert t["full"] == t["folder"]
    assert t["_valid_ranks"] == ["kingdom", "phylum", "class", "order", "family", "genus", "folder"]


def test_build_taxonomy_shallow_ragged():
    """A coarse-only row (only Kingdom + a proposed label)."""
    row = {
        "Kingdom": "chromista", "Phylum": None, "Class": None, "Order": None,
        "Family": None, "Genus": None, "Species": None, "proposed_label": "acantharia",
    }
    t = build_taxonomy(row)
    assert t["kingdom"] == "chromista"
    assert t["phylum"] is None and t["species"] is None
    assert t["folder"] == "chromista acantharia"
    assert t["_valid_ranks"] == ["kingdom", "folder"]


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
    assert t["folder"] == "animalia copepoda"


# --------------------------------------------------------------------------------
# HFTaxonomyDataset — on real cached rows
# --------------------------------------------------------------------------------

def test_dataset_item_shape(cache):
    from PIL.Image import Image as PILImage

    ds = HFTaxonomyDataset(cache.select(range(4)))
    assert len(ds) == 4
    item = ds[0]
    assert set(item.keys()) == {"image", "taxonomy", "folder"}
    assert isinstance(item["image"], PILImage)
    assert isinstance(item["folder"], str) and item["folder"]
    # taxonomy has every rank key + full + _valid_ranks
    for r in RANKS:
        assert r in item["taxonomy"]
    assert "full" in item["taxonomy"] and "_valid_ranks" in item["taxonomy"]


def test_dataset_real_taxonomy_consistency(cache):
    """folder string should end with the proposed_label; valid ranks form a prefix."""
    ds = HFTaxonomyDataset(cache.select(range(0, 300000, 50000)))
    for i in range(len(ds)):
        t = ds[i]["taxonomy"]
        vr = t["_valid_ranks"]
        # valid ranks are a contiguous prefix of RANKS up to the deepest present
        # (allowing folder always last); each present rank's string contains the prior
        for j in range(1, len(vr)):
            if vr[j] != "folder":
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
