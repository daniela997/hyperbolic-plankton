"""Verification for src/hyperbolic_plankton/eval.py (piece 6).

Success criteria (stated before implementing):
  1. `taxonomic_macro_f1` reproduces the paper's `evaluate_taxonomic_metrics` EXACTLY.
     Checked two ways: (a) numeric cross-check against a vendored copy of the paper
     function on random labels; (b) exact hand-computed F1 on a tiny worked example.
  2. `build_unseen_classes` = (held-out `full` strings) minus (seen pool) minus "unknown".
  3. `predict` returns the nearest prototype by Lorentzian distance (argmin); for a class
     whose own prototype is in the set, an image AT that prototype predicts that class.
  4. End-to-end on the real cache: encode prototypes, predict, score — runs, F1 in [0,1],
     and a model's own taxonomy-text "image" surrogate is classified to its own class.

The paper-function cross-check (criterion 1) is the load-bearing test: it pins our metric
to the exact reference, independent of our own reasoning about slice semantics.
"""

import os

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from hyperbolic_plankton import lorentz as L
from hyperbolic_plankton.data import RANKS
from hyperbolic_plankton.eval import (
    build_unseen_classes,
    encode_prototypes,
    predict,
    taxonomic_macro_f1,
)

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"


# --------------------------------------------------------------------------------
# vendored paper reference (notebooks/metrics_paper.ipynb::evaluate_taxonomic_metrics)
# copied verbatim — the ground truth our taxonomic_macro_f1 must match.
# --------------------------------------------------------------------------------

def _paper_evaluate_taxonomic_metrics(y_true, y_pred, class_names):
    TAXONOMIC_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    results = {}
    for level_idx, level_name in enumerate(TAXONOMIC_LEVELS):
        y_true_bin, y_pred_bin = [], []
        for yt, yp in zip(y_true, y_pred):
            true_tokens = class_names[yt].split()
            pred_tokens = class_names[yp].split()
            true_label = " ".join(true_tokens[: level_idx + 1])
            pred_label = " ".join(pred_tokens[: level_idx + 1])
            y_true_bin.append(true_label)
            y_pred_bin.append(pred_label)
        results[level_name] = {
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0, average="macro"),
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0, average="macro"),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0, average="macro"),
        }
    return results


# --------------------------------------------------------------------------------
# 1. metric correctness
# --------------------------------------------------------------------------------

def test_macro_f1_matches_paper_reference():
    """Numeric cross-check vs the vendored paper function on random ragged labels."""
    rng = np.random.default_rng(0)
    class_names = [
        "animalia",
        "animalia arthropoda",
        "animalia arthropoda copepoda",
        "animalia arthropoda copepoda calanoida",
        "chromista heterokontophyta bacillariophyceae",
        "chromista heterokontophyta bacillariophyceae fragilariales fragilariaceae diatoma",
    ]
    n = 400
    yt_idx = rng.integers(0, len(class_names), n)
    yp_idx = rng.integers(0, len(class_names), n)

    paper = _paper_evaluate_taxonomic_metrics(yt_idx, yp_idx, class_names)
    ours = taxonomic_macro_f1(
        [class_names[i] for i in yt_idx], [class_names[i] for i in yp_idx], RANKS
    )
    for rank in RANKS:
        for m in ("precision", "recall", "f1"):
            assert abs(paper[rank][m] - ours[rank][m]) < 1e-12, (rank, m)


def test_macro_f1_exact_hand_values():
    """Tiny worked example with hand-computed kingdom-level F1.

    2 samples. kingdom truncation -> true=[animalia, chromista], pred=[animalia, animalia].
    Per-class F1: animalia P=1/2 R=1 F1=2/3 ; chromista P=0 R=0 F1=0. macro = 1/3.
    """
    true_full = ["animalia arthropoda", "chromista heterokontophyta"]
    pred_full = ["animalia mollusca", "animalia arthropoda"]
    r = taxonomic_macro_f1(true_full, pred_full, RANKS)
    assert abs(r["kingdom"]["f1"] - (1 / 3)) < 1e-12
    # phylum level: true=[animalia arthropoda, chromista heterokontophyta],
    # pred=[animalia mollusca, animalia arthropoda] -> all 4 strings distinct, 0 matches.
    assert r["phylum"]["f1"] == 0.0


# --------------------------------------------------------------------------------
# 2. unseen class set
# --------------------------------------------------------------------------------

def test_build_unseen_classes():
    seen = ["animalia arthropoda", "chromista", "unknown"]
    unseen_full = [
        "animalia arthropoda",        # in seen -> excluded
        "animalia mollusca bivalvia",  # novel -> kept
        "chromista heterokontophyta",  # novel -> kept
        "unknown",                    # no kingdom -> dropped
        "animalia mollusca bivalvia",  # dup -> single
    ]
    out = build_unseen_classes(unseen_full, set(seen))
    assert out == ["animalia mollusca bivalvia", "chromista heterokontophyta"]


# --------------------------------------------------------------------------------
# 3. prediction geometry (no backbone needed — synthetic embeddings on the hyperboloid)
# --------------------------------------------------------------------------------

def test_predict_nearest_prototype():
    curv = torch.tensor(1.0)
    # 3 prototypes = exp_map0 of 3 distinct tangent directions
    tangents = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
    protos = L.exp_map0(tangents, curv)
    # images sitting exactly on prototypes 1, 0, 2 -> should predict 1, 0, 2
    imgs = protos[[1, 0, 2]]
    pred = predict(imgs, protos, curv)
    assert pred.tolist() == [1, 0, 2]


# --------------------------------------------------------------------------------
# 4. end-to-end on the real cache (text-as-image surrogate -> recovers own class)
# --------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(CACHE), reason="plankton cache not present")
def test_end_to_end_real_model():
    from hyperbolic_plankton.model import HyperbolicCLIP

    model = HyperbolicCLIP(backbone="clip").eval()
    classes = [
        "animalia arthropoda copepoda",
        "chromista heterokontophyta bacillariophyceae",
        "animalia cnidaria hydrozoa",
    ]
    protos = encode_prototypes(model, classes)
    assert protos.shape == (3, model.embed_dim)

    # use the same class texts as "image" surrogates (encode_text path): each should be
    # nearest to its OWN prototype (sanity that predict + prototypes are wired right).
    surrogate = model.encode_text([c for c in classes])
    pred = predict(surrogate, protos, model.curvature)
    assert pred.tolist() == [0, 1, 2]

    # metric runs and is in range on these (perfect) predictions
    r = taxonomic_macro_f1(classes, [classes[i] for i in pred.tolist()])
    assert r["full"]["f1"] == 1.0
    for rank in RANKS:
        assert 0.0 <= r[rank]["f1"] <= 1.0


# --------------------------------------------------------------------------------
# 5. geometry diagnostics
# --------------------------------------------------------------------------------

def test_geometry_stats_keys_and_ranges():
    """geometry_stats returns per-rank radius/aperture + per-edge entail_ok, in valid
    ranges, on a tiny synthetic taxonomy batch (no backbone needed for the math, but we
    go through the model so the encode path is exercised)."""
    from hyperbolic_plankton.eval import geometry_stats
    from hyperbolic_plankton.model import HyperbolicCLIP

    model = HyperbolicCLIP(backbone="clip").eval()
    # ragged batch: 2 samples, kingdom+phylum present, class only on sample 0
    tax = {
        "kingdom": ["animalia", "chromista"],
        "phylum": ["animalia arthropoda", "chromista heterokontophyta"],
        "class": ["animalia arthropoda copepoda", None],
        "order": [None, None], "family": [None, None],
        "genus": [None, None], "species": [None, None],
    }
    g = geometry_stats(model, tax)
    assert "geom/curv" in g and g["geom/curv"] > 0
    # ranks with valid entries report radius + aperture; empty ranks do not
    assert "geom/kingdom/radius" in g and "geom/kingdom/aperture" in g
    assert "geom/order/radius" not in g  # all-None rank skipped
    # aperture is a half-angle in (0, pi/2]; radius >= 0
    for k, v in g.items():
        if k.endswith("/aperture"):
            assert 0.0 < v <= 3.1416 / 2 + 1e-4, (k, v)
        if k.endswith("/radius"):
            assert v >= 0.0
        if k.endswith("/entail_ok"):
            assert 0.0 <= v <= 1.0
    # at least one parent->child entailment fraction is reported
    assert any(k.endswith("/entail_ok") for k in g)


def test_geometry_stats_radial_ordering_synthetic():
    """On hand-placed embeddings (root near origin, leaf far out), distance_from_origin
    must reflect the radial ordering the thesis predicts."""
    from hyperbolic_plankton import lorentz as L

    curv = torch.tensor(1.0)
    near = L.exp_map0(torch.tensor([[0.3, 0.0]]), curv)   # small radius
    far = L.exp_map0(torch.tensor([[4.0, 0.0]]), curv)    # large radius
    assert L.distance_from_origin(near, curv).item() < L.distance_from_origin(far, curv).item()
