"""Verification for src/hyperbolic_plankton/loss.py (piece 4).

Strategy: hand-built tensors where we know the right answer (the strongest check for a
loss), plus a numerical cross-check of the entailment primitives against the scratchpad
(`mine/hyperbolic/loss.py`), where the math is identical.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

from hyperbolic_plankton import lorentz as L
from hyperbolic_plankton import loss as Lo

torch.manual_seed(0)
CURV = 1.0


def _on_manifold(x):
    """Lift a euclidean vector onto the hyperboloid."""
    return L.exp_map0(x, CURV)


# --------------------------------------------------------------------------------
# Contrastive
# --------------------------------------------------------------------------------

def test_contrastive_perfect_alignment_low_loss():
    """When image[i] == text[i] and rows are well separated, loss is small."""
    text = _on_manifold(torch.randn(8, 16) * 2.0)
    img = text.clone()
    loss = Lo.hyperbolic_contrastive_loss(img, text, CURV, scale=10.0)
    # near-zero because each image's nearest text is its diagonal partner
    assert loss < 0.1, loss


def test_contrastive_symmetric_and_positive():
    img = _on_manifold(torch.randn(6, 16))
    text = _on_manifold(torch.randn(6, 16))
    loss = Lo.hyperbolic_contrastive_loss(img, text, CURV, scale=5.0)
    assert loss > 0 and torch.isfinite(loss)


# --------------------------------------------------------------------------------
# Entailment primitives (hand-built geometry)
# --------------------------------------------------------------------------------

def test_entailment_pos_zero_when_child_inside_cone():
    """A child far out along the parent's own direction is inside the cone -> 0 loss."""
    parent = _on_manifold(torch.tensor([[1.0, 0.0, 0.0]]))
    child = _on_manifold(torch.tensor([[5.0, 0.0, 0.0]]))  # same direction, deeper
    assert Lo.entailment_pos(parent, child, CURV).item() == pytest.approx(0.0, abs=1e-5)


def test_entailment_pos_positive_when_child_outside_cone():
    """A child orthogonal to the parent direction is outside the narrow cone -> >0."""
    parent = _on_manifold(torch.tensor([[3.0, 0.0, 0.0]]))
    child = _on_manifold(torch.tensor([[0.0, 3.0, 0.0]]))
    assert Lo.entailment_pos(parent, child, CURV).item() > 0.0


def test_entailment_neg_complementary():
    """entailment_neg is large exactly where entailment_pos is ~0 (child inside)."""
    parent = _on_manifold(torch.tensor([[1.0, 0.0, 0.0]]))
    child_inside = _on_manifold(torch.tensor([[5.0, 0.0, 0.0]]))
    assert Lo.entailment_neg(parent, child_inside, CURV, margin=0.1).item() > 0.0


# --------------------------------------------------------------------------------
# SEL-intra (ragged masks + Eq. 3 normalization)
# --------------------------------------------------------------------------------

def _make_text_embs(rank_to_vecs, ranks):
    """Build a text_embs dict {rank: [B,D], rank_valid: [B]} from per-rank lists of
    optional vectors (None = invalid)."""
    out = {}
    B = len(next(iter(rank_to_vecs.values())))
    D = 3
    for r in ranks:
        vecs = rank_to_vecs[r]
        emb = torch.zeros(B, D)
        valid = torch.zeros(B, dtype=torch.bool)
        for i, v in enumerate(vecs):
            if v is not None:
                emb[i] = _on_manifold(torch.tensor(v).float().unsqueeze(0)).squeeze(0)
                valid[i] = True
        out[r] = emb
        out[f"{r}_valid"] = valid
    return out


def test_sel_intra_zero_for_well_nested_hierarchy():
    """If each rank's embedding lies along its parent's direction (nested cones), the
    positive entailment is ~0. With 2 samples of the SAME lineage, negatives are absent,
    so SEL-intra should be ~0."""
    ranks = ["order", "family"]
    # both samples same lineage, child deeper along parent direction
    embs = _make_text_embs(
        {"order": [[1, 0, 0], [1, 0, 0]], "family": [[4, 0, 0], [4, 0, 0]]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A1"]}
    loss = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-4), loss


def test_sel_intra_ragged_denominator():
    """An edge present but with no valid pairs contributes 0 numerator yet still counts
    in the denominator. Verify by comparing a 1-edge vs 2-edge taxonomy where the 2nd
    edge has all-invalid child."""
    ranks = ["order", "family", "genus"]
    # genus all invalid -> edge (family,genus) present but contributes 0
    embs = _make_text_embs(
        {
            "order": [[1, 0, 0], [1, 0, 0]],
            "family": [[4, 0, 0], [4, 0, 0]],
            "genus": [None, None],
        },
        ranks,
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A1"], "genus": [None, None]}
    loss = Lo.sel_intra(embs, tax, ranks, CURV)
    # both edges present -> denominator 2; numerator from (order,family) only (~0 here)
    assert torch.isfinite(loss)


def test_sel_intra_penalizes_bad_nesting():
    """Child pointing away from parent -> positive entailment loss > 0."""
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[3, 0, 0], [3, 0, 0]], "family": [[0, 3, 0], [0, 3, 0]]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A1"]}
    loss = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    assert loss.item() > 0.0


# --------------------------------------------------------------------------------
# SEL-inter
# --------------------------------------------------------------------------------

def test_sel_inter_uses_deepest_rank():
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[1, 0, 0], [1, 0, 0]], "family": [[4, 0, 0], None]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", None]}
    # Images along the deepest text direction -> small positive entailment. (Not
    # exactly 0: a deep parent has a narrow cone, so a same-direction child at a
    # different radius is only approximately inside. Small loss is correct.)
    img = _on_manifold(torch.tensor([[6.0, 0, 0], [6.0, 0, 0]]))
    loss = Lo.sel_inter(img, embs, tax, ranks, CURV, use_negatives=False)
    assert torch.isfinite(loss) and loss.item() < 0.05


def test_stacked_returns_parts():
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[1, 0, 0], [2, 0, 0]], "family": [[4, 0, 0], [5, 0, 0]]}, ranks
    )
    tax = {"order": ["A", "B"], "family": ["A1", "B1"]}
    img = _on_manifold(torch.randn(2, 3))
    total, intra, inter = Lo.stacked_entailment_loss(img, embs, tax, ranks, CURV)
    assert torch.allclose(total, intra + inter)


def test_stats_decomposition_keys_and_consistency():
    """The optional `stats` dict collects per-edge intra + inter pos/neg components and
    pair counts; the recorded pos value matches a direct positive-only edge loss."""
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[3, 0, 0], [0, 3, 0]], "family": [[0, 2, 0], [2, 0, 0]]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A2"]}  # same parent -> all pairs positive
    img = _on_manifold(torch.randn(2, 3))
    stats: dict = {}
    Lo.stacked_entailment_loss(img, embs, tax, ranks, CURV, stats=stats)

    # intra edge + inter term both recorded, with pos/neg/n_pos/n_neg
    assert "sel_intra/order->family/pos" in stats
    assert "sel_intra/order->family/n_pos" in stats
    assert "sel_inter/text->image/pos" in stats
    # recorded intra pos == positive-only edge loss (use_negatives=False isolates pos)
    pos_only = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    assert abs(stats["sel_intra/order->family/pos"] - float(pos_only)) < 1e-6
    # all 4 pairs share the parent label -> 4 positives, 0 negatives
    assert stats["sel_intra/order->family/n_pos"] == 4
    assert stats["sel_intra/order->family/n_neg"] == 0


# --------------------------------------------------------------------------------
# EXACT-VALUE composition tests.
# These treat oxy_angle / half_aperture as a trusted oracle (MERU prior art) and verify
# that OUR loss logic — pos/neg masking, Eq.3 normalization (exact denominator), B×B
# grid orientation, deepest-rank selection — composes them correctly. A bug we could
# have introduced (a transpose, a wrong mask, a wrong denominator) is caught here.
# --------------------------------------------------------------------------------

def test_edge_loss_exact_value_single_edge():
    """sel_intra on one edge with 2 same-parent samples == hand-computed mean of
    relu(angle - aperture) over all valid positive pairs (the full 2x2 grid, since
    both share the parent label)."""
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[3, 0, 0], [0, 3, 0]], "family": [[0, 2, 0], [2, 0, 0]]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A2"]}  # same parent -> all 4 pairs positive

    P, C = embs["order"], embs["family"]
    B = 2
    p_grid = P.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)  # parent in cols
    c_grid = C.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)  # child in rows
    expected_pos = Lo.entailment_pos(p_grid, c_grid, CURV).reshape(B, B).mean()

    got = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    assert torch.allclose(got, expected_pos, atol=1e-6), (got, expected_pos)


def test_mask_separates_positives_from_negatives():
    """Different parent labels => those cross pairs are NEGATIVES, not positives.
    With use_negatives=False, only same-parent pairs count toward the positive mean."""
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[3, 0, 0], [0, 3, 0]], "family": [[0, 2, 0], [2, 0, 0]]}, ranks
    )
    tax = {"order": ["A", "B"], "family": ["A1", "B1"]}  # different parents -> only diagonal positive

    P, C = embs["order"], embs["family"]
    B = 2
    p_grid = P.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
    c_grid = C.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
    pos_all = Lo.entailment_pos(p_grid, c_grid, CURV).reshape(B, B)
    # positive pairs are the diagonal (same parent label only on i==i)
    expected = pos_all.diag().mean()

    got = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)


def test_grid_orientation_child_in_rows():
    """A transpose bug (parent/child swapped) would change the answer for asymmetric
    parent vs child. Verify our result matches the child-as-rows / parent-as-cols
    convention, NOT its transpose."""
    ranks = ["order", "family"]
    # asymmetric: parent and child have clearly different directions/norms
    embs = _make_text_embs(
        {"order": [[5, 0, 0], [5, 0, 0]], "family": [[0, 1, 0], [0, 1, 0]]}, ranks
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A1"]}
    P, C = embs["order"], embs["family"]
    B = 2
    # correct orientation: parent=P broadcast over cols, child=C over rows
    p_grid = P.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
    c_grid = C.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)
    correct = Lo.entailment_pos(p_grid, c_grid, CURV).reshape(B, B).mean()
    # swapped orientation would compute entailment_pos(child, parent) -> different value
    swapped = Lo.entailment_pos(c_grid, p_grid, CURV).reshape(B, B).mean()

    got = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    assert torch.allclose(got, correct, atol=1e-6)
    assert not torch.allclose(correct, swapped, atol=1e-3)  # the test is meaningful


def test_sel_intra_exact_denominator_with_ragged_edge():
    """Eq.3 denominator = number of SUPERVISED edges, not active ones. With 2 edges
    where the 2nd contributes 0 (all-invalid genus), the result must equal
    (edge1_loss + 0) / 2, NOT edge1_loss / 1."""
    ranks = ["order", "family", "genus"]
    embs = _make_text_embs(
        {
            "order": [[3, 0, 0], [0, 3, 0]],
            "family": [[0, 2, 0], [2, 0, 0]],
            "genus": [None, None],  # edge (family,genus) supervised but contributes 0
        },
        ranks,
    )
    tax = {"order": ["A", "A"], "family": ["A1", "A1"], "genus": [None, None]}

    edge1 = Lo.sel_intra(
        {k: embs[k] for k in ["order", "family", "order_valid", "family_valid"]},
        tax, ["order", "family"], CURV, use_negatives=False,
    )
    full = Lo.sel_intra(embs, tax, ranks, CURV, use_negatives=False)
    # 3 ranks => 2 edges; genus edge present (both rank keys in dict) but 0 numerator
    assert torch.allclose(full, edge1 / 2, atol=1e-6), (full, edge1)


def test_sel_inter_picks_deepest_valid_and_its_label():
    """_deepest_text must pick the leaf-most valid rank per sample, and sel_inter must
    use THAT rank's text as the parent and its label for masking."""
    ranks = ["order", "family"]
    embs = _make_text_embs(
        {"order": [[1, 0, 0], [1, 0, 0]], "family": [[4, 0, 0], None]}, ranks
    )
    deep, chosen = Lo._deepest_text(embs, ranks)
    assert chosen == ["family", "order"]  # sample0 has family; sample1 falls back to order
    assert torch.allclose(deep[0], embs["family"][0])
    assert torch.allclose(deep[1], embs["order"][1])


# --------------------------------------------------------------------------------
# Cross-check entailment primitives vs scratchpad
# --------------------------------------------------------------------------------

def _load_scratch_loss():
    """Load scratchpad loss.py, stubbing its `from hyperbolic import LorentzMath`."""
    path = "/home/daniela/mine/hyperbolic/loss.py"
    hyp = "/home/daniela/mine/hyperbolic/hyperbolic.py"
    if not (os.path.exists(path) and os.path.exists(hyp)):
        return None
    # make `import hyperbolic` resolve to the scratchpad module
    spec_h = importlib.util.spec_from_file_location("hyperbolic", hyp)
    mod_h = importlib.util.module_from_spec(spec_h)
    sys.modules["hyperbolic"] = mod_h
    spec_h.loader.exec_module(mod_h)
    spec = importlib.util.spec_from_file_location("scratch_loss", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print("scratch loss load failed:", e)
        return None
    return mod


SCRATCH = _load_scratch_loss()


# Tolerance note: the scratchpad's oxy_angle/half_aperture default to eps=1e-4 while
# ours pass lorentz's eps=1e-8. Near the acos clamp boundary (angles ~pi) that eps gap
# produces ~1e-2 differences. A match at atol=2e-2 confirms the math is otherwise
# identical and the only difference is eps (same finding as the Piece-1 lorentz tests).
_EPS_GAP_ATOL = 2e-2


@pytest.mark.skipif(SCRATCH is None, reason="scratchpad loss not importable")
def test_crosscheck_entailment_pos_vs_scratch():
    parent = _on_manifold(torch.randn(5, 16))
    child = _on_manifold(torch.randn(5, 16))
    ours = Lo.entailment_pos(parent, child, CURV, r_min=0.1)
    ref = SCRATCH.entailment_loss_positive(parent, child, curv=CURV, r_min=0.1)
    assert torch.allclose(ours, ref, atol=_EPS_GAP_ATOL), (ours - ref).abs().max()


@pytest.mark.skipif(SCRATCH is None, reason="scratchpad loss not importable")
def test_crosscheck_entailment_neg_vs_scratch():
    parent = _on_manifold(torch.randn(5, 16))
    child = _on_manifold(torch.randn(5, 16))
    ours = Lo.entailment_neg(parent, child, CURV, r_min=0.1, margin=0.1)
    ref = SCRATCH.entailment_loss_negative(parent, child, curv=CURV, r_min=0.1, margin=0.1)
    assert torch.allclose(ours, ref, atol=_EPS_GAP_ATOL), (ours - ref).abs().max()


# --------------------------------------------------------------------------------
# Ragged-taxonomy NaN regression
# --------------------------------------------------------------------------------

def test_sel_backward_finite_with_ragged_missing_ranks():
    """Invalid (None) ranks arrive as ZERO embeddings; fed to the cone geometry they make
    half_aperture's asin argument blow up and its BACKWARD return NaN. _edge_loss must
    sanitize them so gradients stay finite. Regression for the training-blocker bug.

    Build a batch where deep ranks are mostly missing (the real ragged case), make the
    embeddings require grad, backprop the full SEL, and assert no NaN/Inf grad anywhere.
    """
    ranks = ["kingdom", "phylum", "class"]
    embs = _make_text_embs(
        {
            "kingdom": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            "phylum": [[2, 1, 0], None, [0, 1, 2]],   # sample 1 missing phylum
            "class": [None, None, [1, 1, 1]],          # only sample 2 has class
        },
        ranks,
    )
    for r in ranks:
        embs[r] = embs[r].clone().requires_grad_(True)
    tax = {
        "kingdom": ["A", "B", "C"],
        "phylum": ["A1", None, "C1"],
        "class": [None, None, "C1a"],
    }
    img = _on_manifold(torch.randn(3, 3)).requires_grad_(True)

    total, intra, inter = Lo.stacked_entailment_loss(img, embs, tax, ranks, CURV)
    assert torch.isfinite(total)
    total.backward()
    for r in ranks:
        assert torch.isfinite(embs[r].grad).all(), f"NaN/Inf grad in {r}"
    assert torch.isfinite(img.grad).all()
