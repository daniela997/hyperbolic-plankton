"""Verification for src/hyperbolic_plankton/model.py (piece 2).

Checks (success criteria stated before implementing):
  1. build_backbone returns a frozen module with the expected output dim.
  2. encode_image/encode_text return (B, embed_dim) on the hyperboloid (<x,x>_L = -1/curv).
  3. Frozen backbone gets NO gradient; projection heads + scalars DO.
  4. encode_taxonomy produces correct per-rank embeddings + validity masks, incl. None.
  5. Both inits (CLIP, BioCLIP) load and run.

Backbone loads hit the network/HF cache on first run; these are slow integration tests.
"""

import math

import pytest
import torch

from hyperbolic_plankton import lorentz as L
from hyperbolic_plankton.model import HyperbolicCLIP, build_backbone

# Load each backbone once and share across that backbone's tests.
_MODELS: dict[str, HyperbolicCLIP] = {}


def get_model(backbone: str) -> HyperbolicCLIP:
    if backbone not in _MODELS:
        _MODELS[backbone] = HyperbolicCLIP(backbone=backbone).eval()
    return _MODELS[backbone]


BACKBONES = ["clip", "bioclip"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_backbone_frozen_and_dim(backbone):
    model, dim = build_backbone(backbone)
    assert dim == 512
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("backbone", BACKBONES)
def test_image_on_hyperboloid(backbone):
    model = get_model(backbone)
    pix = torch.randn(3, 3, 224, 224)
    emb = model.encode_image(pix, project=True)
    assert emb.shape == (3, model.embed_dim)
    inner = (emb**2).sum(-1) - L.time_component(emb, model.curvature).squeeze(-1) ** 2
    target = torch.full_like(inner, -1.0 / model.curvature.item())
    assert torch.allclose(inner, target, atol=1e-3), (inner - target).abs().max()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_text_on_hyperboloid(backbone):
    model = get_model(backbone)
    emb = model.encode_text(["a copepod", "a diatom", "marine plankton"], project=True)
    assert emb.shape == (3, model.embed_dim)
    inner = (emb**2).sum(-1) - L.time_component(emb, model.curvature).squeeze(-1) ** 2
    target = torch.full_like(inner, -1.0 / model.curvature.item())
    assert torch.allclose(inner, target, atol=1e-3)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_project_changes_embedding(backbone):
    """project=True applies exp_map0, which changes the embedding vs the raw tangent.

    Note: in the space-components convention every vector trivially satisfies
    <x,x>_L = -1/curv (because time = sqrt(1/curv + ||x||^2) is defined that way), so
    the constraint cannot distinguish tangent from manifold. The real effect of
    project=True is the geodesic placement (norm change), which we check here.
    """
    model = get_model(backbone)
    tangent = model.encode_text(["plankton"], project=False)
    projected = model.encode_text(["plankton"], project=True)
    assert tangent.shape == projected.shape == (1, model.embed_dim)
    assert not torch.allclose(tangent, projected, atol=1e-3)


def test_only_projection_and_scalars_get_grad():
    """After a backward pass, frozen backbone has no grad; proj + scalars do."""
    model = get_model("clip")
    model.zero_grad(set_to_none=True)
    pix = torch.randn(2, 3, 224, 224)
    loss = model.encode_image(pix).sum() + model.encode_text(["a", "b"]).sum()
    loss.backward()

    backbone_grads = [p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.clip.parameters()]
    assert not any(backbone_grads), "frozen backbone unexpectedly got gradients"

    assert model.visual_proj.weight.grad is not None
    assert model.textual_proj.weight.grad is not None
    # at least one MERU scalar should receive grad through the lift
    assert any(s.grad is not None for s in
               [model.visual_alpha, model.textual_alpha, model.curv])


def test_encode_taxonomy_masks_and_shapes():
    model = get_model("clip")
    tax = {
        "order": ["Calanoida", "Calanoida", None],
        "genus": ["Calanus", None, None],
        "_meta": ["ignored"],  # underscore keys skipped
    }
    out = model.encode_taxonomy(tax, project=True)
    assert "_meta" not in out
    assert out["order"].shape == (3, model.embed_dim)
    assert out["order_valid"].tolist() == [True, True, False]
    assert out["genus_valid"].tolist() == [True, False, False]
    # invalid rows are zero; valid rows are on the hyperboloid
    assert torch.allclose(out["order"][2], torch.zeros(model.embed_dim))
    v = out["order"][:2]
    inner = (v**2).sum(-1) - L.time_component(v, model.curvature).squeeze(-1) ** 2
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / model.curvature.item()), atol=1e-3)


def test_clamp_params_keeps_alpha_nonpositive():
    model = get_model("clip")
    with torch.no_grad():
        model.visual_alpha.fill_(1.0)  # would up-scale; must be clamped to <= 0
        model.logit_scale.fill_(10.0)  # > ln(100)
    model.clamp_params()
    assert model.visual_alpha.item() <= 0.0
    assert model.logit_scale.item() <= math.log(100) + 1e-6
