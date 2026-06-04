"""Verification for src/hyperbolic_plankton/train.py (piece 5).

Success criteria (stated before implementing):
  1. TaxonomyCollator: real cached items -> (pixel_values [B,3,224,224] float tensor,
     taxonomy_batch with every rank as a [B] list + full + _valid_ranks, proposed_labels
     [B]). Per-rank lists align 1:1 with the items (None preserved).
  2. train_step runs end to end and returns finite loss parts.
  3. Overfit check: repeatedly stepping on ONE fixed tiny batch drives the loss DOWN
     (the train path actually optimizes the trainable params).
  4. Grads reach the projection heads / MERU scalars, not the frozen backbone.

Uses the real cached plankton subset (tiny .select slices). CPU, projector-only, clip
init — fast enough; the heavier GPU/bioclip/LoRA paths reuse the same code.
"""

import os

import pytest
import torch

from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.train import TaxonomyCollator, train_step

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
pytestmark = pytest.mark.skipif(not os.path.exists(CACHE), reason="plankton cache not present")


@pytest.fixture(scope="module")
def model():
    return HyperbolicCLIP(backbone="clip").eval()


@pytest.fixture(scope="module")
def batch(model):
    from datasets import load_from_disk

    ds = HFTaxonomyDataset(load_from_disk(CACHE).select(range(8)))
    items = [ds[i] for i in range(len(ds))]
    collator = TaxonomyCollator(model.preprocess)
    return items, collator(items)


def test_collator_shapes(batch):
    items, (pixel_values, taxonomy_batch, proposed_labels) = batch
    B = len(items)
    assert pixel_values.shape == (B, 3, 224, 224)
    assert pixel_values.dtype == torch.float32
    for rank in RANKS:
        assert len(taxonomy_batch[rank]) == B
        # per-rank list matches each item's taxonomy entry exactly (None preserved)
        for i, item in enumerate(items):
            assert taxonomy_batch[rank][i] == item["taxonomy"][rank]
    assert len(taxonomy_batch["full"]) == B
    assert len(taxonomy_batch["_valid_ranks"]) == B
    assert proposed_labels == [item["proposed_label"] for item in items]


def test_train_step_runs_and_decreases(model, batch):
    """One fixed batch, several steps -> loss should fall (overfit sanity)."""
    _, (pixel_values, taxonomy_batch, _) = batch
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )

    first = train_step(model, pixel_values, taxonomy_batch, opt)
    assert all(torch.isfinite(torch.tensor(v)) for v in first.values())

    last = first
    for _ in range(20):
        last = train_step(model, pixel_values, taxonomy_batch, opt)

    assert last["loss"] < first["loss"], (first["loss"], last["loss"])


def test_grads_route_to_trainable_only(model, batch):
    _, (pixel_values, taxonomy_batch, _) = batch
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.zero_grad(set_to_none=True)
    train_step(model, pixel_values, taxonomy_batch, opt)

    # projection head + a MERU scalar received gradient this step
    assert model.visual_proj.weight.grad is not None
    assert model.textual_proj.weight.grad is not None
    # frozen backbone conv gets none
    conv = model.clip.visual.conv1.weight
    assert not conv.requires_grad and conv.grad is None
