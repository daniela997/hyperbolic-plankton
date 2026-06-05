"""Collator + single train step (Piece 5).

Bridges the data items (`data.HFTaxonomyDataset` -> `{image, taxonomy, proposed_label}`)
to the model + losses:

  - `TaxonomyCollator`: applies the backbone's open_clip preprocess to each PIL image and
    stacks to `pixel_values [B,3,224,224]`; transposes the per-item taxonomy dicts into the
    per-rank `{rank: [B] list}` shape `encode_taxonomy` / the SEL loss consume.
  - `train_step`: one optimizer step of `contrastive(img, deepest_text) + λ·SEL`.

Loss wiring follows scratchpad `train_all_setups.py::train_epoch_sel_cl` (CL target =
the deepest valid text per sample, then SEL on the full rank stack). v1 stays single-GPU,
projector-only by default; LoRA is just `apply_lora` on the model before training.
"""

from __future__ import annotations

import torch

from .data import RANKS
from .loss import _deepest_text, hyperbolic_contrastive_loss, stacked_entailment_loss


class TaxonomyCollator:
    """Collate `{image, taxonomy, proposed_label}` items into model-ready batches.

    Returns `(pixel_values [B,3,224,224], taxonomy_batch, proposed_labels [B])` where
    `taxonomy_batch` is `{rank: [B] list of str|None}` for each rank in `RANKS`, plus
    `full` ([B] list) and `_valid_ranks` ([B] list of lists) carried through.
    """

    def __init__(self, preprocess, ranks: list[str] = RANKS):
        self.preprocess = preprocess
        self.ranks = ranks

    def __call__(self, batch: list[dict]):
        pixel_values = torch.stack([self.preprocess(item["image"]) for item in batch])

        taxonomy_batch: dict = {}
        for rank in self.ranks:
            taxonomy_batch[rank] = [item["taxonomy"][rank] for item in batch]
            # independent per-rank text ("Rank: Value") for SEL-intra
            taxonomy_batch[f"{rank}_indep"] = [item["taxonomy"][f"{rank}_indep"] for item in batch]
        taxonomy_batch["full"] = [item["taxonomy"]["full"] for item in batch]
        taxonomy_batch["_valid_ranks"] = [item["taxonomy"]["_valid_ranks"] for item in batch]

        proposed_labels = [item["proposed_label"] for item in batch]
        return pixel_values, taxonomy_batch, proposed_labels


def train_step(
    model,
    pixel_values: torch.Tensor,
    taxonomy_batch: dict,
    optimizer,
    lambda_sel: float = 1.0,
    ranks: list[str] = RANKS,
) -> dict:
    """One optimizer step of `contrastive(img, deepest_text) + λ·SEL`.

    The contrastive target is each sample's deepest valid text embedding (so every image
    has a 1:1 positive); SEL then ties the whole rank stack together. Returns the scalar
    loss parts (floats) for logging.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pixel_values = pixel_values.to(model.device)
    img = model.encode_image(pixel_values)
    text_embs = model.encode_taxonomy(taxonomy_batch)
    curv = model.curvature
    scale = model.logit_scale.exp()

    deepest, _ = _deepest_text(text_embs, ranks)
    cl = hyperbolic_contrastive_loss(img, deepest, curv, scale)
    sel, intra, inter = stacked_entailment_loss(img, text_embs, taxonomy_batch, ranks, curv)
    loss = cl + lambda_sel * sel

    loss.backward()
    optimizer.step()
    model.clamp_params()

    return {
        "loss": loss.item(),
        "cl": cl.item(),
        "sel": sel.item(),
        "sel_intra": intra.item(),
        "sel_inter": inter.item(),
    }
