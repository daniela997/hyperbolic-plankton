"""LoRA adaptation of the open_clip backbone (Piece 7b).

Applies the HAC recipe to a `HyperbolicCLIP`: swap fused attention for split q/k/v/o
linears (`plain_mha`), wrap them with LoRA on the last few blocks (text-heavier), and
unfreeze the final LayerNorm of each tower. The projection heads + MERU scalars are
already trainable (they live on the model, outside the frozen backbone).

Recipe source: HAC `configs/train_hac_vit_b_lora.py` + `scripts/train.py` (see
`docs/related-work.md` / `docs/hac-implementation.md`). Targets q,k,v,o (HAC ablation:
dropping `o` hurts); last 4 visual / last 8 text blocks; r=alpha (start small on 24GB).
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .plain_mha import replace_mha_with_plain

__all__ = ["apply_lora", "unfreeze_backbone", "count_trainable"]

_ATTN_SUBMODULES = ("q_proj", "k_proj", "v_proj", "proj")


def _target_regex(n_vis: int, vis_last: int, n_txt: int, txt_last: int) -> str:
    """A single anchored regex matching the attn linears in the last-N blocks of each
    tower, against the FULL module path.

    Why a regex (not a name list): PEFT matches a `target_modules` *list* by name
    SUFFIX, and the text tower's `transformer.resblocks.{i}.attn.q_proj` is a suffix of
    the visual tower's `...visual.transformer.resblocks.{i}.attn.q_proj` — so a list
    would wrongly also adapt visual blocks. A `str` target is matched with `re.fullmatch`
    against the full name, so anchoring `visual.transformer` vs a leading text path
    disambiguates the two towers.
    """
    subs = "|".join(_ATTN_SUBMODULES)
    vis_blocks = "|".join(str(i) for i in range(max(0, n_vis - vis_last), n_vis))
    txt_blocks = "|".join(str(i) for i in range(max(0, n_txt - txt_last), n_txt))
    # PEFT matches `re.fullmatch` against pre-wrap names:
    #   visual:  visual.transformer.resblocks.{i}.attn.{sub}
    #   text:    transformer.resblocks.{i}.attn.{sub}   (no 'visual.' segment)
    vis = rf"visual\.transformer\.resblocks\.({vis_blocks})\.attn\.({subs})"
    txt = rf"transformer\.resblocks\.({txt_blocks})\.attn\.({subs})"
    return rf"(?:{vis})|(?:{txt})"


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 8,
    dropout: float = 0.1,  # HAC lora_dropout
    adapt_visual_blocks: int = 4,
    adapt_text_blocks: int = 8,
    reinit_final_ln: bool = True,
):
    """Swap MHA → split linears, wrap last-N blocks with LoRA, train the final LN.

    Mutates `model.clip` in place. `model` is a `HyperbolicCLIP` whose backbone is
    already frozen. Returns `model`.

    `reinit_final_ln` (HAC `init_final_ln`, default True): reset each tower's final
    LayerNorm to γ=1, β=0 before training it. HAC re-initializes (not just unfreezes) the
    final LN when transitioning CLIP into the new hyperbolic output space — the old LN was
    calibrated for CLIP's output distribution, which no longer applies after the projection
    change. Set False to keep CLIP's pretrained LN params and only unfreeze them.
    """
    clip = model.clip
    replace_mha_with_plain(clip.visual)
    replace_mha_with_plain(clip.transformer)

    n_vis = len(clip.visual.transformer.resblocks)
    n_txt = len(clip.transformer.resblocks)
    targets = _target_regex(n_vis, adapt_visual_blocks, n_txt, adapt_text_blocks)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=targets,  # regex, fullmatch against pre-wrap module names
        lora_dropout=dropout,
        bias="none",
        use_rslora=True,  # rank-stabilized, as in HAC
    )
    # get_peft_model freezes everything it doesn't adapt and marks LoRA params trainable.
    model.clip = get_peft_model(clip, cfg)

    # Final LayerNorm of each tower (HAC §4.4: must train or it gates the output). HAC also
    # re-initializes it (init_final_ln) to fit the new hyperbolic output space.
    for tower in ("visual", "text"):
        ln = _final_ln(model.clip, tower)
        if reinit_final_ln:
            ln.reset_parameters()  # γ=1, β=0 — fresh LN, as HAC does
        for p in ln.parameters():
            p.requires_grad = True

    # The backbone forward must now build a graph so gradients reach the LoRA adapters;
    # freezing is enforced by requires_grad alone (HAC relies on this, not no_grad).
    model.backbone_trainable = True

    return model


def unfreeze_backbone(model) -> None:
    """Make the whole CLIP backbone trainable — the FULL fine-tune setting (Planktonzilla
    recipe). The inverse of the frozen default: every backbone param gets `requires_grad`
    and the forward builds a graph. Use INSTEAD of `apply_lora` for the full-FT baseline.

    (`--no-lora` alone leaves the backbone FROZEN = projector-only; this is the actual
    full fine-tune.)
    """
    for p in model.clip.parameters():
        p.requires_grad = True
    model.backbone_trainable = True


def _final_ln(peft_clip, tower: str) -> nn.Module:
    """Locate the final LayerNorm of the visual (`ln_post`) or text (`ln_final`) tower,
    through PEFT's `base_model.model` wrapper."""
    base = peft_clip.base_model.model
    if tower == "visual":
        return base.visual.ln_post
    return base.ln_final


def count_trainable(model) -> dict[str, int]:
    """Return {trainable, total} parameter counts for sanity/logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total}
