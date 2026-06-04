"""Verification for src/hyperbolic_plankton/lora.py (piece 7b).

Checks:
  1. apply_lora runs; trainable << total.
  2. The trainable set is EXACTLY {lora params, final LNs, projection heads, MERU
     scalars} — no frozen backbone base weight is trainable.
  3. Only the targeted (last-N) blocks get LoRA adapters.
  4. backward gives gradient to LoRA params and to projection heads; none to a frozen
     base weight (e.g. a patch-embed conv).
"""

import pytest
import torch

from hyperbolic_plankton.lora import apply_lora, count_trainable
from hyperbolic_plankton.model import HyperbolicCLIP


@pytest.fixture(scope="module")
def lora_model():
    m = HyperbolicCLIP(backbone="clip").eval()
    return apply_lora(m, r=8, alpha=8, adapt_visual_blocks=4, adapt_text_blocks=8)


def test_trainable_is_small(lora_model):
    c = count_trainable(lora_model)
    frac = c["trainable"] / c["total"]
    assert 0 < frac < 0.1, c  # parameter-efficient: well under 10%


def test_only_expected_params_trainable(lora_model):
    """Every trainable param must be a LoRA param, a final LN, a projection head, or a
    MERU scalar — never a frozen backbone base weight."""
    allowed_substrings = (
        "lora_",            # LoRA A/B
        "ln_post",          # visual final LN
        "ln_final",         # text final LN
        "visual_proj",      # our projection heads
        "textual_proj",
    )
    scalar_names = {"curv", "visual_alpha", "textual_alpha", "logit_scale"}
    for name, p in lora_model.named_parameters():
        if not p.requires_grad:
            continue
        ok = any(s in name for s in allowed_substrings) or name.split(".")[-1] in scalar_names
        assert ok, f"unexpected trainable param: {name}"


def test_only_targeted_blocks_have_lora(lora_model):
    """Last 4 visual blocks (8-11) and last 8 text blocks (4-11) get LoRA; earlier
    blocks do not."""
    lora_a = [n for n, _ in lora_model.named_parameters() if "lora_A" in n]
    # visual: blocks 8-11 present, block 7 absent
    assert any("visual.transformer.resblocks.11.attn.q_proj" in n for n in lora_a)
    assert any("visual.transformer.resblocks.8.attn" in n for n in lora_a)
    assert not any("visual.transformer.resblocks.7.attn" in n for n in lora_a)
    # text: blocks 4-11 present, block 3 absent (note: text path has no 'visual.' prefix)
    text_a = [n for n in lora_a if "visual" not in n]
    assert any("resblocks.11.attn.q_proj" in n for n in text_a)
    assert any("resblocks.4.attn" in n for n in text_a)
    assert not any("resblocks.3.attn" in n for n in text_a)


def test_backward_grad_routing():
    """Fresh model (not the shared fixture) so grad state is clean."""
    model = apply_lora(HyperbolicCLIP(backbone="clip").eval(), r=8, alpha=8)
    model.zero_grad(set_to_none=True)
    pix = torch.randn(2, 3, 224, 224)
    loss = model.encode_image(pix).sum() + model.encode_text(["a", "b"]).sum()
    loss.backward()

    # A LoRA param gets grad. NOTE: lora_B is zero-initialized (standard LoRA), so on the
    # FIRST backward grad(lora_A) = lora_B^T @ grad_out = 0 while grad(lora_B) != 0. We
    # therefore check lora_B (the param that proves the adapter is in the live graph). If
    # this is 0, the backbone no_grad severed the graph (the bug this piece fixed).
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b)
    # projection head gets grad
    assert model.visual_proj.weight.grad is not None
    # a frozen base weight (patch-embed conv) is frozen and gets no grad
    conv = model.clip.base_model.model.visual.conv1.weight
    assert not conv.requires_grad
    assert conv.grad is None
