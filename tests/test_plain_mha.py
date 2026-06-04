"""Verification for src/hyperbolic_plankton/plain_mha.py (piece 7a).

The swap must be NUMERICALLY IDENTICAL to open_clip's original attention on the frozen
weights — otherwise we corrupt the pretrained backbone before LoRA even starts. This is
the gate everything in the LoRA tier depends on. We check:

  1. A single PlainMHA matches nn.MultiheadAttention bit-for-bit (no mask + causal mask).
  2. After replacing all MHAs in a real open_clip model, encode_image/encode_text are
     unchanged (the causal text mask is the risky part).
"""

import open_clip
import pytest
import torch

from hyperbolic_plankton.plain_mha import PlainMultiHeadAttention, replace_mha_with_plain

torch.manual_seed(0)


def test_plain_matches_nn_mha_no_mask():
    mha = torch.nn.MultiheadAttention(64, 8, batch_first=False).eval()
    plain = PlainMultiHeadAttention(64, 8).eval()
    plain.set_parameters(mha)
    x = torch.randn(10, 4, 64)  # (seq, batch, embed)
    with torch.no_grad():
        ref, _ = mha(x, x, x, need_weights=False)
        got, _ = plain(x, x, x, need_weights=False)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_plain_matches_nn_mha_causal_mask():
    """open_clip's text tower uses a causal float mask — the risky path."""
    mha = torch.nn.MultiheadAttention(64, 8, batch_first=False).eval()
    plain = PlainMultiHeadAttention(64, 8).eval()
    plain.set_parameters(mha)
    L = 10
    mask = torch.empty(L, L).fill_(float("-inf")).triu_(1)  # causal
    x = torch.randn(L, 4, 64)
    with torch.no_grad():
        ref, _ = mha(x, x, x, need_weights=False, attn_mask=mask)
        got, _ = plain(x, x, x, need_weights=False, attn_mask=mask)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


@pytest.fixture(scope="module")
def clip_model():
    m, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    return m.eval()


def test_full_model_embeddings_unchanged_after_swap(clip_model):
    """Replacing all MHAs must not change the model's image/text outputs."""
    import copy

    model = copy.deepcopy(clip_model)
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    img = torch.randn(2, 3, 224, 224)
    txt = tok(["a copepod", "a diatom"])

    with torch.no_grad():
        img_before = model.encode_image(img, normalize=False)
        txt_before = model.encode_text(txt, normalize=False)

    n = replace_mha_with_plain(model.visual)
    n += replace_mha_with_plain(model.transformer)  # text transformer
    assert n == 24, f"expected 12 visual + 12 text blocks, replaced {n}"

    with torch.no_grad():
        img_after = model.encode_image(img, normalize=False)
        txt_after = model.encode_text(txt, normalize=False)

    assert torch.allclose(img_before, img_after, atol=1e-4), (img_before - img_after).abs().max()
    assert torch.allclose(txt_before, txt_after, atol=1e-4), (txt_before - txt_after).abs().max()


def test_q_proj_addressable_after_swap(clip_model):
    """After swap, the split q/k/v/o linears exist by name (so PEFT can target them)."""
    import copy

    model = copy.deepcopy(clip_model)
    replace_mha_with_plain(model.visual)
    names = [n for n, _ in model.visual.named_modules()]
    assert any(n.endswith("attn.q_proj") for n in names)
    assert any(n.endswith("attn.k_proj") for n in names)
    assert any(n.endswith("attn.v_proj") for n in names)
    assert any(n.endswith("attn.proj") for n in names)
