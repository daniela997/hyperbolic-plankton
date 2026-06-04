"""Plain (split-QKV) multi-head attention to make open_clip LoRA-targetable.

Piece 7a. open_clip's attention is `torch.nn.MultiheadAttention`, whose Q/K/V live in a
single fused `in_proj_weight`. PEFT's LoRA cannot target a fused projection, so we swap
each `nn.MultiheadAttention` for an equivalent module with separate `q_proj`/`k_proj`/
`v_proj`/`proj` linears that PEFT *can* address.

Ported (narrowed) from HAC's `PlainMultiHeadAttention`
(`/home/daniela/other/HAC/hac/utils/plain_mha.py`). We keep only the `nn.MultiheadAttention`
path (HAC also handles timm `Attention`, which open_clip does not use) and the attn-mask
handling open_clip's *causal* text tower needs. The swap copies weights exactly, so the
replaced attention must be numerically identical to the original — verified in tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PlainMultiHeadAttention", "replace_mha_with_plain"]


class PlainMultiHeadAttention(nn.Module):
    """Drop-in replacement for `nn.MultiheadAttention` with split q/k/v/o linears.

    Supports the self-attention call open_clip makes: `attn(x, x, x, need_weights=False,
    attn_mask=mask)` with `batch_first=False` (seq, batch, embed) — open_clip's default.
    Returns `(output, None)` to match the `nn.MultiheadAttention` signature.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, batch_first: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @torch.no_grad()
    def set_parameters(self, mha: nn.MultiheadAttention) -> None:
        """Copy weights from an `nn.MultiheadAttention` (fused qkv -> split)."""
        assert isinstance(mha, nn.MultiheadAttention)
        assert mha.embed_dim == self.embed_dim and mha.num_heads == self.num_heads
        qw, kw, vw = mha.in_proj_weight.chunk(3, dim=0)
        self.q_proj.weight.copy_(qw)
        self.k_proj.weight.copy_(kw)
        self.v_proj.weight.copy_(vw)
        if mha.in_proj_bias is not None:
            qb, kb, vb = mha.in_proj_bias.chunk(3, dim=0)
            self.q_proj.bias.copy_(qb)
            self.k_proj.bias.copy_(kb)
            self.v_proj.bias.copy_(vb)
        self.proj.weight.copy_(mha.out_proj.weight)
        if mha.out_proj.bias is not None:
            self.proj.bias.copy_(mha.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        **kwargs,  # absorb key_padding_mask/is_causal/average_attn_weights for API compat
    ):
        if self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        L, B, _ = query.shape
        S = key.shape[0]

        def shape(proj_out, n):  # (n, B, E) -> (B, heads, n, head_dim)
            return proj_out.view(n, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        q = shape(self.q_proj(query), L)
        k = shape(self.k_proj(key), S)
        v = shape(self.v_proj(value), S)

        # attn_mask from open_clip is a float/bool (L, S) bias added to the scores;
        # SDPA broadcasts a (L, S) mask over (B, heads, L, S).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.permute(2, 0, 1, 3).reshape(L, B, self.embed_dim)  # (L, B, E)
        out = self.proj(out)

        if self.batch_first:
            out = out.transpose(0, 1)
        return out, None


def replace_mha_with_plain(model: nn.Module) -> int:
    """In-place: replace every `nn.MultiheadAttention` with a weight-copied PlainMHA.

    Returns the number of modules replaced.
    """
    replaced = 0
    modules = dict(model.named_modules())
    for name, module in list(modules.items()):
        if isinstance(module, nn.MultiheadAttention):
            plain = PlainMultiHeadAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
            ).to(module.in_proj_weight.device, module.in_proj_weight.dtype)
            plain.set_parameters(module)
            parent = modules[name.rsplit(".", 1)[0]] if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], plain)
            replaced += 1
    assert replaced > 0, "no nn.MultiheadAttention found to replace"
    return replaced
