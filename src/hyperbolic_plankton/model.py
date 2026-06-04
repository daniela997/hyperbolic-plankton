"""Hyperbolic CLIP model — frozen open_clip backbone + projection + exp_map0 lift.

Piece 2 of the methodical re-implementation. Geometry follows HAC's `AdaptedCLIP`
(`/home/daniela/other/HAC/hac/models.py`) and our verified `lorentz.py`; the per-rank
`encode_taxonomy` (dict of rank -> hyperbolic embeddings + validity masks) follows the
scratchpad `mine/hyperbolic/model.py`, since the SEL loss (Piece 4) consumes it.

Scope (v1, projector-only): frozen backbone, a learnable projection head per modality,
learnable MERU scalars (curv, alpha, logit_scale), and the exp_map0 lift. No LoRA
(Piece 7), no loss inside the model (Piece 4), no parallel-transport/depth variants.
"""

from __future__ import annotations

import math

import open_clip
import torch
import torch.nn as nn

from . import lorentz as L

__all__ = ["build_backbone", "HyperbolicCLIP"]


# open_clip identifiers for the two inits we use. OpenAI weights expect QuickGELU, so
# we load the `-quickgelu` arch variant to avoid the activation mismatch warning.
_BACKBONES = {
    "clip": dict(model_name="ViT-B-16-quickgelu", pretrained="openai"),
    "bioclip": dict(model_name="hf-hub:imageomics/bioclip", pretrained=None),
}


def build_backbone(name: str):
    """Load a frozen open_clip backbone.

    Returns `(model, embed_dim)` where `model` is a frozen open_clip `CLIP` and
    `embed_dim` is its shared image/text output dimension.
    """
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Options: {list(_BACKBONES)}")
    cfg = _BACKBONES[name]
    model, _, _ = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=cfg["pretrained"]
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model.visual.output_dim


class HyperbolicCLIP(nn.Module):
    """Frozen CLIP backbone with a learnable projection into the Lorentz hyperboloid.

    encode_image / encode_text return space components on the hyperboloid (when
    `project=True`) or the Euclidean tangent features (when `project=False`).
    encode_taxonomy returns a dict {rank: [B, embed_dim]} plus {rank}_valid masks,
    encoding only the non-None entries per rank.
    """

    def __init__(
        self,
        backbone: str = "clip",
        embed_dim: int | None = None,
        curv_init: float = 1.0,
        learn_curv: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.clip, backbone_dim = build_backbone(backbone)
        self.tokenizer = open_clip.get_tokenizer(_BACKBONES[backbone]["model_name"])

        # When False (projector-only) the backbone forward runs under no_grad to save
        # memory — correct because every backbone param is frozen. When LoRA is applied
        # (apply_lora sets this True), the graph must stay intact so gradients reach the
        # adapters; freezing is then enforced by requires_grad alone (as HAC does).
        self.backbone_trainable = False

        self.embed_dim = embed_dim or backbone_dim

        # Learnable projection heads (trained from scratch; backbone is frozen).
        # CLIP-style init, matching HAC's AdaptedCLIP.
        self.visual_proj = nn.Linear(backbone_dim, self.embed_dim, bias=False)
        self.textual_proj = nn.Linear(backbone_dim, self.embed_dim, bias=False)
        nn.init.normal_(self.visual_proj.weight, std=backbone_dim**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=backbone_dim**-0.5)

        # MERU scalars.
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }
        self.visual_alpha = nn.Parameter(torch.tensor(self.embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(self.embed_dim**-0.5).log())
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    @property
    def curvature(self) -> torch.Tensor:
        return self.curv.exp()

    def clamp_params(self) -> None:
        """Clamp learnable scalars to their valid ranges (call once per train step)."""
        if self.curv.requires_grad:
            self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=math.log(100))

    def _lift(self, feats: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Scale by alpha and exp-map onto the hyperboloid.

        Force fp32 for the exp map under CUDA autocast (numerical stability); on CPU
        autocast-to-fp32 is unsupported and a no-op, so we skip it there.
        """
        feats = feats * alpha.exp()
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float32):
                return L.exp_map0(feats, self.curvature)
        return L.exp_map0(feats, self.curvature)

    def encode_image(self, pixel_values: torch.Tensor, project: bool = True) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_trainable):
            feats = self.clip.encode_image(pixel_values, normalize=False)
        feats = self.visual_proj(feats)
        return self._lift(feats, self.visual_alpha) if project else feats

    def encode_text(self, texts: list[str], project: bool = True) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.set_grad_enabled(self.backbone_trainable):
            feats = self.clip.encode_text(tokens, normalize=False)
        feats = self.textual_proj(feats)
        return self._lift(feats, self.textual_alpha) if project else feats

    def encode_taxonomy(
        self, taxonomy_batch: dict[str, list[str | None]], project: bool = True
    ) -> dict[str, torch.Tensor]:
        """Encode per-rank taxonomy text into hyperbolic embeddings with validity masks.

        Args:
            taxonomy_batch: {rank: [B] list of strings or None}. Keys starting with
                "_" are skipped (collator metadata).
        Returns:
            {rank: [B, embed_dim]} with zeros for invalid (None) entries, plus
            {f"{rank}_valid": [B] bool} marking which entries were encoded.
        """
        out: dict[str, torch.Tensor] = {}
        for rank, texts in taxonomy_batch.items():
            if rank.startswith("_"):
                continue
            valid = torch.tensor([t is not None for t in texts], dtype=torch.bool, device=self.device)
            emb = torch.zeros(len(texts), self.embed_dim, device=self.device)
            if valid.any():
                idx = valid.nonzero(as_tuple=True)[0]
                sub = self.encode_text([texts[i] for i in idx.tolist()], project=project)
                emb[idx] = sub.to(emb.dtype)
            out[rank] = emb
            out[f"{rank}_valid"] = valid
        return out
