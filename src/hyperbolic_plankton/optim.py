"""Optimizer + LR schedule (HAC recipe).

Re-implemented from HAC `hac/optim.py` (configs/train_hac_vit_b_lora.py):
  - `param_groups`: AdamW weight-decay grouping — wd=0 for LayerNorm/bias and for the
    MERU scalars + LoRA params, wd=`weight_decay` for everything else trainable.
  - `LinearWarmupCosineDecayLR`: linear warmup then cos^2 annealing to 0 (HAC's exact
    multiplier `cos(x * pi/2) ** 2`, NOT the standard 0.5*(1+cos) cosine).
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

# names whose params get NO weight decay (MERU scalars + LoRA), per HAC's exclude_params.
_NO_DECAY_NAMES = ("logit_scale", "visual_alpha", "textual_alpha", "curv", "lora_")
_NORM_CLASSES = (
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """AdamW param groups: `excluded`/`gain_bias` get wd=0, `regular` gets `weight_decay`.

    Mirrors HAC `set_weight_decay_per_param` with `gain_bias_decay=0.0` and
    `exclude_params=[scalars..., "lora_"]`. Only `requires_grad` params are included.
    """
    decay, no_decay = [], []
    seen = set()

    def visit(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in seen:
                continue
            seen.add(p)
            full = f"{prefix}.{name}" if prefix else name
            if any(s in full for s in _NO_DECAY_NAMES):
                no_decay.append(p)               # excluded scalars + LoRA
            elif isinstance(module, _NORM_CLASSES) or "bias" in name:
                no_decay.append(p)               # gain_bias (decay 0.0)
            else:
                decay.append(p)                  # regular
        for cname, child in module.named_children():
            visit(child, f"{prefix}.{cname}" if prefix else cname)

    visit(model)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "name": "regular"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "no_decay"})
    return groups


class LinearWarmupCosineDecayLR(LambdaLR):
    """Linear warmup to step `warmup_steps`, then `cos(x * pi/2) ** 2` decay to 0 by
    `total_steps` (HAC's exact schedule)."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int, last_epoch: int = -1):
        assert warmup_steps < total_steps
        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._mult, last_epoch)

    def _mult(self, step: int) -> float:
        if step < self.wsteps:
            return step / float(max(1, self.wsteps))
        cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
        return max(0.0, math.cos(cos_factor * (math.pi / 2)) ** 2)
