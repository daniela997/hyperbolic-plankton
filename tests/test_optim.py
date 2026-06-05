"""Verification for src/hyperbolic_plankton/optim.py (HAC recipe).

  1. param_groups: LoRA + MERU scalars + LN/bias land in the wd=0 group; projection
     head weights land in the decay group. No param appears twice.
  2. LinearWarmupCosineDecayLR: exact multiplier values at key steps (linear warmup,
     cos^2 endpoints + midpoint) vs HAC's formula.
"""

import math

import torch

from hyperbolic_plankton.lora import apply_lora
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.optim import LinearWarmupCosineDecayLR, param_groups


def test_param_groups_split():
    m = apply_lora(HyperbolicCLIP(backbone="clip").eval(), r=8, alpha=8)
    groups = param_groups(m, weight_decay=0.2)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"regular", "no_decay"}
    assert by_name["regular"]["weight_decay"] == 0.2
    assert by_name["no_decay"]["weight_decay"] == 0.0

    # no param double-counted
    all_params = [p for g in groups for p in g["params"]]
    assert len(all_params) == len({id(p) for p in all_params})

    # every trainable param is covered
    n_trainable = sum(p.requires_grad for p in m.parameters())
    assert len(all_params) == n_trainable

    # MERU scalars + a LoRA param are in no_decay; projection head weight is in regular.
    nd = {id(p) for p in by_name["no_decay"]["params"]}
    reg = {id(p) for p in by_name["regular"]["params"]}
    assert id(m.curv) in nd and id(m.logit_scale) in nd
    assert id(m.visual_proj.weight) in reg
    lora_b = [p for n, p in m.named_parameters() if "lora_B" in n][0]
    assert id(lora_b) in nd


def test_scheduler_exact_values():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = LinearWarmupCosineDecayLR(opt, total_steps=100, warmup_steps=10)

    # step 0 -> multiplier 0 (linear warmup from 0)
    assert abs(opt.param_groups[0]["lr"] - 0.0) < 1e-9
    # advance to step 5 (mid-warmup) -> 5/10 = 0.5
    for _ in range(5):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-9
    # step 10 (warmup end) -> cos(0)^2 = 1.0
    for _ in range(5):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-9
    # step 55 (halfway through decay) -> cos(0.5 * pi/2)^2
    for _ in range(45):
        sched.step()
    expected = math.cos(0.5 * math.pi / 2) ** 2
    assert abs(opt.param_groups[0]["lr"] - expected) < 1e-9
    # step 100 (end) -> cos(pi/2)^2 = 0
    for _ in range(45):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 0.0) < 1e-9
