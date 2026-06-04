"""Verification for src/hyperbolic_plankton/lorentz.py (piece 1).

Two kinds of checks:
  1. Mathematical PROPERTIES (round-trips, points-on-manifold, known values).
  2. Numerical CROSS-CHECK against the two reference implementations:
       - HAC/MERU:  /home/daniela/other/HAC/hac/lorentz.py
       - scratchpad: /home/daniela/mine/hyperbolic/hyperbolic.py (LorentzMath)

The cross-check imports the references by file path so we depend on neither repo's
package layout. If a reference is unavailable the cross-check is skipped (properties
still run), so the suite is informative even in isolation.
"""

import importlib.util
import math
import os

import pytest
import torch

from hyperbolic_plankton import lorentz as Lp

torch.manual_seed(0)

CURVS = [0.5, 1.0, 2.0]
HAC_PATH = "/home/daniela/other/HAC/hac/lorentz.py"
SCRATCH_PATH = "/home/daniela/mine/hyperbolic/hyperbolic.py"


def _ensure_optional_stubs():
    """Stub modules the reference files import only for logging, so they execute
    in the lean test env (HAC's lorentz.py imports loguru only for oxy_angle_eval)."""
    import sys
    import types

    if "loguru" not in sys.modules:
        stub = types.ModuleType("loguru")
        stub.logger = types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )
        sys.modules["loguru"] = stub


def _load_module(path, name):
    import sys

    _ensure_optional_stubs()
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so dataclasses/refs resolve
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — diagnostic, reference is optional
        print(f"[cross-check] could not load {name}: {type(e).__name__}: {e}")
        sys.modules.pop(name, None)
        return None
    return mod


HAC = _load_module(HAC_PATH, "hac_lorentz_ref")
_scratch = _load_module(SCRATCH_PATH, "scratch_hyperbolic_ref")
SCRATCH = getattr(_scratch, "LorentzMath", None) if _scratch else None


def _rand(b=8, d=16, scale=1.0):
    return torch.randn(b, d, dtype=torch.float64) * scale


# --------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("curv", CURVS)
def test_exp_log_roundtrip(curv):
    """log_map0(exp_map0(x)) == x for tangent vectors x."""
    x = _rand(scale=0.5)
    back = Lp.log_map0(Lp.exp_map0(x, curv), curv)
    assert torch.allclose(back, x, atol=1e-6), (back - x).abs().max()


@pytest.mark.parametrize("curv", CURVS)
def test_points_lie_on_hyperboloid(curv):
    """After exp_map0, <p, p>_L == -1/curv (space^2 - time^2)."""
    p = Lp.exp_map0(_rand(scale=0.7), curv)
    p_time = Lp.time_component(p, curv).squeeze(-1)
    inner = (p**2).sum(-1) - p_time**2
    assert torch.allclose(inner, torch.full_like(inner, -1.0 / curv), atol=1e-6)


@pytest.mark.parametrize("curv", CURVS)
def test_self_distance_near_zero(curv):
    """d(p, p) is ~0 up to the arccosh stability floor.

    pairwise_dist clamps its arccosh input to `1 + eps`, so the smallest representable
    non-zero distance is arccosh(1+eps)/sqrt(curv) ~= sqrt(2*eps/curv). This is the
    MERU/HAC reference's intended numerical floor, NOT a bug — assert against it.
    """
    eps = 1e-8
    p = Lp.exp_map0(_rand(scale=0.7), curv)
    d = Lp.pairwise_dist(p, p, curv, eps=eps).diag()
    floor = math.sqrt(2 * eps / curv)
    assert (d <= floor * 1.5).all(), (d.max(), floor)


@pytest.mark.parametrize("curv", CURVS)
def test_distance_from_origin_matches_pairwise(curv):
    """distance_from_origin(p) == pairwise_dist(origin, p)."""
    p = Lp.exp_map0(_rand(scale=0.7), curv)
    origin = torch.zeros(1, p.shape[-1], dtype=p.dtype)
    d_pair = Lp.pairwise_dist(origin, p, curv).squeeze(0)
    d_orig = Lp.distance_from_origin(p, curv)
    assert torch.allclose(d_pair, d_orig, atol=1e-5), (d_pair - d_orig).abs().max()


def test_half_aperture_in_range():
    p = Lp.exp_map0(_rand(scale=1.0), 1.0)
    ha = Lp.half_aperture(p, 1.0)
    assert (ha > 0).all() and (ha < math.pi / 2 + 1e-6).all()


def test_oxy_angle_in_range():
    x = Lp.exp_map0(_rand(scale=1.0), 1.0)
    y = Lp.exp_map0(_rand(scale=1.0), 1.0)
    ang = Lp.oxy_angle(x, y, 1.0)
    assert (ang >= 0).all() and (ang <= math.pi + 1e-6).all()


def test_origin_entails_everything():
    """A point near the origin (wide cone) should entail a far point."""
    near = Lp.exp_map0(_rand(scale=0.05), 1.0)
    far = Lp.exp_map0(_rand(scale=2.0), 1.0)
    # not a guarantee for every pair, but aperture(near) should be large (~pi/2)
    assert Lp.half_aperture(near, 1.0).mean() > Lp.half_aperture(far, 1.0).mean()


# --------------------------------------------------------------------------------
# Cross-check vs HAC reference (eps matched to 1e-8)
# --------------------------------------------------------------------------------

@pytest.mark.skipif(HAC is None, reason="HAC reference not importable")
@pytest.mark.parametrize("curv", CURVS)
def test_crosscheck_hac(curv):
    x = _rand(scale=0.7)
    y = _rand(scale=0.7)
    px, py = Lp.exp_map0(x, curv), Lp.exp_map0(y, curv)

    pairs = [
        (Lp.exp_map0(x, curv), HAC.exp_map0(x, curv)),
        (Lp.log_map0(px, curv), HAC.log_map0(px, curv)),
        (Lp.pairwise_inner(px, py, curv), HAC.pairwise_inner(px, py, curv)),
        (Lp.pairwise_dist(px, py, curv), HAC.pairwise_dist(px, py, curv)),
        (Lp.half_aperture(px, curv), HAC.half_aperture(px, curv)),
        (Lp.oxy_angle(px, py, curv), HAC.oxy_angle(px, py, curv)),
    ]
    for ours, ref in pairs:
        assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().max()


# --------------------------------------------------------------------------------
# Cross-check vs scratchpad (expected to differ slightly: it defaults eps=1e-4).
# We pass eps=1e-8 explicitly so a MATCH confirms the math is identical and the only
# difference is the eps default.
# --------------------------------------------------------------------------------

@pytest.mark.skipif(SCRATCH is None, reason="scratchpad reference not importable")
@pytest.mark.parametrize("curv", CURVS)
def test_crosscheck_scratch_same_eps(curv):
    x = _rand(scale=0.7)
    y = _rand(scale=0.7)
    px, py = Lp.exp_map0(x, curv), Lp.exp_map0(y, curv)
    e = 1e-8
    pairs = [
        (Lp.exp_map0(x, curv, eps=e), SCRATCH.exp_map0(x, curv, eps=e)),
        (Lp.pairwise_dist(px, py, curv, eps=e), SCRATCH.pairwise_dist(px, py, curv, eps=e)),
        (Lp.half_aperture(px, curv, eps=e), SCRATCH.half_aperture(px, curv, eps=e)),
        (Lp.oxy_angle(px, py, curv, eps=e), SCRATCH.oxy_angle(px, py, curv, eps=e)),
    ]
    for ours, ref in pairs:
        assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().max()
