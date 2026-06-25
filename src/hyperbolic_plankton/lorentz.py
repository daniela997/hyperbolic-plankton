"""Lorentz model of hyperbolic geometry — re-implemented from the MERU/HAC spec.

This is piece 1 of the methodical re-implementation. Every function is derived from
the canonical MERU reference (`/home/daniela/other/HAC/hac/lorentz.py`, itself from
facebookresearch/meru) and cross-checked numerically against both that reference and
the existing scratchpad (`/home/daniela/mine/hyperbolic/hyperbolic.py`) in the tests.

Convention (MERU): the Lorentz model represents `d`-dim hyperbolic space on the upper
sheet of a two-sheeted hyperboloid in `(d+1)`-dim Minkowski space. We store only the
`d` **space** components; the **time** component is reconstructed from the hyperboloid
constraint

    x_time = sqrt(1/curv + ||x_space||^2)

so that the Lorentzian inner product <x, x>_L = ||x_space||^2 - x_time^2 = -1/curv.
`curv` is the positive curvature magnitude (hyperboloid curvature is -curv).

Numerical note: we follow MERU/HAC and use `eps=1e-8` (the scratchpad used 1e-4, which
introduces larger error near the origin — see tests/test_lorentz.py for the comparison).
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor

__all__ = [
    "time_component",
    "pairwise_inner",
    "pairwise_dist",
    "exp_map0",
    "log_map0",
    "half_aperture",
    "oxy_angle",
    "distance_from_origin",
    "einstein_midpoint",
]


def _fp32(fn):
    """Run a geometry primitive in fp32, disabling autocast. The acosh/sinh/asinh ops here
    are numerically fragile: under bf16 (7-bit mantissa) a drifting curvature + growing
    embeddings push acosh args into a regime that cascades to NaN (observed: curv -> NaN at
    ~step 500 of a bf16 hyperbolic run). fp32 has the mantissa to stay stable; the cost is
    tiny (these are a small fraction of FLOPs vs the backbone matmuls, which stay bf16).
    No-op on CPU (autocast-to-fp32 unsupported there)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        on_cuda = any(isinstance(a, Tensor) and a.is_cuda for a in args)
        if on_cuda:
            with torch.autocast("cuda", enabled=False):
                args = tuple(a.float() if isinstance(a, Tensor) else a for a in args)
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


@_fp32
def time_component(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Reconstruct the Lorentz time component from space components.

    x_time = sqrt(1/curv + ||x||^2). Returns shape `(..., 1)` (keepdim).
    """
    return torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))


@_fp32
def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Pairwise Lorentzian inner product <x_i, y_j>_L.

    Args:
        x: `(B1, D)` space components.
        y: `(B2, D)` space components.
    Returns:
        `(B1, B2)` matrix of <x_i, y_j>_L = x_i . y_j - x_time_i * y_time_j.
    """
    x_time = time_component(x, curv)  # (B1, 1)
    y_time = time_component(y, curv)  # (B2, 1)
    return x @ y.T - x_time @ y_time.T


@_fp32
def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Pairwise geodesic distance on the hyperboloid.

    d_L(x, y) = (1/sqrt(curv)) * arccosh(-curv * <x, y>_L).
    """
    c_xyl = -curv * pairwise_inner(x, y, curv)
    distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return distance / curv**0.5


@_fp32
def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """Exponential map at the origin: tangent vector -> point on hyperboloid.

    Maps a Euclidean vector `x` (interpreted as a velocity in the tangent space at the
    hyperboloid vertex) onto the manifold, returning only the space components:

        x_space_out = sinh(sqrt(curv) ||x||) * x / (sqrt(curv) ||x||).
    """
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    # Clamp sinh input for stability; max keeps sinh finite in fp32.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)


@_fp32
def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """Logarithmic map at the origin: point on hyperboloid -> tangent vector.

    Inverse of `exp_map0`:

        v = arccosh(sqrt(1 + curv ||x||^2)) * x / (sqrt(curv) ||x||).
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    return distance0 * x / torch.clamp(rc_xnorm, min=eps)


@_fp32
def distance_from_origin(
    x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Geodesic distance (radius) from the hyperboloid vertex.

    d = (1/sqrt(curv)) * arccosh(sqrt(1 + curv ||x||^2)). Returns shape `(...,)`.
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1))
    distance = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    return distance / curv**0.5


@_fp32
def einstein_midpoint(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """Einstein (Klein) midpoint of a set of hyperboloid points -> space components `(D,)`.

    The proper hyperbolic centroid: project to the Klein model (`v = x / x_time`), take the
    Lorentz-factor-weighted mean of the Klein coordinates (the Lorentz factor in this hyperboloid
    convention is the time component `x_time = sqrt(1/curv + ||x||^2)`), then map back. `x` is
    `(N, D)` space components; returns the midpoint's space components, whose `distance_from_origin`
    is the centroid radius used by the radial-ordering loss.

    (ATMG `models.py::einstein_loss` has bugs here — it weights raw features not Klein coords and
    uses ||x||^4 in the factor; it is also disabled in their forward. We implement the correct form.)
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # Lorentz factor γ
    klein = x / x_time                                     # Klein coordinates
    klein_avg = (klein * x_time).sum(0) / x_time.sum(0)    # γ-weighted Klein mean -> [D]
    # map Klein point back to hyperboloid space components: x = v / sqrt(1 - curv*||v||^2)
    kn2 = torch.sum(klein_avg**2)
    return klein_avg / torch.sqrt(torch.clamp(1 - curv * kn2, min=1e-8))


@_fp32
def half_aperture(
    x: Tensor,
    curv: float | Tensor = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-8,
) -> Tensor:
    """Half-aperture of the entailment cone with apex at `x`.

    psi(x) = arcsin( 2 * min_radius / (sqrt(curv) ||x||) ), clamped into (0, pi/2).

    `min_radius` defines a neighbourhood around the vertex where the cone is undefined;
    points inside are projected to the boundary. Do NOT pre-scale `min_radius` by
    1/sqrt(curv) — the curvature already appears in the denominator.
    """
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))


@_fp32
def oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Exterior angle at `x` in the hyperbolic triangle (Origin, x, y).

    Derived from the hyperbolic law of cosines (MERU Eq. 11). Used with `half_aperture`
    to enforce entailment: `x` entails `y` when oxy_angle(x, y) <= half_aperture(x).
    Inputs are paired (same batch size); returns shape `(B,)` in (0, pi).
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # curv * <x, y>_L, diagonal only.
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


@_fp32
def pairwise_oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Pairwise exterior angle at each `x_i` in triangle (Origin, x_i, y_j).

    Batched `[B1, B2]` form of `oxy_angle` (its diagonal equals oxy_angle(x, y)).
    Used by the angle-based contrastive loss for cross-batch negatives.
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))  # (B1,)
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))  # (B2,)

    c_xyl = curv * pairwise_inner(x, y, curv)  # (B1, B2)

    acos_numer = y_time[None, :] + c_xyl * x_time[:, None]
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1)[:, None] * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
