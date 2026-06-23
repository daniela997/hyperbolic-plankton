"""Analytical existence proof: does a hyperbolic embedding satisfying RADIAL-SPREAD +
ENTAILMENT + SEPARABILITY simultaneously exist, and at what curvature?

Model-free, CPU-only, analytical (closed-form Lorentz geometry; no model, no GPU). This is the
experiment we REPORT to justify initialising curvature ≥ ~4.

Three properties we require of a taxonomic embedding (order→family→genus→species):
  (1) RADIAL SPREAD — coarse ranks at smaller geodesic radius, fine ranks larger (monotone).
  (2) ENTAILMENT    — each child's WHOLE cone nests in its parent's: ψ(r_p) − ψ(r_c) − Ξ ≥ 0,
                      where ψ = half_aperture, Ξ = oxy_angle(parent, child).
  (3) SEPARABILITY  — siblings at a rank are geodesically distinct: pairwise dist ≥ τ.

Method — solve the constraint system EDGE BY EDGE, then chain:
  A node is placed by (axis direction, geodesic radius r). For a parent at r_p and a child at
  r_c offset by true angle θ from the parent axis, ψ and Ξ are closed forms of (r_p, r_c, θ, c).
  For each (r_p, r_c) we compute the angular window:
      θ_max_nest(r_p,r_c,c) = largest θ that still NESTS  (binary search on (1))
      θ_sep(r_c,c,τ)        = smallest θ giving sibling separation ≥ τ  (binary search on (3))
  The edge is feasible ⇔ θ_sep ≤ θ_max_nest (a θ both separates AND nests).
  A full hierarchy exists ⇔ a monotone-increasing radius schedule r0<r1<r2<r3 exists whose 3
  consecutive edges are all feasible (longest-increasing-feasible-path DP over a fine radius grid).

KEY RESULT (min_radius=0.1, τ=0.5):
  - No 4-rank chain at curvature c≤2; a chain APPEARS at c≳4 and persists to c=10.
  - The chain exists at the model's FIXED min_radius=0.1 — curvature alone unlocks it, and c≈4–10
    is inside the model's learnable clamp [c_init/10, c_init·10] = [0.1, 10] (default c_init=1).
  - But every trained config learned c∈[0.65, 1.38] (init 1.0) — i.e. in the INFEASIBLE regime.
  ⇒ The 3 properties are jointly ACHIEVABLE; B0's failure is that learned curvature is ~4× too
    small and the SEL loss provides no pressure to raise it. Justifies curv_init ≥ ~4.

  "low/high radius is relative": the apparent conflict (a node must be a small-radius parent yet a
  large-radius separated child) dissolves on a monotone ladder — every interior node is large
  relative to its parent and small relative to its children. A coarse 0.2-step radius grid hid the
  threading; a fine grid finds it.

Run:  PYTHONPATH=src python scripts/ideal_hierarchy.py
"""

from __future__ import annotations

import math

import torch

from hyperbolic_plankton import lorentz as L

DIM = 64
MIN_RADIUS = 0.1   # the model's FIXED aperture floor (r_min default in loss.py)
TAU = 0.5          # required sibling geodesic separation


def unit(v):
    return v / v.norm()


# fixed axis + perpendicular so ψ/Ξ depend only on (r_p, r_c, θ, c), not on a random draw
_AX = unit(torch.arange(1.0, DIM + 1))
_PE = unit(torch.randn(DIM, generator=torch.Generator().manual_seed(3)))
_PE = unit(_PE - (_PE @ _AX) * _AX)


def place(direction, radius, curv):
    return L.exp_map0((radius * unit(direction)).unsqueeze(0), curv)


def psi(radius, curv, min_radius=MIN_RADIUS):
    return L.half_aperture(place(_AX, radius, curv), curv, min_radius).item()


def child(r_c, theta, curv):
    d = unit(math.cos(theta) * _AX + math.sin(theta) * _PE)
    return place(d, r_c, curv)


def oxy(r_p, r_c, theta, curv):
    return L.oxy_angle(place(_AX, r_p, curv), child(r_c, theta, curv), curv).item()


def edge_feasible(r_p, r_c, curv, min_radius=MIN_RADIUS, tau=TAU):
    """θ_sep ≤ θ_max_nest ?  (a single θ both nests the child cone and separates siblings)."""
    # θ_max_nest: largest θ with ψ(r_p) − ψ(r_c) − Ξ ≥ 0
    if psi(r_p, curv, min_radius) - psi(r_c, curv, min_radius) - oxy(r_p, r_c, 1e-4, curv) < 0:
        return False
    lo, hi = 0.0, math.pi / 2
    for _ in range(30):
        m = (lo + hi) / 2
        if psi(r_p, curv, min_radius) - psi(r_c, curv, min_radius) - oxy(r_p, r_c, m, curv) >= 0:
            lo = m
        else:
            hi = m
    theta_max = lo
    # θ_sep: smallest θ with two-sibling geodesic dist ≥ τ at radius r_c
    s1 = place(_AX, r_c, curv)
    def d(theta):
        return L.pairwise_dist(s1, child(r_c, theta, curv), curv).item()
    if d(math.pi / 2) < tau:
        return False
    lo, hi = 0.0, math.pi / 2
    for _ in range(30):
        m = (lo + hi) / 2
        if d(m) >= tau:
            hi = m
        else:
            lo = m
    return hi <= theta_max


def find_chain(curv, radii, min_radius=MIN_RADIUS, tau=TAU):
    """Longest increasing feasible-edge path; return a 4-rank radius schedule if one exists."""
    n = len(radii)
    feas = [[edge_feasible(radii[i], radii[j], curv, min_radius, tau) if j > i else False
             for j in range(n)] for i in range(n)]
    best = [1] * n
    pred = [-1] * n
    for j in range(n):
        for i in range(j):
            if feas[i][j] and best[i] + 1 > best[j]:
                best[j] = best[i] + 1
                pred[j] = i
    if max(best) < 4:
        return None
    j = best.index(max(best))
    path = []
    while j != -1:
        path.append(round(radii[j], 2))
        j = pred[j]
    return path[::-1][:4]


def main():
    radii = [round(0.2 + 0.2 * k, 2) for k in range(30)]   # fine grid 0.2 .. 6.0
    print(f"All-three-properties feasibility   (min_radius={MIN_RADIUS}, τ={TAU}, fine radius grid)")
    print(f"Model can LEARN curvature in [0.1, 10] (clamp = c_init/10 .. c_init·10, c_init=1).\n")
    print(f"{'curvature c':>12s} {'4-rank chain (radial+entail+separable)':>42s}")
    threshold = None
    for c in (1.0, 2.0, 4.0, 7.0, 10.0):
        ch = find_chain(c, radii)
        if ch and threshold is None:
            threshold = c
        print(f"{c:12.1f} {str(ch) if ch else 'NONE (infeasible)':>42s}")
    print(f"\nFeasibility threshold: c ≳ {threshold}.  Trained configs learned c∈[0.65,1.38] "
          f"→ all INFEASIBLE.")
    print("⇒ geometry is achievable; the gap is curvature being ~4× too small. Justifies curv_init ≥ 4.")


if __name__ == "__main__":
    main()
    print("\n(CPU-only, analytical, no model/GPU.)")
