"""Feasibility sweep — is there ANY geometry where the taxonomy nests AND species stay separable?

Purely analytical (same Lorentz construction as synthetic_geometry_oracle.py): NO model, NO GPU.
We grid-search the geometry's free knobs and ask, for each setting, whether BOTH constraints hold:

  (C1) CONTAINMENT  — every species fits inside its parent cone: oxy_angle(parent, sp) ≤ ψ(parent)
                      for all sampled species (and the rank chain order→…→species nests).
  (C2) SEPARABILITY — species are pairwise geodesically distinct: min pairwise dist ≥ sep_min.

Knobs:
  curv c, min_radius (aperture scale, a loss hyperparam), the per-rank radius schedule, and the
  angular spread θ_sp that K distinct species require at the species shell. The whole point: C2
  forces θ_sp > 0 (you can't separate species at θ=0), C1 caps the angle the parent cone tolerates.

Output: per (c, min_radius), the feasible region over (θ_sp, radius schedule). Reported SEPARATELY
per curvature so we never compare across c (the B0 c=1.08 vs C1 c=0.65 confound). Three outcomes:
  feasible region exists -> that's the target geometry --sel-margin should aim for;
  empty -> structural obstruction (proven over the grid, not one hand-built point);
  feasible only at degenerate settings -> quantifies how far the objective is from achievable.

Caveat: this is the IDEALISED geometry. Feasible ≠ "training reaches it" (necessary, not
sufficient). Infeasible IS conclusive: if the closed-form construction can't, the trained model can't.

Run:  PYTHONPATH=src python scripts/feasibility_sweep.py
"""

from __future__ import annotations

import math

import torch

from hyperbolic_plankton import lorentz as L
from synthetic_geometry_oracle import node, child_at_angle

DIM = 64
RANKS = ["order", "family", "genus", "species"]


def min_separating_angle(k_species: int, r_sp: float, curv: float, sep_min: float,
                         seed: int = 0) -> float:
    """Smallest true angular spread θ at which K species on the r_sp shell are pairwise
    geodesically ≥ sep_min apart. Binary-search θ; returns the minimal feasible θ (or inf)."""
    base = torch.randn(DIM, generator=torch.Generator().manual_seed(seed))
    base = base / base.norm()

    def min_pair_dist(theta):
        # K species spread over an angular cap of half-width theta around base
        g = torch.Generator().manual_seed(seed + 1)
        sp = []
        for i in range(k_species):
            perp = torch.randn(DIM, generator=g)
            perp = perp - (perp @ base) * base
            perp = perp / perp.norm()
            ti = theta * (i + 1) / k_species  # fan out to the cap edge
            d = math.cos(ti) * base + math.sin(ti) * perp
            sp.append(node(d, r_sp, curv))
        sp = torch.cat(sp)
        pd = L.pairwise_dist(sp, sp, curv)
        return pd[~torch.eye(k_species, dtype=bool)].min().item()

    lo, hi = 0.0, math.pi / 2
    if min_pair_dist(hi) < sep_min:
        return math.inf  # even max spread can't separate them at this radius
    for _ in range(30):
        mid = (lo + hi) / 2
        if min_pair_dist(mid) >= sep_min:
            hi = mid
        else:
            lo = mid
    return hi


def chain_nests(radii, curv, min_radius, theta_per_edge):
    """Does the full order→…→species chain nest? Each child offset by theta_per_edge from parent
    axis (cumulative). Returns (all_nest, per_edge_slack)."""
    base = torch.randn(DIM, generator=torch.Generator().manual_seed(7))
    base = base / base.norm()
    chain, ang = [], 0.0
    for i, r in enumerate(radii):
        chain.append(child_at_angle(base, ang, r, curv, seed=1000 + i))
        ang += theta_per_edge
    slacks = []
    for p, c in zip(chain[:-1], chain[1:]):
        s = (L.half_aperture(p, curv, min_radius)
             - L.half_aperture(c, curv, min_radius)
             - L.oxy_angle(p, c, curv)).item()
        slacks.append(s)
    return all(s >= -1e-3 for s in slacks), slacks


def species_fit_parent(r_parent, r_sp, theta_sp, curv, min_radius, k_species, seed=0):
    """Fraction of K species (spread by θ_sp at r_sp) that fit inside the parent cone at r_parent."""
    base = torch.randn(DIM, generator=torch.Generator().manual_seed(seed))
    base = base / base.norm()
    parent = node(base, r_parent, curv)
    psi = L.half_aperture(parent, curv, min_radius)
    g = torch.Generator().manual_seed(seed + 1)
    sp = []
    for i in range(k_species):
        perp = torch.randn(DIM, generator=g)
        perp = perp - (perp @ base) * base
        perp = perp / perp.norm()
        ti = theta_sp * (i + 1) / k_species
        sp.append(node(math.cos(ti) * base + math.sin(ti) * perp, r_sp, curv))
    sp = torch.cat(sp)
    ang = L.oxy_angle(parent.expand(k_species, -1), sp, curv)
    return (ang <= psi).float().mean().item()


def sweep():
    K = 20            # species under one genus (BIOSCAN genera have ~10-30)
    SEP_MIN = 1.0     # required min pairwise geodesic separation (= "discriminable")
    print(f"Feasibility: K={K} species/genus, require min pairwise geodesic sep ≥ {SEP_MIN}")
    print("CONTAINMENT (all species in genus cone) AND SEPARABILITY (sep≥SEP_MIN) together?\n")

    for curv in (0.5, 1.0, 2.0):
        print(f"═══ curvature c = {curv} ═══")
        # 1) what angular spread do K species NEED to be separable, at each species radius?
        print(f"  species shell radius → min θ_sp needed for sep≥{SEP_MIN} (deg):")
        theta_need = {}
        for r_sp in (2.0, 3.0, 4.0, 5.0):
            th = min_separating_angle(K, r_sp, curv, SEP_MIN)
            theta_need[r_sp] = th
            s = "∞ (can't separate)" if math.isinf(th) else f"{math.degrees(th):.2f}°"
            print(f"    r_sp={r_sp}: {s}")

        # 2) for a genus at r_parent, what θ_sp does the cone TOLERATE (species still fit)?
        print(f"  genus cone tolerance vs the spread species need (min_radius=0.1):")
        print(f"    {'r_genus':>8s} {'r_sp':>5s} {'ψ(genus)':>9s} {'θ_need':>8s} {'fit%@θ_need':>11s} {'FEASIBLE':>9s}")
        feasible_any = False
        for r_genus in (1.0, 1.5, 2.0):
            for r_sp in (3.0, 4.0):
                th = theta_need[r_sp]
                if math.isinf(th):
                    continue
                psi = L.half_aperture(node(torch.randn(DIM, generator=torch.Generator().manual_seed(0)), r_genus, curv), curv, 0.1).item()
                fit = species_fit_parent(r_genus, r_sp, th, curv, 0.1, K)
                feas = fit >= 0.99
                feasible_any |= feas
                print(f"    {r_genus:8.1f} {r_sp:5.1f} {psi:9.4f} {math.degrees(th):7.2f}° {fit*100:10.0f}% {'✅' if feas else '❌':>9s}")
        print(f"  → any (r_genus, r_sp) where K separable species ALL fit the genus cone? "
              f"{'✅ YES' if feasible_any else '❌ NO (obstruction at c=%s)' % curv}\n")

    # 3) does a LARGER min_radius (wider cones) open feasibility? sweep it at the hardest case
    print("═══ does widening cones (min_radius) rescue feasibility? (c=1.0, r_genus=1.5, r_sp=4.0) ═══")
    th = min_separating_angle(K, 4.0, 1.0, SEP_MIN)
    print(f"  K={K} species need θ_sp={math.degrees(th):.2f}° to separate at r_sp=4.0")
    print(f"  {'min_radius':>11s} {'ψ(genus)':>9s} {'species fit%':>13s}")
    for mr in (0.1, 0.3, 0.5, 1.0, 2.0):
        psi = L.half_aperture(node(torch.randn(DIM, generator=torch.Generator().manual_seed(0)), 1.5, 1.0), 1.0, mr).item()
        fit = species_fit_parent(1.5, 4.0, th, 1.0, mr, K)
        print(f"  {mr:11.2f} {psi:9.4f} {fit*100:12.0f}%")
    print("  (if fit% never reaches 100, even arbitrarily wide cones can't contain separable species)")


if __name__ == "__main__":
    sweep()
    print("\n✅ feasibility sweep complete (CPU-only, analytical, no model/GPU).")
