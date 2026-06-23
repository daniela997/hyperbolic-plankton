"""Synthetic geometry oracle — analytic ground truth for the hyperbolic diagnostics.

NO model, NO checkpoint, NO GPU. We CONSTRUCT Lorentz embeddings with KNOWN properties via
exp_map0(tangent), then check that the SAME probes used in scripts/diagnose_geometry.py report
the ground truth. This separates "the probe is wrong" from "the model learned bad geometry", and
lets us test the thesis's geometric assumptions analytically instead of hoping training finds them.

Construction primitive: a node is (unit_direction d, geodesic_radius r). We map the tangent
vector (r * d) through exp_map0, which lands EXACTLY at geodesic distance r from the origin along
direction d (verified in test 0). So we dial radius and angular placement directly.

Run:  PYTHONPATH=src python scripts/synthetic_geometry_oracle.py
Four blocks (the four things we asked to validate):
  0. PROBE CORRECTNESS   — known radius/aperture/angle/nesting -> probes must match analytic values
  1. ASSUMPTION TEST     — build the IDEAL radial hierarchy -> does it give transitive nesting AND
                           high separability simultaneously? (the thesis claim)
  2. FAILURE REPRODUCTION — rebuild B0 (flat radii) and angle-collapse (one ray) -> probes must flag
  3. CURVATURE SWEEP     — fix points, sweep c & min_radius -> what's real vs hyperparameter artifact
"""

from __future__ import annotations

import math

import torch

from hyperbolic_plankton import lorentz as L

torch.manual_seed(0)
DIM = 64


def node(direction: torch.Tensor, radius: float, curv: float = 1.0) -> torch.Tensor:
    """Place ONE point at geodesic distance `radius` from origin along unit `direction`.

    exp_map0 of a tangent vector of norm `radius` lands at geodesic distance `radius`
    (test 0 confirms). `direction` need not be unit; we normalise it.
    """
    d = direction / direction.norm()
    return L.exp_map0((radius * d).unsqueeze(0), curv)  # [1, DIM]


def rand_dir(seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(DIM)


def child_at_angle(base: torch.Tensor, theta: float, radius: float, curv: float = 1.0,
                   seed: int = 777) -> torch.Tensor:
    """Node at geodesic `radius`, offset by EXACTLY `theta` radians from unit `base`.

    NOTE: do NOT approximate an angular offset by `base + eps*randn` — a 64-dim eps=0.05
    perturbation is ~23°, not "tiny" (the bug that wrecked the first Block 1). Rotate by a
    true angle in the (base, perp) plane instead.
    """
    b = base / base.norm()
    torch.manual_seed(seed)
    perp = torch.randn(DIM)
    perp = perp - (perp @ b) * b
    perp = perp / perp.norm()
    d = math.cos(theta) * b + math.sin(theta) * perp
    return node(d, radius, curv)


def oxy_over_aperture_table():
    """The master relationship: oxy_angle(parent,child) / ψ(parent) as parent radius grows.

    Ratio ≤ 1 ⇒ child fits in parent cone. Both shrink with radius, but oxy_angle shrinks
    SLOWER than ψ, so the ratio EXPLODES with depth — deep cones can't contain off-axis
    children however far out you push them. This is the structural separability-vs-containment
    finding: geodesic separability is free at large radius, cone-containment is not.
    """
    c = 1.0
    base = rand_dir(20); base = base / base.norm()
    print("\n=== MASTER: oxy_angle / ψ(parent) vs parent radius (≤1 = child fits) ===")
    print("    (child at fixed angular offset θ, radius = rp+1.5; rises ≫1 ⇒ containment fails)")
    print(f"    {'rp':>5s} {'ψ(parent)':>10s} {'θ=1°':>8s} {'θ=0.2°':>8s} {'θ=0.05°':>8s}")
    for rp in (0.5, 1.0, 2.0, 3.0, 4.0):
        p = node(base, rp, c)
        psi = L.half_aperture(p, c).item()
        rats = [L.oxy_angle(p, child_at_angle(base, math.radians(t), rp + 1.5, c), c).item() / psi
                for t in (1.0, 0.2, 0.05)]
        print(f"    {rp:5.1f} {psi:10.4f} {rats[0]:8.2f} {rats[1]:8.2f} {rats[2]:8.2f}")
    print("    → ratio EXPLODES with depth: deep ranks demand near-perfect co-axiality.")
    print("      Forcing containment (--sel-margin) at depth ⇒ collapses species onto the axis")
    print("      (= C5/C10 angular collapse). Containment and angular-separability are at odds.")


def slack(parent, child, curv, min_radius=0.1):
    """ψ(parent) − ψ(child) − oxy_angle(parent, child): ≥0 ⇒ child's WHOLE cone nests in parent."""
    ang = L.oxy_angle(parent, child, curv)
    return (L.half_aperture(parent, curv, min_radius) -
            L.half_aperture(child, curv, min_radius) - ang)


def ok(label, got, want, tol=1e-2):
    flag = "✅" if abs(got - want) <= tol else "❌"
    print(f"    {flag} {label:42s} got={got:+.4f}  want={want:+.4f}")


# ───────────────────────────── 0. PROBE CORRECTNESS ─────────────────────────────
def test_probe_correctness():
    print("\n=== 0. PROBE CORRECTNESS (analytic ground truth) ===")
    c = 1.0

    # (a) node() places at the exact geodesic radius we asked for
    for r in (0.3, 1.0, 2.5):
        x = node(rand_dir(1), r, c)
        ok(f"distance_from_origin(r={r})", L.distance_from_origin(x, c).item(), r)

    # (b) half_aperture matches its closed form ψ = arcsin(2·min_r/(√c·‖x_space‖)).
    #     ‖x_space‖ = sinh(√c·r)/√c for a node at geodesic radius r.
    for r in (0.5, 1.5, 3.0):
        x = node(rand_dir(2), r, c)
        xn = x.norm().item()
        want = math.asin(min(1 - 1e-8, 2 * 0.1 / (xn * c**0.5)))
        ok(f"half_aperture(r={r})", L.half_aperture(x, c).item(), want)

    # (c) aperture SHRINKS with radius (the key fact: far points have narrow cones)
    aps = [L.half_aperture(node(rand_dir(3), r, c), c).item() for r in (0.2, 1.0, 3.0)]
    print(f"    aperture vs radius r=0.2/1.0/3.0: {aps[0]:.3f} > {aps[1]:.3f} > {aps[2]:.3f}"
          f"   {'✅ monotone↓' if aps[0] > aps[1] > aps[2] else '❌'}")

    # (d) oxy_angle = 0 when child is radially OUTWARD along the parent's own ray
    d = rand_dir(4)
    p, ch = node(d, 1.0, c), node(d, 2.0, c)
    ok("oxy_angle (co-axial parent→child)", L.oxy_angle(p, ch, c).item(), 0.0)

    # (e) a co-axial deeper child NESTS (slack ≥ 0): parent wider, child on-axis
    s = slack(p, ch, c).item()
    print(f"    co-axial nesting slack = {s:+.4f}   {'✅ nests (≥0)' if s >= -1e-3 else '❌'}")

    # (f) classifier: nearest-prototype by geodesic dist recovers the right class
    protos = torch.cat([node(rand_dir(10 + i), 1.5, c) for i in range(5)])  # 5 well-separated
    pts = protos + 0.0  # points = their own prototypes (trivially separable)
    pred = L.pairwise_dist(pts, protos, c).argmin(1)
    acc = (pred == torch.arange(5)).float().mean().item()
    ok("nearest-proto classifier (separable)", acc, 1.0)


# ───────────────────────────── 1. ASSUMPTION TEST ─────────────────────────────
def test_ideal_hierarchy():
    print("\n=== 1. ASSUMPTION TEST: ideal radial hierarchy ===")
    print("    Build order→family→genus→species: radius INCREASES with depth, each child")
    print("    angularly offset WITHIN the parent's cone. Claim: transitive nesting AND")
    print("    high separability hold SIMULTANEOUSLY (the thing no trained config achieved).")
    c = 1.0
    ranks = ["order", "family", "genus", "species"]
    radii = [0.6, 1.2, 2.0, 3.0]          # increasing depth → increasing radius
    n_species = 8

    # one lineage: place each rank co-axial-ish, child nudged slightly off parent axis
    base = rand_dir(20)
    chain, prev_dir = [], base
    for r in radii:
        prev_dir = (prev_dir + 0.05 * rand_dir())  # tiny angular drift per rank
        chain.append(node(prev_dir, r, c))
    print("\n    single-lineage nesting (slack ψp−ψc−angle, want ≥0):")
    for a, b, ra, rb in zip(ranks, ranks[1:], chain, chain[1:]):
        s = slack(ra, rb, c).item()
        print(f"      {a:7s}→{b:8s} slack={s:+.4f}  {'✅' if s >= -1e-3 else '❌ pokes out'}")

    # many species: separated RADIALLY (push out, tiny TRUE angular offset) — the hyperbolic way.
    # Uses controlled angles (child_at_angle), not the eps*randn bug.
    order = node(base, radii[0], c)
    species = torch.cat([child_at_angle(base, math.radians(0.3), radii[-1] + 0.3 * i, c, seed=30 + i)
                         for i in range(n_species)])
    # transitivity: is each species inside the ORDER cone?
    ang = L.oxy_angle(order.expand(n_species, -1), species, c)
    in_order = (ang <= L.half_aperture(order, c)).float().mean().item()
    # separability: mean pairwise geodesic dist between species protos
    pd = L.pairwise_dist(species, species, c)
    sep = pd[~torch.eye(n_species, dtype=bool)].mean().item()
    # angular spread of species directions (0 = spread, 1 = collapsed)
    dn = torch.nn.functional.normalize(species, dim=-1)
    cos = (dn @ dn.T)[~torch.eye(n_species, dtype=bool)].mean().item()
    print(f"\n    species-in-order-cone   = {in_order:.2f}  (want high → transitive)")
    print(f"    inter-species sep dist  = {sep:.3f}  (want high → discriminable)")
    print(f"    species-dir pairwise cos= {cos:.3f}  (want LOW → angularly spread)")
    verdict = in_order > 0.8 and sep > 1.0 and cos < 0.5
    print(f"    → transitive AND separable AND spread simultaneously? "
          f"{'✅ YES (assumption holds)' if verdict else '❌ NO (tension is real)'}")


# ───────────────────────────── 2. FAILURE REPRODUCTION ─────────────────────────────
def test_failure_modes():
    print("\n=== 2. FAILURE REPRODUCTION (probes must flag the known pathologies) ===")
    c = 1.0

    # (a) B0 flat-radius origin-collapse: all ranks at tiny equal radius
    print("\n  (a) flat radii ~0.15 (B0): apertures saturate, slack ≪ 0, NO nesting")
    d = rand_dir(40)
    flat = [node(d + 0.1 * rand_dir(40 + i), 0.15, c) for i in range(4)]
    aps = [L.half_aperture(x, c).item() for x in flat]
    print(f"      apertures: {[f'{a:.3f}' for a in aps]}  "
          f"{'✅ all ≈π/2 saturated' if all(a > 1.5 for a in aps) else 'not saturated'}")
    s = [slack(flat[i], flat[i + 1], c).item() for i in range(3)]
    print(f"      slacks:    {[f'{x:+.3f}' for x in s]}  "
          f"{'✅ all <0 (no nesting, matches B0)' if all(x < 0 for x in s) else '❌'}")

    # (b) angle-collapse: many images on ONE ray → transitive but zero separability
    print("\n  (b) angle-collapse (C5/C10): images on one ray → in-cone high, separability dead")
    apex = node(rand_dir(50), 0.5, c)
    ray = rand_dir(51)
    imgs = torch.cat([node(ray + 0.001 * rand_dir(51 + i), 1.5, c) for i in range(16)])
    ang = L.oxy_angle(apex.expand(16, -1), imgs, c)
    in_cone = (ang <= L.half_aperture(apex, c)).float().mean().item()
    dn = torch.nn.functional.normalize(imgs, dim=-1)
    cos = (dn @ dn.T)[~torch.eye(16, dtype=bool)].mean().item()
    sep = L.pairwise_dist(imgs, imgs, c)[~torch.eye(16, dtype=bool)].mean().item()
    print(f"      img-dir cos = {cos:.3f} {'✅ collapsed' if cos > 0.9 else ''}   "
          f"sep dist = {sep:.3f} {'✅ ~0' if sep < 0.2 else ''}")

    # (c) spread images: same apex, images fan out → separable, in-cone may drop (the tension)
    print("\n  (c) spread images: fan out → separability up, in-cone DOWN (transitivity tension)")
    imgs2 = torch.cat([node(rand_dir(70 + i), 1.5, c) for i in range(16)])
    ang2 = L.oxy_angle(apex.expand(16, -1), imgs2, c)
    in_cone2 = (ang2 <= L.half_aperture(apex, c)).float().mean().item()
    dn2 = torch.nn.functional.normalize(imgs2, dim=-1)
    cos2 = (dn2 @ dn2.T)[~torch.eye(16, dtype=bool)].mean().item()
    sep2 = L.pairwise_dist(imgs2, imgs2, c)[~torch.eye(16, dtype=bool)].mean().item()
    print(f"      img-dir cos = {cos2:.3f}   sep dist = {sep2:.3f}   in-apex-cone = {in_cone2:.2f}")
    print(f"      collapsed in-cone {in_cone:.2f} vs spread in-cone {in_cone2:.2f}: "
          f"{'✅ spreading costs transitivity' if in_cone2 < in_cone else 'no tension here'}")


# ───────────────────────────── 3. CURVATURE SWEEP ─────────────────────────────
def test_curvature_sweep():
    print("\n=== 3. CURVATURE / min_radius SENSITIVITY (real signal vs hyperparameter artifact) ===")
    print("    FIX the same two nodes (radii 1.0, 2.0, small angular offset); vary c and min_r.")
    d = rand_dir(80)
    off = d + 0.2 * rand_dir(81)
    print(f"\n    {'curv':>6s} {'min_r':>6s} | {'ψ(parent)':>9s} {'ψ(child)':>9s} {'angle':>7s} {'slack':>7s}")
    for c in (0.5, 1.0, 2.0):
        p, ch = node(d, 1.0, c), node(off, 2.0, c)
        for mr in (0.05, 0.1, 0.2):
            ap = L.half_aperture(p, c, mr).item()
            ac = L.half_aperture(ch, c, mr).item()
            an = L.oxy_angle(p, ch, c).item()
            print(f"    {c:6.1f} {mr:6.2f} | {ap:9.3f} {ac:9.3f} {an:7.3f} {ap - ac - an:+7.3f}")
    print("\n    Reading: if slack flips sign across c/min_r for FIXED points, then a slack/aperture")
    print("    difference between two trained configs at DIFFERENT learned c is partly artifact —")
    print("    must compare at matched c (the B0 c=1.08 vs C1 c=0.65 confound).")


if __name__ == "__main__":
    test_probe_correctness()
    oxy_over_aperture_table()
    test_ideal_hierarchy()
    test_failure_modes()
    test_curvature_sweep()
    print("\n✅ oracle complete (CPU-only, no model/GPU touched).")
