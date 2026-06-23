# Curvature feasibility of the taxonomic cone hierarchy (analytical)

**Claim.** A hyperbolic embedding that *simultaneously* satisfies radial spread, cone entailment,
and species separability **exists**, but only when the curvature is high enough (`c ≳ 4`). Every
trained config learned `c ∈ [0.65, 1.38]`, i.e. in the *infeasible* regime. This motivates
**initialising curvature at `c ≥ 4`** (within the model's learnable clamp), rather than the
default `c_init = 1`.

This is a purely analytical, model-free result (closed-form Lorentz geometry, no GPU), reproduced
by `scripts/ideal_hierarchy.py`.

## The three properties

For a taxonomy order→family→genus→species, a *good* embedding for classification needs all three:

1. **Radial spread** — coarse ranks at smaller geodesic radius `r`, fine ranks larger (monotone).
   (Aperture `ψ(x) = arcsin(2·min_radius / (√c·‖x‖))` *shrinks* with radius, so this makes coarse
   cones wide and fine cones narrow — the natural hierarchy shape.)
2. **Entailment (nesting)** — each child's *whole* cone sits inside its parent's:
   `slack = ψ(r_parent) − ψ(r_child) − Ξ(parent, child) ≥ 0`, where `Ξ = oxy_angle`.
3. **Separability** — siblings at every rank are geodesically distinct: pairwise distance `≥ τ`.

## Method: solve per-edge, then chain

The hierarchy is a chain of independent parent→children placements, so feasibility is decided
edge by edge. A node is placed by (axis direction, radius `r`); for a parent at `r_p` and a child
at `r_c` offset by true angle `θ`, both `ψ` and `Ξ` are closed forms of `(r_p, r_c, θ, c)`. For each
`(r_p, r_c)` we find the angular window:

- `θ_max_nest(r_p, r_c, c)` — largest `θ` that still nests (binary search on property 2)
- `θ_sep(r_c, c, τ)` — smallest `θ` that separates siblings (binary search on property 3)

An **edge is feasible** ⇔ `θ_sep ≤ θ_max_nest` (some `θ` both nests the child cone and separates
siblings). A **full hierarchy exists** ⇔ a monotone-increasing radius schedule `r0<r1<r2<r3` exists
whose three consecutive edges are all feasible — found by a longest-increasing-feasible-path DP
over a fine radius grid (0.2…6.0, step 0.2).

*Why "low/high radius is relative" resolves an apparent conflict:* a node must be a small-radius
parent (wide cone, to nest its children) yet a large-radius separated child (big gap from its own
parent, for separation). These do **not** conflict on a monotone ladder — every interior node is
large relative to its parent and small relative to its children. A coarse radius grid hides the
threading; the fine grid finds it. (This corrects an earlier "no chain exists" artifact.)

## Result (min_radius = 0.1, τ = 0.5)

| curvature `c` | 4-rank chain (radial + entail + separable)? |
|---|---|
| 1.0 | **NONE** (infeasible) |
| 2.0 | **NONE** (infeasible) |
| 4.0 | ✅ `[0.2, 1.0, 3.2, 5.0]` |
| 7.0 | ✅ `[0.2, 1.2, 3.8, 4.0]` |
| 10.0 | ✅ `[0.2, 1.0, 2.8, 3.6]` |

- **Threshold `c ≳ 4`**, at the model's *fixed* `min_radius = 0.1` — curvature alone unlocks
  feasibility; no change to `min_radius` is needed.
- `c ∈ [4, 10]` is **inside the model's learnable clamp** `[c_init/10, c_init·10] = [0.1, 10]`
  (`model.py`), so a model initialised at `c_init = 1` *could* reach the feasible regime.

## Why this pins B0's failure

The v3 ladder's learned curvatures (from `docs/diag_all_v3.log`):

| config | E | B0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned `c` | (eucl) | 1.08 | 0.65 | 1.00 | 1.00 | 1.38 | 0.70 | 0.75 | 1.10 | 0.66 | 0.76 | 0.70 |

**Every** hyperbolic config learned `c ∈ [0.65, 1.38]` — all below the `c ≳ 4` threshold, all in the
infeasible regime. They initialise at `c_init = 1` and barely move. So:

> The three properties are jointly **achievable**; B0 (and every variant) fails because the learned
> curvature is **~4× too small** to make the cone hierarchy nest a separable set, and the SEL loss
> provides **no pressure to raise it** (origin-collapse — all ranks near the origin, apertures
> saturated at π/2 — is a low-curvature-friendly lazy basin the optimiser settles into).

This is an *optimisation/parametrisation* failure, **not** a geometric impossibility, and **not** a
high-fan-out containment obstruction (a coarse rank near the origin has a near-π/2 cone that
contains all descendants regardless of fan-out — see retracted note).

## Prediction / proposed experiment

Initialise curvature high (`curv_init ≥ 4`, e.g. 4–8). This (a) starts training in the feasible
regime and (b) shifts the learnable clamp to `[0.4, 40]+`, so the model can stay there. Test on B0:

- **If ranks spread radially and `c` stays ≥ 4** → curvature initialisation was the gap; expect the
  species-in-ancestor-cone transitivity (B0: 0.94→0.49→0.05→0.00) to lift.
- **If it still origin-collapses** → the loss landscape needs a radial driver (e.g. an explicit
  per-rank radius target, or the radial `--sel-margin` term), even though the geometry is feasible.

Either outcome is informative and is *justified by this analytical feasibility result* — which is
why we report the `ideal_hierarchy.py` experiment as the basis for the `curv_init ≥ 4` choice.
