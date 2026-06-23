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
> curvature is **~4× too small** to make the cone hierarchy nest a separable set, and the loss
> provides **no pressure to raise it.**

### Why training doesn't reach `c ≥ 4` (curvature gradient is weak; verified)

Curvature flows through *both* loss terms, yet barely moves. The mechanism, confirmed from the
wandb trajectories:

- **SEL pins curvature.** SEL is satisfied the easy way — origin-collapse (all ranks near origin,
  apertures saturated at π/2) makes the hinge `relu(oxy_angle − τ·ψ) = 0` at *any* curvature. A
  satisfied hinge has **zero gradient**, so SEL exerts no curvature pressure. Decisive evidence:
  **C2 (SEL-only, no CL) holds `c` at exactly 1.00 → 1.00** for all 80 epochs.
- **CL's curvature gradient is weak and not upward.** With CL present, `c` drifts only a little, and
  not consistently up: distance-CL → 1.08 (B0) / 1.38 (C4); **angle-CL → 0.70 (C5) / 0.75 (C6),
  i.e. *down*.** None approach 4.
- **`logit_scale` is NOT a curvature substitute.** It is the standard CLIP softmax temperature; it
  rises 14.3 → ~21 whenever CL is present (B0/C4/C5/C6 alike) as the model sharpens its contrastive
  distribution — normal CLIP behaviour, independent of curvature. (Earlier hypothesis that
  `logit_scale` "absorbs" curvature via a `(c, scale)` degeneracy is **refuted** by C2: removing CL
  freezes *both* `scale` and `curv`, so they are decoupled, not trading off.)

Net: the feasible (c≥4) and collapsed (c≈1) geometries have ~the same loss, and the optimiser has
no gradient incentive to leave the easy-to-reach low-`c` basin. This is an
*optimisation/parametrisation* failure, **not** a geometric impossibility, and **not** a
high-fan-out containment obstruction (a coarse rank near the origin has a near-π/2 cone that
contains all descendants regardless of fan-out — see retracted note).

Practical consequence: because `c` barely drifts from init (C2: 0 drift; others 8–38%),
**initialising at `c_init = 4` should largely STAY near 4** — the weak residual gradient won't undo
a 4× init. So `learn_curv=True` with `curv_init=4` is expected to hold the feasible regime; pinning
(`learn_curv=False`) is the stricter control.

## External corroboration: "Accept the Modality Gap" (2024)

This paper independently observed the SAME curvature mechanism, with numbers that match ours:

- Fig. 13: **geodesic-distance CL drives MERU's curvature DOWN to the clamp floor (0.1)**; their
  loss drives it **UP to ~5** and plateaus. They call geodesic-CL-in-hyperbolic a "fundamental
  mismatch," and report MERU **fails to converge at fixed curvature ≥ 0.5** (Table 5).
- Our analytical feasibility threshold (**c ≳ 4**) and their successful loss's converged curvature
  (**~5**) agree — strong external evidence that the feasible regime is up there, not at c≈1.

**Their fix is a LOSS, not an init** — and it names the missing ingredient. Their final loss
(Eq. 11–13) is `L_angle + λ·L_centroid`:
- `L_angle` = an **angle-based** contrastive loss (replace geodesic similarity with the exterior
  angle α = `oxy_angle`; minimise α for matches, maximise β = π − α). This is *our* angle-CL.
- `L_centroid` = a soft regulariser forcing the **text centroid closer to the origin than the image
  centroid** — an explicit **radial-ordering driver** (coarse=text near origin, fine=image far out).

Crucially, **angle-CL alone is NOT the curvature driver** — our angle-CL configs (C5/C6) drove
curvature DOWN (0.70/0.75) and collapsed angularly (dir-cos 0.91–0.96), the *opposite* of their
c→5. The difference is `L_centroid`: without a radial driver, angle-CL satisfies itself by angular
collapse (our result); with it, embeddings spread radially and curvature rises (their result). They
also report `L_entail` alone gave "no meaningful results" — matching our SEL-only C2/C3 being worst.

## What is ours vs known, and the proposed experiment

- **Novel (ours):** the analytical **feasibility threshold c ≳ 4** (`ideal_hierarchy.py`); the
  diagnosis that every cone-based config sits at c≈1; and the **mechanism** — SEL satisfied-by-
  collapse zeroes the curvature gradient (C2: c frozen 1.00→1.00), `logit_scale` is just CLIP
  temperature, not a curvature substitute.
- **Known (Accept-the-Modality-Gap):** that geodesic-CL suppresses curvature and that an
  `angle-CL + radial-centroid` loss raises it to ~5. So a curvature fix is **not** our contribution;
  cite them. Our `curv_init` test is a *control/ablation*, not a proposed method.

Experiment (after the margin sweep), framed as ablations, not a novel fix:
1. **`curv_init = 4, learn_curv = False`** (pinned) — cleanest test of "is the c≳4 geometry usable
   for classification when we force it." Isolates geometry-feasibility from loss-drift.
2. **`curv_init = 4, learn_curv = True`** — does it *hold* near 4, or drift back down like their
   MERU? (Our C2 low-drift evidence says it may hold under SEL; their Fig. 13 says geodesic-CL
   would pull it down. The B0 mix decides it empirically.)
3. The principled fix to *compare against* is their `L_angle + L_centroid` (angle-CL + a radial
   centroid term) — the radial driver our diagnostics kept pointing at, now with a published form.
