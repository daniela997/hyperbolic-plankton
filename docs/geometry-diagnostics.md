# Geometry diagnostics — BIOSCAN ablation variants

Consolidates every geometric/classification diagnostic run this session. Goal: once GPUs are
free, fill the gaps (`—`) and replace small-set probes (`*`) with full-split numbers, to fully
characterise where each variant's geometry helps or hurts.

**Legend:** `*` = small-set probe (256–1024 imgs, possibly noisy / no CI). `—` = not yet
evaluated. Plain value = full-split / population-level (robust).

**Probe source:** ALL rows below are now the **v3 r64 ladder** checkpoints, full test_seen
(4,878 imgs), via `scripts/diagnose_geometry.py` (run `scripts/run_diagnose_all.sh`). The old
r128-vs-v3 mismatch is resolved — geometry probes and F1 are the same checkpoints. The old r128
findings transferred essentially unchanged (B0 still origin-collapsed, image still non-transitive).

## Classification (full-split v3 r64 ladder — robust)

| metric           |   E   |  B0   |  C1   |  C2   |  C3   |  C4   |  C5   |  C6   |  C7   |  C8   |  C9   |  C10  |
|------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| seen species F1  | **0.765** | 0.628 | 0.466 | 0.383 | 0.147 | 0.614 | 0.621 | 0.583 | 0.619 | 0.450 | 0.602 | 0.648 |
| unseen species F1| **0.071** | 0.050 | 0.048 | 0.045 | 0.025 | 0.053 | 0.059 | 0.054 | **0.060** | 0.046 | 0.052 | 0.053 |

Config key: E euclidean; B0 dist+indepSEL (paper); C1 dist+cumulSEL; C2 SEL-only cumul; C3
SEL-only indep; C4 CL-only; C5 angle+cumulSEL; C6 angle+indepSEL; C7 dist+indepSEL+mask;
C8 dist+cumulSEL+mask; C9 angle+indepSEL+mask; C10 angle+cumulSEL+mask.

Headline so far: on SEEN, E leads all. On UNSEEN, C7 (0.060) and C5 (0.059) BEAT B0 (0.050) and
approach E (0.071) — unseen is the regime where hyperbolic variants start to show value.

## Geometry probes — v3 r64, full test_seen (robust)

All numbers below are population-level over 4,878 test_seen images. Probes run on each config's
**trained SEL text form** (indep vs cumul, per the `sel_text` row). Euclidean (E) has no cones.

| metric                              |  B0   |  C1   |  C2   |  C3   |  C4   |  C5   |  C6   |  C7   |  C8   |  C9   |  C10  |   E   |
|-------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| sel_text                            | indep | cumul | cumul | indep | indep | cumul | indep | indep | cumul | indep | cumul |  —    |
| **species top-1 acc**               | 0.786 | 0.668 | 0.561 | 0.422 | **0.797** | 0.727 | 0.672 | 0.775 | 0.660 | 0.758 | 0.776 | **0.841** |
| learned curvature                   | 1.080 | 0.652 | 0.997 | 0.998 | 1.379 | 0.704 | 0.746 | 1.095 | 0.659 | 0.763 | 0.704 | 1.000 |
| entail_ok (genus→species)           | 1.000 | 0.990 | 1.000 | 1.000 | **0.000** | 0.995 | 1.000 | 1.000 | 0.976 | 1.000 | 0.996 |  —    |
| text radius order→species           |.15→.18|.17→.96|.02→.05|.02→.05|.62→.64|.21→.40|.07→.21|.16→.19|.18→1.0|.08→.21|.21→.36|  —    |
| text aperture order→species         |π/2 flat|1.57→.24|π/2 flat|π/2 flat|.26→.25|1.57→.62|π/2 flat|π/2 flat|1.50→.22|π/2 flat|1.57→.71|  —  |
| nesting slack genus→species         | −1.31 | −0.11 | −1.29 | −1.30 | −1.88 | −0.51 | −1.27 | −1.32 | −0.10 | −1.25 | −0.57 |  —    |
| **fits% (any edge nests)**          | 0.00  | 0.08  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.09  | 0.00  | 0.00  |  —    |
| image-in-cone order/fam/gen/sp      |0/.05/.49/.94|.64/.97/.96/.39|.33/.53/.78/.90|.41/.62/.81/.90|0/0/0/0|1/1/.99/.87|0/.01/.30/.70|.03/.09/.59/.94|.64/.95/.95/.35|0/.01/.32/.74|1/1/.99/.81| — |
| image radius (mean)                 | 0.82  | 1.46  | 0.08  | 0.09  | 0.81  | 0.46  | 0.62  | 0.85  | 1.52  | 0.60  | 0.41  |  —    |
| image-dir pairwise-cos (1=collapsed)| 0.017 | 0.002 | 0.288 | 0.263 | 0.010 | **0.961** | **0.939** | 0.010 | 0.007 | **0.910** | **0.930** |  —  |
| image↔own-species dist              | 0.790 | 0.560 | 0.067 | 0.068 | 0.556 | 0.070 | 0.576 | 0.811 | 0.590 | 0.560 | 0.061 |  —    |
| inter-species proto dist (separab.) | 1.072 | 1.393 | 0.059 | 0.051 | 1.044 | 0.111 | 0.179 | 1.169 | 1.430 | 0.208 | 0.129 | 0.994 |

(top-1 95% CIs are all ±~0.012; the acc ordering above is well-separated except near-ties
C7≈C10 and C1≈C6.)

## Key findings the table encodes
- **No config achieves cone-nesting.** `fits%` is 0.00 for every config except C1/C8 (0.08–0.09,
  and only at the genus→species edge). The taxonomic hierarchy is **never geometrically realized**
  as nested cones — slack stays negative everywhere. This is the core motivation for `--sel-margin`.
- **Cones are inert for classification.** C4 (CL-only) has entail_ok=**0.000** and image-in-cone
  **0/0/0/0**, yet is the **best** hyperbolic config (0.797). SEL/entailment structure contributes
  nothing to accuracy; hyperbolic distance-CL alone does the work. SEL-only configs (C2 0.561,
  C3 0.422) are the worst — the hierarchy without CL is actively bad.
- **B0 origin-collapse** (confirmed v3): all ranks tiny radius (.15→.18), apertures saturated at
  π/2 → no nesting possible. Cumulative-SEL (C1/C5/C8/C10) escapes this for DEEP ranks (species
  aperture →0.22–0.62) but coarse ranks stay π/2-collapsed, so slack is still <0.
- **Image entailment NON-transitive** (B0 .94→.49→.05→.00 sp→gen→fam→order): image lands in its
  species cone but escapes every ancestor. Only cumulative-SEL angle-CL configs (C5/C10) get
  transitive (1/1/.99/.8x) — but they pay for it via angular collapse.
- **angle-CL angular-collapse** is the consistent failure mode of C5/C6/C9/C10: image-dir cos
  **0.91–0.96** (images on one ray) vs distance-CL's **0.01–0.02**, and separability collapses
  (proto dist 0.11–0.21 vs 1.0–1.4). angle-CL buys transitivity by destroying angular spread.
- Use full-split top-1/F1, never subsample probes, for any accuracy claim.

## --sel-margin RESULTS (the cone-containment term WORKED — full diagnostics)

`scripts/run_sel_margin_bioscan.sh` (B0/C1/C5 × {0.5,1.0}) adds the cone-CONTAINMENT hinge
`relu(angle + w·ψ_child − ψ_parent)` so the child's WHOLE cone must fit the parent. All 6 ran;
diagnosed full-split. **The prediction held: it achieves cone-nesting (fits% 0.00→0.6–0.95), which
NO base config did.**

| config            | fits% gen→sp | seen F1 | unseen F1 | separability | img-dir cos (collapse) |
|-------------------|------|------|------|------|------|
| B0 (base)         | 0.00 | 0.628 | 0.050 | 1.07 | 0.017 |
| B0 + margin 0.5   | 0.00 | 0.625 | 0.056 | 1.06 | 0.037 |
| B0 + margin 1.0   | 0.00 | 0.586 | 0.051 | 1.16 | 0.577 |
| C1 (base, dist+cumulSEL) | 0.08 | 0.466 | 0.048 | 1.39 | 0.002 |
| **C1 + margin 1.0** | **0.60–0.85** | 0.615 | 0.044 | **1.58** | **0.093** |
| C5 (base, angle+cumulSEL) | 0.00 | 0.621 | 0.059 | 0.11 | 0.961 |
| C5 + margin 1.0   | **0.61–0.95** | 0.512 | 0.048 | 0.21 | 0.927 |

**Key findings:**
- **Cone-nesting IS achievable** — C1+margin1.0 and C5+margin1.0 reach fits% 0.6–0.95 (vs 0.00 for
  every base config). The margin term does exactly what it was built to do. So "no config nests" was
  true ONLY of the non-margin configs; the margin is the fix and it works.
- **C1+margin1.0 = the geometry we said was impossible: NESTING *with* SEPARABILITY.** It nests
  (fits% 0.6–0.85) AND keeps separability 1.58 (highest of any config) AND stays angularly spread
  (cos 0.093, NOT collapsed). First config to get both. Because C1 is DISTANCE-CL (preserves spread);
  the margin adds nesting on top without collapse.
- **C5+margin nests but stays COLLAPSED** (cos 0.927) — because C5 is ANGLE-CL, which collapses
  regardless of the margin. So the radial driver must pair with DISTANCE-CL, not angle-CL.
- **Nesting did NOT improve classification** — C1+margin seen 0.615 (vs C1's 0.466 is *up*, but its
  geometry-fixed cousin still trails B0 0.628; C5+margin 0.512 *down* from C5 0.621). Achieving the
  cone hierarchy is necessary-but-not-sufficient: the geometry can be right while F1 lags (here
  C1's cumulative-SEL base handicaps it regardless of nesting).

**Implication for the radial-ordering loss** (`radial_ordering_loss`, the explicit curvature driver):
C1+margin is the proof-of-concept that an explicit radial driver + distance-CL yields nesting +
separability together. The radial loss is a cleaner version of the same idea (order radii directly,
no cone-margin coupling). Build it on a DISTANCE-CL + SEL base (B0/C1), NOT angle-CL.
