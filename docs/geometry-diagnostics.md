# Geometry diagnostics — BIOSCAN ablation variants

Consolidates every geometric/classification diagnostic run this session. Goal: once GPUs are
free, fill the gaps (`—`) and replace small-set probes (`*`) with full-split numbers, to fully
characterise where each variant's geometry helps or hurts.

**Legend:** `*` = small-set probe (256–1024 imgs, possibly noisy / no CI). `—` = not yet
evaluated. Plain value = full-split / population-level (robust).

**IMPORTANT — probe source:** the per-rank radius/aperture/slack/transitivity/cos probes were
run on the **OLD r128 June checkpoints** (B0/C1/C5/C8), because they were what existed when we
probed. The full-split test F1 is from the **NEW v3 r64 ladder**. So geometry-probe rows and
F1 rows are from DIFFERENT checkpoints until we re-run the probes on v3 (a TODO below). Curv was
~identical old-vs-new for B0, so the geometry findings likely transfer, but re-confirm.

## Classification (full-split v3 r64 ladder — robust)

| metric          |   E   |  B0   |  C1   |  C2   |  C3   |  C4   |  C5   |  C6   |  C7   | C8 | C9 | C10 |
|-----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----|----|-----|
| seen species F1 | 0.765 | 0.628 | 0.466 | 0.383 | 0.147 | 0.614 | 0.621 | 0.583 | 0.619 | —  | —  | —   |
| unseen species F1| 0.071| 0.050 | 0.048 | 0.045 | 0.025 | 0.053 | 0.059 | 0.054 | **0.060** | — | — | — |

Config key: E euclidean; B0 dist+indepSEL (paper); C1 dist+cumulSEL; C2 SEL-only cumul; C3
SEL-only indep; C4 CL-only; C5 angle+cumulSEL; C6 angle+indepSEL; C7 dist+indepSEL+mask;
C8 dist+cumulSEL+mask; C9 angle+indepSEL+mask; C10 angle+cumulSEL+mask.

Headline so far: on SEEN, E leads all. On UNSEEN, C7 (0.060) and C5 (0.059) BEAT B0 (0.050) and
approach E (0.071) — unseen is the regime where hyperbolic variants start to show value.

## Geometry probes (OLD r128 ckpts — `*` = 256–512 img; re-run on v3 = TODO)

| metric                                  |  B0   |  C1   |  C5   |  C8   | C2/C3/C4/C6/C7/C9/C10 |
|-----------------------------------------|-------|-------|-------|-------|-----------------------|
| species top-1 acc (1024 img)            | 0.66* | 0.66* | 0.66* | 0.63* | C6 0.63* ; others —   |
| bootstrap 95% CI                        |[.63,.69]*|[.63,.68]*| — |[.60,.66]*| C6 [.60,.66]* ; — |
| entail_ok (on SEL-trained text)         | 0.98–1.00* | ~1.0* | 0.997* | 0.995* | —                |
| learned curvature                       | 1.032 | 0.947 | 0.951 | 0.947 | —                     |
| text radius order→species               | 0.11→0.19* | 0.13→0.83* | 0.17→0.55* | — | —             |
| text aperture order→species             | π/2 flat* | 1.57→0.23* | 1.57→0.37* | — | —              |
| nesting slack (ψp−ψc−angle), coarse→deep| −1.4 all* | −1.33/−0.74/−0.09* | −0.82/−0.41/−0.27* | — | — |
| image transitivity (in order/fam/gen/sp cone)| 0.31/0.56/0.84/0.89* | — | — | — | —      |
| image–species Lorentz dist              | 0.314* | 0.575* | — | — | —                       |
| image radius (mean/std)                 | 0.82/0.05* | 1.45/0.05* | 0.76/0.02* | — | —             |
| image direction mean pairwise-cos (angular spread) | 0.032* | — | 0.857* | — | —          |
| inter-species proto Lorentz dist (separability) | 1.07* | 1.39* | — | — | —               |

## Key findings the table encodes
- **entail_ok was a metric bug** (measured on wrong text form); SEL achieves ~0.99 on its own text.
- **B0 origin-collapse**: all ranks tiny radius, apertures saturated π/2 → no nesting possible
  (slack −1.4). Cumulative-SEL (C1/C5) escapes this for DEEP ranks (aperture →0.23/0.37) but
  coarse ranks stay collapsed.
- **Image entailment NON-transitive** in B0 (in-species 0.89 but in-order 0.31).
- **angle-CL angular-collapse** (C5 image-dir cos 0.857 vs B0 0.032) — images pile onto one ray.
- **Subsample top-1 diffs are within noise** (CIs overlap) — don't conclude from those; use F1.

## TODO when GPUs free — full diagnostic sweep
Re-run all geometry probes on the **v3 r64 checkpoints** (and any --sel-margin runs), on the
FULL test_seen split, for EVERY variant (B0..C10 + euclidean + margin), filling all `—`:
1. per-rank radius + aperture (text, on each config's trained SEL text form)
2. nesting slack per edge + fraction-nesting
3. image transitivity (image in each ancestor cone)
4. image radius mean/std + image-direction angular spread (mean pairwise cos)
5. image–species distance + inter-species proto separability
6. entail_ok per edge (on the config's SEL text form)
A single script (extend scripts/visualize_horopca.py's encoder, or a new diagnose_geometry.py)
loads each ckpt, encodes full test_seen, dumps all rows -> regenerate this table with robust
(non-`*`) numbers. The --sel-margin runs additionally re-measure slack/transitivity to confirm
the cone-containment term did what the geometry predicts (slack→≥0, transitivity 0.31→↑).
