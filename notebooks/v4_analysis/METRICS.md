# v4 metrics glossary

Every metric in the v4 summary table (`v4_summary_table.txt`) and the per-run `diag_*.txt` files,
grouped by what it measures. Produced by `scripts/diagnose_geometry.py` (geometry) + wandb (F1).

## Classification quality — the actual goal

| metric | what it is | good | notes |
|---|---|---|---|
| **seenF1** | species macro-F1 on test_seen (trained classes), from wandb eval | high | **THE headline metric.** Everything else is diagnostic for this. |
| **unsF1** | species macro-F1 on test_unseen (novel lineages) | high | generalization to species never seen in training |
| **DIST** | diag top-1 acc, classify by **distance** to species prototype | high | our normal classifier; a proxy for seenF1 but **over-reads cone-heavy runs** (e.g. LS-m0 diag 0.76 vs real F1 0.60) |
| **ANG** | diag top-1, classify by **ATMG angle** (argmin `oxy_angle` to species text) | — | is angle a good classifier for this run? |
| **CONE** | diag top-1, classify by **Dhall cone-energy** (argmin cone violation) | — | do the **cones themselves** classify? ~0 = cones useless/degenerate |

> DIST/ANG/CONE are the diagnostic's own classifiers on cumulative prototypes — trust wandb **seenF1**
> for the honest number.

## Separability — what actually drives classification

| metric | what it is | good | why |
|---|---|---|---|
| **NN** (NN-sep) | nearest-neighbour separation: each species proto's geodesic distance to its **closest** sibling proto | high (0.3–0.7) | **the geometry that predicts F1.** Classification (argmin) is decided by the *nearest confusable* proto, not the average. SEL crushes this to ~0.04. |
| **mean-sep** | mean pairwise inter-species prototype distance (`inter-species proto dist`) | — | **MISLEADS** — global spread, not what the classifier feels. C8 has mean-sep 1.66 (huge) but NN 0.22 (tight) → F1 0.67. Reported only for contrast. |
| per-image margin | per image: d(nearest-wrong-proto) − d(correct-proto); `frac>0` ≈ accuracy | +, high frac>0 | the per-sample version of NN-sep; the direct correlate of top-1 |

## Containment — the cone / hierarchy story

| metric | what it is | good (for retrieval) | why |
|---|---|---|---|
| **inSp** | fraction of images inside their own species cone (`oxy_angle ≤ ψ`) | high | entailment faithfulness. But high inSp with **wide** cones = fake (overlap) |
| **mult** | cone multiplicity: mean # of species cones each image falls inside | **~1** | distinguishes **genuine** containment (mult 1 = image in its own cone only) from **overlap** (mult 6.6 = inside many cones, can't classify). Caught the SEL-only C2 illusion. |
| sat% | fraction of species cones with aperture **saturated at π/2** (half-spaces) | **0%** | saturated cones = degenerate ("containment" is a trivial hemisphere test). 100% = collapse. |
| (psi\*0.5 slack) | does containment survive tightening ψ to ψ·0.5? | > 0 | if ~0, every "contained" image sits *at* the π/2 boundary → degenerate, not genuine separation |

## Collapse — directional / radial health

| metric | what it is | good | why |
|---|---|---|---|
| **ray** | cos-to-mean-ray: how much all image directions align to one shared ray | **low (~0.2)** | the **ray-collapse** metric. 0.8+ = everything strung along one line (hybrid-**angle** pathology). 0.2–0.3 = healthy spread. |
| eff-dim | participation ratio of the centered image SVD spectrum | high (10s) | ~1 = truly 1-D (a literal ray); ~30 = genuinely high-dimensional |
| **imgR** | mean image radius (geodesic distance from origin) | moderate | ~0.003 = origin collapse; too large = pushed to boundary |
| imgcos | mean **pairwise** cosine of image directions | low | older directional-collapse metric (ray is the sharper version) |

## Radial ordering — is the hierarchy laddered outward?

| metric | what it is | good | why |
|---|---|---|---|
| **ordR** | mean radius of **order** (coarsest) prototypes | small (< spR) | order should be near the origin (general) |
| **spR** | mean radius of **species** (finest) prototypes | large (> ordR) | species far out (specific). **ordR < spR = correct ordering** — most runs are flat/inverted |
| **spAp** | species aperture ψ (cone half-width) | small (narrow) | ψ ∝ 1/(√c·r), so small spAp ⟺ species far out. 1.571 = π/2 = saturated |
| curv | learned curvature c | (context) | affects aperture (ψ ∝ 1/√c·r) + angular budget. We've only ever learned c≈0.8–1.3; feasibility of tight non-overlapping cones needs c≳4 |
| edge fits% | fraction of parent→child edges where the child's *whole cone* nests in the parent | — | whole-cone containment (transitivity); 0 = only apex contained, not the cone |

## The three recurring reading patterns

1. **Good classifier** — high NN, high DIST/seenF1, inSp=0 (no cones), low ray.
   → *plain hybrid (0.765), C4, LRCL.* Uses a tight ball efficiently; ignores hierarchy.
2. **Faithful-but-useless containment** — high inSp, high **mult**, low NN, sat 100%.
   → *SEL-only C2/C3.* Contained but can't discriminate (inside many cones).
3. **Collapsed** — imgR→0 **or** ray→0.8, sat 100%, CONE→0.
   → *angle-CL+SEL (C5/C6), hybrid-angle.* Everything on a ray or at the origin.

## The one-line summary

Classification follows **NN-sep**, which is **scale-invariant** (a tight ball with good relative
separation classifies as well as a spread one). **Global spread (mean-sep, HoroPCA appearance) ≠
classification.** Every hierarchy mechanism (cones/SEL, angle-CL, even radial) is a **tax on NN-sep**,
never a gain — so no config gets good classification *and* good hierarchy at c≈1. See the memory notes
`nn-separability-not-mean`, `containment-vs-discrimination`, `angle-collapses-coarse-ranks`.

---

## Note: what exactly-d (RINCE grading) does in hybrid, vs dedup (LRCL)

They fix DIFFERENT halves of the multi-image problem (not redundant):

- **dedup** (LRCL) — on the TEXT/negative axis: one unique prototype per class, so same-species
  images are never false NEGATIVES. Fixes the InfoNCE multi-image problem on the negative side.
- **exactly-d** (RINCE grading) — on the IMAGE/positive axis of T→I. A coarse (genus) prototype's
  T→I positives are ALL its images via `-logp[u, posmask].mean()` (loss.py:311). That mean is
  **per-image**, so it is ABUNDANCE-WEIGHTED: if species-A has 50 imgs and B/C have 3+2, the genus
  prototype is pulled 50/55 toward species-A → genus collapses onto its dominant child. Dedup does
  NOT fix this (positive images aren't deduped). exactly-d DROPS the representative (dominant) species'
  images from the group → genus pulled toward the cross-species residual, staying distinct.

Measured (dist genus_text → its dominant-species text): hybrid 0.212 > LRCL-all 0.176 — exactly-d
keeps genus ~20% further from its biggest child, confirming the mechanism (modest effect, coarse-rank
only). NB the "group-balanced" comment is a per-IMAGE mean, NOT per-sub-species balancing — abundance
still weights within the remaining group. exactly-d is largely redundant for SPECIES F1 (dedup does
that), but matters for keeping COARSE prototypes genuinely coarse (hierarchy).
