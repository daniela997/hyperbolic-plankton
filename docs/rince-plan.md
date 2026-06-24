# RINCE for taxonomy — feasibility + implementation plan

**Idea.** Replace the binary InfoNCE positive/negative split (and the cl-mask patch) with RINCE
(Hoffmann 2022, *Ranking InfoNCE*): a query's batch-mates are **graded positives by taxonomic
shared-depth**, not one-positive-vs-all-negatives. Same-species ≻ same-genus ≻ same-family ≻
same-order ≻ shares-nothing(true negative). Injects the hierarchy **into the contrastive objective**
— a clean alternative to SEL entailment cones (which hit the curvature-feasibility obstruction).

## Feasibility (measured on BIOSCAN train_seen, CPU)

RINCE needs enough graded positives *per query in the negative bank*. Per-query coverage (≥1 partner
at each depth):

| level | B=64 (micro-batch) | **B=768 (cache-accum bank — what RINCE sees)** |
|-------|-----|------|
| same species | 27% | **73%** |
| same genus   | 21% | **47%** |
| same family  | 63% | **88%** |
| same order   | 97% | **100%** |

At B=768: **mean 3.07 of 4 ranked levels populated per query**; only 0.1% of queries have 0 levels
(would degrade to plain InfoNCE). So RINCE is well-supported — **but only with the big effective
batch**. Their reference uses a 4096+ memory bank for exactly this; for us **cache-accum (768) is the
prerequisite** (plain micro-bs=64 is too sparse: the 27%/21% row). Build RINCE ON cache-accum.

## What to port from /home/daniela/other/rince/losses.py (and what to drop)

KEEP (the actual Eq. 3/5 math, ~20 lines):
- `sum_in_log` = RINCE-in (Eq. 3): `softmax(cat[pos,neg]/τ); -log(pos_mass).mean()`; empty levels
  handled by the `>1e-7` filter (matches our sparse fine ranks gracefully). Our density check chose
  the -in variant.
- The per-rank loop (forward l.102-141): at level i, set same-or-more-similar items to -inf in the
  negatives, keep rank-i members as positives, sum over levels = Eq. 5's cumulative denominator.
- `get_dynamic_tau`: τ = min_tau + (1-sim)·(max_tau-min_tau); coarser rank → higher τ (paper's τ_i<τ_{i+1}).

DROP (CIFAR/MoCo baggage, not needed):
- MoCo momentum encoder + memory bank (`backbone_k`, `memorybank_*`, enqueue, update_weights) →
  replaced by in-batch cache-accum negatives (768).
- `set_super_cat_sims` / CIFAR class table / `get_similar_labels` → replaced by taxonomy shared-depth.
- `similarity_threshold`/`below_threshold` noise machinery → our ranks are EXACT, no thresholding.

## Design (decided by data + user)

- **Similarity by depth (`--rince-sim linear`)**: sim = shared_depth/4 → species 1.0, genus 0.75,
  family 0.5, order 0.25, nothing 0 (extends RINCE's 0.75 superclass scheme to 4 levels).
- **`--rince-mode one-per-rank`** (their best, Table 2 RINCE-in / out-in): one ranking term per
  distinct depth (≤4 terms), each its own τ. Mean 3.07 real terms/query at B=768.
- Negatives = in-batch cache-accum bank (768); `h(q,p)` = our `-pairwise_dist` (or `oxy_angle`) so it
  drops into the *_ddp / accum-CL structure. Hyperbolic AND euclidean compatible (swap h).
- Shared-depth `[B,B]` matrix from `taxonomy_batch` lineages — same machinery as `_false_negative_mask`,
  generalised binary→graded.

Implement as `--contrastive ranked` (3rd option alongside distance/angle). Compare vs B0 (plain CL),
cl-mask (binary false-neg removal), and SEL — RINCE is the graded-positive contrastive alternative
to all three, and is geometry-agnostic (no cone feasibility problem).
