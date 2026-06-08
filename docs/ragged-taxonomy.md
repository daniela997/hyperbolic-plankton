# Ragged Taxonomy — Data Statistics & Consequences

Plankton taxonomy labels are **ragged**: each sample is annotated to *some* depth, then
truncated. This doc records the measured depth statistics of our data and the consequences
for the model/losses. Raggedness is the central way our setting differs from the references
(MERU/HAC have no taxonomy; the Hyperbolic-Taxonomies paper uses BIOSCAN, which is *complete*
to species). **Verify claims against the real `train_idx` split — a source-ordered slice
gives badly biased depth stats (see the retraction below).**

## Measured statistics (real train split)

Source: `/scratch/daniela/hyperbolic_plankton_splits/train_idx.npy` (1,755,473 rows; 400k
random sample, seed 0, via `build_taxonomy`). Date: 2026-06-08.

### How deep do lineages go? (deepest valid rank = the CL anchor target)

| rank | **deepest** (this is the leaf) | **present** (annotated at all) |
|---|---|---|
| kingdom | 1.4% | 100.0% |
| phylum | 7.7% | 98.6% |
| class | 16.0% | 90.9% |
| order | 23.0% | 74.9% |
| family | 19.8% | 51.9% |
| genus | 26.0% | 32.1% |
| species | 6.0% | 6.0% |

- **Only ~6% of samples reach species** (matches the Planktonzilla paper's "~6% species,
  ~30% genus"). `present[genus]` = 32% ≈ the paper's ~30%-to-genus.
- **The deepest rank is spread broadly** across order(23%)/genus(26%)/family(20%)/class(16%)
  — NOT concentrated on one or two ranks. So contrastive learning (which anchors each image
  to its *deepest* cumulative text) anchors a wide mix of ranks, not just leaves.
- `present` decays monotonically (100% → 6%): raggedness is pure leaf-truncation.

### Lineages are contiguous (no internal gaps)

**100.0% of lineages are contiguous** — if a rank is missing, *every* deeper rank is also
missing (truncation), never an internal hole (e.g. has-kingdom-and-class-but-no-phylum does
NOT occur). This validates the cumulative-encoding + `_valid_ranks` design: the valid ranks
are always a prefix `kingdom..R` for some depth R.

### Raggedness is strongly source-dependent

Each source dataset truncates at a characteristic depth:

| source | n | deepest mostly |
|---|---|---|
| zooscan | 135,702 | family 46%, order 22% |
| zoocamnet | 117,828 | order 47%, genus 27% |
| whoi | 68,128 | genus 74%, species 11% |
| jedioceans | 32,595 | class 72%, phylum 24% |
| flowcamnet | 13,984 | phylum 48%, genus 16% |
| medplanktonset | 10,400 | genus 46%, species 39% |

Implication: the seen/unseen split (held-out *sources*) also shifts the depth profile — the
4 held-out datasets have their own raggedness, so unseen eval is at characteristic depths.

### Branching factor (avg #children per parent, full-data estimate)

| edge | #distinct parents | avg branch | max branch |
|---|---|---|---|
| kingdom→phylum | 3 | 5.33 | 10 |
| phylum→class | 14 | 2.00 | 5 |
| class→order | 22 | 2.23 | 6 |
| order→family | 38 | 1.34 | 3 |
| family→genus | 36 | 1.22 | 3 |
| genus→species | 11 | 1.00 | 1 |

Branching is **highest at the top** (kingdom→phylum 5.33) and decreases with depth — it does
NOT hump in the middle.

### Tree topology: nodes that are BOTH leaf and parent, and DEAD-END lineages

Because depth varies per sample, a node can be **a leaf** (the deepest rank for some samples
→ gets images via SEL-inter) AND **a parent** (other samples go deeper → gets child-text via
SEL-intra). Counts of distinct nodes per rank (400k sample):

| rank | #nodes | #leaf | #parent | #BOTH |
|---|---|---|---|---|
| phylum | 24 | 14 | 21 | 11 |
| class | 49 | 22 | 43 | 16 |
| order | 111 | 32 | 95 | 16 |
| family | 167 | 45 | 133 | 11 |
| genus | 177 | 129 | 83 | **35** |
| species | 112 | 112 | 0 | 0 |

Dozens of internal nodes are **both** — e.g. 35 genus nodes hold both an image (genus-deep
samples) and a deeper species-text (species-deep samples) in the same cone. These signals
REINFORCE (both say "be a well-formed parent cone"), so it's healthy, not contradictory.

**Dead-end lineages** — a "leaf" node that NO sample ever extends deeper (an annotation
artifact, a shallow terminal node, not a biological species). Fraction of leaf-anchored
images sitting at a dead-end, by leaf rank:

| leaf rank | leaf imgs | dead-end imgs | % dead-end |
|---|---|---|---|
| phylum | 30,700 | 9,285 | 30.2% |
| class | 64,061 | 2,416 | 3.8% |
| order | 92,197 | 12,413 | 13.5% |
| family | 79,239 | 66,116 | **83.4%** |
| genus | 104,014 | 48,168 | 46.3% |
| species | 24,199 | 24,199 | 100% (by def.) |

**Key fact: 83% of family-deep and 46% of genus-deep images are dead-ends** — they anchor at
nodes nothing extends past. These images nest in **shallow, wide cones** (family ≈ 0.37
radius, wide aperture) and CANNOT be tightly localized (no finer structure exists for them).
This is a large, correctly-but-loosely-placed image population, intrinsic to ragged plankton
taxonomy. (It does NOT happen with complete taxonomy like BIOSCAN, where species is always
the leaf.)

## Consequences for the construction

- **CL anchors images to the deepest cumulative text**, which (per the table) is a broad
  rank mix — so most rank embeddings get *some* direct image signal across the dataset.
- **Cumulative encoding + contiguity** ⇒ the per-rank text embeddings form a literal
  taxonomy *tree* (shared-prefix lineages give byte-identical embeddings up to divergence;
  see build-log "construction correctness"). Sibling closeness comes from this prefix
  sharing, not from CL (CL is flat — treats siblings like strangers).
- **SEL-inter entails the image by its deepest text** = a node of varying depth per sample;
  shallow-labeled images nest in *wider* (shallower) cones ⇒ less tightly localized, which is
  correct (less label info ⇒ less constraint).

## OPEN QUESTION — the mid-rank positive-loss hump (UNEXPLAINED)

Observed (curv-slow run, it22200, logged `loss_terms/*`): the SEL-intra **positive** loss
humps in the middle:

| edge | pos | neg |
|---|---|---|
| kingdom→phylum | 0.039 | 0.002 |
| phylum→class | 0.034 | 0.000 |
| **class→order** | **0.661** | 0.001 |
| **order→family** | **0.746** | 0.000 |
| family→genus | 0.108 | 0.002 |
| genus→species | 0.146 | 0.000 |

`class→order` and `order→family` have ~5–20× the positive loss of the other edges.
(Separately: **all negative terms ≈ 0** — non-members are already outside the cones by the
margin; the SEL negatives are inactive at convergence, intra AND inter. So the
"are negatives awkward?" debate is empirically moot in steady state — see build-log.)

**Hypotheses tested and REFUTED:**
- *Branching factor* — refuted; branching is highest at the top, not the middle.
- *Annotation-depth orphan* ("order is rarely the deepest, so CL never anchors it") —
  **REFUTED on real data.** This was based on a biased source-ordered 300k slice that gave
  order=2.2% deepest; the real train split gives **order = 23% deepest** (well-anchored).
  Lesson: always compute depth stats on a random sample of the real split.
- *Dead-end image load* ("mid-ranks carry the most dead-end image anchors, pinning their
  cones wide") — **REFUTED.** Dead-end % does NOT correlate with the hump: `genus→species`
  has 100% dead-end load but LOW loss (0.15); `class→order` has only ~13% dead-end load but
  HIGH loss (0.66). So dead-end load is a real data property (above) but not the hump cause.

**Still-plausible, NOT yet verified:**
- *Radius regime*: mid-rank parent cone width vs child angular spread is the unlucky middle
  (moderate aperture AND genuine directional divergence).
- *Visual divergence*: order/family is biologically where body-plans diversify within a
  class, so image-anchored leaves spread most there, making those cones hardest to satisfy.

**Measurement that would settle it** (needs a GPU; run when free): for each edge on the
trained model, plot the distribution of `oxy_angle` vs `half_aperture` over the positive
pairs. If the hump is driven by large `oxy_angle` → children genuinely diverge in direction;
if by small `half_aperture` → parent cone too tight. That decomposes the hinge and pins the
cause empirically rather than by hypothesis.
