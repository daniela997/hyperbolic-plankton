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

### What's left if you filter to complete-to-species lineages

To isolate raggedness *within* the plankton domain (same images/backbone), one could keep
only rows where all 7 ranks kingdom..species are non-null. Measured on the real train split
(200k random sample, seed 0, rank columns only — no image decode), 2026-06-08:

- **complete-to-species = 5.99%** of train → projected **~105k rows** of 1,755,473. Workable
  size (larger than BIOSCAN's 36k train_seen), BUT:
- **only 100 distinct species / 77 genera** — a narrow, shallow tree (cf. BIOSCAN's thousands
  of species). Heavy head: top species "delicatula" = 24% of complete rows in the sample; 12
  singleton species.
- The surviving rows are a **source-biased slice**: complete-to-species lives almost entirely
  in the few sources that annotate that deep (whoi: species 11%; medplanktonset: species 39%).
  So "complete planktonzilla" ≈ "the whoi + medplanktonset subset" — NOT representative of
  planktonzilla's cross-source diversity.

Implication: a complete-only filter is a valid **stability probe** (does curv collapse when
lineages are complete, dataset held fixed?) but a poor **training set / unseen basis** — the
100-class, 2-source space is degenerate, and holding out a *source* (our unseen definition)
while keeping complete-only would gut the set. BIOSCAN already provides the complete-taxonomy
control with a proper seen/unseen split, so the complete-planktonzilla probe is only worth
running if a plankton-domain/backbone × raggedness interaction is specifically suspected.
**Deferred for now (2026-06-08); stats recorded here.**

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

## Per-term map: what raggedness does to each loss (2026-06-09)

Walking through every loss term on three REAL train lineages of different depths, marking
what is correct-by-design vs an actual bug. Anchors:

- **DEEP** (7): `chromista ciliophora heterotrichea heterotrichida spirostomidae spirostomum ambiguum`
- **MID** (5): `animalia arthropoda copepoda calanoida acartiidae`
- **SHALLOW** (2): `bacteria cyanobacteria`

### CL (image ↔ deepest cumulative `full` text)
Each image anchors to a DIFFERENT-depth node (DEEP→species far out, MID→family mid,
SHALLOW→phylum near origin). The positive target is each sample's own depth.

- ✓ **Varying-depth anchor is CORRECT, not a problem.** A common confusion: "distance-CL
  wants all copepod images at one radius." It does not — a family-deep and a species-deep
  copepod are **different `full` strings = different classes**, each with its own positive
  target at its own depth. There is no single "copepoda class" spanning depths. (Earlier
  framing of a separate "varying-radius" problem was wrong and is retracted.)
- ✗ **Ancestor false-negative (the real, ragged-specific CL bug).** In-batch negatives
  ignore lineage. A SHALLOW lineage that is a strict PREFIX of a DEEP one in the same batch
  is a true relative, yet CL repels them. Measured: **82 distinct ancestor lineage-pairs in
  one B=128 batch.** Real examples:
  - `animalia arthropoda copepoda` (family-deep image) repelled from
    `animalia arthropoda copepoda cyclopoida oncaeidae oncaea` — the shallow image *is* a
    copepod; the deep text *is* a copepod.
  - `bacteria cyanobacteria` repelled from
    `bacteria cyanobacteria cyanophyceae oscillatoriales microcoleaceae trichodesmium`.
  - `animalia chordata appendicularia` repelled from its own descendants
    `...copelata fritillariidae` and `...copelata oikopleuridae`.
  **`--cl-mask same` does NOT catch these** — they are different strings, not identical.
  This bug is RAGGED-SPECIFIC: in complete BIOSCAN every lineage is depth-7, so a shallow
  ancestor never appears as a standalone sample → no in-batch ancestor pairs. Would need an
  ancestor/prefix-aware mask to address.

### SEL-intra (consecutive ranks, `T_r`)
Edge count varies by sample: DEEP supplies all 6 edges, MID supplies 4 (stops at family),
SHALLOW supplies 1 (kingdom→phylum).

- ✓ **Per-sample correct.** A sample only entails edges it has; `_valid` masking adds no
  phantom edges. You cannot entail genus→species for a family-deep sample.
- ⚠️ **Dataset-level deep-edge STARVATION (real, unquantified open issue).** Aggregate edge
  supervision is wildly uneven with depth: kingdom→phylum sees ~100% of samples;
  **genus→species sees only ~6%** (the species-complete fraction). So the deep cones are
  shaped by a fraction of the data the shallow cones get. `sel_intra` normalises each
  *sample's* loss by *its* #edges — this does NOT fix the cross-sample imbalance (deep edges
  still accumulate far fewer gradient contributions overall).
- ⚠️ **Why cumulative text may help raggedness here specifically.** With INDEPENDENT `T_r`,
  each deep edge ("Genus: X" ⊂ "Family: Y") is data-starved in isolation. With CUMULATIVE,
  the deep edge's text ("...family genus" ⊂ "...family") SHARES A PREFIX with the
  well-supervised shallow part, so prefix-sharing transfers shallow supervision down to the
  starved deep edges. This is a concrete, ragged-specific reason cumulative could beat
  independent — testable: if cumulative (C1/C2) beats independent (B0) specifically at the
  DEEP ranks, this is the mechanism. (Worth a per-rank readout.)

### SEL-inter (image ↔ deepest text `T_R'`)
Image entailed by its deepest node: DEEP→tight species cone, MID→wide family cone,
SHALLOW→huge phylum cone.

- ✓ **Correct by design — raggedness handled gracefully.** Image localisation scales with
  annotation depth: deep image tightly placed, shallow image loosely placed in a wide cone.
  This IS the uncertainty encoding. The barely-constrained shallow image contributes little
  gradient — appropriate (we know almost nothing about where it goes). No fix needed.

### Prediction / eval (image → nearest prototype, per-rank truncation)
- ✓ **Correct, verified.** Per-rank truncation scores a sample only at ranks it has; a
  family-deep image is not penalised for "wrong species" (it has none).

### Synthesis — what's open vs settled

| issue | status |
|---|---|
| SEL-inter wide shallow cones | ✓ correct by design |
| eval truncation for ragged depths | ✓ correct, verified |
| CL varying-depth positive anchor | ✓ correct (was wrongly flagged as a problem; retracted) |
| CL same-class false-neg | ✗ bug, fixed by `--cl-mask same` (A1) |
| **CL ancestor false-neg** | ✗ bug, RAGGED-SPECIFIC, **not yet addressed** (82/batch; needs ancestor-aware mask) |
| **SEL-intra deep-edge starvation** | ⚠️ real, unquantified; cumulative text *may* mitigate via prefix-sharing (C1/C2) |

**The unifying observation:** raggedness creates a **depth gradient in supervision density**
— shallow ranks are data-rich, deep ranks data-poor. The two open problems are both
symptoms: SEL-intra starves the deep edges; CL mistreats the shallow-ancestor / deep-
descendant pairs that *only exist because* depths vary. The **cumulative text form** is
interesting precisely because prefix-sharing is a natural channel to move information from
the data-rich shallow ranks to the data-poor deep ones.
