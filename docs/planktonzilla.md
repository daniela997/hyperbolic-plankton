# Planktonzilla — Reference Understanding

> Shared notes on the Planktonzilla paper, dataset, and codebase, written so both
> of us can reference a single source of truth. Everything here is grounded in the
> paper PDF (`/home/daniela/other/planktonzilla/planktonzilla.pdf`), the code repo
> (`/home/daniela/other/planktonzilla/`), and the live HF dataset metadata.
>
> Confidence is flagged inline: **[stated]** = explicit in paper/code,
> **[inferred]** = strong reasonable inference, **[unknown]** = not yet confirmed.

---

## 1. What Planktonzilla is

- **Paper:** "Planktonzilla: Multimodal dataset and models for understanding
  plankton ecosystems" (Inria Chile / OcéanIA project). A dataset + benchmark
  paper. **[stated]**
- **`Planktonzilla-17M`:** a harmonized dataset aggregating **17.4M images from 13
  imaging systems**. Of these, **3.74M are confirmed plankton** (`plankton==True`);
  the rest are detritus/artifacts kept for out-of-distribution work. **[stated]**
- **Taxonomy is harmonized against WoRMS** (World Register of Marine Species) to the
  **deepest *valid* rank**, not forced to species. Only ~6% of plankton samples have
  a full species annotation; ~30% reach genus. Evaluations are therefore done at the
  **deepest valid rank available per sample** (ragged labels). **[stated]**
- **Goal:** a unified, citable benchmark + pre-trained models for plankton
  classification, addressing cross-instrument generalization and inconsistent labels
  across source datasets. **[stated]**

---

## 2. The HF dataset (what we can actually load)

`project-oceania/planktonzilla-17M` — single `train` split, **17.4M rows, ~91GB**
parquet (images embedded). Confirmed schema from the datasets-server `info` endpoint:

| Column | Type | Use |
|---|---|---|
| `image` | Image | the image |
| `dataset` | string | **source dataset** — used to build the held-out/unseen split |
| `original_label` | string | raw per-source label |
| `proposed_label` | string | **WoRMS-harmonized label** (deepest valid rank) — our class identity |
| `root_class` | string | coarse class |
| `Kingdom,Phylum,Class,Order,Family,Genus,Species` | string | **full per-rank lineage** |
| `plankton` | bool | **filter to the 3.74M plankton subset** |
| `qualifier` | string | annotation qualifier |
| `Latitude,Longitude,Depth_min,Depth_max,Temperature,Humidity` | float32 | geo-environmental metadata (unused for now) |
| `ObjID` | string | object id |

**Key consequence:** the per-rank lineage (Kingdom→Species) **already ships in the
dataset** — no WoRMS querying needed. This is what makes hierarchical/entailment
losses feasible. **[stated — confirmed from live metadata]**

Class label choice: we use **`proposed_label`** as the class ("Folder" equivalent)
because it's WoRMS-harmonized and cleaner than `original_label`. Note it sits at the
**deepest valid rank**, so the class space is taxonomically heterogeneous (some
classes are genus-level, some species-level). **[decision]**

---

## 3. The two prediction mechanisms (the conceptual core)

A classifier needs a rule to turn an image into a class. The paper compares two
fundamentally different ones:

### A. Supervised classifier ("Standard Classification")
- Backbone → **linear head, one neuron per class** → softmax → argmax. **[stated]**
- Labels are **integer indices**; the model never sees taxonomy text.
- Trained with **cross-entropy**.
- **Hard limit:** the class set is baked into the architecture. **Cannot classify a
  class it never trained on** (no output neuron) → zero/few-shot is *structurally
  inapplicable*. **[stated]**

### B. CLIP-style (contrastive)
- Image → embedding; each class described by **taxonomic text** → embedding; predict
  the class whose **text embedding is most similar**. **[stated]**
- Trained with **contrastive image–text loss** (lineage as text).
- **Unlocks zero-shot:** a new class is added just by embedding its name as text — no
  new neuron, no retraining. This is what enables unseen-species classification.

> **This is why our hyperbolic method lives in mechanism B** — it matches images to
> taxonomy-text embeddings, so it can do unseen classes. A supervised softmax head
> cannot compete in the unseen setting at all.

---

## 4. The two experiments

### Experiment 1 — In-domain (Table 2)
- **Same class set** for train and test. Plankton subset, exclude the 4 held-out
  datasets, then **stratified 60/20/20** split (by source dataset AND taxonomic
  label). **[stated]**
- Compares CLIP-style vs supervised, on shared backbones (ViT-B/16, ViT-L/14).
- Zero/few-shot "not applicable" here **for the Planktonzilla-trained models** —
  but the table *does* include off-the-shelf BioCLIP zero/few-shot rows as a
  reference floor on the same in-domain test classes. **[stated + inferred]**
- **Finding:** the **fully-supervised classifier wins in-domain** (best at every
  rank). Encoding taxonomy as text does NOT help in the fully-supervised in-domain
  setting. **[stated]**

### Experiment 2 — Out-of-domain / unseen species (Table 3)  ← **OUR ARENA**
- Hold out **4 source datasets**: **GlobalUVP5, PlanktoScope, PlanktonSet1.0,
  SYKE-IFCB-2022**. From these, select **220 plankton classes (113,089 samples)**
  absent from training. **[stated]**
- **Restricted to CLIP-based models** (so a fair comparison with BioCLIP/BioCLIP2).
  Supervised classifier is absent — it's structurally barred from unseen classes.
- Few-shot uses **SimpleShot** (nearest support-image centroid); 1- and 5-shot are
  **averaged over 5 random seeds**. **[stated]**
- **Finding:** CLIP models fine-tuned on Planktonzilla beat off-the-shelf
  BioCLIP/BioCLIP2 on unseen classes (consistent but moderate gains). **[stated]**

---

## 5. "Zero-shot" — the precise meaning (a common confusion)

"Zero-shot" here means **zero examples of the specific held-out classes**, NOT zero
plankton exposure. Two senses get blurred:

| Setting | Plankton domain seen? | These exact classes seen? | Where |
|---|---|---|---|
| Off-the-shelf BioCLIP "zero-shot" | **No** (general bio) | No | Table 3 baseline row |
| Planktonzilla-CLIP "zero-shot" on held-out classes | **Yes** (3.74M plankton) | **No** | Table 3 `+Planktonzilla` rows |
| Planktonzilla-CLIP in-domain | Yes | **Yes** | Table 2 standard |

The paper's contribution is row-2 > row-1: domain fine-tuning helps even for
unseen species. Our hyperbolic method also lives in **row 2** (train on plankton
domain, test on unseen classes). The "shots" refer only to how many labeled
examples of the *held-out* classes you get at eval (0 / 1 / 5).

---

## 6. The models and inits (decoding the table rows)

The paper trains **four models from four inits**, two architectures:

| Init | Arch | Off-the-shelf row? | Fine-tuned `+Planktonzilla` row? |
|---|---|---|---|
| OpenAI CLIP | ViT-B/16 | ✗ none | ✓ Tables 2 & 3 |
| **BioCLIP** | ViT-B/16 | ✓ Table 3 | ✓ Tables 2 & 3 |
| LAION-2B | ViT-L/14 | ✗ none | ✓ Tables 2 & 3 |
| **BioCLIP2** | ViT-L/14 | ✓ Table 3 | ✓ Tables 2 & 3 |

- "BioCLIP" appears in **three roles**: (a) an init they fine-tune, (b) an
  off-the-shelf zero-shot baseline, (c) BioCLIP2 plays both for ViT-L/14. **[stated]**
- **Only bio-pretrained inits get an off-the-shelf baseline row.** Generic CLIP
  (OpenAI/LAION) has **no untrained zero-shot number** in the paper — they only
  appear after fine-tuning. (Generic CLIP on plankton would be near-random and isn't
  an interesting biological-foundation-model baseline.) **[stated + inferred]**
- The **"Supervised classifier" row is a separate, independent model** (own
  hyperparameter table, own training run) — **NOT** a head bolted onto the
  contrastive checkpoint. Evidence: single supervised row per backbone *size* (not
  per init), separate Table 6 hyperparameters, separate 2–3.5h runtime. **[inferred,
  strong]**

---

## 7. Training recipes (Appendix B, Tables 5 & 6)

| Hyperparameter | CLIP-based (Table 5) | Supervised (Table 6) |
|---|---|---|
| Input size | 224×224 | 224×224 |
| **Batch size** | **16,384** | **64** |
| Learning rate | 1e-4 | 1e-4 |
| Warm-up steps | 1,000 | 1,000 |
| Scheduler | Cosine decay | Cosine decay |
| **Max epochs** | **100** | **20** |
| Weight decay | 0.2 | 0.2 |

**Compute (Appendix B):** **[stated]**
- Building the full dataset: ~3 weeks on 1 node (48 CPU cores, 512GB RAM).
- CLIP-based model: **10–15h per run** on **64× NVIDIA H100**.
- Supervised baseline: 2–3.5h per run.
- Total across all configs: ~55h on 64× H100.

> **The batch-size 16,384 on 64× H100 is the single most important number for us.**
> Contrastive learning's strength scales with in-batch negatives. Any gap between our
> numbers and theirs is partly a batch-size confound — which is exactly why our claim
> is **compute-efficiency**, not a controlled accuracy win. See `project-plan.md`.

---

## 8. Evaluation protocol (what makes numbers comparable)

- **Metric: Macro-F1 at each taxonomic rank** (Kingdom→Species). **[stated]**
- Labels **truncated at each level**; a sample is **evaluated only at ranks where it
  has a valid annotation** (ragged). **[stated]**
- CLIP-based predictions: **image–text similarity** (nearest class-prototype text).
- Supervised predictions: softmax over class logits.
- As eval moves to finer ranks, performance reflects ability to distinguish finer
  taxa within the same parent. **[stated]**

> ⚠️ **Macro-F1, not macro-accuracy.** F1 needs per-class precision AND recall.
> (Our current `evaluate.py` computes macro-recall — a known gap to fix.)

---

## 9. The codebase (`/home/daniela/other/planktonzilla/`)

- **Hydra-configured** pipeline; HF `datasets` + `timm`/`open_clip` backbones + HF
  `Trainer`. CLI: `pz_import_dataset`, `pz_train`, `pz_push_model`. ~2.2K LOC.
- `planktonzilla/train.py`: supervised path via `AutoModelForImageClassification`;
  `try/except` fork falls back to `ClipClassifier` for the CLIP path.
- `planktonzilla/clip_model.py`: `ClipClassifier` wraps an `open_clip` visual
  encoder + linear head (this repo's "CLIP" = CLIP-pretrained *backbone*; the actual
  **contrastive** training was run through **OpenCLIP** via `scripts/train_clip.sh`,
  not `pz_train`). **[stated]**
- The contrastive CLIP text captions (taxonomic lineage as a single `.txt` string per
  image in webdataset shards) are built by a cluster-side notebook
  (`notebooks/save_planktonzilla2.py`) **not present in the repo**. **[unknown:
  exact caption format/separators/template]**
- `planktonzilla/loss.py`: imbalanced-learning losses (Focal, LDAM, Asymmetric,
  etc.) for the supervised path.

---

## 10. Open / unverified items

- **[unknown]** Exact text-caption format used for contrastive training (lineage
  join string vs templated prompt) — built off-repo.
- **[unknown]** Exact pretrained init for the supervised classifier row (almost
  certainly fine-tune-from-pretrained, but not literally stated).
- **[unknown]** Whether the live HF `dataset` column strings exactly match the paper
  names "GlobalUVP5 / PlanktoScope / PlanktonSet1.0 / SYKE-IFCB-2022" (need to dump
  unique values before hardcoding the held-out split).
- Planktonzilla v1 trained checkpoints not yet public — for now we compare against
  the **paper's reported Table 2/3 numbers**.
