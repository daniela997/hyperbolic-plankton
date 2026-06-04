# Project Plan — Hyperbolic Plankton

> Our approach, thesis, target venue, and experiment design. Companion to
> `planktonzilla.md` (facts about the paper/dataset) and the hyperbolic codebase at
> `/home/daniela/mine/hyperbolic/`.

---

## 1. Thesis / claim

> **Via hyperbolic approaches we can match or outperform the Planktonzilla CLIP
> models on unseen plankton species, at a fraction of the training compute.**

This is a **compute-efficiency** claim, deliberately. We are NOT claiming a
controlled "hyperbolic beats Euclidean" geometry win (that needs a matched-compute
radial baseline — deferred). We claim: a **frozen-backbone + trainable hyperbolic
projector (+ optional LoRA)** on **2× RTX A5000 (24GB)** reaches the unseen-species
performance the paper got with **64× H100, batch 16,384, 100 epochs**.

**Precedented, not speculative:** HAC 2026 (see `related-work.md`) already showed a
*frozen* CLIP lifted into hyperbolic space via PEFT **matches/surpasses a
fully-trained hyperbolic model (HyCoCLIP) with 85× fewer trainable params and 2/3 the
data**, training on a *single* GPU vs multi-GPU from-scratch. We transplant that
efficiency recipe onto taxonomic SEL (the 2025 paper) + plankton — an unfilled cell.

### Scope discipline (must not overclaim)
- **Winnable:** unseen species / out-of-domain (paper Table 3). Low baseline
  numbers, big headroom, hyperbolic rank-fallback should help.
- **NOT winnable, do not claim:** in-domain supervised classifier (Table 2, species
  ~0.867). A frozen-projector model on 2 GPUs will not beat a full supervised head
  in-domain. The claim is **scoped to unseen generalization**.
- **Honest fallback** (if numbers land below the paper): "hyperbolic entailment
  closes much of the gap at a fraction of the compute" — still a valid
  efficiency-tradeoff result for a workshop.

### Why hyperbolic should help on unseen species
For an unseen species we may fail at the species rank but still **match a correct
higher rank (genus/family) via entailment** — the embedding lands inside the parent's
cone even when the exact leaf is novel. This is read directly off the **coarse-rank
columns of the per-rank macro-F1** on held-out classes.

> Caveat we keep in mind: entailment-based rank fallback is **not unique to
> hyperbolic space** — RCME does it in radial-Euclidean space and beats MERU. So a
> *controlled* geometry claim would require an RCME-style baseline. We are explicitly
> deferring that; v1 is the efficiency claim only.

---

## 2. Target venue

**ECCV 2026 Workshop — "Beyond Euclidean: Hyperbolic Deep Learning for Computer
Vision"** (3rd edition), Malmö, Sweden, Sept 8–9 2026.
<https://sites.google.com/view/beyondeuclidean/>

Scope explicitly includes **hierarchical embeddings** and **vision-language models** —
direct fit.

### Tracks & dates
- **Track 2 (our target): non-archival, ≤4 pages**, ECCV format preferred-not-
  required. Scope: "early-stage results, insightful negative findings, opinion
  pieces, and novel datasets." Non-archival = doesn't burn a future full paper.
- Track 1: archival, ≤14 pages, full paper.
- **Submission deadline: June 24, 2026 (AOE).** Notification Aug 6; camera-ready
  Aug 13. Double-blind via OpenReview.

> v1 plan = **Track 2, 4 pages**. ~3 weeks from now. A controlled radial baseline
> and the L/14 models are the natural Track-1 follow-up.

---

## 3. Compute reality

- **Hardware:** 2× NVIDIA RTX A5000, 24GB each (confirmed via `nvidia-smi`).
- **Env:** `/scratch/daniela/miniconda3/envs/dino_plankton/bin/python`
  (torch 2.10+cu128, open_clip, datasets 2.21, transformers 5.6, CUDA on).
  `fedclip` env also has the stack.
- Implication: **ViT-B/16**, frozen backbone, modest batch (low hundreds, grad-accum
  + cross-GPU negative gather), cached plankton subset. ViT-L/14 likely out of budget
  for v1.

**Compute comparison table (the paper's contribution as much as F1):**

| | Planktonzilla CLIP | Ours (target) |
|---|---|---|
| GPUs | 64× H100 | 2× A5000 |
| Batch size | 16,384 | ~hundreds |
| Epochs | 100 | TBD |
| Backbone | fully fine-tuned | **frozen** + projector |
| Wall-clock | 10–15h/run | hours, 1 node |

---

## 4. Method (v1)

- **Backbone:** frozen **CLIP (OpenAI)** AND **BioCLIP**, both ViT-B/16, via
  `open_clip`. Both inits so we show the efficiency win holds for generic and bio
  pretraining, matching the paper's two B/16 inits.
- **Projector (v1):** a single trainable linear head per modality
  (`visual_proj`/`textual_proj`, `nn.Linear`, MERU init) between the frozen backbone
  and the `exp_map0` lift — **the HAC `AdaptedCLIP` projector**, re-implemented and
  verified in `src/hyperbolic_plankton/model.py` (Piece 2). Backbone frozen → only the
  projection heads + MERU scalars (curv, alphas, logit_scale) train.
  - The scratchpad `model.py` is a *superset* (MLP / depth-factored / parallel-
    transport projector variants); v1 uses only the plain linear one, matching HAC.
  - **Depth-factored projector — Tier-1 experiment, deferred (see §5).** Worth
    exploring because it ties projector output *radius* to taxonomic rank, which is
    exactly the hierarchy signal we care about (hyperbolic radius = specificity).
- **Loss:** hyperbolic contrastive + stacked entailment (SEL), reusing
  `/home/daniela/mine/hyperbolic/loss.py` (already implements SEL-intra/inter,
  MERU contrastive, ragged-rank validity masks, hard negatives, UNCHA).
- **LoRA in backbone:** evidence-based settings from HAC 2026 (see
  `related-work.md`), since HAC is the precedent for our whole efficiency claim:
  - Target **q,k,v,o** attention submatrices (HAC ablation: dropping `o` hurts).
  - Apply to the **last few blocks, text-heavier** (HAC-B: last 4 vision / last 8
    text), not all 12 — saves memory on 24GB.
  - **r = α**: start low (8–16) on A5000; treat rank as a cheap ablation. (HAC-B used
    128, but that's a full-B-with-big-rank result; high rank helped their *bigger*
    model — unclear it transfers to a frozen-projector small-rank budget.)
  - On ViT-B/16 (HAC "B" tier) **LoRA is the indicated PEFT** (adapters won for their
    small tier, LoRA for the bigger).
  - **Also train the final projection heads + final LayerNorm** of each encoder (HAC
    §4.4: the final LN must be trained or it gates the output) — our hyperbolic
    projector is the "projection head"; add the final-LN unfreeze.
  - Requires removing `no_grad` around the frozen backbone forward so adapters get
    gradients (base weights stay frozen). Apply the *same* LoRA config across all
    variants to keep comparisons controlled.
  - **Sequencing:** still **projector-only first** (cheapest, establishes the
    baseline + de-risks data/eval), then add LoRA. HAC shows projector/PEFT-only can
    already beat Euclidean — so projector-only is a legitimate v1 result on its own.

### Reused infrastructure (already exists in `mine/hyperbolic/`)
- `model.py` — Lorentz projector, frozen-backbone forks (TIPS + non-TIPS).
- `loss.py` — full SEL / entailment / contrastive suite. **Mature.**
- `dataset.py` — `build_taxonomy_texts`, ragged-rank handling, collators.
- `train_all_setups_tips_ddp.py` — DDP loop, cross-GPU negatives, setup matrix.
- `evaluate.py` — per-rank, macro, ragged eval (needs F1 + prototype + geometry fixes).

---

## 5. What needs building (v1, smallest real result)

**Tier 0 — must have:**
1. **Data bridge** — stream `planktonzilla-17M`, filter `plankton==True`, map
   columns (`Species`→`species`, `proposed_label`→`Folder`), cache to
   `/scratch/daniela/planktonzilla_cache` via `save_to_disk`. `HFTaxonomyDataset`
   emitting the `{image, taxonomy, folder}` dict the collator expects.
2. **Unseen split** — hold out the 4 paper datasets (GlobalUVP5, PlanktoScope,
   PlanktonSet1.0, SYKE-IFCB-2022) via the `dataset` column → 220 unseen classes.
   *Must dump unique `dataset` values first to match exact strings.*
3. **Wire frozen CLIP + BioCLIP** (open_clip) into the model (non-TIPS path; needs a
   small image/text adapter so `encode_image` returns the pooled CLIP vector).
4. **Eval rewrite** — macro-**F1** per rank, **prototype-based** zero-shot
   classification, geometry-aware distance, on the unseen set. *Critical path.*

**Tier 1 — strengthens the paper:**
5. SimpleShot 1-/5-shot (×5 seeds) to fill the Table-3 shape.
6. Within-hyperbolic ablation: plain-CL vs +SEL/entailment (attributes the win to
   hyperbolic *structure* vs just frozen-projector fine-tuning).
7. **Depth-factored projector** vs the plain linear projector: replace the linear head
   with a unit-direction head + a per-rank depth (radius) head, so taxonomic depth maps
   onto hyperbolic radius explicitly (scratchpad `use_depth_factored`). Hypothesis: a
   natural fit for taxonomy that should help coarse-rank unseen recovery. Add only
   after the v1 linear-projector baseline works.

**Tier 2 — deferred:**
7. RCME-style radial-Euclidean baseline (the controlled geometry comparison → Track-1
   follow-up).
8. ViT-L/14 / BioCLIP2.

---

## 6. Decisions locked in

- Class identity = **`proposed_label`** (WoRMS-harmonized, deepest valid rank).
- Data access = **stream + cache plankton subset to disk** (`/scratch/daniela/
  planktonzilla_cache`).
- Held-out unseen = **the paper's exact 4 datasets**.
- Backbones v1 = **CLIP + BioCLIP**, ViT-B/16, frozen.
- Adaptation = **projector-only first**; LoRA only if needed, applied uniformly.
- Baseline for v1 = **paper's reported Table 2/3 numbers** (+ off-the-shelf BioCLIP).
  RCME radial baseline **deferred**.

---

## 7. Eval gaps to fix (from reading `mine/hyperbolic/evaluate.py`)

1. Computes macro-**recall**, not macro-**F1** → add per-class precision (count false
   positives). **Mandatory** for paper comparability.
2. Training loop calls the **retrieval** variant (nearest other-sample); paper needs
   **prototype** classification (nearest class-name text). The `evaluate_classification`
   function has the right mechanism but returns plain accuracy.
3. **SimpleShot** (support-centroid, ×5 seeds) not implemented.
4. Distance **hardcoded to Lorentz** — fine for v1 (hyperbolic only); will need a
   geometry switch when the radial baseline returns.
