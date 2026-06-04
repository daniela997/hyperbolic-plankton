# Build Log

> Methodical, piece-by-piece re-implementation. We **reference** `mine/hyperbolic/`,
> HAC, and the papers as specs, but **re-derive and verify** each module rather than
> import unverified scratchpad code. Each piece: spec → implement → verify
> (numerical cross-check vs references + property tests) → record here.

Env: `/scratch/daniela/miniconda3/envs/dino_plankton/bin/python` (torch 2.10, CUDA on,
2× RTX A5000). Run tests with: `PYTHONPATH=src <env-python> -m pytest tests/ -v`.

Build order (dependency-respecting): **lorentz → model → data → loss → train → eval →
LoRA**. Projector-only must work end-to-end before the LoRA tier.

---

## Piece 1 — `lorentz.py` (geometry primitives)  ✅ VERIFIED

**Files:** `src/hyperbolic_plankton/lorentz.py`, `tests/test_lorentz.py`.

**Spec source:** MERU/HAC canonical `lorentz.py`
(`/home/daniela/other/HAC/hac/lorentz.py`), cross-checked against the scratchpad
`mine/hyperbolic/hyperbolic.py::LorentzMath`.

**Implemented:** `time_component`, `pairwise_inner`, `pairwise_dist`, `exp_map0`,
`log_map0`, `distance_from_origin`, `half_aperture`, `oxy_angle`. (Deferred:
`log_map`/`exp_map` at arbitrary base points — only needed for the parallel-transport
encoder variant, which is not in the v1 plan.)

**Verification (21 tests, all pass):**
- Properties: exp/log round-trip; points satisfy `<p,p>_L = -1/curv`; self-distance at
  the stability floor; `distance_from_origin == pairwise_dist(origin, ·)`; aperture in
  (0, π/2); angle in (0, π); cone monotonicity.
- **Cross-check vs HAC** (curv 0.5/1/2): all 6 functions match to **atol 1e-6**.
- **Cross-check vs scratchpad** (matched eps): match to **atol 1e-6**.

**Decisions / findings:**
- **eps = 1e-8** (MERU/HAC default), not the scratchpad's 1e-4. The scratchpad math is
  otherwise identical (cross-check confirms) — its looser eps just adds error near the
  origin. We took the principled value.
- The **arccosh stability floor** is intended behaviour: `pairwise_dist` clamps its
  input to `1+eps`, so `d(p,p)` bottoms out at `~sqrt(2*eps/curv)`, not exactly 0. This
  is in the reference too — the test asserts against the floor, not against 0.
- Test loader stubs `loguru` so HAC's `lorentz.py` (which imports it only for an unused
  logging line) executes in the lean env. Without this the cross-checks **silently
  skipped** — worth flagging: a green suite that skips its load-bearing checks is a
  false positive. Always confirm cross-checks actually run.

---

## Piece 2 — `model.py` (frozen open_clip backbone + projection + lift)  ✅ VERIFIED

**Files:** `src/hyperbolic_plankton/model.py`, `tests/test_model.py`.

**Spec source:** HAC `AdaptedCLIP` (geometry: freeze, proj heads, alpha/curv scalars,
exp_map0 lift, clamps) + scratchpad `model.py` (per-rank `encode_taxonomy` dict).

**Implemented:** `build_backbone("clip"|"bioclip")`, `HyperbolicCLIP` with
`encode_image`, `encode_text(list[str])`, `encode_taxonomy(dict)`, `clamp_params`.
- **clip** = `ViT-B-16-quickgelu` / `openai` (quickgelu variant avoids the activation
  mismatch warning for OpenAI weights).
- **bioclip** = `hf-hub:imageomics/bioclip` (downloads on first use; verified working).
- embed_dim = 512 (shared image/text output of ViT-B/16).

**Verification (11 tests, all pass; 32 total across the suite):**
- Both backbones load; frozen (no param has `requires_grad`); output dim 512.
- Image + text embeddings lie on the hyperboloid (`<x,x>_L = -1/curv`, atol 1e-3).
- Backward pass: **frozen backbone gets NO gradient**; projection heads + MERU scalars
  DO. (This is the projector-only guarantee — verified, not assumed.)
- `encode_taxonomy`: correct per-rank shapes, `{rank}_valid` masks, None→zeros, `_`-keys
  skipped, valid rows on the manifold.
- `clamp_params` keeps alpha ≤ 0 and logit_scale ≤ ln(100).

**Decisions / findings:**
- **Loss is NOT in the model** (unlike HAC's `AdaptedCLIP.forward`). The model only
  encodes/projects; loss lives in Piece 4. Keeps the model single-purpose + testable.
- Built `encode_taxonomy` now (one piece ahead of SEL) per user request; tested
  directly without a loss consumer.
- **Conceptual correction worth remembering:** in the space-components representation,
  EVERY vector satisfies `<x,x>_L = -1/curv` by construction (time is defined as
  `sqrt(1/curv+||x||²)`), so that constraint can't distinguish tangent from manifold.
  `project=True`'s real effect is the geodesic placement (norm change via exp_map0).
  Tests check the norm change, not the (always-true) constraint.
- `_lift` autocasts to fp32 only on CUDA (CPU fp32-autocast is unsupported + a no-op).
- Deferred: LoRA (Piece 7), parallel-transport/depth-factored encoders (not in v1).

---

## Piece 4 — `loss.py` (contrastive + SEL)  ✅ VERIFIED

**Files:** `src/hyperbolic_plankton/loss.py`, `tests/test_loss.py`.

**Spec source:** 2025 Hyperbolic Taxonomies paper (SEL Eq. 3) + scratchpad `loss.py`.
**Scope (v1 core):** `hyperbolic_contrastive_loss`, `entailment_pos/neg`, `sel_intra`,
`sel_inter`, `stacked_entailment_loss`. Deferred: UNCHA, hard-neg images, SupCon,
angular alignment (later ablations).

**Verification strategy (important — see the Q&A that shaped it):**
- We **trust** MERU's `oxy_angle`/`half_aperture` math (established prior art; 3
  published codebases — MERU, HAC, scratchpad — agree, and our `lorentz.py` matched
  HAC to 1e-6 in Piece 1).
- We **independently verify OUR loss composition** — the parts we could have gotten
  wrong — with **exact-value tests** that compute the expected loss by hand from the
  trusted primitives:
  - `sel_intra` == hand-computed mean of `relu(angle-aperture)` over masked pairs (1e-6).
  - pos/neg masking by parent-rank label routes cross-pairs correctly.
  - **grid orientation** (child-rows / parent-cols): asserts we match the convention
    AND that the transpose gives a different answer (catches a silent transpose bug).
  - **Eq.3 denominator** = exact supervised-edge count (ragged edge → `edge1/2`, not
    `edge1/1`).
  - `_deepest_text` picks the leaf-most valid rank + its label per sample.
- The scratchpad cross-check is kept but **documented as NON-independent** (same MERU
  lineage; can only confirm we copied the formula identically, not that it's correct).
  Matched at atol 2e-2 due to the known eps gap (scratchpad 1e-4 vs ours 1e-8).

**HAC template note:** HAC's entailment is the SAME `relu(angle - aperture)` hinge
(`AdaptedCLIP.forward`, models.py:828) — confirms our hinge — but HAC multiplies
aperture by a threshold (0.7 inter / 1.2 intra, HyCoCLIP-specific) and has **no SEL**
(it does object-scene/box entailment). So HAC templates the hinge; the stacked-rank
SEL structure is the 2025 paper's, verified by our exact-value tests, not by HAC.

**17 loss tests pass (49 total across the suite).**

---

## Piece 7 — LoRA on the open_clip backbone  ✅ VERIFIED

**Files:** `src/hyperbolic_plankton/plain_mha.py`, `src/hyperbolic_plankton/lora.py`,
`tests/test_plain_mha.py`, `tests/test_lora.py`. Needs `peft` (installed 0.19.1).

**Spec source:** HAC `plain_mha.py` + `scripts/train.py` + `train_hac_vit_b_lora.py`.

### 7a — PlainMHA (numerical-equivalence gate)
open_clip uses `nn.MultiheadAttention` (fused Q/K/V `in_proj_weight`) in BOTH towers,
which PEFT can't target. Ported only HAC's `PlainMultiHeadAttention` path (not the timm
path — open_clip has no timm attention) → split q/k/v/o linears. **Gate verified:** the
swap is bit-for-bit identical to `nn.MultiheadAttention` (no mask AND causal mask), and
a full open_clip model's `encode_image/encode_text` are unchanged after swapping all 24
MHAs (atol 1e-4). This guarantees the pretrained backbone is uncorrupted before LoRA.

### 7b — LoRA application
`apply_lora`: swap MHA → `get_peft_model` (q,k,v,o; last 4 visual / last 8 text blocks;
r=alpha, rsLoRA) → unfreeze final LN. Trainable = LoRA + final LNs + projection heads +
MERU scalars; <10% of params.

**Two real issues found + fixed (methodical approach earned its keep):**
1. **target-module suffix collision.** PEFT matches a `target_modules` *list* by name
   suffix; the text path `transformer.resblocks.{i}...` is a suffix of the visual path
   `visual.transformer.resblocks.{i}...`, so a list wrongly LoRA'd visual blocks 4–11
   instead of 8–11. **Fix:** pass a `str` regex (PEFT uses `re.fullmatch`) anchored to
   distinguish `visual.transformer...` from the bare text `transformer...`.
2. **`no_grad` severed LoRA's gradient graph (real bug in Piece 2).** Our model wrapped
   the backbone forward in `torch.no_grad()` (correct for projector-only, saves memory).
   But LoRA needs the graph intact to receive gradients. **Fix:** conditional
   `set_grad_enabled(self.backbone_trainable)` — `apply_lora` sets `backbone_trainable=
   True`. **This is exactly how HAC does it** (see Q&A below): HAC never uses `no_grad`,
   relying on `requires_grad=False` alone, so the graph flows to adapters while frozen
   base weights accumulate no grad.

**LoRA semantics note (was a test bug, not a code bug):** `lora_B` is zero-initialized
(canonical LoRA → adapter starts as a no-op). So on the FIRST backward,
`grad(lora_A)=lora_B^T·grad_out=0` while `grad(lora_B)≠0`. The test now checks `lora_B`
gets grad (proves the adapter is in the live graph); from step 2 onward `lora_A` trains.

**How HAC loads CLIP + applies PEFT (answer recorded):** HAC does NOT use open_clip — it
builds a timm ViT (visual) + MERU `TransformerTextEncoder` (text) from a local `.pth`.
PEFT sequence (`scripts/train.py:211`): freeze both encoders → `replace_mha_with_plain`
(both, since timm + MERU use different attention) → `get_peft_model` per encoder →
unfreeze final LN. Projection heads/scalars live on `AdaptedCLIP`, never frozen. We use
open_clip (both towers `nn.MultiheadAttention`), so we only need the MHA path, not timm.

**8 tests (4 plain_mha + 4 lora); 57 total across the suite.**

---

## Piece 3 — data bridge (Planktonzilla HF → taxonomy dict)  ✅ VERIFIED

**Files:** `src/hyperbolic_plankton/data.py`, `tests/test_data.py`,
`scripts/cache_planktonzilla.py`.

- **Cache built:** `/scratch/daniela/planktonzilla_cache/plankton` — **3,746,982 rows**
  (matches paper's 3.74M), 41 arrow shards, 30GB.
- **`dataset` values resolved** (planktonzilla.md): held-out 4 = `global_uvp5`,
  `planktoscope`, `planktonset1.0`, `syke_ifcb_2022`; in-domain = other 11.
- **Implemented:** `build_taxonomy(row)` (cumulative-lineage rank strings, ragged None,
  `folder`=proposed_label appended, `full`, `_valid_ranks`); `HFTaxonomyDataset` emitting
  `{image: PIL, taxonomy, folder}`; `split_seen_unseen` (held-out 4 → unseen, rest → seen);
  `RANKS` = kingdom..species + folder.
- **Verification (7 tests, all on the REAL cache):** exact-value taxonomy (full lineage,
  shallow-ragged, all-missing→"unknown", strip/nan/empty→None); dataset item shape +
  PIL image; real-row consistency (each rank string is a prefix-extension of the prior);
  seen/unseen split routes by source. 64 total across the suite.

**Decisions:** cumulative (not independent) rank strings = the SEL paper / scratchpad
default. `folder` appended as deepest rank so image→folder entailment has a leaf. No
hard-negatives / transforms in v1 (model + collator handle preprocessing).

**Still to do (pairs with Piece 5 train loop):** a collator that batches PIL images
through the open_clip preprocess and groups `taxonomy` per-rank into the `{rank: [B]}`
lists the model's `encode_taxonomy` expects + the stratified train/val/test split of the
seen pool.

---

## Piece 5 — collator + train loop  ⬜ NEXT (plan captured for continuity)

**Preprocess finding:** CLIP and BioCLIP use the **identical** open_clip preprocess
(Resize224 bicubic → CenterCrop → RGB → ToTensor → Normalize, same CLIP mean/std). So one
transform serves both inits. `build_backbone` currently DISCARDS the returned preprocess
— thread it out so the collator can apply it (small change to model.py/build_backbone).

**Plan:**
1. `build_backbone` also returns `preprocess`; `HyperbolicCLIP` stores it (or expose it).
2. **Collator**: list of `{image, taxonomy, proposed_label}` →
   `(pixel_values [B,3,224,224], taxonomy_batch {rank: [B] list}, proposed_labels [B])`.
   Transpose per-item taxonomy dicts into per-rank lists; apply preprocess to images.
3. **Train step**: `img = encode_image(pix)`, `text_embs = encode_taxonomy(tax)`,
   `loss = contrastive(img, deepest_text) + λ·SEL(img, text_embs, tax, RANKS)`. Use
   `clamp_params()` each step. Verify: one step decreases loss on a tiny real batch;
   grads reach the right params; runs on GPU for clip + bioclip.
4. **Splits**: stratified train/val/test on the seen pool (11 datasets), unseen = the 4.
5. Keep single-GPU first; DDP/grad-accum later. projector-only first, LoRA via flag.

State at this point: 65 tests pass; HEAD = `efde554`. Pieces 1,2,3,4,7 verified.

**Spec:** HF schema (planktonzilla.md) + scratchpad `dataset.py::build_taxonomy_texts`.

### Environment fix (blocker found + resolved)
- **Blocker:** `dino_plankton` (pyarrow 19) **cannot read** the Planktonzilla parquet
  shards — `OSError: Repetition level histogram size mismatch`. Reproduced reading a
  single shard with raw pyarrow (no datasets), so it's purely a **pyarrow version**
  issue, not a datasets/streaming one. `fedclip` (pyarrow 24) reads them fine.
- **Fix:** upgraded **pyarrow 19 → 24** in `dino_plankton` (`datasets 2.21` pins
  `pyarrow>=15`, no upper bound, so this is allowed). Verified: shard reads now; **all
  32 Piece-1/2 tests still pass** (no regression). The pre-existing pip dependency
  warnings (streamlit/pillow, fsspec, transformers) were already present and untouched.
- **Lesson for the log:** always read these shards with **pyarrow ≥ 24**.

### Caching (running)
- `scripts/cache_planktonzilla.py`: `load_dataset` (downloads all 91GB / 187 shards to
  the HF hub cache on `/scratch`) → `filter(plankton==True)` → `save_to_disk`
  `/scratch/daniela/planktonzilla_cache/plankton`. Launched in background (multi-hour).
- `/scratch` has 6.2TB free; `/home` and `/` are nearly full → everything on `/scratch`.

### Still to do (once cache exists)
- Dump the **complete set of `dataset`-column values** from the local cache (the
  streaming sample only surfaced `global_uvp5`; values are **lowercase** like
  `global_uvp5`, so the paper's "GlobalUVP5/PlanktoScope/PlanktonSet1.0/SYKE-IFCB-2022"
  need mapping to the real strings — the open `[unknown]` in planktonzilla.md).
- `HFTaxonomyDataset`: reads the cached subset, maps `Species`→`species`,
  `proposed_label`→`Folder`, emits the `{image, taxonomy, folder}` dict the model's
  `encode_taxonomy` + (future) collator expect. Reuse `build_taxonomy_texts` logic.
- Unseen split: hold out the 4 paper datasets via the `dataset` column.
- Verify: load real rows, taxonomy dict shape, ragged handling, split sizes.
