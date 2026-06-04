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

## Piece 3 — data bridge (Planktonzilla HF → taxonomy dict)  🚧 IN PROGRESS

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
