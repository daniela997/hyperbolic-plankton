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

## Next: Piece 2 — model (frozen open_clip backbone + projection + exp_map0 lift)
Spec: HAC `AdaptedCLIP` (geometry) + our `model.py`. Verify: frozen params get no grad;
projected points lie on the hyperboloid (`<x,x>_L ≈ -1/curv`); forward runs on a dummy
batch for both CLIP and BioCLIP open_clip backbones.
