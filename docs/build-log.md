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

## Entailment/hyperparam audit vs HAC, MERU, UNCHA (2026-06-06)

Read all four sources directly. Findings:

**Correct in ours (verified vs source):** entailment direction `oxy_angle(parent,child) ≤
half_aperture(parent)` with parent=apex=more-general (matches MERU/HAC/UNCHA); curv/alpha/
logit_scale init + clamps (curv∈[init/10,init·10]; alpha.data≤0 so exp≤1; logit_scale≤ln100)
— byte-for-byte MERU/HAC; projector init `Linear(width,dim,bias=False)`, `normal_(std=width^-0.5)`
— identical; no-wd param grouping (LN/bias + scalars + lora) — matches HAC `set_weight_decay_per_param`.

**Legitimate divergences (we follow Taxonomies/RCME for SEL, not MERU/HAC):** B×B grid +
negative-entailment hinge (MERU/HAC/UNCHA do positives-only on the matched diagonal);
SEL-intra across taxonomy ranks (no analogue in HAC/UNCHA). Inter/intra in HAC/UNCHA means
text↔image vs part↔whole — NOT our rank↔rank; so HAC's 0.7/1.2 aperture thresholds do NOT
map onto our ranks (earlier suggestion to copy them = retracted).

**Two gaps FIXED this commit:**
1. **Cross-GPU contrastive negatives (`dist_gather`).** MERU/HAC gather text/image across all
   ranks (differentiable `torch.distributed.nn.all_gather`) so each image is scored vs the
   GLOBAL batch; targets shifted by `B·rank`. We were contrastive-training on local
   negatives only (128, not the effective batch). Added `hyperbolic_contrastive_loss_ddp`;
   verified on a 2-GPU smoke (no deadlock under no_sync accumulation; cl rises with the
   larger negative set, as expected). Caveat: grad-accum means this enlarges negatives per
   micro-batch (→ micro_bs·world), not across accum steps — still a real gain, not the full 768.
5. **Re-init final LN (HAC `init_final_ln`).** HAC resets each tower's final LN (γ=1,β=0)
   when transitioning CLIP into the hyperbolic output space, then trains it. We only
   unfroze it. Now `apply_lora(reinit_final_ln=True)` default; `--no-reinit-final-ln` to A/B.

**Known remaining gaps (not adopted — candidate levers):** UNCHA continuous-angle term
`+α·φ` (gradient even when child inside cone; scratchpad has it as `angle_alpha`); HAC/UNCHA
aperture thresholds (semantics don't map); UNCHA uncertainty calibration (deferred).

---

## Training plan — λ_sel sweep (recorded before first launch, 2026-06-05)

Loss = `contrastive(img, deepest_text) + λ_sel · SEL`. `λ_sel` is logged to wandb so it
is ablatable. **Planned sweep: λ_sel ∈ {1.0, 0.5, 0.2}**, BioCLIP init, HAC recipe
(30k iters, batch 768, lr 2.5e-4, 4k warmup + cos², LoRA r=128 rslora).

**Headline run first at λ_sel = 1.0**, then 0.5 and 0.2 as the ablation arm. Rationale:
- The thesis rests on **SEL** (entailment cones → higher-rank fallback on unseen species);
  the contrastive term alone is "hyperbolic CLIP", not "hyperbolic *taxonomy* CLIP". Start
  where SEL has real influence.
- Our SEL magnitude is already small in practice (smoke: `sel`≈0.05–0.15 vs `cl`≈2–4), so
  even at λ=1.0 SEL is a minority of the total loss; λ=0.2 → 0.01–0.03 may be too weak to
  shape the geometry. Starting at 0.2 would tell us less.
- HAC uses `entail_weight=0.2`, but HAC's entailment is a **different object** (single
  text⊐image hinge regularizing a contrastive main objective; ours is a *stack* of
  rank→rank hinges closer to a primary objective). 0.2 isn't directly transferable.
- λ=0.2 doubles as the "does SEL even matter?" near-pure-contrastive comparison: if 1.0
  and 0.2 land together, SEL isn't doing much (a finding); if 1.0 wins, it supports the
  thesis.

**Caveat / what to watch (first ~2k iters):** SEL behaviour past ~200 iters is unverified.
If λ=1.0 over-constrains (curvature collapse, contrastive alignment stalling), it may be
too aggressive — the live curves will show this early; kill/adjust if so.

### CORRECTION (2026-06-05): collapse was the LR/WD recipe, NOT the text encoding

The "cumulative text → collapse" diagnosis below was **wrong** and is retracted. Decisive
counter-evidence from the user's scratchpad (`mine/hyperbolic`):
- The scratchpad is **frozen backbone + projector-only + CUMULATIVE text + learnable curv**
  — and it is **stable AND best-performing** (cumulative consistently beat independent there).
- That setup differs from ours in only one axis bundle: **the optimiser recipe (+ LoRA)**, NOT
  the text. So cumulative text is exonerated; the collinearity-causes-collapse theory is refuted
  by a near-identical stable setup using cumulative.

**The actual cause — recipe mismatch.** We copied **HAC's** LoRA optimiser (tuned for VQA/GRIT)
wholesale; the scratchpad used the **Hyperbolic-Taxonomies (2025) SEL recipe**:

| | scratchpad (STABLE) | our run (COLLAPSE) |
|---|---|---|
| peak LR | **5e-5** (one-cycle) | 2.5e-4 (HAC) — **5×** |
| weight decay | **1e-4** | 0.2 — **2000×** |
| batch | 256 | 768 |
| LoRA | none (projector only) | r=128 |

5× the LR yanks the `curv` scalar; wd 0.2 actively drags trained params. That, not the text,
is why curv collapsed (and why MERU/scratchpad with gentler recipes don't).

**Correction applied (partial — deliberately one change at a time):** **cumulative text is the
default** again (independent demoted to `--independent-intra` ablation flag). Curvature stays
learnable. The LR/wd recipe is left at HAC's **2.5e-4 / 0.2** for now — we will change ONE
variable at a time from this clean base (cumulative + HAC LR) rather than bundle the full SEL
recipe in at once. LoRA stays central (projector-only capped out in the scratchpad, so it is
NOT an ablation). The SEL recipe (5e-5 / 1e-4 / one-cycle) is the leading candidate for the
first controlled probe but is not yet the default.

**Why cumulative is also the *principled* choice (user's point):** a deeper rank's cumulative
string is its parent's string + one token (prefix-extension) → the partial order is encoded in
the text itself; child = parent + a small textual step → a small radial step out. The
parent/child similarity that the retracted section called "collinear/bad" is just adjacency in
the tree being preserved. Independent text discards that structural signal (and the measured
performance advantage).

> NOTE: the section below is RETRACTED — kept for the record of the (wrong) reasoning path.

### [RETRACTED] Root cause found: cumulative SEL text → curvature collapse (2026-06-05)

After runs 1 (λ=1.0) and 2 (λ=0.2) BOTH collapsed identically (curv 1.0→0.32, F1
peak~it2000 then halves), and a CL-only probe showed curv RISES without SEL, the cause
was traced to **how SEL-intra encodes rank text**, not the SEL weight.

**The bug:** our SEL-intra used **cumulative** lineage text per rank (genus =
"animalia arthropoda copepoda", i.e. family-text + one word). Consecutive parent/child
are then near-identical strings → near-collinear embeddings (measured cos: order→family
**0.94**, genus→species 0.84, kingdom→phylum 0.60). SEL needs to push the child OUTWARD
along the parent's ray, but near-collinear points give almost no radial-separation
gradient. So the optimiser can't satisfy SEL by moving embeddings → it takes the cheap
global shortcut: **shrink curvature to widen all cones at once**. (AdamW normalises the
curv-gradient magnitude, so even λ=0.2 supplies enough *directional* pressure — explains
why both λ collapsed alike.)

**The paper says use per-rank text.** Hyperbolic-Taxonomies §3.1 gives the label format as
structured per-rank fields ("Order: Diptera; Family: Syrphidae; Genus: …"), and Eq. 3's
`T_r` is "the embedding **at rank** r"; §3.3 designs SEL for the **"nested and
NON-overlapping"** hierarchy. Cumulative text is maximally *overlapping* (prefix nesting) —
the opposite. The paper uses per-rank text for SEL and a separate concatenated "full text"
only for the contrastive term. So our cumulative-for-SEL was the departure.

**Measured fix (independent per-rank text):** cos drops to 0.18–0.42 (order→family 0.94→
0.42, genus→species 0.84→0.18), dist ~doubles — distinct points SEL can actually separate
radially, analogous to HAC's distinct object/scene inputs. **Curvature stays learnable**
(MERU/HAC both learn it; freezing it would treat the symptom and just push the optimiser to
the next shortcut, e.g. alpha). `--freeze-curv` kept only as a diagnostic, not the fix.

**Resolved design (corrected):** **BOTH** SEL terms use **independent per-rank text** `T_r`.
Re-reading Eq. 4: SEL-inter entails the image by `T_R'` = the deepest *per-rank* label
(`T_Species`/`T_Genus`/…) — the SAME `T_r` objects as Eq. 3, NOT the cumulative `full`
string. The concatenated "full text" is used **only for contrastive alignment** ("Full-text
supervision", p.4). So: SEL-intra + SEL-inter → independent `T_r`; **contrastive + eval →
cumulative `full`** (paper-faithful class label, validated vs Table 3). Pos/neg masking in
both SEL terms still keys on the cumulative lineage label. (Initially we mistakenly kept
SEL-inter on cumulative text; fixed.)

### Run 1 result: λ_sel=1.0 → CURVATURE COLLAPSE (killed at it 7250, 2026-06-05)

The caveat materialised. Trajectory (BioCLIP, λ=1.0, HAC recipe):

| it | lr | curv | cl | seen_sp F1 | unseen_sp F1 |
|---|---|---|---|---|---|
| 1000 | 8e-5 | 0.96 | 1.92 | 0.219 | 0.016 |
| 2000 | 1.4e-4 | 0.83 | 1.84 | **0.297** | **0.020** |
| 3000 | 2.4e-4 | 0.66 | 1.93 | 0.293 | 0.022 |
| 5000 | 2.5e-4 | 0.50 | 2.12 | 0.214 | 0.017 |
| 7000 | 2.4e-4 | 0.34 | 2.65 | 0.104 | 0.010 |

**Everything peaks at it~2000 (when warmup finishes ramping LR toward 2.5e-4), then
degrades as `curv` falls monotonically 0.96→0.33.** `cl` rises as curv shrinks; both F1
halve from peak. Checkpoints saved at it2000 (peak region), it4000, it6000.

**Mechanism (a real finding, not a bug):** SEL has a built-in incentive to *shrink
curvature*. Cone half-aperture `asin(2·r_min/(‖x‖·√curv))` widens as curv→0, so smaller
curvature makes the entailment hinges trivially satisfiable — a cheap way to cut SEL loss
without organising the hierarchy. At λ=1.0 + full LR this incentive wins; curvature
collapses and the contrastive geometry degrades as collateral. HAC avoids this despite an
identical `[curv/10, curv*10]` clamp because its single text⊐image hinge lacks our stacked
SEL pressure.

**Action:** killed; relaunch at **λ_sel=0.2** (sweep's other end — now a directed test of
the curvature-collapse hypothesis, not just an ablation). Change ONE variable: if 0.2 is
stable, confirmed it's the SEL weight; if it still collapses, add a reduced LR for the
geometric scalars (curv/alpha) — a MERU-style guard. Geometry logging added (see below) to
watch radius/aperture/entailment per rank directly, not just the scalar curv.

### SEL correctness audit (before relaunch) — ✅ SEL IS CORRECT

Reviewed what each SEL term operates on + measured the geometry on a real BioCLIP batch
(no training). Verdict: **the loss is implemented correctly; run-1's failure was
optimization (curvature collapse), not a loss bug.** Evidence:

- **What each term operates on:** `sel_intra` = entailment between *consecutive-rank
  cumulative-lineage texts* (parent=coarser, child=finer); positive if same lineage at
  the parent rank, pushing finer text inside coarser text's cone. `sel_inter` =
  deepest-valid-text ⊐ image. Class label = cumulative `full` string (matches contrastive
  + eval + the paper's `tax_label`).
- **`oxy_angle` verified** on controlled cases: child deeper along parent's ray →
  angle 0 (inside); child *shallower* → angle π (outside). So entailment needs child
  (a) along parent's ray AND (b) at LARGER radius.
- **Cumulative encoding is NOT degenerate** (Concern A refuted): parent/child cosines
  0.59–0.94, distances 0.09–0.24 — distinct points, real gradient.
- **Root cause of `entail_ok=0`:** at init the per-rank **radii are flat (~0.26–0.30)**
  with no increasing trend (kingdom even slightly larger). Children aren't further out
  than parents, so they're outside the cones. SEL's *job* is to create that radial
  ordering; entail_ok=0 pre-training is correct/expected.
- **Per-term decomposition (logged as `loss_terms/*`):** at init **`neg=0` at every edge**
  (non-children already far outside cones, nothing to push), **`pos≈0.9–1.3`** (real
  positive pressure, larger at deeper edges). `n_pos` shrinks with depth (kingdom→phylum
  134K → genus→species 721) since fine ranks have mostly-distinct labels → noisy deep
  edges. So SEL = essentially all positive pressure at init, competing with the (easier)
  curvature-shrink escape.

**Instrumentation added:** `loss.stacked_entailment_loss(..., stats=dict)` decomposes
intra-per-edge + inter into pos/neg + pair counts; driver logs `loss_terms/*`. Combined
with `geom/*`, the λ=0.2 run can distinguish *healthy* (sel pos ↓ **while** entail_ok ↑,
radii spread) from *cheating* (pos ↓ but entail_ok stays 0, apertures widen uniformly).

### SEL-inter operand: deepest rank (leaf), NOT all ranks — ✅ PAPER-FAITHFUL

Question raised: shouldn't the image be entailed by *every* ancestor rank (transitivity),
not just the leaf? Checked the paper (Hyperbolic Multimodal Repr. for Bio Taxonomies,
2025), **Eq. 4**:

> `L_SEL-inter = (1/3)·[ L_ent(I, T_R') + L_ent(D, T_R') + L_ent(I, D) ]`, where `T_R'` is
> **the deepest available taxonomic label** (T_species if known, else T_genus, …).

So SEL-inter **is leaf-only by design**, and our `_deepest_text`-based `sel_inter` matches
it exactly. (We drop the DNA `D` terms — no DNA modality here; image↔deepest-text is the
applicable part.) Given cumulative encoding, the deepest-rank embedding == the `full`
embedding (same vector); `sel_inter` uses `_deepest_text` only to also get the rank id +
validity mask for pos/neg masking.

**Why leaf-only is sufficient (the transitivity rationale, which the question correctly
identified):** the paper splits the labor — **SEL-intra (Eq. 3)** builds the nested chain
`kingdom ⊐ phylum ⊐ … ⊐ species` among the *text* ranks, so containment is transitive;
**SEL-inter** then anchors the image at the leaf only and *relies on* that chain to
propagate the image into all ancestor cones. Entailing the image by every rank would be
redundant **at convergence**.

**Parked ablation — "stacked image-entailment":** the transitivity argument only holds
once the intra chain has converged. Our diagnostic shows it has NOT (entail_ok=0,
curvature collapse). A variant that entails the image by *every valid rank* (not just the
leaf) would give direct gradient at each level during the un-converged phase — more robust,
but a **departure from the paper** (non-comparable to its SEL). Decision: do NOT switch
now. First get the intra chain to actually converge (λ tuning / curvature-collapse fix);
if intra converges, leaf-only inter works as designed and this is moot. Keep stacked
image-entailment as a later ablation only.

**Aside (paper, §3.1):** stacked entailment is memory-heavy — they could only fit batch
1520 on 4×A100-80GB "for experiments using stacked entailment". Confirms the B×B SEL grids
are our memory driver; relevant if we ever push batch size past 768.

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

## Piece 6 — unseen-species eval  ✅ VERIFIED (split reproduces paper exactly)

`src/.../eval.py`: Planktonzilla-faithful Table-3 protocol, hyperbolic-distance prediction
+ a Euclidean-cosine path for the paper-matching baseline. `scripts/run_unseen_eval.py`.

**Protocol (verified against the repo, not assumed):**
- Class string = `" ".join([Kingdom..Species] non-empty)` = our `build_taxonomy(row)["full"]`
  (paper `gen_datasets.py::build_tax_string`). Both skip gaps; no contiguity requirement.
- Unseen classes = held-out `full` strings absent from the seen pool, Kingdom required.
- Predict: encode each class string with prompt `"a photo of a {label}"` (paper's exact
  template, confirmed in metrics_paper.ipynb), nearest prototype. Paper: argmax cosine;
  ours: argmin Lorentzian distance. Restrict label space to the unseen set.
- Per-rank macro-F1: truncate true+pred `full` to k tokens, sklearn `f1_score(macro)` —
  re-implements `evaluate_taxonomic_metrics`. Plus overall full-string F1.

**Split reproduces the paper EXACTLY (independent of any model):**
- **220 unseen classes / 113,089 samples** — matches paper §3.1 verbatim. Built from raw
  cache with our own `full`-string + `HELD_OUT_DATASETS` logic. (Held-out total 821,212.)

**Euclidean-cosine baseline vs paper Table 3 (ViT-B/16 BioCLIP, zero-shot, n=113,089):**

| rank | paper | ours | | rank | paper | ours |
|---|---|---|---|---|---|---|
| kingdom | 0.346 | 0.259 | | order | 0.018 | **0.018** |
| phylum | 0.102 | 0.081 | | family | 0.013 | **0.013** |
| class | 0.032 | 0.035 | | genus | 0.011 | 0.010 |
| | | | | species | 0.010 | 0.007 |

Order/Family **identical to 3 dp**; class/genus within rounding. Coarse ranks (kingdom,
phylum) run a bit lower — second-order (BioCLIP snapshot / fp autocast / argmax ties),
not a pipeline error. **This validates the data split + class strings + similarity
prediction + truncated macro-F1 against the published numbers.** CLIP (OpenAI) baseline
also runs (kingdom 0.328, then decays) — consistent regime.

**Verified (`tests/test_eval.py`, 5 tests):**
- `taxonomic_macro_f1` matches a **vendored verbatim copy** of the paper's
  `evaluate_taxonomic_metrics` to 1e-12 on random ragged labels (load-bearing check) +
  exact hand-computed kingdom F1 (=1/3 on a worked 2-sample example).
- `build_unseen_classes` set algebra; `predict` argmin geometry (synthetic hyperboloid);
  end-to-end on the real model (text-as-image surrogate → recovers own class, full F1=1).

**Note:** for the paper-comparable eval the class identity is the `full` lineage string —
NOT `proposed_label` (that was the training-positive choice in Piece 5; separate axis).

**Still to do:** SimpleShot 1/5-shot (image-centroid, 5 seeds) as added Table-3 columns;
fold the fast columnar `_full_strings` into the lib (currently inlined in the run script).
The headline comparison needs a *trained* projector/LoRA model run through `run_unseen_eval`
(Piece 5 output) — the untrained projector floor is ~0 (random projection breaks alignment).

State at this point: **73 tests pass**. Pieces 1,2,3,4,5,6,7 verified.

---

## Piece 5 — collator + train step  ✅ VERIFIED

`src/.../train.py`: `TaxonomyCollator` + `train_step`. Bridges data items → model+losses.

**Preprocess finding (confirmed):** CLIP and BioCLIP use the **identical** open_clip
preprocess (Resize224 bicubic → CenterCrop → RGB → ToTensor → Normalize, same CLIP
mean/std). One transform serves both inits. `build_backbone` previously DISCARDED it →
now returns `(model, embed_dim, preprocess)` and `HyperbolicCLIP` stores `self.preprocess`.

**What's built:**
- `TaxonomyCollator(preprocess, ranks=RANKS)`: list of `{image, taxonomy, proposed_label}`
  → `(pixel_values [B,3,224,224], taxonomy_batch {rank: [B] list} + full + _valid_ranks,
  proposed_labels [B])`. Applies preprocess per-image + stacks; transposes per-item
  taxonomy dicts into per-rank lists (None preserved). Shape mirrors scratchpad
  `TaxonomyCollator`, but uses the open_clip transform (not an HF processor).
- `train_step(model, pix, tax, optimizer, lambda_sel=1.0)`: one Adam step of
  `contrastive(img, deepest_text) + λ·SEL`. CL target = each sample's **deepest valid
  text** (`loss._deepest_text`, 1:1 positive per image); then `stacked_entailment_loss`.
  Calls `model.clamp_params()` after `optimizer.step()`. Returns loss parts for logging.
  Wiring follows scratchpad `train_epoch_sel_cl`.

**Verified (`tests/test_train.py`, 3 tests + 1 folded into test_model):**
- Collator shapes + 1:1 per-rank alignment with items (None preserved), pixel dtype.
- `train_step` runs, returns finite parts; **overfit check**: 21 steps on one fixed real
  batch drives loss DOWN (clip, projector-only, CPU).
- Grads reach `visual_proj`/`textual_proj`; frozen `visual.conv1` gets none.
- `build_backbone` returns a working preprocess (PIL→[3,224,224]).
- **GPU smoke (off-CI, real A5000):** train_step finite for clip AND bioclip; fp32
  exp-map autocast active, no NaNs. **LoRA path composes**: `apply_lora` → 0.65%
  trainable, `backbone_trainable=True`, loss 4.21→2.10 over 11 steps, lora_B gets grad.

**Test convention:** package is NOT pip-installed; run with `PYTHONPATH=src python -m pytest`.

State at this point: **68 tests pass**. Pieces 1,2,3,4,5,7 verified.

**Still to do (Piece 5 leftovers → roll into Piece 6 / data plumbing):**
- **Splits:** stratified train/val/test on the seen pool (11 datasets); unseen = the 4
  held-out. Not yet built (the collator/step don't need it; eval does).
- DataLoader wiring + a real multi-step training run; DDP/grad-accum deferred.

**Spec:** HF schema (planktonzilla.md) + scratchpad `dataset.py` / `train_all_setups.py`.

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

## Euclidean-LoRA (E0c) tuning campaign — vs full-FT (2026-06-12 → 06-15)

**Goal:** does Euclidean LoRA match the full-FT Planktonzilla CLIP on SEEN (the
LoRA-vs-full-FT control). Full-FT bar in `docs/baseline-planktonzilla-clip.md` (commit
`be58144`). All runs: `--no-proj` (architecture-identical to their CLIP), all 12+12 blocks,
attention-only, planktonzilla.

**Best result so far — E0c r=32, 20ep, lr 2e-4 warmupcos, fp16 (run `7r7cvoa3`):**
full-split SEEN species 0.634 vs full-FT 0.818 (**−0.184**); the seen gap widens monotonically
with depth (kingdom −0.04 → species −0.18). UNSEEN gaps small in absolute terms (both near
the floor). LoRA recovers coarse taxonomy, not fine-grained discrimination. This is the
"before" the bf16/OneCycle/sweep changes must beat.

**Levers tried, attributed one at a time:**
- **LoRA rank ✗** — r=64/α=64 DIVERGED (seen F1 climbed then collapsed). Cause: with
  `use_rslora=True` the update scale is α/√r, so α=r inflates the step by √2 when r doubles
  (not capacity). Fixed: `--lora-alpha` flag (commit `617bc9a`) to hold scale constant
  (r=64/α=45). At matched scale r=64 ≈ or slightly BELOW r=32 → rank is NOT the bottleneck.
- **Training duration** — r=32/30ep still running mid-campaign; user constrained to ≤20ep,
  so duration deprioritized. (The 20ep flat tail was warmupcos decaying LR→0, not
  convergence — motivated the OneCycle switch.)
- **MLP adaptation** — `--lora-mlp` added (commit `7b3a4d9`, 3.9M→8.9M trainable params),
  held in reserve: doubles params (hurts the efficiency story) and HAC adapts attention-only.
  Untested; the fallback if LR/schedule don't close the gap.

**Infra fixes from the campaign:**
- Eval OOM (`33c23f4`): `encode_image/text` re-enabled grad under `no_grad` for LoRA models
  (backbone_trainable=True) → graph accumulated over the eval loop → 23 GiB OOM. Gated on
  `backbone_trainable and torch.is_grad_enabled()`. Affected ALL LoRA/hyperbolic evals.
- NCCL final-eval timeout (`2e7f9f1`): rank-0-only final eval ran > NCCL's 600s barrier
  timeout, so idle ranks aborted the job AFTER training/ckpt saved. Dropped the cross-rank
  barrier (each rank tears down independently).
- Adapter-only checkpoints (`6cfb44e`): 615 MB → 16 MB (frozen backbone recreated at load).
- Per-run checkpoint dirs (`4858cd6`): `{tag}__{run_id}/` — same-tag reruns no longer clobber
  (this had destroyed run `759e1par`'s checkpoints when a later same-tag run overwrote them).
- bf16 training (`2bc49fb`): autocast fp16+GradScaler → bf16, scaler dropped (no underflow).
  NOTE: old runs are fp16; new runs bf16 — not precision-matched.
- OneCycleLR default (`5949fc2`): warmupcos → onecycle (Taxonomy-paper super-convergence;
  `--lr` is now the PEAK). Aimed at faster convergence + the flat-tail problem.

**Next:** wandb LR sweep `scripts/sweeps/e0c_lr_phase1.yaml` (OneCycle peak ∈
{3e-4,6e-4,1e-3,2e-3}, 5ep proxy, bf16) — does a swept peak LR + OneCycle close the seen gap.
Then full-split confirm the winner (fresh bf16+onecycle series, NOT comparable to the fp16
0.634). MLP adaptation only if the gap persists.
