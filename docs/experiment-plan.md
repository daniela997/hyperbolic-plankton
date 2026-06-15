# Experiment Plan — methodical, one variable at a time

> The discipline layer for **runs**. `project-plan.md` has the thesis/venue/method;
> this doc fixes the **benchmark, the baseline config, and the ablation ladder** so every
> run changes exactly ONE variable against a fixed reference, measured by a frozen eval.
> Written 2026-06-09 after a reset (prior experimentation sprawled across datasets,
> losses, and three curvature theories — two of which were wrong). Checkpoints were
> deleted; this is the clean restart.

---

## 0. Why this doc exists (lessons from the sprawl)

- Runs changed multiple variables at once → wins/losses were unattributable.
- The seen eval was **biased** (predicted among subsample classes) → every reported
  number was misleading until fixed (commit `361a9ec`, `c6d6771`).
- Geometry was read off inconsistent checkpoints; "curvature collapse" had three
  competing explanations, two refuted by later data.
- **Rule from now on: fixed benchmark + fixed baseline + one variable per run + the
  trustworthy eval as the only metric. No run without a hypothesis and a success
  criterion written down first.**

---

## 1. The frozen benchmark (do not change)

**Eval harness** (commits `361a9ec`, `c6d6771` — DONE):
- **Periodic monitor** (during training): stratified subsample, `--eval-cap` rows per
  `proposed_label`, scored against the **full present-class set** (classes present in the
  full eval split — Planktonzilla's `torch.unique` CLIP protocol). Unbiased, low-variance.
- **Final number** (`scripts/final_eval.py`): full test/unseen splits, present-classes
  over the full split, per-rank macro-F1 (`average="macro", zero_division=0` — identical
  to Planktonzilla's `f1_score` call).
- **The metric:** per-rank macro-F1, seen and unseen separately. **Unseen is the claim**
  (seen is monitored but not the headline — see `project-plan.md` §1 scope).

**Datasets:**
- **BIOSCAN-1M** (CLIBD split, complete-to-species, 36,279 train) — the clean control.
  Run here FIRST: simpler, faster, complete taxonomy isolates method behaviour from
  raggedness.
- **Planktonzilla** (ragged, 1.76M train, 220-class / 113,089 unseen = paper's Exp-2) —
  THE thesis benchmark. Run after the recipe is proven on BIOSCAN.

---

## 2. The baseline config (the fixed reference point)

Everything is measured as a delta from this. It is the **faithful Taxonomies-paper
recipe** (the one that trained stably), with all new machinery OFF:

```
--backbone clip                    # OpenAI CLIP ViT-B/16 (the "CLIP-style" reference)
--epochs 50                        # paper: 50 full passes (drives total_steps from dataset size)
--optimizer adam --scheduler onecycle --lr 5e-5 --wd 1e-4
--micro-bs 128 --accum 3            # eff. batch 768 across 2 GPUs
--lambda-sel 1.0                    # L = CL + SEL (paper writes equal weight)
--contrastive distance             # MERU-style InfoNCE (the standard)
--cl-mask none                     # no false-negative masking
--sel-text independent             # paper: SEL (intra Eq.3 + inter Eq.4) uses per-rank T_r;
                                   #   CL always uses the cumulative `full` string
--sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0   # plain hinge SEL, no UNCHA terms
# NO --curv-lr-scale               # free curvature
--lora-r 128                       # HAC LoRA recipe (LoRA on by default; --no-lora to disable)
```

Length is **epochs**, not a fixed iter count (a fixed `--iters` silently means very
different #epochs across datasets). 50 epochs = `50 * (len(loader)//accum)` steps:
- BIOSCAN (36k): ~2,360 steps.
- Planktonzilla (1.76M): **~114,000 steps** — a long run. Consider fewer epochs for the
  first pass if iteration speed matters; record whatever is used.

Per-dataset: `--dataset {bioscan,planktonzilla}`, `--eval-every` 200 (bioscan) / 1000 (pz).

> **SEL text form is also an ablation axis** (A6): `--sel-text cumulative` vs the
> independent default — the paper uses independent, but our frozen+projector regime may
> differ. Baseline uses the paper-faithful `independent`.

**Baseline is run ONCE per dataset and frozen as the reference.** Its `final_eval.py`
numbers are the bar every ablation must beat (or be measured against).

**The baseline matrix (two datasets × adaptation × geometry):**

We run BOTH datasets — **BIOSCAN** (small, complete taxonomy: we produce *every* cell
ourselves as the clean control) and **Planktonzilla** (large, ragged: the headline
benchmark, where we lean on the paper's numbers + their released full-FT weights).

| | Euclidean (flat CLIP InfoNCE) | Hyperbolic (Lorentz lift) |
|---|---|---|
| **Full FT** | BIOSCAN: **FT** (`train_euclidean_ft.py`) · PZ: paper macro-F1 **+ released weights** | — (we don't full-FT hyperbolic) |
| **LoRA** | **E0a/E0b** (`--geometry euclidean`) | B0 + ablation ladder (our method) |

- **BIOSCAN** is the dataset where we own all three Euclidean/hyperbolic corners. The
  **FT** cell uses `train_euclidean_ft.py` (backbone unfrozen, `unfreeze_backbone`, no LoRA)
  — the Planktonzilla recipe (lr 1e-4, wd 0.2, adamw) *adapted* to our compute (batch 768,
  50 epochs, not PZ's 16,384 / 100). It is NOT a literal PZ reproduction; the batch/epochs
  differ by necessity, so the FT-vs-LoRA gap on BIOSCAN must be read as adaptation-only
  given those fixed deltas. Comparator backing for BIOSCAN's Euclidean numbers: CLIBD
  (2024) trained a Euclidean model on this split, but evaluated on *retrieval*, not our
  per-rank macro-F1 — so it's context, not a directly comparable number. The Taxonomies
  paper (2025) that introduced the hyperbolic method on BIOSCAN released **no code/weights**.
- **Planktonzilla** supplies the Euclidean full-FT reference for free (paper Table 2/3
  macro-F1 + released weights we can re-eval through our harness), so we do NOT retrain it
  first; the PZ full-FT cell is "given." Our PZ runs are the LoRA + hyperbolic rows.

Comparing E0 against Planktonzilla/FT isolates **LoRA vs full-FT** (same flat-space InfoNCE);
comparing E0 against B0 isolates **Euclidean vs hyperbolic** (same LoRA, same backbone,
same projector). Without E0, B0-vs-Planktonzilla conflates the two changes. E0 trains the
identical frozen-backbone + LoRA + projector as B0 but skips the exp-map lift and SEL,
using cosine InfoNCE (open_clip `ClipLoss`: `logit_scale * img_n @ text_n.T`, symmetric CE,
cross-GPU shifted diagonal). Eval is cosine-argmax (`--geometry euclidean` in `final_eval.py`).

---

## 3. The ablation ladder (one variable per run)

Each row changes exactly one flag from the baseline. Run on **BIOSCAN first**, promote
the winners to **planktonzilla**. Every run reports the same thing: per-rank seen+unseen
macro-F1 (`final_eval.py`) + the geometry summary (curv, per-rank radius/aperture,
entail_ok).

| # | change vs baseline | hypothesis | success criterion |
|---|---|---|---|
| **B0** | none (baseline) | establishes the reference | trains stably; record all numbers |
| **E0** | `--geometry euclidean` (two sub-runs E0a/E0b, see below) | the missing corner of the 2×2: LoRA in **flat** space (cosine CLIP InfoNCE = open_clip `ClipLoss`, no SEL). Isolates LoRA-vs-full-FT from hyperbolic-vs-Euclidean | E0b (B0-matched) vs B0 = pure geometry delta; E0a (HAC LoRA recipe) vs Planktonzilla = LoRA-vs-full-FT. CL-only, `--lambda-sel` forced 0. Launch: `scripts/run_ablations_euclidean.sh`. **STATUS (06-15):** E0c (no-proj, =their arch) best so far = full-split SEEN species 0.634 vs full-FT 0.818 (**−0.18**, gap widens with depth). Rank ruled out as the cause; LR/schedule sweep in progress. Full record in `build-log.md` → "Euclidean-LoRA (E0c) tuning campaign". |
| **A1** | `--cl-mask same` | same-class false negatives hurt; masking helps (esp. unseen, where intra-clade structure matters) | unseen macro-F1 ↑ vs B0 at ≥1 rank, no seen regression |
| **A2** | `--contrastive angle` | distance-CL shrinks curvature + distorts hierarchy (ATMG); angle-CL is radius-free, SEL-aligned | curv stops gliding OR unseen ↑; watch seen doesn't crater |
| **A3** | `--sel-leak 0.1 --sel-tau 0.7` | leaky+tighter cones un-saturate apertures, spread upper ranks | order/family aperture < π/2; radii more stratified; entail_ok meaningful (not trivially 1.0) |
| **A4** | `--sel-uncertainty 0.5` | radius=uncertainty penalty pushes parents out → ranks deeper, ragged leaves depth-appropriate | upper-rank radii spread; no seen regression |
| **A6** | `--sel-text cumulative` | does the paper's independent-T_r choice actually help us, or does cumulative (shared-prefix tree) work better in the frozen+projector regime? | compare unseen F1 + geometry vs B0 (independent) |
| **A7** | `--lora-r ∈ {8, 32, 128}` (sub-sweep) | r=128 is HAC-B's *large*-config capacity setting, inherited as our default — **not** geometry-specific (LoRA acts on the Euclidean backbone matrices; rank is orthogonal to the hyperbolic output). HAC's own rank ablation (their Tab. 3) is nearly flat: r 8→128 buys only ~0.7% on VQA. So a smaller rank likely matches r=128 here at far fewer params — which **strengthens the parameter-efficiency thesis** (vs 64×H100 full-FT) | a low rank (8 or 32) within ~noise of r=128 on unseen F1 → adopt the smaller rank as the headline "parameter-efficient" config. Cheap; run on BIOSCAN, and ideally also on the Euclidean E0 (the lever is geometry-agnostic, so E0 is a clean place to measure it). **STATUS (06-15):** partially done on E0c — r=64 (at matched rsLoRA scale α/√r) ≈ or slightly BELOW r=32, so **rank is not the lever for the seen gap**; confirms the "flat" prior. CAUTION: α=r inflates the rsLoRA step √2× when r doubles → r=64/α=64 diverges; use `--lora-alpha` to hold scale. |
| **A5** | best-of(A1–A4,A6) combined | the winning terms compose | beats B0 on unseen; geometry clean |

Notes:
- **Do NOT combine until each is attributed.** A5 only after A1–A4 each measured alone.
- If an ablation needs a sub-sweep (e.g. `sel-tau` ∈ {0.5,0.7}, `cl-mask` level), that is
  a *named sub-experiment*, still one-variable, logged separately.
- Curvature: baseline is **free** (no guardrail). The `--curv-lr-scale` guardrail is NOT
  in the ladder — it was a patch for a problem A2/A4 may dissolve. Only reintroduce it as
  an explicit experiment IF free-curvature proves harmful AND the loss-side fixes don't.

---

## 4. What we already know (don't re-litigate)

- **Curvature glides to ~0.55–0.58 on planktonzilla at lr 5e-5 regardless of UNCHA terms
  or guardrail** (verified: UNCHA and no-UNCHA runs had near-identical curv curves). The
  earlier "UNCHA prevents collapse" claim was WRONG. Open: whether **angle-CL** (A2)
  changes this — ATMG says distance-CL is the curvature driver; untested in our setup.
- **False negatives are real**: 4.4% same-class / 15.7% ancestor pairs in a B=128
  planktonzilla batch (notebook `notebooks/false_negatives.ipynb`). `--cl-mask same`
  excludes the 4.4%; ancestor masking deliberately deferred.
- **UNCHA terms un-saturate mid-rank apertures** (A3) — measured: order aperture
  1.26→0.60, family 0.98→0.43 vs baseline. But this was at-a-glance on deleted
  checkpoints; re-establish cleanly in the ladder.
- **Planktonzilla (the paper) found supervised > their unmasked CLIP on SEEN.** Our lane
  is UNSEEN (supervised can't do it) + compute efficiency (2×A5000 vs 64×H100). Their
  CLIP ate the full false-negative penalty at batch 16,384 — A1 tests whether that's
  recoverable.

---

## 5. Execution order

**BIOSCAN first (own every cell), then Planktonzilla (headline + given full-FT ref):**

1. **Euclidean baselines on BIOSCAN** (the Euclidean column of the matrix):
   - **FT** — `scripts/run_euclidean_ft.sh` (full fine-tune, `train_euclidean_ft.py`).
   - **E0a / E0b** — `scripts/run_ablations_euclidean.sh` (LoRA, HAC recipe + B0-matched).
2. **B0 on BIOSCAN** → `final_eval.py` → reference for the hyperbolic ladder.
3. A1–A4 on BIOSCAN, each one-variable, attributed.
4. A5 (best combo) on BIOSCAN.
5. **Promote to Planktonzilla**: B0 + winning ablations + the Euclidean-LoRA baseline. The
   Euclidean **full-FT** reference is already given (paper Table 2/3 + released weights —
   re-eval the weights through our harness rather than retraining).
6. Planktonzilla `final_eval.py` (full 113,089 unseen) vs the paper numbers.

Each step: write the hypothesis + success criterion (already in §3), run, record the
result + verdict in `build-log.md`, decide promote/drop. No silent multi-variable runs.
