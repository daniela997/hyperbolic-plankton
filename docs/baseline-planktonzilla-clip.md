# Baseline: official Planktonzilla CLIP (full-FT) — the bar our models are measured against

The paper's full-FT CLIP (`hf-hub:project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt`,
ViT-B/16 OpenAI+PZ), run **off-the-shelf zero-shot through OUR eval** (cosine-argmax over
present classes, per-rank truncate-deepest macro-F1 = `taxonomic_macro_f1`), on the **FULL**
seen-test and unseen splits, unannotated rows excluded. This is the calibrated full-FT
reference for every LoRA / hyperbolic comparison.

Measured 2026-06-12 on the rebuilt split (global-ClassLabel fix, commit `f731e34`). Numbers
are stable to <±0.006 across split rebuilds and identical fp16/fp32.

## SEEN (full test split, 584,966 imgs, 362 present classes)

| rank | ours (baseline) | paper Table 2 | Δ |
|---|---|---|---|
| kingdom | 0.955 | 0.961 | −0.006 |
| phylum  | 0.935 | 0.905 | +0.030 |
| class   | 0.889 | 0.865 | +0.024 |
| order   | 0.873 | 0.853 | +0.020 |
| family  | 0.857 | 0.822 | +0.035 |
| genus   | 0.832 | 0.797 | +0.035 |
| species | 0.818 | 0.786 | +0.032 |

## UNSEEN (full split, 113,089 imgs, 220 novel-lineage classes)

| rank | ours (baseline) | paper Table 3 | Δ |
|---|---|---|---|
| kingdom | 0.376 | 0.408 | −0.032 |
| phylum  | 0.183 | 0.204 | −0.021 |
| class   | 0.113 | 0.118 | −0.005 |
| order   | 0.080 | 0.088 | −0.008 |
| family  | 0.065 | 0.072 | −0.007 |
| genus   | 0.052 | 0.065 | −0.013 |
| species | 0.041 | 0.055 | −0.014 |

## Best LoRA so far vs this baseline (E0c, r=32, all-blocks, no-proj, 20ep)

Run `7r7cvoa3` (Euclidean-LoRA, `--no-proj`, r=32/α=32, all 12+12 blocks, lr 2e-4 warmupcos,
fp16). Full-split `test/*`, same eval as the baseline. **This is the "before" — the bar the
bf16/OneCycle/LR-sweep changes must beat.**

SEEN (584,966 / 362):

| rank | full-FT | LoRA E0c | gap |
|---|---|---|---|
| kingdom | 0.955 | 0.915 | −0.040 |
| phylum  | 0.935 | 0.877 | −0.058 |
| class   | 0.889 | 0.773 | −0.116 |
| order   | 0.873 | 0.745 | −0.128 |
| family  | 0.857 | 0.698 | −0.159 |
| genus   | 0.832 | 0.669 | −0.163 |
| species | 0.818 | 0.634 | −0.184 |

UNSEEN (113,089 / 220):

| rank | full-FT | LoRA E0c | gap |
|---|---|---|---|
| kingdom | 0.376 | 0.294 | −0.082 |
| phylum  | 0.183 | 0.155 | −0.028 |
| class   | 0.113 | 0.086 | −0.027 |
| order   | 0.080 | 0.056 | −0.024 |
| family  | 0.065 | 0.051 | −0.014 |
| genus   | 0.052 | 0.046 | −0.006 |
| species | 0.041 | 0.033 | −0.008 |

**The seen gap widens monotonically with depth** (kingdom −0.04 → species −0.18): LoRA
recovers coarse taxonomy but not fine-grained discrimination. Ruled out as the cause so far:
LoRA rank (r=64 at matched rsLoRA scale ≈ r=32; r=64/α=64 diverged on step-size), and
training duration is being tested (≤20ep constraint). Next levers: OneCycle + a swept peak
LR (in progress), then MLP adaptation if needed. Unseen gaps are small in absolute terms
(both near the floor) — closing unseen is the hyperbolic method's job, not LoRA's.

## How to read this baseline

- **Use OUR numbers (the "ours" columns), not the paper's**, as the bar. They are the full-FT
  model scored through the *identical* eval our models use, so the comparison is
  confound-free. The paper Δ is shown only to document fidelity (±0.035, all snapshot floor —
  see `[[planktonzilla-eval-fidelity]]`: weights/precision/F1/split all verified, residual is
  dataset-version skew + macro-F1 tail bias, not a pipeline bug).
- A trained LoRA / hyperbolic model **matching these "ours" numbers ≈ matching full-FT**. The
  thesis claim (LoRA can match full-FT; hyperbolic helps) is judged against THIS, per rank,
  seen and unseen — not against a single mean-F1.
- Reproduce with: `PYTHONPATH=src:scripts python scripts/acceptance_test_eval.py`.
