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
