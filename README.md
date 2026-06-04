# hyperbolic-plankton

Hyperbolic vision-language representation learning for plankton taxonomy, targeting
unseen-species generalization at low compute. Built on the existing hyperbolic
codebase at `/home/daniela/mine/hyperbolic/`.

## One-line thesis

Via hyperbolic approaches we can match/outperform the Planktonzilla CLIP models on
**unseen plankton species** at a **fraction of the training compute** (2× RTX A5000
vs the paper's 64× H100).

## Docs (the shared source of truth)

- [`docs/planktonzilla.md`](docs/planktonzilla.md) — our reference understanding of
  the Planktonzilla paper, dataset, codebase, and eval protocol. Confidence-flagged
  (`[stated]` / `[inferred]` / `[unknown]`).
- [`docs/project-plan.md`](docs/project-plan.md) — our approach, thesis, target venue
  (ECCV 2026 *Beyond Euclidean* workshop), experiment design, decisions, and what
  needs building.
- [`docs/related-work.md`](docs/related-work.md) — the four-paper arc (MERU → CLIBD →
  Hyperbolic Taxonomies → HAC), what each contributes, our novelty cell, and the
  HAC-derived LoRA recipe we copy.
- [`docs/hac-implementation.md`](docs/hac-implementation.md) — the exact HAC reference
  wiring (`/home/daniela/other/HAC`): freeze→PEFT→final-LN sequence, the hyperbolic
  lift, the working ViT-B LoRA config, and what we copy vs. change.

## Status

Planning / understanding phase. No code yet — capturing shared understanding first.

## Key external references

- Planktonzilla repo: `/home/daniela/other/planktonzilla/` (+ `planktonzilla.pdf`)
- HF dataset: `project-oceania/planktonzilla-17M`
- Hyperbolic codebase (reused): `/home/daniela/mine/hyperbolic/`
- Papers: `/home/daniela/mine/hyperbolic/papers/` (MERU, RCME, CLIBD, Hyperbolic
  Taxonomies, Global/Local Entailment, UNCHA)
- Venue: <https://sites.google.com/view/beyondeuclidean/>
