# How to count Planktonzilla taxonomy classes (and reproduce the paper's numbers)

The paper reports: **5 kingdoms, 29 phyla, 62 classes, 155 orders, 242 families,
289 genera, 201 species** (and "601 distinct plankton classes"). Reproducing these from
the raw `Kingdom..Species` columns is subtle — naive counts disagree with the paper. This
doc fixes the **one correct counting rule**.

## The rule: distinct **lineage paths**, ragged-tolerant, case-sensitive

A class at rank *R* = a **distinct path** from Kingdom down to *R*, built from the
**populated** ancestor ranks (skipping gaps), compared **case-sensitively**.

```python
def rank_count(rows, depth):  # depth: 0=Kingdom .. 6=Species
    paths = set()
    for r in rows:
        if r[TAX[depth]] is None:
            continue
        # tuple of populated ranks from Kingdom..depth (skip None gaps = ragged-tolerant)
        path = tuple(r[TAX[k]] for k in range(depth + 1) if r[TAX[k]] is not None)
        paths.add(path)
    return len(paths)
```

This reproduces the paper at every rank (Family/Genus −1 = the single missing rare taxon,
see below):

| rank | this rule | paper |
|---|---|---|
| Kingdom | 5 | 5 |
| Phylum | 29 | 29 |
| Class | 62 | 62 |
| Order | 155 | 155 |
| Family | 241 | 242 |
| Genus | 288 | 289 |
| Species | 201 | 201 |

## Why the naive counts are wrong

- **Bare field distinct** (`set(df[col])`): undercounts the deep ranks because a species
  epithet (or genus name) is **reused across different parents** — e.g. the same Species
  string under two genera is two species, but one bare value. Gives Species=192 (wrong).
- **Contiguous-path** (require *all* ranks 0..R populated): undercounts by excluding
  **ragged / incertae sedis** lineages — taxa with a genuinely undefined intermediate rank
  (e.g. `katablepharis remigera` has no Order; its Class is literally
  `cryptophyta incertae sedis`). Gives Species=200 (off by 1). The path rule above is
  *ragged-tolerant* (skips the None gap), recovering it → 201.
- **Case-insensitive**: the source 17M itself contains case-collisions for a few taxa
  (`Bacillariophyceae`/`bacillariophyceae`, `Lithodesmiales`/`lithodesmiales`,
  `Lithodesmiaceae`/`lithodesmiaceae`). The paper counts them **case-sensitively** (so
  Class=62, Order=155 only match with the case-dups kept). This is a known data-quality
  quirk in the published 17M; neither their pipeline nor ours normalizes case.

## Residual vs the paper

After the correct rule: **Family −1, Genus −1**, everything else exact. These 2 taxa are
ultra-rare (single-specimen) and absent from our public-17M snapshot — a dataset-version
difference in the rare tail, not a pipeline error. The empty markers in the data are
**`None` only** (no `""`, `"nan"`, or whitespace). The eval reproduces the paper's
Table 2/3 within ±0.03 using our class set, confirming the residual is immaterial.

## Split definition (paper §3.1, Experiments 1 & 2)

- **Unseen (Exp-2):** the 4 held-out source datasets (`global_uvp5`, `planktoscope`,
  `planktonset1.0`, `syke_ifcb_2022`); select the **220 plankton classes / 113,089 samples**
  absent from training. We match this **row-exact**.
- **Seen (Exp-1):** the remaining (in-domain) data, **60/20/20 train/val/test**, stratified
  by **both source dataset and taxonomic label** — i.e. their `stratified_split_by_dataset`
  (per-`dataset` stratified `train_test_split`, seed 42, singletons→train, stratify with
  unstratified fallback). Reproduced in `build_splits.py` / `stratified_split_seen`.
