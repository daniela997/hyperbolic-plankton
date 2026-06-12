"""Planktonzilla-faithful seen-pool train/val/test split.

Re-implements `notebooks/gen_datasets.py::stratified_split_by_dataset`: per source
`dataset`, hold out (val+test) stratified by the class label (`full` lineage string),
forcing singleton classes (count==1) entirely into train so every val/test class has
training support. Defaults to 60/20/20, seed=42 (the paper's values).

The class label the paper stratifies on is `label` = the integer-encoded `tax_label`
(== our `full` string). We add a transient `full` column, stratify on it, and split.
"""

from __future__ import annotations

from collections import Counter

from datasets import concatenate_datasets

from .data import _clean

_RANK_COLS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def _add_full_column(ds, num_proc=8):
    """Attach the `full` lineage string (paper's class label) as a column for stratify."""

    def _full(batch):
        out = []
        n = len(batch[_RANK_COLS[0]])
        for i in range(n):
            toks = [t for t in (_clean(batch[c][i]) for c in _RANK_COLS) if t is not None]
            out.append(" ".join(toks) if toks else "unknown")
        return {"full": out}

    return ds.map(_full, batched=True, num_proc=num_proc, desc="add full-class column")


def stratified_split_seen(
    seen_ds,
    seed: int = 42,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    num_proc: int = 8,
):
    """Return (train_ds, val_ds, test_ds) over the seen pool — paper recipe.

    Per `dataset`: singleton classes -> train; remainder split train / (val+test) then
    val / test, both stratified by `full` class. `seen_ds` must already exclude the 4
    held-out datasets (use `data.split_seen_unseen` first).
    """
    from datasets import ClassLabel

    seen_ds = _add_full_column(seen_ds, num_proc=num_proc)

    # Encode `full` as ONE global ClassLabel over the whole seen pool, exactly as the paper
    # does at dataset-build time (gen_datasets: ClassLabel(names=sorted(set(tax_label)))).
    # HF's stratified shuffle groups by the integer label, so the int encoding ORDER changes
    # which rows land in each split even at the same seed. Building the encoding once globally
    # (not per-dataset, per-call) is what makes our split reproduce theirs row-for-row; a
    # per-call sorted(set(sub)) gives each sub-dataset a different int map -> different rows.
    global_names = sorted(set(seen_ds["full"]))
    seen_ds = seen_ds.cast_column("full", ClassLabel(names=global_names))

    train_parts, val_parts, test_parts = [], [], []
    dataset_names = sorted(set(seen_ds["dataset"]))

    for dname in dataset_names:
        sub = seen_ds.filter(lambda b: [d == dname for d in b["dataset"]], batched=True, num_proc=num_proc)
        labels = sub["full"]  # now integer class ids (global encoding), like their `label`
        counts = Counter(labels)
        singletons = {k for k, v in counts.items() if v == 1}

        sing_idx = [i for i, y in enumerate(labels) if y in singletons]
        rem_idx = [i for i in range(len(sub)) if y_not_singleton(labels[i], singletons)]
        ds_sing = sub.select(sing_idx) if sing_idx else None
        ds_rem = sub.select(rem_idx) if rem_idx else None

        if ds_rem is None or len(ds_rem) == 0:
            train_parts.append(sub)
            continue

        n = len(ds_rem)
        # train / (val+test), stratified (fall back to unstratified if a class is too rare)
        sp = _safe_split(ds_rem, int(n * (test_frac + val_frac)), seed)
        train_part, valtest = sp["train"], sp["test"]
        # val / test
        sp2 = _safe_split(valtest, int(n * val_frac), seed)
        test_part, val_part = sp2["train"], sp2["test"]

        if ds_sing is not None:
            train_part = concatenate_datasets([train_part, ds_sing])
        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

    train_ds = concatenate_datasets(train_parts)
    val_ds = concatenate_datasets(val_parts) if val_parts else None
    test_ds = concatenate_datasets(test_parts) if test_parts else None
    return train_ds, val_ds, test_ds


def y_not_singleton(y, singletons):
    return y not in singletons


def _safe_split(ds, test_size, seed):
    """train_test_split stratified by the pre-encoded global `full` ClassLabel, falling back
    to unstratified on ValueError (HF raises when a class has too few members to stratify) —
    the paper's exact `try/except ValueError`.

    `ds.full` must ALREADY be a ClassLabel typed with the global encoding (done once in
    stratified_split_seen); we do NOT re-derive a per-call encoding here, so the int label
    map is identical across every sub-dataset — matching how the paper stratifies on its one
    global `label` column.
    """
    try:
        out = ds.train_test_split(
            test_size=test_size, shuffle=True, seed=seed, stratify_by_column="full"
        )
    except ValueError:
        out = ds.train_test_split(test_size=test_size, shuffle=True, seed=seed)
    return out
