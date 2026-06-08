"""BIOSCAN-1M data bridge — CLIBD HDF5 splits -> taxonomy items.

Complete-taxonomy (order/family/genus/species, species-complete) counterpart to the
ragged planktonzilla loader in `data.py`. Used as the clean testbed for training-recipe
ablations (e.g. distance vs angle contrastive) before reintroducing raggedness.

Reads CLIBD's `BioScan_data_in_splits.hdf5` (download: `bioscan-ml/clibd`, merge the
`split_data/splitted_files/*.part*` parts). The seen/unseen partition IS the file: it
ships named HDF5 groups, so the split is exactly the one CLIBD and the Hyperbolic-
Taxonomies (2025) paper used. Groups we consume:
  - `train_seen`   : the paper's 36k training set (complete species labels)
  - `test_seen`    : seen-taxa test set
  - `test_unseen`  : unseen-taxa test set
  - `val_seen`/`val_unseen` : optional periodic-eval sets

Each group stores: `image` (byte-encoded, padded) + `image_mask` (true length) and the
rank strings `order`/`family`/`genus`/`species` as utf-8 byte arrays (DATA.md).
"""

from __future__ import annotations

import io

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .data import _BLANK_IMAGE, build_taxonomy

# BIOSCAN ranks (coarse->fine). No kingdom/phylum/class — a 4-rank tree, complete to
# species. These are both the HDF5 dataset names and (capitalised) the `build_taxonomy`
# columns, so we map lower<->Title here.
BIOSCAN_RANK_COLUMNS = ["Order", "Family", "Genus", "Species"]
BIOSCAN_RANKS = [c.lower() for c in BIOSCAN_RANK_COLUMNS]


class BioscanHDF5Dataset(Dataset):
    """One CLIBD HDF5 split group, emitting `{image, taxonomy, proposed_label}` — the same
    item shape as `HFTaxonomyDataset`, so the existing collator/loss/eval consume it
    unchanged. `proposed_label` is the species string (the species-complete class identity).
    """

    def __init__(self, hdf5_path: str, group: str):
        self.hdf5_path = hdf5_path
        self.group = group
        # Open lazily per-worker (h5py handles are not fork-safe); cache None until first use.
        self._h5: h5py.File | None = None
        with h5py.File(hdf5_path, "r") as f:
            self._len = len(f[group]["image"])

    def _grp(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5[self.group]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        g = self._grp()
        enc = g["image"][idx].astype(np.uint8)
        enc = enc[: g["image_mask"][idx]]  # strip padding (CLIBD dataset.py:243-246)
        try:
            image = Image.open(io.BytesIO(enc)).convert("RGB")
        except Exception:
            image = _BLANK_IMAGE.copy()
        # rank strings -> a row dict keyed by the Title-case columns build_taxonomy expects
        row = {col: g[col.lower()][idx].decode("utf-8") for col in BIOSCAN_RANK_COLUMNS}
        return {
            "image": image,
            "taxonomy": build_taxonomy(row, rank_columns=BIOSCAN_RANK_COLUMNS),
            "proposed_label": row["Species"] or "unknown",
        }
