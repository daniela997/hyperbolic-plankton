"""Gromov δ-hyperbolicity of the BIOSCAN taxonomy — per subtree, tied to fan-out.

Model-free, CPU-only. Reads ONLY the label hierarchy from the BIOSCAN HDF5 (order/family/genus/
species strings); no embeddings, no checkpoints, no GPU.

WHY per-subtree, not global: an UNWEIGHTED tree path metric is exactly 0-hyperbolic (δ=0) by
construction, so a global δ on the bare taxonomy is trivially ~0 and says nothing. The interesting
structure is HOW δ_rel behaves on the ULTRAMETRIC leaf metric of each subtree, which exposes the
shape: a "bushy" star-like subtree (one parent, many leaves at equal depth = high fan-out, e.g. an
order with ~100 species) vs a "deep" chain. This ties δ to the FAN-OUT variable the feasibility
sweep (scripts/feasibility_sweep.py) found binding: high-fan-out coarse ranks are where cone
containment fails. We report δ_rel(subtree) against subtree fan-out to test that link.

δ is computed by the standard 4-point sampling estimator from Lars Mennen's reference
(https://lars76.github.io/2020/07/22/computing-gromov-hyperbolicity.html), vendored as
`sample_hyperbolicity` below — we do NOT reimplement the Gromov machinery. δ_rel = 2δ/diam.

Leaf metric: ULTRAMETRIC from taxonomy depth — d(leaf_i, leaf_j) = 2 * (D - depth(LCA_ij)) where
D=4 ranks (order..species). Pure tree ⇒ additive tree metric ⇒ δ≈0 (sanity check the estimator).
(A learned-embedding δ would replace this leaf metric with a pairwise-embedding-distance matrix.)

Run:  PYTHONPATH=src python scripts/delta_hyperbolicity.py
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np


RANKS = ["order", "family", "genus", "species"]


def load_lineages(hdf5, split="train_seen"):
    import h5py
    f = h5py.File(hdf5, "r")
    g = f[split]
    cols = {}
    for r in RANKS:
        cols[r] = [x.decode() if isinstance(x, bytes) else str(x) for x in g[r][:]]
    f.close()
    # one lineage tuple per row; drop rows with any empty rank
    lin = []
    for i in range(len(cols["order"])):
        t = tuple(cols[r][i] for r in RANKS)
        if all(x and x != "" and x.lower() != "nan" for x in t):
            lin.append(t)
    return lin


def lca_depth(a, b):
    """Depth (0..4) of the lowest common ancestor of two lineage tuples (0 = share nothing)."""
    d = 0
    for x, y in zip(a, b):
        if x == y:
            d += 1
        else:
            break
    return d


def ultrametric(leaves):
    """Pairwise ultrametric leaf distance matrix: d_ij = 2*(D - depth(LCA))."""
    D = len(RANKS)
    n = len(leaves)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 2.0 * (D - lca_depth(leaves[i], leaves[j]))
            M[i, j] = M[j, i] = d
    return M


def sample_hyperbolicity(dist, num_samples=50000, seed=0):
    """4-point-sampling Gromov δ estimator. Vendored from Lars Mennen's reference:
    https://lars76.github.io/2020/07/22/computing-gromov-hyperbolicity.html

    For a random 4-tuple, sort the three perfect-matching pairwise-distance sums S1≥S2≥S3;
    the 4-point δ is (S1−S2)/2. Returns max over samples (the δ estimate). δ_rel = 2δ/diam."""
    n = dist.shape[0]
    if n < 4:
        return 0.0, 0.0
    diam = dist.max()
    if diam == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    delta_max = 0.0
    for _ in range(num_samples):
        a, b, c, d = rng.integers(0, n, 4)
        s = sorted([dist[a, b] + dist[c, d],
                    dist[a, c] + dist[b, d],
                    dist[a, d] + dist[b, c]], reverse=True)
        delta_max = max(delta_max, (s[0] - s[1]) / 2)
    return float(delta_max), float(2 * delta_max / diam)


def main():
    hdf5 = os.environ.get(
        "BIOSCAN_HDF5",
        "/scratch/daniela/bioscan1m/data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5")
    lin = load_lineages(hdf5)
    uniq = sorted(set(lin))
    print(f"BIOSCAN train_seen: {len(lin)} rows, {len(uniq)} unique species-lineages\n")

    # ---- global δ_rel on the unique-lineage ultrametric (sanity: tree ⇒ ~0) ----
    if len(uniq) <= 1200:
        d_glob, dr_glob = sample_hyperbolicity(ultrametric(uniq))
        print(f"GLOBAL (unique lineages, ultrametric): δ={d_glob:.3f}  δ_rel={dr_glob:.4f}  "
              f"{'✅ ~0 (tree-like, probe sane)' if dr_glob < 0.05 else '(not ~0 — investigate)'}")

    # ---- per-subtree: δ_rel vs fan-out, at each internal rank ----
    # A subtree = all species under one parent at a given rank. Its fan-out = # leaf species.
    print("\nPer-subtree δ_rel vs fan-out (does bushiness ↔ δ?):")
    for parent_rank in ("order", "family", "genus"):
        depth = RANKS.index(parent_rank) + 1
        groups = defaultdict(set)
        for t in uniq:
            groups[t[:depth]].add(t)  # key by ancestor prefix
        rows = []
        for key, leaves in groups.items():
            leaves = sorted(leaves)
            if len(leaves) < 4:
                continue
            _, dr = sample_hyperbolicity(ultrametric(leaves))
            rows.append((len(leaves), dr))
        if not rows:
            print(f"  {parent_rank:7s}: (no subtree with ≥4 species)")
            continue
        rows.sort()
        fanouts = np.array([r[0] for r in rows])
        drs = np.array([r[1] for r in rows])
        # correlation of fan-out with δ_rel
        corr = np.corrcoef(fanouts, drs)[0, 1] if len(rows) > 2 else float("nan")
        print(f"  {parent_rank:7s}: {len(rows):3d} subtrees  fan-out[{fanouts.min()}..{fanouts.max()}]"
              f"  δ_rel mean={drs.mean():.3f} max={drs.max():.3f}  corr(fanout,δ_rel)={corr:+.2f}")
        # show the bushiest few
        big = sorted(rows, reverse=True)[:3]
        print(f"          bushiest: " + ", ".join(f"K={k}→δ_rel={d:.3f}" for k, d in big))


if __name__ == "__main__":
    main()
    print("\n✅ δ-hyperbolicity probe complete (CPU-only, taxonomy labels only, no GPU).")
