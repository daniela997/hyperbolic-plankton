"""Losses — hyperbolic contrastive + stacked entailment (SEL).

Piece 4 of the methodical re-implementation. The v1 core of the 2025 paper
"Hyperbolic Multimodal Representation Learning for Biological Taxonomies":
  - `hyperbolic_contrastive_loss`: MERU-style InfoNCE on negative Lorentzian distance.
  - `entailment_pos` / `entailment_neg`: the cone hinges (child in/out of parent's cone).
  - `sel_intra`: stacked entailment between consecutive taxonomic ranks (paper Eq. 3).
  - `sel_inter`: image entailed by its deepest available text rank.

All inputs are **space components** on the hyperboloid (from `model.encode_*`). Ragged
taxonomy is handled via per-rank `{rank}_valid` masks (from `model.encode_taxonomy`).

Deferred (later ablations, not v1): UNCHA uncertainty calibration, RCME hard-negative
images, SupCon, angular alignment. See scratchpad `mine/hyperbolic/loss.py`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import lorentz as L

__all__ = [
    "hyperbolic_contrastive_loss",
    "entailment_pos",
    "entailment_neg",
    "sel_intra",
    "sel_inter",
    "stacked_entailment_loss",
]


def hyperbolic_contrastive_loss(
    img: torch.Tensor, text: torch.Tensor, curv: torch.Tensor | float, scale: torch.Tensor | float
) -> torch.Tensor:
    """Symmetric InfoNCE with logits = -pairwise_dist * scale (MERU)."""
    B = img.shape[0]
    dist = L.pairwise_dist(img, text, curv)
    logits = -dist * scale
    targets = torch.arange(B, device=img.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def entailment_pos(
    parent: torch.Tensor, child: torch.Tensor, curv: torch.Tensor | float, r_min: float = 0.1
) -> torch.Tensor:
    """Positive entailment hinge: child should lie inside parent's cone.

    L = relu(oxy_angle(parent, child) - half_aperture(parent)). Returns shape (...,).
    """
    angle = L.oxy_angle(parent, child, curv)
    aperture = L.half_aperture(parent, curv, min_radius=r_min)
    return F.relu(angle - aperture)


def entailment_neg(
    parent: torch.Tensor,
    child: torch.Tensor,
    curv: torch.Tensor | float,
    r_min: float = 0.1,
    margin: float = 0.1,
) -> torch.Tensor:
    """Negative entailment hinge: a non-child should lie outside parent's cone.

    L = relu(half_aperture(parent) - oxy_angle(parent, child) + margin).
    """
    angle = L.oxy_angle(parent, child, curv)
    aperture = L.half_aperture(parent, curv, min_radius=r_min)
    return F.relu(aperture - angle + margin)


def _dense_ids(labels: list[str | None], device: torch.device) -> torch.Tensor:
    """Map hashable labels to dense int ids; None -> -1 (unknown)."""
    key_to_idx: dict[str, int] = {}
    out: list[int] = []
    for k in labels:
        if k is None:
            out.append(-1)
            continue
        out.append(key_to_idx.setdefault(k, len(key_to_idx)))
    return torch.tensor(out, dtype=torch.long, device=device)


def _edge_loss(
    parent: torch.Tensor,
    child: torch.Tensor,
    parent_valid: torch.Tensor,
    child_valid: torch.Tensor,
    parent_labels: list[str | None] | None,
    curv: torch.Tensor | float,
    r_min: float,
    margin: float,
    use_negatives: bool,
    stats: dict | None = None,
    stats_key: str = "",
) -> torch.Tensor:
    """Entailment loss for one (parent_rank -> child_rank) edge over the B×B grid.

    Positive pairs share the parent-rank label; negatives differ. Falls back to
    diagonal-only positives when no parent labels are given. Returns a scalar; if no
    valid positive pairs exist in the batch, returns 0 (the edge still counts in the
    SEL denominator — handled by the caller).
    """
    B = parent.shape[0]
    device = parent.device

    # Invalid (ragged-missing) entries arrive as ZERO vectors from encode_taxonomy. Fed
    # into the cone geometry they make ||x||->0, so half_aperture's asin argument -> inf
    # and its backward returns NaN (acos in oxy_angle likewise). These entries are masked
    # out of the loss below, so replacing them with a safe unit-norm placeholder leaves
    # every loss value unchanged while keeping zero vectors out of the geometry. We detect
    # invalid by mask (not by norm) so valid small-norm embeddings are untouched.
    def _sanitize(emb, valid_mask):
        unit = torch.zeros_like(emb)
        unit[..., 0] = 1.0
        return torch.where(valid_mask.unsqueeze(-1), emb, unit)

    parent = _sanitize(parent, parent_valid)
    child = _sanitize(child, child_valid)

    # valid pairs: parent valid in its row, child valid in its column
    valid = parent_valid.unsqueeze(0).expand(B, B) & child_valid.unsqueeze(1).expand(B, B)

    if parent_labels is not None:
        ids = _dense_ids(parent_labels, device)
        same = (ids.unsqueeze(0) == ids.unsqueeze(1)) & (ids.unsqueeze(0) >= 0) & (ids.unsqueeze(1) >= 0)
    else:
        same = torch.eye(B, dtype=torch.bool, device=device)
    pos_mask = valid & same
    neg_mask = valid & ~same

    # grid: child in rows, parent in cols
    p_grid = parent.unsqueeze(0).expand(B, -1, -1).reshape(B * B, -1)
    c_grid = child.unsqueeze(1).expand(-1, B, -1).reshape(B * B, -1)

    pos_all = entailment_pos(p_grid, c_grid, curv, r_min).reshape(B, B)
    if pos_mask.any():
        pos_loss = pos_all[pos_mask].mean()
    else:
        pos_loss = parent.new_zeros(())
    loss = pos_loss

    neg_loss = None
    if use_negatives and neg_mask.any():
        neg_all = entailment_neg(p_grid, c_grid, curv, r_min, margin).reshape(B, B)
        neg_loss = neg_all[neg_mask].mean()
        loss = 0.5 * (loss + neg_loss)

    if stats is not None:
        k = stats_key
        stats[f"{k}/pos"] = pos_loss.detach().item()
        stats[f"{k}/n_pos"] = int(pos_mask.sum())
        stats[f"{k}/neg"] = neg_loss.detach().item() if neg_loss is not None else 0.0
        stats[f"{k}/n_neg"] = int(neg_mask.sum())

    return loss


def sel_intra(
    text_embs: dict[str, torch.Tensor],
    taxonomy_batch: dict[str, list[str | None]],
    ranks: list[str],
    curv: torch.Tensor | float,
    r_min: float = 0.1,
    margin: float = 0.1,
    use_negatives: bool = True,
    stats: dict | None = None,
) -> torch.Tensor:
    """Stacked entailment between consecutive ranks (paper Eq. 3).

    Aggregate = (1 / #supervised_edges) * sum over edges of the per-edge loss. An edge
    (r-1, r) is "supervised" iff both ranks' embeddings are present; edges with no valid
    in-batch positive pairs contribute 0 to the numerator but still count in the
    denominator (so deeper edges aren't re-weighted when shallow ranks dominate).

    `stats` (if given) collects per-edge pos/neg components for logging.
    """
    edges = [
        (ranks[i - 1], ranks[i])
        for i in range(1, len(ranks))
        if ranks[i - 1] in text_embs and ranks[i] in text_embs
    ]
    if not edges:
        ref = next(iter(text_embs.values()))
        return ref.new_zeros(())

    total = None
    for parent_rank, child_rank in edges:
        loss = _edge_loss(
            parent=text_embs[parent_rank],
            child=text_embs[child_rank],
            parent_valid=text_embs[f"{parent_rank}_valid"],
            child_valid=text_embs[f"{child_rank}_valid"],
            parent_labels=taxonomy_batch.get(parent_rank),
            curv=curv,
            r_min=r_min,
            margin=margin,
            use_negatives=use_negatives,
            stats=stats,
            stats_key=f"sel_intra/{parent_rank}->{child_rank}",
        )
        total = loss if total is None else total + loss
    return total / len(edges)


def _deepest_text(
    text_embs: dict[str, torch.Tensor], ranks: list[str]
) -> tuple[torch.Tensor, list[str | None]]:
    """Per-sample deepest valid text embedding, and which rank-key it came from.

    Returns (embs [B, D], rank_per_sample [B]) where rank_per_sample[i] is the rank name
    used for sample i (None if no rank was valid)."""
    present = [r for r in ranks if r in text_embs]
    ref = text_embs[present[0]]
    B, D = ref.shape
    embs = torch.zeros(B, D, device=ref.device, dtype=ref.dtype)
    chosen: list[str | None] = [None] * B
    for i in range(B):
        for r in reversed(present):  # leaf -> root
            if bool(text_embs[f"{r}_valid"][i]):
                embs[i] = text_embs[r][i]
                chosen[i] = r
                break
    return embs, chosen


def sel_inter(
    img: torch.Tensor,
    text_embs: dict[str, torch.Tensor],
    taxonomy_batch: dict[str, list[str | None]],
    ranks: list[str],
    curv: torch.Tensor | float,
    r_min: float = 0.1,
    margin: float = 0.1,
    use_negatives: bool = True,
    stats: dict | None = None,
) -> torch.Tensor:
    """Image entailed by its deepest available text (text is parent, image is child).

    Positive pairs share the deepest-text label; negatives differ.
    """
    deepest, chosen_rank = _deepest_text(text_embs, ranks)
    valid = torch.tensor([r is not None for r in chosen_rank], dtype=torch.bool, device=img.device)
    # label per sample = the (rank, text) the deepest embedding came from
    labels = [
        taxonomy_batch[chosen_rank[i]][i] if chosen_rank[i] is not None else None
        for i in range(len(chosen_rank))
    ]
    return _edge_loss(
        parent=deepest,
        child=img,
        parent_valid=valid,
        child_valid=torch.ones_like(valid),
        parent_labels=labels,
        curv=curv,
        r_min=r_min,
        margin=margin,
        use_negatives=use_negatives,
        stats=stats,
        stats_key="sel_inter/text->image",
    )


def stacked_entailment_loss(
    img: torch.Tensor,
    text_embs: dict[str, torch.Tensor],
    taxonomy_batch: dict[str, list[str | None]],
    ranks: list[str],
    curv: torch.Tensor | float,
    r_min: float = 0.1,
    margin: float = 0.1,
    use_negatives: bool = True,
    stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full SEL = SEL-intra + SEL-inter. Returns (total, intra, inter).

    `stats` (if given) collects per-edge / per-term pos+neg components for logging.
    """
    intra = sel_intra(text_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives, stats=stats)
    inter = sel_inter(img, text_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives, stats=stats)
    return intra + inter, intra, inter
