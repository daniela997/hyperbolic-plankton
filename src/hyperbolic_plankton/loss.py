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
    "euclidean_contrastive_loss_ddp",
    "entailment_pos",
    "entailment_neg",
    "sel_intra",
    "sel_inter",
    "stacked_entailment_loss",
]


def hyperbolic_contrastive_loss(
    img: torch.Tensor, text: torch.Tensor, curv: torch.Tensor | float, scale: torch.Tensor | float,
    class_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with logits = -pairwise_dist * scale (MERU).

    Single-process / square (B×B) form. For DDP with cross-GPU negatives use
    `hyperbolic_contrastive_loss_ddp`, which gathers text/image across processes so each
    image is scored against the GLOBAL batch (MERU/HAC behaviour).

    `class_ids` [B] (optional): same-class off-diagonal cells are masked out of the
    negatives (true positives mis-treated as negatives). None = standard InfoNCE.
    """
    B = img.shape[0]
    dist = L.pairwise_dist(img, text, curv)
    logits = -dist * scale
    if class_ids is not None:
        mask = _false_negative_mask(class_ids, class_ids, 0)
        logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(B, device=img.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def _gather_across_processes(t: torch.Tensor) -> torch.Tensor:
    """Differentiable all-gather → [B*world, D] (gradients scatter back to each rank).

    Mirrors HAC's `gather_across_processes` (uses `torch.distributed.nn.all_gather`, the
    autograd-aware variant — plain `dist.all_gather` would detach the other ranks). Returns
    the input unchanged when not running under DDP.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t
    from torch.distributed.nn import all_gather as nn_all_gather

    return torch.cat(list(nn_all_gather(t)), dim=0)


def _gather_labels(ids: torch.Tensor) -> torch.Tensor:
    """All-gather an int label tensor [B] -> [B*world] (non-differentiable; labels carry no
    grad). Returns the input unchanged off-DDP. Used to build the cross-GPU negative mask."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return ids
    out = [torch.zeros_like(ids) for _ in range(dist.get_world_size())]
    dist.all_gather(out, ids)
    return torch.cat(out, dim=0)


def _false_negative_mask(local_ids: torch.Tensor, all_ids: torch.Tensor, rank: int) -> torch.Tensor:
    """Bool mask [B, B*world] marking negative cells to SUPPRESS: same-class pairs that are
    not the matched diagonal. `*_ids` are per-sample class ids; cell (i,j) is suppressed when
    local_ids[i] == all_ids[j] and j is not i's own diagonal (i + B*rank)."""
    B = local_ids.shape[0]
    same = local_ids[:, None] == all_ids[None, :]            # [B, B*world]
    # id -1 = unknown (None lineage): two unknowns are NOT the same class, so never mask
    # them against each other (else we'd drop legitimate negatives). Only matters on ragged
    # datasets like Planktonzilla; BIOSCAN is complete-to-species so has no -1.
    same &= local_ids[:, None] != -1
    diag = torch.arange(B, device=local_ids.device) + B * rank
    same[torch.arange(B, device=local_ids.device), diag] = False  # keep the true positive
    return same


def hyperbolic_contrastive_loss_ddp(
    img: torch.Tensor, text: torch.Tensor, curv: torch.Tensor | float, scale: torch.Tensor | float,
    class_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """MERU/HAC contrastive loss with cross-GPU negatives.

    Each rank's local `img`/`text` (B) are scored against the GLOBALLY gathered
    text/image (B*world), so the negative set is the full effective batch — not just the
    local micro-batch. Targets are shifted by `B * rank` to hit the matched diagonal in the
    gathered set. Falls back to the local square loss when not under DDP.

    `class_ids` [B] (optional): per-sample class id. When given, same-class off-diagonal
    cells are masked OUT of the negatives (they are true positives mis-treated as negatives —
    pervasive in clade-imbalanced plankton batches). None = no masking (standard InfoNCE).
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return hyperbolic_contrastive_loss(img, text, curv, scale, class_ids)

    B = img.shape[0]
    rank = dist.get_rank()
    all_img = _gather_across_processes(img)
    all_text = _gather_across_processes(text)

    # local image vs all text; local text vs all image (MERU/HAC orientation)
    img_logits = -L.pairwise_dist(img, all_text, curv) * scale   # [B, B*world]
    text_logits = -L.pairwise_dist(text, all_img, curv) * scale  # [B, B*world]
    if class_ids is not None:
        mask = _false_negative_mask(class_ids, _gather_labels(class_ids), rank)
        img_logits = img_logits.masked_fill(mask, float("-inf"))
        text_logits = text_logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(B, device=img.device) + B * rank
    return 0.5 * (F.cross_entropy(img_logits, targets) + F.cross_entropy(text_logits, targets))


def hyperbolic_angle_contrastive_loss_ddp(
    img: torch.Tensor, text: torch.Tensor, curv: torch.Tensor | float, scale: torch.Tensor | float,
    class_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Angle-based contrastive loss (ATMG, "Accept the Modality Gap"), cross-GPU negatives.

    Logits are exterior angles instead of distances, so the loss is radius-free (does not
    pin images to a fixed shell) and speaks the same geometric quantity (oxy_angle) as SEL.
    Asymmetric apex convention (ATMG models.py): text is the apex (minimise angle at text),
    image fans out (maximise angle at image) — i.e. the image is an instance of its deepest
    text. Targets shifted by B*rank; falls back to the local square form off-DDP.

    `class_ids` [B] (optional): same-class off-diagonal cells masked out of negatives.
    """
    import torch.distributed as dist

    B = img.shape[0]
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        all_img, all_text, rank = img, text, 0
        all_ids = class_ids
    else:
        all_img = _gather_across_processes(img)
        all_text = _gather_across_processes(text)
        rank = dist.get_rank()
        all_ids = _gather_labels(class_ids) if class_ids is not None else None

    # angle at image (maximise for match) / angle at text (minimise for match)
    img_logits = L.pairwise_oxy_angle(img, all_text, curv) * scale    # [B, B*world]
    text_logits = -L.pairwise_oxy_angle(text, all_img, curv) * scale  # [B, B*world]
    if class_ids is not None:
        mask = _false_negative_mask(class_ids, all_ids, rank)
        img_logits = img_logits.masked_fill(mask, float("-inf"))
        text_logits = text_logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(B, device=img.device) + B * rank
    return 0.5 * (F.cross_entropy(img_logits, targets) + F.cross_entropy(text_logits, targets))


def euclidean_contrastive_loss_ddp(
    img: torch.Tensor, text: torch.Tensor, curv: torch.Tensor | float, scale: torch.Tensor | float,
    class_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flat-space CLIP InfoNCE with cross-GPU negatives (open_clip `ClipLoss`).

    The Euclidean baseline: cosine-similarity InfoNCE, identical to open_clip's `ClipLoss`
    (`logit_scale * img_n @ text_n.T`, symmetric CE, labels shifted by B*rank). Isolates the
    LoRA variable from the hyperbolic-geometry variable — same frozen backbone + LoRA +
    projector as the hyperbolic model, but flat space and no entailment.

    `img`/`text` are the projected Euclidean features (`encode_*(project=False)`); they are
    L2-normalised here. `curv` is accepted and ignored (signature-compatible with the
    hyperbolic CL functions so the train loop can swap losses without special-casing args).
    `class_ids` masks same-class off-diagonal negatives as in the hyperbolic versions.
    """
    import torch.distributed as dist

    B = img.shape[0]
    img = F.normalize(img, dim=-1)
    text = F.normalize(text, dim=-1)
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        all_img, all_text, rank, all_ids = img, text, 0, class_ids
    else:
        all_img = _gather_across_processes(img)
        all_text = _gather_across_processes(text)
        rank = dist.get_rank()
        all_ids = _gather_labels(class_ids) if class_ids is not None else None

    img_logits = scale * img @ all_text.T   # [B, B*world]
    text_logits = scale * text @ all_img.T  # [B, B*world]
    if class_ids is not None:
        mask = _false_negative_mask(class_ids, all_ids, rank)
        img_logits = img_logits.masked_fill(mask, float("-inf"))
        text_logits = text_logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(B, device=img.device) + B * rank
    return 0.5 * (F.cross_entropy(img_logits, targets) + F.cross_entropy(text_logits, targets))


def entailment_pos(
    parent: torch.Tensor, child: torch.Tensor, curv: torch.Tensor | float,
    r_min: float = 0.1, tau: float = 1.0, leak: float = 0.0, lam_u: float = 0.0,
) -> torch.Tensor:
    """Positive entailment hinge: child should lie inside parent's cone.

    Base: L = relu(oxy_angle(parent, child) - tau * half_aperture(parent)).

    Two optional terms (adapted from UNCHA, arXiv 2603.22042) that fix the upper-rank
    collapse — once a child is contained the bare hinge gives ZERO gradient, so ranks pile
    at the origin where the cone saturates to pi/2:

    - `leak` (>0): always-on `leak * oxy_angle`, a Leaky-ReLU-style continued gradient that
      keeps pulling the child onto the parent's radial axis even when already contained
      (UNCHA Eq. 14). Aligning children to the axis + their distinctness from the parent
      forces them apart in RADIUS (the only free dimension left on a shared ray).
    - `tau` (<1): tightens the effective cone (tau * aperture), countering the pi/2
      saturation so the hinge stays active longer.
    - `lam_u` (>0): `lam_u * softplus(-||parent||)` = a radius/uncertainty penalty (UNCHA
      Eq. 7/15 core). softplus(-||x||) is high near the origin, so this pushes the PARENT
      outward — breaking the radial-direction symmetry so children end up DEEPER, and giving
      ragged leaves a depth-appropriate (uncertainty-appropriate) radius for free.

    All three are scale-free / angular (no radial margin tuned to the geometry). Defaults
    (tau=1, leak=0, lam_u=0) reproduce the plain hinge exactly.
    """
    angle = L.oxy_angle(parent, child, curv)
    aperture = L.half_aperture(parent, curv, min_radius=r_min)
    loss = F.relu(angle - tau * aperture)
    if leak > 0.0:
        loss = loss + leak * angle
    if lam_u > 0.0:
        loss = loss + lam_u * F.softplus(-torch.linalg.norm(parent, dim=-1))
    return loss


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
    tau: float = 1.0,
    leak: float = 0.0,
    lam_u: float = 0.0,
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

    pos_all = entailment_pos(p_grid, c_grid, curv, r_min, tau, leak, lam_u).reshape(B, B)
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
    tau: float = 1.0,
    leak: float = 0.0,
    lam_u: float = 0.0,
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
            tau=tau,
            leak=leak,
            lam_u=lam_u,
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

    `text_embs` are the per-rank embeddings to draw the deepest text from. Paper Eq. 4
    entails the image by `T_R'` = the deepest *per-rank* (independent) text — the SAME
    `T_r` objects SEL-intra uses, just selected at the deepest valid rank (NOT the
    cumulative `full` string; that is used only for contrastive alignment). So pass the
    independent embeddings here. Positive pairs share the deepest-rank label; the labels
    still come from the cumulative `taxonomy_batch` (two images share a leaf iff their
    cumulative lineage matches).
    """
    deepest, chosen_rank = _deepest_text(text_embs, ranks)
    valid = torch.tensor([r is not None for r in chosen_rank], dtype=torch.bool, device=img.device)
    # label per sample = the cumulative lineage at the chosen deepest rank (for masking)
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
    sel_text_embs: dict[str, torch.Tensor] | None = None,
    tau: float = 1.0,
    leak: float = 0.0,
    lam_u: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full SEL = SEL-intra + SEL-inter. Returns (total, intra, inter).

    Per the paper, BOTH SEL terms use the **per-rank (independent)** text embeddings `T_r`:
    SEL-intra entails consecutive ranks (Eq. 3); SEL-inter entails the image by the deepest
    per-rank text `T_R'` (Eq. 4). The cumulative/`full` text is used ONLY for contrastive
    alignment, not for SEL. Pass those independent embeddings as `sel_text_embs`; if omitted
    we fall back to `text_embs` (legacy cumulative behaviour). Positive/negative MASKING
    still uses the cumulative labels in `taxonomy_batch` (two samples share a parent iff
    their cumulative lineage matches), independent of which text form is embedded.

    `stats` (if given) collects per-edge / per-term pos+neg components for logging.
    """
    sel_embs = sel_text_embs if sel_text_embs is not None else text_embs
    intra = sel_intra(sel_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives,
                      stats=stats, tau=tau, leak=leak, lam_u=lam_u)
    inter = sel_inter(img, sel_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives, stats=stats)
    return intra + inter, intra, inter
