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
    "accum_contrastive_loss_ddp",
    "ranked_contrastive_loss_ddp",
    "ranked_infonce",
    "shared_depth_matrix",
    "stable_lineage_ids",
    "radial_ordering_loss",
    "level_restricted_loss",
    "level_restricted_accum",
    "hybrid_graded_loss",
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


def ranked_contrastive_loss_ddp(
    img, text, lineage_ids, curv, scale, kind="distance",
    max_depth=4, min_tau=0.1, max_tau=0.5,
):
    """RINCE (ranked InfoNCE) cross-modal CL with taxonomic graded positives + cross-GPU bank.

    `img`/`text` [B,D] are this rank's matched image/deepest-text features. `lineage_ids` [B,R] are
    per-rank STABLE int ids (same string -> same id on every GPU; -1 = unknown) coarse->fine, used
    to grade bank items by shared taxonomic depth. Query = local image; bank = ALL gathered texts.
    Computes sim = h(image_i, text_j) and applies `ranked_infonce` (RINCE-in). `kind` selects the
    similarity (distance/angle/euclidean). Falls back to the local square form off-DDP."""
    import torch.distributed as dist
    ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if ddp:
        all_text = _gather_across_processes(text)              # autograd-aware (grad to other ranks)
        all_lin = _gather_int_rows(lineage_ids)                # [B*world, R], non-diff
    else:
        all_text, all_lin = text, lineage_ids
    # image-vs-all-text similarity (query=image). ranked_infonce requires "higher sim = more
    # similar". _cl_logits' image side gives -dist (✓ higher=closer) but +oxy_angle for angle
    # (larger angle = LESS similar) — negate it so the convention holds (else ranked learns the
    # INVERTED ranking: loss→0 by ranking dissimilar texts as positives, but F1=0).
    sim, _ = _cl_logits(kind, img, text, img, all_text, curv, scale)   # [B, B*world]
    if kind == "angle":
        sim = -sim
    depth = shared_depth_matrix(lineage_ids, all_lin)          # [B, B*world]
    return ranked_infonce(sim, depth, max_depth=max_depth, min_tau=min_tau, max_tau=max_tau)


def _gather_int_rows(t: torch.Tensor) -> torch.Tensor:
    """Non-differentiable all-gather of an int tensor [B, R] -> [B*world, R]. Returns input off-DDP."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, dim=0)


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


def level_restricted_loss(img, text_embs, taxonomy_batch, ranks, curv, scale,
                          sim="distance", tau=1.0):
    """Level-Restricted Contrastive Learning (Tao et al. 2026, "Beyond Flat Labels"), Eq. 1-3.

    One InfoNCE PER taxonomic level, contrasting only WITHIN that level (negatives = other
    same-level labels), summed equally over levels. This removes cross-level false negatives
    (a same-genus image is never a wrong-level negative) by construction — the partitioning
    alternative to RINCE's grading. Uses our negative-Lorentzian-distance similarity (their
    supplementary uses the same), so it drops into the hyperbolic setup.

    Per level ℓ: build the set of UNIQUE level-ℓ labels in the batch (level-restriction). The
    image's positive is its own level-ℓ label's text; negatives are the other unique level-ℓ
    texts. Symmetric I→T (Eq. 1) + group-balanced T→I (Eq. 2, each label's positives = the SET
    of images carrying it). `text_embs` = encode_taxonomy(...) cumulative per-rank embeddings.
    """
    levels = [r for r in ranks if r in text_embs]
    total = img.new_zeros(())
    n = 0
    for r in levels:
        v = text_embs[f"{r}_valid"]
        labels = taxonomy_batch[r]
        keep = [i for i in range(len(labels)) if bool(v[i]) and labels[i] is not None]
        if len(keep) < 2:
            continue
        keep_t = torch.tensor(keep, device=img.device)
        lab = [labels[i] for i in keep]
        uniq = list(dict.fromkeys(lab))               # unique level-ℓ labels (the restricted set K^ℓ)
        if len(uniq) < 2:
            continue                                   # need ≥2 classes for a contrast
        pos = {u: j for j, u in enumerate(uniq)}
        target = torch.tensor([pos[labels[i]] for i in keep], device=img.device)  # [Nk] img -> its label idx
        uemb = text_embs[r][keep_t][[lab.index(u) for u in uniq]]  # [U, D] one text per unique label
        im = img[keep_t]                                # [Nk, D]
        if sim == "euclidean":
            s = scale * F.normalize(im, dim=-1) @ F.normalize(uemb, dim=-1).T
        elif sim == "angle":
            s = -L.pairwise_oxy_angle(uemb, im, curv).T * scale  # text/species apex, negated (see hybrid)
        else:
            s = -L.pairwise_dist(im, uemb, curv) * scale   # [Nk, U] negative distance
        s = s / tau
        # I->T: each image picks its label among the U unique level texts
        i2t = F.cross_entropy(s, target)
        # T->I: each unique level text vs all images; positives = SET of images with that label
        #       (group-balanced). log-softmax over images, average the positive rows' mass.
        logp = F.log_softmax(s.T, dim=1)               # [U, Nk]
        t2i_terms = []
        for u in range(len(uniq)):
            posmask = target == u
            if posmask.any():
                t2i_terms.append(-logp[u, posmask].mean())
        t2i = torch.stack(t2i_terms).mean() if t2i_terms else s.new_zeros(())
        total = total + 0.5 * (i2t + t2i)
        n += 1
    return total / max(n, 1)


def hybrid_graded_loss(img, text_embs, taxonomy_batch, ranks, curv, scale,
                       sim="distance", min_tau=0.1, max_tau=0.5):
    """Hybrid: RINCE's exactly-d tier partitioning + LRCL's deduped-prototype symmetric I<->T.

    Per rank ℓ (= depth d): prototypes = UNIQUE cumulative rank-ℓ labels (the deduped depth-d
    prefixes; `text_embs[ℓ]` cumulative). Symmetric I->T + group-balanced T->I (LRCL form), BUT with
    RINCE's exactly-d exclusion in T->I: a prototype's positive images are those matching it at rank ℓ
    AND differing at the next finer rank (depth exactly d) — images that also share a deeper prototype
    are excluded here (counted at their own deeper tier), so coarse prototypes model the residual
    cross-finer-rank relatedness, not the species-mate mass. Per-tier temperature (coarser=hotter,
    RINCE schedule). Pass CUMULATIVE text_embs.
    """
    levels = [r for r in ranks if r in text_embs]
    total = img.new_zeros(())
    n = 0
    for di, r in enumerate(levels):
        d = di + 1
        finer = levels[di + 1] if di + 1 < len(levels) else None
        v = text_embs[f"{r}_valid"]
        labels = taxonomy_batch[r]
        keep = [i for i in range(len(labels)) if bool(v[i]) and labels[i] is not None]
        if len(keep) < 2:
            continue
        keep_t = torch.tensor(keep, device=img.device)
        lab = [labels[i] for i in keep]
        uniq = list(dict.fromkeys(lab))
        if len(uniq) < 2:
            continue
        pos = {u: j for j, u in enumerate(uniq)}
        target = torch.tensor([pos[labels[i]] for i in keep], device=img.device)  # [Nk] -> rank-ℓ proto
        uemb = text_embs[r][keep_t][[lab.index(u) for u in uniq]]  # [U,D] cumulative text per unique label
        im = img[keep_t]
        if sim == "euclidean":
            s = scale * F.normalize(im, dim=-1) @ F.normalize(uemb, dim=-1).T
        elif sim == "angle":
            # entailment convention: angle at the TEXT/species apex (uemb is the parent), oxy_angle ~0
            # when the image is aligned on the species' outward ray. NEGATE so larger score = more
            # similar (matches -pairwise_dist below; softmax/CE want larger=positive). pairwise_oxy_angle
            # (uemb, im) -> [U, Nk]; .T -> [Nk, U]. (Image-apex order, the un-negated form, is the WRONG
            # vertex — it gives pi for an aligned image; that bug broke the early RINCE-angle runs.)
            s = -L.pairwise_oxy_angle(uemb, im, curv).T * scale
        else:
            s = -L.pairwise_dist(im, uemb, curv) * scale
        tau = min_tau + (1.0 - d / len(levels)) * (max_tau - min_tau)  # coarser -> hotter
        s = s / tau
        i2t = F.cross_entropy(s, target)                       # LRCL I->T
        logp = F.log_softmax(s.T, dim=1)                       # [U,Nk]
        U = len(uniq)
        # group membership mask [U, Nk]: image j belongs to prototype u iff target[j]==u
        grp = (target[None, :] == torch.arange(U, device=img.device)[:, None])  # [U,Nk] bool
        if finer is not None:
            # exactly-d exclusion (VECTORISED): drop images in u's group whose NEXT-FINER rank label
            # matches the group representative's (rep = first image in the group). Build dense finer
            # ids once, take each prototype's rep-finer id, mask out same-finer images.
            finer_lab = [taxonomy_batch[finer][i] for i in keep]
            finer_id = _dense_ids(finer_lab, img.device)        # [Nk]
            first_idx = grp.float().argmax(dim=1)               # [U] first image index in each group
            rep_finer = finer_id[first_idx]                     # [U] rep's finer id per prototype
            exclude = grp & (finer_id[None, :] == rep_finer[:, None])  # same-finer-as-rep -> drop
            grp_excl = grp & ~exclude
            grp = torch.where(grp_excl.any(dim=1, keepdim=True), grp_excl, grp)  # fall back if all dropped
        # group-balanced T->I: per prototype, mean log-prob over its (masked) positive images
        cnt = grp.float().sum(dim=1).clamp(min=1)               # [U]
        per_u = -(logp * grp.float()).sum(dim=1) / cnt          # [U]
        t2i = per_u[grp.any(dim=1)].mean() if grp.any() else s.new_zeros(())
        total = total + 0.5 * (i2t + t2i)
        n += 1
    return total / max(n, 1)


def radial_ordering_loss(text_embs, img, ranks, curv, margin=0.2, use_centroid=False):
    """Radial-ordering driver: push each level's MEAN radius to INCREASE down the hierarchy
    (order < family < ... < species < image). Generalises ATMG's 2-level L_centroid (text centroid
    nearer origin than image centroid, paper Eq. 12) to our per-rank hierarchy.

    Why: SEL is satisfied-by-collapse (zero curvature gradient once origin-collapsed), so nothing
    drives the radial spread that the cone hierarchy needs (analytically feasible only at curv≳4;
    see docs/curvature-feasibility.md). This term's gradient EXPLICITLY pushes coarse ranks inward
    and fine ranks/images outward — the radial driver SEL lacks — coupling to higher curvature.

    Uses MEAN per-point radius per level, NOT the Einstein-centroid radius: the centroid of a
    concentric RING collapses toward the origin (a fully-spread ring's centroid is at the origin by
    symmetry in ANY geometry; but for a PARTIALLY-spread ring the HYPERBOLIC Einstein centroid is
    dragged toward the origin MORE than the Euclidean mean — Klein compression down-weights far-out
    points. Measured: ring at radius 4, 120° spread -> Euclid centroid r=3.8 but Einstein r=1.2). So
    rank centroids squash into a narrow near-origin band regardless of true shell radius and can't
    distinguish the fine ranks — exactly where we need the spread. MEAN radius averages the SCALAR
    radii (no vector cancellation, no Klein compression) -> tracks true shell radius exactly (ring at
    radius 4 -> mean radius 4.0 for any spread).

    For each consecutive level pair (coarser, finer): `relu(ρ̄(coarser) − ρ̄(finer) + margin)`,
    ρ̄ = mean distance_from_origin. Zero only when each finer level is ≥ margin further out; cannot
    be satisfied by collapse (all radii equal → all hinges active). Returns a scalar.

    `use_centroid=True`: use the Einstein-CENTROID radius per rank instead of the mean radius — this
    is ATMG's actual Eq.12 form. The A/B control that empirically confirms the centroid collapses the
    fine ranks (see docstring above); expected to be MUCH weaker than the mean-radius default.
    """
    def rank_radius(pts):
        if use_centroid:
            return L.distance_from_origin(L.einstein_midpoint(pts, curv)[None], curv)[0]
        return L.distance_from_origin(pts, curv).mean()
    radii = []
    for r in ranks:
        if r in text_embs and bool(text_embs[f"{r}_valid"].any()):
            v = text_embs[f"{r}_valid"]
            radii.append(rank_radius(text_embs[r][v]))
    radii.append(rank_radius(img))  # image = deepest level
    if len(radii) < 2:
        return img.new_zeros(())
    loss = img.new_zeros(())
    for coarser, finer in zip(radii[:-1], radii[1:]):
        loss = loss + F.relu(coarser - finer + margin)
    return loss / (len(radii) - 1)


def stable_lineage_ids(taxonomy_batch, ranks, device):
    """[B, R] int tensor of per-rank STABLE ids from a taxonomy_batch (coarse->fine rank order).
    Same string -> same id on every GPU (DETERMINISTIC hash, since Python's built-in hash() is
    per-process randomised and would break cross-rank shared-depth under DDP); None/'' -> -1
    (unknown, never matches). Used by ranked (RINCE) CL for graded positives."""
    import hashlib

    def hid(rank, s):  # deterministic 63-bit id of "rank:value" (rank prefix disambiguates ranks)
        h = hashlib.blake2b(f"{rank}:{s}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF

    B = len(taxonomy_batch[ranks[0]])
    out = torch.full((B, len(ranks)), -1, dtype=torch.long, device=device)
    for r, rank in enumerate(ranks):
        for i, s in enumerate(taxonomy_batch[rank]):
            if s is not None and s != "":
                out[i, r] = hid(rank, s)
    return out


def shared_depth_matrix(query_lineages, bank_lineages):
    """[Q, K] int matrix: number of LEADING ranks shared between query i and bank j (taxonomy
    order->...->species). 4 = same species, 0 = shares nothing. `*_lineages` are [N, R] LongTensors
    of per-rank dense ids (rank order coarse->fine); -1 (unknown) never matches (treated distinct)."""
    q = query_lineages[:, None, :]            # [Q,1,R]
    k = bank_lineages[None, :, :]             # [1,K,R]
    eq = (q == k) & (q != -1)                 # [Q,K,R] per-rank equality, unknowns never equal
    # cumulative AND from the coarsest rank: depth = #leading ranks that all match
    cum = torch.cumprod(eq.long(), dim=2)     # stays 1 only while every prior rank matched
    return cum.sum(dim=2)                     # [Q,K] in 0..R


def ranked_infonce(sim, depth, max_depth, min_tau=0.1, max_tau=0.5, one_per_rank=True, eps=1e-7):
    """RINCE-in (Hoffmann 2022, sum_in_log) over taxonomic shared-depth ranks.

    Ported from /home/daniela/other/rince/losses.py (sum_in_log + the per-rank loop), with their
    CIFAR similarity table replaced by `depth` (our taxonomy shared-depth) and their MoCo memory
    bank replaced by the in-batch/cache-accum bank already encoded in `sim`.

    Args:
      sim   : [Q, K] similarity logits h(query_i, bank_j) (already * scale). Higher = more similar.
      depth : [Q, K] shared taxonomic depth (from shared_depth_matrix), 0..max_depth.
      max_depth : R (=4 for BIOSCAN). Ranking levels are depths max_depth..1; depth 0 = true negative.
    Per level d (fine->coarse): positives = {j: depth==d}; negatives (denominator) = positives at
    this level + everything LESS similar (depth < d, incl. true negatives). Items MORE similar
    (depth > d) are excluded from both (set to -inf) so they don't act as negatives. RINCE-in:
    softmax over [pos, neg]; loss = -log(sum of softmax mass on positives). Empty levels skipped.

    tau per level: linear in dissimilarity (their get_dynamic_tau): sim_level = d/max_depth,
    tau = min_tau + (1 - sim_level)*(max_tau - min_tau) -> coarser ranks get higher tau.
    `one_per_rank=True` = one loss term per distinct depth level (their best variant).
    """
    total = sim.new_zeros(())
    n_terms = 0
    for d in range(max_depth, 0, -1):
        # level d: positives = {depth==d}; denominator = positives + everything LESS similar
        # (depth<d, incl. true negatives). Items MORE similar (depth>d) -> -inf (excluded from both).
        logits = sim.masked_fill(depth > d, float("-inf"))
        tau = min_tau + (1.0 - d / max_depth) * (max_tau - min_tau)  # coarser rank -> higher tau
        pos_mass = (F.softmax(logits / tau, dim=1) * (depth == d)).sum(dim=1)  # [Q]
        valid = pos_mass > eps                             # queries with >=1 rank-d positive
        if valid.any():
            total = total - torch.log(pos_mass[valid]).mean()  # valid guards log(0); no +eps
            n_terms += 1
    return total / max(n_terms, 1)


def _cl_logits(kind, img, text, all_img, all_text, curv, scale):
    """The [B, N] image- and text-side logit matrices for each CL variant (the only part that
    differs between the three *_ddp losses). `all_*` is the full negative bank (gathered + cached)."""
    if kind == "euclidean":
        return scale * F.normalize(img, dim=-1) @ F.normalize(all_text, dim=-1).T, \
               scale * F.normalize(text, dim=-1) @ F.normalize(all_img, dim=-1).T
    if kind == "angle":
        return L.pairwise_oxy_angle(img, all_text, curv) * scale, \
               -L.pairwise_oxy_angle(text, all_img, curv) * scale
    return -L.pairwise_dist(img, all_text, curv) * scale, \
           -L.pairwise_dist(text, all_img, curv) * scale


def _gather_local_loss(t: torch.Tensor) -> torch.Tensor:
    """OpenCLIP gather_features(gather_with_grad=False, local_loss=False): non-differentiable
    all_gather, then splice THIS rank's grad-carrying tensor back in. Gradient flows only through
    the local slice (DDP all-reduce handles cross-rank sync) — NOT autograd-aware all_gather,
    which double-counts with DDP and zeroes the gradient. Returns input unchanged off-DDP."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t
    world, rank = dist.get_world_size(), dist.get_rank()
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous())
    gathered[rank] = t  # local rank keeps its grad-carrying tensor
    return torch.cat(gathered, dim=0)


def accum_contrastive_loss_ddp(
    kind, local_img, local_text, curv, scale, local_ids=None,
):
    """Accumulation-aware CL for one grad-pass micro-step (faithful OpenCLIP cache-recompute).

    `local_img`/`local_text` are the FULL local accum set for this rank
    (cat(cached[:j] + [fresh_j] + cached[j+1:]), size accum*mb), where only the active
    micro-batch slice carries gradient (the rest are detached cached features). This mirrors
    OpenCLIP train.py line 154 exactly: the whole set is queried against itself.

    Cross-rank: gather the full local sets NON-differentiably and splice the local (grad-carrying)
    set back in (OpenCLIP gather_features, gather_with_grad=False, local_loss=False). Gradient
    flows only through the local slice; DDP all-reduce syncs across ranks. `kind` ∈ {distance,
    angle, euclidean}. Targets are the diagonal of the local block in the gathered bank.
    """
    import torch.distributed as dist
    ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    N = local_img.shape[0]                       # accum*mb (the full local set)
    rank = dist.get_rank() if ddp else 0
    all_img = _gather_local_loss(local_img)      # [N*world, D]; local slice keeps grad
    all_text = _gather_local_loss(local_text)
    img_logits, text_logits = _cl_logits(kind, local_img, local_text, all_img, all_text, curv, scale)
    if local_ids is not None:
        all_ids = _gather_labels(local_ids) if ddp else local_ids
        mask = _false_negative_mask(local_ids, all_ids, rank)   # [N, N*world]
        img_logits = img_logits.masked_fill(mask, float("-inf"))
        text_logits = text_logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(N, device=local_img.device) + N * rank  # local block's diagonal
    return 0.5 * (F.cross_entropy(img_logits, targets) + F.cross_entropy(text_logits, targets))


def level_restricted_accum(local_img, local_text_per_rank, local_labels_per_rank, ranks,
                           curv, scale, sim="distance", tau=1.0):
    """Accumulation-aware LRCL for one grad-pass micro-step (same GradCache pattern as
    `accum_contrastive_loss_ddp`). LRCL is SYMMETRIC: its T->I direction scores each label-text
    against ALL IMAGES, so — like plain CL — the image negatives must span the full effective
    batch, requiring the cache-recompute.

    `local_img` [N,D] is the FULL local set (cat(cached[:j]+[fresh_j]+cached[j+1:]), N=accum*mb),
    only the active micro's slice carries grad. `local_text_per_rank` = {rank: [N,D]} the per-rank
    cumulative text embeddings for the SAME full local set (also fresh-spliced). `local_labels_per_rank`
    = {rank: [N] list of label strings} for dedup. Cross-rank: gather images + per-rank texts + labels
    non-differentiably, splice the local grad-slice back in (OpenCLIP gather_with_grad=False)."""
    import torch.distributed as dist
    ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    all_img = _gather_local_loss(local_img)              # [N*world, D]; local slice keeps grad
    total = local_img.new_zeros(())
    n = 0
    for r in ranks:
        if r not in local_text_per_rank:
            continue
        all_text = _gather_local_loss(local_text_per_rank[r])      # [N*world, D]
        labels = (_gather_str(local_labels_per_rank[r]) if ddp else local_labels_per_rank[r])
        keep = [i for i in range(len(labels)) if labels[i] is not None]
        if len(keep) < 2:
            continue
        lab = [labels[i] for i in keep]
        uniq = list(dict.fromkeys(lab))
        if len(uniq) < 2:
            continue
        pos = {u: j for j, u in enumerate(uniq)}
        target = torch.tensor([pos[labels[i]] for i in keep], device=all_img.device)  # [Nk]
        keep_t = torch.tensor(keep, device=all_img.device)
        # one text per unique label; .index picks the FIRST occurrence (a grad-carrying copy if the
        # label is in the active micro, since fresh is spliced before its cached duplicates only when
        # active — acceptable: same string -> same embedding, grad flows via the active copies).
        uemb = all_text[keep_t][[lab.index(u) for u in uniq]]      # [U, D]
        im = all_img[keep_t]                                        # [Nk, D]
        if sim == "euclidean":
            s = scale * F.normalize(im, dim=-1) @ F.normalize(uemb, dim=-1).T
        elif sim == "angle":
            s = -L.pairwise_oxy_angle(uemb, im, curv).T * scale  # text/species apex, negated (see hybrid)
        else:
            s = -L.pairwise_dist(im, uemb, curv) * scale            # [Nk, U]
        s = s / tau
        i2t = F.cross_entropy(s, target)
        logp = F.log_softmax(s.T, dim=1)                            # [U, Nk]; T->I over all images
        t2i_terms = [-logp[u, target == u].mean() for u in range(len(uniq)) if (target == u).any()]
        t2i = torch.stack(t2i_terms).mean() if t2i_terms else s.new_zeros(())
        total = total + 0.5 * (i2t + t2i)
        n += 1
    return total / max(n, 1)


def _gather_str(lst):
    """All-gather a list of (hashable) label strings across ranks -> concatenated list. Non-diff."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return lst
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, lst)
    return [x for sub in out for x in sub]


def entailment_pos(
    parent: torch.Tensor, child: torch.Tensor, curv: torch.Tensor | float,
    r_min: float = 0.1, tau: float = 1.0, leak: float = 0.0, lam_u: float = 0.0,
    cone_margin: float = 0.0,
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

    `cone_margin` (>0): the CONE-CONTAINMENT term (our derivation, not UNCHA). The bare hinge only
    needs the child's APEX inside the parent — so child cones hang off the parent's edge and a
    grandchild (or the image) inside the child pokes outside the parent: entailment is NOT
    transitive (measured: image in species 0.89 -> in order 0.31). Requiring the child's WHOLE
    cone inside the parent's — angle + psi(child) <= psi(parent) — restores transitivity. This
    is meaningful only when child/parent are both CONES (text-text, sel_intra); skip for
    sel_inter where the child is an image point. It also self-induces radial spread: the
    condition is only satisfiable when psi(child) < psi(parent), i.e. child at larger radius
    (psi ∝ 1/||x||), so it pushes the chain outward — a principled alternative to lam_u.

    Defaults (tau=1, leak=0, lam_u=0, cone_margin=0) reproduce the plain hinge.
    """
    angle = L.oxy_angle(parent, child, curv)
    aperture = L.half_aperture(parent, curv, min_radius=r_min)
    eff = angle - tau * aperture
    if cone_margin > 0.0:
        eff = eff + cone_margin * L.half_aperture(child, curv, min_radius=r_min)
    loss = F.relu(eff)
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
    cone_margin: float = 0.0,
) -> torch.Tensor:
    """Negative entailment hinge: a non-child should lie outside parent's cone.

    Base: L = relu(half_aperture(parent) - oxy_angle(parent, child) + margin) — pushes the child's
    APEX outside the wrong parent's cone (apex-only, mirrors the plain positive hinge).

    `cone_margin` (>0): the SEPARATION counterpart to entailment_pos's cone_margin. Adds
    cone_margin*psi(child), requiring the child's WHOLE CONE outside the parent's
    (oxy_angle >= psi(parent) + cone_margin*psi(child) + margin), not just its apex. This is the
    symmetric negative to whole-cone containment: as the positive pulls correct children's whole
    cones IN, this pushes wrong-parent children's whole cones OUT — the separation force that the
    plain (apex-only) negative lacks (measured: ~85% of negatives already apex-satisfied, so the
    apex hinge gives no gradient; the whole-cone version keeps pushing non-children further apart).
    """
    angle = L.oxy_angle(parent, child, curv)
    aperture = L.half_aperture(parent, curv, min_radius=r_min)
    eff = aperture - angle + margin
    if cone_margin > 0.0:
        eff = eff + cone_margin * L.half_aperture(child, curv, min_radius=r_min)
    return F.relu(eff)


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
    cone_margin: float = 0.0,
    neg_cone_margin: float = 0.0,
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

    pos_all = entailment_pos(p_grid, c_grid, curv, r_min, tau, leak, lam_u,
                             cone_margin).reshape(B, B)
    if pos_mask.any():
        pos_loss = pos_all[pos_mask].mean()
    else:
        pos_loss = parent.new_zeros(())
    loss = pos_loss

    neg_loss = None
    if use_negatives and neg_mask.any():
        neg_all = entailment_neg(p_grid, c_grid, curv, r_min, margin,
                                 cone_margin=neg_cone_margin).reshape(B, B)
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
    cone_margin: float = 0.0,
    neg_cone_margin: float = 0.0,
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
            cone_margin=cone_margin,
            neg_cone_margin=neg_cone_margin,
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
    cone_margin: float = 0.0,
    neg_cone_margin: float = 0.0,
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
    # Both margins apply to SEL-INTRA (text-text rank cones) ONLY — sel_inter's child is an image
    # POINT (its cone psi(image) is geometrically defined but SEMANTICALLY vacuous: an instance has no
    # descendants to contain), so inter keeps the plain hinges. cone_margin (POSITIVE/containment):
    # child rank's whole cone INSIDE its parent. neg_cone_margin (NEGATIVE/separation): child rank's
    # whole cone OUTSIDE wrong parents — separates the COARSE rank cones (genus-vs-genus within a
    # family, etc.), which propagates to species separation by containment (species ⊂ separated genus
    # cones). This is the hierarchy-aware complement to CL's direct flat species separation.
    intra = sel_intra(sel_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives,
                      stats=stats, tau=tau, leak=leak, lam_u=lam_u, cone_margin=cone_margin,
                      neg_cone_margin=neg_cone_margin)
    inter = sel_inter(img, sel_embs, taxonomy_batch, ranks, curv, r_min, margin, use_negatives,
                      stats=stats)
    return intra + inter, intra, inter
