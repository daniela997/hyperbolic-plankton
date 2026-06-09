"""Train HyperbolicCLIP with LoRA on the seen pool (HAC recipe), DDP over 2 GPUs.

Launch:
  PYTHONPATH=src torchrun --nproc_per_node=2 scripts/train_lora.py \
      --backbone bioclip --iters 30000 --micro-bs 128 --accum 3

Effective batch = micro_bs * world_size * accum (default 128*2*3 = 768, HAC).
Recipe (HAC configs/train_hac_vit_b_lora.py): AdamW lr 2.5e-4 betas (0.9,0.98) wd 0.2
(disabled for LN/bias + MERU scalars + LoRA), LinearWarmupCosineDecay 4k warmup, AMP,
LoRA r=128 alpha=128 rslora on last 4 visual / last 8 text blocks (q,k,v,o).
"""

# A5000-specific: NCCL peer-to-peer is broken on these cards and hangs DDP. Must be set
# BEFORE torch/NCCL init (NCCL reads the env var at process group creation).
import os

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import argparse
import time

import json

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hyperbolic_plankton.bioscan import BIOSCAN_RANKS, BioscanHDF5Dataset
from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.eval import (
    flatten_metrics,
    geometry_stats,
    run_unseen_eval,
)
from hyperbolic_plankton.loss import (
    _deepest_text,
    _dense_ids,
    hyperbolic_angle_contrastive_loss_ddp,
    hyperbolic_contrastive_loss_ddp,
    stacked_entailment_loss,
)
from hyperbolic_plankton.lora import apply_lora, count_trainable
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.optim import LinearWarmupCosineDecayLR, param_groups
from hyperbolic_plankton.train import TaxonomyCollator

CACHE = "/scratch/daniela/planktonzilla_cache/plankton"
CKPT_DIR = "/scratch/daniela/hyperbolic_plankton_ckpts"
SPLIT_DIR = "/scratch/daniela/hyperbolic_plankton_splits"
WANDB_DIR = "/scratch/daniela/wandb"  # keep wandb's local files off the repo/home


def _stratified_subsample(cache, idx, cap, seed):
    """Class-balanced subset of `idx`: up to `cap` rows per `proposed_label`, sampled `seed`.

    Stratify by `proposed_label` (the WoRMS-harmonised CLASS IDENTITY the eval scores), not
    `full` — `full` conflates annotation depth with identity (two depths of the same species
    get different `full` strings). ~equal samples per class gives the periodic MACRO-F1
    (which weights classes equally) a low-variance estimate of the full macro-F1, unlike a
    uniform subsample where rare classes are under-counted exactly where macro weights most.
    Reads only the `proposed_label` column (no image decode)."""
    labels = np.array(cache.select(idx.tolist())["proposed_label"], dtype=object)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        rows = idx[labels == cls]
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        keep.append(rows)
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def _build_eval_sets(cache, args):
    """Fixed, seeded subsamples of seen-val and unseen for periodic eval (rank 0 only).

    Returns {seen: (ds, classes), unseen: (ds, classes)}. BOTH predict among their FULL
    frozen class set (seen_classes.json / unseen_classes.json) — the subsample only reduces
    the number of *images* scored (a time-saver), NOT the candidate class space.

    Subsampling is STRATIFIED (`--eval-cap` rows per `proposed_label`), not uniform, so the
    periodic macro-F1 tracks the full-split macro-F1 with low variance. Final paper numbers
    come from scripts/final_eval.py over the full splits.
    """
    with open(f"{SPLIT_DIR}/seen_classes.json") as f:
        seen_classes = json.load(f)
    val_idx = np.load(f"{SPLIT_DIR}/val_idx.npy")
    sel = _stratified_subsample(cache, val_idx, args.eval_cap, seed=0)
    seen_val = cache.select(sorted(sel.tolist()))
    seen_ds = HFTaxonomyDataset(seen_val)

    unseen_idx = np.load(f"{SPLIT_DIR}/unseen_idx.npy")
    sel = _stratified_subsample(cache, unseen_idx, args.eval_cap, seed=0)
    unseen_sub = cache.select(sorted(sel.tolist()))
    unseen_ds = HFTaxonomyDataset(unseen_sub)
    with open(f"{SPLIT_DIR}/unseen_classes.json") as f:
        unseen_classes = json.load(f)

    # a fixed taxonomy batch (from seen-val) for per-rank geometry diagnostics — built once
    # so the radius/aperture/entailment curves track the SAME samples over training.
    geom_items = [seen_ds[i] for i in range(min(512, len(seen_ds)))]

    return {
        "sets": {"seen": (seen_ds, seen_classes), "unseen": (unseen_ds, unseen_classes)},
        "geom_items": geom_items,
    }


BIOSCAN_HDF5 = "/scratch/daniela/bioscan1m/data/BIOSCAN_1M/split_data/BioScan_data_in_splits.hdf5"


def _bioscan_classes(group: str) -> list[str]:
    """Distinct `full` lineage strings in a BIOSCAN HDF5 group (the candidate class set)."""
    ds = BioscanHDF5Dataset(BIOSCAN_HDF5, group)
    return sorted({ds[i]["taxonomy"]["full"] for i in range(len(ds))} - {"unknown"})


def _build_eval_sets_bioscan(args):
    """BIOSCAN periodic eval from the CLIBD test_seen / test_unseen groups.

    Candidate classes = classes PRESENT in each full test group (Planktonzilla CLIP protocol:
    `unique` over the eval texts, taken over the FULL group so it is unbiased). The periodic
    monitor scores an image subsample against this full class set.
    """
    seen_ds = BioscanHDF5Dataset(BIOSCAN_HDF5, "test_seen")
    unseen_ds = BioscanHDF5Dataset(BIOSCAN_HDF5, "test_unseen")
    seen_classes = _bioscan_classes("test_seen")
    unseen_classes = _bioscan_classes("test_unseen")
    geom_items = [seen_ds[i] for i in range(min(512, len(seen_ds)))]
    return {
        "sets": {"seen": (seen_ds, seen_classes), "unseen": (unseen_ds, unseen_classes)},
        "geom_items": geom_items,
    }


def _run_periodic_eval(model, eval_sets, num_workers, sel_indep=True, ranks=RANKS):
    """Eval seen-val + unseen subsamples + per-rank geometry; flat wandb-loggable dict.

    `sel_indep` must match training so the logged loss_terms reflect the SAME SEL the
    optimiser sees (independent per-rank text iff training uses it; cumulative otherwise).
    """
    from hyperbolic_plankton.train import TaxonomyCollator

    was_training = model.training
    model.eval()
    out = {}
    for name, (ds, classes) in eval_sets["sets"].items():
        res = run_unseen_eval(model, ds, classes, num_workers=num_workers, ranks=ranks)
        out.update(flatten_metrics(res["metrics"], prefix=f"eval/{name}"))
        out[f"eval/{name}/n_classes"] = res["n_classes"]

    # geometry diagnostics + SEL term decomposition on the fixed batch
    pixel_values, taxonomy_batch, _ = TaxonomyCollator(model.preprocess, ranks=ranks)(
        eval_sets["geom_items"]
    )
    out.update(geometry_stats(model, taxonomy_batch, ranks=ranks))

    # per-term SEL (intra per edge + inter, pos/neg components) for understanding which
    # part of the loss is active — logged under loss_terms/*. Use the SAME text form as
    # training (independent per-rank by default; cumulative under the ablation).
    with torch.no_grad():
        img = model.encode_image(pixel_values.to(model.device))
        cum_embs = model.encode_taxonomy(taxonomy_batch)
        sel_embs = model.encode_taxonomy(taxonomy_batch, indep=True) if sel_indep else cum_embs
        sel_stats: dict = {}
        _, intra, inter = stacked_entailment_loss(
            img, cum_embs, taxonomy_batch, ranks, model.curvature,
            stats=sel_stats, sel_text_embs=sel_embs,
        )
        out["loss_terms/sel_intra"] = float(intra)
        out["loss_terms/sel_inter"] = float(inter)
        for k, v in sel_stats.items():
            out[f"loss_terms/{k}"] = v

    if was_training:
        model.train()
    return out


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def forward_loss(model, pixel_values, taxonomy_batch, lambda_sel, stats=None,
                 sel_indep=True, contrastive="distance", ranks=RANKS,
                 sel_tau=1.0, sel_leak=0.0, sel_uncertainty=0.0, cl_mask="none"):
    """contrastive(img, deepest_text) + lambda*SEL. `model` may be a DDP wrapper; geometry
    helpers live on the underlying module.

    If `stats` (a dict) is given, the SEL per-edge pos/neg decomposition is collected into
    it (cheap float reads) for per-step logging alongside cl/sel.
    """
    core = model.module if isinstance(model, DDP) else model
    img = core.encode_image(pixel_values)
    curv = core.curvature
    scale = core.logit_scale.exp()
    # Contrastive: align image to its deepest CUMULATIVE (`full`) text — the paper's
    # full-text-for-CL.
    cum_embs = core.encode_taxonomy(taxonomy_batch)
    deepest, _ = _deepest_text(cum_embs, ranks)
    # cl_mask: suppress same-class off-diagonal negatives (true positives mis-treated as
    # negatives — ~4.4% of pairs in clade-imbalanced plankton batches). class id = dense id
    # of the deepest cumulative `full` lineage string.
    class_ids = None
    if cl_mask == "same":
        class_ids = _dense_ids(taxonomy_batch["full"], img.device)
    cl_fn = (hyperbolic_angle_contrastive_loss_ddp if contrastive == "angle"
             else hyperbolic_contrastive_loss_ddp)
    cl = cl_fn(img, deepest, curv, scale, class_ids=class_ids)
    # SEL — both intra (Eq.3) and inter (Eq.4) use the SAME text form. Paper-faithful is
    # INDEPENDENT per-rank `T_r` ('Rank: Value'): distinct per-rank concepts give SEL the
    # radial-separation gradient it needs (cumulative ranks are near-collinear). `cumulative`
    # is the ablation. CL above always uses the cumulative `full` string regardless.
    sel_embs = core.encode_taxonomy(taxonomy_batch, indep=True) if sel_indep else cum_embs
    sel, intra, inter = stacked_entailment_loss(
        img, cum_embs, taxonomy_batch, ranks, curv, stats=stats, sel_text_embs=sel_embs,
        tau=sel_tau, leak=sel_leak, lam_u=sel_uncertainty,
    )
    if stats is not None:
        stats["loss_terms/sel_intra"] = intra.detach().item()
        stats["loss_terms/sel_inter"] = inter.detach().item()
    return cl + lambda_sel * sel, cl.detach(), sel.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="planktonzilla", choices=["planktonzilla", "bioscan"],
                    help="planktonzilla=ragged 7-rank (default); bioscan=complete 4-rank "
                         "(order..species) CLIBD split, the clean recipe testbed")
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip"])
    ap.add_argument("--epochs", type=float, default=50,
                    help="full passes over the train set (Taxonomy paper: 50). Drives "
                         "total_steps = epochs * (len(loader)//accum). Primary length control.")
    ap.add_argument("--iters", type=int, default=None,
                    help="override total optimizer steps (debug); if set, ignores --epochs")
    ap.add_argument("--warmup-frac", type=float, default=0.1,
                    help="warmup as a fraction of total steps (paper warms up proportionally)")
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--wd", type=float, default=0.2)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"],
                    help="adamw=HAC; adam=Taxonomy-paper recipe")
    ap.add_argument("--scheduler", default="warmupcos", choices=["warmupcos", "onecycle"],
                    help="warmupcos=HAC linear-warmup+cos^2; onecycle=Taxonomy-paper OneCycleLR")
    ap.add_argument("--onecycle-pct-start", type=float, default=0.3)
    ap.add_argument("--onecycle-min-lr", type=float, default=1e-6)
    ap.add_argument("--curv-lr-scale", type=float, default=1.0,
                    help="LR multiplier for geometry scalars (curv, alphas); <1 slows them "
                         "so the hierarchy is learned via embeddings, not curvature collapse")
    ap.add_argument("--lambda-sel", type=float, default=1.0)
    ap.add_argument("--contrastive", default="distance", choices=["distance", "angle"],
                    help="distance=MERU InfoNCE on -pairwise_dist; angle=ATMG exterior-angle "
                         "InfoNCE (radius-free, same oxy_angle quantity as SEL)")
    ap.add_argument("--cl-mask", default="none", choices=["none", "same"],
                    help="mask same-class off-diagonal CL negatives (true positives "
                         "mis-treated as negatives, ~4.4%% of plankton batch pairs). "
                         "none=standard InfoNCE")
    # SEL-intra anti-collapse terms (UNCHA-inspired). Defaults reproduce the plain hinge.
    ap.add_argument("--sel-leak", type=float, default=0.0,
                    help="Leaky-entailment factor: always-on `leak*oxy_angle` keeps pulling "
                         "children onto the parent axis after containment (UNCHA Eq.14). "
                         "Aligned axis + distinctness => radial separation. 0=off")
    ap.add_argument("--sel-tau", type=float, default=1.0,
                    help="Aperture threshold (<1 tightens the cone: tau*aperture), countering "
                         "the pi/2 saturation so the hinge stays active. 1.0=off")
    ap.add_argument("--sel-uncertainty", type=float, default=0.0,
                    help="Radius/uncertainty penalty weight: `lam_u*softplus(-||parent||)` "
                         "pushes parents off the origin so children end up deeper, and gives "
                         "ragged leaves depth-appropriate radius (UNCHA Eq.7/15). 0=off")
    ap.add_argument("--sel-text", default="independent", choices=["independent", "cumulative"],
                    help="text form for BOTH SEL terms (intra Eq.3 + inter Eq.4). Paper uses "
                         "independent per-rank embeddings T_r ('Rank: Value') for SEL and the "
                         "cumulative/full string ONLY for CL — so 'independent' is paper-"
                         "faithful (default). 'cumulative' is the ablation. CL always uses full.")
    ap.add_argument("--freeze-curv", action="store_true",
                    help="hold curvature fixed at init (removes the curvature-collapse "
                         "shortcut so SEL must update embeddings, not shrink cones)")
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--no-lora", action="store_true",
                    help="projector-only (skip LoRA) = the scratchpad regime; for the "
                         "disentangling probe of whether LoRA (not LR) drives the collapse")
    ap.add_argument("--no-reinit-final-ln", action="store_true",
                    help="keep CLIP's pretrained final-LN params (only unfreeze); default "
                         "re-inits to fresh LN as HAC does")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-cap", type=int, default=50,
                    help="periodic eval: max rows per proposed_label class (stratified "
                         "subsample, so macro-F1 tracks the full-split value with low variance)")
    ap.add_argument("--tag", default="bioclip_lora")
    ap.add_argument("--wandb-project", default="hyperbolic-plankton")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world = 1
    eff_batch = args.micro_bs * world * args.accum
    log(f"backbone={args.backbone} world={world} micro_bs={args.micro_bs} "
        f"accum={args.accum} -> effective batch={eff_batch}")

    # model (+ LoRA unless --no-lora, which gives the scratchpad's projector-only regime:
    # frozen backbone under no_grad, only projector + MERU scalars trainable).
    model = HyperbolicCLIP(backbone=args.backbone, learn_curv=not args.freeze_curv)
    if not args.no_lora:
        model = apply_lora(
            model, r=args.lora_r, alpha=args.lora_r,
            reinit_final_ln=not args.no_reinit_final_ln,
        )
    model.to(device)
    if is_main():
        c = count_trainable(model)
        log(f"trainable params: {c['trainable']:,} / {c['total']:,} "
            f"({100 * c['trainable'] / c['total']:.2f}%)")
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True) if ddp else model

    # data: load the prebuilt Planktonzilla-faithful splits (scripts/build_splits.py).
    # Train on seen-TRAIN only; seen-val + unseen are held out for periodic eval.
    # ranks: planktonzilla = ragged 7-rank; bioscan = complete 4-rank (order..species).
    log(f"loading {args.dataset} train split...")
    if args.dataset == "bioscan":
        ranks = BIOSCAN_RANKS
        train_ds = BioscanHDF5Dataset(BIOSCAN_HDF5, "train_seen")
        eval_sets = _build_eval_sets_bioscan(args) if is_main() else None
    else:
        ranks = RANKS
        cache = load_from_disk(CACHE)
        train_idx = np.load(f"{SPLIT_DIR}/train_idx.npy")
        train_ds = HFTaxonomyDataset(cache.select(train_idx.tolist()))
        eval_sets = _build_eval_sets(cache, args) if is_main() else None
    log(f"seen-train rows: {len(train_ds):,}")

    collate = TaxonomyCollator(model.preprocess, ranks=ranks)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=0) if ddp else None
    loader = DataLoader(
        train_ds, batch_size=args.micro_bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )

    # Length in OPTIMIZER STEPS: --epochs (default, paper-faithful) -> epochs * steps/epoch,
    # where one step = `accum` micro-batches. --iters overrides for debug. Computed here so
    # it scales correctly with dataset size and batch (a fixed --iters silently means very
    # different #epochs on BIOSCAN 36k vs planktonzilla 1.76M).
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_iters = args.iters if args.iters is not None else int(args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_frac * total_iters))
    log(f"steps/epoch={steps_per_epoch}  epochs={args.epochs}  total_steps={total_iters}  "
        f"warmup={warmup_steps}")

    # Optimizer: AdamW (HAC) or Adam (Taxonomy-paper recipe — wd added to grad, not decoupled).
    # --curv-lr-scale (<1) puts the geometry scalars (curv, alphas) in a slower group so the
    # model learns the hierarchy via embeddings instead of cheaply shrinking curv to widen cones.
    geom_scale = args.curv_lr_scale if args.curv_lr_scale < 1.0 else None
    pg = param_groups(model, args.wd, base_lr=args.lr, geom_lr_scale=geom_scale)
    if args.optimizer == "adam":
        opt = torch.optim.Adam(pg, lr=args.lr, betas=(0.9, 0.98))
    else:
        opt = torch.optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.98))
    if geom_scale is not None:
        log(f"geom scalars (curv/alpha) at lr x{geom_scale} = {args.lr * geom_scale:.2e}")

    # Scheduler: HAC linear-warmup+cos², or the Taxonomy-paper OneCycleLR. Per-group max_lr
    # preserves the geom group's slower LR through the one-cycle schedule.
    if args.scheduler == "onecycle":
        max_lrs = [g.get("lr_scale", 1.0) * args.lr for g in pg]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lrs, total_steps=total_iters,
            pct_start=args.onecycle_pct_start, anneal_strategy="cos",
            div_factor=args.lr / args.onecycle_min_lr, final_div_factor=1.0,
        )
    else:
        sched = LinearWarmupCosineDecayLR(opt, total_steps=total_iters, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda")

    # wandb (rank 0 only): config = all hyperparams incl lambda_sel + effective batch.
    wb = None
    if is_main() and not args.no_wandb:
        import wandb

        os.makedirs(WANDB_DIR, exist_ok=True)
        wb = wandb.init(
            project=args.wandb_project, name=args.tag, dir=WANDB_DIR,
            config={**vars(args), "effective_batch": eff_batch, "world_size": world,
                    "trainable_params": count_trainable(model)["trainable"]},
        )

    os.makedirs(CKPT_DIR, exist_ok=True)
    ddp_model.train()
    it = 0
    t0 = time.perf_counter()
    run_cl = run_sel = run_loss = 0.0
    data_iter = iter(loader)
    epoch = 0
    sel_terms: dict = {}  # last-micro-step SEL decomposition, refreshed before each log
    while it < total_iters:
        opt.zero_grad(set_to_none=True)
        # collect the SEL term breakdown on the last micro-step of an iter that will log.
        is_log_iter = (it + 1) % args.log_every == 0
        # gradient accumulation: `accum` micro-batches per optimizer step
        for micro in range(args.accum):
            try:
                pixel_values, taxonomy_batch, _ = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)
                pixel_values, taxonomy_batch, _ = next(data_iter)
            pixel_values = pixel_values.to(device, non_blocking=True)
            # only sync grads on the last micro-step (DDP no_sync on the others)
            sync_ctx = (
                ddp_model.no_sync()
                if ddp and micro < args.accum - 1
                else _nullctx()
            )
            last_micro = micro == args.accum - 1
            step_stats = sel_terms if (is_log_iter and last_micro and is_main()) else None
            with sync_ctx, torch.amp.autocast("cuda"):
                loss, cl, sel = forward_loss(
                    ddp_model, pixel_values, taxonomy_batch, args.lambda_sel,
                    stats=step_stats, sel_indep=(args.sel_text == "independent"),
                    contrastive=args.contrastive, ranks=ranks,
                    sel_tau=args.sel_tau, sel_leak=args.sel_leak,
                    sel_uncertainty=args.sel_uncertainty, cl_mask=args.cl_mask,
                )
                loss = loss / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.accum
            run_cl += cl.item()
            run_sel += sel.item()

        scaler.step(opt)
        scaler.update()
        sched.step()
        model.clamp_params()
        it += 1

        if it % args.log_every == 0:
            # run_loss/run_cl/run_sel each accumulate `accum` micro-step values per iter,
            # so all three average over n = log_every * accum contributions.
            n = args.log_every * args.accum
            ips = args.log_every * eff_batch / (time.perf_counter() - t0)
            avg_loss, avg_cl, avg_sel = run_loss / n, run_cl / n, run_sel / n
            lr = sched.get_last_lr()[0]
            log(f"it {it:>6}/{total_iters} | loss {avg_loss:.4f} cl {avg_cl:.4f} "
                f"sel {avg_sel:.4f} | lr {lr:.2e} | curv {model.curvature.item():.3f} | "
                f"{ips:.0f} img/s")
            if wb is not None:
                payload = {
                    "train/loss": avg_loss, "train/cl": avg_cl, "train/sel": avg_sel,
                    "train/lr": lr, "train/curv": model.curvature.item(),
                    "train/logit_scale": model.logit_scale.exp().item(),
                    "train/lambda_sel": args.lambda_sel, "train/img_per_s": ips,
                    "train/epoch": epoch,
                }
                # SEL term breakdown from this iter's last micro-step (train/loss_terms/*)
                payload.update({f"train/{k}": v for k, v in sel_terms.items()})
                wb.log(payload, step=it)
            sel_terms.clear()
            run_cl = run_sel = run_loss = 0.0
            t0 = time.perf_counter()

        # periodic eval (rank 0). Other ranks wait at a barrier so DDP stays in lockstep.
        if it % args.eval_every == 0:
            if is_main():
                metrics = _run_periodic_eval(
                    model, eval_sets, args.num_workers, sel_indep=(args.sel_text == "independent"),
                    ranks=ranks,
                )
                log(f"  [eval it {it}] "
                    f"unseen species_f1={metrics.get('eval/unseen/species_f1', 0):.4f} "
                    f"seen species_f1={metrics.get('eval/seen/species_f1', 0):.4f}")
                if wb is not None:
                    wb.log(metrics, step=it)
            if ddp:
                dist.barrier()
            t0 = time.perf_counter()  # don't count eval time in img/s

        if is_main() and it % args.ckpt_every == 0:
            path = os.path.join(CKPT_DIR, f"{args.tag}_it{it}.pt")
            torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, path)
            log(f"  saved {path}")

    if is_main():
        path = os.path.join(CKPT_DIR, f"{args.tag}_final.pt")
        torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, path)
        log(f"DONE. saved {path}")
        if wb is not None:
            wb.finish()
    if ddp:
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
