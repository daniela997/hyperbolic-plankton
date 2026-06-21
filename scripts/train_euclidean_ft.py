"""Euclidean full fine-tune baseline — the Planktonzilla CLIP recipe, on our splits.

The full-FT corner of the baseline matrix: CLIP ViT-B/16, **backbone unfrozen**, flat-space
cosine InfoNCE (open_clip `ClipLoss`), taxonomic lineage as text. No LoRA, no hyperbolic
lift, no SEL. Trains the SAME data splits + eval harness as `train_lora.py` so the
full-FT-vs-LoRA-vs-hyperbolic comparison differs only in adaptation/geometry.

Kept SEPARATE from `train_lora.py` (which is the LoRA/hyperbolic path) on purpose: full-FT
needs the backbone trainable (`--no-lora` in train_lora leaves it FROZEN = projector-only),
and has none of the SEL/curvature/geom-scalar machinery. Reuses train_lora's data/eval
helpers by import to avoid duplicating them.

  PYTHONPATH=src torchrun --nproc_per_node=2 scripts/train_euclidean_ft.py \
    --dataset bioscan --lr 1e-4 --wd 0.2 --epochs 50 --tag bioscan_FT_euclidean
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from hyperbolic_plankton.bioscan import BIOSCAN_RANKS, BioscanHDF5Dataset
from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.lora import count_trainable, unfreeze_backbone
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.optim import param_groups
from hyperbolic_plankton.train import TaxonomyCollator

# Reuse train_lora's data/eval helpers + constants verbatim (single source of truth).
from train_lora import (  # noqa: E402
    BIOSCAN_HDF5,
    CACHE,
    CKPT_DIR,
    SPLIT_DIR,
    WANDB_DIR,
    _build_eval_sets,
    _build_eval_sets_bioscan,
    _nullctx,
    _run_periodic_eval,
    forward_loss,
    is_main,
    log,
)
from final_eval import run_final_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bioscan", choices=["planktonzilla", "bioscan"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip"])
    ap.add_argument("--epochs", type=float, default=50)
    ap.add_argument("--iters", type=int, default=None, help="override total steps (debug)")
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4, help="Planktonzilla CLIP full-FT lr")
    ap.add_argument("--wd", type=float, default=0.2, help="Planktonzilla CLIP weight decay")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    ap.add_argument("--scheduler", default="onecycle", choices=["warmupcos", "onecycle"])
    ap.add_argument("--onecycle-pct-start", type=float, default=0.3)
    ap.add_argument("--onecycle-min-lr", type=float, default=1e-6)
    ap.add_argument("--cl-mask", default="none", choices=["none", "same"])
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-epochs", type=float, default=None,
                    help="periodic eval cadence in EPOCHS (overrides --eval-every); derived "
                         "from steps_per_epoch so it's consistent across datasets")
    ap.add_argument("--eval-every", type=int, default=200,
                    help="periodic eval cadence in optimizer STEPS (used unless --eval-epochs)")
    ap.add_argument("--eval-cap", type=int, default=50)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the backbone forward (model.clip) for ~1.2-1.8x")
    ap.add_argument("--tag", default="euclidean_ft")
    ap.add_argument("--wandb-project", default="hyperbolic-plankton")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="global seed (torch/numpy/random), same on every DDP rank")
    args = ap.parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere (A5000) — free fp32 speedup
    # Fixed for the full-FT euclidean baseline: flat cosine InfoNCE, no SEL. These let us
    # reuse train_lora.forward_loss / _run_periodic_eval unchanged.
    args.geometry = "euclidean"
    args.lambda_sel = 0.0
    args.lambda_cl = 1.0
    args.contrastive = "distance"  # ignored under geometry=euclidean
    args.sel_text = "independent"  # ignored (no SEL); kept for _run_periodic_eval signature
    args.sel_tau, args.sel_leak, args.sel_uncertainty = 1.0, 0.0, 0.0

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
    log(f"FULL-FT euclidean | backbone={args.backbone} world={world} "
        f"micro_bs={args.micro_bs} accum={args.accum} -> effective batch={eff_batch}")

    # model: full fine-tune = backbone UNFROZEN, no LoRA. learn_curv off (no hyperbolic lift).
    model = HyperbolicCLIP(backbone=args.backbone, learn_curv=False)
    unfreeze_backbone(model)
    model.to(device)
    if is_main():
        c = count_trainable(model)
        log(f"trainable params: {c['trainable']:,} / {c['total']:,} "
            f"({100 * c['trainable'] / c['total']:.2f}%)  [full fine-tune]")
    if args.compile:
        # compile the methods actually called by encode_* (compiling model.clip is a no-op)
        model.clip.encode_image = torch.compile(model.clip.encode_image)
        model.clip.encode_text = torch.compile(model.clip.encode_text)
        log("torch.compile applied to backbone encode_image/encode_text")
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True) if ddp else model

    # data: identical splits + loader as train_lora (Planktonzilla-faithful).
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

    steps_per_epoch = max(1, len(loader) // args.accum)
    total_iters = args.iters if args.iters is not None else int(args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(args.warmup_frac * total_iters))
    if args.eval_epochs is not None:
        args.eval_every = max(1, round(args.eval_epochs * steps_per_epoch))
    log(f"steps/epoch={steps_per_epoch}  epochs={args.epochs}  total_steps={total_iters}  "
        f"warmup={warmup_steps}  eval_every={args.eval_every}")

    pg = param_groups(model, args.wd, base_lr=args.lr)
    if args.optimizer == "adam":
        opt = torch.optim.Adam(pg, lr=args.lr, betas=(0.9, 0.98))
    else:
        opt = torch.optim.AdamW(pg, lr=args.lr, betas=(0.9, 0.98))

    if args.scheduler == "onecycle":
        max_lrs = [g.get("lr_scale", 1.0) * args.lr for g in pg]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lrs, total_steps=total_iters,
            pct_start=args.onecycle_pct_start, anneal_strategy="cos",
            div_factor=args.lr / args.onecycle_min_lr, final_div_factor=1.0,
        )
    else:
        from hyperbolic_plankton.optim import LinearWarmupCosineDecayLR
        sched = LinearWarmupCosineDecayLR(opt, total_steps=total_iters, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda")

    wb = None
    if is_main() and not args.no_wandb:
        import wandb

        os.makedirs(WANDB_DIR, exist_ok=True)
        wb = wandb.init(
            project=args.wandb_project, name=args.tag, dir=WANDB_DIR,
            config={**vars(args), "effective_batch": eff_batch, "world_size": world,
                    "trainable_params": count_trainable(model)["trainable"],
                    "adaptation": "full_ft"},
        )

    os.makedirs(CKPT_DIR, exist_ok=True)
    ddp_model.train()
    it = 0
    t0 = time.perf_counter()
    run_loss = 0.0
    best_unseen = -1.0  # track best periodic unseen mean-F1 -> save _best.pt
    data_iter = iter(loader)
    epoch = 0
    while it < total_iters:
        opt.zero_grad(set_to_none=True)
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
            sync_ctx = (
                ddp_model.no_sync()
                if ddp and micro < args.accum - 1
                else _nullctx()
            )
            with sync_ctx, torch.amp.autocast("cuda"):
                loss, cl, _sel = forward_loss(
                    ddp_model, pixel_values, taxonomy_batch, args.lambda_sel,
                    contrastive=args.contrastive, ranks=ranks, cl_mask=args.cl_mask,
                    lambda_cl=args.lambda_cl, geometry=args.geometry,
                )
                loss = loss / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.accum

        scaler.step(opt)
        scaler.update()
        sched.step()
        model.clamp_params()
        it += 1

        if it % args.log_every == 0:
            n = args.log_every * args.accum
            ips = args.log_every * eff_batch / (time.perf_counter() - t0)
            avg_loss = run_loss / n
            lr = sched.get_last_lr()[0]
            log(f"it {it:>6}/{total_iters} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"{ips:.0f} img/s")
            if wb is not None:
                wb.log({"train/loss": avg_loss, "train/lr": lr,
                        "train/logit_scale": model.logit_scale.exp().item(),
                        "train/img_per_s": ips, "train/epoch": epoch}, step=it)
            run_loss = 0.0
            t0 = time.perf_counter()

        if it % args.eval_every == 0:
            if is_main():
                metrics = _run_periodic_eval(
                    model, eval_sets, args.num_workers, ranks=ranks, geometry=args.geometry,
                )
                log(f"  [eval it {it}] "
                    f"unseen species_f1={metrics.get('eval/unseen/species_f1', 0):.4f} "
                    f"seen species_f1={metrics.get('eval/seen/species_f1', 0):.4f}")
                if wb is not None:
                    wb.log(metrics, step=it)
                # best-unseen checkpoint (peak is often mid-run): score = mean unseen F1.
                unseen_fs = [v for k, v in metrics.items()
                             if k.startswith("eval/unseen/") and k.endswith("_f1")]
                cur = sum(unseen_fs) / len(unseen_fs) if unseen_fs else -1.0
                if cur > best_unseen:
                    best_unseen = cur
                    path = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
                    torch.save({"model": model.state_dict(), "it": it, "args": vars(args),
                                "unseen_mean_f1": cur}, path)
                    log(f"  ↑ best unseen mean-F1 {cur:.4f} -> saved {path}")
            if ddp:
                dist.barrier()
            t0 = time.perf_counter()

        if is_main() and it % args.ckpt_every == 0:
            path = os.path.join(CKPT_DIR, f"{args.tag}_it{it}.pt")
            torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, path)
            log(f"  saved {path}")

    if is_main():
        path = os.path.join(CKPT_DIR, f"{args.tag}_final.pt")
        torch.save({"model": model.state_dict(), "it": it, "args": vars(args)}, path)
        log(f"DONE. saved {path}")

        # Final test eval (the paper numbers): FULL seen+unseen splits, present-classes,
        # per-rank macro-F1 — logged to wandb under test/* alongside the run.
        log("running final test eval (full seen/unseen splits)...")
        test_metrics = run_final_eval(
            model, args.dataset, geometry=args.geometry, num_workers=args.num_workers,
        )
        if wb is not None:
            wb.log(test_metrics, step=it)
            wb.summary.update(test_metrics)
            wb.finish()
    # non-rank-0 ranks wait here while rank 0 runs the (rank-0-only) final eval, so the
    # process group isn't torn down underneath it.
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
