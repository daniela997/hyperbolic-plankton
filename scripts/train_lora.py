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

from hyperbolic_plankton.data import RANKS, HFTaxonomyDataset
from hyperbolic_plankton.eval import (
    class_set_from_dataset,
    flatten_metrics,
    geometry_stats,
    run_unseen_eval,
)
from hyperbolic_plankton.loss import (
    _deepest_text,
    hyperbolic_contrastive_loss,
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


def _build_eval_sets(cache, args):
    """Fixed, seeded subsamples of seen-val and unseen for periodic eval (rank 0 only).

    Returns {seen: (ds, classes), unseen: (ds, classes)}. seen-val predicts among the
    classes present in the seen-val subsample; unseen uses the prebuilt 220-class set.
    """
    rng = np.random.default_rng(0)

    val_idx = np.load(f"{SPLIT_DIR}/val_idx.npy")
    sel = rng.choice(val_idx, size=min(args.eval_n, len(val_idx)), replace=False)
    seen_val = cache.select(sorted(sel.tolist()))
    seen_classes = class_set_from_dataset(seen_val)
    seen_ds = HFTaxonomyDataset(seen_val)

    unseen_idx = np.load(f"{SPLIT_DIR}/unseen_idx.npy")
    sel = rng.choice(unseen_idx, size=min(args.eval_n, len(unseen_idx)), replace=False)
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


def _run_periodic_eval(model, eval_sets, num_workers):
    """Eval seen-val + unseen subsamples + per-rank geometry; flat wandb-loggable dict."""
    from hyperbolic_plankton.train import TaxonomyCollator

    was_training = model.training
    model.eval()
    out = {}
    for name, (ds, classes) in eval_sets["sets"].items():
        res = run_unseen_eval(model, ds, classes, num_workers=num_workers)
        out.update(flatten_metrics(res["metrics"], prefix=f"eval/{name}"))
        out[f"eval/{name}/n_classes"] = res["n_classes"]

    # geometry diagnostics + SEL term decomposition on the fixed batch
    pixel_values, taxonomy_batch, _ = TaxonomyCollator(model.preprocess)(eval_sets["geom_items"])
    out.update(geometry_stats(model, taxonomy_batch))

    # per-term SEL (intra per edge + inter, pos/neg components) for understanding which
    # part of the loss is active — logged under loss_terms/*.
    with torch.no_grad():
        img = model.encode_image(pixel_values.to(model.device))
        text_embs = model.encode_taxonomy(taxonomy_batch)
        sel_stats: dict = {}
        _, intra, inter = stacked_entailment_loss(
            img, text_embs, taxonomy_batch, RANKS, model.curvature, stats=sel_stats
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


def forward_loss(model, pixel_values, taxonomy_batch, lambda_sel, stats=None):
    """contrastive(img, deepest_text) + lambda*SEL. `model` may be a DDP wrapper; geometry
    helpers live on the underlying module.

    If `stats` (a dict) is given, the SEL per-edge pos/neg decomposition is collected into
    it (cheap float reads) for per-step logging alongside cl/sel.
    """
    core = model.module if isinstance(model, DDP) else model
    img = core.encode_image(pixel_values)
    text_embs = core.encode_taxonomy(taxonomy_batch)
    curv = core.curvature
    scale = core.logit_scale.exp()
    deepest, _ = _deepest_text(text_embs, RANKS)
    cl = hyperbolic_contrastive_loss(img, deepest, curv, scale)
    sel, intra, inter = stacked_entailment_loss(
        img, text_embs, taxonomy_batch, RANKS, curv, stats=stats
    )
    if stats is not None:
        stats["loss_terms/sel_intra"] = intra.detach().item()
        stats["loss_terms/sel_inter"] = inter.detach().item()
    return cl + lambda_sel * sel, cl.detach(), sel.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="bioclip", choices=["clip", "bioclip"])
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--wd", type=float, default=0.2)
    ap.add_argument("--lambda-sel", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=10000, help="subsample size per eval set")
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
        f"accum={args.accum} -> effective batch={eff_batch}  iters={args.iters}")

    # model + LoRA
    model = apply_lora(HyperbolicCLIP(backbone=args.backbone), r=args.lora_r, alpha=args.lora_r)
    model.to(device)
    if is_main():
        c = count_trainable(model)
        log(f"trainable params: {c['trainable']:,} / {c['total']:,} "
            f"({100 * c['trainable'] / c['total']:.2f}%)")
    ddp_model = DDP(model, device_ids=[device.index], find_unused_parameters=True) if ddp else model

    # data: load the prebuilt Planktonzilla-faithful splits (scripts/build_splits.py).
    # Train on seen-TRAIN only; seen-val + unseen are held out for periodic eval.
    log("loading cache + prebuilt split indices...")
    cache = load_from_disk(CACHE)
    train_idx = np.load(f"{SPLIT_DIR}/train_idx.npy")
    train_ds = HFTaxonomyDataset(cache.select(train_idx.tolist()))
    log(f"seen-train rows: {len(train_ds):,}")

    # eval subsamples (fixed, seeded) built once on rank 0's view; every rank can build
    # them but only rank 0 evaluates.
    eval_sets = _build_eval_sets(cache, args) if is_main() else None

    collate = TaxonomyCollator(model.preprocess)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=0) if ddp else None
    loader = DataLoader(
        train_ds, batch_size=args.micro_bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )

    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.98))
    sched = LinearWarmupCosineDecayLR(opt, total_steps=args.iters, warmup_steps=args.warmup)
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
    while it < args.iters:
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
                    ddp_model, pixel_values, taxonomy_batch, args.lambda_sel, stats=step_stats
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
            n = args.log_every * args.accum
            ips = args.log_every * eff_batch / (time.perf_counter() - t0)
            avg_loss, avg_cl, avg_sel = run_loss / args.log_every, run_cl / n, run_sel / n
            lr = sched.get_last_lr()[0]
            log(f"it {it:>6}/{args.iters} | loss {avg_loss:.4f} cl {avg_cl:.4f} "
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
                metrics = _run_periodic_eval(model, eval_sets, args.num_workers)
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
