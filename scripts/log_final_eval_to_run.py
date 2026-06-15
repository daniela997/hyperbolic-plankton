"""Resume an existing wandb run and log the full-split final eval into it under test/*.

Use when a run's end-of-training final eval didn't log (e.g. the DDP barrier crashed after
training, before run_final_eval completed). Runs the SAME run_final_eval path (identical to
the calibration baseline) on a saved checkpoint and writes test/<split>/<rank>_f1 into the
ORIGINAL run, so the dashboard shows the headline numbers next to the training curves.

  PYTHONPATH=src:scripts python scripts/log_final_eval_to_run.py \
    --run-id 7r7cvoa3 --project hyperbolic-plankton-euclidean --entity uofg \
    --ckpt <ckpt.pt> --lora --lora-r 32 --lora-visual-blocks 12 --lora-text-blocks 12 \
    --no-proj --geometry euclidean
"""

from __future__ import annotations

import argparse

import torch
import wandb

from hyperbolic_plankton.lora import apply_lora
from hyperbolic_plankton.model import HyperbolicCLIP
from final_eval import run_final_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="wandb run id to resume (e.g. 7r7cvoa3)")
    ap.add_argument("--project", default="hyperbolic-plankton-euclidean")
    ap.add_argument("--entity", default="uofg")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="planktonzilla", choices=["planktonzilla", "bioscan"])
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip"])
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (default =r); MUST match training")
    ap.add_argument("--lora-visual-blocks", type=int, default=4)
    ap.add_argument("--lora-text-blocks", type=int, default=8)
    ap.add_argument("--lora-mlp", action="store_true")
    ap.add_argument("--no-proj", action="store_true")
    ap.add_argument("--geometry", default="euclidean", choices=["hyperbolic", "euclidean"])
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HyperbolicCLIP(backbone=args.backbone, use_proj=not args.no_proj)
    if args.lora:
        model = apply_lora(model, r=args.lora_r,
                           alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_r,
                           adapt_visual_blocks=args.lora_visual_blocks,
                           adapt_text_blocks=args.lora_text_blocks,
                           include_mlp=args.lora_mlp)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(device).eval()
    print(f"loaded {args.ckpt}")

    metrics = run_final_eval(model, args.dataset, geometry=args.geometry,
                             num_workers=args.num_workers)

    # resume the ORIGINAL run and log test/* (resume="must" so we never create a stray new
    # run). No explicit step: these are final-summary numbers, and an out-of-order step (e.g.
    # _best.pt's it < the run's last logged step) would be rejected. summary.update is what
    # the dashboard reads for headline values.
    run = wandb.init(entity=args.entity, project=args.project, id=args.run_id,
                     resume="must")
    run.log(metrics)
    run.summary.update(metrics)
    run.finish()
    print(f"logged {len(metrics)} test/* metrics into run {args.run_id}")


if __name__ == "__main__":
    main()
