#!/bin/bash

# Hyperbolic ablations on PLANKTONZILLA — the thesis runs (Lorentz lift + entailment).
# Two runs, isolating the SEL term:
#   H_cl     CL-only      (--lambda-sel 0): hyperbolic contrastive, NO entailment. The
#            geometry-without-hierarchy control — is the lift alone worth anything?
#   H_clsel  CL + SEL     (--lambda-sel 1): the full method (distance-CL + stacked
#            entailment). The headline hyperbolic model.
#
# vs the Euclidean E0c (same backbone+LoRA, FLAT) = geometry delta.
# vs the full-FT Planktonzilla bar (docs/baseline-planktonzilla-clip.md) = the absolute ref.
#
# RECIPE = the one the Euclidean campaign LEARNED (NOT the stale paper-faithful B0 in
# experiment-plan §2). Deviations from documented-B0, deliberate:
#   - scheduler warmupcos (NOT onecycle): onecycle's long ramp walked E0c off its early
#     optimum; lower-LR warmupcos converged best.
#   - lr 1e-4 (NOT 5e-5): the best Euclidean LR; hyperbolic CL is the same InfoNCE scale.
#   - lora-r 32 (NOT 128): r=32 ≈ r=128 on E0c, far fewer params (efficiency thesis).
#   - 20 epochs (NOT 50): E0c converged well within 20; pz at 50ep = ~114k steps is huge.
#   - bf16 (kept: more stable than fp16 for the acosh/cone ops, same speed).
# HYPERBOLIC essentials (NOT shared with the Euclidean E0c):
#   - projector ON (NO --no-proj): the lift needs a projection into the hyperboloid;
#     --no-proj is euclidean-only and would SystemExit here.
#   - --geometry hyperbolic (default), --contrastive distance, --sel-text independent,
#     plain hinge SEL (no UNCHA: --sel-tau 1 --sel-leak 0 --sel-uncertainty 0).
#   - free curvature (no --curv-lr-scale); clamp_params guards collapse. WATCH the curv
#     curve — λ_sel=1 caused curvature collapse in early runs (see build-log).
#
# Logged to the hyperbolic-plankton project. Each run does the full seen/unseen TEST eval
# at the end (auto-logged test/*, now that the NCCL barrier bug is fixed).
#
# Eval a checkpoint: PYTHONPATH=src python scripts/final_eval.py \
#   --ckpt <dir>/<tag>_best.pt --dataset planktonzilla --backbone clip --lora --lora-r 32 \
#   --lora-visual-blocks 12 --lora-text-blocks 12 --geometry hyperbolic

set -e
cd /home/daniela/mine/hyperbolic-plankton

WANDB_PROJECT="hyperbolic-plankton"

run_hyperbolic() {
    local TAG=$1
    shift 1
    echo -e "\n================================================================="
    echo "🚀 Starting hyperbolic run: $TAG"
    echo "================================================================="
    # micro-bs/accum are passed PER CALL (effective batch = micro-bs * accum * 2 GPUs = 768).
    # CL+SEL needs micro-bs 64 (SEL adds a 2nd 7-rank text encode -> 128 OOMs the 24GB A5000);
    # CL-only skips SEL (forward_loss early-returns at lambda_sel==0), so it fits at 128/3 and
    # runs ~2x fewer optimizer-step iterations -> finishes quicker.
    PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29557 scripts/train_lora.py \
        --dataset planktonzilla --backbone clip --epochs 20 \
        --lr 1e-4 --wd 0.2 --optimizer adamw --scheduler warmupcos --warmup-frac 0.1 \
        --lora-r 32 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --geometry hyperbolic --contrastive distance --cl-mask none \
        --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
        --compile --eval-epochs 1.0 --eval-cap 50 \
        --wandb-project "$WANDB_PROJECT" \
        "$@" \
        --tag "$TAG"
}

# H_cl — CL-only (lift, no entailment): isolates whether the hyperbolic contrastive alone
# helps over Euclidean E0c, before SEL. Skips SEL -> fits micro-bs 128/accum 3.
run_hyperbolic "planktonzilla_H_cl_lora_r32_20ep" \
    --micro-bs 128 --accum 3 --lambda-cl 1.0 --lambda-sel 0.0

# H_clsel — CL + SEL: the full hyperbolic method. Now fits micro-bs 128/accum 3 too:
# encode_taxonomy deduplicates text (batches are ~80% duplicate per rank), so SEL's text
# encode no longer forces the smaller batch — same speed as H_cl.
run_hyperbolic "planktonzilla_H_clsel_lora_r32_20ep" \
    --micro-bs 128 --accum 3 --lambda-cl 1.0 --lambda-sel 1.0
