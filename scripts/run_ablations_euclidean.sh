#!/bin/bash

# Euclidean-LoRA baseline (E0) — the missing 2×2 corner: same frozen backbone + LoRA +
# projector as the hyperbolic runs, but FLAT space (cosine CLIP InfoNCE = open_clip
# ClipLoss, no exp-map lift, no SEL). Isolates LoRA-vs-full-FT (vs Planktonzilla) and
# Euclidean-vs-hyperbolic (vs B0). `--geometry euclidean` forces --lambda-sel 0.
#
# Logged to a SEPARATE wandb project (hyperbolic-plankton-euclidean) so the Euclidean
# baselines don't clutter the hyperbolic experiment project. Each run does a full
# seen/unseen TEST eval at the end (the paper numbers), logged under test/*.
#
# Two recipe variants, because we do NOT have a validated LoRA-CLIP-contrastive recipe and
# the two roles want different optimizer settings:
#   E0a — HAC-grounded LoRA recipe (lr 2.5e-4 / wd 0.2 / adamw, from HAC's
#         train_hac_vit_b_lora.py — the closest LoRA-on-CLIP precedent). The "good LoRA
#         recipe" answer; serves the LoRA-vs-full-FT-vs-Planktonzilla comparison.
#         (param_groups already excludes LoRA params from weight decay, as HAC does.)
#   E0b — B0-matched (lr 5e-5 / wd 1e-4 / adam). Identical to the hyperbolic B0 except
#         geometry, so E0b-vs-B0 is a clean Euclidean-vs-hyperbolic delta.
#
# We START on PLANKTONZILLA: it is the "expensive" dataset, but we already have the
# full-FT reference (paper Table 2/3 macro-F1 + released weights), so the first LoRA
# number lands against a real comparator. BIOSCAN runs follow.
#
# Read each with: PYTHONPATH=src python scripts/final_eval.py \
#   --ckpt <tag>_final.pt --dataset <ds> --backbone clip --lora --geometry euclidean

set -e
cd /home/daniela/mine/hyperbolic-plankton

WANDB_PROJECT="hyperbolic-plankton-euclidean"

# Per-run launch: dataset is passed per call. Eval cadence is once per epoch
# (--eval-epochs 1, derived from steps_per_epoch) so it means the same on both datasets.
# Everything else matches the hyperbolic ablations.
run_euclidean() {
    local TAG=$1
    local DATASET=$2
    shift 2

    echo -e "\n================================================================="
    echo "🚀 Starting run: $TAG  (dataset=$DATASET)"
    echo "================================================================="

    # NOTE: short --epochs + HAC-shape schedule. The first 50-epoch planktonzilla E0a run
    # showed BOTH seen and unseen F1 peak at epoch ~4 then DECLINE while OneCycle's LR was
    # still RAMPING toward its peak (pct_start=0.3 → 30% of training spent climbing) — the
    # long ramp walked the model off the early optimum. Fix: HAC's own schedule shape
    # (--scheduler warmupcos = LinearWarmupCosineDecayLR, short --warmup-frac then decay),
    # so peak LR is hit early (~epoch 0.5) and the useful window is in the decay phase, not
    # the ramp. LoRA-CLIP fine-tunes are short (cf. AMD clipora ~3 ep): 5 ep, lr 1e-4.
    # --eval-epochs 0.5 so the early peak is sampled. Pass --epochs to override per run.
    PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29556 scripts/train_lora.py \
        --dataset "$DATASET" --backbone clip --epochs 5 --micro-bs 128 --accum 3 \
        --scheduler warmupcos --warmup-frac 0.1 --lora-r 128 --eval-epochs 0.5 \
        --geometry euclidean --lambda-cl 1.0 --cl-mask none --compile \
        --wandb-project "$WANDB_PROJECT" \
        "$@" \
        --tag "$TAG"
}

# ==============================================================================
# PLANKTONZILLA — start here (full-FT reference already known from the paper)
# ==============================================================================

# E0a — short LoRA recipe: 5 epochs, lr 1e-4 (replaces the 50-epoch/2.5e-4 run that
# declined after epoch ~4). wd 0.2 / adamw retained from the HAC LoRA recipe.
run_euclidean "planktonzilla_E0a_euclidean_lora_5ep_lr1e-4" planktonzilla \
    --lr 1e-4 --wd 0.2 --optimizer adamw

# E0c — like E0a but --no-proj: drops our visual_proj/textual_proj heads so the model is
# architecturally IDENTICAL to a bare full-FT CLIP (CLIP + LoRA only -> cosine). This is the
# clean LoRA-vs-full-FT calibration against the Planktonzilla CLIP weights (no extra-projector
# confound). Differs from E0a by the projector ONLY.
# run_euclidean "planktonzilla_E0c_euclidean_lora_noproj" planktonzilla \
#     --lr 1e-4 --wd 0.2 --optimizer adamw --no-proj

# E0b — B0-matched recipe (lr 5e-5 / wd 1e-4 / adam)
# run_euclidean "planktonzilla_E0b_euclidean_b0matched" planktonzilla \
#     --lr 5e-5 --wd 1e-4 --optimizer adam

# ==============================================================================
# BIOSCAN — the clean complete-taxonomy control (run after planktonzilla)
# ==============================================================================

# E0a — HAC-grounded LoRA recipe
# run_euclidean "bioscan_E0a_euclidean_lora_hac" bioscan \
#     --lr 2.5e-4 --wd 0.2 --optimizer adamw

# E0b — B0-matched recipe (clean geometry control vs B0)
# run_euclidean "bioscan_E0b_euclidean_b0matched" bioscan \
#     --lr 5e-5 --wd 1e-4 --optimizer adam

echo -e "\n✅ Euclidean baseline run(s) completed successfully!"
