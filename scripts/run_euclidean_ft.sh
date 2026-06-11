#!/bin/bash

# Euclidean full fine-tune baseline (FT) — the Planktonzilla CLIP recipe on our splits.
# CLIP ViT-B/16, backbone UNFROZEN, flat cosine InfoNCE (open_clip ClipLoss), lineage text.
# No LoRA, no hyperbolic lift, no SEL. The top-left corner of the baseline 2×2:
#
#                  Euclidean              Hyperbolic
#   Full FT    -> FT (this script) <-     (n/a)
#   LoRA          E0a/E0b                 B0 + ladder
#
# Planktonzilla's reported recipe (paper p.7 + scripts/train_clip.sh): lr 1e-4, wd 0.2,
# adamw, 100 epochs, global batch 16,384, 64×H100. We adapt to 2×A5000: batch 768, 50
# epochs (the project's fixed budget). So this is the Planktonzilla recipe ADAPTED to our
# compute, not a literal reproduction — the batch/epochs differ by necessity.
#
# Read with: PYTHONPATH=src python scripts/final_eval.py \
#   --ckpt <tag>_final.pt --dataset bioscan --backbone clip --geometry euclidean
#   (NOTE: no --lora flag — this is a full fine-tune.)

set -e
cd /home/daniela/mine/hyperbolic-plankton

run_ft() {
    local TAG=$1
    shift

    echo -e "\n================================================================="
    echo "🚀 Starting full-FT run: $TAG"
    echo "================================================================="

    PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29557 scripts/train_euclidean_ft.py \
        --backbone clip --micro-bs 128 --accum 3 --epochs 50 \
        --scheduler onecycle --eval-epochs 1 --compile \
        --wandb-project hyperbolic-plankton-euclidean \
        "$@" \
        --tag "$TAG"
}

# ==============================================================================
# BIOSCAN — Euclidean full fine-tune (Planktonzilla recipe: lr 1e-4 / wd 0.2 / adamw)
# ==============================================================================
run_ft "bioscan_FT_euclidean" \
    --dataset bioscan --lr 1e-4 --wd 0.2 --optimizer adamw

# Planktonzilla full-FT can be added here later (much longer run):
# run_ft "planktonzilla_FT_euclidean" \
#     --dataset planktonzilla --lr 1e-4 --wd 0.2 --optimizer adamw

echo -e "\n✅ Euclidean full-FT run(s) completed successfully!"
