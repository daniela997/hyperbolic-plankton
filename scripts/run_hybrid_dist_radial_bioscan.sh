#!/bin/bash
# hybrid-DISTANCE + radial — the "spread the winner to the rim WITHOUT the ray" run (never done).
# hybrid-distance is the best classifier (F1 0.765, NN 0.302, no ray). radial pushes species protos +
# images OUTWARD. Question: does radial reach the rim (species r>~0.7, toward r>1.5) on the DISTANCE base
# while KEEPING NN-sep? (angle+radial reached species 0.72 at lambda=2 with NN UP, but angle rays;
# distance is what we actually want.) This is the precursor to "contained at the rim" — if radial can
# push species far out here keeping NN, adding image containment at the rim becomes worth building.
#
# lambda 1.0 and 2.0 to see the trend (angle version: species 0.68 @1.0, 0.72 @2.0, NN rose 0.33->0.36).
# v4 recipe. Idempotent; waits for GPUs (queues behind the tau runs).
#   nohup bash scripts/run_hybrid_dist_radial_bioscan.sh > /tmp/hybrid_dist_radial.log 2>&1 &

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

wait_for_gpus() {
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
        [ "$used" -lt 2000 ] && break
        echo "⏳ waiting for GPUs to free (currently ${used} MiB used)..."; sleep 120
    done
}

run() {
    local TAG=$1; shift
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done — skipping"; return
    fi
    wait_for_gpus
    echo -e "\n========================== 🚀 $TAG =========================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29579 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 0.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

run "bioscan_H_hybrid_radial1_r64_v4" --lambda-radial 1.0
run "bioscan_H_hybrid_radial2_r64_v4" --lambda-radial 2.0

echo -e "\n✅ hybrid-distance radial runs complete."
