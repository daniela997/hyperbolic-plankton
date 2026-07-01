#!/bin/bash
# ATMG-motivated radial sweep: hybrid-ANGLE CL + radial_ordering_loss (L_centroid), NO SEL.
# ATMG argues angle-CL IS the smoothed entailment loss and needs the CENTROID/radial term (not the
# hard SEL hinge, which Prop.2 shows collapses). We found hybrid-angle+SEL collapses (100% saturated
# psi, dir-cos 0.64); this tests whether angle-CL + radial (no SEL) gives non-degenerate hierarchy.
# No reference lambda exists (ATMG paper unstated; their code ships L_centroid DISABLED), so sweep it.
# rad value ~0.10, cl ~0.15 -> lambda 0.5/1.0/2.0 spans half..2x CL weight.
#
# Baseline (lambda_radial=0) is bioscan_H_hybrid_angle_r64_v4 (run separately).
# v4 recipe: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12/50ep/seed0/cache-accum, micro-bs64 accum6.
# Idempotent skip-if-done; waits for free GPUs.
#
#   nohup bash scripts/run_hybrid_angle_radial_bioscan.sh > /tmp/hybrid_angle_radial.log 2>&1 &

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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29576 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        --contrastive hybrid --rince-sim angle --lambda-cl 1.0 --lambda-sel 0.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

run "bioscan_H_hybrid_angle_radial05_r64_v4" --lambda-radial 0.5
run "bioscan_H_hybrid_angle_radial1_r64_v4"  --lambda-radial 1.0
run "bioscan_H_hybrid_angle_radial2_r64_v4"  --lambda-radial 2.0

echo -e "\n✅ hybrid-angle radial sweep complete."
