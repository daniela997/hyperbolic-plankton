#!/bin/bash
# Fill the obvious SEL gaps: hybrid+SEL-cumul (SEL on the WINNER) and LRCL-all+SEL-cumul (completes
# the LRCL grid). Waits for GPUs to be free before each run (so it can be queued behind another job).
#
# v4 recipe: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12/50ep/seed0/cache-accum, micro-bs64 accum6.
# Idempotent skip-if-done.
#
#   nohup bash scripts/run_sel_gaps_bioscan.sh > /tmp/sel_gaps.log 2>&1 &

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

wait_for_gpus() {
    # block until BOTH GPUs are <1GB used (no other training job)
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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29572 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# hybrid + SEL-cumulative (SEL on the winner)
run "bioscan_H_hybrid_selcumul_r64_v4" \
    --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative

# LRCL-all + SEL-cumulative (completes the LRCL grid)
run "bioscan_L_lrcl_all_selcumul_r64_v4" \
    --contrastive level-restricted --lrcl-ranks all --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative

# hybrid with ANGLE similarity (ATMG-motivated): does angle-CL classify well when the hybrid's
# dedup/symmetric/grading structure protects it from the collapse that killed bare angle-CL (C5/C6)?
# (rince-sim angle now fixed: species-apex, negated.)
run "bioscan_H_hybrid_angle_r64_v4" \
    --contrastive hybrid --rince-sim angle --lambda-cl 1.0 --lambda-sel 0.0

# hybrid-angle + SEL-cumulative
run "bioscan_H_hybrid_angle_selcumul_r64_v4" \
    --contrastive hybrid --rince-sim angle --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative

echo -e "\n✅ SEL-gaps + hybrid-angle ablation complete."
