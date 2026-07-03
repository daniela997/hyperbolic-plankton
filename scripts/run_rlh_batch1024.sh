#!/bin/bash
# Rerun the core RANKED (RINCE) / LRCL / HYBRID configs at BATCH 1024 with SEL-GradCache.
# Mirrors the 13 core v4 configs (originals ran batch 768, PER-MICRO SEL ~110 sibling pairs/step).
# Now: batch 1024 (micro 128 x accum 4 x 2 GPUs) with SEL on the FULL GradCached batch (~9954 pairs).
# Companion to run_b0c_batch1024.sh. micro-bs 128 is the max that fits (256 OOMs); needs expandable_segments.
# v4 recipe otherwise. Idempotent; waits for GPUs. New tags: *_b1024_r64_v4.
#   nohup bash scripts/run_rlh_batch1024.sh > /tmp/rlh_1024.log 2>&1 &

cd /home/daniela/mine/hyperbolic-plankton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29596 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 4 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 --lambda-cl 1.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# ---- RANKED (RINCE) ----
run "bioscan_R_ranked_distance_clonly_b1024_r64_v4"    --contrastive ranked --rince-sim distance --lambda-sel 0.0
run "bioscan_R_ranked_distance_selcumul_b1024_r64_v4"  --contrastive ranked --rince-sim distance --lambda-sel 1.0 --sel-text cumulative
run "bioscan_R_ranked_distance_selindep_b1024_r64_v4"  --contrastive ranked --rince-sim distance --lambda-sel 1.0 --sel-text independent
run "bioscan_R_ranked_angle_clonly_b1024_r64_v4"       --contrastive ranked --rince-sim angle    --lambda-sel 0.0
run "bioscan_R_ranked_angle_selcumul_b1024_r64_v4"     --contrastive ranked --rince-sim angle    --lambda-sel 1.0 --sel-text cumulative
run "bioscan_R_ranked_angle_selindep_b1024_r64_v4"     --contrastive ranked --rince-sim angle    --lambda-sel 1.0 --sel-text independent

# ---- LRCL ----
run "bioscan_L_lrcl_all_b1024_r64_v4"                  --contrastive level-restricted --lrcl-ranks all     --lambda-sel 0.0
run "bioscan_L_lrcl_all_selindep_b1024_r64_v4"         --contrastive level-restricted --lrcl-ranks all     --lambda-sel 1.0 --sel-text independent
run "bioscan_L_lrcl_species_b1024_r64_v4"              --contrastive level-restricted --lrcl-ranks species --lambda-sel 0.0

# ---- HYBRID ----
run "bioscan_H_hybrid_b1024_r64_v4"                    --contrastive hybrid --rince-sim distance --lambda-sel 0.0
run "bioscan_H_hybrid_selcumul_b1024_r64_v4"           --contrastive hybrid --rince-sim distance --lambda-sel 1.0 --sel-text cumulative
run "bioscan_H_hybrid_angle_b1024_r64_v4"              --contrastive hybrid --rince-sim angle    --lambda-sel 0.0
run "bioscan_H_hybrid_angle_selcumul_b1024_r64_v4"     --contrastive hybrid --rince-sim angle    --lambda-sel 1.0 --sel-text cumulative

echo -e "\n✅ RANKED + LRCL + HYBRID batch-1024 SEL-GradCache reruns complete."
