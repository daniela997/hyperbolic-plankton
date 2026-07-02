#!/bin/bash
# Rerun the B0 baseline + C1-C10 ablation configs at BATCH 1024 with SEL-GradCache (loss-agnostic).
# The originals ran at batch 768 (micro 64 x accum 6 x 2 GPUs) with PER-MICRO SEL (starved: ~110
# sibling pairs/step). These rerun at batch 1024 (micro 128 x accum 4 x 2 GPUs) with SEL on the FULL
# GradCached batch (~9954 sibling pairs/step) — so SEL can finally SEE + orient the sibling structure.
# micro-bs 128 is the biggest that fits (256 OOMs). Needs expandable_segments.
# v4 recipe otherwise. Idempotent; waits for GPUs. New tags: *_b1024_r64_v4.
#   nohup bash scripts/run_b0c_batch1024.sh > /tmp/b0c_1024.log 2>&1 &

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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29595 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 4 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# B0 baseline: distance CL + SEL-independent
run "bioscan_B0_baseline_b1024_r64_v4"                 --contrastive distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent

# C1: distance CL + SEL-cumulative
run "bioscan_C1_seltext_cumulative_b1024_r64_v4"       --contrastive distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative
# C2: SEL-only cumulative (no CL)
run "bioscan_C2_selonly_cumulative_b1024_r64_v4"       --contrastive distance --lambda-cl 0.0 --lambda-sel 1.0 --sel-text cumulative
# C3: SEL-only independent (no CL)
run "bioscan_C3_selonly_independent_b1024_r64_v4"      --contrastive distance --lambda-cl 0.0 --lambda-sel 1.0 --sel-text independent
# C4: CL-only (no SEL)
run "bioscan_C4_clonly_b1024_r64_v4"                   --contrastive distance --lambda-cl 1.0 --lambda-sel 0.0 --sel-text independent
# C5: angle CL + SEL-cumulative
run "bioscan_C5_selcumulative_clangle_b1024_r64_v4"    --contrastive angle    --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative
# C6: angle CL + SEL-independent
run "bioscan_C6_selindependent_clangle_b1024_r64_v4"   --contrastive angle    --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent
# C7: distance CL + SEL-independent + cl-mask=same
run "bioscan_C7_selindependent_cldistance_masksame_b1024_r64_v4" --contrastive distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent --cl-mask same
# C8: distance CL + SEL-cumulative + cl-mask=same
run "bioscan_C8_selcumulative_cldistance_masksame_b1024_r64_v4"  --contrastive distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative --cl-mask same
# C9: angle CL + SEL-independent + cl-mask=same
run "bioscan_C9_selindependent_clangle_masksame_b1024_r64_v4"    --contrastive angle    --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent --cl-mask same
# C10: angle CL + SEL-cumulative + cl-mask=same
run "bioscan_C10_selcumulative_clangle_masksame_b1024_r64_v4"    --contrastive angle    --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative --cl-mask same

echo -e "\n✅ B0 + C1-C10 batch-1024 SEL-GradCache reruns complete."
