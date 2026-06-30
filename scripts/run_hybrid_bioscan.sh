#!/bin/bash
# Hybrid loss ablation (v4 recipe): RINCE exactly-d grading + LRCL deduped-prototype symmetric I<->T,
# on CUMULATIVE text. Two runs: hybrid (CL-only) and hybrid + SEL-cumulative.
#
# Same recipe as run_ablations_bioscan_v4.sh: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12 blocks/50ep/
# seed0/fp16/cache-accum, micro-bs64 accum6 (eff 768). Idempotent skip-if-done.
#
#   tmux new-session -d -s hybrid 'bash scripts/run_hybrid_bioscan.sh 2>&1 | tee /tmp/hybrid.log'

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

run() {
    local TAG=$1; shift
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done — skipping"; return
    fi
    echo -e "\n========================== 🚀 $TAG =========================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29563 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# (1) hybrid CL-only
run "bioscan_H_hybrid_r64_v4" \
    --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 0.0

# (2) hybrid + SEL-cumulative
run "bioscan_H_hybrid_selcumul_r64_v4" \
    --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative

echo -e "\n✅ hybrid ablation complete."
