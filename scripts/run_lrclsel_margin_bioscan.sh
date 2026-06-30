#!/bin/bash
# C1-with-LRCL-CL ablation: swap C1's distance-CL for lrcl_species CL, + SEL-cumulative, across
# sel-margin {0, 0.5, 1.0}. Tests whether the dedup/group-balanced CL + cone-containment margin
# rescues SEL-cumulative (C1 = distance-CL + SEL-cumul, sel_margin=0, classified 0.60).
# Reference (CL-only, no SEL) = the existing v4 L_lrcl_species (0.694).
#
# v4 recipe: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12/50ep/seed0/cache-accum, micro-bs64 accum6.
# Idempotent skip-if-done.
#
#   bash scripts/run_lrclsel_margin_bioscan.sh 2>&1 | tee /tmp/lrclsel.log

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

run() {
    local TAG=$1; shift
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done — skipping"; return
    fi
    echo -e "\n========================== 🚀 $TAG =========================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29568 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# lrcl_species CL + SEL-cumulative, across sel-margin
BASE="--contrastive level-restricted --lrcl-ranks species --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative"

run "bioscan_LS_lrclsp_selcumul_m0_r64_v4"   $BASE --sel-margin 0.0
run "bioscan_LS_lrclsp_selcumul_m05_r64_v4"  $BASE --sel-margin 0.5
run "bioscan_LS_lrclsp_selcumul_m1_r64_v4"   $BASE --sel-margin 1.0

echo -e "\n✅ lrcl_species + SEL-cumulative × sel-margin ablation complete."

# (appended) hybrid + SEL-cumulative — runs LAST, after the margin sweep above.
run "bioscan_H_hybrid_selcumul_r64_v4" \
    --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative

echo -e "\n✅ hybrid + SEL-cumulative complete."
