#!/bin/bash
# 2x2 sel_margin (containment) x sel_neg_margin (separation) ablation on lrcl_species + SEL-cumulative.
# Tests the NEW separation lever: does whole-cone separation of the coarse rank cones (sel_neg_margin)
# improve species classification, alone and with containment (sel_margin)?
#
# The 2x2 corners: (m,n) in {0,1}^2. (0,0)=m0 and (1,0)=m1 ALREADY EXIST from run_lrclsel_margin.
# This runs the 2 NEW corners with sel_neg_margin=1:
#   (margin 0, neg 1) — separation only
#   (margin 1, neg 1) — containment + separation (symmetric)
#
# v4 recipe: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12/50ep/seed0/cache-accum, micro-bs64 accum6.
# Idempotent skip-if-done.
#
#   bash scripts/run_negmargin_bioscan.sh 2>&1 | tee /tmp/negmargin.log

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

run() {
    local TAG=$1; shift
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done — skipping"; return
    fi
    echo -e "\n========================== 🚀 $TAG =========================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29570 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

BASE="--contrastive level-restricted --lrcl-ranks species --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative"

# the 2 NEW corners (sel_neg_margin=1); (0,0) and (1,0) already exist as m0/m1.
run "bioscan_LS_lrclsp_selcumul_m0_n1_r64_v4"  $BASE --sel-margin 0.0 --sel-neg-margin 1.0
run "bioscan_LS_lrclsp_selcumul_m1_n1_r64_v4"  $BASE --sel-margin 1.0 --sel-neg-margin 1.0

echo -e "\n✅ neg-margin 2x2 ablation complete (2 new corners; m0/m1 are the neg=0 corners)."
