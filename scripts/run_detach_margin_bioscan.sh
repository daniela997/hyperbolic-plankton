#!/bin/bash
# detach-parent / cone-margin sub-grid (v4 recipe). Tests whether fixing SEL's origin-collapse
# (detach-parent: stop the parent-shrinking gradient) or forcing whole-cone containment
# (cone-margin) makes the cone hierarchy actually HELP — the controlled test of "is SEL bad, or
# is collapsed SEL bad?" vs the v4 plain B0/C1/LRCL+SEL.
#
# Same recipe as run_ablations_bioscan_v4.sh: adam/onecycle/lr2.5e-4/wd1e-4/r64/50ep/cache-accum,
# micro-bs64 accum6. Idempotent skip-if-done.
#
#   tmux new-session -d -s detach 'bash scripts/run_detach_margin_bioscan.sh 2>&1 | tee /tmp/detach.log'

cd /home/daniela/mine/hyperbolic-plankton
PY=/scratch/daniela/miniconda3/envs/dino_plankton/bin/python
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

run() {
    local TAG=$1; shift
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done — skipping"; return
    fi
    echo -e "\n========================== 🚀 $TAG =========================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29561 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# B0 = distCL + SEL-independent.  C1 = distCL + SEL-cumulative.
B0="--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text independent"
C1="--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text cumulative"

# (1) B0 / C1 × {detach-parent, cone-margin}
run "bioscan_B0_detach_r64_v4"      $B0 --sel-detach-parent
run "bioscan_B0_margin1_r64_v4"     $B0 --sel-margin 1.0
run "bioscan_C1_detach_r64_v4"      $C1 --sel-detach-parent
run "bioscan_C1_margin1_r64_v4"     $C1 --sel-margin 1.0

# (2) LRCL-all + SEL × {independent, cumulative}, all with detach-parent
LRCL="--contrastive level-restricted --lrcl-ranks all --lambda-cl 1.0 --lambda-sel 1.0"
run "bioscan_L_lrcl_all_selindep_detach_r64_v4" $LRCL --sel-text independent --sel-detach-parent
run "bioscan_L_lrcl_all_selcumul_detach_r64_v4" $LRCL --sel-text cumulative  --sel-detach-parent

echo -e "\n✅ detach/margin sub-grid complete."
