#!/bin/bash
# --sel-margin (cone-CONTAINMENT) experiment on BIOSCAN. The SEL-intra hinge becomes
# relu(angle + w*psi(child) - psi(parent)), requiring each child rank's WHOLE cone inside its
# parent's, not just the apex. Goal: restore TRANSITIVE entailment (image in species => in all
# ancestors; B0 had 0.89->0.31 dropoff) + self-induce radial spread (only satisfiable at
# psi(child)<psi(parent) = child deeper). Derived from the target geometry, not UNCHA.
#
# Tested on B0 (indep-SEL, origin-collapsed -> margin should push ranks out), C1 (dist-CL +
# cumul-SEL, deep ranks already spread, coarse ranks origin-collapsed -> margin completes the
# chain; the best candidate), C5 (angle-CL + cumul-SEL). Two weights each {0.5, 1.0}; w=0 is the
# baseline config (already in the main ladder). Same v3 recipe (r64/wd1e-3/lr2.5e-4/adam/
# onecycle/80ep/seed0/fp16) so only --sel-margin varies vs the ladder's B0/C1/C5.
#
#   tmux new-session -d -s margin 'bash scripts/run_sel_margin_bioscan.sh 2>&1 | tee /tmp/margin.log'

cd /home/daniela/mine/hyperbolic-plankton
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun

run() {
    local TAG=$1; shift
    echo -e "\n================== $TAG =================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29556 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 80 --micro-bs 128 --accum 3 \
        --lr 2.5e-4 --wd 1e-3 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 1.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED — continuing"
    fi
}

# base loss configs (vary --sel-margin):
#   B0  = distance CL + independent SEL
#   C1  = distance CL + cumulative  SEL
#   C5  = angle    CL + cumulative  SEL
B0="--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --sel-text independent --cl-mask none"
C1="--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --sel-text cumulative --cl-mask none"
C5="--lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle    --sel-text cumulative --cl-mask none"

for w in 0.5 1.0; do
    run "bioscan_B0_selmargin${w}_r64_v3" $B0 --sel-margin $w
    run "bioscan_C1_selmargin${w}_r64_v3" $C1 --sel-margin $w
    run "bioscan_C5_selmargin${w}_r64_v3" $C5 --sel-margin $w
done

echo -e "\n✅ sel-margin experiment complete."
