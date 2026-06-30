#!/bin/bash
# Margin sweep on the HYBRID+SEL-cumul base (the healthiest geometry: inSp 0.82, non-collapsed,
# but ranks clustered in the center / cones wide). Goal: push embeddings to the rim + narrow cones
# WITHOUT desyncing image containment. The whole-cone constraint (sel-margin) was only ever run on
# the LS/lrcl-species base (where it desynced images, inSp 0.82->0.16) — NEVER on hybrid. This tests
# it on the correct base. Adds: (1) sel-margin 0.25/0.5 (positive whole-cone containment),
# (2) sel-margin 0.25 + sel-neg-margin 0.25 (also push WRONG cones apart -> determinism).
#
# Identical to bioscan_H_hybrid_selcumul_r64_v4 (m0 control) except the margin flags.
# v4 recipe: adam/onecycle/lr2.5e-4/wd1e-4/r64/12+12/50ep/seed0/cache-accum, micro-bs64 accum6.
# Idempotent skip-if-done; waits for free GPUs before each run.
#
#   nohup bash scripts/run_hybrid_margin_bioscan.sh > /tmp/hybrid_margin.log 2>&1 &

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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29574 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        --contrastive hybrid --rince-sim distance --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# positive whole-cone containment margin (push to rim, narrow cones — does it keep images inside?)
run "bioscan_H_hybrid_selcumul_m025_r64_v4"      --sel-margin 0.25
run "bioscan_H_hybrid_selcumul_m05_r64_v4"       --sel-margin 0.5

# positive + negative whole-cone margin (also push WRONG cones apart -> more deterministic)
run "bioscan_H_hybrid_selcumul_m025_n025_r64_v4" --sel-margin 0.25 --sel-neg-margin 0.25

echo -e "\n✅ hybrid margin sweep complete."
