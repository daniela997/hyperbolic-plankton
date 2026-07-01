#!/bin/bash
# tau-matched control: RINCE-dist and hybrid-dist with FLAT tau=1.0 (= LRCL's temperature), everything
# else identical to their baselines. Tests whether RINCE/hybrid's TIGHT clustering (proto radius ~0.5)
# vs LRCL's spread (radius ~1.2) is caused by TEMPERATURE (per-tier tau 0.1-0.5 vs LRCL's flat 1.0).
# Prediction: tau=1.0 -> larger radius + higher NN-sep, approaching LRCL. Baselines: R_ranked_distance
# (radius 0.52, NN 0.17) and H_hybrid (radius 0.44, NN 0.30) vs LRCL-all (radius 1.20, NN 0.61).
#
# min_tau=max_tau=1.0 makes the per-tier schedule flat (tau = 1.0 at every rank), matching LRCL.
# v4 recipe. Idempotent skip-if-done; waits for GPUs.
#   nohup bash scripts/run_tau_matched_bioscan.sh > /tmp/tau_matched.log 2>&1 &

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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29577 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 --lambda-sel 0.0 \
        --rince-min-tau 1.0 --rince-max-tau 1.0 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# RINCE-dist, flat tau=1.0 (baseline: rince_min_tau=0.1 max_tau=0.5)
run "bioscan_R_ranked_distance_tau1_r64_v4" --contrastive ranked --rince-sim distance

# hybrid-dist, flat tau=1.0
run "bioscan_H_hybrid_tau1_r64_v4"          --contrastive hybrid --rince-sim distance

# hybrid-dist, flat tau=1.0 + radial (does the spread-out version also get the radial ladder?)
run "bioscan_H_hybrid_tau1_radial05_r64_v4" --contrastive hybrid --rince-sim distance --lambda-radial 0.5

echo -e "\n✅ tau-matched (+radial) runs complete."
