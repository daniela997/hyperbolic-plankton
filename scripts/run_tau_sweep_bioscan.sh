#!/bin/bash
# max_tau sweep for RINCE and hybrid (min_tau fixed at 0.1 = RINCE's sharp species temp).
# Sweeps the COARSE-END temperature width. Our baseline max_tau=0.5 was an undocumented guess
# (RINCE's actual default is 0.2); this brackets it: 0.2 (RINCE), 0.35 (mid), 0.7 (wide).
# 0.5 (baseline) already exists as R_ranked_distance_clonly / H_hybrid. Flat tau=1.0 is a SEPARATE
# axis (run_tau_matched). Keeping species tau=0.1 fixed isolates the coarse-rank effect (species
# NN-sep/F1 should be ~unchanged; coarse-rank prototype placement is what varies).
#
#   max_tau  species genus family order
#   0.2      0.10    0.12  0.15   0.18   (RINCE default)
#   0.35     0.10    0.16  0.22   0.29
#   0.7      0.10    0.25  0.40   0.55   (wide)
#
# v4 recipe. Idempotent skip-if-done; waits for GPUs. Runs after the tau_matched queue frees them.
#   nohup bash scripts/run_tau_sweep_bioscan.sh > /tmp/tau_sweep.log 2>&1 &

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
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29578 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 --lambda-sel 0.0 --rince-sim distance \
        --rince-min-tau 0.1 \
        "$@" --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing"
    fi
}

# RINCE (ranked) — max_tau sweep
run "bioscan_R_ranked_distance_maxtau02_r64_v4"  --contrastive ranked --rince-max-tau 0.2
run "bioscan_R_ranked_distance_maxtau035_r64_v4" --contrastive ranked --rince-max-tau 0.35
run "bioscan_R_ranked_distance_maxtau07_r64_v4"  --contrastive ranked --rince-max-tau 0.7

# hybrid — max_tau sweep
run "bioscan_H_hybrid_maxtau02_r64_v4"  --contrastive hybrid --rince-max-tau 0.2
run "bioscan_H_hybrid_maxtau035_r64_v4" --contrastive hybrid --rince-max-tau 0.35
run "bioscan_H_hybrid_maxtau07_r64_v4"  --contrastive hybrid --rince-max-tau 0.7

echo -e "\n✅ max_tau sweep complete."
