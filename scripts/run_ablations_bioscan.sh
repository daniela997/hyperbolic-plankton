#!/bin/bash
# BIOSCAN ablation ladder — the DEVELOPMENT testbed (complete-to-species, ~2.5h/run vs ~20h
# on Planktonzilla). Full C0-C10 grid + Euclidean baseline. Develop here; confirm only the
# WINNER + Euclidean baseline on Planktonzilla. 2 GPUs per run, sequential (~30h total).
#
# ONE shared recipe so every difference reflects the LOSS under test, not tuning:
# onecycle / lr 5e-5 (peak) / 80 epochs / r32 / seed 0 / bf16. The first r32 ladder used
# warmupcos/lr1e-4/50ep and UNDERPERFORMED: on BIOSCAN's short ~2350-step runs, warmupcos
# decayed the LR to ~0 by ~85% of training, cutting off the slower-converging hyperbolic
# configs (Euclidean finished first and falsely "won"). The June good runs used onecycle@5e-5;
# this matches that schedule (the one axis that mattered) + 80 epochs so nothing is cut off.
# Differs from June only in rank (r32 vs r128) and precision (bf16 vs fp16), both shown OK.
#
# The grid (B0 = the Taxonomy-paper method; C1-C10 vary one axis each):
#   axes: lambda_cl/sel (CL-only / SEL-only / both), contrastive (distance/angle),
#         sel-text (independent/cumulative), cl-mask (none/same = false-negative suppression)
#
# Read results: PYTHONPATH=src python scripts/final_eval.py --ckpt <dir>/<tag>_best.pt \
#   --dataset bioscan --backbone clip --lora --lora-r 32 --lora-visual-blocks 12 \
#   --lora-text-blocks 12 --geometry {euclidean|hyperbolic}

cd /home/daniela/mine/hyperbolic-plankton

# absolute paths so this works in a non-interactive tmux shell (no conda activation)
PY=/scratch/daniela/miniconda3/envs/dino_plankton/bin/python
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun

# NO `set -e`: one config crashing (e.g. a NaN like C10 last time) must NOT abort the whole
# overnight ladder. Each run is wrapped so failures are logged and the ladder continues.
run() {
    local TAG=$1; shift
    echo -e "\n=================================================================="
    echo "🚀 $TAG"
    echo "=================================================================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29555 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 80 --micro-bs 128 --accum 3 \
        --lr 5e-5 --wd 0.2 --optimizer adamw --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 32 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 1.0 \
        "$@" \
        --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing to next config"
    fi
}

# E — Euclidean baseline (LoRA-vs-full-FT + flat-vs-hyperbolic control). No SEL (forced).
run "bioscan_E_euclidean_r32_oc80" \
    --geometry euclidean --lambda-cl 1.0 --cl-mask none

# B0 — baseline = the Taxonomy-paper method (CL distance + SEL independent, lambda_cl=sel=1)
run "bioscan_B0_baseline_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C1 — SEL text cumulative (vs B0's independent)
run "bioscan_C1_seltext_cumulative_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C2 — SEL-only, cumulative (no contrastive)
run "bioscan_C2_selonly_cumulative_r32_oc80" \
    --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C3 — SEL-only, independent (no contrastive)
run "bioscan_C3_selonly_independent_r32_oc80" \
    --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C4 — CL-only (no SEL) = the hyperbolic lift without entailment
run "bioscan_C4_clonly_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 0.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C5 — SEL cumulative + CL angle
run "bioscan_C5_selcumulative_clangle_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C6 — SEL independent + CL angle
run "bioscan_C6_selindependent_clangle_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C7 — SEL independent + CL distance + false-negative mask
run "bioscan_C7_selindependent_cldistance_masksame_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C8 — SEL cumulative + CL distance + false-negative mask
run "bioscan_C8_selcumulative_cldistance_masksame_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C9 — SEL independent + CL angle + false-negative mask
run "bioscan_C9_selindependent_clangle_masksame_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C10 — SEL cumulative + CL angle + false-negative mask
run "bioscan_C10_selcumulative_clangle_masksame_r32_oc80" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

echo -e "\n✅ BIOSCAN C0-C10 + Euclidean ablation ladder complete."
