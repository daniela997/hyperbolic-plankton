#!/bin/bash
# BIOSCAN ablation ladder — the DEVELOPMENT testbed (complete-to-species, ~2.5h/run vs ~20h
# on Planktonzilla). Full C0-C10 grid + Euclidean baseline. Develop here; confirm only the
# WINNER + Euclidean baseline on Planktonzilla. 2 GPUs per run, sequential (~30h total).
#
# ONE shared recipe so every difference reflects the LOSS under test, not tuning:
# warmupcos / lr 1e-4 / r32 / seed 0 / bf16 (the Euclidean-campaign learnings). This SUPERSEDES
# the old run_ablations.sh recipe (adam/onecycle/lr5e-5/r128); the June C7/C9/C10 runs used
# that old recipe and are NOT comparable, so the whole grid re-runs fresh.
#
# The grid (B0 = the Taxonomy-paper method; C1-C10 vary one axis each):
#   axes: lambda_cl/sel (CL-only / SEL-only / both), contrastive (distance/angle),
#         sel-text (independent/cumulative), cl-mask (none/same = false-negative suppression)
#
# Read results: PYTHONPATH=src python scripts/final_eval.py --ckpt <dir>/<tag>_best.pt \
#   --dataset bioscan --backbone clip --lora --lora-r 32 --lora-visual-blocks 12 \
#   --lora-text-blocks 12 --geometry {euclidean|hyperbolic}

set -e
cd /home/daniela/mine/hyperbolic-plankton

run() {
    local TAG=$1; shift
    echo -e "\n=================================================================="
    echo "🚀 $TAG"
    echo "=================================================================="
    PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29555 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 3 \
        --lr 1e-4 --wd 0.2 --optimizer adamw --scheduler warmupcos --warmup-frac 0.1 \
        --lora-r 32 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 1.0 \
        "$@" \
        --tag "$TAG"
}

# E — Euclidean baseline (LoRA-vs-full-FT + flat-vs-hyperbolic control). No SEL (forced).
run "bioscan_E_euclidean_r32" \
    --geometry euclidean --lambda-cl 1.0 --cl-mask none

# B0 — baseline = the Taxonomy-paper method (CL distance + SEL independent, lambda_cl=sel=1)
run "bioscan_B0_baseline_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C1 — SEL text cumulative (vs B0's independent)
run "bioscan_C1_seltext_cumulative_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C2 — SEL-only, cumulative (no contrastive)
run "bioscan_C2_selonly_cumulative_r32" \
    --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C3 — SEL-only, independent (no contrastive)
run "bioscan_C3_selonly_independent_r32" \
    --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C4 — CL-only (no SEL) = the hyperbolic lift without entailment
run "bioscan_C4_clonly_r32" \
    --lambda-cl 1.0 --lambda-sel 0.0 --contrastive distance --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C5 — SEL cumulative + CL angle
run "bioscan_C5_selcumulative_clangle_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C6 — SEL independent + CL angle
run "bioscan_C6_selindependent_clangle_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C7 — SEL independent + CL distance + false-negative mask
run "bioscan_C7_selindependent_cldistance_masksame_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C8 — SEL cumulative + CL distance + false-negative mask
run "bioscan_C8_selcumulative_cldistance_masksame_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C9 — SEL independent + CL angle + false-negative mask
run "bioscan_C9_selindependent_clangle_masksame_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# C10 — SEL cumulative + CL angle + false-negative mask
run "bioscan_C10_selcumulative_clangle_masksame_r32" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

echo -e "\n✅ BIOSCAN C0-C10 + Euclidean ablation ladder complete."
