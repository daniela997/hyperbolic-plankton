#!/bin/bash
# BIOSCAN ablation ladder — the DEVELOPMENT testbed (complete-to-species taxonomy, ~2.5h/run
# vs ~20h on Planktonzilla). Develop here, then confirm only the WINNER + Euclidean baseline
# on Planktonzilla. 2 GPUs per run, sequential.
#
# ONE shared recipe (the Euclidean-campaign learnings) so every difference reflects the LOSS
# under test, not tuning: warmupcos / lr 1e-4 / r32 / seed 0 / bf16. NOT the old paper-faithful
# adam/onecycle/lr5e-5/r128 in run_ablations.sh.
#
# The 5 flag-flip configs (all already-implemented flags). Novel losses (random-truncation)
# are added later as separate runs once developed.
#   E   Euclidean baseline      --geometry euclidean (flat CLIP InfoNCE, no SEL) — the control
#   H1  hyperbolic CL-only      --lambda-sel 0 (lift, no entailment)
#   H2  hyperbolic CL+SEL       the Taxonomy-paper method (distance CL + independent SEL)
#   H3  + angle CL              --contrastive angle (radius-free CL, SEL-aligned)
#   H4  + false-negative mask   --cl-mask same (suppress same-class off-diagonal negatives)
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

# E — Euclidean baseline (the LoRA-vs-full-FT + flat-vs-hyperbolic control)
run "bioscan_E_euclidean_r32" \
    --geometry euclidean --lambda-cl 1.0

# H1 — hyperbolic CL-only (lift, no entailment)
run "bioscan_H1_clonly_r32" \
    --geometry hyperbolic --lambda-cl 1.0 --lambda-sel 0.0 \
    --contrastive distance

# H2 — hyperbolic CL + SEL (the Taxonomy-paper method)
run "bioscan_H2_clsel_r32" \
    --geometry hyperbolic --lambda-cl 1.0 --lambda-sel 1.0 \
    --contrastive distance --sel-text independent --cl-mask none

# H3 — + angle CL (radius-free contrastive)
run "bioscan_H3_clsel_angle_r32" \
    --geometry hyperbolic --lambda-cl 1.0 --lambda-sel 1.0 \
    --contrastive angle --sel-text independent --cl-mask none

# H4 — + false-negative mask (suppress same-class off-diagonal negatives)
run "bioscan_H4_clsel_masksame_r32" \
    --geometry hyperbolic --lambda-cl 1.0 --lambda-sel 1.0 \
    --contrastive distance --sel-text independent --cl-mask same

echo -e "\n✅ BIOSCAN ablation ladder complete."
