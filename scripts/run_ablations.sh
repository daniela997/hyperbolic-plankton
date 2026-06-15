#!/bin/bash

# Exit immediately if any training run fails (e.g., out of memory, syntax error)
# Remove this line if you want subsequent runs to continue even if one fails.
set -e

# Navigate to the correct working directory
cd /home/daniela/mine/hyperbolic-plankton

# Define a function to handle the common launch prefix and arguments
run_ablation() {
    local TAG=$1
    shift # Shift the arguments so $@ only contains the specific ablation flags

    echo -e "\n================================================================="
    echo "🚀 Starting run: $TAG"
    echo "================================================================="

    PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29555 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 3 \
        --lr 5e-5 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --lora-r 128 --eval-every 200 \
        "$@" \
        --tag "$TAG"
}

# # ==============================================================================
# # B0 — baseline (CL distance + SEL independent, λ_cl=λ_sel=1)
# # ==============================================================================
# run_ablation "bioscan_B0_baseline" \
#     --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
#     --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C1 — SEL text cumulative
# # ==============================================================================
# run_ablation "bioscan_C1_seltext_cumulative" \
#     --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
#     --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C2 — SEL-only, cumulative (no contrastive)
# # ==============================================================================
# run_ablation "bioscan_C2_selonly_cumulative" \
#     --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
#     --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C3 — SEL-only, independent (no contrastive)
# # ==============================================================================
# run_ablation "bioscan_C3_selonly_independent" \
#     --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
#     --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C4 — CL-only (no SEL)
# # ==============================================================================
# run_ablation "bioscan_C4_clonly" \
#     --lambda-cl 1.0 --lambda-sel 0.0 --contrastive distance --cl-mask none \
#     --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C5 — SEL cumulative + CL angle
# # ==============================================================================
# run_ablation "bioscan_C5_selcumulative_clangle" \
#     --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
#     --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

# # ==============================================================================
# # C6 — SEL independent + CL angle
# # ==============================================================================
# run_ablation "bioscan_C6_selindependent_clangle" \
#     --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
#     --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0


# ==============================================================================
# C7 — SEL independent + CL distance masked
# ==============================================================================
run_ablation "bioscan_C7_selindependent_cldistance_masksame" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0


# ==============================================================================
# C8 — SEL cumulative + CL distance masked
# ==============================================================================
run_ablation "bioscan_C8_selcumulative_cldistance_masksame" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0


# ==============================================================================
# C9 — SEL independent + CL angle masked
# ==============================================================================
run_ablation "bioscan_C9_selindependent_clangle_masksame" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0


# ==============================================================================
# C10 — SEL cumulative + CL angle masked
# ==============================================================================
run_ablation "bioscan_C10_selcumulative_clangle_masksame" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same \
    --sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0

echo -e "\n✅ All ablation runs completed successfully!"