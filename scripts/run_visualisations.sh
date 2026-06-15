#!/bin/bash

# Exit immediately if a command fails
set -e

# Navigate to the correct working directory
cd /home/daniela/mine/hyperbolic-plankton

# Ensure the output directory exists before trying to save files there
mkdir -p /scratch/daniela/horopca/

# Define the exact tags used in the updated training script
TAGS=(
    # "bioscan_B0_baseline"
    # "bioscan_C1_seltext_cumulative"
    # "bioscan_C2_selonly_cumulative"
    # "bioscan_C3_selonly_independent"
    # "bioscan_C4_clonly"
    # "bioscan_C5_selcumulative_clangle"
    # "bioscan_C6_selindependent_clangle"
    "bioscan_C7_selindependent_cldistance_masksame"
    "bioscan_C8_selcumulative_cldistance_masksame"
    "bioscan_C9_selindependent_clangle_masksame"
    "bioscan_C10_selcumulative_clangle_masksame"
)

echo -e "\n================================================================="
echo "Starting HoroPCA Visualizations"
echo "================================================================="

for TAG in "${TAGS[@]}"; do
    # Construct the file paths based on the current tag
    CKPT_PATH="/scratch/daniela/hyperbolic_plankton_ckpts/${TAG}_final.pt"
    OUT_PATH="/scratch/daniela/horopca/clip_${TAG}_final.png"

    # Check if the checkpoint exists before running
    if [ ! -f "$CKPT_PATH" ]; then
        echo "Warning: Checkpoint not found -> $CKPT_PATH"
        echo "Skipping $TAG..."
        echo "-----------------------------------------------------------------"
        continue
    fi

    # Determine the correct --sel-text argument based on the tag name
    if [[ "$TAG" == *"cumulative"* ]]; then
        SEL_TEXT="cumulative"
    else
        SEL_TEXT="independent"
    fi

    echo "Generating visualization for: $TAG (sel-text: $SEL_TEXT)"
    
    PYTHONPATH=src python scripts/visualize_horopca.py \
        --ckpt "$CKPT_PATH" \
        --dataset bioscan \
        --backbone clip \
        --lora \
        --lora-r 128 \
        --n 250 \
        --sel-text "$SEL_TEXT" \
        --out "$OUT_PATH"
        
    echo "Saved to $OUT_PATH"
    echo "-----------------------------------------------------------------"
done

echo -e "\n🎉 All visualizations complete!"