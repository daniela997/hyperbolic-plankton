#!/bin/bash
# BIOSCAN ablation ladder — V4: cache-accum CL (true 768-way InfoNCE) + the RINCE sub-grid.
#
# Recipe LOCKED from the v4 cache-accum sweep (zojuv2lz, best trial seen 0.640 @ 50ep, beats the
# v3 80ep B0 0.628): adam / onecycle / lr 2.5e-4 (sweep best 2.32e-4, rounded) / wd 1e-4 / r64 /
# 12+12 blocks / 50 epochs /
# seed 0 / fp16 / --cache-accum-cl. micro-bs 64 / accum 6 (eff batch 768; micro-bs 128 OOMs with
# cache-accum). vs v3 the only recipe change is wd 1e-3 -> 1e-4 (cache-accum wants less decay).
#
# Eval every 5 epochs (--eval-epochs 5.0) to save time; _best.pt = best of those evals, _final.pt
# always saved. E (Euclidean) skipped (v3 E is the baseline). 17 configs.
#
# Two blocks:
#  (1) C0-C10 — the loss ablation, now on the cache-accum (768-neg) objective.
#  (2) RINCE sub-grid — ranked-positive CL (graded by taxonomic depth, geometry-agnostic
#      alternative to SEL cones): ranked x {CL-only, +SEL-indep, +SEL-cumul} x {distance,angle sim}.
#
# Read results: PYTHONPATH=src python scripts/final_eval.py --ckpt <dir>/<tag>_best.pt \
#   --dataset bioscan --backbone clip --lora --lora-r 64 --lora-visual-blocks 12 \
#   --lora-text-blocks 12 --geometry {euclidean|hyperbolic}
#
#   tmux new-session -d -s ablate4 'bash scripts/run_ablations_bioscan_v4.sh 2>&1 | tee /tmp/ablate_v4.log'

cd /home/daniela/mine/hyperbolic-plankton
PY=/scratch/daniela/miniconda3/envs/dino_plankton/bin/python
TORCHRUN=/scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun

CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

# NO `set -e`: one config crashing (e.g. a NaN) must NOT abort the whole ladder.
run() {
    local TAG=$1; shift
    # idempotent resume: skip if a completed _final.pt already exists for this tag
    if ls "$CKDIR/${TAG}__"*/"${TAG}_final.pt" >/dev/null 2>&1; then
        echo "⏭️  $TAG already done (final.pt exists) — skipping"
        return
    fi
    echo -e "\n=================================================================="
    echo "🚀 $TAG"
    echo "=================================================================="
    if ! PYTHONPATH=src "$TORCHRUN" --nproc_per_node=2 --master_port=29557 scripts/train_lora.py \
        --dataset bioscan --backbone clip --epochs 50 --micro-bs 64 --accum 6 --cache-accum-cl \
        --lr 2.5e-4 --wd 1e-4 --optimizer adam --scheduler onecycle \
        --onecycle-pct-start 0.3 --onecycle-min-lr 1e-6 \
        --lora-r 64 --lora-visual-blocks 12 --lora-text-blocks 12 \
        --seed 0 --compile --eval-epochs 5.0 \
        "$@" \
        --tag "$TAG"; then
        echo "⚠️  $TAG FAILED (exit $?) — continuing to next config"
    fi
}

# ============================ (1) C0-C10 (loss ablation) ============================
# E (Euclidean) SKIPPED — v3's E (seen 0.765) is the baseline; flat CLIP doesn't use the
# cache-accum hierarchy, so re-running it at v4 adds little. Re-add if a v4 E control is wanted.

# RESUME (independent-text fix 4c3528d): re-run the configs that used INDEPENDENT SEL text
# (B0, C3 — trained with the old 'Rank:' prefix, now invalid). E/C1/C2 already completed and are
# VALID (E=no SEL; C1/C2=cumulative text, unaffected by the fix) -> commented out, kept on disk.
run "bioscan_B0_baseline_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text independent

# C1 done + valid (cumulative): skip
# run "bioscan_C1_seltext_cumulative_r64_v4" \
#     --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text cumulative

# C2 done + valid (cumulative): skip
# run "bioscan_C2_selonly_cumulative_r64_v4" \
#     --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text cumulative

run "bioscan_C3_selonly_independent_r64_v4" \
    --lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none --sel-text independent

run "bioscan_C4_clonly_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 0.0 --contrastive distance --cl-mask none --sel-text independent

run "bioscan_C5_selcumulative_clangle_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none --sel-text cumulative

run "bioscan_C6_selindependent_clangle_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none --sel-text independent

run "bioscan_C7_selindependent_cldistance_masksame_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same --sel-text independent

run "bioscan_C8_selcumulative_cldistance_masksame_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask same --sel-text cumulative

run "bioscan_C9_selindependent_clangle_masksame_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same --sel-text independent

run "bioscan_C10_selcumulative_clangle_masksame_r64_v4" \
    --lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask same --sel-text cumulative

# ============================ (2) RINCE sub-grid ============================
# ranked-CL graded by taxonomic depth. --cl-mask irrelevant (ranking subsumes it). sim x SEL grid.
for SIM in distance angle; do
    # R*-clonly: ranked CL alone (no SEL) — does graded-positive CL beat plain CL/cl-mask?
    run "bioscan_R_ranked_${SIM}_clonly_r64_v4" \
        --contrastive ranked --rince-sim $SIM --lambda-cl 1.0 --lambda-sel 0.0 --sel-text independent

    # R*-selindep: ranked CL + SEL independent — does the entailment hierarchy add to RINCE?
    run "bioscan_R_ranked_${SIM}_selindep_r64_v4" \
        --contrastive ranked --rince-sim $SIM --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent

    # R*-selcumul: ranked CL + SEL cumulative
    run "bioscan_R_ranked_${SIM}_selcumul_r64_v4" \
        --contrastive ranked --rince-sim $SIM --lambda-cl 1.0 --lambda-sel 1.0 --sel-text cumulative
done

# ============================ (3) LRCL sub-grid ============================
# Level-Restricted CL (Tao 2026, "Beyond Flat Labels"): per-level unique-label InfoNCE, the
# PARTITIONING alternative to RINCE's grading. With cache-accum (LRCL is symmetric -> T->I needs
# the full image negative set). lrcl-ranks: species = generic single-level baseline; all = multirank.

# L1: generic single-level (unique-label + group-balanced) InfoNCE baseline (CL-only)
run "bioscan_L_lrcl_species_r64_v4" \
    --contrastive level-restricted --lrcl-ranks species --lambda-cl 1.0 --lambda-sel 0.0

# L2: full multi-rank LRCL (the paper's method, CL-only)
run "bioscan_L_lrcl_all_r64_v4" \
    --contrastive level-restricted --lrcl-ranks all --lambda-cl 1.0 --lambda-sel 0.0

# L3: full LRCL + SEL independent (does the entailment hierarchy add anything on top of LRCL?)
run "bioscan_L_lrcl_all_selindep_r64_v4" \
    --contrastive level-restricted --lrcl-ranks all --lambda-cl 1.0 --lambda-sel 1.0 --sel-text independent

echo -e "\n✅ BIOSCAN v4 ladder (C0-C10 + Euclidean + RINCE + LRCL sub-grids) complete."
