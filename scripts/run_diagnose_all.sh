#!/bin/bash
# Run diagnose_geometry.py on EVERY finished v3 BIOSCAN checkpoint (+ any sel-margin runs),
# full test_seen split, to fill docs/geometry-diagnostics.md with robust (non-*) numbers.
# Run when GPUs are free (it's eval-only, ~minutes/ckpt on GPU).
#
#   tmux new-session -d -s diag 'bash scripts/run_diagnose_all.sh 2>&1 | tee /tmp/diag_all.log'

cd /home/daniela/mine/hyperbolic-plankton
PY=/scratch/daniela/miniconda3/envs/dino_plankton/bin/python
CKDIR=/scratch/daniela/hyperbolic_plankton_ckpts

# one _final.pt per run-dir matching the v3 / sel-margin tags
for d in "$CKDIR"/bioscan_*_r64_v3__* "$CKDIR"/bioscan_*selmargin*_r64_v3__*; do
    [ -d "$d" ] || continue
    f=$(ls "$d"/*_final.pt 2>/dev/null | head -1)
    [ -z "$f" ] && f=$(ls "$d"/*_best.pt 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    PYTHONPATH=src "$PY" scripts/diagnose_geometry.py --ckpt "$f" --n 4878 \
        2>&1 | grep -vE "QuickGELU|warnings.warn|^[[:space:]]*$"
done
echo -e "\n✅ all diagnostics done — update docs/geometry-diagnostics.md"
