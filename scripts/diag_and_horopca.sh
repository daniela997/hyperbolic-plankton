#!/bin/bash
# Pull diagnostics + HoroPCA for a set of run tags. Idempotent: skips a run if it has no _final.pt yet,
# and (unless --force) skips diag/horopca outputs that already exist. Waits for a GPU with headroom.
# Also pulls seen/unseen species F1 from wandb into the cache. Safe to run alongside training (uses one
# GPU, batched inference). Default set = the currently-pending runs (centroid1 + tau-matched).
#
#   bash scripts/diag_and_horopca.sh                 # default pending set
#   bash scripts/diag_and_horopca.sh TAG1 TAG2 ...   # explicit tags
#   FORCE=1 bash scripts/diag_and_horopca.sh TAG     # redo even if outputs exist
#
# Env: GPU=<id> to pin (default: auto-pick the GPU with most free mem).

cd /home/daniela/mine/hyperbolic-plankton
PY=/scratch/daniela/miniconda3/envs/dino_plankton/bin/python
CK=/scratch/daniela/hyperbolic_plankton_ckpts
OUT=notebooks/v4_analysis

DEFAULT_TAGS=(
  bioscan_H_hybrid_angle_centroid1_r64_v4
  bioscan_R_ranked_distance_tau1_r64_v4
  bioscan_H_hybrid_tau1_r64_v4
  bioscan_H_hybrid_tau1_radial05_r64_v4
)
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=("${DEFAULT_TAGS[@]}")

pick_gpu() {
  if [ -n "$GPU" ]; then echo "$GPU"; return; fi
  # GPU with the most free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' '
}

for TAG in "${TAGS[@]}"; do
  fp=$(ls "$CK/${TAG}__"*/"${TAG}_final.pt" 2>/dev/null | head -1)
  if [ -z "$fp" ]; then echo "⏳ $TAG — no _final.pt yet, skipping"; continue; fi
  G=$(pick_gpu)
  diag="$OUT/diag_${TAG}.txt"; horo="$OUT/horopca_${TAG}.png"

  # --- diagnostics ---
  if [ -n "$FORCE" ] || [ ! -s "$diag" ] || ! grep -q "RAY-collapse" "$diag" 2>/dev/null; then
    echo "🔬 diag $TAG (GPU $G)"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" scripts/diagnose_geometry.py --ckpt "$fp" 2>&1 \
      | grep -vE "Warning|warn|UserWarning|will be removed" > "$diag"
    grep -E "DISTANCE|ANGLE/ATMG|CONE-ENERGY|RAY-collapse|SATURATED|NN-sep|species " "$diag" | grep -vE "n_neg|loss"
  else
    echo "⏭️  diag $TAG exists (has ray metric) — skipping"
  fi

  # --- HoroPCA (n=150 recipe). Do NOT pass --sel-text: let visualize_horopca auto-detect it from the
  # ckpt (it correctly falls back to cumulative when lambda_sel==0, so CL-only runs don't get the
  # spurious "species (classifier)" protos — those belong only to independent-SEL runs). ---
  if [ -n "$FORCE" ] || [ ! -s "$horo" ]; then
    echo "🖼️  horopca $TAG (GPU $G)"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=$G PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$PY" scripts/visualize_horopca.py --ckpt "$fp" --dataset bioscan --backbone clip --lora --lora-r 64 \
      --split test_seen --n 150 --out "$horo" 2>&1 \
      | grep -vE "Warning|warn|UserWarning|will be removed|FutureWarning" | tail -2
  else
    echo "⏭️  horopca $TAG exists — skipping"
  fi
done

# --- pull F1 from wandb into the cache for all requested tags ---
echo "📊 pulling F1 from wandb..."
PYTHONPATH=src "$PY" - "${TAGS[@]}" <<'PYEOF' 2>&1 | grep -vE "wandb: |Warning|warn"
import sys, json, os
import wandb
os.chdir("notebooks/v4_analysis")
api=wandb.Api(); cache=json.load(open("wandb_f1_cache.json"))
for name in sys.argv[1:]:
    rs=list(api.runs("uofg/hyperbolic-plankton",filters={"display_name":name}))
    if not rs: print(f"  {name}: not in wandb"); continue
    raw=rs[0].summary._json_dict; sd=json.loads(raw) if isinstance(raw,str) else dict(raw)
    out={}
    for rk in ["order","family","genus","species"]:
        for sp in ["seen","unseen"]:
            for k in (f"eval/{sp}/{rk}_f1", f"{sp}_{rk}_f1", f"test/{sp}/{rk}_f1"):
                if k in sd: out[f"{sp}_{rk}_f1"]=sd[k]; break
    if out:
        cache[name]=out
        print(f"  {name}: seenSp={out.get('seen_species_f1',float('nan')):.3f} unsSp={out.get('unseen_species_f1',float('nan')):.3f}")
    else:
        print(f"  {name}: no F1 keys yet")
json.dump(cache,open("wandb_f1_cache.json","w"),indent=2)
PYEOF

echo "✅ diag+horopca done for: ${TAGS[*]}"
