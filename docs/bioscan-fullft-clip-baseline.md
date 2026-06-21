# BIOSCAN full-FT CLIP baseline (the Planktonzilla-matched control)

The BIOSCAN analogue of the Planktonzilla full-fine-tuned "CLIP-style" model. This is the
**top-left corner of the 2×2** and the bar our LoRA / hyperbolic runs are measured against —
it answers "how close does parameter-efficient adaptation get to a full fine-tune?" on the
clean (complete-to-species) BIOSCAN testbed, the same way the released Planktonzilla CLIP
weights answer it on Planktonzilla.

|            | Euclidean (flat InfoNCE)            | Hyperbolic (Lorentz + SEL) |
|------------|-------------------------------------|----------------------------|
| **Full FT**| **this baseline** (`train_euclidean_ft.py`) | n/a                |
| **LoRA**   | E0 / euclidean ablations            | B0 + the ladder            |

## What it is (and is NOT)

- **Full fine-tune**, not LoRA: the ENTIRE CLIP ViT-B/16 backbone is unfrozen and trained
  end-to-end (`unfreeze_backbone(model)`), ~154M trainable params (vs ~4M for LoRA).
- **Flat cosine InfoNCE** (open_clip `ClipLoss` form, MERU-style), image ↔ the cumulative
  taxonomic lineage string. NO hyperbolic lift, NO SEL, NO projection-only freezing.
- Starts from **pretrained CLIP** (OpenAI ViT-B/16), not from scratch.
- Uses the **same data splits, the same `forward_loss`, and the same eval harness** as the
  LoRA/hyperbolic runs (`train_euclidean_ft.py` imports them from `train_lora.py`), so the
  full-FT-vs-LoRA comparison differs ONLY in adaptation — no confound.

## Recipe (Planktonzilla CLIP, adapted to 2×A5000)

Planktonzilla's reported full-FT CLIP recipe (paper §; `scripts/train_clip.sh`):
lr 1e-4, wd 0.2, AdamW, 100 epochs, global batch 16,384 on 64×H100. We keep their
**lr / wd / optimizer** and adapt the **batch / epochs** to our compute:

| knob        | Planktonzilla | this baseline (BIOSCAN) | why |
|-------------|---------------|-------------------------|-----|
| backbone    | CLIP ViT-B/16 | CLIP ViT-B/16           | same |
| lr          | 1e-4          | 1e-4                    | matched |
| weight decay| 0.2           | 0.2                     | matched (full-FT tolerates it; LoRA did not) |
| optimizer   | AdamW         | AdamW                   | matched |
| scheduler   | —             | onecycle (pct 0.3)      | our short-run default |
| effective batch | 16,384    | 768 (128×3×2 GPU)       | compute limit |
| epochs      | 100           | 50                      | BIOSCAN is 36k imgs; 50ep ≈ converged |
| precision   | amp           | fp16 + GradScaler       | A5000; NOT bf16 (bf16 caused hyperbolic NaNs) |
| seed        | —             | 0 (torch/numpy/random)  | reproducible |

So this is the Planktonzilla recipe **adapted**, not a literal reproduction — the batch and
epochs differ by necessity. The lr/wd/optimizer (the parts that define the optimisation) are
matched.

## How to run

```bash
cd /home/daniela/mine/hyperbolic-plankton
PYTHONPATH=src /scratch/daniela/miniconda3/envs/dino_plankton/bin/torchrun \
  --nproc_per_node=2 --master_port=29557 scripts/train_euclidean_ft.py \
  --dataset bioscan --backbone clip --micro-bs 128 --accum 3 --epochs 50 \
  --lr 1e-4 --wd 0.2 --optimizer adamw --scheduler onecycle \
  --seed 0 --compile --eval-epochs 1 \
  --wandb-project hyperbolic-plankton-euclidean \
  --tag bioscan_FT_euclidean
```

Or use the launcher (same thing): `bash scripts/run_euclidean_ft.sh`. Run it in tmux so it
survives disconnects:
```bash
tmux new-session -d -s ft 'bash scripts/run_euclidean_ft.sh 2>&1 | tee /tmp/bioscan_ft.log'
```

Full-FT is heavier than LoRA (all 154M params train + their optimizer state), so it uses more
VRAM and runs slower per step than the LoRA ladder. If it OOMs at micro-bs 128, drop to 64
and raise --accum to 6 (keeps effective batch 768).

## How to evaluate (the headline numbers)

```bash
PYTHONPATH=src python scripts/final_eval.py \
  --ckpt /scratch/daniela/hyperbolic_plankton_ckpts/bioscan_FT_euclidean__<runid>/bioscan_FT_euclidean_best.pt \
  --dataset bioscan --backbone clip --geometry euclidean
```
NOTE: **no `--lora` flag** — this is a full fine-tune, so the checkpoint is the whole model.
(Adapter-only saving in train_lora does not apply here; this script saves the full
state_dict.) Per-rank seen/unseen macro-F1, same metric as everything else.

## Reading the result

- **FT vs LoRA (same Euclidean loss)** = the cost of parameter efficiency. If LoRA lands
  close to FT, the efficiency thesis holds on BIOSCAN.
- **FT (Euclidean) vs B0 (hyperbolic LoRA)** is NOT a clean single-variable comparison (it
  changes both adaptation and geometry) — use the LoRA-Euclidean (E0) runs for the geometry
  delta. FT is the absolute reference, not the geometry control.
