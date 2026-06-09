# BIOSCAN ablation commands

> Copy-paste runs, each ONE variable from the B0 baseline (see `experiment-plan.md` §2).
> Run on BIOSCAN (clean complete-taxonomy control) first; promote winners to planktonzilla.
> After each run: `final_eval.py` for the numbers, record verdict in `build-log.md`.
>
> Common to all: `--dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 3
> --lr 5e-5 --wd 1e-4 --optimizer adam --scheduler onecycle --lora-r 128 --eval-every 200`.
> BIOSCAN 50 epochs ≈ 2,360 steps (startup prints the exact `total_steps`).

Launch prefix (all runs):
```bash
cd /home/daniela/mine/hyperbolic-plankton
PYTHONPATH=src torchrun --nproc_per_node=2 --master_port=29555 scripts/train_lora.py \
  --dataset bioscan --backbone clip --epochs 50 --micro-bs 128 --accum 3 \
  --lr 5e-5 --wd 1e-4 --optimizer adam --scheduler onecycle \
  --lora-r 128 --eval-every 200 \
  <ABLATION FLAGS> --tag <TAG>
```

---

## B0 — baseline (CL distance + SEL independent, λ_cl=λ_sel=1)
*(you are running this)*
```
--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
--sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
--tag bioscan_B0_baseline
```

---

## C1 — SEL text cumulative
B0 but SEL uses the cumulative `full` string for both SEL terms (CL already uses full).
Tests whether the paper's independent-`T_r` choice helps us, or cumulative (shared-prefix
tree) is better in the frozen+projector regime.
```
--lambda-cl 1.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
--sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
--tag bioscan_C1_seltext_cumulative
```

## C2 — SEL-only, cumulative (no contrastive)
Drop CL entirely (`--lambda-cl 0`), SEL cumulative. The paper's SEL-only rows are its
strongest unseen results; this is the cumulative-text version of that.
```
--lambda-cl 0.0 --lambda-sel 1.0 --contrastive distance --cl-mask none \
--sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
--tag bioscan_C2_selonly_cumulative
```

## C3 — CL-only (no SEL)
Drop SEL (`--lambda-sel 0`), CL distance. Isolates what the contrastive term alone learns —
the pure "CLIP-style" point, comparable to Planktonzilla's CLIP baseline.
```
--lambda-cl 1.0 --lambda-sel 0.0 --contrastive distance --cl-mask none \
--sel-text independent --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
--tag bioscan_C3_clonly
```

## C4 — SEL cumulative + CL angle
B0 with SEL cumulative AND the angle-based contrastive (ATMG). Tests the combination:
angle-CL (radius-free, SEL-aligned) with cumulative SEL text.
```
--lambda-cl 1.0 --lambda-sel 1.0 --contrastive angle --cl-mask none \
--sel-text cumulative --sel-tau 1.0 --sel-leak 0.0 --sel-uncertainty 0.0 \
--tag bioscan_C4_selcumulative_clangle
```

---

## After these
Decide next runs from the results — the natural follow-ups (cl-mask same, UNCHA
leak/tau/lam_u) are A1/A3/A4 in `experiment-plan.md`, layered onto whichever
CL/SEL/text combination wins here.

## Reading each run
```bash
# final numbers (full test/unseen splits)
PYTHONPATH=src python scripts/final_eval.py \
  --ckpt /scratch/daniela/hyperbolic_plankton_ckpts/<tag>_it<N>.pt \
  --dataset bioscan --backbone clip --lora
```
Compare per-rank seen+unseen macro-F1 vs B0, plus the periodic geometry (curv, per-rank
radius/aperture, entail_ok) from wandb.
