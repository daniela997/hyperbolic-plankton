"""Calibration: eval the OFFICIAL Planktonzilla-trained CLIP on OUR seen/unseen splits.

Loads project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt (the paper's full-FT
"CLIP-style" ViT-B/16, gated on HF — accept the terms + `huggingface-cli login` first) and
runs it through OUR eval (cosine-argmax zero-shot, present-classes, per-rank macro-F1 =
taxonomic_macro_f1). This is the true full-FT calibration bar — every protocol confound
removed (no reliance on the paper's own subsample/protocol).

Off-the-shelf zero-shot: the weights ARE the model (no projector/LoRA/lift). We wrap the
open_clip model in a tiny shim exposing the surface run_unseen_eval_cosine needs, so the
eval path is byte-identical to ours.

Two eval sets, both reported (they answer different questions):
  --mode full       FULL seen(test)/unseen splits, present-classes -> compare to your LoRA
                    runs' end-of-training run_final_eval (the headline bar). Slow (113k unseen).
  --mode subsample  the SAME stratified subsample (--eval-cap/class) + seen(VAL) split the
                    training monitor uses -> compare to the wandb eval/unseen/* curves. Fast.
  --mode both       (default) run both.

  PYTHONPATH=src python scripts/eval_planktonzilla_clip.py --mode both
"""

from __future__ import annotations

import argparse
import types

import open_clip
import torch
from datasets import load_from_disk

from hyperbolic_plankton.eval import flatten_metrics, run_unseen_eval_cosine

# reuse the SAME split builders as the LoRA runs so sets/classes are identical:
#   _planktonzilla_sets  -> full test_seen/unseen (final_eval.py)
#   _build_eval_sets     -> stratified subsample, seen-VAL (the periodic monitor)
from final_eval import _planktonzilla_sets  # noqa: E402
from train_lora import CACHE, RANKS, _build_eval_sets  # noqa: E402

HF_ID = "hf-hub:project-oceania/CLIP-ViT-B-16.openai-pt.planktonzilla-pt"


class _OpenClipShim:
    """Minimal surface run_unseen_eval_cosine expects: .clip / .preprocess / .tokenizer /
    .device. Lets us drive their bare open_clip model through our exact eval path."""

    def __init__(self, model, preprocess, tokenizer, device):
        self.clip = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self._device = device

    @property
    def device(self):
        return self._device


def _report(shim, name, ds, classes, num_workers):
    print(f"\n=== {name}: {len(ds):,} images, {len(classes)} classes ===")
    res = run_unseen_eval_cosine(shim, ds, classes, num_workers=num_workers)
    m = flatten_metrics(res["metrics"], prefix=name)
    for r in RANKS:
        k = f"{name}/{r}_f1"
        if k in m:
            print(f"  {r:8s} macro-F1 = {m[k]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["full", "subsample", "both"])
    ap.add_argument("--eval-cap", type=int, default=50,
                    help="subsample mode: rows/class (match the training run's --eval-cap)")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {HF_ID} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(HF_ID)
    tokenizer = open_clip.get_tokenizer(HF_ID)
    model.to(device).eval()
    shim = _OpenClipShim(model, preprocess, tokenizer, device)

    if args.mode in ("full", "both"):
        print("\n########## FULL splits (test_seen / unseen) — headline bar ##########")
        (seen_ds, seen_cls), (unseen_ds, unseen_cls), _ = _planktonzilla_sets()
        _report(shim, "seen", seen_ds, seen_cls, args.num_workers)
        _report(shim, "unseen", unseen_ds, unseen_cls, args.num_workers)

    if args.mode in ("subsample", "both"):
        print("\n########## SUBSAMPLE (seen-VAL + stratified) — matches wandb monitor ##########")
        cache = load_from_disk(CACHE)
        sets = _build_eval_sets(cache, types.SimpleNamespace(eval_cap=args.eval_cap))
        (seen_ds, seen_cls) = sets["sets"]["seen"]
        (unseen_ds, unseen_cls) = sets["sets"]["unseen"]
        _report(shim, "seen", seen_ds, seen_cls, args.num_workers)
        _report(shim, "unseen", unseen_ds, unseen_cls, args.num_workers)


if __name__ == "__main__":
    main()
