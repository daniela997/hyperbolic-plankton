"""Push the Euclidean-LoRA E0c checkpoint (the best LoRA-vs-full-FT seen result) to a PRIVATE
HF model repo, with a README (metrics + load instructions) and a runnable load snippet.

We do NOT use standard PEFT format: apply_lora first runs replace_mha_with_plain (open_clip's
fused MultiheadAttention -> split q/k/v linears), so the adapter targets modules that only
exist AFTER that surgery — a plain PeftModel.from_pretrained onto stock open_clip would find
no q_proj to attach to. The checkpoint also carries 8 non-LoRA trainable tensors (curv,
alphas, logit_scale, re-init final LN) that PEFT format would drop. So we ship the raw
trainable state_dict + the exact load path (apply_lora + load_state_dict, strict=False).

  PYTHONPATH=src python scripts/push_to_hf.py --repo danielaivanova/<name> [--public]
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, create_repo, upload_file

CKPT = ("/scratch/daniela/hyperbolic_plankton_ckpts/"
        "planktonzilla_E0c_euclidean_lora_allblocks_r32_noproj_20ep_lr1e4_bf16__0o1s0wpx/"
        "planktonzilla_E0c_euclidean_lora_allblocks_r32_noproj_20ep_lr1e4_bf16_final.pt")

README = """\
---
license: mit
tags:
  - plankton
  - taxonomy
  - clip
  - lora
  - biology
library_name: open_clip
---

# Euclidean-LoRA CLIP for Plankton Taxonomy (E0c)

Parameter-efficient (LoRA) adaptation of OpenAI CLIP ViT-B/16 for hierarchical plankton
taxonomy on **Planktonzilla**, trained with a flat (Euclidean) cosine-InfoNCE contrastive
objective over the cumulative taxonomic lineage string. This is the **Euclidean-LoRA control**
for a hyperbolic-adaptation study: it isolates *LoRA vs full fine-tuning* (same flat-space
objective as the Planktonzilla full-FT CLIP), before adding hyperbolic geometry.

- Backbone: `ViT-B-16` (OpenAI), **frozen**; LoRA on attention q/k/v/proj of all 12+12 blocks.
- LoRA: r=32, α=32, rsLoRA. No projection head (architecture-identical to the Planktonzilla
  full-FT CLIP). ~3.9M trainable params (vs full fine-tune).
- Trained: 20 epochs, lr 1e-4 warmup+cos, bf16, 2×A5000.

## Results (full test split, per-rank macro-F1; full-FT Planktonzilla CLIP for reference)

| rank | this LoRA | full-FT |
|---|---|---|
| kingdom | 0.911 | 0.955 |
| phylum  | 0.869 | 0.935 |
| class   | 0.774 | 0.889 |
| order   | 0.748 | 0.873 |
| family  | 0.699 | 0.857 |
| genus   | 0.675 | 0.832 |
| species | 0.642 | 0.818 |

## Loading

The checkpoint is the **trainable state only** (LoRA adapters + re-init final LN + geometry
scalars). It is NOT a stock PEFT adapter: `apply_lora` splits open_clip's fused attention
into q/k/v linears first, so it must be loaded through the matching code path:

```python
import torch
from hyperbolic_plankton.model import HyperbolicCLIP
from hyperbolic_plankton.lora import apply_lora

model = HyperbolicCLIP(backbone="clip", use_proj=False)   # no projector (Euclidean E0c)
model = apply_lora(model, r=32, alpha=32,
                   adapt_visual_blocks=12, adapt_text_blocks=12)  # splits MHA, adds LoRA
sd = torch.load("E0c_final.pt", map_location="cpu")
model.load_state_dict(sd["model"], strict=False)          # frozen backbone filled from CLIP
model.eval()
```

Eval (cosine-argmax over present classes, per-rank macro-F1) is in
`scripts/final_eval.py --geometry euclidean` in the source repo.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. danielaivanova/bioclip-plankton-lora")
    ap.add_argument("--public", action="store_true", help="default is PRIVATE")
    args = ap.parse_args()

    assert os.path.exists(CKPT), f"checkpoint not found: {CKPT}"
    api = HfApi()
    create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    print(f"repo: {args.repo}  (private={not args.public})")

    upload_file(path_or_fileobj=CKPT, path_in_repo="E0c_final.pt",
                repo_id=args.repo, repo_type="model")
    print("uploaded E0c_final.pt")

    upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                repo_id=args.repo, repo_type="model")
    print("uploaded README.md")
    print(f"\nDone -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
