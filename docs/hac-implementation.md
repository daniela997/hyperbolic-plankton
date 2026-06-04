# HAC Implementation Notes (`/home/daniela/other/HAC`)

> Concrete wiring from the HAC reference code — the exact freeze/PEFT/projection
> recipe we mirror. Resolves several things the paper left implicit. Source files
> cited inline. `[code]` = read directly from the repo.

---

## TL;DR of the reference architecture

HAC = a **frozen MERU/HyCoCLIP backbone** + **trainable hyperbolic projection** +
**PEFT (LoRA or adapters) on selected backbone blocks**. The model class is
`AdaptedCLIP` in [`hac/models.py`](../../other/HAC/hac/models.py). The freeze/PEFT
wiring lives in [`scripts/train.py`](../../other/HAC/scripts/train.py) (~L211–294),
not in the model. This separation (model = geometry, script = what's trainable) is
worth copying.

Our `mine/hyperbolic/model.py` already mirrors `AdaptedCLIP`'s **geometry** (frozen
backbone, `visual_proj`/`textual_proj`, `visual_alpha`/`textual_alpha`, learnable
`curv`, `exp_map0`, the clamps). What it lacks is the **PEFT wiring** + **final-LN
unfreeze**, which HAC does in the training script.

---

## The exact freeze → PEFT → unfreeze sequence  `[code]`

From `scripts/train.py` L211–294, in this order:

1. **Freeze the whole backbone** — both encoders:
   ```python
   for p in model.visual.parameters():  p.requires_grad = False
   for p in model.textual.parameters(): p.requires_grad = False
   ```
2. **Apply PEFT** (if configured): replace MHA with a plain MHA
   (`replace_mha_with_plain`) so attention q/k/v/o become addressable submodules,
   then `model.visual = get_peft_model(model.visual, lora_config)` (PEFT/LoRA), or
   `get_adapted_encoder(...)` (adapters lib). LoRA params are trainable by default;
   backbone stays frozen.
3. **Unfreeze the final LayerNorm** of each encoder (only if `init_final_ln=True`):
   ```python
   for p in model.visual.norm.parameters():     p.requires_grad = True
   for p in model.textual.ln_final.parameters(): p.requires_grad = True
   ```
4. The **projection heads** (`visual_proj`, `textual_proj`) and the MERU scalars
   (`curv`, `visual_alpha`, `textual_alpha`, `logit_scale`) are **module-level
   params of `AdaptedCLIP`, not inside the frozen encoders**, so they are trainable
   by default — never frozen in step 1.

> **Net trainable set:** LoRA adapters + final LN (×2 encoders) + projection heads
> (×2) + {curv, visual_alpha, textual_alpha, logit_scale}. Everything else frozen.

This is exactly the "projector-only first, then add LoRA" plan — projector-only =
skip step 2.

---

## Projection / final-LN re-init  `[code]`

`AdaptedCLIP.init_model` ([models.py](../../other/HAC/hac/models.py) L592–647):
- Loads a CLIP/MERU checkpoint into the backbone.
- If `init_proj=True`: **re-initializes** `visual_proj`/`textual_proj` CLIP-style
  (`std = width**-0.5`) — the projection into hyperbolic space is learned fresh, not
  inherited.
- If `init_final_ln=True`: **replaces** `visual.norm` and `textual.ln_final` with
  fresh `nn.LayerNorm`. Combined with step 3 above, the final LN is trained from
  scratch. The paper's rationale: the final LN isn't bypassed by a skip connection,
  so leaving it frozen imposes a fixed gate on the encoder output.

Both default to `True` in the ViT-B LoRA config.

---

## The hyperbolic lift  `[code]`

`AdaptedCLIP.encode_image/encode_text` (L650–723), identical structure to MERU:
1. backbone → pooled feature (image: `visual(images)` on pixel-normalized input;
   text: features at the EOS/argmax token).
2. `visual_proj` / `textual_proj` → `embed_dim`.
3. **project to hyperboloid:** `feats * alpha.exp()` then `L.exp_map0(feats, curv)`
   under `autocast(float32)`. (`project=False` returns the Euclidean tangent feature.)

`forward` clamps `curv` to `[log(curv/10), log(curv*10)]`, clamps `alpha ≤ 0` (so it
never up-scales), clamps `logit_scale ≤ ln(100)`. Loss = contrastive (neg-Lorentzian
distance, in-batch + cross-GPU negatives) + `entail_weight ×` entailment. **Same
clamps/structure already in our `model.py`.**

> Note: HAC's entailment is the **HyCoCLIP object–scene** form (global/local aperture
> thresholds 0.7/1.2, box terms). **We replace this with our taxonomic SEL** from
> `mine/hyperbolic/loss.py`. The *geometry/lift/clamps* transfer; the *loss* is ours.

---

## LoRA config that worked (ViT-B)  `[code]`

[`configs/train_hac_vit_b_lora.py`](../../other/HAC/configs/train_hac_vit_b_lora.py):

```python
LoraConfig(
    r=128, lora_alpha=128,
    target_modules=["attn.q_proj","attn.k_proj","attn.v_proj","attn.proj"],  # q,k,v,o
    lora_dropout=0.1, bias="none",
    use_rslora=True,                       # rank-stabilized LoRA
    exclude_modules=generate_lora_param_names(
        visual_blocks=range(0,8),          # exclude visual blocks 0-7  => adapt 8-11 (last 4)
        textual_blocks=range(0,4),         # exclude textual blocks 0-3 => adapt 4-11 (last 8)
        separate_qkv=True, components=["attn","proj","ffn"]),
)
```
- **Same LoRA config applied to both encoders**, but `exclude_modules` adapts only the
  **last 4 vision / last 8 text blocks** (text-heavier — matches the QAP finding).
- `separate_qkv=True`: MHA is split into q/k/v/o linears so LoRA can target each.
- `use_rslora=True` (rank-stabilized).
- Optim: AdamW lr 2.5e-4, betas (0.9,0.98), wd 0.2 but **0.0 for norm/bias and for
  `logit_scale, visual_alpha, textual_alpha, curv` and any `lora_` param**
  (`set_weight_decay_per_param`); 4k warmup + cosine; 30k iters; batch 768; AMP.

### For our 2×A5000 build
- **r=128 is heavy** — start r=α∈{8,16,32}; rank is a cheap ablation. (HAC's r=128 was
  their full-B result; the "high rank helps" finding is for the bigger model.)
- Keep **q,k,v,o** targets and **last-few-blocks, text-heavier** masking — both backed
  by their ablations.
- **Reuse `set_weight_decay_per_param` logic**: no wd on norms/biases/scalars/LoRA.

---

## Dependencies this implies

- `peft` (LoraConfig / get_peft_model) — for the LoRA path.
- `adapters` (adapterhub) — only for the bottleneck/sequential-adapter path; **not
  needed** if we do LoRA-only.
- A **plain-MHA replacement** so attention submatrices are addressable
  (`hac/utils/plain_mha.py::replace_mha_with_plain`). open_clip's attention is a fused
  `nn.MultiheadAttention`; LoRA needs separate q/k/v/o linears. **This is the one
  non-trivial plumbing piece** we'd port or re-implement for our open_clip backbone.

> Open question for our build: our `mine/hyperbolic/model.py` uses HF/`AutoModel`
> (TIPS) or a DINO+CLIP pair. To LoRA an **open_clip** CLIP/BioCLIP we need its
> attention exposed as q/k/v/o — either via this plain-MHA swap or by targeting
> open_clip's own attention module names with PEFT. Resolve before the LoRA tier.

---

## What we copy vs. change

| Component | HAC | Us |
|---|---|---|
| Backbone | frozen MERU/HyCoCLIP CLIP (ViT-B) | frozen **open_clip CLIP + BioCLIP** (ViT-B/16) |
| Freeze→PEFT→LN sequence | `scripts/train.py` L211–294 | **copy verbatim** |
| Projection + alpha + curv + exp_map0 | `AdaptedCLIP` | **already have** in `model.py` |
| Final-LN re-init + unfreeze | `init_final_ln` | **add** (we currently freeze whole backbone) |
| LoRA (q,k,v,o, last blocks, rsLoRA) | r=128 | **copy structure**, smaller r |
| Loss | HyCoCLIP object–scene entailment | **our taxonomic SEL** (`loss.py`) |
| Data | GRIT webdataset (boxes) | **Planktonzilla** (taxonomy, no boxes) → `use_boxes=False` |
| wd-per-param exclusion | `set_weight_decay_per_param` | **reuse the rule** |
