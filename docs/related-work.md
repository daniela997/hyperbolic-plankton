# Related Work — the four-paper arc

> The literature lineage our method sits in, and what each paper gives us. PDFs in
> `/home/daniela/mine/hyperbolic/papers/`. Confidence flags as in `planktonzilla.md`.

---

## The arc (our related-work spine)

| Paper | Space | Hierarchy mechanism | Backbone training | What it gives us |
|---|---|---|---|---|
| **MERU** (2023) | Hyperbolic (Lorentz) | entailment cone, *text entails image* | **from scratch** | geometry + cone/entailment primitives (exp_map, oxy_angle, half_aperture) |
| **CLIBD** (2024) | **Euclidean** | taxonomic ranks (+DNA) as text, contrastive | full fine-tune | biology task framing; taxonomy-as-supervision; seen/unseen eval protocol |
| **Hyperbolic Taxonomies** (2025) | Hyperbolic | **Stacked Entailment Loss (SEL)** across ranks | full fine-tune | maps CLIBD→hyperbolic; the SEL loss our `loss.py` implements |
| **HAC** (2026) | Hyperbolic | HyCoCLIP compositional entailment | **frozen + PEFT (LoRA)** | proof you can lift a *frozen* CLIP into hyperbolic space cheaply |

### The gap we fill (our novelty — keep this crisp)

> **HAC's PEFT efficiency × the 2025 paper's taxonomic SEL × plankton imaging.**

- HAC shows PEFT→hyperbolic works — but for **object–scene** hierarchy (HyCoCLIP /
  GRIT boxes), evaluated on **VQA**, *not* taxonomy.
- The 2025 paper does **taxonomic-rank** hierarchy with SEL — but **full fine-tuning**,
  on **insects + DNA** (BIOSCAN-1M), not plankton, not PEFT.
- **No prior work does taxonomic-rank PEFT-hyperbolic adaptation.** That's the
  unfilled cell. Do NOT let it read as "just HAC on plankton" (different hierarchy +
  different loss) or "just the 2025 paper, cheaper" (they full-fine-tuned).

---

## Per-paper notes

### MERU (2023) — Hyperbolic Image-Text Representations
- First large-scale hyperbolic contrastive CLIP. Lorentz hyperboloid; embeddings
  lifted via exp map from the origin; contrastive on **negative Lorentzian distance**
  + an **entailment loss** enforcing *text entails image* (image inside text's cone).
- Trained **from scratch** (12M RedCaps). Competitive with Euclidean CLIP, better at
  capturing hierarchy; gains at low embedding dimensions.
- **Gives us:** the geometric toolkit (`hyperbolic.py`'s `LorentzMath`), the cone /
  half-aperture / oxy_angle machinery, the `text entails image` idea SEL generalizes.

### CLIBD (2024) — Bridging Vision and Genomics
- Tri-modal (image, DNA barcode, taxonomic text) **Euclidean** contrastive alignment
  on BIOSCAN-1M insects. Classifies by nearest reference embedding (image/DNA/text).
- **Seen/unseen split** by holding out species — the protocol the plankton paper and
  we reuse. Concatenates taxonomic ranks into a text string ("full text").
- **Gives us:** the biological-taxonomy task framing, the seen/unseen generalization
  setup, taxonomy-as-text supervision. Stays Euclidean — the thing 2025 hyperbolizes.

### Hyperbolic Taxonomies (2025) — Hyperbolic Multimodal Rep. for Biological Taxonomies
- Takes the CLIBD setup into **hyperbolic** space (MERU framework) and adds the
  **Stacked Entailment Loss (SEL)**: enforce entailment between *consecutive*
  taxonomic ranks (intra-modal) + image↔deepest-text (inter-modal), with pos+neg
  cones. Transitivity across ranks.
- Full fine-tune, 4×A100, BIOSCAN-1M. Finding: hyperbolic matches/exceeds Euclidean
  CLIBD at higher ranks; fine-grained species + open-world still hard.
- **Gives us:** the SEL loss itself — **already implemented** in
  `/home/daniela/mine/hyperbolic/loss.py` (SEL-intra/inter, stacked, ragged-rank
  validity, hard negatives). This is the loss we apply to plankton.

### HAC (2026) — Parameter-Efficient Hyperbolic Adaptation of CLIP for Zero-Shot VQA
**The methodological keystone for our efficiency claim.**

- **Core result:** lift a *frozen* pretrained CLIP into hyperbolic space via **PEFT**
  (LoRA / adapters / BitFit / LayerNorm tuning) — no from-scratch training. Beats
  Euclidean CLIP and **approaches/surpasses fully-trained hyperbolic HyCoCLIP with
  85× fewer trainable params and 2/3 the data** (HAC-S avg 37.9 vs HyCoCLIP-S 38.1;
  HAC-B LoRA surpasses HyCoCLIP-B on 4/6 tasks). **[stated]**
- **Compute (§5.3):** HAC-S < 1 day on a **single A6000**; HAC-B < 1 day on a
  **single A100**. vs MERU 8×V100 ~1 day, HyCoCLIP 4×A100. → the single-GPU-adapt vs
  multi-GPU-from-scratch contrast we want against Planktonzilla's 64×H100. **[stated]**
- **Loss:** HyCoCLIP's hierarchical compositional contrastive + entailment
  (`L_hCC + λ L_hCE`, λ=0.1). Hierarchy is **object–scene**, not taxonomic. Removing
  `L_hCE` "collapses the learned hyperbolic geometry" → independent evidence the
  **entailment term is load-bearing** (supports our CL-vs-SEL ablation). **[stated]**

#### HAC recipe specifics we copy (evidence-based, de-risks our build)
- **What's trained vs frozen** (Fig 1, §4.4): backbone **frozen**; trainable =
  PEFT adapters in selected blocks **+ fully-retrained final projection heads + final
  LayerNorm of each encoder**. They stress the final LayerNorm *must* be trained (not
  skip-bypassed) or it gates the output. → our hyperbolic projector = the "projection
  head"; **also unfreeze the final LN**. **[stated]**
- **Hyperbolic setup:** coordinate order `[x_space, x_time]`; Euclidean feature lifted
  with a **zero time component** then exp_map0; learnable projection scalars
  `α_img, α_txt` init `1/√n`, optimized in log-space; learnable curvature κ. (Same as
  our `model.py`.) **[stated]**
- **LoRA config — scale dependent (Table 1–3 + ablations):**
  - HAC-B (≈ our ViT-B tier): LoRA on **all attention submatrices q,k,v,o**,
    **r = α = 128**, on the **last 4 vision blocks / last 8 text blocks** (not all 12).
  - HAC-S (smaller): r = α = 8; *increasing* rank/matrices does **not** help the small
    model → optimal hyperparams depend on model scale.
  - Ablations: q,k,v,o all matter (dropping `o` or using MLP-LoRA hurts); high rank
    helps the **bigger** model; **adapt more text than vision blocks** (QAP analysis:
    vision layers are largely "unaligned feature extractors," text aligns from
    mid-depth, so text needs more adaptation). **[stated]**
  - For HAC-S the best PEFT was **sequential adapters**; for HAC-B it was **LoRA**.
    → on ViT-B/16 (B tier), **LoRA is the indicated choice**. **[stated]**
- **Optimization:** 30K iters, batch 768, AdamW (β 0.9/0.98), wd 0.2 (off for
  gains/biases/scalars), lr 2.5e-4, 4K linear warmup + cosine. Reusable starting
  config. **[stated]**
- **Augmentations (Supp.):** resize-shorter-224 + center-crop, hflip p0.5, color
  jitter, gaussian blur, autocontrast, hist-eq, sharpness, gamma; NEFTune α=0.1 on
  text token embeddings. **[stated]**

---

## How this maps onto our decisions

- **LoRA target** was an open question; HAC settles it: **q,k,v,o** (not just q,v),
  on the **last few blocks, text-heavier**, plus train **projector + final LN**.
- **LoRA vs adapters:** B tier → **LoRA**.
- **Rank:** HAC-B used 128 (heavy for 24GB); start lower (8–16) on 2×A5000, treat rank
  as a cheap ablation. The "high rank helps bigger models" finding is about their full
  B with 128 — we may not see the same with a frozen-projector + small-rank budget.
- **The entailment loss matters** (HAC + 2025 both show it) → keep SEL; run the
  CL-vs-SEL ablation to attribute our gains to *structure*, not just adaptation.
- **Efficiency framing is precedented**, not speculative — cite HAC §5.3 + the 85×
  param / 2-3 data result as the foundation for "match big-compute cheaply."
