# %% [markdown]
# # V4 BIOSCAN — geometry vs classification, all configs
#
# Persistent analysis of every v4 run: per-rank geometry (radii, apertures, cone nesting,
# collapse) from `diagnose_geometry.py`, joined to wandb test-F1, plus the HoroPCA gallery.
#
# Run as a notebook (jupytext/`# %%` cells) or `python v4_geometry_analysis.py`.
# Results are cached to `geom_cache.json` so the 24 diagnostics run once.

# %%
import glob, json, os, re, subprocess, sys
import pandas as pd

REPO = "/home/daniela/mine/hyperbolic-plankton"
CKDIR = "/scratch/daniela/hyperbolic_plankton_ckpts"
PY = "/scratch/daniela/miniconda3/envs/dino_plankton/bin/python"
HERE = os.path.join(REPO, "notebooks/v4_analysis")
CACHE = os.path.join(HERE, "geom_cache.json")
os.chdir(REPO)

# %% [markdown]
# ## 1. Locate one checkpoint per config (newest run per tag)

# %%
def find_ckpts():
    tags = {}
    for d in sorted(glob.glob(f"{CKDIR}/bioscan_*_v4__*/")):
        m = re.match(r".*/(bioscan_.*_v4)__[^/]+/$", d)
        if not m:
            continue
        tag = m.group(1)
        best = os.path.join(d, f"{tag}_best.pt")
        if os.path.exists(best):
            tags.setdefault(tag, []).append((os.path.getmtime(best), best))
    return {t: sorted(v)[-1][1] for t, v in tags.items()}

ckpts = find_ckpts()
print(f"{len(ckpts)} configs")

# %% [markdown]
# ## 2. Run diagnose_geometry on each (cached) and parse the per-rank geometry

# %%
def parse_diag(out):
    g = lambda p, d="?": (re.search(p, out).group(1) if re.search(p, out) else d)
    radii = {r: g(rf"{r}\s+([\d.]+)\s+[\d.]+") for r in ["order","family","genus","species"]}
    aps   = {r: g(rf"{r}\s+[\d.]+\s+([\d.]+)") for r in ["order","family","genus","species"]}
    fits  = {f"fit_{c}": g(rf"{p}->{c}\s+[\d.]+\s+[-\d.]+\s+([\d.]+)")
             for p, c in [("order","family"),("family","genus"),("genus","species")]}
    incone = dict(re.findall(r"(order|family|genus|species)=([\d.]+)", out))
    return dict(
        curv=g(r"curv=([\d.]+)"),
        top1=g(r"top-1 acc \(DISTANCE\) = ([\d.]+)"),           # distance-to-prototype (our classifier)
        top1_angle=g(r"top-1 acc \(ANGLE/ATMG\) = ([\d.]+)"),   # ATMG argmin-exterior-angle
        top1_cone=g(r"top-1 acc \(CONE-ENERGY/Dhall\) = ([\d.]+)"),  # Dhall argmin cone-violation
        cone_contained=g(r"img inside own-species cone ([\d.]+)"),   # frac img with zero species-cone violation
        sep=g(r"inter-species proto dist = ([\d.]+)"),          # MEAN sep — misleads, kept for contrast
        nn_sep=g(r"NN-sep ([\d.]+)"),                           # NEAREST-NEIGHBOUR sep — predicts F1
        margin=g(r"per-image margin = ([+\-\d.]+)"),            # per-image classification margin
        proto_r=g(r"CLASSIFIER protos.*radius ([\d.]+)"),       # classifier-prototype radius
        **{f"r_{r}": radii[r] for r in radii}, **{f"a_{r}": aps[r] for r in aps}, **fits,
        img_r=g(r"image radius = ([\d.]+)"), img_cos=g(r"pairwise cos = ([\d.]+)"),
        img_in_species=incone.get("species","?"), img_in_order=incone.get("order","?"),
    )

def run_all(force=False):
    if os.path.exists(CACHE) and not force:
        return json.load(open(CACHE))
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    res = {}
    for tag, ck in sorted(ckpts.items()):
        out = subprocess.run([PY, "scripts/diagnose_geometry.py", "--ckpt", ck,
                              "--dataset", "bioscan"], capture_output=True, text=True, env=env).stdout
        res[tag] = parse_diag(out)
        print(f"  {tag}", file=sys.stderr)
    json.dump(res, open(CACHE, "w"), indent=2)
    return res

geom = run_all(force=False)

# %% [markdown]
# ## 3. Join wandb test-F1 (per rank, seen + unseen)

# %%
def pull_wandb_f1():
    import wandb, math
    api = wandb.Api()
    runs = list(api.runs("hyperbolic-plankton", filters={"display_name": {"$regex": ".*_v4$"}}))
    def lv(h, c):
        if c not in h.columns: return float("nan")
        s = h[c].dropna(); return float(s.iloc[-1]) if len(s) else float("nan")
    out = {}
    for r in runs:
        h = r.history(samples=3000, pandas=True)
        out[r.name] = {f"{sp}_{rk}_f1": lv(h, f"test/{sp}/{rk}_f1")
                       for sp in ["seen","unseen"] for rk in ["order","family","genus","species"]}
    return out

WANDB_CACHE = os.path.join(HERE, "wandb_f1_cache.json")
if os.path.exists(WANDB_CACHE):
    wf1 = json.load(open(WANDB_CACHE))
else:
    wf1 = pull_wandb_f1(); json.dump(wf1, open(WANDB_CACHE, "w"), indent=2)

# %% [markdown]
# ## 4. Assemble the master table

# %%
def fnum(x):
    try: return float(x)
    except (ValueError, TypeError): return float("nan")

rows = []
for tag, d in geom.items():
    short = tag.replace("bioscan_","").replace("_r64_v4","")
    w = wf1.get(tag, {})
    rows.append(dict(
        config=short,
        seen_sp_f1=w.get("seen_species_f1", float("nan")),     # authoritative (Planktonzilla macro-F1)
        unseen_sp_f1=w.get("unseen_species_f1", float("nan")),
        # geometry metrics that PREDICT classification (measured B0 vs C1):
        nn_sep=fnum(d["nn_sep"]),          # nearest-neighbour proto separation
        margin=fnum(d["margin"]),          # per-image classification margin
        proto_r=fnum(d["proto_r"]),        # classifier-prototype radius
        # metrics that DON'T predict (kept to show they mislead):
        mean_sep=fnum(d["sep"]), diag_top1=fnum(d["top1"]), curv=fnum(d["curv"]),
        r_order=fnum(d["r_order"]), r_species=fnum(d["r_species"]),
        radial_ok=fnum(d["r_order"]) < fnum(d["r_species"]),  # coarse inside fine?
        collapsed=fnum(d["r_species"]) < 0.05,                # everything at origin
        img_cos=fnum(d["img_cos"]), img_in_species=fnum(d["img_in_species"]),
    ))
df = pd.DataFrame(rows).sort_values("seen_sp_f1", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 220, "display.max_columns", 30)
# nn_sep = geometric (prototype NN distance); mean_sep = misleading; margin kept in df but NOT shown
# here (it's ~circular with accuracy — see §5).
print(df[["config","seen_sp_f1","unseen_sp_f1","nn_sep","mean_sep","curv","collapsed"]]
      .round(3).to_string(index=False))

# %% [markdown]
# ## 5. Which geometry metric predicts seen-species F1?
#
# CAVEATS (read before trusting any r below):
# - `margin` = mean over images of (dist to nearest WRONG proto − dist to correct proto). Its sign
#   per image IS the classification decision, so `frac(margin>0) == top-1 acc`. Correlating margin
#   with F1 is therefore ~CIRCULAR — it restates accuracy, not an independent geometric cause. EXCLUDED
#   from the predictive read below.
# - `diag_top1` is the diagnostic's own accuracy proxy — also not an independent predictor.
# - The only NON-circular geometric metrics here are nn_sep / mean_sep / curv / radii (properties of
#   the embedding alone, computed without reference to predictions).
# With n=23 configs and several candidate metrics, a lone r≈0.5 is weak evidence. The DEFENSIBLE
# claim is the CONTROLLED two-point one: B0 vs C1 (same loss family, differ only in SEL text) have
# ~equal mean_sep but B0 nn_sep 0.49 vs C1 0.20 → B0 classifies better. nn_sep is the right metric;
# mean_sep misleads. The across-all-configs correlation only WEAKLY supports this (nn_sep r≈+0.47 >
# mean_sep +0.34), and curv (−0.62) is just as strong — so NO single metric robustly explains F1.

# %%
num = df.select_dtypes("number").drop(columns=["margin", "diag_top1"], errors="ignore")  # drop circular
corr = num.corr(numeric_only=True)["seen_sp_f1"].drop("seen_sp_f1").sort_values(key=abs, ascending=False)
print("Correlation with seen F1 (CIRCULAR metrics margin/diag_top1 excluded; |r| desc):")
print(corr.round(3).to_string())
print("\nThe non-circular geometric metrics, head-to-head:")
for k in ["nn_sep", "mean_sep", "curv"]:
    if k in corr:
        print(f"  {k:10} r = {corr[k]:+.3f}")

# %% [markdown]
# ## 6. HoroPCA gallery — generate any missing plots

# %%
def horopca(tag, sel_text):
    ck = ckpts[f"bioscan_{tag}_r64_v4"]
    out = os.path.join(HERE, f"horopca_{tag}.png")
    if os.path.exists(out):
        return out
    env = {**os.environ, "PYTHONPATH": "src", "CUDA_VISIBLE_DEVICES": "0",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run([PY, "scripts/visualize_horopca.py", "--ckpt", ck, "--dataset", "bioscan",
                    "--backbone", "clip", "--lora", "--lora-r", "64", "--sel-text", sel_text,
                    "--split", "test_seen", "--n", "150", "--out", out],
                   capture_output=True, text=True, env=env)
    return out

# sel_text per config (from the tag): cumulative configs use cumulative, else independent
# (E is euclidean - skip horopca). Generate on demand in the notebook.
print("call horopca('C4_clonly','independent') etc. to render a plot inline")
