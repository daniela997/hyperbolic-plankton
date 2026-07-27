"""Push the best checkpoint of a W&B sweep to W&B as a model artifact. Run ON THE POD.

Ranks every run in the sweep by a summary metric, locates the winner's `_best.pt` under
`$HP_CKPT_DIR/{tag}__{run_id}/`, and uploads it as an artifact ATTACHED TO the winning run
(so the artifact keeps its lineage back to the config/metrics that produced it).

Pull it on another machine with `scripts/pull_ckpt.py`.

  # inspect the ranking + resolve the ckpt path, WITHOUT uploading:
  /root/hp-venv/bin/python scripts/push_sweep_ckpt.py daiqxa8h --dry-run

  # upload:
  /root/hp-venv/bin/python scripts/push_sweep_ckpt.py daiqxa8h

`lora_r` is printed for every top run because it identifies the recipe generation: r=32 is
the v4 (post-`8e6144f`) config pinned to E0c capacity, r=64 is the older pre-fix recipe
whose rankings tuned a broken objective and are not comparable.
"""

import argparse
import glob
import os
import sys

import wandb

ENTITY = os.environ.get("WANDB_ENTITY", "uofg")
PROJECT = os.environ.get("WANDB_PROJECT", "hyperbolic-plankton-sweep")
CKPT_DIR = os.environ.get("HP_CKPT_DIR", "/mnt/resources/hyperbolic_plankton_ckpts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_id", help="the sweep id, e.g. daiqxa8h")
    ap.add_argument("--metric", default="eval/seen/species_f1",
                    help="summary metric to rank runs by (default: %(default)s)")
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--artifact-name", default=None,
                    help="default: ranked-dedup-<sweep_id>-best")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the ranking and resolved ckpt path, upload nothing")
    args = ap.parse_args()

    api = wandb.Api()
    sweep = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")
    print(f"sweep {args.sweep_id}: {len(sweep.runs)} runs")

    # Rank by the summary metric. Runs that never logged it (crashed early, still starting)
    # are skipped rather than treated as 0 — a 0 would outrank nothing but clutters the list.
    scored = []
    for r in sweep.runs:
        v = r.summary.get(args.metric)
        if v is None:
            continue
        scored.append((float(v), r))
    if not scored:
        sys.exit(f"no run in {args.sweep_id} has summary metric '{args.metric}'")
    scored.sort(key=lambda t: -t[0])

    print(f"\ntop runs by {args.metric}:")
    for v, r in scored[:8]:
        lr = r.config.get("lr")
        lr_s = f"{lr:.3e}" if isinstance(lr, (int, float)) else str(lr)
        print(f"  {v:.4f}  {r.id}  {r.state:9s}  lr={lr_s}  wd={r.config.get('wd')}  "
              f"r={r.config.get('lora_r')}  {r.config.get('tag', '')}")

    best_v, best = scored[0]
    lr = best.config.get("lr")
    lr_s = f"{lr:.3e}" if isinstance(lr, (int, float)) else str(lr)
    print(f"\nBEST: {best.id}  {args.metric}={best_v:.4f}")
    print(f"  config: lr={lr_s} wd={best.config.get('wd')} "
          f"lora_r={best.config.get('lora_r')} alpha={best.config.get('lora_alpha')} "
          f"sched={best.config.get('scheduler')} contrastive={best.config.get('contrastive')}")
    print(f"  summary: seen/species_f1={best.summary.get('eval/seen/species_f1')} "
          f"unseen_mean={best.summary.get('eval/unseen/mean_f1')}")

    # train_lora saves to {CKPT_DIR}/{tag}__{run_id}/{tag}_best.pt
    tag = best.config.get("tag", "bioclip_lora")
    cand = os.path.join(args.ckpt_dir, f"{tag}__{best.id}", f"{tag}_best.pt")
    if not os.path.exists(cand):
        hits = glob.glob(os.path.join(args.ckpt_dir, f"*__{best.id}", "*_best.pt"))
        if not hits:
            print(f"\nNOT FOUND: {cand}")
            print("dirs containing that run id:")
            for d in glob.glob(os.path.join(args.ckpt_dir, f"*{best.id}*")):
                print("   ", d, os.listdir(d))
            sys.exit(1)
        cand = hits[0]
    print(f"\ncheckpoint: {cand}  ({os.path.getsize(cand) / 1e6:.1f} MB)")

    if args.dry_run:
        print("(dry run - not uploading)")
        return

    name = args.artifact_name or f"ranked-dedup-{args.sweep_id}-best"
    # resume="must" attaches the artifact to the winning run itself, preserving lineage.
    run = wandb.init(entity=args.entity, project=args.project, id=best.id,
                     resume="must", job_type="export")
    art = wandb.Artifact(
        name, type="model",
        description=f"best of sweep {args.sweep_id} by {args.metric}={best_v:.4f} (run {best.id})",
        metadata={
            "sweep": args.sweep_id, "run_id": best.id, args.metric: best_v,
            **{k: best.config.get(k) for k in
               ("lr", "wd", "lora_r", "lora_alpha", "scheduler", "contrastive",
                "geometry", "micro_bs", "accum", "epochs", "rince_min_tau", "rince_max_tau")},
        },
    )
    art.add_file(cand, name="bioclip_lora_best.pt")
    run.log_artifact(art)
    art.wait()
    print(f"\nuploaded: {args.entity}/{args.project}/{name}:{art.version}")
    print(f"PULL WITH:  python scripts/pull_ckpt.py "
          f"{args.entity}/{args.project}/{name}:latest")
    run.finish()


if __name__ == "__main__":
    main()
