"""Pull a W&B model artifact (uploaded by `push_sweep_ckpt.py`) onto this machine.

  python scripts/pull_ckpt.py uofg/hyperbolic-plankton-sweep/ranked-dedup-daiqxa8h-best:latest

Refuses to overwrite an existing destination file: checkpoints from different sweeps are
NOT comparable (pre-`8e6144f` runs tuned a broken objective), so silently replacing one
with another would destroy the ability to tell them apart. Pass `--as-name` to choose a
different filename, or delete the old one deliberately.

After downloading it re-opens the checkpoint and prints the recipe it was trained under,
so a mislabelled artifact is caught here rather than three experiments later.
"""

import argparse
import os
import shutil
import sys

import torch
import wandb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact", help="entity/project/name:version")
    ap.add_argument("--out", default="/scratch/daniela/hyperbolic_plankton_ckpts/from_cluster",
                    help="destination directory (default: %(default)s)")
    ap.add_argument("--as-name", default=None,
                    help="filename to save as (default: the artifact name + .pt)")
    args = ap.parse_args()

    api = wandb.Api()
    art = api.artifact(args.artifact, type="model")
    print(f"artifact {args.artifact}")
    print(f"  description: {art.description}")
    print("  metadata:")
    for k, v in (art.metadata or {}).items():
        print(f"    {k}: {v}")

    d = art.download()
    srcs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")]
    if not srcs:
        sys.exit(f"no .pt file in {d}")
    src = srcs[0]

    os.makedirs(args.out, exist_ok=True)
    name = args.as_name or f"{art.name.split(':')[0]}.pt"
    dst = os.path.join(args.out, name)
    if os.path.exists(dst):
        print(f"\nWARNING: {dst} exists - NOT overwriting.")
        print(f"  existing: {os.path.getsize(dst) / 1e6:.1f} MB")
        print(f"  new     : {os.path.getsize(src) / 1e6:.1f} MB")
        print("  re-run with --as-name to pick a different filename, or delete it first.")
        sys.exit(1)
    shutil.copy2(src, dst)
    print(f"\nsaved -> {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB)")

    ck = torch.load(dst, map_location="cpu", weights_only=False)
    cfg = ck.get("args", {})
    print(f"  it={ck.get('it')}  unseen_mean_f1={ck.get('unseen_mean_f1')}")
    print(f"  contrastive={cfg.get('contrastive')} geometry={cfg.get('geometry')} "
          f"no_proj={cfg.get('no_proj')}")
    print(f"  lora_r={cfg.get('lora_r')} alpha={cfg.get('lora_alpha')} "
          f"lr={cfg.get('lr')} wd={cfg.get('wd')} sched={cfg.get('scheduler')}")
    print(f"  lora keys: {sum(1 for k in ck['model'] if 'lora' in k.lower())}")


if __name__ == "__main__":
    main()
