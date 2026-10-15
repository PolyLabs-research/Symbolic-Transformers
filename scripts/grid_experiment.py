#!/usr/bin/env python3
"""
Run the "do the grids matter" ablation end to end.

1. Generate a dataset whose variable/predicate indices come from a band with a
   contiguous hold-out removed (plus a never-seen "far" band).
2. Train one model per embedding arm on identical data and seed.
3. Evaluate every arm on the seen / held-out / far splits with evaluate_grid.py.
4. Collate a results table (markdown + JSON).

Reduced CPU run (about an hour, single seed):
    python scripts/grid_experiment.py --out-dir runs/grid_small --n-train 2000 --epochs 8

Full run (GPU / Colab):
    python scripts/grid_experiment.py --out-dir runs/grid_full --n-train 10000 --epochs 30 --seeds 1 2 3

Arms run in parallel processes by default (one per arm, CPU threads split
between them); pass --sequential to run them one after another.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "learned": ["--embedding", "learned", "--tie-weights"],
    "grid_linear": ["--embedding", "grid_linear", "--tie-weights"],
    "grid_conv": ["--embedding", "grid_conv", "--tie-weights"],
    # Cell weights generated from smooth position features through one shared
    # matrix: no parameter belongs to a single cell.
    "grid_shared": ["--embedding", "grid_shared", "--tie-weights"],
    # Smoother basis: index features only, periods >= 128, so increments cannot
    # change sign inside a gap narrower than ~64 cells.
    "grid_smooth": ["--embedding", "grid_shared", "--tie-weights", "--shared-min-period", "128",
                    "--shared-axis-freq", "0"],
    # Smooth basis plus a zero-initialised per-cell residual for precision on trained cells.
    "grid_hybrid": ["--embedding", "grid_shared", "--tie-weights", "--shared-min-period", "128",
                    "--shared-axis-freq", "0", "--shared-residual"],
    # Every cell gets the same generated weight: embedding = count × one direction.
    # The walk through any gap, and beyond the trained range, is a straight line
    # by construction. grid_tied_res adds the per-cell residual for precision.
    "grid_tied": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "constant"],
    "grid_tied_res": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "constant",
                      "--shared-residual"],
    # [1, i/625] basis: embedding = k·a + (k²/1250)·c + b, whose direction rotates
    # uniformly with k, so LayerNorm cannot collapse it. Straight walk through gaps
    # and beyond the range by construction; grid_ramp_res adds the residual.
    "grid_ramp": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp"],
    "grid_ramp_res": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                      "--shared-residual"],
    # Ramp + residual with the residual in its own optimiser group: 10x learning
    # rate and 10x weight decay, so trained cells sharpen fast and untrained
    # cells decay back to the smooth ramp.
    "grid_ramp_res_fast": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                           "--shared-residual", "--residual-lr-mult", "10", "--residual-wd", "0.1"],
    # Residual projected orthogonal to the ramp plane: texture for trained cells,
    # magnitude only from the ramp (shared with unseen cells).
    "grid_ramp_res_orth": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                           "--shared-residual", "--residual-orthogonal"],
    "grid_ramp_res_fast_orth": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                "--shared-residual", "--residual-orthogonal",
                                "--residual-lr-mult", "10", "--residual-wd", "0.1"],
    "grid_ramp_res_mid_orth": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                               "--shared-residual", "--residual-orthogonal",
                               "--residual-lr-mult", "3", "--residual-wd", "0.03"],
    # Residual dropout: half the training batches see the ramp alone, so the
    # comparator cannot abandon the shared magnitude route for per-cell texture.
    "grid_ramp_res_orth_drop": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                "--shared-residual", "--residual-orthogonal", "--residual-dropout", "0.5"],
    # Residual anneal: full residual scaffolds the first 30% of training, then it
    # is scaled linearly to zero by 70%, leaving the comparator the ramp alone
    # for the last 30%. best_model.pt comes from the residual-free epochs.
    "grid_ramp_res_orth_anneal": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                  "--shared-residual", "--residual-orthogonal", "--residual-anneal", "0.3", "0.7"],
    # Geometric anneal: same window, but the scale falls log-linearly from 1 to
    # 1e-3, so the comparator spends many epochs with a small residual instead of
    # one (a linear anneal only bites once the scale is below 1 / logit margin).
    "grid_ramp_res_orth_anneal_geo": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                      "--shared-residual", "--residual-orthogonal", "--residual-anneal", "0.3", "0.7",
                                      "--residual-anneal-floor", "0.001"],
    # Same geometric anneal pinned to epochs 24-56 of a 120-epoch run (--epochs 120),
    # so the residual-free phase is 64 epochs at a learning rate still well above the
    # cosine floor instead of 24 epochs at the floor.
    "grid_ramp_res_orth_anneal_geo_long": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                           "--shared-residual", "--residual-orthogonal", "--residual-anneal", "0.2", "0.4667",
                                           "--residual-anneal-floor", "0.001"],
    # Anneal and hold: geometric to 0.01 by 70% of training, then held there, so
    # the comparator keeps a small residual it has tuned itself to read.
    "grid_ramp_res_orth_anneal_hold": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                       "--shared-residual", "--residual-orthogonal", "--residual-anneal", "0.3", "0.7",
                                       "--residual-anneal-floor", "0.01", "--residual-anneal-hold"],
    # Per-cell variant: each cell's residual row is dropped independently.
    "grid_ramp_res_orth_cdrop": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                                 "--shared-residual", "--residual-orthogonal", "--residual-dropout", "0.5",
                                 "--residual-dropout-cell"],
    # Middle residual strength between grid_ramp_res (x1, 0.01) and _fast (x10, 0.1).
    "grid_ramp_res_mid": ["--embedding", "grid_shared", "--tie-weights", "--shared-basis", "ramp",
                          "--shared-residual", "--residual-lr-mult", "3", "--residual-wd", "0.03"],
    # Control: the original untied model. Unseen rows keep their random init
    # instead of being pushed away from hidden states by the tied softmax.
    "learned_untied": ["--embedding", "learned"],
    # Control: fixed sinusoid features of k for numerals, same linear layer as grid_linear.
    "sinusoid": ["--embedding", "sinusoid", "--tie-weights"],
    # Same, without the high frequencies (periods 32..1250): a smooth ordinal code
    # that cannot memorise individual values. Separates nesting from smoothness.
    "sinusoid_smooth": ["--embedding", "sinusoid", "--tie-weights", "--sinusoid-min-period", "32"],
}
DEFAULT_ARMS = ["learned", "grid_linear", "grid_conv"]
SPLITS = ("test", "test_holdout", "test_far")
CATEGORIES = ("overall", "structure", "after_numeral", "numeral", "copy", "fresh")


def run(cmd, log_path: Path, env=None, wait: bool = True):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable] + cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if wait:
        rc = proc.wait()
        log.close()
        if rc != 0:
            raise SystemExit(f"command failed ({rc}): {' '.join(cmd)}\n  see {log_path}")
    return proc


def generate_data(args, data_dir: Path):
    if (data_dir / "metadata.json").exists() and not args.regenerate:
        print(f"✓ Data already present in {data_dir}")
        return
    cmd = ["data/dataset_generator.py", "--out", str(data_dir),
           "--n-train", str(args.n_train), "--n-val", str(args.n_val), "--n-test", str(args.n_test),
           "--n-test-holdout", str(args.n_test), "--n-test-far", str(args.n_test),
           "--index-range", args.index_range, "--holdout", args.holdout, "--far-range", args.far_range,
           "--seed", str(args.data_seed)]
    print(f"→ generating data in {data_dir}")
    run(cmd, data_dir / "generate.log")


def train_arm(args, arm: str, seed: int, data_dir: Path, run_dir: Path, threads: int, wait: bool):
    cmd = ["training/train.py", "--model-size", args.model_size, "--num-epochs", str(args.epochs),
           "--batch-size", str(args.batch_size), "--warmup-steps", str(args.warmup_steps),
           "--learning-rate", str(args.learning_rate), "--seed", str(seed),
           "--train-data", str(data_dir / "train.json"), "--val-data", str(data_dir / "val.json"),
           "--checkpoint-dir", str(run_dir), "--device", args.device] + ARMS[arm]
    if getattr(args, "num_workers", None) is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if getattr(args, "train_extra", None):
        cmd += shlex.split(args.train_extra)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    print(f"→ training {arm} seed {seed} ({threads} threads) → {run_dir}")
    return run(cmd, run_dir / "train.log", env=env, wait=wait)


def evaluate_arm(args, run_dir: Path, data_dir: Path):
    splits = [str(data_dir / f"{s}.json") for s in SPLITS if (data_dir / f"{s}.json").exists()]
    cmd = ["evaluate_grid.py", "--checkpoint", str(run_dir / "best_model.pt"), "--splits", *splits,
           "--noise-flip", *[str(p) for p in args.noise_flip], "--out", str(run_dir / "eval.json"),
           "--device", args.device]
    if args.max_eval_samples:
        cmd += ["--max-samples", str(args.max_eval_samples)]
    run(cmd, run_dir / "eval.log")
    return json.load(open(run_dir / "eval.json"))


def _label(key: str, r: dict) -> str:
    """Arm label from the run directory name, marking tied output weights."""
    arm = key.rsplit("_s", 1)[0]
    return f"{arm}{' (tied)' if r.get('tie_weights') else ''}"


def collate(results, out_dir: Path):
    lines = ["# Grid embedding ablation", "",
             "Top-1 accuracy (%) per split and target category. `copy` is the bound-variable reuse case,",
             "the one place the model must emit a specific index.", ""]
    for split in SPLITS:
        if not any(split in r["splits"] for r in results.values()):
            continue
        lines += [f"## {split}", "", "| arm | seed | " + " | ".join(CATEGORIES) + " |",
                  "|---|---|" + "---|" * len(CATEGORIES)]
        for key, r in results.items():
            if split not in r["splits"]:
                continue
            res = r["splits"][split]
            lines.append(f"| {_label(key, r)} | {key.rsplit('_s', 1)[-1]} | "
                         + " | ".join(f"{res[c]['acc']*100:.1f}" for c in CATEGORIES) + " |")
        lines.append("")
    lines += ["## Embedding geometry (numerals)", "",
              "| arm | seed | range | spearman(dist, gap) | nearest-neighbour adjacent |", "|---|---|---|---|---|"]
    for key, r in results.items():
        for rng, g in r["geometry"].items():
            lines.append(f"| {_label(key, r)} | {key.rsplit('_s', 1)[-1]} | {rng} | "
                         f"{g['spearman_dist_vs_gap']:.3f} | {g['nearest_neighbour_is_adjacent']*100:.1f}% |")
    lines.append("")
    noisy = {k: r for k, r in results.items() if r["noise"]}
    if noisy:
        lines += ["## Panel noise (grid arms): overall / copy accuracy on `test`", "",
                  "| arm | seed | flip p | overall | copy |", "|---|---|---|---|---|"]
        for key, r in noisy.items():
            for p, per_split in r["noise"].items():
                res = per_split.get("test")
                if res:
                    lines.append(f"| {_label(key, r)} | {key.rsplit('_s', 1)[-1]} | {p} | "
                                 f"{res['overall']['acc']*100:.1f} | {res['copy']['acc']*100:.1f} |")
        lines.append("")
    lines += ["## Training", "", "| arm | seed | params | best val loss | epochs |", "|---|---|---|---|---|"]
    for key, r in results.items():
        lines.append(f"| {_label(key, r)} | {key.rsplit('_s', 1)[-1]} | {r['params']:,} | "
                     f"{r['best_val_loss']:.4f} | {r['epoch']} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    json.dump(results, open(out_dir / "results.json", "w"), indent=2)
    print("\n".join(lines))
    print(f"\nSaved {out_dir / 'results.md'} and results.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid embedding ablation driver")
    ap.add_argument("--out-dir", default="runs/grid_experiment")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--index-range", default="0-99")
    ap.add_argument("--holdout", default="60-79")
    ap.add_argument("--far-range", default="100-199")
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--model-size", default="tiny")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--noise-flip", nargs="*", type=float, default=[0.01, 0.05])
    ap.add_argument("--max-eval-samples", type=int, default=None)
    ap.add_argument("--sequential", action="store_true", help="Train arms one after another")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads per training process")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per training process")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing checkpoints")
    ap.add_argument("--collate-only", action="store_true", help="Only rebuild results.md from existing eval.json files")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)
    generate_data(args, data_dir)

    jobs = [] if args.collate_only else [(arm, seed) for seed in args.seeds for arm in args.arms]
    if not args.eval_only and not args.collate_only:
        n_par = 1 if args.sequential else len(args.arms)
        threads = args.threads or max(1, (os.cpu_count() or 1) // n_par)
        t0 = time.time()
        for seed in args.seeds:
            procs = []
            for arm in args.arms:
                run_dir = out_dir / f"{arm}_s{seed}"
                proc = train_arm(args, arm, seed, data_dir, run_dir, threads, wait=args.sequential)
                if not args.sequential:
                    procs.append((arm, proc))
            for arm, proc in procs:
                rc = proc.wait()
                if rc != 0:
                    raise SystemExit(f"training {arm} seed {seed} failed, see {out_dir / f'{arm}_s{seed}' / 'train.log'}")
        print(f"✓ training finished in {(time.time() - t0) / 60:.1f} min")

    for arm, seed in jobs:
        run_dir = out_dir / f"{arm}_s{seed}"
        print(f"→ evaluating {run_dir}")
        evaluate_arm(args, run_dir, data_dir)
    # Collate every evaluated arm in the output directory, so arms added in
    # later invocations (e.g. --arms learned_untied) join the same table.
    results = {}
    for eval_path in sorted(out_dir.glob("*_s*/eval.json")):
        results[eval_path.parent.name] = json.load(open(eval_path))
    collate(results, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
