#!/usr/bin/env python3
"""
Magnitude comparison ablation: does the base-625 panel give the model magnitude?

1. Generate NUM a <op> NUM b -> TRUE/FALSE data with a held-out value band and
   a never-seen far band (data/magnitude_generator.py).
2. Train each embedding arm on identical data per seed.
3. Evaluate with evaluate_magnitude.py and collate mean ± std over seeds.

    python scripts/magnitude_experiment.py --out-dir runs/magnitude --seeds 1 2 3 --epochs 40

Shares the training/process helpers with grid_experiment.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_experiment import ARMS, run, train_arm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = ["learned_untied", "learned", "grid_linear", "grid_conv"]
SPLITS = ("test", "test_holdout", "test_far")
GAPS = ("0", "1", "2-5", "6-20", "21-100", "101+")


def generate_data(args, data_dir: Path):
    if (data_dir / "metadata.json").exists() and not args.regenerate:
        print(f"✓ Data already present in {data_dir}")
        return
    cmd = ["data/magnitude_generator.py", "--out", str(data_dir), "--n-train", str(args.n_train),
           "--n-val", str(args.n_val), "--n-test", str(args.n_test), "--value-range", args.value_range,
           "--holdout", args.holdout, "--far-range", args.far_range, "--p-equal", str(args.p_equal),
           "--small-gap-frac", str(args.small_gap_frac), "--small-gap-max", str(args.small_gap_max),
           "--seed", str(args.data_seed), "--ops", *args.ops, "--task", args.task]
    print(f"→ generating data in {data_dir}")
    run(cmd, data_dir / "generate.log")


def evaluate_arm(args, run_dir: Path, data_dir: Path):
    splits = [str(data_dir / f"{s}.json") for s in SPLITS if (data_dir / f"{s}.json").exists()]
    cmd = ["evaluate_magnitude.py", "--checkpoint", str(run_dir / "best_model.pt"), "--splits", *splits,
           "--out", str(run_dir / "eval.json"), "--device", args.device]
    run(cmd, run_dir / "eval.log")


def _arm_of(key: str) -> str:
    return key.rsplit("_s", 1)[0]


def _ms(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]*100:.1f}"
    return f"{np.mean(values)*100:.1f} ± {np.std(values)*100:.1f}"


def _ms_raw(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return "n/a"
    return f"{np.mean(values):.3f}" if len(values) == 1 else f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def collate(results, out_dir: Path):
    arms = []
    for key in results:
        if _arm_of(key) not in arms:
            arms.append(_arm_of(key))
    by_arm = {arm: [r for k, r in results.items() if _arm_of(k) == arm] for arm in arms}
    n_seeds = {arm: len(rs) for arm, rs in by_arm.items()}

    lines = ["# Magnitude comparison ablation", "",
             "`NUM a <op> NUM b ANSWER -> TRUE/FALSE`, ops LESS_THAN / GREATER_THAN. Accuracy (%) is argmax over the",
             "full vocabulary; mean ± std over seeds where more than one seed ran.", ""]
    lines += ["## Accuracy per split", "", "| arm | seeds | " + " | ".join(SPLITS) + " |", "|---|---|" + "---|" * len(SPLITS)]
    for arm, rs in by_arm.items():
        cells = [_ms([r["splits"][s]["overall"]["acc"] for r in rs if s in r["splits"]]) for s in SPLITS]
        lines.append(f"| {arm} | {n_seeds[arm]} | " + " | ".join(cells) + " |")
    lines.append("")

    for split in ("test_holdout", "test_far"):
        if not any(split in r["splits"] for r in results.values()):
            continue
        lines += [f"## {split}: accuracy by number of unseen operands", "",
                  "| arm | 1 unseen | 2 unseen |", "|---|---|---|"]
        for arm, rs in by_arm.items():
            lines.append(f"| {arm} | " + " | ".join(
                _ms([r["splits"][split]["by_unseen"][k]["acc"] for r in rs if split in r["splits"]]) for k in ("1", "2")) + " |")
        lines.append("")

    for split in ("test_holdout", "test_far"):
        if not any(split in r["splits"] and "by_gap_unseen2" in r["splits"][split] for r in results.values()):
            continue
        lines += [f"## {split}: both operands unseen, accuracy by |a - b|", "",
                  "The clean test: no seen operand to anchor on.", "",
                  "| arm | " + " | ".join(f"gap {g}" for g in GAPS) + " |", "|---|" + "---|" * len(GAPS)]
        for arm, rs in by_arm.items():
            lines.append(f"| {arm} | " + " | ".join(
                _ms([r["splits"][split]["by_gap_unseen2"][g]["acc"] for r in rs if split in r["splits"]]) for g in GAPS) + " |")
        lines.append("")

    for split in SPLITS:
        if not any(split in r["splits"] for r in results.values()):
            continue
        lines += [f"## {split}: accuracy by |a - b|", "", "| arm | " + " | ".join(f"gap {g}" for g in GAPS) + " |",
                  "|---|" + "---|" * len(GAPS)]
        for arm, rs in by_arm.items():
            lines.append(f"| {arm} | " + " | ".join(
                _ms([r["splits"][split]["by_gap"][g]["acc"] for r in rs if split in r["splits"]]) for g in GAPS) + " |")
        lines.append("")

    lines += ["## Is magnitude in the embedding?", "",
              "Linear readout of k fitted on training-pool numerals, scored on each pool (Spearman); PC1 of the",
              "training-pool rows correlated with k.", "",
              "| arm | readout train | readout holdout | readout far | PC1 train | PC1 holdout | PC1 far |",
              "|---|---|---|---|---|---|---|"]
    for arm, rs in by_arm.items():
        def pick(section, pool, field=None):
            vals = []
            for r in rs:
                sec = r["magnitude_axis"][section]
                if pool in sec:
                    vals.append(sec[pool][field] if field else sec[pool])
            return _ms_raw(vals)
        lines.append(f"| {arm} | {pick('readout', 'train', 'spearman')} | {pick('readout', 'holdout', 'spearman')} | "
                     f"{pick('readout', 'far', 'spearman')} | {pick('pc1_spearman', 'train')} | "
                     f"{pick('pc1_spearman', 'holdout')} | {pick('pc1_spearman', 'far')} |")
    lines.append("")

    lines += ["## Training", "", "| arm | seeds | params | best val loss | epochs |", "|---|---|---|---|---|"]
    for arm, rs in by_arm.items():
        lines.append(f"| {arm} | {n_seeds[arm]} | {rs[0]['params']:,} | "
                     f"{_ms_raw([r['best_val_loss'] for r in rs])} | {', '.join(str(r['epoch']) for r in rs)} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    json.dump(results, open(out_dir / "results.json", "w"), indent=2)
    print("\n".join(lines))
    print(f"\nSaved {out_dir / 'results.md'} and results.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Magnitude comparison ablation driver")
    ap.add_argument("--out-dir", default="runs/magnitude")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--value-range", default="0-499")
    ap.add_argument("--holdout", default="200-299")
    ap.add_argument("--far-range", default="500-624")
    ap.add_argument("--ops", nargs="+", default=["LESS_THAN", "GREATER_THAN"])
    ap.add_argument("--p-equal", type=float, default=0.02)
    ap.add_argument("--small-gap-frac", type=float, default=0.0,
                    help="fraction of pairs drawn with |a-b| <= --small-gap-max (oversample close values)")
    ap.add_argument("--small-gap-max", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--task", default="compare", choices=["compare", "sum"],
                    help="compare: NUM_a OP NUM_b; sum: NUM_a PLUS NUM_b OP NUM_c PLUS NUM_d")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--model-size", default="tiny")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers per training process (0: five-token samples need none)")
    ap.add_argument("--train-extra", default=None,
                    help='extra flags for training/train.py as one string, e.g. "--save-every 4 --keep-checkpoints 0"')
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--collate-only", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)
    if not args.collate_only:
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
                if proc.wait() != 0:
                    raise SystemExit(f"training {arm} seed {seed} failed, see {out_dir / f'{arm}_s{seed}' / 'train.log'}")
        print(f"✓ training finished in {(time.time() - t0) / 60:.1f} min")

    for arm, seed in jobs:
        run_dir = out_dir / f"{arm}_s{seed}"
        print(f"→ evaluating {run_dir}")
        evaluate_arm(args, run_dir, data_dir)
    results = {p.parent.name: json.load(open(p)) for p in sorted(out_dir.glob("*_s*/eval.json"))}
    collate(results, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
