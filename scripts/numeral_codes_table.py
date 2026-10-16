#!/usr/bin/env python3
"""Tables for the numeral-code head-to-head (experiments/numeral_codes.md).

For each task and arm, reads every <run_dir>/<arm>_s<seed>/eval.json and prints mean ± population std over
seeds (n = 3 unless stated) of: seen accuracy (test), held-out and far all-operands-unseen accuracy with the
held-out gap breakdown, the whole held-out and far bands, and the TRUE/FALSE-restricted accuracy on both
bands (a full-vocabulary miss can be a non-answer token, most likely in the far band for the xVal code,
whose tied numeral logits grow with |value|). The last column is each seed's escape epoch: the first epoch
with validation loss below 0.6 (from train.log), or "never" if the seed stayed at chance.

Rank arms on the all-unseen columns: about half of each unseen split can be answered by treating an
unknown operand as "big", and on the all-unseen subsets a constant guess is at the base rate.

usage: numeral_codes_table.py [--json out.json]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# (task label, {arm label: glob of eval.json}); the comparison-task sinusoid runs live in an older directory
# with byte-identical data (see experiments/withdrawal_routes.md).
SOURCES = [
    ("comparison: NUM_a OP NUM_b", {
        "learned table": "runs/magnitude_precision_routes/learned_untied_s*/eval.json",
        "smooth sinusoid features": "runs/magnitude_precision_anneal_v2/sinusoid_smooth_s*/eval.json",
        "xVal-style (value x one direction)": "runs/magnitude_precision_routes/xval_s*/eval.json",
        "ramp, numerals only (ours)": "runs/magnitude_precision_routes/grid_ramp_numonly_s*/eval.json",
    }),
    ("sum comparison: NUM_a + NUM_b OP NUM_c + NUM_d", {
        "learned table": "runs/numeral_codes_sum/learned_untied_s*/eval.json",
        "smooth sinusoid features": "runs/numeral_codes_sum/sinusoid_smooth_s*/eval.json",
        "xVal-style (value x one direction)": "runs/numeral_codes_sum/xval_s*/eval.json",
        "ramp, numerals only (ours)": "runs/numeral_codes_sum/grid_ramp_numonly_s*/eval.json",
        "control: xVal-style, lr 1e-3": "runs/numeral_codes_sum/xval_lr1e3_s*/eval.json",
        "control: ramp, lr 1e-3": "runs/numeral_codes_sum/grid_ramp_numonly_lr1e3_s*/eval.json",
        "control: ramp, warm start from the comparison task": "runs/numeral_codes_sum/grid_ramp_numonly_warm_s*/eval.json",
    }),
]


def escape_epoch(eval_path: str, threshold: float = 0.6) -> str:
    log = Path(eval_path).parent / "train.log"
    if not log.exists():
        return "?"
    vals = [float(line.split()[2]) for line in open(log) if line.strip().startswith("Val Loss:")]
    for i, v in enumerate(vals, 1):
        if v < threshold:
            return str(i)
    return "never"


def all_unseen(split: dict) -> dict:
    return split.get("by_gap_all_unseen") or split["by_gap_unseen2"]


def metrics(e: dict) -> dict:
    sp = e["splits"]
    au = lambda name: max(sp[name]["by_unseen"], key=int)
    g = lambda d, k: d[k]["acc"] if k in d and d[k]["n"] else float("nan")
    return {
        "seen": sp["test"]["overall"]["acc"],
        "seen_gap1": g(sp["test"]["by_gap"], "1"),
        "seen_gap2_5": g(sp["test"]["by_gap"], "2-5"),
        "held": sp["test_holdout"]["overall"]["acc"],
        "held_tf": sp["test_holdout"]["overall_tf"]["acc"],
        "held_all": sp["test_holdout"]["by_unseen"][au("test_holdout")]["acc"],
        "held_all_gap1": g(all_unseen(sp["test_holdout"]), "1"),
        "held_all_gap2_5": g(all_unseen(sp["test_holdout"]), "2-5"),
        "held_all_gap6_20": g(all_unseen(sp["test_holdout"]), "6-20"),
        "far": sp["test_far"]["overall"]["acc"],
        "far_tf": sp["test_far"]["overall_tf"]["acc"],
        "far_all": sp["test_far"]["by_unseen"][au("test_far")]["acc"],
        "epoch": e["epoch"],
    }


def fmt(vals, pct=True, sd=True):
    v = np.array([x for x in vals if x == x], dtype=float)
    if not len(v):
        return "n/a"
    k = 100.0 if pct else 1.0
    return f"{v.mean()*k:.1f}" + (f" ± {v.std()*k:.1f}" if sd and len(v) > 1 else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dump = {}
    for task, arms in SOURCES:
        print(f"\n### {task}\n")
        print("| numeral code | seeds | seen | seen gap 1 / 2-5 | held-out, all operands unseen (gap 1 / 2-5 / 6-20) | "
              "far, all unseen | held-out band | far band | TRUE/FALSE only: held-out / far | escape epoch per seed |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        dump[task] = {}
        for arm, pattern in arms.items():
            files = sorted(glob.glob(str(ROOT / pattern)))
            if not files:
                print(f"| {arm} | 0 | (not run yet) | | | | | | | |")
                continue
            ms = [metrics(json.load(open(f))) for f in files]
            esc = [escape_epoch(f) for f in files]
            col = lambda k: [m[k] for m in ms]
            dump[task][arm] = {"files": files, "per_seed": ms, "escape_epoch": esc}
            print(f"| {arm} | {len(ms)} | {fmt(col('seen'))} | {fmt(col('seen_gap1'), sd=False)} / {fmt(col('seen_gap2_5'), sd=False)} | "
                  f"{fmt(col('held_all'))} ({fmt(col('held_all_gap1'), sd=False)} / "
                  f"{fmt(col('held_all_gap2_5'), sd=False)} / {fmt(col('held_all_gap6_20'), sd=False)}) | "
                  f"{fmt(col('far_all'))} | {fmt(col('held'))} | {fmt(col('far'))} | "
                  f"{fmt(col('held_tf'), sd=False)} / {fmt(col('far_tf'), sd=False)} | {', '.join(esc)} |")
    if args.json:
        json.dump(dump, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
