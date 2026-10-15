#!/usr/bin/env python3
"""Decision curves from evaluator row dumps on a psychometric split (scripts/psychometric_split.py).

For each dump, comparison value c and operator, P(TRUE | NUM_k OP NUM_c) as a function of k gives the
model's decision curve around c. Reported per (model, c, op): accuracy over all k, the transition
width (number of k with 0.25 < P(TRUE) < 0.75), the largest |k - c| at which the model is wrong,
and the fitted threshold (the k where the curve crosses 0.5, by linear interpolation). Prints a
table and optionally writes a JSON with the full curves.

usage: psychometric_curves.py DUMP.jsonl [DUMP.jsonl ...] [--out curves.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    curves = defaultdict(dict)   # (c, op) -> {k: (p_true, correct)}
    for line in open(path):
        r = json.loads(line)
        curves[(r["b"], r["op"])][r["a"]] = (r["p_true"], r["correct"])
    return curves


POOLS = {"train": lambda k: k < 350 or 450 <= k < 500, "holdout": lambda k: 350 <= k < 450, "far": lambda k: k >= 500}


def analyse(curve: dict, c: int, op: str) -> dict:
    ks = np.array(sorted(curve)); p = np.array([curve[k][0] for k in ks]); ok = np.array([curve[k][1] for k in ks])
    wrong = ks[ok == 0]
    errors_by_pool = {name: int(sum(1 for k in wrong if f(k))) for name, f in POOLS.items()}
    # crossing of 0.5, interpolated on the first sign change nearest to c
    diff = p - 0.5
    cross = None
    for i in range(len(ks) - 1):
        if diff[i] == 0 or diff[i] * diff[i + 1] < 0:
            x = ks[i] + (0 - diff[i]) / (diff[i + 1] - diff[i]) * (ks[i + 1] - ks[i]) if diff[i + 1] != diff[i] else ks[i]
            if cross is None or abs(x - c) < abs(cross - c):
                cross = float(x)
    return {"c": c, "op": op, "acc": float(ok.mean()),
            "transition_width": int(((p > 0.25) & (p < 0.75)).sum()),
            "max_wrong_distance": int(np.abs(wrong - c).max()) if len(wrong) else 0,
            "n_wrong": int(len(wrong)), "errors_by_pool": errors_by_pool, "threshold": cross,
            "curve": {"k": ks.tolist(), "p_true": p.tolist()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = {}
    print("| model | c | op | acc (%) | transition width (k) | furthest error (|k-c|) | errors train/holdout/far | threshold |")
    print("|---|---|---|---|---|---|---|---|")
    for d in args.dumps:
        path = Path(d); name = path.stem
        res = []
        for (c, op), curve in sorted(load(path).items()):
            r = analyse(curve, c, op); res.append(r)
            thr = "n/a" if r["threshold"] is None else f"{r['threshold']:.1f}"
            e = r["errors_by_pool"]
            print(f"| {name} | {c} | {op} | {r['acc']*100:.1f} | {r['transition_width']} | {r['max_wrong_distance']} | {e['train']}/{e['holdout']}/{e['far']} | {thr} |")
        out[name] = res
    if args.out:
        json.dump(out, open(args.out, "w"))
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
