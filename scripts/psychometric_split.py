#!/usr/bin/env python3
"""Write a psychometric split: every value k = 0..624 against a few fixed comparison values c.

Evaluating it with evaluate_magnitude.py --dump-rows gives P(TRUE | NUM_k LESS_THAN NUM_c) as a
function of k, i.e. the model's decision curve around c. With c inside the held-out band or past
the trained range, the sharpness of the step at a never-seen c is the readable picture.

usage: psychometric_split.py --metadata runs/.../data/metadata.json --out runs/.../data/test_psycho.json
                             [--values 375 425 520 580] [--ops LESS_THAN GREATER_THAN]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.magnitude_generator import MagnitudeGenerator  # noqa: E402
from evaluate_magnitude import pools_from_metadata  # noqa: E402
from utils.grid_bank import GRID_CELLS  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--values", nargs="+", type=int, default=None,
                    help="comparison values c (default: the quartiles of the held-out band and of the far band)")
    ap.add_argument("--vocab", default=None, help="vocabulary path (default: the metadata's, resolved against the repository root)")
    ap.add_argument("--ops", nargs="+", default=["LESS_THAN", "GREATER_THAN"])
    args = ap.parse_args()

    meta = json.load(open(args.metadata))
    vp = Path(args.vocab or meta.get("vocab_path", "unified_vocabulary.json"))
    vocab = Vocabulary(str(vp if vp.is_absolute() else ROOT / vp))
    pools = pools_from_metadata(meta)
    train_pool = set(pools["train"])
    bands = {name: (min(v), max(v)) for name, v in pools.items() if v and name != "train"}
    if args.values is None:
        args.values = []
        for name in ("holdout", "far"):
            if name in bands:
                lo, hi = bands[name]
                args.values += [lo + (hi - lo) // 4, lo + 3 * (hi - lo) // 4]
        if not args.values:
            ap.error("no held-out or far band in the metadata; pass --values")
    lo_n, hi_n = vocab.numeral_range
    bad = [c for c in args.values if not lo_n <= c <= hi_n]
    if bad:
        ap.error(f"values must be numerals in {lo_n}..{hi_n}; got {bad}")
    for c in args.values:
        where = next((name for name, (lo, hi) in bands.items() if lo <= c <= hi), "train")
        print(f"  c = {c}: {where} band" + (" (a trained value, not a never-seen threshold)" if where == "train" else ""))
    gen = MagnitudeGenerator(vocab, ops=args.ops)
    samples = [gen.make_sample(k, c, op, train_pool) for c in args.values for op in args.ops for k in range(GRID_CELLS)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"num_samples": len(samples), "max_seq_len": 5, "samples": samples,
               "psychometric": {"values": args.values, "ops": args.ops}}, open(args.out, "w"))
    print(f"Wrote {len(samples)} pairs ({len(args.values)} values x {len(args.ops)} ops x {GRID_CELLS} k) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
