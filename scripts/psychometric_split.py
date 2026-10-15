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
    ap.add_argument("--values", nargs="+", type=int, default=[375, 425, 520, 580])
    ap.add_argument("--ops", nargs="+", default=["LESS_THAN", "GREATER_THAN"])
    args = ap.parse_args()

    meta = json.load(open(args.metadata))
    vocab = Vocabulary(meta.get("vocab_path", str(ROOT / "unified_vocabulary.json")))
    train_pool = set(pools_from_metadata(meta)["train"])
    gen = MagnitudeGenerator(vocab, ops=args.ops)
    samples = [gen.make_sample(k, c, op, train_pool) for c in args.values for op in args.ops for k in range(GRID_CELLS)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"num_samples": len(samples), "max_seq_len": 5, "samples": samples,
               "psychometric": {"values": args.values, "ops": args.ops}}, open(args.out, "w"))
    print(f"Wrote {len(samples)} pairs ({len(args.values)} values x {len(args.ops)} ops x {GRID_CELLS} k) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
