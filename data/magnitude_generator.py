#!/usr/bin/env python3
"""
Magnitude comparison task: does the model know that 17 < 42?

Each sample is one five-token context and a one-token answer:

    MESSAGE_START  NUM_a  LESS_THAN  NUM_b  ANSWER   ->   TRUE | FALSE

Numerals are bare numeral tokens (ids 0-624), the operator is one of the
vocabulary's relation tokens, and the target is TRUE or FALSE. Nothing in FOL
syntax ever depends on which of two indices is larger, so this is the first
task where the numeral embedding has to carry *magnitude*. The base-625 panel
of numeral k is the first k cells filled (a thermometer code), so a grid
embedding gets that ordering for free; a learned table has to infer it from
the pairs it sees and cannot know it for values it never saw.

Splits (values = numerals that may appear as operands):

    train, val, test   operands from the training pool; val/test pairs are
                       disjoint from train pairs
    test_holdout       at least one operand from the held-out band (values in
                       range, never seen in training)
    test_far           at least one operand from the far band (beyond the
                       training range)

Every sample records a, b, op, label and n_unseen (0/1/2 operands outside the
training pool) so evaluate_magnitude.py can bin accuracy by |a-b| and by how
many operands were unseen.

Usage:
    python data/magnitude_generator.py --out runs/magnitude/data \
        --value-range 0-499 --holdout 200-299 --far-range 500-624
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.vocabulary import Vocabulary  # noqa: E402

OPS = {
    "LESS_THAN": lambda a, b: a < b,
    "GREATER_THAN": lambda a, b: a > b,
    "LESS_EQUAL": lambda a, b: a <= b,
    "GREATER_EQUAL": lambda a, b: a >= b,
    "EQUALS": lambda a, b: a == b,
    "NOT_EQUALS": lambda a, b: a != b,
}
DEFAULT_OPS = ("LESS_THAN", "GREATER_THAN")


def parse_band(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if text is None:
        return None
    lo, hi = text.split("-")
    return int(lo), int(hi)


class MagnitudeGenerator:
    def __init__(self, vocab: Vocabulary, ops: Sequence[str] = DEFAULT_OPS, p_equal: float = 0.02):
        self.vocab = vocab
        self.ops = list(ops)
        for op in self.ops:
            if op not in OPS:
                raise ValueError(f"unknown op {op}; choose from {list(OPS)}")
            vocab.encode_label(op)  # raises if the vocabulary lacks it
        self.p_equal = p_equal
        self.ms = vocab.message_start
        self.answer = vocab.answer_token
        self.true_id = vocab.encode_label("TRUE")
        self.false_id = vocab.encode_label("FALSE")
        if self.ms is None or self.answer is None:
            raise ValueError("vocabulary needs MESSAGE_START and ANSWER tokens")

    def make_sample(self, a: int, b: int, op: str, train_pool: Set[int]) -> Dict:
        label = OPS[op](a, b)
        return {
            "context": [self.ms, a, self.vocab.encode_label(op), b, self.answer],
            "target": self.true_id if label else self.false_id,
            "a": a, "b": b, "op": op, "label": "TRUE" if label else "FALSE",
            "n_unseen": int(a not in train_pool) + int(b not in train_pool),
        }

    def sample_pairs(self, n: int, pool_a: List[int], pool_b: List[int], train_pool: Set[int],
                     exclude: Set[Tuple[int, int, str]], require_unseen: bool = False) -> List[Dict]:
        """Draw n distinct (a, b, op) triples; a from pool_a, b from pool_b, order randomised."""
        seen: Set[Tuple[int, int, str]] = set()
        out: List[Dict] = []
        attempts = 0
        max_attempts = n * 50
        while len(out) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"could not draw {n} distinct pairs (got {len(out)}); pools too small?")
            a = random.choice(pool_a)
            b = a if random.random() < self.p_equal else random.choice(pool_b)
            if random.random() < 0.5:
                a, b = b, a
            op = random.choice(self.ops)
            key = (a, b, op)
            if key in seen or key in exclude:
                continue
            sample = self.make_sample(a, b, op, train_pool)
            if require_unseen and sample["n_unseen"] == 0:
                continue
            seen.add(key)
            out.append(sample)
        return out


def _save(path: Path, samples: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump({"num_samples": len(samples), "max_seq_len": 5, "samples": samples}, f)
    n_true = sum(1 for s in samples if s["label"] == "TRUE")
    print(f"✓ {path.name}: {len(samples)} samples, {n_true / max(1, len(samples)) * 100:.1f}% TRUE")


def generate_magnitude_data(vocab_path: str, output_dir: str, n_train: int = 20000, n_val: int = 2000,
                            n_test: int = 4000, value_range: Tuple[int, int] = (0, 499),
                            holdout: Optional[Tuple[int, int]] = (200, 299),
                            far_range: Optional[Tuple[int, int]] = (500, 624),
                            ops: Sequence[str] = DEFAULT_OPS, p_equal: float = 0.02, seed: int = 42) -> Dict:
    random.seed(seed)
    vocab = Vocabulary(vocab_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hi_all = vocab.numeral_range[1]

    def band(lo_hi):
        if lo_hi is None:
            return []
        lo, hi = lo_hi
        if not (0 <= lo <= hi <= hi_all):
            raise ValueError(f"band {lo_hi} outside 0..{hi_all}")
        return list(range(lo, hi + 1))

    holdout_vals = band(holdout)
    far_vals = band(far_range)
    train_vals = [v for v in band(value_range) if v not in set(holdout_vals)]
    if set(far_vals) & set(band(value_range)):
        raise ValueError("far_range overlaps value_range")
    if not train_vals:
        raise ValueError("training value pool is empty")
    train_pool = set(train_vals)

    gen = MagnitudeGenerator(vocab, ops=ops, p_equal=p_equal)
    print("=" * 60)
    print("MAGNITUDE COMPARISON DATASET")
    print("=" * 60)
    print(f"ops={list(ops)}  train values={value_range} minus {holdout}  far={far_range}")

    train = gen.sample_pairs(n_train, train_vals, train_vals, train_pool, exclude=set())
    train_keys = {(s["a"], s["b"], s["op"]) for s in train}
    val = gen.sample_pairs(n_val, train_vals, train_vals, train_pool, exclude=train_keys)
    val_keys = train_keys | {(s["a"], s["b"], s["op"]) for s in val}
    test = gen.sample_pairs(n_test, train_vals, train_vals, train_pool, exclude=val_keys)
    _save(out / "train.json", train)
    _save(out / "val.json", val)
    _save(out / "test.json", test)

    splits = {"train": len(train), "val": len(val), "test": len(test)}
    if holdout_vals:
        mixed = train_vals + holdout_vals
        th = gen.sample_pairs(n_test, holdout_vals, mixed, train_pool, exclude=set(), require_unseen=True)
        _save(out / "test_holdout.json", th)
        splits["test_holdout"] = len(th)
    if far_vals:
        mixed = train_vals + far_vals
        tf = gen.sample_pairs(n_test, far_vals, mixed, train_pool, exclude=set(), require_unseen=True)
        _save(out / "test_far.json", tf)
        splits["test_far"] = len(tf)

    metadata = {
        "task": "magnitude_comparison_v1",
        "vocab_path": vocab_path,
        "vocab_size": vocab.vocab_size,
        "ops": list(ops),
        "p_equal": p_equal,
        "seed": seed,
        "value_pools": {
            "train": [value_range[0], value_range[1]],
            "holdout": [holdout[0], holdout[1]] if holdout else None,
            "far": [far_range[0], far_range[1]] if far_range else None,
            "train_pool_size": len(train_vals),
        },
        "splits": splits,
        "format": "MESSAGE_START NUM_a OP NUM_b ANSWER -> TRUE|FALSE",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Wrote {out}")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the magnitude comparison dataset")
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--out", default="datasets/magnitude")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--value-range", default="0-499", metavar="LO-HI")
    ap.add_argument("--holdout", default="200-299", metavar="LO-HI", help="held-out value band ('none' to disable)")
    ap.add_argument("--far-range", default="500-624", metavar="LO-HI", help="never-seen value band ('none' to disable)")
    ap.add_argument("--ops", nargs="+", default=list(DEFAULT_OPS), choices=list(OPS))
    ap.add_argument("--p-equal", type=float, default=0.02, help="fraction of pairs with a == b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_magnitude_data(
        args.vocab, args.out, args.n_train, args.n_val, args.n_test,
        value_range=parse_band(args.value_range),
        holdout=None if args.holdout.lower() == "none" else parse_band(args.holdout),
        far_range=None if args.far_range.lower() == "none" else parse_band(args.far_range),
        ops=args.ops, p_equal=args.p_equal, seed=args.seed,
    )
