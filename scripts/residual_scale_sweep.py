#!/usr/bin/env python3
"""Readability curves: evaluate saved models with the grid_shared residual re-applied at several scales.

A comparator that reads the ramp is flat across scales; one that still reads the residual falls off
below some scale; a model with no residual is evaluated once. Writes a JSON of every number and prints
one table per split (held-out both-unseen accuracy by default).

usage: residual_scale_sweep.py CKPT [CKPT ...] --splits test.json test_holdout.json test_far.json
                               [--scales 0 1e-6 ...] [--metadata metadata.json] [--out sweep.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_magnitude import evaluate_split, pools_from_metadata, saved_residual_scale, set_residual_scale  # noqa: E402
from models.transformer import load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

DEFAULT_SCALES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]


def label_for(path: Path) -> str:
    """runs/x/arm_s1/best_model.pt -> x/arm_s1; runs/x/arm_s1/checkpoint_epoch_52.pt -> x/arm_s1@52."""
    run = path.parent
    tag = f"{run.parent.name}/{run.name}"
    if path.stem.startswith("checkpoint_epoch_"):
        tag += "@" + path.stem.rsplit("_", 1)[-1]
    return tag


def summarize(split_result: dict) -> dict:
    g = lambda d, k: float(d[k]["acc"]) if k in d else float("nan")
    return {
        "acc": float(split_result["overall"]["acc"]),
        "unseen2": g(split_result["by_unseen"], "2"),
        "unseen2_gap1": g(split_result["by_gap_unseen2"], "1"),
        "unseen2_gap2_5": g(split_result["by_gap_unseen2"], "2-5"),
        "unseen2_gap6_20": g(split_result["by_gap_unseen2"], "6-20"),
        "gap1": g(split_result["by_gap"], "1"),
        "gap2_5": g(split_result["by_gap"], "2-5"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    ap.add_argument("--vocab", default=str(ROOT / "unified_vocabulary.json"))
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="unseen2", choices=["acc", "unseen2", "unseen2_gap1", "unseen2_gap2_5", "unseen2_gap6_20", "gap1", "gap2_5"],
                    help="which number the printed tables show (the JSON has all of them)")
    args = ap.parse_args()

    device = torch.device(args.device)
    vocab = Vocabulary(args.vocab)
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    datasets = {Path(p).stem: FOLDataset(p, pad_id=pad_id) for p in args.splits}
    scales = sorted(set(args.scales))

    results = {"scales": scales, "splits": list(datasets), "models": {}}
    for ck in args.checkpoints:
        path = Path(ck)
        model, ckpt = load_model_from_checkpoint(str(path), vocab.vocab_size, device, verbose=False)
        model.eval()
        saved = saved_residual_scale(model)
        entry = {"checkpoint": str(path), "epoch": ckpt.get("epoch"), "saved_scale": saved, "per_scale": {}}
        eval_scales = scales if saved is not None else [None]
        for s in eval_scales:
            if s is not None:
                set_residual_scale(model, s)
            with torch.no_grad():
                per_split = {name: summarize(evaluate_split(model, ds, vocab, device, args.batch_size, args.max_samples))
                             for name, ds in datasets.items()}
            entry["per_scale"]["none" if s is None else repr(float(s))] = per_split
            print(f"  {label_for(path)} scale {'n/a' if s is None else f'{s:g}'}: "
                  + " | ".join(f"{n} {r['acc']*100:.1f}" for n, r in per_split.items()), flush=True)
        results["models"][label_for(path)] = entry

    for name in datasets:
        print(f"\n### {name}: {args.metric} (%) by re-applied residual scale\n")
        print("| model | saved | " + " | ".join(f"{s:g}" for s in scales) + " |")
        print("|---|---|" + "---|" * len(scales))
        for label, entry in results["models"].items():
            if entry["saved_scale"] is None:
                v = entry["per_scale"]["none"][name][args.metric] * 100
                print(f"| {label} | no residual | " + " | ".join(f"{v:.1f}" for _ in scales) + " |")
                continue
            cells = [f"{entry['per_scale'][repr(float(s))][name][args.metric]*100:.1f}" for s in scales]
            print(f"| {label} | {entry['saved_scale']:g} | " + " | ".join(cells) + " |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
