#!/usr/bin/env python3
"""Readability curves: evaluate saved models with the grid_shared residual re-applied at several scales.

A comparator that reads the ramp is flat across scales; one that still reads the residual falls off
below some scale; a model with no residual is evaluated once. --rows chooses which grid rows take the
override (the operator and answer tokens are grid panels through the same encoder; with --rows numerals
they keep the saved scale, which separates numeral readability from operator separability). The JSON
records every number plus, per model and scale, the residual's share of the normalised numeral
embeddings and of the four operator/answer rows, so curves can be plotted against share as well as
scale. Printed tables parenthesise cells above the saved scale (a comparator was never trained there).

usage: residual_scale_sweep.py CKPT [CKPT ...] --splits test.json test_holdout.json test_far.json
                               [--rows all|numerals|glyphs] [--scales 0 1e-6 ...] [--out sweep.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_magnitude import ROW_CHOICES, clear_residual_row_scale, evaluate_split, saved_residual_scale, set_residual_scale  # noqa: E402
from models.transformer import load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

DEFAULT_SCALES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
METRICS = ["acc", "acc_tf", "unseen2", "unseen2_gap1", "unseen2_gap2_5", "unseen2_gap6_20", "gap1", "gap2_5"]


def label_for(path: Path) -> str:
    """runs/x/arm_s1/best_model.pt -> x/arm_s1; a periodic checkpoint adds @epoch; other stems are kept."""
    path = path.resolve()
    run = path.parent
    tag = f"{run.parent.name}/{run.name}"
    if path.stem.startswith("checkpoint_epoch_"):
        tag += "@" + path.stem.rsplit("_", 1)[-1]
    elif path.stem != "best_model":
        tag += ":" + path.stem
    return tag


def summarize(split_result: dict) -> dict:
    """The "unseen2" metrics mean every operand unseen (two of two on the comparison task, four of four on the
    sum task); the names are kept for the existing sweep JSON files."""
    g = lambda d, k: float(d[k]["acc"]) if k in d else float("nan")
    all_key = max(split_result["by_unseen"], key=int)
    au = split_result.get("by_gap_all_unseen") or split_result["by_gap_unseen2"]
    return {
        "acc": float(split_result["overall"]["acc"]),
        "acc_tf": float(split_result["overall_tf"]["acc"]),
        "unseen2": g(split_result["by_unseen"], all_key),
        "unseen2_gap1": g(au, "1"),
        "unseen2_gap2_5": g(au, "2-5"),
        "unseen2_gap6_20": g(au, "6-20"),
        "gap1": g(split_result["by_gap"], "1"),
        "gap2_5": g(split_result["by_gap"], "2-5"),
    }


def residual_share(model, vocab: Vocabulary, scale: float) -> dict:
    """Median over rows of |scale * residual part| / |generated part + bias| before LayerNorm, for the
    numeral rows and for the operator/answer rows (LESS_THAN, GREATER_THAN, TRUE, FALSE)."""
    emb = model.symbol_embed
    enc = emb.encoder
    with torch.no_grad():
        bits = emb.panels()
        gen = bits @ (enc.phi @ enc.proj.weight.t()) + enc.proj.bias
        res = scale * (bits @ enc.effective_residual(unscaled=True))
        ratio = res.norm(dim=1) / gen.norm(dim=1).clamp_min(1e-12)
    lo, hi = vocab.numeral_range
    glyphs = [vocab.encode_label(t) for t in ("LESS_THAN", "GREATER_THAN", "TRUE", "FALSE")]
    return {"numerals": float(ratio[lo:hi + 1].median()), "glyphs": float(ratio[glyphs].median())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES, help="residual scales in [0, 1]")
    ap.add_argument("--rows", default="all", choices=ROW_CHOICES, help="rows the override applies to (default all)")
    ap.add_argument("--vocab", default=str(ROOT / "unified_vocabulary.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="unseen2", choices=METRICS,
                    help="which number the printed tables show (the JSON has all of them); a split where it is "
                         "undefined for every model falls back to acc")
    args = ap.parse_args()
    bad = [s for s in args.scales if not 0.0 <= s <= 1.0]
    if bad:
        ap.error(f"--scales must lie in [0, 1]; got {bad}")
    stems = [Path(p).stem for p in args.splits]
    if len(set(stems)) != len(stems):
        ap.error(f"split file names must be distinct (got {stems})")
    labels = [label_for(Path(c)) for c in args.checkpoints]
    dups = sorted({l for l in labels if labels.count(l) > 1})
    if dups:
        ap.error(f"checkpoints map to the same label: {dups}")

    device = torch.device(args.device)
    vocab = Vocabulary(args.vocab)
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    datasets = {Path(p).stem: FOLDataset(p, pad_id=pad_id) for p in args.splits}
    scales = sorted(set(args.scales))

    results = {"scales": scales, "rows": args.rows, "splits": list(datasets), "models": {}}
    for ck, label in zip(args.checkpoints, labels):
        path = Path(ck)
        model, ckpt = load_model_from_checkpoint(str(path), vocab.vocab_size, device, verbose=False)
        model.eval()
        saved = saved_residual_scale(model)
        entry = {"checkpoint": str(path), "epoch": ckpt.get("epoch"), "saved_scale": saved, "rows": args.rows,
                 "per_scale": {}, "share": {}}
        eval_scales = sorted(set(scales) | {saved}) if saved is not None else [None]
        for s in eval_scales:
            if s is not None:
                clear_residual_row_scale(model)
                set_residual_scale(model, s, args.rows, vocab)
                entry["share"][repr(float(s))] = residual_share(model, vocab, s)
            with torch.no_grad():
                per_split = {name: summarize(evaluate_split(model, ds, vocab, device, args.batch_size, args.max_samples))
                             for name, ds in datasets.items()}
            entry["per_scale"]["none" if s is None else repr(float(s))] = per_split
            print(f"  {label} scale {'n/a' if s is None else f'{s:g}'}: "
                  + " | ".join(f"{n} {r['acc']*100:.1f}" for n, r in per_split.items()), flush=True)
        results["models"][label] = entry

    for name in datasets:
        metric = args.metric
        vals = [e["per_scale"][k][name][metric] for e in results["models"].values() for k in e["per_scale"]]
        if all(np.isnan(v) for v in vals):
            metric = "acc"
        print(f"\n### {name}: {metric} (%) by re-applied residual scale ({args.rows} rows); cells above the saved scale in parentheses\n")
        print("| model | saved | " + " | ".join(f"{s:g}" for s in scales) + " |")
        print("|---|---|" + "---|" * len(scales))
        for label, entry in results["models"].items():
            if entry["saved_scale"] is None:
                v = entry["per_scale"]["none"][name][metric] * 100
                print(f"| {label} | no residual | " + " | ".join(f"{v:.1f}" for _ in scales) + " |")
                continue
            cells = []
            for s in scales:
                v = entry["per_scale"][repr(float(s))][name][metric] * 100
                cells.append(f"({v:.1f})" if s > entry["saved_scale"] else f"{v:.1f}")
            print(f"| {label} | {entry['saved_scale']:g} | " + " | ".join(cells) + " |")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
