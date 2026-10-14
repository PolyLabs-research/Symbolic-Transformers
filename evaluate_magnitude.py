#!/usr/bin/env python3
"""
Evaluate a checkpoint on the magnitude comparison task (data/magnitude_generator.py).

Reports, per split:
- acc        argmax over the full vocabulary equals the TRUE/FALSE target
- acc_tf     argmax restricted to {TRUE, FALSE} (ignores any mass on other tokens)
- by |a-b|   accuracy binned by the gap between the operands (adjacent values are hardest)
- by unseen  accuracy by how many operands were outside the training value pool
- P(TRUE) curve as a function of the signed difference in the operator's direction

and two embedding probes on the numeral rows of the model's embedding table:
- magnitude axis: Spearman correlation between the first principal component and k
- linear readout: fit k ~ w·e_k on training-pool values, then score Spearman/R² on
  held-out and far values. This asks directly whether magnitude is decodable for
  numerals the model never trained on.

Usage:
    python evaluate_magnitude.py --checkpoint runs/magnitude/grid_linear_s1/best_model.pt \
        --splits runs/magnitude/data/test.json runs/magnitude/data/test_holdout.json \
                 runs/magnitude/data/test_far.json --out eval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_grid import geometry_probe, spearman  # noqa: E402
from models.transformer import load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

GAP_BINS = [(0, 0), (1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
SIGNED_BINS = [(-10**9, -101), (-100, -21), (-20, -6), (-5, -2), (-1, -1), (0, 0),
               (1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]


def bin_label(lo: int, hi: int) -> str:
    if lo == hi:
        return str(lo)
    if hi >= 10**9:
        return f"{lo}+"
    if lo <= -10**9:
        return f"<={hi}"
    return f"{lo}-{hi}"


def gap_bin(gap: int) -> str:
    for lo, hi in GAP_BINS:
        if lo <= gap <= hi:
            return bin_label(lo, hi)
    raise ValueError(gap)


def signed_bin(diff: int) -> str:
    for lo, hi in SIGNED_BINS:
        if lo <= diff <= hi:
            return bin_label(lo, hi)
    raise ValueError(diff)


def signed_difference(sample: Dict) -> int:
    """Difference in the direction that makes the operator TRUE: positive means TRUE."""
    a, b, op = sample["a"], sample["b"], sample["op"]
    if op in ("LESS_THAN", "LESS_EQUAL"):
        return b - a
    if op in ("GREATER_THAN", "GREATER_EQUAL"):
        return a - b
    return -abs(a - b) if op == "EQUALS" else abs(a - b)


@torch.no_grad()
def evaluate_split(model, dataset: FOLDataset, vocab: Vocabulary, device: torch.device,
                   batch_size: int = 512, max_samples: Optional[int] = None) -> Dict:
    true_id, false_id = vocab.encode_label("TRUE"), vocab.encode_label("FALSE")
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    rows = []
    for start in range(0, n, batch_size):
        idxs = range(start, min(n, start + batch_size))
        items = [dataset[i] for i in idxs]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        lengths = torch.stack([it["length"] for it in items]).to(device)
        logits = model(input_ids)
        last = logits[torch.arange(input_ids.size(0), device=device), lengths - 1, :]
        pred_full = last.argmax(dim=-1).cpu().tolist()
        tf = torch.softmax(last[:, [true_id, false_id]], dim=-1)[:, 0].cpu().tolist()
        for j, i in enumerate(idxs):
            s = dataset.samples[i]
            target = int(s["target"])
            rows.append({
                "correct": int(pred_full[j] == target),
                "correct_tf": int((tf[j] >= 0.5) == (target == true_id)),
                "gap": abs(s["a"] - s["b"]),
                "n_unseen": int(s.get("n_unseen", 0)),
                "signed": signed_difference(s),
                "p_true": tf[j],
            })

    def acc(sel, key="correct"):
        sel = list(sel)
        return {"acc": float(np.mean([r[key] for r in sel])) if sel else float("nan"), "n": len(sel)}

    out = {
        "overall": acc(rows),
        "overall_tf": acc(rows, "correct_tf"),
        "by_gap": {bin_label(lo, hi): acc(r for r in rows if lo <= r["gap"] <= hi) for lo, hi in GAP_BINS},
        "by_unseen": {str(k): acc(r for r in rows if r["n_unseen"] == k) for k in (0, 1, 2)},
        # Gap breakdown restricted to pairs where BOTH operands are unseen: the
        # only comparison that cannot be answered by treating an unknown token
        # as "an average value" (a held-out band in the middle of the range
        # makes that guess right for most one-unseen pairs).
        "by_gap_unseen2": {bin_label(lo, hi): acc(r for r in rows if r["n_unseen"] == 2 and lo <= r["gap"] <= hi)
                           for lo, hi in GAP_BINS},
        "p_true_by_signed": {bin_label(lo, hi): {
            "p_true": float(np.mean([r["p_true"] for r in rows if lo <= r["signed"] <= hi]))
            if any(lo <= r["signed"] <= hi for r in rows) else float("nan"),
            "n": sum(1 for r in rows if lo <= r["signed"] <= hi)} for lo, hi in SIGNED_BINS},
    }
    return out


def magnitude_axis_probe(table: np.ndarray, pools: Dict[str, List[int]], ridge: float = 1e-2) -> Dict:
    """Is numeral magnitude encoded along a direction of the embedding table?

    table: (vocab, d) numeral rows are 0..624. pools: name -> list of values.
    """
    train_vals = np.array(pools["train"])
    E = table[: max(max(v) for v in pools.values()) + 1]
    out = {}

    # PC1 of the training-pool rows vs k
    X = E[train_vals] - E[train_vals].mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = E @ vt[0]
    out["pc1_spearman"] = {name: spearman(pc1[np.array(v)], np.array(v, dtype=float)) for name, v in pools.items() if v}

    # Ridge readout k ~ w·e + c fitted on the training pool, scored elsewhere
    Xtr = np.hstack([E[train_vals], np.ones((len(train_vals), 1))])
    ytr = train_vals.astype(np.float64)
    w = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    out["readout"] = {}
    for name, v in pools.items():
        if not v:
            continue
        v = np.array(v)
        pred = np.hstack([E[v], np.ones((len(v), 1))]) @ w
        ss_res = float(((v - pred) ** 2).sum())
        ss_tot = float(((v - v.mean()) ** 2).sum()) or 1.0
        out["readout"][name] = {"spearman": spearman(pred, v.astype(float)), "r2": 1 - ss_res / ss_tot,
                                "mae": float(np.mean(np.abs(v - pred)))}
    return out


def pools_from_metadata(meta: Dict) -> Dict[str, List[int]]:
    vp = meta["value_pools"]
    lo, hi = vp["train"]
    hold = vp.get("holdout")
    far = vp.get("far")
    hold_set = set(range(hold[0], hold[1] + 1)) if hold else set()
    pools = {"train": [k for k in range(lo, hi + 1) if k not in hold_set]}
    if hold:
        pools["holdout"] = list(range(hold[0], hold[1] + 1))
    if far:
        pools["far"] = list(range(far[0], far[1] + 1))
    return pools


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the magnitude comparison task")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--metadata", default=None, help="metadata.json (default: next to the first split)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    vocab = Vocabulary(args.vocab)
    model, ckpt = load_model_from_checkpoint(args.checkpoint, vocab.vocab_size, device)
    cfg = ckpt.get("config", {})
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0

    meta_path = Path(args.metadata) if args.metadata else Path(args.splits[0]).parent / "metadata.json"
    pools = pools_from_metadata(json.load(open(meta_path))) if meta_path.exists() else {"train": list(range(625))}

    results = {
        "checkpoint": args.checkpoint,
        "embedding_type": cfg.get("embedding_type", "learned"),
        "tie_weights": bool(cfg.get("tie_weights", False)),
        "epoch": ckpt.get("epoch"), "best_val_loss": ckpt.get("best_val_loss"),
        "params": model.count_parameters(),
        "splits": {},
        "geometry": geometry_probe(model, vocab),
    }
    with torch.no_grad():
        table = model.embedding_table().detach().cpu().numpy()
    results["magnitude_axis"] = magnitude_axis_probe(table, pools)

    for path in args.splits:
        ds = FOLDataset(path, pad_id=pad_id)
        results["splits"][Path(path).stem] = evaluate_split(model, ds, vocab, device, args.batch_size, args.max_samples)

    tag = f"{results['embedding_type']}{' (tied)' if results['tie_weights'] else ''}"
    print(f"\n{tag}  epoch {results['epoch']}  val {results['best_val_loss']}")
    gaps = [bin_label(lo, hi) for lo, hi in GAP_BINS]
    print("| split | acc | acc_tf | " + " | ".join(f"gap {g}" for g in gaps) + " | 1 unseen | 2 unseen |")
    print("|" + "---|" * (len(gaps) + 5))
    for name, r in results["splits"].items():
        print(f"| {name} | {r['overall']['acc']*100:.1f} | {r['overall_tf']['acc']*100:.1f} | "
              + " | ".join(f"{r['by_gap'][g]['acc']*100:.0f}" for g in gaps)
              + f" | {r['by_unseen']['1']['acc']*100:.1f} | {r['by_unseen']['2']['acc']*100:.1f} |")
    ma = results["magnitude_axis"]
    print("\nMagnitude axis: PC1 spearman", {k: round(v, 3) for k, v in ma["pc1_spearman"].items()})
    print("Linear readout of k fitted on train values:",
          {k: f"spearman {v['spearman']:.3f}, R2 {v['r2']:.2f}" for k, v in ma["readout"].items()})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
