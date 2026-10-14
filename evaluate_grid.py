#!/usr/bin/env python3
"""
Split-aware evaluation for the "do the grids matter" experiment.

For each dataset split it reports next-token accuracy broken down by what the
target is, because index identity only matters in a few places in this grammar:

- overall        every sample
- structure      target is an operator / boundary token
- after_numeral  structure targets whose previous token is a numeral (does an
                 unfamiliar index derail the syntax?)
- numeral        target is a numeral
- copy           numeral target that already appeared in the context right
                 after VAR, i.e. the bound variable is being reused. This is the
                 one place the model must *produce* a specific index, and the
                 only numeral prediction that is not chance for unseen indices.
- fresh          numeral target not seen earlier in the context (chance level
                 by construction; reported for completeness)

It also probes the embedding geometry of the numeral tokens (does embedding
distance track numeric distance?) and, for grid models, re-runs the splits with
random bit flips applied to the panels.

Usage:
    python evaluate_grid.py --checkpoint runs/grid_linear/best_model.pt \
        --splits datasets/grid_holdout/test.json datasets/grid_holdout/test_holdout.json \
        --noise-flip 0.01 0.05 --out results.json
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

from models.transformer import GridEmbedding, load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

CATEGORIES = ("overall", "structure", "after_numeral", "numeral", "copy", "fresh")


def categorize(context: List[int], target: int, vocab: Vocabulary, var_id: int) -> List[str]:
    """Which accuracy buckets a (context, target) pair belongs to."""
    cats = ["overall"]
    if vocab.is_numeral(target):
        cats.append("numeral")
        bound_before = any(context[i] == var_id and context[i + 1] == target for i in range(len(context) - 1))
        cats.append("copy" if bound_before else "fresh")
    else:
        cats.append("structure")
        if context and vocab.is_numeral(context[-1]):
            cats.append("after_numeral")
    return cats


@torch.no_grad()
def evaluate_split(model, dataset: FOLDataset, vocab: Vocabulary, device: torch.device,
                   batch_size: int = 256, max_samples: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    var_id = vocab.compositional_tokens["VAR"]
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    hits = {c: 0 for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    for start in range(0, n, batch_size):
        idxs = range(start, min(n, start + batch_size))
        items = [dataset[i] for i in idxs]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        lengths = torch.stack([it["length"] for it in items]).to(device)
        targets = torch.stack([it["target"] for it in items]).to(device)
        logits = model(input_ids)
        last = logits[torch.arange(input_ids.size(0), device=device), lengths - 1, :]
        preds = last.argmax(dim=-1)
        correct = (preds == targets).cpu().tolist()
        for j, i in enumerate(idxs):
            ctx = dataset.samples[i]["context"]
            tgt = int(dataset.samples[i]["target"])
            for c in categorize(ctx, tgt, vocab, var_id):
                counts[c] += 1
                hits[c] += int(correct[j])
    return {c: {"acc": (hits[c] / counts[c]) if counts[c] else float("nan"), "n": counts[c]} for c in CATEGORIES}


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), no scipy needed."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    uniq, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=ranks)
    cnts = np.bincount(inv)
    return (sums / cnts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def geometry_probe(model, vocab: Vocabulary, ranges=((0, 99), (0, 624)), n_pairs: int = 4000, seed: int = 0) -> Dict:
    """Does distance between numeral embeddings track |i - j|?"""
    table = model.embedding_table().detach().cpu().numpy()
    lo_all, hi_all = vocab.numeral_range
    rng = np.random.default_rng(seed)
    out = {}
    for lo, hi in ranges:
        hi = min(hi, hi_all)
        emb = table[lo:hi + 1]
        i = rng.integers(0, hi - lo + 1, size=n_pairs)
        j = rng.integers(0, hi - lo + 1, size=n_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        dist = np.linalg.norm(emb[i] - emb[j], axis=1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        cos = np.sum((emb[i] / norms[i]) * (emb[j] / norms[j]), axis=1)
        gap = np.abs(i - j).astype(np.float64)
        # Nearest neighbour (excluding self): is it an adjacent index?
        d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_idx = d2.argmin(axis=1)
        adjacent = float(np.mean(np.abs(nn_idx - np.arange(len(emb))) == 1))
        out[f"{lo}-{hi}"] = {
            "spearman_dist_vs_gap": spearman(dist, gap),
            "spearman_cos_vs_gap": spearman(-cos, gap),
            "nearest_neighbour_is_adjacent": adjacent,
            "n_pairs": int(len(i)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Split-aware evaluation for grid-embedding experiments")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--splits", nargs="+", required=True, help="Dataset JSON files to evaluate")
    ap.add_argument("--noise-flip", nargs="*", type=float, default=[],
                    help="Panel bit-flip probabilities to re-evaluate with (grid models only)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="Write results JSON here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    vocab = Vocabulary(args.vocab)
    model, ckpt = load_model_from_checkpoint(args.checkpoint, vocab.vocab_size, device)
    cfg = ckpt.get("config", {})
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0

    results = {
        "checkpoint": args.checkpoint,
        "embedding_type": cfg.get("embedding_type", "learned"),
        "tie_weights": bool(cfg.get("tie_weights", False)),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "params": model.count_parameters(),
        "splits": {},
        "geometry": geometry_probe(model, vocab, seed=args.seed),
        "noise": {},
    }

    datasets = {}
    for path in args.splits:
        name = Path(path).stem
        datasets[name] = FOLDataset(path, pad_id=pad_id)
        results["splits"][name] = evaluate_split(model, datasets[name], vocab, device,
                                                 args.batch_size, args.max_samples)

    grid_embed = model.symbol_embed if isinstance(model.symbol_embed, GridEmbedding) else None
    for p in args.noise_flip:
        if grid_embed is None:
            print(f"(skipping noise {p}: not a grid model)")
            continue
        torch.manual_seed(args.seed)
        grid_embed.noise_flip = p
        results["noise"][str(p)] = {name: evaluate_split(model, ds, vocab, device, args.batch_size, args.max_samples)
                                    for name, ds in datasets.items()}
        grid_embed.noise_flip = 0.0

    # Report
    print(f"\n{results['embedding_type']}{' (tied)' if results['tie_weights'] else ''}  "
          f"epoch {results['epoch']}  val {results['best_val_loss']}")
    header = "| split | " + " | ".join(CATEGORIES) + " |"
    print(header)
    print("|" + "---|" * (len(CATEGORIES) + 1))
    for name, res in results["splits"].items():
        print(f"| {name} | " + " | ".join(f"{res[c]['acc']*100:.1f} (n={res[c]['n']})" for c in CATEGORIES) + " |")
    for p, per_split in results["noise"].items():
        for name, res in per_split.items():
            print(f"| {name} @flip {p} | " + " | ".join(f"{res[c]['acc']*100:.1f}" for c in CATEGORIES) + " |")
    print("\nGeometry (numeral embeddings):")
    for rng_name, g in results["geometry"].items():
        print(f"  {rng_name}: spearman(dist, |i-j|)={g['spearman_dist_vs_gap']:.3f}  "
              f"spearman(1-cos, |i-j|)={g['spearman_cos_vs_gap']:.3f}  "
              f"nearest-neighbour adjacent={g['nearest_neighbour_is_adjacent']*100:.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
