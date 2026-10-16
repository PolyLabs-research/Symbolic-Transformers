#!/usr/bin/env python3
"""Assess formula-evaluation models (Stage A data from data/eval_generator.py).

    python scripts/assess_eval.py --data datasets/eval_stage_a \\
        --checkpoint direct=runs/eval_a_direct/best_model.pt:direct \\
        --checkpoint scratch=runs/eval_a_scratch/best_model.pt:scratch --json experiments/eval_stage_a.json

For every test split it reports accuracy and minimal-pair accuracy (both twins right; any predictor that
ignores the world scores 0 on pairs, because twins share the formula), broken down by formula height,
number of objects and outer operator. Direct models answer from the logits after ANSWER (TRUE against
FALSE); scratchpad models decode greedily after QUERY and are scored on the token after the ANSWER they
write, with the trace checked against the gold one (step 0 holds the atom lookups).

Surface baselines, trained on the training split: logistic regression and a small MLP on features a
shallow predictor could use without composing the formula (how many of its atoms are true in the world,
connective counts, height, number of objects, world density, outer operator).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.eval_generator import CONNECTIVES, MAX_LEN, Tokens  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

TEST_SPLITS = ["test_iid", "test_deep", "test_large_world", "test_shift", "test_negation"]
OUTERS = ["NOT"] + CONNECTIVES
FEATURES = ["n_atoms", "n_true", "frac_true", "n_not"] + [f"n_{c}" for c in CONNECTIVES] + \
           ["density", "height", "objects"]


def feature_matrix(items):
    rows = [[float(it[f]) for f in FEATURES] + [float(it["outer"] == o) for o in OUTERS] for it in items]
    return torch.tensor(rows)


def fit_baselines(train_items, seed=0):
    """Logistic regression and an MLP on surface features; returns name -> predict(items) -> labels."""
    torch.manual_seed(seed)
    x = feature_matrix(train_items)
    y = torch.tensor([float(it["label"]) for it in train_items])
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    out = {}
    for name, net in (("surface features, logistic regression", nn.Linear(x.size(1), 1)),
                      ("surface features, MLP", nn.Sequential(nn.Linear(x.size(1), 64), nn.ReLU(),
                                                               nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)))):
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        xs = (x - mu) / sd
        for _ in range(400):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(xs).squeeze(1), y)
            loss.backward()
            opt.step()
        net.eval()
        out[name] = (lambda items, net=net: ((net((feature_matrix(items) - mu) / sd).squeeze(1) > 0)
                                             .long().tolist()))
    return out


@torch.no_grad()
def predict_direct(model, items, tok, device, batch=256):
    order = sorted(range(len(items)), key=lambda i: items[i]["direct_from"])
    out = [None] * len(items)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        n = max(items[i]["direct_from"] for i in idx)
        ids = torch.full((len(idx), n), tok.v.pad_token, dtype=torch.long)
        lens = []
        for r, i in enumerate(idx):
            prefix = items[i]["direct_seq"][:items[i]["direct_from"]]
            ids[r, :len(prefix)] = torch.tensor(prefix)
            lens.append(len(prefix))
        logits = model(ids.to(device))[torch.arange(len(idx)), torch.tensor(lens) - 1]
        for r, i in enumerate(idx):
            out[i] = int(logits[r, tok.TRUE] > logits[r, tok.FALSE])
    return out, [None] * len(items)


@torch.no_grad()
def predict_scratch(model, items, tok, device, batch=64):
    """Greedy decoding after QUERY, in groups of equal prefix length (no padding, absolute positions)."""
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it["scratch_from"]].append(i)
    preds, traces = [None] * len(items), [None] * len(items)
    for plen, idx_all in groups.items():
        for s in range(0, len(idx_all), batch):
            idx = idx_all[s:s + batch]
            seqs = torch.tensor([items[i]["scratch_seq"][:plen] for i in idx], device=device)
            budget = min(MAX_LEN - plen, max(len(items[i]["scratch_seq"]) - plen for i in idx) + 16)
            done = torch.zeros(len(idx), dtype=torch.bool, device=device)
            for _ in range(budget):
                nxt = model(seqs)[:, -1].argmax(-1)
                seqs = torch.cat([seqs, nxt[:, None]], 1)
                done |= (seqs[:, -2] == tok.ANSWER)
                if done.all():
                    break
            for r, i in enumerate(idx):
                gen = seqs[r, plen:].tolist()
                k = gen.index(tok.ANSWER) if tok.ANSWER in gen else None
                ans = gen[k + 1] if k is not None and k + 1 < len(gen) else None
                preds[i] = 1 if ans == tok.TRUE else (0 if ans == tok.FALSE else -1)
                traces[i] = gen[:k] if k is not None else gen
    return preds, traces


def score(items, preds, traces=None, tok=None, fmt="direct"):
    def acc(ids):
        return sum(preds[i] == items[i]["label"] for i in ids) / len(ids) if ids else float("nan")

    res = {"n": len(items), "accuracy": acc(range(len(items)))}
    pairs = defaultdict(list)
    for i, it in enumerate(items):
        if it["pair"] is not None:
            pairs[it["pair"]].append(i)
    if pairs:
        res["pair_accuracy"] = sum(all(preds[i] == items[i]["label"] for i in g) for g in pairs.values()) / len(pairs)
    for key in ("height", "objects", "outer", "rho"):
        groups = defaultdict(list)
        for i, it in enumerate(items):
            groups[it[key]].append(i)
        res[f"by_{key}"] = {str(k): acc(v) for k, v in sorted(groups.items())}
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[50 * ((len(it[f"{fmt}_seq"]) - 1) // 50)].append(i)
    res["by_length"] = {f"{k + 1}-{k + 50}": acc(v) for k, v in sorted(groups.items())}
    if traces is not None and traces[0] is not None:
        gold = [it["scratch_seq"][it["scratch_from"]:-2] for it in items]
        step0 = lambda t: t[:t.index(tok.SEP)] if tok.SEP in t else t
        res["trace_exact"] = sum(t == g for t, g in zip(traces, gold)) / len(items)
        res["step0_exact"] = sum(step0(t) == step0(g) for t, g in zip(traces, gold)) / len(items)
        res["invalid_answer"] = sum(p == -1 for p in preds) / len(items)
    return res


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/eval_stage_a")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH:FORMAT")
    ap.add_argument("--splits", nargs="*", default=TEST_SPLITS)
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = Vocabulary(args.vocab)
    tok = Tokens(vocab)
    data = Path(args.data)
    splits = {s: json.load(open(data / f"{s}.json"))["items"] for s in args.splits}
    results = {}

    baselines = fit_baselines(json.load(open(data / "train.json"))["items"])
    for name, predict in baselines.items():
        results[name] = {s: score(items, predict(items)) for s, items in splits.items()}

    from models.transformer import load_model_from_checkpoint
    for spec in args.checkpoint:
        name, rest = spec.split("=", 1)
        path, fmt = rest.rsplit(":", 1)
        model, ck = load_model_from_checkpoint(path, vocab.vocab_size, device)
        model.eval()
        results[name] = {"_checkpoint": {"path": path, "format": fmt, "epoch": ck.get("epoch"),
                                         "best_val_loss": ck.get("best_val_loss")}}
        for s, items in splits.items():
            preds, traces = (predict_scratch if fmt == "scratch" else predict_direct)(model, items, tok, device)
            results[name][s] = score(items, preds, traces, tok, fmt)
            print(f"  {name} {s}: accuracy {pct(results[name][s]['accuracy'])}", file=sys.stderr)

    names = list(results)
    print("Accuracy / minimal-pair accuracy (%). A predictor that ignores the world gets 0 pairs right.\n")
    print("| method | " + " | ".join(args.splits) + " |")
    print("|---|" + "---|" * len(args.splits))
    for name in names:
        cells = [f"{pct(results[name][s]['accuracy'])} / {pct(results[name][s].get('pair_accuracy'))}"
                 for s in args.splits]
        print(f"| {name} | " + " | ".join(cells) + " |")
    for key in ("height", "objects"):
        print(f"\nAccuracy (%) by {key}, all test splits pooled per value:\n")
        vals = sorted({v for n in names for s in args.splits for v in results[n][s][f"by_{key}"]}, key=float)
        print(f"| method | " + " | ".join(vals) + " |")
        print("|---|" + "---|" * len(vals))
        for name in names:
            cells = []
            for v in vals:
                tot = [(results[name][s][f"by_{key}"].get(v), sum(1 for it in splits[s] if str(it[key]) == v))
                       for s in args.splits]
                n = sum(c for a, c in tot if a is not None)
                cells.append(pct(sum(a * c for a, c in tot if a is not None) / n) if n else "n/a")
            print(f"| {name} | " + " | ".join(cells) + " |")
    scratch = [n for n in names if results[n].get("_checkpoint", {}).get("format") == "scratch"]
    if scratch:
        print("\nScratchpad traces (%): step 0 (lookups) exactly right / whole trace exactly right / no answer:\n")
        print("| method | " + " | ".join(args.splits) + " |")
        print("|---|" + "---|" * len(args.splits))
        for name in scratch:
            r = results[name]
            print(f"| {name} | " + " | ".join(f"{pct(r[s]['step0_exact'])} / {pct(r[s]['trace_exact'])} / "
                                              f"{pct(r[s]['invalid_answer'])}" for s in args.splits) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
