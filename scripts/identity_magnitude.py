#!/usr/bin/env python3
"""Identity against magnitude: can one numeral code both look a key up and compare two numbers?

    python scripts/identity_magnitude.py --task recall --code xval --steps 10000

Both tasks use fresh batches over the same numerals. Training uses 0-499, minus a held-out set of 100
drawn once (seed 0, scattered, so every row and column of the grid still appears in training); 500-624
are never seen.

- **recall**: ``k1 v1 ... k8 v8 ANSWER q -> v`` (the ``pairs`` layout of sanity_recall.py), values
  TRUE/FALSE, 8 queries per training sequence. Half the training sequences draw 8 random keys, half 8
  consecutive members of the training pool, so neighbouring numerals have to be told apart.
- **compare**: ``a OP b ANSWER -> TRUE/FALSE`` with OP < or >, 8 comparisons per training sequence
  separated by SEP; 40% of pairs have |a - b| <= 10.

``--code`` sets how numerals are embedded (every other token has a learned row):

| code | numeral k is |
|---|---|
| learned | its own learned row |
| xval | s(k) * h: one learned direction scaled by k's standardised value |
| ramp | its count panel through two generated directions, then LayerNorm (grid_shared, ramp basis) |
| grid_linear | its count panel through a learned linear map |
| grid_cell | the k-th cell of the grid: a learned row vector for k // 25 plus a column vector for k % 25 |
| grid_cell_xval | both channels: the grid cell plus s(k) * h |

Evaluation uses 512 fresh one-query sequences per set. Recall sets: ``seen`` (random training-pool
keys), ``seen_adjacent`` (8 consecutive pool members), ``heldout`` and ``far`` (all keys from the held-out
set or from 500-624). Compare sets: ``seen``, ``heldout`` and ``far`` (both numbers from that set).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transformer import create_model  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

CODES = {
    "learned": {"embedding_type": "learned"},
    "xval": {"embedding_type": "xval"},
    "ramp": {"embedding_type": "grid_shared", "shared_basis": "ramp", "grid_numerals_only": True},
    "grid_linear": {"embedding_type": "grid_linear", "grid_numerals_only": True},
    "grid_cell": {"embedding_type": "grid_cell"},
    "grid_cell_xval": {"embedding_type": "grid_cell_xval"},
}
HELDOUT = sorted(random.Random(0).sample(range(500), 100))
POOL = [k for k in range(500) if k not in set(HELDOUT)]
FAR = list(range(500, 625))
N_KEYS = 8
CLOSE_FRACTION, CLOSE_MAX = 0.4, 10
GAP_BINS = [(1, 1), (2, 5), (6, 20), (21, 625)]


def draw_keys(numbers, adjacent):
    if adjacent:
        s = random.randrange(len(numbers) - N_KEYS + 1)
        keys = numbers[s:s + N_KEYS]
        return random.sample(keys, N_KEYS)
    return random.sample(numbers, N_KEYS)


def recall_batch(vocab, bs, queries, sets):
    """Each row draws its keys from one of ``sets``: (numbers, adjacent) pairs, chosen uniformly."""
    lab = vocab.encode_label
    tf = [lab("TRUE"), lab("FALSE")]
    rows, targets = [], None
    for _ in range(bs):
        keys = draw_keys(*random.choice(sets))
        vals = [random.choice(tf) for _ in keys]
        seq = [vocab.message_start]
        for k, v in zip(keys, vals):
            seq += [k, v]
        tg = []
        for _ in range(queries):
            i = random.randrange(N_KEYS)
            seq += [lab("ANSWER"), keys[i]]
            tg.append(len(seq))
            seq.append(vals[i])
        rows.append(seq)
        targets = tg
    return torch.tensor(rows), targets


_NEAR = {}


def draw_pair(numbers):
    """Two distinct numbers, |a - b| <= CLOSE_MAX with probability CLOSE_FRACTION."""
    if random.random() < CLOSE_FRACTION:
        key = tuple(numbers)
        if key not in _NEAR:
            _NEAR[key] = {a: [b for b in numbers if b != a and abs(b - a) <= CLOSE_MAX] for a in numbers}
        near = _NEAR[key]
        while True:
            a = random.choice(numbers)
            if near[a]:
                return a, random.choice(near[a])
    return tuple(random.sample(numbers, 2))


def compare_batch(vocab, bs, queries, numbers):
    lab = vocab.encode_label
    lt, gt, tf = lab("LESS_THAN"), lab("GREATER_THAN"), (lab("TRUE"), lab("FALSE"))
    rows, gaps, targets = [], [], None
    for _ in range(bs):
        seq, tg = [vocab.message_start], []
        for q in range(queries):
            a, b = draw_pair(numbers)
            op = random.choice([lt, gt])
            truth = a < b if op == lt else a > b
            seq += ([lab("SEP")] if q else []) + [a, op, b, lab("ANSWER")]
            tg.append(len(seq))
            seq.append(tf[0] if truth else tf[1])
            if q == 0:
                gaps.append(abs(a - b))
        rows.append(seq)
        targets = tg
    return torch.tensor(rows), targets, gaps


def eval_sets(task):
    if task == "recall":
        return {"seen": [(POOL, False)], "seen_adjacent": [(POOL, True)], "heldout": [(HELDOUT, False)],
                "far": [(FAR, False)]}
    return {"seen": POOL, "heldout": HELDOUT, "far": FAR}


def make_batch(vocab, task, bs, queries, source):
    if task == "recall":
        seq, targets = recall_batch(vocab, bs, queries, source)
        return seq, targets, None
    return compare_batch(vocab, bs, queries, source)


@torch.no_grad()
def evaluate(model, vocab, task, device, n=512):
    model.eval()
    out = {}
    for name, source in eval_sets(task).items():
        seq, targets, gaps = make_batch(vocab, task, n, 1, source)
        seq = seq.to(device)
        right = model(seq[:, :-1])[:, targets[0] - 1].argmax(-1) == seq[:, targets[0]]
        out[name] = right.float().mean().item()
        if gaps is not None:
            for lo, hi in GAP_BINS:
                pick = torch.tensor([lo <= g <= hi for g in gaps], device=device)
                if pick.any():
                    out[f"{name}_gap{lo}-{hi if hi < 625 else ''}"] = right[pick].float().mean().item()
    model.train()
    return out


def run(task, code, steps, seed, device, log_every, queries=8):
    random.seed(seed)
    torch.manual_seed(seed)
    vocab = Vocabulary(str(ROOT / "unified_vocabulary.json"))
    model = create_model(vocab.vocab_size, "tiny", verbose=False, **CODES[code]).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    train_source = [(POOL, False), (POOL, True)] if task == "recall" else POOL
    history = []
    for step in range(1, steps + 1):
        seq, targets, _ = make_batch(vocab, task, 64, queries, train_source)
        seq = seq.to(device)
        logits = model(seq[:, :-1])
        pos = torch.tensor(targets) - 1
        loss = F.cross_entropy(logits[:, pos].reshape(-1, logits.size(-1)), seq[:, targets].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            acc = evaluate(model, vocab, task, device)
            history.append({"step": step, "loss": round(loss.item(), 4), **{k: round(v, 3) for k, v in acc.items()}})
            main = ", ".join(f"{k} {v:.3f}" for k, v in acc.items() if "_gap" not in k)
            print(f"{task} {code} step {step}: loss {loss.item():.3f}, {main}", flush=True)
    final = history[-1]
    gaps = {k: v for k, v in final.items() if "_gap" in k}
    if gaps:
        print(f"{task} {code} final by gap: " + ", ".join(f"{k} {v:.3f}" for k, v in gaps.items()), flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="recall", choices=["recall", "compare"])
    ap.add_argument("--code", default="learned", choices=list(CODES))
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None, help="append this run's history to a JSON list file")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    history = run(args.task, args.code, args.steps, args.seed, device, args.log_every)
    if args.json:
        path = Path(args.json)
        runs = json.loads(path.read_text()) if path.exists() else []
        runs.append({"task": args.task, "code": args.code, "seed": args.seed, "steps": args.steps,
                     "history": history})
        path.write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
