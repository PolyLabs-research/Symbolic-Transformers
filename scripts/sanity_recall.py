#!/usr/bin/env python3
"""Sanity check: can the project's transformer learn minimal associative recall?

    python scripts/sanity_recall.py --steps 3000 --variant default

Each sequence lists n key-value pairs, then a query key, and the target is that key's value:

    MESSAGE_START k1 v1 k2 v2 ... kn vn ANSWER q   ->   v(q)

``--layout`` changes where the key and the value sit, and ``--queries`` sets how many queries each training
sequence carries (see LAYOUTS).

Keys are distinct numerals from 0-99; values are TRUE/FALSE (``--values binary``) or numerals 100-199
(``--values numerals``). Every batch is freshly sampled, so the model cannot memorise. Small transformers
learn this task reliably; if this model does not, the problem is in the model or the training setup,
not in the formula-evaluation data. ``--variant`` changes one thing at a time from the trainer's setup.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transformer import create_model  # noqa: E402
from training.train import make_lr_lambda  # noqa: E402
from utils.fused import N_INDEX, SymbolFuser  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

VARIANTS = {
    "default": {},                    # the trainer's setup: post-LN, dropout 0.1, lr 3e-4
    "pre_ln": {"norm_first": True},
    "no_dropout": {"dropout": 0.0},
    "lr_1e-3": {"lr": 1e-3},
    "pre_ln_lr_1e-3": {"norm_first": True, "lr": 1e-3},
}


# k: a key, v: its value, q: the queried key. The answer is predicted at the token before "->".
LAYOUTS = {
    "pairs": "k1 v1 ... kn vn ANSWER q -> v",
    "pairs_offset2": "k ) v ... ANSWER q ) -> v",          # a filler token in the facts and in the query
    "pairs_gap_context": "k ) v ... ANSWER q -> v",        # the filler in the facts only
    "pairs_gap_query": "k v ... ANSWER q ) -> v",          # the filler in the query only
    "table": "CONST k1 ... CONST kn SEP PRED 0 ( CONST k ) v ... MODELS PRED 0 ( CONST q ) ANSWER -> v",
    "table_nodomain": "SEP PRED 0 ( CONST k ) v ... MODELS PRED 0 ( CONST q ) ANSWER -> v",
    "table_keylast": "CONST k1 ... CONST kn SEP PRED 0 ( CONST k v ) ... MODELS PRED 0 ( CONST q -> v )",
}


def batch(vocab, n_pairs, bs, values, queries, device, layout="pairs", balanced=False):
    """A batch of ``layout`` (see LAYOUTS). The table layouts list the facts in a random order, and
    separate queries with SEP. Returns the sequences and the positions of the answers.

    Queries pick a fact uniformly, so answering with the majority of the listed values beats chance. With
    ``balanced`` (binary values only) each answer is TRUE or FALSE with probability 1/2 whatever the table
    holds, as in the lookup ladder's training items, and that shortcut is worth nothing."""
    lab = vocab.encode_label
    true_false = torch.tensor([lab("TRUE"), lab("FALSE")])
    keys = torch.stack([torch.randperm(100)[:n_pairs] for _ in range(bs)])
    if values == "binary":
        vals = true_false[torch.randint(0, 2, (bs, n_pairs))]
    else:
        vals = torch.randint(100, 200, (bs, n_pairs))
    full = lambda tok, n=1: torch.full((bs, n), tok)
    interleave = lambda parts: torch.stack(parts, 2).reshape(bs, len(parts) * n_pairs)
    const, pred = vocab.compositional_tokens["CONST"], vocab.compositional_tokens["PRED"]
    lp, rp = lab("LPAREN"), lab("RPAREN")
    seq = [full(vocab.message_start)]
    if layout.startswith("pairs"):
        gap = layout in ("pairs_offset2", "pairs_gap_context")
        seq.append(interleave([keys, full(rp, n_pairs), vals] if gap else [keys, vals]))
    else:
        if layout != "table_nodomain":
            seq.append(interleave([full(const, n_pairs), keys]))
        order = torch.stack([torch.randperm(n_pairs) for _ in range(bs)])
        k, v = keys.gather(1, order), vals.gather(1, order)
        entry = [full(pred, n_pairs), full(0, n_pairs), full(lp, n_pairs), full(const, n_pairs), k]
        entry += [v, full(rp, n_pairs)] if layout == "table_keylast" else [full(rp, n_pairs), v]
        seq += [full(lab("SEP")), interleave(entry), full(lab("MODELS"))]
    targets = []
    for i in range(queries):
        if balanced:
            if values != "binary":
                raise ValueError("balanced queries need binary values")
            want = true_false[torch.randint(0, 2, (bs,))]
            match = (vals == want[:, None]).float()
            match[match.sum(1) == 0] = 1.0  # every value is the other one: query any fact
            qi = torch.multinomial(match, 1).squeeze(1)
        else:
            qi = torch.randint(0, n_pairs, (bs,))
        q = keys[torch.arange(bs), qi][:, None]
        v = vals[torch.arange(bs), qi][:, None]
        if layout in ("pairs", "pairs_gap_context"):
            seq += [full(lab("ANSWER")), q]
        elif layout in ("pairs_offset2", "pairs_gap_query"):
            seq += [full(lab("ANSWER")), q, full(rp)]
        else:
            seq += ([full(lab("SEP"))] if i else []) + [full(pred), full(0), full(lp), full(const), q]
            if layout != "table_keylast":
                seq += [full(rp), full(lab("ANSWER"))]
        seq.append(v)
        targets.append(sum(s.size(1) for s in seq) - 1)
        if layout == "table_keylast":
            seq.append(full(rp))
    seq = torch.cat(seq, 1).to(device)
    return seq, targets


def fuse_batch(fuser, seq, targets):
    """One token per symbol. Every row of a batch has its category tokens at the same positions."""
    first = seq[0].tolist()
    cats = [j for j, t in enumerate(first) if t in fuser.cat_index]
    seq = seq.clone()
    for j in cats:
        seq[:, j] = fuser.base + fuser.cat_index[first[j]] * N_INDEX + seq[:, j + 1]
    numerals = {j + 1 for j in cats}
    _, where = fuser.fuse(first)
    return seq[:, [j for j in range(seq.size(1)) if j not in numerals]], [where[t] for t in targets]


def export_items(out, vocab, pool, targets, fresh, n_pairs):
    """Write the training pool and fresh single-query val/test sets as data/eval_generator item files, so
    training/train.py and scripts/assess_eval.py run on exactly these sequences."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    true = vocab.encode_label("TRUE")

    def items(seq, tg):
        return [{"direct_seq": row, "direct_from": tg[0], "direct_targets": tg, "label": int(row[tg[0]] == true),
                 "pair": None, "height": 0, "objects": n_pairs, "outer": "ATOM", "rho": 0.5}
                for row in seq.tolist()]
    splits = {"train": items(pool, targets)}
    for name in ("val", "test_iid"):
        splits[name] = items(*fresh(2000, 1))
    for name, its in splits.items():
        (out / f"{name}.json").write_text(json.dumps({"split": name, "items": its}))
        print(f"{name}: {len(its)} items of length {len(its[0]['direct_seq'])}")


def run(variant, steps, n_pairs, values, queries, seed, device, log_every, layout="pairs", fuse=False,
        fixed=0, schedule="constant", balanced=False, export=None):
    """``fixed``: train on a pool of that many sequences, drawn once (evaluation stays fresh), as the
    trainer does with a dataset. ``schedule="cosine"``: the trainer's learning-rate schedule, warm-up
    then cosine decay to 10% of the rate over ``steps``; the default holds the rate after warm-up.
    ``export``: write that pool (drawn exactly as a run with these settings would draw it) and fresh
    val/test sets to a directory, and stop."""
    torch.manual_seed(seed)
    cfg = VARIANTS[variant]
    vocab = Vocabulary(str(ROOT / "unified_vocabulary.json"))
    fuser = SymbolFuser(vocab) if fuse else None
    make = lambda bs, q: (lambda b: fuse_batch(fuser, *b) if fuse else b)(
        batch(vocab, n_pairs, bs, values, q, device, layout, balanced))
    model = create_model(vocab.vocab_size, "tiny", verbose=False,
                         fused_categories=fuser.categories if fuse else None)
    if "norm_first" in cfg or "dropout" in cfg:
        for layer in model.transformer.layers:
            if "norm_first" in cfg:
                layer.norm_first = cfg["norm_first"]
            if "dropout" in cfg:
                for m in layer.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = cfg["dropout"]
                layer.self_attn.dropout = cfg["dropout"]
        if "dropout" in cfg:
            model.pos_encoder.dropout.p = cfg["dropout"]
    model.to(device).train()
    lr = cfg.get("lr", 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = 500
    lr_lambda = make_lr_lambda(steps, warm) if schedule == "cosine" else (lambda s: min(1.0, (s + 1) / warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if fixed:
        pool, targets = make(fixed, queries)
    if export:
        if not fixed or fuse:
            raise ValueError("--export writes the --fixed pool in split tokens (the trainer fuses on its own)")
        export_items(export, vocab, pool.cpu(), targets, lambda bs, q: (lambda b: (b[0].cpu(), b[1]))(make(bs, q)),
                     n_pairs)
        return []
    history = []
    for step in range(1, steps + 1):
        if fixed:
            seq = pool[torch.randint(0, fixed, (64,), device=pool.device)]
        else:
            seq, targets = make(64, queries)
        logits = model(seq[:, :-1])
        pos = torch.tensor(targets) - 1
        loss = F.cross_entropy(logits[:, pos].reshape(-1, logits.size(-1)), seq[:, targets].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                s, t = make(512, 1)
                acc = (model(s[:, :-1])[:, t[0] - 1].argmax(-1) == s[:, t[0]]).float().mean().item()
            model.train()
            history.append((step, round(loss.item(), 3), round(acc, 3)))
            print(f"{variant} step {step}: loss {loss.item():.3f}, query accuracy {acc:.3f}", flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="default", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--values", default="binary", choices=["binary", "numerals"])
    ap.add_argument("--queries", type=int, default=1)
    ap.add_argument("--layout", default="pairs", choices=list(LAYOUTS))
    ap.add_argument("--fuse", action="store_true", help="One token per symbol: CONST k as a single token")
    ap.add_argument("--fixed", type=int, default=0, metavar="N",
                    help="Train on a fixed pool of N sequences instead of fresh batches (evaluation stays fresh)")
    ap.add_argument("--balanced", action="store_true",
                    help="Answers TRUE or FALSE with probability 1/2 whatever the table holds, as in the ladder's items")
    ap.add_argument("--export", default=None, metavar="DIR",
                    help="With --fixed: write the training pool and fresh val/test sets as item files, then stop")
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine"],
                    help="cosine: the trainer's warm-up then cosine decay to 10%% over --steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run(args.variant, args.steps, args.pairs, args.values, args.queries, args.seed, device, args.log_every,
        args.layout, args.fuse, args.fixed, args.schedule, args.balanced, args.export)


if __name__ == "__main__":
    main()
