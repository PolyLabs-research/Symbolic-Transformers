#!/usr/bin/env python3
"""Sanity check: can the project's transformer learn minimal associative recall?

    python scripts/sanity_recall.py --steps 3000 --variant default

Each sequence lists n key-value pairs, then a query key, and the target is that key's value:

    MESSAGE_START k1 v1 k2 v2 ... kn vn ANSWER q   ->   v(q)

Keys are distinct numerals from 0-99; values are TRUE/FALSE (``--values binary``) or numerals 100-199
(``--values numerals``). Every batch is freshly sampled, so the model cannot memorise. Small transformers
learn this task reliably; if this model does not, the problem is in the model or the training setup,
not in the formula-evaluation data. ``--variant`` changes one thing at a time from the trainer's setup.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transformer import create_model  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

VARIANTS = {
    "default": {},                    # the trainer's setup: post-LN, dropout 0.1, lr 3e-4
    "pre_ln": {"norm_first": True},
    "no_dropout": {"dropout": 0.0},
    "lr_1e-3": {"lr": 1e-3},
    "pre_ln_lr_1e-3": {"norm_first": True, "lr": 1e-3},
}


def batch(vocab, n_pairs, bs, values, queries, device, layout="pairs"):
    """``pairs``: k1 v1 ... kn vn ANSWER q -> v. ``table``: the formula-evaluation full-table layout,
    CONST k1 ... CONST kn SEP PRED 0 ( CONST k ) v ... MODELS PRED 0 ( CONST q ) ANSWER -> v, with
    queries separated by SEP."""
    lab = vocab.encode_label
    true_false = torch.tensor([lab("TRUE"), lab("FALSE")])
    keys = torch.stack([torch.randperm(100)[:n_pairs] for _ in range(bs)])
    if values == "binary":
        vals = true_false[torch.randint(0, 2, (bs, n_pairs))]
    else:
        vals = torch.randint(100, 200, (bs, n_pairs))
    full = lambda tok: torch.full((bs, 1), tok)
    if layout == "pairs":
        seq = [full(vocab.message_start), torch.stack([keys, vals], 2).reshape(bs, 2 * n_pairs)]
    else:
        const, pred = vocab.compositional_tokens["CONST"], vocab.compositional_tokens["PRED"]
        order = torch.stack([torch.randperm(n_pairs) for _ in range(bs)])
        domain = torch.stack([full(const)[:, 0].expand(bs, n_pairs), keys], 2).reshape(bs, 2 * n_pairs)
        k, v = keys.gather(1, order), vals.gather(1, order)
        entry = [full(pred).expand(bs, n_pairs), torch.zeros(bs, n_pairs, dtype=torch.long),
                 full(lab("LPAREN")).expand(bs, n_pairs), full(const).expand(bs, n_pairs), k,
                 full(lab("RPAREN")).expand(bs, n_pairs), v]
        seq = [full(vocab.message_start), domain, full(lab("SEP")),
               torch.stack(entry, 2).reshape(bs, 7 * n_pairs), full(lab("MODELS"))]
    targets = []
    for i in range(queries):
        qi = torch.randint(0, n_pairs, (bs,))
        q = keys[torch.arange(bs), qi]
        v = vals[torch.arange(bs), qi]
        if layout == "pairs":
            seq += [full(lab("ANSWER")), q[:, None], v[:, None]]
        else:
            const, pred = vocab.compositional_tokens["CONST"], vocab.compositional_tokens["PRED"]
            seq += ([full(lab("SEP"))] if i else []) + [full(pred), full(0), full(lab("LPAREN")), full(const),
                                                       q[:, None], full(lab("RPAREN")), full(lab("ANSWER")),
                                                       v[:, None]]
        targets.append(sum(s.size(1) for s in seq) - 1)
    seq = torch.cat(seq, 1).to(device)
    return seq, targets


def run(variant, steps, n_pairs, values, queries, seed, device, log_every, layout="pairs"):
    torch.manual_seed(seed)
    cfg = VARIANTS[variant]
    vocab = Vocabulary(str(ROOT / "unified_vocabulary.json"))
    model = create_model(vocab.vocab_size, "tiny", verbose=False)
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
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    history = []
    for step in range(1, steps + 1):
        seq, targets = batch(vocab, n_pairs, 64, values, queries, device, layout)
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
                s, t = batch(vocab, n_pairs, 512, values, 1, device, layout)
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
    ap.add_argument("--layout", default="pairs", choices=["pairs", "table"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run(args.variant, args.steps, args.pairs, args.values, args.queries, args.seed, device, args.log_every,
        args.layout)


if __name__ == "__main__":
    main()
