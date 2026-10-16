#!/usr/bin/env python3
"""Assess FOL next-symbol models against count baselines, by grammatical role and on unseen contexts.

    python scripts/assess_fol.py --checkpoint retrained=runs/fol_retrain/best_model.pt \\
        --checkpoint shipped=checkpoints/best_model.pt --json experiments/fol_retrain.json

Every test target gets a role from utils/fol_grammar.py (forced token, formula-start choice, connective,
arity, predicate index, quantified variable, free variable, bound-variable reference) and a flag for
whether its context occurs in the training set. Accuracy is reported overall, on contexts not in the
training set, and per role next to the best achievable value where the grammar fixes it. Baselines: the
most frequent token, a bigram, and the longest matching context suffix in the training set (for a context
seen in training, that is the training set's own most frequent answer). Each model also samples formulas,
scored for well-formedness and for referring to variables that are in scope.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.transformer import load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset, trim_padding  # noqa: E402
from utils.fol_grammar import ROLES, FOLGrammar, ceiling  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

K = 5


def load_samples(path):
    return json.load(open(path))["samples"]


def annotate_samples(samples, grammar, vocab):
    """Role and bound-variable distribution of every sample's target, from its formula's parse."""
    roles, dists, formulas = [None] * len(samples), [None] * len(samples), [None] * len(samples)
    # Samples of one formula are consecutive; the one whose target is MESSAGE_END holds the formula.
    start = 0
    for i, s in enumerate(samples):
        if s["target"] == vocab.message_end:
            formula = s["context"][1:]
            r, d = grammar.annotate_with_boundaries(formula)
            for j in range(start, i + 1):
                pos = len(samples[j]["context"])  # target position in [MS] + formula + [ME]
                roles[j] = r[pos - 1]
                dists[j] = d.get(pos)
                formulas[j] = tuple(formula)
            start = i + 1
    assert all(r is not None for r in roles), "every sample should belong to a complete formula"
    return roles, dists, formulas


class SuffixCounts:
    """Next-token counts for every suffix of every training context."""

    def __init__(self, samples, max_order=None):
        self.table = defaultdict(Counter)
        self.unigram = Counter()
        self.max_order = max_order
        for s in samples:
            c, t = tuple(s["context"]), s["target"]
            self.unigram[t] += 1
            top = len(c) if max_order is None else min(len(c), max_order)
            for n in range(1, top + 1):
                self.table[c[-n:]][t] += 1
        self.unigram_ranked = [t for t, _ in self.unigram.most_common(K)]

    def ranked(self, context):
        """Top-K tokens: longest matching suffix first, filled from shorter suffixes, then unigram."""
        c = tuple(context)
        out = []
        top = len(c) if self.max_order is None else min(len(c), self.max_order)
        for n in range(top, 0, -1):
            counts = self.table.get(c[-n:])
            if counts:
                for t, _ in counts.most_common():
                    if t not in out:
                        out.append(t)
                        if len(out) == K:
                            return out
        for t in self.unigram_ranked:
            if t not in out:
                out.append(t)
        return out[:K]


def baseline_hits(samples, predictor):
    hit1, hit5 = [], []
    for s in samples:
        r = predictor(s["context"])
        hit1.append(r[0] == s["target"])
        hit5.append(s["target"] in r)
    return {"hit1": hit1, "hit5": hit5}


@torch.no_grad()
def model_hits(model, dataset, device, batch_size=512):
    nll, hit1, hit5 = [], [], []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
        ids = torch.stack([it["input_ids"] for it in items]).to(device)
        lengths = torch.stack([it["length"] for it in items]).to(device)
        targets = torch.stack([it["target"] for it in items]).to(device)
        logits = model(trim_padding(ids, lengths))
        last = logits[torch.arange(ids.size(0), device=device), lengths - 1]
        logp = F.log_softmax(last, dim=-1)
        nll += (-logp.gather(1, targets[:, None]).squeeze(1)).tolist()
        top = logp.topk(K, dim=-1).indices
        hit1 += (top[:, 0] == targets).tolist()
        hit5 += (top == targets[:, None]).any(1).tolist()
    return {"nll": nll, "hit1": hit1, "hit5": hit5}


@torch.no_grad()
def sample_formulas(model, vocab, device, n, max_len=128, batch=250, seed=0):
    """Ancestral samples at temperature 1, from MESSAGE_START until MESSAGE_END or max_len tokens."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    while len(out) < n:
        b = min(batch, n - len(out))
        seqs = torch.full((b, 1), vocab.message_start, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = model(seqs)[:, -1]
            nxt = torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1, generator=g).squeeze(1).to(device)
            nxt = torch.where(done, torch.full_like(nxt, vocab.pad_token), nxt)
            seqs = torch.cat([seqs, nxt[:, None]], dim=1)
            done |= nxt == vocab.message_end
            if done.all():
                break
        for row, d in zip(seqs.tolist(), done.tolist()):
            row = row[1:]
            out.append(row[:row.index(vocab.message_end)] if d else None)  # None: never ended
    return out


def generation_scores(formulas, grammar, train_formulas):
    ended = [f for f in formulas if f is not None]
    ok = [f for f in ended if grammar.is_well_formed(f)]
    in_scope = total_bound = 0
    for f in ok:
        roles, dists = grammar.annotate(f)
        for i, d in dists.items():
            total_bound += 1
            in_scope += d.get(f[i], 0) > 0
    return {
        "n": len(formulas),
        "ended": len(ended) / len(formulas),
        "well_formed": len(ok) / len(formulas),
        "bound_refs_in_scope": in_scope / total_bound if total_bound else None,
        "novel": sum(tuple(f) not in train_formulas for f in ok) / len(ok) if ok else None,
        "distinct": len({tuple(f) for f in ok}) / len(ok) if ok else None,
        "mean_len": sum(map(len, ok)) / len(ok) if ok else None,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(hits, roles, seen, formula_seen, targets):
    def subset(ids):
        r = {"n": len(ids), "top1": mean([hits["hit1"][i] for i in ids]), "top5": mean([hits["hit5"][i] for i in ids])}
        if "nll" in hits:
            r["loss"] = mean([hits["nll"][i] for i in ids])
        return r

    every = range(len(roles))
    out = {"all": subset(every),
           "formula_not_in_train": subset([i for i in every if not formula_seen[i]]),
           "formula_in_train": subset([i for i in every if formula_seen[i]]),
           "unseen_context": subset([i for i in every if not seen[i]]),
           "seen_context": subset([i for i in every if seen[i]]),
           "num_000_target": subset([i for i in every if targets[i] == 0]),
           "by_role": {role: subset([i for i in every if roles[i] == role]) for role in ROLES}}
    return out


def role_ceilings(roles, dists, pool):
    res = {}
    for role in ROLES:
        ids = [i for i, r in enumerate(roles) if r == role]
        c1 = [ceiling(role, dists[i], pool, 1) for i in ids]
        c5 = [ceiling(role, dists[i], pool, K) for i in ids]
        res[role] = {"share": len(ids) / len(roles),
                     "top1": None if c1 and c1[0] is None else mean(c1),
                     "top5": None if c5 and c5[0] is None else mean(c5)}
    return res


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--train-data", default="datasets/fol_next_symbol/train.json")
    ap.add_argument("--test-data", default="datasets/fol_next_symbol/test.json")
    ap.add_argument("--metadata", default="datasets/fol_next_symbol/metadata.json")
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--n-generate", type=int, default=1000)
    ap.add_argument("--json", default=None, help="Write all numbers here")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    vocab = Vocabulary(args.vocab)
    grammar = FOLGrammar(vocab)
    meta = json.load(open(args.metadata))
    pool = {"var": meta["config"]["max_variables"], "pred": meta["config"]["max_predicates"]}

    train, test = load_samples(args.train_data), load_samples(args.test_data)
    roles, dists, sample_formula = annotate_samples(test, grammar, vocab)
    train_contexts = {tuple(s["context"]) for s in train}
    seen = [tuple(s["context"]) in train_contexts for s in test]
    train_formulas = {tuple(s["context"][1:]) for s in train if s["target"] == vocab.message_end}
    formula_seen = [f in train_formulas for f in sample_formula]
    test_formulas = [s["context"][1:] for s in test if s["target"] == vocab.message_end]

    counts = SuffixCounts(train)
    bigram = SuffixCounts(train, max_order=1)
    methods = {
        "most frequent token": baseline_hits(test, lambda c: counts.unigram_ranked),
        "bigram counts": baseline_hits(test, bigram.ranked),
        "longest matching suffix (training-set lookup)": baseline_hits(test, counts.ranked),
    }
    generation = {"generator (reference: test formulas)": generation_scores(test_formulas, grammar, train_formulas)}
    checkpoints = {}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        model, ck = load_model_from_checkpoint(path, vocab.vocab_size, device)
        model.eval()
        checkpoints[name] = {"path": path, "epoch": ck.get("epoch"), "best_val_loss": ck.get("best_val_loss"),
                             "params": sum(p.numel() for p in model.parameters())}
        ds = FOLDataset(args.test_data, max_seq_len=128, pad_id=vocab.pad_token)
        methods[name] = model_hits(model, ds, device)
        generation[name] = generation_scores(sample_formulas(model, vocab, device, args.n_generate),
                                             grammar, train_formulas)

    targets = [s["target"] for s in test]
    results = {name: summarise(h, roles, seen, formula_seen, targets) for name, h in methods.items()}
    ceil = role_ceilings(roles, dists, pool)
    n_unseen = sum(not s for s in seen)
    n_old_formula = sum(formula_seen)
    n_formulas_in_train = sum(tuple(f) in train_formulas for f in test_formulas)

    print(f"Test set: {len(test)} targets from {len(test_formulas)} formulas. {n_formulas_in_train} formulas "
          f"({n_old_formula} targets) also occur in the training set; {n_unseen} targets "
          f"({100 * n_unseen / len(test):.1f}%) have a context not in the training set.\n")
    print("| method | loss | top-1 | top-5 | formulas not in training: top-1 / in training: top-1 "
          "| unseen contexts: loss / top-1 / top-5 | seen contexts: top-1 | NUM_000 targets: top-1 / top-5 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        a, u, s = r["all"], r["unseen_context"], r["seen_context"]
        fn, fo = r["formula_not_in_train"], r["formula_in_train"]
        loss = f"{a['loss']:.3f}" if "loss" in a else "n/a"
        uloss = f"{u['loss']:.3f}" if "loss" in u else "n/a"
        print(f"| {name} | {loss} | {pct(a['top1'])} | {pct(a['top5'])} | {pct(fn['top1'])} / {pct(fo['top1'])} | "
              f"{uloss} / {pct(u['top1'])} / {pct(u['top5'])} | {pct(s['top1'])} | "
              f"{pct(r['num_000_target']['top1'])} / {pct(r['num_000_target']['top5'])} |")
    for k in ("top1", "top5"):
        print(f"\n{'Top-1' if k == 'top1' else 'Top-5'} accuracy (%) by the role of the target:\n")
        print("| method | " + " | ".join(ROLES) + " |")
        print("|---|" + "---|" * len(ROLES))
        print("| share of targets | " + " | ".join(pct(ceil[r]["share"]) for r in ROLES) + " |")
        print("| best achievable | " + " | ".join(pct(ceil[r][k]) for r in ROLES) + " |")
        for name, r in results.items():
            print(f"| {name} | " + " | ".join(pct(r["by_role"][x][k]) for x in ROLES) + " |")
    print("\nSampled formulas (temperature 1):\n")
    print("| source | n | ended | well-formed | bound-position variables in scope | novel | distinct | mean length |")
    print("|---|---|---|---|---|---|---|---|")
    for name, gsc in generation.items():
        ml = f"{gsc['mean_len']:.1f}" if gsc["mean_len"] is not None else "n/a"
        print(f"| {name} | {gsc['n']} | {pct(gsc['ended'])} | {pct(gsc['well_formed'])} | "
              f"{pct(gsc['bound_refs_in_scope'])} | {pct(gsc['novel'])} | {pct(gsc['distinct'])} | {ml} |")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "test_data": args.test_data, "train_data": args.train_data, "n_targets": len(test),
            "n_unseen_context": n_unseen, "n_test_formulas_in_train": n_formulas_in_train,
            "n_targets_from_train_formulas": n_old_formula, "pool": pool, "checkpoints": checkpoints,
            "results": results, "role_ceilings": ceil, "generation": generation}, indent=2))


if __name__ == "__main__":
    main()
