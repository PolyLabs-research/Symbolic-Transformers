"""Tests for the Stage A formula-evaluation data (data/eval_generator.py) and the trainer's sequence mode."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.eval_generator import (MAX_LEN, Sampler, Tokens, evaluate, generate, height,  # noqa: E402
                                 reduce_innermost, substitute, trace)
from utils.vocabulary import Vocabulary  # noqa: E402

VOCAB = str(ROOT / "unified_vocabulary.json")


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(VOCAB)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval_a")
    generate(str(out), VOCAB, seed=3, scale=0.02)
    return out, {p.stem: json.load(open(p))["items"] for p in out.glob("*.json") if p.stem != "metadata"}


def test_evaluator_and_trace_on_a_hand_case():
    p, q = ("atom", (0, (5,))), ("atom", (4, (5, 7)))
    f = ("bin", "IMPLIES", ("not", p), ("bin", "AND", p, q))  # (NOT P(5) IMPLIES (P(5) AND R(5,7)))
    assert evaluate(f, {(0, (5,))}) is True  # NOT P false -> implication true
    assert evaluate(f, set()) is False  # NOT P true, (P AND R) false
    steps = trace(f, set())
    assert len(steps) == height(f) + 1 == 3
    assert steps[0] == ("bin", "IMPLIES", ("not", ("val", False)), ("bin", "AND", ("val", False), ("val", False)))
    assert steps[1] == ("bin", "IMPLIES", ("val", True), ("val", False))
    assert steps[2] == ("val", False)


def test_every_reduction_step_preserves_the_value():
    random.seed(0)
    s = Sampler(max_height=(1, 2, 3, 4, 5))
    for _ in range(300):
        objects, true_atoms, _, f = s.draw()
        w = set(true_atoms)
        steps = trace(f, w)
        assert len(steps) == height(f) + 1
        assert all(evaluate(st, set()) == evaluate(f, w) for st in steps)
        assert steps[-1] == ("val", evaluate(f, w))
        for a, b in zip(steps[1:], steps[2:]):
            assert height(b) == height(a) - 1


def _parse(seq, vocab):
    """World atoms and formula tokens back from a direct sequence."""
    tok = Tokens(vocab)
    sep, models = seq.index(tok.SEP), seq.index(tok.MODELS)
    objects = [seq[i + 1] for i in range(1, sep, 2)]
    atoms, i = set(), sep + 1
    while i < models:
        j = seq.index(tok.RP, i)
        body = seq[i:j + 1]
        consts = tuple(body[k + 1] for k in range(len(body)) if body[k] == tok.CONST)
        atoms.add((body[1], consts))
        i = j + 1
    return objects, atoms, seq[models + 1:-2]


def test_labels_are_balanced_and_items_unique(data):
    _, splits = data
    seen = set()
    for name, items in splits.items():
        strata = defaultdict(Counter)
        for it in items:
            strata[(it["outer"], it["height"], it["objects"])][it["label"]] += 1
            key = tuple(it["direct_seq"][:-1])
            assert key not in seen, f"duplicate item in {name}"
            seen.add(key)
        for c in strata.values():
            assert abs(c[0] - c[1]) <= (0 if items[0]["pair"] is not None else 1)


def test_twins_differ_in_one_atom_of_the_formula_and_flip_the_answer(data, vocab):
    _, splits = data
    for name, items in splits.items():
        if not name.startswith("test"):
            assert all(it["pair"] is None for it in items)
            continue
        groups = defaultdict(list)
        for it in items:
            groups[it["pair"]].append(it)
        assert all(len(g) == 2 for g in groups.values())
        for a, b in groups.values():
            oa, wa, fa = _parse(a["direct_seq"], vocab)
            ob, wb, fb = _parse(b["direct_seq"], vocab)
            assert oa == ob and fa == fb
            assert len(wa ^ wb) == 1
            assert a["label"] != b["label"]


def test_split_properties_and_formats(data, vocab):
    _, splits = data
    tok = Tokens(vocab)
    assert max(it["height"] for it in splits["train"]) <= 3
    assert min(it["height"] for it in splits["test_deep"]) >= 4
    assert min(it["objects"] for it in splits["test_large_world"]) >= 5
    for items in splits.values():
        for it in items:
            d, s = it["direct_seq"], it["scratch_seq"]
            answer = tok.TRUE if it["label"] else tok.FALSE
            assert d[-1] == s[-1] == answer and d[-2] == s[-2] == tok.ANSWER
            assert it["direct_from"] == len(d) - 1
            assert s[it["scratch_from"] - 1] == tok.QUERY and s[:it["scratch_from"] - 1] == d[:-2]
            assert len(s) <= MAX_LEN and max(s) < vocab.vocab_size
            # The scratchpad has one SEP-separated step per level plus the lookups.
            trace_part = s[it["scratch_from"]:-2]
            assert trace_part.count(tok.SEP) == it["height"]


def test_negation_split_negates_compound_formulas(data, vocab):
    _, splits = data
    tok = Tokens(vocab)
    for it in splits["test_negation"]:
        f = _parse(it["direct_seq"], vocab)[2]
        assert any(f[i] == tok.NOT and f[i + 1] in (tok.NOT, tok.LP) for i in range(len(f) - 1))


def test_sequence_dataset_labels_and_loss(data, vocab):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from models.transformer import create_model
    from training.train import SequenceDataset, Trainer, TrainingConfig

    out, splits = data
    raw = json.load(open(out / "val.json"))
    for fmt in ("direct", "scratch"):
        ds = SequenceDataset(raw, fmt, vocab.pad_token)
        it, item = ds[0], raw["items"][0]
        seq, start = item[f"{fmt}_seq"], item[f"{fmt}_from"]
        labels = it["labels"].tolist()
        n = len(seq) - 1
        assert it["input_ids"].tolist()[:n] == seq[:-1] and it["length"].item() == n
        assert labels[:start - 1] == [-100] * (start - 1)
        assert labels[start - 1:n] == seq[start:]
        assert all(x == -100 for x in labels[n:])
    torch.manual_seed(0)
    model = create_model(vocab.vocab_size, "tiny").eval()
    ds = SequenceDataset(raw, "direct", vocab.pad_token)
    loader = DataLoader(ds, batch_size=4)
    trainer = Trainer(model, loader, loader, TrainingConfig(model_size="tiny", vocab_size=vocab.vocab_size,
                                                           device="cpu", warmup_steps=1), vocab)
    batch = next(iter(loader))
    with torch.no_grad():
        loss = trainer.batch_loss(batch)
        # For the direct format the loss is exactly the cross-entropy of the answer token.
        rows = []
        for i in range(4):
            n = batch["length"][i].item()
            logits = model(batch["input_ids"][i:i + 1, :n])[0, -1]
            rows.append(torch.nn.functional.cross_entropy(logits[None], batch["labels"][i, n - 1][None]))
    assert torch.isfinite(loss) and abs(loss.item() - torch.stack(rows).mean().item()) < 1e-4


def test_length_bucket_sampler_covers_every_item_once():
    pytest.importorskip("torch")
    from training.train import LengthBucketSampler

    lengths = [random.Random(i).randint(5, 300) for i in range(1037)]
    for shuffle in (True, False):
        s = LengthBucketSampler(lengths, 32, shuffle=shuffle, bucket=20)
        batches = list(s)
        assert len(batches) == len(s)
        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
        assert all(len(b) <= 32 for b in batches)
        spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches) / len(batches)
        assert spread < 40, "batches should hold items of similar length"


@pytest.mark.parametrize("encoding", ["true_atoms", "full_table"])
def test_lookup_items(tmp_path, vocab, encoding):
    from data.eval_generator import possible_atoms

    generate(str(tmp_path), VOCAB, seed=4, scale=0.01, lookup=True, encoding=encoding)
    tok = Tokens(vocab)
    for split in ("train", "test_iid", "test_large_world"):
        items = json.load(open(tmp_path / f"{split}.json"))["items"]
        assert all(it["height"] == 0 for it in items)
        assert abs(sum(it["label"] for it in items) / len(items) - 0.5) < 0.02
        for it in items:
            seq = it["direct_seq"]
            models = seq.index(tok.MODELS)
            assert seq[models + 1] == tok.PRED and seq[-2] == tok.ANSWER  # the formula is one atom
            if encoding == "full_table":
                sep = seq.index(tok.SEP)
                objects = [seq[i + 1] for i in range(1, sep, 2)]
                world = seq[sep + 1:models]
                values = [world[i + 1] for i in range(len(world) - 1) if world[i] == tok.RP]
                assert len(values) == len(possible_atoms(objects))
                assert all(v in (tok.TRUE, tok.FALSE) for v in values)
                formula = seq[models + 1:-2]
                k = next(i for i in range(len(world)) if world[i:i + len(formula)] == formula)
                assert world[k + len(formula)] == seq[-1], "the atom's listed value is the answer"
        if split.startswith("test"):
            pairs = defaultdict(list)
            for it in items:
                pairs[it["pair"]].append(it)
            for a, b in pairs.values():
                da, db = a["direct_seq"][:-1], b["direct_seq"][:-1]
                if encoding == "full_table":
                    assert len(da) == len(db) and sum(x != y for x, y in zip(da, db)) == 1
                else:
                    assert abs(len(da) - len(db)) in (6, 9)  # one unary or binary atom added or removed
                assert a["label"] != b["label"]


@pytest.mark.parametrize("signature,preds,objects", [("single", {0}, {12, 21, 32}), ("unary", {0, 1, 2, 3}, {3, 5, 8})])
def test_lookup_ladder_signatures(tmp_path, vocab, signature, preds, objects):
    generate(str(tmp_path), VOCAB, seed=6, scale=0.01, lookup=True, signature=signature)
    tok = Tokens(vocab)
    assert not (tmp_path / "test_large_world.json").exists()
    for split in ("train", "test_iid"):
        items = json.load(open(tmp_path / f"{split}.json"))["items"]
        assert {it["objects"] for it in items} <= objects
        assert abs(sum(it["label"] for it in items) / len(items) - 0.5) < 0.03
        for it in items:
            seq = it["direct_seq"]
            used = {seq[i + 1] for i in range(len(seq) - 1) if seq[i] == tok.PRED}
            assert used <= preds and tok.COMMA not in seq, "no binary atoms on this rung"


def test_multi_query_training_items(tmp_path, vocab):
    from training.train import SequenceDataset

    generate(str(tmp_path), VOCAB, seed=7, scale=0.01, lookup=True, signature="single",
             encoding="full_table", queries=8)
    tok = Tokens(vocab)
    train = json.load(open(tmp_path / "train.json"))
    values = []
    for it in train["items"]:
        seq, targets = it["direct_seq"], it["direct_targets"]
        assert len(targets) == it["queries"] == 8
        assert all(seq[t - 1] == tok.ANSWER and seq[t] in (tok.TRUE, tok.FALSE) for t in targets)
        values += [seq[t] == tok.TRUE for t in targets]
        # Each answer is the value the full table lists for that query's atom.
        models = seq.index(tok.MODELS)
        world = seq[seq.index(tok.SEP) + 1:models]
        for t in targets:
            start = max(i for i in range(models, t) if seq[i] == tok.PRED)
            atom = seq[start:t - 1]
            k = next(i for i in range(len(world)) if world[i:i + len(atom)] == atom)
            assert world[k + len(atom)] == seq[t]
    assert abs(sum(values) / len(values) - 0.5) < 0.05
    val = json.load(open(tmp_path / "val.json"))["items"]
    assert all("direct_targets" not in it for it in val), "validation items keep a single query"
    ds = SequenceDataset(train, "direct", vocab.pad_token)
    it, item = ds[0], train["items"][0]
    labelled = [j + 1 for j, x in enumerate(it["labels"].tolist()) if x != -100]
    assert labelled == item["direct_targets"]
