"""Tests for one token per symbol (utils/fused.py), the fused model, and the key-last lookup encoding."""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.eval_generator import (Sampler, Tokens, build_multi_query_split, build_split,  # noqa: E402
                                 possible_atoms)
from models.transformer import create_model, load_model_from_checkpoint  # noqa: E402
from training.train import SequenceDataset  # noqa: E402
from utils.fused import N_INDEX, SymbolFuser  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(ROOT / "unified_vocabulary.json"))


@pytest.fixture(scope="module")
def fuser(vocab):
    return SymbolFuser(vocab)


@pytest.fixture(scope="module")
def tok(vocab):
    return Tokens(vocab)


def test_fuse_and_unfuse_round_trip(vocab, fuser, tok):
    seq = [tok.start] + tok.atom((4, (3, 7))) + [tok.ANSWER, tok.TRUE]
    fused, where = fuser.fuse(seq)
    assert fuser.unfuse(fused) == seq
    assert len(fused) == len(seq) - 3
    assert fused[1] == fuser.fused_id(tok.PRED, 4) == vocab.vocab_size + 2 * N_INDEX + 4
    assert [fuser.label(t, vocab) for t in fused[1:7]] == ["PRED_4", "LPAREN", "CONST_3", "COMMA", "CONST_7",
                                                           "RPAREN"]
    for j, t in enumerate(seq):  # every token lands in the fused token that holds it
        assert t in ([fused[where[j]]] if fused[where[j]] < fuser.base else list(fuser.split(fused[where[j]])))


def test_a_category_without_its_numeral_is_rejected(fuser, tok):
    with pytest.raises(ValueError):
        fuser.fuse([tok.start, tok.CONST, tok.RP])
    with pytest.raises(ValueError):
        fuser.fuse([tok.start, tok.CONST])


def test_fused_items_keep_their_targets(fuser, tok):
    random.seed(0)
    items, _ = build_multi_query_split(tok, Sampler(objects=(3,), max_height=(0,), min_height=0), 20, set(), 4,
                                       "full_table")
    for it in items:
        f = fuser.fuse_item(it)
        assert fuser.unfuse(f["direct_seq"]) == it["direct_seq"]
        assert [f["direct_seq"][t] for t in f["direct_targets"]] == [it["direct_seq"][t] for t in it["direct_targets"]]
        assert f["direct_seq"][f["direct_from"]] == it["direct_seq"][it["direct_from"]]


def test_sequence_dataset_labels_the_fused_tokens(vocab, fuser, tok):
    random.seed(1)
    items, _ = build_multi_query_split(tok, Sampler(objects=(2,), max_height=(0,), min_height=0), 8, set(), 3,
                                       "keylast")
    ds = SequenceDataset({"items": items}, "direct", vocab.pad_token, fuser=fuser)
    for k, it in enumerate(ds.items):
        row = ds[k]
        labelled = [j + 1 for j, y in enumerate(row["labels"].tolist()) if y != -100]
        assert labelled == it["direct_targets"]
        assert [row["labels"][j - 1].item() for j in labelled] == [it["direct_seq"][j] for j in labelled]


def test_fuse_samples_matches_the_split_samples(fuser):
    data = json.load(open(ROOT / "datasets/fol_next_symbol/val.json"))["samples"][:3000]
    fused = fuser.fuse_samples(data)
    prefixes = {tuple(s["context"] + [s["target"]]) for s in data}
    after_category = sum(1 for s in data if s["context"] and s["context"][-1] in fuser.cat_index)
    assert len(fused) == len(data) - after_category
    assert all(tuple(fuser.unfuse(s["context"] + [s["target"]])) in prefixes for s in fused)


@pytest.mark.parametrize("tie", [False, True])
def test_fused_output_is_category_times_index(vocab, fuser, tie):
    """With the per-category maps at their initial identity, P(c, n) = P_split(c) * softmax over numerals."""
    torch.manual_seed(0)
    fused = create_model(vocab.vocab_size, "tiny", tie_weights=tie, verbose=False,
                         fused_categories=fuser.categories).eval()
    split = create_model(vocab.vocab_size, "tiny", tie_weights=tie, verbose=False).eval()
    missing = split.load_state_dict(fused.state_dict(), strict=False)
    assert set(missing.unexpected_keys) == {"fused_categories", "index_maps"} and not missing.missing_keys
    x = torch.randint(0, vocab.vocab_size, (2, 9))
    with torch.no_grad():
        p_f = fused(x).softmax(-1)
        z = split(x)
    p_s = z.softmax(-1)
    cats = fuser.categories
    others = [t for t in range(vocab.vocab_size) if t not in cats]
    assert torch.allclose(p_f.sum(-1), torch.ones(2, 9), atol=1e-5)
    assert torch.equal(p_f[..., cats], torch.zeros(2, 9, len(cats)))
    assert torch.allclose(p_f[..., others], p_s[..., others], atol=1e-6)
    index = z[..., :N_INDEX].softmax(-1)
    for c, cat in enumerate(cats):
        block = p_f[..., fuser.base + c * N_INDEX: fuser.base + (c + 1) * N_INDEX]
        assert torch.allclose(block, p_s[..., cat, None] * index, atol=1e-6)


def test_fused_input_row_is_category_plus_numeral(vocab, fuser):
    torch.manual_seed(0)
    m = create_model(vocab.vocab_size, "tiny", verbose=False, fused_categories=fuser.categories)
    table = m.embedding_table()
    full = m.input_table(table)
    assert full.shape == (fuser.size, m.d_model)
    t = fuser.fused_id(fuser.categories[1], 57)
    assert torch.allclose(full[t], table[fuser.categories[1]] + table[57])
    assert torch.equal(full[:vocab.vocab_size], table)


def test_fused_checkpoint_round_trip(vocab, fuser, tmp_path):
    torch.manual_seed(0)
    m = create_model(vocab.vocab_size, "tiny", verbose=False, fused_categories=fuser.categories).eval()
    with torch.no_grad():
        m.index_maps.normal_(0, 0.1)
    path = tmp_path / "ck.pt"
    torch.save({"model_state_dict": m.state_dict(), "config": {"model_size": "tiny", "fuse_symbols": True}}, path)
    loaded, _ = load_model_from_checkpoint(path, vocab.vocab_size, verbose=False)
    x = torch.randint(0, fuser.size, (2, 7))
    with torch.no_grad():
        assert torch.allclose(loaded(x), m(x))


def test_keylast_items_answer_at_the_key(vocab, tok):
    random.seed(2)
    sampler = Sampler(objects=(3,), max_height=(0,), min_height=0)
    items, _ = build_split(tok, sampler, 30, set(), True, "keylast")
    by_pair = {}
    for it in items:
        seq = it["direct_seq"]
        assert it["scratch_seq"] == seq and it["direct_from"] == len(seq) - 1
        models = seq.index(tok.MODELS)
        query = seq[models + 1:-1]
        assert query[-2] == tok.CONST and seq[-1] == (tok.TRUE if it["label"] else tok.FALSE)
        # the fact for the queried atom: the same open atom, then its value, then RPAREN
        facts = seq[:models]
        hits = [j for j in range(len(facts) - len(query)) if facts[j:j + len(query)] == query]
        assert len(hits) == 1
        j = hits[0] + len(query)
        assert facts[j] == seq[-1] and facts[j + 1] == tok.RP
        by_pair.setdefault(it["pair"], []).append(it)
    for twins in by_pair.values():  # a twin flips only the queried atom's value
        assert len(twins) == 2 and twins[0]["label"] != twins[1]["label"]
        a, b = twins[0]["direct_seq"], twins[1]["direct_seq"]
        assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 2


def test_keylast_multi_query_targets(tok):
    random.seed(3)
    sampler = Sampler(objects=(2, 3), max_height=(0,), min_height=0)
    items, _ = build_multi_query_split(tok, sampler, 10, set(), 4, "keylast")
    for it in items:
        seq = it["direct_seq"]
        models = seq.index(tok.MODELS)
        facts = seq[:models]
        for t in it["direct_targets"]:
            start = max(j for j in range(models, t) if seq[j] == tok.PRED)
            query = seq[start:t]
            hits = [j for j in range(len(facts) - len(query)) if facts[j:j + len(query)] == query]
            assert len(hits) == 1 and facts[hits[0] + len(query)] == seq[t]
        assert len(seq) <= 512


def test_keylast_tables_list_every_atom(tok):
    objects = [5, 9]
    listing = [(a, i % 2 == 0) for i, a in enumerate(possible_atoms(objects))]
    table = tok.table(objects, listing, keylast=True)
    assert table.count(tok.RP) == len(listing) and table.count(tok.TRUE) + table.count(tok.FALSE) == len(listing)


@pytest.mark.parametrize("sampling, low, high", [("balanced", 0.45, 0.55), ("uniform", 0.55, 0.70)])
def test_training_query_sampling_and_the_majority_shortcut(tok, sampling, low, high):
    """Uniform queries let 'answer with the world's majority value' beat chance; balanced queries do not."""
    random.seed(4)
    sampler = Sampler(objects=(12,), max_height=(0,), min_height=0, unary=[0], binary=[])
    items, _ = build_multi_query_split(tok, sampler, 300, set(), 8, "keylast", query_sampling=sampling)
    right = total = 0
    for it in items:
        seq = it["direct_seq"]
        world = seq[:seq.index(tok.MODELS)]
        majority = tok.TRUE if world.count(tok.TRUE) > world.count(tok.FALSE) else tok.FALSE
        for t in it["direct_targets"]:
            right += seq[t] == majority
            total += 1
    assert low < right / total < high
