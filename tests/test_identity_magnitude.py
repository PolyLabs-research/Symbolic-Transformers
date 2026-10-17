"""Tests for the grid-cell numeral codes and scripts/identity_magnitude.py."""
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import identity_magnitude as im  # noqa: E402
from models.transformer import create_model, load_model_from_checkpoint  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(ROOT / "unified_vocabulary.json"))


@pytest.mark.parametrize("code", ["grid_cell", "grid_cell_xval"])
def test_grid_cell_rows(vocab, code):
    torch.manual_seed(0)
    emb = create_model(vocab.vocab_size, "tiny", embedding_type=code, verbose=False).symbol_embed
    table = emb.table()
    for k in (0, 24, 25, 377, 624):
        want = emb.rows[k // 25] + emb.cols[k % 25]
        if code == "grid_cell_xval":
            want = want + (k - 249.5) / 144.3 * emb.direction
        assert torch.allclose(table[k], want)
    plus = vocab.encode_label("PLUS")
    assert torch.equal(table[plus], emb.fallback.weight[plus])


def test_xval_keeps_its_parameters(vocab):
    emb = create_model(vocab.vocab_size, "tiny", embedding_type="xval", verbose=False).symbol_embed
    assert emb.rows is None and not any("rows" in n or "cols" in n for n, _ in emb.named_parameters())


def test_grid_cell_checkpoint_round_trip(vocab, tmp_path):
    torch.manual_seed(0)
    m = create_model(vocab.vocab_size, "tiny", embedding_type="grid_cell_xval", verbose=False).eval()
    path = tmp_path / "ck.pt"
    torch.save({"model_state_dict": m.state_dict(),
                "config": {"model_size": "tiny", "embedding_type": "grid_cell_xval"}}, path)
    loaded, _ = load_model_from_checkpoint(path, vocab.vocab_size, verbose=False)
    x = torch.randint(0, vocab.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.allclose(loaded(x), m(x))


def test_heldout_set_leaves_every_row_and_column_in_training():
    assert len(im.HELDOUT) == 100 and not set(im.HELDOUT) & set(im.POOL)
    assert sorted(im.HELDOUT + im.POOL) == list(range(500))
    assert {k // 25 for k in im.POOL} == set(range(20)) and {k % 25 for k in im.POOL} == set(range(25))


def test_recall_batch_answers_and_adjacent_keys(vocab):
    random.seed(0)
    seq, targets, _ = im.make_batch(vocab, "recall", 32, 4, [(im.POOL, True)])
    for row in seq.tolist():
        facts = dict(zip(row[1:17:2], row[2:17:2]))
        keys = sorted(facts)
        i = im.POOL.index(keys[0])
        assert keys == im.POOL[i:i + 8]
        for t in targets:
            assert row[t - 2] == vocab.encode_label("ANSWER") and row[t] == facts[row[t - 1]]


def test_compare_batch_labels_and_close_pairs(vocab):
    random.seed(1)
    seq, targets, gaps = im.make_batch(vocab, "compare", 2000, 2, im.HELDOUT)
    lt, true = vocab.encode_label("LESS_THAN"), vocab.encode_label("TRUE")
    for row in seq[:200].tolist():
        for t in targets:
            a, op, b = row[t - 4], row[t - 3], row[t - 2]
            assert a in im.HELDOUT and b in im.HELDOUT and a != b
            assert (row[t] == true) == ((a < b) if op == lt else (a > b))
    close = sum(g <= im.CLOSE_MAX for g in gaps) / len(gaps)
    assert 0.38 < close < 0.50   # 40% drawn close, plus random pairs that land close by chance


@pytest.mark.parametrize("code", list(im.CODES))
def test_every_code_builds_and_runs(vocab, code):
    model = create_model(vocab.vocab_size, "tiny", verbose=False, **im.CODES[code])
    random.seed(2)
    seq, targets, _ = im.make_batch(vocab, "recall", 2, 2, [(im.POOL, False)])
    assert model(seq[:, :-1]).shape == (2, seq.size(1) - 1, vocab.vocab_size)
