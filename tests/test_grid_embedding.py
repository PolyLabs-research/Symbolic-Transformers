"""Tests for the grid bank, grid embeddings, index pools and the split-aware evaluator."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.grid_bank import GRID_CELLS, GridBank, build_grid_bank, thermometer_grid  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402
from data.dataset_generator import FOLFormulaGenerator, FormulaConfig, generate_training_data  # noqa: E402

VOCAB_PATH = ROOT / "unified_vocabulary.json"
BANK_PATH = ROOT / "grids" / "grid_bank.npz"


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(VOCAB_PATH))


@pytest.fixture(scope="module")
def bank():
    return GridBank.load(BANK_PATH)


# ----------------------------------------------------------------------------
# Grid bank
# ----------------------------------------------------------------------------

def test_thermometer_is_nested():
    prev = thermometer_grid(0)
    for k in range(1, GRID_CELLS + 1):
        g = thermometer_grid(k)
        assert g.sum() == k and np.all(prev <= g)
        prev = g


def test_shipped_bank_matches_vocabulary(vocab, bank):
    assert bank.vocab_size == vocab.vocab_size
    raw = json.load(open(VOCAB_PATH, encoding="utf-8"))
    for tok in raw["tokens"]:
        tid = tok["id"]
        if tok["type"] == "numeral":
            assert bank.has_grid[tid] and bank.grids[tid].sum() == tid
        elif tok.get("path"):
            assert bank.has_grid[tid] and bank.grids[tid].sum() == tok["black_cells"], tok["label"]
        else:
            assert not bank.has_grid[tid] and bank.grids[tid].sum() == 0
    assert bank.summary() == {"vocab_size": 671, "grid_backed": 662, "numerals": 625, "glyphs": 37}


def test_bank_rebuild_is_identical(bank):
    rebuilt = build_grid_bank(VOCAB_PATH, repo_root=ROOT, verify_pngs=True)
    assert np.array_equal(rebuilt.grids, bank.grids) and np.array_equal(rebuilt.has_grid, bank.has_grid)


# ----------------------------------------------------------------------------
# Generator index pools
# ----------------------------------------------------------------------------

def _numerals(formulas, vocab):
    return {t for f in formulas for t in f if vocab.is_numeral(t)}


def test_pools_restrict_indices(vocab):
    random.seed(3)
    pool = [i for i in range(0, 100) if not 60 <= i <= 79]
    cfg = FormulaConfig(max_depth=3, variable_pool=pool, predicate_pool=pool)
    gen = FOLFormulaGenerator(vocab, cfg)
    seen = _numerals(gen.generate_batch(300, complexity=3), vocab)
    assert seen and seen <= set(pool)
    assert not any(60 <= i <= 79 for i in seen)
    gen2 = FOLFormulaGenerator(vocab, FormulaConfig(max_depth=3, variable_pool=list(range(60, 80)),
                                                    predicate_pool=list(range(60, 80))))
    seen2 = _numerals(gen2.generate_batch(100, complexity=3), vocab)
    assert seen2 and seen2 <= set(range(60, 80))


def test_generate_training_data_with_holdout(tmp_path):
    generate_training_data(str(VOCAB_PATH), str(tmp_path), n_train=60, n_val=10, n_test=10,
                           index_range=(0, 49), holdout=(20, 29), far_range=(100, 119), seed=5)
    meta = json.load(open(tmp_path / "metadata.json"))
    assert meta["index_pools"]["holdout"] == [20, 29] and meta["index_pools"]["far"] == [100, 119]
    assert meta["index_pools"]["train_pool_size"] == 40
    v = Vocabulary(str(VOCAB_PATH))

    def numerals_in(split):
        d = json.load(open(tmp_path / f"{split}.json"))
        return {t for s in d["samples"] for t in s["context"] + [s["target"]] if v.is_numeral(t)}

    assert not numerals_in("train") & set(range(20, 30))
    assert numerals_in("train") <= set(range(0, 50))
    assert numerals_in("test_holdout") <= set(range(20, 30))
    assert numerals_in("test_far") <= set(range(100, 120))


# ----------------------------------------------------------------------------
# Model (needs torch)
# ----------------------------------------------------------------------------

def test_grid_embedding_table_and_tying(vocab, bank):
    torch = pytest.importorskip("torch")
    from models.transformer import GridEmbedding, create_model

    emb = GridEmbedding(bank, d_model=32, kind="grid_linear")
    table = emb.table()
    assert table.shape == (vocab.vocab_size, 32)
    # Tokens without a panel come from the learned fallback rows.
    pad = vocab.pad_token
    assert torch.allclose(table[pad], emb.fallback.weight[pad])
    # Numeral rows are a function of the panel: adjacent k's differ by one cell,
    # so they must be closer than distant k's.
    d_adj = (table[10] - table[11]).norm().item()
    d_far = (table[10] - table[600]).norm().item()
    assert d_adj < d_far

    for kind in ("grid_linear", "grid_conv"):
        model = create_model(vocab.vocab_size, "tiny", embedding_type=kind, tie_weights=True,
                             grid_bank=bank, verbose=False)
        ids = torch.tensor([[vocab.message_start, 641, 653, 624, 655, 99, 630]])  # FORALL VAR 624 PRED 99 (
        logits = model(ids)
        assert logits.shape == (1, 7, vocab.vocab_size) and torch.isfinite(logits).all()
        assert model.output_proj is None and model.output_bias.shape == (vocab.vocab_size,)


def test_learned_tied_and_untied_shapes(vocab):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model

    tied = create_model(vocab.vocab_size, "tiny", embedding_type="learned", tie_weights=True, verbose=False)
    untied = create_model(vocab.vocab_size, "tiny", verbose=False)
    ids = torch.randint(0, vocab.vocab_size, (2, 9))
    assert tied(ids).shape == untied(ids).shape == (2, 9, vocab.vocab_size)
    assert tied.count_parameters() < untied.count_parameters()


def test_checkpoint_roundtrip_restores_grid_buffers(tmp_path, vocab, bank):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model, load_model_from_checkpoint

    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_linear", tie_weights=True,
                         grid_bank=bank, verbose=False)
    model.eval()
    ids = torch.tensor([[vocab.message_start, 641, 653, 70]])
    with torch.no_grad():
        before = model(ids)
    ckpt = {"model_state_dict": model.state_dict(),
            "config": {"model_size": "tiny", "embedding_type": "grid_linear", "tie_weights": True}}
    torch.save(ckpt, tmp_path / "m.pt")
    loaded, _ = load_model_from_checkpoint(tmp_path / "m.pt", vocab.vocab_size, "cpu", verbose=False)
    with torch.no_grad():
        after = loaded(ids)
    assert torch.allclose(before, after)
    assert bool(loaded.symbol_embed.has_grid[70]) and loaded.symbol_embed.grids[70].sum().item() == 70

    # Old-style checkpoints (no embedding keys) still load as learned/untied.
    old = create_model(vocab.vocab_size, "tiny", verbose=False)
    torch.save({"model_state_dict": old.state_dict(), "config": {"model_size": "tiny"}}, tmp_path / "old.pt")
    loaded_old, _ = load_model_from_checkpoint(tmp_path / "old.pt", vocab.vocab_size, "cpu", verbose=False)
    assert loaded_old.embedding_type == "learned" and loaded_old.output_proj is not None


def test_noise_flip_only_touches_grid_rows(vocab, bank):
    torch = pytest.importorskip("torch")
    from models.transformer import GridEmbedding

    emb = GridEmbedding(bank, d_model=16)
    clean = emb.table().detach()
    emb.noise_flip = 0.2
    torch.manual_seed(0)
    noisy = emb.table().detach()
    pad = vocab.pad_token
    assert torch.allclose(clean[pad], noisy[pad])
    assert not torch.allclose(clean[:625], noisy[:625])


# ----------------------------------------------------------------------------
# Evaluator categories
# ----------------------------------------------------------------------------

def test_evaluator_categories(vocab):
    pytest.importorskip("torch")
    from evaluate_grid import categorize

    var = vocab.compositional_tokens["VAR"]
    pred = vocab.compositional_tokens["PRED"]
    ms = vocab.message_start
    # FORALL VAR 7 PRED 3 ( VAR -> 7  : copy of a bound variable
    assert set(categorize([ms, 641, var, 7, pred, 3, 630, var], 7, vocab, var)) == {"overall", "numeral", "copy"}
    # PRED 3 ( VAR -> 9 with no earlier VAR 9 : fresh numeral
    assert set(categorize([ms, pred, 3, 630, var], 9, vocab, var)) == {"overall", "numeral", "fresh"}
    # ... VAR 7 -> RPAREN : structure right after a numeral
    assert set(categorize([ms, pred, 3, 630, var, 7], 631, vocab, var)) == {"overall", "structure", "after_numeral"}
    # MESSAGE_START -> FORALL : plain structure
    assert set(categorize([ms], 641, vocab, var)) == {"overall", "structure"}
