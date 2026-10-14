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


# ----------------------------------------------------------------------------
# Sinusoid control arm
# ----------------------------------------------------------------------------

def test_sinusoid_bank_covers_numerals_only(vocab):
    from utils.grid_bank import sinusoid_bank, sinusoid_features

    bank = sinusoid_bank(vocab.vocab_size)
    assert bank.grids.shape == (vocab.vocab_size, 64) and bank.n_features == 64
    assert bank.has_grid[:625].all() and not bank.has_grid[625:].any()
    f = sinusoid_features(np.arange(625))
    assert np.allclose(bank.grids[:625], f)
    # Smooth in k: adjacent values are closer than distant ones, and the slowest
    # cosine (period 1250, half a cycle over 0..624) is monotone over the range.
    assert np.linalg.norm(f[10] - f[11]) < np.linalg.norm(f[10] - f[300])
    slowest_cos = f[:, 63]
    assert np.all(np.diff(slowest_cos) < 0)


def test_sinusoid_model_forward_and_roundtrip(tmp_path, vocab):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model, load_model_from_checkpoint

    model = create_model(vocab.vocab_size, "tiny", embedding_type="sinusoid", tie_weights=True, verbose=False)
    model.eval()
    assert model.symbol_embed.grids.shape == (vocab.vocab_size, 64)
    ids = torch.tensor([[vocab.message_start, 17, vocab.encode_label("LESS_THAN"), 599, vocab.answer_token]])
    with torch.no_grad():
        before = model(ids)
    assert before.shape == (1, 5, vocab.vocab_size) and torch.isfinite(before).all()
    # Glyph / special tokens come from the learned fallback rows.
    table = model.embedding_table()
    assert torch.allclose(table[vocab.answer_token], model.symbol_embed.fallback.weight[vocab.answer_token])
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"model_size": "tiny", "embedding_type": "sinusoid", "tie_weights": True}}, tmp_path / "s.pt")
    loaded, _ = load_model_from_checkpoint(tmp_path / "s.pt", vocab.vocab_size, "cpu", verbose=False)
    with torch.no_grad():
        after = loaded(ids)
    assert torch.allclose(before, after)
    # noise_flip is a panel-only concept and must leave sinusoid features alone.
    loaded.symbol_embed.noise_flip = 0.5
    with torch.no_grad():
        assert torch.allclose(loaded(ids), after)


# ----------------------------------------------------------------------------
# Shared-structure encoder
# ----------------------------------------------------------------------------

def test_cell_position_features_are_smooth():
    from utils.grid_bank import cell_position_features, GRID_N

    phi = cell_position_features()
    assert phi.shape[0] == 625 and np.isfinite(phi).all()
    # Neighbouring cells along the fill order are closer than distant ones.
    d_adj = np.linalg.norm(phi[100] - phi[101])
    d_far = np.linalg.norm(phi[100] - phi[400])
    assert d_adj < d_far
    # Cells in the same column share column features; the ramp columns are exact.
    assert np.allclose(phi[3, -4], 3 / 625) and np.allclose(phi[3 + GRID_N, -3], 1 / GRID_N)


def test_grid_shared_generates_cell_weights_and_roundtrips(tmp_path, vocab, bank):
    torch = pytest.importorskip("torch")
    from models.transformer import SharedCellEncoder, create_model, load_model_from_checkpoint

    enc = SharedCellEncoder(d_model=16)
    W = enc.cell_weights()
    assert W.shape == (625, 16)
    # No parameter belongs to a single cell: adjacent cells get similar weights.
    adj = (W[1:] - W[:-1]).norm(dim=1).mean().item()
    rnd = (W[torch.randperm(625)] - W).norm(dim=1).mean().item()
    assert adj < 0.5 * rnd
    # Encoder output is the sum of generated weights over filled cells (plus bias).
    bits = torch.tensor(bank.grids[:5], dtype=torch.float32)
    assert torch.allclose(enc(bits), bits @ W + enc.proj.bias, atol=1e-5)

    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_shared", tie_weights=True,
                         grid_bank=bank, verbose=False)
    model.eval()
    ids = torch.tensor([[vocab.message_start, 17, vocab.encode_label("LESS_THAN"), 599, vocab.answer_token]])
    with torch.no_grad():
        before = model(ids)
    assert torch.isfinite(before).all()
    assert model.count_parameters() < create_model(vocab.vocab_size, "tiny", embedding_type="grid_linear",
                                                   tie_weights=True, grid_bank=bank, verbose=False).count_parameters()
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"model_size": "tiny", "embedding_type": "grid_shared", "tie_weights": True}}, tmp_path / "g.pt")
    loaded, _ = load_model_from_checkpoint(tmp_path / "g.pt", vocab.vocab_size, "cpu", verbose=False)
    with torch.no_grad():
        assert torch.allclose(before, loaded(ids))


def test_sinusoid_rejects_panel_bank(vocab, bank):
    pytest.importorskip("torch")
    from models.transformer import create_model

    with pytest.raises(ValueError):
        create_model(vocab.vocab_size, "tiny", embedding_type="sinusoid", grid_bank=bank, verbose=False)
