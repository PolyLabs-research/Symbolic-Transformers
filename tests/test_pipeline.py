"""Fast regression tests for the Symbolic-Transformers data pipeline.

Run from the repo root:  python -m pytest tests
Tests that need PyTorch are skipped automatically when it is not installed.
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.vocabulary import Vocabulary  # noqa: E402
from data.dataset_generator import (FOLFormulaGenerator, FormulaConfig, NextSymbolDataset,  # noqa: E402
                                    generate_training_data)

VOCAB_PATH = ROOT / "unified_vocabulary.json"


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(VOCAB_PATH))


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------

def test_vocab_layout(vocab):
    assert vocab.vocab_size == 671
    assert vocab.numeral_range == (0, 624)
    assert vocab.message_start == 662 and vocab.message_end == 663
    assert vocab.pad_token == 664 and vocab.pad_token != 0
    assert vocab.decode_id(0) == "NUM_000"
    assert set(vocab.compositional_tokens) == {"VAR", "CONST", "PRED", "FUNC", "SORT"}


def test_compositional_encoding_is_base_625(vocab):
    var = vocab.compositional_tokens["VAR"]
    assert vocab.encode_compositional("VAR", 0) == [var, 0]
    assert vocab.encode_compositional("VAR", 42) == [var, 42]
    assert vocab.encode_compositional("VAR", 624) == [var, 624]
    assert vocab.encode_compositional("VAR", 700) == [var, 1, 75]
    for idx in (0, 1, 42, 624, 625, 700, 390_624, 390_625):
        cat, value, nxt = vocab.decode_compositional(vocab.encode_compositional("PRED", idx))
        assert (cat, value) == ("PRED", idx)
        assert nxt == len(vocab.encode_compositional("PRED", idx))


def test_formula_round_trip(vocab):
    text = "FORALL VAR 1 LPAREN PRED 5 LPAREN VAR 1 RPAREN RPAREN"
    tokens = vocab.encode_formula_simple(text)
    assert vocab.decode_formula_simple(tokens) == text
    wrapped = vocab.wrap_with_boundaries(tokens)
    assert wrapped[0] == vocab.message_start and wrapped[-1] == vocab.message_end


def test_asset_paths_are_portable_and_exist(vocab):
    raw = json.load(open(VOCAB_PATH, encoding="utf-8"))
    for tok in raw["tokens"]:
        if tok.get("filename"):
            assert "\\" not in tok["path"], tok["label"]
            assert (ROOT / tok["path"]).exists(), tok["path"]


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------

def _generate(vocab, n, max_variables, max_predicates, seed=0):
    random.seed(seed)
    cfg = FormulaConfig(max_depth=3, max_variables=max_variables, max_predicates=max_predicates)
    gen = FOLFormulaGenerator(vocab, cfg)
    return gen.generate_batch(n, complexity=3)


def test_generator_covers_configured_index_range(vocab):
    formulas = _generate(vocab, 300, max_variables=100, max_predicates=100)
    numerals = {t for f in formulas for t in f if vocab.is_numeral(t)}
    assert max(numerals) >= 50, "indices should span well beyond 0-4"
    assert min(numerals) == 0
    assert all(n < 100 for n in numerals)


def test_generator_is_deterministic_given_seed(vocab):
    a = _generate(vocab, 50, 100, 100, seed=7)
    b = _generate(vocab, 50, 100, 100, seed=7)
    c = _generate(vocab, 50, 100, 100, seed=8)
    assert a == b
    assert a != c


def test_next_symbol_dataset_contexts_start_with_message_start(vocab):
    formulas = _generate(vocab, 20, 10, 10)
    ds = NextSymbolDataset(formulas, vocab, max_seq_len=128, use_boundaries=True)
    assert len(ds) == sum(len(f) + 1 for f in formulas)  # every position incl. MESSAGE_END
    assert all(s["context"][0] == vocab.message_start for s in ds.samples)
    assert any(s["target"] == vocab.message_end for s in ds.samples)
    assert any(s["target"] == 0 for s in ds.samples), "NUM_000 must appear as a target"


def _split_formulas(data_dir, split, vocab):
    """The formulas of a saved split: the context of every sample whose target is MESSAGE_END."""
    samples = json.load(open(Path(data_dir) / f"{split}.json"))["samples"]
    return [tuple(s["context"]) for s in samples if s["target"] == vocab.message_end]


def _assert_no_shared_formulas(data_dir, splits, vocab):
    seen = {}
    for split in splits:
        formulas = _split_formulas(data_dir, split, vocab)
        assert len(set(formulas)) == len(formulas), f"{split} repeats a formula"
        for f in formulas:
            assert f not in seen, f"{split} shares a formula with {seen[f]}"
            seen[f] = split


def test_generated_splits_share_no_formulas(tmp_path, vocab):
    # A three-index pool makes short formulas collide often, so the redraw path is exercised.
    generate_training_data(str(VOCAB_PATH), str(tmp_path), n_train=200, n_val=50, n_test=50,
                           index_range=(0, 2), holdout=(3, 4), seed=3)
    meta = json.load(open(tmp_path / "metadata.json"))
    assert meta["splits"]["train"]["duplicates_redrawn"] > 0
    assert meta["splits"]["val"]["duplicates_redrawn"] > 0
    splits = ["train", "val", "test", "test_holdout"]
    for split in splits:
        assert len(_split_formulas(tmp_path, split, vocab)) == meta["splits"][split]["num_formulas"]
    _assert_no_shared_formulas(tmp_path, splits, vocab)


def test_generator_refuses_a_split_its_index_pool_cannot_fill(tmp_path):
    # One index admits only a few dozen distinct depth-1 formulas.
    with pytest.raises(RuntimeError, match="distinct"):
        generate_training_data(str(VOCAB_PATH), str(tmp_path), n_train=500, n_val=10, n_test=10,
                               index_range=(0, 0), complexity_distribution=[(1, 1.0)], seed=1)


def test_shipped_dataset_splits_share_no_formulas(vocab):
    data_dir = ROOT / "datasets" / "fol_next_symbol"
    if not (data_dir / "train.json").exists():
        pytest.skip("shipped dataset not present")
    _assert_no_shared_formulas(data_dir, ["train", "val", "test"], vocab)


def test_benchmark_inputs_tokenize_without_torch_model(vocab):
    """BenchmarkRunner.tokenize must not require vocab.encode (which does not exist)."""
    from symbolic_benchmarks import BenchmarkRunner, BENCHMARKS

    class _Dummy:
        def eval(self):
            return self

    runner = BenchmarkRunner(_Dummy(), vocab, device="cpu")
    for category in BENCHMARKS.values():
        for test in category["tests"]:
            toks = runner.tokenize(test["input"])
            assert toks[0] == vocab.message_start
            assert toks.count(vocab.message_start) == 1
            if "compare_with" in test:
                assert runner.tokenize(test["compare_with"])[0] == vocab.message_start


# ----------------------------------------------------------------------------
# Training-side behaviour (needs torch)
# ----------------------------------------------------------------------------

def _write_dataset(tmp_path, vocab, formulas):
    ds = NextSymbolDataset(formulas, vocab, max_seq_len=512, use_boundaries=True)
    path = tmp_path / "train.json"
    ds.save(str(path))
    return path, ds


def test_fol_dataset_pads_with_pad_token_and_keeps_tail(tmp_path, vocab):
    torch = pytest.importorskip("torch")
    from training.train import FOLDataset

    formulas = _generate(vocab, 30, 10, 10)
    path, raw = _write_dataset(tmp_path, vocab, formulas)
    longest = max(len(s["context"]) for s in raw.samples)

    ds = FOLDataset(str(path), max_seq_len=512, pad_id=vocab.pad_token)
    assert ds.max_seq_len == longest, "should pad only as far as the data needs"
    for i in range(len(ds)):
        item = ds[i]
        length = int(item["length"])
        ctx = item["input_ids"].tolist()
        assert ctx[:length] == raw.samples[i]["context"]
        assert all(t == vocab.pad_token for t in ctx[length:])
        assert int(item["target"]) == raw.samples[i]["target"]

    short = FOLDataset(str(path), max_seq_len=5, pad_id=vocab.pad_token)
    for i in range(len(short)):
        item = short[i]
        full = raw.samples[i]["context"]
        assert item["input_ids"].tolist()[: int(item["length"])] == full[-5:]


def test_trainer_loss_counts_num_000_targets(vocab):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from models.transformer import create_model
    from training.train import Trainer, TrainingConfig

    model = create_model(vocab.vocab_size, "tiny")
    cfg = TrainingConfig(model_size="tiny", vocab_size=vocab.vocab_size, device="cpu", warmup_steps=1)
    loader = DataLoader([{"input_ids": torch.zeros(4, dtype=torch.long),
                          "target": torch.tensor(0), "length": torch.tensor(4)}], batch_size=1)
    trainer = Trainer(model, loader, loader, cfg, vocab)
    assert trainer.criterion.ignore_index == -100, "numeral 0 must contribute to the loss"
    logits = torch.zeros(2, vocab.vocab_size)
    loss = trainer.criterion(logits, torch.tensor([0, 5]))
    assert torch.isfinite(loss) and loss.item() > 0


def test_trimming_padding_leaves_last_token_logits_unchanged(vocab):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model
    from training.train import trim_padding

    torch.manual_seed(0)
    model = create_model(vocab.vocab_size, "tiny").eval()
    lengths = torch.tensor([3, 7, 5])
    ids = torch.full((3, 20), vocab.pad_token, dtype=torch.long)
    for i, n in enumerate(lengths.tolist()):
        ids[i, :n] = torch.randint(0, 625, (n,))
    trimmed = trim_padding(ids, lengths)
    assert trimmed.shape == (3, 7)
    rows = torch.arange(3)
    with torch.no_grad():
        full = model(ids)[rows, lengths - 1]
        short = model(trimmed)[rows, lengths - 1]
    assert torch.allclose(full, short, atol=1e-5)


def test_model_forward_and_benchmark_runner_end_to_end(vocab):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model
    from symbolic_benchmarks import BENCHMARKS, BenchmarkRunner

    model = create_model(vocab.vocab_size, "tiny")
    runner = BenchmarkRunner(model, vocab, device="cpu")
    preds = runner.get_predictions("FORALL VAR 624")
    assert len(preds) == vocab.vocab_size
    assert abs(sum(preds.values()) - 100.0) < 1e-3
    report = runner.run_category("deterministic")
    assert len(report["tests"]) == len(BENCHMARKS["deterministic"]["tests"])
    assert all("top5" in t and "passed" in t for t in report["tests"])
