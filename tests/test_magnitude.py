"""Tests for the magnitude comparison task: generator, evaluator bins and embedding probes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.magnitude_generator import OPS, MagnitudeGenerator, generate_magnitude_data  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

VOCAB_PATH = ROOT / "unified_vocabulary.json"


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(VOCAB_PATH))


def test_sample_labels_and_format(vocab):
    gen = MagnitudeGenerator(vocab)
    pool = set(range(0, 100))
    s = gen.make_sample(17, 42, "LESS_THAN", pool)
    assert s["context"] == [vocab.message_start, 17, vocab.encode_label("LESS_THAN"), 42, vocab.answer_token]
    assert s["target"] == vocab.encode_label("TRUE") and s["label"] == "TRUE" and s["n_unseen"] == 0
    assert gen.make_sample(42, 17, "LESS_THAN", pool)["label"] == "FALSE"
    assert gen.make_sample(42, 42, "LESS_THAN", pool)["label"] == "FALSE"
    assert gen.make_sample(42, 17, "GREATER_THAN", pool)["label"] == "TRUE"
    assert gen.make_sample(150, 17, "GREATER_THAN", pool)["n_unseen"] == 1
    assert gen.make_sample(150, 170, "GREATER_THAN", pool)["n_unseen"] == 2
    for op, fn in OPS.items():
        assert gen.make_sample(3, 5, op, pool)["label"] == ("TRUE" if fn(3, 5) else "FALSE")


def test_generate_splits_are_disjoint_and_pooled(tmp_path):
    meta = generate_magnitude_data(str(VOCAB_PATH), str(tmp_path), n_train=800, n_val=100, n_test=200,
                                   value_range=(0, 99), holdout=(40, 59), far_range=(100, 139), seed=3)
    assert meta["value_pools"]["train_pool_size"] == 80

    def load(name):
        return json.load(open(tmp_path / f"{name}.json"))["samples"]

    train, val, test, hold, far = (load(n) for n in ("train", "val", "test", "test_holdout", "test_far"))
    key = lambda s: (s["a"], s["b"], s["op"])  # noqa: E731
    assert not {key(s) for s in train} & {key(s) for s in test}
    assert not {key(s) for s in train} & {key(s) for s in val}
    pool = set(range(0, 100)) - set(range(40, 60))
    for s in train + val + test:
        assert s["a"] in pool and s["b"] in pool and s["n_unseen"] == 0
    for s in hold:
        assert s["n_unseen"] >= 1 and any(40 <= x <= 59 for x in (s["a"], s["b"]))
        assert all(x in pool or 40 <= x <= 59 for x in (s["a"], s["b"]))
    for s in far:
        assert s["n_unseen"] >= 1 and any(100 <= x <= 139 for x in (s["a"], s["b"]))
    labels = [s["label"] for s in train]
    assert 0.35 < labels.count("TRUE") / len(labels) < 0.65
    # every sample's label matches the operator applied to its operands
    for s in train + hold + far:
        assert s["label"] == ("TRUE" if OPS[s["op"]](s["a"], s["b"]) else "FALSE")


def test_evaluator_bins_and_signed_difference():
    pytest.importorskip("torch")
    from evaluate_magnitude import gap_bin, signed_bin, signed_difference

    assert gap_bin(0) == "0" and gap_bin(1) == "1" and gap_bin(3) == "2-5" and gap_bin(500) == "101+"
    assert signed_bin(-500) == "<=-101" and signed_bin(0) == "0" and signed_bin(7) == "6-20"
    assert signed_difference({"a": 3, "b": 10, "op": "LESS_THAN"}) == 7
    assert signed_difference({"a": 3, "b": 10, "op": "GREATER_THAN"}) == -7


def test_magnitude_axis_probe_detects_a_magnitude_axis():
    pytest.importorskip("torch")
    from evaluate_magnitude import magnitude_axis_probe

    rng = np.random.default_rng(0)
    k = np.arange(671, dtype=float)
    axis = rng.normal(size=16)
    # A table whose numeral rows move along one direction with k (plus noise)
    # must be readable everywhere; a random table must not.
    structured = np.outer(k / 625.0, axis) + 0.002 * rng.normal(size=(671, 16))
    random_table = rng.normal(size=(671, 16))
    pools = {"train": [v for v in range(0, 200) if not 80 <= v <= 119], "holdout": list(range(80, 120)),
             "far": list(range(200, 300))}
    t = magnitude_axis_probe(structured, pools)
    r = magnitude_axis_probe(random_table, pools)
    assert t["readout"]["train"]["spearman"] > 0.99
    assert t["readout"]["holdout"]["spearman"] > 0.95 and t["readout"]["far"]["spearman"] > 0.95
    assert t["readout"]["far"]["r2"] > 0.9
    assert abs(t["pc1_spearman"]["holdout"]) > 0.95
    assert abs(r["readout"]["holdout"]["spearman"]) < 0.5 and r["readout"]["far"]["r2"] < 0.5


def test_probe_on_untrained_grid_linear_interpolates(vocab):
    """An untrained grid_linear table already carries magnitude for held-out values (interpolation)."""
    torch = pytest.importorskip("torch")
    from evaluate_magnitude import magnitude_axis_probe
    from models.transformer import create_model
    from utils.grid_bank import GridBank

    torch.manual_seed(0)
    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_linear", tie_weights=True,
                         grid_bank=GridBank.load(ROOT / "grids" / "grid_bank.npz"), verbose=False)
    with torch.no_grad():
        table = model.embedding_table().cpu().numpy()
    pools = {"train": [v for v in range(0, 500) if not 200 <= v <= 299], "holdout": list(range(200, 300))}
    probe = magnitude_axis_probe(table, pools)
    assert probe["readout"]["holdout"]["spearman"] > 0.8


def test_small_gap_oversampling(vocab):
    import random
    from data.magnitude_generator import MagnitudeGenerator

    random.seed(0)
    pool = list(range(0, 500))
    plain = MagnitudeGenerator(vocab, small_gap_frac=0.0).sample_pairs(3000, pool, pool, set(pool), set())
    rich = MagnitudeGenerator(vocab, small_gap_frac=0.4, small_gap_max=10).sample_pairs(3000, pool, pool, set(pool), set())
    frac = lambda s: sum(1 for x in s if 1 <= abs(x["a"] - x["b"]) <= 10) / len(s)  # noqa: E731
    assert frac(plain) < 0.08
    assert 0.35 < frac(rich) < 0.5
    # Labels still follow the operator, and operands stay inside the pool.
    from data.magnitude_generator import OPS
    for s in rich:
        assert s["label"] == ("TRUE" if OPS[s["op"]](s["a"], s["b"]) else "FALSE")
        assert 0 <= s["a"] <= 499 and 0 <= s["b"] <= 499


def test_residual_optimiser_group(vocab):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from models.transformer import create_model
    from training.train import Trainer, TrainingConfig
    from utils.grid_bank import GridBank

    bank = GridBank.load(ROOT / "grids" / "grid_bank.npz")
    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_shared", tie_weights=True, grid_bank=bank,
                         verbose=False, shared_basis="ramp", shared_residual=True)
    cfg = TrainingConfig(model_size="tiny", vocab_size=vocab.vocab_size, device="cpu", warmup_steps=1,
                         learning_rate=1e-3, residual_lr_mult=10.0, residual_weight_decay=0.5)
    loader = DataLoader([{"input_ids": torch.zeros(4, dtype=torch.long), "target": torch.tensor(0),
                          "length": torch.tensor(4)}], batch_size=1)
    trainer = Trainer(model, loader, loader, cfg, vocab)
    groups = trainer.optimizer.param_groups
    assert len(groups) == 2
    assert groups[1]["params"][0] is model.symbol_embed.encoder.residual
    assert groups[1]["lr"] == pytest.approx(1e-2) and groups[1]["weight_decay"] == 0.5
    assert groups[0]["weight_decay"] == cfg.weight_decay
    n_params = sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for g in groups for p in g["params"]) == n_params
