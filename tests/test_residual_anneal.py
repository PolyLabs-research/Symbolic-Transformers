"""The grid_shared residual scale and the trainer's --residual-anneal schedule."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.vocabulary import Vocabulary  # noqa: E402

VOCAB_PATH = ROOT / "unified_vocabulary.json"


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(VOCAB_PATH))


def test_residual_scale_multiplies_the_residual_everywhere():
    torch = pytest.importorskip("torch")
    from models.transformer import SharedCellEncoder
    from utils.grid_bank import GRID_CELLS

    torch.manual_seed(0)
    enc = SharedCellEncoder(8, basis="ramp", residual=True)
    with torch.no_grad():
        enc.residual.normal_()
    generated = enc.phi @ enc.proj.weight.t()
    assert enc.effective_residual() is enc.residual              # scale 1: untouched
    enc.set_residual_scale(0.5)
    assert torch.allclose(enc.cell_weights(), generated + 0.5 * enc.residual)
    enc.set_residual_scale(0.0)
    assert torch.allclose(enc.cell_weights(), generated)
    # The forward pass (training or not) sees the scaled residual too.
    bits = torch.zeros(2, GRID_CELLS)
    bits[1, :10] = 1.0
    plain = SharedCellEncoder(8, basis="ramp", residual=False)
    plain.load_state_dict({"phi": enc.phi, "proj.weight": enc.proj.weight, "proj.bias": enc.proj.bias})
    enc.train()
    assert torch.allclose(enc(bits), plain(bits))
    enc.eval()
    assert torch.allclose(enc(bits), plain(bits))
    with pytest.raises(ValueError):
        enc.set_residual_scale(1.5)
    with pytest.raises(ValueError):
        plain.set_residual_scale(0.5)
    # Orthogonal projection and scale compose.
    orth = SharedCellEncoder(8, basis="ramp", residual=True, residual_orthogonal=True)
    with torch.no_grad():
        orth.residual.normal_()
    full = orth.effective_residual().clone()
    orth.set_residual_scale(0.25)
    assert torch.allclose(orth.effective_residual(), 0.25 * full)


def test_residual_scale_survives_checkpoints_and_defaults_to_one():
    torch = pytest.importorskip("torch")
    from models.transformer import SharedCellEncoder

    enc = SharedCellEncoder(8, basis="ramp", residual=True)
    enc.set_residual_scale(0.25)
    state = enc.state_dict()
    assert "residual_scale" in state
    fresh = SharedCellEncoder(8, basis="ramp", residual=True)
    fresh.load_state_dict(state)
    assert float(fresh.residual_scale) == 0.25
    # A checkpoint written before the scale existed loads strictly, at full residual.
    del state["residual_scale"]
    old = SharedCellEncoder(8, basis="ramp", residual=True)
    old.load_state_dict(state, strict=True)
    assert float(old.residual_scale) == 1.0
    # Residual-free encoders carry no scale.
    assert "residual_scale" not in SharedCellEncoder(8, basis="ramp", residual=False).state_dict()


def test_residual_scale_schedule():
    from training.train import residual_scale_at

    assert residual_scale_at(1, 80, 0.3, 0.7) == 1.0
    assert residual_scale_at(24, 80, 0.3, 0.7) == 1.0
    assert residual_scale_at(25, 80, 0.3, 0.7) == pytest.approx(31 / 32)
    assert residual_scale_at(40, 80, 0.3, 0.7) == pytest.approx(0.5)
    assert residual_scale_at(56, 80, 0.3, 0.7) == 0.0
    assert residual_scale_at(80, 80, 0.3, 0.7) == 0.0
    # Degenerate (start == end) is a step.
    assert residual_scale_at(39, 80, 0.5, 0.5) == 1.0
    assert residual_scale_at(40, 80, 0.5, 0.5) == 0.0
    # Never leaves [0, 1].
    assert all(0.0 <= residual_scale_at(e, 10, 0.0, 1.0) <= 1.0 for e in range(1, 11))
    # Geometric: log-linear from 1 to the floor, then 0.
    assert residual_scale_at(24, 80, 0.3, 0.7, floor=1e-3) == 1.0
    assert residual_scale_at(40, 80, 0.3, 0.7, floor=1e-3) == pytest.approx(10 ** -1.5)
    assert residual_scale_at(55, 80, 0.3, 0.7, floor=1e-3) == pytest.approx(10 ** (-3 * 31 / 32))
    assert residual_scale_at(56, 80, 0.3, 0.7, floor=1e-3) == 0.0
    geo = [residual_scale_at(e, 80, 0.3, 0.7, floor=1e-3) for e in range(25, 56)]
    assert all(a > b for a, b in zip(geo, geo[1:])) and geo[-1] > 1e-3
    # Hold: the floor is kept after the window instead of 0.
    assert residual_scale_at(56, 80, 0.3, 0.7, floor=0.01, hold=True) == 0.01
    assert residual_scale_at(80, 80, 0.3, 0.7, floor=0.01, hold=True) == 0.01
    assert residual_scale_at(40, 80, 0.3, 0.7, floor=0.01, hold=True) == pytest.approx(0.1)
    assert residual_scale_at(80, 80, 0.3, 0.7, floor=None, hold=True) == 0.0   # no floor: nothing to hold


def test_trainer_anneals_and_picks_a_residual_free_best(tmp_path, vocab):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from models.transformer import create_model, load_model_from_checkpoint
    from training.train import Trainer, TrainingConfig, residual_scale_at
    from utils.grid_bank import GridBank

    torch.manual_seed(0)
    bank = GridBank.load(ROOT / "grids" / "grid_bank.npz")
    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_shared", tie_weights=True, grid_bank=bank,
                         verbose=False, shared_basis="ramp", shared_residual=True, shared_residual_orthogonal=True)
    n_epochs = 10
    cfg = TrainingConfig(model_size="tiny", vocab_size=vocab.vocab_size, device="cpu", warmup_steps=1,
                         learning_rate=1e-3, num_epochs=n_epochs, residual_anneal=(0.3, 0.7),
                         checkpoint_dir=str(tmp_path), save_every=100, log_every=1000,
                         embedding_type="grid_shared", tie_weights=True, shared_basis="ramp",
                         shared_residual=True, shared_residual_orthogonal=True)
    lt = vocab.encode_label("LESS_THAN")
    samples = [{"input_ids": torch.tensor([vocab.message_start, a, lt, b, vocab.answer_token]),
                "target": torch.tensor(vocab.encode_label("TRUE") if a < b else vocab.encode_label("FALSE")),
                "length": torch.tensor(5)} for a, b in [(3, 40), (50, 7), (12, 200), (300, 5)]]
    loader = DataLoader(samples, batch_size=2)
    trainer = Trainer(model, loader, loader, cfg, vocab)
    trainer.train()

    scales = trainer.history["residual_scale"]
    assert scales == [residual_scale_at(e, n_epochs, 0.3, 0.7) for e in range(1, n_epochs + 1)]
    assert scales[0] == 1.0 and scales[-1] == 0.0 and 0.0 < scales[4] < 1.0
    # The model ends residual-free, and best_model.pt comes from that phase.
    assert float(model.symbol_embed.encoder.residual_scale) == 0.0
    best = torch.load(tmp_path / "best_model.pt", map_location="cpu")
    assert residual_scale_at(best["epoch"], n_epochs, 0.3, 0.7) == 0.0
    assert float(best["model_state_dict"]["symbol_embed.encoder.residual_scale"]) == 0.0
    assert best["config"]["residual_anneal"] == [0.3, 0.7] or tuple(best["config"]["residual_anneal"]) == (0.3, 0.7)
    loaded, _ = load_model_from_checkpoint(tmp_path / "best_model.pt", vocab.vocab_size, "cpu", verbose=False)
    assert float(loaded.symbol_embed.encoder.residual_scale) == 0.0
    # ... so its numeral rows are the generated weights alone, whatever the residual holds.
    enc = loaded.symbol_embed.encoder
    assert torch.count_nonzero(enc.residual) > 0
    assert torch.allclose(enc.cell_weights(), enc.phi @ enc.proj.weight.t())


def test_anneal_requires_a_residual(vocab):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from models.transformer import create_model
    from training.train import Trainer, TrainingConfig
    from utils.grid_bank import GridBank

    bank = GridBank.load(ROOT / "grids" / "grid_bank.npz")
    model = create_model(vocab.vocab_size, "tiny", embedding_type="grid_shared", tie_weights=True, grid_bank=bank,
                         verbose=False, shared_basis="ramp", shared_residual=False)
    cfg = TrainingConfig(model_size="tiny", vocab_size=vocab.vocab_size, device="cpu", warmup_steps=1,
                         num_epochs=1, residual_anneal=(0.0, 0.5))
    loader = DataLoader([{"input_ids": torch.zeros(4, dtype=torch.long), "target": torch.tensor(0),
                          "length": torch.tensor(4)}], batch_size=1)
    with pytest.raises(ValueError, match="residual"):
        Trainer(model, loader, loader, cfg, vocab).train()
