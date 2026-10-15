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


def test_rewarm_restarts_the_schedule_at_the_anneal_end():
    from training.train import anneal_end_epoch, make_lr_lambda

    assert anneal_end_epoch(80, 0.7) == 56 and anneal_end_epoch(120, 0.4667) == 57 and anneal_end_epoch(10, 0.7) == 7
    steps_per_epoch, n_epochs = 313, 80
    total = steps_per_epoch * n_epochs
    restart = (anneal_end_epoch(n_epochs, 0.7) - 1) * steps_per_epoch
    plain = make_lr_lambda(total, 100)
    rewarm = make_lr_lambda(total, 100, restart, 100)
    # Identical before the restart, including the initial warm-up and the cosine.
    for s in (0, 50, 99, 100, 5000, restart - 1):
        assert rewarm(s) == plain(s)
    assert plain(0) == pytest.approx(0.01) and plain(99) == 1.0
    assert plain(restart) < 0.3                       # deep in the cosine tail by epoch 56
    # At the restart the rate drops to the start of a fresh warm-up and climbs back to the base rate.
    assert rewarm(restart) == pytest.approx(0.01)
    assert rewarm(restart + 99) == 1.0
    assert rewarm(restart + 101) < 1.0
    # ... then decays by cosine to the 10% floor by the last step.
    assert rewarm(total - 1) == pytest.approx(0.1, abs=1e-3)
    mid = restart + 100 + (total - restart - 100) // 2
    assert 0.4 < rewarm(mid) < 0.6
    # Without a restart step the schedule is the plain one.
    assert make_lr_lambda(total, 100, None, 100)(restart) == plain(restart)


def test_staged_schedule_and_parser():
    from training.train import parse_anneal_stages, staged_residual_scale_at, TrainingConfig, residual_schedule_scale, residual_schedule_end

    stages = parse_anneal_stages("0.2:1,0.4667:0.01,0.5667:0.01,0.9:0.000001,1:0.000001")
    assert stages[0] == (0.2, 1.0) and stages[-1] == (1.0, 1e-6)
    f = lambda e: staged_residual_scale_at(e, 120, stages)
    assert f(1) == 1.0 and f(24) == 1.0
    assert f(40) == pytest.approx(10 ** -1.0, rel=1e-3)            # halfway (in log) between 1 and 0.01
    assert f(56) == pytest.approx(0.01, rel=1e-3) and f(60) == pytest.approx(0.01) and f(68) == pytest.approx(0.01)
    assert f(88) == pytest.approx(10 ** -4.0, rel=1e-3)            # halfway between 1e-2 and 1e-6
    assert f(108) == pytest.approx(1e-6, rel=1e-3) and f(120) == pytest.approx(1e-6)
    vals = [f(e) for e in range(1, 121)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    # A zero endpoint is a step at its breakpoint.
    step = parse_anneal_stages("0.5:0.01,0.75:0")
    assert staged_residual_scale_at(59, 80, step) == 0.01 and staged_residual_scale_at(60, 80, step) == 0.0
    # The trainer-facing helpers route to the staged form and know where it ends.
    cfg = TrainingConfig(num_epochs=120, residual_anneal_stages=stages)
    assert residual_schedule_end(cfg) == 120 and residual_schedule_scale(cfg, 56) == pytest.approx(0.01, rel=1e-3)
    # The deep arm ends its breakpoints at the start of the hold: the scale holds at 1e-6 from epoch 108
    # to the end and best_model.pt is chosen across that hold.
    sys.path.insert(0, str(ROOT / "scripts"))
    from grid_experiment import ARMS
    arm = ARMS["grid_ramp_res_orth_anneal_deep"]
    deep = parse_anneal_stages(arm[arm.index("--residual-anneal-stages") + 1])
    cfg_deep = TrainingConfig(num_epochs=120, residual_anneal_stages=deep)
    assert residual_schedule_end(cfg_deep) == 108
    assert residual_schedule_scale(cfg_deep, 108) == pytest.approx(1e-6, rel=1e-3)
    assert residual_schedule_scale(cfg_deep, 120) == pytest.approx(1e-6)
    cfg2 = TrainingConfig(num_epochs=80, residual_anneal=(0.3, 0.7), residual_anneal_floor=0.01, residual_anneal_hold=True)
    assert residual_schedule_end(cfg2) == 56 and residual_schedule_scale(cfg2, 80) == 0.01
    assert residual_schedule_end(TrainingConfig()) is None
    for bad in ("0.5:1", "0.2:1,0.1:0.5", "0.2:2,0.5:0", "0.2:1,1.5:0"):
        with pytest.raises(ValueError):
            parse_anneal_stages(bad)
