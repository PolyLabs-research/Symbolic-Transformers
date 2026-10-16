"""The xVal-style numeral embedding and the sum-comparison task (numeral-code head-to-head)."""
import json
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


def test_xval_rows_are_scaled_copies_of_one_direction(tmp_path, vocab):
    torch = pytest.importorskip("torch")
    from models.transformer import create_model, load_model_from_checkpoint

    torch.manual_seed(0)
    model = create_model(vocab.vocab_size, "tiny", embedding_type="xval", tie_weights=True, verbose=False)
    emb = model.symbol_embed
    lo, hi = vocab.numeral_range
    with torch.no_grad():
        table = model.embedding_table()
    h = emb.direction
    for k in (0, 17, 249, 250, 499, 624):
        s = (k - 249.5) / 144.3
        assert torch.allclose(table[k], s * h, atol=1e-6)
    assert float(emb.values[624]) > 2.5 and float(emb.values[0]) < -1.7          # far band extrapolates
    lt = vocab.encode_label("LESS_THAN")
    assert torch.equal(table[lt], emb.fallback.weight[lt]) and not emb.is_numeral[lt]
    # One direction shared by every numeral: the numeral block has rank one.
    assert torch.linalg.matrix_rank(table[lo:hi + 1]) == 1
    # Gradients reach the direction through numeral tokens.
    ids = torch.tensor([[vocab.message_start, 17, lt, 420, vocab.answer_token]])
    model.train()
    model(ids)[0, -1, vocab.encode_label("TRUE")].backward()
    assert emb.direction.grad is not None and emb.direction.grad.abs().sum() > 0
    # Checkpoint round trip, including a non-default standardisation.
    m2 = create_model(vocab.vocab_size, "tiny", embedding_type="xval", tie_weights=True, verbose=False,
                      xval_center=100.0, xval_scale=50.0)
    torch.save({"model_state_dict": m2.state_dict(),
                "config": {"model_size": "tiny", "embedding_type": "xval", "tie_weights": True,
                           "xval_center": 100.0, "xval_scale": 50.0}}, tmp_path / "x.pt")
    loaded, _ = load_model_from_checkpoint(tmp_path / "x.pt", vocab.vocab_size, "cpu", verbose=False)
    loaded.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(loaded.embedding_table(), m2.embedding_table())
        assert torch.allclose(loaded(ids), m2(ids))
    assert float(loaded.symbol_embed.values[150]) == pytest.approx(1.0)


def _gen(tmp_path, task, **kw):
    from data.magnitude_generator import generate_magnitude_data
    out = tmp_path / task
    meta = generate_magnitude_data(str(VOCAB_PATH), str(out), n_train=kw.get("n_train", 3000), n_val=300,
                                   n_test=kw.get("n_test", 1200), value_range=(0, 499), holdout=(350, 449),
                                   far_range=(500, 624), small_gap_frac=0.4, small_gap_max=10, seed=7, task=task)
    load = lambda name: json.load(open(out / f"{name}.json"))["samples"]
    return out, meta, load


def test_sum_task_labels_pools_and_unseen_counts(tmp_path, vocab):
    from data.magnitude_generator import OPS
    out, meta, load = _gen(tmp_path, "sum")
    assert meta["task"] == "sum_comparison_v1" and meta["n_operands"] == 4
    plus = vocab.encode_label("PLUS")
    train_pool = set(range(0, 350)) | set(range(450, 500))
    hold, far = set(range(350, 450)), set(range(500, 625))
    for name in ("train", "val", "test", "test_holdout", "test_far"):
        samples = load(name)
        assert json.load(open(out / f"{name}.json"))["max_seq_len"] == 9
        keys = set()
        for s in samples:
            a, b, c, d = s["operands"]
            ctx = s["context"]
            assert ctx == [vocab.message_start, a, plus, b, vocab.encode_label(s["op"]), c, plus, d, vocab.answer_token]
            assert (s["lhs"], s["rhs"]) == (a + b, c + d)
            truth = OPS[s["op"]](a + b, c + d)
            assert s["label"] == ("TRUE" if truth else "FALSE")
            assert s["target"] == vocab.encode_label(s["label"])
            assert s["n_unseen"] == sum(x not in train_pool for x in (a, b, c, d))
            keys.add((a, b, c, d, s["op"]))
            ops = {a, b, c, d}
            if name in ("train", "val", "test"):
                assert ops <= train_pool
            elif name == "test_holdout":
                assert ops <= train_pool | hold and ops & hold
            else:
                assert ops <= train_pool | far and ops & far
        assert len(keys) == len(samples)                                    # distinct within a split
        frac_true = sum(s["label"] == "TRUE" for s in samples) / len(samples)
        assert 0.4 < frac_true < 0.6
        small = sum(1 <= abs(s["lhs"] - s["rhs"]) <= 10 for s in samples) / len(samples)
        assert small > 0.25                                                 # close sums are oversampled
        if name in ("test_holdout", "test_far"):
            all_unseen = sum(s["n_unseen"] == 4 for s in samples) / len(samples)
            assert 0.4 < all_unseen < 0.65
    tr = {(*s["operands"], s["op"]) for s in load("train")}
    assert not tr & {(*s["operands"], s["op"]) for s in load("test")}       # test disjoint from train


def test_compare_task_regenerates_the_existing_dataset_exactly(tmp_path):
    ref = ROOT / "runs" / "magnitude_precision" / "data"
    if not (ref / "test_holdout.json").exists():
        pytest.skip("reference dataset not present")
    from data.magnitude_generator import generate_magnitude_data
    out = tmp_path / "compare"
    generate_magnitude_data(str(VOCAB_PATH), str(out), n_train=20000, n_val=2000, n_test=4000, value_range=(0, 499),
                            holdout=(350, 449), far_range=(500, 624), small_gap_frac=0.4, small_gap_max=10, seed=42,
                            task="compare")
    for name in ("train", "val", "test", "test_holdout", "test_far"):
        assert json.load(open(out / f"{name}.json")) == json.load(open(ref / f"{name}.json")), name


def test_evaluator_breakdowns_for_both_tasks(tmp_path, vocab):
    torch = pytest.importorskip("torch")
    from evaluate_magnitude import evaluate_split
    from models.transformer import create_model
    from training.train import FOLDataset

    torch.manual_seed(0)
    model = create_model(vocab.vocab_size, "tiny", embedding_type="xval", tie_weights=True, verbose=False)
    model.eval()
    device = torch.device("cpu")
    out, _, _ = _gen(tmp_path, "sum", n_train=300, n_test=300)
    with torch.no_grad():
        r_sum, rows = evaluate_split(model, FOLDataset(str(out / "test_holdout.json"), pad_id=vocab.pad_token or 0),
                                     vocab, device, return_rows=True)
    assert set(r_sum["by_unseen"]) == {"0", "1", "2", "3", "4"} and "by_gap_unseen2" not in r_sum
    assert r_sum["by_gap_all_unseen"]["101+"]["n"] + sum(v["n"] for k, v in r_sum["by_gap_all_unseen"].items() if k != "101+") \
        == r_sum["by_unseen"]["4"]["n"]
    assert all(r["n_operands"] == 4 and len(r["operands"]) == 4 and r["gap"] == abs(r["lhs"] - r["rhs"])
               and r["lhs"] == sum(r["operands"][:2]) and "a" not in r for r in rows)
    out_c, _, _ = _gen(tmp_path, "compare", n_train=300, n_test=300)
    with torch.no_grad():
        r_cmp = evaluate_split(model, FOLDataset(str(out_c / "test_holdout.json"), pad_id=vocab.pad_token or 0), vocab, device)
    assert set(r_cmp["by_unseen"]) == {"0", "1", "2"} and r_cmp["by_gap_unseen2"] == r_cmp["by_gap_all_unseen"]
