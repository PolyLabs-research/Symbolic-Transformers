"""Tests for the associative-recall layouts in scripts/sanity_recall.py."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sanity_recall import LAYOUTS, batch  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

# layout: (tokens from a key to its value in the facts, tokens from the query key to the answer)
GAPS = {"pairs": (1, 1), "pairs_offset2": (2, 2), "pairs_gap_context": (2, 1), "pairs_gap_query": (1, 2),
        "table": (2, 3), "table_nodomain": (2, 3), "table_keylast": (1, 1)}


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(ROOT / "unified_vocabulary.json"))


def test_every_layout_has_its_gaps_stated():
    assert set(GAPS) == set(LAYOUTS)


@pytest.mark.parametrize("values", ["binary", "numerals"])
@pytest.mark.parametrize("layout", sorted(GAPS))
def test_each_answer_is_the_value_of_the_queried_key(vocab, layout, values):
    fact_gap, query_gap = GAPS[layout]
    if values == "binary":
        is_value = lambda tok: tok in (vocab.encode_label("TRUE"), vocab.encode_label("FALSE"))
    else:
        is_value = lambda tok: 100 <= tok < 200
    queries_start = {vocab.encode_label("ANSWER"), vocab.encode_label("MODELS")}
    torch.manual_seed(0)
    seq, targets = batch(vocab, 8, 4, values, 3, "cpu", layout)
    for row in seq.tolist():
        end = next(i for i, tok in enumerate(row) if tok in queries_start)
        facts = {row[j]: row[j + fact_gap] for j in range(1, end - fact_gap) if is_value(row[j + fact_gap])
                 and not is_value(row[j])}
        assert len(facts) == 8
        for t in targets:
            assert row[t] == facts[row[t - query_gap]]


@pytest.mark.parametrize("layout", ["table", "table_keylast", "pairs"])
def test_fused_batches_unfuse_to_the_split_batches(vocab, layout):
    from sanity_recall import fuse_batch
    from utils.fused import SymbolFuser
    fuser = SymbolFuser(vocab)
    torch.manual_seed(1)
    seq, targets = batch(vocab, 8, 4, "binary", 3, "cpu", layout)
    fused, ftargets = fuse_batch(fuser, seq, targets)
    for row, frow in zip(seq.tolist(), fused.tolist()):
        assert fuser.unfuse(frow) == row
        assert [frow[t] for t in ftargets] == [row[t] for t in targets]
    assert fused.size(1) == seq.size(1) - (0 if layout == "pairs" else sum(t in fuser.cat_index for t in seq[0].tolist()))
