"""Tests for the FOL grammar roles (utils/fol_grammar.py) and the assessment helpers (scripts/assess_fol.py)."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.dataset_generator import FOLFormulaGenerator, FormulaConfig, NextSymbolDataset  # noqa: E402
from utils.fol_grammar import FOLGrammar, ceiling  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(ROOT / "unified_vocabulary.json"))


@pytest.fixture(scope="module")
def grammar(vocab):
    return FOLGrammar(vocab)


def _formulas(vocab, n, pool, seed):
    random.seed(seed)
    gen = FOLFormulaGenerator(vocab, FormulaConfig(max_depth=4, variable_pool=pool, predicate_pool=pool))
    return [f for c in (1, 2, 3, 4) for f in gen.generate_batch(n, c)]


def test_every_generated_formula_parses_and_bound_draws_are_in_scope(vocab, grammar):
    for pool, seed in ((list(range(100)), 0), ([0, 1, 2], 1)):
        for f in _formulas(vocab, 100, pool, seed):
            roles, dists = grammar.annotate(f)
            assert None not in roles
            for i, d in dists.items():
                assert roles[i] == "bound_var" and d.get(f[i], 0) > 0
                assert abs(sum(d.values()) - 1) < 1e-9


def test_roles_of_a_hand_built_formula(vocab, grammar):
    lab, var, pred = vocab.encode_label, vocab.compositional_tokens["VAR"], vocab.compositional_tokens["PRED"]
    # FORALL x3 FORALL x5 ( P1(x3) AND x3 = x5 )
    f = [lab("FORALL"), var, 3, lab("FORALL"), var, 5, lab("LPAREN"),
         pred, 1, lab("LPAREN"), var, 3, lab("RPAREN"), lab("AND"),
         var, 3, lab("EQUALS"), var, 5, lab("RPAREN")]
    roles, dists = grammar.annotate(f)
    assert roles == ["formula_start", "forced", "quant_var", "formula_start", "forced", "quant_var", "formula_start",
                     "formula_start", "pred_index", "forced", "forced", "bound_var", "arity", "connective",
                     "formula_start", "bound_var", "forced", "forced", "bound_var", "forced"]
    assert dists[11] == {3: 0.5, 5: 0.5}
    assert dists[15] == {3: 0.5, 5: 0.5}
    assert dists[18] == {5: 1.0}, "the second side of an equality is a different bound position"
    pool = {"var": 100, "pred": 100}
    assert ceiling("bound_var", dists[18], pool, 1) == 1.0
    assert ceiling("bound_var", dists[11], pool, 1) == 0.5
    assert ceiling("free_var", None, pool, 5) == 0.05
    assert ceiling("formula_start", None, pool, 1) is None
    wrapped, wdists = grammar.annotate_with_boundaries(f)
    assert wrapped[-1] == "forced" and len(wrapped) == len(f) + 1 and 12 in wdists


def test_well_formedness_rejects_broken_formulas(vocab, grammar):
    f = _formulas(vocab, 5, list(range(100)), 2)[-1]
    assert grammar.is_well_formed(f)
    assert grammar.is_well_formed([vocab.message_start] + f + [vocab.message_end])
    assert not grammar.is_well_formed(f[:-1])
    assert not grammar.is_well_formed(f + f)
    assert not grammar.is_well_formed([vocab.encode_label("RPAREN")] + f)


def test_sample_roles_align_with_the_dataset(vocab, grammar):
    from assess_fol import annotate_samples

    formulas = _formulas(vocab, 10, list(range(100)), 3)
    ds = NextSymbolDataset(formulas, vocab, max_seq_len=128, use_boundaries=True)
    roles, dists, owner = annotate_samples(ds.samples, grammar, vocab)
    assert len(roles) == len(ds.samples) == sum(len(f) + 1 for f in formulas)
    for s, r, d, f in zip(ds.samples, roles, dists, owner):
        if s["target"] == vocab.message_end:
            assert r == "forced"
        if r == "bound_var":
            assert d[s["target"]] > 0
        assert list(s["context"][1:]) == list(f[:len(s["context"]) - 1])


def test_suffix_lookup_answers_from_training_and_backs_off(vocab):
    from assess_fol import SuffixCounts

    train = [{"context": [1, 2, 3], "target": 7}, {"context": [1, 2, 3], "target": 7},
             {"context": [1, 2, 3], "target": 8}, {"context": [9, 3], "target": 5}]
    counts = SuffixCounts(train)
    assert counts.ranked([1, 2, 3])[:2] == [7, 8]
    assert counts.ranked([4, 2, 3])[:2] == [7, 8], "backs off to the longest seen suffix"
    assert counts.ranked([4, 4, 3])[0] == 7, "suffix (3,) was followed by 7 twice, 8 and 5 once"
    assert set(counts.ranked([42])) == {7, 8, 5}, "unseen token falls back to unigram counts"
    bigram = SuffixCounts(train, max_order=1)
    assert bigram.ranked([1, 2, 3])[0] == 7
