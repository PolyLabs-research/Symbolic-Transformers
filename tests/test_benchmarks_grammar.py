"""Every scored benchmark in symbolic_benchmarks.py must agree with the generator's grammar."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from symbolic_benchmarks import BENCHMARKS, score_test, total_variation  # noqa: E402
from utils.fol_grammar import FOLGrammar  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402

CASES = [(cat, i, t) for cat, c in BENCHMARKS.items() for i, t in enumerate(c["tests"]) if not t.get("observational")]


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(str(ROOT / "unified_vocabulary.json"))


@pytest.fixture(scope="module")
def grammar(vocab):
    return FOLGrammar(vocab)


def _continuations(text, vocab, grammar):
    ids = vocab.encode_formula_simple(text)
    if ids and ids[0] == vocab.message_start:
        ids = ids[1:]
    return {"NUMERAL" if t == grammar.NUMERAL else vocab.id_to_label[t] for t in grammar.continuations(ids)}


def _label_set(labels):
    return {"NUMERAL" if lab.startswith("NUM_") else lab for lab in labels}


@pytest.mark.parametrize("cat,i,test", CASES, ids=[f"{c}-{i}" for c, i, _ in CASES])
def test_benchmark_expectation_matches_grammar(cat, i, test, vocab, grammar):
    cont = _continuations(test["input"], vocab, grammar)
    assert cont, f"{test['input']!r} is not a prefix of any formula the generator produces"
    if "expected" in test:
        assert _label_set([test["expected"]]) <= cont
        if test["expected_min"] >= 90 and not test["expected"].startswith("NUM_"):
            assert cont == {test["expected"]}, "a >= 90% expectation needs the token to be forced"
    if "expected_one_of" in test:
        assert _label_set(test["expected_one_of"]) == cont, "list exactly the grammatical continuations"
    if "expected_distribution" in test:
        assert set(test["expected_distribution"]) == cont
    if "top_max" in test:
        assert cont == {"NUMERAL"}
    for tok, key in (("RPAREN", "rparen_max"), ("MESSAGE_END", "message_end_max")):
        if key in test:
            assert tok not in cont, f"{tok} is grammatical here, so a cap on it is not a grammar fact"
    if "compare_with" in test:
        assert _continuations(test["compare_with"], vocab, grammar) == cont


def test_continuations_of_simple_prefixes(vocab, grammar):
    assert _continuations("FORALL", vocab, grammar) == {"VAR"}
    assert _continuations("FORALL VAR", vocab, grammar) == {"NUMERAL"}
    assert _continuations("PRED 0 LPAREN VAR 0 RPAREN", vocab, grammar) == {"MESSAGE_END"}
    assert _continuations("PRED 0 LPAREN VAR 0", vocab, grammar) == {"COMMA", "RPAREN"}
    starts = {"PRED", "VAR", "LPAREN", "FORALL", "EXISTS", "NOT"}
    assert {"NOT", "FORALL", "LPAREN"} <= _continuations("MESSAGE_START", vocab, grammar) == starts
    assert grammar.continuations(vocab.encode_formula_simple("RPAREN")) == set()


def test_score_test_criteria():
    flat = {"A": 50.0, "B": 50.0}
    assert score_test({"input": "x", "rationale": "", "expected": "A", "expected_min": 40}, lambda s: flat)["passed"]
    assert not score_test({"input": "x", "rationale": "", "expected": "A", "expected_min": 60}, lambda s: flat)["passed"]
    dist = {"input": "x", "rationale": "", "expected_distribution": {"A": [40, 60], "B": [40, 60]}}
    assert score_test(dist, lambda s: flat)["passed"]
    assert not score_test(dist, lambda s: {"A": 70.0, "B": 30.0})["passed"]
    one_of = {"input": "x", "rationale": "", "expected_one_of": ["A", "B"], "total_min": 80, "rparen_max": 5}
    assert score_test(one_of, lambda s: flat)["passed"]
    assert not score_test(one_of, lambda s: {"A": 50.0, "B": 20.0, "RPAREN": 30.0})["passed"]
    assert score_test({"input": "x", "rationale": "", "top_max": 5}, lambda s: {"A": 4.0})["passed"]
    cmp = {"input": "x", "compare_with": "y", "rationale": ""}
    assert score_test(cmp, lambda s: flat)["passed"]
    assert not score_test(cmp, lambda s: flat if s == "x" else {"A": 100.0})["passed"]
    assert total_variation(flat, {"A": 100.0}) == 50.0
    obs = score_test({**cmp, "observational": True}, lambda s: flat)
    assert obs["observational"] and "passed" not in obs
