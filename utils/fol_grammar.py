"""Prefix parser for the formulas ``data/dataset_generator.py`` produces.

Grammar, at token level (indices below 625 are one numeral token):

    formula := atomic | QUANT VAR n formula | NOT formula | LPAREN formula CONN formula RPAREN
    atomic  := PRED n LPAREN VAR n [COMMA VAR n] RPAREN | VAR n EQUALS VAR n
    QUANT in {FORALL, EXISTS}, CONN in {AND, OR, IMPLIES, IFF}

The grammar is LL(1), so a prefix fixes the parse state and every next token has a role that the
prefix alone determines. ``annotate`` labels each position of a formula with that role:

    forced          the grammar allows exactly one token here (MESSAGE_END included)
    formula_start   first token of a (sub)formula: the generator's choice of formula type
    connective      the binary connective, uniform over four
    arity           COMMA or RPAREN after a predicate's first argument (unary or binary, 50/50)
    pred_index      predicate index, uniform over the predicate pool
    quant_var       the variable a quantifier binds, uniform over the variable pool
    free_var        a variable with no usable bound variables in scope, uniform over the pool
    bound_var       a variable the generator draws from the bound variables in scope

For ``bound_var`` positions it also returns the generator's distribution over values, so the best
achievable top-k accuracy of any predictor is known per position.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

ROLES = ["forced", "formula_start", "connective", "arity", "pred_index", "quant_var", "free_var", "bound_var"]


class FormulaSyntaxError(ValueError):
    pass


class FOLGrammar:
    def __init__(self, vocab):
        lab = vocab.encode_label
        self.quant = {lab("FORALL"), lab("EXISTS")}
        self.conn = {lab("AND"), lab("OR"), lab("IMPLIES"), lab("IFF")}
        self.NOT, self.LP, self.RP = lab("NOT"), lab("LPAREN"), lab("RPAREN")
        self.COMMA, self.EQ = lab("COMMA"), lab("EQUALS")
        self.VAR, self.PRED = vocab.compositional_tokens["VAR"], vocab.compositional_tokens["PRED"]
        self.is_numeral = vocab.is_numeral
        self.start, self.end = vocab.message_start, vocab.message_end

    # ------------------------------------------------------------------ parsing
    def annotate(self, formula: List[int]) -> Tuple[List[str], Dict[int, Dict[int, float]]]:
        """Role of every token of ``formula`` (no boundaries) and value distributions at bound_var positions."""
        roles: List[Optional[str]] = [None] * len(formula)
        dists: Dict[int, Dict[int, float]] = {}
        end = self._formula(formula, 0, [], roles, dists)
        if end != len(formula):
            raise FormulaSyntaxError(f"trailing tokens after position {end}")
        return roles, dists

    def annotate_with_boundaries(self, formula: List[int]):
        """Roles for the targets of [MESSAGE_START] + formula + [MESSAGE_END], i.e. positions 1..len+1."""
        roles, dists = self.annotate(formula)
        return roles + ["forced"], {i + 1: d for i, d in dists.items()}

    def is_well_formed(self, tokens: List[int]) -> bool:
        """True if ``tokens`` (optionally wrapped in MESSAGE_START/END) is one complete formula."""
        if tokens and tokens[0] == self.start:
            tokens = tokens[1:]
        if tokens and tokens[-1] == self.end:
            tokens = tokens[:-1]
        try:
            self.annotate(tokens)
            return True
        except FormulaSyntaxError:
            return False

    def _tok(self, f, i):
        if i >= len(f):
            raise FormulaSyntaxError(f"formula ends at position {i}")
        return f[i]

    def _expect(self, f, i, tok, roles):
        if self._tok(f, i) != tok:
            raise FormulaSyntaxError(f"position {i}: expected {tok}, got {f[i]}")
        roles[i] = "forced"
        return i + 1

    def _numeral(self, f, i):
        if not self.is_numeral(self._tok(f, i)):
            raise FormulaSyntaxError(f"position {i}: expected an index numeral, got {f[i]}")
        return f[i]

    def _formula(self, f, i, bound, roles, dists):
        t = self._tok(f, i)
        roles[i] = "formula_start"
        if t in self.quant:
            i = self._expect(f, i + 1, self.VAR, roles)
            n = self._numeral(f, i)
            roles[i] = "quant_var"
            return self._formula(f, i + 1, bound + [n], roles, dists)
        if t == self.NOT:
            return self._formula(f, i + 1, bound, roles, dists)
        if t == self.LP:
            j = self._formula(f, i + 1, bound, roles, dists)
            if self._tok(f, j) not in self.conn:
                raise FormulaSyntaxError(f"position {j}: expected a connective, got {f[j]}")
            roles[j] = "connective"
            k = self._formula(f, j + 1, bound, roles, dists)
            return self._expect(f, k, self.RP, roles)
        if t == self.PRED:
            self._numeral(f, i + 1)
            roles[i + 1] = "pred_index"
            j = self._expect(f, i + 2, self.LP, roles)
            j = self._expect(f, j, self.VAR, roles)
            self._arg(f, j, bound, roles, dists)
            t2 = self._tok(f, j + 1)
            roles[j + 1] = "arity"
            if t2 == self.RP:
                return j + 2
            if t2 != self.COMMA:
                raise FormulaSyntaxError(f"position {j + 1}: expected COMMA or RPAREN, got {t2}")
            j = self._expect(f, j + 2, self.VAR, roles)
            self._arg(f, j, bound, roles, dists)
            return self._expect(f, j + 1, self.RP, roles)
        if t == self.VAR:
            # Equality: two distinct bound positions when at least two are in scope, else uniform draws.
            n1 = self._numeral(f, i + 1)
            if len(bound) >= 2:
                roles[i + 1] = "bound_var"
                dists[i + 1] = _distribution(bound)
            else:
                roles[i + 1] = "free_var"
            j = self._expect(f, i + 2, self.EQ, roles)
            j = self._expect(f, j, self.VAR, roles)
            self._numeral(f, j)
            if len(bound) >= 2:
                rest = list(bound)
                if n1 in rest:
                    rest.remove(n1)
                roles[j] = "bound_var"
                dists[j] = _distribution(rest)
            else:
                roles[j] = "free_var"
            return j + 1
        raise FormulaSyntaxError(f"position {i}: token {t} cannot start a formula")

    def _arg(self, f, i, bound, roles, dists):
        self._numeral(f, i)
        if bound:
            roles[i] = "bound_var"
            dists[i] = _distribution(bound)
        else:
            roles[i] = "free_var"


def _distribution(values: List[int]) -> Dict[int, float]:
    c = Counter(values)
    return {v: n / len(values) for v, n in c.items()}


def ceiling(role: str, dist: Optional[Dict[int, float]], pool_size: Dict[str, int], k: int) -> Optional[float]:
    """Best achievable top-k accuracy at one position, or None where the grammar alone does not fix it."""
    if role == "forced":
        return 1.0
    if role == "connective":
        return min(1.0, k / 4)
    if role == "arity":
        return min(1.0, k / 2)
    if role == "pred_index":
        return min(1.0, k / pool_size["pred"])
    if role in ("quant_var", "free_var"):
        return min(1.0, k / pool_size["var"])
    if role == "bound_var":
        return sum(sorted(dist.values(), reverse=True)[:k])
    return None  # formula_start: depends on the generator's depth and complexity mixture
