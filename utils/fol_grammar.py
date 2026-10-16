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

For ``bound_var`` positions it also returns the generator's distribution over values, and
``formula_start_distributions`` gives the Bayes-optimal distribution at every formula start, so the best
achievable top-k accuracy and loss of any predictor are known at every position.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

ROLES = ["forced", "formula_start", "connective", "arity", "pred_index", "quant_var", "free_var", "bound_var"]

# The dataset's default complexity mixture (generate_training_data): FOLFormulaGenerator.generate_batch
# sets the formula's maximum depth to its complexity level.
COMPLEXITY_PRIOR = {1: 0.2, 2: 0.4, 3: 0.3, 4: 0.1}


class FormulaSyntaxError(ValueError):
    pass


class FOLGrammar:
    def __init__(self, vocab):
        lab = vocab.encode_label
        self.quant = {lab("FORALL"), lab("EXISTS")}
        self.conn = {lab("AND"), lab("OR"), lab("IMPLIES"), lab("IFF")}
        self.NOT, self.LP, self.RP = lab("NOT"), lab("LPAREN"), lab("RPAREN")
        self.FORALL, self.EXISTS = lab("FORALL"), lab("EXISTS")
        self.COMMA, self.EQ = lab("COMMA"), lab("EQUALS")
        self.VAR, self.PRED = vocab.compositional_tokens["VAR"], vocab.compositional_tokens["PRED"]
        self.is_numeral = vocab.is_numeral
        self.start, self.end = vocab.message_start, vocab.message_end

    # ------------------------------------------------------------------ parsing
    def annotate(self, formula: List[int]) -> Tuple[List[str], Dict[int, Dict[int, float]]]:
        """Role of every token of ``formula`` (no boundaries) and value distributions at bound_var positions."""
        roles, dists, _ = self._parse(formula)
        return roles, dists

    def _parse(self, formula):
        roles: List[Optional[str]] = [None] * len(formula)
        dists: Dict[int, Dict[int, float]] = {}
        depths: Dict[int, int] = {}
        end = self._formula(formula, 0, [], roles, dists, 0, depths)
        if end != len(formula):
            raise FormulaSyntaxError(f"trailing tokens after position {end}")
        return roles, dists, depths

    def _type_probs(self, depth: int, max_depth: int) -> Dict[int, float]:
        """First-token distribution of a (sub)formula, mirroring FOLFormulaGenerator.generate_formula.

        Below the maximum depth a formula is atomic with p 0.3, else a quantifier or a connective with
        p 1/2 each (FORALL / EXISTS 1/2 each; NOT is 1 of the 5 connectives, the other 4 are binary and
        start with LPAREN). At the maximum depth it is atomic. Atomic formulas are equalities (start with
        VAR) with p 0.3, else predicates.
        """
        atomic = 1.0 if depth >= max_depth else 0.3
        rest = 1.0 - atomic
        probs = {self.PRED: 0.7 * atomic, self.VAR: 0.3 * atomic}
        if rest:
            probs.update({self.FORALL: rest / 4, self.EXISTS: rest / 4,
                          self.NOT: rest / 2 * 0.2, self.LP: rest / 2 * 0.8})
        return probs

    def formula_start_distributions(self, formula: List[int],
                                    prior: Dict[int, float] = COMPLEXITY_PRIOR) -> Dict[int, Dict[int, float]]:
        """Bayes-optimal next-token distribution at every formula_start position of ``formula``.

        The formula's maximum depth (its complexity level) is hidden, so the prediction averages the
        generator's first-token distribution over levels, weighted by how well each level explains the
        formula types already seen in the prefix. Index values carry no information about structure.
        """
        _, _, depths = self._parse(formula)
        weights = dict(prior)
        out = {}
        for i in sorted(depths):
            total = sum(weights.values())
            post = {c: w / total for c, w in weights.items()}
            dist = defaultdict(float)
            for c, w in post.items():
                for tok, p in self._type_probs(depths[i], c).items():
                    dist[tok] += w * p
            out[i] = dict(dist)
            weights = {c: w * self._type_probs(depths[i], c).get(formula[i], 0.0) for c, w in weights.items()}
        return out

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

    # Stands for "any index numeral" in continuations().
    NUMERAL = -1

    def continuations(self, prefix: List[int]) -> set:
        """Tokens that can follow ``prefix`` (no MESSAGE_START) in some well-formed formula.

        Any index numeral is reported as ``FOLGrammar.NUMERAL``; MESSAGE_END is included when the prefix
        is already a complete formula.
        """
        out = set()
        if self._viability(list(prefix)) == "complete":
            out.add(self.end)
        candidates = sorted(self.quant | self.conn | {self.NOT, self.LP, self.RP, self.COMMA, self.EQ,
                                                      self.VAR, self.PRED})
        for tok in candidates:
            if self._viability(list(prefix) + [tok]) is not None:
                out.add(tok)
        if self._viability(list(prefix) + [0]) is not None:
            out.add(self.NUMERAL)
        return out

    def _viability(self, tokens):
        """'complete', 'open' (a proper prefix of some formula) or None (no formula starts this way)."""
        try:
            self.annotate(tokens)
            return "complete"
        except FormulaSyntaxError as e:
            return "open" if str(e).startswith("formula ends") else None

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

    def _formula(self, f, i, bound, roles, dists, depth, depths):
        t = self._tok(f, i)
        roles[i] = "formula_start"
        depths[i] = depth
        if t in self.quant:
            i = self._expect(f, i + 1, self.VAR, roles)
            n = self._numeral(f, i)
            roles[i] = "quant_var"
            return self._formula(f, i + 1, bound + [n], roles, dists, depth + 1, depths)
        if t == self.NOT:
            return self._formula(f, i + 1, bound, roles, dists, depth + 1, depths)
        if t == self.LP:
            j = self._formula(f, i + 1, bound, roles, dists, depth + 1, depths)
            if self._tok(f, j) not in self.conn:
                raise FormulaSyntaxError(f"position {j}: expected a connective, got {f[j]}")
            roles[j] = "connective"
            k = self._formula(f, j + 1, bound, roles, dists, depth + 1, depths)
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
    if role in ("bound_var", "formula_start") and dist is not None:
        return sum(sorted(dist.values(), reverse=True)[:k])
    return None  # formula_start without its distribution


def optimal_loss(role: str, dist: Optional[Dict[int, float]], pool_size: Dict[str, int]) -> Optional[float]:
    """Expected cross-entropy (nats) of the Bayes-optimal predictor at one position."""
    if role == "forced":
        return 0.0
    if role == "connective":
        return math.log(4)
    if role == "arity":
        return math.log(2)
    if role == "pred_index":
        return math.log(pool_size["pred"])
    if role in ("quant_var", "free_var"):
        return math.log(pool_size["var"])
    if dist is not None:
        return -sum(p * math.log(p) for p in dist.values() if p > 0)
    return None
