"""
Formula evaluation, Stage A (propositional): is this ground formula true in this finite world?

An item is a world (a set of objects and the atoms true about them) and a formula built from ground
atoms with NOT and the binary connectives; the label is its truth value (closed world: unlisted atoms
are false). Every item is written in two formats over the same tokens:

    direct:   MESSAGE_START <world> MODELS <formula> ANSWER <TRUE|FALSE>
    scratch:  MESSAGE_START <world> MODELS <formula> QUERY <step 0> SEP <step 1> ... SEP <step h>
              ANSWER <TRUE|FALSE>

    world:    CONST a CONST b ... SEP <true atoms, in random order>
    atoms:    PRED p ( CONST a )   with p in 0-3 (unary)   |   PRED 4 ( CONST a , CONST b )   (binary)
    formula:  atom | NOT formula | ( formula CONN formula ),  CONN in AND, OR, IMPLIES, IFF

The scratchpad's step 0 is the formula with every atom replaced by its truth value (the lookups); each
later step evaluates every innermost operator at once, so a formula of height h takes h steps and every
step is local. Loss is on the tokens from ``targets_from`` on: the answer alone (direct) or the whole
scratchpad and answer (scratch).

Design notes (experiments/plan_formula_evaluation.md):
  - every formula height is equally common within a split (1-3 in training, 4-5 in test_deep): random
    formulas mostly fall short of their maximum depth, and height-1 items can be answered by counting
    true atoms, so an unbalanced mix would reward that shortcut;
  - labels are balanced 50/50 within each stratum (outer operator, height, number of objects);
  - every test item has a minimal-pair twin: the same formula in a world that differs in one atom of the
    formula, with the opposite answer, so a predictor that ignores the world gets no pair right;
  - object names are drawn from 0-99 per world, so no name is ever unseen in training;
  - no (world, formula) item occurs twice anywhere.

Lookup diagnostic (``--lookup``): every formula is a single atom, so an item asks only whether one atom
holds in the world. ``--encoding full_table`` lists every possible atom followed by its value instead of
only the true atoms, which turns a lookup into "find the matching atom, copy the next token".

Usage:
    python data/eval_generator.py --out datasets/eval_stage_a
    python data/eval_generator.py --lookup --encoding full_table --out datasets/eval_lookup_full_table
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from utils.vocabulary import Vocabulary  # noqa: E402

UNARY_PREDS = [0, 1, 2, 3]
BINARY_PREDS = [4]
CONNECTIVES = ["AND", "OR", "IMPLIES", "IFF"]
MAX_LEN = 512  # the model's positional encoding covers 512 positions

TRUTH = {
    "AND": lambda a, b: a and b,
    "OR": lambda a, b: a or b,
    "IMPLIES": lambda a, b: (not a) or b,
    "IFF": lambda a, b: a == b,
}

# Formula trees: ("atom", (pred, args)) | ("not", sub) | ("bin", conn, left, right) | ("val", bool)


def evaluate(node, world: set) -> bool:
    kind = node[0]
    if kind == "atom":
        return node[1] in world
    if kind == "val":
        return node[1]
    if kind == "not":
        return not evaluate(node[1], world)
    return TRUTH[node[1]](evaluate(node[2], world), evaluate(node[3], world))


def height(node) -> int:
    if node[0] in ("atom", "val"):
        return 0
    if node[0] == "not":
        return 1 + height(node[1])
    return 1 + max(height(node[2]), height(node[3]))


def atoms_of(node) -> List[Tuple]:
    if node[0] == "atom":
        return [node[1]]
    if node[0] == "val":
        return []
    if node[0] == "not":
        return atoms_of(node[1])
    return atoms_of(node[2]) + atoms_of(node[3])


def substitute(node, world: set):
    """Step 0 of the scratchpad: every atom replaced by its truth value."""
    if node[0] == "atom":
        return ("val", node[1] in world)
    if node[0] == "not":
        return ("not", substitute(node[1], world))
    if node[0] == "bin":
        return ("bin", node[1], substitute(node[2], world), substitute(node[3], world))
    return node


def reduce_innermost(node):
    """One scratchpad step: every operator whose operands are all values is evaluated."""
    if node[0] == "val":
        return node
    if node[0] == "not":
        sub = node[1]
        return ("val", not sub[1]) if sub[0] == "val" else ("not", reduce_innermost(sub))
    left, right = node[2], node[3]
    if left[0] == "val" and right[0] == "val":
        return ("val", TRUTH[node[1]](left[1], right[1]))
    return ("bin", node[1], reduce_innermost(left), reduce_innermost(right))


def trace(node, world: set) -> List:
    """The scratchpad steps: step 0 (lookups), then one reduction per level until a single value."""
    steps = [substitute(node, world)]
    while steps[-1][0] != "val":
        steps.append(reduce_innermost(steps[-1]))
    return steps


class Tokens:
    def __init__(self, vocab: Vocabulary):
        self.v = vocab
        lab = vocab.encode_label
        self.CONST, self.PRED = vocab.compositional_tokens["CONST"], vocab.compositional_tokens["PRED"]
        self.LP, self.RP, self.COMMA, self.NOT = lab("LPAREN"), lab("RPAREN"), lab("COMMA"), lab("NOT")
        self.SEP, self.MODELS, self.QUERY, self.ANSWER = lab("SEP"), lab("MODELS"), lab("QUERY"), lab("ANSWER")
        self.TRUE, self.FALSE = lab("TRUE"), lab("FALSE")
        self.conn = {c: lab(c) for c in CONNECTIVES}
        self.start = vocab.message_start

    def atom(self, a) -> List[int]:
        return self.atom_open(a) + [self.RP]

    def atom_open(self, a) -> List[int]:
        """The atom without its closing parenthesis, so it ends with the last token of its key."""
        pred, args = a
        out = [self.PRED, pred, self.LP]
        for i, c in enumerate(args):
            if i:
                out.append(self.COMMA)
            out += [self.CONST, c]
        return out

    def formula(self, node) -> List[int]:
        kind = node[0]
        if kind == "atom":
            return self.atom(node[1])
        if kind == "val":
            return [self.TRUE if node[1] else self.FALSE]
        if kind == "not":
            return [self.NOT] + self.formula(node[1])
        return [self.LP] + self.formula(node[2]) + [self.conn[node[1]]] + self.formula(node[3]) + [self.RP]

    def world(self, objects: List[int], true_atoms: List[Tuple]) -> List[int]:
        out = []
        for c in objects:
            out += [self.CONST, c]
        out.append(self.SEP)
        for a in true_atoms:
            out += self.atom(a)
        return out

    def table(self, objects: List[int], listing: List[Tuple], keylast: bool = False) -> List[int]:
        """Full-table world: every possible atom, each followed by TRUE or FALSE. With ``keylast`` the value
        comes right after the atom's last key token, inside the parenthesis: PRED p ( CONST a v )."""
        out = []
        for c in objects:
            out += [self.CONST, c]
        out.append(self.SEP)
        for a, value in listing:
            v = self.TRUE if value else self.FALSE
            out += self.atom_open(a) + [v, self.RP] if keylast else self.atom(a) + [v]
        return out

    def item(self, objects, true_atoms, formula, world_set, table: Optional[List[Tuple]] = None,
             keylast: bool = False) -> Dict:
        """With ``keylast`` (lookup items only) the world is a key-last table and the query is the atom
        without its closing parenthesis, so the answer is predicted at the query's last key token:
        MODELS PRED p ( CONST q -> v. Both formats are then that one sequence."""
        label = evaluate(formula, world_set)
        answer = self.TRUE if label else self.FALSE
        if keylast:
            if formula[0] != "atom" or table is None:
                raise ValueError("the key-last encoding is defined for single-atom items with a full table")
            seq = [self.start] + self.table(objects, table, keylast=True) + [self.MODELS] + \
                self.atom_open(formula[1]) + [answer]
            return {"direct_seq": seq, "direct_from": len(seq) - 1, "scratch_seq": seq,
                    "scratch_from": len(seq) - 1, "label": int(label)}
        world = self.table(objects, table) if table is not None else self.world(objects, true_atoms)
        body = [self.start] + world + [self.MODELS] + self.formula(formula)
        direct = body + [self.ANSWER, answer]
        scratch = body + [self.QUERY]
        scratch_from = len(scratch)
        for k, step in enumerate(trace(formula, world_set)):
            if k:
                scratch.append(self.SEP)
            scratch += self.formula(step)
        scratch += [self.ANSWER, answer]
        return {"direct_seq": direct, "direct_from": len(direct) - 1,
                "scratch_seq": scratch, "scratch_from": scratch_from, "label": int(label)}


def possible_atoms(objects: List[int], unary=UNARY_PREDS, binary=BINARY_PREDS) -> List[Tuple]:
    """Every ground atom over the objects, in a fixed order."""
    out = [(p, (c,)) for p in unary for c in objects]
    return out + [(p, (c, d)) for p in binary for c in objects for d in objects]


def multi_query_item(tok: "Tokens", objects, true_atoms, queries, table=None, keylast: bool = False) -> Dict:
    """A lookup item with several queries: MODELS q1 ANSWER v1 SEP q2 ANSWER v2 ..., loss on every v.
    With ``keylast``: MODELS PRED p ( CONST q1 v1 ) SEP PRED p ( CONST q2 v2 ...

    The first query has exactly the single-query layout, so models trained on these are tested on
    ordinary single-query items.
    """
    world = tok.table(objects, table, keylast) if table is not None else tok.world(objects, true_atoms)
    seq = [tok.start] + world + [tok.MODELS]
    targets = []
    for k, (a, value) in enumerate(queries):
        if k:
            seq += [tok.RP, tok.SEP] if keylast else [tok.SEP]
        seq += tok.atom_open(a) if keylast else tok.atom(a) + [tok.ANSWER]
        targets.append(len(seq))
        seq.append(tok.TRUE if value else tok.FALSE)
    return {"direct_seq": seq, "direct_from": targets[0], "direct_targets": targets,
            "scratch_seq": seq, "scratch_from": targets[0], "scratch_targets": targets,
            "label": int(queries[0][1]), "queries": len(queries)}


def build_multi_query_split(tok: "Tokens", sampler: "Sampler", n: int, seen: set, k: int,
                            encoding: str) -> Tuple[List[Dict], Dict]:
    """n training items with k lookup queries each, half of them true, in random order."""
    items, stats = [], {"too_long": 0, "duplicates": 0}
    while len(items) < n:
        objects, true_atoms, rho = sampler.world()
        possible = possible_atoms(objects, sampler.unary, sampler.binary)
        true_set = set(true_atoms)
        false_atoms = [a for a in possible if a not in true_set]
        queries = []
        for _ in range(k):
            want = random.random() < 0.5
            pool = true_atoms if (want and true_atoms) or not false_atoms else false_atoms
            a = random.choice(pool)
            queries.append((a, a in true_set))
        table = None
        if encoding in ("full_table", "keylast"):
            order = list(possible)
            random.shuffle(order)
            table = [(x, x in true_set) for x in order]
        it = multi_query_item(tok, objects, true_atoms, queries, table, keylast=encoding == "keylast")
        key = tuple(it["direct_seq"])
        if len(it["direct_seq"]) > MAX_LEN:
            stats["too_long"] += 1
            continue
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        it.update({"pair": None, "outer": "ATOM", "height": 0, "objects": len(objects), "rho": rho})
        items.append(it)
    return items, stats


class Sampler:
    """Worlds and formulas for one split."""

    def __init__(self, objects=(2, 3, 4), rho=(0.5,), max_height=(1, 2, 3), height_weights=None,
                 p_atom=0.3, p_not=0.25, conn_weights=None, not_mode="any", min_height=1, index_pool=100,
                 unary=UNARY_PREDS, binary=BINARY_PREDS):
        self.unary, self.binary = list(unary), list(binary)
        self.objects, self.rho = list(objects), list(rho)
        self.max_height = list(max_height)
        self.height_weights = height_weights or [1.0] * len(self.max_height)
        self.p_atom, self.p_not = p_atom, p_not
        self.conn_weights = conn_weights or [1.0] * len(CONNECTIVES)
        self.not_mode, self.min_height, self.index_pool = not_mode, min_height, index_pool

    def world(self):
        m = random.choice(self.objects)
        rho = random.choice(self.rho)
        objects = random.sample(range(self.index_pool), m)
        possible = possible_atoms(objects, self.unary, self.binary)
        true_atoms = [a for a in possible if random.random() < rho]
        random.shuffle(true_atoms)
        return objects, true_atoms, rho

    def atom(self, objects):
        pred = random.choice(self.unary + self.binary)
        arity = 2 if pred in self.binary else 1
        return ("atom", (pred, tuple(random.choice(objects) for _ in range(arity))))

    def formula(self, objects, depth, max_depth):
        if max_depth == 0:  # lookup items: the formula is one atom
            return self.atom(objects)
        if depth > 0 and (depth >= max_depth or random.random() < self.p_atom):
            return self.atom(objects)
        allow_not = self.not_mode != "none"
        if allow_not and random.random() < self.p_not:
            sub = self.formula(objects, depth + 1, max_depth)
            if self.not_mode == "atoms" and sub[0] != "atom":
                sub = self.atom(objects)
            if self.not_mode == "compound" and sub[0] == "atom":
                return self.formula(objects, depth, max_depth)
            return ("not", sub)
        conn = random.choices(CONNECTIVES, weights=self.conn_weights)[0]
        return ("bin", conn, self.formula(objects, depth + 1, max_depth), self.formula(objects, depth + 1, max_depth))

    def draw(self):
        """A world and a formula whose height is exactly a target drawn from ``max_height``, so every
        height is equally common (most random formulas fall short of their maximum depth)."""
        while True:
            objects, true_atoms, rho = self.world()
            target = random.choices(self.max_height, weights=self.height_weights)[0]
            f = self.formula(objects, 0, target)
            if height(f) != target or height(f) < self.min_height:
                continue
            if self.not_mode == "compound" and not _has_not_over_compound(f):
                continue
            return objects, true_atoms, rho, f


def _has_not_over_compound(node) -> bool:
    if node[0] in ("atom", "val"):
        return False
    if node[0] == "not":
        return node[1][0] != "atom" or _has_not_over_compound(node[1])
    return _has_not_over_compound(node[2]) or _has_not_over_compound(node[3])


def outer(node) -> str:
    return node[1] if node[0] == "bin" else node[0].upper()


def features(objects, true_atoms, formula, world_set, unary=UNARY_PREDS, binary=BINARY_PREDS) -> Dict:
    """Surface features for the shallow baseline: what a predictor could use without composing."""
    atoms = atoms_of(formula)
    n_possible = len(possible_atoms(objects, unary, binary))
    toks = []

    def walk(n):
        if n[0] == "not":
            toks.append("NOT")
            walk(n[1])
        elif n[0] == "bin":
            toks.append(n[1])
            walk(n[2])
            walk(n[3])
    walk(formula)
    return {"n_atoms": len(atoms), "n_true": sum(a in world_set for a in atoms),
            "frac_true": sum(a in world_set for a in atoms) / len(atoms),
            "n_not": toks.count("NOT"), **{f"n_{c}": toks.count(c) for c in CONNECTIVES},
            "density": len(true_atoms) / n_possible}


def build_split(tok: Tokens, sampler: Sampler, n: int, seen: set, pairs: bool,
                encoding: str = "true_atoms") -> Tuple[List[Dict], Dict]:
    """n base items, labels balanced within (outer operator, height, objects); twins added if pairs.

    With ``encoding="full_table"`` the world lists every possible atom (in a random order shared by the
    twins) with its value, and a twin flips that value instead of adding or removing the atom.
    ``encoding="keylast"`` is the same table with each value right after its atom's last key token.
    """
    counts = defaultdict(lambda: [0, 0])
    items, stats = [], {"too_long": 0, "duplicates": 0, "no_twin": 0, "balance_rejects": 0}
    base = 0
    while base < n:
        objects, true_atoms, rho, f = sampler.draw()
        world_set = set(true_atoms)
        label = int(evaluate(f, world_set))
        stratum = (outer(f), height(f), len(objects))
        if counts[stratum][label] > counts[stratum][1 - label]:
            stats["balance_rejects"] += 1
            continue
        twin = None
        if pairs:
            flips = [a for a in set(atoms_of(f)) if evaluate(f, world_set ^ {a}) != bool(label)]
            if not flips:
                stats["no_twin"] += 1
                continue
            a = random.choice(flips)
            twin_atoms = [x for x in true_atoms if x != a]
            if a not in world_set:
                twin_atoms.insert(random.randrange(len(twin_atoms) + 1), a)
            twin = (twin_atoms, set(twin_atoms))
        group = [(true_atoms, world_set)] + ([twin] if twin else [])
        if encoding in ("full_table", "keylast"):
            order = possible_atoms(objects, sampler.unary, sampler.binary)
            random.shuffle(order)
            built = [tok.item(objects, ta, f, ws, table=[(x, x in ws) for x in order],
                              keylast=encoding == "keylast") for ta, ws in group]
        else:
            built = [tok.item(objects, ta, f, ws) for ta, ws in group]
        keys = [tuple(b["direct_seq"][:-1]) for b in built]
        if any(len(b["scratch_seq"]) > MAX_LEN for b in built):
            stats["too_long"] += 1
            continue
        if any(k in seen for k in keys) or len(set(keys)) < len(keys):
            stats["duplicates"] += 1
            continue
        counts[stratum][label] += 1
        base += 1
        pair_id = base if pairs else None
        for b, k, (ta, ws) in zip(built, keys, group):
            seen.add(k)
            b.update({"pair": pair_id, "outer": stratum[0], "height": stratum[1], "objects": stratum[2],
                      "rho": rho, **features(objects, ta, f, ws, sampler.unary, sampler.binary)})
            items.append(b)
    return items, stats


SPLITS = {
    # name: (sampler kwargs, number of base items, with twins)
    "train": ({}, 50000, False),
    "val": ({}, 2000, False),
    "test_iid": ({}, 1000, True),
    "test_deep": ({"max_height": (4, 5)}, 1000, True),
    "test_large_world": ({"objects": (5, 6)}, 1000, True),
    "test_shift": ({"rho": (0.2, 0.8), "p_not": 0.5, "conn_weights": [0.1, 0.1, 0.4, 0.4]}, 1000, True),
    "test_negation": ({"not_mode": "compound"}, 1000, True),
}


LOOKUP = {"max_height": (0,), "min_height": 0}
LOOKUP_SPLITS = {
    # Single-atom items: does the model find an atom in the world? 5 objects is the largest world whose
    # full table fits in MAX_LEN, so both encodings are tested at 5.
    "train": (LOOKUP, 50000, False),
    "val": (LOOKUP, 2000, False),
    "test_iid": (LOOKUP, 1000, True),
    "test_large_world": ({**LOOKUP, "objects": (5,)}, 1000, True),
}


# Key-complexity ladder for lookups. Each rung keeps the number of possible atoms per world at about
# 12, 21 or 32 (the full signature's counts for 2, 3, 4 objects), so only the key's structure changes.
SIGNATURES = {
    "single": ([0], [], (12, 21, 32)),        # one predicate: the key is the object token alone
    "unary": ([0, 1, 2, 3], [], (3, 5, 8)),   # (predicate, object) at fixed offsets
    "full": (UNARY_PREDS, BINARY_PREDS, (2, 3, 4)),  # Stage A: unary and binary atoms
}


def generate(out_dir: str, vocab_path: str = "unified_vocabulary.json", seed: int = 0,
             scale: float = 1.0, train_not_mode: str = "any", lookup: bool = False,
             encoding: str = "true_atoms", signature: str = "full", queries: int = 1) -> Dict:
    random.seed(seed)
    vocab = Vocabulary(vocab_path)
    tok = Tokens(vocab)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    meta = {"generator": "eval_stage_a_v1", "seed": seed, "unary_preds": UNARY_PREDS,
            "binary_preds": BINARY_PREDS, "max_len": MAX_LEN, "train_not_mode": train_not_mode, "splits": {}}
    if lookup or encoding != "true_atoms":
        meta.update({"generator": "eval_lookup_v1" if lookup else "eval_stage_a_v1", "encoding": encoding})
    splits = LOOKUP_SPLITS if lookup else SPLITS
    if signature != "full":
        if not lookup:
            raise ValueError("a reduced signature is only defined for the lookup diagnostic")
        unary, binary, objects = SIGNATURES[signature]
        splits = {name: ({**LOOKUP, "objects": objects, "unary": unary, "binary": binary}, n, pairs)
                  for name, (_, n, pairs) in LOOKUP_SPLITS.items() if name != "test_large_world"}
        meta.update({"signature": signature, "unary_preds": unary, "binary_preds": binary})
    if encoding == "keylast" and not lookup:
        raise ValueError("the key-last encoding is only defined for the lookup diagnostic")
    if queries > 1:
        if not lookup:
            raise ValueError("several queries per item are only defined for the lookup diagnostic")
        meta["train_queries"] = queries
    for name, (kwargs, n, pairs) in splits.items():
        kw = dict(kwargs)
        if name in ("train", "val", "test_iid") and train_not_mode != "any":
            kw["not_mode"] = train_not_mode
        n_items = max(1, int(n * scale))
        if name == "train" and queries > 1:
            items, stats = build_multi_query_split(tok, Sampler(**kw), n_items, seen, queries, encoding)
        else:
            items, stats = build_split(tok, Sampler(**kw), n_items, seen, pairs, encoding)
        text = json.dumps({"split": name, "items": items})
        (out / f"{name}.json").write_text(text)
        meta["splits"][name] = {"sampler": kw, "base_items": n_items, "items": len(items), "pairs": pairs,
                                "true_fraction": sum(i["label"] for i in items) / len(items),
                                "max_scratch_len": max(len(i["scratch_seq"]) for i in items),
                                "sha256": hashlib.sha256(text.encode()).hexdigest(), **stats}
        print(f"{name}: {len(items)} items, {stats}")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the Stage A formula-evaluation dataset")
    ap.add_argument("--out", default="datasets/eval_stage_a")
    ap.add_argument("--vocab", default="unified_vocabulary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every split's size (for quick runs)")
    ap.add_argument("--train-not-mode", default="any", choices=["any", "atoms"],
                    help="'atoms': NOT only over atoms in train/val/test_iid (test_negation is then held out)")
    ap.add_argument("--lookup", action="store_true", help="Single-atom items only (the lookup diagnostic)")
    ap.add_argument("--encoding", default="true_atoms", choices=["true_atoms", "full_table", "keylast"],
                    help="World as the list of true atoms, or every possible atom with its value; 'keylast' "
                         "(lookup only): each value right after its atom's last key token, and the answer "
                         "predicted at the query's last key token")
    ap.add_argument("--signature", default="full", choices=list(SIGNATURES),
                    help="Lookup ladder: 'single' (one predicate), 'unary' (four unary), 'full' (Stage A)")
    ap.add_argument("--queries", type=int, default=1,
                    help="Lookup diagnostic: queries per training item (val and test items keep one)")
    args = ap.parse_args()
    generate(args.out, args.vocab, args.seed, args.scale, args.train_not_mode, args.lookup, args.encoding,
             args.signature, args.queries)
