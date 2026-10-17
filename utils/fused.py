"""One token per symbol: a compositional symbol (VAR, CONST, PRED, FUNC or SORT plus its base-625
index) becomes a single token instead of two.

The split vocabulary writes ``CONST 5`` as the category token followed by the numeral. Fused ids come
after the base vocabulary: (category c, index n) -> vocab_size + c * 625 + n, with the categories in the
vocabulary's order. The model embeds a fused token as its category row plus its numeral row, so the
index is still a base-625 numeral (see ``SymbolicTransformer(fused_categories=...)``).

Indices are single base-625 digits (0-624), which is all any dataset here uses; a category token is
fused with exactly the one numeral after it.
"""

from typing import Dict, List, Sequence, Tuple

N_INDEX = 625


class SymbolFuser:
    def __init__(self, vocab):
        if tuple(vocab.numeral_range) != (0, N_INDEX - 1):
            raise ValueError(f"fused ids assume numerals 0-{N_INDEX - 1}, got {vocab.numeral_range}")
        self.base = vocab.vocab_size
        self.categories = list(vocab.compositional_tokens.values())
        self.names = list(vocab.compositional_tokens)
        self.cat_index = {t: i for i, t in enumerate(self.categories)}
        self.size = self.base + len(self.categories) * N_INDEX

    def fused_id(self, category: int, index: int) -> int:
        return self.base + self.cat_index[category] * N_INDEX + index

    def split(self, token: int) -> Tuple[int, int]:
        """(category token, numeral) of a fused token."""
        c, n = divmod(token - self.base, N_INDEX)
        return self.categories[c], n

    def label(self, token: int, vocab) -> str:
        if token < self.base:
            return vocab.decode_id(token)
        c, n = divmod(token - self.base, N_INDEX)
        return f"{self.names[c]}_{n}"

    def fuse(self, ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        """The fused sequence, and for every input position the position of the token that holds it."""
        out, where = [], []
        j = 0
        while j < len(ids):
            t = ids[j]
            if t in self.cat_index:
                if j + 1 >= len(ids) or not 0 <= ids[j + 1] < N_INDEX:
                    raise ValueError(f"category token {t} at position {j} is not followed by a numeral")
                where += [len(out), len(out)]
                out.append(self.fused_id(t, ids[j + 1]))
                j += 2
            else:
                where.append(len(out))
                out.append(t)
                j += 1
        return out, where

    def unfuse(self, ids: Sequence[int]) -> List[int]:
        out = []
        for t in ids:
            out += list(self.split(t)) if t >= self.base else [t]
        return out

    def fuse_item(self, item: Dict, formats: Sequence[str] = ("direct", "scratch")) -> Dict:
        """A copy of an eval_generator item with each format's sequence fused and its loss positions
        (``_from``, ``_targets``) moved to the tokens that now hold them."""
        out = dict(item)
        for fmt in formats:
            if f"{fmt}_seq" not in item:
                continue
            seq, where = self.fuse(item[f"{fmt}_seq"])
            out[f"{fmt}_seq"] = seq
            out[f"{fmt}_from"] = where[item[f"{fmt}_from"]]
            if f"{fmt}_targets" in item:
                out[f"{fmt}_targets"] = sorted({where[t] for t in item[f"{fmt}_targets"]})
        return out

    def fuse_samples(self, samples: List[Dict]) -> List[Dict]:
        """Next-symbol samples (context -> target) at symbol level.

        A sample whose target is a category token takes the numeral from the next sample, which must
        continue the same sequence; a sample whose target is that numeral is then dropped.
        """
        out = []
        for i, s in enumerate(samples):
            ctx = list(s.get("context", s.get("input_ids", [])))
            if ctx and ctx[-1] in self.cat_index:
                continue
            target = s["target"]
            if target in self.cat_index:
                nxt = samples[i + 1] if i + 1 < len(samples) else None
                nctx = None if nxt is None else list(nxt.get("context", nxt.get("input_ids", [])))
                if nctx != ctx + [target] or not 0 <= nxt["target"] < N_INDEX:
                    raise ValueError(f"sample {i}: target {target} is a category, but the next sample does "
                                     "not supply its numeral")
                target = self.fused_id(target, nxt["target"])
            fused = dict(s)
            fused.pop("input_ids", None)
            fused["context"] = self.fuse(ctx)[0]
            fused["target"] = target
            out.append(fused)
        return out
