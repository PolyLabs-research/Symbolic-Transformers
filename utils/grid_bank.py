"""
Grid bank: the 25x25 base-625 panel behind every grid-backed token.

Numerals 0-624 are the canonical count encoding (k filled cells in row-major
order, a thermometer code: the panel for k is a subset of the panel for k+1).
Operator glyphs are read from the painted PNGs shipped in ``out/symbols``.
Special tokens (MESSAGE_START, PAD, ...) have no panel; ``has_grid`` is False
for them and the model falls back to a learned embedding.

The bank is built once by ``scripts/build_grid_bank.py`` into
``grids/grid_bank.npz`` so nothing at training time depends on base-625.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

GRID_N = 25
GRID_CELLS = GRID_N * GRID_N  # 625
DEFAULT_BANK_PATH = Path(__file__).resolve().parents[1] / "grids" / "grid_bank.npz"


def thermometer_grid(k: int) -> np.ndarray:
    """Row-major count panel for value k: the first k cells filled. Shape (625,), uint8."""
    if not (0 <= k <= GRID_CELLS):
        raise ValueError(f"k must be in [0, {GRID_CELLS}], got {k}")
    g = np.zeros(GRID_CELLS, dtype=np.uint8)
    g[:k] = 1
    return g


def sample_png_to_bits(png_path: Path) -> np.ndarray:
    """Reconstruct the 625-bit row-major vector from a panel PNG by sampling cell centres."""
    from PIL import Image  # local import: PIL only needed when building the bank

    im = Image.open(png_path).convert("RGB")
    w, h = im.width, im.height
    cell_px = max(1, round((w - 1) / GRID_N))
    bits = np.zeros((GRID_N, GRID_N), dtype=np.uint8)
    for r in range(GRID_N):
        for c in range(GRID_N):
            x = min(w - 1, c * cell_px + cell_px // 2)
            y = min(h - 1, r * cell_px + cell_px // 2)
            R, G, B = im.getpixel((x, y))
            bits[r, c] = 1 if (R + G + B) < (255 * 3 / 2) else 0
    return bits.reshape(-1)


@dataclass
class GridBank:
    """Per-token feature rows aligned by token id.

    For the real grid bank a row is the 625-bit panel (uint8). The same container
    also carries other fixed per-token features (see ``sinusoid_bank``), so
    ``grids`` may be float and have a different width; ``has_grid`` marks the
    tokens that have a feature row at all.
    """

    grids: np.ndarray      # (vocab_size, n_features); uint8 panels or float features
    has_grid: np.ndarray   # (vocab_size,) bool
    labels: List[str]      # token labels, for sanity checks

    @property
    def vocab_size(self) -> int:
        return int(self.grids.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.grids.shape[1])

    @classmethod
    def empty(cls, vocab_size: int, n_features: int = GRID_CELLS) -> "GridBank":
        """A bank of all-zero rows; used when a checkpoint will supply the real buffers."""
        return cls(
            grids=np.zeros((vocab_size, n_features), dtype=np.float32),
            has_grid=np.zeros(vocab_size, dtype=bool),
            labels=[""] * vocab_size,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, grids=self.grids, has_grid=self.has_grid,
                            labels=np.array(self.labels, dtype=object))

    @classmethod
    def load(cls, path: Path = DEFAULT_BANK_PATH) -> "GridBank":
        with np.load(Path(path), allow_pickle=True) as z:
            return cls(grids=z["grids"].astype(np.uint8), has_grid=z["has_grid"].astype(bool),
                       labels=[str(x) for x in z["labels"].tolist()])

    def summary(self) -> Dict[str, int]:
        return {
            "vocab_size": self.vocab_size,
            "grid_backed": int(self.has_grid.sum()),
            "numerals": int(sum(1 for i, l in enumerate(self.labels) if l.startswith("NUM_") and self.has_grid[i])),
            "glyphs": int(sum(1 for i, l in enumerate(self.labels) if not l.startswith("NUM_") and self.has_grid[i])),
        }


def sinusoid_features(k: np.ndarray, n_freq: int = 32, max_period: float = 2 * GRID_CELLS,
                      min_period: float = 2.0) -> np.ndarray:
    """Fixed multi-frequency sin/cos features of integer values k, shape (len(k), 2*n_freq).

    Periods run geometrically from ``min_period`` to ``max_period`` (default 2 to
    1250: half a cycle over 0..624, so the slowest cosine is monotone across the
    whole numeral range). A larger ``min_period`` drops the high frequencies that
    let a linear layer memorise individual values, giving a smoother code. This is the standard positional-encoding
    recipe applied to a numeral's value instead of a sequence position: an
    ordinal code that is smooth in k, with no 2D structure and no per-cell
    parameters, which makes it the control for "is it the grid or just an
    ordinal encoding that helps?".
    """
    k = np.asarray(k, dtype=np.float64)[:, None]
    periods = np.geomspace(min_period, max_period, n_freq)[None, :]
    angles = 2 * np.pi * k / periods
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def cell_position_features(n_index_freq: int = 8, n_axis_freq: int = 4,
                           index_periods: Tuple[float, float] = (32.0, 2 * GRID_CELLS),
                           axis_periods: Tuple[float, float] = (5.0, 2 * GRID_N)) -> np.ndarray:
    """Fixed smooth features of every panel cell's position, shape (625, F).

    For cell i at (row, col): sin/cos of the row-major index i at ``n_index_freq``
    periods, sin/cos of row and of col at ``n_axis_freq`` periods each, plus
    linear ramps (i/625, row/25, col/25) and a constant. All periods are long
    relative to a single cell, so neighbouring cells get similar features.

    Used by the ``grid_shared`` encoder, which derives each cell's weight from
    these features through one shared learned matrix instead of keeping 625
    independent weight columns. A cell that is never filled in training still
    gets a weight consistent with its neighbours.
    """
    i = np.arange(GRID_CELLS, dtype=np.float64)
    row = i // GRID_N
    col = i % GRID_N

    def sincos(x: np.ndarray, lo: float, hi: float, n: int) -> List[np.ndarray]:
        periods = np.geomspace(lo, hi, n)
        ang = 2 * np.pi * x[:, None] / periods[None, :]
        return [np.sin(ang), np.cos(ang)]

    parts = sincos(i, index_periods[0], index_periods[1], n_index_freq)
    parts += sincos(row, axis_periods[0], axis_periods[1], n_axis_freq)
    parts += sincos(col, axis_periods[0], axis_periods[1], n_axis_freq)
    parts.append(np.stack([i / GRID_CELLS, row / GRID_N, col / GRID_N, np.ones_like(i)], axis=1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def sinusoid_bank(vocab_size: int, n_numerals: int = GRID_CELLS, n_freq: int = 32,
                  labels: Optional[List[str]] = None, min_period: float = 2.0) -> GridBank:
    """A bank whose numeral rows (ids 0..n_numerals-1) are sinusoid features of k.

    Every other token (glyphs, special tokens) has no feature row and uses the
    model's learned fallback embedding, exactly like special tokens do in the
    grid arms.
    """
    feats = np.zeros((vocab_size, 2 * n_freq), dtype=np.float32)
    feats[:n_numerals] = sinusoid_features(np.arange(n_numerals), n_freq=n_freq, min_period=min_period)
    has = np.zeros(vocab_size, dtype=bool)
    has[:n_numerals] = True
    return GridBank(grids=feats, has_grid=has, labels=labels or [""] * vocab_size)


def build_grid_bank(vocab_path: Path, repo_root: Optional[Path] = None, verify_pngs: bool = True) -> GridBank:
    """Build the bank from unified_vocabulary.json and the glyph PNGs it references.

    Numeral panels are generated (thermometer code); when ``verify_pngs`` is set
    they are also checked against the exported digit PNGs so the programmatic
    encoding and the shipped artwork cannot drift apart.
    """
    vocab_path = Path(vocab_path)
    repo_root = Path(repo_root) if repo_root is not None else vocab_path.resolve().parent
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    V = int(vocab["vocab_size"])
    grids = np.zeros((V, GRID_CELLS), dtype=np.uint8)
    has_grid = np.zeros(V, dtype=bool)
    labels = [""] * V
    lo, hi = vocab["numeral_range"]

    for tok in vocab["tokens"]:
        tid = int(tok["id"])
        labels[tid] = tok["label"]
        if tok["type"] == "numeral":
            k = tid - lo
            grids[tid] = thermometer_grid(k)
            has_grid[tid] = True
            if verify_pngs and tok.get("path"):
                png = repo_root / tok["path"]
                if png.exists():
                    bits = sample_png_to_bits(png)
                    if not np.array_equal(bits, grids[tid]):
                        raise ValueError(f"numeral {tid}: PNG {png} does not match the row-major count panel")
        elif tok.get("path"):
            png = repo_root / tok["path"]
            if not png.exists():
                raise FileNotFoundError(f"glyph PNG missing for {tok['label']}: {png}")
            bits = sample_png_to_bits(png)
            expected = tok.get("black_cells")
            if expected is not None and int(bits.sum()) != int(expected):
                raise ValueError(f"glyph {tok['label']}: PNG has {int(bits.sum())} cells, vocabulary says {expected}")
            grids[tid] = bits
            has_grid[tid] = True
        # else: special token, no panel

    assert lo == 0 and has_grid[lo:hi + 1].all()
    return GridBank(grids=grids, has_grid=has_grid, labels=labels)
