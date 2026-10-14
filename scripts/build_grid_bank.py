#!/usr/bin/env python3
"""Build grids/grid_bank.npz from unified_vocabulary.json and the glyph PNGs in out/.

Usage (from the repo root):
    python scripts/build_grid_bank.py [--vocab unified_vocabulary.json] [--out grids/grid_bank.npz]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.grid_bank import build_grid_bank, DEFAULT_BANK_PATH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the base-625 grid bank for the vocabulary")
    ap.add_argument("--vocab", default=str(ROOT / "unified_vocabulary.json"))
    ap.add_argument("--out", default=str(DEFAULT_BANK_PATH))
    ap.add_argument("--no-verify", action="store_true", help="Skip checking numeral panels against digit PNGs")
    args = ap.parse_args()

    bank = build_grid_bank(Path(args.vocab), repo_root=ROOT, verify_pngs=not args.no_verify)
    bank.save(Path(args.out))
    print(f"✓ Wrote {args.out}: {bank.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
