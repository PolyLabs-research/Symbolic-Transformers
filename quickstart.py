#!/usr/bin/env python3
"""
Quick-start script for the FOL Symbolic Transformer.
Checks prerequisites and runs one of a few small end-to-end steps.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           FOL SYMBOLIC TRANSFORMER - QUICK START         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")

# Check prerequisites
print("\n🔍 Checking prerequisites...")

if sys.version_info < (3, 10):
    print("❌ Python 3.10+ required")
    sys.exit(1)
print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        device_str = "cuda"
    else:
        print("⚠️  No GPU detected - will use CPU (slower but works)")
        device_str = "cpu"
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install with: pip install torch")
    sys.exit(1)

if not Path("unified_vocabulary.json").exists():
    print("\n❌ unified_vocabulary.json not found (it ships with the repo; re-clone if missing)")
    sys.exit(1)
print("✅ Unified vocabulary found")

print("\n" + "=" * 60)
print("✅ All prerequisites met!")
print("=" * 60)


def run(cmd):
    """Run a command in this interpreter and stream its output."""
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call([sys.executable] + cmd)


print("\nWhat would you like to do?")
print("1. Test vocabulary loading (seconds)")
print("2. Generate a small dataset (1,000 formulas, ~1 min)")
print("3. Train a tiny model for a few epochs (CPU ok, ~10 min)")
print("4. Full pipeline: large advanced dataset + small model (GPU recommended)")
print("5. Exit")

choice = input("\nEnter choice (1-5): ").strip()

if choice == "1":
    from utils.vocabulary import Vocabulary

    vocab = Vocabulary("unified_vocabulary.json")
    print("\n✅ Numerals:", [vocab.encode_numeral(v) for v in (0, 1, 24, 100, 624)])
    print("✅ Symbols:", {label: vocab.encode_label(label) for label in ('FORALL', 'EXISTS', 'AND', 'IMPLIES')})
    print("✅ Compositional: VAR 12 →", vocab.encode_compositional('VAR', 12),
          "| VAR 700 →", vocab.encode_compositional('VAR', 700))
    print("\n✅ Vocabulary system working correctly!")

elif choice == "2":
    rc = run(["data/dataset_generator.py", "--out", "datasets/fol_next_symbol",
              "--n-train", "1000", "--n-val", "200", "--n-test", "200"])
    if rc == 0:
        print("\n✅ Dataset generated at datasets/fol_next_symbol/")

elif choice == "3":
    if not Path("datasets/fol_next_symbol/train.json").exists():
        print("\n❌ Dataset not found. Run option 2 first.")
        sys.exit(1)
    epochs = input("Epochs [5]: ").strip() or "5"
    rc = run(["training/train.py", "--model-size", "tiny", "--num-epochs", epochs, "--batch-size", "32"])
    if rc == 0:
        print("\n✅ Training complete! Checkpoints saved to checkpoints/")
        print("   Evaluate with: python evaluate_model.py")
        print("   Explore with:  python demo.py")

elif choice == "4":
    print("\nThis generates 50,000 formulas with the advanced generator and trains the 'small' model.")
    if device_str == "cpu":
        print("⚠️  No GPU detected - this will be very slow on CPU.")
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)
    rc = run(["data/advanced_generator.py"])
    if rc == 0:
        rc = run(["training/train.py", "--model-size", "small", "--num-epochs", "100"])
    if rc == 0:
        rc = run(["evaluate_model.py"])

else:
    print("\nExiting.")
    sys.exit(0)

print("\n" + "=" * 60)
print("Next steps:")
print("=" * 60)
print("- See README.md for full documentation")
print("- See TECHNICAL_ARCHITECTURE.md for details")
print("- python symbolic_benchmarks.py --checkpoint checkpoints/best_model.pt --vocab unified_vocabulary.json")
print("- Adjust hyperparameters in training/train.py")
print("=" * 60)
