#!/usr/bin/env python3
"""
Evaluate a trained Symbolic Transformer checkpoint on a next-symbol dataset.

Usage:
    python evaluate_model.py
    python evaluate_model.py --checkpoint checkpoints/best_model.pt \
        --test-data datasets/fol_next_symbol/test.json --max-samples 5000

Reports top-1/5/10 accuracy, shows a few sample predictions, and prints the
checkpoint's own training metadata (no hard-coded numbers).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Make the repo importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transformer import load_model_from_checkpoint  # noqa: E402
from training.train import FOLDataset, trim_padding  # noqa: E402
from utils.vocabulary import Vocabulary  # noqa: E402


def load_model_checkpoint(checkpoint_path: str, vocab_size: int, device: torch.device):
    """Load a trained model (and its checkpoint dict) from disk."""
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, vocab_size, device)
    config = checkpoint['config']
    print(f"✓ Loaded {config['model_size']} model from epoch {checkpoint.get('epoch')}")
    if 'best_val_loss' in checkpoint:
        print(f"✓ Best val loss: {checkpoint['best_val_loss']:.4f}")
    return model, checkpoint


@torch.no_grad()
def compute_topk_accuracy(model, dataset: FOLDataset, device: torch.device,
                          ks=(1, 5, 10), batch_size: int = 256, max_samples: int = None):
    """Compute top-k accuracies, reading logits at the last real token."""
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    correct = {k: 0 for k in ks}
    kmax = max(ks)
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(n, start + batch_size))]
        input_ids = torch.stack([it['input_ids'] for it in items]).to(device)
        lengths = torch.stack([it['length'] for it in items]).to(device)
        targets = torch.stack([it['target'] for it in items]).to(device)
        logits = model(trim_padding(input_ids, lengths))
        last = logits[torch.arange(input_ids.size(0), device=device), lengths - 1, :]
        topk = torch.topk(last, k=kmax, dim=-1).indices
        hits = topk == targets.unsqueeze(1)
        for k in ks:
            correct[k] += int(hits[:, :k].any(dim=1).sum().item())
    return {k: correct[k] / max(1, n) for k in ks}, n


@torch.no_grad()
def show_samples(model, dataset: FOLDataset, vocab: Vocabulary, device: torch.device, n_show: int = 10):
    """Print a few predictions with their contexts."""
    for i in range(min(n_show, len(dataset))):
        item = dataset[i]
        length = int(item['length'])
        context = item['input_ids'].tolist()[:length]
        target = int(item['target'])

        logits = model(torch.tensor([context], dtype=torch.long, device=device))
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_ids = top5_ids.tolist()
        top5_probs = top5_probs.tolist()

        context_str = vocab.decode_formula_simple(context[-10:])
        target_str = vocab.decode_id(target)
        pred_str = vocab.decode_id(top5_ids[0])
        correct = top5_ids[0] == target

        print(f"\n{'=' * 60}")
        print(f"Example {i + 1}:")
        print(f"  Context: ...{context_str}")
        print(f"  Target:     {target_str:15s} {'✓' if correct else '✗'}")
        print(f"  Predicted:  {pred_str:15s} (confidence: {top5_probs[0]:.1%})")
        if not correct and target in top5_ids:
            print(f"  (Target was #{top5_ids.index(target) + 1} in top-5)")
        print("  Top-5 predictions:")
        for j, (tid, prob) in enumerate(zip(top5_ids, top5_probs)):
            marker = "← TARGET" if tid == target else ""
            print(f"    {j + 1}. {vocab.decode_id(tid):15s} {prob:6.1%} {marker}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Symbolic Transformer checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--test-data", default="datasets/fol_next_symbol/test.json")
    parser.add_argument("--vocab", default="unified_vocabulary.json")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate at most this many samples (default: all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-show", type=int, default=10, help="Number of sample predictions to print")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("EVALUATING TRAINED MODEL")
    print("=" * 60)

    for path in [args.vocab, args.checkpoint, args.test_data]:
        if not Path(path).exists():
            print(f"❌ File not found: {path}")
            return 1

    device = torch.device(args.device)
    print("\nLoading vocabulary...")
    vocab = Vocabulary(args.vocab)

    print("\nLoading model...")
    model, checkpoint = load_model_checkpoint(args.checkpoint, vocab.vocab_size, device)
    n_params = model.count_parameters()

    print("\nLoading test data...")
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    dataset = FOLDataset(args.test_data, pad_id=pad_id)

    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)
    accs, n_eval = compute_topk_accuracy(model, dataset, device, batch_size=args.batch_size,
                                         max_samples=args.max_samples)
    for k in (1, 5, 10):
        print(f"✓ Top-{k:<2} Accuracy: {accs[k]:.2%}")
    random_top1 = 1.0 / vocab.vocab_size
    print(f"\n📊 Random top-1 would be {random_top1:.2%}; model is {accs[1] / random_top1:.1f}× better "
          f"(evaluated on {n_eval} samples)")

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    show_samples(model, dataset, vocab, device, n_show=args.n_show)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    cfg = checkpoint.get('config', {})
    print(f"Model size:      {cfg.get('model_size', '?')} ({n_params:,} params, "
          f"{cfg.get('embedding_type', 'learned')} embedding{', tied' if cfg.get('tie_weights') else ''})")
    print(f"Checkpoint:      {args.checkpoint} (epoch {checkpoint.get('epoch', '?')}, "
          f"step {checkpoint.get('global_step', '?')})")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss:   {checkpoint['best_val_loss']:.4f}")
    print(f"Test data:       {args.test_data}")
    print("")
    print("Performance:")
    for k in (1, 5, 10):
        print(f"  Top-{k}:  {accs[k]:.1%}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
