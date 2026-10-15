"""
Training script for Symbolic FOL Transformer.
Optimized for AMD Radeon GPU with ROCm backend.
CORRECTED: Fixes padding bug by using sequence lengths for loss calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import sys
import argparse
import re

import numpy as np

# Ensure we can import from parent directory
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import EMBEDDING_TYPES, create_model
from utils.grid_bank import DEFAULT_BANK_PATH, GridBank
from utils.vocabulary import Vocabulary


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_config(model_size='tiny', num_epochs=50, batch_size=None):
    """Helper to get training config for different model sizes."""
    if batch_size is None:
        batch_size = {
            'tiny': 64,
            'small': 32,
            'base': 16,
            'large': 8,
        }[model_size]

    return TrainingConfig(
        model_size=model_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        warmup_steps=2000,
        checkpoint_dir='checkpoints',
    )


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_size: str = 'base'
    vocab_size: int = 663
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    max_seq_len: int = 512
    train_data_path: str = "datasets/fol_next_symbol/train.json"
    val_data_path: str = "datasets/fol_next_symbol/val.json"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    log_every: int = 100
    device: str = "cuda"
    mixed_precision: bool = False
    # Embedding experiment knobs (see models/transformer.py)
    embedding_type: str = "learned"
    tie_weights: bool = False
    grid_bank: str = str(DEFAULT_BANK_PATH)
    sinusoid_min_period: float = 2.0
    shared_min_period: float = 32.0
    shared_axis_freq: int = 4
    shared_residual: bool = False
    shared_basis: str = "full"
    seed: int = 42


def _sample_context(sample: Dict) -> List[int]:
    """Return the (unpadded) context token list of a dataset sample."""
    if 'input_ids' in sample:
        return sample['input_ids']
    if 'context' in sample:
        return sample['context']
    return sample.get('input', [])


class FOLDataset(Dataset):
    """PyTorch Dataset for FOL next-symbol prediction.

    Each item is a right-padded context plus its true length, so the trainer
    reads the logits at the last *real* token. Contexts are padded with
    ``pad_id`` (the vocabulary's PAD token, not numeral 0) and only as far as
    the longest context in the file, capped at ``max_seq_len``. A context that
    is still too long keeps its *last* tokens, because the target follows the
    end of the context.
    """
    
    def __init__(self, data_path: str, max_seq_len: int = 512, pad_id: int = 0):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.pad_id = pad_id
        longest = max((len(_sample_context(s)) for s in self.samples), default=1)
        self.max_seq_len = max(1, min(max_seq_len, longest))
        n_truncated = sum(1 for s in self.samples if len(_sample_context(s)) > self.max_seq_len)
        print(f"✓ Loaded {len(self.samples)} samples from {data_path} "
              f"(padded to {self.max_seq_len}, pad_id={pad_id})")
        if n_truncated:
            print(f"⚠ {n_truncated} contexts longer than {self.max_seq_len} will keep only their last tokens")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        context = list(_sample_context(sample))
        target = sample['target']
        
        # Real length before padding; truncate from the front so the target
        # still follows the final context token.
        context_len = len(context)
        if context_len > self.max_seq_len:
            context = context[-self.max_seq_len:]
            context_len = self.max_seq_len
        elif context_len < self.max_seq_len:
            context = context + [self.pad_id] * (self.max_seq_len - context_len)
        
        return {
            'input_ids': torch.tensor(context, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'length': torch.tensor(context_len, dtype=torch.long)
        }


class Trainer:
    """Trainer for Symbolic Transformer."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        vocab: Vocabulary
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.vocab = vocab
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler: linear warmup, then cosine decay to 10%.
        total_steps = max(1, len(train_loader) * config.num_epochs)
        warmup = max(1, config.warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # No ignore_index: targets are never padding, and token 0 is the
        # numeral NUM_000 (ignoring it silently dropped ~6% of training targets).
        self.criterion = nn.CrossEntropyLoss()
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['target'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # CRITICAL FIX: Select logits at the actual end of the sequence
            # Create indices [0, 1, 2, ... batch_size-1]
            batch_indices = torch.arange(input_ids.size(0), device=self.device)
            
            # Select [batch_i, length_i - 1, :]
            # We subtract 1 because lengths are 1-based, indices are 0-based
            last_token_logits = logits[batch_indices, lengths - 1, :]
            
            # Compute loss
            loss = self.criterion(last_token_logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                logits = self.model(input_ids)
                
                # CRITICAL FIX: Same logic as training
                batch_indices = torch.arange(input_ids.size(0), device=self.device)
                last_token_logits = logits[batch_indices, lengths - 1, :]
                
                loss = self.criterion(last_token_logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'history': self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        print(f"✓ Resumed from checkpoint: {checkpoint_path}")

    def train(self, start_epoch: int = 0):
        print("\n" + "="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Model: {self.config.model_size} | embedding: {self.config.embedding_type}"
              f"{' (tied output)' if self.config.tie_weights else ''} | seed: {self.config.seed}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            print(f"Epoch {self.epoch}/{self.config.num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            epoch_time = time.time() - start_time
            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} {'[BEST]' if is_best else ''}")
            print(f"  Time:       {epoch_time:.1f}s\n")
            
            if self.epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(is_best)

def main():
    parser = argparse.ArgumentParser(description="Train the Symbolic FOL Transformer")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume from checkpoint")
    parser.add_argument("--num-epochs", type=int, help="Total epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--train-data", default=None, help="Path to train.json")
    parser.add_argument("--val-data", default=None, help="Path to val.json")
    parser.add_argument("--checkpoint-dir", default=None, help="Where to write checkpoints (default: checkpoints/)")
    parser.add_argument("--vocab", default="unified_vocabulary.json")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding", default="learned", choices=list(EMBEDDING_TYPES),
                        help="learned (original) | grid_linear | grid_conv (base-625 panel encoders)")
    parser.add_argument("--tie-weights", action="store_true",
                        help="Tie the output layer to the embedding table (recommended with grid embeddings)")
    parser.add_argument("--grid-bank", default=str(DEFAULT_BANK_PATH), help="Grid bank .npz for grid embeddings")
    parser.add_argument("--sinusoid-min-period", type=float, default=2.0,
                        help="Shortest period of the sinusoid features (--embedding sinusoid)")
    parser.add_argument("--shared-min-period", type=float, default=32.0,
                        help="grid_shared: shortest period of the cell-index features")
    parser.add_argument("--shared-axis-freq", type=int, default=4,
                        help="grid_shared: number of row/col frequencies (0 = index features only)")
    parser.add_argument("--shared-residual", action="store_true",
                        help="grid_shared: add a zero-initialised per-cell residual weight")
    parser.add_argument("--shared-basis", default="full", choices=["full", "constant"],
                        help="grid_shared: 'constant' gives every cell the same generated weight (count × one direction)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (use 0 for tiny samples or when running several trainings in parallel)")
    args = parser.parse_args()

    def resolve_checkpoint_path(resume_arg: str, checkpoint_dir: Path) -> str:
        if resume_arg == "latest":
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            def checkpoint_epoch(path: Path) -> int:
                match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
                return int(match.group(1)) if match else -1
            return str(max(checkpoints, key=checkpoint_epoch))
        return resume_arg

    config = get_training_config(model_size=args.model_size, num_epochs=args.num_epochs or 50, batch_size=args.batch_size)
    if args.train_data: config.train_data_path = args.train_data
    if args.val_data: config.val_data_path = args.val_data
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    if args.warmup_steps is not None: config.warmup_steps = args.warmup_steps
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.device: config.device = args.device
    config.seed = args.seed
    config.embedding_type = args.embedding
    config.tie_weights = args.tie_weights
    config.grid_bank = args.grid_bank
    config.sinusoid_min_period = args.sinusoid_min_period
    config.shared_min_period = args.shared_min_period
    config.shared_axis_freq = args.shared_axis_freq
    config.shared_residual = args.shared_residual
    config.shared_basis = args.shared_basis
    set_seed(config.seed)
    vocab = Vocabulary(args.vocab)
    config.vocab_size = vocab.vocab_size
    
    resume_path: Optional[str] = None
    if args.resume:
        resume_path = resolve_checkpoint_path(args.resume, Path(config.checkpoint_dir))
        checkpoint_meta = torch.load(resume_path, map_location="cpu")
        ckpt_config = checkpoint_meta.get('config', {})
        if ckpt_config.get('model_size'): config.model_size = ckpt_config['model_size']
        if ckpt_config.get('vocab_size'): config.vocab_size = ckpt_config['vocab_size']
        # The architecture must match the checkpoint, whatever the CLI said.
        config.embedding_type = ckpt_config.get('embedding_type', 'learned')
        config.tie_weights = bool(ckpt_config.get('tie_weights', False))
        if ckpt_config.get('grid_bank'): config.grid_bank = ckpt_config['grid_bank']
        for key in ('shared_min_period', 'shared_axis_freq', 'shared_residual', 'shared_basis', 'sinusoid_min_period'):
            if key in ckpt_config: setattr(config, key, ckpt_config[key])
        if ckpt_config.get('num_epochs') and args.num_epochs is None: config.num_epochs = ckpt_config['num_epochs']

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    print("\nLoading datasets...")
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    train_dataset = FOLDataset(config.train_data_path, config.max_seq_len, pad_id=pad_id)
    val_dataset = FOLDataset(config.val_data_path, config.max_seq_len, pad_id=pad_id)
    
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_cuda)
    
    print("\nCreating model...")
    bank = GridBank.load(config.grid_bank) if config.embedding_type in ('grid_linear', 'grid_conv', 'grid_shared') else None
    model = create_model(config.vocab_size, config.model_size, embedding_type=config.embedding_type,
                         tie_weights=config.tie_weights, grid_bank=bank,
                         sinusoid_min_period=config.sinusoid_min_period,
                         shared_min_period=config.shared_min_period,
                         shared_axis_freq=config.shared_axis_freq,
                         shared_residual=config.shared_residual,
                         shared_basis=config.shared_basis)
    trainer = Trainer(model, train_loader, val_loader, config, vocab)

    start_epoch = 0
    if resume_path:
        trainer.load_checkpoint(resume_path)
        start_epoch = trainer.epoch

    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()
