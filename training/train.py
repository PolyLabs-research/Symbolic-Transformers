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
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
import sys
import argparse
import re

import numpy as np

# Ensure we can import from parent directory
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import EMBEDDING_TYPES, create_model
from utils.grid_bank import DEFAULT_BANK_PATH, GridBank
from utils.fused import SymbolFuser
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
    # Periodic checkpoints (checkpoint_epoch_N.pt) to keep; 0 keeps all. best_model.pt is never pruned.
    keep_checkpoints: int = 2
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
    shared_residual_orthogonal: bool = False
    shared_residual_dropout: float = 0.0
    shared_residual_dropout_cell: bool = False
    # Only numeral rows go through the panel encoder; operators and other glyphs use learned rows.
    grid_numerals_only: bool = False
    # xval embedding: numeral k -> ((k - xval_center) / xval_scale) * one learned direction.
    xval_center: float = 249.5
    xval_scale: float = 144.3
    # Warm start: model weights loaded from this checkpoint before training (optimizer and schedule fresh).
    init_from: Optional[str] = None
    # Item datasets (data/eval_generator.py): which written format to train on.
    seq_format: str = "direct"
    # One token per symbol (utils/fused.py): CONST 5 is read and predicted as one token.
    fuse_symbols: bool = False
    # Separate optimiser group for the grid_shared per-cell residual.
    residual_lr_mult: float = 1.0
    residual_weight_decay: Optional[float] = None
    # (start, end) as fractions of num_epochs: the residual is scaled by 1 up to
    # start, linearly down to 0 at end, and 0 after. best_model.pt is then chosen
    # among residual-free epochs only.
    residual_anneal: Optional[Tuple[float, float]] = None
    # With a floor in (0, 1) the anneal is geometric: the scale falls from 1 to
    # floor log-linearly between start and end (then 0), so most of the anneal
    # is spent where the residual is small rather than large.
    residual_anneal_floor: Optional[float] = None
    # Keep the scale at the floor after end instead of dropping it to zero.
    residual_anneal_hold: bool = False
    # Restart the learning-rate schedule when the anneal window ends: linear
    # warm-up over this many steps back to the base rate, then cosine decay to
    # 10% over the remaining steps. Gives the comparator a fresh rate to find
    # a route without the residual instead of the tail of the cosine.
    residual_anneal_rewarm: Optional[int] = None
    # General form: (fraction of num_epochs, scale) breakpoints. The scale is the
    # first value before the first breakpoint, the last value after the last one,
    # and geometric in between (a step where either side is zero). Mutually
    # exclusive with residual_anneal.
    residual_anneal_stages: Optional[List[Tuple[float, float]]] = None
    seed: int = 42


def anneal_end_epoch(num_epochs: int, end: float) -> int:
    """First 1-based epoch at which the anneal has reached its final value."""
    return int(math.ceil(end * num_epochs - 1e-9))


def parse_anneal_stages(text: str) -> List[Tuple[float, float]]:
    """'0.2:1,0.47:0.01,0.9:1e-6' -> [(0.2, 1.0), (0.47, 0.01), (0.9, 1e-6)], validated."""
    stages = []
    for item in text.split(","):
        t, s = item.split(":")
        stages.append((float(t), float(s)))
    if len(stages) < 2:
        raise ValueError("--residual-anneal-stages needs at least two breakpoints")
    for (t, s) in stages:
        if not (0.0 <= t <= 1.0) or not (0.0 <= s <= 1.0):
            raise ValueError("--residual-anneal-stages: fractions and scales must lie in [0, 1]")
    if any(b[0] < a[0] for a, b in zip(stages, stages[1:])):
        raise ValueError("--residual-anneal-stages: fractions must be non-decreasing")
    return stages


def staged_residual_scale_at(epoch: int, num_epochs: int, stages: List[Tuple[float, float]]) -> float:
    """Residual multiplier for 1-based ``epoch`` under breakpoint ``stages``."""
    pts = [(t * num_epochs, s) for t, s in stages]
    if epoch <= pts[0][0]:
        return float(pts[0][1])
    for (e0, s0), (e1, s1) in zip(pts, pts[1:]):
        if epoch <= e1:
            if e1 <= e0:
                return float(s1)
            if s0 <= 0.0 or s1 <= 0.0:
                return float(s0) if epoch < e1 else float(s1)   # a step at the later breakpoint
            return float(s0 * (s1 / s0) ** ((epoch - e0) / (e1 - e0)))
    return float(pts[-1][1])


def residual_schedule_end(config) -> Optional[int]:
    """First epoch at which the residual schedule holds its final value, or None without one."""
    if config.residual_anneal_stages:
        return anneal_end_epoch(config.num_epochs, config.residual_anneal_stages[-1][0])
    if config.residual_anneal is not None:
        return anneal_end_epoch(config.num_epochs, config.residual_anneal[1])
    return None


def residual_schedule_scale(config, epoch: int) -> float:
    if config.residual_anneal_stages:
        return staged_residual_scale_at(epoch, config.num_epochs, config.residual_anneal_stages)
    if config.residual_anneal is not None:
        return residual_scale_at(epoch, config.num_epochs, config.residual_anneal[0], config.residual_anneal[1],
                                 config.residual_anneal_floor, config.residual_anneal_hold)
    return 1.0


def describe_residual_schedule(config) -> str:
    n = config.num_epochs
    if config.residual_anneal_stages:
        return "Residual anneal (stages): " + " -> ".join(f"{s:g} at epoch {t * n:.0f}" for t, s in config.residual_anneal_stages)
    a = config.residual_anneal
    shape = f"geometric to {config.residual_anneal_floor:g}" if config.residual_anneal_floor else "linear"
    final = (f"held at {config.residual_anneal_floor:g}" if config.residual_anneal_hold and config.residual_anneal_floor
             else "0")
    return f"Residual anneal ({shape}): scale 1 until epoch {a[0] * n:.0f}, {final} from epoch {a[1] * n:.0f}"


def make_lr_lambda(total_steps: int, warmup: int, restart_step: Optional[int] = None,
                   rewarm: Optional[int] = None):
    """Linear warm-up then cosine decay to 10%; with ``restart_step`` the same
    shape starts again from that step with a ``rewarm``-step warm-up."""
    total_steps = max(1, total_steps)
    warmup = max(1, warmup)

    def segment(step: int, warm: int, length: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, length - warm))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda(step: int) -> float:
        if restart_step is not None and step >= restart_step:
            return segment(step - restart_step, max(1, rewarm or 1), total_steps - restart_step)
        return segment(step, warmup, total_steps)
    return lr_lambda


def residual_scale_at(epoch: int, num_epochs: int, start: float, end: float,
                      floor: Optional[float] = None, hold: bool = False) -> float:
    """Residual multiplier for 1-based ``epoch`` under a (start, end) anneal.

    1 while epoch <= start * num_epochs; from epoch >= end * num_epochs, 0 (or
    ``floor`` with ``hold``); in between linear, or geometric from 1 down to
    ``floor`` when one is given. start == end is a step.
    """
    e0, e1 = start * num_epochs, end * num_epochs
    final = floor if (hold and floor) else 0.0
    if e1 <= e0:
        return 1.0 if epoch < e1 else final
    if epoch <= e0:
        return 1.0
    if epoch >= e1:
        return final
    t = (epoch - e0) / (e1 - e0)
    if floor is None:
        return float(1.0 - t)
    return float(floor ** t)


def _sample_context(sample: Dict) -> List[int]:
    """Return the (unpadded) context token list of a dataset sample."""
    if 'input_ids' in sample:
        return sample['input_ids']
    if 'context' in sample:
        return sample['context']
    return sample.get('input', [])


def trim_padding(input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Drop the padding columns past the batch's longest real context.

    Padding is on the right and attention is causal, so the logits at every real
    position are unchanged; the batch just costs less than one padded to the file's
    longest context.
    """
    return input_ids[:, :int(lengths.max())]


class SequenceDataset(Dataset):
    """Whole sequences with loss on a suffix: one forward pass per item.

    Items come from data/eval_generator.py: ``<fmt>_seq`` is the full token sequence and
    ``<fmt>_from`` the index of its first target token. Each item gives ``input_ids`` (the sequence
    without its last token, right-padded) and ``labels`` (the sequence shifted left by one, with -100
    before the target suffix and on padding), so the loss covers exactly the target tokens. An item may
    instead list its target positions in ``<fmt>_targets``. With a ``fuser`` the items are read one token
    per symbol, and their target positions move with them.
    """

    def __init__(self, data: Dict, fmt: str, pad_id: int, source: str = "", fuser: Optional[SymbolFuser] = None):
        self.items = data["items"]
        if fuser is not None:
            self.items = [fuser.fuse_item(it, (fmt,)) for it in self.items]
        self.fmt, self.pad_id = fmt, pad_id
        self.max_len = max(len(it[f"{fmt}_seq"]) for it in self.items) - 1
        print(f"✓ Loaded {len(self.items)} {fmt} items from {source} (inputs padded to {self.max_len}, "
              f"pad_id={pad_id})")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        seq, start = it[f"{self.fmt}_seq"], it[f"{self.fmt}_from"]
        n = len(seq) - 1
        targets = it.get(f"{self.fmt}_targets")
        if targets is not None:  # loss only on these positions (e.g. several answers in one item)
            want = set(targets)
            labels = [seq[j + 1] if j + 1 in want else -100 for j in range(n)]
        else:
            labels = [-100] * (start - 1) + seq[start:]
        pad = self.max_len - n
        return {
            "input_ids": torch.tensor(seq[:-1] + [self.pad_id] * pad, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad, dtype=torch.long),
            "length": torch.tensor(n, dtype=torch.long),
        }


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar-length items, so little of each batch is padding.

    Each epoch the items are shuffled, cut into chunks of ``bucket`` batches, sorted by length within a
    chunk and batched; the batch order is then shuffled. Without shuffling, batches follow length order.
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, bucket: int = 50):
        self.lengths, self.batch_size, self.shuffle, self.bucket = lengths, batch_size, shuffle, bucket

    def _batches(self) -> List[List[int]]:
        n = len(self.lengths)
        order = torch.randperm(n).tolist() if self.shuffle else sorted(range(n), key=self.lengths.__getitem__)
        chunk = self.batch_size * self.bucket
        batches = []
        for s in range(0, n, chunk):
            part = sorted(order[s:s + chunk], key=self.lengths.__getitem__)
            batches += [part[i:i + self.batch_size] for i in range(0, len(part), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        n, chunk = len(self.lengths), self.batch_size * self.bucket
        return sum(-(-min(chunk, n - s) // self.batch_size) for s in range(0, n, chunk))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool):
    """Item datasets are batched by length; next-symbol sample files as before."""
    if isinstance(dataset, SequenceDataset):
        lengths = [len(it[f"{dataset.fmt}_seq"]) for it in dataset.items]
        return DataLoader(dataset, batch_sampler=LengthBucketSampler(lengths, batch_size, shuffle),
                          num_workers=num_workers, pin_memory=pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory)


def open_dataset(path: str, max_seq_len: int, pad_id: int, seq_format: str = "direct",
                 fuser: Optional[SymbolFuser] = None) -> Dataset:
    """A next-symbol sample file (FOLDataset) or an item file from data/eval_generator.py."""
    with open(path) as f:
        data = json.load(f)
    if "items" in data:
        return SequenceDataset(data, seq_format, pad_id, path, fuser)
    return FOLDataset(path, max_seq_len, pad_id=pad_id, fuser=fuser)


class FOLDataset(Dataset):
    """PyTorch Dataset for FOL next-symbol prediction.

    Each item is a right-padded context plus its true length, so the trainer
    reads the logits at the last *real* token. Contexts are padded with
    ``pad_id`` (the vocabulary's PAD token, not numeral 0) and only as far as
    the longest context in the file, capped at ``max_seq_len``. A context that
    is still too long keeps its *last* tokens, because the target follows the
    end of the context.
    """
    
    def __init__(self, data_path: str, max_seq_len: int = 512, pad_id: int = 0,
                 fuser: Optional[SymbolFuser] = None):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples'] if fuser is None else fuser.fuse_samples(data['samples'])
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
        
        # Setup optimizer. The grid_shared per-cell residual (if any) gets its own
        # group so it can learn faster and decay harder than the rest: trained
        # cells sharpen their increment, untrained cells shrink back to the
        # smooth part.
        residual = getattr(getattr(getattr(model, 'symbol_embed', None), 'encoder', None), 'residual', None)
        if residual is not None and (config.residual_lr_mult != 1.0 or config.residual_weight_decay is not None):
            others = [p for p in model.parameters() if p is not residual]
            groups = [
                {'params': others},
                {'params': [residual], 'lr': config.learning_rate * config.residual_lr_mult,
                 'weight_decay': config.weight_decay if config.residual_weight_decay is None
                 else config.residual_weight_decay},
            ]
            print(f"✓ Residual group: lr x{config.residual_lr_mult}, weight_decay {groups[1]['weight_decay']}")
        else:
            groups = [{'params': list(model.parameters())}]
        self.optimizer = optim.AdamW(
            groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler: linear warmup, then cosine decay to 10%,
        # restarted at the end of the residual anneal when --residual-anneal-rewarm is set.
        total_steps = max(1, len(train_loader) * config.num_epochs)
        restart_step = None
        if residual_schedule_end(config) is not None and config.residual_anneal_rewarm is not None:
            restart_step = (residual_schedule_end(config) - 1) * len(train_loader)
        self.restart_step = restart_step
        self.lr_lambda = make_lr_lambda(total_steps, config.warmup_steps, restart_step, config.residual_anneal_rewarm)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        # No ignore_index: targets are never padding, and token 0 is the
        # numeral NUM_000 (ignoring it silently dropped ~6% of training targets).
        self.criterion = nn.CrossEntropyLoss()
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    
    def batch_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-entropy for one batch: at the last real token (next-symbol samples) or on every
        labelled position (SequenceDataset items)."""
        lengths = batch['length'].to(self.device)
        input_ids = trim_padding(batch['input_ids'].to(self.device), lengths)
        logits = self.model(input_ids)
        if 'labels' in batch:
            labels = batch['labels'][:, :input_ids.size(1)].to(self.device)
            return self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        # Logits at the actual end of each context (lengths are 1-based)
        rows = torch.arange(input_ids.size(0), device=self.device)
        return self.criterion(logits[rows, lengths - 1, :], batch['target'].to(self.device))

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.batch_loss(batch)
            
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
                loss = self.batch_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def prune_checkpoints(self):
        """Keep only the newest ``keep_checkpoints`` periodic checkpoints (resume uses the newest)."""
        keep = self.config.keep_checkpoints
        if keep <= 0:
            return
        def epoch_of(path: Path) -> int:
            match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
            return int(match.group(1)) if match else -1
        periodic = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=epoch_of)
        for old in periodic[:-keep]:
            old.unlink(missing_ok=True)

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
        self.prune_checkpoints()
        
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
        
        anneal_end = residual_schedule_end(self.config)
        encoder = getattr(getattr(self.model, 'symbol_embed', None), 'encoder', None)
        if anneal_end is not None and getattr(encoder, 'residual', None) is None:
            raise ValueError("--residual-anneal needs a grid_shared model with --shared-residual")
        if anneal_end is not None:
            print(f"{describe_residual_schedule(self.config)}; best_model.pt from epoch {anneal_end} on only")
            if self.restart_step is not None:
                print(f"Learning-rate re-warm-up at step {self.restart_step} (epoch {anneal_end}): "
                      f"{self.config.residual_anneal_rewarm} warm-up steps, then cosine")

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch + 1
            start_time = time.time()
            print(f"Epoch {self.epoch}/{self.config.num_epochs}")
            scale = 1.0
            if anneal_end is not None:
                scale = residual_schedule_scale(self.config, self.epoch)
                encoder.set_residual_scale(scale)
                print(f"  Residual scale: {scale:.3g}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            self.history.setdefault('residual_scale', []).append(scale)
            
            anneal_done = anneal_end is None or self.epoch >= anneal_end
            is_best = val_loss < self.best_val_loss and anneal_done
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
    parser.add_argument("--seq-format", default="direct", choices=["direct", "scratch"],
                        help="For item datasets (data/eval_generator.py): train on the direct answer or the scratchpad")
    parser.add_argument("--fuse-symbols", action="store_true",
                        help="One token per symbol: read and predict CONST 5, VAR 7, ... as single tokens (utils/fused.py)")
    parser.add_argument("--init-from", default=None, metavar="CKPT",
                        help="warm start: load model weights from CKPT (strict; optimizer, schedule and epochs start fresh)")
    parser.add_argument("--xval-center", type=float, default=249.5,
                        help="xval embedding: numeral value standardisation centre (default: mean of 0..499)")
    parser.add_argument("--xval-scale", type=float, default=144.3,
                        help="xval embedding: standardisation scale (default: std of 0..499)")
    parser.add_argument("--grid-numerals-only", action="store_true",
                        help="grid embeddings: only numeral tokens use panels; operators, answers and other glyphs get learned rows")
    parser.add_argument("--shared-residual", action="store_true",
                        help="grid_shared: add a zero-initialised per-cell residual weight")
    parser.add_argument("--residual-orthogonal", action="store_true",
                        help="grid_shared residual: project it orthogonal to the generated directions (cannot carry magnitude)")
    parser.add_argument("--save-every", type=int, default=5,
                        help="write a periodic checkpoint every N epochs (best_model.pt is written on every improvement)")
    parser.add_argument("--keep-checkpoints", type=int, default=2,
                        help="periodic checkpoints to keep per run (0 = keep all; best_model.pt is always kept)")
    parser.add_argument("--residual-dropout", type=float, default=0.0,
                        help="grid_shared residual: probability of training a batch without the residual "
                             "(the model must keep a magnitude route through the shared part)")
    parser.add_argument("--residual-dropout-cell", action="store_true",
                        help="grid_shared residual: drop each cell's residual row independently instead")
    parser.add_argument("--residual-anneal", type=float, nargs=2, default=None, metavar=("START", "END"),
                        help="grid_shared residual: scale it by 1 until START x epochs, linearly to 0 at END x epochs, "
                             "0 after (fractions in [0, 1]); best_model.pt is then chosen among residual-free epochs")
    parser.add_argument("--residual-anneal-floor", type=float, default=None, metavar="F",
                        help="with --residual-anneal: anneal geometrically from 1 to F (0 < F < 1) instead of linearly to 0")
    parser.add_argument("--residual-anneal-hold", action="store_true",
                        help="with --residual-anneal-floor: keep the scale at F after END instead of dropping it to 0")
    parser.add_argument("--residual-anneal-stages", type=str, default=None, metavar="T:S,T:S,...",
                        help="grid_shared residual: breakpoints 'fraction:scale,...' (geometric between them, a step "
                             "where a side is 0); replaces --residual-anneal/--floor/--hold")
    parser.add_argument("--residual-anneal-rewarm", type=int, default=None, metavar="N",
                        help="with --residual-anneal: restart the learning-rate schedule when the anneal ends "
                             "(N warm-up steps back to the base rate, then cosine decay to 10%%)")
    parser.add_argument("--residual-lr-mult", type=float, default=1.0,
                        help="grid_shared residual: learning-rate multiplier for its own optimiser group")
    parser.add_argument("--residual-wd", type=float, default=None,
                        help="grid_shared residual: weight decay for its own group (default: global)")
    parser.add_argument("--shared-basis", default="full", choices=["full", "constant", "ramp"],
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
    config.grid_numerals_only = args.grid_numerals_only
    config.xval_center = args.xval_center
    config.xval_scale = args.xval_scale
    config.init_from = args.init_from
    config.seq_format = args.seq_format
    config.fuse_symbols = args.fuse_symbols
    config.shared_basis = args.shared_basis
    config.shared_residual_orthogonal = args.residual_orthogonal
    config.shared_residual_dropout = args.residual_dropout
    config.keep_checkpoints = args.keep_checkpoints
    config.shared_residual_dropout_cell = args.residual_dropout_cell
    config.residual_lr_mult = args.residual_lr_mult
    config.residual_weight_decay = args.residual_wd
    if args.residual_anneal is not None:
        start, end = args.residual_anneal
        if not (0.0 <= start <= end <= 1.0):
            parser.error("--residual-anneal START END needs 0 <= START <= END <= 1")
        config.residual_anneal = (start, end)
    if args.residual_anneal_floor is not None:
        if args.residual_anneal is None:
            parser.error("--residual-anneal-floor needs --residual-anneal")
        if not (0.0 < args.residual_anneal_floor < 1.0):
            parser.error("--residual-anneal-floor F needs 0 < F < 1")
        config.residual_anneal_floor = args.residual_anneal_floor
    if args.residual_anneal_hold:
        if args.residual_anneal_floor is None:
            parser.error("--residual-anneal-hold needs --residual-anneal-floor")
        config.residual_anneal_hold = True
    if args.residual_anneal_stages is not None:
        if args.residual_anneal is not None or args.residual_anneal_floor is not None or args.residual_anneal_hold:
            parser.error("--residual-anneal-stages replaces --residual-anneal, --residual-anneal-floor and --residual-anneal-hold")
        try:
            config.residual_anneal_stages = parse_anneal_stages(args.residual_anneal_stages)
        except ValueError as exc:
            parser.error(str(exc))
    if args.residual_anneal_rewarm is not None:
        if args.residual_anneal is None and args.residual_anneal_stages is None:
            parser.error("--residual-anneal-rewarm needs --residual-anneal or --residual-anneal-stages")
        if args.residual_anneal_rewarm < 1:
            parser.error("--residual-anneal-rewarm N needs N >= 1")
        config.residual_anneal_rewarm = args.residual_anneal_rewarm
    config.save_every = args.save_every
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
        for key in ('shared_min_period', 'shared_axis_freq', 'shared_residual', 'shared_basis',
                    'shared_residual_orthogonal', 'shared_residual_dropout',
                    'shared_residual_dropout_cell', 'sinusoid_min_period', 'residual_anneal',
                    'residual_anneal_floor', 'residual_anneal_hold', 'residual_anneal_rewarm',
                    'residual_anneal_stages', 'grid_numerals_only', 'xval_center', 'xval_scale', 'seq_format',
                    'fuse_symbols'):
            if key in ckpt_config: setattr(config, key, ckpt_config[key])
        if ckpt_config.get('num_epochs') and args.num_epochs is None: config.num_epochs = ckpt_config['num_epochs']

    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    print("\nLoading datasets...")
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    fuser = SymbolFuser(vocab) if config.fuse_symbols else None
    train_dataset = open_dataset(config.train_data_path, config.max_seq_len, pad_id, config.seq_format, fuser)
    val_dataset = open_dataset(config.val_data_path, config.max_seq_len, pad_id, config.seq_format, fuser)
    
    use_cuda = torch.cuda.is_available()
    train_loader = make_loader(train_dataset, config.batch_size, True, args.num_workers, use_cuda)
    val_loader = make_loader(val_dataset, config.batch_size, False, args.num_workers, use_cuda)
    
    print("\nCreating model...")
    bank = GridBank.load(config.grid_bank) if config.embedding_type in ('grid_linear', 'grid_conv', 'grid_shared') else None
    model = create_model(config.vocab_size, config.model_size, embedding_type=config.embedding_type,
                         tie_weights=config.tie_weights, grid_bank=bank,
                         sinusoid_min_period=config.sinusoid_min_period,
                         shared_min_period=config.shared_min_period,
                         shared_axis_freq=config.shared_axis_freq,
                         shared_residual=config.shared_residual,
                         shared_basis=config.shared_basis,
                         shared_residual_orthogonal=config.shared_residual_orthogonal,
                         shared_residual_dropout=config.shared_residual_dropout,
                         shared_residual_dropout_cell=config.shared_residual_dropout_cell,
                         grid_numerals_only=config.grid_numerals_only,
                         xval_center=config.xval_center, xval_scale=config.xval_scale,
                         fused_categories=fuser.categories if fuser is not None else None)
    if config.init_from and not resume_path:
        init_ck = torch.load(config.init_from, map_location="cpu")
        model.load_state_dict(init_ck["model_state_dict"])
        print(f"✓ Warm start: weights from {config.init_from} (epoch {init_ck.get('epoch')})")
    trainer = Trainer(model, train_loader, val_loader, config, vocab)

    start_epoch = 0
    if resume_path:
        trainer.load_checkpoint(resume_path)
        start_epoch = trainer.epoch

    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()
