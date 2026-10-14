"""
Symbolic Transformer Model for FOL Next-Symbol Prediction.

Embedding options (``embedding_type``):

- ``learned``      a plain nn.Embedding table (the original model).
- ``grid_linear``  every grid-backed token is embedded by one shared linear
                   projection of its 625-bit base-625 panel. Numerals use the
                   row-major count (thermometer) code, so panel k is a subset
                   of panel k+1 and embeddings move along a path as k grows.
- ``grid_conv``    a small 2D conv encoder over the 25x25 panel.

Tokens without a panel (MESSAGE_START, PAD, ...) use a learned fallback row.
With ``tie_weights`` the output logits are the dot product with the same
embedding table, so a numeral the model never trained on can still be
predicted from its panel.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python models/transformer.py` as well as `from models.transformer import ...`
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.grid_bank import GRID_CELLS, GRID_N, DEFAULT_BANK_PATH, GridBank  # noqa: E402

EMBEDDING_TYPES = ("learned", "grid_linear", "grid_conv")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    Standard implementation from "Attention is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class ConvGridEncoder(nn.Module):
    """Small 2D conv encoder: (N, 625) bits -> (N, d_model)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 25 -> 12 -> 6
        self.proj = nn.Linear(8 * 6 * 6, d_model)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        x = bits.view(-1, 1, GRID_N, GRID_N)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.proj(x.flatten(1))


class GridEmbedding(nn.Module):
    """Embed tokens from their base-625 panels.

    The full (vocab_size, d_model) table is recomputed from the panels on every
    forward pass (cheap: a few hundred rows), which keeps weight tying trivial
    and lets ``noise_flip`` corrupt the panels at evaluation time to measure
    robustness. Rows whose token has no panel come from ``fallback``.
    """

    def __init__(self, grid_bank: GridBank, d_model: int, kind: str = "grid_linear", init_std: float = 0.02):
        super().__init__()
        if kind not in ("grid_linear", "grid_conv"):
            raise ValueError(f"Unknown grid embedding kind: {kind}")
        self.kind = kind
        self.d_model = d_model
        vocab_size = grid_bank.vocab_size
        self.register_buffer("grids", torch.tensor(grid_bank.grids, dtype=torch.float32))
        self.register_buffer("has_grid", torch.tensor(grid_bank.has_grid, dtype=torch.bool))
        self.encoder = nn.Linear(GRID_CELLS, d_model) if kind == "grid_linear" else ConvGridEncoder(d_model)
        # LayerNorm keeps every panel embedding at the same scale regardless of
        # how many cells are filled; its gain starts at init_std so grid rows
        # match the scale of a freshly initialised learned table.
        self.norm = nn.LayerNorm(d_model)
        nn.init.constant_(self.norm.weight, init_std)
        self.fallback = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.fallback.weight, mean=0.0, std=init_std)
        self.noise_flip: float = 0.0  # evaluation-time bit-flip probability

    @property
    def num_embeddings(self) -> int:
        return self.fallback.num_embeddings

    def panels(self) -> torch.Tensor:
        g = self.grids
        if self.noise_flip > 0.0:
            flip = (torch.rand_like(g) < self.noise_flip) & self.has_grid.unsqueeze(1)
            g = torch.where(flip, 1.0 - g, g)
        return g

    def table(self) -> torch.Tensor:
        """Full embedding table, shape (vocab_size, d_model)."""
        encoded = self.norm(self.encoder(self.panels()))
        return torch.where(self.has_grid.unsqueeze(1), encoded, self.fallback.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.table())


class SymbolicTransformer(nn.Module):
    """
    Transformer model for symbolic FOL reasoning.
    
    Architecture:
    - Symbol embedding (vocab_size → d_model): learned table or grid encoder
    - Positional encoding
    - N transformer encoder layers
    - Output projection (d_model → vocab_size), optionally tied to the embedding
    
    Key design choices:
    - Smaller d_model than typical (256-512) since we have discrete symbols
    - Causal attention mask for autoregressive next-token prediction
    - Layer normalization for stable training
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        embedding_type: str = "learned",
        tie_weights: bool = False,
        grid_bank: Optional[GridBank] = None,
    ):
        """
        Args:
            vocab_size: Size of symbol vocabulary
            d_model: Embedding dimension
            n_heads: Number of attention heads (must divide d_model)
            n_layers: Number of transformer blocks
            d_ff: Feedforward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_causal_mask: Use causal attention for autoregressive generation
            embedding_type: 'learned', 'grid_linear' or 'grid_conv'
            tie_weights: Compute output logits against the embedding table
            grid_bank: Panels for grid embeddings (required unless 'learned')
        """
        super().__init__()
        
        if embedding_type not in EMBEDDING_TYPES:
            raise ValueError(f"embedding_type must be one of {EMBEDDING_TYPES}, got {embedding_type!r}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_causal_mask = use_causal_mask
        self.embedding_type = embedding_type
        self.tie_weights = tie_weights
        
        # Verify d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # 1. Symbol embedding
        if embedding_type == "learned":
            self.symbol_embed = nn.Embedding(vocab_size, d_model)
        else:
            if grid_bank is None:
                raise ValueError("grid embeddings need a GridBank (pass grid_bank=GridBank.load())")
            if grid_bank.vocab_size != vocab_size:
                raise ValueError(f"grid bank has {grid_bank.vocab_size} rows, vocab has {vocab_size}")
            self.symbol_embed = GridEmbedding(grid_bank, d_model, kind=embedding_type)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # GELU works well for language models
            batch_first=True,   # [batch, seq, feature] format
            norm_first=False    # Post-norm (more stable)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 4. Output projection (or tied output with a per-token bias)
        if tie_weights:
            self.output_proj = None
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        if isinstance(self.symbol_embed, nn.Embedding):
            nn.init.normal_(self.symbol_embed.weight, mean=0.0, std=0.02)
        
        if self.output_proj is not None:
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
    
    def embedding_table(self) -> torch.Tensor:
        """The (vocab_size, d_model) table currently used to embed tokens."""
        if isinstance(self.symbol_embed, nn.Embedding):
            return self.symbol_embed.weight
        return self.symbol_embed.table()
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal attention mask.
        Prevents attending to future positions.
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask of shape [sz, sz] with -inf for masked positions
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Input token IDs [batch_size, seq_len]
            src_mask: Attention mask [seq_len, seq_len] (optional)
            
        Returns:
            Logits for next token [batch_size, seq_len, vocab_size]
        """
        # 1. Embed symbols (one table lookup, shared with the tied output)
        table = self.embedding_table()
        x = F.embedding(src, table) * math.sqrt(self.d_model)  # Scale embedding
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Generate causal mask if needed
        if self.use_causal_mask and src_mask is None:
            seq_len = src.size(1)
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 4. Apply transformer
        x = self.transformer(x, mask=src_mask)
        
        # 5. Project to vocabulary
        if self.output_proj is not None:
            logits = self.output_proj(x)
        else:
            logits = x @ table.t() + self.output_bias
        
        return logits
    
    def generate(
        self, 
        prompt: torch.Tensor, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            prompt: Initial token sequence [batch_size, prompt_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample from top-k tokens only
            
        Returns:
            Generated sequence [batch_size, prompt_len + max_new_tokens]
        """
        self.eval()
        
        generated = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[:, -1, None]] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MODEL_CONFIGS = {
    'tiny': {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'dropout': 0.1
    },
    'small': {
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1
    },
    'base': {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1
    },
    'large': {
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'dropout': 0.1
    }
}


def create_model(
    vocab_size: int,
    model_size: str = 'base',
    embedding_type: str = 'learned',
    tie_weights: bool = False,
    grid_bank: Optional[Union[GridBank, str, Path]] = None,
    verbose: bool = True,
) -> SymbolicTransformer:
    """
    Factory function for creating models of different sizes.
    
    Args:
        vocab_size: Vocabulary size
        model_size: 'tiny', 'small', 'base', 'large'
        embedding_type: 'learned', 'grid_linear' or 'grid_conv'
        tie_weights: Tie the output layer to the embedding table
        grid_bank: GridBank, path to a bank .npz, or None to load the default
            bank (only used for grid embeddings)
        
    Returns:
        SymbolicTransformer model
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")
    
    bank: Optional[GridBank] = None
    if embedding_type != 'learned':
        if isinstance(grid_bank, GridBank):
            bank = grid_bank
        else:
            bank = GridBank.load(grid_bank if grid_bank is not None else DEFAULT_BANK_PATH)
    
    config = MODEL_CONFIGS[model_size]
    model = SymbolicTransformer(
        vocab_size=vocab_size,
        embedding_type=embedding_type,
        tie_weights=tie_weights,
        grid_bank=bank,
        **config
    )
    
    if verbose:
        n_params = model.count_parameters()
        extra = f", {embedding_type}{', tied' if tie_weights else ''}"
        print(f"✓ Created {model_size} model with {n_params:,} parameters{extra}")
    
    return model


def load_model_from_checkpoint(
    checkpoint: Union[str, Path, dict],
    vocab_size: int,
    device: Union[str, torch.device] = 'cpu',
    verbose: bool = True,
) -> Tuple[SymbolicTransformer, dict]:
    """Rebuild a model from a checkpoint written by training/train.py.

    Works for checkpoints from before the grid embeddings existed (they carry
    no ``embedding_type``/``tie_weights`` keys and default to the original
    learned, untied model). Grid panels are restored from the checkpoint's own
    buffers, so no grid bank file is needed at load time.
    """
    device = torch.device(device)
    if not isinstance(checkpoint, dict):
        checkpoint = torch.load(checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    embedding_type = config.get('embedding_type', 'learned')
    tie_weights = bool(config.get('tie_weights', False))
    bank = GridBank.empty(vocab_size) if embedding_type != 'learned' else None
    model = create_model(vocab_size, config.get('model_size', 'tiny'), embedding_type=embedding_type,
                         tie_weights=tie_weights, grid_bank=bank, verbose=verbose)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


if __name__ == "__main__":
    # Test model creation
    print("Testing Symbolic Transformer model...\n")
    
    vocab_size = 671  # matches unified_vocabulary.json (625 numerals + 37 symbols + 9 special)
    
    for size in ['tiny', 'small', 'base', 'large']:
        print(f"\n{size.upper()} model:")
        model = create_model(vocab_size, size)
        
        # Test forward pass
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Parameters:   {model.count_parameters():,}")
        
        # Test generation
        prompt = torch.randint(0, vocab_size, (1, 5))
        generated = model.generate(prompt, max_new_tokens=10)
        print(f"  Generation:   {prompt.shape} → {generated.shape}")
    
    print("\nGrid embedding variants (tiny, tied):")
    for emb in ('grid_linear', 'grid_conv'):
        model = create_model(vocab_size, 'tiny', embedding_type=emb, tie_weights=True)
        logits = model(torch.randint(0, vocab_size, (2, 16)))
        print(f"  {emb}: logits {tuple(logits.shape)}")
    
    print("\n✓ Model tests passed!")
