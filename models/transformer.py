"""
Symbolic Transformer Model for FOL Next-Symbol Prediction.

Optimized for AMD Radeon GPU (ROCm backend).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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


class SymbolicTransformer(nn.Module):
    """
    Transformer model for symbolic FOL reasoning.
    
    Architecture:
    - Symbol embedding (vocab_size → d_model)
    - Positional encoding
    - N transformer encoder layers
    - Output projection (d_model → vocab_size)
    
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
        use_causal_mask: bool = True
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
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_causal_mask = use_causal_mask
        
        # Verify d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # 1. Symbol embedding
        self.symbol_embed = nn.Embedding(vocab_size, d_model)
        
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
        
        # 4. Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Xavier/Glorot initialization for embeddings
        nn.init.normal_(self.symbol_embed.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
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
        # 1. Embed symbols
        x = self.symbol_embed(src) * math.sqrt(self.d_model)  # Scale embedding
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Generate causal mask if needed
        if self.use_causal_mask and src_mask is None:
            seq_len = src.size(1)
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # 4. Apply transformer
        x = self.transformer(x, mask=src_mask)
        
        # 5. Project to vocabulary
        logits = self.output_proj(x)
        
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


def create_model(vocab_size: int, model_size: str = 'base') -> SymbolicTransformer:
    """
    Factory function for creating models of different sizes.
    
    Args:
        vocab_size: Vocabulary size
        model_size: 'tiny', 'small', 'base', 'large'
        
    Returns:
        SymbolicTransformer model
    """
    configs = {
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
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = configs[model_size]
    model = SymbolicTransformer(
        vocab_size=vocab_size,
        **config
    )
    
    n_params = model.count_parameters()
    print(f"✓ Created {model_size} model with {n_params:,} parameters")
    
    return model


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
    
    print("\n✓ Model tests passed!")
