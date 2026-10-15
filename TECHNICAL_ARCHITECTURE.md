# Technical Architecture: FOL Symbolic Transformer

## Executive Summary

This document describes the technical architecture of a transformer-based model for First-Order Logic (FOL) reasoning using discrete symbolic representations. The key innovation is replacing high-dimensional continuous embeddings with a compact vocabulary of 671 discrete tokens (625 numerals, 37 operator glyphs, 9 special tokens), each numeral and glyph having a 25×25 grid rendering.

---

## 1. Motivation & Hypothesis

### Current Paradigm: High-Dimensional Embeddings

Modern language models use:
- **Vocabulary**: 30K-50K tokens
- **Embeddings**: 768-4096 dimensions per token
- **Model size**: 100M-100B+ parameters
- **Compositionality**: Implicit in vector arithmetic

**Issues:**
- Dimensionality bloat (most dimensions may be noise)
- Poor interpretability (what does dimension 743 mean?)
- Inefficient for structured reasoning tasks
- Requires massive datasets to learn

### Our Approach: Discrete Symbolic Encoding

- **Vocabulary**: 671 tokens (625 numerals + 37 operator glyphs + 9 special tokens)
- **Embeddings**: 256-768 dimensions (much smaller)
- **Model size**: 500K-80M parameters (10-100× smaller)
- **Compositionality**: Explicit in symbol sequences

**Hypothesis:**  
For formal reasoning tasks, **discrete symbolic representations with explicit compositional structure** can match or exceed the performance of high-dimensional continuous embeddings while being more efficient and interpretable.

---

## 2. Symbol Vocabulary Design

### 2.1 Base-625 Numeral System

**Encoding:** Each number 0-624 maps to a unique 25×25 grid pattern

```
Number 0:   All cells empty
Number 1:   Top-left cell filled
Number 24:  Entire first row filled (25 cells)
Number 624: All cells filled
```

**Properties:**
- **Bijective mapping**: One-to-one correspondence
- **Positional encoding**: Row-major order (left→right, top→bottom)
- **Visual grounding**: Humans can recognize patterns
- **Deterministic**: No learning required for numerals

**Use cases:**
- Variable indices: `VAR 12` means "variable x₁₂"
- Predicate indices: `PRED 5` means "predicate P₅"
- Arity specification
- Position markers

### 2.2 FOL Operator Symbols

**Arithmetic glyphs** (IDs 625-635, painted in base-625): X, PLUS, MINUS, DIV, EQUALS, LPAREN, RPAREN, POW, DOT, SQRT, FACTORIAL. FOL reuses EQUALS (629), LPAREN (630), RPAREN (631) and DOT (633).

**Logical Connectives** (IDs 636-640):
```
NOT     (¬)  - Negation
AND     (∧)  - Conjunction
OR      (∨)  - Disjunction
IMPLIES (→)  - Implication
IFF     (↔)  - Biconditional
```

**Quantifiers** (IDs 641-643):
```
FORALL      (∀)  - Universal quantification
EXISTS      (∃)  - Existential quantification
EXISTS1     (∃!) - Unique existence
```

**Relations** (IDs 644-648, plus EQUALS at 629):
```
EQUALS        (=)  - Equality (ID 629)
NOT_EQUALS    (≠)  - Inequality
LESS_THAN     (<)
GREATER_THAN  (>)
LESS_EQUAL    (≤)
GREATER_EQUAL (≥)
```

**Structural Tokens** (COMMA 649, COLON 650; LPAREN 630, RPAREN 631, DOT 633):
```
LPAREN  (()  - Left parenthesis
RPAREN  ())  - Right parenthesis
COMMA   (,)  - Argument separator
COLON   (:)  - Type annotation
DOT     (.)  - Statement terminator
```

**Category Tokens** (IDs 653-657) - **CRITICAL**:
```
VAR    - Variable introducer
CONST  - Constant introducer
PRED   - Predicate introducer
FUNC   - Function introducer
SORT   - Type/sort introducer
```

**Truth Constants** (IDs 651-652):
```
TRUE   (⊤)
FALSE  (⊥)
```

**Meta-Logic** (IDs 658-661):
```
ENTAILS    (⊢)  - Syntactic entailment
MODELS     (⊨)  - Semantic entailment
DEFINE     (≔)  - Definition
EQUIVALENT (≡)  - Logical equivalence
```

### 2.3 Compositional Encoding

**Key Design Decision:** Instead of creating separate symbols for each variable/predicate/function, use **category token + numeral sequence**.

**Example 1: Variables**
```
Variable x₁:   [VAR, 1]
Variable x₁₂:  [VAR, 12]       (any index below 625 is a single numeral token)
Variable x₅₀₀: [VAR, 500]
Variable x₇₀₀: [VAR, 1, 75]    (1 × 625 + 75; digits are most-significant first)
```

Because the base is 625, composition only appears for indices ≥ 625. The generators draw indices from
0–99 by default, so in practice each index is learned as an atomic embedding; a configurable base is the
planned way to make multi-digit indices common.

**Example 2: Predicates**
```
P₁(x):      [PRED, 1, LPAREN, VAR, 1, RPAREN]
P₅(x, y):   [PRED, 5, LPAREN, VAR, 1, COMMA, VAR, 2, RPAREN]
```

**Benefits:**
- **Scalable**: Support unlimited entities without vocabulary explosion
- **Compositional**: Model learns structure naturally
- **Efficient**: Reuses numeral infrastructure
- **Interpretable**: Clear semantic structure

---

## 3. Model Architecture

### 3.1 Overview

```
Input: Symbol sequence [s₁, s₂, ..., sₙ]
  ↓
Embedding Layer: vocab_size → d_model
  ↓
Positional Encoding: Add position information
  ↓
Transformer Encoder: N layers of multi-head attention + FFN
  ↓
Output Projection: d_model → vocab_size
  ↓
Logits: Probability distribution over next symbol
```

### 3.2 Embedding Layer

```python
self.symbol_embed = nn.Embedding(vocab_size=671, embedding_dim=d_model)
```

**Design choices:**
- **No pre-trained embeddings**: Learn from scratch
- **Xavier initialization**: `N(0, 0.02)`
- **Scaling**: Multiply by `√d_model` (standard practice)

**Why smaller d_model?**
- 671 discrete tokens vs. 50K continuous tokens
- Less semantic variation to capture
- Empirically: 256-512 dimensions sufficient

### 3.2.1 Grid embeddings (`--embedding grid_linear | grid_conv`)

The learned table treats numerals as unrelated symbols. The grid arms derive each grid-backed
token's embedding from its 625-bit panel `g_t` (row-major, from `grids/grid_bank.npz`):

```
grid_linear:  e_t = LN(W g_t + b)            W: 625 × d_model, shared by all tokens
grid_conv:    e_t = LN(MLP(Conv2d(g_t)))     4 and 8 channels, two 2× max-pools, 288 → d_model
grid_shared:  e_t = LN((g_t Φ) A + b)        Φ: 625 × 36 fixed smooth features of each cell's
                                             position (sin/cos of index, row, col; ramps); A: 36 × d_model
                                             learned, so cell i's weight is Φ_i A and no parameter
                                             belongs to one cell
special:      e_t = F[t]                     learned fallback rows for tokens without a panel
```

`grid_linear` only trains column `W_i` from panels that fill cell `i`, so cells beyond the largest trained
value keep their random init and the embedding of a larger value points in a random direction.
`grid_shared` generates every cell's weight from the same `A`, so cells never filled in training get weights
consistent with their neighbours.

Measured on the magnitude task (`experiments/magnitude_*.md`): with the default smooth basis `grid_shared`
extrapolates order past the trained range but inverts comparisons inside a held-out gap. A basis of one
shared direction (`--shared-basis constant`) collapses: LayerNorm is scale-invariant, so `k·u` normalises to
the same vector for every k. The basis that works is `ramp` = [1, i/625] with the per-cell residual
(`--shared-residual`):

```
grid_ramp_res:  e_t = LN( k·a + (k²/1250)·c + b + Σ_{filled i} r_i )     a, c: d_model; r_i: 625 × d_model, zero-init
```

The direction rotates uniformly with k, so the walk through any gap and the continuation past the range are
segments of the same curve; the residuals give trained cells distinctive increments. Without the residual
the path is too flat to learn from; without the ramp the path cannot leave the trained range.

The residual has three knobs, all used in the precision study (`experiments/magnitude_precision.md`):

- `--residual-orthogonal`: `r_i` is projected off span{a, c} every forward pass, so it can give trained
  cells texture but cannot carry magnitude; magnitude then comes only from the ramp, which unseen cells
  share.
- `--residual-lr-mult M --residual-wd W`: the residual gets its own AdamW group (learning rate ×M, weight
  decay W). Its strength decides how much the comparator can lean on per-cell texture.
- `--residual-dropout p [--residual-dropout-cell]`: with probability p a training batch is embedded with
  the ramp alone (or each cell's residual row is dropped independently), so the comparator must keep a
  route through the shared part. Evaluation always uses the full weights.

For numerals `g_k` is the thermometer code (first k cells set), so `W g_k` is the cumulative sum of
the first k columns of `W`: adjacent indices get adjacent embeddings and an index absent from training
still lands between its neighbours. The LayerNorm gain is initialised to 0.02 so grid rows start at the
same scale as a learned table. The full table is recomputed each forward pass (671 rows, negligible).

With `--tie-weights` the output logits are `h · e_tᵀ + c_t` against the same table, which is what lets
an unseen numeral be predicted (its row exists because its panel exists). Checkpoints store the panels
as buffers, so `load_model_from_checkpoint` needs no bank file.

### 3.3 Positional Encoding

Standard sinusoidal encoding:

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Alternative considered:** Grid-based encoding (using 25×25 structure)
- Could encode row/column position
- Would be specific to numeral symbols
- Decided against to maintain generality

### 3.4 Transformer Encoder

**Layer structure:**
```python
for layer in layers:
    # 1. Multi-head self-attention
    x' = MultiHeadAttention(x, mask=causal_mask)
    x = LayerNorm(x + x')  # Residual connection
    
    # 2. Position-wise feedforward
    x' = FFN(x)  # Linear → GELU → Linear
    x = LayerNorm(x + x')  # Residual connection
```

**Key parameters:**
- **d_model**: 128-768 (embedding dimension)
- **n_heads**: 4-12 (attention heads)
- **d_ff**: 4×d_model (feedforward hidden dim)
- **n_layers**: 2-12 (depth)

**Causal masking:**
```python
mask[i, j] = -∞  if j > i  (prevent attending to future)
           = 0    if j ≤ i  (allow attending to past)
```

This ensures autoregressive next-token prediction.

### 3.5 Output Projection

```python
logits = Linear(d_model → vocab_size)
```

Projects hidden states to vocabulary space, giving probability distribution:

```python
probs = softmax(logits)
next_symbol = argmax(probs)  # or sample
```

### 3.6 Model Configurations

| Size  | d_model | Heads | Layers | FFN  | Params | VRAM  |
|-------|---------|-------|--------|------|--------|-------|
| Tiny  | 128     | 4     | 2      | 512  | 500K   | 2GB   |
| Small | 256     | 4     | 4      | 1024 | 2M     | 4GB   |
| Base  | 512     | 8     | 6      | 2048 | 15M    | 8GB   |
| Large | 768     | 12    | 12     | 3072 | 80M    | 16GB  |

**Parameter calculation example (Base):**
```
Embedding:  671 × 512           = 344K
Attention:  6 × (4×512×512)     = 6.3M
FFN:        6 × (2×512×2048)    = 12.6M
Output:     512 × 671           = 344K
Total:                          ≈ 15M
```

---

## 4. Training Procedure

### 4.1 Task: Next-Symbol Prediction

**Objective:** Given formula prefix, predict next symbol

```
Input:  [FORALL, VAR, 1, PRED, 5]
Target: LPAREN
```

**Loss function:**
```python
loss = CrossEntropyLoss(logits, target)
```

**Why this task?**
- Tests syntactic understanding
- Tests semantic coherence
- Autoregressive generation capability
- Easily evaluated (accuracy metrics)

### 4.2 Dataset Generation

**Formula complexity levels:**

**Level 1: Atomic** (20% of dataset)
```
P(x)
P(x) = Q(x)
```

**Level 2: Simple Compounds** (40%)
```
P(x) ∧ Q(x)
¬P(x)
P(x) → Q(x)
```

**Level 3: Quantified** (30%)
```
∀x P(x)
∃x (P(x) ∧ Q(x))
```

**Level 4: Nested** (10%)
```
∀x (P(x) → ∃y Q(x, y))
∀x ∀y (P(x) ∧ Q(y) → R(x, y))
```

**Generation algorithm:**
```python
def generate_formula(depth=0, bound_vars=[]):
    if depth >= max_depth:
        return generate_atomic(bound_vars)
    
    choice = random.choice(['quantifier', 'connective', 'atomic'])
    
    if choice == 'quantifier':
        var = new_variable()
        return [FORALL/EXISTS, var, 
                generate_formula(depth+1, bound_vars + [var])]
    
    elif choice == 'connective':
        if connective is binary:
            return [LPAREN, 
                    generate_formula(depth+1, bound_vars),
                    connective,
                    generate_formula(depth+1, bound_vars),
                    RPAREN]
        else:  # unary (NOT)
            return [NOT, generate_formula(depth+1, bound_vars)]
    
    else:
        return generate_atomic(bound_vars)
```

### 4.3 Training Configuration

**Optimizer:** AdamW
```python
lr = 1e-4
betas = (0.9, 0.98)
weight_decay = 0.01
eps = 1e-9
```

**Learning rate schedule:** Warmup + Cosine Annealing
```python
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

**Batch size:** 32 (adjust based on GPU memory)

**Gradient clipping:** Max norm = 1.0

**Epochs:** 50 (early stopping if val loss plateaus)

### 4.4 Regularization

- **Dropout:** 0.1 (in attention and FFN)
- **Weight decay:** 0.01 (L2 regularization)
- **Label smoothing:** Optional (reduces overconfidence)

### 4.5 Hardware Optimization (AMD GPU)

**ROCm backend:**
```python
device = torch.device("cuda")  # Uses ROCm on AMD
model = model.to(device)
```

**Memory optimization:**
- Mixed precision training (FP16) - may not be fully supported on all AMD GPUs
- Gradient accumulation for effective larger batches
- Gradient checkpointing for deep models

**Batch size tuning:**
```
Radeon 8060S (96GB VRAM):
- Tiny:  batch_size = 128
- Small: batch_size = 64
- Base:  batch_size = 32
- Large: batch_size = 16
```

---

## 5. Evaluation Metrics

### 5.1 Perplexity

Measure of model uncertainty:

```python
perplexity = exp(cross_entropy_loss)
```

**Interpretation:**
- Perfect model: perplexity = 1.0
- Random guessing (671 tokens): perplexity = 671
- Good model: perplexity < 20

### 5.2 Accuracy Metrics

**Top-k Accuracy:**
```python
correct = (target in top_k_predictions)
accuracy = correct_predictions / total_predictions
```

**Expected performance:**
- Top-1: 60-70% (exact match)
- Top-5: 85-92% (target in top 5)
- Top-10: 90-95% (target in top 10)

### 5.3 Generative Quality

**Syntactic validity:**
- Balanced parentheses
- Valid operator arity
- Proper quantifier-variable binding

**Semantic coherence:**
- Variables used after declaration
- Predicates applied to correct types
- Logical consistency

### 5.4 Benchmark Comparisons

Compare against:
1. **Random baseline**: Uniform distribution over 671 tokens
2. **N-gram model**: Statistical prediction (e.g., trigrams)
3. **LSTM baseline**: Recurrent architecture
4. **Standard transformer**: Same size but with continuous embeddings

---

## 6. Implementation Details

### 6.1 Data Pipeline

```python
# 1. Generate formulas
generator = FOLFormulaGenerator(vocab, config)
formulas = generator.generate_batch(n=10000)

# 2. Create training samples
dataset = NextSymbolDataset(formulas, max_seq_len=128)
# Each formula creates len(formula)-1 training samples

# 3. Create DataLoader
loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)
```

### 6.2 Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    
    for batch in train_loader:
        # Forward
        logits = model(batch['input_ids'])
        loss = criterion(logits[:, -1, :], batch['target'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    
    # Checkpointing
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch)
```

### 6.3 Generation Algorithm

**Greedy decoding:**
```python
next_token = argmax(softmax(logits))
```

**Top-k sampling:**
```python
top_k_logits = topk(logits, k=50)
probs = softmax(top_k_logits / temperature)
next_token = sample(probs)
```

**Beam search:** (for future)
```python
# Maintain k best sequences
# Expand each by vocab_size
# Keep top k by cumulative log-probability
```

---

## 7. Future Extensions

### 7.1 Inference Task (Phase 2)

**Objective:** Given premises, predict conclusion

```
Input:  P(x), P(x) → Q(x)
Output: Q(x)
```

**Architecture changes:**
- Encoder-decoder structure
- Premise encoding
- Conclusion generation
- Multi-step reasoning

### 7.2 Symbol Invention (Phase 3)

**Objective:** Model creates its own symbols

```
Model proposes: PRIME = FUNC 42
Meaning: PRIME(x) ≡ "x is prime"
```

**Implementation:**
- Symbol proposal module
- Consistency enforcement
- Compression reward
- Interpretability metrics

### 7.3 Integration with Criticality Theory

**Moderator layer:**
```python
class ModeratorLayer(nn.Module):
    def forward(self, x, criticality_state):
        # Control information flow based on criticality
        # Near-critical: max information processing
        # Subcritical: dampen activity
        # Supercritical: enhance activity
        return modulated_output
```

### 7.4 Structured Attention

Replace flat attention with **graph attention**:
```
Variables attend to quantifiers
Predicates attend to arguments
Connectives attend to operands
```

This respects FOL structure explicitly.

---

## 8. Theoretical Foundations

### 8.1 Compositionality Principle

**Frege's Principle:** Meaning of compound = function of meanings of parts

In our system:
```
Meaning([FORALL, VAR, 1, PRED, 5, LPAREN, VAR, 1, RPAREN])
  = FORALL(VAR(1), PRED(5, [VAR(1)]))
  = ∀x₁ P₅(x₁)
```

Compositionality is **explicit** in symbol sequence, not **implicit** in embeddings.

### 8.2 Symbol Grounding

**Symbol Grounding Problem:** How do symbols acquire meaning?

**Our approach:**
- **Numerals**: Grounded in visual patterns (25×25 grid)
- **Operators**: Grounded in formal semantics (FOL rules)
- **Compositional**: Grounded in structural relationships

### 8.3 Efficiency Argument

**Information-theoretic perspective:**

Number of FOL formulas expressible with n symbols:
```
Traditional: O(vocab_size^n) = O(50000^n)
Our system:  O(vocab_size^n) = O(671^n)
```

But with compositional structure:
```
Effective vocabulary: 671 base + unlimited composites
Real expressions: Still O(671^n) but with structured meaning
```

**Claim:** Compositional structure provides compression without sacrificing expressiveness.

---

## 9. Experimental Validation

### Hypothesis Testing

**H1:** Symbolic representation achieves comparable accuracy to continuous embeddings  
**Test:** Train both on same task, compare perplexity and accuracy

**H2:** Symbolic representation is more sample-efficient  
**Test:** Learning curves with varying dataset sizes

**H3:** Symbolic representation is more interpretable  
**Test:** Analyze attention patterns, symbol co-occurrence

**H4:** Model learns compositional structure  
**Test:** Evaluate on formulas with unseen combinations of known components

### Ablation Studies

1. **Remove positional encoding**: Test importance of position
2. **Remove compositional structure**: Treat all symbols as atomic
3. **Vary embedding dimension**: Find optimal d_model
4. **Vary model depth**: Test 2-layer vs 12-layer performance

---

## 10. Conclusion

This architecture demonstrates a novel approach to symbolic AI:

**Key innovations:**
1. Base-625 visual symbolic encoding
2. Compositional symbol sequences
3. Compact vocabulary (671 vs 50K+)
4. Smaller models (15M vs 100M+ params)
5. Explicit structure preservation

**Expected outcomes:**
- Competitive performance on FOL tasks
- Better interpretability
- More efficient training
- Foundation for symbol invention and meta-reasoning

**Path forward:**
- Validate on next-symbol prediction ✓
- Extend to inference task
- Enable symbol creation
- Integrate with criticality-based reasoning

This represents a step toward **discrete symbolic AI** that combines the learning power of neural networks with the interpretability and efficiency of symbolic systems.
