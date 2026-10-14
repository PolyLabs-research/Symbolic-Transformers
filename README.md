# Symbolic Transformers

**Tiny Transformers for First-Order Logic — Edge-Deployable, Interpretable, Data-Efficient**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripptytrip/Symbolic-Transformers/blob/main/train_symbolic_transformer.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

![Screencast](Screencast%20from%2021-12-25%2022_22_36.gif)

## Who is this for?


- **Researchers** testing how far tiny transformers can go on formal reasoning tasks
- **TinyML/edge developers** who need reasoning, or pattern matching models in MBs, not GBs
- **Educators** building interactive logic tutors or puzzle tools
- **Neuro-symbolic builders** experimenting with custom symbolic alphabets

## What this is NOT (yet)

- ❌ Not a full theorem prover, its very much a toy in its presant state
- ❌ Not a verified proof pipeline (no external prover integration yet)
- ❌ Not trained on natural language — this is formal-symbol sequence learning
- ❌ Not claiming to replace grammars for pure syntax checking

Its a personal project and something I'm exploring.

## Why not just write a grammar?

A grammar can tell you what strings are *syntactically valid* with 100% accuracy. This project asks a different question: 

**what can a tiny model learn from data beyond validity?**

- Frequency priors (which constructions are common vs rare)
- Soft semantic constraints (which predicates tend to co-occur)
- Generalisation under distribution shift (unseen variable indices, longer formulas)
- Foundation for inference tasks where grammars can't help (Phase 4+)

---

## 🎯 Overview

**Symbolic Transformers** achieve competitive performance on logical reasoning tasks using only **566K parameters** and **671 discrete tokens** — 200× more parameter-efficient than traditional language models.

### Key Innovation

Instead of high-dimensional continuous embeddings (768-4096 dims × 50K tokens), we use:

- **Compositional category tokens**: `VAR 12 → [VAR, 12]`, `VAR 700 → [VAR, 1, 75]` — a category token followed by base-625 digits, so a finite vocabulary names unlimited variables
- **Compact vocabulary**: 671 tokens (625 numerals + 37 operator glyphs + 9 special tokens)
- **Symbolic grounding**: Each numeral has a corresponding 25×25 grid pattern via [base-625](https://github.com/tripptytrip/base-625)

**Result:** A **2.2 MB model** that runs on edge devices and learns FOL syntax rules with near-perfect accuracy.

### How base-625 fits in

The companion repo [**base-625**](https://github.com/tripptytrip/base-625) provides a 25×25 glyph/grid system for visual symbol representations.

**Current status:** The original model trains on **discrete token IDs** (learned embeddings) and uses the grids only for rendering and inspection. The `grid_linear` and `grid_conv` embedding arms (see [Making the grids matter](#-making-the-grids-matter)) feed the 25×25 panels into the model directly, so the question "do the grids help?" is now a flag on the trainer rather than a future plan.

---

## 🚀 Quick Start

### Option 1: Train in Google Colab (Recommended)

No local setup required — GPU training in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tripptytrip/Symbolic-Transformers/blob/main/train_symbolic_transformer.ipynb)

### Option 2: Local Installation

```bash
git clone https://github.com/tripptytrip/Symbolic-Transformers.git
cd Symbolic-Transformers
pip install torch numpy tqdm rich
python quickstart.py
```

### Option 3: Train Directly

```bash
python data/dataset_generator.py                    # Generate data
python training/train.py --model-size tiny --num-epochs 100  # Train
python evaluate_model.py                            # Evaluate
```

---

## 📊 Results

### Model Performance

| Model | Parameters | Size | Val Loss | Top-1 Acc | Top-5 Acc |
|-------|-----------|------|----------|-----------|-----------|
| **Tiny** | 566K | 2.2 MB | 0.88 | ~44% | ~75% |
| **Small** | 3.5M | 14 MB | 0.75* | ~52%* | ~82%* |
| **Base** | 19.6M | 78 MB | 0.65* | ~58%* | ~86%* |

*Projected based on scaling experiments. Base model training in progress (epoch 1: val loss 0.96).

> **Note on the shipped checkpoint.** `checkpoints/best_model.pt` was trained before two fixes landed: the
> dataset only used variable/predicate indices 0–4, and the loss ignored every `NUM_000` target. The numbers
> above come from that run. Retrain (Colab or `python training/train.py`) after regenerating data to get
> figures that reflect the current code.

### Baselines & Context

To understand what 44% accuracy means:

| Method | Top-1 Accuracy | Notes |
|--------|---------------|-------|
| Random | 0.15% | 1/671 tokens |
| Unigram frequency | ~8% | Always predict most common token |
| Bigram model | ~18% | Conditional on previous token |
| Grammar-constrained random | ~25-30% | Random among valid next tokens |
| **Symbolic Transformer (tiny)** | **~44%** | Learns soft preferences beyond grammar |

The model's value is in learning *which* valid continuations are likely, not just *what* is valid.

### What the Model Learns

The model achieves **99.6% confidence** on hard syntax rules:

```
Input: FORALL
Prediction: VAR (99.6%)  ← Learned that ∀ must bind a variable

Input: FORALL VAR
Prediction: [numeral] (uniform ~2% each)  ← Appropriate uncertainty over indices

Input: FORALL VAR 1
Prediction: PRED (75%), FORALL (10%), EXISTS (5%)  ← Soft preferences among valid options
```

### Key Finding: Data Quality > Model Size

Our experiments revealed that **more data beats more epochs**:

- Tiny model (566K params) + 14K samples → overfits after ~170 epochs
- Best checkpoint: epoch 167 (val loss 0.879)
- Continued training to epoch 700 made val loss *worse* (1.18)
- 10× more data (144K samples) → base model hits val loss 0.96 in epoch 1

**Lesson:** In the low-parameter regime, data efficiency matters more than architecture scaling.

---

## 🏗️ Architecture

### Compositional Token Design

The key insight: instead of infinite embeddings for every variable/predicate, use compositional sequences:

```
Traditional:  VAR_0, VAR_1, VAR_2, ... VAR_999  (1000 embeddings)
Ours:         VAR + base-625 digits              (626 embeddings, unlimited range)

VAR 0:   [VAR, 0]
VAR 42:  [VAR, 42]         (one base-625 digit)
VAR 700: [VAR, 1, 75]      (1 × 625 + 75)
PRED 5:  [PRED, 5]
```

This is reusable for any symbolic domain with indexed entities. Note that every index below 625 is a
single numeral token, so multi-digit composition only appears from index 625 upwards; the default
generators draw indices from 0–99, so the model currently learns indices as atomic tokens. Making the
base configurable (for example base 25, so `VAR 42 → [VAR, 1, 17]`) is the planned way to exercise
composition in ordinary training data.

### Model Configurations

| Config | d_model | Layers | Heads | FF dim | Parameters |
|--------|---------|--------|-------|--------|------------|
| tiny   | 128     | 2      | 4     | 512    | 566K |
| small  | 256     | 4      | 4     | 1024   | 3.5M |
| base   | 512     | 6      | 8     | 2048   | 19.6M |

### Vocabulary (671 tokens)

- **Numerals (0-624):** IDs 0–624, one per base-625 digit; each has a 25×25 grid rendering
- **Arithmetic glyphs (625–635):** X, PLUS, MINUS, DIV, EQUALS, LPAREN, RPAREN, POW, DOT, SQRT, FACTORIAL (painted in base-625; FOL reuses EQUALS and the parentheses)
- **Connectives (636–640):** NOT (¬), AND (∧), OR (∨), IMPLIES (→), IFF (↔)
- **Quantifiers (641–643):** FORALL (∀), EXISTS (∃), EXISTS1 (∃!)
- **Relations (644–648):** NOT_EQUALS (≠), LESS_THAN (<), GREATER_THAN (>), LESS_EQUAL (≤), GREATER_EQUAL (≥)
- **Structural (649–650):** COMMA, COLON
- **Truth and meta (651–661):** TRUE, FALSE, VAR, CONST, PRED, FUNC, SORT, ENTAILS, MODELS, DEFINE, EQUIVALENT
- **Special (662–670):** MESSAGE_START, MESSAGE_END, PAD, UNK, and the reasoning tokens SEP, QUERY, ANSWER, END, UNKNOWN

Every training context starts with `MESSAGE_START`, so the demos and benchmarks prepend it automatically.

---

## 🧩 Making the grids matter

A numeral's panel is the **count encoding**: numeral *k* is the first *k* cells filled in row-major order.
That is a thermometer code, so panel *k* is a subset of panel *k+1*, which a learned embedding table knows
nothing about. The grid arms replace the table for every grid-backed token:

| `--embedding` | numeral / glyph tokens | special tokens | notes |
|---|---|---|---|
| `learned` | `nn.Embedding` row | `nn.Embedding` row | original model |
| `grid_linear` | one shared `Linear(625 → d_model)` over the panel bits | learned fallback row | embeddings move along a path as *k* grows |
| `grid_conv` | small 2D conv encoder over the 25×25 panel | learned fallback row | sees shape, not just count |
| `sinusoid` | numerals: fixed sin/cos features of *k* through `Linear(64 → d_model)`; glyphs: learned row | learned fallback row | control: an ordinal code with no 2D structure |

All arms can use `--tie-weights`, which computes output logits against the same embedding table. With a
grid embedding that means an index the model never trained on can still be **predicted** from its panel,
which is what copying a bound variable into a predicate body requires.

Panels come from `grids/grid_bank.npz` (625 generated numeral panels + the 37 painted glyphs from `out/symbols`,
verified against the PNGs by `python scripts/build_grid_bank.py`). Checkpoints carry their panels, so no bank
file is needed to load one.

### The experiment

```bash
# reduced CPU run (~1 h): 2k formulas, indices 0-99 with 60-79 held out, 100-199 never seen
python scripts/grid_experiment.py --out-dir runs/grid_small --n-train 2000 --epochs 8

# full run (GPU / Colab)
python scripts/grid_experiment.py --out-dir runs/grid_full --n-train 10000 --epochs 30 --seeds 1 2 3
```

The driver generates the data (`data/dataset_generator.py --index-range 0-99 --holdout 60-79 --far-range 100-199`),
trains the three arms on identical data and seed, and evaluates each with `evaluate_grid.py`, which reports
top-1 accuracy per split and per target category:

- `structure` / `after_numeral`: operator targets, and operator targets right after a numeral (does an unseen index derail the syntax?)
- `copy`: numeral targets that reuse a variable bound earlier in the context, the one place the model must emit a specific index
- `fresh`: numeral targets with no earlier occurrence (chance by construction)
- embedding geometry: does distance between numeral embeddings track |i − j|, and is a numeral's nearest neighbour an adjacent index
- panel noise: grid arms re-evaluated with 1% and 5% of panel bits flipped

Results land in `runs/<name>/results.md`. The reduced run (`experiments/grid_small_seed1.md`) showed that
FOL syntax does not need the grids: an untied learned table generalises structure to unseen indices just as
well, because no syntactic decision depends on which index is larger. It also showed that tying the output
layer to a *learned* table breaks on unseen tokens while grid rows are immune.

### The magnitude task

The first task where the numeral embedding has to carry magnitude:

```
MESSAGE_START  NUM_a  LESS_THAN  NUM_b  ANSWER   ->   TRUE | FALSE
```

Operands are bare numerals, operators are the vocabulary's relation tokens, and the answer is one token.
Training values come from 0-499 with 200-299 held out; test pairs are disjoint from training pairs;
`test_holdout` has at least one operand from the held-out band and `test_far` at least one from 500-624.

```bash
python scripts/magnitude_experiment.py --out-dir runs/magnitude --seeds 1 2 3 --epochs 30
```

`evaluate_magnitude.py` reports accuracy by |a − b| (adjacent values are hardest) and by how many
operands were unseen, plus two probes of the embedding table itself: whether a linear readout of *k*
fitted on training values transfers to held-out and far values, and how well the first principal
component tracks *k*. Results: `experiments/magnitude_*.md`.

Manual equivalents for the FOL experiment:

```bash
python training/train.py --embedding grid_linear --tie-weights --train-data runs/grid_small/data/train.json \
    --val-data runs/grid_small/data/val.json --checkpoint-dir runs/mine --num-epochs 8
python evaluate_grid.py --checkpoint runs/mine/best_model.pt \
    --splits runs/grid_small/data/test.json runs/grid_small/data/test_holdout.json runs/grid_small/data/test_far.json
```

## 📚 Training Guide

### Recommended Configurations

| Goal | Model | Formulas | Epochs | Time |
|------|-------|----------|--------|------|
| Quick test | tiny | 500 | 30 | ~5 min |
| Standard | tiny | 1,000 | 100-150 | ~30 min |
| Best results | small | 5,000+ | 150-200 | 2+ hours |
| Research scale | base | 10,000+ | 100-200 | 6+ hours |

### ⚠️ Avoiding Overfitting

**Warning signs:**
- Train loss keeps dropping, val loss plateaus
- Val loss starts **increasing**
- Train/val gap exceeds 0.2

**Rule of thumb:**

| Data Size | Max Epochs (tiny) | Max Epochs (small) |
|-----------|-------------------|-------------------|
| 1K formulas | 100-150 | 80-120 |
| 5K formulas | 150-200 | 120-150 |
| 10K formulas | 200-300 | 150-200 |

### Interpreting Loss

- **Val Loss > 2.0**: Learning basic patterns
- **Val Loss 1.0-2.0**: Learning syntax rules
- **Val Loss < 1.0**: Good FOL structure understanding
- **Val Loss < 0.85**: Excellent (approaching capacity)

---

## 🎮 Interactive UI

Explore the model in real-time with the Gradio demos (`pip install gradio`):

```bash
python demo.py                  # live next-token explorer
python demo_with_benchmarks.py  # explorer + structural benchmark suite in the browser
```

Type a prefix such as `FORALL VAR 1` and the top next-token probabilities update as you type.
`MESSAGE_START` is prepended automatically; numerals must be 0–624.

The same benchmark suite runs from the command line:

```bash
python symbolic_benchmarks.py --checkpoint checkpoints/best_model.pt --vocab unified_vocabulary.json
```

---

## 🚀 Deployment

### Model Sizes

| Model | FP32 | INT8 (quantized) |
|-------|------|------------------|
| Tiny  | 2.2 MB | ~0.6 MB |
| Small | 14 MB | ~3.5 MB |
| Base  | 78 MB | ~20 MB |

### Edge Targets

The tiny model (2.2 MB / 0.6 MB quantized) runs on:
- ✅ Raspberry Pi Zero
- ✅ ESP32 (with INT8 quantization)
- ✅ Web browser (ONNX.js / WebAssembly)
- ✅ Mobile devices (iOS/Android)
- ✅ Any system with Python

### Export to ONNX

```python
torch.onnx.export(
    model, sample_input, "symbolic_transformer.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 1: 'seq_len'}}
)
```

---

## 📁 Project Structure

```
Symbolic-Transformers/
├── train_symbolic_transformer.ipynb  # 🚀 Colab notebook
├── quickstart.py                     # Interactive setup
├── evaluate_model.py                 # Evaluation script (top-k accuracy, samples)
├── demo.py                           # Gradio next-token explorer
├── demo_with_benchmarks.py           # Gradio explorer + benchmark suite
├── symbolic_benchmarks.py            # Structural benchmark definitions + CLI runner
├── evaluate_grid.py                  # Split-aware evaluation for the FOL grid experiment
├── evaluate_magnitude.py             # Magnitude task evaluation + embedding magnitude probes
├── unified_vocabulary.json           # 671-token vocabulary
├── grids/grid_bank.npz               # 25×25 panels per token (built by scripts/build_grid_bank.py)
│
├── data/
│   ├── magnitude_generator.py        # NUM a <op> NUM b -> TRUE/FALSE with held-out value bands
│   ├── dataset_generator.py          # Basic FOL formula generation (seeded, CLI)
│   ├── advanced_generator.py         # Functions, fixed arities, structural diversity
│   ├── generate_diverse_formulas.py  # Horn clauses + recursive terms
│   └── transitivity_generator.py     # Phase 4 reasoning chains (generator only)
├── models/
│   └── transformer.py                # Model architecture (learned / grid_linear / grid_conv embeddings)
├── training/
│   └── train.py                      # Training loop
├── evaluation/
│   └── evaluate.py                   # Metrics
├── utils/
│   ├── vocabulary.py                 # Encoding/decoding
│   └── grid_bank.py                  # Panel bank: thermometer numerals + painted glyphs
├── scripts/
│   ├── update_vocabulary_complete.py # Adds special tokens to the vocabulary
│   ├── build_grid_bank.py            # Rebuilds grids/grid_bank.npz from the PNGs
│   ├── grid_experiment.py            # FOL: generate → train arms → evaluate → results.md
│   └── magnitude_experiment.py       # Magnitude task, multi-seed, mean ± std
├── tests/                            # pytest suite (python -m pytest tests)
├── out/                              # 25×25 glyph PNGs + decode tables (from base-625)
│
├── datasets/                         # Generated data
└── checkpoints/                      # Saved models
```

---

## 📈 Roadmap

- [x] **Phase 1:** Next-symbol prediction ✓
- [x] **Phase 2:** Interactive UI ✓
- [x] **Phase 3:** Colab training notebook ✓
- [ ] **Phase 4:** Inference task (premise → conclusion)
- [ ] **Phase 5:** Multi-step proof generation
- [ ] **Phase 6:** Symbol invention (learn new operators)

**Phase 4** is where this project moves beyond what grammars can do — learning to predict valid conclusions from premises.

---

## 🔗 Related Projects

- [**base-625**](https://github.com/tripptytrip/base-625) — The 25×25 glyph/grid symbol system used for visual grounding
- [**Quantum Rat**](https://github.com/tripptytrip/quantum-rat) — Computational neuroethology platform (related research)

---

## 📝 Citation

```bibtex
@software{symbolic_transformers_2025,
  author = {Michael Kerr},
  title = {Symbolic Transformers: Tiny Models for First-Order Logic 
           with Compositional Token Design},
  year = {2025},
  url = {https://github.com/tripptytrip/Symbolic-Transformers},
  note = {566K parameter model achieving 44\% next-token accuracy on FOL}
}
```

---

## 🤝 Contributing

Contributions welcome! High-impact areas:

- **Baselines:** Implement n-gram, LSTM, grammar-constrained baselines
- **Inference tasks:** Premise → conclusion dataset and training
- **Benchmarks:** Generalisation tests (longer formulas, held-out symbols)
- **Domains:** Arithmetic, set theory, type theory vocabularies
- **Deployment:** ONNX export, quantization, browser demo

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 📞 Contact

- **Issues:** [GitHub Issues](https://github.com/tripptytrip/Symbolic-Transformers/issues)
- **X/Twitter:** [@_trippitytrip_](https://x.com/_trippitytrip)

---

**Built for efficient, interpretable, edge-deployable AI reasoning** 🧠
