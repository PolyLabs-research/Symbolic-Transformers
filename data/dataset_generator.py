"""
FOL Formula Dataset Generator (with MESSAGE boundaries)
Generates training data for next-symbol prediction on FOL formulas.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.vocabulary import Vocabulary


@dataclass
class FormulaConfig:
    """Configuration for formula generation."""
    max_depth: int = 3
    max_variables: int = 100
    max_predicates: int = 100
    max_functions: int = 3
    use_quantifiers: bool = True
    use_connectives: bool = True
    use_equality: bool = True
    use_arithmetic: bool = False  # For future extension
    use_message_boundaries: bool = True  # NEW: Control MESSAGE_START/END


class FOLFormulaGenerator:
    """
    Generate random well-formed FOL formulas for training.
    
    Generates formulas with increasing complexity:
    - Level 1: Atomic predicates P(x)
    - Level 2: Simple connectives P(x) ∧ Q(x)
    - Level 3: Quantifiers ∀x P(x)
    - Level 4: Nested structures ∀x(P(x) → Q(x))
    """
    
    def __init__(self, vocab: Vocabulary, config: FormulaConfig):
        self.vocab = vocab
        self.config = config
        
        # Check if MESSAGE_START/MESSAGE_END are available
        self.has_boundaries = (
            vocab.has_special_token('MESSAGE_START') and 
            vocab.has_special_token('MESSAGE_END')
        )
        
        if config.use_message_boundaries and not self.has_boundaries:
            print("⚠ Warning: MESSAGE_START/MESSAGE_END not in vocabulary, disabling boundaries")
            config.use_message_boundaries = False
        
        # Get token IDs for generation
        self.connectives = ['NOT', 'AND', 'OR', 'IMPLIES', 'IFF']
        self.quantifiers = ['FORALL', 'EXISTS']
        self.relations = ['EQUALS', 'NOT_EQUALS']
        
        self.connective_ids = [vocab.encode_label(c) for c in self.connectives]
        self.quantifier_ids = [vocab.encode_label(q) for q in self.quantifiers]
        
        # Structural tokens
        self.lparen = vocab.encode_label('LPAREN')
        self.rparen = vocab.encode_label('RPAREN')
        self.comma = vocab.encode_label('COMMA')
        
        # Category tokens
        self.var_token = vocab.compositional_tokens['VAR']
        self.pred_token = vocab.compositional_tokens['PRED']
        self.func_token = vocab.compositional_tokens.get('FUNC')
    
    def generate_variable(self, var_idx: Optional[int] = None) -> List[int]:
        """Generate variable token sequence VAR n."""
        if var_idx is None:
            var_idx = random.randint(0, self.config.max_variables - 1)
        return self.vocab.encode_compositional('VAR', var_idx)
    
    def generate_predicate(
        self, 
        pred_idx: Optional[int] = None,
        arity: int = 1,
        var_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate predicate token sequence PRED n (VAR m1, VAR m2, ...).
        
        Args:
            pred_idx: Predicate index (random if None)
            arity: Number of arguments
            var_indices: Specific variable indices (random if None)
        """
        if pred_idx is None:
            pred_idx = random.randint(0, self.config.max_predicates - 1)
        
        tokens = self.vocab.encode_compositional('PRED', pred_idx)
        tokens.append(self.lparen)
        
        if var_indices is None:
            var_indices = [random.randint(0, self.config.max_variables - 1) 
                          for _ in range(arity)]
        
        for i, var_idx in enumerate(var_indices):
            tokens.extend(self.generate_variable(var_idx))
            if i < len(var_indices) - 1:
                tokens.append(self.comma)
        
        tokens.append(self.rparen)
        return tokens
    
    def generate_atomic(self, bound_vars: Optional[List[int]] = None) -> List[int]:
        """
        Generate atomic formula (predicate or equality).
        
        Args:
            bound_vars: List of variable indices currently in scope
        """
        if self.config.use_equality and random.random() < 0.3:
            # Generate equality: VAR n = VAR m
            if bound_vars and len(bound_vars) >= 2:
                var1, var2 = random.sample(bound_vars, 2)
            else:
                var1 = random.randint(0, self.config.max_variables - 1)
                var2 = random.randint(0, self.config.max_variables - 1)
            
            tokens = self.generate_variable(var1)
            tokens.append(self.vocab.encode_label('EQUALS'))
            tokens.extend(self.generate_variable(var2))
            return tokens
        else:
            # Generate predicate
            arity = random.randint(1, 2)  # Mostly unary or binary
            
            if bound_vars:
                var_indices = [random.choice(bound_vars) for _ in range(arity)]
            else:
                var_indices = None
            
            return self.generate_predicate(arity=arity, var_indices=var_indices)
    
    def generate_formula(
        self, 
        depth: int = 0, 
        bound_vars: Optional[List[int]] = None
    ) -> List[int]:
        """
        Recursively generate formula with given maximum depth.
        
        Args:
            depth: Current nesting depth
            bound_vars: Variables currently bound by quantifiers
        """
        if bound_vars is None:
            bound_vars = []
        
        # Base case: generate atomic formula
        if depth >= self.config.max_depth or random.random() < 0.3:
            return self.generate_atomic(bound_vars)
        
        # Recursive cases
        formula_type = random.choice(['quantifier', 'connective'])
        
        if formula_type == 'quantifier' and self.config.use_quantifiers:
            # Generate quantified formula: ∀x φ or ∃x φ
            quantifier = random.choice(self.quantifiers)
            var_idx = random.randint(0, self.config.max_variables - 1)
            
            tokens = [self.vocab.encode_label(quantifier)]
            tokens.extend(self.generate_variable(var_idx))
            
            # Generate body with this variable bound
            new_bound = bound_vars + [var_idx]
            body = self.generate_formula(depth + 1, new_bound)
            
            tokens.extend(body)
            return tokens
        
        elif formula_type == 'connective' and self.config.use_connectives:
            # Generate compound formula with connective
            connective = random.choice(self.connectives)
            
            if connective == 'NOT':
                # Unary: ¬φ
                tokens = [self.vocab.encode_label('NOT')]
                subformula = self.generate_formula(depth + 1, bound_vars)
                tokens.extend(subformula)
                return tokens
            else:
                # Binary: (φ ∧ ψ), (φ → ψ), etc.
                tokens = [self.lparen]
                left = self.generate_formula(depth + 1, bound_vars)
                tokens.extend(left)
                tokens.append(self.vocab.encode_label(connective))
                right = self.generate_formula(depth + 1, bound_vars)
                tokens.extend(right)
                tokens.append(self.rparen)
                return tokens
        
        # Fallback to atomic
        return self.generate_atomic(bound_vars)
    
    def generate_batch(self, n_samples: int, complexity: int = 2) -> List[List[int]]:
        """
        Generate batch of formulas.
        
        Args:
            n_samples: Number of formulas to generate
            complexity: 1-5 scale (affects depth)
        """
        # Map complexity to depth
        depth_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        max_depth = depth_map.get(complexity, 2)
        
        old_depth = self.config.max_depth
        self.config.max_depth = max_depth
        
        formulas = []
        for _ in range(n_samples):
            formula = self.generate_formula()
            formulas.append(formula)
        
        self.config.max_depth = old_depth
        return formulas


class NextSymbolDataset:
    """
    Dataset for next-symbol prediction task with MESSAGE boundaries.
    
    Given a formula prefix, predict the next token.
    Example: [MESSAGE_START, FORALL, VAR, 1, PRED] → next token
    """
    
    def __init__(
        self, 
        formulas: List[List[int]],
        vocab: Vocabulary,
        max_seq_len: int = 512,
        use_boundaries: bool = True
    ):
        """
        Create dataset from formula token sequences.
        
        Args:
            formulas: List of token sequences (without boundaries)
            vocab: Vocabulary instance
            max_seq_len: Maximum sequence length
            use_boundaries: Whether to wrap formulas with MESSAGE_START/END
        """
        self.formulas = formulas
        self.max_seq_len = max_seq_len
        self.samples = []
        
        # Check if boundaries are available
        has_boundaries = (
            vocab.has_special_token('MESSAGE_START') and 
            vocab.has_special_token('MESSAGE_END')
        )
        
        if use_boundaries and not has_boundaries:
            print("⚠ Warning: MESSAGE_START/MESSAGE_END not available, proceeding without boundaries")
            use_boundaries = False
        
        skipped_count = 0
        
        # Create training samples: (context, target) pairs
        for formula in formulas:
            # Wrap with message boundaries if enabled
            if use_boundaries:
                full_formula = [vocab.message_start] + formula + [vocab.message_end]
            else:
                full_formula = formula
            
            # Skip if too long
            if len(full_formula) > max_seq_len:
                skipped_count += 1
                continue
            
            # Create samples at each position INCLUDING MESSAGE_END as target
            for i in range(1, len(full_formula)):
                context = full_formula[:i]
                target = full_formula[i]
                
                self.samples.append({
                    'context': context,
                    'target': target,
                    'formula_len': len(formula)
                })
        
        if skipped_count > 0:
            print(f"⚠ Warning: Skipped {skipped_count} formulas exceeding max_seq_len ({max_seq_len})")
        
        # Report boundary usage
        if use_boundaries and self.samples:
            boundary_samples = sum(1 for s in self.samples 
                                 if s['target'] in [vocab.message_start, vocab.message_end])
            print(f"✓ Generated {len(self.samples)} samples ({boundary_samples} boundary targets)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def save(self, output_path: str):
        """Save dataset to JSON."""
        with open(output_path, 'w') as f:
            json.dump({
                'num_formulas': len(self.formulas),
                'num_samples': len(self.samples),
                'max_seq_len': self.max_seq_len,
                'samples': self.samples
            }, f)
        print(f"✓ Saved {len(self.samples)} samples to {output_path}")
    
    @staticmethod
    def load(input_path: str) -> 'NextSymbolDataset':
        """Load dataset from JSON."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct formulas from samples
        # (This is approximate - we just need something for the constructor)
        formulas = []
        dataset = NextSymbolDataset.__new__(NextSymbolDataset)
        dataset.formulas = formulas
        dataset.max_seq_len = data['max_seq_len']
        dataset.samples = data['samples']
        
        print(f"✓ Loaded {len(dataset.samples)} samples from {input_path}")
        return dataset


def generate_training_data(
    vocab_path: str,
    output_dir: str,
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 1000,
    complexity_distribution: Optional[List[Tuple[int, float]]] = None,
    use_boundaries: bool = True,
    max_variables: int = FormulaConfig.max_variables,
    max_predicates: int = FormulaConfig.max_predicates,
    max_depth: int = 3,
    max_seq_len: int = 128,
    seed: Optional[int] = 42,
):
    """
    Generate complete training/validation/test datasets.
    
    Args:
        vocab_path: Path to unified vocabulary
        output_dir: Directory to save datasets
        n_train: Number of training formulas
        n_val: Number of validation formulas
        n_test: Number of test formulas
        complexity_distribution: List of (complexity, weight) tuples
        use_boundaries: Whether to wrap formulas with MESSAGE_START/END
        max_variables: Variable indices are drawn from 0..max_variables-1
        max_predicates: Predicate indices are drawn from 0..max_predicates-1
        max_depth: Maximum nesting depth of generated formulas
        max_seq_len: Formulas longer than this (with boundaries) are skipped
        seed: Random seed for reproducible datasets (None = unseeded)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        random.seed(seed)
    
    print("="*60)
    print("FOL DATASET GENERATION (BASIC with MESSAGE BOUNDARIES)")
    print("="*60)
    
    # Load vocabulary
    vocab = Vocabulary(vocab_path)
    
    # Check boundary availability
    has_boundaries = (
        vocab.has_special_token('MESSAGE_START') and 
        vocab.has_special_token('MESSAGE_END')
    )
    
    if use_boundaries and not has_boundaries:
        print("⚠ Warning: MESSAGE_START/MESSAGE_END not in vocabulary")
        print("   Run update_vocabulary_complete.py first!")
        print("   Proceeding without boundaries...")
        use_boundaries = False
    elif use_boundaries:
        print(f"✓ Using MESSAGE_START (ID {vocab.message_start}) and MESSAGE_END (ID {vocab.message_end})")
    
    # Setup generator
    config = FormulaConfig(
        max_depth=max_depth,
        max_variables=max_variables,
        max_predicates=max_predicates,
        use_quantifiers=True,
        use_connectives=True,
        use_equality=True,
        use_message_boundaries=use_boundaries
    )
    generator = FOLFormulaGenerator(vocab, config)
    
    # Default complexity distribution
    if complexity_distribution is None:
        complexity_distribution = [
            (1, 0.2),  # 20% simple
            (2, 0.4),  # 40% medium
            (3, 0.3),  # 30% complex
            (4, 0.1),  # 10% very complex
        ]
    
    def generate_split(n_formulas: int, split_name: str) -> NextSymbolDataset:
        """Generate one data split."""
        print(f"\nGenerating {split_name} set ({n_formulas} formulas)...")
        
        formulas = []
        for complexity, weight in complexity_distribution:
            n_for_complexity = int(n_formulas * weight)
            batch = generator.generate_batch(n_for_complexity, complexity)
            formulas.extend(batch)
            print(f"  Complexity {complexity}: {n_for_complexity} formulas")
        
        dataset = NextSymbolDataset(formulas, vocab, max_seq_len=max_seq_len, use_boundaries=use_boundaries)
        
        output_path = output_dir / f"{split_name}.json"
        dataset.save(str(output_path))
        
        return dataset
    
    # Generate splits
    train_dataset = generate_split(n_train, 'train')
    val_dataset = generate_split(n_val, 'val')
    test_dataset = generate_split(n_test, 'test')
    
    # Save metadata
    metadata = {
        'generator': 'basic_with_boundaries_v2',
        'vocab_path': vocab_path,
        'vocab_size': vocab.vocab_size,
        'uses_boundaries': use_boundaries,
        'seed': seed,
        'max_seq_len': max_seq_len,
        'complexity_distribution': complexity_distribution,
        'config': {
            'max_depth': config.max_depth,
            'max_variables': config.max_variables,
            'max_predicates': config.max_predicates,
        },
        'splits': {
            'train': {'num_formulas': n_train, 'num_samples': len(train_dataset)},
            'val': {'num_formulas': n_val, 'num_samples': len(val_dataset)},
            'test': {'num_formulas': n_test, 'num_samples': len(test_dataset)},
        }
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ Dataset generation complete!")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    if use_boundaries:
        print(f"✓ Message boundaries: ENABLED")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FOL next-symbol datasets")
    parser.add_argument("--vocab", default="unified_vocabulary.json")
    parser.add_argument("--out", default="datasets/fol_next_symbol")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-variables", type=int, default=FormulaConfig.max_variables)
    parser.add_argument("--max-predicates", type=int, default=FormulaConfig.max_predicates)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-boundaries", action="store_true", help="Do not wrap formulas in MESSAGE_START/END")
    args = parser.parse_args()

    generate_training_data(
        vocab_path=args.vocab,
        output_dir=args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        complexity_distribution=[
            (1, 0.2),  # 20% simple
            (2, 0.4),  # 40% medium
            (3, 0.3),  # 30% complex
            (4, 0.1),  # 10% very complex
        ],
        use_boundaries=not args.no_boundaries,
        max_variables=args.max_variables,
        max_predicates=args.max_predicates,
        max_depth=args.max_depth,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
