#!/usr/bin/env python3
"""
Advanced FOL Formula Generator 2.0
----------------------------------
Key Improvements:
1. Fixed Signatures: PRED_X and FUNC_X have permanent arities (consistency).
2. Recursive Terms: Generates f(g(x), y) structure.
3. Horn Clauses: Distinct branch for (A & B) -> C reasoning chains.
4. Tuned Noise: Low probability shadowing/vacuous quantification.
"""

import random
import sys
from pathlib import Path
import json

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset_generator import FOLFormulaGenerator, FormulaConfig, NextSymbolDataset
from utils.vocabulary import Vocabulary

class AdvancedFormulaGenerator(FOLFormulaGenerator):
    """
    Generator that enforces signature consistency and adds recursive term structures.
    """
    def __init__(self, vocab, config):
        super().__init__(vocab, config)
        
        # ---------------------------------------------------------
        # 1. FIXED SIGNATURES
        # ---------------------------------------------------------
        # PRED_1 is *always* binary. FUNC_2 is *always* unary.
        # This allows the model to learn the "identity" of symbols.
        
        self.pred_arities = {}
        for i in range(config.max_predicates):
            # 50% Binary, 40% Unary, 10% Ternary
            self.pred_arities[i] = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]
            
        self.func_arities = {}
        for i in range(config.max_functions):
            # 70% Unary (f(x)), 30% Binary (f(x,y))
            self.func_arities[i] = random.choices([1, 2], weights=[0.7, 0.3])[0]

        # ---------------------------------------------------------
        # 2. HYPERPARAMETERS
        # ---------------------------------------------------------
        self.prob_term_complexity = 0.3   # Chance a term is a function f(...) rather than var
        self.prob_shadowing = 0.05        # LOW: Chance to reuse bound variable name (confusing)
        self.prob_vacuous = 0.05          # LOW: Quantify variable but don't use it
        self.prob_horn = 0.20             # MEDIUM: Generate Horn Clause structure (implication chain)

    def generate_formula(self, depth=0, bound_vars=None):
        if bound_vars is None:
            bound_vars = []

        # Hard stop on depth
        if depth >= self.config.max_depth:
            return self.generate_atomic(bound_vars)

        # Archetypes (only at root or low depth)
        if depth == 0 and random.random() < self.prob_horn:
            return self.generate_horn_clause(depth, bound_vars)
        
        # Standard recursive generation
        # Weights: Quantifier (30%), Binary (40%), Unary (15%), Atomic (15%)
        node_type = random.choices(
            ['quantifier', 'binary', 'unary', 'atomic'], 
            weights=[0.3, 0.4, 0.15, 0.15]
        )[0]
        
        if node_type == 'quantifier' and self.config.use_quantifiers:
            return self.generate_quantified(depth, bound_vars)
        elif node_type == 'binary' and self.config.use_connectives:
            return self.generate_binary(depth, bound_vars)
        elif node_type == 'unary' and self.config.use_connectives:
            return self.generate_unary(depth, bound_vars)
        else:
            return self.generate_atomic(bound_vars)

    # ---------------------------------------------------------
    # NEW: Recursive Term Generation (The "Stack" Test)
    # ---------------------------------------------------------
    def generate_term(self, depth, bound_vars):
        """Generates either a Variable or a Function f(t1, t2)."""
        
        # Base case: Max depth reached, or random choice, or no functions available
        use_func = (random.random() < self.prob_term_complexity)
        if not use_func or depth > 2 or self.config.max_functions == 0:
            # Return Variable
            if bound_vars:
                # 80% chance to pick recently bound var, 20% random free var (if needed)
                var_idx = random.choice(bound_vars)
            else:
                var_idx = random.randint(0, self.config.max_variables - 1)
            return self.vocab.encode_compositional('VAR', var_idx)
        
        # Recursive case: Function
        func_idx = random.randint(0, self.config.max_functions - 1)
        arity = self.func_arities[func_idx]
        
        tokens = self.vocab.encode_compositional('FUNC', func_idx)
        tokens.append(self.lparen)
        
        for i in range(arity):
            tokens.extend(self.generate_term(depth + 1, bound_vars))
            if i < arity - 1:
                tokens.append(self.comma)
        
        tokens.append(self.rparen)
        return tokens

    def generate_atomic(self, bound_vars):
        """Generates PRED(t1...) or t1 = t2 using recursive terms."""
        
        # Equality (t1 = t2)
        if self.config.use_equality and random.random() < 0.25:
            tokens = self.generate_term(0, bound_vars)
            op = 'EQUALS' if random.random() < 0.8 else 'NOT_EQUALS'
            tokens.append(self.vocab.encode_label(op))
            tokens.extend(self.generate_term(0, bound_vars))
            return tokens
        
        # Predicate (PRED(t1, t2...))
        pred_idx = random.randint(0, self.config.max_predicates - 1)
        arity = self.pred_arities[pred_idx] # Use FIXED arity
        
        tokens = self.vocab.encode_compositional('PRED', pred_idx)
        tokens.append(self.lparen)
        
        for i in range(arity):
            tokens.extend(self.generate_term(0, bound_vars))
            if i < arity - 1:
                tokens.append(self.comma)
        
        tokens.append(self.rparen)
        return tokens

    def generate_quantified(self, depth, bound_vars):
        quantifier = random.choice(['FORALL', 'EXISTS'])
        
        # Shadowing check (rarely reuse an existing variable name)
        if bound_vars and random.random() < self.prob_shadowing:
            var_idx = random.choice(bound_vars)
        else:
            var_idx = random.randint(0, self.config.max_variables - 1)
            
        tokens = [self.vocab.encode_label(quantifier)]
        tokens.extend(self.vocab.encode_compositional('VAR', var_idx))
        
        # Vacuous check (rarely don't add var to scope)
        if random.random() < self.prob_vacuous:
            new_bound = bound_vars # Scope doesn't change
        else:
            new_bound = bound_vars + [var_idx]
            
        tokens.extend(self.generate_formula(depth + 1, new_bound))
        return tokens

    def generate_horn_clause(self, depth, bound_vars):
        """Generates (Atomic & Atomic) -> Atomic"""
        num_conditions = random.randint(1, 3)
        tokens = []
        
        if num_conditions > 1: tokens.append(self.lparen)
            
        for i in range(num_conditions):
            tokens.extend(self.generate_atomic(bound_vars))
            if i < num_conditions - 1:
                tokens.append(self.vocab.encode_label('AND'))
        
        if num_conditions > 1: tokens.append(self.rparen)
            
        tokens.append(self.vocab.encode_label('IMPLIES'))
        tokens.extend(self.generate_atomic(bound_vars))
        return tokens
        
    # Standard wrappers
    def generate_binary(self, depth, bound_vars):
        connective = random.choice(['AND', 'OR', 'IMPLIES', 'IFF'])
        tokens = [self.lparen]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.vocab.encode_label(connective))
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        tokens.append(self.rparen)
        return tokens

    def generate_unary(self, depth, bound_vars):
        tokens = [self.vocab.encode_label('NOT')]
        tokens.extend(self.generate_formula(depth + 1, bound_vars))
        return tokens

def generate_diverse_dataset():
    print("\n" + "="*60)
    print("GENERATING ADVANCED DIVERSE FOL DATASET (2.0)")
    print("="*60)
    
    vocab = Vocabulary("unified_vocabulary.json")
    
    # Config: Enable Functions!
    config = FormulaConfig(
        max_depth=5,
        max_variables=8,
        max_predicates=10,
        max_functions=4, # >0 enables function generation
        use_quantifiers=True,
        use_connectives=True,
        use_equality=True
    )
    
    generator = AdvancedFormulaGenerator(vocab, config)
    
    # Print signatures to confirm consistency
    print(f"Fixed Predicate Arities: {generator.pred_arities}")
    print(f"Fixed Function Arities:  {generator.func_arities}")
    
    # -----------------------------------------------------
    # Generate Splits (Standard Logic)
    # -----------------------------------------------------
    output_dir = Path("datasets/fol_diverse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': 10000,
        'val': 1000,
        'test': 1000
    }
    
    for split_name, n_samples in splits.items():
        print(f"\nGenerating {split_name} ({n_samples} formulas)...")
        formulas = generator.generate_batch(n_samples, complexity=3)
        
        # Convert to NextSymbolDataset
        dataset = NextSymbolDataset(formulas, vocab, max_seq_len=128)
        dataset.save(str(output_dir / f"{split_name}.json"))
        
    print(f"\n✓ Saved all datasets to {output_dir}")

if __name__ == "__main__":
    generate_diverse_dataset()
    