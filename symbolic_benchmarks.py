"""
Symbolic Transformer Structural Benchmarks

These tests verify that the model has learned FOL syntax rules,
not just pattern matching. They form the foundation for reasoning capabilities.

Usage:
    from symbolic_benchmarks import BENCHMARKS, BenchmarkRunner
    
    runner = BenchmarkRunner(model, vocab)
    report = runner.run_all()
    print(runner.generate_report(report))
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import json


# ============================================================
# BENCHMARK DEFINITIONS
# ============================================================

BENCHMARKS = {
    
    # ============================================================
    # CATEGORY 1: DETERMINISTIC STRUCTURE
    # These should be near 100% - the grammar requires it
    # ============================================================
    
    "deterministic": {
        "name": "Deterministic Structure",
        "description": "Hard syntactic constraints - model should be ~100% confident",
        "tests": [
            {
                "input": "FORALL",
                "expected": "VAR",
                "expected_min": 95,
                "rationale": "Universal quantifier must bind a variable"
            },
            {
                "input": "EXISTS",
                "expected": "VAR",
                "expected_min": 95,
                "rationale": "Existential quantifier must bind a variable"
            },
            {
                "input": "PRED 3",
                "expected": "LPAREN",
                "expected_min": 95,
                "rationale": "Predicate with index must open argument list"
            },
            {
                "input": "PRED 0 LPAREN VAR 0 COMMA VAR 1",
                "expected": "RPAREN",
                "expected_min": 90,
                "rationale": "Binary predicate with two arguments must close"
            },
            {
                "input": "FUNC 2",
                "expected": "LPAREN",
                "expected_min": 90,
                "rationale": "Function with index must open argument list"
            }
        ]
    },

    # ============================================================
    # CATEGORY 2: CALIBRATED UNCERTAINTY
    # Model should show appropriate uncertainty across valid options
    # ============================================================
    
    "calibrated_uncertainty": {
        "name": "Calibrated Uncertainty",
        "description": "Grammar branches here - model should distribute probability across valid continuations",
        "tests": [
            {
                "input": "NOT",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS"],
                "expected_max": 40,
                "rationale": "After NOT, multiple valid continuations: predicate, grouped formula, or quantified formula"
            },
            {
                "input": "FORALL VAR 1",
                "expected_one_of": ["LPAREN", "PRED", "FORALL", "EXISTS", "NOT"],
                "expected_max": 50,
                "rationale": "After binding a variable, can start body: predicate, group, nested quantifier, negation"
            },
            {
                "input": "FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN",
                "expected_one_of": ["AND", "OR", "IMPLIES", "IFF"],
                "expected_max": 40,
                "rationale": "Complete formula can extend with any connective"
            },
            {
                "input": "LPAREN PRED 1 LPAREN VAR 0 RPAREN",
                "expected_one_of": ["AND", "OR", "IMPLIES", "IFF"],
                "expected_max": 35,
                "rationale": "After first clause in compound, need binary connective"
            }
        ]
    },

    # ============================================================
    # CATEGORY 3: PREDICATE ARITY
    # Model should learn distribution of unary vs binary predicates
    # ============================================================
    
    "predicate_arity": {
        "name": "Predicate Arity Distribution",
        "description": "Model should learn that predicates can be unary or binary",
        "tests": [
            {
                "input": "PRED 7 LPAREN VAR 2",
                "expected_one_of": ["RPAREN", "COMMA"],
                "expected_distribution": {"RPAREN": [50, 80], "COMMA": [20, 50]},
                "rationale": "After first argument: close (unary) or continue (binary)"
            },
            {
                "input": "PRED 0 LPAREN VAR 0",
                "expected_one_of": ["RPAREN", "COMMA"],
                "rationale": "Same test with different indices - should show similar distribution"
            },
            {
                "input": "PRED 99 LPAREN VAR 50",
                "expected_one_of": ["RPAREN", "COMMA"],
                "rationale": "Compositional test - large indices should behave same as small"
            }
        ]
    },

    # ============================================================
    # CATEGORY 4: PARENTHESIS TRACKING
    # Critical test: does the model track structural depth?
    # ============================================================
    
    "parenthesis_tracking": {
        "name": "Parenthesis Balance Tracking",
        "description": "Model must track open/close parens across sequence - tests structural memory",
        "tests": [
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1",
                "expected": "RPAREN",
                "expected_min": 55,
                "rationale": "Inside PRED 1's argument list - must close predicate paren first"
            },
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
                "expected_one_of": ["RPAREN", "AND", "OR", "IMPLIES", "IFF"],
                "rparen_min": 25,
                "rationale": "Compound complete, one outer LPAREN unclosed - can close or extend"
            },
            {
                "input": "LPAREN LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN RPAREN",
                "expected_one_of": ["RPAREN", "AND", "OR", "IMPLIES", "IFF"],
                "rparen_min": 20,
                "rationale": "Two levels of nesting - still one unclosed outer paren"
            },
            {
                "input": "PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
                "expected_one_of": ["AND", "OR", "IMPLIES", "IFF"],
                "rparen_max": 15,
                "rationale": "All parens balanced - RPAREN should be LOW probability"
            }
        ]
    },

    # ============================================================
    # CATEGORY 5: COMPOSITIONAL GENERALISATION
    # Same structure, different indices - tests true learning vs memorisation
    # ============================================================
    
    "compositional_generalisation": {
        "name": "Compositional Generalisation",
        "description": "Model should treat VAR 0 and VAR 624 identically in structural context",
        "tests": [
            {
                "input": "FORALL VAR 0",
                "compare_with": "FORALL VAR 99",
                "rationale": "Different variable index should yield same structural predictions"
            },
            {
                "input": "FORALL VAR 0",
                "compare_with": "FORALL VAR 624",
                "rationale": "Max single-digit base-625 index should behave identically"
            },
            {
                "input": "PRED 0 LPAREN VAR 0",
                "compare_with": "PRED 50 LPAREN VAR 100",
                "rationale": "Different pred and var indices should yield same arity distribution"
            },
            {
                "input": "EXISTS VAR 1 PRED 1 LPAREN VAR 1 RPAREN",
                "compare_with": "EXISTS VAR 500 PRED 200 LPAREN VAR 500 RPAREN",
                "rationale": "Complete formulas with different indices should have same continuation distribution"
            }
        ]
    },

    # ============================================================
    # CATEGORY 6: NESTED QUANTIFIERS
    # Tests understanding of scope and nesting
    # ============================================================
    
    "nested_quantifiers": {
        "name": "Nested Quantifier Handling",
        "description": "Model should handle multiple levels of quantification",
        "tests": [
            {
                "input": "FORALL VAR 0 FORALL VAR 1",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
                "rationale": "After two quantifiers, start body - same options as after one"
            },
            {
                "input": "FORALL VAR 0 EXISTS VAR 1",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
                "rationale": "Mixed quantifiers should behave same as repeated"
            },
            {
                "input": "FORALL VAR 0 EXISTS VAR 1 FORALL VAR 2",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
                "rationale": "Triple nesting - should still understand body comes next"
            }
        ]
    },

    # ============================================================
    # CATEGORY 7: IMPLICATION STRUCTURE
    # Critical for reasoning - model must understand implies
    # ============================================================
    
    "implication_structure": {
        "name": "Implication Structure",
        "description": "Understanding of antecedent → consequent structure",
        "tests": [
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
                "rationale": "RHS of implication needs a formula"
            },
            {
                "input": "FORALL VAR 0 LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES PRED 1 LPAREN VAR 0 RPAREN",
                "expected": "RPAREN",
                "expected_min": 30,
                "rationale": "Complete implication inside quantifier scope - should close"
            }
        ]
    },

    # ============================================================
    # CATEGORY 8: ERROR RECOVERY / EDGE CASES
    # What does the model do with unusual inputs?
    # ============================================================
    
    "edge_cases": {
        "name": "Edge Cases",
        "description": "Unusual or malformed inputs - observational, not pass/fail",
        "tests": [
            {
                "input": "RPAREN",
                "observational": True,
                "rationale": "Starting with close paren - what does model predict?"
            },
            {
                "input": "AND",
                "observational": True,
                "rationale": "Starting with connective - what does model predict?"
            },
            {
                "input": "VAR",
                "observational": True,
                "rationale": "Incomplete compositional token - how does model handle?"
            },
            {
                "input": "LPAREN RPAREN",
                "observational": True,
                "rationale": "Empty parens - what continuation does model expect?"
            }
        ]
    }
}


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class BenchmarkRunner:
    """Runs structural benchmarks against a Symbolic Transformer model."""
    
    def __init__(self, model, vocab, device='cpu', prepend_start: bool = True):
        """
        Args:
            model: Trained Symbolic Transformer model
            vocab: Vocabulary object with encode/decode methods
            device: torch device
            prepend_start: Prepend MESSAGE_START to every input (when the
                vocabulary has it). Every training context begins with
                MESSAGE_START, so probing without it is out of distribution.
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.prepend_start = prepend_start and vocab.message_start is not None
        self.model.eval()
    
    def tokenize(self, input_str: str) -> List[int]:
        """Encode a space-separated label string, adding MESSAGE_START if needed."""
        tokens = self.vocab.encode_formula_simple(input_str)
        if self.prepend_start and (not tokens or tokens[0] != self.vocab.message_start):
            tokens = [self.vocab.message_start] + tokens
        return tokens
    
    def get_predictions(self, input_str: str) -> Dict[str, float]:
        """
        Get prediction probabilities for an input string.
        
        Args:
            input_str: Space-separated token string (e.g., "FORALL VAR 0")
            
        Returns:
            Dict mapping token names to probabilities (as percentages)
        """
        import torch
        import torch.nn.functional as F
        
        # Tokenize input
        tokens = self.tokenize(input_str)
        input_tensor = torch.tensor([tokens], device=self.device)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits[0, -1, :], dim=-1)
        
        # Convert to dict with token names
        predictions = {}
        for idx, prob in enumerate(probs.cpu().numpy()):
            token_name = self.vocab.id_to_label.get(idx, f"UNK_{idx}")
            predictions[token_name] = float(prob) * 100  # Convert to percentage
        
        return predictions
    
    def run_all(self) -> Dict:
        """Run all benchmark categories and return report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "summary": {"passed": 0, "failed": 0, "observational": 0}
        }
        
        for category_key, category in BENCHMARKS.items():
            report["categories"][category_key] = {
                "name": category["name"],
                "description": category["description"],
                "tests": []
            }
            
            for test in category["tests"]:
                result = self._run_single_test(test)
                report["categories"][category_key]["tests"].append(result)
                
                if result.get("observational"):
                    report["summary"]["observational"] += 1
                elif result.get("passed"):
                    report["summary"]["passed"] += 1
                else:
                    report["summary"]["failed"] += 1
        
        total_tests = report["summary"]["passed"] + report["summary"]["failed"]
        report["summary"]["total"] = total_tests + report["summary"]["observational"]
        report["summary"]["pass_rate"] = (
            f"{(report['summary']['passed'] / total_tests * 100):.1f}"
            if total_tests > 0 else "N/A"
        )
        
        return report
    
    def _run_single_test(self, test: Dict) -> Dict:
        """Run a single benchmark test."""
        predictions = self.get_predictions(test["input"])
        
        # Get top 5 predictions for display
        top5 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = {
            "input": test["input"],
            "rationale": test["rationale"],
            "top5": top5,
            "observational": test.get("observational", False)
        }
        
        if test.get("observational"):
            result["note"] = "Observational test - no pass/fail criteria"
            return result
        
        # Check deterministic expectation
        if "expected" in test and "expected_min" in test:
            expected_token = test["expected"]
            min_prob = test["expected_min"]
            actual_prob = predictions.get(expected_token, 0)
            
            result["passed"] = actual_prob >= min_prob
            result["detail"] = f"Expected {expected_token} >= {min_prob}%, got {actual_prob:.1f}%"
        
        # Check distribution expectation
        elif "expected_one_of" in test:
            expected_tokens = test["expected_one_of"]
            probs = [predictions.get(t, 0) for t in expected_tokens]
            total_prob = sum(probs)
            
            result["passed"] = total_prob >= 50
            result["detail"] = f"Expected one of {expected_tokens}, total prob: {total_prob:.1f}%"
            
            # Additional checks
            if "expected_max" in test:
                max_prob = max(probs)
                result["passed"] = result["passed"] and max_prob <= test["expected_max"]
                result["detail"] += f" (max <= {test['expected_max']}%: {max_prob:.1f}%)"
            
            if "rparen_min" in test:
                rparen_prob = predictions.get("RPAREN", 0)
                result["passed"] = result["passed"] and rparen_prob >= test["rparen_min"]
                result["detail"] += f" (RPAREN >= {test['rparen_min']}%: {rparen_prob:.1f}%)"
            
            if "rparen_max" in test:
                rparen_prob = predictions.get("RPAREN", 0)
                result["passed"] = result["passed"] and rparen_prob <= test["rparen_max"]
                result["detail"] += f" (RPAREN <= {test['rparen_max']}%: {rparen_prob:.1f}%)"
        
        # Compositional comparison
        elif "compare_with" in test:
            predictions2 = self.get_predictions(test["compare_with"])
            top5_2 = sorted(predictions2.items(), key=lambda x: x[1], reverse=True)[:5]
            
            result["compare_with"] = test["compare_with"]
            result["top5_compare"] = top5_2
            
            # Check if top-5 tokens overlap
            top5_tokens_1 = set(t for t, _ in top5)
            top5_tokens_2 = set(t for t, _ in top5_2)
            overlap = len(top5_tokens_1 & top5_tokens_2)
            
            result["passed"] = overlap >= 4
            result["detail"] = f"Top-5 overlap: {overlap}/5 tokens match"
        
        return result
    
    def run_category(self, category_key: str) -> Dict:
        """Run benchmarks for a single category."""
        if category_key not in BENCHMARKS:
            raise ValueError(f"Unknown category: {category_key}")
        
        category = BENCHMARKS[category_key]
        results = {
            "name": category["name"],
            "description": category["description"],
            "tests": []
        }
        
        for test in category["tests"]:
            result = self._run_single_test(test)
            results["tests"].append(result)
        
        return results
    
    def generate_report(self, report: Dict) -> str:
        """Generate a human-readable report string."""
        lines = []
        
        lines.append("═" * 60)
        lines.append("SYMBOLIC TRANSFORMER STRUCTURAL BENCHMARKS")
        lines.append("═" * 60)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Pass Rate: {report['summary']['pass_rate']}% ({report['summary']['passed']}/{report['summary']['passed'] + report['summary']['failed']})")
        lines.append(f"Observational: {report['summary']['observational']}")
        lines.append("")
        
        for key, category in report["categories"].items():
            lines.append("─" * 60)
            lines.append(f"{category['name'].upper()}")
            lines.append(f"{category['description']}")
            lines.append("─" * 60)
            
            for test in category["tests"]:
                if test.get("observational"):
                    status = "👁️"
                elif test.get("passed"):
                    status = "✓"
                else:
                    status = "✗"
                
                lines.append(f"{status} Input: {test['input']}")
                lines.append(f"  {test['rationale']}")
                
                # Show top 5 predictions
                if test.get("top5"):
                    top5_str = ", ".join(f"{t}: {p:.1f}%" for t, p in test["top5"])
                    lines.append(f"  Top 5: {top5_str}")
                
                if test.get("detail"):
                    lines.append(f"  Result: {test['detail']}")
                
                if test.get("compare_with"):
                    lines.append(f"  Compared with: {test['compare_with']}")
                    if test.get("top5_compare"):
                        top5_str = ", ".join(f"{t}: {p:.1f}%" for t, p in test["top5_compare"])
                        lines.append(f"  Compare Top 5: {top5_str}")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def save_report(self, report: Dict, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


# ============================================================
# QUICK TEST FUNCTION
# ============================================================

def quick_test(input_str: str, model, vocab, device='cpu') -> None:
    """
    Quick test function for interactive use.
    
    Usage:
        quick_test("FORALL", model, vocab)
    """
    runner = BenchmarkRunner(model, vocab, device)
    predictions = runner.get_predictions(input_str)
    
    print(f"\nInput: {input_str}")
    print("Top 5 predictions:")
    for token, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]:
        bar = "█" * int(prob / 2)
        print(f"  {token:12} {prob:5.1f}% {bar}")


# ============================================================
# MAIN - Run benchmarks from command line
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Symbolic Transformer benchmarks")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary JSON")
    parser.add_argument("--output", help="Path to save JSON report")
    parser.add_argument("--category", help="Run only specific category")
    
    parser.add_argument("--no-prepend-start", action="store_true",
                        help="Do not prepend MESSAGE_START to benchmark inputs")
    args = parser.parse_args()
    
    import os
    import sys
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.transformer import load_model_from_checkpoint
    from utils.vocabulary import Vocabulary
    
    vocab = Vocabulary(args.vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, vocab.vocab_size, device)
    
    runner = BenchmarkRunner(model, vocab, device, prepend_start=not args.no_prepend_start)
    if args.category:
        category = runner.run_category(args.category)
        report = {"timestamp": datetime.now().isoformat(), "categories": {args.category: category},
                  "summary": {"passed": 0, "failed": 0, "observational": 0}}
        for t in category["tests"]:
            key = "observational" if t.get("observational") else ("passed" if t.get("passed") else "failed")
            report["summary"][key] += 1
        total = report["summary"]["passed"] + report["summary"]["failed"]
        report["summary"]["pass_rate"] = f"{report['summary']['passed'] / total * 100:.1f}" if total else "N/A"
    else:
        report = runner.run_all()
    print(runner.generate_report(report))
    if args.output:
        runner.save_report(report, args.output)
        print(f"Saved report to {args.output}")
