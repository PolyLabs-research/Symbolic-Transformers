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

# Every scored expectation below is checked against the generator's grammar by
# tests/test_benchmarks_grammar.py (utils/fol_grammar.py lists the grammatical continuations of a
# prefix). In that grammar a binary connective always sits inside its own brackets, "( A AND B )",
# so a complete top-level formula can only be followed by MESSAGE_END. Variable and predicate
# indices are 0-99 in the shipped data; cases using larger indices are observational, because those
# embedding rows are never trained.
FORMULA_STARTS = ["PRED", "VAR", "LPAREN", "FORALL", "EXISTS", "NOT"]
CONNECTIVES = ["AND", "OR", "IMPLIES", "IFF"]

BENCHMARKS = {

    # ============================================================
    # CATEGORY 1: MESSAGE BOUNDARIES
    # ============================================================

    "message_boundaries": {
        "name": "Message Boundaries",
        "description": "MESSAGE_END must follow a complete formula and nothing else",
        "tests": [
            {
                "input": "MESSAGE_START",
                "expected_one_of": FORMULA_STARTS,
                "message_end_max": 5,
                "rationale": "A message must contain a formula, so MESSAGE_END is impossible here"
            },
            {
                "input": "PRED 0 LPAREN VAR 0 RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 90,
                "rationale": "A complete top-level formula ends the message (connectives only appear inside brackets)"
            },
            {
                "input": "FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 90,
                "rationale": "The quantifier's body is complete, so the whole formula is: the message ends"
            },
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN",
                "expected_one_of": CONNECTIVES,
                "message_end_max": 5,
                "rationale": "An open bracket holds a left operand that needs a connective: ending now is malformed"
            },
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 90,
                "rationale": "Every bracket is closed and the formula is complete"
            }
        ]
    },

    # ============================================================
    # CATEGORY 2: DETERMINISTIC STRUCTURE
    # The grammar allows exactly one token - model should be ~100% confident
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
                "input": "PRED 3 LPAREN",
                "expected": "VAR",
                "expected_min": 95,
                "rationale": "Arguments are variables"
            },
            {
                "input": "PRED 0 LPAREN VAR 0 COMMA VAR 1",
                "expected": "RPAREN",
                "expected_min": 90,
                "rationale": "Binary predicate with two arguments must close"
            },
            {
                "input": "VAR 3",
                "expected": "EQUALS",
                "expected_min": 90,
                "rationale": "An atomic formula that starts with a variable is an equality"
            }
        ]
    },

    # ============================================================
    # CATEGORY 3: CALIBRATED UNCERTAINTY
    # Model should show appropriate uncertainty across valid options
    # ============================================================

    "calibrated_uncertainty": {
        "name": "Calibrated Uncertainty",
        "description": "Grammar branches here - model should distribute probability across valid continuations",
        "tests": [
            {
                "input": "NOT",
                "expected_one_of": FORMULA_STARTS,
                "expected_max": 40,
                "rationale": "Any formula can follow NOT (the generator's most likely start, PRED, has 31%)"
            },
            {
                "input": "FORALL VAR 1",
                "expected_one_of": FORMULA_STARTS,
                "expected_max": 50,
                "rationale": "After binding a variable, any formula can start the body"
            },
            {
                "input": "LPAREN PRED 1 LPAREN VAR 0 RPAREN",
                "expected_one_of": CONNECTIVES,
                "expected_max": 35,
                "rationale": "After the first operand, one of four connectives, each with probability 1/4"
            }
        ]
    },

    # ============================================================
    # CATEGORY 4: PREDICATE ARITY
    # Predicates are unary or binary with equal probability
    # ============================================================

    "predicate_arity": {
        "name": "Predicate Arity Distribution",
        "description": "Predicates are unary or binary, 50/50, whatever the indices",
        "tests": [
            {
                "input": "PRED 7 LPAREN VAR 2",
                "expected_distribution": {"RPAREN": [40, 60], "COMMA": [40, 60]},
                "rationale": "After the first argument: close (unary) or continue (binary), 50/50"
            },
            {
                "input": "PRED 0 LPAREN VAR 0",
                "expected_distribution": {"RPAREN": [40, 60], "COMMA": [40, 60]},
                "rationale": "Same test with index 0"
            },
            {
                "input": "PRED 99 LPAREN VAR 50",
                "expected_distribution": {"RPAREN": [40, 60], "COMMA": [40, 60]},
                "rationale": "Same test at the top of the trained index range"
            }
        ]
    },

    # ============================================================
    # CATEGORY 5: PARENTHESIS TRACKING
    # Critical test: does the model track structural depth?
    # ============================================================

    "parenthesis_tracking": {
        "name": "Parenthesis Balance Tracking",
        "description": "Model must track open brackets and operands across the sequence",
        "tests": [
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1",
                "expected_distribution": {"RPAREN": [40, 60], "COMMA": [40, 60]},
                "rationale": "Inside PRED 1's argument list: unary or binary, 50/50; the outer bracket must wait"
            },
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
                "expected": "RPAREN",
                "expected_min": 90,
                "rationale": "Both operands are complete, so the bracket must close (a bracket holds two operands)"
            },
            {
                "input": "LPAREN LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN RPAREN",
                "expected_one_of": CONNECTIVES,
                "rparen_max": 5,
                "message_end_max": 5,
                "rationale": "The inner compound is the left operand of the outer bracket: a connective must follow"
            },
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN AND LPAREN PRED 1 LPAREN VAR 1 RPAREN OR PRED 2 LPAREN VAR 2 RPAREN RPAREN",
                "expected": "RPAREN",
                "expected_min": 90,
                "rationale": "The nested compound closed as the right operand, so the outer bracket closes next"
            }
        ]
    },

    # ============================================================
    # CATEGORY 6: COMPOSITIONAL GENERALISATION
    # Same structure, different indices - tests true learning vs memorisation
    # ============================================================

    "compositional_generalisation": {
        "name": "Compositional Generalisation",
        "description": "Structural predictions should not depend on index values (scored within the trained range 0-99)",
        "tests": [
            {
                "input": "FORALL VAR 0",
                "compare_with": "FORALL VAR 99",
                "rationale": "Different variable index should yield the same structural predictions"
            },
            {
                "input": "PRED 0 LPAREN VAR 0",
                "compare_with": "PRED 50 LPAREN VAR 90",
                "rationale": "Different predicate and variable indices should yield the same arity distribution"
            },
            {
                "input": "EXISTS VAR 1 PRED 1 LPAREN VAR 1 RPAREN",
                "compare_with": "EXISTS VAR 70 PRED 40 LPAREN VAR 70 RPAREN",
                "rationale": "Complete formulas with different indices should have the same continuation"
            },
            {
                "input": "FORALL VAR 0",
                "compare_with": "FORALL VAR 624",
                "observational": True,
                "rationale": "Index 624 is outside the trained range: its embedding row was never trained"
            },
            {
                "input": "EXISTS VAR 1 PRED 1 LPAREN VAR 1 RPAREN",
                "compare_with": "EXISTS VAR 500 PRED 200 LPAREN VAR 500 RPAREN",
                "observational": True,
                "rationale": "Indices 200 and 500 are outside the trained range"
            }
        ]
    },

    # ============================================================
    # CATEGORY 7: VARIABLE SCOPE
    # The generator always refers to a bound variable when one is in scope
    # ============================================================

    "variable_scope": {
        "name": "Variable Scope",
        "description": "Variable references should go to the variables bound in scope",
        "tests": [
            {
                "input": "FORALL VAR 7 PRED 3 LPAREN VAR",
                "expected": "NUM_007",
                "expected_min": 80,
                "rationale": "The only variable in scope is 7"
            },
            {
                "input": "FORALL VAR 7 EXISTS VAR 42 PRED 3 LPAREN VAR",
                "expected_one_of": ["NUM_007", "NUM_042"],
                "total_min": 80,
                "expected_max": 75,
                "rationale": "Variables 7 and 42 are in scope, 50/50; little probability should go elsewhere"
            },
            {
                "input": "PRED 3 LPAREN VAR",
                "top_max": 5,
                "rationale": "No variable is in scope, so the index is uniform over 100 values (1% each)"
            }
        ]
    },

    # ============================================================
    # CATEGORY 8: NESTED QUANTIFIERS
    # Tests understanding of scope and nesting
    # ============================================================

    "nested_quantifiers": {
        "name": "Nested Quantifier Handling",
        "description": "Model should handle multiple levels of quantification",
        "tests": [
            {
                "input": "FORALL VAR 0 FORALL VAR 1",
                "expected_one_of": FORMULA_STARTS,
                "rationale": "After two quantifiers, start body - same options as after one"
            },
            {
                "input": "FORALL VAR 0 EXISTS VAR 1",
                "expected_one_of": FORMULA_STARTS,
                "rationale": "Mixed quantifiers should behave same as repeated"
            },
            {
                "input": "FORALL VAR 0 EXISTS VAR 1 FORALL VAR 2",
                "expected_one_of": FORMULA_STARTS,
                "rationale": "Triple nesting - should still understand body comes next"
            }
        ]
    },

    # ============================================================
    # CATEGORY 9: IMPLICATION STRUCTURE
    # ============================================================

    "implication_structure": {
        "name": "Implication Structure",
        "description": "Understanding of antecedent → consequent structure",
        "tests": [
            {
                "input": "LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES",
                "expected_one_of": FORMULA_STARTS,
                "rationale": "RHS of implication needs a formula"
            },
            {
                "input": "FORALL VAR 0 LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES PRED 1 LPAREN VAR 0 RPAREN",
                "expected": "RPAREN",
                "expected_min": 90,
                "rationale": "Both sides of the implication are complete, so its bracket must close"
            }
        ]
    },

    # ============================================================
    # CATEGORY 10: ERROR RECOVERY / EDGE CASES
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
            },
            {
                "input": "MESSAGE_START MESSAGE_END",
                "observational": True,
                "rationale": "Empty message - what does model predict?"
            },
            {
                "input": "FUNC 2",
                "observational": True,
                "rationale": "FUNC is in the vocabulary but never occurs in the training data"
            }
        ]
    }
}


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Total variation distance between two prediction dicts given in percent (0-100)."""
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys) / 2


def score_test(test: Dict, predict: Callable[[str], Dict[str, float]]) -> Dict:
    """Score one benchmark case; ``predict`` maps an input string to {token label: percent}."""
    predictions = predict(test["input"])
    top5 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    result = {
        "input": test["input"],
        "rationale": test["rationale"],
        "top5": top5,
        "observational": test.get("observational", False)
    }

    if "compare_with" in test:
        predictions2 = predict(test["compare_with"])
        tv = total_variation(predictions, predictions2)
        result["compare_with"] = test["compare_with"]
        result["top5_compare"] = sorted(predictions2.items(), key=lambda x: x[1], reverse=True)[:5]
        result["passed"] = tv <= test.get("max_tv", 10)
        result["detail"] = f"Total variation distance {tv:.1f}%"
        if not test.get("observational"):
            result["detail"] += f" (pass <= {test.get('max_tv', 10)}%)"
    elif "expected" in test:
        actual = predictions.get(test["expected"], 0)
        result["passed"] = actual >= test["expected_min"]
        result["detail"] = f"Expected {test['expected']} >= {test['expected_min']}%, got {actual:.1f}%"
    elif "expected_one_of" in test:
        probs = [predictions.get(t, 0) for t in test["expected_one_of"]]
        total_min = test.get("total_min", 50)
        result["passed"] = sum(probs) >= total_min
        result["detail"] = f"Expected one of {test['expected_one_of']}, total prob: {sum(probs):.1f}% (>= {total_min}%)"
        if "expected_max" in test:
            result["passed"] = result["passed"] and max(probs) <= test["expected_max"]
            result["detail"] += f" (max <= {test['expected_max']}%: {max(probs):.1f}%)"
    elif "expected_distribution" in test:
        result["passed"] = True
        parts = []
        for tok, (lo, hi) in test["expected_distribution"].items():
            actual = predictions.get(tok, 0)
            result["passed"] = result["passed"] and lo <= actual <= hi
            parts.append(f"{tok} {actual:.1f}% (want {lo}-{hi})")
        result["detail"] = ", ".join(parts)
    elif "top_max" in test:
        top = top5[0][1] if top5 else 0
        result["passed"] = top <= test["top_max"]
        result["detail"] = f"Largest single-token probability {top:.1f}% (want <= {test['top_max']}%)"

    for tok, key in (("RPAREN", "rparen_max"), ("MESSAGE_END", "message_end_max")):
        if key in test and "passed" in result:
            actual = predictions.get(tok, 0)
            result["passed"] = result["passed"] and actual <= test[key]
            result["detail"] += f" ({tok} <= {test[key]}%: {actual:.1f}%)"

    if result["observational"]:
        result.pop("passed", None)
        result["note"] = "Observational test - no pass/fail criteria"
    return result


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
        return score_test(test, self.get_predictions)

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
