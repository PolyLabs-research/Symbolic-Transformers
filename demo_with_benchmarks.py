"""
Symbolic Transformer Demo with Structural Benchmarks (Phase 1.1 - with MESSAGE boundaries)

Integrated demo that includes:
1. Interactive prediction explorer
2. Structural benchmark suite (including message boundaries)
3. Visual result display

Usage:
    python demo_with_benchmarks.py
"""

import gradio as gr
import torch
import torch.nn.functional as F
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Ensure we can import from the current directory
sys.path.insert(0, os.getcwd())

from utils.vocabulary import Vocabulary
from models.transformer import create_model


# ============================================================
# BENCHMARK DEFINITIONS
# ============================================================

BENCHMARKS = {
    
    "message_boundaries": {
        "name": "Message Boundaries (NEW)",
        "description": "Tests that MESSAGE_START/MESSAGE_END tokens work correctly for formula termination",
        "tests": [
            {
                "input": "MESSAGE_START PRED 0 LPAREN VAR 0 RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 60,
                "rationale": "Complete formula with balanced parens should end with MESSAGE_END"
            },
            {
                "input": "MESSAGE_START FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 60,
                "rationale": "Complete quantified formula should end with MESSAGE_END"
            },
            {
                "input": "MESSAGE_START LPAREN PRED 0 LPAREN VAR 0 RPAREN",
                "expected_one_of": ["AND", "OR", "IMPLIES", "IFF"],
                "message_end_max": 5,
                "rationale": "Unbalanced parens - MESSAGE_END should be very unlikely"
            },
            {
                "input": "MESSAGE_START",
                "expected_one_of": ["PRED", "LPAREN", "FORALL", "EXISTS", "NOT"],
                "message_end_max": 5,
                "rationale": "After MESSAGE_START, formula must begin (MESSAGE_END impossible)"
            },
            {
                "input": "MESSAGE_START LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN RPAREN",
                "expected": "MESSAGE_END",
                "expected_min": 70,
                "rationale": "Complex formula with all parens balanced should strongly predict MESSAGE_END"
            }
        ]
    },
    
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
                "input": "MESSAGE_START FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN",
                "expected_one_of": ["MESSAGE_END", "AND", "OR", "IMPLIES", "IFF"],
                "message_end_min": 40,
                "rationale": "Complete formula can end or extend - MESSAGE_END should be strong option"
            },
            {
                "input": "LPAREN PRED 1 LPAREN VAR 0 RPAREN",
                "expected_one_of": ["AND", "OR", "IMPLIES", "IFF"],
                "expected_max": 35,
                "rationale": "After first clause in compound, need binary connective"
            }
        ]
    },

    "predicate_arity": {
        "name": "Predicate Arity Distribution",
        "description": "Model should learn that predicates can be unary or binary",
        "tests": [
            {
                "input": "PRED 7 LPAREN VAR 2",
                "expected_one_of": ["RPAREN", "COMMA"],
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
                "input": "MESSAGE_START PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN",
                "expected_one_of": ["MESSAGE_END", "AND", "OR", "IMPLIES", "IFF"],
                "rationale": "All parens balanced with MESSAGE_START - should strongly prefer MESSAGE_END or connectives"
            }
        ]
    },

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
            }
        ]
    }
}


# ============================================================
# SETUP & LOAD MODEL
# ============================================================

print("🔄 Initializing GUI with Benchmarks...")

if not os.path.exists('unified_vocabulary.json'):
    raise FileNotFoundError("unified_vocabulary.json not found")

vocab = Vocabulary('unified_vocabulary.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if MESSAGE_START/MESSAGE_END are available
HAS_MESSAGE_BOUNDARIES = vocab.has_special_token('MESSAGE_START') and vocab.has_special_token('MESSAGE_END')

if not HAS_MESSAGE_BOUNDARIES:
    print("⚠ Warning: MESSAGE_START/MESSAGE_END not in vocabulary")
    print("   Message boundary tests will be skipped")
    # Remove message boundary category
    if "message_boundaries" in BENCHMARKS:
        del BENCHMARKS["message_boundaries"]

if not os.path.exists('checkpoints/best_model.pt'):
    raise FileNotFoundError("checkpoints/best_model.pt not found")

checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
config = checkpoint['config']
model = create_model(vocab_size=vocab.vocab_size, model_size=config['model_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✓ Vocabulary loaded: {vocab.vocab_size} tokens")
print(f"✓ Model loaded: {config['model_size']} configuration")
print(f"✓ Device: {device}")
if HAS_MESSAGE_BOUNDARIES:
    print(f"✓ Message boundaries: MESSAGE_START ({vocab.message_start}), MESSAGE_END ({vocab.message_end})")


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def parse_tokens(sequence_text: str) -> List[int]:
    """Parse 'FORALL VAR 1' into token IDs.

    Numerals must be 0-624 (anything else would index past the embedding
    table). MESSAGE_START is prepended unless already present, because every
    training context starts with it.
    """
    token_ids: List[int] = []
    for t in sequence_text.strip().split():
        if t in vocab.label_to_id:
            token_ids.append(vocab.encode_label(t))
        elif t.isdigit() and 0 <= int(t) <= vocab.numeral_range[1]:
            token_ids.append(int(t))
        else:
            raise ValueError(f"Unknown token {t!r} (labels or numerals 0-{vocab.numeral_range[1]} only)")
    if HAS_MESSAGE_BOUNDARIES and (not token_ids or token_ids[0] != vocab.message_start):
        token_ids = [vocab.message_start] + token_ids
    return token_ids


def predict_next_token(sequence_text: str) -> Dict[str, float]:
    """Takes a string like 'FORALL VAR', returns a dict of top tokens with probabilities."""
    if not sequence_text.strip():
        return None
    
    try:
        token_ids = parse_tokens(sequence_text)
    except ValueError as exc:
        return {str(exc): 0.0}
    
    x = torch.tensor([token_ids], device=device, dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
    
    top_probs, top_ids = torch.topk(probs, 10)
    
    results = {}
    for prob, tid in zip(top_probs, top_ids):
        label = vocab.decode_id(tid.item())
        results[label] = float(prob) * 100  # Convert to percentage
    
    return results


def get_full_predictions(sequence_text: str) -> Dict[str, float]:
    """Get all predictions as percentages."""
    if not sequence_text.strip():
        return {}
    
    try:
        token_ids = parse_tokens(sequence_text)
    except ValueError:
        return {}
    
    x = torch.tensor([token_ids], device=device, dtype=torch.long)
    
    with torch.no_grad():
        logits = model(x)
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
    
    results = {}
    for idx, prob in enumerate(probs.cpu().numpy()):
        label = vocab.decode_id(idx)
        results[label] = float(prob) * 100
    
    return results


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_single_test(test: Dict) -> Dict:
    """Run a single benchmark test."""
    predictions = get_full_predictions(test["input"])
    
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
        
        if "expected_max" in test:
            max_prob = max(probs)
            result["passed"] = result["passed"] and max_prob <= test["expected_max"]
            result["detail"] += f" (max <= {test['expected_max']}%: {max_prob:.1f}%)"
        
        # NEW: MESSAGE_END specific checks
        if "message_end_min" in test:
            msg_end_prob = predictions.get("MESSAGE_END", 0)
            result["passed"] = result["passed"] and msg_end_prob >= test["message_end_min"]
            result["detail"] += f" (MESSAGE_END >= {test['message_end_min']}%: {msg_end_prob:.1f}%)"
        
        if "message_end_max" in test:
            msg_end_prob = predictions.get("MESSAGE_END", 0)
            result["passed"] = result["passed"] and msg_end_prob <= test["message_end_max"]
            result["detail"] += f" (MESSAGE_END <= {test['message_end_max']}%: {msg_end_prob:.1f}%)"
        
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
        predictions2 = get_full_predictions(test["compare_with"])
        top5_2 = sorted(predictions2.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result["compare_with"] = test["compare_with"]
        result["top5_compare"] = top5_2
        
        top5_tokens_1 = set(t for t, _ in top5)
        top5_tokens_2 = set(t for t, _ in top5_2)
        overlap = len(top5_tokens_1 & top5_tokens_2)
        
        result["passed"] = overlap >= 4
        result["detail"] = f"Top-5 overlap: {overlap}/5 tokens match"
    
    return result


def run_category(category_key: str) -> str:
    """Run all benchmarks in a category and return formatted markdown."""
    if category_key not in BENCHMARKS:
        return f"Unknown category: {category_key}"
    
    category = BENCHMARKS[category_key]
    results = []
    
    for test in category["tests"]:
        results.append(run_single_test(test))
    
    passed = sum(1 for t in results if t.get("passed") and not t.get("observational"))
    failed = sum(1 for t in results if not t.get("passed") and not t.get("observational"))
    observational = sum(1 for t in results if t.get("observational"))
    
    output = f"# {category['name']}\n\n"
    output += f"{category['description']}\n\n"
    output += f"**Results:** ✅ {passed} passed, ❌ {failed} failed, 👁️ {observational} observational\n\n"
    output += "---\n\n"
    
    for i, result in enumerate(results):
        if result.get("observational"):
            status = "👁️"
        elif result.get("passed"):
            status = "✅"
        else:
            status = "❌"
        
        output += f"### Test {i+1}: {status}\n\n"
        output += f"**Input:** `{result['input']}`\n\n"
        output += f"**Rationale:** {result['rationale']}\n\n"
        
        if result.get("top5"):
            output += "| Token | Probability |\n|-------|-------------|\n"
            for token, prob in result["top5"]:
                bar = "█" * int(prob / 5)
                output += f"| {token} | {prob:.1f}% {bar} |\n"
            output += "\n"
        
        if result.get("detail"):
            output += f"**Result:** {result['detail']}\n\n"
        
        if result.get("compare_with"):
            output += f"**Compared with:** `{result['compare_with']}`\n\n"
            if result.get("top5_compare"):
                output += "| Token | Probability |\n|-------|-------------|\n"
                for token, prob in result["top5_compare"]:
                    bar = "█" * int(prob / 5)
                    output += f"| {token} | {prob:.1f}% {bar} |\n"
                output += "\n"
        
        output += "---\n\n"
    
    return output


def run_all_benchmarks() -> Tuple[str, str]:
    """Run all benchmarks and return summary + detailed results."""
    timestamp = datetime.now().isoformat()
    all_results = {}
    total_passed = 0
    total_failed = 0
    total_observational = 0
    
    for category_key, category in BENCHMARKS.items():
        category_results = []
        for test in category["tests"]:
            result = run_single_test(test)
            category_results.append(result)
            
            if result.get("observational"):
                total_observational += 1
            elif result.get("passed"):
                total_passed += 1
            else:
                total_failed += 1
        
        all_results[category_key] = {
            "name": category["name"],
            "results": category_results
        }
    
    # Summary
    total_tests = total_passed + total_failed
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    summary = f"# 📊 Benchmark Summary\n\n"
    summary += f"**Timestamp:** {timestamp}\n\n"
    summary += f"**Pass Rate:** {pass_rate:.1f}% ({total_passed}/{total_tests})\n\n"
    
    if HAS_MESSAGE_BOUNDARIES:
        summary += f"**Message Boundaries:** ✅ Enabled (MESSAGE_START ID: {vocab.message_start}, MESSAGE_END ID: {vocab.message_end})\n\n"
    else:
        summary += f"**Message Boundaries:** ❌ Not available in vocabulary\n\n"
    
    summary += "| Metric | Count |\n|--------|-------|\n"
    summary += f"| ✅ Passed | {total_passed} |\n"
    summary += f"| ❌ Failed | {total_failed} |\n"
    summary += f"| 👁️ Observational | {total_observational} |\n"
    summary += f"| **Total** | {total_tests + total_observational} |\n\n"
    
    summary += "## Category Breakdown\n\n"
    summary += "| Category | Passed | Failed | Status |\n|----------|--------|--------|--------|\n"
    
    for category_key, data in all_results.items():
        passed = sum(1 for r in data["results"] if r.get("passed") and not r.get("observational"))
        failed = sum(1 for r in data["results"] if not r.get("passed") and not r.get("observational"))
        status = "✅" if failed == 0 else "⚠️" if passed > failed else "❌"
        summary += f"| {data['name']} | {passed} | {failed} | {status} |\n"
    
    # Detailed results
    detailed = "=" * 60 + "\n"
    detailed += "SYMBOLIC TRANSFORMER STRUCTURAL BENCHMARKS (v1.1)\n"
    detailed += "=" * 60 + "\n"
    detailed += f"Timestamp: {timestamp}\n"
    detailed += f"Pass Rate: {pass_rate:.1f}% ({total_passed}/{total_tests})\n"
    if HAS_MESSAGE_BOUNDARIES:
        detailed += f"Message Boundaries: ENABLED\n"
    detailed += "\n"
    
    for category_key, data in all_results.items():
        detailed += "-" * 60 + "\n"
        detailed += f"{data['name'].upper()}\n"
        detailed += "-" * 60 + "\n"
        
        for result in data["results"]:
            if result.get("observational"):
                status = "👁️"
            elif result.get("passed"):
                status = "✓"
            else:
                status = "✗"
            
            detailed += f"\n{status} Input: {result['input']}\n"
            detailed += f"  {result['rationale']}\n"
            
            if result.get("top5"):
                detailed += "  Top 5: " + ", ".join(f"{t}: {p:.1f}%" for t, p in result["top5"]) + "\n"
            
            if result.get("detail"):
                detailed += f"  Result: {result['detail']}\n"
    
    return summary, detailed


# ============================================================
# EXAMPLE INPUTS
# ============================================================

EXAMPLES = [
    ["FORALL"],
    ["EXISTS"],
    ["PRED 3"],
    ["NOT"],
    ["PRED 7 LPAREN VAR 2"],
    ["LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1"],
    ["LPAREN PRED 0 LPAREN VAR 0 RPAREN AND PRED 1 LPAREN VAR 1 RPAREN"],
    ["FORALL VAR 0 FORALL VAR 1"],
    ["FORALL VAR 0 LPAREN PRED 0 LPAREN VAR 0 RPAREN IMPLIES PRED 1 LPAREN VAR 0 RPAREN"],
]

if HAS_MESSAGE_BOUNDARIES:
    EXAMPLES.extend([
        ["MESSAGE_START"],
        ["MESSAGE_START PRED 0 LPAREN VAR 0 RPAREN"],
        ["MESSAGE_START FORALL VAR 0 PRED 0 LPAREN VAR 0 RPAREN"],
    ])


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="Symbolic Transformer Demo", theme=gr.themes.Soft()) as demo:
    
    boundary_status = "✅ With MESSAGE Boundaries" if HAS_MESSAGE_BOUNDARIES else "⚠️ Without MESSAGE Boundaries"
    
    gr.Markdown(f"""
    # 🧠 Symbolic Transformer Phase 1.1
    
    Interactive explorer for the FOL syntax model with structural benchmarks.
    
    **Model:** {config['model_size']} configuration | **Vocab:** {vocab.vocab_size} tokens | **Device:** {device} | **Status:** {boundary_status}
    """)
    
    with gr.Tabs():
        
        # ============================================================
        # TAB 1: Interactive Prediction
        # ============================================================
        with gr.TabItem("🔮 Prediction Explorer"):
            gr.Markdown("""
            Type a First-Order Logic formula prefix to see what the model predicts next.
            
            **Examples:** `FORALL` → VAR | `PRED 3` → LPAREN | `NOT` → distributed
            
            **With MESSAGE boundaries:** Try `MESSAGE_START PRED 0 LPAREN VAR 0 RPAREN` to see termination
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Context",
                        placeholder="e.g., MESSAGE_START FORALL VAR 0 PRED 1 LPAREN VAR 0 RPAREN",
                        lines=1
                    )
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=input_text,
                        label="Quick Examples"
                    )
                
                with gr.Column(scale=3):
                    output_label = gr.Label(
                        num_top_classes=5,
                        label="Next Token Probability"
                    )
            
            input_text.change(
                fn=lambda x: {k: v/100 for k, v in predict_next_token(x).items()} if predict_next_token(x) else None,
                inputs=input_text,
                outputs=output_label
            )
        
        # ============================================================
        # TAB 2: Structural Benchmarks
        # ============================================================
        with gr.TabItem("📊 Structural Benchmarks"):
            benchmark_intro = """
            ## Structural Benchmark Suite
            
            These tests verify that the model has learned FOL syntax rules, not just pattern matching.
            
            **Categories:**
            """
            
            if HAS_MESSAGE_BOUNDARIES:
                benchmark_intro += "- **Message Boundaries (NEW)**: Tests MESSAGE_START/MESSAGE_END termination\n"
            
            benchmark_intro += """
            - **Deterministic Structure**: Hard constraints (expect ~100%)
            - **Calibrated Uncertainty**: Appropriate distribution across valid options
            - **Predicate Arity**: Learned unary vs binary distribution
            - **Parenthesis Tracking**: Structural memory across sequence
            - **Compositional Generalisation**: Same structure, different indices
            - **Nested Quantifiers**: Multiple levels of quantification
            - **Implication Structure**: Antecedent → consequent understanding
            - **Edge Cases**: Observational tests for unusual inputs
            """
            
            gr.Markdown(benchmark_intro)
            
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in BENCHMARKS.items()],
                    label="Select Category",
                    value="message_boundaries" if HAS_MESSAGE_BOUNDARIES else "deterministic"
                )
                run_category_btn = gr.Button("Run Category", variant="primary")
                run_all_btn = gr.Button("Run All Benchmarks", variant="secondary")
            
            benchmark_output = gr.Markdown(label="Benchmark Results")
            
            run_category_btn.click(
                fn=run_category,
                inputs=category_dropdown,
                outputs=benchmark_output
            )
            run_all_btn.click(
                fn=lambda: run_all_benchmarks()[0],
                outputs=benchmark_output
            )
        
        # ============================================================
        # TAB 3: Full Report
        # ============================================================
        with gr.TabItem("📋 Full Report"):
            gr.Markdown("## Complete Benchmark Report")
            
            generate_report_btn = gr.Button("Generate Full Report", variant="primary")
            full_report_output = gr.Textbox(
                label="Detailed Report",
                lines=30,
                max_lines=50
            )
            
            generate_report_btn.click(
                fn=lambda: run_all_benchmarks()[1],
                outputs=full_report_output
            )
        
        # ============================================================
        # TAB 4: About
        # ============================================================
        with gr.TabItem("ℹ️ About"):
            about_text = f"""
            ## About This Demo
            
            This is the **Symbolic Transformer Phase 1.1** demo - a {config['model_size']} parameter model
            that has learned First-Order Logic syntax through next-token prediction.
            
            ### Key Capabilities Demonstrated
            
            | Capability | What It Means |
            |------------|---------------|
            | Quantifier Binding | FORALL/EXISTS must be followed by VAR |
            | Predicate Structure | PRED n must open parenthesis for arguments |
            | Calibrated Uncertainty | Model knows when multiple options are valid |
            | Parenthesis Tracking | Maintains structural state across sequence |
            | Compositional Generalisation | VAR 0 and VAR 624 treated identically |
            """
            
            if HAS_MESSAGE_BOUNDARIES:
                about_text += f"""| **Message Boundaries (NEW)** | MESSAGE_START/MESSAGE_END mark formula completion |
            """
            
            about_text += f"""
            
            ### Model Details
            
            - **Configuration**: {config['model_size']}
            - **Vocabulary**: {vocab.vocab_size} tokens
            - **Device**: {device}
            """
            
            if HAS_MESSAGE_BOUNDARIES:
                about_text += f"""- **Message Boundaries**: ✅ Enabled (START: {vocab.message_start}, END: {vocab.message_end})
            """
            else:
                about_text += """- **Message Boundaries**: ❌ Not in vocabulary (retrain with updated data generator)
            """
            
            about_text += """
            
            ### What's Next
            
            This syntax model is the foundation for **reasoning capabilities**:
            1. Equality substitution (Stage 1)
            2. Transitivity chains (Stage 2)
            3. Modus ponens (Stage 3)
            4. Multi-step deduction (Stage 4)
            
            ---
            
            *Built with compositional tokenization and careful thinking about what it means to reason.*
            """
            
            gr.Markdown(about_text)


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Launching Symbolic Transformer Demo with Benchmarks (v1.1)")
    if HAS_MESSAGE_BOUNDARIES:
        print("✓ Message boundary tests ENABLED")
    else:
        print("⚠ Message boundary tests DISABLED (not in vocabulary)")
    print("=" * 60 + "\n")
    demo.launch(share=True)