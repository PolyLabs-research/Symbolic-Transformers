import gradio as gr
import torch
import torch.nn.functional as F
import sys
import os

# Ensure we can import from the current directory
sys.path.insert(0, os.getcwd())

from utils.vocabulary import Vocabulary
from models.transformer import create_model

# 1. SETUP & LOAD MODEL (Global scope for speed)
print("🔄 Initializing GUI...")
if not os.path.exists('unified_vocabulary.json'):
    raise FileNotFoundError("unified_vocabulary.json not found")

vocab = Vocabulary('unified_vocabulary.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('checkpoints/best_model.pt'):
    raise FileNotFoundError("checkpoints/best_model.pt not found")

checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
config = checkpoint['config']

model = create_model(vocab_size=vocab.vocab_size, model_size=config['model_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# 2. PREDICTION FUNCTION
def parse_tokens(sequence_text):
    """Parse 'FORALL VAR 1' into token IDs.

    Numerals must be 0-624 (anything else would index past the embedding
    table). MESSAGE_START is prepended because every training context starts
    with it; type it explicitly if you want to control that yourself.
    """
    token_ids = []
    for t in sequence_text.strip().split():
        if t in vocab.label_to_id:
            token_ids.append(vocab.encode_label(t))
        elif t.isdigit() and 0 <= int(t) <= vocab.numeral_range[1]:
            token_ids.append(int(t))
        else:
            raise ValueError(f"Unknown token {t!r} (labels or numerals 0-{vocab.numeral_range[1]} only)")
    if vocab.message_start is not None and (not token_ids or token_ids[0] != vocab.message_start):
        token_ids = [vocab.message_start] + token_ids
    return token_ids


def predict_next_token(sequence_text):
    """
    Takes a string like "FORALL VAR", returns a dict of top 5 next tokens.
    """
    if not sequence_text.strip():
        return None

    try:
        token_ids = parse_tokens(sequence_text)
    except ValueError as exc:
        return {str(exc): 0.0}

    # Prepare tensor
    x = torch.tensor([token_ids], device=device, dtype=torch.long)

    # Inference
    with torch.no_grad():
        logits = model(x)
        # We only care about the prediction for the *last* token in the sequence
        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
    
    # Get Top 10 for visualization
    top_probs, top_ids = torch.topk(probs, 10)
    
    # Format for Gradio (Label component expects {label: confidence})
    results = {}
    for prob, tid in zip(top_probs, top_ids):
        label = vocab.decode_id(tid.item())
        results[label] = float(prob)
        
    return results

# 3. DEFINE INTERFACE
description = """
### Symbolic Transformer Explorer
Type a First-Order Logic formula prefix to see what the model predicts next.
MESSAGE_START is added automatically (every training formula begins with it).
**Examples:**
* `FORALL` (Should predict VAR)
* `PRED 1 LPAREN` (Should predict VAR or FUNC)
* `VAR 1 EQUALS` (Should predict VAR)
"""

demo = gr.Interface(
    fn=predict_next_token,
    inputs=gr.Textbox(
        label="Input Context", 
        placeholder="e.g., FORALL VAR 1",
        lines=1
    ),
    outputs=gr.Label(
        num_top_classes=5, 
        label="Next Token Probability"
    ),
    title="🧠 Symbolic Transformer Phase 1",
    description=description,
    live=True, # Updates instantly as you type!
)

if __name__ == "__main__":
    # share=True creates a public link you can access from your phone/browser
    demo.launch(share=True)