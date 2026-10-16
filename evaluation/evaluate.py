"""
Evaluation script for Symbolic FOL Transformer.
Tests model on various metrics and generates sample completions.
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import SymbolicTransformer
from utils.vocabulary import Vocabulary
from training.train import FOLDataset, TrainingConfig


class Evaluator:
    """Evaluate trained model."""
    
    def __init__(
        self,
        model: SymbolicTransformer,
        vocab: Vocabulary,
        device: str = "cuda"
    ):
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def compute_perplexity(self, dataset: FOLDataset) -> float:
        """
        Compute perplexity on dataset.
        Lower is better (perfect prediction = perplexity 1.0).
        """
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                target = sample['target'].unsqueeze(0).to(self.device)
                last = int(sample['length']) - 1
                
                logits = self.model(input_ids)
                # Read logits at the last real token, not the last padded slot.
                loss = F.cross_entropy(logits[:, last, :], target)
                
                total_loss += loss.item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def compute_accuracy(self, dataset: FOLDataset, top_k: int = 1) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            dataset: Test dataset
            top_k: Consider prediction correct if target in top-k predictions
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                target = sample['target'].item()
                last = int(sample['length']) - 1
                
                logits = self.model(input_ids)
                top_k_preds = torch.topk(logits[0, last, :], k=top_k).indices.tolist()
                
                if target in top_k_preds:
                    correct += 1
                total += 1
        
        return correct / total
    
    def generate_completion(
        self,
        prompt: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 50
    ) -> Tuple[List[int], List[float]]:
        """
        Generate completion for prompt with confidence scores.
        
        Returns:
            (generated_tokens, confidence_scores)
        """
        generated = list(prompt)
        confidences = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[-1]] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                confidence = probs[next_token].item()
                
                generated.append(next_token)
                confidences.append(confidence)
                
                # Stop if we hit a natural stopping point
                # (You could add logic here based on token type)
        
        return generated[len(prompt):], confidences
    
    def evaluate_benchmark(self, test_dataset: FOLDataset) -> Dict:
        """
        Run complete evaluation benchmark.
        
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*60)
        print("EVALUATION BENCHMARK")
        print("="*60)
        
        # Perplexity
        print("\nComputing perplexity...")
        perplexity = self.compute_perplexity(test_dataset)
        print(f"✓ Perplexity: {perplexity:.4f}")
        
        # Accuracy metrics
        print("\nComputing accuracy...")
        top1_acc = self.compute_accuracy(test_dataset, top_k=1)
        top5_acc = self.compute_accuracy(test_dataset, top_k=5)
        top10_acc = self.compute_accuracy(test_dataset, top_k=10)
        
        print(f"✓ Top-1 Accuracy:  {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"✓ Top-5 Accuracy:  {top5_acc:.4f} ({top5_acc*100:.2f}%)")
        print(f"✓ Top-10 Accuracy: {top10_acc:.4f} ({top10_acc*100:.2f}%)")
        
        # Sample generations
        print("\nSample completions:")
        print("-" * 60)
        
        for i in range(min(5, len(test_dataset))):
            sample = test_dataset[i]
            length = int(sample['length'])
            prompt = sample['input_ids'].tolist()[:length][:10]
            target = sample['target'].item()
            
            generated, confidences = self.generate_completion(
                prompt, 
                max_new_tokens=10,
                temperature=0.7
            )
            
            # Decode
            prompt_str = self.vocab.decode_formula_simple(prompt)
            target_str = self.vocab.decode_id(target)
            generated_str = self.vocab.decode_formula_simple(generated)
            
            print(f"\nExample {i+1}:")
            print(f"  Prompt:    {prompt_str}")
            print(f"  Target:    {target_str}")
            print(f"  Generated: {generated_str}")
            print(f"  Avg Conf:  {sum(confidences)/len(confidences):.3f}")
        
        print("\n" + "="*60)
        
        return {
            'perplexity': perplexity,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'top10_accuracy': top10_acc
        }


def load_checkpoint(checkpoint_path: str, vocab_size: int, device: str = "cuda") -> SymbolicTransformer:
    """Load model from checkpoint (any embedding type)."""
    from models.transformer import load_model_from_checkpoint
    device = device if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, vocab_size, device)
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"✓ Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main():
    """Main evaluation function."""
    # Paths
    checkpoint_path = "checkpoints/best_model.pt"
    vocab_path = "unified_vocabulary.json"
    test_data_path = "datasets/fol_next_symbol/test.json"
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Vocabulary(vocab_path)
    
    # Load model
    print("\nLoading model...")
    model = load_checkpoint(checkpoint_path, vocab.vocab_size)
    
    # Load test data
    print("\nLoading test dataset...")
    pad_id = vocab.pad_token if vocab.pad_token is not None else 0
    test_dataset = FOLDataset(test_data_path, pad_id=pad_id)
    
    # Create evaluator
    evaluator = Evaluator(model, vocab)
    
    # Run evaluation
    metrics = evaluator.evaluate_benchmark(test_dataset)
    
    # Save results
    results_path = Path("evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
