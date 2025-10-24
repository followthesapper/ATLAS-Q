#!/usr/bin/env python
"""Train the AI rank predictor with substantial data for GPU utilization."""
import sys
from pathlib import Path

# Direct import to avoid __init__.py issues
import torch

# Import AI predictor directly
ai_predictor_path = Path(__file__).parent.parent / 'src' / 'quantum_hybrid_system' / 'tools_qih' / 'ai_rank_predictor.py'
import importlib.util
spec = importlib.util.spec_from_file_location("ai_rank_predictor", ai_predictor_path)
ai_predictor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_predictor_module)

RankPredictorWrapper = ai_predictor_module.RankPredictorWrapper
generate_training_data = ai_predictor_module.generate_training_data

def main():
    print("=" * 70)
    print("Training AI Rank Predictor for Tensor Network Compression")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Generate substantial training data for GPU utilization
    print("\n" + "=" * 70)
    print("Generating Training Data")
    print("=" * 70)
    train_loader, val_loader = generate_training_data(
        n_samples=50000,  # Much larger dataset
        max_rank=2048,
        device=device,
        batch_size=1024  # Large batches to utilize GPU
    )
    
    # Train predictor
    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)
    predictor = RankPredictorWrapper(device=device)
    predictor.train(train_loader, val_loader, epochs=100, lr=1e-3)
    
    # Save model
    model_path = models_dir / 'rank_predictor.pt'
    predictor.save(str(model_path))
    
    # Test predictions
    print("\n" + "=" * 70)
    print("Testing Predictions")
    print("=" * 70)
    test_cases = [
        ("Slow decay", torch.exp(-torch.linspace(0, 2, 100))),
        ("Medium decay", torch.exp(-torch.linspace(0, 4, 200))),
        ("Fast decay", torch.exp(-torch.linspace(0, 8, 500))),
        ("Very fast decay", torch.exp(-torch.linspace(0, 12, 1000))),
    ]
    
    for name, test_sigmas in test_cases:
        test_sigmas = test_sigmas.to(device)
        predicted_rank = predictor.predict(test_sigmas)
        print(f"  {name:20s}: {predicted_rank:4d}/{len(test_sigmas):4d} SVs ({100*predicted_rank/len(test_sigmas):.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Model saved to: {model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
