#!/usr/bin/env python
"""
Quick Test Script for DDoS Detection and Adversarial Evasion Experiment

This script validates that the defender-attacker framework works correctly
by running a quick experiment with the medium difficulty setting.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import create_synthetic_ddos_data, prepare_data_loaders
from src.models import create_model
from src.trainer import Trainer
from src.adversarial_gan import AdversarialAttacker, evaluate_evasion_success


def run_quick_experiment(difficulty='medium', n_samples=5000, epochs=10):
    """Run a quick experiment to validate the framework."""
    
    print("=" * 60)
    print(f"DDoS Detection & Adversarial Evasion Experiment")
    print(f"Difficulty: {difficulty}, Samples: {n_samples}, Epochs: {epochs}")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic DDoS traffic data...")
    X, y = create_synthetic_ddos_data(n_samples=n_samples, difficulty=difficulty)
    
    n_benign = (y == 0).sum()
    n_attack = (y == 1).sum()
    print(f"  Generated {n_samples} samples: {n_benign} benign, {n_attack} attack")
    
    # Step 2: Prepare data loaders
    print("\n[2/5] Preparing data loaders...")
    loaders = prepare_data_loaders(X, y, batch_size=64, model_type='cnn')
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val: {len(loaders['val'].dataset)} samples")
    print(f"  Test: {len(loaders['test'].dataset)} samples")
    
    # Step 3: Train CNN defender
    print(f"\n[3/5] Training CNN Defender for {epochs} epochs...")
    model = create_model('cnn', input_features=X.shape[1])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model, 
        device=device, 
        checkpoint_dir='./test_experiment_results',
        experiment_name='test_cnn'
    )
    
    # Handle class imbalance
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=epochs,
        learning_rate=0.001,
        class_weights=class_weights,
        patience=epochs + 5  # Disable early stopping for quick test
    )
    
    print(f"\n  Training Results:")
    print(f"    Final Train Accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"    Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"    Final Val F1 Score: {history['val_f1'][-1]:.4f}")
    print(f"    Final Val FPR: {history['val_fpr'][-1]:.6f}")
    
    # Step 4: Evaluate on test set
    print("\n[4/5] Evaluating on test set...")
    test_metrics = trainer.evaluate(loaders['test'])
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    
    # Step 5: Test adversarial attacks
    print("\n[5/5] Testing Adversarial Attacks (FGSM & PGD)...")
    
    # Get attack samples from test set
    X_test = loaders['test'].dataset.features.numpy()
    if X_test.ndim == 3:  # CNN format (batch, 1, features)
        X_test = X_test.squeeze(1)
    y_test = loaders['test'].dataset.labels.numpy()
    attack_samples = X_test[y_test == 1][:200]  # Take 200 attack samples
    
    if len(attack_samples) > 0:
        attacker = AdversarialAttacker(model, device=device)
        
        # Test FGSM
        print("\n  FGSM Attack:")
        adv_fgsm = attacker.generate_adversarial_examples(attack_samples, method='fgsm')
        fgsm_metrics = evaluate_evasion_success(model, attack_samples, adv_fgsm, device=device)
        print(f"    Original Detection Rate: {fgsm_metrics['original_detection_rate']:.4f}")
        print(f"    Adversarial Detection Rate: {fgsm_metrics['adversarial_detection_rate']:.4f}")
        print(f"    Evasion Success Rate: {fgsm_metrics['evasion_success_rate']:.4f}")
        
        # Test PGD
        print("\n  PGD Attack:")
        adv_pgd = attacker.generate_adversarial_examples(attack_samples, method='pgd')
        pgd_metrics = evaluate_evasion_success(model, attack_samples, adv_pgd, device=device)
        print(f"    Original Detection Rate: {pgd_metrics['original_detection_rate']:.4f}")
        print(f"    Adversarial Detection Rate: {pgd_metrics['adversarial_detection_rate']:.4f}")
        print(f"    Evasion Success Rate: {pgd_metrics['evasion_success_rate']:.4f}")
    else:
        print("  [Warning] No attack samples in test set for adversarial testing")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Difficulty: {difficulty}")
    print(f"Defender (CNN) Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Defender Test F1 Score: {test_metrics['f1']:.4f}")
    if len(attack_samples) > 0:
        print(f"Attacker (PGD) Evasion Success: {pgd_metrics['evasion_success_rate']:.4f}")
        print(f"Detection Rate Drop: {pgd_metrics['detection_rate_drop']:.4f}")
    
    return {
        'training_history': history,
        'test_metrics': test_metrics,
        'fgsm_metrics': fgsm_metrics if len(attack_samples) > 0 else None,
        'pgd_metrics': pgd_metrics if len(attack_samples) > 0 else None
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick experiment test')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'])
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=10)
    
    args = parser.parse_args()
    
    results = run_quick_experiment(
        difficulty=args.difficulty,
        n_samples=args.n_samples,
        epochs=args.epochs
    )
