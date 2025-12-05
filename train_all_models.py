#!/usr/bin/env python
"""
Full Training Experiment: Compare All Models on Hard Difficulty

This script trains CNN, LSTM, Transformer, and Hybrid models on 
synthetic DDoS data with hard difficulty and compares their performance.

Usage:
    python train_all_models.py --epochs 50 --n-samples 50000
"""

import sys
import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import create_synthetic_ddos_data, prepare_data_loaders
from src.models import create_model
from src.trainer import Trainer
from src.adversarial_gan import AdversarialAttacker, evaluate_evasion_success


def train_and_evaluate_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    device: str,
    output_dir: str
):
    """Train and evaluate a single model."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Prepare data loaders
    loaders = prepare_data_loaders(
        X, y, 
        batch_size=64, 
        model_type=model_type,
        sequence_length=10 if model_type in ['lstm', 'transformer'] else 1
    )
    
    # Create model
    model = create_model(model_type, input_features=X.shape[1])
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create trainer
    experiment_name = f"{model_type}_hard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model,
        device=device,
        checkpoint_dir=output_dir,
        experiment_name=experiment_name
    )
    
    # Handle class imbalance
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    
    # Train
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=epochs,
        learning_rate=0.001,
        class_weights=class_weights,
        patience=15
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print(f"\nEvaluating {model_type.upper()} on test set...")
    test_metrics = trainer.evaluate(loaders['test'])
    
    # Adversarial evaluation
    print(f"\nAdversarial evaluation for {model_type.upper()}...")
    
    # Get attack samples - handle different dataset types
    dataset = loaders['test'].dataset
    
    # Handle different dataset types
    if hasattr(dataset, 'features'):
        # DDoSDataset
        X_test = dataset.features.numpy()
        y_test = dataset.labels.numpy()
    elif hasattr(dataset, 'sequences'):
        # SequentialDataset
        X_test = dataset.sequences.numpy()
        y_test = dataset.sequence_labels.numpy()
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")
    
    if X_test.ndim == 3 and model_type == 'cnn':
        X_test = X_test.squeeze(1)
    
    # For sequential models, use last timestep for adversarial attack
    if model_type in ['lstm', 'transformer']:
        # Get last timestep features for adversarial generation
        if X_test.ndim == 3:
            attack_samples = X_test[y_test == 1][:300, -1, :]  # Last timestep
        else:
            attack_samples = X_test[y_test == 1][:300]
    else:
        attack_samples = X_test[y_test == 1][:300]
    
    adv_metrics = {}
    if len(attack_samples) > 0:
        attacker = AdversarialAttacker(model, device=device)
        
        # FGSM
        try:
            adv_fgsm = attacker.generate_adversarial_examples(attack_samples, method='fgsm')
            adv_metrics['fgsm'] = evaluate_evasion_success(model, attack_samples, adv_fgsm, device=device)
        except Exception as e:
            print(f"  FGSM failed: {e}")
            adv_metrics['fgsm'] = None
        
        # PGD
        try:
            adv_pgd = attacker.generate_adversarial_examples(attack_samples, method='pgd')
            adv_metrics['pgd'] = evaluate_evasion_success(model, attack_samples, adv_pgd, device=device)
        except Exception as e:
            print(f"  PGD failed: {e}")
            adv_metrics['pgd'] = None
    
    results = {
        'model_type': model_type,
        'n_parameters': n_params,
        'training_time_seconds': training_time,
        'epochs_trained': len(history['train_loss']),
        'final_train_accuracy': history['train_accuracy'][-1],
        'final_val_accuracy': history['val_accuracy'][-1],
        'final_val_f1': history['val_f1'][-1],
        'test_metrics': test_metrics,
        'adversarial_metrics': adv_metrics,
        'checkpoint_dir': str(trainer.checkpoint_dir)
    }
    
    # Save results
    results_path = trainer.checkpoint_dir / 'full_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{model_type.upper()} Results:")
    print(f"  Training Time: {training_time/60:.2f} minutes")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Test FPR: {test_metrics['fpr']:.6f}")
    if adv_metrics.get('pgd'):
        print(f"  PGD Evasion Rate: {adv_metrics['pgd']['evasion_success_rate']:.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models on hard difficulty')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--n-samples', type=int, default=50000, help='Number of samples')
    parser.add_argument('--difficulty', type=str, default='hard', 
                       choices=['easy', 'medium', 'hard'])
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./results_comparison',
                       help='Output directory')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['cnn', 'lstm', 'transformer', 'hybrid'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    # Check GPU
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    print("="*60)
    print("DDoS Detection Model Comparison Experiment")
    print("="*60)
    print(f"Difficulty: {args.difficulty}")
    print(f"Samples: {args.n_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Models: {', '.join(args.models)}")
    
    # Generate data once
    print(f"\nGenerating synthetic data (difficulty={args.difficulty})...")
    X, y = create_synthetic_ddos_data(
        n_samples=args.n_samples,
        difficulty=args.difficulty
    )
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: Benign={np.sum(y==0)}, Attack={np.sum(y==1)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    all_results = {}
    
    for model_type in args.models:
        try:
            results = train_and_evaluate_model(
                model_type=model_type,
                X=X,
                y=y,
                epochs=args.epochs,
                device=args.device,
                output_dir=str(output_dir)
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"\nERROR training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_type] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<12} {'Params':<12} {'Test Acc':<10} {'Test F1':<10} {'FPR':<10} {'PGD Evasion':<12} {'Time (min)':<10}")
    print("-"*80)
    
    best_model = None
    best_f1 = 0
    
    for model_type, results in all_results.items():
        if 'error' in results:
            print(f"{model_type:<12} ERROR: {results['error'][:50]}")
            continue
            
        params = f"{results['n_parameters']:,}"
        test_acc = f"{results['test_metrics']['accuracy']:.4f}"
        test_f1 = f"{results['test_metrics']['f1']:.4f}"
        fpr = f"{results['test_metrics']['fpr']:.6f}"
        
        pgd_evasion = "N/A"
        if results['adversarial_metrics'].get('pgd'):
            pgd_evasion = f"{results['adversarial_metrics']['pgd']['evasion_success_rate']:.4f}"
        
        time_min = f"{results['training_time_seconds']/60:.2f}"
        
        print(f"{model_type:<12} {params:<12} {test_acc:<10} {test_f1:<10} {fpr:<10} {pgd_evasion:<12} {time_min:<10}")
        
        if results['test_metrics']['f1'] > best_f1:
            best_f1 = results['test_metrics']['f1']
            best_model = model_type
    
    print("-"*80)
    print(f"\nBest Model: {best_model.upper()} (F1 Score: {best_f1:.4f})")
    
    # Save overall results
    summary_path = output_dir / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment_config': {
                'difficulty': args.difficulty,
                'n_samples': args.n_samples,
                'epochs': args.epochs,
                'device': args.device
            },
            'results': all_results,
            'best_model': best_model,
            'best_f1': best_f1
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {summary_path}")
    
    return all_results


if __name__ == '__main__':
    main()
