#!/usr/bin/env python
"""
Train on Real CICDDoS2019 Dataset with Proper Stratification

This script trains all models on the actual CICDDoS2019 dataset with:
- Stratified sampling across attack types
- Balanced class distribution
- Robust preprocessing and evaluation

Usage:
    python train_real_data.py --data-path data/01-12 --n-samples 100000 --epochs 30
"""

import sys
import os
import json
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_cicddos2019, DataPreprocessor, prepare_data_loaders
from src.models import create_model
from src.trainer import Trainer
from src.adversarial_gan import AdversarialAttacker, evaluate_evasion_success


def analyze_data_distribution(df: pd.DataFrame, detailed: bool = True) -> dict:
    """Analyze and log data distribution."""
    label_col = 'Label' if 'Label' in df.columns else ' Label'
    
    label_counts = df[label_col].value_counts().sort_values(ascending=False)
    
    print("\n" + "=" * 60)
    print("Data Distribution Analysis")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"\nClass distribution (sorted by frequency):")
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {label:20s}: {count:10,} ({pct:5.2f}%)")
    
    # Check for benign vs attack balance
    benign_count = label_counts.get('BENIGN', 0)
    attack_count = len(df) - benign_count
    print(f"\nBinary classification:")
    print(f"  BENIGN: {benign_count:,} ({benign_count/len(df)*100:.2f}%)")
    print(f"  ATTACK: {attack_count:,} ({attack_count/len(df)*100:.2f}%)")
    
    if detailed:
        # Show attack type diversity
        attack_types = [label for label in label_counts.index if label != 'BENIGN']
        print(f"\nAttack types present: {len(attack_types)}")
        print(f"Attack types: {', '.join(attack_types)}")
    
    return {
        'total_samples': len(df),
        'label_distribution': label_counts.to_dict(),
        'benign_count': benign_count,
        'attack_count': attack_count,
        'imbalance_ratio': benign_count / max(attack_count, 1),
        'num_attack_types': len([l for l in label_counts.index if l != 'BENIGN'])
    }


def train_and_evaluate_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    device: str,
    output_dir: str,
    class_weights: torch.Tensor = None
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
        test_size=0.2,
        val_size=0.1
    )
    
    # Create model
    model = create_model(model_type, input_features=X.shape[1])
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create trainer
    experiment_name = f"{model_type}_cicddos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model,
        device=device,
        checkpoint_dir=output_dir,
        experiment_name=experiment_name
    )
    
    # Train
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=epochs,
        learning_rate=1e-3,
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
    
    if hasattr(dataset, 'features'):
        X_test = dataset.features.numpy()
        y_test = dataset.labels.numpy()
    elif hasattr(dataset, 'sequences'):
        X_test = dataset.sequences.numpy()
        y_test = dataset.sequence_labels.numpy()
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")
    
    if X_test.ndim == 3 and model_type == 'cnn':
        X_test = X_test.squeeze(1)
    
    # For sequential models, use last timestep for adversarial attack
    if model_type in ['lstm', 'transformer']:
        if X_test.ndim == 3:
            attack_samples = X_test[y_test == 1][:300, -1, :]
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
        'training_time_minutes': training_time / 60,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'fpr': test_metrics.get('fpr', 0)
        },
        'adversarial_metrics': adv_metrics,
        'history': history
    }
    
    # Print summary
    print(f"\n{model_type.upper()} Results:")
    print(f"  Training Time: {training_time/60:.2f} minutes")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Test FPR: {test_metrics.get('fpr', 0):.6f}")
    if adv_metrics.get('pgd'):
        print(f"  PGD Evasion Rate: {adv_metrics['pgd']['evasion_success_rate']:.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models on CICDDoS2019 dataset')
    parser.add_argument('--data-path', type=str, required=True, help='Path to CICDDoS data directory')
    parser.add_argument('--n-samples', type=int, default=100000, help='Total samples to load (stratified)')
    parser.add_argument('--balance-classes', action='store_true', help='Balance benign/attack to 50/50')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--models', type=str, nargs='+', default=['cnn', 'lstm', 'hybrid'],
                       help='Models to train (cnn, lstm, transformer, hybrid)')
    parser.add_argument('--output-dir', type=str, default='results_cicddos', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("CICDDoS2019 Model Training Experiment")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Target samples: {args.n_samples:,}")
    print(f"Balance classes: {args.balance_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Models: {', '.join(args.models)}")
    
    # Load data with stratification
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    df = load_cicddos2019(
        args.data_path,
        sample_size=args.n_samples,
        stratified=True,
        balance_classes=args.balance_classes
    )
    
    # Analyze distribution
    distribution_stats = analyze_data_distribution(df)
    
    # Preprocess data
    print("\n" + "=" * 60)
    print("Preprocessing Data")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution after preprocessing:")
    print(f"  Benign: {np.sum(y == 0):,}")
    print(f"  Attack: {np.sum(y == 1):,}")
    
    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor.joblib')
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    print(f"\nClass weights: Benign={class_weights[0]:.4f}, Attack={class_weights[1]:.4f}")
    
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
                output_dir=str(output_dir),
                class_weights=class_weights
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"\nERROR training {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save comprehensive results
    summary_file = output_dir / 'comparison_summary.json'
    
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(summary_file, 'w') as f:
        json.dump({
            'data_stats': convert_to_serializable(distribution_stats),
            'model_results': convert_to_serializable(all_results),
            'config': vars(args)
        }, f, indent=2)
    
    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<15} {'Params':<12} {'Test Acc':<10} {'Test F1':<10} {'FPR':<12} {'PGD Evasion':<12} {'Time (min)':<10}")
    print("-" * 80)
    
    best_f1 = 0
    best_model = None
    
    for model_type, results in all_results.items():
        metrics = results['test_metrics']
        adv = results['adversarial_metrics'].get('pgd')
        pgd_evasion = adv['evasion_success_rate'] if adv else float('nan')
        
        print(f"{model_type:<15} {results['n_parameters']:<12,} {metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['fpr']:<12.6f} {pgd_evasion:<12.4f} "
              f"{results['training_time_minutes']:<10.2f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model_type
    
    print("-" * 80)
    if best_model:
        print(f"\nBest Model: {best_model.upper()} (F1 Score: {best_f1:.4f})")
    print(f"\nResults saved to: {summary_file}")


if __name__ == '__main__':
    main()
