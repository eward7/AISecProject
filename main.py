#!/usr/bin/env python
"""
Main entry point for DDoS Detection and Adversarial Evasion Research

This script provides a command-line interface for:
- Training detection models
- Evaluating model performance
- Running adversarial attacks
- Generating reports
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def train_detector(args):
    """Train a DDoS detection model."""
    from src.data_loader import (
        create_synthetic_ddos_data, load_cicddos2019,
        DataPreprocessor, prepare_data_loaders
    )
    from src.models import create_model
    from src.trainer import Trainer
    import torch
    import numpy as np
    
    print("=" * 60)
    print("DDoS Detection Model Training")
    print("=" * 60)
    
    # Load data
    if args.data_path:
        print(f"\nLoading data from {args.data_path}...")
        df = load_cicddos2019(args.data_path, sample_size=args.sample_size)
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df)
        
        # Save preprocessor
        preprocessor.save(Path(args.output_dir) / 'preprocessor.joblib')
    else:
        difficulty = getattr(args, 'difficulty', 'medium')
        print(f"\nUsing synthetic data for training (difficulty: {difficulty})...")
        X, y = create_synthetic_ddos_data(
            n_samples=args.n_samples or 50000,
            attack_ratio=0.3,
            difficulty=difficulty
        )
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Prepare data loaders
    loaders = prepare_data_loaders(
        X, y,
        batch_size=args.batch_size,
        model_type=args.model_type
    )
    
    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    model = create_model(args.model_type, input_features=X.shape[1])
    
    # Create trainer
    trainer = Trainer(
        model,
        device=args.device,
        checkpoint_dir=args.output_dir,
        experiment_name=f"{args.model_type}_detector"
    )
    
    # Handle class imbalance
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        patience=args.patience
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)
    test_metrics = trainer.evaluate(loaders['test'])
    
    # Save results
    results = {
        'model_type': args.model_type,
        'training_history': history,
        'test_metrics': test_metrics,
        'model_path': str(trainer.checkpoint_dir / 'best_model.pt')
    }
    
    results_path = Path(args.output_dir) / f"{args.model_type}_detector" / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {results['model_path']}")


def evaluate_model(args):
    """Evaluate a trained detection model."""
    from src.data_loader import create_synthetic_ddos_data, DataPreprocessor, load_cicddos2019
    from src.models import load_pretrained_model
    from src.evaluation import run_full_evaluation
    import numpy as np
    
    print("=" * 60)
    print("DDoS Detection Model Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_pretrained_model(args.model_path, args.model_type, args.device)
    
    # Load data
    if args.data_path:
        print(f"Loading test data from {args.data_path}...")
        df = load_cicddos2019(args.data_path)
        preprocessor = DataPreprocessor()
        
        # Load preprocessor if available
        preprocessor_path = Path(args.model_path).parent / 'preprocessor.joblib'
        if preprocessor_path.exists():
            preprocessor.load(str(preprocessor_path))
            X, y = preprocessor.transform(df)
        else:
            X, y = preprocessor.fit_transform(df)
    else:
        difficulty = getattr(args, 'difficulty', 'medium')
        print(f"Using synthetic data for evaluation (difficulty: {difficulty})...")
        X, y = create_synthetic_ddos_data(
            n_samples=args.n_samples or 10000,
            difficulty=difficulty
        )
    
    # Run evaluation
    results = run_full_evaluation(
        model, X, y,
        output_dir=args.output_dir,
        model_name=f"{args.model_type}_detector",
        run_adversarial=args.run_adversarial,
        run_gan=args.run_gan
    )
    
    print("\nEvaluation complete!")


def run_adversarial(args):
    """Run adversarial attack experiments."""
    from src.data_loader import create_synthetic_ddos_data
    from src.models import load_pretrained_model
    from src.adversarial_gan import AdversarialGAN, AdversarialAttacker, evaluate_evasion_success
    import numpy as np
    
    print("=" * 60)
    print("Adversarial Attack Experiments")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading target model from {args.model_path}...")
    model = load_pretrained_model(args.model_path, args.model_type, args.device)
    
    # Load data
    difficulty = getattr(args, 'difficulty', 'medium')
    print(f"Loading data (difficulty: {difficulty})...")
    X, y = create_synthetic_ddos_data(
        n_samples=args.n_samples or 10000,
        difficulty=difficulty
    )
    attack_data = X[y == 1]
    benign_data = X[y == 0]
    
    print(f"Attack samples: {len(attack_data)}")
    print(f"Benign samples: {len(benign_data)}")
    
    # Create attacker
    attacker = AdversarialAttacker(model, device=args.device)
    
    results = {}
    
    # Test different attack methods
    for method in ['fgsm', 'pgd']:
        print(f"\n{'=' * 40}")
        print(f"Testing {method.upper()} Attack")
        print('=' * 40)
        
        for epsilon in [0.05, 0.1, 0.2]:
            print(f"\nEpsilon = {epsilon}")
            adv_examples = attacker.generate_adversarial_examples(
                attack_data[:500], method=method
            )
            metrics = evaluate_evasion_success(
                model, attack_data[:500], adv_examples, device=args.device
            )
            results[f'{method}_eps_{epsilon}'] = metrics
    
    # Train GAN-based attack
    if args.run_gan:
        print(f"\n{'=' * 40}")
        print("Training GAN-based Attack")
        print('=' * 40)
        
        gan = AdversarialGAN(feature_dim=X.shape[1], device=args.device)
        gan.train(benign_data, epochs=args.gan_epochs)
        
        # Train perturbation generator
        attacker.train_perturbation_generator(
            attack_data, np.ones(len(attack_data)),
            epochs=args.gan_epochs
        )
        
        adv_learned = attacker.generate_adversarial_examples(
            attack_data[:500], method='learned'
        )
        metrics = evaluate_evasion_success(
            model, attack_data[:500], adv_learned, device=args.device
        )
        results['learned_perturbation'] = metrics
    
    # Save results
    results_path = Path(args.output_dir) / 'adversarial_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


def run_simulation(args):
    """Run network simulation with attack scenarios."""
    from src.network_simulator import SimulatedNetwork, AttackScenario
    import json
    
    print("=" * 60)
    print("Network Simulation")
    print("=" * 60)
    
    # Create network
    network = SimulatedNetwork()
    scenario = AttackScenario(network)
    
    # Run scenarios
    scenarios = ['syn_flood', 'udp_flood', 'http_flood', 'dns_amplification', 'mixed']
    results = {}
    
    for scenario_name in scenarios:
        print(f"\n{'=' * 40}")
        print(f"Running {scenario_name} scenario")
        print('=' * 40)
        
        network.clear()
        stats = scenario.run_scenario(
            scenario_name,
            duration=args.duration,
            attack_start=args.duration * 0.3,
            attack_duration=args.duration * 0.4
        )
        results[scenario_name] = stats
        
        print(f"Total packets: {stats['total_packets']}")
        print(f"Attack packets: {stats['attack_packets']}")
        print(f"Total flows: {stats['total_flows']}")
    
    # Export flow data
    if args.export_flows:
        export_path = Path(args.output_dir) / 'simulated_flows.csv'
        network.export_flows_to_csv(str(export_path))
        print(f"\nFlows exported to {export_path}")
    
    # Save results
    results_path = Path(args.output_dir) / 'simulation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='DDoS Detection and Adversarial Evasion Research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a CNN detector:
    python main.py train --model-type cnn --epochs 100

  Evaluate a model:
    python main.py evaluate --model-path ./results/cnn_detector/best_model.pt

  Run adversarial attacks:
    python main.py adversarial --model-path ./results/cnn_detector/best_model.pt

  Run network simulation:
    python main.py simulate --duration 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a detection model')
    train_parser.add_argument('--model-type', type=str, default='cnn',
                              choices=['cnn', 'lstm', 'transformer', 'hybrid'],
                              help='Type of model to train')
    train_parser.add_argument('--data-path', type=str, default=None,
                              help='Path to CICDDoS2019 dataset')
    train_parser.add_argument('--n-samples', type=int, default=50000,
                              help='Number of synthetic samples')
    train_parser.add_argument('--sample-size', type=int, default=None,
                              help='Sample size per file')
    train_parser.add_argument('--difficulty', type=str, default='medium',
                              choices=['easy', 'medium', 'hard'],
                              help='Difficulty of synthetic data classification (easy/medium/hard)')
    train_parser.add_argument('--epochs', type=int, default=100,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=64,
                              help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3,
                              help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=15,
                              help='Early stopping patience')
    train_parser.add_argument('--device', type=str, default='auto',
                              help='Device to use')
    train_parser.add_argument('--output-dir', type=str, default='./results',
                              help='Output directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--model-type', type=str, default='cnn',
                             choices=['cnn', 'lstm', 'transformer', 'hybrid'],
                             help='Type of model')
    eval_parser.add_argument('--data-path', type=str, default=None,
                             help='Path to test data')
    eval_parser.add_argument('--n-samples', type=int, default=10000,
                             help='Number of synthetic samples')
    eval_parser.add_argument('--difficulty', type=str, default='medium',
                             choices=['easy', 'medium', 'hard'],
                             help='Difficulty of synthetic data classification')
    eval_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use')
    eval_parser.add_argument('--output-dir', type=str, default='./results',
                             help='Output directory')
    eval_parser.add_argument('--run-adversarial', action='store_true',
                             help='Run adversarial evaluation')
    eval_parser.add_argument('--run-gan', action='store_true',
                             help='Run GAN evaluation')
    
    # Adversarial command
    adv_parser = subparsers.add_parser('adversarial', help='Run adversarial attacks')
    adv_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to model checkpoint')
    adv_parser.add_argument('--model-type', type=str, default='cnn',
                            choices=['cnn', 'lstm', 'transformer', 'hybrid'],
                            help='Type of model')
    adv_parser.add_argument('--n-samples', type=int, default=10000,
                            help='Number of synthetic samples')
    adv_parser.add_argument('--difficulty', type=str, default='medium',
                            choices=['easy', 'medium', 'hard'],
                            help='Difficulty of synthetic data classification')
    adv_parser.add_argument('--device', type=str, default='auto',
                            help='Device to use')
    adv_parser.add_argument('--output-dir', type=str, default='./results',
                            help='Output directory')
    adv_parser.add_argument('--run-gan', action='store_true',
                            help='Run GAN-based attacks')
    adv_parser.add_argument('--gan-epochs', type=int, default=50,
                            help='GAN training epochs')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run network simulation')
    sim_parser.add_argument('--duration', type=float, default=60.0,
                            help='Simulation duration in seconds')
    sim_parser.add_argument('--output-dir', type=str, default='./results',
                            help='Output directory')
    sim_parser.add_argument('--export-flows', action='store_true',
                            help='Export flow data to CSV')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_detector(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'adversarial':
        run_adversarial(args)
    elif args.command == 'simulate':
        run_simulation(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
