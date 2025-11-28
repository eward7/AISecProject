"""
Comprehensive Evaluation Module for DDoS Detection and Adversarial Attacks

This module provides tools for:
- Baseline model evaluation
- Adversarial robustness testing
- Comprehensive reporting
- Visualization of results
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """
    Comprehensive evaluator for DDoS detection models.
    
    Measures baseline performance including:
    - Accuracy, Precision, Recall, F1
    - False Positive Rate (FPR)
    - Inference latency
    - ROC and PR curves
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Detection model to evaluate
            device: Device to use
        """
        self.model = model
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Evaluate baseline detection performance.
        
        Args:
            X: Test features
            y: True labels
            batch_size: Batch size for inference
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating baseline detection performance...")
        
        # Get predictions
        predictions, probabilities = self._get_predictions(X, batch_size)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
        }
        
        # Calculate confusion matrix and derived metrics
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # False Positive Rate
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # True Positive Rate (same as recall)
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC AUC
        if probabilities is not None and len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, probabilities[:, 1])
            metrics['roc_auc'] = auc(fpr, tpr)
            
            # PR AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y, probabilities[:, 1])
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        logger.info(f"Baseline Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        
        return metrics
    
    def measure_latency(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        n_warmup: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference latency.
        
        Args:
            X: Sample features
            n_samples: Number of samples for measurement
            n_warmup: Number of warmup iterations
            
        Returns:
            Latency statistics
        """
        logger.info("Measuring inference latency...")
        
        self.model.eval()
        x = torch.FloatTensor(X[:1]).to(self.device)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(x)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for i in range(min(n_samples, len(X))):
                sample = torch.FloatTensor(X[i:i+1]).to(self.device)
                if sample.dim() == 2:
                    sample = sample.unsqueeze(1)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(sample)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append(end - start)
        
        latencies = np.array(latencies) * 1000  # Convert to ms
        
        metrics = {
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'throughput_per_sec': float(1000 / np.mean(latencies))
        }
        
        logger.info(f"Latency Metrics:")
        logger.info(f"  Mean: {metrics['latency_mean_ms']:.3f} ms")
        logger.info(f"  P95: {metrics['latency_p95_ms']:.3f} ms")
        logger.info(f"  P99: {metrics['latency_p99_ms']:.3f} ms")
        logger.info(f"  Throughput: {metrics['throughput_per_sec']:.1f} samples/sec")
        
        return metrics
    
    def _get_predictions(
        self,
        X: np.ndarray,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions."""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                
                if batch.dim() == 2:
                    batch = batch.unsqueeze(1)
                
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)


class AdversarialEvaluator:
    """
    Evaluator for adversarial robustness.
    
    Tests model robustness against various adversarial attacks
    and evasion techniques.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto'
    ):
        """
        Initialize adversarial evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_robustness(
        self,
        attack_data: np.ndarray,
        attack_method: str = 'all',
        epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3]
    ) -> Dict[str, Dict]:
        """
        Evaluate model robustness against adversarial attacks.
        
        Args:
            attack_data: Attack traffic features
            attack_method: Attack method ('fgsm', 'pgd', 'all')
            epsilon_values: List of perturbation magnitudes to test
            
        Returns:
            Robustness metrics for each attack and epsilon
        """
        from .adversarial_gan import AdversarialAttacker
        
        attacker = AdversarialAttacker(self.model, device=str(self.device))
        
        methods = ['fgsm', 'pgd'] if attack_method == 'all' else [attack_method]
        results = {}
        
        for method in methods:
            method_results = {}
            logger.info(f"\nEvaluating {method.upper()} attack...")
            
            for epsilon in epsilon_values:
                logger.info(f"  Testing epsilon = {epsilon}")
                
                # Get original detection rate
                orig_preds, _ = self._get_predictions(attack_data)
                orig_detection_rate = np.mean(orig_preds == 1)
                
                # Generate adversarial examples
                x = torch.FloatTensor(attack_data).to(self.device)
                y = torch.ones(len(attack_data), dtype=torch.long).to(self.device)
                
                if method == 'fgsm':
                    x_adv = attacker.fgsm_attack(x, y, epsilon=epsilon)
                else:
                    x_adv = attacker.pgd_attack(x, y, epsilon=epsilon)
                
                # Evaluate adversarial examples
                adv_preds, _ = self._get_predictions(x_adv.cpu().numpy())
                adv_detection_rate = np.mean(adv_preds == 1)
                evasion_rate = np.mean(adv_preds == 0)
                
                # Calculate perturbation magnitude
                perturbation = x_adv.cpu().numpy() - attack_data
                l2_norm = np.mean(np.linalg.norm(perturbation, axis=1))
                linf_norm = np.mean(np.abs(perturbation).max(axis=1))
                
                method_results[epsilon] = {
                    'original_detection_rate': float(orig_detection_rate),
                    'adversarial_detection_rate': float(adv_detection_rate),
                    'evasion_rate': float(evasion_rate),
                    'detection_drop': float(orig_detection_rate - adv_detection_rate),
                    'mean_l2_perturbation': float(l2_norm),
                    'mean_linf_perturbation': float(linf_norm)
                }
            
            results[method] = method_results
        
        return results
    
    def evaluate_gan_evasion(
        self,
        attack_data: np.ndarray,
        benign_data: np.ndarray,
        gan_epochs: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate GAN-based evasion attacks.
        
        Args:
            attack_data: Attack traffic features
            benign_data: Benign traffic features for GAN training
            gan_epochs: Number of GAN training epochs
            
        Returns:
            GAN evasion metrics
        """
        from .adversarial_gan import AdversarialGAN, AdversarialAttacker
        
        logger.info("Training GAN for evasion attack...")
        
        # Train GAN on benign data
        gan = AdversarialGAN(feature_dim=benign_data.shape[1], device=str(self.device))
        gan.train(benign_data, epochs=gan_epochs)
        
        # Generate synthetic benign-like traffic
        synthetic_traffic = gan.generate(len(attack_data))
        
        # Evaluate synthetic traffic
        synth_preds, _ = self._get_predictions(synthetic_traffic)
        
        # Original attack detection
        orig_preds, _ = self._get_predictions(attack_data)
        
        # Train perturbation generator
        logger.info("Training perturbation generator...")
        attacker = AdversarialAttacker(self.model, device=str(self.device))
        attacker.train_perturbation_generator(
            attack_data, np.ones(len(attack_data)),
            epochs=gan_epochs
        )
        
        # Generate adversarial examples using learned perturbations
        adv_learned = attacker.generate_adversarial_examples(attack_data, method='learned')
        learned_preds, _ = self._get_predictions(adv_learned)
        
        metrics = {
            'synthetic_benign_accuracy': float(np.mean(synth_preds == 0)),
            'original_attack_detection': float(np.mean(orig_preds == 1)),
            'learned_perturbation_evasion': float(np.mean(learned_preds == 0)),
            'detection_rate_drop': float(np.mean(orig_preds == 1) - np.mean(learned_preds == 1))
        }
        
        logger.info(f"GAN Evasion Results:")
        logger.info(f"  Synthetic Benign Accuracy: {metrics['synthetic_benign_accuracy']:.4f}")
        logger.info(f"  Original Attack Detection: {metrics['original_attack_detection']:.4f}")
        logger.info(f"  Learned Perturbation Evasion: {metrics['learned_perturbation_evasion']:.4f}")
        
        return metrics
    
    def _get_predictions(
        self,
        X: np.ndarray,
        batch_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions."""
        self.model.eval()
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                
                if batch.dim() == 2:
                    batch = batch.unsqueeze(1)
                
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)


class ComprehensiveReport:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self, output_dir: str = './results'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_data = {}
    
    def add_baseline_results(self, results: Dict[str, float]):
        """Add baseline evaluation results."""
        self.report_data['baseline'] = results
    
    def add_latency_results(self, results: Dict[str, float]):
        """Add latency measurement results."""
        self.report_data['latency'] = results
    
    def add_adversarial_results(self, results: Dict):
        """Add adversarial evaluation results."""
        self.report_data['adversarial'] = results
    
    def add_gan_results(self, results: Dict[str, float]):
        """Add GAN evasion results."""
        self.report_data['gan_evasion'] = results
    
    def generate_report(
        self,
        model_name: str = "DDoS_Detector",
        save_json: bool = True,
        save_txt: bool = True
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            save_json: Whether to save JSON report
            save_txt: Whether to save text report
            
        Returns:
            Report as string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "=" * 70,
            f"DDoS Detection Model Evaluation Report",
            f"Model: {model_name}",
            f"Generated: {timestamp}",
            "=" * 70,
            ""
        ]
        
        # Baseline Performance
        if 'baseline' in self.report_data:
            report_lines.extend([
                "BASELINE DETECTION PERFORMANCE",
                "-" * 40,
            ])
            baseline = self.report_data['baseline']
            report_lines.extend([
                f"  Accuracy:           {baseline.get('accuracy', 0):.4f}",
                f"  Precision:          {baseline.get('precision', 0):.4f}",
                f"  Recall:             {baseline.get('recall', 0):.4f}",
                f"  F1 Score:           {baseline.get('f1_score', 0):.4f}",
                f"  False Positive Rate: {baseline.get('false_positive_rate', 0):.4f}",
                f"  ROC AUC:            {baseline.get('roc_auc', 0):.4f}",
                "",
                "  Confusion Matrix:",
                f"    True Negatives:   {baseline.get('true_negatives', 0)}",
                f"    False Positives:  {baseline.get('false_positives', 0)}",
                f"    False Negatives:  {baseline.get('false_negatives', 0)}",
                f"    True Positives:   {baseline.get('true_positives', 0)}",
                ""
            ])
        
        # Latency Performance
        if 'latency' in self.report_data:
            report_lines.extend([
                "INFERENCE LATENCY",
                "-" * 40,
            ])
            latency = self.report_data['latency']
            report_lines.extend([
                f"  Mean Latency:       {latency.get('latency_mean_ms', 0):.3f} ms",
                f"  Std Deviation:      {latency.get('latency_std_ms', 0):.3f} ms",
                f"  P50 Latency:        {latency.get('latency_p50_ms', 0):.3f} ms",
                f"  P95 Latency:        {latency.get('latency_p95_ms', 0):.3f} ms",
                f"  P99 Latency:        {latency.get('latency_p99_ms', 0):.3f} ms",
                f"  Throughput:         {latency.get('throughput_per_sec', 0):.1f} samples/sec",
                ""
            ])
        
        # Adversarial Robustness
        if 'adversarial' in self.report_data:
            report_lines.extend([
                "ADVERSARIAL ROBUSTNESS",
                "-" * 40,
            ])
            
            for method, method_results in self.report_data['adversarial'].items():
                report_lines.append(f"\n  {method.upper()} Attack:")
                for epsilon, results in method_results.items():
                    report_lines.extend([
                        f"    Epsilon = {epsilon}:",
                        f"      Original Detection Rate:    {results['original_detection_rate']:.4f}",
                        f"      Adversarial Detection Rate: {results['adversarial_detection_rate']:.4f}",
                        f"      Evasion Rate:               {results['evasion_rate']:.4f}",
                        f"      Detection Drop:             {results['detection_drop']:.4f}",
                    ])
            report_lines.append("")
        
        # GAN Evasion
        if 'gan_evasion' in self.report_data:
            report_lines.extend([
                "GAN-BASED EVASION",
                "-" * 40,
            ])
            gan = self.report_data['gan_evasion']
            report_lines.extend([
                f"  Synthetic Benign Accuracy:    {gan.get('synthetic_benign_accuracy', 0):.4f}",
                f"  Original Attack Detection:    {gan.get('original_attack_detection', 0):.4f}",
                f"  Learned Perturbation Evasion: {gan.get('learned_perturbation_evasion', 0):.4f}",
                f"  Detection Rate Drop:          {gan.get('detection_rate_drop', 0):.4f}",
                ""
            ])
        
        report_lines.extend([
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save reports
        if save_json:
            json_path = self.output_dir / f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)
            logger.info(f"JSON report saved to {json_path}")
        
        if save_txt:
            txt_path = self.output_dir / f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(txt_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Text report saved to {txt_path}")
        
        return report_text


def run_full_evaluation(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = './results',
    model_name: str = 'DDoS_Detector',
    run_adversarial: bool = True,
    run_gan: bool = True
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        model: Model to evaluate
        X_test: Test features
        y_test: Test labels
        output_dir: Output directory for reports
        model_name: Name of the model
        run_adversarial: Whether to run adversarial evaluation
        run_gan: Whether to run GAN evaluation
        
    Returns:
        Complete evaluation results
    """
    logger.info("Starting comprehensive evaluation...")
    
    # Initialize evaluators
    det_evaluator = DetectionEvaluator(model)
    report = ComprehensiveReport(output_dir)
    
    # Baseline evaluation
    baseline_results = det_evaluator.evaluate_baseline(X_test, y_test)
    report.add_baseline_results(baseline_results)
    
    # Latency evaluation
    latency_results = det_evaluator.measure_latency(X_test)
    report.add_latency_results(latency_results)
    
    # Adversarial evaluation
    if run_adversarial:
        attack_data = X_test[y_test == 1]
        if len(attack_data) > 0:
            adv_evaluator = AdversarialEvaluator(model)
            adv_results = adv_evaluator.evaluate_robustness(
                attack_data[:1000],  # Limit for speed
                epsilon_values=[0.01, 0.05, 0.1, 0.2]
            )
            report.add_adversarial_results(adv_results)
    
    # GAN evaluation
    if run_gan:
        attack_data = X_test[y_test == 1]
        benign_data = X_test[y_test == 0]
        
        if len(attack_data) > 0 and len(benign_data) > 0:
            adv_evaluator = AdversarialEvaluator(model)
            gan_results = adv_evaluator.evaluate_gan_evasion(
                attack_data[:500],
                benign_data[:1000],
                gan_epochs=30
            )
            report.add_gan_results(gan_results)
    
    # Generate report
    report_text = report.generate_report(model_name)
    print("\n" + report_text)
    
    return report.report_data


if __name__ == '__main__':
    # Demo evaluation
    print("DDoS Detection Evaluation Demo")
    print("=" * 50)
    
    from data_loader import create_synthetic_ddos_data
    from models import create_model
    import torch.optim as optim
    
    # Create synthetic data
    X, y = create_synthetic_ddos_data(n_samples=5000)
    
    # Quick train a model for demo
    model = create_model('cnn', input_features=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training model for demo...")
    for epoch in range(10):
        model.train()
        x_batch = torch.FloatTensor(X).unsqueeze(1)
        y_batch = torch.LongTensor(y)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print("\nRunning evaluation...")
    results = run_full_evaluation(
        model, X, y,
        output_dir='./demo_results',
        model_name='Demo_CNN',
        run_adversarial=True,
        run_gan=False  # Skip GAN for speed in demo
    )
