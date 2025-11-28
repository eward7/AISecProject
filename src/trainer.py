"""
Training Pipeline for DDoS Detection Models

This module provides comprehensive training functionality including:
- Training loop with validation
- Learning rate scheduling
- Early stopping
- Checkpointing
- Logging and metrics tracking
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            model: Model to save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.should_stop
    
    def restore_weights(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_fpr': [],  # False positive rate
            'learning_rate': [],
            'epoch_time': []
        }
        self.best_metrics: Dict[str, float] = {}
    
    def update(self, epoch_metrics: Dict[str, float]):
        """Update metrics history."""
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'max') -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        if metric not in self.history or not self.history[metric]:
            return 0.0, 0
        
        values = self.history[metric]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        return values[best_idx], best_idx
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate false positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # AUC if probabilities available
        auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else 0.0
    else:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        fpr = 0.0
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'auc': auc
    }


class Trainer:
    """Training orchestrator for DDoS detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        checkpoint_dir: str = './checkpoints',
        experiment_name: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on ('auto', 'cuda', 'cpu', 'mps')
            checkpoint_dir: Directory for saving checkpoints
            experiment_name: Name for this training experiment
        """
        self.model = model
        
        # Set device
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
        logger.info(f"Using device: {self.device}")
        
        # Setup directories
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        scheduler_type: str = 'reduce_on_plateau',
        class_weights: Optional[torch.Tensor] = None,
        gradient_clip: float = 1.0
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            scheduler_type: LR scheduler type
            class_weights: Optional class weights for imbalanced data
            gradient_clip: Gradient clipping value
            
        Returns:
            Training history
        """
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        if scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Setup loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, gradient_clip
            )
            
            # Validation phase
            val_metrics = self._validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
            
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            epoch_metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_fpr': val_metrics['fpr'],
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            self.metrics_tracker.update(epoch_metrics)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, FPR: {val_metrics['fpr']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, optimizer, is_best=True)
            
            # Check early stopping
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        self.early_stopping.restore_weights(self.model)
        
        # Save final metrics
        self.metrics_tracker.save(self.checkpoint_dir / 'metrics.json')
        
        return self.metrics_tracker.history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        gradient_clip: float
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_proba = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                
                proba = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        
        metrics = calculate_metrics(all_labels, all_preds, all_proba)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'input_features': getattr(self.model, 'input_features', 76),
                'num_classes': getattr(self.model, 'num_classes', 2)
            },
            'metrics': self.metrics_tracker.history
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        measure_latency: bool = True,
        n_latency_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            measure_latency: Whether to measure inference latency
            n_latency_samples: Number of samples for latency measurement
            
        Returns:
            Evaluation metrics including latency
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_proba = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc='Evaluating'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                proba = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)
        
        metrics = calculate_metrics(all_labels, all_preds, all_proba)
        
        # Measure inference latency
        if measure_latency:
            latencies = self._measure_latency(test_loader, n_latency_samples)
            metrics['latency_mean_ms'] = np.mean(latencies) * 1000
            metrics['latency_std_ms'] = np.std(latencies) * 1000
            metrics['latency_p99_ms'] = np.percentile(latencies, 99) * 1000
        
        logger.info(f"Test Results - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1']:.4f}, FPR: {metrics['fpr']:.4f}")
        
        if measure_latency:
            logger.info(f"Latency - Mean: {metrics['latency_mean_ms']:.2f}ms, "
                       f"P99: {metrics['latency_p99_ms']:.2f}ms")
        
        return metrics
    
    def _measure_latency(
        self,
        test_loader: DataLoader,
        n_samples: int = 100
    ) -> List[float]:
        """Measure per-sample inference latency."""
        self.model.eval()
        latencies = []
        
        # Get sample batch
        sample_iterator = iter(test_loader)
        batch_x, _ = next(sample_iterator)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(batch_x[:1].to(self.device))
        
        # Measure latency
        with torch.no_grad():
            for i in range(min(n_samples, len(batch_x))):
                sample = batch_x[i:i+1].to(self.device)
                
                # Synchronize for accurate timing on GPU
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(sample)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                latencies.append(end - start)
        
        return latencies


def train_model(
    model_type: str = 'cnn',
    data_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = 'auto',
    output_dir: str = './results'
) -> Dict:
    """
    Convenience function to train a DDoS detection model.
    
    Args:
        model_type: Type of model to train
        data_path: Path to dataset (if None, uses synthetic data)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        output_dir: Output directory for results
        
    Returns:
        Training results
    """
    from .models import create_model
    from .data_loader import create_synthetic_ddos_data, prepare_data_loaders
    
    # Prepare data
    if data_path is None:
        logger.info("Using synthetic data for training")
        X, y = create_synthetic_ddos_data(n_samples=50000)
    else:
        from .data_loader import load_cicddos2019, DataPreprocessor
        df = load_cicddos2019(data_path)
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df)
    
    # Create data loaders
    loaders = prepare_data_loaders(X, y, batch_size=batch_size, model_type=model_type)
    
    # Create model
    model = create_model(model_type, input_features=X.shape[1])
    
    # Create trainer
    trainer = Trainer(model, device=device, checkpoint_dir=output_dir)
    
    # Handle class imbalance
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
    class_weights = class_weights / class_weights.sum()
    
    # Train
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=epochs,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    # Evaluate
    test_metrics = trainer.evaluate(loaders['test'])
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'model_path': str(trainer.checkpoint_dir / 'best_model.pt')
    }


if __name__ == '__main__':
    # Demo training
    print("DDoS Detection Model Training Demo")
    print("=" * 50)
    
    # This demo uses synthetic data
    from models import create_model
    from data_loader import create_synthetic_ddos_data, prepare_data_loaders
    
    # Create synthetic data
    X, y = create_synthetic_ddos_data(n_samples=5000)
    
    # Prepare loaders
    loaders = prepare_data_loaders(X, y, batch_size=32, model_type='cnn')
    
    # Create model
    model = create_model('cnn', input_features=X.shape[1])
    
    # Create trainer
    trainer = Trainer(model, checkpoint_dir='./demo_checkpoints')
    
    print("\nStarting training (5 epochs for demo)...")
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        epochs=5,
        learning_rate=1e-3
    )
    
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(loaders['test'])
    
    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
