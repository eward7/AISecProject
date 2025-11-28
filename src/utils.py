"""
Utility functions for the DDoS Detection project.
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'cpu', 'mps')
        
    Returns:
        torch.device object
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon) device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
    else:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Output file path
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Configuration saved to {path}")


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {path}")
    return config


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> Path:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to created directory
    """
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics (in GB)
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    stats = {
        'rss_gb': mem_info.rss / (1024 ** 3),
        'vms_gb': mem_info.vms / (1024 ** 3),
    }
    
    if torch.cuda.is_available():
        stats['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
        stats['cuda_cached_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
    
    return stats


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start).total_seconds()
        if self.name:
            logger.info(f"{self.name}: {format_time(self.elapsed)}")


if __name__ == '__main__':
    # Demo utilities
    print("Utility Functions Demo")
    print("=" * 50)
    
    # Set seed
    set_seed(42)
    print(f"Random number: {random.random()}")
    
    # Get device
    device = get_device('auto')
    print(f"Device: {device}")
    
    # Timer demo
    with Timer("Test operation"):
        import time
        time.sleep(0.5)
    
    # Memory usage
    try:
        mem = get_memory_usage()
        print(f"Memory usage: {mem}")
    except ImportError:
        print("psutil not installed, skipping memory check")
