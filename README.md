# AI Security Research: DDoS Detection and Adversarial Evasion

A comprehensive research framework for studying DDoS attack detection using deep learning models and evaluating their robustness against adversarial evasion techniques.

## Overview

This project provides:

1. **Simulated Network Environment**: A virtual network for testing DDoS attacks and prevention mechanisms
2. **Deep Learning Detection Models**: CNN, LSTM, Transformer, and Hybrid CNN-LSTM architectures for DDoS traffic classification
3. **Adversarial Attack Framework**: GAN-based and gradient-based methods for generating evasive traffic patterns
4. **Comprehensive Evaluation**: Metrics for detection accuracy, false positive rate, latency, and adversarial robustness

## Project Structure

```
AISecProject/
├── configs/
│   └── default_config.yaml      # Default configuration settings
├── data/
│   └── (CICDDoS2019 dataset files - download separately)
├── models/
│   └── (trained model checkpoints)
├── notebooks/
│   └── (Jupyter notebooks for analysis)
├── results/
│   └── (evaluation reports and metrics)
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── network_simulator.py     # Simulated network environment
│   ├── models.py                # Deep learning model architectures
│   ├── trainer.py               # Training pipeline
│   ├── adversarial_gan.py       # GAN and adversarial attack framework
│   └── evaluation.py            # Comprehensive evaluation module
├── main.py                      # Command-line interface
├── environment.yml              # Conda environment specification
└── README.md                    # This file
```

## Installation

### 1. Create Conda Environment

```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate aisecproject
```

### 2. GPU Support (Optional)

For NVIDIA GPU support, edit `environment.yml` and uncomment the appropriate CUDA version:

```yaml
# For CUDA 11.8
- pytorch-cuda=11.8

# For CUDA 12.1
- pytorch-cuda=12.1
```

Then recreate the environment.

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dataset

### Option 1: CICDDoS2019 Dataset (Recommended)

The CICDDoS2019 dataset is the primary dataset for this research.

**Download Instructions:**
1. Visit: https://www.unb.ca/cic/datasets/ddos-2019.html
2. Download the CSV files
3. Place them in the `data/` directory

**Attack Types in CICDDoS2019:**
- DrDoS_DNS, DrDoS_LDAP, DrDoS_MSSQL, DrDoS_NTP
- DrDoS_NetBIOS, DrDoS_SNMP, DrDoS_SSDP, DrDoS_UDP
- Syn, TFTP, UDP-lag, WebDDoS, Portmap, LDAP

### Option 2: Synthetic Data

For testing without downloading the dataset, the framework includes a synthetic data generator that creates realistic DDoS traffic patterns:

```python
from src.data_loader import create_synthetic_ddos_data

X, y = create_synthetic_ddos_data(n_samples=50000, attack_ratio=0.3)
```

## Usage

### Training a Detection Model

#### Using Command Line

```bash
# Train a CNN detector with synthetic data
python main.py train --model-type cnn --epochs 100 --batch-size 64

# Train an LSTM detector
python main.py train --model-type lstm --epochs 100

# Train a Transformer detector
python main.py train --model-type transformer --epochs 100

# Train a Hybrid CNN-LSTM detector
python main.py train --model-type hybrid --epochs 100

# Train with CICDDoS2019 dataset
python main.py train --model-type cnn --data-path ./data --epochs 100
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-type` | Model architecture (cnn, lstm, transformer, hybrid) | cnn |
| `--data-path` | Path to CICDDoS2019 dataset | None (uses synthetic) |
| `--n-samples` | Number of synthetic samples | 50000 |
| `--epochs` | Training epochs | 100 |
| `--batch-size` | Batch size | 64 |
| `--learning-rate` | Learning rate | 0.001 |
| `--patience` | Early stopping patience | 15 |
| `--device` | Device (auto, cuda, cpu, mps) | auto | note: auto option doesn't seem to work when doing evaluation
| `--output-dir` | Output directory | ./results |

### Evaluating a Model

```bash
# Basic evaluation
python main.py evaluate --model-path ./results/cnn_detector/best_model.pt --model-type cnn

# With adversarial evaluation
python main.py evaluate --model-path ./results/cnn_detector/best_model.pt --run-adversarial

# With GAN-based evasion evaluation
python main.py evaluate --model-path ./results/cnn_detector/best_model.pt --run-adversarial --run-gan
```

### Running Adversarial Attacks

```bash
# Run FGSM and PGD attacks
python main.py adversarial --model-path ./results/cnn_detector/best_model.pt

# Include GAN-based attacks
python main.py adversarial --model-path ./results/cnn_detector/best_model.pt --run-gan --gan-epochs 50
```

### Network Simulation

```bash
# Run attack scenarios
python main.py simulate --duration 60 --export-flows

# This will run the following scenarios:
# - SYN Flood
# - UDP Flood
# - HTTP Flood
# - DNS Amplification
# - Mixed attacks
```

## Python API Usage

### Training a Model

```python
from src.data_loader import create_synthetic_ddos_data, prepare_data_loaders
from src.models import create_model
from src.trainer import Trainer

# Create data
X, y = create_synthetic_ddos_data(n_samples=50000)
loaders = prepare_data_loaders(X, y, batch_size=64, model_type='cnn')

# Create model
model = create_model('cnn', input_features=76)

# Train
trainer = Trainer(model, checkpoint_dir='./results')
history = trainer.train(loaders['train'], loaders['val'], epochs=100)

# Evaluate
metrics = trainer.evaluate(loaders['test'], measure_latency=True)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"FPR: {metrics['fpr']:.4f}")
print(f"Latency: {metrics['latency_mean_ms']:.2f}ms")
```

### Running Adversarial Attacks

```python
from src.adversarial_gan import AdversarialAttacker, AdversarialGAN, evaluate_evasion_success
from src.models import load_pretrained_model

# Load trained model
model = load_pretrained_model('./results/cnn_detector/best_model.pt', 'cnn')

# Create attacker
attacker = AdversarialAttacker(model)

# FGSM Attack
adv_fgsm = attacker.generate_adversarial_examples(attack_data, method='fgsm')
metrics_fgsm = evaluate_evasion_success(model, attack_data, adv_fgsm)

# PGD Attack
adv_pgd = attacker.generate_adversarial_examples(attack_data, method='pgd')
metrics_pgd = evaluate_evasion_success(model, attack_data, adv_pgd)

# Train GAN for evasion
gan = AdversarialGAN(feature_dim=76)
gan.train(benign_data, epochs=100)

# Train perturbation generator
attacker.train_perturbation_generator(attack_data, labels, epochs=50)
adv_learned = attacker.generate_adversarial_examples(attack_data, method='learned')
```

### Using the Network Simulator

```python
from src.network_simulator import SimulatedNetwork, AttackScenario, TrafficGenerator

# Create network
network = SimulatedNetwork()
scenario = AttackScenario(network)

# Run attack scenario
stats = scenario.run_scenario('syn_flood', duration=60, attack_start=20, attack_duration=20)

# Get flow features for ML
features = network.get_flow_features()
labels = network.get_flow_labels()

# Export data
network.export_flows_to_csv('flows.csv')
```

## Model Architectures

### CNN Detector
- 1D Convolutional layers for feature extraction
- Batch normalization and dropout for regularization
- Best for: Pattern recognition in network features

### LSTM Detector
- Bidirectional LSTM with attention mechanism
- Captures temporal dependencies in traffic sequences
- Best for: Sequential traffic analysis

### Transformer Detector
- Self-attention mechanism for feature relationships
- Positional encoding for sequence awareness
- Best for: Complex pattern recognition

### Hybrid CNN-LSTM
- CNN for local feature extraction
- LSTM for temporal modeling
- Best for: Combined spatial-temporal patterns

## Adversarial Attack Methods

### FGSM (Fast Gradient Sign Method)
- Single-step gradient attack
- Fast but less powerful
- Parameters: `epsilon` (perturbation magnitude)

### PGD (Projected Gradient Descent)
- Iterative gradient attack
- More powerful, finds better adversarial examples
- Parameters: `epsilon`, `alpha` (step size), `num_iter`

### GAN-based Attacks
- Generator learns to create benign-like traffic
- Perturbation generator learns to evade specific detectors
- Most realistic evasion technique

## Evaluation Metrics

### Detection Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of true attacks among detected attacks
- **Recall**: Ratio of detected attacks among all attacks
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: Benign traffic misclassified as attack
- **ROC AUC**: Area under ROC curve

### Latency Metrics
- **Mean Latency**: Average inference time
- **P95/P99 Latency**: Tail latency percentiles
- **Throughput**: Samples processed per second

### Adversarial Metrics
- **Evasion Rate**: Percentage of attacks classified as benign
- **Detection Drop**: Reduction in detection rate after perturbation
- **Perturbation Magnitude**: L2/L∞ norm of added perturbation

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
model:
  type: "cnn"
  cnn:
    conv_channels: [64, 128, 256]
    dropout: 0.5

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

adversarial:
  pgd:
    epsilon: 0.1
    num_iter: 40
```

## Results

After training and evaluation, results are saved to the `results/` directory:

```
results/
├── cnn_detector/
│   ├── best_model.pt           # Best model checkpoint
│   ├── metrics.json            # Training metrics
│   └── results.json            # Final results
├── report_DDoS_Detector_*.json # Evaluation report (JSON)
├── report_DDoS_Detector_*.txt  # Evaluation report (Text)
└── adversarial_results.json    # Adversarial attack results
```

## Research Applications

This framework can be used for:

1. **Baseline Establishment**: Measure detection model performance on DDoS traffic
2. **Robustness Testing**: Evaluate model vulnerability to adversarial attacks
3. **Defense Development**: Test and improve adversarial defense mechanisms
4. **Attack Simulation**: Generate realistic attack traffic for testing

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 32`
- Use CPU: `--device cpu`

### Slow Training
- Enable GPU if available
- Reduce model complexity in config
- Use smaller dataset sample

### Import Errors
- Ensure conda environment is activated
- Check all dependencies are installed: `conda list`

## License

This project is for educational and research purposes.

## Acknowledgments

- CICDDoS2019 dataset: Canadian Institute for Cybersecurity, University of New Brunswick
- PyTorch team for the deep learning framework
