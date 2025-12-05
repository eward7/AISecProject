# AI Security Research: DDoS Detection and Adversarial Robustness

Deep learning-based DDoS attack detection with comprehensive adversarial robustness evaluation on the CICDDoS2019 dataset.

## Overview

This research framework evaluates deep learning architectures (CNN, LSTM, Transformer, Hybrid CNN-LSTM) for DDoS detection and measures their vulnerability to adversarial attacks (FGSM, PGD). The project demonstrates that while models achieve >99.9% accuracy on balanced CICDDoS2019 data, this reflects dataset oversimplification rather than production readiness—real-world performance would likely be 70-85%.

**Key Findings:**
- **CNN**: 99.98% accuracy, 0.05% FPR, **24.7% PGD evasion** (most robust)
- **LSTM**: 99.94% accuracy, 0.12% FPR, **47.7% PGD evasion** (most vulnerable)
- **Hybrid**: 99.93% accuracy, 0.15% FPR, 41.3% PGD evasion (4× fewer parameters)

**Critical Insight:** High accuracy indicates CICDDoS2019's laboratory-generated attacks (extreme volumetric signatures) are trivially separable from clean benign traffic—not representative of sophisticated real-world evasive attacks.

## Project Structure

```
AISecProject/
├── src/                         # Core modules
│   ├── data_loader.py           # CICDDoS2019 + synthetic data loading
│   ├── models.py                # CNN, LSTM, Transformer, Hybrid architectures
│   ├── trainer.py               # Training pipeline with early stopping
│   ├── adversarial_gan.py       # FGSM, PGD, GAN-based attacks
│   ├── evaluation.py            # Comprehensive metrics + reporting
│   └── network_simulator.py     # Simulated DDoS attack scenarios
│
├── configs/
│   └── default_config.yaml      # Model hyperparameters
│
├── data/                        # Place CICDDoS2019 CSV files here
│   └── CICDDoS2019/             # 01-12 subdirectories with CSV files
│
├── results_cicddos/             # Training outputs
│   ├── best_cnn_model.pth       # Trained CNN checkpoint
│   ├── best_lstm_model.pth      # Trained LSTM checkpoint
│   ├── best_hybrid_model.pth    # Trained Hybrid checkpoint
│   ├── comparison_summary.json  # Final results with all metrics
│   └── cnn_training_log.json    # Per-epoch training history
│
├── paper_figures/               # Publication-ready visualizations
│   ├── fig1_model_comparison.pdf
│   ├── fig2_confusion_matrix.pdf
│   ├── fig4_adversarial_analysis.pdf
│   └── fig5_efficiency.pdf
│
├── train_real_data.py           # Main training script for CICDDoS2019
├── generate_summary.py          # Generate results comparison table
├── generate_paper_figures.py    # Create publication figures
├── generate_attack_table.py     # Per-attack-type performance analysis
└── main.py                      # CLI for synthetic data experiments
```

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate aisecproject
```

### 2. Verify GPU Support (Recommended)

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Training on NVIDIA RTX 4070 takes ~12-16 minutes per model. CPU training possible but slower.

## Dataset Setup

### CICDDoS2019 (Required for Main Experiments)

**Download:** https://www.unb.ca/cic/datasets/ddos-2019.html

**Structure:**
```
data/CICDDoS2019/
├── 01-12/                       # 11 directories for Jan 12, 2019 attacks
│   ├── DrDoS_DNS.csv
│   ├── DrDoS_LDAP.csv
│   ├── DrDoS_MSSQL.csv
│   ├── DrDoS_NTP.csv
│   ├── DrDoS_NetBIOS.csv
│   ├── DrDoS_SNMP.csv
│   ├── DrDoS_SSDP.csv
│   ├── DrDoS_UDP.csv
│   ├── Syn.csv
│   ├── TFTP.csv
│   └── UDP-lag.csv
└── SAT-01-12-2018_0.pcap_ISCX.csv  # Benign traffic
```

**Attack Types:**
- **DrDoS (Distributed Reflection DoS):** DNS (28-54× amp), LDAP (46-55×), NTP (556× - highest!), MSSQL, NetBIOS, SNMP, SSDP, UDP, TFTP
- **Direct Volumetric:** SYN Flood, UDP-lag
- **Application-Layer:** WebDDoS

**Dataset Stats:**
- 47.8 million flows (Jan 12, 2019)
- 12 attack types + benign
- 79 features (CICFlowMeter extraction)
- Original imbalance: 98.82% attack, 1.18% benign

## Training on CICDDoS2019

### Main Training Script

```bash
# Train all 3 models (CNN, LSTM, Hybrid) with balanced data
python train_real_data.py --epochs 50 --n-samples 100000 --device cuda

# Options:
#   --epochs        : Training epochs (default: 50)
#   --n-samples     : Total samples (default: 100000, balanced 50/50)
#   --device        : cuda/cpu (default: cuda)
#   --batch-size    : Batch size (default: 128)
#   --learning-rate : Initial LR (default: 1e-3)
```

**Output:**
- `results_cicddos/best_cnn_model.pth` (747K params)
- `results_cicddos/best_lstm_model.pth` (736K params)
- `results_cicddos/best_hybrid_model.pth` (181K params)
- `results_cicddos/comparison_summary.json` (all metrics)

### Data Balancing Strategy

The script implements intelligent oversampling:
1. **50/50 benign-to-attack ratio** (fixes original 1.18% benign imbalance)
2. **Minimum 1,000 samples per attack type** (ensures all 12 types represented)
3. **Stratified splitting** (maintains balance in train/val/test)

Result: 200,290 samples (100,145 benign, 100,145 attack across 12 types)

### Training Details

**Optimizer:** AdamW (lr=1e-3)  
**Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)  
**Early Stopping:** Patience=15 epochs  
**Hardware:** NVIDIA RTX 4070 Laptop GPU (8GB VRAM)  
**Training Time:** 12-16 minutes per model (30 epochs typical)

## Adversarial Evaluation

### Automatic (Included in Training)

`train_real_data.py` automatically runs adversarial evaluation:
- **FGSM:** Fast Gradient Sign Method (ε=0.3)
- **PGD:** Projected Gradient Descent (10 iterations, α=0.01, ε=0.3)
- **Metrics:** Evasion success rate, mean L2 perturbation

### Results Interpretation

```json
{
  "cnn": {
    "fgsm": {"evasion_success_rate": 0.07},   // 7% evasion
    "pgd":  {"evasion_success_rate": 0.247}   // 24.7% evasion - BEST
  },
  "lstm": {
    "fgsm": {"evasion_success_rate": 0.297},
    "pgd":  {"evasion_success_rate": 0.477}   // 47.7% evasion - WORST
  }
}
```

**Key Insight:** CNN's spatial feature aggregation creates more robust decision boundaries than LSTM's sequential processing.

## Results Analysis

### Generate Summary Table

```bash
python generate_summary.py
```

Output: Formatted comparison table + `comparison_summary.json`

### Generate Publication Figures

```bash
python generate_paper_figures.py
```

Creates 5 publication-quality figures (300 DPI PDF + PNG):
- `fig1_model_comparison.pdf` (3-panel: accuracy, FPR, adversarial)
- `fig2_confusion_matrix.pdf` (CNN classification matrix)
- `fig4_adversarial_analysis.pdf` (perturbation vs evasion)
- `fig5_efficiency.pdf` (accuracy vs parameters)

### Per-Attack-Type Analysis

```bash
python generate_attack_table.py
```

Output: CSV + LaTeX table showing CNN performance on each of 12 attack types (e.g., DrDoS_NTP: 100% perfect detection, Syn: 99.93%)

## Synthetic Data Experiments

For testing without CICDDoS2019 dataset:

```bash
# Train on synthetic data with configurable difficulty
python main.py train --model-type cnn --n-samples 50000 --difficulty medium

# Difficulty levels:
#   easy   : ~97% accuracy (clearly separable)
#   medium : ~93% accuracy (realistic overlap)
#   hard   : ~82% accuracy (sophisticated attacks)
```

## Model Architectures

### CNN Detector (747,650 parameters)
- **Structure:** 3 Conv1D layers (64→128→256 filters) + 2 FC layers
- **Features:** Batch normalization, dropout (p=0.5), adaptive pooling
- **Best for:** Spatial feature correlations (packet size vs inter-arrival time)
- **Advantages:** Fastest inference (1.73ms), most robust to PGD (24.7%)

### LSTM Detector (735,875 parameters)
- **Structure:** 2 Bidirectional LSTM layers (128 units) + FC output
- **Features:** Dropout (p=0.3), temporal sequence modeling
- **Best for:** Sequential traffic patterns, low-and-slow attacks
- **Disadvantages:** Most vulnerable to PGD (47.7% evasion)

### Hybrid CNN-LSTM (180,802 parameters)
- **Structure:** 2 Conv layers + 1 LSTM layer + FC
- **Features:** 4.1× fewer parameters than CNN/LSTM
- **Best for:** Resource-constrained edge deployment
- **Trade-off:** 99.93% accuracy (marginal drop) for 4× size reduction

### Transformer Detector (experimental)
- **Structure:** Self-attention with positional encoding
- **Best for:** Complex feature relationships
- **Note:** Higher computational cost, not included in main experiments

## Critical Dataset Limitations

**⚠️ Important:** The 99.98% accuracy is **not production-ready**. CICDDoS2019 contains laboratory-generated attacks with extreme statistical signatures:

**Lab Attacks (Easy to Detect):**
- DrDoS: 10,000+ identical packets/sec from spoofed IPs
- SYN Flood: Pure TCP SYN with no payload diversity
- Benign: Clean HTTP browsing with regular timing

**Real Attacks (Hard to Detect):**
- Low-and-slow attacks spread over time
- Polymorphic payloads with encryption
- Botnet diversity mixing legitimate and attack traffic
- Application-layer mimicking user sessions

**Expected Production Performance:** 70-85% accuracy (not 99.98%), with 10-100× FPR increase.

**Recommendation:** Use CICDDoS2019 for:
✓ Comparative architecture analysis (CNN vs LSTM)  
✓ Adversarial robustness methodology  
✓ Baseline establishment  

**Do NOT use for:**
✗ Claiming production deployment readiness  
✗ Estimating real-world performance  
✗ Security product validation  

## Configuration

Edit `configs/default_config.yaml`:

```yaml
model:
  cnn:
    conv_channels: [64, 128, 256]
    dropout: 0.5
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  patience: 15

adversarial:
  fgsm:
    epsilon: 0.3
  pgd:
    epsilon: 0.3
    alpha: 0.01
    num_iter: 10
```

## Evaluation Metrics

### Detection Performance
- **Accuracy:** Overall classification correctness
- **Precision:** True attacks / detected attacks
- **Recall:** Detected attacks / all attacks  
- **F1 Score:** Harmonic mean of precision and recall
- **FPR:** False Positive Rate (benign misclassified as attack)
- **AUC-ROC:** Area under ROC curve

### Adversarial Robustness
- **Evasion Success Rate:** Attacks misclassified as benign after perturbation
- **Mean L2 Perturbation:** Average Euclidean distance of perturbations
- **Detection Drop:** Reduction in recall after attack

### Inference Performance
- **Latency:** Mean inference time (ms)
- **Throughput:** Samples processed per second

## Python API Usage

### Training Example

```python
from src.data_loader import load_cicddos2019, prepare_data_loaders
from src.models import CNNDetector
from src.trainer import Trainer
import torch

# Load and balance data
data = load_cicddos2019(
    data_dir='data/CICDDoS2019',
    sample_size=100000,
    balance_classes=True,
    min_samples_per_attack=1000
)

# Prepare dataloaders
loaders = prepare_data_loaders(
    data, 
    batch_size=128, 
    model_type='cnn'
)

# Create model
model = CNNDetector(input_size=79).to('cuda')

# Train
trainer = Trainer(model, checkpoint_dir='results_cicddos')
history = trainer.train(
    loaders['train'], 
    loaders['val'], 
    epochs=50,
    patience=15
)

# Evaluate
metrics = trainer.evaluate(loaders['test'])
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"FPR: {metrics['fpr']:.4f}")
```

### Adversarial Attack Example

```python
from src.adversarial_gan import AdversarialAttacker
from src.models import load_model

# Load trained model
model = load_model('results_cicddos/best_cnn_model.pth')

# Create attacker
attacker = AdversarialAttacker(model, device='cuda')

# FGSM attack
X_adv_fgsm = attacker.fgsm(X_attack, y_attack, epsilon=0.3)

# PGD attack  
X_adv_pgd = attacker.pgd(
    X_attack, 
    y_attack, 
    epsilon=0.3, 
    alpha=0.01, 
    num_iter=10
)

# Evaluate evasion
with torch.no_grad():
    preds_original = model(X_attack) > 0.5
    preds_adversarial = model(X_adv_pgd) > 0.5
    
evasion_rate = (preds_original & ~preds_adversarial).float().mean()
print(f"PGD Evasion Rate: {evasion_rate:.2%}")
```

## Command-Line Reference

### Training Commands

```bash
# CICDDoS2019 training (recommended)
python train_real_data.py --epochs 50 --n-samples 100000 --device cuda

# Synthetic data training
python main.py train --model-type cnn --n-samples 50000 --difficulty medium

# Train all models sequentially
python train_all_models.py --epochs 50 --n-samples 100000
```

### Analysis Commands

```bash
# Generate results summary
python generate_summary.py

# Create paper figures
python generate_paper_figures.py

# Per-attack-type analysis
python generate_attack_table.py
```

### Evaluation Commands

```bash
# Evaluate specific model
python main.py evaluate --model-path results_cicddos/best_cnn_model.pth --model-type cnn

# Run adversarial attacks
python main.py adversarial --model-path results_cicddos/best_cnn_model.pth --device cuda
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_real_data.py --batch-size 64 --device cuda

# Or use CPU (slower but works)
python train_real_data.py --device cpu
```

### CICDDoS2019 Loading Errors
```bash
# Check file structure
ls -R data/CICDDoS2019/

# Ensure CSVs are in subdirectories (01-12, etc.)
# Benign file should be at root: SAT-01-12-2018_0.pcap_ISCX.csv
```

### Import Errors
```bash
# Ensure environment is activated
conda activate aisecproject

# Reinstall if needed
conda env update -f environment.yml
```

### Slow Training
- Enable CUDA: `--device cuda`
- Increase batch size: `--batch-size 256` (if GPU memory allows)
- Reduce samples: `--n-samples 50000`

## Research Context

**Associated Paper:** "The Dual-Edged Sword: AI in the Execution and Defense of DDoS Attacks" (NeurIPS 2024 submission)

**Key Contributions:**
1. Balanced CICDDoS2019 evaluation (fixes class imbalance)
2. Adversarial robustness comparison (CNN > LSTM counterintuitively)
3. Critical assessment of benchmark dataset limitations
4. Production deployment guidance

**Citation:** (Add when published)

## License

Educational and research use only. CICDDoS2019 dataset © Canadian Institute for Cybersecurity, University of New Brunswick.

## Acknowledgments

- **Dataset:** UNB Canadian Institute for Cybersecurity (CICDDoS2019)
- **Framework:** PyTorch deep learning library
- **Hardware:** NVIDIA RTX 4070 Laptop GPU
- **Research:** AI Security course project, 2024

---

**Last Updated:** December 5, 2024  
**Status:** Complete experimental framework with published results  
**Contact:** See paper for author information
