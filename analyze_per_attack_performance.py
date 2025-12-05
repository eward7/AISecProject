#!/usr/bin/env python
"""
Analyze per-attack-type performance for each model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.models import CNNDetector, LSTMDetector, HybridCNNLSTM
from src.data_loader import load_cicddos2019

# Load test data
print("Loading CICDDoS2019 test data...")
X_train, y_train, X_test, y_test, attack_train, attack_test = load_cicddos2019(
    data_dir='data/CICDDoS2019',
    balance_classes=True,
    sample_size=100000,
    min_samples_per_attack=1000
)

print(f"Test set: {len(X_test)} samples")
print(f"Attack types in test set: {np.unique(attack_test)}")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configurations
models_config = {
    'cnn': {'class': CNNDetector, 'path': 'results_cicddos/best_cnn_model.pth'},
    'lstm': {'class': LSTMDetector, 'path': 'results_cicddos/best_lstm_model.pth'},
    'hybrid': {'class': HybridCNNLSTM, 'path': 'results_cicddos/best_hybrid_model.pth'}
}

# Attack type descriptions
attack_descriptions = {
    'DrDoS_DNS': 'DNS Amplification',
    'DrDoS_LDAP': 'LDAP Amplification',
    'DrDoS_MSSQL': 'MSSQL Amplification',
    'DrDoS_NTP': 'NTP Amplification',
    'DrDoS_NetBIOS': 'NetBIOS Amplification',
    'DrDoS_SNMP': 'SNMP Amplification',
    'DrDoS_SSDP': 'SSDP Amplification',
    'DrDoS_UDP': 'UDP Reflection',
    'Syn': 'SYN Flood',
    'TFTP': 'TFTP Amplification',
    'UDP-lag': 'UDP Lag Attack',
    'WebDDoS': 'Web Application DDoS'
}

results_per_model = {}

for model_name, config in models_config.items():
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name.upper()} model")
    print(f"{'='*60}")
    
    # Load model
    model = config['class'](input_size=X_train.shape[1]).to(device)
    checkpoint = torch.load(config['path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).cpu().numpy().astype(int)
    
    # Overall metrics
    overall_acc = accuracy_score(y_test, predictions)
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
    
    # Per-attack-type analysis
    attack_types = sorted(np.unique(attack_test[attack_test != 'BENIGN']))
    
    per_attack_results = []
    
    # Benign class
    benign_mask = (attack_test == 'BENIGN')
    benign_true = y_test[benign_mask]
    benign_pred = predictions[benign_mask]
    benign_acc = accuracy_score(benign_true, benign_pred)
    benign_count = benign_mask.sum()
    
    per_attack_results.append({
        'Attack Type': 'BENIGN',
        'Description': 'Legitimate Traffic',
        'Samples': int(benign_count),
        'Accuracy (%)': benign_acc * 100,
        'Precision (%)': '-',
        'Recall (%)': '-',
        'F1 (%)': '-'
    })
    
    # Each attack type
    for attack_type in attack_types:
        attack_mask = (attack_test == attack_type)
        attack_true = y_test[attack_mask]
        attack_pred = predictions[attack_mask]
        
        if len(attack_true) == 0:
            continue
        
        attack_acc = accuracy_score(attack_true, attack_pred)
        
        # Calculate precision, recall, F1 for this attack type
        # Note: For attack detection, we care about detecting attacks (class 1)
        if len(np.unique(attack_pred)) > 1:
            prec, rec, f1, _ = precision_recall_fscore_support(
                attack_true, attack_pred, average='binary', zero_division=0
            )
        else:
            # All predictions same class
            if attack_pred[0] == 1:  # All predicted as attack
                rec = 1.0
                prec = attack_acc
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            else:  # All predicted as benign
                rec = 0.0
                prec = 0.0
                f1 = 0.0
        
        per_attack_results.append({
            'Attack Type': attack_type,
            'Description': attack_descriptions.get(attack_type, 'Unknown'),
            'Samples': int(attack_mask.sum()),
            'Accuracy (%)': attack_acc * 100,
            'Precision (%)': prec * 100,
            'Recall (%)': rec * 100,
            'F1 (%)': f1 * 100
        })
    
    # Create DataFrame
    df = pd.DataFrame(per_attack_results)
    
    print("\n" + "="*80)
    print(f"{model_name.upper()} - Per-Attack-Type Performance")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save results
    results_per_model[model_name] = per_attack_results
    
    # Save to CSV
    output_path = f'results_cicddos/{model_name}_per_attack_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")

# Generate LaTeX table for best model (CNN)
print("\n" + "="*80)
print("LaTeX Table for Paper (CNN Model)")
print("="*80)

cnn_df = pd.DataFrame(results_per_model['cnn'])

latex_lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Per-Attack-Type Detection Performance (CNN Model)}",
    "\\label{tab:per_attack_performance}",
    "\\begin{tabular}{llrrrr}",
    "\\toprule",
    "\\textbf{Attack Type} & \\textbf{Description} & \\textbf{Samples} & \\textbf{Accuracy} & \\textbf{Recall} & \\textbf{F1} \\\\",
    " & & & (\\%) & (\\%) & (\\%) \\\\",
    "\\midrule"
]

for _, row in cnn_df.iterrows():
    attack = row['Attack Type']
    desc = row['Description']
    samples = row['Samples']
    acc = row['Accuracy (%)']
    rec = row['Recall (%)'] if isinstance(row['Recall (%)'], float) else '-'
    f1 = row['F1 (%)'] if isinstance(row['F1 (%)'], float) else '-'
    
    if attack == 'BENIGN':
        latex_lines.append(f"{attack} & {desc} & {samples:,} & {acc:.2f} & - & - \\\\")
    else:
        latex_lines.append(f"{attack} & {desc} & {samples:,} & {acc:.2f} & {rec:.2f} & {f1:.2f} \\\\")

latex_lines.extend([
    "\\midrule",
    f"\\textbf{{Overall}} & All Traffic & {len(y_test):,} & {overall_acc*100:.2f} & - & - \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
])

latex_table = "\n".join(latex_lines)
print(latex_table)

with open('results_cicddos/per_attack_performance_table.tex', 'w') as f:
    f.write(latex_table)

print("\n✓ LaTeX table saved to results_cicddos/per_attack_performance_table.tex")

# Save all results to JSON
with open('results_cicddos/per_attack_analysis.json', 'w') as f:
    json.dump(results_per_model, f, indent=2)

print("\n✅ Per-attack-type analysis complete!")
