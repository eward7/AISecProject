"""
Data Loader for CICDDoS2019 Dataset

This module provides utilities for loading, preprocessing, and preparing
the CICDDoS2019 dataset for DDoS detection model training.

The CICDDoS2019 dataset contains network traffic features extracted from
pcap files, including various types of DDoS attacks.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CICDDoS2019 feature columns (80 features + label)
CICDDOS2019_FEATURES = [
    'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
    'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
]

# Numeric features for model input (excluding IPs, Flow ID, Timestamp)
NUMERIC_FEATURES = [
    'Source Port', 'Destination Port', 'Protocol', 'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
    'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# Attack types in CICDDoS2019
ATTACK_TYPES = [
    'BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP',
    'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP',
    'Syn', 'TFTP', 'UDP-lag', 'WebDDoS', 'Portmap', 'LDAP'
]


class CICDDoS2019Dataset(Dataset):
    """PyTorch Dataset for CICDDoS2019 data."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 1,
        for_cnn: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            sequence_length: Sequence length for LSTM/Transformer models
            for_cnn: If True, reshape features for CNN input
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length
        self.for_cnn = for_cnn
        
        if for_cnn:
            # Reshape for CNN: (batch, channels, features)
            self.features = self.features.unsqueeze(1)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SequentialDataset(Dataset):
    """Dataset for sequential models (LSTM, Transformer)."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 10
    ):
        """
        Initialize sequential dataset.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            sequence_length: Number of time steps in each sequence
        """
        self.sequence_length = sequence_length
        self.n_features = features.shape[1]
        
        # Create sequences
        self.sequences = []
        self.sequence_labels = []
        
        for i in range(len(features) - sequence_length + 1):
            self.sequences.append(features[i:i + sequence_length])
            # Use the label of the last element in the sequence
            self.sequence_labels.append(labels[i + sequence_length - 1])
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.sequence_labels = torch.LongTensor(np.array(self.sequence_labels))
    
    def __len__(self) -> int:
        return len(self.sequence_labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.sequence_labels[idx]


class DataPreprocessor:
    """Preprocessor for CICDDoS2019 dataset."""
    
    def __init__(self, feature_columns: Optional[List[str]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            feature_columns: List of feature columns to use. If None, uses NUMERIC_FEATURES.
        """
        self.feature_columns = feature_columns or NUMERIC_FEATURES
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and infinities.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        # Handle column name variations (spaces, case)
        df.columns = df.columns.str.strip()
        
        # Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Remove any remaining rows with NaN
        df = df.dropna()
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        label_column: str = 'Label'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: DataFrame with features and labels
            label_column: Name of the label column
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        df = self.clean_data(df)
        
        # Get available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        if len(available_features) < len(self.feature_columns):
            logger.warning(f"Some features not found. Using {len(available_features)} features.")
        
        self.feature_columns = available_features
        
        # Extract features
        X = df[self.feature_columns].values.astype(np.float32)
        
        # Extract and encode labels
        y = df[label_column].values
        
        # Binary classification: BENIGN vs Attack
        y_binary = np.array(['BENIGN' if label == 'BENIGN' else 'ATTACK' for label in y])
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y_binary)
        
        self.is_fitted = True
        
        logger.info(f"Features shape: {X_scaled.shape}")
        logger.info(f"Class distribution: {np.bincount(y_encoded)}")
        
        return X_scaled, y_encoded
    
    def transform(
        self,
        df: pd.DataFrame,
        label_column: str = 'Label'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame with features and labels
            label_column: Name of the label column
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = self.clean_data(df)
        
        X = df[self.feature_columns].values.astype(np.float32)
        y = df[label_column].values
        
        y_binary = np.array(['BENIGN' if label == 'BENIGN' else 'ATTACK' for label in y])
        
        X_scaled = self.scaler.transform(X)
        y_encoded = self.label_encoder.transform(y_binary)
        
        return X_scaled, y_encoded
    
    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features."""
        return self.scaler.inverse_transform(X)
    
    def save(self, path: str):
        """Save preprocessor state."""
        import joblib
        state = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor state."""
        import joblib
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.label_encoder = state['label_encoder']
        self.feature_columns = state['feature_columns']
        self.is_fitted = state['is_fitted']
        logger.info(f"Preprocessor loaded from {path}")


def load_cicddos2019(
    data_dir: str,
    sample_size: Optional[int] = None,
    attack_types: Optional[List[str]] = None,
    stratified: bool = True,
    balance_classes: bool = True,
    min_samples_per_attack: int = 1000
) -> pd.DataFrame:
    """
    Load CICDDoS2019 dataset from directory with stratified sampling.
    
    Args:
        data_dir: Directory containing CSV files
        sample_size: Total samples to load (will be distributed across classes)
        attack_types: If provided, filter to these attack types only
        stratified: If True, sample proportionally from each file/attack type
        balance_classes: If True, balance benign vs attack samples to 50/50
        min_samples_per_attack: Minimum samples per attack type for stratification
        
    Returns:
        Combined DataFrame with stratified samples and balanced classes
    """
    data_path = Path(data_dir)
    # Look for CSV files recursively in subdirectories
    csv_files = list(data_path.rglob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # First pass: count rows in each file to determine sampling strategy
    file_sizes = {}
    if stratified and sample_size:
        for csv_file in csv_files:
            # Quick row count using wc -l for speed
            try:
                import subprocess
                result = subprocess.run(['wc', '-l', str(csv_file)], capture_output=True, text=True)
                row_count = int(result.stdout.split()[0]) - 1  # -1 for header
            except:
                # Fallback to slower method  
                row_count = sum(1 for _ in open(csv_file)) - 1
            file_sizes[csv_file] = row_count
            logger.info(f"{csv_file.name}: {row_count:,} rows")
        
        total_rows = sum(file_sizes.values())
        logger.info(f"Total rows across all files: {total_rows:,}")
    
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}...")
        try:
            # Determine sample size for this file
            if stratified and sample_size and csv_file in file_sizes:
                # Proportional sampling based on file size
                file_sample_size = max(100, int(sample_size * (file_sizes[csv_file] / total_rows)))
            else:
                file_sample_size = sample_size if sample_size else None
            
            # Memory-efficient chunked loading with sampling
            if file_sample_size and file_sizes.get(csv_file, 0) > file_sample_size:
                sampled_chunks = []
                rows_needed = file_sample_size
                chunk_size = 50000
                
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                    # Sample from this chunk proportionally
                    chunk_sample_size = min(len(chunk), rows_needed)
                    sampled = chunk.sample(n=chunk_sample_size, random_state=42)
                    sampled_chunks.append(sampled)
                    rows_needed -= chunk_sample_size
                    
                    if rows_needed <= 0:
                        break
                
                df = pd.concat(sampled_chunks, ignore_index=True)
            else:
                # File is small enough to load entirely
                df = pd.read_csv(csv_file, low_memory=False)
            
            # Sample from this file
            if file_sample_size and len(df) > file_sample_size:
                df = df.sample(n=file_sample_size, random_state=42)
            
            if attack_types:
                label_col = 'Label' if 'Label' in df.columns else ' Label'
                df = df[df[label_col].isin(attack_types)]
            
            dfs.append(df)
            logger.info(f"  Loaded {len(df):,} samples from {csv_file.name}")
        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total samples loaded: {len(combined_df):,}")
    
    # Standardize label column
    label_col = 'Label' if 'Label' in combined_df.columns else ' Label'
    
    # Balance classes with attack type stratification
    if balance_classes:
        benign_df = combined_df[combined_df[label_col] == 'BENIGN']
        attack_df = combined_df[combined_df[label_col] != 'BENIGN']
        
        logger.info(f"\nOriginal distribution:")
        logger.info(f"  Benign: {len(benign_df):,}")
        logger.info(f"  Attack: {len(attack_df):,}")
        
        # Get all unique attack types
        attack_types_present = attack_df[label_col].unique()
        logger.info(f"\nAttack types present: {list(attack_types_present)}")
        
        # Strategy: Oversample benign to match attack samples
        # and ensure all attack types are well represented
        
        # Ensure each attack type has minimum representation
        attack_dfs = []
        for attack_type in attack_types_present:
            attack_subset = attack_df[attack_df[label_col] == attack_type]
            # Ensure minimum samples per attack type
            n_samples = max(len(attack_subset), min_samples_per_attack)
            sampled = attack_subset.sample(n=n_samples, random_state=42, replace=(n_samples > len(attack_subset)))
            attack_dfs.append(sampled)
            logger.info(f"  {attack_type}: {len(sampled):,} samples (original: {len(attack_subset):,})")
        
        # Combine all attack samples
        attack_df_balanced = pd.concat(attack_dfs, ignore_index=True)
        
        # Oversample benign to match total attack samples (50/50 balance)
        target_benign = len(attack_df_balanced)
        if len(benign_df) < target_benign:
            # Oversample benign with replacement
            benign_df_balanced = benign_df.sample(n=target_benign, random_state=42, replace=True)
            logger.info(f"\nOversampled benign from {len(benign_df):,} to {target_benign:,}")
        else:
            # Downsample benign if we have more than needed
            benign_df_balanced = benign_df.sample(n=target_benign, random_state=42)
            logger.info(f"\nDownsampled benign from {len(benign_df):,} to {target_benign:,}")
        
        combined_df = pd.concat([benign_df_balanced, attack_df_balanced], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"\nBalanced dataset: {len(benign_df_balanced):,} benign, {len(attack_df_balanced):,} attack")
        logger.info(f"Final dataset size: {len(combined_df):,}")
        logger.info(f"Attack types represented: {len(attack_types_present)}")
    
    return combined_df


def create_synthetic_ddos_data(
    n_samples: int = 10000,
    n_features: int = 76,
    attack_ratio: float = 0.3,
    random_state: int = 42,
    difficulty: str = 'medium'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic DDoS data for testing when real dataset is not available.
    
    This generates data with patterns similar to real DDoS traffic with
    configurable difficulty levels to create more realistic and challenging
    classification tasks.
    
    Difficulty levels:
    - 'easy': Clearly separable classes (for debugging/baseline)
    - 'medium': Moderate overlap between classes (realistic scenario)  
    - 'hard': Significant overlap, mimics sophisticated attacks
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        attack_ratio: Ratio of attack samples
        random_state: Random seed
        difficulty: Classification difficulty ('easy', 'medium', 'hard')
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    n_attacks = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attacks
    
    # Difficulty settings control class separation
    difficulty_params = {
        'easy': {'mean_sep': 0.4, 'std_benign': 0.15, 'std_attack': 0.2, 'noise': 0.05},
        'medium': {'mean_sep': 0.2, 'std_benign': 0.25, 'std_attack': 0.3, 'noise': 0.15},
        'hard': {'mean_sep': 0.1, 'std_benign': 0.3, 'std_attack': 0.35, 'noise': 0.25}
    }
    params = difficulty_params.get(difficulty, difficulty_params['medium'])
    
    # ===== BENIGN TRAFFIC =====
    # Base features with realistic variance
    benign_base = np.random.normal(loc=0.5, scale=params['std_benign'], size=(n_benign, n_features))
    
    # Add feature correlations (real traffic has correlated features)
    # E.g., packet count correlates with byte count
    correlation_matrix = np.eye(n_features)
    for i in range(min(10, n_features-1)):
        correlation_matrix[i, i+1] = 0.5
        correlation_matrix[i+1, i] = 0.5
    
    # Apply mild correlations to benign
    benign_features = benign_base + np.random.multivariate_normal(
        np.zeros(n_features), 
        correlation_matrix * 0.01, 
        size=n_benign
    )
    
    # Some benign traffic has burst patterns (legitimate high traffic)
    burst_mask = np.random.random(n_benign) < 0.1  # 10% burst traffic
    benign_features[burst_mask, :10] *= 1.5  # Higher values for burst
    
    # ===== ATTACK TRAFFIC =====
    # Different attack types with varying characteristics
    n_volumetric = int(n_attacks * 0.4)  # 40% volumetric (easier to detect)
    n_protocol = int(n_attacks * 0.3)     # 30% protocol attacks
    n_application = n_attacks - n_volumetric - n_protocol  # 30% application layer (hardest)
    
    # --- Volumetric attacks (e.g., UDP flood) ---
    # Higher mean, but with significant overlap with benign bursts
    volumetric = np.random.normal(
        loc=0.5 + params['mean_sep'] * 1.5, 
        scale=params['std_attack'], 
        size=(n_volumetric, n_features)
    )
    # High packet rates and byte counts
    volumetric[:, :5] = np.random.exponential(scale=0.8, size=(n_volumetric, 5))
    # Add noise to make some look more benign
    volumetric += np.random.normal(0, params['noise'], volumetric.shape)
    
    # --- Protocol attacks (e.g., SYN flood) ---
    protocol = np.random.normal(
        loc=0.5 + params['mean_sep'], 
        scale=params['std_attack'], 
        size=(n_protocol, n_features)
    )
    # Specific flag patterns (SYN, RST flags elevated)
    if n_features > 50:
        protocol[:, 46:52] = np.random.exponential(scale=1.0, size=(n_protocol, min(6, n_features-46)))
    # Shorter flow durations
    protocol[:, 0] = np.random.exponential(scale=0.3, size=n_protocol)
    protocol += np.random.normal(0, params['noise'], protocol.shape)
    
    # --- Application layer attacks (e.g., HTTP flood, Slowloris) ---
    # These INTENTIONALLY look very similar to benign traffic
    application = np.random.normal(
        loc=0.5 + params['mean_sep'] * 0.5,  # Much closer to benign
        scale=params['std_benign'],  # Same variance as benign!
        size=(n_application, n_features)
    )
    # Subtle differences: slightly higher inter-arrival time variance
    if n_features > 25:
        application[:, 22:26] *= 1.2  # IAT features
    # Some samples are nearly indistinguishable
    stealth_mask = np.random.random(n_application) < 0.3  # 30% very stealthy
    application[stealth_mask] = np.random.normal(
        loc=0.52,  # Almost same as benign
        scale=params['std_benign'],
        size=(stealth_mask.sum(), n_features)
    )
    application += np.random.normal(0, params['noise'] * 1.5, application.shape)
    
    # Combine attack types
    attack_features = np.vstack([volumetric, protocol, application])
    
    # Shuffle attack types
    attack_shuffle = np.random.permutation(n_attacks)
    attack_features = attack_features[attack_shuffle]
    
    # Clip to reasonable ranges
    benign_features = np.clip(benign_features, 0, 2)
    attack_features = np.clip(attack_features, 0, 3)
    
    # Add label noise (some mislabeled samples, simulating real-world data)
    if difficulty in ['medium', 'hard']:
        noise_rate = 0.02 if difficulty == 'medium' else 0.05
        # Some benign samples that look like attacks
        noisy_benign = np.random.random(n_benign) < noise_rate
        benign_features[noisy_benign] = benign_features[noisy_benign] * 1.3 + 0.2
    
    # Combine and create labels
    X = np.vstack([benign_features, attack_features]).astype(np.float32)
    y = np.array([0] * n_benign + [1] * n_attacks)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    logger.info(f"Generated synthetic data: {n_benign} benign, {n_attacks} attack samples")
    
    return X, y


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    test_size: float = 0.2,
    val_size: float = 0.1,
    model_type: str = 'cnn',
    sequence_length: int = 10,
    random_state: int = 42
) -> Dict[str, DataLoader]:
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        X: Feature array
        y: Label array
        batch_size: Batch size for data loaders
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        model_type: Type of model ('cnn', 'lstm', 'transformer')
        sequence_length: Sequence length for sequential models
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets based on model type
    if model_type in ['lstm', 'transformer']:
        train_dataset = SequentialDataset(X_train, y_train, sequence_length)
        val_dataset = SequentialDataset(X_val, y_val, sequence_length)
        test_dataset = SequentialDataset(X_test, y_test, sequence_length)
    else:
        train_dataset = CICDDoS2019Dataset(X_train, y_train, for_cnn=(model_type == 'cnn'))
        val_dataset = CICDDoS2019Dataset(X_val, y_val, for_cnn=(model_type == 'cnn'))
        test_dataset = CICDDoS2019Dataset(X_test, y_test, for_cnn=(model_type == 'cnn'))
    
    # Create data loaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    return loaders


def download_sample_data(output_dir: str = './data'):
    """
    Provide instructions for downloading CICDDoS2019 dataset.
    
    The actual dataset must be downloaded from:
    https://www.unb.ca/cic/datasets/ddos-2019.html
    
    Args:
        output_dir: Directory to save data
    """
    instructions = """
    ============================================================
    CICDDoS2019 Dataset Download Instructions
    ============================================================
    
    The CICDDoS2019 dataset is available from the Canadian Institute 
    for Cybersecurity (CIC) at the University of New Brunswick.
    
    1. Visit: https://www.unb.ca/cic/datasets/ddos-2019.html
    
    2. Download the CSV files for the desired attack types
    
    3. Place the CSV files in the data directory:
       {output_dir}
    
    Alternatively, you can use the synthetic data generator:
    
    >>> from data_loader import create_synthetic_ddos_data
    >>> X, y = create_synthetic_ddos_data(n_samples=50000)
    
    ============================================================
    """
    print(instructions.format(output_dir=output_dir))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


if __name__ == '__main__':
    # Demo: Create synthetic data and prepare loaders
    print("Creating synthetic DDoS data for demonstration...")
    
    X, y = create_synthetic_ddos_data(n_samples=10000)
    
    print("\nPreparing data loaders for CNN model...")
    loaders = prepare_data_loaders(X, y, model_type='cnn')
    
    print(f"\nTrain batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Test a batch
    for batch_X, batch_y in loaders['train']:
        print(f"\nBatch shape: {batch_X.shape}")
        print(f"Labels shape: {batch_y.shape}")
        break
