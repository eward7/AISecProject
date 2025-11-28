"""
Deep Learning Models for DDoS Detection

This module implements various deep learning architectures for detecting
DDoS attacks in network traffic:
- CNN (Convolutional Neural Network)
- LSTM (Long Short-Term Memory)
- Transformer
- Hybrid CNN-LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNDetector(nn.Module):
    """
    1D Convolutional Neural Network for DDoS Detection.
    
    This model treats network flow features as a 1D signal and applies
    convolutional filters to extract patterns indicative of attacks.
    """
    
    def __init__(
        self,
        input_features: int = 76,
        num_classes: int = 2,
        conv_channels: Tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 3,
        fc_hidden: int = 256,
        dropout: float = 0.5
    ):
        """
        Initialize the CNN detector.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes (2 for binary classification)
            conv_channels: Number of channels in each conv layer
            kernel_size: Size of convolutional kernels
            fc_hidden: Size of fully connected hidden layer
            dropout: Dropout rate
        """
        super(CNNDetector, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Convolutional layers
        layers = []
        in_channels = 1
        current_length = input_features
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2) if current_length > 4 else nn.Identity(),
            ])
            in_channels = out_channels
            if current_length > 4:
                current_length = current_length // 2
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after conv layers
        self.conv_output_size = conv_channels[-1] * current_length
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, features) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.conv_layers(x)
        x = self.fc(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class LSTMDetector(nn.Module):
    """
    LSTM-based Network for DDoS Detection.
    
    This model is designed to process sequential network traffic data
    and capture temporal patterns in attack traffic.
    """
    
    def __init__(
        self,
        input_features: int = 76,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize the LSTM detector.
        
        Args:
            input_features: Number of input features per time step
            num_classes: Number of output classes
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
        """
        super(LSTMDetector, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            return_attention: Whether to return attention weights
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle 2D input (single timestep)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDetector(nn.Module):
    """
    Transformer-based Network for DDoS Detection.
    
    This model uses self-attention to capture relationships between
    different network flow features and temporal patterns.
    """
    
    def __init__(
        self,
        input_features: int = 76,
        num_classes: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize the Transformer detector.
        
        Args:
            input_features: Number of input features per time step
            num_classes: Number of output classes
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super(TransformerDetector, self).__init__()
        
        self.d_model = d_model
        self.input_features = input_features
        
        # Input projection
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM Network for DDoS Detection.
    
    This model combines CNN for feature extraction with LSTM for
    temporal pattern recognition.
    """
    
    def __init__(
        self,
        input_features: int = 76,
        num_classes: int = 2,
        cnn_channels: Tuple[int, ...] = (32, 64),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize the hybrid model.
        
        Args:
            input_features: Number of input features
            num_classes: Number of output classes
            cnn_channels: CNN channel configurations
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(HybridCNNLSTM, self).__init__()
        
        self.input_features = input_features
        
        # CNN feature extractor
        cnn_layers = []
        in_channels = 1
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) or (batch, features)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        batch_size, seq_len, features = x.shape
        
        # Reshape for CNN: process each timestep
        x = x.view(batch_size * seq_len, 1, features)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch*seq, channels, features)
        
        # Global average pooling over features
        x = x.mean(dim=2)  # (batch*seq, channels)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        x = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def create_model(
    model_type: str = 'cnn',
    input_features: int = 76,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create detection models.
    
    Args:
        model_type: Type of model ('cnn', 'lstm', 'transformer', 'hybrid')
        input_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    models = {
        'cnn': CNNDetector,
        'lstm': LSTMDetector,
        'transformer': TransformerDetector,
        'hybrid': HybridCNNLSTM
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    model_class = models[model_type]
    model = model_class(input_features=input_features, num_classes=num_classes, **kwargs)
    
    logger.info(f"Created {model_type.upper()} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model


def load_pretrained_model(
    model_path: str,
    model_type: str = 'cnn',
    device: str = 'cpu'
) -> nn.Module:
    """
    Load a pretrained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {})
    input_features = config.get('input_features', 76)
    num_classes = config.get('num_classes', 2)
    
    # Create model
    model = create_model(model_type, input_features, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained {model_type} model from {model_path}")
    
    return model


if __name__ == '__main__':
    # Demo: Create and test each model
    print("Testing DDoS Detection Models\n")
    
    batch_size = 32
    seq_len = 10
    input_features = 76
    
    # Test CNN
    print("=" * 50)
    print("Testing CNN Detector")
    cnn = create_model('cnn', input_features=input_features)
    x_cnn = torch.randn(batch_size, 1, input_features)
    out_cnn = cnn(x_cnn)
    print(f"Input shape: {x_cnn.shape}")
    print(f"Output shape: {out_cnn.shape}")
    
    # Test LSTM
    print("\n" + "=" * 50)
    print("Testing LSTM Detector")
    lstm = create_model('lstm', input_features=input_features)
    x_lstm = torch.randn(batch_size, seq_len, input_features)
    out_lstm = lstm(x_lstm)
    print(f"Input shape: {x_lstm.shape}")
    print(f"Output shape: {out_lstm.shape}")
    
    # Test Transformer
    print("\n" + "=" * 50)
    print("Testing Transformer Detector")
    transformer = create_model('transformer', input_features=input_features)
    x_trans = torch.randn(batch_size, seq_len, input_features)
    out_trans = transformer(x_trans)
    print(f"Input shape: {x_trans.shape}")
    print(f"Output shape: {out_trans.shape}")
    
    # Test Hybrid
    print("\n" + "=" * 50)
    print("Testing Hybrid CNN-LSTM Detector")
    hybrid = create_model('hybrid', input_features=input_features)
    x_hybrid = torch.randn(batch_size, seq_len, input_features)
    out_hybrid = hybrid(x_hybrid)
    print(f"Input shape: {x_hybrid.shape}")
    print(f"Output shape: {out_hybrid.shape}")
