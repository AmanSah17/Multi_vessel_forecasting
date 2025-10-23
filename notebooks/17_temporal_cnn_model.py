"""
Temporal CNN Model for Maritime Vessel Forecasting
- 1D Convolutional layers for temporal pattern extraction
- Dilated convolutions for multi-scale temporal patterns
- Residual connections for better gradient flow
- Batch normalization for training stability
- Comparison with LSTM model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TemporalCNNBlock(nn.Module):
    """Temporal CNN block with dilated convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TemporalCNNBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else \
                       nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class TemporalCNNModel(nn.Module):
    """Temporal CNN for time series prediction."""
    
    def __init__(self, input_size, output_size=4, num_filters=64, num_layers=4, 
                 kernel_size=3, dropout=0.2):
        super(TemporalCNNModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Input projection
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        
        # Temporal CNN blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            self.blocks.append(
                TemporalCNNBlock(
                    num_filters, num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Global average pooling + FC layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, time, features)
        # Convert to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Temporal CNN blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # FC layers
        x = self.fc(x)
        
        return x


class TemporalCNNWithAttention(nn.Module):
    """Temporal CNN with attention mechanism."""
    
    def __init__(self, input_size, output_size=4, num_filters=64, num_layers=4,
                 kernel_size=3, dropout=0.2, num_heads=4):
        super(TemporalCNNWithAttention, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Input projection
        self.input_proj = nn.Conv1d(input_size, num_filters, 1)
        
        # Temporal CNN blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                TemporalCNNBlock(
                    num_filters, num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            num_filters, num_heads, dropout=dropout, batch_first=True
        )
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, time, features)
        batch_size, seq_len, _ = x.shape
        
        # Convert to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Temporal CNN blocks
        for block in self.blocks:
            x = block(x)
        
        # Convert back to (batch, time, features) for attention
        x = x.transpose(1, 2)
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, features)
        
        # FC layers
        x = self.fc(x)
        
        return x


class HybridLSTMCNN(nn.Module):
    """Hybrid model combining LSTM and CNN."""
    
    def __init__(self, input_size, output_size=4, hidden_size=128, num_layers=2,
                 num_filters=64, kernel_size=3, dropout=0.2):
        super(HybridLSTMCNN, self).__init__()
        
        # CNN branch
        self.cnn_proj = nn.Conv1d(input_size, num_filters, 1)
        self.cnn_block = TemporalCNNBlock(
            num_filters, num_filters, kernel_size=kernel_size, dropout=dropout
        )
        
        # LSTM branch
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_filters + hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, time, features)
        
        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, features, time)
        x_cnn = self.cnn_proj(x_cnn)
        x_cnn = self.cnn_block(x_cnn)
        x_cnn = F.adaptive_avg_pool1d(x_cnn, 1).squeeze(-1)  # (batch, filters)
        
        # LSTM branch
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden)
        x_lstm = h_n[-1]  # (batch, hidden)
        
        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)
        x_out = self.fusion(x_fused)
        
        return x_out


if __name__ == "__main__":
    # Test models
    batch_size = 32
    seq_len = 60
    input_size = 50  # Advanced features
    output_size = 4
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Test Temporal CNN
    print("Testing Temporal CNN...")
    cnn_model = TemporalCNNModel(input_size, output_size)
    y_cnn = cnn_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_cnn.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
    
    # Test Temporal CNN with Attention
    print("\nTesting Temporal CNN with Attention...")
    cnn_attn_model = TemporalCNNWithAttention(input_size, output_size)
    y_cnn_attn = cnn_attn_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_cnn_attn.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cnn_attn_model.parameters()):,}")
    
    # Test Hybrid LSTM-CNN
    print("\nTesting Hybrid LSTM-CNN...")
    hybrid_model = HybridLSTMCNN(input_size, output_size)
    y_hybrid = hybrid_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_hybrid.shape}")
    print(f"  Parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")

