"""
EEG-TCN: Temporal Convolutional Network for Single-Channel EEG

Designed for small-data, single-channel EEG cognitive load classification.

Key advantages over CNN-Transformer and MBCN-BiLSTM:
  - NO pooling anywhere: full temporal resolution preserved (all 1024 steps)
  - Dilated convolutions: receptive field covers entire 2s window in ~7 layers
  - Residual connections: raw signal info never permanently lost
  - Pure convolutions: stable training from step 1 (no warmup needed)
  - ~15K parameters: properly sized for ~3000 training samples

Architecture:
    Raw EEG (1, 1024)
        |
    Input Projection (1 -> 16 channels)
        |
    TCN Block (d=1)   + residual   |
    TCN Block (d=2)   + residual   | Receptive field
    TCN Block (d=4)   + residual   | grows exponentially
    TCN Block (d=8)   + residual   | with each block
    TCN Block (d=16)  + residual   |
    TCN Block (d=32)  + residual   |
    TCN Block (d=64)  + residual   v  -> covers full 1024 samples
        |
    SE Block (adaptive feature weighting)
        |
    Global Avg Pool + Global Max Pool
        |
    Dropout -> Linear -> n_classes
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns which feature channels matter most."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1)
        return x * y


class TCNBlock(nn.Module):
    """
    Single TCN residual block: two dilated convolutions + skip connection.
    
    No pooling, no downsampling. Output has the exact same temporal
    length as the input. The dilation parameter controls how far
    back in time this block can "see".
    
    At 512 Hz with kernel_size=5:
        dilation=1  -> sees  4 samples  ->  7.8 ms
        dilation=2  -> sees  8 samples  -> 15.6 ms
        dilation=4  -> sees 16 samples  -> 31.3 ms
        dilation=8  -> sees 32 samples  -> 62.5 ms
        dilation=16 -> sees 64 samples  -> 125 ms
        dilation=32 -> sees 128 samples -> 250 ms
        dilation=64 -> sees 256 samples -> 500 ms
    """
    def __init__(self, n_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # "Same" padding so output length == input length
        padding = dilation * (kernel_size - 1) // 2
        
        self.net = nn.Sequential(
            # First dilated conv
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            # Second dilated conv
            nn.Conv1d(n_channels, n_channels, kernel_size,
                      dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Residual connection: original signal always preserved
        return x + self.net(x)


class EEG_TCN(nn.Module):
    """
    Temporal Convolutional Network for single-channel EEG classification.
    
    Args:
        n_classes:   Number of output classes (2 for binary)
        n_samples:   Samples per window (1024 = 2s @ 512Hz)
        n_channels:  Feature channels throughout the TCN (default 16)
        kernel_size: Convolution kernel size (default 5)
        dropout:     Dropout rate (default 0.15, light for small model)
    """
    def __init__(self, n_classes=2, n_samples=1024, n_channels=16,
                 kernel_size=5, dropout=0.15):
        super().__init__()
        
        # --- 1. Input Projection ---
        # Lift raw EEG from 1 channel to n_channels feature space
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.ELU(inplace=True),
        )
        
        # --- 2. TCN Backbone ---
        # Each block doubles the dilation -> receptive field grows exponentially
        # Total RF = 1 + 2*(k-1)*(1+2+4+8+16+32+64) = 1 + 8*127 = 1017
        # That covers 99.3% of the 1024-sample window
        dilations = [1, 2, 4, 8, 16, 32, 64]
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(n_channels, kernel_size, d, dropout)
            for d in dilations
        ])
        
        # --- 3. SE Block ---
        # After TCN extraction, adaptively weight which channels matter
        self.se = SEBlock(n_channels, reduction=4)
        
        # --- 4. Global Pooling ---
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # --- 5. Classifier ---
        # Concat avg+max = n_channels*2 features -> n_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_channels * 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, 1, 1024)
        
        # 1. Project to feature space
        x = self.input_proj(x)          # (batch, 16, 1024)
        
        # 2. TCN blocks with residual connections
        for block in self.tcn_blocks:
            x = block(x)               # (batch, 16, 1024) — same size throughout!
        
        # 3. Adaptive feature weighting
        x = self.se(x)                 # (batch, 16, 1024)
        
        # 4. Global pooling
        avg = self.global_avg(x).squeeze(-1)  # (batch, 16)
        mx  = self.global_max(x).squeeze(-1)  # (batch, 16)
        pooled = torch.cat([avg, mx], dim=1)  # (batch, 32)
        
        # 5. Classify
        logits = self.classifier(pooled)      # (batch, n_classes)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EEG_TCN(n_classes=2)
    print(f"Model: EEG-TCN")
    print(f"Total Parameters: {model.count_parameters():,}")
    
    # Verify receptive field covers the window
    k, dilations = 5, [1, 2, 4, 8, 16, 32, 64]
    rf = 1 + 2 * (k - 1) * sum(dilations)
    print(f"Receptive Field: {rf} / 1024 samples ({rf/1024*100:.1f}%)")
    
    dummy = torch.randn(4, 1, 1024)
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
