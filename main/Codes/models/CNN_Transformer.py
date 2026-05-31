import torch
import torch.nn as nn
import math

class ConvBranch(nn.Module):
    """
    Single convolutional branch for extracting features at a specific temporal scale.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for adaptive feature recalibration.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excite
        y = self.excite(y).view(b, c, 1)
        # Recalibrate
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model by conditionally applying to 1::2
        pe[:, 1::2] = torch.cos(position * div_term)[:pe[:, 1::2].shape[0]]
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class CNN_Transformer(nn.Module):
    """
    CNN-Transformer Hybrid for single-channel EEG classification.
    
    1. Multi-scale 1D-CNN (preserves more temporal resolution)
    2. Squeeze-and-Excitation (adaptive feature weighting)
    3. Transformer Encoder (global temporal dependencies)
    4. Global Pooling & Classifier
    """
    def __init__(self, n_classes=2, n_samples=1024, filters_per_branch=16, d_model=64, n_heads=4, tf_layers=2, dropout=0.3):
        super().__init__()
        
        # --- 1. Multi-Branch CNN (Spatial/Frequency Feature Extraction) ---
        # 4 branches with different kernel sizes
        self.branch_theta = ConvBranch(1, filters_per_branch, kernel_size=64)  # ~8 Hz
        self.branch_alpha = ConvBranch(1, filters_per_branch, kernel_size=32)  # ~16 Hz
        self.branch_beta  = ConvBranch(1, filters_per_branch, kernel_size=16)  # ~32 Hz
        self.branch_gamma = ConvBranch(1, filters_per_branch, kernel_size=8)   # ~64 Hz
        
        cnn_out_channels = filters_per_branch * 4  # e.g., 64 channels
        
        # Gentle pooling (downsample by 4 instead of 64+ like in MBCN)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Projection to d_model (if needed, but here cnn_out_channels == d_model)
        if cnn_out_channels != d_model:
            self.proj = nn.Conv1d(cnn_out_channels, d_model, 1)
        else:
            self.proj = nn.Identity()
            
        # --- 2. Squeeze-and-Excitation Block ---
        self.se = SEBlock(d_model, reduction=4)
        
        # --- 3. Transformer Encoder ---
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=n_samples // 4 + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        
        # --- 4. Global Pooling ---
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # --- 5. Classifier ---
        pool_out = d_model * 2  # avg + max
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_out, 64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, n_classes)
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
        # x shape: (batch_size, 1, 1024)
        
        # 1. CNN Extraction (Multi-scale)
        bt = self.branch_theta(x)
        ba = self.branch_alpha(x)
        bb = self.branch_beta(x)
        bg = self.branch_gamma(x)
        
        # Concatenate branches along channel dimension
        x_cnn = torch.cat([bt, ba, bb, bg], dim=1)  # (batch, 64, 1024)
        
        # Gentle Pooling
        x_pool = self.pool(x_cnn)                   # (batch, 64, 256)
        
        # Project & SE Recalibration
        x_proj = self.proj(x_pool)                  # (batch, d_model, 256)
        x_se = self.se(x_proj)                      # (batch, d_model, 256)
        
        # 2. Transformer
        # Transformer expects (batch, seq_len, features) for batch_first=True
        x_trans_in = x_se.transpose(1, 2)           # (batch, 256, d_model)
        x_pos = self.pos_encoder(x_trans_in)        # add positional encoding
        x_trans_out = self.transformer(x_pos)       # (batch, 256, d_model)
        
        # 3. Global Pooling
        # Transpose back for 1D pooling: (batch, d_model, 256)
        x_trans_out = x_trans_out.transpose(1, 2)
        
        avg_pool = self.global_avg(x_trans_out).squeeze(-1)  # (batch, d_model)
        max_pool = self.global_max(x_trans_out).squeeze(-1)  # (batch, d_model)
        pooled = torch.cat([avg_pool, max_pool], dim=1)      # (batch, d_model * 2)
        
        # 4. Classifier
        logits = self.classifier(pooled)                     # (batch, n_classes)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model shapes and parameters
    model = CNN_Transformer(n_classes=3)
    print(f"Model: CNN-Transformer Hybrid")
    print(f"Total Parameters: {model.count_parameters():,}")
    
    dummy_input = torch.randn(4, 1, 1024)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
