import torch
import torch.nn as nn

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
        # Pool by 4 to reduce temporal dimension early (1024 -> 256)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return x

class MBCN_BiLSTM(nn.Module):
    """
    Multi-Branch CNN combined with Bidirectional LSTM for EEG classification.
    
    1. Multi-Branch CNN extracts frequency-band spatial features.
    2. BiLSTM models the forward and backward temporal sequence of those features.
    3. Global pooling + Classifier makes the final prediction.
    """
    def __init__(self, n_classes=2, n_samples=1024, filters_per_branch=16, lstm_hidden=32, dropout=0.4):
        super().__init__()
        
        # --- 1. Multi-Branch CNN (Spatial/Frequency Feature Extraction) ---
        # 4 branches with different kernel sizes to capture different EEG sub-bands
        self.branch_theta = ConvBranch(1, filters_per_branch, kernel_size=128)  # Theta/Delta (~4 Hz)
        self.branch_alpha = ConvBranch(1, filters_per_branch, kernel_size=64)   # Alpha (~8 Hz)
        self.branch_beta  = ConvBranch(1, filters_per_branch, kernel_size=32)   # Beta (~16 Hz)
        self.branch_gamma = ConvBranch(1, filters_per_branch, kernel_size=16)   # Gamma (~32 Hz)
        
        cnn_out_channels = filters_per_branch * 4  # 64 channels
        
        # --- 2. Bidirectional LSTM (Temporal Sequence Modeling) ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # BiLSTM outputs hidden_size * 2 because it combines forward and backward passes
        lstm_out_channels = lstm_hidden * 2  # 64 channels
        
        # --- 3. Global Pooling ---
        # Capture both the average signal trend and the peak signal spikes
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)
        
        # --- 4. Classifier ---
        pool_out = lstm_out_channels * 2  # 128 features (64 avg + 64 max)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_out, 64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout * 0.5), # slightly lighter dropout in hidden layer
            nn.Linear(64, n_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Convs, Xavier for Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, 1, 1024)
        
        # 1. CNN Extraction
        bt = self.branch_theta(x)  # (batch, 16, 256)
        ba = self.branch_alpha(x)  # (batch, 16, 256)
        bb = self.branch_beta(x)   # (batch, 16, 256)
        bg = self.branch_gamma(x)  # (batch, 16, 256)
        
        # Concatenate branches along channel dimension
        cnn_out = torch.cat([bt, ba, bb, bg], dim=1)  # shape: (batch, 64, 256)
        
        # 2. Prepare for LSTM
        # LSTM batch_first=True expects shape: (batch, seq_len, features)
        # So we transpose the channels and time dimensions
        lstm_in = cnn_out.transpose(1, 2)  # shape: (batch, 256, 48)
        
        # 3. BiLSTM Pass
        # lstm_out contains all hidden states across time
        lstm_out, (hn, cn) = self.lstm(lstm_in)  # lstm_out shape: (batch, 256, 64)
        
        # 4. Transpose back for 1D pooling
        lstm_out = lstm_out.transpose(1, 2)  # shape: (batch, 64, 256)
        
        # 5. Pooling
        avg_pool = self.global_avg(lstm_out).squeeze(-1)  # (batch, 64)
        max_pool = self.global_max(lstm_out).squeeze(-1)  # (batch, 64)
        pooled = torch.cat([avg_pool, max_pool], dim=1)   # (batch, 128)
        
        # 6. Classifier
        logits = self.classifier(pooled)  # (batch, n_classes)
        return logits

    def count_parameters(self):
        """Count total trainable parameters to match the API in train_mstcnn.py."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Quick standard test to verify shapes and parameters
    model = MBCN_BiLSTM(n_classes=3)
    print(f"Model: MBCN-BiLSTM")
    print(f"Total Parameters: {model.count_parameters():,}")
    
    dummy_input = torch.randn(4, 1, 1024)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") 
