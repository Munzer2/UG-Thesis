"""

MSTCNN-A: Multi scale Temporal CNN with attention


Custom DL model for single channel EEG cognitive load classification

Architecture: 
1) Multi scale convolution --> 5 parallel branches with different kernel sizes
to capture the bands 

2) Squeeze excitation attention ---> learns which frequency scales matter most 
for each input. 

3) Temporal refinement ---> further temporal pattern extraction 

4) Dual global pooling ---> captures both average and peak info 

5) Classifier ---> compact fully connected head 

Designed for: 
- Single channel EEG (1 electrode) 
- Small datasets (8-15 subjects) 
- 2 second windows @ 512 Hz (1024 samples) 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Squeeze extraction block ( channel attention ) 
# purpose: After concatenating features from 5 frequency scales, 
# not all scales are equally useful. SE learns to weight them
# if theta/beta ratio matters most for cognitive load, it will upweight 
# those channels.

class SqueezeExcitation(nn.Module): 
    """
    Channel attention mechanism. 
    Learns to emphasize important frequency-scale channels
    and suppress less useful ones.
    
    Args:
        channels: Number of input channels (= total filters
                  from all multi-scale branches)
        reduction: Compression ratio for the bottleneck
                   (higher = fewer params, more compression)
    """
    def __init__(self, channels, reduction = 4): 
        super().__init__() 
        mid = max(channels // reduction, 4) # bottleneck size.
        self.squeeze = nn.AdaptiveAvgPool1d(1) # Global pooling
        self.excite = nn.Sequential( 
            nn.Linear(channels, mid),  
            nn.ReLU(inplace = True), 
            nn.Linear(mid, channels ), 
            nn.Sigmoid() 
        )

    def forward(self, x ): 
        b,c, _ = x.shape 
        # Squeeze : (batch, channels, time) --> (batch, channels, 1) -> (batch, channels)
        w = self.squeeze(x).view(b,c) 
        w = self.excite(w).view(b,c,1) # (batch, channels, 1)
        # multiply each channel by its learned weight
        return x * w 


# ==========================================================
# 2. MULTI-SCALE CONVOLUTION BRANCH
# ==========================================================
# PURPOSE: Different EEG frequency bands have different
# temporal characteristics. A convolution with kernel_size=k
# is most sensitive to oscillations with period ≈ k samples.
#
# At 512 Hz:
#   kernel=128 → sensitive to ~4 Hz   (δ band)
#   kernel=64  → sensitive to ~8 Hz   (θ band)
#   kernel=32  → sensitive to ~16 Hz  (α band)
#   kernel=16  → sensitive to ~32 Hz  (β band)
#   kernel=8   → sensitive to ~64 Hz  (γ band)
#
# Each branch independently learns filters for its scale,
# then outputs are concatenated before attention.
# ==========================================================

class ConvBranch(nn.Module):
    """
    Single conv branch for one temporal scale. 
    Each branch --> conv1D --> batchNorm ---> ELU ---> AvgPool
    Args: 
        in_channels : 1 for raw EEG
        out_channels: Number of filters to learn
        kernel_size : Temporal filter width ( determines which frequency 
                        band this captures.)
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = kernel_size // 2, # keep time dim same
            bias = False
        )
        self.bn = nn.BatchNorm1d(out_channels) 
        self.elu = nn.ELU(inplace = True) 
        # Pool by 4 to reduce temporal dimension early
        # 1024 → 256 samples after this
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        x = self.conv(x)     # Learn frequency-specific filters
        x = self.bn(x)       # Normalize activations
        x = self.elu(x)      # Non-linearity (ELU handles negatives well)
        x = self.pool(x)     # Reduce temporal dimension
        return x 

        
        
#  Main model: MSTCNN-A
class MSTCNN_A(nn.Module):
    """
    Multi-Scale Temporal CNN with Attention for single-channel
    EEG classification.
    
    Architecture:
        Input (1, 1024)
            │
        ┌───┼──────┬──────┬──────┬──────┐
        │   │      │      │      │      │
      Conv Conv  Conv  Conv  Conv     (5 parallel branches)
      k128  k64   k32   k16   k8     (δ  θ  α  β  γ)
        │   │      │      │      │
        └───┴──────┴──────┴──────┘
            │
        Concatenate  (40 channels × 256 time)
            │
        Squeeze-Excitation Attention
            │
        Temporal Conv (refine patterns)
            │
        Global Avg Pool + Global Max Pool
            │
        Dropout(0.5)
            │
        Dense(64) → ELU → Dropout(0.3)
            │
        Dense(n_classes)
    
    Args:
        n_classes:      Number of output classes (2 or 3)
        n_samples:      Samples per window (default 1024 = 2s @ 512Hz)
        filters_per_branch: Convolution filters per scale branch
        dropout:        Dropout rate after global pooling
    """

    def __init__(self,n_classes = 2, n_samples = 1024, filters_per_branch = 8, dropout= 0.4): 
        super().__init__()
        # --- Multi-Scale Branches ---
        # 5 branches, each tuned to a different EEG frequency band
        # kernel sizes chosen to match EEG band periods at 512 Hz
        self.branch_delta = ConvBranch(1, filters_per_branch, kernel_size=128)  # δ: 1-4 Hz
        self.branch_theta = ConvBranch(1, filters_per_branch, kernel_size=64)   # θ: 4-8 Hz
        self.branch_alpha = ConvBranch(1, filters_per_branch, kernel_size=32)   # α: 8-13 Hz
        self.branch_beta  = ConvBranch(1, filters_per_branch, kernel_size=16)   # β: 13-30 Hz
        self.branch_gamma = ConvBranch(1, filters_per_branch, kernel_size=8)    # γ: 30-50 Hz

        total_channels = filters_per_branch * 5  # 40 channels after concat
        # --- Squeeze-Excitation Attention ---
        # Learns which of the 40 channels (across 5 scales) are most
        # discriminative for cognitive load classification
        self.se = SqueezeExcitation(total_channels, reduction=4)

        # --- Temporal Refinement ---
        # After attention, this block extracts higher-level temporal
        # patterns from the multi-scale representation
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=16,
                      padding=8, groups=total_channels, bias=False),  # Depthwise
            nn.Conv1d(total_channels, 32, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm1d(32),
            nn.ELU(inplace=True),
            nn.AvgPool1d(kernel_size=4, stride=4),  # 256 → 64 time steps
        )

        # --- Global Pooling ---
        # Two complementary views of the feature map:
        #   AvgPool: captures the average activation (stable, smooth)
        #   MaxPool: captures the peak activation (signal highlights)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

        # --- Classifier ---
        # Input: 32 (avg) + 32 (max) = 64 features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),           # Heavy dropout (0.5) to prevent overfitting
            nn.Linear(64, 64),
            nn.ELU(inplace=True),
            nn.Dropout(dropout * 0.6),     # Lighter dropout (0.3) in hidden layer
            nn.Linear(64, n_classes),
        )

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Conv layers, Xavier for Linear.
        This helps the network train faster and more stably on
        small datasets by starting from a good weight distribution.
        """
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
        """
        Forward pass.
        
        Args:
            x: Raw EEG tensor, shape (batch, 1, 1024)
        
        Returns:
            logits: (batch, n_classes) — raw class scores
        """
        # --- Stage 1: Multi-Scale Feature Extraction ---
        # Each branch processes the same raw EEG with a different
        # kernel size, capturing different frequency information
        d = self.branch_delta(x)   # (batch, 8, 256)
        t = self.branch_theta(x)   # (batch, 8, 256)
        a = self.branch_alpha(x)   # (batch, 8, 256)
        b = self.branch_beta(x)    # (batch, 8, 256)
        g = self.branch_gamma(x)   # (batch, 8, 256)

        # Concatenate along channel dimension
        # → (batch, 40, 256) — 40 channels of multi-scale features
        out = torch.cat([d, t, a, b, g], dim=1)

        # --- Stage 2: Channel Attention ---
        # SE block learns which of the 40 channels matter most
        # for distinguishing Simple vs Complex UI cognitive load
        out = self.se(out)  # (batch, 40, 256) — re-weighted

        # --- Stage 3: Temporal Refinement ---
        # Extract higher-level patterns from the attended features
        out = self.temporal_conv(out)  # (batch, 32, 64)

        # --- Stage 4: Dual Global Pooling ---
        # Compress time dimension into a fixed-size vector
        avg = self.global_avg(out).squeeze(-1)  # (batch, 32)
        mx  = self.global_max(out).squeeze(-1)  # (batch, 32)
        out = torch.cat([avg, mx], dim=1)       # (batch, 64)

        # --- Stage 5: Classification ---
        out = self.classifier(out)  # (batch, n_classes)
        return out

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================================
# 4. DATA AUGMENTATION (TRAINING ONLY)
# ==========================================================
# PURPOSE: With only ~400 windows, augmentation prevents
# overfitting by creating realistic variations of the EEG.
# Each technique simulates a real-world EEG variation:
#
#   Noise:     Session-to-session recording noise
#   Scaling:   Electrode impedance / amplitude variations
#   Shift:     Temporal alignment uncertainty
#   Dropout:   Momentary signal loss / micro-artifacts
# ==========================================================
class EEGAugmenter:
    """
    On-the-fly data augmentation for EEG training data.
    
    IMPORTANT: Only applied during training, never during
    evaluation. This ensures test results are unbiased.
    
    Args:
        noise_std:   Std dev of Gaussian noise (relative to signal std)
        scale_range: Min/max amplitude scaling factors
        shift_max:   Max circular shift in samples
        drop_prob:   Probability of zeroing out random samples
        mixup_alpha: Alpha parameter for Beta distribution in Mixup
    """
    def __init__(self, noise_std=0.05, scale_range=(0.8, 1.2),
                 shift_max=50, drop_prob=0.1, mixup_alpha=0.2):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.drop_prob = drop_prob
        self.mixup_alpha = mixup_alpha

    def __call__(self, x, y=None):
        """
        Apply random augmentations to a batch of EEG windows.
        
        Args:
            x: Tensor of shape (batch, 1, 1024)
            y: Optional labels tensor for Mixup (batch,)
        
        Returns:
            Augmented tensor of same shape (and mixed labels if Mixup applied)
        """
        x = x.clone()

        # 1. Gaussian Noise (50% chance)
        # Simulates session-to-session recording variability
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 2. Amplitude Scaling (50% chance)
        # Simulates electrode impedance changes between sessions
        if torch.rand(1).item() < 0.5:
            lo, hi = self.scale_range
            scale = torch.empty(x.size(0), 1, 1).uniform_(lo, hi).to(x.device)
            x = x * scale

        # 3. Circular Time Shift (30% chance)
        # Simulates temporal alignment uncertainty in windowing
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)

        # 4. Random Sample Dropout (20% chance)
        # Simulates momentary signal loss or micro-artifacts
        if torch.rand(1).item() < 0.2:
            mask = torch.rand_like(x) > self.drop_prob
            x = x * mask

        # 5. Mixup (40% chance, only when labels are provided)
        # Interpolates between random pairs in the batch — proven
        # effective for small EEG datasets as it creates virtual
        # training samples in the feature space.
        mixed_y = None
        if y is not None and torch.rand(1).item() < 0.4 and x.size(0) > 1:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)
            perm = torch.randperm(x.size(0)).to(x.device)
            x = lam * x + (1 - lam) * x[perm]
            mixed_y = (y, y[perm], lam)

        if y is not None:
            return x, mixed_y
        return x





# ==========================================================
# QUICK TEST
# ==========================================================
if __name__ == "__main__":
    # Test with dummy data
    model = MSTCNN_A(n_classes=2, n_samples=1024)
    print(f"Model: MSTCNN-A")
    print(f"Parameters: {model.count_parameters():,}")
    print()

    # Print architecture
    print(model)
    print()

    # Test forward pass
    dummy = torch.randn(4, 1, 1024)  # Batch of 4, 1 channel, 1024 samples
    output = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output.detach().numpy()}")
    print()

    # Test augmentation
    aug = EEGAugmenter()
    augmented = aug(dummy)
    print(f"Augmented shape: {augmented.shape}")
    print(f"Max diff from augmentation: {(augmented - dummy).abs().max():.4f}")
