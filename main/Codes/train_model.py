"""
=============================================================
EEG Cognitive Load Classification Pipeline
=============================================================
Models:
  1. Random Forest  — handcrafted features (27 per window)
  2. EEGNet (CNN)   — raw 2-sec EEG windows

Evaluation:
  Leave-One-Subject-Out (LOSO) Cross-Validation

Pipeline:
  Load CSVs → TASK phase → Remove artifacts
  → 2-sec windows (50% overlap)
  → Per-subject Z-score normalization
  → Train/Evaluate with LOSO
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
import time
warnings.filterwarnings('ignore')

from scipy import signal, stats
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR     = os.path.join("..", "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR  = os.path.join("..", "results_model")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLING_RATE = 512
WINDOW_SEC    = 2
OVERLAP       = 0.5
WINDOW_SIZE   = WINDOW_SEC * SAMPLING_RATE  # 1024 samples
STEP_SIZE     = int(WINDOW_SIZE * (1 - OVERLAP))  # 512 samples

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

LABEL_MAP = {
    'design_A_simple':   'Simple',
    'design_B_complex':  'Complex',
    'design_C_moderate': 'Moderate',
}
CLASS_ORDER = ['Simple', 'Moderate', 'Complex']

# Merge repeat sessions into one participant
PARTICIPANT_MERGE = {
    'adnan2':   'adnan',
    'Mushfiq2': 'Mushfiq',
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 5,
    'min_samples_split': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

# EEGNet hyperparameters
EEGNET_EPOCHS  = 50
EEGNET_LR      = 0.001
EEGNET_BATCH   = 32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. DATA LOADING & WINDOWING
# ==========================================
def load_and_window():
    """Load all cleaned CSVs, extract TASK-phase raw EEG,
       segment into 2-sec windows with 50% overlap."""
    files = glob.glob(FILE_PATTERN)
    if not files:
        raise FileNotFoundError(f"No files found matching {FILE_PATTERN}")

    print(f"[DATA] Loading {len(files)} files...")

    all_windows = []  # list of (participant, label, raw_values_array)

    for f in files:
        df = pd.read_csv(f)
        name = os.path.basename(f).replace('UI_Exp_', '').replace('.csv', '').rsplit('_', 1)[0]
        name = PARTICIPANT_MERGE.get(name, name)

        # Filter: raw type, TASK phase, no artifacts
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()

        if task_df.empty:
            print(f"  [SKIP] {name}: no valid TASK samples")
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        # Window per complexity level within this participant
        for complexity in CLASS_ORDER:
            subset = task_df[task_df['complexity'] == complexity]
            values = subset['value'].values.astype(np.float64)

            n_windows = 0
            for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
                window = values[start:start + WINDOW_SIZE]
                if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                    all_windows.append({
                        'participant': name,
                        'label': complexity,
                        'data': window,
                    })
                    n_windows += 1

        p_windows = sum(1 for w in all_windows if w['participant'] == name)
        print(f"  {name}: {p_windows} windows")

    print(f"[DATA] Total windows: {len(all_windows)}")

    # Count per class
    for c in CLASS_ORDER:
        n = sum(1 for w in all_windows if w['label'] == c)
        print(f"  {c}: {n} windows")

    return all_windows


# ==========================================
# 2. FEATURE EXTRACTION (for Random Forest)
# ==========================================
def compute_band_power(values, band, fs=SAMPLING_RATE):
    """Compute absolute band power using Welch's PSD."""
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(512, len(values)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])


def sample_entropy(data, m=2, r_factor=0.2):
    """Compute sample entropy (simplified, fast version)."""
    N = len(data)
    r = r_factor * np.std(data)
    if r == 0:
        return 0.0

    # Subsample for speed if long
    if N > 512:
        data = data[:512]
        N = 512

    def count_matches(template_len):
        count = 0
        templates = np.array([data[i:i + template_len] for i in range(N - template_len)])
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i] - templates[i + 1:]), axis=1)
            count += np.sum(dists < r)
        return count

    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0:
        return 0.0
    return -np.log((A + 1e-10) / (B + 1e-10))


def permutation_entropy(data, order=3, delay=1):
    """Compute permutation entropy."""
    n = len(data)
    permutations_count = {}

    for i in range(n - (order - 1) * delay):
        indices = [i + j * delay for j in range(order)]
        pattern = tuple(np.argsort([data[idx] for idx in indices]))

        if pattern in permutations_count:
            permutations_count[pattern] += 1
        else:
            permutations_count[pattern] = 1

    total = sum(permutations_count.values())
    probs = np.array(list(permutations_count.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-10))


def extract_features(window):
    """Extract 27 handcrafted features from a 2-sec EEG window."""
    features = {}

    # --- Band Powers (5 absolute + 5 relative) ---
    band_powers = {}
    for band_name, (fmin, fmax) in BANDS.items():
        bp = compute_band_power(window, (fmin, fmax))
        band_powers[band_name] = bp
        features[f'bp_{band_name}'] = bp

    total_power = sum(band_powers.values()) + 1e-10
    for band_name, bp in band_powers.items():
        features[f'rp_{band_name}'] = bp / total_power

    # --- Band Ratios (4) ---
    alpha = band_powers['alpha'] + 1e-10
    beta  = band_powers['beta'] + 1e-10
    theta = band_powers['theta'] + 1e-10
    gamma = band_powers['gamma'] + 1e-10

    features['ratio_engagement']   = beta / alpha         # β/α
    features['ratio_cogload']      = theta / beta         # θ/β
    features['ratio_taskload']     = theta / alpha        # θ/α
    features['ratio_relaxation']   = (alpha + theta) / (beta + gamma)  # (α+θ)/(β+γ)

    # --- Statistical Features (6) ---
    features['stat_mean']     = np.mean(window)
    features['stat_std']      = np.std(window)
    features['stat_skewness'] = float(skew(window))
    features['stat_kurtosis'] = float(kurtosis(window))
    features['stat_rms']      = np.sqrt(np.mean(window ** 2))
    features['stat_ptp']      = np.ptp(window)  # peak-to-peak

    # --- Hjorth Parameters (3) ---
    diff1 = np.diff(window)
    diff2 = np.diff(diff1)

    activity   = np.var(window)
    mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

    features['hjorth_activity']   = activity
    features['hjorth_mobility']   = mobility
    features['hjorth_complexity'] = complexity

    # --- Entropy Features (4) ---
    # Shannon entropy (on histogram)
    hist, _ = np.histogram(window, bins=50, density=True)
    hist = hist[hist > 0]
    features['entropy_shannon'] = -np.sum(hist * np.log2(hist + 1e-10))

    # Spectral entropy
    freqs, psd = signal.welch(window, fs=SAMPLING_RATE, nperseg=min(512, len(window)))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    features['entropy_spectral'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Sample entropy
    features['entropy_sample'] = sample_entropy(window)

    # Permutation entropy
    features['entropy_permutation'] = permutation_entropy(window)

    return features


def build_feature_matrix(windows):
    """Extract features from all windows, return X matrix, y labels, subjects."""
    print(f"\n[FEATURES] Extracting 27 features from {len(windows)} windows...")
    t0 = time.time()

    rows = []
    labels = []
    subjects = []

    for i, w in enumerate(windows):
        feats = extract_features(w['data'])
        rows.append(feats)
        labels.append(w['label'])
        subjects.append(w['participant'])

        if (i + 1) % 200 == 0 or i == len(windows) - 1:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(windows)} ({elapsed:.1f}s)")

    feature_names = list(rows[0].keys())
    X = np.array([[r[f] for f in feature_names] for r in rows])
    y = np.array(labels)
    subjects = np.array(subjects)

    print(f"[FEATURES] Done. X shape: {X.shape}, feature names: {len(feature_names)}")
    return X, y, subjects, feature_names


# ==========================================
# 3. PER-SUBJECT NORMALIZATION
# ==========================================
def normalize_per_subject(X, subjects):
    """Z-score normalize features within each participant."""
    X_norm = X.copy()
    for subj in np.unique(subjects):
        mask = subjects == subj
        scaler = StandardScaler()
        X_norm[mask] = scaler.fit_transform(X_norm[mask])
    return X_norm


# ==========================================
# 4. RANDOM FOREST — LOSO
# ==========================================
def train_random_forest_loso(X, y, subjects, feature_names):
    """Train Random Forest with Leave-One-Subject-Out CV."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST — LOSO Cross-Validation")
    print("=" * 60)

    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    unique_subjects = np.unique(subjects)
    fold_results = []
    all_y_true, all_y_pred = [], []
    feature_importances = np.zeros(len(feature_names))

    for fold, test_subj in enumerate(unique_subjects):
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, labels=CLASS_ORDER, average='macro', zero_division=0)

        fold_results.append({
            'subject': test_subj,
            'accuracy': acc,
            'f1_macro': f1,
            'n_test': len(y_test),
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        feature_importances += clf.feature_importances_

        print(f"  Fold {fold+1} (test={test_subj:>10}): "
              f"Acc={acc:.3f}, F1={f1:.3f} (n={len(y_test)})")

    feature_importances /= len(unique_subjects)

    # Aggregate metrics
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc  = np.std([r['accuracy'] for r in fold_results])
    mean_f1  = np.mean([r['f1_macro'] for r in fold_results])
    std_f1   = np.std([r['f1_macro'] for r in fold_results])

    print(f"\n  === RANDOM FOREST AVERAGE ===")
    print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"\n  Classification Report (aggregated):")
    print(classification_report(all_y_true, all_y_pred,
                                labels=CLASS_ORDER, zero_division=0))

    return {
        'model': 'Random Forest',
        'fold_results': fold_results,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'y_true': all_y_true, 'y_pred': all_y_pred,
        'feature_importances': feature_importances,
        'feature_names': feature_names,
    }


# ==========================================
# 5. EEGNet MODEL
# ==========================================
class EEGNet(nn.Module):
    """
    Compact EEGNet for single-channel EEG classification.
    Adapted from Lawhern et al., 2018.
    """
    def __init__(self, n_classes=3, n_samples=1024, F1=4, D=2, F2=8, dropout=0.25):
        super(EEGNet, self).__init__()

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)

        # Block 2: Depthwise convolution (spatial — trivial for single channel)
        self.depthwise = nn.Conv2d(F1, F1 * D, (1, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1 * D)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropout)

        # Block 3: Separable convolution
        self.separable1 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3        = nn.BatchNorm2d(F2)
        self.pool2      = nn.AvgPool2d((1, 8))
        self.drop2      = nn.Dropout(dropout)

        # Classifier — compute flattened size dynamically
        self._flat_size = self._get_flat_size(n_samples)
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_samples):
        x = torch.zeros(1, 1, 1, n_samples)
        x = self._forward_features(x)
        return x.shape[1]

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = nn.ELU()(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable1(x)
        x = self.bn3(x)
        x = nn.ELU()(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.classifier(x)
        return x


def train_eegnet_loso(windows):
    """Train EEGNet with Leave-One-Subject-Out CV."""
    print("\n" + "=" * 60)
    print("EEGNet — LOSO Cross-Validation")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # Prepare data: normalize raw windows per subject
    subjects_list = np.array([w['participant'] for w in windows])
    labels_list   = np.array([w['label'] for w in windows])
    raw_data      = np.array([w['data'] for w in windows])  # (N, 1024)

    # Per-subject Z-score on raw signal
    for subj in np.unique(subjects_list):
        mask = subjects_list == subj
        mu  = raw_data[mask].mean()
        std = raw_data[mask].std() + 1e-10
        raw_data[mask] = (raw_data[mask] - mu) / std

    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    labels_encoded = le.transform(labels_list)

    unique_subjects = np.unique(subjects_list)
    fold_results = []
    all_y_true, all_y_pred = [], []

    for fold, test_subj in enumerate(unique_subjects):
        train_mask = subjects_list != test_subj
        test_mask  = subjects_list == test_subj

        # Shape: (N, 1, 1, 1024) for EEGNet
        X_train = torch.FloatTensor(raw_data[train_mask]).unsqueeze(1).unsqueeze(2)
        X_test  = torch.FloatTensor(raw_data[test_mask]).unsqueeze(1).unsqueeze(2)
        y_train = torch.LongTensor(labels_encoded[train_mask])
        y_test_np = labels_list[test_mask]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader  = DataLoader(train_dataset, batch_size=EEGNET_BATCH,
                                   shuffle=True, drop_last=False)

        # Initialize model
        model = EEGNet(n_classes=len(CLASS_ORDER), n_samples=WINDOW_SIZE).to(DEVICE)
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(_compute_class_weights(y_train.numpy(), len(CLASS_ORDER))).to(DEVICE)
        )
        optimizer = optim.Adam(model.parameters(), lr=EEGNET_LR, weight_decay=1e-4)

        # Train
        model.train()
        for epoch in range(EEGNET_EPOCHS):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test.to(DEVICE)
            outputs = model(X_test_dev)
            _, predicted = torch.max(outputs, 1)
            y_pred = le.inverse_transform(predicted.cpu().numpy())

        acc = accuracy_score(y_test_np, y_pred)
        f1  = f1_score(y_test_np, y_pred, labels=CLASS_ORDER, average='macro', zero_division=0)

        fold_results.append({
            'subject': test_subj,
            'accuracy': acc,
            'f1_macro': f1,
            'n_test': len(y_test_np),
        })
        all_y_true.extend(y_test_np)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1} (test={test_subj:>10}): "
              f"Acc={acc:.3f}, F1={f1:.3f} (n={len(y_test_np)})")

    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc  = np.std([r['accuracy'] for r in fold_results])
    mean_f1  = np.mean([r['f1_macro'] for r in fold_results])
    std_f1   = np.std([r['f1_macro'] for r in fold_results])

    print(f"\n  === EEGNet AVERAGE ===")
    print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"\n  Classification Report (aggregated):")
    print(classification_report(all_y_true, all_y_pred,
                                labels=CLASS_ORDER, zero_division=0))

    return {
        'model': 'EEGNet',
        'fold_results': fold_results,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'y_true': all_y_true, 'y_pred': all_y_pred,
    }


def _compute_class_weights(y, n_classes):
    """Compute balanced class weights."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    weights = len(y) / (n_classes * counts)
    return weights


# ==========================================
# 6. VISUALIZATION
# ==========================================
def plot_confusion_matrix(results, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(results['y_true'], results['y_pred'], labels=CLASS_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[0])
    axes[0].set_title(f'{model_name} — Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized (recall per class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=axes[1])
    axes[1].set_title(f'{model_name} — Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    fname = f'{RESULTS_DIR}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {fname}")


def plot_feature_importance(results, top_n=15):
    """Plot top feature importances from Random Forest."""
    if 'feature_importances' not in results:
        return

    imp = results['feature_importances']
    names = results['feature_names']

    # Sort
    idx = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for i in idx:
        name = names[i]
        if 'ratio' in name:
            colors.append('#e74c3c')  # Red for ratios
        elif 'bp_' in name:
            colors.append('#3498db')  # Blue for band powers
        elif 'rp_' in name:
            colors.append('#2ecc71')  # Green for relative powers
        elif 'stat_' in name:
            colors.append('#f39c12')  # Orange for statistical
        elif 'hjorth' in name:
            colors.append('#9b59b6')  # Purple for Hjorth
        elif 'entropy' in name:
            colors.append('#1abc9c')  # Teal for entropy
        else:
            colors.append('#95a5a6')

    ax.barh(range(len(idx)), imp[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1])
    ax.set_xlabel('Feature Importance (MDI)')
    ax.set_title('Random Forest — Top 15 Feature Importances', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Band Ratios'),
        Patch(facecolor='#3498db', label='Absolute Band Power'),
        Patch(facecolor='#2ecc71', label='Relative Band Power'),
        Patch(facecolor='#f39c12', label='Statistical'),
        Patch(facecolor='#9b59b6', label='Hjorth Parameters'),
        Patch(facecolor='#1abc9c', label='Entropy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/feature_importance_rf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/feature_importance_rf.png")


def plot_model_comparison(rf_results, eegnet_results):
    """Side-by-side model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = [rf_results, eegnet_results]
    model_names = ['Random Forest', 'EEGNet']
    colors = ['#3498db', '#e74c3c']

    # 1. Accuracy comparison
    accs = [r['mean_acc'] for r in models]
    stds = [r['std_acc'] for r in models]
    bars = axes[0].bar(model_names, accs, yerr=stds, color=colors,
                       capsize=8, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for bar, acc, std in zip(bars, accs, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                     f'{acc:.3f}±{std:.3f}', ha='center', fontsize=11, fontweight='bold')

    # 2. F1 comparison
    f1s = [r['mean_f1'] for r in models]
    f1_stds = [r['std_f1'] for r in models]
    bars = axes[1].bar(model_names, f1s, yerr=f1_stds, color=colors,
                       capsize=8, alpha=0.8, edgecolor='black')
    axes[1].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance')
    axes[1].set_ylabel('F1 Score (Macro)')
    axes[1].set_title('Model F1 Comparison', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for bar, f1, std in zip(bars, f1s, f1_stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                     f'{f1:.3f}±{std:.3f}', ha='center', fontsize=11, fontweight='bold')

    # 3. Per-subject accuracy
    subjects_rf = [r['subject'] for r in rf_results['fold_results']]
    acc_rf = [r['accuracy'] for r in rf_results['fold_results']]
    acc_en = [r['accuracy'] for r in eegnet_results['fold_results']]

    x = np.arange(len(subjects_rf))
    width = 0.35
    axes[2].bar(x - width/2, acc_rf, width, label='Random Forest',
                color='#3498db', alpha=0.8, edgecolor='black')
    axes[2].bar(x + width/2, acc_en, width, label='EEGNet',
                color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[2].axhline(y=1/3, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(subjects_rf, rotation=45, ha='right')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Per-Subject Accuracy', fontweight='bold')
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/model_comparison.png")


def save_results_csv(rf_results, eegnet_results):
    """Save all metrics to CSV."""
    rows = []
    for results in [rf_results, eegnet_results]:
        for fold in results['fold_results']:
            rows.append({
                'Model': results['model'],
                'Test_Subject': fold['subject'],
                'Accuracy': fold['accuracy'],
                'F1_Macro': fold['f1_macro'],
                'N_Test_Windows': fold['n_test'],
            })
        rows.append({
            'Model': results['model'],
            'Test_Subject': 'AVERAGE',
            'Accuracy': results['mean_acc'],
            'F1_Macro': results['mean_f1'],
            'N_Test_Windows': '',
        })

    df = pd.DataFrame(rows)
    path = f'{RESULTS_DIR}/model_results.csv'
    df.to_csv(path, index=False)
    print(f"  → Saved: {path}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("EEG COGNITIVE LOAD CLASSIFICATION PIPELINE")
    print("=" * 60)

    # 1. Load and window
    windows = load_and_window()

    # 2. Feature extraction for RF
    X, y, subjects, feature_names = build_feature_matrix(windows)

    # 3. Per-subject normalization
    X_norm = normalize_per_subject(X, subjects)

    # Replace any NaN/Inf from feature extraction
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. Random Forest LOSO
    rf_results = train_random_forest_loso(X_norm, y, subjects, feature_names)

    # 5. EEGNet LOSO
    eegnet_results = train_eegnet_loso(windows)

    # 6. Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_confusion_matrix(rf_results, 'Random Forest')
    plot_confusion_matrix(eegnet_results, 'EEGNet')
    plot_feature_importance(rf_results)
    plot_model_comparison(rf_results, eegnet_results)
    save_results_csv(rf_results, eegnet_results)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Random Forest:  Acc = {rf_results['mean_acc']:.3f} ± {rf_results['std_acc']:.3f}  |  "
          f"F1 = {rf_results['mean_f1']:.3f} ± {rf_results['std_f1']:.3f}")
    print(f"  EEGNet:         Acc = {eegnet_results['mean_acc']:.3f} ± {eegnet_results['std_acc']:.3f}  |  "
          f"F1 = {eegnet_results['mean_f1']:.3f} ± {eegnet_results['std_f1']:.3f}")
    print(f"  Chance level:   33.3%")
    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
