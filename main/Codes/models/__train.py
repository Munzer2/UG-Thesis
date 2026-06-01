

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score)

# Import our custom model
# from MSTCNN_A import MSTCNN_A, EEGAugmenter

from MBCN_BiLSTM import MBCN_BiLSTM
from EEG_TCN import EEG_TCN
from MSTCNN_A import EEGAugmenter

# ==========================================
# CONFIGURATION
# ==========================================
# Paths (relative to Codes/models/)
DATA_DIR     = os.path.join("..", "..", "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR  = os.path.join("..", "..", "results_models")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Signal parameters
SAMPLING_RATE = 512
WINDOW_SEC    = 2
OVERLAP       = 0.5
WINDOW_SIZE   = WINDOW_SEC * SAMPLING_RATE   # 1024 samples
STEP_SIZE     = int(WINDOW_SIZE * (1 - OVERLAP))  # 512 samples

# Label mapping
LABEL_MAP = {
    'design_A_simple':   'Simple',
    'design_B_complex':  'Complex',
    'design_C_moderate': 'Moderate',
}

# Merge repeat sessions into one participant
PARTICIPANT_MERGE = {
    'adnan2':   'adnan',
    'Mushfiq2': 'Mushfiq',
}

# Training hyperparameters
SEED         = 42
EPOCHS       = 50        # Increased from 30 — more time to converge with cosine annealing
BATCH_SIZE   = 16        # Small batch for small dataset
LEARNING_RATE = 0.0005   # Lowered from 0.001 — gentler updates for small data
WEIGHT_DECAY  = 1e-4     # Reduced from 1e-3 — avoid over-regularization
LABEL_SMOOTH  = 0.1      # Label smoothing factor
DROPOUT       = 0.4      # Reduced from 0.5 — prevent underfitting

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ==========================================
# 1. DATA LOADING & WINDOWING (TRIAL-AWARE)
# ==========================================
def load_and_window():
    """
    Load all cleaned CSVs, filter TASK-phase raw EEG,
    and segment into 2-sec windows with 50% overlap.
    
    IMPORTANT: Windows are created WITHIN each individual trial
    (identified by the 'image' column). This prevents cross-trial
    contamination where a window could span two different task
    segments with a break in between.
    
    Returns:
        List of dicts: [{participant, label, data (1024,)}, ...]
    """
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found: {FILE_PATTERN}")

    print(f"[DATA] Loading {len(files)} files...")
    all_windows = []

    for f in files:
        df = pd.read_csv(f)
        name = os.path.basename(f).replace('UI_Exp_', '').replace('.csv', '').rsplit('_', 1)[0]
        name = PARTICIPANT_MERGE.get(name, name)
        # base_subject groups repeat sessions (adnan2 -> adnan) for LOSO
        base_subject = name

        # Filter: raw EEG, TASK phase, no artifacts
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()

        if task_df.empty:
            print(f"  [SKIP] {name}: no valid TASK samples")
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        # Window within each INDIVIDUAL TRIAL (label + image combination)
        # This prevents windows from spanning across different task trials
        for complexity in ['Simple', 'Moderate', 'Complex']:
            comp_df = task_df[task_df['complexity'] == complexity]
            if comp_df.empty:
                continue

            # Group by image (each image = one trial for this complexity)
            for image_name, trial_group in comp_df.groupby('image'):
                values = trial_group['value'].values.astype(np.float64)

                # Only window within this single trial's data
                for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
                    window = values[start:start + WINDOW_SIZE]
                    if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                        all_windows.append({
                            'participant': name,
                            'base_subject': base_subject,
                            'label': complexity,
                            'data': window,
                        })

        p_total = sum(1 for w in all_windows if w['participant'] == name)
        print(f"  {name}: {p_total} windows")

    print(f"\n[DATA] Total: {len(all_windows)} windows")
    for c in ['Simple', 'Moderate', 'Complex']:
        n = sum(1 for w in all_windows if w['label'] == c)
        print(f"  {c}: {n}")
    return all_windows


# ==========================================
# 2. PER-SUBJECT NORMALIZATION
# ==========================================
def normalize_raw_per_subject(raw_data, subjects):
    """
    Z-score normalize raw EEG windows per subject.
    
    Each participant's brain signals have different baseline
    amplitudes. This normalizes so all subjects are on the
    same scale, making the model focus on relative patterns
    rather than absolute amplitudes.
    """
    raw_norm = raw_data.copy()
    for subj in np.unique(subjects):
        mask = subjects == subj
        mu  = raw_norm[mask].mean()
        std = raw_norm[mask].std() + 1e-10
        raw_norm[mask] = (raw_norm[mask] - mu) / std
    return raw_norm


# ==========================================
# 3. COMPUTE CLASS WEIGHTS (replaces undersampling)
# ==========================================
# DESIGN DECISION: Removed per-subject undersampling.
# With only 11 subjects, discarding data to balance classes
# wastes precious training samples. Instead, we use
# class-weighted loss which gives minority classes higher
# gradient weight without throwing away any data.
# ==========================================
def compute_class_weights(labels, n_classes):
    """
    Compute balanced class weights for the loss function.
    Classes with fewer samples get higher weight.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    weights = len(labels) / (n_classes * counts)
    return weights


# ==========================================
# 4. MIXUP-AWARE LOSS COMPUTATION
# ==========================================
def mixup_criterion(criterion, outputs, mixed_y):
    """
    Compute loss for Mixup-augmented batches.
    
    When Mixup is applied, the target is a weighted combination
    of two labels. The loss is computed as:
        loss = lam * L(output, y_a) + (1-lam) * L(output, y_b)
    
    When Mixup was NOT applied (mixed_y is None), falls back
    to standard loss computation.
    """
    if mixed_y is None:
        # No Mixup was applied — shouldn't reach here, but safe fallback
        return criterion(outputs, outputs)  # placeholder, won't be called
    
    y_a, y_b, lam = mixed_y
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


# ==========================================
# 5. SINGLE FOLD TRAINING (FIXED EPOCHS + VALIDATION)
# ==========================================
def train_one_fold(model, train_loader, val_loader, test_data, test_labels,
                   criterion, optimizer, scheduler, augmenter,
                   n_epochs, device):
    """
    Train the model for one LOSO fold using fixed epochs.
    
    NO EARLY STOPPING — trains for exactly n_epochs.
    The test subject is NEVER seen during training;
    it is only evaluated once after all epochs complete.
    This guarantees zero data leakage.
    
    A within-fold validation set (one held-out training subject)
    is monitored to track convergence — for diagnostics only,
    it does NOT influence training decisions.
    
    Args:
        train_loader: DataLoader for training subjects
        val_loader:   DataLoader for within-fold validation (1 subject)
        test_data:    LOSO test subject (untouched until final eval)
        test_labels:  Labels for test subject
        n_epochs:     Fixed number of training epochs
        ...
    
    Returns:
        test_preds: Predictions on test subject after training
        history:    Dict of loss/accuracy curves (train + val)
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(n_epochs):
        # --- TRAINING ---
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Apply augmentation with Mixup (training only!)
            if augmenter is not None:
                batch_X, mixed_y = augmenter(batch_X, batch_y)
            else:
                mixed_y = None

            optimizer.zero_grad()
            outputs = model(batch_X)

            # Compute loss (Mixup-aware: weighted combination of two targets)
            if mixed_y is not None:
                loss = mixup_criterion(criterion, outputs, mixed_y)
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_correct += (predicted == batch_y).sum().item()
            epoch_total += batch_y.size(0)

        # Step the LR scheduler
        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- VALIDATION (diagnostic only, does NOT affect training) ---
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vout = model(vx)
                    vloss = criterion(vout, vy)
                    val_loss_sum += vloss.item() * vx.size(0)
                    _, vpred = torch.max(vout, 1)
                    val_correct += (vpred == vy).sum().item()
                    val_total += vy.size(0)
            history['val_loss'].append(val_loss_sum / val_total)
            history['val_acc'].append(val_correct / val_total)

    # --- FINAL TEST EVALUATION (after all epochs) ---
    # Test subject was NEVER seen during training
    X_test = torch.FloatTensor(test_data).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        _, test_pred = torch.max(test_out, 1)
        test_preds = test_pred.cpu().numpy()

    return test_preds, history


# ==========================================
# 6. LOSO CROSS-VALIDATION (LEAKAGE-FREE + VALIDATION)
# ==========================================
def run_loso_cv(raw_data, labels, subjects, base_subjects, class_order, experiment_name):
    """
    Run Group-LOSO cross-validation.
    
    LEAKAGE-FREE DESIGN (fixed epochs + within-fold validation):
      For each fold:
        1. Leave one BASE subject out as TEST (completely untouched)
           (Both 'Mushfiq' and 'Mushfiq2' go together)
        2. From remaining BASE subjects, hold out 1 as VALIDATION
           (for convergence monitoring only — no model selection)
        3. Train on the rest for fixed EPOCHS
        4. Evaluate final model on TEST subject
    
    No early stopping = no data leakage.
    The test subject is NEVER seen during training.
    Validation is purely diagnostic (shows if model is overfitting).
    
    Returns:
        results: Dict with fold results, predictions, metrics
        all_histories: Training histories for plotting
    """
    print(f"\n{'='*60}")
    print(f"Model Training — {experiment_name}")
    print(f"Classes: {class_order}")
    print(f"Device: {DEVICE}")
    print(f"Training: {EPOCHS} fixed epochs (no early stopping, no leakage)")
    print(f"LR: {LEARNING_RATE}, WD: {WEIGHT_DECAY}, Dropout: {DROPOUT}")
    print(f"{'='*60}")

    n_classes = len(class_order)
    le = LabelEncoder()
    le.fit(class_order)

    unique_base_subjects = np.unique(base_subjects)
    fold_results = []
    all_y_true, all_y_pred = [], []
    all_histories = []

    for fold, test_subj in enumerate(unique_base_subjects):
        print(f"\n  Fold {fold+1}/{len(unique_base_subjects)} — test: {test_subj}")

        # Split: test base_subject vs. everyone else
        test_mask  = base_subjects == test_subj
        train_mask = base_subjects != test_subj

        X_train_all = raw_data[train_mask]
        y_train_all_str = labels[train_mask]
        subj_train_all = base_subjects[train_mask]  # Important: use base_subjects for validation split too
        X_test_raw  = raw_data[test_mask]
        y_test_str  = labels[test_mask]

        # Skip if test subject lacks all classes
        if len(np.unique(y_test_str)) < 2:
            print(f"    [SKIP] Only one class present")
            continue

        # --- Within-fold validation split ---
        # Hold out 1 BASE training subject as validation (for monitoring only)
        train_base_subjects = np.unique(subj_train_all)
        # Pick the validation base_subject deterministically (rotate based on fold)
        val_subj = train_base_subjects[fold % len(train_base_subjects)]

        val_mask_inner = subj_train_all == val_subj
        train_mask_inner = subj_train_all != val_subj

        X_train_raw = X_train_all[train_mask_inner]
        y_train_str = y_train_all_str[train_mask_inner]
        X_val_raw = X_train_all[val_mask_inner]
        y_val_str = y_train_all_str[val_mask_inner]

        # Encode labels
        y_train = le.transform(y_train_str)
        y_test  = le.transform(y_test_str)
        y_val   = le.transform(y_val_str)

        # Compute class weights from training data
        class_weights = compute_class_weights(y_train, n_classes)

        print(f"    train: {len(y_train)}, val: {len(y_val)} (subj={val_subj}), test: {len(y_test)}")

        # Prepare PyTorch data
        X_train_t = torch.FloatTensor(X_train_raw).unsqueeze(1)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val_raw).unsqueeze(1)
        y_val_t = torch.LongTensor(y_val)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader  = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, drop_last=False
        )
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader  = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, drop_last=False
        )

        # Initialize fresh model for each fold
        # model = MSTCNN_A(
        #     n_classes=n_classes, n_samples=WINDOW_SIZE, dropout=DROPOUT
        # ).to(DEVICE)

        # model = MBCN_BiLSTM(n_classes = n_classes, n_samples = WINDOW_SIZE, dropout = DROPOUT).to(DEVICE)
        # model = CNN_Transformer(n_classes = n_classes, n_samples = WINDOW_SIZE, dropout = DROPOUT).to(DEVICE)
        model = EEG_TCN(n_classes = n_classes, n_samples = WINDOW_SIZE).to(DEVICE)

        # Loss with label smoothing + class weights
        class_weights_t = torch.FloatTensor(class_weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTH)

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Cosine annealing: gradually reduces LR over EPOCHS
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        # Data augmentation with Mixup (training only)
        # augmenter = EEGAugmenter(
        #     noise_std=0.05,
        #     scale_range=(0.8, 1.2),
        #     shift_max=50,
        #     drop_prob=0.1,
        #     mixup_alpha=0.2
        # )
        augmenter = None

        # Train this fold (fixed epochs, test subject untouched)
        test_preds, history = train_one_fold(
            model, train_loader, val_loader,
            X_test_raw, y_test,
            criterion, optimizer, scheduler, augmenter,
            EPOCHS, DEVICE
        )

        # Convert predictions back to strings
        y_pred_str = le.inverse_transform(test_preds)
        y_test_actual = y_test_str

        acc = accuracy_score(y_test_actual, y_pred_str)
        f1  = f1_score(y_test_actual, y_pred_str, labels=class_order,
                       average='macro', zero_division=0)

        # Report final val accuracy for diagnostics
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0

        fold_results.append({
            'subject': test_subj,
            'accuracy': acc,
            'f1_macro': f1,
            'n_test': len(y_test_actual),
            'epochs_trained': EPOCHS,
            'final_val_acc': final_val_acc,
        })
        all_y_true.extend(y_test_actual)
        all_y_pred.extend(y_pred_str)
        all_histories.append(history)

        print(f"    Acc={acc:.3f}, F1={f1:.3f}, ValAcc={final_val_acc:.3f} "
              f"(n={len(y_test_actual)}, epochs={len(history['train_loss'])})")

    # Aggregate
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc  = np.std([r['accuracy'] for r in fold_results])
    mean_f1  = np.mean([r['f1_macro'] for r in fold_results])
    std_f1   = np.std([r['f1_macro'] for r in fold_results])

    print(f"\n  === AVERAGE METRICS ===")
    print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                labels=class_order, zero_division=0))

    results = {
        'model': 'Model',
        'experiment': experiment_name,
        'fold_results': fold_results,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'y_true': all_y_true, 'y_pred': all_y_pred,
    }
    return results, all_histories


# ==========================================
# 7. VISUALIZATION
# ==========================================
def plot_training_curves(histories, experiment_name):
    """Plot training + validation loss and accuracy curves for all folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, hist in enumerate(histories):
        epochs = range(1, len(hist['train_loss']) + 1)
        # Training curves
        axes[0][0].plot(epochs, hist['train_loss'], alpha=0.5, label=f'Fold {i+1}')
        axes[0][1].plot(epochs, hist['train_acc'], alpha=0.5, label=f'Fold {i+1}')
        # Validation curves
        if hist['val_loss']:
            axes[1][0].plot(epochs, hist['val_loss'], alpha=0.5, label=f'Fold {i+1}')
            axes[1][1].plot(epochs, hist['val_acc'], alpha=0.5, label=f'Fold {i+1}')

    for ax, ylabel, title in [
        (axes[0][0], 'Loss', f'{experiment_name} — Training Loss'),
        (axes[0][1], 'Accuracy', f'{experiment_name} — Training Accuracy'),
        (axes[1][0], 'Loss', f'{experiment_name} — Validation Loss'),
        (axes[1][1], 'Accuracy', f'{experiment_name} — Validation Accuracy'),
    ]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tag = experiment_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    fname = f'{RESULTS_DIR}/training_curves_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {fname}")


def plot_confusion_matrix(results, class_order, experiment_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(results['y_true'], results['y_pred'], labels=class_order)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_order, yticklabels=class_order, ax=axes[0])
    axes[0].set_title(f'{experiment_name}\nConfusion Matrix (Counts)',
                      fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples',
                xticklabels=class_order, yticklabels=class_order, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title(f'{experiment_name}\nConfusion Matrix (Normalized)',
                      fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    tag = experiment_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    fname = f'{RESULTS_DIR}/confusion_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {fname}")


def plot_per_subject_accuracy(results_binary):
    """Per-subject accuracy for experiment."""
    fig, ax = plt.subplots(figsize=(10, 6))

    results = results_binary
    exp_name = 'Binary (S vs C)'
    chance = 0.5
    subjects = [r['subject'] for r in results['fold_results']]
    accs = [r['accuracy'] for r in results['fold_results']]

    colors = ['#9b59b6' if a > chance else '#e74c3c' for a in accs]
    bars = ax.bar(subjects, accs, color=colors, alpha=0.85, edgecolor='black')
    ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.6,
                label=f'Chance ({chance:.1%})')
    ax.axhline(y=results['mean_acc'], color='#9b59b6', linestyle='-',
                alpha=0.4, label=f'Mean ({results["mean_acc"]:.1%})')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.0%}', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy')
    ax.set_title(f'{exp_name} Per-Subject', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_xticklabels(subjects, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/per_subject_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/per_subject_accuracy.png")


def plot_comparison_with_baselines(results_dict):
    """Compare accuracy across all experiments run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    exp_names = list(results_dict.keys())
    accs = [results_dict[e]['mean_acc'] for e in exp_names]
    f1s  = [results_dict[e]['mean_f1'] for e in exp_names]

    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c'][:len(exp_names)]

    # Accuracy
    bars = axes[0].bar(exp_names, accs, color=colors, alpha=0.85, edgecolor='black')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (50%)')
    axes[0].axhline(y=1/3, color='lightgray', linestyle=':', alpha=0.6, label='Chance 3-class (33%)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('EEG-TCN — Experiment Comparison\nAccuracy',
                      fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{acc:.1%}', ha='center', fontsize=10, fontweight='bold')

    # F1
    bars = axes[1].bar(exp_names, f1s, color=colors, alpha=0.85, edgecolor='black')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Chance')
    axes[1].set_ylabel('F1 Score (Macro)')
    axes[1].set_title('EEG-TCN — Experiment Comparison\nF1 Macro',
                      fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for bar, f1 in zip(bars, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{f1:.1%}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_experiments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/comparison_experiments.png")


def save_results_csv(all_results):
    """Save all metrics to CSV."""
    rows = []
    for exp_name, results in all_results.items():
        for fold in results['fold_results']:
            rows.append({
                'Experiment': exp_name,
                'Model': 'EEG-TCN',
                'Test_Subject': fold['subject'],
                'Accuracy': fold['accuracy'],
                'F1_Macro': fold['f1_macro'],
                'N_Windows': fold['n_test'],
                'Epochs': fold['epochs_trained'],
            })
        rows.append({
            'Experiment': exp_name,
            'Model': 'EEG-TCN',
            'Test_Subject': 'AVERAGE',
            'Accuracy': results['mean_acc'],
            'F1_Macro': results['mean_f1'],
            'N_Windows': '',
            'Epochs': '',
        })

    df = pd.DataFrame(rows)
    path = f'{RESULTS_DIR}/results.csv'
    df.to_csv(path, index=False)
    print(f"  → Saved: {path}")


# ==========================================
# 8. MAIN
# ==========================================
def main():
    print("=" * 60)
    print("EEG Cognitive Load Classification — EEG-TCN")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS} (fixed, no early stopping)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Label smoothing: {LABEL_SMOOTH}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Windowing: trial-aware (no cross-trial contamination)")
    print(f"  Class balance: class-weighted loss (no undersampling)")

    # ---- Load data (trial-aware windowing) ----
    windows = load_and_window()

    # ---- Prepare raw data arrays ----
    all_data     = np.array([w['data'] for w in windows])
    all_labels   = np.array([w['label'] for w in windows])
    all_subjects = np.array([w['participant'] for w in windows])
    all_base_subjects = np.array([w['base_subject'] for w in windows])

    # ---- Per-subject normalization ----
    all_data_norm = normalize_raw_per_subject(all_data, all_subjects)

    all_results = {}
    all_histories = {}

    '''
    # ==================================
    # EXPERIMENT 1: 3-Class (Simple vs Moderate vs Complex)
    # ==================================
    class_order_3 = ['Simple', 'Moderate', 'Complex']
    mask_3 = np.isin(all_labels, class_order_3)
    data_3 = all_data_norm[mask_3]
    labels_3 = all_labels[mask_3]
    subj_3 = all_subjects[mask_3]
    base_subj_3 = all_base_subjects[mask_3]

    print(f"\n[DATA] 3-Class distribution:")
    for c in class_order_3:
        print(f"  {c}: {np.sum(labels_3 == c)}")

    results_3class, hist_3 = run_loso_cv(
        data_3, labels_3, subj_3, base_subj_3, class_order_3, "3-Class"
    )
    all_results['3-Class'] = results_3class
    all_histories['3-Class'] = hist_3
    '''

    # ==================================
    # EXPERIMENT 2: Binary (Simple vs Complex)
    # ==================================
    class_order_2 = ['Simple', 'Complex']
    mask_2 = np.isin(all_labels, class_order_2)
    data_2 = all_data_norm[mask_2]
    labels_2 = all_labels[mask_2]
    subj_2 = all_subjects[mask_2]
    base_subj_2 = all_base_subjects[mask_2]

    print(f"\n[DATA] Binary distribution:")
    for c in class_order_2:
        print(f"  {c}: {np.sum(labels_2 == c)}")

    results_binary, hist_2 = run_loso_cv(
        data_2, labels_2, subj_2, base_subj_2, class_order_2, "Binary Simple vs Complex"
    )
    all_results['Binary (S vs C)'] = results_binary
    all_histories['Binary (S vs C)'] = hist_2

    # ==================================
    # VISUALIZATIONS
    # ==================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # plot_training_curves(hist_3, '3-Class')
    plot_training_curves(hist_2, 'Binary Simple vs Complex')
    # plot_confusion_matrix(results_3class, class_order_3, '3-Class (S vs M vs C)')
    plot_confusion_matrix(results_binary, class_order_2, 'Binary (Simple vs Complex)')
    plot_per_subject_accuracy(results_binary)
    plot_comparison_with_baselines(all_results)
    save_results_csv(all_results)

    # ==================================
    # FINAL SUMMARY
    # ==================================
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    # print(f"\n  3-Class (Simple vs Moderate vs Complex):")
    # print(f"    Accuracy: {results_3class['mean_acc']:.1%} ± {results_3class['std_acc']:.1%}")
    # print(f"    F1 Macro: {results_3class['mean_f1']:.1%} ± {results_3class['std_f1']:.1%}")
    print(f"\n  Binary (Simple vs Complex):")
    print(f"    Accuracy: {results_binary['mean_acc']:.1%} ± {results_binary['std_acc']:.1%}")
    print(f"    F1 Macro: {results_binary['mean_f1']:.1%} ± {results_binary['std_f1']:.1%}")
    print(f"\n  Chance: 3-class = 33.3%  |  Binary = 50.0%")
    print(f"  Regularization: dropout={DROPOUT}, WD={WEIGHT_DECAY}, LS={LABEL_SMOOTH}")
    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
