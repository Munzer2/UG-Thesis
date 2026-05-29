"""
=============================================================
EEG Model Comparison with Class Balancing
=============================================================
Compares: Random Forest, SVM, XGBoost
Experiments:
  A) 3-Class balanced (Simple vs Moderate vs Complex)
  B) Binary (Simple vs Complex only)
Both with:
  - Random undersampling for class balance
  - Leave-One-Subject-Out (LOSO) cross-validation
  - Per-subject Z-score normalization
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
import time
import warnings
warnings.filterwarnings('ignore')

from scipy import signal, stats
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR     = os.path.join("..", "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR  = os.path.join("..", "results_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLING_RATE = 512
WINDOW_SEC    = 2
OVERLAP       = 0.5
WINDOW_SIZE   = WINDOW_SEC * SAMPLING_RATE   # 1024
STEP_SIZE     = int(WINDOW_SIZE * (1 - OVERLAP))  # 512

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

# Merge repeat sessions into one participant
PARTICIPANT_MERGE = {
    'adnan2':   'adnan',
    'Mushfiq2': 'Mushfiq',
}

SEED = 42
np.random.seed(SEED)

# ==========================================
# MODEL DEFINITIONS
# ==========================================
MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        min_samples_split=10, class_weight='balanced',
        random_state=SEED, n_jobs=-1
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', random_state=SEED
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=SEED,
        use_label_encoder=False, verbosity=0
    ),
}

# ==========================================
# DATA LOADING & WINDOWING
# ==========================================
def load_and_window():
    """Load CSVs, filter TASK/raw/no-artifact, segment into 2-sec windows."""
    files = glob.glob(FILE_PATTERN)
    if not files:
        raise FileNotFoundError(f"No files found: {FILE_PATTERN}")

    print(f"[DATA] Loading {len(files)} files...")
    all_windows = []

    for f in files:
        df = pd.read_csv(f)
        name = os.path.basename(f).replace('UI_Exp_', '').replace('.csv', '').rsplit('_', 1)[0]
        name = PARTICIPANT_MERGE.get(name, name)

        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()

        if task_df.empty:
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        for complexity in ['Simple', 'Moderate', 'Complex']:
            values = task_df[task_df['complexity'] == complexity]['value'].values.astype(np.float64)
            for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
                window = values[start:start + WINDOW_SIZE]
                if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                    all_windows.append({
                        'participant': name,
                        'label': complexity,
                        'data': window,
                    })

    print(f"[DATA] Total: {len(all_windows)} windows")
    for c in ['Simple', 'Moderate', 'Complex']:
        n = sum(1 for w in all_windows if w['label'] == c)
        print(f"  {c}: {n}")
    return all_windows


# ==========================================
# FEATURE EXTRACTION
# ==========================================
def compute_band_power(values, band, fs=SAMPLING_RATE):
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(512, len(values)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])


def sample_entropy(data, m=2, r_factor=0.2):
    N = min(len(data), 512)
    data = data[:N]
    r = r_factor * np.std(data)
    if r == 0:
        return 0.0

    def count_matches(tl):
        templates = np.array([data[i:i + tl] for i in range(N - tl)])
        count = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i] - templates[i + 1:]), axis=1)
            count += np.sum(dists < r)
        return count

    A, B = count_matches(m + 1), count_matches(m)
    return -np.log((A + 1e-10) / (B + 1e-10)) if B > 0 else 0.0


def permutation_entropy(data, order=3, delay=1):
    n = len(data)
    counts = {}
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort([data[i + j * delay] for j in range(order)]))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-10))


def extract_features(window):
    """27 handcrafted features per window."""
    f = {}

    # Band powers (5 abs + 5 rel)
    bp = {}
    for name, (lo, hi) in BANDS.items():
        bp[name] = compute_band_power(window, (lo, hi))
        f[f'bp_{name}'] = bp[name]

    total = sum(bp.values()) + 1e-10
    for name, val in bp.items():
        f[f'rp_{name}'] = val / total

    # Ratios (4)
    a, b, t, g = bp['alpha']+1e-10, bp['beta']+1e-10, bp['theta']+1e-10, bp['gamma']+1e-10
    f['ratio_engagement'] = b / a
    f['ratio_cogload']    = t / b
    f['ratio_taskload']   = t / a
    f['ratio_relaxation'] = (a + t) / (b + g)

    # Statistical (6)
    f['stat_mean']     = np.mean(window)
    f['stat_std']      = np.std(window)
    f['stat_skewness'] = float(skew(window))
    f['stat_kurtosis'] = float(kurtosis(window))
    f['stat_rms']      = np.sqrt(np.mean(window ** 2))
    f['stat_ptp']      = np.ptp(window)

    # Hjorth (3)
    d1, d2 = np.diff(window), np.diff(np.diff(window))
    act = np.var(window)
    mob = np.sqrt(np.var(d1) / (act + 1e-10))
    f['hjorth_activity']   = act
    f['hjorth_mobility']   = mob
    f['hjorth_complexity'] = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mob + 1e-10)

    # Entropy (4)
    hist, _ = np.histogram(window, bins=50, density=True)
    hist = hist[hist > 0]
    f['entropy_shannon'] = -np.sum(hist * np.log2(hist + 1e-10))

    freqs, psd = signal.welch(window, fs=SAMPLING_RATE, nperseg=min(512, len(window)))
    pn = psd / (np.sum(psd) + 1e-10)
    pn = pn[pn > 0]
    f['entropy_spectral'] = -np.sum(pn * np.log2(pn + 1e-10))
    f['entropy_sample']      = sample_entropy(window)
    f['entropy_permutation'] = permutation_entropy(window)

    return f


def build_feature_matrix(windows):
    """Extract features from all windows."""
    print(f"\n[FEATURES] Extracting from {len(windows)} windows...")
    t0 = time.time()
    rows, labels, subjects = [], [], []

    for i, w in enumerate(windows):
        rows.append(extract_features(w['data']))
        labels.append(w['label'])
        subjects.append(w['participant'])
        if (i + 1) % 200 == 0 or i == len(windows) - 1:
            print(f"  {i+1}/{len(windows)} ({time.time()-t0:.1f}s)")

    feat_names = list(rows[0].keys())
    X = np.array([[r[f] for f in feat_names] for r in rows])
    return X, np.array(labels), np.array(subjects), feat_names


# ==========================================
# PER-SUBJECT NORMALIZATION
# ==========================================
def normalize_per_subject(X, subjects):
    X_norm = X.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        sc = StandardScaler()
        X_norm[mask] = sc.fit_transform(X_norm[mask])
    return X_norm


# ==========================================
# CLASS BALANCING (RANDOM UNDERSAMPLING)
# ==========================================
def undersample(X, y, subjects, seed=SEED):
    """Undersample to match the smallest class count per participant."""
    rng = np.random.RandomState(seed)
    indices = []

    for subj in np.unique(subjects):
        subj_mask = subjects == subj
        subj_idx = np.where(subj_mask)[0]
        subj_labels = y[subj_idx]

        classes, counts = np.unique(subj_labels, return_counts=True)
        min_count = counts.min()

        for cls in classes:
            cls_idx = subj_idx[subj_labels == cls]
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            indices.extend(chosen)

    indices = np.array(sorted(indices))
    return X[indices], y[indices], subjects[indices]


# ==========================================
# LOSO CROSS-VALIDATION
# ==========================================
def run_loso(model_name, model_template, X, y, subjects, class_order):
    """Run LOSO cross-validation for a single model."""
    unique_subjects = np.unique(subjects)
    fold_results = []
    all_y_true, all_y_pred = [], []

    for fold, test_subj in enumerate(unique_subjects):
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Skip if test subject has no samples for some class
        if len(np.unique(y_test)) < 2:
            continue

        # Clone model (fresh instance)
        from sklearn.base import clone
        clf = clone(model_template)

        # XGBoost needs numeric labels
        if model_name == 'XGBoost':
            le_local = LabelEncoder()
            le_local.fit(class_order)
            y_train_enc = le_local.transform(y_train)
            clf.fit(X_train, y_train_enc)
            y_pred_enc = clf.predict(X_test)
            y_pred = le_local.inverse_transform(y_pred_enc.astype(int))
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, labels=class_order, average='macro', zero_division=0)

        fold_results.append({
            'subject': test_subj, 'accuracy': acc,
            'f1_macro': f1, 'n_test': len(y_test),
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc  = np.std([r['accuracy'] for r in fold_results])
    mean_f1  = np.mean([r['f1_macro'] for r in fold_results])
    std_f1   = np.std([r['f1_macro'] for r in fold_results])

    return {
        'model': model_name,
        'fold_results': fold_results,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'y_true': all_y_true, 'y_pred': all_y_pred,
    }


# ==========================================
# RUN ONE EXPERIMENT (3-class or binary)
# ==========================================
def run_experiment(windows, experiment_name, class_order):
    """Run all models on given windows for specified classes."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Classes: {class_order}")
    print(f"{'='*60}")

    # Filter windows to requested classes
    filtered = [w for w in windows if w['label'] in class_order]
    print(f"Windows after filtering: {len(filtered)}")

    # Extract features
    X, y, subjects, feat_names = build_feature_matrix(filtered)

    # Per-subject normalization
    X_norm = normalize_per_subject(X, subjects)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # Report class distribution BEFORE balancing
    print(f"\n  Before balancing:")
    for c in class_order:
        print(f"    {c}: {np.sum(y == c)} windows")

    # Undersample
    X_bal, y_bal, subj_bal = undersample(X_norm, y, subjects)

    print(f"\n  After balancing:")
    for c in class_order:
        print(f"    {c}: {np.sum(y_bal == c)} windows")
    print(f"  Total: {len(y_bal)} windows")

    # Run each model
    all_results = {}
    for model_name, model_template in MODELS.items():
        print(f"\n--- {model_name} ---")
        results = run_loso(model_name, model_template, X_bal, y_bal, subj_bal, class_order)

        print(f"  Accuracy: {results['mean_acc']:.3f} ± {results['std_acc']:.3f}")
        print(f"  F1 Macro: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}")

        # Per-fold detail
        for fold in results['fold_results']:
            print(f"    {fold['subject']:>10}: Acc={fold['accuracy']:.3f}, "
                  f"F1={fold['f1_macro']:.3f} (n={fold['n_test']})")

        print(f"\n  Classification Report:")
        print(classification_report(results['y_true'], results['y_pred'],
                                    labels=class_order, zero_division=0))

        all_results[model_name] = results

    return all_results, feat_names, X_bal, y_bal, subj_bal


# ==========================================
# VISUALIZATION
# ==========================================
def plot_comparison_bars(results_3class, results_binary):
    """Model comparison: accuracy & F1 for both experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, (results, exp_name) in enumerate([
        (results_3class, '3-Class (Balanced)'),
        (results_binary, 'Binary (Simple vs Complex)')
    ]):
        model_names = list(results.keys())
        accs = [results[m]['mean_acc'] for m in model_names]
        acc_stds = [results[m]['std_acc'] for m in model_names]
        f1s = [results[m]['mean_f1'] for m in model_names]
        f1_stds = [results[m]['std_f1'] for m in model_names]

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        chance = 1.0 / len(next(iter(results.values()))['fold_results'][0:1] or [{}])

        # Accuracy
        ax = axes[0][col]
        chance_level = 1/3 if '3-Class' in exp_name else 0.5
        bars = ax.bar(model_names, accs, yerr=acc_stds, color=colors[:len(model_names)],
                      capsize=8, alpha=0.85, edgecolor='black')
        ax.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.6,
                   label=f'Chance ({chance_level:.1%})')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{exp_name} — Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        for bar, acc, std in zip(bars, accs, acc_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                    f'{acc:.1%}', ha='center', fontsize=11, fontweight='bold')

        # F1
        ax = axes[1][col]
        bars = ax.bar(model_names, f1s, yerr=f1_stds, color=colors[:len(model_names)],
                      capsize=8, alpha=0.85, edgecolor='black')
        ax.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.6,
                   label=f'Chance')
        ax.set_ylabel('F1 Score (Macro)')
        ax.set_title(f'{exp_name} — F1 Macro', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        for bar, f1, std in zip(bars, f1s, f1_stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                    f'{f1:.1%}', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → Saved: {RESULTS_DIR}/model_comparison.png")


def plot_per_subject(results_3class, results_binary):
    """Per-subject accuracy across all models for both experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, (results, exp_name) in zip(axes, [
        (results_3class, '3-Class (Balanced)'),
        (results_binary, 'Binary (Simple vs Complex)')
    ]):
        model_names = list(results.keys())
        subjects = [r['subject'] for r in results[model_names[0]]['fold_results']]
        x = np.arange(len(subjects))
        width = 0.25

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        for i, model_name in enumerate(model_names):
            accs = [r['accuracy'] for r in results[model_name]['fold_results']]
            offset = (i - 1) * width
            ax.bar(x + offset, accs, width, label=model_name,
                   color=colors[i], alpha=0.85, edgecolor='black')

        chance = 1/3 if '3-Class' in exp_name else 0.5
        ax.axhline(y=chance, color='gray', linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{exp_name} — Per-Subject Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/per_subject_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/per_subject_accuracy.png")


def plot_confusion_matrices(results_3class, results_binary):
    """Confusion matrices for all models in both experiments."""
    all_experiments = [
        (results_3class, '3-Class', ['Simple', 'Moderate', 'Complex']),
        (results_binary, 'Binary', ['Simple', 'Complex']),
    ]

    for results, exp_tag, class_order in all_experiments:
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, res) in zip(axes, results.items()):
            cm = confusion_matrix(res['y_true'], res['y_pred'], labels=class_order)
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_order, yticklabels=class_order, ax=ax,
                        vmin=0, vmax=1)
            ax.set_title(f'{model_name}\nAcc={res["mean_acc"]:.1%}',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')

        plt.suptitle(f'{exp_tag} — Normalized Confusion Matrices',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = f'{RESULTS_DIR}/confusion_{exp_tag.lower().replace("-","")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {fname}")


def save_summary_csv(results_3class, results_binary):
    """Save all metrics to a single CSV."""
    rows = []
    for exp_name, results in [('3-Class Balanced', results_3class),
                               ('Binary (S vs C)', results_binary)]:
        for model_name, res in results.items():
            for fold in res['fold_results']:
                rows.append({
                    'Experiment': exp_name,
                    'Model': model_name,
                    'Test_Subject': fold['subject'],
                    'Accuracy': fold['accuracy'],
                    'F1_Macro': fold['f1_macro'],
                    'N_Windows': fold['n_test'],
                })
            rows.append({
                'Experiment': exp_name,
                'Model': model_name,
                'Test_Subject': 'AVERAGE',
                'Accuracy': res['mean_acc'],
                'F1_Macro': res['mean_f1'],
                'N_Windows': '',
            })

    df = pd.DataFrame(rows)
    path = f'{RESULTS_DIR}/comparison_results.csv'
    df.to_csv(path, index=False)
    print(f"  → Saved: {path}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("EEG MODEL COMPARISON — Class Balanced")
    print("=" * 60)

    windows = load_and_window()

    # Experiment A: 3-class balanced
    results_3class, feat_names_3, *_ = run_experiment(
        windows, "3-Class Balanced", ['Simple', 'Moderate', 'Complex']
    )

    # Experiment B: Binary (Simple vs Complex)
    results_binary, feat_names_b, *_ = run_experiment(
        windows, "Binary: Simple vs Complex", ['Simple', 'Complex']
    )

    # Visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    plot_comparison_bars(results_3class, results_binary)
    plot_per_subject(results_3class, results_binary)
    plot_confusion_matrices(results_3class, results_binary)
    save_summary_csv(results_3class, results_binary)

    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Model':<18} {'3-Class Acc':>12} {'3-Class F1':>12} "
          f"{'Binary Acc':>12} {'Binary F1':>12}")
    print(f"  {'-'*70}")
    for model_name in MODELS:
        r3 = results_3class[model_name]
        rb = results_binary[model_name]
        print(f"  {model_name:<18} "
              f"{r3['mean_acc']:.1%} ± {r3['std_acc']:.1%}  "
              f"{r3['mean_f1']:.1%} ± {r3['std_f1']:.1%}  "
              f"{rb['mean_acc']:.1%} ± {rb['std_acc']:.1%}  "
              f"{rb['mean_f1']:.1%} ± {rb['std_f1']:.1%}")

    print(f"\n  Chance: 3-class = 33.3%  |  Binary = 50.0%")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
