"""
=============================================================
Random Forest Binary Classifier — Simple vs Complex UI
=============================================================
Uses features proven significant by our statistical analysis:
  - Beta power (p=0.005)  → active engagement marker
  - Gamma power (p=0.042) → visual feature binding
  - Engagement Index β/α  → peaks at Moderate, drops at overload
  - Cognitive Load θ/β    → frontal executive control
  - Attention (alpha suppression) → strongest finding (p<0.001)

Pipeline:
  1. Trial-aware windowing (no cross-trial contamination)
  2. Extract 27 handcrafted features per window
  3. Per-subject Z-score normalization
  4. Per-subject random undersampling (balance classes)
  5. LOSO cross-validation with Random Forest
  6. Feature importance analysis → validates analysis findings

Results saved to: results_rf_binary/
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

from scipy import signal
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR     = os.path.join("..", "..", "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR  = os.path.join("..", "..", "results_rf_binary")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLING_RATE = 512
WINDOW_SEC    = 2
OVERLAP       = 0.5
WINDOW_SIZE   = WINDOW_SEC * SAMPLING_RATE   # 1024 samples
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

# Merge repeat sessions into one participant
PARTICIPANT_MERGE = {
    'adnan2':   'adnan',
    'Mushfiq2': 'Mushfiq',
}

# Only include subjects with at least this many windows PER CLASS
MIN_WINDOWS_PER_CLASS = 5

SEED = 42
np.random.seed(SEED)

# Random Forest — tuned for small EEG datasets
RF_PARAMS = {
    'n_estimators': 500,          # More trees = more stable
    'max_depth': 12,              # Slightly deeper for richer features
    'min_samples_leaf': 4,
    'min_samples_split': 8,
    'class_weight': 'balanced',   # Additional class weighting on top of undersampling
    'random_state': SEED,
    'n_jobs': -1,
}


# ==========================================
# 1. DATA LOADING — TRIAL-AWARE WINDOWING
# ==========================================
def load_and_window():
    """
    Load cleaned CSVs, filter TASK-phase raw EEG, and segment
    into 2-sec windows with 50% overlap.

    TRIAL-AWARE: Windows are created WITHIN each individual trial
    (identified by 'image' column). Prevents cross-trial contamination.

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

        # Filter: raw EEG, TASK phase, no artifacts
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()

        if task_df.empty:
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        # BINARY: Only keep Simple and Complex
        for complexity in ['Simple', 'Complex']:
            comp_df = task_df[task_df['complexity'] == complexity]
            if comp_df.empty:
                continue

            # Window within each individual trial (by image)
            if 'image' in comp_df.columns:
                groups = comp_df.groupby('image')
            else:
                groups = [(f"{name}_{complexity}", comp_df)]

            for trial_id, trial_group in groups:
                values = trial_group['value'].values.astype(np.float64)
                for start in range(0, len(values) - WINDOW_SIZE + 1, STEP_SIZE):
                    window = values[start:start + WINDOW_SIZE]
                    if len(window) == WINDOW_SIZE and not np.any(np.isnan(window)):
                        all_windows.append({
                            'participant': name,
                            'label': complexity,
                            'trial': str(trial_id),
                            'data': window,
                        })

    # Count per participant
    participants = {}
    for w in all_windows:
        key = w['participant']
        if key not in participants:
            participants[key] = {'Simple': 0, 'Complex': 0}
        participants[key][w['label']] += 1

    print(f"\n[DATA] Total: {len(all_windows)} windows")
    for c in ['Simple', 'Complex']:
        n = sum(1 for w in all_windows if w['label'] == c)
        print(f"  {c}: {n}")

    # Filter out subjects with too few windows in either class
    valid_subjects = set()
    skipped = []
    for subj, counts in participants.items():
        if counts['Simple'] >= MIN_WINDOWS_PER_CLASS and counts['Complex'] >= MIN_WINDOWS_PER_CLASS:
            valid_subjects.add(subj)
        else:
            skipped.append(f"{subj} (S={counts['Simple']}, C={counts['Complex']})")

    if skipped:
        print(f"\n[DATA] Skipping {len(skipped)} subjects with <{MIN_WINDOWS_PER_CLASS} windows/class:")
        for s in skipped:
            print(f"  {s}")

    all_windows = [w for w in all_windows if w['participant'] in valid_subjects]

    # Count unique trials
    trial_set = set()
    for w in all_windows:
        trial_set.add((w['participant'], w['trial'], w['label']))

    print(f"\n[DATA] After filtering: {len(all_windows)} windows from {len(valid_subjects)} subjects")
    print(f"  Unique trials: {len(trial_set)}")
    for c in ['Simple', 'Complex']:
        n = sum(1 for w in all_windows if w['label'] == c)
        t = sum(1 for s, tr, l in trial_set if l == c)
        print(f"  {c}: {n} windows across {t} trials")

    return all_windows


# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def compute_band_power(values, band, fs=SAMPLING_RATE):
    """Welch's PSD → band power."""
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(512, len(values)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapezoid(psd[idx], freqs[idx])


def sample_entropy(data, m=2, r_factor=0.2):
    """Fast sample entropy."""
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
    """Permutation entropy."""
    n = len(data)
    counts = {}
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort([data[i + j * delay] for j in range(order)]))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-10))


def extract_features(window):
    """
    Extract 27 handcrafted features from a 2-sec EEG window.

    Features are organized by category:
      Band Powers (10): absolute + relative for δ, θ, α, β, γ
      Band Ratios (4):  engagement(β/α), cogload(θ/β), taskload(θ/α), relaxation
      Statistical (6):  mean, std, skewness, kurtosis, RMS, peak-to-peak
      Hjorth (3):       activity, mobility, complexity
      Entropy (4):      Shannon, spectral, sample, permutation

    The ratios are directly motivated by our analysis findings:
      - β/α (engagement) peaked at Moderate → distinguishes overload
      - θ/β (cognitive load) → frontal executive control marker
    """
    f = {}

    # --- Band Powers (5 absolute + 5 relative = 10) ---
    bp = {}
    for band_name, (lo, hi) in BANDS.items():
        bp[band_name] = compute_band_power(window, (lo, hi))
        f[f'bp_{band_name}'] = bp[band_name]

    total = sum(bp.values()) + 1e-10
    for band_name, val in bp.items():
        f[f'rp_{band_name}'] = val / total

    # --- Band Ratios (4) — directly from analysis findings ---
    a = bp['alpha'] + 1e-10
    b = bp['beta']  + 1e-10
    t = bp['theta'] + 1e-10
    g = bp['gamma'] + 1e-10

    f['ratio_engagement'] = b / a                # β/α — our strongest ratio finding
    f['ratio_cogload']    = t / b                # θ/β — cognitive load index
    f['ratio_taskload']   = t / a                # θ/α — task demand index
    f['ratio_relaxation'] = (a + t) / (b + g)    # relaxation vs activation

    # --- Statistical Features (6) ---
    f['stat_mean']     = np.mean(window)
    f['stat_std']      = np.std(window)
    f['stat_skewness'] = float(skew(window))
    f['stat_kurtosis'] = float(kurtosis(window))
    f['stat_rms']      = np.sqrt(np.mean(window ** 2))
    f['stat_ptp']      = np.ptp(window)

    # --- Hjorth Parameters (3) ---
    d1 = np.diff(window)
    d2 = np.diff(d1)
    act = np.var(window)
    mob = np.sqrt(np.var(d1) / (act + 1e-10))
    f['hjorth_activity']   = act
    f['hjorth_mobility']   = mob
    f['hjorth_complexity'] = np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mob + 1e-10)

    # --- Entropy Features (4) ---
    hist, _ = np.histogram(window, bins=50, density=True)
    hist = hist[hist > 0]
    f['entropy_shannon'] = -np.sum(hist * np.log2(hist + 1e-10))

    freqs, psd = signal.welch(window, fs=SAMPLING_RATE, nperseg=min(512, len(window)))
    pn = psd / (np.sum(psd) + 1e-10)
    pn = pn[pn > 0]
    f['entropy_spectral']    = -np.sum(pn * np.log2(pn + 1e-10))
    f['entropy_sample']      = sample_entropy(window)
    f['entropy_permutation'] = permutation_entropy(window)

    return f


def build_feature_matrix(windows):
    """Extract features from all windows."""
    print(f"\n[FEATURES] Extracting 27 features from {len(windows)} windows...")
    t0 = time.time()
    rows, labels, subjects, trials = [], [], [], []

    for i, w in enumerate(windows):
        rows.append(extract_features(w['data']))
        labels.append(w['label'])
        subjects.append(w['participant'])
        trials.append(w['trial'])
        if (i + 1) % 500 == 0 or i == len(windows) - 1:
            print(f"  {i+1}/{len(windows)} ({time.time()-t0:.1f}s)")

    feat_names = list(rows[0].keys())
    X = np.array([[r[fn] for fn in feat_names] for r in rows])
    print(f"[FEATURES] Done. Shape: {X.shape}")
    return X, np.array(labels), np.array(subjects), np.array(trials), feat_names


# ==========================================
# 3. NORMALIZATION + CLASS BALANCING
# ==========================================
def normalize_per_subject(X, subjects):
    """Z-score normalize features within each participant."""
    X_norm = X.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        sc = StandardScaler()
        X_norm[mask] = sc.fit_transform(X_norm[mask])
    return X_norm


def undersample_per_subject(X, y, subjects, trials):
    """
    Random undersampling to balance classes within each subject.
    Ensures equal representation per class per participant.
    """
    rng = np.random.RandomState(SEED)
    indices = []

    for subj in np.unique(subjects):
        subj_idx = np.where(subjects == subj)[0]
        subj_labels = y[subj_idx]

        classes, counts = np.unique(subj_labels, return_counts=True)
        if len(classes) < 2:
            continue
        min_count = counts.min()

        for cls in classes:
            cls_idx = subj_idx[subj_labels == cls]
            chosen = rng.choice(cls_idx, size=min_count, replace=False)
            indices.extend(chosen)

    indices = np.array(sorted(indices))
    return X[indices], y[indices], subjects[indices], trials[indices]


def majority_vote_trials(y_true_list, y_pred_list, subjects_list, trials_list):
    """
    Majority voting at the trial level.

    Groups all window predictions by (subject, trial), then takes
    the most-predicted class as the trial-level prediction.
    This simulates real usage: a user views a UI for several seconds,
    and we aggregate all EEG windows from that viewing.

    Returns:
        trial_true: list of true labels per trial
        trial_pred: list of majority-vote predictions per trial
    """
    from collections import Counter

    # Group by (subject, trial)
    trial_groups = {}
    for yt, yp, subj, trial in zip(y_true_list, y_pred_list,
                                    subjects_list, trials_list):
        key = (subj, trial)
        if key not in trial_groups:
            trial_groups[key] = {'true': yt, 'preds': []}
        trial_groups[key]['preds'].append(yp)

    trial_true, trial_pred = [], []
    for key, info in trial_groups.items():
        trial_true.append(info['true'])
        # Majority vote
        counts = Counter(info['preds'])
        trial_pred.append(counts.most_common(1)[0][0])

    return trial_true, trial_pred


# ==========================================
# 4. LOSO CROSS-VALIDATION
# ==========================================
def run_loso(X, y, subjects, trials, feature_names, experiment_name=""):
    """
    Leave-One-Subject-Out cross-validation with Random Forest.
    Returns both window-level and trial-level (majority vote) metrics.
    """
    print(f"\n{'='*60}")
    print(f"LOSO — {experiment_name}")
    print(f"{'='*60}")

    unique_subjects = np.unique(subjects)
    fold_results = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_trials_test = []  # track trial IDs for majority voting
    all_subjects_test = []  # track subjects for majority voting
    feature_importances = np.zeros(len(feature_names))

    for fold, test_subj in enumerate(unique_subjects):
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f"  Fold {fold+1} ({test_subj}): SKIP — only one class")
            continue

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, labels=['Simple', 'Complex'],
                       average='macro', zero_division=0)

        fold_results.append({
            'subject': test_subj,
            'accuracy': acc,
            'f1_macro': f1,
            'n_test': len(y_test),
            'n_simple': np.sum(y_test == 'Simple'),
            'n_complex': np.sum(y_test == 'Complex'),
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_trials_test.extend(trials[test_mask])
        all_subjects_test.extend(subjects[test_mask])
        feature_importances += clf.feature_importances_

        status = "✓" if acc > 0.5 else "✗"
        print(f"  {status} Fold {fold+1:2d} ({test_subj:>30s}): "
              f"Acc={acc:.3f}, F1={f1:.3f} "
              f"(S={np.sum(y_test=='Simple')}, C={np.sum(y_test=='Complex')})")

    feature_importances /= len([r for r in fold_results])

    # Aggregate
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc  = np.std([r['accuracy'] for r in fold_results])
    mean_f1  = np.mean([r['f1_macro'] for r in fold_results])
    std_f1   = np.std([r['f1_macro'] for r in fold_results])

    # Count subjects above chance
    above_chance = sum(1 for r in fold_results if r['accuracy'] > 0.5)
    total_folds = len(fold_results)

    print(f"\n  === RESULTS ===")
    print(f"  Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  F1 Macro: {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"  Subjects above chance (>50%): {above_chance}/{total_folds}")
    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                labels=['Simple', 'Complex'], zero_division=0))

    # === TRIAL-LEVEL MAJORITY VOTING ===
    trial_true, trial_pred = majority_vote_trials(
        all_y_true, all_y_pred, all_subjects_test, all_trials_test
    )
    trial_acc = accuracy_score(trial_true, trial_pred)
    trial_f1  = f1_score(trial_true, trial_pred, labels=['Simple', 'Complex'],
                         average='macro', zero_division=0)
    trial_above = sum(1 for t, p in zip(trial_true, trial_pred) if t == p)

    print(f"\n  === TRIAL-LEVEL (Majority Vote) ===")
    print(f"  Trial Accuracy: {trial_acc:.3f}")
    print(f"  Trial F1 Macro: {trial_f1:.3f}")
    print(f"  Trials correct: {trial_above}/{len(trial_true)}")
    print(f"\n  Trial-Level Classification Report:")
    print(classification_report(trial_true, trial_pred,
                                labels=['Simple', 'Complex'], zero_division=0))

    return {
        'fold_results': fold_results,
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_f1': mean_f1, 'std_f1': std_f1,
        'y_true': all_y_true, 'y_pred': all_y_pred, 'y_prob': all_y_prob,
        'feature_importances': feature_importances,
        'feature_names': feature_names,
        'above_chance': above_chance, 'total_folds': total_folds,
        # Trial-level results
        'trial_true': trial_true, 'trial_pred': trial_pred,
        'trial_acc': trial_acc, 'trial_f1': trial_f1,
    }


# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_feature_importance(results):
    """Plot top feature importances — validates which EEG features matter."""
    imp = results['feature_importances']
    names = results['feature_names']
    idx = np.argsort(imp)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = []
    for i in idx:
        name = names[i]
        if 'ratio' in name:      colors.append('#e74c3c')
        elif 'bp_' in name:      colors.append('#3498db')
        elif 'rp_' in name:      colors.append('#2ecc71')
        elif 'stat_' in name:    colors.append('#f39c12')
        elif 'hjorth' in name:   colors.append('#9b59b6')
        elif 'entropy' in name:  colors.append('#1abc9c')
        else:                    colors.append('#95a5a6')

    ax.barh(range(len(idx)), imp[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx][::-1], fontsize=10)
    ax.set_xlabel('Feature Importance (MDI)', fontsize=12)
    ax.set_title('Random Forest — Feature Importance\n'
                 '(Which EEG features distinguish Simple vs Complex UIs?)',
                 fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Band Ratios (from analysis)'),
        Patch(facecolor='#3498db', label='Absolute Band Power'),
        Patch(facecolor='#2ecc71', label='Relative Band Power'),
        Patch(facecolor='#f39c12', label='Statistical'),
        Patch(facecolor='#9b59b6', label='Hjorth Parameters'),
        Patch(facecolor='#1abc9c', label='Entropy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/feature_importance.png")


def plot_confusion_matrix(results):
    """Plot confusion matrix."""
    cm = confusion_matrix(results['y_true'], results['y_pred'],
                          labels=['Simple', 'Complex'])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Simple', 'Complex'],
                yticklabels=['Simple', 'Complex'], ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Simple', 'Complex'],
                yticklabels=['Simple', 'Complex'], ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.suptitle(f'Random Forest Binary — Acc={results["mean_acc"]:.1%}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/confusion_matrix.png")


def plot_per_subject(results):
    """Per-subject accuracy chart."""
    subjects = [r['subject'] for r in results['fold_results']]
    accs = [r['accuracy'] for r in results['fold_results']]

    # Sort by accuracy for readability
    sorted_pairs = sorted(zip(subjects, accs), key=lambda x: x[1], reverse=True)
    subjects_sorted = [p[0] for p in sorted_pairs]
    accs_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#2ecc71' if a > 0.5 else '#e74c3c' for a in accs_sorted]
    bars = ax.barh(range(len(subjects_sorted)), accs_sorted,
                   color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (50%)')
    ax.axvline(x=results['mean_acc'], color='#3498db', linestyle='-',
               alpha=0.6, label=f'Mean ({results["mean_acc"]:.1%})')

    ax.set_yticks(range(len(subjects_sorted)))
    ax.set_yticklabels(subjects_sorted, fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Per-Subject Accuracy — Random Forest Binary\n'
                 f'{results["above_chance"]}/{results["total_folds"]} subjects above chance',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/per_subject_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/per_subject_accuracy.png")


def save_results_csv(results):
    """Save detailed results."""
    rows = []
    for fold in results['fold_results']:
        rows.append({
            'Subject': fold['subject'],
            'Accuracy': fold['accuracy'],
            'F1_Macro': fold['f1_macro'],
            'N_Test': fold['n_test'],
            'N_Simple': fold['n_simple'],
            'N_Complex': fold['n_complex'],
        })
    rows.append({
        'Subject': 'AVERAGE',
        'Accuracy': results['mean_acc'],
        'F1_Macro': results['mean_f1'],
        'N_Test': '',
        'N_Simple': '',
        'N_Complex': '',
    })

    df = pd.DataFrame(rows)
    path = f'{RESULTS_DIR}/results.csv'
    df.to_csv(path, index=False)
    print(f"  → Saved: {path}")

    # Feature importance CSV
    imp_df = pd.DataFrame({
        'Feature': results['feature_names'],
        'Importance': results['feature_importances'],
    }).sort_values('Importance', ascending=False)
    imp_path = f'{RESULTS_DIR}/feature_importance.csv'
    imp_df.to_csv(imp_path, index=False)
    print(f"  → Saved: {imp_path}")


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 60)
    print("RANDOM FOREST — Binary (Simple vs Complex)")
    print("Experiment: class_weight ONLY  vs  undersampling + class_weight")
    print("=" * 60)
    print(f"  Features: 27 handcrafted (band power, ratios, entropy, Hjorth)")
    print(f"  Evaluation: Leave-One-Subject-Out (LOSO)")
    print(f"  Min windows/class/subject: {MIN_WINDOWS_PER_CLASS}")

    # 1. Load & window
    windows = load_and_window()

    if len(windows) == 0:
        print("[ERROR] No valid windows. Check data paths.")
        return

    # 2. Extract features
    X, y, subjects, trials, feat_names = build_feature_matrix(windows)

    # 3. Per-subject normalization
    X_norm = normalize_per_subject(X, subjects)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # ==========================================
    # EXPERIMENT A: No undersampling (class_weight='balanced' only)
    # ==========================================
    print(f"\n{'='*60}")
    print("EXPERIMENT A: class_weight='balanced' only (all data)")
    print(f"{'='*60}")
    for c in ['Simple', 'Complex']:
        print(f"  {c}: {np.sum(y == c)} windows")

    results_A = run_loso(X_norm, y, subjects, trials, feat_names,
                         "A: class_weight='balanced' (no undersampling)")

    # ==========================================
    # EXPERIMENT B: Per-subject undersampling + class_weight='balanced'
    # ==========================================
    X_bal, y_bal, subj_bal, trial_bal = undersample_per_subject(X_norm, y, subjects, trials)

    print(f"\n{'='*60}")
    print("EXPERIMENT B: Per-subject undersampling + class_weight='balanced'")
    print(f"{'='*60}")
    for c in ['Simple', 'Complex']:
        print(f"  {c}: {np.sum(y_bal == c)} windows")
    print(f"  Total: {len(y_bal)} windows from {len(np.unique(subj_bal))} subjects")

    results_B = run_loso(X_bal, y_bal, subj_bal, trial_bal, feat_names,
                         "B: undersampling + class_weight='balanced'")

    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    # Use the better result for detailed plots
    best = results_B if results_B['mean_f1'] > results_A['mean_f1'] else results_A
    best_name = 'B (undersampled)' if best is results_B else 'A (all data)'

    plot_confusion_matrix(best)
    plot_feature_importance(best)
    plot_per_subject(best)
    save_results_csv(best)

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['A: All data\n(class_weight only)', 'B: Undersampled\n+ class_weight']
    colors = ['#3498db', '#2ecc71']

    # Accuracy
    accs = [results_A['mean_acc'], results_B['mean_acc']]
    stds = [results_A['std_acc'], results_B['std_acc']]
    bars = axes[0].bar(labels, accs, yerr=stds, color=colors,
                       capsize=8, alpha=0.85, edgecolor='black')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (50%)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for bar, acc, std in zip(bars, accs, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                     f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')

    # F1
    f1s = [results_A['mean_f1'], results_B['mean_f1']]
    f1_stds = [results_A['std_f1'], results_B['std_f1']]
    bars = axes[1].bar(labels, f1s, yerr=f1_stds, color=colors,
                       capsize=8, alpha=0.85, edgecolor='black')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.6, label='Chance')
    axes[1].set_ylabel('F1 Score (Macro)')
    axes[1].set_title('F1 Macro Comparison', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for bar, f1, std in zip(bars, f1s, f1_stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                     f'{f1:.1%}', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('Effect of Undersampling on RF Binary Classification',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/comparison_AB.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {RESULTS_DIR}/comparison_AB.png")

    # ==========================================
    # FEATURE IMPORTANCE (from best model)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE (from {best_name})")
    print(f"{'='*60}")

    imp = best['feature_importances']
    names = best['feature_names']
    sorted_idx = np.argsort(imp)[::-1]

    analysis_features = {
        'bp_beta': 'Beta power (p=0.005 in analysis)',
        'bp_gamma': 'Gamma power (p=0.042 in analysis)',
        'ratio_engagement': 'β/α Engagement Index (p=0.040 in analysis)',
        'ratio_cogload': 'θ/β Cognitive Load ratio',
        'rp_beta': 'Relative Beta power',
        'rp_gamma': 'Relative Gamma power',
    }

    print(f"\n  Top 10 features by RF importance:")
    for rank, i in enumerate(sorted_idx[:10]):
        marker = " ← ANALYSIS HIT" if names[i] in analysis_features else ""
        print(f"    {rank+1:2d}. {names[i]:<25s} {imp[i]:.4f}{marker}")

    print(f"\n  Analysis-significant features — their RF rank:")
    for feat, desc in analysis_features.items():
        if feat in names:
            rank = list(sorted_idx).index(names.index(feat)) + 1
            print(f"    #{rank:2d} {feat:<25s} → {desc}")

    # ==========================================
    # SIDE-BY-SIDE COMPARISON
    # ==========================================
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"")
    print(f"  {'Metric':<30s} {'A: All data':>15s} {'B: Undersampled':>15s}")
    print(f"  {'-'*60}")
    print(f"  {'WINDOW-LEVEL':<30s}")
    print(f"  {'  Accuracy':<30s} {results_A['mean_acc']:>14.1%} {results_B['mean_acc']:>14.1%}")
    print(f"  {'  F1 Macro':<30s} {results_A['mean_f1']:>14.1%} {results_B['mean_f1']:>14.1%}")
    print(f"  {'  Subjects > chance':<30s} {results_A['above_chance']:>7d}/{results_A['total_folds']:<6d} {results_B['above_chance']:>7d}/{results_B['total_folds']:<6d}")
    print(f"  {'-'*60}")
    print(f"  {'TRIAL-LEVEL (Majority Vote)':<30s}")
    print(f"  {'  Trial Accuracy':<30s} {results_A['trial_acc']:>14.1%} {results_B['trial_acc']:>14.1%}")
    print(f"  {'  Trial F1 Macro':<30s} {results_A['trial_f1']:>14.1%} {results_B['trial_f1']:>14.1%}")
    print(f"  {'-'*60}")
    print(f"  {'N windows':<30s} {np.sum(y != ''):>15d} {np.sum(y_bal != ''):>15d}")
    print(f"")

    winner = 'B (undersampled)' if results_B['trial_f1'] > results_A['trial_f1'] else 'A (all data)'
    print(f"  → Winner (by Trial F1): {winner}")
    print(f"  Chance level: 50.0%")
    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
