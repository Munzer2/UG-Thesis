"""
=============================================================
EEG Sub-Band Power Analysis
=============================================================
Computes detailed sub-band power (low, mid, high) for all
EEG frequency bands across participants and complexity levels.

Bands analyzed:
  - Low Delta:   1-2 Hz
  - Mid Delta:   2-3 Hz
  - High Delta:  3-4 Hz
  - Low Theta:   4-5.5 Hz   (frontal cognitive control)
  - Mid Theta:   5.5-6.5 Hz
  - High Theta:  6.5-8 Hz   (memory encoding)
  - Low Alpha:   8-10 Hz
  - Mid Alpha:   10-11.5 Hz
  - High Alpha:  11.5-13 Hz
  - Low Beta:    13-20 Hz
  - Mid Beta:    20-25 Hz
  - High Beta:   25-30 Hz
  - Low Gamma:   30-37 Hz
  - Mid Gamma:   37-44 Hz
  - High Gamma:  44-50 Hz

Derived Ratios:
  - Engagement Index:   β / α
  - Cognitive Load:     θ / β
  - Task Load:          θ / α
  - Theta/Beta Ratio:   θ / highβ
  - Alpha Ratio:        highα / lowα
  - Beta Ratio:         highβ / lowβ
  - Mid Alpha Ratio:    midα / (lowα + highα)
  - Mid Beta Ratio:     midβ / (lowβ + highβ)
  - Delta Ratio:        highδ / lowδ
  - Theta Ratio:        highθ / lowθ

Output:
  - CSV:  results_analysis/subband_analysis.csv
  - Plots in results_analysis/

Usage:
  python analyze_subbands.py
============================================================="""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as sig
from scipy.stats import kruskal, mannwhitneyu, spearmanr

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # main/ directory
DATA_DIR = os.path.join(BASE_DIR, "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLING_RATE = 512
WINDOW_SEC = 2
WINDOW_SIZE = WINDOW_SEC * SAMPLING_RATE  # 1024

LABEL_MAP = {
    'design_A_simple':   'Simple',
    'design_B_complex':  'Complex',
    'design_C_moderate': 'Moderate',
}

# Sub-band definitions (Hz) — Low / Mid / High for ALL bands
SUBBANDS = {
    'Low Delta':  (1, 2),
    'Mid Delta':  (2, 3),
    'High Delta': (3, 4),
    'Low Theta':  (4, 5.5),
    'Mid Theta':  (5.5, 6.5),
    'High Theta': (6.5, 8),
    'Low Alpha':  (8, 10),
    'Mid Alpha':  (10, 11.5),
    'High Alpha': (11.5, 13),
    'Low Beta':   (13, 20),
    'Mid Beta':   (20, 25),
    'High Beta':  (25, 30),
    'Low Gamma':  (30, 37),
    'Mid Gamma':  (37, 44),
    'High Gamma': (44, 50),
}

# Aggregated bands (for ratio computation)
AGG_BANDS = {
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Theta_full': (4, 8),
}

# Plot styling
sns.set_style("whitegrid")
COLORS_3 = {'Simple': '#2ecc71', 'Moderate': '#f39c12', 'Complex': '#e74c3c'}
COLORS_2 = {'Simple': '#2ecc71', 'Complex': '#e74c3c'}


# ==========================================
# 1. COMPUTE PSD PER TRIAL
# ==========================================
def compute_trial_psd(raw_values, fs=512, nperseg=1024):
    """
    Compute Welch PSD for a single trial's raw EEG values.
    Returns (frequencies, psd_values).
    """
    if len(raw_values) < nperseg:
        return None, None
    freqs, psd = sig.welch(raw_values, fs=fs, nperseg=nperseg)
    return freqs, psd


def band_power(freqs, psd, low, high):
    """Extract mean power in a frequency band."""
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        return 0.0
    return np.mean(psd[idx])


# ==========================================
# 2. LOAD DATA AND COMPUTE SUB-BAND POWERS
# ==========================================
def load_and_analyze():
    """
    Load all cleaned CSVs, compute per-trial PSD,
    extract sub-band powers and derived ratios.
    """
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found: {FILE_PATTERN}")

    print(f"[DATA] Loading {len(files)} files...")
    all_trials = []

    for f in files:
        df = pd.read_csv(f)
        name = os.path.basename(f).replace('UI_Exp_', '').replace('.csv', '').split('_')[0]

        # Filter: raw EEG, TASK phase, no artifacts
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()

        if task_df.empty:
            print(f"  [SKIP] {name}: no valid TASK samples")
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        # Process each trial (unique image within each complexity)
        for complexity in ['Simple', 'Moderate', 'Complex']:
            comp_df = task_df[task_df['complexity'] == complexity]
            if comp_df.empty:
                continue

            for image_name, trial_group in comp_df.groupby('image'):
                values = trial_group['value'].values.astype(np.float64)
                values = values[~np.isnan(values)]

                if len(values) < WINDOW_SIZE:
                    continue

                # Compute PSD for this trial
                freqs, psd = compute_trial_psd(values, fs=SAMPLING_RATE, nperseg=WINDOW_SIZE)
                if freqs is None:
                    continue

                # Extract sub-band powers
                row = {
                    'Participant': name,
                    'Complexity': complexity,
                    'Image': image_name,
                    'N_Samples': len(values),
                }

                for band_name, (lo, hi) in SUBBANDS.items():
                    row[band_name] = band_power(freqs, psd, lo, hi)

                # Aggregated bands for ratios
                alpha_power = band_power(freqs, psd, 8, 13)
                beta_power = band_power(freqs, psd, 13, 30)
                theta_power = band_power(freqs, psd, 4, 8)
                delta_power = band_power(freqs, psd, 1, 4)
                low_delta = row['Low Delta']
                high_delta = row['High Delta']
                low_theta = row['Low Theta']
                high_theta = row['High Theta']
                low_alpha = row['Low Alpha']
                mid_alpha = row['Mid Alpha']
                high_alpha = row['High Alpha']
                low_beta = row['Low Beta']
                mid_beta = row['Mid Beta']
                high_beta = row['High Beta']

                # Derived ratios
                row['Engagement (β/α)'] = beta_power / (alpha_power + 1e-10)
                row['CogLoad (θ/β)'] = theta_power / (beta_power + 1e-10)
                row['TaskLoad (θ/α)'] = theta_power / (alpha_power + 1e-10)
                row['Theta/HighBeta'] = theta_power / (high_beta + 1e-10)
                row['Alpha Ratio (H/L)'] = high_alpha / (low_alpha + 1e-10)
                row['Beta Ratio (H/L)'] = high_beta / (low_beta + 1e-10)
                row['Delta Ratio (H/L)'] = high_delta / (low_delta + 1e-10)
                row['Theta Ratio (H/L)'] = high_theta / (low_theta + 1e-10)
                row['Mid Alpha Ratio'] = mid_alpha / (low_alpha + high_alpha + 1e-10)
                row['Mid Beta Ratio'] = mid_beta / (low_beta + high_beta + 1e-10)
                row['Total Power'] = sum(row[b] for b in SUBBANDS)

                # Relative powers (proportion of total)
                total = row['Total Power'] + 1e-10
                for band_name in SUBBANDS:
                    row[f'{band_name} (%)'] = (row[band_name] / total) * 100

                all_trials.append(row)

        n_trials = sum(1 for t in all_trials if t['Participant'] == name)
        print(f"  {name}: {n_trials} trials")

    df_result = pd.DataFrame(all_trials)
    print(f"\n[DATA] Total trials analyzed: {len(df_result)}")
    for c in ['Simple', 'Moderate', 'Complex']:
        n = len(df_result[df_result['Complexity'] == c])
        print(f"  {c}: {n} trials")

    return df_result


# ==========================================
# 3. STATISTICAL TESTS
# ==========================================
def run_statistics(df):
    """Run Kruskal-Wallis and Mann-Whitney tests on all sub-bands."""
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    bands_to_test = list(SUBBANDS.keys()) + [
        'Engagement (β/α)', 'CogLoad (θ/β)', 'TaskLoad (θ/α)',
        'Theta/HighBeta', 'Alpha Ratio (H/L)', 'Beta Ratio (H/L)',
        'Delta Ratio (H/L)', 'Theta Ratio (H/L)',
        'Mid Alpha Ratio', 'Mid Beta Ratio',
    ]

    stats_rows = []

    # --- 3-class Kruskal-Wallis ---
    print(f"\n--- Kruskal-Wallis (3-class: Simple vs Moderate vs Complex) ---")
    print(f"{'Feature':<20s} {'H-stat':>8s} {'p-value':>10s} {'Sig':>5s} {'S_mean':>10s} {'M_mean':>10s} {'C_mean':>10s}")
    print("-" * 75)

    for feat in bands_to_test:
        s_vals = df[df.Complexity == 'Simple'][feat].dropna().values
        m_vals = df[df.Complexity == 'Moderate'][feat].dropna().values
        c_vals = df[df.Complexity == 'Complex'][feat].dropna().values

        if len(s_vals) < 3 or len(m_vals) < 3 or len(c_vals) < 3:
            continue

        h, p = kruskal(s_vals, m_vals, c_vals)
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{feat:<20s} {h:8.2f} {p:10.4f} {sig_str:>5s} {np.mean(s_vals):10.2f} {np.mean(m_vals):10.2f} {np.mean(c_vals):10.2f}")

        stats_rows.append({
            'Feature': feat, 'Test': 'Kruskal-Wallis (3-class)',
            'Statistic': h, 'p_value': p, 'Significant': sig_str,
            'Simple_mean': np.mean(s_vals), 'Moderate_mean': np.mean(m_vals),
            'Complex_mean': np.mean(c_vals),
        })

    # --- Binary Mann-Whitney ---
    print(f"\n--- Mann-Whitney U (Binary: Simple vs Complex) ---")
    print(f"{'Feature':<20s} {'U-stat':>10s} {'p-value':>10s} {'Sig':>5s} {'S_mean':>10s} {'C_mean':>10s} {'Diff%':>8s} {'Dir':>5s}")
    print("-" * 80)

    for feat in bands_to_test:
        s_vals = df[df.Complexity == 'Simple'][feat].dropna().values
        c_vals = df[df.Complexity == 'Complex'][feat].dropna().values

        if len(s_vals) < 3 or len(c_vals) < 3:
            continue

        u, p = mannwhitneyu(s_vals, c_vals, alternative='two-sided')
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        s_mean = np.mean(s_vals)
        c_mean = np.mean(c_vals)
        diff_pct = ((c_mean - s_mean) / (s_mean + 1e-10)) * 100
        direction = 'C>S' if c_mean > s_mean else 'S>C'

        print(f"{feat:<20s} {u:10.1f} {p:10.4f} {sig_str:>5s} {s_mean:10.2f} {c_mean:10.2f} {diff_pct:+7.1f}% {direction:>5s}")

        stats_rows.append({
            'Feature': feat, 'Test': 'Mann-Whitney U (Binary)',
            'Statistic': u, 'p_value': p, 'Significant': sig_str,
            'Simple_mean': s_mean, 'Complex_mean': c_mean,
            'Diff_percent': diff_pct, 'Direction': direction,
        })

    return pd.DataFrame(stats_rows)


# ==========================================
# 4. VISUALIZATIONS
# ==========================================
def plot_subband_absolute(df):
    """Bar plot of absolute sub-band powers by complexity."""
    band_names = list(SUBBANDS.keys())
    complexities = ['Simple', 'Moderate', 'Complex']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(band_names))
    width = 0.25

    for i, comp in enumerate(complexities):
        means = [df[df.Complexity == comp][b].mean() for b in band_names]
        sems = [df[df.Complexity == comp][b].sem() for b in band_names]
        bars = ax.bar(x + i * width, means, width, label=comp,
                      color=COLORS_3[comp], alpha=0.85, edgecolor='black',
                      yerr=sems, capsize=3)

    ax.set_xlabel('Sub-Band', fontsize=12)
    ax.set_ylabel('Power (µV²/Hz)', fontsize=12)
    ax.set_title('Absolute Sub-Band Power by UI Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(band_names, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_absolute_power.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_subband_relative(df):
    """Bar plot of relative (%) sub-band powers by complexity."""
    band_names = [f'{b} (%)' for b in SUBBANDS.keys()]
    display_names = list(SUBBANDS.keys())
    complexities = ['Simple', 'Moderate', 'Complex']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(display_names))
    width = 0.25

    for i, comp in enumerate(complexities):
        means = [df[df.Complexity == comp][b].mean() for b in band_names]
        sems = [df[df.Complexity == comp][b].sem() for b in band_names]
        ax.bar(x + i * width, means, width, label=comp,
               color=COLORS_3[comp], alpha=0.85, edgecolor='black',
               yerr=sems, capsize=3)

    ax.set_xlabel('Sub-Band', fontsize=12)
    ax.set_ylabel('Relative Power (%)', fontsize=12)
    ax.set_title('Relative Sub-Band Power by UI Complexity', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_relative_power.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_subband_boxplots(df):
    """Box plots for each sub-band split by complexity (Simple vs Complex only)."""
    band_names = list(SUBBANDS.keys())
    n_bands = len(band_names)
    n_cols = 4
    n_rows = (n_bands + n_cols - 1) // n_cols  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    binary_df = df[df.Complexity.isin(['Simple', 'Complex'])]

    for i, band in enumerate(band_names):
        ax = axes[i]
        data_s = binary_df[binary_df.Complexity == 'Simple'][band].values
        data_c = binary_df[binary_df.Complexity == 'Complex'][band].values

        # Mann-Whitney for annotation
        u, p = mannwhitneyu(data_s, data_c, alternative='two-sided')
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        bp = ax.boxplot([data_s, data_c], labels=['Simple', 'Complex'],
                        patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(COLORS_2['Simple'])
        bp['boxes'][1].set_facecolor(COLORS_2['Complex'])
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax.set_title(f'{band}\np={p:.4f} ({sig_str})', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power (µV²/Hz)')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_bands, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Sub-Band Power: Simple vs Complex', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_boxplots.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_ratio_boxplots(df):
    """Box plots for derived EEG ratios by complexity."""
    ratios = ['Engagement (β/α)', 'CogLoad (θ/β)', 'TaskLoad (θ/α)',
              'Theta/HighBeta', 'Alpha Ratio (H/L)', 'Beta Ratio (H/L)',
              'Delta Ratio (H/L)', 'Theta Ratio (H/L)',
              'Mid Alpha Ratio', 'Mid Beta Ratio']
    complexities = ['Simple', 'Moderate', 'Complex']

    n_ratios = len(ratios)
    n_cols = 5
    n_rows = (n_ratios + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 5 * n_rows))
    axes = axes.flatten()

    for i, ratio in enumerate(ratios):
        ax = axes[i]
        data = [df[df.Complexity == c][ratio].dropna().values for c in complexities]

        # Kruskal-Wallis
        h, p = kruskal(*data)
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        bp = ax.boxplot(data, labels=complexities, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
        for j, comp in enumerate(complexities):
            bp['boxes'][j].set_facecolor(COLORS_3[comp])
            bp['boxes'][j].set_alpha(0.7)

        # Overlay individual points
        for j, d in enumerate(data):
            jitter = np.random.normal(0, 0.04, len(d))
            ax.scatter(np.full_like(d, j + 1) + jitter, d,
                       alpha=0.15, s=10, color='gray', zorder=0)

        ax.set_title(f'{ratio}\nH={h:.2f}, p={p:.4f} ({sig_str})',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(len(ratios), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('EEG Derived Ratios by UI Complexity', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_ratios.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_per_participant_heatmap(df):
    """Heatmap of sub-band power difference (Complex - Simple) per participant."""
    band_names = list(SUBBANDS.keys())
    binary_df = df[df.Complexity.isin(['Simple', 'Complex'])]
    participants = sorted(binary_df['Participant'].unique())

    diff_matrix = []
    for p in participants:
        pdf = binary_df[binary_df.Participant == p]
        s_means = pdf[pdf.Complexity == 'Simple'][band_names].mean()
        c_means = pdf[pdf.Complexity == 'Complex'][band_names].mean()
        # Percent difference
        diff = ((c_means - s_means) / (s_means.abs() + 1e-10)) * 100
        diff_matrix.append(diff)

    diff_df = pd.DataFrame(diff_matrix, index=participants)

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(diff_df, cmap='RdBu_r', center=0, annot=True, fmt='.0f',
                linewidths=0.5, ax=ax, cbar_kws={'label': '% Change (Complex − Simple)'},
                vmin=-50, vmax=50, annot_kws={'size': 8})
    ax.set_title('Per-Participant Sub-Band Power Change\n(Complex − Simple, % difference)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sub-Band')
    ax.set_ylabel('Participant')

    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_per_participant_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_psd_curves(df_raw_files):
    """
    Plot average PSD curves for Simple vs Complex,
    with shaded sub-band regions.
    """
    files = sorted(glob.glob(FILE_PATTERN))
    all_psds = {'Simple': [], 'Complex': []}

    for f in files:
        df = pd.read_csv(f)
        mask = (df['type'] == 'raw') & (df['phase'] == 'TASK')
        if 'is_artifact' in df.columns:
            mask &= (df['is_artifact'] == False)
        task_df = df[mask].copy()
        if task_df.empty:
            continue

        task_df['complexity'] = task_df['label'].map(LABEL_MAP)

        for comp in ['Simple', 'Complex']:
            cdf = task_df[task_df['complexity'] == comp]
            vals = cdf['value'].values.astype(np.float64)
            vals = vals[~np.isnan(vals)]
            if len(vals) < WINDOW_SIZE:
                continue
            freqs, psd = sig.welch(vals, fs=SAMPLING_RATE, nperseg=WINDOW_SIZE)
            all_psds[comp].append(psd)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Shade sub-band regions
    band_colors = {
        'Low Delta': '#c8e6c9', 'Mid Delta': '#a5d6a7', 'High Delta': '#81c784',
        'Low Theta': '#ffe0b2', 'Mid Theta': '#ffcc80', 'High Theta': '#ffb74d',
        'Low Alpha': '#e3f2fd', 'Mid Alpha': '#bbdefb', 'High Alpha': '#90caf9',
        'Low Beta': '#fce4ec', 'Mid Beta': '#f8bbd0', 'High Beta': '#f48fb1',
        'Low Gamma': '#fff9c4', 'Mid Gamma': '#ffecb3', 'High Gamma': '#ffe082',
    }
    for band, (lo, hi) in SUBBANDS.items():
        ax.axvspan(lo, hi, alpha=0.3, color=band_colors[band], label=f'_{band}')
        ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200,
                band.replace(' ', '\n'), ha='center', va='top', fontsize=7, alpha=0.6)

    # Plot mean PSD with SEM shading
    freqs_ref = None
    for comp, color in [('Simple', '#2ecc71'), ('Complex', '#e74c3c')]:
        psds = np.array(all_psds[comp])
        mean_psd = np.mean(psds, axis=0)
        sem_psd = np.std(psds, axis=0) / np.sqrt(len(psds))
        freqs_ref = freqs

        ax.semilogy(freqs, mean_psd, color=color, linewidth=2, label=f'{comp} (n={len(psds)})')
        ax.fill_between(freqs, mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.2, color=color)

    # Add sub-band region labels on top
    for band, (lo, hi) in SUBBANDS.items():
        ax.axvspan(lo, hi, alpha=0.15, color=band_colors[band])
        mid = (lo + hi) / 2
        ax.annotate(band.replace(' ', '\n'), xy=(mid, ax.get_ylim()[0]),
                    fontsize=7, ha='center', va='bottom', alpha=0.7,
                    xytext=(mid, 0.02), textcoords=('data', 'axes fraction'))

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=12)
    ax.set_title('PSD: Simple vs Complex (with Sub-Band Regions)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 55)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_psd_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


def plot_3class_subband_comparison(df):
    """Grouped bar chart comparing all 3 classes across sub-bands with significance markers."""
    band_names = list(SUBBANDS.keys())
    complexities = ['Simple', 'Moderate', 'Complex']
    n_bands = len(band_names)
    n_cols = 4
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    axes = axes.flatten()

    for i, band in enumerate(band_names):
        ax = axes[i]
        data = {c: df[df.Complexity == c][band].values for c in complexities}

        # Plot with jitter
        positions = [1, 2, 3]
        for j, (comp, pos) in enumerate(zip(complexities, positions)):
            vals = data[comp]
            jitter = np.random.normal(0, 0.08, len(vals))
            ax.scatter(np.full_like(vals, pos) + jitter, vals,
                       alpha=0.15, s=8, color=COLORS_3[comp], zorder=0)

        bp = ax.boxplot([data[c] for c in complexities],
                        labels=complexities, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        for j, comp in enumerate(complexities):
            bp['boxes'][j].set_facecolor(COLORS_3[comp])
            bp['boxes'][j].set_alpha(0.65)

        # Kruskal-Wallis
        h, p = kruskal(data['Simple'], data['Moderate'], data['Complex'])
        sig_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        ax.set_title(f'{band}\nH={h:.2f}, p={p:.4f} ({sig_str})', fontsize=10, fontweight='bold')
        ax.set_ylabel('Power (µV²/Hz)', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_bands, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Sub-Band Power: Simple vs Moderate vs Complex',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f'{RESULTS_DIR}/subband_3class_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {path}")


# ==========================================
# 5. MAIN
# ==========================================
def main():
    print("=" * 60)
    print("EEG Sub-Band Power Analysis")
    print("=" * 60)

    # Load and compute
    df = load_and_analyze()

    # Save raw results
    csv_path = f'{RESULTS_DIR}/subband_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  → Raw data saved: {csv_path}")

    # Statistics
    stats_df = run_statistics(df)
    stats_path = f'{RESULTS_DIR}/subband_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  → Statistics saved: {stats_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE: Mean Sub-Band Power")
    print(f"{'='*60}")
    band_names = list(SUBBANDS.keys())
    print(f"{'Band':<15s} {'Simple':>10s} {'Moderate':>10s} {'Complex':>10s}")
    print("-" * 50)
    for b in band_names:
        s = df[df.Complexity == 'Simple'][b].mean()
        m = df[df.Complexity == 'Moderate'][b].mean()
        c = df[df.Complexity == 'Complex'][b].mean()
        print(f"{b:<15s} {s:10.2f} {m:10.2f} {c:10.2f}")

    # Visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    plot_subband_absolute(df)
    plot_subband_relative(df)
    plot_subband_boxplots(df)
    plot_ratio_boxplots(df)
    plot_per_participant_heatmap(df)
    plot_psd_curves(df)
    plot_3class_comparison(df)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


# Fix function name reference
plot_3class_comparison = plot_3class_subband_comparison

if __name__ == "__main__":
    main()
