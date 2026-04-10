"""
=============================================================
Comprehensive EEG Analysis for Cognitive Load Study
=============================================================
Analyses:
  1. Behavioral: Reaction time by complexity
  2. Spectral:   PSD from cleaned raw EEG per complexity level
  3. Band Power: Per-band comparison (delta, theta, alpha, beta, gamma)
  4. EEG Indices: Engagement (beta/alpha), Cognitive Load (theta/beta),
                  Task Load Index (theta/alpha)
  5. Per-Subject: Individual participant breakdowns
  6. Statistical: ANOVA + post-hoc + effect sizes
  7. Temporal:    Band power evolution within trials
  8. Correlation: Neural vs Behavioral measures
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import signal, stats
from itertools import combinations

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "dataset_clean")
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR = os.path.join("..", "results_analysis")
SAMPLING_RATE = 512

# Frequency bands
BANDS = {
    'Delta (1-4 Hz)':  (1, 4),
    'Theta (4-8 Hz)':  (4, 8),
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
    'Gamma (30-50 Hz)':(30, 50),
}

# Clean label mapping
LABEL_MAP = {
    'design_A_simple':   'Simple',
    'design_B_complex':  'Complex',
    'design_C_moderate': 'Moderate',
}
# Ordered for plotting
LABEL_ORDER = ['Simple', 'Moderate', 'Complex']

# Color palette
PALETTE = {'Simple': '#2ecc71', 'Moderate': '#f39c12', 'Complex': '#e74c3c'}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# DATA LOADING
# ==========================================
def load_all_data():
    """Load all cleaned CSVs, return raw-only + power-only DataFrames."""
    files = glob.glob(FILE_PATTERN)
    if not files:
        print(f"[ERROR] No files found in {DATA_DIR}")
        return None, None

    print(f"Loading {len(files)} files...")
    raw_list, power_list = [], []

    for f in files:
        df = pd.read_csv(f)
        participant = os.path.basename(f).replace('UI_Exp_', '').replace('.csv', '')
        # Extract just the name part before underscore+digits
        name = participant.split('_')[0]
        df['participant'] = name
        df['complexity'] = df['label'].map(LABEL_MAP)

        # Separate raw and power rows
        raw_df = df[df['type'] == 'raw'].copy()
        power_df = df[df['type'] == 'power'].copy()

        # Filter artifacts from raw
        if 'is_artifact' in raw_df.columns:
            raw_df = raw_df[raw_df['is_artifact'] == False]

        raw_list.append(raw_df)
        power_list.append(power_df)

    return pd.concat(raw_list, ignore_index=True), pd.concat(power_list, ignore_index=True)


# ==========================================
# HELPER: COMPUTE PSD FOR A SEGMENT
# ==========================================
def compute_band_powers(values, fs=SAMPLING_RATE):
    """Compute absolute and relative band powers from raw EEG values."""
    freqs, psd = signal.welch(values, fs=fs, nperseg=min(512, len(values)))
    total_power = np.trapezoid(psd, freqs)

    powers = {}
    for band_name, (fmin, fmax) in BANDS.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        abs_power = np.trapezoid(psd[idx], freqs[idx])
        powers[band_name] = abs_power
        powers[f'{band_name}_rel'] = abs_power / (total_power + 1e-10)

    return powers, freqs, psd


def get_sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def cohens_d(g1, g2):
    """Effect size: Cohen's d."""
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / (pooled_std + 1e-10)


# ==========================================
# ANALYSIS 1: REACTION TIME
# ==========================================
def analyze_reaction_time(raw_df):
    """Behavioral analysis: reaction time by UI complexity."""
    print("\n" + "="*60)
    print("ANALYSIS 1: REACTION TIME BY COMPLEXITY")
    print("="*60)

    # Get unique trials
    task_df = raw_df[raw_df['phase'] == 'TASK']
    trials = task_df.drop_duplicates(subset=['participant', 'image', 'target_instruction'])
    trials = trials[trials['reaction_time'] > 0]  # Remove invalid

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1a: Overall
    sns.boxplot(data=trials, x='complexity', y='reaction_time', order=LABEL_ORDER,
                palette=PALETTE, showfliers=False, ax=axes[0])
    sns.stripplot(data=trials, x='complexity', y='reaction_time', order=LABEL_ORDER,
                  color='black', alpha=0.3, size=3, ax=axes[0])
    axes[0].set_title('Reaction Time by UI Complexity', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Reaction Time (s)')
    axes[0].set_xlabel('')

    # 1b: Per participant
    sns.boxplot(data=trials, x='complexity', y='reaction_time', hue='participant',
                order=LABEL_ORDER, showfliers=False, ax=axes[1])
    axes[1].set_title('Reaction Time by Participant', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Reaction Time (s)')
    axes[1].set_xlabel('')
    axes[1].legend(title='Participant', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/1_reaction_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stats
    groups = [trials[trials['complexity'] == c]['reaction_time'].values for c in LABEL_ORDER]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) >= 2:
        H, p_kw = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H={H:.3f}, p={p_kw:.4f} {get_sig_stars(p_kw)}")

        # Pairwise Mann-Whitney U
        for (i, c1), (j, c2) in combinations(enumerate(LABEL_ORDER), 2):
            g1 = trials[trials['complexity'] == c1]['reaction_time'].values
            g2 = trials[trials['complexity'] == c2]['reaction_time'].values
            if len(g1) > 0 and len(g2) > 0:
                U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                d = cohens_d(g1, g2)
                print(f"  {c1} vs {c2}: U={U:.0f}, p={p:.4f} {get_sig_stars(p)}, Cohen's d={d:.3f}")

    print("  → Saved: 1_reaction_time.png")


# ==========================================
# ANALYSIS 2: PSD BY COMPLEXITY
# ==========================================
def analyze_psd(raw_df):
    """Spectral analysis: PSD from cleaned raw EEG per complexity level."""
    print("\n" + "="*60)
    print("ANALYSIS 2: POWER SPECTRAL DENSITY BY COMPLEXITY")
    print("="*60)

    task_df = raw_df[raw_df['phase'] == 'TASK']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2a: Overlay PSD
    for complexity in LABEL_ORDER:
        vals = task_df[task_df['complexity'] == complexity]['value'].values.astype(float)
        if len(vals) < 512:
            continue
        freqs, psd = signal.welch(vals, fs=SAMPLING_RATE, nperseg=1024)
        mask = freqs <= 50
        axes[0].semilogy(freqs[mask], psd[mask], label=complexity,
                         color=PALETTE[complexity], linewidth=2)

    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density (µV²/Hz)')
    axes[0].set_title('PSD by UI Complexity (All Participants)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add band shading
    band_colors = ['#3498db', '#9b59b6', '#2ecc71', '#f1c40f', '#e74c3c']
    for (band_name, (fmin, fmax)), color in zip(BANDS.items(), band_colors):
        axes[0].axvspan(fmin, fmax, alpha=0.08, color=color)

    # 2b: Per-participant PSD (just Simple vs Complex)
    participants = task_df['participant'].unique()
    for i, participant in enumerate(participants):
        p_df = task_df[task_df['participant'] == participant]
        for complexity in ['Simple', 'Complex']:
            vals = p_df[p_df['complexity'] == complexity]['value'].values.astype(float)
            if len(vals) < 512:
                continue
            freqs, psd = signal.welch(vals, fs=SAMPLING_RATE, nperseg=1024)
            mask = freqs <= 50
            style = '-' if complexity == 'Simple' else '--'
            axes[1].semilogy(freqs[mask], psd[mask], style, alpha=0.6,
                             label=f'{participant} ({complexity})')

    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (µV²/Hz)')
    axes[1].set_title('PSD: Simple vs Complex (Per Participant)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/2_psd_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 2_psd_comparison.png")


# ==========================================
# ANALYSIS 3: BAND POWER COMPARISON
# ==========================================
def analyze_band_powers(raw_df):
    """Per-band absolute and relative power comparison across complexity levels."""
    print("\n" + "="*60)
    print("ANALYSIS 3: BAND POWER BY COMPLEXITY")
    print("="*60)

    task_df = raw_df[raw_df['phase'] == 'TASK']

    # Segment into 2-second windows and compute band powers
    WINDOW = 2 * SAMPLING_RATE
    records = []

    for participant in task_df['participant'].unique():
        for complexity in LABEL_ORDER:
            subset = task_df[(task_df['participant'] == participant) &
                             (task_df['complexity'] == complexity)]
            values = subset['value'].values.astype(float)

            for start in range(0, len(values) - WINDOW, WINDOW):
                segment = values[start:start + WINDOW]
                powers, _, _ = compute_band_powers(segment)
                record = {
                    'participant': participant,
                    'complexity': complexity,
                }
                record.update(powers)
                records.append(record)

    bp_df = pd.DataFrame(records)

    if bp_df.empty:
        print("  [SKIP] No valid segments found")
        return None

    # Plot: Relative band power (grouped bar chart)
    rel_cols = [c for c in bp_df.columns if '_rel' in c]
    band_names = [c.replace('_rel', '') for c in rel_cols]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 3a: Relative band power
    plot_data = []
    for _, row in bp_df.iterrows():
        for col, name in zip(rel_cols, band_names):
            plot_data.append({
                'Band': name, 'Relative Power': row[col],
                'Complexity': row['complexity']
            })
    plot_df = pd.DataFrame(plot_data)

    sns.barplot(data=plot_df, x='Band', y='Relative Power', hue='Complexity',
                hue_order=LABEL_ORDER, palette=PALETTE, errorbar='se',
                capsize=0.05, ax=axes[0])
    axes[0].set_title('Relative Band Power by UI Complexity', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Relative Power (proportion)')
    axes[0].tick_params(axis='x', rotation=15)

    # 3b: Absolute band power (log scale)
    abs_cols = [c for c in bp_df.columns if c in [b for b in BANDS.keys()]]
    plot_data2 = []
    for _, row in bp_df.iterrows():
        for col in abs_cols:
            plot_data2.append({
                'Band': col, 'Absolute Power': row[col],
                'Complexity': row['complexity']
            })
    plot_df2 = pd.DataFrame(plot_data2)

    sns.barplot(data=plot_df2, x='Band', y='Absolute Power', hue='Complexity',
                hue_order=LABEL_ORDER, palette=PALETTE, errorbar='se',
                capsize=0.05, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Absolute Band Power by UI Complexity (Log Scale)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Absolute Power (µV²)')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/3_band_power.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical tests per band
    print("\n  Band-by-Band Statistical Tests:")
    print(f"  {'Band':<25} {'Kruskal H':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*55}")

    for band in abs_cols:
        groups = [bp_df[bp_df['complexity'] == c][band].values for c in LABEL_ORDER]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups)
            print(f"  {band:<25} {H:>10.3f} {p:>10.4f} {get_sig_stars(p):>5}")

    print("  → Saved: 3_band_power.png")
    return bp_df


# ==========================================
# ANALYSIS 4: EEG COGNITIVE INDICES
# ==========================================
def analyze_eeg_indices(bp_df):
    """Compute and compare EEG-derived cognitive indices."""
    if bp_df is None:
        return

    print("\n" + "="*60)
    print("ANALYSIS 4: EEG COGNITIVE INDICES")
    print("="*60)

    # Compute indices from band powers
    bp_df = bp_df.copy()
    theta_col = 'Theta (4-8 Hz)'
    alpha_col = 'Alpha (8-13 Hz)'
    beta_col  = 'Beta (13-30 Hz)'

    bp_df['Engagement (β/α)'] = bp_df[beta_col] / (bp_df[alpha_col] + 1e-10)
    bp_df['Cognitive Load (θ/β)'] = bp_df[theta_col] / (bp_df[beta_col] + 1e-10)
    bp_df['Task Load (θ/α)'] = bp_df[theta_col] / (bp_df[alpha_col] + 1e-10)

    indices = ['Engagement (β/α)', 'Cognitive Load (θ/β)', 'Task Load (θ/α)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, idx_name in zip(axes, indices):
        # Remove outliers per index using IQR
        Q1 = bp_df[idx_name].quantile(0.05)
        Q3 = bp_df[idx_name].quantile(0.95)
        clean = bp_df[(bp_df[idx_name] >= Q1) & (bp_df[idx_name] <= Q3)]

        sns.boxplot(data=clean, x='complexity', y=idx_name, order=LABEL_ORDER,
                    palette=PALETTE, showfliers=False, ax=ax)
        sns.stripplot(data=clean, x='complexity', y=idx_name, order=LABEL_ORDER,
                      color='black', alpha=0.15, size=2, ax=ax)
        ax.set_title(idx_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('')

        # Stats
        groups = [clean[clean['complexity'] == c][idx_name].values for c in LABEL_ORDER]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups)
            ax.text(0.02, 0.98, f'H={H:.2f}, p={p:.4f} {get_sig_stars(p)}',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('EEG Cognitive Indices by UI Complexity', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/4_eeg_indices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print detailed stats
    print("\n  Index Statistics:")
    for idx_name in indices:
        print(f"\n  {idx_name}:")
        for c in LABEL_ORDER:
            vals = bp_df[bp_df['complexity'] == c][idx_name]
            print(f"    {c:>10}: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

        # Pairwise
        for c1, c2 in combinations(LABEL_ORDER, 2):
            g1 = bp_df[bp_df['complexity'] == c1][idx_name].values
            g2 = bp_df[bp_df['complexity'] == c2][idx_name].values
            if len(g1) > 0 and len(g2) > 0:
                U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                d = cohens_d(g1, g2)
                print(f"    {c1} vs {c2}: p={p:.4f} {get_sig_stars(p)}, d={d:.3f}")

    print("\n  → Saved: 4_eeg_indices.png")


# ==========================================
# ANALYSIS 5: PER-PARTICIPANT HEATMAP
# ==========================================
def analyze_per_participant(bp_df):
    """Per-participant band power comparison."""
    if bp_df is None:
        return

    print("\n" + "="*60)
    print("ANALYSIS 5: PER-PARTICIPANT BREAKDOWN")
    print("="*60)

    abs_cols = [c for c in bp_df.columns if c in BANDS.keys()]
    participants = bp_df['participant'].unique()

    fig, axes = plt.subplots(len(participants), 1, figsize=(14, 4 * len(participants)))
    if len(participants) == 1:
        axes = [axes]

    for ax, participant in zip(axes, participants):
        p_df = bp_df[bp_df['participant'] == participant]

        # Compute mean band powers per complexity
        means = p_df.groupby('complexity')[abs_cols].mean()
        means = means.reindex(LABEL_ORDER)

        # Normalize per band for visualization
        normalized = means.div(means.max(axis=0), axis=1)

        sns.heatmap(normalized, annot=means.round(1), fmt='.1f', cmap='YlOrRd',
                    ax=ax, cbar_kws={'label': 'Normalized Power'})
        ax.set_title(f'Band Power — {participant}', fontsize=13, fontweight='bold')
        ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/5_per_participant.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 5_per_participant.png")


# ==========================================
# ANALYSIS 6: FATIGUE / TIME-ON-TASK
# ==========================================
def analyze_fatigue(raw_df):
    """Fatigue analysis: EEG changes over time + reaction time trend."""
    print("\n" + "="*60)
    print("ANALYSIS 6: FATIGUE / TIME-ON-TASK EFFECTS")
    print("="*60)

    task_df = raw_df[raw_df['phase'] == 'TASK']

    # Get unique trials with reaction time
    trials = task_df.drop_duplicates(subset=['participant', 'image', 'target_instruction']).copy()
    trials = trials[trials['reaction_time'] > 0].sort_values('timestamp')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 6a: Reaction time over experiment (per participant)
    for participant in trials['participant'].unique():
        p_trials = trials[trials['participant'] == participant].copy()
        p_trials['trial_order'] = range(len(p_trials))
        axes[0].scatter(p_trials['trial_order'], p_trials['reaction_time'],
                        alpha=0.3, s=20, label=participant)
        # Trend line
        if len(p_trials) > 2:
            z = np.polyfit(p_trials['trial_order'], p_trials['reaction_time'], 1)
            p = np.poly1d(z)
            axes[0].plot(p_trials['trial_order'], p(p_trials['trial_order']),
                         '--', alpha=0.7, linewidth=2)

    axes[0].set_title('Reaction Time Trend Over Experiment', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Trial Order')
    axes[0].set_ylabel('Reaction Time (s)')
    axes[0].legend(fontsize=9)

    # 6b: Alpha power evolution (fatigue indicator)
    # Split each participant's TASK data into early, middle, late thirds
    alpha_data = []
    for participant in task_df['participant'].unique():
        p_df = task_df[task_df['participant'] == participant]
        values = p_df['value'].values.astype(float)
        n = len(values)
        thirds = [('Early', values[:n//3]),
                  ('Middle', values[n//3:2*n//3]),
                  ('Late', values[2*n//3:])]

        for period, vals in thirds:
            if len(vals) < 512:
                continue
            powers, _, _ = compute_band_powers(vals)
            alpha_data.append({
                'participant': participant,
                'Period': period,
                'Alpha Power': powers['Alpha (8-13 Hz)'],
                'Theta Power': powers['Theta (4-8 Hz)'],
            })

    alpha_df = pd.DataFrame(alpha_data)
    if not alpha_df.empty:
        period_order = ['Early', 'Middle', 'Late']
        sns.barplot(data=alpha_df, x='Period', y='Alpha Power', hue='participant',
                    order=period_order, errorbar=None, ax=axes[1])
        axes[1].set_title('Alpha Power Over Time (Fatigue Indicator)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Alpha Power (µV²)')
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/6_fatigue.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 6_fatigue.png")


# ==========================================
# ANALYSIS 7: NEURAL-BEHAVIORAL CORRELATION
# ==========================================
def analyze_correlation(raw_df):
    """Correlation between EEG features and reaction time."""
    print("\n" + "="*60)
    print("ANALYSIS 7: NEURAL ↔ BEHAVIORAL CORRELATION")
    print("="*60)

    task_df = raw_df[raw_df['phase'] == 'TASK']

    # For each trial, compute EEG features and match with reaction time
    trial_features = []
    grouped = task_df.groupby(['participant', 'image', 'target_instruction'])

    for (participant, image, instruction), group in grouped:
        rt = group['reaction_time'].iloc[0]
        complexity = group['complexity'].iloc[0]
        if rt <= 0 or pd.isna(rt):
            continue

        values = group['value'].values.astype(float)
        if len(values) < 256:
            continue

        powers, _, _ = compute_band_powers(values)

        theta = powers['Theta (4-8 Hz)']
        alpha = powers['Alpha (8-13 Hz)']
        beta = powers['Beta (13-30 Hz)']

        trial_features.append({
            'participant': participant,
            'complexity': complexity,
            'reaction_time': rt,
            'theta_power': theta,
            'alpha_power': alpha,
            'beta_power': beta,
            'engagement': beta / (alpha + 1e-10),
            'cognitive_load': theta / (beta + 1e-10),
            'task_load': theta / (alpha + 1e-10),
        })

    feat_df = pd.DataFrame(trial_features)

    if feat_df.empty:
        print("  [SKIP] No valid trial features computed")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    correlates = [
        ('engagement', 'Engagement (β/α)'),
        ('cognitive_load', 'Cognitive Load (θ/β)'),
        ('task_load', 'Task Load (θ/α)'),
    ]

    for ax, (col, label) in zip(axes, correlates):
        # Remove extreme outliers
        q_low, q_high = feat_df[col].quantile(0.05), feat_df[col].quantile(0.95)
        clean = feat_df[(feat_df[col] >= q_low) & (feat_df[col] <= q_high)]

        for complexity in LABEL_ORDER:
            subset = clean[clean['complexity'] == complexity]
            ax.scatter(subset[col], subset['reaction_time'],
                       c=PALETTE[complexity], label=complexity, alpha=0.5, s=30)

        # Overall correlation
        r, p = stats.spearmanr(clean[col], clean['reaction_time'])
        ax.set_title(f'{label}\nr={r:.3f}, p={p:.4f} {get_sig_stars(p)}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel('Reaction Time (s)')
        ax.legend(fontsize=9)

        # Trend line
        z = np.polyfit(clean[col], clean['reaction_time'], 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(clean[col].min(), clean[col].max(), 50)
        ax.plot(x_range, p_line(x_range), 'k--', alpha=0.5, linewidth=2)

    plt.suptitle('Neural-Behavioral Correlation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/7_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved: 7_correlation.png")


# ==========================================
# ANALYSIS 8: SUMMARY TABLE
# ==========================================
def generate_summary(raw_df):
    """Generate a summary CSV with all key metrics."""
    print("\n" + "="*60)
    print("ANALYSIS 8: GENERATING SUMMARY TABLE")
    print("="*60)

    task_df = raw_df[raw_df['phase'] == 'TASK']
    summary_rows = []

    for participant in task_df['participant'].unique():
        for complexity in LABEL_ORDER:
            subset = task_df[(task_df['participant'] == participant) &
                             (task_df['complexity'] == complexity)]
            if len(subset) == 0:
                continue

            values = subset['value'].values.astype(float)
            rt_vals = subset.drop_duplicates(
                subset=['image', 'target_instruction']
            )['reaction_time']
            rt_vals = rt_vals[rt_vals > 0]

            row = {
                'Participant': participant,
                'Complexity': complexity,
                'N_samples': len(values),
                'N_trials': len(rt_vals),
                'RT_mean': rt_vals.mean() if len(rt_vals) > 0 else np.nan,
                'RT_std': rt_vals.std() if len(rt_vals) > 0 else np.nan,
                'EEG_mean': np.mean(values),
                'EEG_std': np.std(values),
            }

            if len(values) >= 512:
                powers, _, _ = compute_band_powers(values)
                for band_name, power in powers.items():
                    if '_rel' not in band_name:
                        row[band_name] = power

                theta = powers['Theta (4-8 Hz)']
                alpha = powers['Alpha (8-13 Hz)']
                beta = powers['Beta (13-30 Hz)']
                row['Engagement'] = beta / (alpha + 1e-10)
                row['CogLoad'] = theta / (beta + 1e-10)
                row['TaskLoad'] = theta / (alpha + 1e-10)

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = f'{RESULTS_DIR}/analysis_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  → Saved: {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df


# ==========================================
# MAIN
# ==========================================
def main():
    raw_df, power_df = load_all_data()
    if raw_df is None:
        return

    print(f"\nLoaded: {len(raw_df)} raw samples, {len(power_df)} power samples")
    print(f"Participants: {raw_df['participant'].unique()}")
    print(f"TASK raw samples: {len(raw_df[raw_df['phase'] == 'TASK'])}")

    # Run all analyses
    analyze_reaction_time(raw_df)         # 1. Behavioral
    analyze_psd(raw_df)                   # 2. Spectral
    bp_df = analyze_band_powers(raw_df)   # 3. Band powers
    analyze_eeg_indices(bp_df)            # 4. Cognitive indices
    analyze_per_participant(bp_df)        # 5. Individual differences
    analyze_fatigue(raw_df)               # 6. Time-on-task
    analyze_correlation(raw_df)           # 7. Neural ↔ Behavioral
    generate_summary(raw_df)              # 8. Summary table

    print("\n" + "="*60)
    print(f"ALL ANALYSES COMPLETE → {RESULTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()