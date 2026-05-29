"""Compare effect size vs statistical significance for Low Theta and Alpha Ratio."""
import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal, mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
df = pd.read_csv(os.path.join(BASE_DIR, "results_analysis", "subband_analysis.csv"))

feat = 'Alpha Ratio (H/L)'
s_vals = df[df.Complexity=='Simple'][feat].dropna().values
m_vals = df[df.Complexity=='Moderate'][feat].dropna().values
c_vals = df[df.Complexity=='Complex'][feat].dropna().values

print('=== Alpha Ratio (H/L) - Why the 3-class test is significant ===')
print(f'  Simple mean:    {np.mean(s_vals):.4f}  (n={len(s_vals)})')
print(f'  Moderate mean:  {np.mean(m_vals):.4f}  (n={len(m_vals)})')
print(f'  Complex mean:   {np.mean(c_vals):.4f}  (n={len(c_vals)})')
print(f'  Simple median:  {np.median(s_vals):.4f}')
print(f'  Moderate median:{np.median(m_vals):.4f}')
print(f'  Complex median: {np.median(c_vals):.4f}')
print()

# 3-class test
h, p = kruskal(s_vals, m_vals, c_vals)
print(f'  Kruskal-Wallis (S vs M vs C):  H={h:.2f}, p={p:.4f}')

# Pairwise tests
u1, p1 = mannwhitneyu(s_vals, m_vals, alternative='two-sided')
u2, p2 = mannwhitneyu(s_vals, c_vals, alternative='two-sided')
u3, p3 = mannwhitneyu(m_vals, c_vals, alternative='two-sided')
print(f'  Simple vs Moderate:   U={u1:.0f}, p={p1:.4f}')
print(f'  Simple vs Complex:    U={u2:.0f}, p={p2:.4f}')
print(f'  Moderate vs Complex:  U={u3:.0f}, p={p3:.4f}')

mod_to_comp = ((np.mean(c_vals) - np.mean(m_vals)) / np.mean(m_vals)) * 100
sim_to_mod = ((np.mean(m_vals) - np.mean(s_vals)) / np.mean(s_vals)) * 100
print()
print(f'PATTERN: Moderate=HIGHEST ({np.mean(m_vals):.4f}), drops to Complex ({np.mean(c_vals):.4f})')
print(f'  Moderate-to-Complex drop: {mod_to_comp:.1f}%')
print(f'  Simple-to-Moderate rise:  {sim_to_mod:+.1f}%')

print()
print('=== Now compare: Low Theta ===')
feat2 = 'Low Theta'
s2 = df[df.Complexity=='Simple'][feat2].dropna().values
m2 = df[df.Complexity=='Moderate'][feat2].dropna().values
c2 = df[df.Complexity=='Complex'][feat2].dropna().values
print(f'  Simple mean:    {np.mean(s2):.2f}  (n={len(s2)})')
print(f'  Moderate mean:  {np.mean(m2):.2f}  (n={len(m2)})')
print(f'  Complex mean:   {np.mean(c2):.2f}  (n={len(c2)})')
h2, p2k = kruskal(s2, m2, c2)
print(f'  Kruskal-Wallis: H={h2:.2f}, p={p2k:.4f}')
print(f'  Simple CV:      {np.std(s2)/np.mean(s2)*100:.0f}%')
print(f'  Complex CV:     {np.std(c2)/np.mean(c2)*100:.0f}%')

# Pairwise for Low Theta
u1t, p1t = mannwhitneyu(s2, m2, alternative='two-sided')
u2t, p2t = mannwhitneyu(s2, c2, alternative='two-sided')
u3t, p3t = mannwhitneyu(m2, c2, alternative='two-sided')
print(f'  Simple vs Moderate:   U={u1t:.0f}, p={p1t:.4f}')
print(f'  Simple vs Complex:    U={u2t:.0f}, p={p2t:.4f}')
print(f'  Moderate vs Complex:  U={u3t:.0f}, p={p3t:.4f}')

# Per-participant consistency
binary = df[df.Complexity.isin(['Simple','Complex'])]
participants = sorted(binary.Participant.unique())

print()
print('='*80)
print('PER-PARTICIPANT COMPARISON: Effect size vs Consistency')
print('='*80)
header = f"{'Feature':<20s} | {'Spread':<25s} | {'StdDev':<8s} | {'Consistency':<15s}"
print(header)
print('-'*75)

for feat_name in ['Low Theta', 'High Theta', 'Mid Theta', 'Low Alpha',
                   'Alpha Ratio (H/L)', 'Mid Beta']:
    diffs = []
    n_exp = 0
    n_tot = 0
    for part in participants:
        pdf = binary[binary.Participant == part]
        sv = pdf[pdf.Complexity=='Simple'][feat_name].mean()
        cv = pdf[pdf.Complexity=='Complex'][feat_name].mean()
        if pd.isna(sv) or pd.isna(cv):
            continue
        n_tot += 1
        pct = ((cv - sv) / (abs(sv) + 1e-10)) * 100
        diffs.append(pct)
        if feat_name == 'Mid Beta':
            if cv > sv: n_exp += 1  # expected: increase
        else:
            if cv < sv: n_exp += 1  # expected: decrease
    diffs = np.array(diffs)
    d_min = diffs.min()
    d_max = diffs.max()
    spread_str = f"{d_min:+.0f}% to {d_max:+.0f}%"
    cons_str = f"{n_exp}/{n_tot} ({n_exp/n_tot*100:.0f}%)"
    print(f"{feat_name:<20s} | {spread_str:<25s} | {diffs.std():<8.1f} | {cons_str:<15s}")

print()
print("="*80)
print("TRIAL-LEVEL ANALYSIS: Distribution overlap")
print("="*80)
for feat_name in ['Low Theta', 'Alpha Ratio (H/L)']:
    s_v = binary[binary.Complexity=='Simple'][feat_name].dropna().values
    c_v = binary[binary.Complexity=='Complex'][feat_name].dropna().values
    # Cohen's d
    pooled_std = np.sqrt((np.std(s_v)**2 + np.std(c_v)**2) / 2)
    cohens_d = (np.mean(s_v) - np.mean(c_v)) / pooled_std
    print(f"  {feat_name}:")
    print(f"    Cohen's d = {cohens_d:.3f}")
    print(f"    Simple:  mean={np.mean(s_v):.3f}, std={np.std(s_v):.3f}, IQR=[{np.percentile(s_v,25):.3f}, {np.percentile(s_v,75):.3f}]")
    print(f"    Complex: mean={np.mean(c_v):.3f}, std={np.std(c_v):.3f}, IQR=[{np.percentile(c_v,25):.3f}, {np.percentile(c_v,75):.3f}]")
    print()
