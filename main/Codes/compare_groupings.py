"""Compare how current images score on both mean_response (appeal) and complexitymodel."""
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(BASE_DIR, 'Ratings', 'complexity.csv')

FOLDERS = {
    'design_A_simple': os.path.join(BASE_DIR, 'design_A_simple'),
    'design_B_complex': os.path.join(BASE_DIR, 'design_B_complex'),
    'design_C_moderate': os.path.join(BASE_DIR, 'design_C_moderate'),
}

df = pd.read_csv(CSV_FILE, low_memory=False)

# Get per-website stats for BOTH measures
site_stats = df.groupby('website').agg({
    'mean_response': 'mean',
    'complexitymodel': 'first',  # Same for all rows of same website
}).reset_index()

# Convert complexitymodel to numeric
site_stats['complexitymodel'] = pd.to_numeric(site_stats['complexitymodel'], errors='coerce')

print("="*80)
print("COMPARISON: Visual Appeal (mean_response) vs Complexity Model (complexitymodel)")
print("="*80)

all_rows = []
for folder_name, folder_path in FOLDERS.items():
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for f in sorted(files, key=lambda x: int(x.replace('.png', ''))):
        img_id = f.replace('.png', '')
        match = site_stats[site_stats['website'] == f'english_{img_id}']
        if match.empty:
            match = site_stats[site_stats['website'] == img_id]
        if not match.empty:
            all_rows.append({
                'folder': folder_name,
                'image': f,
                'appeal': match['mean_response'].values[0],
                'complexity': match['complexitymodel'].values[0],
            })

result_df = pd.DataFrame(all_rows)

for folder in ['design_A_simple', 'design_B_complex', 'design_C_moderate']:
    fdf = result_df[result_df['folder'] == folder]
    print(f"\n--- {folder} ---")
    print(f"{'Image':<12} {'Appeal':>8} {'Complexity':>12}")
    print("-" * 35)
    for _, row in fdf.sort_values('appeal').iterrows():
        print(f"{row['image']:<12} {row['appeal']:>8.2f} {row['complexity']:>12.2f}")
    print(f"{'AVERAGE':<12} {fdf['appeal'].mean():>8.2f} {fdf['complexity'].mean():>12.2f}")

# Correlation between the two measures across ALL images
print(f"\n{'='*80}")
print("CORRELATION between Appeal and Complexity across all images:")
valid = result_df.dropna()
from scipy.stats import pearsonr, spearmanr
r_p, p_p = pearsonr(valid['appeal'], valid['complexity'])
r_s, p_s = spearmanr(valid['appeal'], valid['complexity'])
print(f"  Pearson:  r = {r_p:.3f}, p = {p_p:.4f}")
print(f"  Spearman: r = {r_s:.3f}, p = {p_s:.4f}")

# Check separation quality for both measures
print(f"\n{'='*80}")
print("GROUP SEPARATION QUALITY:")
a = result_df[result_df['folder'] == 'design_A_simple']
b = result_df[result_df['folder'] == 'design_B_complex']

from scipy.stats import mannwhitneyu
# Appeal separation
u, p = mannwhitneyu(a['appeal'], b['appeal'])
print(f"\n  Appeal (A_simple vs B_complex):")
print(f"    A_simple avg: {a['appeal'].mean():.2f}  |  B_complex avg: {b['appeal'].mean():.2f}")
print(f"    Mann-Whitney U={u:.0f}, p={p:.6f}")
cohens_d = (a['appeal'].mean() - b['appeal'].mean()) / np.sqrt((a['appeal'].std()**2 + b['appeal'].std()**2)/2)
print(f"    Cohen's d = {cohens_d:.2f}")

# Complexity separation
a_c = a['complexity'].dropna()
b_c = b['complexity'].dropna()
if len(a_c) > 0 and len(b_c) > 0:
    u, p = mannwhitneyu(a_c, b_c)
    print(f"\n  Complexity Model (A_simple vs B_complex):")
    print(f"    A_simple avg: {a_c.mean():.2f}  |  B_complex avg: {b_c.mean():.2f}")
    print(f"    Mann-Whitney U={u:.0f}, p={p:.6f}")
    cohens_d = (a_c.mean() - b_c.mean()) / np.sqrt((a_c.std()**2 + b_c.std()**2)/2)
    print(f"    Cohen's d = {cohens_d:.2f}")
