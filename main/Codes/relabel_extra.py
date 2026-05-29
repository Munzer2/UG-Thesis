"""Relabel the 4 extra mislabeled images in participant CSVs."""
import pandas as pd
import glob
import os

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

RELABEL_MAP = {
    '4.png':   'design_A_simple',    # complexity 2.97 <= 4.25
    '311.png': 'design_C_moderate',  # complexity 4.69, 4.25 < x <= 5.24
    '312.png': 'design_A_simple',    # complexity 3.89 <= 4.25
    '313.png': 'design_A_simple',    # complexity 3.50 <= 4.25
}

files = sorted(glob.glob(os.path.join(DATASET_DIR, 'UI_Exp_*.csv')))
print(f"Scanning {len(files)} CSVs for images: {list(RELABEL_MAP.keys())}\n")

total_changes = 0
for f in files:
    fname = os.path.basename(f)
    df = pd.read_csv(f, low_memory=False)
    
    changed = 0
    for img_file, correct_label in RELABEL_MAP.items():
        mask = (df['image'] == img_file) & (df['label'] != correct_label) & df['image'].notna()
        n = mask.sum()
        if n > 0:
            old = df.loc[mask, 'label'].unique()
            df.loc[mask, 'label'] = correct_label
            changed += n
            print(f"  {fname}: {img_file} ({n} rows) {list(old)} -> {correct_label}")
    
    if changed > 0:
        df.to_csv(f, index=False)
        total_changes += changed

print(f"\nDONE. {total_changes} rows relabeled.")
