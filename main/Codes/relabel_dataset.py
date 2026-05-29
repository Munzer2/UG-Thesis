"""
Relabel participant CSV files based on correct complexity-model tertile grouping.

The 'label' column contains folder names like 'design_A_simple', 'design_B_complex', 'design_C_moderate'.
Images that moved between classes need their labels updated.

Image -> correct label mapping based on complexitymodel tertile splits (<=4.25, <=5.24, >5.24):
"""
import pandas as pd
import glob
import os

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# Images that changed class. Map: filename -> (old_label, new_label)
# From the 19 misclassified images we identified earlier:
RELABEL_MAP = {
    # Was in Complex (design_B_complex), now Simple (design_A_simple)
    '0.png':   'design_A_simple',
    # Was in Complex (design_B_complex), now Moderate (design_C_moderate)
    '68.png':  'design_C_moderate',
    '87.png':  'design_C_moderate',
    '116.png': 'design_C_moderate',
    '223.png': 'design_C_moderate',
    '308.png': 'design_C_moderate',
    # Was in Moderate (design_C_moderate), now Complex (design_B_complex)
    '1.png':   'design_B_complex',
    '7.png':   'design_B_complex',
    '8.png':   'design_B_complex',
    '9.png':   'design_B_complex',
    '10.png':  'design_B_complex',
    '13.png':  'design_B_complex',
    '29.png':  'design_B_complex',
    '286.png': 'design_B_complex',
    # Was in Moderate (design_C_moderate), now Simple (design_A_simple)
    '14.png':  'design_A_simple',
    # Was in Simple (design_A_simple), now Complex (design_B_complex)
    '298.png': 'design_B_complex',
    # Was in Simple (design_A_simple), now Moderate (design_C_moderate)
    '242.png': 'design_C_moderate',
    '250.png': 'design_C_moderate',
    '269.png': 'design_C_moderate',
}

# ---- PHASE 1: SCAN ----
files = sorted(glob.glob(os.path.join(DATASET_DIR, 'UI_Exp_*.csv')))
print(f"Found {len(files)} participant CSVs in {DATASET_DIR}\n")

total_changes = 0
file_changes = {}

for f in files:
    fname = os.path.basename(f)
    df = pd.read_csv(f, low_memory=False)
    
    # Find rows that reference moved images
    changes = 0
    for img_file, correct_label in RELABEL_MAP.items():
        mask = (df['image'] == img_file) & (df['label'] != correct_label) & (df['image'].notna())
        n = mask.sum()
        if n > 0:
            old_labels = df.loc[mask, 'label'].unique()
            print(f"  {fname}: {img_file} -> {n} rows: {list(old_labels)} -> {correct_label}")
            changes += n
    
    if changes > 0:
        file_changes[f] = changes
        total_changes += changes

print(f"\n{'='*60}")
print(f"SCAN COMPLETE")
print(f"{'='*60}")
print(f"  Files affected:  {len(file_changes)} / {len(files)}")
print(f"  Total rows to relabel: {total_changes}")

# ---- PHASE 2: APPLY ----
if total_changes > 0:
    confirm = input(f"\nApply {total_changes} label changes across {len(file_changes)} files? (yes/no): ").strip().lower()
    if confirm == 'yes':
        for f in file_changes:
            fname = os.path.basename(f)
            df = pd.read_csv(f, low_memory=False)
            
            changed = 0
            for img_file, correct_label in RELABEL_MAP.items():
                mask = (df['image'] == img_file) & (df['label'] != correct_label) & (df['image'].notna())
                n = mask.sum()
                if n > 0:
                    df.loc[mask, 'label'] = correct_label
                    changed += n
            
            df.to_csv(f, index=False)
            print(f"  [SAVED] {fname}: {changed} rows relabeled")
        
        print(f"\nDONE. {total_changes} rows relabeled across {len(file_changes)} files.")
    else:
        print("Aborted. No changes made.")
else:
    print("\nNo relabeling needed!")
