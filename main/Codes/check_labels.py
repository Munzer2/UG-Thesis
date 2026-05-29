"""Extract unique image -> label mappings from all participant CSVs for cross-checking."""
import pandas as pd
import glob
import os

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# Correct mapping based on complexitymodel tertile splits
CORRECT = {
    '0.png': ('design_A_simple', 4.23),
    '1.png': ('design_B_complex', 5.81),
    '3.png': ('design_A_simple', 2.94),
    '7.png': ('design_B_complex', 7.43),
    '8.png': ('design_B_complex', 5.91),
    '9.png': ('design_B_complex', 5.62),
    '10.png': ('design_B_complex', 5.61),
    '11.png': ('design_A_simple', 2.85),
    '12.png': ('design_B_complex', 6.43),
    '13.png': ('design_B_complex', 5.80),
    '14.png': ('design_A_simple', 3.94),
    '15.png': ('design_C_moderate', 5.12),
    '16.png': ('design_B_complex', 6.77),
    '17.png': ('design_C_moderate', 4.62),
    '29.png': ('design_B_complex', 5.80),
    '38.png': ('design_B_complex', 5.72),
    '45.png': ('design_A_simple', 4.24),
    '56.png': ('design_B_complex', 7.97),
    '68.png': ('design_C_moderate', 4.91),
    '75.png': ('design_B_complex', 6.10),
    '79.png': ('design_A_simple', 3.24),
    '87.png': ('design_C_moderate', 5.02),
    '105.png': ('design_B_complex', 7.42),
    '116.png': ('design_C_moderate', 5.14),
    '120.png': ('design_A_simple', 3.27),
    '124.png': ('design_A_simple', 3.81),
    '138.png': ('design_B_complex', 7.57),
    '142.png': ('design_A_simple', 4.14),
    '222.png': ('design_B_complex', 8.35),
    '223.png': ('design_C_moderate', 4.68),
    '230.png': ('design_A_simple', 4.18),
    '236.png': ('design_B_complex', 9.85),
    '242.png': ('design_C_moderate', 4.75),
    '246.png': ('design_A_simple', 1.38),
    '250.png': ('design_C_moderate', 4.87),
    '269.png': ('design_C_moderate', 4.64),
    '272.png': ('design_A_simple', 3.46),
    '286.png': ('design_B_complex', 5.40),
    '298.png': ('design_B_complex', 5.41),
    '299.png': ('design_A_simple', 3.81),
    '302.png': ('design_A_simple', 3.05),
    '308.png': ('design_C_moderate', 4.58),
    '309.png': ('design_B_complex', 6.18),
    '323.png': ('design_B_complex', 6.49),
    '334.png': ('design_A_simple', 2.54),
    '339.png': ('design_A_simple', 3.20),
    '346.png': ('design_A_simple', 4.05),
}

files = sorted(glob.glob(os.path.join(DATASET_DIR, 'UI_Exp_*.csv')))
print(f"Scanning {len(files)} participant CSVs...\n")

# Collect all unique (image, label) pairs across all files
all_mappings = {}  # image -> set of labels found
per_file = {}      # file -> {image: label}

for f in files:
    fname = os.path.basename(f)
    df = pd.read_csv(f, low_memory=False)
    
    # Get rows with non-empty image and label
    labeled = df[df['image'].notna() & (df['image'] != '') & df['label'].notna() & (df['label'] != '')]
    pairs = labeled[['image', 'label']].drop_duplicates()
    
    file_map = {}
    for _, row in pairs.iterrows():
        img = str(row['image'])
        lbl = str(row['label'])
        file_map[img] = lbl
        
        if img not in all_mappings:
            all_mappings[img] = set()
        all_mappings[img].add(lbl)
    
    per_file[fname] = file_map

# Print the consolidated mapping
print("=" * 80)
print("  IMAGE -> LABEL MAPPING (from all participant CSVs after relabeling)")
print("  Tertile boundaries: Simple <= 4.25 | 4.25 < Moderate <= 5.24 | Complex > 5.24")
print("=" * 80)
print(f"{'Image':<12} {'Label in CSVs':<25} {'Expected Label':<25} {'Score':>6} {'Match':>6}")
print("-" * 80)

errors = []
for img in sorted(all_mappings.keys(), key=lambda x: int(x.replace('.png', '')) if x.replace('.png', '').isdigit() else 999):
    labels_found = all_mappings[img]
    expected_label, score = CORRECT.get(img, ('UNKNOWN', 0))
    
    if len(labels_found) == 1:
        actual = list(labels_found)[0]
        match = 'OK' if actual == expected_label else 'WRONG'
    else:
        actual = ' / '.join(sorted(labels_found))
        match = 'MIXED'
    
    print(f"{img:<12} {actual:<25} {expected_label:<25} {score:>6.2f} {match:>6}")
    
    if match != 'OK':
        errors.append((img, actual, expected_label, score))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Unique images found across all CSVs: {len(all_mappings)}")
print(f"  Images with correct label:  {len(all_mappings) - len(errors)}")
print(f"  Images with wrong/mixed label: {len(errors)}")

if errors:
    print("\n  PROBLEMS:")
    for img, actual, expected, score in errors:
        print(f"    {img}: found '{actual}', expected '{expected}' (score={score:.2f})")
    
    # Show which files have the issue
    print("\n  AFFECTED FILES:")
    for img, actual, expected, score in errors:
        for fname, fmap in per_file.items():
            if img in fmap and fmap[img] != expected:
                print(f"    {fname}: {img} -> '{fmap[img]}' (should be '{expected}')")
else:
    print("\n  ALL IMAGES HAVE CORRECT LABELS ACROSS ALL FILES!")
