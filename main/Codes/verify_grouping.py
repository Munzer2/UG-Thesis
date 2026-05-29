"""Quick verification: Check mean_response scores for images in each design folder."""
import pandas as pd
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

# Get per-website mean scores
site_stats = df.groupby('website')['mean_response'].agg(['mean', 'std', 'count']).reset_index()

for folder_name, folder_path in FOLDERS.items():
    print(f"\n{'='*60}")
    print(f"  {folder_name}")
    print(f"{'='*60}")
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    scores = []
    
    for f in sorted(files, key=lambda x: int(x.replace('.png', ''))):
        img_id = f.replace('.png', '')
        # Try matching with english_ prefix (as groupImages.py strips it)
        match = site_stats[site_stats['website'].str.replace('english_', '') == img_id]
        if match.empty:
            match = site_stats[site_stats['website'] == img_id]
        if match.empty:
            match = site_stats[site_stats['website'] == f'english_{img_id}']
        
        if not match.empty:
            score = match['mean'].values[0]
            std = match['std'].values[0]
            n = match['count'].values[0]
            scores.append(score)
            print(f"  {f:<12s}  mean_response = {score:.2f}  (std={std:.2f}, n={n})")
        else:
            print(f"  {f:<12s}  NOT FOUND in CSV")
    
    if scores:
        print(f"  -------")
        print(f"  Folder avg: {sum(scores)/len(scores):.2f}  |  min: {min(scores):.2f}  |  max: {max(scores):.2f}  |  count: {len(scores)}")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print("  mean_response = visual APPEAL rating (1-9 scale)")
print("  LOW score  = unappealing/ugly website")
print("  HIGH score = attractive/beautiful website")
print(f"{'='*60}")
