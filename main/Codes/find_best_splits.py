import pandas as pd
import numpy as np
import os

CSV_FILE = os.path.join('Ratings', 'complexity.csv')
df = pd.read_csv(CSV_FILE, low_memory=False)

# Clean complexity model values
df['complexitymodel'] = pd.to_numeric(df['complexitymodel'], errors='coerce')
valid_complexity = df.dropna(subset=['complexitymodel'])

# 1. Dataset stats
print("="*60)
print("1. GLOBAL DATASET RANGE (`complexitymodel`)")
print("="*60)
c_data = valid_complexity['complexitymodel']
print(f"Overall Datapoints: {len(c_data)}")
print(f"Min:   {c_data.min():.2f}")
print(f"Max:   {c_data.max():.2f}")
print(f"Mean:  {c_data.mean():.2f}")
print(f"StDev: {c_data.std():.2f}")
print("")

p33 = c_data.quantile(0.33)
p67 = c_data.quantile(0.67)
print(f"33rd Percentile (Tertile 1 boundary): {p33:.2f}")
print(f"67th Percentile (Tertile 2 boundary): {p67:.2f}")
print("")
print("BEST SPLIT STRATEGY (Tertiles):")
print(f"  Simple:   X <= {p33:.2f}")
print(f"  Moderate: {p33:.2f} < X <= {p67:.2f}")
print(f"  Complex:  X > {p67:.2f}")

# 2. Re-evaluate the 47 currently chosen images
print("\n" + "="*60)
print("2. EVALUATING CURRENT IMAGES UNDER THESE TERTILES")
print("="*60)

FOLDERS = {
    'design_A_simple': 'Simple',
    'design_B_complex': 'Complex',
    'design_C_moderate': 'Moderate',
}

# Get per-website stats for complexity
site_stats = valid_complexity.groupby('website')['complexitymodel'].first().reset_index()

current_images = []
for folder_name, class_name in FOLDERS.items():
    folder_path = folder_name # running in main
    if not os.path.exists(folder_path): continue
    for f in os.listdir(folder_path):
        if not f.endswith('.png'): continue
        
        img_id = f.replace('.png', '')
        
        match = site_stats[site_stats['website'] == f'english_{img_id}']
        if match.empty:
            match = site_stats[site_stats['website'] == img_id]
            
        if not match.empty:
            c_val = match['complexitymodel'].values[0]
            
            # Determine correct class based on tertiles
            if c_val <= p33:
                correct_class = 'Simple'
            elif c_val <= p67:
                correct_class = 'Moderate'
            else:
                correct_class = 'Complex'
                
            current_images.append({
                'image': f,
                'complexity': c_val,
                'current_class': class_name,
                'correct_class': correct_class,
                'is_wrong': class_name != correct_class
            })

res_df = pd.DataFrame(current_images)

wrong_count = res_df['is_wrong'].sum()
print(f"Total current images analyzed: {len(res_df)}")
print(f"Total misclassified images: {wrong_count} ({(wrong_count/len(res_df))*100:.1f}%)")

print("\n--- SHIFTS REQUIRED ---")
for _, row in res_df[res_df['is_wrong']].sort_values(['current_class', 'correct_class']).iterrows():
    print(f"Image {row['image']:>7}: Complexity {row['complexity']:>5.2f} | Moves from {row['current_class']:>8} -> {row['correct_class']:>8}")

print("\n--- NEW CLASS COUNTS (If you re-sorted only these 47 images) ---")
print(res_df['correct_class'].value_counts())
