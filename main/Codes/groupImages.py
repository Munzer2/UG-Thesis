import pandas as pd
import shutil
import os

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = 'Ratings/complexity.csv'
SOURCE_IMG_DIR = 'Images'
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

# STRICTNESS FILTER
# We only accept images where the Standard Deviation is < 2.0
# This removes "controversial" images that confused users.
MAX_STD_DEV = 2.0 

FOLDERS = {
    'simple': 'design_B_simple',    
    'moderate': 'design_C_moderate',
    'complex': 'design_A_complex'  
}

for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

def sort_images_reliably():
    print("--- LOADING DATA ---")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {CSV_FILE}")
        return

    # 1. Calculate Mean AND Standard Deviation
    stats = df.groupby('website')['mean_response'].agg(['mean', 'std']).reset_index()
    
    # 2. Filter out unreliable data
    reliable_sites = stats[stats['std'] < MAX_STD_DEV]
    
    print(f"Total Sites: {len(stats)}")
    print(f"Reliable Sites (High Consensus): {len(reliable_sites)}")
    print(f"Rejected {len(stats) - len(reliable_sites)} sites due to disagreement.")

    # --- NEW: PRINT HIGHEST AND LOWEST SCORES ---
    if not reliable_sites.empty:
        min_score = reliable_sites['mean'].min()
        max_score = reliable_sites['mean'].max()
        
        print("\n" + "="*40)
        print("DATASET EXTREMES (Among Reliable Sites)")
        print("="*40)
        print(f"LOWEST Score (Simplest):   {min_score:.2f} / 9.0")
        print(f"HIGHEST Score (Most Complex): {max_score:.2f} / 9.0")
        
        # Optional: Print the names of the most extreme sites
        simplest_site = reliable_sites.loc[reliable_sites['mean'].idxmin(), 'website']
        complex_site = reliable_sites.loc[reliable_sites['mean'].idxmax(), 'website']
        print(f" -> Simplest Website ID: {simplest_site}")
        print(f" -> Most Complex Website ID: {complex_site}")
        print("="*40 + "\n")
    else:
        print("[ERROR] No reliable sites found. Check your CSV or Std Dev filter.")
        return

    # 3. Sort Images
    counts = {'simple': 0, 'moderate': 0, 'complex': 0}

    for index, row in reliable_sites.iterrows():
        csv_name = str(row['website'])
        score = row['mean'] 
        
        # Categorize (Using your Extreme Logic)
        target_folder = None
        category = ""
        
        # Simple: 1.0 to 2.5
        if 1.0 <= score < 2.5:
            target_folder = FOLDERS['simple']
            category = 'simple'
        # Complex: 6 to 9.0
        elif 6 <= score <= 9.0:
            target_folder = FOLDERS['complex']
            category = 'complex'
        # Moderate: Everything else (2.5 to 6)
        else:
            target_folder = FOLDERS['moderate']
            category = 'moderate'

        # Find File (Smart Match)
        search_name = csv_name.replace("english_", "")
        found_src = None
        for ext in IMAGE_EXTENSIONS:
            test_path = os.path.join(SOURCE_IMG_DIR, search_name + ext)
            if os.path.exists(test_path):
                found_src = test_path
                break
        
        if found_src:
            filename = os.path.basename(found_src)
            dst_path = os.path.join(target_folder, filename)
            shutil.copy2(found_src, dst_path)
            counts[category] += 1

    print("--- SORTING COMPLETE ---")
    print(f"Simple (1.0 - 2.5):   {counts['simple']} images")
    print(f"Moderate (2.5 - 7.5): {counts['moderate']} images")
    print(f"Complex (7.5 - 9.0):  {counts['complex']} images")

if __name__ == "__main__":
    sort_images_reliably()