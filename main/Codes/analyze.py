import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats

# ==========================================
# CONFIGURATION
# ==========================================
# POINT TO THE CLEAN DATASET
DATA_DIR = os.path.join("..", "dataset_clean") 
FILE_PATTERN = os.path.join(DATA_DIR, "UI_Exp_*.csv")
RESULTS_DIR = os.path.join("..", "results_analysis")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_clean_data():
    """Loads clean files and removes artifacts safely."""
    files = glob.glob(FILE_PATTERN)
    if not files:
        print(f"[ERROR] No files found in {DATA_DIR}. Did you run preprocessing?")
        return None

    print(f"Loading {len(files)} clean files...")
    df_list = []
    
    for f in files:
        try:
            temp_df = pd.read_csv(f)
            temp_df['source_file'] = os.path.basename(f)
            
            if 'is_artifact' in temp_df.columns:
                temp_df['is_artifact'] = temp_df['is_artifact'].fillna(False).astype(bool)
                
                # Filter out the marked artifacts
                temp_df = temp_df[temp_df['is_artifact'] == False]
            
            df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list: return None
    df = pd.concat(df_list, ignore_index=True)
    return df

def get_sig_text(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def analyze_all(df):
    # Filter for TASK phase only
    task_df = df[df['phase'] == 'TASK'].copy()
    print(f"Total Clean Task Samples: {len(task_df)}")

    # ==========================================
    # 1. BEHAVIORAL (Reaction Time)
    # ==========================================
    trials = task_df.drop_duplicates(subset=['subject_id', 'image', 'target_instruction'])
    
    plt.figure(figsize=(8, 6))
    # FIX 2: Added hue='label' and legend=False
    sns.barplot(data=trials, x='label', y='reaction_time', hue='label', palette="viridis", errorbar='sd', capsize=0.1, legend=False)
    plt.title("Reaction Time (Clean Data)")
    plt.ylabel("Time (s)")
    plt.savefig(f"{RESULTS_DIR}/1_Reaction_Time.png")
    plt.close()

    # ==========================================
    # 2. EEG: ENGAGEMENT INDEX (Beta / Alpha)
    # ==========================================
    
    power_df = task_df[task_df['type'] == 'power'].copy()
    
    if power_df.empty:
        print("[WARNING] No power data found (Did you delete it?). Skipping EEG plots.")
        return

    # Calculate Engagement (Thinking vs Idling)
    # We add 1e-6 to avoid division by zero
    power_df['Engagement'] = power_df['beta'] / (power_df['alpha'] + 1e-6)
    
    # Remove Outliers (Standard IQR method)
    Q1 = power_df['Engagement'].quantile(0.25)
    Q3 = power_df['Engagement'].quantile(0.75)
    IQR = Q3 - Q1
    clean_power = power_df[
        (power_df['Engagement'] >= Q1 - 1.5 * IQR) & 
        (power_df['Engagement'] <= Q3 + 1.5 * IQR)
    ]
    
    # Stats: T-Test Complex vs Simple
    simple = clean_power[clean_power['label'] == 'design_B_simple']['Engagement']
    complex_ = clean_power[clean_power['label'] == 'design_A_complex']['Engagement']
    
    if len(simple) > 0 and len(complex_) > 0:
        t, p = stats.ttest_ind(simple, complex_, equal_var=False)
        print(f"T-Test (Engagement Simple vs Complex): t={t:.2f}, p={p:.4f} {get_sig_text(p)}")

    # Plot
    plt.figure(figsize=(8, 6))
    # FIX 3: Added hue='label' and legend=False
    sns.boxplot(data=clean_power, x='label', y='Engagement', hue='label', palette="coolwarm", showfliers=False, legend=False)
    plt.title("Visual Engagement Index (Beta / Alpha)")
    plt.ylabel("Engagement (Higher = More Focus)")
    plt.savefig(f"{RESULTS_DIR}/2_Engagement_Index.png")
    plt.close()

    # ==========================================
    # 3. FATIGUE CHECK (Time on Task)
    # ==========================================
    trials = trials.sort_values('timestamp')
    trials['trial_order'] = range(len(trials))
    
    plt.figure(figsize=(10, 5))
    sns.regplot(data=trials, x='trial_order', y='reaction_time', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title("Fatigue Check: Reaction Time over Experiment")
    plt.savefig(f"{RESULTS_DIR}/3_Fatigue_Check.png")
    plt.close()

    print(f"\n[DONE] Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    df = load_clean_data()
    if df is not None:
        analyze_all(df)