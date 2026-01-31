import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
FILE_PATTERN = "UI_Exp_*.csv"
RESULTS_DIR = "figures"  

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_data():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print("No CSV files found!")
        return None

    print(f"Loading {len(files)} files...")
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        temp_df['source_file'] = os.path.basename(f)
        df_list.append(temp_df)
    
    df = pd.concat(df_list, ignore_index=True)
    return df

def analyze_all():
    df = load_data()
    if df is None: return

    # Filter for TASK phase only
    task_df = df[df['phase'] == 'TASK'].copy()
    
    print("\n" + "="*40)
    print("STARTING ANALYSIS")
    print("="*40)

    # ==========================================
    # 1. BEHAVIORAL: REACTION TIME
    # ==========================================
    print("Generating 1. Reaction Time Analysis...")
    trials = task_df.drop_duplicates(subset=['subject_id', 'image', 'target_instruction'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=trials, x='label', y='reaction_time', palette="viridis", errorbar='sd')
    plt.title("Reaction Time by Complexity")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Website Category")
    plt.savefig(f"{RESULTS_DIR}/1_Reaction_Time.png")
    plt.close()

    # ==========================================
    # 2. SIGNAL QUALITY CHECK
    # ==========================================
    print("Generating 2. Signal Quality Check...")
    raw_df = task_df[task_df['type'] == 'raw']
    if not raw_df.empty:
        sample_trial = raw_df['image'].unique()[0]
        sample_data = raw_df[raw_df['image'] == sample_trial].iloc[:2000]
        
        plt.figure(figsize=(12, 4))
        plt.plot(sample_data['timestamp'], sample_data['value'], color='black', linewidth=0.5)
        plt.axhline(y=200, color='r', linestyle='--')
        plt.axhline(y=-200, color='r', linestyle='--')
        plt.title(f"Raw Signal Sample ({sample_trial})")
        plt.savefig(f"{RESULTS_DIR}/2_Signal_Quality.png")
        plt.close()

    # ==========================================
    # 3. COGNITIVE LOAD (THETA/ALPHA)
    # ==========================================
    print("Generating 3. Cognitive Load Ratio...")
    power_df = task_df[task_df['type'] == 'power'].copy()
    
    if not power_df.empty:
        # Calculate Ratio
        power_df['theta_alpha_ratio'] = power_df['theta'] / (power_df['alpha'] + 1)
        
        # Clean Outliers (Interquartile Range)
        Q1 = power_df['theta_alpha_ratio'].quantile(0.25)
        Q3 = power_df['theta_alpha_ratio'].quantile(0.75)
        IQR = Q3 - Q1
        clean_data = power_df[
            (power_df['theta_alpha_ratio'] >= Q1 - 1.5 * IQR) & 
            (power_df['theta_alpha_ratio'] <= Q3 + 1.5 * IQR)
        ]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=clean_data, x='label', y='theta_alpha_ratio', palette="coolwarm", showfliers=False)
        plt.title("Cognitive Load (Theta/Alpha Ratio)")
        plt.ylabel("Workload Index")
        plt.savefig(f"{RESULTS_DIR}/3_Cognitive_Load_Ratio.png")
        plt.close()

        # ==========================================
        # 4: BAND POWER BREAKDOWN
        # ==========================================
        print("Generating 4. Band Power Breakdown...")
        possible_bands = [
            'delta', 'theta', 'alpha', 'beta', 'gamma',  # Simplified names
            'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma' # Raw names
        ]
        
        # 2. Filter: Only keep the bands that ACTUALLY exist in your CSV
        existing_bands = [b for b in possible_bands if b in clean_data.columns]
        
        if existing_bands:
            print(f"   -> Found bands: {existing_bands}")
            
            # Melt only the columns that exist
            melted_power = clean_data.melt(
                id_vars=['label'], 
                value_vars=existing_bands, 
                var_name='Band', 
                value_name='Power'
            )
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_power, x='Band', y='Power', hue='label', palette="magma")
            plt.title("Brainwave Power Spectrum: Simple vs Complex")
            plt.ylabel("Raw Power")
            plt.savefig(f"{RESULTS_DIR}/4_Band_Power_Comparison.png")
            plt.close()
        else:
            print("   -> [SKIP] No band columns found! Check your CSV column names.")
            print(f"   -> Your columns are: {clean_data.columns.tolist()}")

        # ==========================================
        # 5. ALPHA SUPPRESSION CHECK
        # ==========================================
        print("Generating 5. Alpha Suppression Check...")
        plt.figure(figsize=(10, 6))
        # We focus specifically on Alpha
        sns.violinplot(data=clean_data, x='label', y='alpha', palette="Blues", inner="quartile")
        plt.title("Alpha Suppression Check (Lower Alpha = Higher Focus)")
        plt.ylabel("Alpha Power")
        plt.savefig(f"{RESULTS_DIR}/5_Alpha_Suppression.png")
        plt.close()

    # ==========================================
    # 6. FATIGUE / LEARNING EFFECT
    # ==========================================
    print("Generating 6. Fatigue Analysis...")
    # We use the index as a proxy for time/order
    trials = trials.sort_values(by=['timestamp'])
    trials['trial_order'] = range(1, len(trials) + 1)
    
    plt.figure(figsize=(12, 5))
    sns.regplot(data=trials, x='trial_order', y='reaction_time', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title("Fatigue Analysis: Reaction Time over Experiment Duration")
    plt.xlabel("Trial Order (1 = First Website, N = Last)")
    plt.ylabel("Reaction Time (s)")
    plt.savefig(f"{RESULTS_DIR}/6_Fatigue_Analysis.png")
    plt.close()

if __name__ == "__main__":
    analyze_all()