import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


FILE_PATTERN = "UI_Exp_*.csv"

def inspect_data(): 
    # load data
    files = glob.glob(FILE_PATTERN)
    if not files:
        print("No files found matching the pattern.")
        return

    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        temp_df['source_file'] = os.path.basename(f)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    # filter for only relevant phase

    task_df = df[df['phase'] == 'TASK'].copy()

    print("DATASET SUMMARY:")
    print(f"TOTAL ROWS: {len(df)}")
    print(f"TASK PHASE ROWS: {len(task_df)}")
    print(f"UNIQUE SUBJECTS: {task_df['subject_id'].unique()}")
    print(f"Websites visited: {task_df['image_name'].nunique()}")

    trials = task_df.drop_duplicates(subset=['subject_id', 'image', 'target_instruction'])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=trials, x='label', y='reaction_time', palette = 'viridis', errorbar='sd')
    plt.title("Check 1: Did 'Complex' sites take longer?")
    plt.ylabel("Reaction Time (seconds)")
    plt.xlabel("Website Category")
    plt.show()


    # Now let's check the signal quality

    raw_df = task_df[task_df['type'] == 'raw']

    if not raw_df.empty:
        sample_trial = raw_df['image'].unique()[0]
        sample_data = raw_df[raw_df['image'] == sample_trial].iloc[:2000] 

        plt.figure(figsize=(15, 4))
        plt.plot(sample_data['timestamp'], sample_data['value'], color='black', linewidth=0.8)
        
        # Draw threshold lines to show where we might need to "cut" data later
        plt.axhline(y=200, color='r', linestyle='--', label='Blink Threshold (+200)')
        plt.axhline(y=-200, color='r', linestyle='--', label='Blink Threshold (-200)')
        
        plt.title(f"Check 2: Raw Signal Quality (Snippet from {sample_trial})")
        plt.ylabel("Amplitude (uV)")
        plt.xlabel("Time")
        plt.legend()
        plt.show()

    # Cognitive load analysis
    power_df = task_df[task_df['type'] == 'power'].copy()

    if not power_df.empty:
        # Calculate the "Workload Ratio" (Theta / Alpha)
        # Avoid division by zero
        power_df['theta_alpha_ratio'] = power_df['theta'] / (power_df['alpha'] + 1)
        
        # Remove massive outliers for the plot (e.g., blinks causing 1000x spikes)
        q_low = power_df['theta_alpha_ratio'].quantile(0.05)
        q_high = power_df['theta_alpha_ratio'].quantile(0.95)
        clean_plot_data = power_df[(power_df['theta_alpha_ratio'] < q_high) & 
                                   (power_df['theta_alpha_ratio'] > q_low)]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=clean_plot_data, x='label', y='theta_alpha_ratio', palette="coolwarm")
        plt.title("Check 3: Cognitive Load Ratio (Theta/Alpha)")
        plt.ylabel("Workload Index (Higher = More Effort)")
        plt.show()
        
        print("\n" + "="*40)
        print("PRE-PROCESSING PLAN")
        print("="*40)
        print(f"1. RAW DATA: Found {len(raw_df)} samples.")
        print(f"   - Max Amplitude: {raw_df['value'].max()} uV")
        print(f"   - Min Amplitude: {raw_df['value'].min()} uV")
        if raw_df['value'].max() > 500 or raw_df['value'].min() < -500:
            print("   -> WARNING: Large artifacts detected (Blinks/Jaw). \n   -> ACTION: We MUST apply the 'Amplitude Threshold' filter (drop > 200uV).")
        else:
            print("   -> Signal looks relatively clean.")
            
        print(f"\n2. POWER DATA: Found {len(power_df)} samples.")
        avg_diff = power_df.groupby('label')['theta_alpha_ratio'].mean()
        print("   - Average Workload Ratio by Category:")
        print(avg_diff)
        if avg_diff.get('design_A_complex', 0) > avg_diff.get('design_B_simple', 0):
            print("   -> SUCCESS: Complex sites show higher average load. The data is valid.")
        else:
            print("   -> CAUTION: Simple sites show higher load. Check for noise or short task duration.")

if __name__ == "__main__":
    inspect_data()