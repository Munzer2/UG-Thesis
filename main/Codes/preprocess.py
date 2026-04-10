import pandas as pd
import numpy as np
import scipy.signal as signal
import pywt
import os
import glob
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "dataset")
OUTPUT_DIR = os.path.join("..", "dataset_clean")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")  # New folder for images

SAMPLING_RATE = 512
MAINS_FREQ = 50      # Dhaka = 50Hz
AMPLITUDE_THRESHOLD = 100  # Used for final residual artifact marking

# Wavelet Denoising Config
WAVELET = 'db4'       # Daubechies-4 wavelet (matches EEG morphology)
WAVELET_LEVEL = 5     # Decomposition depth (captures 1-16 Hz at 512 Hz)

# Adaptive Z-Score Config
Z_THRESHOLD = 3.0     # Samples beyond this many SDs are interpolated (3.0 for single-channel)
Z_WINDOW_SEC = 5      # Sliding window size in seconds for local stats

# Create directories
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_notch(freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    freq_norm = freq / nyq
    b, a = signal.iirnotch(freq_norm, quality_factor)
    return b, a

def wavelet_denoise(data, wavelet=WAVELET, level=WAVELET_LEVEL):
    """
    Wavelet-based artifact removal using DWT with soft universal thresholding.
    
    How it works:
    1. Decompose the signal into frequency sub-bands (wavelet coefficients)
    2. Apply Donoho's universal threshold to shrink noisy coefficients
    3. Reconstruct the signal from the cleaned coefficients
    
    The key idea is that EEG signal energy concentrates in a few large
    coefficients, while artifact/noise energy spreads across many small ones.
    Soft thresholding shrinks the small (noisy) coefficients toward zero
    without touching the large (signal) coefficients.
    """
    # Decompose: signal -> [approx, detail_1, detail_2, ..., detail_N]
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Estimate noise level from the finest detail coefficients
    # (Donoho & Johnstone, 1994: Median Absolute Deviation / 0.6745)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Universal threshold: sqrt(2 * log(N)) * sigma
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply soft thresholding to detail coefficients only
    # (keep the approximation coefficients unchanged — they carry the slow EEG trends)
    denoised_coeffs = [coeffs[0]]  # approximation untouched
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
    
    # Reconstruct the clean signal
    clean = pywt.waverec(denoised_coeffs, wavelet)
    return clean[:len(data)]  # trim to original length (waverec may add 1 sample)

def adaptive_zscore_clean(data, z_thresh=Z_THRESHOLD, window_sec=Z_WINDOW_SEC):
    """
    Adaptive sliding-window z-score artifact correction.
    
    For each window, computes local mean and std, identifies samples
    that deviate more than z_thresh standard deviations, and replaces
    them with linearly interpolated values from neighboring clean samples.
    
    This handles residual artifacts that survive wavelet denoising
    (e.g., eye blinks, jaw clenches) and also adapts to slow drifts
    in signal baseline that occur over the course of a recording.
    """
    clean = data.copy()
    win_size = window_sec * SAMPLING_RATE
    total_interpolated = 0

    for start in range(0, len(data), win_size):
        end = min(start + win_size, len(data))
        segment = data[start:end]
        
        if len(segment) < 10:
            continue
        
        mu = np.mean(segment)
        sigma = np.std(segment)
        
        if sigma < 1e-6:  # flat signal, nothing to do
            continue
        
        z_scores = np.abs((segment - mu) / sigma)
        artifact_mask = z_scores > z_thresh
        
        if not np.any(artifact_mask):
            continue
        
        # Get global indices
        artifact_indices = np.where(artifact_mask)[0] + start
        good_indices = np.where(~artifact_mask)[0] + start
        
        if len(good_indices) > 2:
            # Interpolate artifact samples from surrounding clean samples
            clean[artifact_indices] = np.interp(
                artifact_indices, good_indices, data[good_indices]
            )
            total_interpolated += len(artifact_indices)
    
    return clean, total_interpolated

def visualize_cleaning(raw, bp_notch, wavelet_clean, zscore_clean, filename, save_path):
    """Saves a 3-row plot comparing each preprocessing stage."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"Preprocessing: {filename}", fontsize=14, fontweight='bold')
    
    # Plot a 5-second snippet from the middle
    mid_point = len(raw) // 2
    n_samples = 5 * SAMPLING_RATE  # 5 seconds
    start = mid_point
    end = mid_point + n_samples
    if len(raw) < n_samples:
        start, end = 0, len(raw)

    time_axis = np.arange(end - start) / SAMPLING_RATE

    # --- Row 1: Raw vs Bandpass+Notch ---
    axes[0].plot(time_axis, raw[start:end], color='lightgray', alpha=0.8, label='Original Raw')
    axes[0].plot(time_axis, bp_notch[start:end], color='#3498db', linewidth=0.9, label='After BP + Notch')
    axes[0].axhline(y=AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4)
    axes[0].axhline(y=-AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4)
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Stage 1-2: Bandpass (1-50 Hz) + Notch (50 Hz)')

    # --- Row 2: BP+Notch vs Wavelet Denoised ---
    axes[1].plot(time_axis, bp_notch[start:end], color='lightgray', alpha=0.8, label='After BP + Notch')
    axes[1].plot(time_axis, wavelet_clean[start:end], color='#2ecc71', linewidth=0.9, label='After Wavelet Denoise')
    axes[1].axhline(y=AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4, label='±100 µV')
    axes[1].axhline(y=-AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4)
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Stage 3: Wavelet Denoising (db4, level 5)')

    # --- Row 3: Wavelet vs Z-Score Cleaned ---
    axes[2].plot(time_axis, wavelet_clean[start:end], color='lightgray', alpha=0.8, label='After Wavelet Denoise')
    axes[2].plot(time_axis, zscore_clean[start:end], color='#e67e22', linewidth=0.9, label='After Adaptive Z-Score')
    axes[2].axhline(y=AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4, label='±100 µV')
    axes[2].axhline(y=-AMPLITUDE_THRESHOLD, color='r', linestyle='--', alpha=0.4)
    axes[2].set_ylabel('Amplitude (µV)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper right')
    axes[2].set_title(f'Stage 4: Adaptive Z-Score (z > {Z_THRESHOLD}, {Z_WINDOW_SEC}s window)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def clean_file(filepath):
    filename = os.path.basename(filepath)
    print(f"Processing {filename}...")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"   [ERROR] Could not read file: {e}")
        return None

    # Filter 1: Isolate RAW data
    raw_indices = df['type'] == 'raw'
    raw_data = df.loc[raw_indices, 'value'].values
    
    if len(raw_data) == 0:
        print("   [SKIP] No raw data found.")
        return None

    # --- STEP 1: BANDPASS (1-50Hz) ---
    b, a = butter_bandpass(1.0, 50.0, SAMPLING_RATE)
    bp_notch_signal = signal.filtfilt(b, a, raw_data)

    # --- STEP 2: NOTCH (50Hz) ---
    b_notch, a_notch = butter_notch(MAINS_FREQ, SAMPLING_RATE)
    bp_notch_signal = signal.filtfilt(b_notch, a_notch, bp_notch_signal)

    # --- STEP 3: WAVELET DENOISING ---
    wavelet_signal = wavelet_denoise(bp_notch_signal)
    print(f"   Wavelet denoise applied ({WAVELET}, level {WAVELET_LEVEL})")

    # --- STEP 4: ADAPTIVE Z-SCORE CLEANING ---
    clean_signal, n_interpolated = adaptive_zscore_clean(wavelet_signal)
    interp_pct = n_interpolated / len(clean_signal) * 100
    print(f"   Z-score cleanup: {n_interpolated:,} samples interpolated ({interp_pct:.1f}%)")

    # --- STEP 5: FINAL ARTIFACT MARKING ---
    # Flag any samples still above threshold after both cleaning stages
    is_artifact = np.abs(clean_signal) > AMPLITUDE_THRESHOLD
    artifact_pct = np.sum(is_artifact) / len(is_artifact) * 100
    print(f"   Final residual artifacts: {np.sum(is_artifact):,} samples ({artifact_pct:.1f}%)")

    df.loc[raw_indices, 'value'] = clean_signal
    df.loc[raw_indices, 'is_artifact'] = is_artifact

    # Save CSV
    out_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(out_path, index=False)
    
    return raw_data, bp_notch_signal, wavelet_signal, clean_signal

def main():
    files = glob.glob(os.path.join(DATA_DIR, "UI_Exp_*.csv"))
    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files. Starting preprocessing...")

    for f in files:
        result = clean_file(f)
        
        if result is not None:
            raw, bp_notch, wavelet_clean, zscore_clean = result
            
            # Generate and Save Plot
            image_name = os.path.basename(f).replace('.csv', '.png')
            save_path = os.path.join(PLOTS_DIR, image_name)
            
            visualize_cleaning(raw, bp_notch, wavelet_clean, zscore_clean,
                               os.path.basename(f), save_path)

    print(f"\n[DONE] Clean data saved to '{OUTPUT_DIR}'")
    print(f"[DONE] Pipeline: Bandpass → Notch → Wavelet Denoise → Adaptive Z-Score (z>{Z_THRESHOLD})")
    print(f"[DONE] Quality Check plots saved to '{PLOTS_DIR}'")

if __name__ == "__main__":
    main()