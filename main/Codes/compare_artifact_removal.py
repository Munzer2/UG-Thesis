"""
=======================================================
Artifact Removal Algorithm Comparison
=======================================================
Compares 4 artifact removal methods on single-channel EEG:
  1. Adaptive Z-Score Thresholding
  2. Wavelet Denoising (DWT with db4)
  3. Empirical Mode Decomposition (EMD)
  4. Artifact Subspace Reconstruction (ASR)

Generates comparison plots saved to ../dataset_clean/plots/
=======================================================
"""


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# --- Artifact Removal Imports ---
import pywt
from PyEMD import EMD as PyEMD_EMD
from meegkit.asr import ASR

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "dataset")
CLEAN_DIR = os.path.join("..", "dataset_clean")
PLOTS_DIR = os.path.join(CLEAN_DIR, "plots", "artifact_comparison")
SAMPLING_RATE = 512
MAINS_FREQ = 50

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

# ==========================================
# BASELINE FILTERS (applied before all methods)
# ==========================================
def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=512, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, freq=50, fs=512, quality=30):
    b, a = signal.iirnotch(freq / (0.5 * fs), quality)
    return signal.filtfilt(b, a, data)

# ==========================================
# METHOD 1: Adaptive Z-Score Thresholding
# ==========================================
def method_zscore(data, z_thresh=3.0, window_sec=5):
    """Sliding-window z-score: marks and interpolates samples > z_thresh SDs."""
    clean = data.copy()
    win_size = window_sec * SAMPLING_RATE

    for start in range(0, len(data), win_size):
        end = min(start + win_size, len(data))
        segment = data[start:end]
        if len(segment) < 10:
            continue
        mu = np.mean(segment)
        sigma = np.std(segment)
        if sigma < 1e-6:
            continue
        z = np.abs((segment - mu) / sigma)
        artifact_mask = z > z_thresh
        if np.any(artifact_mask):
            indices = np.where(artifact_mask)[0] + start
            good_indices = np.where(~artifact_mask)[0] + start
            if len(good_indices) > 2:
                clean[indices] = np.interp(indices, good_indices, data[good_indices])

    return clean

# ==========================================
# METHOD 2: Wavelet Denoising (db4)
# ==========================================
def method_wavelet(data, wavelet='db4', level=5):
    """DWT decomposition with soft universal thresholding."""
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Donoho's universal threshold
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Soft threshold detail coefficients (keep approximation)
    denoised = [coeffs[0]]
    for c in coeffs[1:]:
        denoised.append(pywt.threshold(c, threshold, mode='soft'))

    clean = pywt.waverec(denoised, wavelet)
    return clean[:len(data)]

# ==========================================
# METHOD 3: EMD-Based Removal
# ==========================================
def method_emd(data, n_artifact_imfs=2):
    """Remove first N IMFs (typically contain high-freq artifacts)."""
    emd = PyEMD_EMD()
    emd.FIXE_H = 10  # Limit iterations for speed

    try:
        imfs = emd(data)
    except Exception:
        return data.copy()

    if len(imfs) <= n_artifact_imfs:
        return data.copy()

    # Reconstruct without the first n_artifact_imfs
    clean = np.sum(imfs[n_artifact_imfs:], axis=0)
    return clean

# ==========================================
# METHOD 4: ASR (Artifact Subspace Reconstruction)
# ==========================================
def method_asr(data, sfreq=512, cutoff=20):
    """Single-channel ASR via meegkit."""
    signal_2d = data.reshape(-1, 1)

    asr = ASR(sfreq=sfreq, cutoff=cutoff)

    # Use a clean calibration segment (first 10s or 1/4 of signal)
    cal_len = min(10 * sfreq, len(data) // 4)
    try:
        asr.fit(signal_2d[:cal_len])
        clean_2d = asr.transform(signal_2d)
        return clean_2d.flatten()
    except Exception:
        return data.copy()

# ==========================================
# BASELINE (current method from preprocess.py)
# ==========================================
def method_baseline_threshold(data, threshold=100):
    """Original method: mark |value| > threshold as artifact, set to 0."""
    clean = data.copy()
    clean[np.abs(clean) > threshold] = 0
    return clean

# ==========================================
# METRICS
# ==========================================
def compute_metrics(original, cleaned):
    """Compute quality metrics comparing original vs cleaned signal."""
    # SNR improvement estimate (signal = cleaned, noise = what was removed)
    removed = original - cleaned
    signal_power = np.mean(cleaned**2)
    noise_power = np.mean(removed**2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Correlation with original (how much signal is preserved)
    corr = np.corrcoef(original, cleaned)[0, 1]

    # Artifact residual: % of samples still > 100uV after cleaning
    artifact_pct = np.sum(np.abs(cleaned) > 100) / len(cleaned) * 100

    # RMS of removed component (how much was altered)
    rms_removed = np.sqrt(np.mean(removed**2))

    return {
        'SNR (dB)': round(snr_db, 2),
        'Correlation': round(corr, 4),
        'Artifact %': round(artifact_pct, 2),
        'RMS Removed': round(rms_removed, 2)
    }

# ==========================================
# PSD (Power Spectral Density) computation
# ==========================================
def compute_psd(data, fs=512):
    """Compute PSD using Welch's method."""
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(1024, len(data)), 
                               noverlap=min(512, len(data)//2))
    return freqs, psd

# ==========================================
# VISUALIZATION
# ==========================================
def plot_comparison(raw_original, bp_notch, results, participant, save_path):
    """
    Generate a comprehensive 4-panel comparison plot:
      - Panel 1: Time-domain overlay (5-second snippet)
      - Panel 2: Time-domain zoomed on a noisy segment
      - Panel 3: PSD comparison
      - Panel 4: Metrics bar chart
    """
    methods = list(results.keys())
    colors = {
        'Baseline (Threshold)': '#e74c3c',
        'Adaptive Z-Score':     '#e67e22',
        'Wavelet Denoise':      '#2ecc71',
        'EMD':                  '#3498db',
        'ASR':                  '#9b59b6'
    }

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Artifact Removal Comparison — {participant}', 
                 fontsize=18, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # --- Find a segment with visible artifacts for zooming ---
    # Look for the region with the highest amplitude (most artifacts)
    window = 5 * SAMPLING_RATE  # 5 seconds
    max_amp = 0
    artifact_start = len(bp_notch) // 2  # fallback: use middle
    for i in range(0, len(bp_notch) - window, window // 2):
        amp = np.max(np.abs(bp_notch[i:i+window]))
        if amp > max_amp:
            max_amp = amp
            artifact_start = i
    artifact_end = artifact_start + window

    # === PANEL 1: Full 5-second snippet from middle ===
    ax1 = fig.add_subplot(gs[0, :])
    mid = len(bp_notch) // 2
    snip_start, snip_end = mid, mid + 5 * SAMPLING_RATE
    if snip_end > len(bp_notch):
        snip_start = max(0, len(bp_notch) - 5 * SAMPLING_RATE)
        snip_end = len(bp_notch)
    t = np.arange(snip_end - snip_start) / SAMPLING_RATE

    ax1.plot(t, bp_notch[snip_start:snip_end], 
             color='lightgray', alpha=0.7, linewidth=1, label='After BP+Notch only')
    for name in methods:
        cleaned = results[name]['cleaned']
        ax1.plot(t, cleaned[snip_start:snip_end], 
                 color=colors[name], alpha=0.8, linewidth=0.8, label=name)

    ax1.set_title('Time Domain — 5s Snippet (Middle of Recording)', fontsize=13)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.set_xlim([0, t[-1]])

    # === PANEL 2: Zoomed on artifact-heavy region ===
    ax2 = fig.add_subplot(gs[1, 0])
    t2 = np.arange(artifact_end - artifact_start) / SAMPLING_RATE

    ax2.plot(t2, bp_notch[artifact_start:artifact_end], 
             color='lightgray', alpha=0.7, linewidth=1.2, label='After BP+Notch only')
    for name in methods:
        cleaned = results[name]['cleaned']
        ax2.plot(t2, cleaned[artifact_start:artifact_end], 
                 color=colors[name], alpha=0.85, linewidth=0.9, label=name)

    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.4, label='±100 µV')
    ax2.axhline(y=-100, color='red', linestyle='--', alpha=0.4)
    ax2.set_title(f'Zoomed — Noisiest 5s Segment (max amp: {max_amp:.0f} µV)', fontsize=13)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (µV)')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)

    # === PANEL 3: PSD comparison ===
    ax3 = fig.add_subplot(gs[1, 1])
    freqs_ref, psd_ref = compute_psd(bp_notch)
    ax3.semilogy(freqs_ref, psd_ref, color='lightgray', alpha=0.7, 
                 linewidth=1.5, label='After BP+Notch only')

    for name in methods:
        freqs, psd = compute_psd(results[name]['cleaned'])
        ax3.semilogy(freqs, psd, color=colors[name], alpha=0.85, linewidth=1, label=name)

    # Shade EEG bands
    bands = [(4,8,'θ','#fff3bf'), (8,13,'α','#d0ebff'), 
             (13,30,'β','#d3f9d8'), (30,50,'γ','#ffe3e3')]
    for lo, hi, lbl, clr in bands:
        ax3.axvspan(lo, hi, alpha=0.15, color=clr)
        ax3.text((lo+hi)/2, ax3.get_ylim()[1]*0.3, lbl, 
                 ha='center', fontsize=10, fontweight='bold', alpha=0.5)

    ax3.set_title('Power Spectral Density (Welch)', fontsize=13)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (µV²/Hz)')
    ax3.set_xlim([0.5, 55])
    ax3.legend(loc='upper right', fontsize=8, ncol=2)

    # === PANEL 4: Metrics comparison ===
    ax4 = fig.add_subplot(gs[2, 0])
    
    metric_names = ['SNR (dB)', 'Correlation', 'Artifact %', 'RMS Removed']
    x = np.arange(len(metric_names))
    width = 0.15
    
    for i, name in enumerate(methods):
        metrics = results[name]['metrics']
        vals = [metrics[m] for m in metric_names]
        bars = ax4.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    ax4.set_xticks(x + width * (len(methods)-1) / 2)
    ax4.set_xticklabels(metric_names, fontsize=10)
    ax4.set_title('Quality Metrics Comparison', fontsize=13)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_ylabel('Value')

    # === PANEL 5: Artifact count per method ===
    ax5 = fig.add_subplot(gs[2, 1])
    
    method_labels = []
    artifact_counts = []
    total_samples = len(bp_notch)
    
    # Reference: how many artifacts in the BP+Notch-only signal
    ref_artifacts = np.sum(np.abs(bp_notch) > 100)
    method_labels.append('BP+Notch\n(no removal)')
    artifact_counts.append(ref_artifacts)
    
    for name in methods:
        cleaned = results[name]['cleaned']
        count = np.sum(np.abs(cleaned) > 100)
        method_labels.append(name.replace(' ', '\n'))
        artifact_counts.append(count)
    
    bar_colors = ['lightgray'] + [colors[m] for m in methods]
    bars = ax5.bar(method_labels, artifact_counts, color=bar_colors, alpha=0.85, edgecolor='gray')
    
    for bar, count in zip(bars, artifact_counts):
        pct = count / total_samples * 100
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_samples*0.005,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
    
    ax5.set_title(f'Samples > ±100 µV (out of {total_samples:,} total)', fontsize=13)
    ax5.set_ylabel('Artifact Sample Count')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {save_path}")

# ==========================================
# PER-TRIAL COMPARISON PLOT
# ==========================================
def plot_per_trial(df, raw_signal, results, participant, save_path):
    """
    Shows artifact % by complexity level (simple/moderate/complex) for each method.
    """
    methods = list(results.keys())
    colors_map = {
        'Baseline (Threshold)': '#e74c3c',
        'Adaptive Z-Score':     '#e67e22',
        'Wavelet Denoise':     '#2ecc71',
        'EMD':                  '#3498db',
        'ASR':                  '#9b59b6'
    }

    # Get raw indices
    raw_mask = df['type'] == 'raw'
    raw_df = df[raw_mask].copy()
    raw_df = raw_df.reset_index(drop=True)

    # Map labels to simple categories
    def simplify_label(label):
        if isinstance(label, str):
            if 'simple' in label.lower() or 'a_simple' in label.lower():
                return 'Simple'
            elif 'moderate' in label.lower() or 'c_moderate' in label.lower():
                return 'Moderate'
            elif 'complex' in label.lower() or 'b_complex' in label.lower():
                return 'Complex'
        return 'Other'

    raw_df['complexity'] = raw_df['label'].apply(simplify_label)
    task_mask = raw_df['phase'] == 'TASK'
    task_raw = raw_df[task_mask]

    complexities = ['Simple', 'Moderate', 'Complex']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Artifact % by UI Complexity — {participant}', fontsize=16, fontweight='bold')

    for idx, comp in enumerate(complexities):
        ax = axes[idx]
        comp_indices = task_raw[task_raw['complexity'] == comp].index.values

        if len(comp_indices) == 0:
            ax.set_title(f'{comp} (no data)')
            continue

        method_pcts = {}
        for name in methods:
            cleaned = results[name]['cleaned']
            # Get the cleaned values at these indices
            valid = comp_indices[comp_indices < len(cleaned)]
            if len(valid) > 0:
                vals = cleaned[valid]
                pct = np.sum(np.abs(vals) > 100) / len(vals) * 100
            else:
                pct = 0
            method_pcts[name] = pct

        bars = ax.bar(range(len(methods)), list(method_pcts.values()),
                      color=[colors_map[m] for m in methods], alpha=0.85, edgecolor='gray')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)

        for bar, pct in zip(bars, method_pcts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'{comp} UI ({len(comp_indices):,} samples)', fontsize=13)
        ax.set_ylabel('Artifact %' if idx == 0 else '')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {save_path}")

# ==========================================
# MAIN
# ==========================================
def main():
    files = glob.glob(os.path.join(DATA_DIR, "UI_Exp_*.csv"))
    if not files:
        print("[ERROR] No files found in dataset/. Check DATA_DIR.")
        return

    print(f"Found {len(files)} files.\n")
    print("=" * 60)
    print("ARTIFACT REMOVAL COMPARISON")
    print("=" * 60)

    summary_rows = []

    for filepath in files:
        filename = os.path.basename(filepath)
        participant = filename.replace('UI_Exp_', '').replace('.csv', '')
        print(f"\n▶ Processing {participant}...")

        # Load data
        df = pd.read_csv(filepath)
        raw_mask = df['type'] == 'raw'
        raw_data = df.loc[raw_mask, 'value'].values.astype(float)

        if len(raw_data) == 0:
            print(f"   [SKIP] No raw data in {filename}")
            continue

        print(f"   Samples: {len(raw_data):,}")

        # === Apply baseline filters (common to all methods) ===
        print("   Applying bandpass (1-50 Hz) + notch (50 Hz)...")
        bp_notch = bandpass_filter(raw_data)
        bp_notch = notch_filter(bp_notch)

        # === Apply each artifact removal method ===
        results = {}

        print("   [1/5] Baseline threshold (±100 µV)...")
        cleaned = method_baseline_threshold(bp_notch)
        results['Baseline (Threshold)'] = {
            'cleaned': cleaned,
            'metrics': compute_metrics(bp_notch, cleaned)
        }

        print("   [2/5] Adaptive z-score...")
        cleaned = method_zscore(bp_notch)
        results['Adaptive Z-Score'] = {
            'cleaned': cleaned,
            'metrics': compute_metrics(bp_notch, cleaned)
        }

        print("   [3/5] Wavelet denoising (db4, level 5)...")
        cleaned = method_wavelet(bp_notch)
        results['Wavelet Denoise'] = {
            'cleaned': cleaned,
            'metrics': compute_metrics(bp_notch, cleaned)
        }

        print("   [4/5] EMD (removing first 2 IMFs)...")
        # EMD can be slow on long signals, so process in chunks
        chunk_size = 30 * SAMPLING_RATE  # 30-second chunks
        emd_clean = np.zeros_like(bp_notch)
        for i in range(0, len(bp_notch), chunk_size):
            end = min(i + chunk_size, len(bp_notch))
            chunk = bp_notch[i:end]
            if len(chunk) > 100:
                emd_clean[i:end] = method_emd(chunk)
            else:
                emd_clean[i:end] = chunk
        results['EMD'] = {
            'cleaned': emd_clean,
            'metrics': compute_metrics(bp_notch, emd_clean)
        }

        print("   [5/5] ASR (cutoff=20)...")
        cleaned = method_asr(bp_notch)
        results['ASR'] = {
            'cleaned': cleaned,
            'metrics': compute_metrics(bp_notch, cleaned)
        }

        # === Print metrics table ===
        print(f"\n   {'Method':<25} {'SNR (dB)':>10} {'Corr':>8} {'Artif %':>10} {'RMS Rem':>10}")
        print(f"   {'-'*65}")
        for name, data in results.items():
            m = data['metrics']
            print(f"   {name:<25} {m['SNR (dB)']:>10.2f} {m['Correlation']:>8.4f} "
                  f"{m['Artifact %']:>9.2f}% {m['RMS Removed']:>10.2f}")
            summary_rows.append({
                'Participant': participant,
                'Method': name,
                **m
            })

        # === Generate comparison plots ===
        print(f"\n   Generating plots...")
        save_main = os.path.join(PLOTS_DIR, f"comparison_{participant}.png")
        plot_comparison(raw_data, bp_notch, results, participant, save_main)

        save_trial = os.path.join(PLOTS_DIR, f"by_complexity_{participant}.png")
        plot_per_trial(df, bp_notch, results, participant, save_trial)

    # === Summary across all participants ===
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(PLOTS_DIR, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Metrics summary saved to: {summary_path}")

        # === Aggregate bar chart ===
        print("\n   Generating aggregate comparison...")
        plot_aggregate_summary(summary_df, os.path.join(PLOTS_DIR, "aggregate_comparison.png"))

    print("\n" + "=" * 60)
    print(f"ALL DONE! Plots saved to: {PLOTS_DIR}")
    print("=" * 60)

def plot_aggregate_summary(df, save_path):
    """Bar chart of average metrics across all participants for each method."""
    methods = df['Method'].unique()
    colors_map = {
        'Baseline (Threshold)': '#e74c3c',
        'Adaptive Z-Score':     '#e67e22',
        'Wavelet Denoise':      '#2ecc71',
        'EMD':                  '#3498db',
        'ASR':                  '#9b59b6'
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Aggregate Comparison Across All Participants', fontsize=16, fontweight='bold')

    metrics = ['SNR (dB)', 'Correlation', 'Artifact %', 'RMS Removed']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []
        clrs = []
        for m in methods:
            vals = df[df['Method'] == m][metric]
            means.append(vals.mean())
            stds.append(vals.std())
            clrs.append(colors_map.get(m, 'gray'))

        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
                      color=clrs, alpha=0.85, edgecolor='gray')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight='bold')

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {save_path}")


if __name__ == "__main__":
    main()
