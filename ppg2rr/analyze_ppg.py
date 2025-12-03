import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import glob
import os
import re

def load_data(file_path):
    """Loads PPG data from a text file."""
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

def parse_filename(filename):
    """Parses the filename to extract the ground truth respiratory rate."""
    basename = os.path.basename(filename)
    # Handle the specific case of '5d'
    if '5d' in basename:
        return 5
    
    match = re.search(r'data_(\d+)in', basename)
    if match:
        return int(match.group(1))
    return None

def get_peaks(data, fs=100):
    """Detects peaks in the PPG signal."""
    # Simple peak detector with distance constraint
    # Heart rate is typically 40-180 bpm -> 0.6 - 3 Hz. 
    # Min distance between peaks ~ 0.3s (30 samples at 100Hz)
    distance = int(0.3 * fs)
    # Lower prominence to 1 (raw data is integer, variations might be small)
    peaks, _ = signal.find_peaks(data, distance=distance, prominence=1)
    print(f"  Debug: Found {len(peaks)} peaks")
    return peaks

def extract_respiratory_signals(data, fs=100):
    """
    Extracts respiratory signals using RIIV, RIAV, and RIFV.
    """
    # 1. Bandpass to get clean cardiac signal
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 5.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    cardiac_ppg = signal.filtfilt(b, a, data)
    
    # 2. Detect Peaks
    peaks = get_peaks(cardiac_ppg, fs)
    
    if len(peaks) < 2:
        return None, None, None
    
    # Time vector for peaks
    peak_times = peaks / fs
    
    # RIIV: Peak Amplitudes (Baseline + Amplitude)
    # Actually, strictly RIIV is baseline, RIAV is amplitude. 
    # But peak values capture both. Let's use Peak values as one signal.
    riiv_raw = cardiac_ppg[peaks]
    
    # RIFV: Inter-Beat Intervals
    ibis = np.diff(peak_times)
    # Associated times for IBIs (midpoints)
    ibi_times = peak_times[:-1] + np.diff(peak_times)/2
    
    # Resample to constant Hz for FFT (e.g., 4 Hz)
    resample_fs = 4.0
    duration = len(data) / fs
    t_interp = np.arange(0, duration, 1/resample_fs)
    
    # Interpolate RIIV
    riiv_interp = np.interp(t_interp, peak_times, riiv_raw)
    
    # Interpolate RIFV
    rifv_interp = np.interp(t_interp, ibi_times, ibis)
    
    # 3. Raw Baseline (Bandpass 0.1 - 1.0 Hz)
    # This avoids aliasing for high respiratory rates (e.g. 36 bpm) which might be close to HR/2
    nyquist = 0.5 * fs
    b_base, a_base = signal.butter(2, [0.1/nyquist, 1.0/nyquist], btype='band')
    raw_baseline = signal.filtfilt(b_base, a_base, data)
    # Downsample baseline to same resample_fs for consistency in FFT
    # Simple decimation or interpolation
    t_raw = np.arange(len(data)) / fs
    baseline_interp = np.interp(t_interp, t_raw, raw_baseline)
    
    return riiv_interp, rifv_interp, baseline_interp, resample_fs

def estimate_rate_from_signal(sig, fs):
    """Estimates rate from a 1D signal."""
    if sig is None:
        return 0
        
    # Detrend
    sig = signal.detrend(sig)
    
    # Windowing
    window = signal.windows.hann(len(sig))
    sig = sig * window
    
    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(sig))
    
    # Range: 4 bpm (0.06 Hz) to 60 bpm (1.0 Hz)
    mask = (freqs >= 0.06) & (freqs <= 1.0)
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[mask]
    
    if len(valid_fft) == 0:
        return 0
        
    peak_idx = np.argmax(valid_fft)
    return valid_freqs[peak_idx] * 60

def main():
    data_dir = "/Users/josh/Documents/ppg"
    files = glob.glob(os.path.join(data_dir, "data_*.txt"))
    
    results = []
    
    plt.figure(figsize=(20, 12))
    
    for i, file_path in enumerate(sorted(files)):
        print(f"Processing {os.path.basename(file_path)}...")
        raw_data = load_data(file_path)
        true_rate = parse_filename(file_path)
        
        fs = len(raw_data) / 60.0
        
        riiv, rifv, baseline, resample_fs = extract_respiratory_signals(raw_data, fs=fs)
        
        rate_riiv = estimate_rate_from_signal(riiv, resample_fs)
        rate_rifv = estimate_rate_from_signal(rifv, resample_fs)
        rate_base = estimate_rate_from_signal(baseline, resample_fs)
        
        # Fusion logic:
        # If rates disagree, which one to trust?
        # High rates (>20) are better seen in baseline.
        # Low rates (<10) are seen in all, but RIIV is usually clean.
        # Let's just report all for now.
        
        results.append({
            'file': os.path.basename(file_path),
            'true_rate': true_rate,
            'est_riiv': rate_riiv,
            'est_rifv': rate_rifv,
            'est_base': rate_base
        })
        
        # Plotting
        plt.subplot(len(files), 4, i*4 + 1)
        plt.plot(signal.detrend(raw_data))
        plt.title(f"{os.path.basename(file_path)} (Detrended, True: {true_rate})")
        
        # Plot RIIV FFT
        plt.subplot(len(files), 4, i*4 + 2)
        if riiv is not None:
            f, Pxx = signal.periodogram(signal.detrend(riiv), resample_fs)
            plt.plot(f*60, Pxx)
            plt.xlim(0, 60)
            plt.title(f"RIIV: {rate_riiv:.1f}")
            plt.axvline(x=true_rate, color='r', linestyle='--')
        
        # Plot RIFV FFT
        plt.subplot(len(files), 4, i*4 + 3)
        if rifv is not None:
            f, Pxx = signal.periodogram(signal.detrend(rifv), resample_fs)
            plt.plot(f*60, Pxx)
            plt.xlim(0, 60)
            plt.title(f"RIFV: {rate_rifv:.1f}")
            plt.axvline(x=true_rate, color='r', linestyle='--')

        # Plot Baseline FFT
        plt.subplot(len(files), 4, i*4 + 4)
        if baseline is not None:
            f, Pxx = signal.periodogram(signal.detrend(baseline), resample_fs)
            plt.plot(f*60, Pxx)
            plt.xlim(0, 60)
            plt.title(f"Base: {rate_base:.1f}")
            plt.axvline(x=true_rate, color='r', linestyle='--')

    plt.tight_layout()
    plt.savefig('ppg_analysis_plot_v3.png')
    print("Analysis plot saved to ppg_analysis_plot_v3.png")
    
    print("\nResults Summary:")
    print(f"{'File':<20} | {'True':<5} | {'RIIV':<6} | {'RIFV':<6} | {'Base':<6} | {'Best Diff':<10}")
    print("-" * 75)
    for res in results:
        # Find closest estimate
        estimates = [res['est_riiv'], res['est_rifv'], res['est_base']]
        diffs = [abs(res['true_rate'] - e) for e in estimates]
        best_diff = min(diffs)
        print(f"{res['file']:<20} | {res['true_rate']:<5} | {res['est_riiv']:<6.1f} | {res['est_rifv']:<6.1f} | {res['est_base']:<6.1f} | {best_diff:<10.1f}")

if __name__ == "__main__":
    main()
