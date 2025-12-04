import numpy as np
import scipy.signal as signal
import glob
import os
import sys

def load_data(file_path):
    """Loads PPG data from a text file."""
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

def get_peaks(data, fs=100):
    """Detects peaks in the PPG signal."""
    # Heart rate is typically 40-180 bpm -> 0.6 - 3 Hz. 
    # Min distance between peaks ~ 0.3s (30 samples at 100Hz)
    distance = int(0.3 * fs)
    # Low prominence to catch small peaks
    peaks, _ = signal.find_peaks(data, distance=distance, prominence=1)
    return peaks

def extract_signals(data, fs=100):
    """
    Extracts respiratory signals:
    1. RIIV: Respiratory Induced Intensity Variation (Peak Amplitudes)
    2. RIFV: Respiratory Induced Frequency Variation (Inter-Beat Intervals)
    3. Baseline: Raw signal filtered to respiratory band (0.1-1.0 Hz)
    """
    # 1. Bandpass to get clean cardiac signal for peak detection
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 5.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    cardiac_ppg = signal.filtfilt(b, a, data)
    
    # 2. Detect Peaks
    peaks = get_peaks(cardiac_ppg, fs)
    
    if len(peaks) < 2:
        return None, None, None, 4.0
    
    # Time vector for peaks
    peak_times = peaks / fs
    
    # RIIV: Peak Amplitudes
    riiv_raw = cardiac_ppg[peaks]
    
    # RIFV: Inter-Beat Intervals
    ibis = np.diff(peak_times)
    # Associated times for IBIs (midpoints)
    ibi_times = peak_times[:-1] + np.diff(peak_times)/2
    
    # Resample to constant Hz for FFT
    resample_fs = 4.0
    duration = len(data) / fs
    t_interp = np.arange(0, duration, 1/resample_fs)
    
    # Interpolate RIIV
    riiv_interp = np.interp(t_interp, peak_times, riiv_raw)
    
    # Interpolate RIFV
    rifv_interp = np.interp(t_interp, ibi_times, ibis)
    
    # 3. Raw Baseline (Bandpass 0.1 - 1.0 Hz)
    b_base, a_base = signal.butter(2, [0.1/nyquist, 1.0/nyquist], btype='band')
    # Detrend first to avoid edge ripples
    detrended_data = signal.detrend(data)
    raw_baseline = signal.filtfilt(b_base, a_base, detrended_data)
    
    # Downsample baseline to same resample_fs
    t_raw = np.arange(len(data)) / fs
    baseline_interp = np.interp(t_interp, t_raw, raw_baseline)
    
    return riiv_interp, rifv_interp, baseline_interp, resample_fs

def get_fft_peak(sig, fs):
    """
    Calculates the dominant frequency and its power in the respiratory range.
    Range: 4 bpm (0.06 Hz) to 60 bpm (1.0 Hz)
    """
    if sig is None:
        return 0, 0
        
    # Detrend and Window
    sig = signal.detrend(sig)
    window = signal.windows.hann(len(sig))
    sig_win = sig * window
    
    # FFT
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(sig_win))
    
    # Mask for respiratory range
    mask = (freqs >= 0.06) & (freqs <= 1.0)
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[mask]
    
    if len(valid_fft) == 0:
        return 0, 0
        
    peak_idx = np.argmax(valid_fft)
    peak_freq = valid_freqs[peak_idx]
    peak_power = valid_fft[peak_idx]
    
    return peak_freq * 60, peak_power

def calculate_respiratory_rate(data, fs=100):
    """
    Calculates respiratory rate using fusion of RIIV, RIFV, and Baseline.
    Also detects chaotic breathing.
    """
    riiv, rifv, baseline, resample_fs = extract_signals(data, fs)
    
    # Chaos Detection (Spectral Entropy on RIIV)
    if riiv is not None:
        f, Pxx = signal.welch(riiv, resample_fs, nperseg=len(riiv)//2)
        Pxx_norm = Pxx / np.sum(Pxx)
        entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
    else:
        entropy = 0
        
    # Lower threshold to catch chaos_1 (Entropy ~3.91)
    # Normal files are usually < 3.7 (except extreme ratios which are also "abnormal")
    is_chaos = entropy > 3.8
    
    rate_riiv, power_riiv = get_fft_peak(riiv, resample_fs)
    rate_rifv, power_rifv = get_fft_peak(rifv, resample_fs)
    rate_base, power_base = get_fft_peak(baseline, resample_fs)
    
    # Fusion Logic
    final_rate = rate_riiv # Default to RIIV
    
    # If RIIV and RIFV are close, that increases confidence in them
    if abs(rate_riiv - rate_rifv) < 3:
        confidence_riiv = 1.0
    else:
        confidence_riiv = 0.5
        
    # Check for high rate in Baseline
    if 25 < rate_base < 45:
        # If Base is in this "fast breathing" band, and RIIV is low, switch to Base
        if rate_riiv < 15:
            final_rate = rate_base
            
    return {
        'fused': final_rate,
        'riiv': rate_riiv,
        'rifv': rate_rifv,
        'base': rate_base,
        'entropy': entropy,
        'is_chaos': is_chaos
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calculate Respiratory Rate from PPG data.')
    parser.add_argument('file', nargs='?', help='Path to the PPG text file.')
    args = parser.parse_args()

    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found.")
            return
            
        raw_data = load_data(args.file)
        fs = len(raw_data) / 60.0
        
        results = calculate_respiratory_rate(raw_data, fs)
        
        chaos_str = " (CHAOTIC)" if results['is_chaos'] else ""
        print(f"Estimated Respiratory Rate: {results['fused']:.2f} BPM{chaos_str}")
        print(f"Details: RIIV={results['riiv']:.2f}, RIFV={results['rifv']:.2f}, Base={results['base']:.2f}, Entropy={results['entropy']:.2f}")
        
    else:
        # Process all files in default dir (Demo mode)
        data_dir = "/Users/jim94/Desktop/ECG/ppg2rr"
        files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        print(f"{'File':<25} | {'RIIV':<6} | {'RIFV':<6} | {'Base':<6} | {'Entropy':<8} | {'Calculated':<10}")
        print("-" * 80)
        
        for file_path in sorted(files):
            raw_data = load_data(file_path)
            fs = len(raw_data) / 60.0
            
            results = calculate_respiratory_rate(raw_data, fs)
            
            chaos_mark = " (!)" if results['is_chaos'] else ""
            print(f"{os.path.basename(file_path):<25} | {results['riiv']:<6.1f} | {results['rifv']:<6.1f} | {results['base']:<6.1f} | {results['entropy']:<8.2f} | {results['fused']:<6.1f}{chaos_mark}")

if __name__ == "__main__":
    main()
