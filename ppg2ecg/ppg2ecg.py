import argparse
import re
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.signal import resample
from math import pi

# 1. FORCE CPU (Fixes the GPU crash/noise)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. FORCE FLOAT64 (Fixes the "expected dtype float..." error)
# This MUST run before the model is loaded!
tf.keras.backend.set_floatx('float64')

# --- Import CardioGAN modules ---
# These files must be in the same folder as this script
import module
import tflib

# --- CONFIGURATION ---
ORIGINAL_FS = 81.7     # Your sensor's speed
TARGET_FS = 128.0      # Model's required speed
WINDOW_SEC = 4         # Model window duration
SEGMENT_LEN = int(TARGET_FS * WINDOW_SEC) # 512 points
MODEL_DIR = './weights' # Path to the unzipped weights

# --- 1. DATA READING & FILTERING (Your Code) ---

def read_integers(path: Path):
    text = path.read_text(encoding="utf-8")
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    return [float(n) for n in nums]

def DC_filter(values, mode='iir', fs=250.0, cutoff=0.5):
    vals = list(values)
    if not vals: return []
    # IIR high-pass
    fc = float(cutoff); fs = float(fs)
    rc = 1.0 / (2.0 * pi * fc); dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out = [0.0] * len(vals)
    prev_out = 0.0; prev_in = float(vals[0])
    for i in range(1, len(vals)):
        x = float(vals[i])
        y = alpha * (prev_out + x - prev_in)
        out[i] = y
        prev_out = y; prev_in = x
    return out

def moving_average_filter(values, window_size=5):
    vals = list(values)
    if not vals: return []
    window_size = max(1, int(window_size))
    out = []
    for i in range(len(vals)):
        start = max(0, i - window_size + 1)
        window = vals[start:i + 1]
        avg = sum(window) / len(window)
        out.append(avg)
    return out

# --- 2. AI MODEL SETUP ---

def load_cardiogan():
    """Initializes the TensorFlow model and loads weights."""
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Weights folder '{MODEL_DIR}' not found!", file=sys.stderr)
        print("Please download 'cardiogan_ppg2ecg_generator.zip' and extract it here.", file=sys.stderr)
        sys.exit(1)
        
    print("Loading CardioGAN model...")
    # Initialize the generator structure
    Gen_PPG2ECG = module.generator_attention()
    # Restore weights
    checkpoint = tflib.Checkpoint(dict(Gen_PPG2ECG=Gen_PPG2ECG), MODEL_DIR)
    checkpoint.restore()
    print("Model loaded successfully.")
    return Gen_PPG2ECG

# --- 3. MAIN PIPELINE ---

def main():
    p = argparse.ArgumentParser(description="Convert Arduino PPG to ECG using CardioGAN.")
    p.add_argument("file", nargs="?", default="raw_data.txt", help="Input text file")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    # A. Load Raw Data
    raw_values = read_integers(path)
    if not raw_values:
        print("No data found.")
        sys.exit(1)

    # === NEW CODE TO CROP DATA ===
    TARGET_TIME_SECONDS = 8.1
    # Calculate the number of samples needed for exactly 8 seconds at 81.7 Hz
    num_samples_to_keep = int(TARGET_TIME_SECONDS * ORIGINAL_FS)
    
    # Crop the list to contain only the first 8.0 seconds worth of samples
    raw_values = raw_values[:num_samples_to_keep]
    print(f"Cropped raw data to {TARGET_TIME_SECONDS} seconds ({num_samples_to_keep} samples).")
    # B. Apply Your Filters (Cleaning)
    # Using your specific filter chain that worked well
    print("Filtering signal...")
    v_dc = DC_filter(raw_values, fs=ORIGINAL_FS, cutoff=0.5)
    # Note: Skipping the LP filter since 125Hz > Nyquist of 81Hz (it does nothing)
    clean_ppg = moving_average_filter(v_dc, window_size=5)

    # C. Resampling (Crucial Step: 81.7 Hz -> 128 Hz)
    print(f"Resampling from {ORIGINAL_FS} Hz to {TARGET_FS} Hz...")
    num_samples_target = int(len(clean_ppg) * (TARGET_FS / ORIGINAL_FS))
    resampled_ppg = resample(clean_ppg, num_samples_target)

    # D. Prepare Batches for Model
    # The model expects chunks of 512 (4 seconds)
    # We will step forward by 512 samples (non-overlapping for simplicity)
    
    model = load_cardiogan()
    ecg_reconstructed = []
    
    print("Running Inference...")
    for i in range(0, len(resampled_ppg) - SEGMENT_LEN, SEGMENT_LEN):
        # 1. Slice
        segment = resampled_ppg[i : i + SEGMENT_LEN]
        
        # 2. Normalize Segment to [-1, 1] (Required by GAN)
        seg_min = np.min(segment)
        seg_max = np.max(segment)
        if seg_max - seg_min == 0:
            segment_norm = segment # Avoid div by zero
        else:
            segment_norm = 2 * ((segment - seg_min) / (seg_max - seg_min)) - 1
            
        # 3. Reshape for TF: (1, 512, 1)
        segment_input = segment_norm.reshape(1, SEGMENT_LEN, 1)
        
        # 4. Predict
        # training=False turns off dropout
        fake_ecg = model(segment_input, training=False)
        
        # 5. Store result (flatten to 1D array)
        ecg_reconstructed.extend(fake_ecg.numpy().flatten())

    # E. Visualization
    if not ecg_reconstructed:
        print("Not enough data to generate ECG (Need at least 4 seconds).")
        sys.exit(1)

    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # Plot Input PPG
    # Create time axis for original signal
    t_ppg = np.linspace(0, len(resampled_ppg)/TARGET_FS, len(resampled_ppg))
    ax1.plot(t_ppg, resampled_ppg, color='#1f77b4', label='Resampled PPG')
    ax1.set_title("Input: Processed PPG (128 Hz)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Plot Output ECG
    # Create time axis for ECG
    t_ecg = np.linspace(0, len(ecg_reconstructed)/TARGET_FS, len(ecg_reconstructed))
    ax2.plot(t_ecg, ecg_reconstructed, color='#d62728', label='Synthesized ECG')
    ax2.set_title("Output: Reconstructed ECG (CardioGAN)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (Normalized)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
