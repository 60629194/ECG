import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return np.array(data)

def main():
    file_path = "data_36in60.txt"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    data = load_data(file_path)
    fs = len(data) / 60.0  # Assuming 60 seconds recording
    time = np.arange(len(data)) / fs

    # Linear Detrend
    detrended_data = signal.detrend(data)
    
    # Calculate the trend line
    trend = data - detrended_data

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, data, label='Raw Data')
    plt.plot(time, trend, label='Trend (Linear)', color='red', linestyle='--')
    plt.title(f'Raw PPG Data with Trend - {file_path}')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, detrended_data, label='Detrended Data', color='green')
    plt.title('Detrended PPG Data')
    plt.legend()
    plt.grid(True)
    
    # Also show the bandpass filtered version used in analysis
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 5.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, filtered_data, label='Bandpass Filtered (0.5-5.0 Hz)', color='orange')
    plt.title('Bandpass Filtered Data (Used for Analysis)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('trend_analysis.png')
    print("Trend analysis plot saved to trend_analysis.png")

if __name__ == "__main__":
    main()
