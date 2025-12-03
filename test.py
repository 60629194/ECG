
import matplotlib.pyplot as plt
import numpy as np
import time
import serial
from collections import deque
from scipy.signal import butter, filtfilt

# Butterworth low-pass filter design
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# Class for managing data plotting
class AppendMode:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)

class UpdateMode:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def append(self, x, y, i):
        self.axis_x[i] = x
        self.axis_y[i] = y


# Set up the plot
fig, (ax, ax2) = plt.subplots(2, 1)
line, = ax.plot(np.random.randn(100))
line2, = ax2.plot(np.random.randn(100))
plt.show(block=False)
plt.setp(line2, color='r')

PData = AppendMode(500)
AData = UpdateMode(500)
ax.set_ylim(-10, 10)
ax2.set_ylim(-10, 10)

print('Plotting data...')

# Serial port setup
strPort = 'com3'
ser = serial.Serial(strPort, 115200)
ser.flush()
start = time.time()

# Low-pass filter parameters
fs = 50  # Sample rate (Hz)
cutoff = 4  # Cutoff frequency (Hz)
alpha = 0.95
prevx = 0
prevy = 0

# Filter coefficients (pre-compute)
b, a = butter_lowpass(cutoff, fs)

# Accumulating buffer for filtering

signal_buffer = deque(maxlen=50)  # Buffer size larger than 15
while True:
    for ii in range(10):
        try:
            raw = ser.readline().strip()
            if not raw:
                continue

            data = float(raw)

            # Apply the DC filter (which you are already using)
            y = alpha * (prevy + data - prevx)
            prevx = data
            prevy = y

            # Add the filtered value to the buffer
            signal_buffer.append(y)

            # Only apply the low-pass filter when the buffer is filled with enough data
                            
            # Ensure the buffer has enough data (e.g., 30 or more samples)
            if len(signal_buffer) >= 30:
                # Apply the low-pass filter to the buffer
                filtered_y = butter_lowpass_filter(list(signal_buffer), cutoff, fs)

                # Get the latest filtered value from the filtered buffer
                filtered_value = filtered_y[-1]
                # Add the filtered value to the plot data
                PData.add(time.time() - start, filtered_value)

        except Exception as e:
            print("Error:", e)
            pass

    if len(PData.axis_x) > 1:
        ax.set_xlim(PData.axis_x[0], PData.axis_x[0] + 5)
        ax2.set_xlim(PData.axis_x[0], PData.axis_x[0] + 5)
        line.set_xdata(PData.axis_x)
        line.set_ydata(PData.axis_y)
        line2.set_xdata(PData.axis_x)
        line2.set_ydata(PData.axis_y)
        fig.canvas.draw()
        fig.canvas.flush_events()
