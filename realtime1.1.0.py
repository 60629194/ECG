import matplotlib.pyplot as plt
import numpy as np
import time
import serial
from collections import deque

# 模式設定
DCcutoff = 0.5  # Hz
MODE = "diagnostic"  # "diagnostic" or "monitoring"
AVERAGE_WINDOW = 5  # 移動平均視窗大小

if MODE == "diagnostic":
    HPcutoff = 0.67  # Hz
    LPcutoff = 125.0  # Hz
else:
    HPcutoff = 0.5  # Hz
    LPcutoff = 70.0  # Hz

# 定義資料緩衝
class PlotData:
    def __init__(self, max_entries=500):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)

PData = PlotData(500)

# 初始化繪圖（ECG + FFT）
fig, (ax_ecg, ax_fft) = plt.subplots(2, 1, figsize=(10, 8))
line_ecg, = ax_ecg.plot([0], [0], color='r')
line_fft, = ax_fft.plot([0], [0])

peak_dot, = ax_fft.plot([], [], 'o')        # 標示最大峰值的點
peak_text = ax_fft.text(0, 0, "", color='blue')
heartrate_text = ax_fft.text(20, 800, "", color='green') #(x, y) 位置可調整

ax_ecg.set_ylim(-2, 3)
ax_fft.set_xlim(0, 125)   # 頻率範圍（依你的 LPcutoff 調整）
ax_fft.set_ylim(0, 1000)  # 你可以之後依結果調整
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Magnitude")

plt.show(block=False)

# 串口設定
strPort = 'COM5'
ser = serial.Serial(strPort, 115200)
ser.flush()

start = time.time()
last_time = time.time()

# 濾波函數
def DC_filter(values, fs=250.0, cutoff=0.5):
    from math import pi
    vals = list(values)
    if not vals: return []
    rc = 1.0 / (2.0 * pi * cutoff)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)
    out = [0.0] * len(vals)
    prev_out = out[0]
    prev_in = float(vals[0])
    for i in range(1, len(vals)):
        x = float(vals[i])
        y = alpha * (prev_out + x - prev_in)
        out[i] = y
        prev_out = y
        prev_in = x
    return out

def highpass_filter(values, fs=250.0, cutoff=0.67):
    from math import pi
    vals = list(values)
    if not vals: return []
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * pi * cutoff)
    alpha = dt / (rc + dt)
    out = [0.0] * len(vals)
    y_lp = float(vals[0])
    out[0] = 0.0
    for i in range(1, len(vals)):
        x = float(vals[i])
        y_lp = y_lp + alpha * (x - y_lp)
        out[i] = x - y_lp
    return out

def lowpass_filter(values, fs=250.0, cutoff=40.0):
    from math import pi
    vals = list(values)
    if not vals: return []
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * pi * cutoff)
    alpha = dt / (rc + dt)
    out = [0.0] * len(vals)
    y = float(vals[0])
    out[0] = y
    for i in range(1, len(vals)):
        x = float(vals[i])
        y = y + alpha * (x - y)
        out[i] = y
    return out

def moving_average_filter(values, window_size=5):
    vals = list(values)
    if not vals: return []
    window_size = max(1, int(window_size))
    out = []
    for i in range(len(vals)):
        start = max(0, i - window_size + 1)
        avg = sum(vals[start:i+1]) / len(vals[start:i+1])
        out.append(avg)
    return out

#頻域函數
def compute_fft(values, fs):
    v = np.array(values, dtype=float)
    n = len(v)
    fft_vals = np.fft.rfft(v)
    fft_freqs = np.fft.rfftfreq(n, 1.0/fs)
    magnitude = np.abs(fft_vals)
    return fft_freqs, magnitude


# 主迴圈
raw_data = []  # 用於 IIR 濾波
while True:
    # 讀取新資料
    N = 10
    for _ in range(N):
        try:
            raw = ser.readline().strip()
            if not raw:
                continue
            data = float(raw)
            raw_data.append(data)
            if len(raw_data) > 2000:  # 限制長度
                raw_data = raw_data[-2000:]
        except:
            pass

    # 計算動態取樣頻率
    now = time.time()
    elapsed = now - last_time
    last_time = now
    FS = N / elapsed if elapsed > 0 else 100
    dt = 1.0 / FS

    # 如果資料不足就跳過
    if len(raw_data) < 5:
        continue

    # 濾波
    filtered = DC_filter(raw_data, fs=FS, cutoff=DCcutoff)
    filtered = highpass_filter(filtered, fs=FS, cutoff=HPcutoff)
    filtered = lowpass_filter(filtered, fs=FS, cutoff=LPcutoff)
    filtered = moving_average_filter(filtered, window_size=AVERAGE_WINDOW)

    # 取最新 N 筆
    new_samples = filtered[-N:]
    now_t = time.time() - start
    t0 = now_t - dt * (len(new_samples) - 1)
    for i, y in enumerate(new_samples):
        PData.add(t0 + i*dt, y)

    # === 更新 ECG ===
    ax_ecg.set_xlim(PData.axis_x[0], PData.axis_x[-1])
    line_ecg.set_xdata(PData.axis_x)
    line_ecg.set_ydata(PData.axis_y)

    # === 計算 FFT（使用最新 500 筆資料） ===
    fft_window = 100
    if len(raw_data) > fft_window:
        segment = np.array(raw_data[-fft_window:], dtype=float)
    else:
        segment = np.array(raw_data, dtype=float)

    # FFT 前扣平均值避免假 DC
    segment -= np.mean(segment)

    freqs, mags = compute_fft(segment, FS)

    # === 更新 FFT 曲線 ===
    line_fft.set_xdata(freqs)
    line_fft.set_ydata(mags)
    ax_fft.set_xlim(0, FS/2)

    # === 找最高峰 (排除 <1 Hz) ===
    freqs_arr = np.array(freqs)
    mags_arr = np.array(mags)
    valid = freqs_arr >= 0.5    # 避免 DC 峰干擾  ->heartbeat 30 bpm = 0.5 Hz

    if np.any(valid):
        idx = np.argmax(mags_arr[valid])
        peak_freq = freqs_arr[valid][idx]
        peak_mag = mags_arr[valid][idx]

        # 點
        peak_dot.set_xdata([peak_freq])
        peak_dot.set_ydata([peak_mag])

        # 文字
        peak_text.set_position((peak_freq, peak_mag))
        peak_text.set_text(f"{peak_freq:.2f} Hz")
        heartrate_text.set_text(f"Heart Rate: {peak_freq*60:.1f} bpm")



    # === 更新 FFT 曲線 ===
    line_fft.set_xdata(freqs)
    line_fft.set_ydata(mags)
    ax_fft.set_xlim(0, FS/2)    # Nyquist

    # === 重繪兩張圖 ===
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)

