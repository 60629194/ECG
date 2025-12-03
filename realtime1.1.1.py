import matplotlib.pyplot as plt
import numpy as np
import time
import serial
from collections import deque

# ==========================================
# 1. 參數與設定
# ==========================================
DCcutoff = 0.5  # Hz
MODE = "diagnostic"  # "diagnostic" or "monitoring"
AVERAGE_WINDOW = 5  # 移動平均視窗大小

if MODE == "diagnostic":
    HPcutoff = 0.67   # Hz
    LPcutoff = 125.0  # Hz
else:
    HPcutoff = 0.5    # Hz
    LPcutoff = 70.0   # Hz

# ==========================================
# 2. 資料結構與繪圖
# ==========================================
class PlotData:
    def __init__(self, max_entries=500):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)

PData = PlotData(500)

# 初始化繪圖（ECG + FFT）
# 這裡加了 plt.ion() 確保互動模式開啟
plt.ion() 
fig, (ax_ecg, ax_fft) = plt.subplots(2, 1, figsize=(10, 8))

# ECG 線條
line_ecg, = ax_ecg.plot([], [], color='r', lw=1.2) 

# FFT 圖形
line_fft, = ax_fft.plot([], [], color='k', lw=1)
peak_dot, = ax_fft.plot([], [], 'o', color='red')        # 標示峰值
peak_text = ax_fft.text(0, 0, "", color='blue', fontsize=12)
heartrate_text = ax_fft.text(0.7, 0.9, "", transform=ax_fft.transAxes, 
                             color='green', fontsize=12, fontweight='bold') # 改用相對座標固定位置

# 設定軸範圍
ax_ecg.set_title("Real-time ECG")
ax_ecg.set_ylim(-2, 3) # 這裡之後可以根據訊號大小動態調整
ax_ecg.set_ylabel("Amplitude")

ax_fft.set_title("Frequency Spectrum (FFT)")
ax_fft.set_xlim(0, 40)   # 目測
ax_fft.set_ylim(0, 1000)       # 初始範圍
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Magnitude")

plt.tight_layout()
plt.show(block=False)

# ==========================================
# 3. port 設定
# ==========================================
strPort = 'COM5'
try:
    ser = serial.Serial(strPort, 115200)
    ser.flush() # 清空緩衝區
except Exception as e:
    print(f"fail to open serial port: {e}")
    exit()

# ==========================================
# 4. filter 函式
# ==========================================
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
    # 簡單優化：直接用卷積計算會比迴圈快，但為了保持你的結構，維持原樣
    # 但這裡稍微修一下邊界處理，避免錯誤
    for i in range(len(vals)):
        start = max(0, i - window_size + 1)
        segment = vals[start:i+1]
        avg = sum(segment) / len(segment)
        out.append(avg)
    return out

def compute_fft(values, fs):
    v = np.array(values, dtype=float)
    n = len(v)
    fft_vals = np.fft.rfft(v)
    fft_freqs = np.fft.rfftfreq(n, 1.0/fs)
    magnitude = np.abs(fft_vals)
    return fft_freqs, magnitude

# ==========================================
# 5. main loop
# ==========================================

raw_data = []         # 原始資料暫存區 (未濾波)
start_time = time.time()
last_process_time = time.time() # 用來計算兩次繪圖間的時間差


# === 修改這裡：定義一個連續的時間計數器 ===
current_t = 0.0  # 這是我們的內部時鐘，從 0 開始
FS = 81.6        # 你測出來的數值
dt = 1.0 / FS    # 每一步的時間間隔


print("開始監測...")

try:
    while True:
        # --- [關鍵修改] 貪婪讀取：把 buffer 裡的資料一次清空 ---
        bytes_waiting = ser.in_waiting
        if bytes_waiting == 0:
            # 沒資料就稍微休息，避免 CPU 燒起來
            time.sleep(0.001) 
            continue
            
        # 讀取這一批次的所有資料
        new_batch = []
        # 為了保險，設定一個單次最大讀取量 (例如 500 筆)，避免卡死
        # 但通常 ser.readlines() 或迴圈讀取 in_waiting 即可
        read_count = 0
        while ser.in_waiting > 0 and read_count < 200: 
            try:
                line = ser.readline().strip()
                if line:
                    val = float(line)
                    new_batch.append(val)
                read_count += 1
            except ValueError:
                continue # 忽略壞掉的資料
        
        if not new_batch:
            continue

        # 將新資料加入總暫存區
        raw_data.extend(new_batch)
        
        # 限制 raw_data 長度，避免記憶體爆炸和濾波運算過慢
        # 因為你的濾波器是每次重算整個 list，保持在 2000 左右是極限了
        if len(raw_data) > 2000:
            raw_data = raw_data[-2000:]

        
        # 資料量太少就不畫圖
        if len(raw_data) < 50:
            continue

        # --- 訊號處理 (因為要用你的函式，所以每次都全量運算) ---
        # 注意：這一步比較耗時，資料量越大越慢
        filtered = DC_filter(raw_data, fs=FS, cutoff=DCcutoff)
        filtered = highpass_filter(filtered, fs=FS, cutoff=HPcutoff)
        filtered = lowpass_filter(filtered, fs=FS, cutoff=LPcutoff)
        filtered = moving_average_filter(filtered, window_size=AVERAGE_WINDOW)

        # --- 更新繪圖資料 ---
        # 我們只取最新處理好的 N 筆資料放入 PlotData
        # 這裡取 len(new_batch) 筆，確保畫圖速度跟上讀取速度
        num_new = len(new_batch)
        samples_to_plot = filtered[-num_new:]
        
        # 不要用 time.time() 了，那是不準的！
        # 直接根據資料點數，嚴格地增加時間
        for val in samples_to_plot:
            current_t += dt       # 時間往前走一小步
            PData.add(current_t, val)

        # --- 繪圖更新 ---
        # 1. ECG
        if len(PData.axis_x) > 1:
            line_ecg.set_xdata(PData.axis_x)
            line_ecg.set_ydata(PData.axis_y)
            ax_ecg.set_xlim(PData.axis_x[0], PData.axis_x[-1])
            
            # 動態調整 Y 軸範圍 (看最近的資料)
            y_data = list(PData.axis_y)
            if y_data: # 確保有資料
                y_min, y_max = min(y_data), max(y_data)
                if y_max != y_min: # 避免全為 0 出錯
                    margin = (y_max - y_min) * 0.1
                    ax_ecg.set_ylim(y_min - margin, y_max + margin)

        # 2. FFT (每隔幾次迴圈更新一次就好，不用每次都算，省資源)
        # 這裡設定：每處理 10 次數據更新一次 FFT，或者你可以每次都更
        if len(raw_data) >= 163: # 至少有2秒資料才做 FFT
            # 取最近的 N 點做 FFT (例如 400 點)
            n_fft = 400
            if len(filtered) < n_fft:
                segment = np.array(filtered)
            else:
                segment = np.array(filtered[-n_fft:])
            
            # 扣除平均值 (去直流)
            segment -= np.mean(segment)
            
            freqs, mags = compute_fft(segment, FS)
            
            line_fft.set_xdata(freqs)
            line_fft.set_ydata(mags)
            
            # 找峰值 (心率)
            valid_mask = (freqs >= 0.6) & (freqs <= 3.0) # 36 BPM ~ 180 BPM
            if np.any(valid_mask):
                target_mags = mags[valid_mask]
                target_freqs = freqs[valid_mask]
                
                peak_idx = np.argmax(target_mags)
                p_freq = target_freqs[peak_idx]
                p_mag = target_mags[peak_idx]
                
                peak_dot.set_data([p_freq], [p_mag])
                peak_text.set_position((p_freq, p_mag))
                peak_text.set_text(f"{p_freq:.2f}Hz")
                
                bpm = p_freq * 60
                heartrate_text.set_text(f"HR: {bpm:.1f} BPM")
            
            # 動態調整 FFT Y軸
            if np.max(mags) > 10:
                ax_fft.set_ylim(0, np.max(mags[1:]) * 1.2) # 忽略 DC

        # --- 刷新畫面 ---
        # pause 時間設極短，讓介面有反應即可
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # 這裡不使用 plt.pause()，因為它比較慢，改用 flush_events + sleep 
        # 但為了保險起見，如果是標準 matplotlib 後端：
        #plt.pause(0.001)

except KeyboardInterrupt:
    ser.close()
    print("\nSerial port closed.")