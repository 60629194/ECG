import matplotlib.pyplot as plt
import numpy as np
import time
import serial

# 高通濾波器參數
alpha = 0.95
prev_x = 0.0
prev_y = 0.0

# 儲存資料
raw_data = []
filtered_data = []
timestamps = []

# 設定序列埠
strPort = 'COM5'
ser = serial.Serial(strPort, 115200)
ser.flush()

print("Reading data from serial...")

start = time.time()
duration = 10  # 設定讀取時間（秒）

while time.time() - start < duration:
    try:
        raw = ser.readline().strip()
        if not raw:
            continue
        data = float(raw)
        raw_data.append(data)

        # 高通濾波
        y = alpha * (prev_y + data - prev_x)
        prev_x = data
        prev_y = y
        filtered_data.append(y)
        timestamps.append(time.time() - start)
    except Exception as e:
        pass

ser.close()
print("Finished reading. Total samples:", len(raw_data))
with open("raw_data.txt", "w") as f:
    for value in raw_data:
        f.write(f"{value}\n")


# 繪圖
fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
ax.plot(timestamps, raw_data, label='Raw Data')
ax.set_title('Raw Data')
ax.set_ylim(0, 700)
ax.legend()

ax2.plot(timestamps, filtered_data, color='r', label='High-pass Filtered')
ax2.set_title('Filtered Data (DC Removed)')
ax2.set_ylim(-10, 10)
ax2.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()