import matplotlib.pyplot as plt
import numpy as np
import time
import serial
from collections import deque

# 定義資料緩衝
class PlotData:
    def __init__(self, max_entries=30):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)

# 初始化繪圖
fig, (ax, ax2) = plt.subplots(2, 1)
line,  = ax.plot(np.random.randn(100))
line2, = ax2.plot(np.random.randn(100))
plt.show(block=False)
plt.setp(line2, color='r')

PData = PlotData(500)
ax.set_ylim(0, 20)
ax2.set_ylim(-10, 10)  # 改一下範圍，因為濾掉 DC 後資料中心在 0

print('plotting data...')
strPort = 'COM5'
ser = serial.Serial(strPort, 115200)
ser.flush()



start = time.time()

# 高通濾波器參數
alpha = 0.95  # 調整這個值改變濾波強度 (0.9~0.99)
prev_x = 0.0
prev_y = 0.0

while True:
    for ii in range(10):
        try:
            raw = ser.readline().strip()
            if not raw:
                continue
            data = float(raw)

            # --- 高通濾波 (去除 DC 分量) ---
            y = alpha * (prev_y + data - prev_x)
            prev_x = data
            prev_y = y

            # 加入繪圖資料
            PData.add(time.time() - start, y)
        except Exception as e:
            # print("Read error:", e)
            pass

    # 更新畫面
    if len(PData.axis_x) > 0:
        ax.set_xlim(PData.axis_x[0], PData.axis_x[-1])
        ax2.set_xlim(PData.axis_x[0], PData.axis_x[-1])
        line.set_xdata(PData.axis_x)
        line.set_ydata(PData.axis_y)
        line2.set_xdata(PData.axis_x)
        line2.set_ydata(PData.axis_y)
        fig.canvas.draw()
        fig.canvas.flush_events()
