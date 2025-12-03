import tkinter as tk
from tkinter import font
import math
import collections
import serial
import numpy as np
import time
import threading

# --- Configuration and Colors ---
COLORS = {
    "BG_MAIN": "#050814",
    "BG_PANEL": "#0c1229",
    "NEON_CYAN": "#00fff9",
    "NEON_GREEN": "#39ff14",
    "NEON_RED": "#ff3333",     
    "TEXT_MAIN": "#e6f0ff",
    "TEXT_DIM": "#6b7c99",
    "GRID_LINE": "#1a264f"
}

# --- Settings ---
GRAPH_HEIGHT = 400
GRAPH_WIDTH = 800

# [關鍵修改 1] 這裡改小一點，波形就會被拉開，不會擠在一起
DATA_POINTS = 100       # 建議改成 100 ~ 150
REFRESH_RATE_MS = 20
SERIAL_PORT = 'COM5'    # <--- 記得改成你的 Port
BAUD_RATE = 115200
FS = 81.6               
SCALE_FACTOR = 100      

# --- Filter Parameters ---
DC_CUTOFF = 0.5
MODE = "diagnostic"

if MODE == "diagnostic":
    HP_CUTOFF = 0.5
    LP_CUTOFF = 150.0  # Hz
elif MODE == "monitoring":
    HP_CUTOFF = 0.67    # Hz
    LP_CUTOFF = 40.0   # Hz

AVG_WINDOW = 5

class FuturisticPPGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("BIO-METRIC SENSOR INTERFACE // DUAL MODE")
        self.root.geometry("1024x600")
        self.root.configure(bg=COLORS["BG_MAIN"])
        
        # --- Data Structures ---
        self.raw_buffer = collections.deque(maxlen=600) 
        self.display_buffer = collections.deque([GRAPH_HEIGHT/2] * DATA_POINTS, maxlen=DATA_POINTS)
        self.fft_buffer = [] # 用來存更長的數據做 FFT 計算

        
        # [新增] 掃描模式專用的固定緩衝區 (初始化為中間值)
        self.sweep_buffer = [GRAPH_HEIGHT/2] * DATA_POINTS
        self.sweep_idx = 0 # 目前寫入的位置 (0 ~ DATA_POINTS-1)

        # --- View Control ---
        self.graph_mode = "WAVE"     # "WAVE" (捲動) 或 "SWEEP" (掃描)
        self.fft_enabled = False     # 頻譜圖開關
        self.filter_enabled = True   # 濾波器開關
        
        # [掃描模式專用]
        self.sweep_buffer = [GRAPH_HEIGHT/2] * DATA_POINTS
        self.sweep_idx = 0
        self.line_tail = None

        # Individual filter toggles
        self.hp_enabled = True
        self.lp_enabled = True
        self.ma_enabled = True
        
        # --- Serial Connection ---
        self.ser = None
        self.is_running = True
        self.connect_serial()

        # --- Fonts ---
        self.header_font = font.Font(family="Consolas", size=14, weight="bold")
        self.digit_font = font.Font(family="Courier New", size=28, weight="bold")
        self.label_font = font.Font(family="Calibri", size=10)
        self.btn_font = font.Font(family="Consolas", size=11, weight="bold")

        self.setup_gui()
        
        # Threads
        self.read_thread = threading.Thread(target=self.read_serial_loop, daemon=True)
        self.read_thread.start()

        self.animate_graph()

    def connect_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"[{SERIAL_PORT}] Connected. ")
        except Exception as e:
            print(f"Serial Error: {e}")
            self.ser = None

    def create_neon_border(self, parent, accent_color, padding=2):
        outer = tk.Frame(parent, bg=accent_color, padx=padding, pady=padding)
        inner = tk.Frame(outer, bg=COLORS["BG_PANEL"])
        inner.pack(fill="both", expand=True)
        return outer, inner

    def setup_gui(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # 1. Header
        header_frame = tk.Frame(self.root, bg=COLORS["BG_MAIN"], height=40)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10,5))
        
        self.status_label = tk.Label(header_frame, text="STATUS: INITIALIZING...", 
                 bg=COLORS["BG_MAIN"], fg=COLORS["NEON_CYAN"], font=self.label_font)
        self.status_label.pack(side="left")
        
        tk.Label(header_frame, text=f"SOURCE: {SERIAL_PORT}", 
                 bg=COLORS["BG_MAIN"], fg=COLORS["TEXT_DIM"], font=self.label_font).pack(side="right")

        # 2. Sidebar
        left_sidebar_outer, self.left_sidebar = self.create_neon_border(self.root, COLORS["NEON_CYAN"])
        left_sidebar_outer.grid(row=1, column=0, sticky="ns", padx=(10, 5), pady=5)
        self.left_sidebar.configure(width=180, height=500)
        self.left_sidebar.pack_propagate(False)

        self.bpm_label = self.add_stat_module(self.left_sidebar, "HEART RATE (BPM)", COLORS["NEON_GREEN"])
        self.freq_label = self.add_stat_module(self.left_sidebar, "PEAK FREQ (Hz)", COLORS["NEON_CYAN"])
        self.add_decorative_text(self.left_sidebar)

        # 3. Graph Area
        graph_outer, graph_inner = self.create_neon_border(self.root, COLORS["NEON_CYAN"], padding=3)
        graph_outer.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.canvas = tk.Canvas(graph_inner, bg="black", 
                                height=GRAPH_HEIGHT, width=GRAPH_WIDTH, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)

        # 4. Bottom Control Bar [關鍵修改 2：加入按鈕]
        bottom_bar_outer, bottom_bar = self.create_neon_border(self.root, COLORS["TEXT_DIM"], padding=1)
        bottom_bar_outer.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(5,10))
        
        # --- 按鈕區 ---
        
        # [按鈕 1] FFT 獨立開關
        self.fft_btn = tk.Button(bottom_bar, text="[ FFT: OFF ]", 
                                    bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], 
                                    font=self.btn_font, relief="flat", 
                                    activebackground=COLORS["GRID_LINE"],
                                    activeforeground=COLORS["NEON_CYAN"],
                                    command=self.toggle_fft)
        self.fft_btn.pack(side="right", padx=5, pady=2)

        # [按鈕 2] 波形模式切換 (WAVE / SWEEP)
        self.mode_btn = tk.Button(bottom_bar, text="[ VIEW: WAVE ]", 
                                    bg=COLORS["BG_PANEL"], fg=COLORS["NEON_CYAN"], 
                                    font=self.btn_font, relief="flat", 
                                    activebackground=COLORS["GRID_LINE"],
                                    activeforeground=COLORS["NEON_RED"],
                                    command=self.toggle_graph_mode)
        self.mode_btn.pack(side="right", padx=5, pady=2)

        # Individual filter buttons: High-pass, Low-pass, Moving Average
        self.hp_btn = tk.Button(bottom_bar, text="[ Hi-Pass: ON ]",
                        bg=COLORS["BG_PANEL"], fg=COLORS["NEON_GREEN"],
                        font=self.btn_font, relief="flat", activebackground=COLORS["GRID_LINE"],
                        command=self.toggle_hp)
        self.hp_btn.pack(side="right", padx=6, pady=2)

        self.lp_btn = tk.Button(bottom_bar, text="[ Lo-Pass: ON ]",
                        bg=COLORS["BG_PANEL"], fg=COLORS["NEON_GREEN"],
                        font=self.btn_font, relief="flat", activebackground=COLORS["GRID_LINE"],
                        command=self.toggle_lp)
        self.lp_btn.pack(side="right", padx=6, pady=2)

        self.ma_btn = tk.Button(bottom_bar, text="[ SMOOTH: ON ]",
                        bg=COLORS["BG_PANEL"], fg=COLORS["NEON_GREEN"],
                        font=self.btn_font, relief="flat", activebackground=COLORS["GRID_LINE"],
                        command=self.toggle_ma)
        self.ma_btn.pack(side="right", padx=6, pady=2)

        self.console_msg = tk.Label(bottom_bar, text=">> SYSTEM READY.", 
                 bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], font=self.label_font, anchor="w")
        self.console_msg.pack(side="left", fill="x", padx=5)

    def toggle_graph_mode(self):
        """切換波形顯示模式：WAVE (捲動) <-> SWEEP (掃描)"""
        if self.graph_mode == "WAVE":
            self.graph_mode = "SWEEP"
            self.mode_btn.config(text="[ VIEW: SWEEP ]", fg=COLORS["NEON_RED"])
        else:
            self.graph_mode = "WAVE"
            self.mode_btn.config(text="[ VIEW: WAVE ]", fg=COLORS["NEON_CYAN"])
        
        # 如果目前沒有在看 FFT，就立即刷新畫布狀態
        if not self.fft_enabled:
            self.reset_canvas_for_mode()

    def toggle_fft(self):
        """獨立切換 FFT 頻譜視圖"""
        self.fft_enabled = not self.fft_enabled
        
        if self.fft_enabled:
            self.fft_btn.config(text="[ FFT: ON ]", fg=COLORS["NEON_GREEN"])
            # 清除畫布，準備畫 FFT
            self.canvas.delete("all")
            self.line_id = None
            self.line_tail = None
        else:
            self.fft_btn.config(text="[ FFT: OFF ]", fg=COLORS["TEXT_DIM"])
            # 關閉 FFT，回到波形顯示
            self.reset_canvas_for_mode()

    def reset_canvas_for_mode(self):
        """根據目前的 graph_mode 重置畫布線條"""
        self.canvas.delete("all")
        self.line_id = None
        self.line_tail = None
        self.draw_grid()

        if self.graph_mode == "WAVE":
            self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
        elif self.graph_mode == "SWEEP":
            self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
            self.line_tail = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
            self.sweep_idx = 0 # 重置掃描位置比較好看

    def toggle_hp(self):
        """Toggle high-pass filter on/off"""
        self.hp_enabled = not getattr(self, 'hp_enabled', True)
        if self.hp_enabled:
            self.hp_btn.config(text="[ Hi-Pass: ON ]", fg=COLORS["NEON_GREEN"])
            self.console_msg.config(text=">> HP FILTER ENABLED.")
        else:
            self.hp_btn.config(text="[ Hi-Pass: OFF ]", fg=COLORS["NEON_RED"])
            self.console_msg.config(text=">> HP FILTER DISABLED.")

    def toggle_lp(self):
        """Toggle low-pass filter on/off"""
        self.lp_enabled = not getattr(self, 'lp_enabled', True)
        if self.lp_enabled:
            self.lp_btn.config(text="[ Lo-Pass: ON ]", fg=COLORS["NEON_GREEN"])
            self.console_msg.config(text=">> LP FILTER ENABLED.")
        else:
            self.lp_btn.config(text="[ Lo-Pass: OFF ]", fg=COLORS["NEON_RED"])
            self.console_msg.config(text=">> LP FILTER DISABLED.")

    def toggle_ma(self):
        """Toggle moving-average filter on/off"""
        self.ma_enabled = not getattr(self, 'ma_enabled', True)
        if self.ma_enabled:
            self.ma_btn.config(text="[ SMOOTH: ON ]", fg=COLORS["NEON_GREEN"])
            self.console_msg.config(text=">> MA FILTER ENABLED.")
        else:
            self.ma_btn.config(text="[ SMOOTH: OFF ]", fg=COLORS["NEON_RED"])
            self.console_msg.config(text=">> MA FILTER DISABLED.")

    def add_stat_module(self, parent, title, glow_color):
        frame = tk.Frame(parent, bg=COLORS["BG_PANEL"], pady=15)
        frame.pack(fill="x", padx=5)
        tk.Label(frame, text=title, bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], 
                 font=self.label_font, anchor="w").pack(fill="x")
        value_label = tk.Label(frame, text="--", bg=COLORS["BG_PANEL"], fg=glow_color, font=self.digit_font)
        value_label.pack(anchor="e")
        tk.Frame(frame, bg=COLORS["GRID_LINE"], height=2).pack(fill="x", pady=(5,0))
        return value_label

    def add_decorative_text(self, parent):
        frame = tk.Frame(parent, bg=COLORS["BG_PANEL"], pady=20)
        frame.pack(fill="both", expand=True, padx=5)
        txt = "MODE SELECT\nMANUAL\n\nPROCESSING\nREAL-TIME"
        tk.Label(frame, text=txt, bg=COLORS["BG_PANEL"], fg=COLORS["GRID_LINE"], 
                 font=self.label_font, justify="left", anchor="nw").pack(fill="both", expand=True)

    def draw_grid(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        # 避免視窗還沒縮放完成時拿到 1x1
        if w < 10: w = GRAPH_WIDTH
        if h < 10: h = GRAPH_HEIGHT

        # 1. 畫垂直線 (Vertical Lines)
        for i in range(0, w, 40):
            color = COLORS["GRID_LINE"]
            if (i % 160) == 0: color = COLORS["TEXT_DIM"]
            self.canvas.create_line(i, 0, i, h, fill=color, dash=(4, 4))
        
        # 2. 畫中間的基準線 (Center Line)
        center_y = h / 2
        self.canvas.create_line(0, center_y, w, center_y, fill=COLORS["TEXT_DIM"])

        # 3. 畫水平網格 (Horizontal Lines)
        # [修正] 這裡原本檢查 self.view_mode，現在改成 self.graph_mode
        # 而且不管是 WAVE 還是 SWEEP 模式，我們都讓它顯示網格
        if self.graph_mode in ["WAVE", "SWEEP"]: 
            for i in range(0, h, 40):
                if abs(i - center_y) < 5: continue
                self.canvas.create_line(0, i, w, i, fill=COLORS["GRID_LINE"], dash=(4, 4))
        else:
            # FFT 模式通常不需要橫線，因為它是頻率強度
            pass

    # --- Signal Processing ---
    def process_signal(self, data_list):
        # Normalize to zero-mean first (applies for both filtered and unfiltered paths)
        vals = np.array(data_list)
        vals = vals - np.mean(vals)

        # Apply filters conditionally (HP, LP, MA)
        vals_list = vals.tolist()

        if getattr(self, 'hp_enabled', True):
            vals_list = self.highpass_filter(vals_list, FS, HP_CUTOFF)
        if getattr(self, 'lp_enabled', True):
            vals_list = self.lowpass_filter(vals_list, FS, LP_CUTOFF)
        if getattr(self, 'ma_enabled', True):
            vals_list = self.moving_average_filter(vals_list, AVG_WINDOW)

        return vals_list
        
    def lowpass_filter(self, vals, fs, cutoff):
        if len(vals) < 2: return vals
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff)
        alpha = dt / (rc + dt)
        out = [0.0] * len(vals)
        y = float(vals[0])
        out[0] = y
        for i in range(1, len(vals)):
            y = y + alpha * (vals[i] - y)
            out[i] = y
        return out

    def highpass_filter(self, vals, fs, cutoff):
        """
        IIR High-pass filter based on RC time constant.
        這就是切斷低頻漂移的刀。
        """
        if len(vals) < 2: return vals
        
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff)
        alpha = rc / (rc + dt)
        
        out = [0.0] * len(vals)
        out[0] = 0.0 # 初始值設為 0
        
        # 遞迴運算
        for i in range(1, len(vals)):
            # y[i] = alpha * (y[i-1] + x[i] - x[i-1])
            out[i] = alpha * (out[i-1] + vals[i] - vals[i-1])
            
        return out

    def moving_average_filter(self, vals, window_size):
        if len(vals) == 0: return []
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(vals, window, 'same').tolist()

    # --- Background Read Loop ---
    def read_serial_loop(self):
        while self.is_running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting:
                        # 讀取所有數據
                        chunk = self.ser.read(self.ser.in_waiting)
                        lines = chunk.split(b'\n')
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            try:
                                val = float(line.decode('utf-8'))
                                self.raw_buffer.append(val)
                                self.fft_buffer.append(val)
                            except ValueError:
                                continue
                    else:
                        time.sleep(0.005)
                except Exception as e:
                    print(f"Read Error: {e}")
                    time.sleep(1)
            else:
                time.sleep(1)

    # --- Main Animation Loop ---
    def animate_graph(self):
        # 1. Update Signal / FFT Logic
        if len(self.raw_buffer) >= AVG_WINDOW * 2:
            self.status_label.config(text="STATUS: DATA STREAMING", fg=COLORS["NEON_GREEN"])
            
            # --- 優先權判斷 ---
            if self.fft_enabled:
                # 如果 FFT 開啟，只畫 FFT
                self.draw_fft_mode()
            else:
                # 如果 FFT 關閉，根據模式畫波形
                if self.graph_mode == "WAVE":
                    self.draw_wave_mode()
                elif self.graph_mode == "SWEEP":
                    self.draw_sweep_mode()
        else:
            self.status_label.config(text="STATUS: WAITING FOR SIGNAL...", fg=COLORS["NEON_RED"])

        # 2. Background BPM check (Always runs)
        if not hasattr(self, 'tick_count'): self.tick_count = 0
        self.tick_count += 1
        if self.tick_count > 15:
            self.tick_count = 0
            self.calculate_bpm_update()

        self.root.after(REFRESH_RATE_MS, self.animate_graph)

    def draw_sweep_mode(self):
        """
        掃描模式 (Sweep/Erase Bar)：
        類似醫院心電圖，從左寫到右，寫滿後從左邊開始覆蓋舊資料。
        較舊的資料顏色會更淡 (Dimmer)。
        """
        # 1. 取得最新的一個點
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        latest_val = processed_data[-1]
        
        # 2. 映射 Y 座標
        y_pos = (GRAPH_HEIGHT / 2) - (latest_val * SCALE_FACTOR)
        y_pos = max(10, min(GRAPH_HEIGHT - 10, y_pos))
        
        # 3. 更新 Buffer (覆蓋舊資料)
        self.sweep_buffer[self.sweep_idx] = y_pos
        
        # 4. 移動游標
        self.sweep_idx += 1
        if self.sweep_idx >= DATA_POINTS:
            self.sweep_idx = 0  # 回到起點 (Wrap around)
            
        # --- 繪圖：完整的線，新資料亮，舊資料暗 ---
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        x_stretch = w / DATA_POINTS
        
        # 清除舊的線段
        self.canvas.delete("sweep_line")
        
        # 繪製每一段線段，根據年齡調整透明度 (或顏色亮度)
        # Tkinter 不直接支援透明度，所以用顏色漸層代替
        base_color_rgb = (0, 255, 249)  # NEON_CYAN RGB
        for i in range(DATA_POINTS - 1):
            # 計算這個點的年齡 (0 = 最新，DATA_POINTS-1 = 最舊)
            age = (self.sweep_idx - i) % DATA_POINTS
            
            # 根據年齡計算 Dimming 係數 (0 到 1)
            # 最新的點 (age 小) 最亮，最舊的點 (age 接近 DATA_POINTS) 最暗
            dim_factor = 1 - (age / DATA_POINTS)**3  # 使用平方根曲線讓變化更平滑
            
            # 調整顏色亮度 (線性插值到更暗的顏色)
            # 暗色背景是 #050814，所以我們往那個方向淡出
            dark_bg_rgb = (5, 8, 20)
            r = int(base_color_rgb[0] * dim_factor + dark_bg_rgb[0] * (1 - dim_factor))
            g = int(base_color_rgb[1] * dim_factor + dark_bg_rgb[1] * (1 - dim_factor))
            b = int(base_color_rgb[2] * dim_factor + dark_bg_rgb[2] * (1 - dim_factor))
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            x0 = i * x_stretch
            y0 = self.sweep_buffer[i]
            x1 = (i + 1) * x_stretch
            y1 = self.sweep_buffer[(i + 1) % DATA_POINTS]
            
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2, tags="sweep_line")

    def draw_wave_mode(self):
        # 處理數據
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        latest_val = processed_data[-1]
        
        # 座標映射
        y_pos = (GRAPH_HEIGHT / 2) - (latest_val * SCALE_FACTOR)
        y_pos = max(10, min(GRAPH_HEIGHT - 10, y_pos)) # Clamp
        self.display_buffer.append(y_pos)

        # 畫線
        w = self.canvas.winfo_width()
        x_stretch = w / DATA_POINTS # 自動計算每個點的間距
        
        coords = []
        for i, y_val in enumerate(self.display_buffer):
            coords.append(i * x_stretch)
            coords.append(y_val)
        
        if self.line_id is None:
             self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
             
        if len(coords) > 4:
             self.canvas.coords(self.line_id, *coords)

    def draw_fft_mode(self):
        # 畫頻譜圖 (直方圖效果)
        if len(self.fft_buffer) < 200: return
        
        # 取最近 N 點做 FFT
        data = list(self.fft_buffer)[-400:] 
        v = np.array(data)
        v = v - np.mean(v)
        
        fft_vals = np.fft.rfft(v)
        fft_freqs = np.fft.rfftfreq(len(v), 1.0/FS)
        magnitude = np.abs(fft_vals)

        # 只顯示 0 ~ 40Hz
        mask = (fft_freqs < 40)
        display_freqs = fft_freqs[mask]
        display_mags = magnitude[mask]

        # 清除舊的 Bar
        self.canvas.delete("fft_bar")

        # 繪製新的 Bar
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        num_bars = len(display_freqs)
        bar_width = w / num_bars
        
        # 簡單縮放 Magnitude 到高度
        max_mag = np.max(display_mags) if np.max(display_mags) > 0 else 1
        
        for i, mag in enumerate(display_mags):
            # 高度計算 (加一點 log 效果或是線性縮放)
            # 這裡用線性縮放，把最大值映射到畫面高度的 80%
            bar_h = (mag / max_mag) * (h * 0.8)
            
            x0 = i * bar_width
            y0 = h
            x1 = x0 + bar_width - 1
            y1 = h - bar_h
            
            # 根據頻率改變顏色 (酷炫效果)
            bar_color = COLORS["NEON_CYAN"]
            freq = display_freqs[i]
            if 0.6 <= freq <= 3.0: # 心跳範圍亮綠色
                bar_color = COLORS["NEON_GREEN"]

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=bar_color, outline="", tags="fft_bar")


    def calculate_bpm_update(self):
        # 這裡跟上面 FFT 繪圖邏輯分開，純粹為了更新數字
        if len(self.fft_buffer) < 200: return
        data = list(self.fft_buffer)[-400:]
        v = np.array(data)
        v = v - np.mean(v)
        fft_vals = np.fft.rfft(v)
        freqs = np.fft.rfftfreq(len(v), 1.0/FS)
        mags = np.abs(fft_vals)
        
        valid_mask = (freqs >= 0.6) & (freqs <= 3.0)
        if np.any(valid_mask):
            target_mags = mags[valid_mask]
            target_freqs = freqs[valid_mask]
            peak_idx = np.argmax(target_mags)
            p_freq = target_freqs[peak_idx]
            bpm = p_freq * 60
            
            self.bpm_label.config(text=f"{bpm:.1f}")
            self.freq_label.config(text=f"{p_freq:.2f}")

        if len(self.fft_buffer) > 1000:
            self.fft_buffer = self.fft_buffer[-500:]

if __name__ == "__main__":
    root = tk.Tk()
    app = FuturisticPPGViewer(root)
    root.update_idletasks()
    # 確保一開始網格正確
    app.draw_grid()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.is_running = False
        if app.ser: app.ser.close()