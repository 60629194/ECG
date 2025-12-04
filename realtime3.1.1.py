import tkinter as tk
from tkinter import font
import math
import collections
import serial
import numpy as np
import time
import threading
import scipy.signal as signal

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
GRAPH_HEIGHT = 250  
GRAPH_WIDTH = 800

PPG_DATA_POINTS = 400
ECG_DATA_POINTS = 512       
REFRESH_RATE_MS = 20
SERIAL_PORT = 'COM5'    
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
        self.root.geometry("1024x700") 
        self.root.configure(bg=COLORS["BG_MAIN"])
        
        # --- Data Structures ---
        self.raw_buffer = collections.deque(maxlen=600) 
        self.display_buffer = collections.deque([GRAPH_HEIGHT/2] * PPG_DATA_POINTS, maxlen=PPG_DATA_POINTS)
        self.fft_buffer = [] 
        
        # [New] Total sample counter to avoid buffer trimming issues
        self.total_sample_count = 0
        self.last_ppg_sample_count = 0

        self.sweep_buffer = [GRAPH_HEIGHT/2] * PPG_DATA_POINTS
        self.sweep_idx = 0 

        # --- RR Data Structures ---
        # 60 seconds at 81.6Hz is ~4900 samples. Set to 5000 for safety.
        self.rr_buffer = collections.deque(maxlen=5000)
        self.rr_lock = threading.Lock()

        # --- ECG Data Structures ---
        # ecg_playback_queue: Stores the generated ECG points waiting to be displayed
        self.ecg_playback_queue = collections.deque() 
        self.ecg_display_buffer = collections.deque([GRAPH_HEIGHT/2] * ECG_DATA_POINTS, maxlen=ECG_DATA_POINTS) 
        self.ecg_sweep_buffer = [GRAPH_HEIGHT/2] * ECG_DATA_POINTS 
        self.ecg_sweep_idx = 0
        self.ecg_lock = threading.Lock()
        
        # Playback timing control
        self.ecg_playback_accumulator = 0.0
        
        # Inference Control
        self.inference_running = True
        self.inference_thread = None

        # --- View Control ---
        self.graph_mode = "WAVE"     
        self.fft_enabled = False     
        self.filter_enabled = True   
        
        # [New] ECG View Toggle
        self.ecg_view_enabled = True

        self.line_tail = None
        self.ecg_line_tail = None

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

        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

        self.rr_thread = threading.Thread(target=self.rr_loop, daemon=True)
        self.rr_thread.start()

        self.animate_graph()
        
        # [New] Time tracking for smooth playback
        self.last_frame_time = time.time()
        
        # [New] Auto-scaling state
        self.current_scale = SCALE_FACTOR
        self.target_scale = SCALE_FACTOR

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
        # [New] RR Label
        self.rr_label = self.add_stat_module(self.left_sidebar, "RESP RATE (BPM)", COLORS["NEON_CYAN"])
        
        self.add_decorative_text(self.left_sidebar)

        # 3. Graph Area (Split into PPG and ECG)
        graph_outer, graph_inner = self.create_neon_border(self.root, COLORS["NEON_CYAN"], padding=3)
        graph_outer.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # PPG Canvas (Top)
        self.ppg_frame = tk.Frame(graph_inner, bg="black")
        self.ppg_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        tk.Label(self.ppg_frame, text="RAW PPG", bg="black", fg=COLORS["NEON_CYAN"], font=("Consolas", 8)).place(x=5, y=5)

        self.canvas = tk.Canvas(self.ppg_frame, bg="black", 
                                height=GRAPH_HEIGHT, width=GRAPH_WIDTH, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)

        # Separator [New] Store reference
        self.separator = tk.Frame(graph_inner, bg=COLORS["GRID_LINE"], height=1)
        self.separator.pack(fill="x", padx=5, pady=2)

        # ECG Canvas (Bottom)
        self.ecg_frame = tk.Frame(graph_inner, bg="black")
        self.ecg_frame.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(self.ecg_frame, text="SYNTHESIZED ECG (AI)", bg="black", fg=COLORS["NEON_RED"], font=("Consolas", 8)).place(x=5, y=5)

        self.ecg_canvas = tk.Canvas(self.ecg_frame, bg="black",
                                    height=GRAPH_HEIGHT, width=GRAPH_WIDTH, highlightthickness=0)
        self.ecg_canvas.pack(fill="both", expand=True)

        self.ecg_line_id = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)

        # 4. Bottom Control Bar
        bottom_bar_outer, bottom_bar = self.create_neon_border(self.root, COLORS["TEXT_DIM"], padding=1)
        bottom_bar_outer.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(5,10))
        
        # --- 按鈕區 ---
        self.fft_btn = tk.Button(bottom_bar, text="[ FFT: OFF ]", 
                                    bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], 
                                    font=self.btn_font, relief="flat", 
                                    activebackground=COLORS["GRID_LINE"],
                                    activeforeground=COLORS["NEON_CYAN"],
                                    command=self.toggle_fft)
        self.fft_btn.pack(side="right", padx=5, pady=2)

        self.mode_btn = tk.Button(bottom_bar, text="[ VIEW: WAVE ]", 
                                    bg=COLORS["BG_PANEL"], fg=COLORS["NEON_CYAN"], 
                                    font=self.btn_font, relief="flat", 
                                    activebackground=COLORS["GRID_LINE"],
                                    activeforeground=COLORS["NEON_RED"],
                                    command=self.toggle_graph_mode)
        self.mode_btn.pack(side="right", padx=5, pady=2)

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

        # [New] ECG View Toggle Button
        self.ecg_btn = tk.Button(bottom_bar, text="[ ECG VIEW: ON ]",
                        bg=COLORS["BG_PANEL"], fg=COLORS["NEON_CYAN"],
                        font=self.btn_font, relief="flat", activebackground=COLORS["GRID_LINE"],
                        command=self.toggle_ecg_view)
        self.ecg_btn.pack(side="right", padx=6, pady=2)

        self.console_msg = tk.Label(bottom_bar, text=">> SYSTEM READY.", 
                 bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], font=self.label_font, anchor="w")
        self.console_msg.pack(side="left", fill="x", padx=5)

    def toggle_ecg_view(self):
        self.ecg_view_enabled = not self.ecg_view_enabled
        
        if self.ecg_view_enabled:
            self.ecg_btn.config(text="[ ECG VIEW: ON ]", fg=COLORS["NEON_CYAN"])
            self.separator.pack(fill="x", padx=5, pady=2)
            self.ecg_frame.pack(fill="both", expand=True, padx=2, pady=2)
            self.console_msg.config(text=">> ECG VIEW ENABLED.")
        else:
            self.ecg_btn.config(text="[ ECG VIEW: OFF ]", fg=COLORS["TEXT_DIM"])
            self.ecg_frame.pack_forget()
            self.separator.pack_forget()
            self.console_msg.config(text=">> ECG VIEW DISABLED.")
            
        # Force update to recalculate heights
        self.root.update_idletasks()
        self.draw_grid()

    def toggle_graph_mode(self):
        if self.graph_mode == "WAVE":
            self.graph_mode = "SWEEP"
            self.mode_btn.config(text="[ VIEW: SWEEP ]", fg=COLORS["NEON_RED"])
        else:
            self.graph_mode = "WAVE"
            self.mode_btn.config(text="[ VIEW: WAVE ]", fg=COLORS["NEON_CYAN"])
        
        if not self.fft_enabled:
            self.reset_canvas_for_mode()

    def toggle_fft(self):
        self.fft_enabled = not self.fft_enabled
        
        if self.fft_enabled:
            self.fft_btn.config(text="[ FFT: ON ]", fg=COLORS["NEON_GREEN"])
            self.canvas.delete("all")
            self.ecg_canvas.delete("all")
            self.line_id = None
            self.line_tail = None
            self.ecg_line_id = None
            self.ecg_line_tail = None
        else:
            self.fft_btn.config(text="[ FFT: OFF ]", fg=COLORS["TEXT_DIM"])
            self.reset_canvas_for_mode()

    def reset_canvas_for_mode(self):
        self.canvas.delete("all")
        self.ecg_canvas.delete("all")
        self.line_id = None
        self.line_tail = None
        self.ecg_line_id = None
        self.ecg_line_tail = None
        
        self.draw_grid()

        if self.graph_mode == "WAVE":
            self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
            self.ecg_line_id = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
        elif self.graph_mode == "SWEEP":
            self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
            self.line_tail = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
            
            self.ecg_line_id = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
            self.ecg_line_tail = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
            
            self.sweep_idx = 0 
            self.ecg_sweep_idx = 0

    def toggle_hp(self):
        self.hp_enabled = not getattr(self, 'hp_enabled', True)
        if self.hp_enabled:
            self.hp_btn.config(text="[ Hi-Pass: ON ]", fg=COLORS["NEON_GREEN"])
            self.console_msg.config(text=">> HP FILTER ENABLED.")
        else:
            self.hp_btn.config(text="[ Hi-Pass: OFF ]", fg=COLORS["NEON_RED"])
            self.console_msg.config(text=">> HP FILTER DISABLED.")

    def toggle_lp(self):
        self.lp_enabled = not getattr(self, 'lp_enabled', True)
        if self.lp_enabled:
            self.lp_btn.config(text="[ Lo-Pass: ON ]", fg=COLORS["NEON_GREEN"])
            self.console_msg.config(text=">> LP FILTER ENABLED.")
        else:
            self.lp_btn.config(text="[ Lo-Pass: OFF ]", fg=COLORS["NEON_RED"])
            self.console_msg.config(text=">> LP FILTER DISABLED.")

    def toggle_ma(self):
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
        def _draw_on(cv):
            w = cv.winfo_width()
            h = cv.winfo_height()
            if w < 10: w = GRAPH_WIDTH
            if h < 10: h = GRAPH_HEIGHT
            
            cv.delete("grid") 

            for i in range(0, w, 40):
                color = COLORS["GRID_LINE"]
                if (i % 160) == 0: color = COLORS["TEXT_DIM"]
                cv.create_line(i, 0, i, h, fill=color, dash=(4, 4), tags="grid")
            
            center_y = h / 2
            cv.create_line(0, center_y, w, center_y, fill=COLORS["TEXT_DIM"], tags="grid")

            if self.graph_mode in ["WAVE", "SWEEP"]: 
                for i in range(0, h, 40):
                    if abs(i - center_y) < 5: continue
                    cv.create_line(0, i, w, i, fill=COLORS["GRID_LINE"], dash=(4, 4), tags="grid")

        _draw_on(self.canvas)
        if self.ecg_view_enabled:
            _draw_on(self.ecg_canvas)

    # --- Signal Processing ---
    def process_signal(self, data_list):
        vals = np.array(data_list)
        vals = vals - np.mean(vals)
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
        if len(vals) < 2: return vals
        dt = 1.0 / fs
        rc = 1.0 / (2.0 * math.pi * cutoff)
        alpha = rc / (rc + dt)
        out = [0.0] * len(vals)
        out[0] = 0.0 
        for i in range(1, len(vals)):
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
                        chunk = self.ser.read(self.ser.in_waiting)
                        lines = chunk.split(b'\n')
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            try:
                                val = float(line.decode('utf-8'))
                                self.raw_buffer.append(val)
                                self.fft_buffer.append(val)
                                
                                # [New] Append to RR buffer
                                with self.rr_lock:
                                    self.rr_buffer.append(val)
                                    
                                self.total_sample_count += 1 
                            except ValueError:
                                continue
                    else:
                        time.sleep(0.005)
                except Exception as e:
                    print(f"Read Error: {e}")
                    time.sleep(1)
            else:
                time.sleep(1)

    # --- RR Calculation Logic (Ported) ---
    def get_peaks(self, data, fs=100):
        distance = int(0.3 * fs)
        peaks, _ = signal.find_peaks(data, distance=distance, prominence=1)
        return peaks

    def get_fft_peak(self, sig, fs):
        if sig is None: return 0, 0
        sig = signal.detrend(sig)
        window = signal.windows.hann(len(sig))
        sig_win = sig * window
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(sig_win))
        mask = (freqs >= 0.06) & (freqs <= 1.0)
        valid_freqs = freqs[mask]
        valid_fft = fft_vals[mask]
        if len(valid_fft) == 0: return 0, 0
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        peak_power = valid_fft[peak_idx]
        return peak_freq * 60, peak_power

    def extract_signals(self, data, fs=100):
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 5.0 / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        cardiac_ppg = signal.filtfilt(b, a, data)
        peaks = self.get_peaks(cardiac_ppg, fs)
        if len(peaks) < 2: return None, None, None, 4.0
        peak_times = peaks / fs
        riiv_raw = cardiac_ppg[peaks]
        ibis = np.diff(peak_times)
        ibi_times = peak_times[:-1] + np.diff(peak_times)/2
        resample_fs = 4.0
        duration = len(data) / fs
        t_interp = np.arange(0, duration, 1/resample_fs)
        riiv_interp = np.interp(t_interp, peak_times, riiv_raw)
        rifv_interp = np.interp(t_interp, ibi_times, ibis)
        b_base, a_base = signal.butter(2, [0.1/nyquist, 1.0/nyquist], btype='band')
        detrended_data = signal.detrend(data)
        raw_baseline = signal.filtfilt(b_base, a_base, detrended_data)
        t_raw = np.arange(len(data)) / fs
        baseline_interp = np.interp(t_interp, t_raw, raw_baseline)
        return riiv_interp, rifv_interp, baseline_interp, resample_fs

    def calculate_respiratory_rate(self, data, fs=100):
        riiv, rifv, baseline, resample_fs = self.extract_signals(data, fs)
        if riiv is None: return 0
        rate_riiv, _ = self.get_fft_peak(riiv, resample_fs)
        rate_rifv, _ = self.get_fft_peak(rifv, resample_fs)
        rate_base, _ = self.get_fft_peak(baseline, resample_fs)
        
        final_rate = rate_riiv 
        if abs(rate_riiv - rate_rifv) < 3:
            pass
        if 25 < rate_base < 45:
            if rate_riiv < 15:
                final_rate = rate_base
        return final_rate

    def rr_loop(self):
        """Background thread to calculate RR every 2 seconds"""
        while self.is_running:
            time.sleep(2.0)
            
            with self.rr_lock:
                if len(self.rr_buffer) < 2000: # Need at least ~25s of data
                    continue
                data_snapshot = list(self.rr_buffer)
                
            try:
                # Calculate RR
                rr_val = self.calculate_respiratory_rate(np.array(data_snapshot), fs=FS)
                
                # Update UI safely
                def update_ui(val):
                    self.rr_label.config(text=f"{val:.1f}")
                    
                self.root.after(0, update_ui, rr_val)
                
            except Exception as e:
                print(f"RR Calc Error: {e}")

    # --- Background Inference Loop ---
    def inference_loop(self):
        """
        Background thread to run PPG -> ECG inference continuously.
        Optimization: Run only once every 4 seconds (non-overlapping) to save CPU.
        """
        import os
        import sys
        import time
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ppg_dir = os.path.join(current_dir, 'ppg2ecg')
        if ppg_dir not in sys.path:
            sys.path.append(ppg_dir)
            
        try:
            import ppg2ecg as p2e
            import tensorflow as tf
            from scipy.signal import resample
            
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            
            p2e.MODEL_DIR = os.path.join(ppg_dir, 'weights')
            tf.keras.backend.set_floatx('float64')
            
            print("[Inference] Loading CardioGAN...")
            self.ecg_model = p2e.load_cardiogan()
            print("[Inference] Model Loaded.")
            
        except Exception as e:
            print(f"[Inference] Setup Error: {e}")
            return

        last_processed_count = 0
        
        while self.is_running:
            # [Fix] Use total_sample_count instead of buffer length
            current_count = self.total_sample_count
            
            if current_count < 350:
                time.sleep(0.1)
                continue
                
            # Wait for ~4 seconds of NEW data (320 samples)
            if current_count - last_processed_count < 320:
                time.sleep(0.1)
                continue
            
            try:
                t0 = time.time()
                
                # Snapshot the last 4 seconds
                raw_snapshot = list(self.fft_buffer)[-450:]
                
                # Update our progress marker
                last_processed_count = current_count

                # --- Preprocessing ---
                v_dc = p2e.DC_filter(raw_snapshot, fs=FS, cutoff=0.5)
                clean_ppg = p2e.moving_average_filter(v_dc, window_size=5)
                
                num_input_samples = int(4.0 * FS) 
                if len(clean_ppg) < num_input_samples: continue
                
                segment_src = clean_ppg[-num_input_samples:]
                resampled_ppg = resample(segment_src, 512)
                
                seg_min = np.min(resampled_ppg)
                seg_max = np.max(resampled_ppg)
                if seg_max - seg_min == 0:
                    segment_norm = resampled_ppg
                else:
                    segment_norm = 2 * ((resampled_ppg - seg_min) / (seg_max - seg_min)) - 1
                
                segment_input = segment_norm.reshape(1, 512, 1)
                fake_ecg = self.ecg_model(segment_input, training=False)
                ecg_out = fake_ecg.numpy().flatten()
                
                # --- Post-processing ---
                # Add the entire 4-second chunk (512 points) to the playback queue
                with self.ecg_lock:
                    self.ecg_playback_queue.extend(ecg_out)
                    
                dt = time.time() - t0
                print(f"[Inference] Updated 4s ECG in {dt:.3f}s")
                    
            except Exception as e:
                print(f"[Inference] Error: {e}")
                time.sleep(1)

    # --- Main Animation Loop ---
    def animate_graph(self):
        if len(self.raw_buffer) >= AVG_WINDOW * 2:
            self.status_label.config(text="STATUS: DATA STREAMING", fg=COLORS["NEON_GREEN"])
            
            # Update scale factor based on recent data
            if len(self.raw_buffer) > 0:
                 current_raw = list(self.raw_buffer)
                 processed = self.process_signal(current_raw)
                 # Use last 400 points for scaling context
                 context_data = processed[-PPG_DATA_POINTS:] if len(processed) > PPG_DATA_POINTS else processed
                 self.update_scale_factor(context_data, self.canvas.winfo_height())

            if self.fft_enabled:
                self.draw_fft_mode()
            else:
                if self.graph_mode == "WAVE":
                    self.draw_wave_mode()
                    if self.ecg_view_enabled:
                        self.draw_ecg_wave_mode()
                elif self.graph_mode == "SWEEP":
                    self.draw_sweep_mode()
                    if self.ecg_view_enabled:
                        self.draw_ecg_sweep_mode()
        else:
            self.status_label.config(text="STATUS: WAITING FOR SIGNAL...", fg=COLORS["NEON_RED"])

        if not hasattr(self, 'tick_count'): self.tick_count = 0
        self.tick_count += 1
        if self.tick_count > 15:
            self.tick_count = 0
            self.calculate_bpm_update()

        self.root.after(REFRESH_RATE_MS, self.animate_graph)
        self.last_frame_time = time.time()

    def update_scale_factor(self, data, height):
        if not data: return
        
        max_val = np.max(np.abs(data))
        if max_val < 0.01: # Noise floor
            target = 100.0
        else:
            # Target: Max value should occupy ~40% of height (so peak-to-peak is ~80%)
            target = (height * 0.4) / max_val
            
        # Smooth transition
        self.current_scale = self.current_scale * 0.9 + target * 0.1

    def draw_sweep_mode(self):
        # Calculate how many NEW samples arrived since last frame
        current_count = self.total_sample_count
        new_samples = current_count - self.last_ppg_sample_count
        
        if new_samples <= 0: return
        
        # Fetch the new data points
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        
        # We only want the *new* points. 
        # Be careful if buffer size is smaller than new_samples (unlikely but possible)
        if len(processed_data) < new_samples:
            data_to_add = processed_data
        else:
            data_to_add = processed_data[-new_samples:]
            
        # [Fix] Use actual canvas height
        h = self.canvas.winfo_height()
        
        for val in data_to_add:
            y_pos = (h / 2) - (val * self.current_scale)
            y_pos = max(10, min(h - 10, y_pos))
            
            self.sweep_buffer[self.sweep_idx] = y_pos
            self.sweep_idx += 1
            if self.sweep_idx >= PPG_DATA_POINTS:
                self.sweep_idx = 0
        
        self.last_ppg_sample_count = current_count

        w = self.canvas.winfo_width()
        x_stretch = w / PPG_DATA_POINTS
        
        self.canvas.delete("sweep_line")
        
        base_color_rgb = (0, 255, 249) 
        for i in range(PPG_DATA_POINTS - 1):
            age = (self.sweep_idx - i) % PPG_DATA_POINTS
            dim_factor = 1 - (age / PPG_DATA_POINTS)**3
            
            dark_bg_rgb = (5, 8, 20)
            r = int(base_color_rgb[0] * dim_factor + dark_bg_rgb[0] * (1 - dim_factor))
            g = int(base_color_rgb[1] * dim_factor + dark_bg_rgb[1] * (1 - dim_factor))
            b = int(base_color_rgb[2] * dim_factor + dark_bg_rgb[2] * (1 - dim_factor))
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            x0 = i * x_stretch
            y0 = self.sweep_buffer[i]
            x1 = (i + 1) * x_stretch
            y1 = self.sweep_buffer[(i + 1) % PPG_DATA_POINTS]
            
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2, tags="sweep_line")

    def draw_ecg_sweep_mode(self):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0.1: dt = 0.1
        
        base_samples = 128.0 * dt
        
        target_buffer = 512
        current_buffer = len(self.ecg_playback_queue)
        error = current_buffer - target_buffer
        speed_factor = 1.0 + (error * 0.001)
        speed_factor = max(0.8, min(1.2, speed_factor))
        
        samples_needed = base_samples * speed_factor
        
        self.ecg_playback_accumulator += samples_needed
        points_to_pop = int(self.ecg_playback_accumulator)
        self.ecg_playback_accumulator -= points_to_pop
        
        new_points = []
        with self.ecg_lock:
            for _ in range(points_to_pop):
                if len(self.ecg_playback_queue) > 0:
                    new_points.append(self.ecg_playback_queue.popleft())
                else:
                    break
                    
        if not new_points: return

        # [Fix] Use actual canvas height
        h = self.ecg_canvas.winfo_height()

        for val in new_points:
            y_pos = (h / 2) - (val * 80) 
            y_pos = max(10, min(h - 10, y_pos))
            
            self.ecg_sweep_buffer[self.ecg_sweep_idx] = y_pos
            self.ecg_sweep_idx = (self.ecg_sweep_idx + 1) % ECG_DATA_POINTS
            
        self.ecg_canvas.delete("sweep_line")
        w = self.ecg_canvas.winfo_width()
        x_stretch = w / ECG_DATA_POINTS
        
        base_color_rgb = (255, 51, 51) 
        dark_bg_rgb = (5, 8, 20)
        
        for i in range(ECG_DATA_POINTS - 1):
            age = (self.ecg_sweep_idx - i) % ECG_DATA_POINTS
            dim_factor = 1 - (age / ECG_DATA_POINTS)**3
            
            r = int(base_color_rgb[0] * dim_factor + dark_bg_rgb[0] * (1 - dim_factor))
            g = int(base_color_rgb[1] * dim_factor + dark_bg_rgb[1] * (1 - dim_factor))
            b = int(base_color_rgb[2] * dim_factor + dark_bg_rgb[2] * (1 - dim_factor))
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            x0 = i * x_stretch
            y0 = self.ecg_sweep_buffer[i]
            x1 = (i + 1) * x_stretch
            y1 = self.ecg_sweep_buffer[(i + 1) % ECG_DATA_POINTS]
            
            self.ecg_canvas.create_line(x0, y0, x1, y1, fill=color, width=2, tags="sweep_line")

    def draw_wave_mode(self):
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        
        # [Fix] Show exactly PPG_DATA_POINTS from the end of the buffer
        # This ensures time scale is constant regardless of FPS
        if len(processed_data) < PPG_DATA_POINTS:
            data_to_show = processed_data
        else:
            data_to_show = processed_data[-PPG_DATA_POINTS:]
            
        # [Fix] Use actual canvas height
        h = self.canvas.winfo_height()

        self.display_buffer.clear()
        for val in data_to_show:
            y_pos = (h / 2) - (val * self.current_scale)
            y_pos = max(10, min(h - 10, y_pos)) 
            self.display_buffer.append(y_pos)

        w = self.canvas.winfo_width()
        x_stretch = w / PPG_DATA_POINTS 
        
        coords = []
        for i, y_val in enumerate(self.display_buffer):
            coords.append(i * x_stretch)
            coords.append(y_val)
        
        if self.line_id is None:
             self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
             
        if len(coords) > 4:
             self.canvas.coords(self.line_id, *coords)

    def draw_ecg_wave_mode(self):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0.1: dt = 0.1
        
        base_samples = 128.0 * dt
        
        target_buffer = 512
        current_buffer = len(self.ecg_playback_queue)
        error = current_buffer - target_buffer
        speed_factor = 1.0 + (error * 0.001)
        speed_factor = max(0.8, min(1.2, speed_factor))
        
        samples_needed = base_samples * speed_factor
        
        self.ecg_playback_accumulator += samples_needed
        points_to_pop = int(self.ecg_playback_accumulator)
        self.ecg_playback_accumulator -= points_to_pop
        
        new_points = []
        with self.ecg_lock:
            for _ in range(points_to_pop):
                if len(self.ecg_playback_queue) > 0:
                    new_points.append(self.ecg_playback_queue.popleft())
                else:
                    break
        
        # [Fix] Use actual canvas height
        h = self.ecg_canvas.winfo_height()

        for val in new_points:
            y_pos = (h / 2) - (val * 80)
            y_pos = max(10, min(h - 10, y_pos))
            self.ecg_display_buffer.append(y_pos)
            
        w = self.ecg_canvas.winfo_width()
        x_stretch = w / ECG_DATA_POINTS
        
        coords = []
        for i, y_val in enumerate(self.ecg_display_buffer):
            coords.append(i * x_stretch)
            coords.append(y_val)
            
        if self.ecg_line_id is None:
             self.ecg_line_id = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
             
        if len(coords) > 4:
             self.ecg_canvas.coords(self.ecg_line_id, *coords)

    def draw_fft_mode(self):
        if len(self.fft_buffer) < 200: return
        
        data = list(self.fft_buffer)[-400:] 
        v = np.array(data)
        v = v - np.mean(v)
        
        fft_vals = np.fft.rfft(v)
        fft_freqs = np.fft.rfftfreq(len(v), 1.0/FS)
        magnitude = np.abs(fft_vals)

        mask = (fft_freqs < 40)
        display_freqs = fft_freqs[mask]
        display_mags = magnitude[mask]

        self.canvas.delete("fft_bar")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        num_bars = len(display_freqs)
        bar_width = w / num_bars
        
        max_mag = np.max(display_mags) if np.max(display_mags) > 0 else 1
        
        for i, mag in enumerate(display_mags):
            bar_h = (mag / max_mag) * (h * 0.8)
            
            x0 = i * bar_width
            y0 = h
            x1 = x0 + bar_width - 1
            y1 = h - bar_h
            
            bar_color = COLORS["NEON_CYAN"]
            freq = display_freqs[i]
            if 0.6 <= freq <= 3.0: 
                bar_color = COLORS["NEON_GREEN"]

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=bar_color, outline="", tags="fft_bar")


    def calculate_bpm_update(self):
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

    def run_ppg2ecg(self):
        """
        執行 PPG -> ECG 轉換 (Non-realtime Snapshot)
        使用 ppg2ecg/ppg2ecg.py 的邏輯與模型
        """
        if len(self.fft_buffer) < 350: 
            self.console_msg.config(text=">> NEED MORE DATA (WAIT A FEW SECONDS)")
            return

        self.console_msg.config(text=">> LOADING AI MODEL... (MAY FREEZE)")
        self.root.update() 

        try:
            import os
            import sys
            
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ppg_dir = os.path.join(current_dir, 'ppg2ecg')
            if ppg_dir not in sys.path:
                sys.path.append(ppg_dir)
            
            import ppg2ecg as p2e
            import tensorflow as tf
            import matplotlib.pyplot as plt
            from scipy.signal import resample
            
            p2e.MODEL_DIR = os.path.join(ppg_dir, 'weights')
            
            tf.keras.backend.set_floatx('float64')

            if not hasattr(self, 'ecg_model'):
                print("Initializing CardioGAN...")
                self.ecg_model = p2e.load_cardiogan()
            
            target_fs = 128.0
            window_sec = 4.0
            
            raw_data = list(self.fft_buffer)[-450:] 
            
            v_dc = p2e.DC_filter(raw_data, fs=FS, cutoff=0.5)
            clean_ppg = p2e.moving_average_filter(v_dc, window_size=5)
            
            num_input_samples = int(window_sec * FS)
            
            if len(clean_ppg) < num_input_samples:
                 self.console_msg.config(text=">> DATA BUFFER UNDERRUN")
                 return
            
            segment_src = clean_ppg[-num_input_samples:]
            
            resampled_ppg = resample(segment_src, 512)
            
            seg_min = np.min(resampled_ppg)
            seg_max = np.max(resampled_ppg)
            if seg_max - seg_min == 0:
                segment_norm = resampled_ppg
            else:
                segment_norm = 2 * ((resampled_ppg - seg_min) / (seg_max - seg_min)) - 1
                
            self.console_msg.config(text=">> RUNNING INFERENCE...")
            self.root.update()
            
            segment_input = segment_norm.reshape(1, 512, 1)
            fake_ecg = self.ecg_model(segment_input, training=False)
            ecg_out = fake_ecg.numpy().flatten()
            
            self.console_msg.config(text=">> SHOWING RESULT...")
            
            plt.figure(figsize=(10, 7))
            
            plt.subplot(2, 1, 1)
            t_ppg = np.linspace(0, window_sec, len(resampled_ppg))
            plt.plot(t_ppg, resampled_ppg, color='#1f77b4', label='PPG (Resampled 128Hz)')
            plt.title("Input: PPG (4s Snapshot)")
            plt.ylabel("Amplitude")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            t_ecg = np.linspace(0, window_sec, len(ecg_out))
            plt.plot(t_ecg, ecg_out, color='#d62728', label='AI Generated ECG')
            plt.title("Output: CardioGAN Reconstructed ECG")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (Normalized)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            self.console_msg.config(text=">> READY.")

        except Exception as e:
            print(f"PPG2ECG Error: {e}")
            self.console_msg.config(text=">> AI ERROR. SEE CONSOLE.")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = FuturisticPPGViewer(root)
    root.update_idletasks()
    app.draw_grid()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.is_running = False
        if app.ser: app.ser.close()