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
    "NEON_YELLOW": "#f4e242",  # New color for age/HRV
    "TEXT_MAIN": "#e6f0ff",
    "TEXT_DIM": "#6b7c99",
    "GRID_LINE": "#1a264f"
}

# --- Settings ---
GRAPH_HEIGHT = 250    
GRAPH_WIDTH = 800

PPG_DATA_POINTS = 400
ECG_DATA_POINTS = 627         
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

# ==============================================================================
# --- STANDALONE FILTER FUNCTIONS (Replacing imported 'filters' module) ---
# ==============================================================================
def highpass_filter_func(data, fs, cutoff):
    """Standalone highpass filter mimicking the logic needed for analysis."""
    if len(data) < 2: return data
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = rc / (rc + dt)
    out = np.zeros(len(data))
    out[0] = 0.0 
    for i in range(1, len(data)):
        out[i] = alpha * (out[i-1] + data[i] - data[i-1])
    return out

def lowpass_filter_func(data, fs, cutoff):
    """Standalone lowpass filter."""
    if len(data) < 2: return data
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = dt / (rc + dt)
    out = np.zeros(len(data))
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = out[i-1] + alpha * (data[i] - out[i-1])
    return out

# ==============================================================================
# --- MERGED PPG ANALYSIS LOGIC ---
# ==============================================================================

def find_peaks_simple(signal_data, distance=None, height=None):
    """Simple peak detection using numpy."""
    dy = np.diff(signal_data)
    peaks = []
    for i in range(1, len(dy)):
        if dy[i - 1] > 0 and dy[i] <= 0:
            if height is not None and signal_data[i] < height:
                continue
            peaks.append(i)

    if distance is not None and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= distance:
                filtered_peaks.append(p)
        return filtered_peaks
    return peaks

def calculate_sdppg_index(data, fs):
    """Calculates Second Derivative PPG (SDPPG) indices."""
    # 1. Preprocessing
    hp_filtered = highpass_filter_func(data, fs, 0.5)
    lp_filtered = lowpass_filter_func(hp_filtered, fs, 10.0)

    # 2. Derivatives
    d1 = np.gradient(lp_filtered)
    sdppg = np.gradient(d1)

    # 3. Detect 'a' waves
    threshold_a = np.mean(sdppg) + 1.0 * np.std(sdppg)
    min_dist = int(0.5 * fs)  #nyquist
    a_peaks = find_peaks_simple(sdppg, distance=min_dist, height=threshold_a)

    ratios_b_a = []
    ratios_c_a = []
    ratios_d_a = []
    ratios_e_a = []
    valid_beats = 0

    for i in range(len(a_peaks) - 1):
        a_idx = a_peaks[i]
        next_a_idx = a_peaks[i + 1]

        search_window = int(0.4 * fs)
        end_search = min(a_idx + search_window, next_a_idx)

        if end_search <= a_idx:
            continue

        window = sdppg[a_idx:end_search]

        # Find b (min)
        neg_window = -window
        minima_indices = find_peaks_simple(neg_window)
        if not minima_indices: continue
        b_idx = a_idx + minima_indices[0]

        # Find c (max)
        c_window = sdppg[b_idx:end_search]
        c_cands = find_peaks_simple(c_window)
        if not c_cands: continue
        c_idx = b_idx + c_cands[0]

        # Find d (min)
        d_window = sdppg[c_idx:end_search]
        d_cands = find_peaks_simple(-d_window)
        if not d_cands: continue
        d_idx = c_idx + d_cands[0]

        # Find e (max)
        e_window = sdppg[d_idx:end_search]
        e_cands = find_peaks_simple(e_window)
        if not e_cands: continue
        e_idx = d_idx + e_cands[0]

        val_a = sdppg[a_idx]
        val_b = sdppg[b_idx]
        val_c = sdppg[c_idx]
        val_d = sdppg[d_idx]
        val_e = sdppg[e_idx]

        if val_a != 0:
            ratios_b_a.append(val_b / val_a)
            ratios_c_a.append(val_c / val_a)
            ratios_d_a.append(val_d / val_a)
            ratios_e_a.append(val_e / val_a)
            valid_beats += 1

    if valid_beats > 0:
        return {
            "b_a": np.mean(ratios_b_a),
            "c_a": np.mean(ratios_c_a),
            "d_a": np.mean(ratios_d_a),
            "e_a": np.mean(ratios_e_a),
            "aging_index": (
                np.mean(ratios_b_a)
                - np.mean(ratios_c_a)
                - np.mean(ratios_d_a)
                - np.mean(ratios_e_a)
            ),
            "valid_beats": valid_beats,
        }
    else:
        return {"b_a": None, "c_a": None, "d_a": None, "e_a": None, "aging_index": None, "valid_beats": 0}

def calculate_hrv(data, fs):
    """Calculates Heart Rate Variability (HRV) metrics."""
    hp_filtered = highpass_filter_func(data, fs, 0.5)
    lp_filtered = lowpass_filter_func(hp_filtered, fs, 5.0)

    min_dist = int(0.5 * fs) 
    threshold = np.mean(lp_filtered)
    peaks = find_peaks_simple(lp_filtered, distance=min_dist, height=threshold)

    if len(peaks) < 2:
        return {"mean_nn": None, "sdnn": None, "rmssd": None, "pnn50": None, "bpm": None}

    nn_intervals = np.diff(peaks) / fs * 1000.0
    nn_intervals = [nn for nn in nn_intervals if 300 <= nn <= 1500]

    if len(nn_intervals) < 2:
        return {"mean_nn": None, "sdnn": None, "rmssd": None, "pnn50": None, "bpm": None}

    mean_nn = np.mean(nn_intervals)
    sdnn = np.std(nn_intervals, ddof=1)
    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    nn50 = np.sum(np.abs(diff_nn) > 50)
    pnn50 = (nn50 / len(diff_nn)) * 100.0 if len(diff_nn) > 0 else 0.0
    bpm = 60000.0 / mean_nn

    return {"mean_nn": mean_nn, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50, "bpm": bpm}

def calculate_ipad_index(data, fs):
    """Calculates IPAD (Inflection Point Area and d-peak) index."""
    hp_filtered = highpass_filter_func(data, fs, 0.5)
    lp_filtered = lowpass_filter_func(hp_filtered, fs, 10.0)

    p5 = np.percentile(lp_filtered, 5)
    p95 = np.percentile(lp_filtered, 95)

    if p95 > p5:
        normalized = (np.array(lp_filtered) - p5) / (p95 - p5)
        lp_filtered = 0.8 + normalized * 1.0
        lp_filtered = np.clip(lp_filtered, 0.8, 1.8)

    d1 = np.gradient(lp_filtered)
    sdppg = np.gradient(d1)

    threshold_a = np.mean(sdppg) + 1.0 * np.std(sdppg)
    min_dist = int(0.5 * fs)
    a_peaks = find_peaks_simple(sdppg, distance=min_dist, height=threshold_a)

    ipad_values = []
    valid_beats = 0

    for i in range(len(a_peaks) - 1):
        a_idx = a_peaks[i]
        next_a_idx = a_peaks[i + 1]
        search_window = int(0.6 * fs)
        end_search = min(a_idx + search_window, next_a_idx)

        if end_search <= a_idx: continue
        window = sdppg[a_idx:end_search]

        zero_crossings = np.where(np.diff(np.signbit(window)))[0]
        if len(zero_crossings) < 4: continue

        z1 = zero_crossings[0]
        z2 = zero_crossings[1]
        z3 = zero_crossings[2]
        z4 = zero_crossings[3]

        s1_area = np.trapz(np.abs(window[0 : z1 + 1]))
        s2_area = np.trapz(np.abs(window[z1 : z2 + 1]))

        d_search_end = min(z4 + 10, len(window))
        d_region = window[z3:d_search_end]
        if len(d_region) == 0: continue

        d_val = np.min(d_region)
        a_val = sdppg[a_idx]

        if a_val == 0 or s1_area == 0: continue
        d_a_ratio = d_val / a_val
        ipa = s2_area / s1_area
        ipad = ipa + d_a_ratio
        ipad_values.append(ipad)
        valid_beats += 1

    if valid_beats > 0:
        return {"ipad": np.mean(ipad_values), "valid_beats": valid_beats}
    else:
        return {"ipad": None, "valid_beats": 0}

def estimate_age(ipad_index, sdppg_indices):
    """Estimates age using multiple methods."""
    estimates = {"age_ipad": None, "age_takazawa": None, "age_vessel": None}

    # 1. IPAD Method
    if ipad_index is not None:
        estimates["age_ipad"] = (ipad_index - 0.325) / -0.00748

    # 2. Takazawa Method (Aging Index)
    ai = sdppg_indices.get("aging_index")
    if ai is not None:
        estimates["age_takazawa"] = (ai + 1.556) / 0.019

    # 3. Vessel Age Method (IJBEM)
    b_a = sdppg_indices.get("b_a")
    c_a = sdppg_indices.get("c_a")
    d_a = sdppg_indices.get("d_a")
    e_a = sdppg_indices.get("e_a")

    if all(v is not None for v in [b_a, c_a, d_a, e_a]):
        estimates["age_vessel"] = (36.89 + (6.62 * b_a) - (27.05 * c_a) - (24.68 * d_a) + (2.44 * e_a))

    return estimates

# ==============================================================================
# --- MAIN APPLICATION ---
# ==============================================================================

class FuturisticPPGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("BIO-METRIC SENSOR INTERFACE // DUAL MODE // ADVANCED ANALYTICS")
        self.root.geometry("1024x768") 
        self.root.configure(bg=COLORS["BG_MAIN"])
        
        # --- Data Structures ---
        self.raw_buffer = collections.deque(maxlen=600) 
        self.display_buffer = collections.deque([GRAPH_HEIGHT/2] * PPG_DATA_POINTS, maxlen=PPG_DATA_POINTS)
        self.fft_buffer = [] 
        
        self.total_sample_count = 0
        self.last_ppg_sample_count = 0

        self.sweep_buffer = [GRAPH_HEIGHT/2] * PPG_DATA_POINTS
        self.sweep_idx = 0 

        # --- RR & Analysis Data Structures ---
        # Increased buffer for stable HRV/Age analysis
        self.rr_buffer = collections.deque(maxlen=5000)
        self.rr_lock = threading.Lock()

        # --- ECG Data Structures ---
        self.ecg_playback_queue = collections.deque() 
        self.ecg_display_buffer = collections.deque([GRAPH_HEIGHT/2] * ECG_DATA_POINTS, maxlen=ECG_DATA_POINTS) 
        self.ecg_sweep_buffer = [GRAPH_HEIGHT/2] * ECG_DATA_POINTS 
        self.ecg_sweep_idx = 0
        self.ecg_lock = threading.Lock()
        
        self.ecg_playback_accumulator = 0.0
        
        # Inference Control
        self.inference_running = True
        
        # --- View Control ---
        self.graph_mode = "WAVE"     
        self.fft_enabled = False     
        self.filter_enabled = True   
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
        self.digit_font = font.Font(family="Courier New", size=24, weight="bold") # Reduced slightly
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

        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        self.analysis_thread.start()

        self.animate_graph()
        
        self.last_frame_time = time.time()
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
        self.left_sidebar.configure(width=200, height=600)
        self.left_sidebar.pack_propagate(False)

        # Standard Metrics
        self.bpm_label = self.add_stat_module(self.left_sidebar, "HEART RATE (BPM)", COLORS["NEON_GREEN"])
        self.rr_label = self.add_stat_module(self.left_sidebar, "RESP RATE (BPM)", COLORS["NEON_CYAN"])
        
        # New Advanced Metrics
        self.hrv_label = self.add_stat_module(self.left_sidebar, "HRV (SDNN)", COLORS["NEON_YELLOW"])
        self.age_label = self.add_stat_module(self.left_sidebar, "VASCULAR AGE", COLORS["NEON_RED"])

        self.add_decorative_text(self.left_sidebar)

        # 3. Graph Area
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
        
        # Buttons
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
        self.hp_btn.config(text=f"[ Hi-Pass: {'ON' if self.hp_enabled else 'OFF'} ]", 
                           fg=COLORS["NEON_GREEN"] if self.hp_enabled else COLORS["NEON_RED"])

    def toggle_lp(self):
        self.lp_enabled = not getattr(self, 'lp_enabled', True)
        self.lp_btn.config(text=f"[ Lo-Pass: {'ON' if self.lp_enabled else 'OFF'} ]", 
                           fg=COLORS["NEON_GREEN"] if self.lp_enabled else COLORS["NEON_RED"])

    def toggle_ma(self):
        self.ma_enabled = not getattr(self, 'ma_enabled', True)
        self.ma_btn.config(text=f"[ SMOOTH: {'ON' if self.ma_enabled else 'OFF'} ]", 
                           fg=COLORS["NEON_GREEN"] if self.ma_enabled else COLORS["NEON_RED"])

    def add_stat_module(self, parent, title, glow_color):
        frame = tk.Frame(parent, bg=COLORS["BG_PANEL"], pady=10)
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

    # --- Signal Processing Wrappers ---
    def process_signal(self, data_list):
        if not data_list: return []
        
        # dc byebye
        vals = np.array(data_list)
        vals = vals - np.mean(vals)
        vals_list = vals.tolist()
        
        # 1. High Pass
        if getattr(self, 'hp_enabled', True):
            # return numpy array
            vals_list = highpass_filter_func(vals_list, FS, HP_CUTOFF)
            
        # 2. Low Pass
        if getattr(self, 'lp_enabled', True):
            # return numpy array
            vals_list = lowpass_filter_func(vals_list, FS, LP_CUTOFF)
            
        # 3. Moving Average
        if getattr(self, 'ma_enabled', True):
            # return list
            vals_list = self.moving_average_filter(vals_list, AVG_WINDOW)
        
        # if the final result is numpy array, force it back to python list
        if isinstance(vals_list, np.ndarray):
            vals_list = vals_list.tolist()
            
        return vals_list
        
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

    # --- RR Logic ---
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
        ibis = np.diff(peak_times)
        ibi_times = peak_times[:-1] + np.diff(peak_times)/2
        resample_fs = 4.0
        duration = len(data) / fs
        t_interp = np.arange(0, duration, 1/resample_fs)
        if len(peaks) < 2: return None, None, None, 4.0
        riiv_raw = cardiac_ppg[peaks]
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
        if riiv is None: return {'fused': 0, 'is_chaos': False}
        
        f, Pxx = signal.welch(riiv, resample_fs, nperseg=len(riiv)//2)
        Pxx_norm = Pxx / np.sum(Pxx)
        entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
        is_chaos = entropy > 3.9
        
        rate_riiv, _ = self.get_fft_peak(riiv, resample_fs)
        rate_rifv, _ = self.get_fft_peak(rifv, resample_fs)
        rate_base, _ = self.get_fft_peak(baseline, resample_fs)
        
        final_rate = rate_riiv 
        if abs(rate_riiv - rate_rifv) < 3: pass
        if 25 < rate_base < 45:
            if rate_riiv < 15: final_rate = rate_base
                
        return {'fused': final_rate, 'is_chaos': is_chaos}

    def rr_loop(self):
        while self.is_running:
            time.sleep(2.0)
            with self.rr_lock:
                if len(self.rr_buffer) < 2000: continue
                data_snapshot = list(self.rr_buffer)
            try:
                rr_results = self.calculate_respiratory_rate(np.array(data_snapshot), fs=FS)
                def update_ui(res):
                    if res['is_chaos']:
                        self.rr_label.config(text=f"{res['fused']:.1f}", fg=COLORS["NEON_RED"])
                    else:
                        self.rr_label.config(text=f"{res['fused']:.1f}", fg=COLORS["NEON_CYAN"])
                self.root.after(0, update_ui, rr_results)
            except Exception as e:
                print(f"RR Calc Error: {e}")

    # --- New Analysis Loop for HRV and Age ---
    def analysis_loop(self):
        """Background thread for advanced analytics (Age, HRV)."""
        while self.is_running:
            time.sleep(3.0)  # Update less frequently than main graph
            
            with self.rr_lock:
                # Need sufficient data for reliable HRV/Age (e.g., 30s = ~2500 samples)
                if len(self.rr_buffer) < 2500: continue
                data_snapshot = list(self.rr_buffer)[-3000:] # Take last ~35s
            
            try:
                data_arr = np.array(data_snapshot)
                
                # 1. HRV
                hrv_res = calculate_hrv(data_arr, FS)
                
                # 2. SDPPG & Age
                sdppg_res = calculate_sdppg_index(data_arr, FS)
                ipad_res = calculate_ipad_index(data_arr, FS)
                
                age_res = estimate_age(ipad_res.get('ipad'), sdppg_res)

                '''
                # Use Vessel Age first, then Takazawa, then IPAD as fallback
                estimated_age_val = age_res.get('age_vessel')
                if estimated_age_val is None: estimated_age_val = age_res.get('age_takazawa')
                if estimated_age_val is None: estimated_age_val = age_res.get('age_ipad')
                '''

                # Use Vessel Age first, then Takazawa, then IPAD as fallback
                estimated_age_val = age_res.get('age_takazawa')
                if estimated_age_val is None: estimated_age_val = age_res.get('age_vessel')
                if estimated_age_val is None: estimated_age_val = age_res.get('age_ipad')

                def update_analysis_ui(h_res, age_val):
                    # Update HRV (SDNN is standard)
                    if h_res and h_res.get('sdnn') is not None:
                        self.hrv_label.config(text=f"{h_res['sdnn']:.1f} ms", fg=COLORS["NEON_YELLOW"])
                    else:
                        self.hrv_label.config(text="--", fg=COLORS["TEXT_DIM"])
                        
                    # Update Age
                    if age_val is not None and 10 < age_val < 100:
                        self.age_label.config(text=f"{age_val:.1f} yr", fg=COLORS["NEON_RED"])
                    else:
                        self.age_label.config(text="CALC...", fg=COLORS["TEXT_DIM"])

                self.root.after(0, update_analysis_ui, hrv_res, estimated_age_val)
                
            except Exception as e:
                print(f"Analysis Error: {e}")

    # --- Background Inference Loop ---
    def inference_loop(self):
        import os, sys
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ppg_dir = os.path.join(current_dir, 'ppg2ecg')
        if ppg_dir not in sys.path: sys.path.append(ppg_dir)
            
        try:
            import ppg2ecg as p2e
            import tensorflow as tf
            from scipy.signal import resample
            tf.config.threading.set_intra_op_parallelism_threads(1)
            p2e.MODEL_DIR = os.path.join(ppg_dir, 'weights')
            tf.keras.backend.set_floatx('float64')
            self.ecg_model = p2e.load_cardiogan()
        except Exception:
            return

        last_processed_count = 0
        while self.is_running:
            current_count = self.total_sample_count
            if current_count < 350:
                time.sleep(0.1)
                continue
            if current_count - last_processed_count < 320:
                time.sleep(0.1)
                continue
            
            try:
                raw_snapshot = list(self.fft_buffer)[-450:]
                last_processed_count = current_count
                v_dc = p2e.DC_filter(raw_snapshot, fs=FS, cutoff=0.5)
                clean_ppg = p2e.moving_average_filter(v_dc, window_size=5)
                num_input_samples = int(4.0 * FS) 
                if len(clean_ppg) < num_input_samples: continue
                segment_src = clean_ppg[-num_input_samples:]
                resampled_ppg = resample(segment_src, 512)
                seg_min, seg_max = np.min(resampled_ppg), np.max(resampled_ppg)
                if seg_max - seg_min == 0: segment_norm = resampled_ppg
                else: segment_norm = 2 * ((resampled_ppg - seg_min) / (seg_max - seg_min)) - 1
                segment_input = segment_norm.reshape(1, 512, 1)
                fake_ecg = self.ecg_model(segment_input, training=False)
                ecg_out = fake_ecg.numpy().flatten()
                with self.ecg_lock:
                    self.ecg_playback_queue.extend(ecg_out)
            except Exception:
                time.sleep(1)

    # --- Main Animation Loop ---
    def animate_graph(self):
        if len(self.raw_buffer) >= AVG_WINDOW * 2:
            self.status_label.config(text="STATUS: DATA STREAMING", fg=COLORS["NEON_GREEN"])
            if len(self.raw_buffer) > 0:
                 current_raw = list(self.raw_buffer)
                 processed = self.process_signal(current_raw)
                 context_data = processed[-PPG_DATA_POINTS:] if len(processed) > PPG_DATA_POINTS else processed
                 self.update_scale_factor(context_data, self.canvas.winfo_height())

            if self.fft_enabled: self.draw_fft_mode()
            else:
                if self.graph_mode == "WAVE":
                    self.draw_wave_mode()
                    if self.ecg_view_enabled: self.draw_ecg_wave_mode()
                elif self.graph_mode == "SWEEP":
                    self.draw_sweep_mode()
                    if self.ecg_view_enabled: self.draw_ecg_sweep_mode()
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
        if max_val < 0.01: target = 100.0
        else: target = (height * 0.4) / max_val
        self.current_scale = self.current_scale * 0.9 + target * 0.1

    def draw_sweep_mode(self):
        current_count = self.total_sample_count
        new_samples = current_count - self.last_ppg_sample_count
        if new_samples <= 0: return
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        if len(processed_data) < new_samples: data_to_add = processed_data
        else: data_to_add = processed_data[-new_samples:]
        h = self.canvas.winfo_height()
        for val in data_to_add:
            y_pos = (h / 2) - (val * self.current_scale)
            y_pos = max(10, min(h - 10, y_pos))
            self.sweep_buffer[self.sweep_idx] = y_pos
            self.sweep_idx += 1
            if self.sweep_idx >= PPG_DATA_POINTS: self.sweep_idx = 0
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
            x0, y0 = i * x_stretch, self.sweep_buffer[i]
            x1, y1 = (i + 1) * x_stretch, self.sweep_buffer[(i + 1) % PPG_DATA_POINTS]
            self.canvas.create_line(x0, y0, x1, y1, fill=color, width=2, tags="sweep_line")

    def draw_ecg_sweep_mode(self):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        if dt > 0.1: dt = 0.1
        base_samples = 128.0 * dt
        target_buffer = 512
        current_buffer = len(self.ecg_playback_queue)
        speed_factor = max(0.8, min(1.2, 1.0 + (current_buffer - target_buffer) * 0.001))
        self.ecg_playback_accumulator += base_samples * speed_factor
        points_to_pop = int(self.ecg_playback_accumulator)
        self.ecg_playback_accumulator -= points_to_pop
        new_points = []
        with self.ecg_lock:
            for _ in range(points_to_pop):
                if len(self.ecg_playback_queue) > 0: new_points.append(self.ecg_playback_queue.popleft())
                else: break
        if not new_points: return
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
        for i in range(ECG_DATA_POINTS - 1):
            age = (self.ecg_sweep_idx - i) % ECG_DATA_POINTS
            dim_factor = 1 - (age / ECG_DATA_POINTS)**3
            dark_bg_rgb = (5, 8, 20)
            r = int(base_color_rgb[0] * dim_factor + dark_bg_rgb[0] * (1 - dim_factor))
            g = int(base_color_rgb[1] * dim_factor + dark_bg_rgb[1] * (1 - dim_factor))
            b = int(base_color_rgb[2] * dim_factor + dark_bg_rgb[2] * (1 - dim_factor))
            color = f'#{r:02x}{g:02x}{b:02x}'
            x0, y0 = i * x_stretch, self.ecg_sweep_buffer[i]
            x1, y1 = (i + 1) * x_stretch, self.ecg_sweep_buffer[(i + 1) % ECG_DATA_POINTS]
            self.ecg_canvas.create_line(x0, y0, x1, y1, fill=color, width=2, tags="sweep_line")

    def draw_wave_mode(self):
        current_raw = list(self.raw_buffer)
        processed_data = self.process_signal(current_raw)
        if len(processed_data) < PPG_DATA_POINTS: data_to_show = processed_data
        else: data_to_show = processed_data[-PPG_DATA_POINTS:]
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
            coords.extend([i * x_stretch, y_val])
        if self.line_id is None:
             self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)
        if len(coords) > 4:
             self.canvas.coords(self.line_id, *coords)

    def draw_ecg_wave_mode(self):
        current_ppg_count = self.total_sample_count
        if not hasattr(self, 'last_ecg_sync_ppg_count'):
            self.last_ecg_sync_ppg_count = current_ppg_count
            return
        ppg_delta = current_ppg_count - self.last_ecg_sync_ppg_count
        samples_needed = ppg_delta * (128.0 / FS)
        self.ecg_playback_accumulator += samples_needed
        points_to_pop = int(self.ecg_playback_accumulator)
        self.ecg_playback_accumulator -= points_to_pop
        self.last_ecg_sync_ppg_count = current_ppg_count
        new_points = []
        with self.ecg_lock:
            for _ in range(points_to_pop):
                if len(self.ecg_playback_queue) > 0: new_points.append(self.ecg_playback_queue.popleft())
                else: break
        h = self.ecg_canvas.winfo_height()
        for val in new_points:
            y_pos = (h / 2) - (val * 80)
            y_pos = max(10, min(h - 10, y_pos))
            self.ecg_display_buffer.append(y_pos)
        w = self.ecg_canvas.winfo_width()
        target_ecg_points = int(PPG_DATA_POINTS * (128.0 / FS))
        buffer_list = list(self.ecg_display_buffer)
        if len(buffer_list) > target_ecg_points: data_to_show = buffer_list[-target_ecg_points:]
        else: data_to_show = buffer_list
        x_stretch = w / target_ecg_points if target_ecg_points > 0 else 1
        coords = []
        for i, y_val in enumerate(data_to_show):
            coords.extend([i * x_stretch, y_val])
        if self.ecg_line_id is None:
             self.ecg_line_id = self.ecg_canvas.create_line(0,0,0,0, fill=COLORS["NEON_RED"], width=2, smooth=True)
        if len(coords) > 4:
             self.ecg_canvas.coords(self.ecg_line_id, *coords)

    def draw_fft_mode(self):
        if len(self.fft_buffer) < 200: return
        data = list(self.fft_buffer)[-400:] 
        v = np.array(data) - np.mean(data)
        fft_vals = np.fft.rfft(v)
        fft_freqs = np.fft.rfftfreq(len(v), 1.0/FS)
        magnitude = np.abs(fft_vals)
        mask = (fft_freqs < 40)
        display_freqs = fft_freqs[mask]
        display_mags = magnitude[mask]
        self.canvas.delete("fft_bar")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        bar_width = w / len(display_freqs)
        max_mag = np.max(display_mags) if np.max(display_mags) > 0 else 1
        for i, mag in enumerate(display_mags):
            bar_h = (mag / max_mag) * (h * 0.8)
            x0, y0 = i * bar_width, h
            x1, y1 = x0 + bar_width - 1, h - bar_h
            bar_color = COLORS["NEON_CYAN"]
            if 0.6 <= display_freqs[i] <= 3.0: bar_color = COLORS["NEON_GREEN"]
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=bar_color, outline="", tags="fft_bar")

    def calculate_bpm_update(self):
        if len(self.fft_buffer) < 200: return
        data = list(self.fft_buffer)[-400:]
        v = np.array(data) - np.mean(data)
        fft_vals = np.fft.rfft(v)
        freqs = np.fft.rfftfreq(len(v), 1.0/FS)
        mags = np.abs(fft_vals)
        valid_mask = (freqs >= 0.6) & (freqs <= 3.0)
        if np.any(valid_mask):
            peak_idx = np.argmax(mags[valid_mask])
            p_freq = freqs[valid_mask][peak_idx]
            self.bpm_label.config(text=f"{p_freq * 60:.1f}")
        if len(self.fft_buffer) > 1000:
            self.fft_buffer = self.fft_buffer[-500:]

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