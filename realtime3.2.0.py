import tkinter as tk
from tkinter import font
import math
import collections
import serial
import numpy as np
import time
import threading
import scipy.signal as signal
import sys
import os

# Add ppg_age to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ppg_age_dir = os.path.join(current_dir, 'ppg_age')
if ppg_age_dir not in sys.path:
    sys.path.append(ppg_age_dir)

import ppg_analysis

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
        # [Sync] Drive ECG update based on PPG data arrival
        current_ppg_count = self.total_sample_count
        
        # Initialize if first run
        if not hasattr(self, 'last_ecg_sync_ppg_count'):
            self.last_ecg_sync_ppg_count = current_ppg_count
            return

        ppg_delta = current_ppg_count - self.last_ecg_sync_ppg_count
        
        # Calculate how many ECG samples correspond to the new PPG samples
        # Ratio = 128Hz (ECG) / FS (PPG)
        samples_needed = ppg_delta * (128.0 / FS)
        
        self.ecg_playback_accumulator += samples_needed
        points_to_pop = int(self.ecg_playback_accumulator)
        self.ecg_playback_accumulator -= points_to_pop
        
        # Update sync tracker
        self.last_ecg_sync_ppg_count = current_ppg_count
        
        new_points = []
        with self.ecg_lock:
            for _ in range(points_to_pop):
                if len(self.ecg_playback_queue) > 0:
                    new_points.append(self.ecg_playback_queue.popleft())
                else:
                    break
        
        h = self.ecg_canvas.winfo_height()

        for val in new_points:
            y_pos = (h / 2) - (val * 80)
            y_pos = max(10, min(h - 10, y_pos))
            self.ecg_display_buffer.append(y_pos)
            
        w = self.ecg_canvas.winfo_width()
        
        # [Sync] Ensure X-axis scale matches PPG time window
        target_ecg_points = int(PPG_DATA_POINTS * (128.0 / FS))
        
        buffer_list = list(self.ecg_display_buffer)
        
        if len(buffer_list) > target_ecg_points:
            data_to_show = buffer_list[-target_ecg_points:]
        else:
            data_to_show = buffer_list
            
        x_stretch = w / target_ecg_points if target_ecg_points > 0 else 1
        
        coords = []
        for i, y_val in enumerate(data_to_show):
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
            # self.freq_label.config(text=f"{p_freq:.2f}")

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