import tkinter as tk
from tkinter import font
import math
import random
import collections

# --- Configuration and Colors ---
# The color palette defines the futuristic look.
COLORS = {
    "BG_MAIN": "#050814",      # Deepest dark blue/black background
    "BG_PANEL": "#0c1229",     # Slightly lighter panel background
    "NEON_CYAN": "#00fff9",    # Primary accent glow color
    "NEON_GREEN": "#39ff14",   # Secondary accent for healthy stats
    "NEON_RED": "#ff3333",     # Accent for warnings (not used heavily here)
    "TEXT_MAIN": "#e6f0ff",    # Whitish-blue text
    "TEXT_DIM": "#6b7c99",     # Dimmed text for labels
    "GRID_LINE": "#1a264f"     # Very faint lines for the graph grid
}

# Settings for the graph simulation
GRAPH_HEIGHT = 400
GRAPH_WIDTH = 800
DATA_POINTS = 200 # How many points to keep in history
REFRESH_RATE_MS = 30 # Lower = smoother but higher CPU usage

class FuturisticPPGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("BIO-METRIC SENSOR INTERFACE // PPG MODULE")
        self.root.geometry("1024x600")
        self.root.configure(bg=COLORS["BG_MAIN"])
        
        # Initialize data structure for plotting
        # Deque is efficient for sliding windows of data
        self.data_buffer = collections.deque([GRAPH_HEIGHT/2] * DATA_POINTS, maxlen=DATA_POINTS)
        self.sim_angle = 0 # Used for generating fake sine wave data

        # Custom Fonts
        # Trying to find fonts that look technical. Fallbacks included.
        self.header_font = font.Font(family="Consolas", size=14, weight="bold")
        self.digit_font = font.Font(family="Courier New", size=28, weight="bold")
        self.label_font = font.Font(family="Calibri", size=10)

        self.setup_gui()
        
        # Start the simulation loop
        self.animate_graph()

    def create_neon_border(self, parent, accent_color, padding=2):
        """
        Helper function to create a glowing border effect.
        It creates an outer frame (the glow) holding an inner frame (the content background).
        """
        outer = tk.Frame(parent, bg=accent_color, padx=padding, pady=padding)
        inner = tk.Frame(outer, bg=COLORS["BG_PANEL"])
        inner.pack(fill="both", expand=True)
        return outer, inner

    def setup_gui(self):
        # --- Main Layout Structure ---
        # Using grid layout for precision structuring
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # 1. Top Header Bar
        header_frame = tk.Frame(self.root, bg=COLORS["BG_MAIN"], height=40)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10,5))
        
        tk.Label(header_frame, text="STATUS: ACTIVE // SCANNING...", 
                 bg=COLORS["BG_MAIN"], fg=COLORS["NEON_CYAN"], 
                 font=self.label_font).pack(side="left")
        tk.Label(header_frame, text="SYSTEM ID: PPG-IR-X99", 
                 bg=COLORS["BG_MAIN"], fg=COLORS["TEXT_DIM"], 
                 font=self.label_font).pack(side="right")

        # 2. Left Sidebar (Stats)
        left_sidebar_outer, self.left_sidebar = self.create_neon_border(self.root, COLORS["NEON_CYAN"])
        left_sidebar_outer.grid(row=1, column=0, sticky="ns", padx=(10, 5), pady=5)
        # Fix width of sidebar
        self.left_sidebar.configure(width=180, height=500)
        self.left_sidebar.pack_propagate(False)

        # Add Stat Modules to Left Sidebar
        self.bpm_label = self.add_stat_module(self.left_sidebar, "HEART RATE (BPM)", COLORS["NEON_GREEN"])
        self.spo2_label = self.add_stat_module(self.left_sidebar, "OXYGEN SAT (SpO2)", COLORS["NEON_CYAN"])
        self.add_decorative_text(self.left_sidebar)

        # 3. Center Graph Area (The Core)
        graph_outer, graph_inner = self.create_neon_border(self.root, COLORS["NEON_CYAN"], padding=3)
        graph_outer.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Canvas for drawing the PPG signal
        self.canvas = tk.Canvas(graph_inner, bg="black", 
                                height=GRAPH_HEIGHT, width=GRAPH_WIDTH,
                                highlightthickness=0) # Remove default ugly white border
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Draw initial grid lines on canvas
        self.draw_grid()
        # Initialize the line object on the canvas
        self.line_id = self.canvas.create_line(0,0,0,0, fill=COLORS["NEON_CYAN"], width=2, smooth=True)

        # 4. Bottom Control Bar (Placeholder)
        bottom_bar_outer, bottom_bar = self.create_neon_border(self.root, COLORS["TEXT_DIM"], padding=1)
        bottom_bar_outer.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(5,10))
        tk.Label(bottom_bar, text=">> DATA STREAM ESTABLISHED. WAITING FOR USER INPUT.", 
                 bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], font=self.label_font, anchor="w").pack(fill="x", padx=5)


    def add_stat_module(self, parent, title, glow_color):
        """Creates a single statistic display block."""
        frame = tk.Frame(parent, bg=COLORS["BG_PANEL"], pady=15)
        frame.pack(fill="x", padx=5)

        tk.Label(frame, text=title, bg=COLORS["BG_PANEL"], fg=COLORS["TEXT_DIM"], 
                 font=self.label_font, anchor="w").pack(fill="x")
        
        # The digital readout label
        value_label = tk.Label(frame, text="--", bg=COLORS["BG_PANEL"], fg=glow_color, 
                               font=self.digit_font)
        value_label.pack(anchor="e")
        
        # Decorative separator line
        tk.Frame(frame, bg=COLORS["GRID_LINE"], height=2).pack(fill="x", pady=(5,0))
        return value_label

    def add_decorative_text(self, parent):
        """Adds some "technobabble" text to fill space nicely."""
        frame = tk.Frame(parent, bg=COLORS["BG_PANEL"], pady=20)
        frame.pack(fill="both", expand=True, padx=5)
        txt = "SENSOR CALIBRATION\nOK\n\nIR EMITTER STATUS\nNOMINAL\n\nAMBIENT NOISE\nFILTERING ACTIVE"
        tk.Label(frame, text=txt, bg=COLORS["BG_PANEL"], fg=COLORS["GRID_LINE"], 
                 font=self.label_font, justify="left", anchor="nw").pack(fill="both", expand=True)


    def draw_grid(self):
        """Draws a technical looking grid on the canvas."""
        w = self.canvas.winfo_reqwidth()
        h = self.canvas.winfo_reqheight()
        
        # Vertical grid lines
        for i in range(0, w, 40):
            color = COLORS["GRID_LINE"]
            # Make every 4th line slightly brighter
            if (i % 160) == 0: color = COLORS["TEXT_DIM"]
            self.canvas.create_line(i, 0, i, h, fill=color, dash=(4, 4))
            
        # Horizontal grid lines (center line brighter)
        center_y = h / 2
        self.canvas.create_line(0, center_y, w, center_y, fill=COLORS["TEXT_DIM"])
        
        for i in range(0, h, 40):
             if i == int(center_y): continue
             self.canvas.create_line(0, i, w, i, fill=COLORS["GRID_LINE"], dash=(4, 4))

    # --- SIMULATION AREA --- 
    # REPLACE THIS SECTION WITH YOUR REAL SENSOR DATA LATER
    def get_fake_sensor_data(self):
        """Generates a PPG-like waveform combining sine waves and noise."""
        self.sim_angle += 0.15
        # Main pulse wave
        wave = math.sin(self.sim_angle) * 100 
        # Dicrotic notch simulation (secondary bump)
        notch = math.sin(self.sim_angle * 2 + 0.5) * 30
        # Random sensor noise
        noise = random.uniform(-5, 5)
        
        base_line = GRAPH_HEIGHT / 2
        # Invert because canvas Y coordinates go down
        val = base_line - (wave + notch + noise)
        return val

    def update_stats_display(self, val):
        # Periodically update BPM display with fake data just for visual effect
        if random.random() > 0.95:
             fake_bpm = random.randint(65, 85)
             self.bpm_label.config(text=str(fake_bpm))
             fake_spo2 = random.randint(96, 99)
             self.spo2_label.config(text=str(fake_spo2) + "%")

    def animate_graph(self):
        """The main loop that redraws the canvas."""
        # 1. Get new data point
        new_val = self.get_fake_sensor_data()
        self.data_buffer.append(new_val)
        
        # 2. Update stats (optional)
        self.update_stats_display(new_val)

        # 3. Prepare coordinates for drawing line
        # We need a flat list of [x1, y1, x2, y2, x3, y3...]
        w = self.canvas.winfo_width()
        x_stretch = w / DATA_POINTS
        
        coords = []
        for i, y_val in enumerate(self.data_buffer):
            coords.append(i * x_stretch)
            coords.append(y_val)
            
        # 4. Update the pre-existing line on the canvas
        # Using coordinates change is much faster than deleting/creating lines constantly
        if len(coords) > 4:
             self.canvas.coords(self.line_id, *coords)

        # Schedule next update
        self.root.after(REFRESH_RATE_MS, self.animate_graph)

if __name__ == "__main__":
    root = tk.Tk()
    app = FuturisticPPGViewer(root)
    # Ensure the grid is drawn correctly after window manager initializes
    root.update_idletasks()
    app.draw_grid()
    root.mainloop()
