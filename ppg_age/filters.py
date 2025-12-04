import math
import numpy as np

HP_CUTOFF = 0.5
LP_CUTOFF = 150.0

def lowpass_filter(vals, fs, cutoff):
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

def highpass_filter(vals, fs, cutoff):
    if len(vals) < 2: return vals
    
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = rc / (rc + dt)
    
    out = [0.0] * len(vals)
    out[0] = 0.0 
    
    for i in range(1, len(vals)):
        out[i] = alpha * (out[i-1] + vals[i] - vals[i-1])
    
    return out

def moving_average_filter(vals, window_size):
    if len(vals) == 0: 
        return []
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(vals, window, 'same').tolist()
