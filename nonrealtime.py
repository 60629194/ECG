import argparse
import re
import sys
from pathlib import Path
import time
#!/usr/bin/env python3

import matplotlib.pyplot as plt

FS = 81.7


def read_integers(path: Path):
    text = path.read_text(encoding="utf-8")
    # Match integers, decimals and scientific notation so values like `357.0`
    # are parsed as a single number instead of two integers ("357" and "0").
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    return [float(n) for n in nums]

def read_realtime_from_com(port="COM5", baud=115200, window_seconds=5):
    """Generator: yields (values_list, fs) repeatedly in real time."""
    ser = serial.Serial(port, baudrate=baud, timeout=0.1)

    # First: estimate FS using ~1 second of data
    print("Estimating sampling rate (FS) ...")
    t0 = time.time()
    timestamps = []

    while time.time() - t0 < 1.0:
        line = ser.readline().strip()
        if not line:
            continue
        try:
            v = float(line)
        except:
            continue
        timestamps.append(time.time())

    if len(timestamps) < 2:
        raise RuntimeError("Not enough data to determine FS")

    diffs = np.diff(timestamps)
    fs = 1 / np.mean(diffs)
    print(f"Estimated FS = {fs:.2f} Hz")

    # Rolling buffer (5 seconds)
    maxlen = int(fs * window_seconds)
    buffer_vals = deque(maxlen=maxlen)

    # Now continue reading forever
    while True:
        line = ser.readline().strip()
        if not line:
            continue

        try:
            value = float(line)
        except:
            continue

        buffer_vals.append(value)

        yield list(buffer_vals), fs


def plot_data(*series, labels=None, title=None, out_path=None, show=True):
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4.5 * n), sharex=True)

    # If there is only one series, axes is not a list
    if n == 1:
        axes = [axes]

    for idx, values in enumerate(series):
        ax = axes[idx]
        x = list(range(len(values)))

        # Pick label if provided
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = f"Series {idx+1}"

        ax.plot(x, values, linestyle='-', linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True)

    if title:
        fig.suptitle(title)

    axes[-1].set_xlabel("Index")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

    if out_path:
        plt.savefig(out_path, dpi=150)
    if show:
        plt.show()




def DC_filter(values, mode='iir', fs=250.0, cutoff=0.5):
    """Remove DC / baseline wander from a 1D sequence of samples.

    Parameters
    - values: sequence of numeric samples (list or iterable)
    - mode: 'mean' for simple DC (mean) subtraction, or 'iir' for a
      first-order high-pass IIR filter which is suitable for ECG baseline
      wander removal. Default is 'iir'.
    - fs: sampling frequency in Hz (used by 'iir' mode). Default 250.0.
    - cutoff: high-pass cutoff frequency in Hz (used by 'iir' mode). Default 0.5 Hz.

    Returns
    - A list of filtered float samples (same length as input).

    Notes
    - The IIR implementation uses a single-pole high-pass derived from the
      bilinear transform of an RC high-pass. It's simple and low-cost for
      streaming signals like ECG. For production, consider a zero-phase
      forward-backward filter (filtfilt) if you need no phase distortion.
    """
    vals = list(values)
    if not vals:
        return []

    if mode == 'mean':
        mu = sum(vals) / len(vals)
        return [float(v - mu) for v in vals]

    # IIR high-pass: first-order
    # compute RC = 1/(2*pi*fc), alpha = RC/(RC + 1/fs)
    from math import pi

    fc = float(cutoff)
    fs = float(fs)
    rc = 1.0 / (2.0 * pi * fc)
    dt = 1.0 / fs
    alpha = rc / (rc + dt)

    out = [0.0] * len(vals)
    out[0] = 0.0
    prev_out = out[0]
    prev_in = float(vals[0])
    for i in range(1, len(vals)):
        x = float(vals[i])
        # y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        y = alpha * (prev_out + x - prev_in)
        out[i] = y
        prev_out = y
        prev_in = x

    return out


def highpass_filter(values, fs=250.0, cutoff=0.67):
    """
    Simple high-pass filter: HP(x) = x - lowpass(x)
    using a single-pole exponential moving average (EMA) low-pass.

    Parameters
    ----------
    values : iterable of numbers
        Input samples.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        High-pass cutoff frequency in Hz.

    Returns
    -------
    list of float : filtered values, same length as input.
    """
    vals = list(values)
    if not vals:
        return []

    from math import pi

    dt = 1.0 / float(fs)
    rc = 1.0 / (2.0 * pi * float(cutoff))
    alpha = dt / (rc + dt)

    out = [0.0] * len(vals)

    # Low-pass initial state
    y_lp = float(vals[0])
    out[0] = 0.0

    for i in range(1, len(vals)):
        x = float(vals[i])
        y_lp = y_lp + alpha * (x - y_lp)   # low-pass
        out[i] = x - y_lp                  # high-pass output

    return out



def lowpass_filter(values, fs=250.0, cutoff=40.0, order=1):
    """Simple single-pole low-pass filter (EMA).

    Parameters
    - values: iterable of numeric samples
    - fs: sampling frequency in Hz (used with cutoff). If caller passes a
      small number (e.g. fs < 10) intending a cutoff, prefer explicit cutoff.
    - cutoff: cutoff frequency in Hz. Default 40 Hz (typical for ECG low-pass)
    - order: ignored (kept for API compatibility)

    Returns a list of floats (same length as input).
    """
    vals = list(values)
    if not vals:
        return []

    # If caller mistakenly passed cutoff in the fs parameter (very small),
    # don't override cutoff when fs is large (>10).
    if cutoff is None:
        cutoff = 40.0

    from math import pi

    fc = float(cutoff)
    fs = float(fs)
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * pi * fc)
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
    """Causal moving average (MA) filter.

    Computes the mean of the last window_size samples (including current).
    First window_size-1 outputs are partial-window averages.

    Parameters
    - values: iterable of numeric samples
    - window_size: number of samples per window (default 5)

    Returns a list of floats (same length as input).
    """
    vals = list(values)
    if not vals:
        return []

    window_size = max(1, int(window_size))
    out = []
    for i in range(len(vals)):
        # Take the window from max(0, i - window_size + 1) to i+1 (inclusive)
        start = max(0, i - window_size + 1)
        window = vals[start:i + 1]
        avg = sum(window) / len(window)
        out.append(avg)

    return out


def main():
    p = argparse.ArgumentParser(description="Read integers from a file and plot them with matplotlib.")
    p.add_argument("file", nargs="?", default="data.txt", help="Input text file (default: raw_data.txt)")
    p.add_argument("--no-show", action="store_true", help="Do not show the interactive window")
    p.add_argument("--out", "-o", help="Save plot to a file (e.g. plot.png)")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    
    values = read_integers(path)
    values_dc = DC_filter(values, fs=FS)
    values_hp = highpass_filter(values_dc, fs=FS, cutoff=0.67)
    values_lp = lowpass_filter(values_hp, fs=FS, cutoff=125)
    values_smooth = moving_average_filter(values_lp, window_size=5)

    if not values:
        print(f"No integers found in {path}", file=sys.stderr)
        sys.exit(1)

    plot_data(
    values_hp,
    values_lp,
    values_smooth,
    labels=["After High-pass Filter", "After Low-pass Filter", "After Moving Average Filter"],
    title=f"Data from {path.name}",
    show=not args.no_show
    )
    plot_data(values_smooth, labels=["After Moving Average Filter"], title="Smoothed Data",out_path="result.png" ,show=not args.no_show)

    print(f"Plotted {len(values)} values.")


if __name__ == "__main__":
    main()