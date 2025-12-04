import numpy as np
import filters
import matplotlib.pyplot as plt


def load_data(filepath):
    with open(filepath, "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data


def find_peaks_simple(signal, distance=None, height=None):
    """
    Simple peak detection using numpy.
    Returns indices of peaks.
    """
    dy = np.diff(signal)
    peaks = []
    for i in range(1, len(dy)):
        if dy[i - 1] > 0 and dy[i] <= 0:
            if height is not None and signal[i] < height:
                continue
            peaks.append(i)

    if distance is not None and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= distance:
                filtered_peaks.append(p)
        return filtered_peaks
    return peaks


def analyze_segment(segment, fs, segment_idx):
    # 1. Bandpass Filter
    # Highpass 0.5Hz
    hp_filtered = filters.highpass_filter(segment, fs, 0.5)
    # Lowpass 10Hz (crucial for 2nd derivative stability)
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 10.0)

    # 2. Calculate 2nd Derivative (Acceleration)
    # APG = d2/dt2 (PPG)
    # Using simple finite differences:
    # d1[i] = x[i] - x[i-1]
    # d2[i] = d1[i] - d1[i-1] = x[i] - 2x[i-1] + x[i-2]
    # Or use np.gradient twice
    d1 = np.gradient(lp_filtered)
    sdppg = np.gradient(d1)

    # 3. Identify Waves (a, b, c, d, e)
    # 'a' is the systolic peak of APG (max positive)
    # We first find 'a' waves.

    # Threshold for 'a' wave: assume it's significant positive peak
    # Use 0.6 * max(sdppg) as rough threshold or mean + std
    threshold_a = np.mean(sdppg) + 1.0 * np.std(sdppg)
    min_dist = int(0.5 * fs)  # One beat per 0.5s min

    a_peaks = find_peaks_simple(sdppg, distance=min_dist, height=threshold_a)

    ratios_b_a = []
    ratios_c_a = []
    ratios_d_a = []
    ratios_e_a = []

    print(f"\n--- Segment {segment_idx + 1} Analysis ---")
    print(f"Detected {len(a_peaks)} 'a' waves.")

    valid_beats = 0

    for i in range(len(a_peaks) - 1):
        a_idx = a_peaks[i]
        next_a_idx = a_peaks[i + 1]

        # Search for b, c, d, e between a and next_a
        # But usually they are very close to 'a'.
        # b: first local min after a
        # c: first local max after b
        # d: first local min after c
        # e: first local max after d

        # Define search window: e.g., 400ms after 'a'
        search_window = int(0.4 * fs)
        end_search = min(a_idx + search_window, next_a_idx)

        if end_search <= a_idx:
            continue

        window = sdppg[a_idx:end_search]

        # Find 'b' (min)
        # We look for the first significant valley.
        # Simple approach: find all local minima in window, take the first one.

        # Local minima = local maxima of -signal
        neg_window = -window
        minima_indices = find_peaks_simple(neg_window)

        if not minima_indices:
            continue

        b_local_idx = minima_indices[0]
        b_idx = a_idx + b_local_idx

        # Find 'c' (max) after 'b'
        c_window = sdppg[b_idx:end_search]
        c_candidates = find_peaks_simple(c_window)
        if not c_candidates:
            continue
        c_local_idx = c_candidates[0]
        c_idx = b_idx + c_local_idx

        # Find 'd' (min) after 'c'
        d_window = sdppg[c_idx:end_search]
        d_candidates = find_peaks_simple(-d_window)
        if not d_candidates:
            continue
        d_local_idx = d_candidates[0]
        d_idx = c_idx + d_local_idx

        # Find 'e' (max) after 'd'
        e_window = sdppg[d_idx:end_search]
        e_candidates = find_peaks_simple(e_window)
        if not e_candidates:
            continue
        e_local_idx = e_candidates[0]
        e_idx = d_idx + e_local_idx

        # Get values
        val_a = sdppg[a_idx]
        val_b = sdppg[b_idx]
        val_c = sdppg[c_idx]
        val_d = sdppg[d_idx]
        val_e = sdppg[e_idx]

        # Calculate ratios
        if val_a != 0:
            ratios_b_a.append(val_b / val_a)
            ratios_c_a.append(val_c / val_a)
            ratios_d_a.append(val_d / val_a)
            ratios_e_a.append(val_e / val_a)
            valid_beats += 1

    if valid_beats > 0:
        print(f"  Analyzed {valid_beats} valid beats for SDPPG.")
        print(f"  Average b/a: {np.mean(ratios_b_a):.4f}")
        print(f"  Average c/a: {np.mean(ratios_c_a):.4f}")
        print(f"  Average d/a: {np.mean(ratios_d_a):.4f}")
        print(f"  Average e/a: {np.mean(ratios_e_a):.4f}")

        # Aging Index (b-c-d-e)/a is also common, but user asked for ratios.
        aging_index = (
            np.mean(ratios_b_a)
            - np.mean(ratios_c_a)
            - np.mean(ratios_d_a)
            - np.mean(ratios_e_a)
        )
        print(f"  Aging Index ((b-c-d-e)/a): {aging_index:.4f}")

        # Plotting logic for the first segment only
        if segment_idx == 0:
            plt.figure(figsize=(12, 6))
            # Plot 3 seconds of data (SDPPG is high freq, 3s is enough to see details)
            plot_samples = int(3 * fs)
            start_plot = 1000
            end_plot = start_plot + plot_samples

            time_axis = np.arange(start_plot, end_plot) / fs
            plt.plot(
                time_axis,
                sdppg[start_plot:end_plot],
                label="SDPPG (2nd Deriv)",
                color="purple",
            )

            # Helper to plot points in range
            def plot_points(indices, values, label, color, marker):
                valid_pts = [
                    (i, v)
                    for i, v in zip(indices, values)
                    if start_plot <= i < end_plot
                ]
                if valid_pts:
                    plt.plot(
                        [p[0] / fs for p in valid_pts],
                        [p[1] for p in valid_pts],
                        marker,
                        label=label,
                        color=color,
                    )

            # Re-collect indices for plotting
            # This is a bit inefficient but safe.
            # Ideally we should have stored them in lists during the loop.
            # Let's just iterate the loop again for the plot range?
            # Or better, just store them in the main loop if we are in segment 0.

            # Since I can't easily change the main loop structure without a large replace,
            # I will just re-run the detection logic for the plot window.

            plot_a = []
            plot_b = []
            plot_c = []
            plot_d = []
            plot_e = []

            # Find 'a' peaks in the plot window
            window_sdppg = sdppg[start_plot:end_plot]
            # We need absolute indices
            # Let's just use the already detected 'a_peaks' that fall in range
            a_in_range = [p for p in a_peaks if start_plot <= p < end_plot]

            for a_idx in a_in_range:
                # Find b, c, d, e for this a_idx
                # (Copying logic from main loop)
                try:
                    next_a_idx = -1
                    # Find next a
                    try:
                        idx_in_peaks = a_peaks.index(a_idx)
                        if idx_in_peaks < len(a_peaks) - 1:
                            next_a_idx = a_peaks[idx_in_peaks + 1]
                    except ValueError:
                        pass

                    if next_a_idx == -1:
                        next_a_idx = len(sdppg)

                    search_window = int(0.4 * fs)
                    end_search = min(a_idx + search_window, next_a_idx)

                    if end_search <= a_idx:
                        continue

                    window = sdppg[a_idx:end_search]
                    neg_window = -window
                    minima_indices = find_peaks_simple(neg_window)

                    if not minima_indices:
                        continue
                    b_idx = a_idx + minima_indices[0]

                    c_window = sdppg[b_idx:end_search]
                    c_cands = find_peaks_simple(c_window)
                    if not c_cands:
                        continue
                    c_idx = b_idx + c_cands[0]

                    d_window = sdppg[c_idx:end_search]
                    d_cands = find_peaks_simple(-d_window)
                    if not d_cands:
                        continue
                    d_idx = c_idx + d_cands[0]

                    e_window = sdppg[d_idx:end_search]
                    e_cands = find_peaks_simple(e_window)
                    if not e_cands:
                        continue
                    e_idx = d_idx + e_cands[0]

                    plot_a.append((a_idx, sdppg[a_idx]))
                    plot_b.append((b_idx, sdppg[b_idx]))
                    plot_c.append((c_idx, sdppg[c_idx]))
                    plot_d.append((d_idx, sdppg[d_idx]))
                    plot_e.append((e_idx, sdppg[e_idx]))

                except Exception:
                    pass

            # Plot them
            if plot_a:
                plt.plot(
                    [p[0] / fs for p in plot_a],
                    [p[1] for p in plot_a],
                    "o",
                    label="a",
                    color="red",
                )
            if plot_b:
                plt.plot(
                    [p[0] / fs for p in plot_b],
                    [p[1] for p in plot_b],
                    "o",
                    label="b",
                    color="green",
                )
            if plot_c:
                plt.plot(
                    [p[0] / fs for p in plot_c],
                    [p[1] for p in plot_c],
                    "o",
                    label="c",
                    color="orange",
                )
            if plot_d:
                plt.plot(
                    [p[0] / fs for p in plot_d],
                    [p[1] for p in plot_d],
                    "o",
                    label="d",
                    color="blue",
                )
            if plot_e:
                plt.plot(
                    [p[0] / fs for p in plot_e],
                    [p[1] for p in plot_e],
                    "o",
                    label="e",
                    color="cyan",
                )

            plt.title("SDPPG Analysis (a, b, c, d, e waves)")
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration")
            plt.legend()
            plt.grid(True)
            plt.savefig("sdppg_plot.png")
            print("  Saved visualization to 'sdppg_plot.png'")
            plt.close()

    else:
        print("  Could not detect enough valid a-b-c-d-e sequences.")


def main():
    filepath = "data_180s.txt"
    raw_data = load_data(filepath)

    fs = len(raw_data) / 180.0
    print(f"Estimated Sampling Rate: {fs:.2f} Hz")

    samples_per_60s = int(60 * fs)

    seg1 = raw_data[0:samples_per_60s]
    seg2 = raw_data[samples_per_60s : 2 * samples_per_60s]
    seg3 = raw_data[2 * samples_per_60s :]

    segments = [seg1, seg2, seg3]

    for i, seg in enumerate(segments):
        if len(seg) < samples_per_60s * 0.5:
            continue
        analyze_segment(seg, fs, i)


if __name__ == "__main__":
    main()
