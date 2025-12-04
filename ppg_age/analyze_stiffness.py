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
    # Find local maxima
    # dy > 0 then dy < 0
    dy = np.diff(signal)
    peaks = []
    for i in range(1, len(dy)):
        if dy[i - 1] > 0 and dy[i] <= 0:
            # potential peak at i
            if height is not None and signal[i] < height:
                continue
            peaks.append(i)

    # Filter by distance
    if distance is not None and len(peaks) > 0:
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= distance:
                filtered_peaks.append(p)
        return filtered_peaks
    return peaks


def analyze_segment(segment, fs, segment_idx):
    # 1. Remove trend (High-pass)
    # The user mentioned an upward trend. 0.5Hz HP filter is good for removing baseline wander/trend.
    hp_filtered = filters.highpass_filter(segment, fs, 0.5)

    # 2. Smooth (Low-pass)
    # To remove high freq noise for better peak detection.
    # PPG main energy is < 10Hz.
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 8.0)

    # 3. Detect Systolic Peaks (Main peaks)
    # Heart rate is typically 60-100 bpm (1-1.6 Hz).
    # Min distance between peaks ~ 0.5s (for 120bpm) -> 0.5 * fs samples.
    min_dist = int(0.5 * fs)
    # Use a dynamic height threshold (e.g., above mean)
    threshold = np.mean(lp_filtered)
    systolic_peaks = find_peaks_simple(lp_filtered, distance=min_dist, height=threshold)

    ris = []
    valid_beats = 0

    print(f"\n--- Segment {segment_idx + 1} Analysis ---")
    print(f"Detected {len(systolic_peaks)} beats.")

    # For plotting a few examples
    debug_plots = 0

    for i in range(len(systolic_peaks) - 1):
        p1_idx = systolic_peaks[i]
        next_p1_idx = systolic_peaks[i + 1]

        # Look for diastolic peak/inflection between p1 and next_p1
        # Usually it's within the first 400ms after P1? Or just search the interval.
        # Let's search in the interval [p1_idx + offset, next_p1_idx]
        # We need to avoid the immediate descent of P1.

        # Search window: start a bit after P1, end before next P1
        search_start = p1_idx + int(0.1 * fs)  # 100ms after P1
        search_end = next_p1_idx - int(0.1 * fs)

        if search_start >= search_end:
            continue

        interval = lp_filtered[search_start:search_end]
        if len(interval) == 0:
            continue

        # Find local maxima in this interval (Reflected wave)
        # Note: indices returned are relative to search_start
        # We might find multiple, usually the first significant one or the highest one is P2.
        # Sometimes P2 is just an inflection point (2nd derivative needed), but user asked for "second wave peak".
        # We will look for a local max.
        diastolic_candidates = find_peaks_simple(interval)

        p2_idx = -1
        p2_val = -9999

        if diastolic_candidates:
            # Take the highest peak in the interval as P2
            # Or the first one? Usually P2 is the dicrotic wave.
            # Let's take the highest for now as it's "Reflection Index".
            # Actually, often P2 is lower than P1.
            best_cand = -1
            max_val = -float("inf")
            for cand in diastolic_candidates:
                val = interval[cand]
                if val > max_val:
                    max_val = val
                    best_cand = cand

            if best_cand != -1:
                p2_idx = search_start + best_cand
                p2_val = max_val

        if p2_idx != -1:
            p1_val = lp_filtered[p1_idx]

            # RI = P2 / P1
            # Note: signals are AC coupled (centered around 0 approx due to HP filter).
            # But RI is usually calculated on the absolute amplitude from the baseline *foot* of the pulse.
            # Since we HP filtered, the baseline is roughly 0.
            # However, strictly speaking, we should measure height from the pulse onset (foot).
            # Let's find the foot (minima before P1).

            # Find foot: minimum between previous peak and current P1
            # For the first peak, we might not have a previous peak.
            # Let's look back from P1.
            lookback_limit = int(0.5 * fs)
            search_foot_start = max(0, p1_idx - lookback_limit)
            # The foot is the minimum in [search_foot_start, p1_idx]
            # But we need to be careful not to pick the previous beat's dicrotic notch.
            # Usually foot is the global min in the window between beats.

            # Let's define the beat window as [prev_peak, current_peak].
            # If i=0, we can't find prev peak easily.

            foot_val = 0  # Default fallback
            if i > 0:
                prev_p1 = systolic_peaks[i - 1]
                # Search min between prev_p1 and p1
                # Usually foot is closer to p1.
                # Let's search in [p1 - 0.4s, p1]
                window_min_idx = np.argmin(
                    lp_filtered[max(prev_p1, p1_idx - int(0.4 * fs)) : p1_idx]
                )
                foot_idx = max(prev_p1, p1_idx - int(0.4 * fs)) + window_min_idx
                foot_val = lp_filtered[foot_idx]
            else:
                # For first beat, just search a bit back
                if p1_idx > 10:
                    window_min_idx = np.argmin(
                        lp_filtered[max(0, p1_idx - int(0.4 * fs)) : p1_idx]
                    )
                    foot_idx = max(0, p1_idx - int(0.4 * fs)) + window_min_idx
                    foot_val = lp_filtered[foot_idx]

            # Amplitudes relative to foot
            amp_p1 = p1_val - foot_val
            amp_p2 = p2_val - foot_val

            if amp_p1 > 0:  # Avoid div by zero
                ri = amp_p2 / amp_p1
                ris.append(ri)
                valid_beats += 1

    if valid_beats > 0:
        avg_ri = np.mean(ris)
        print(f"  Average Reflection Index (RI): {avg_ri:.4f}")
        print(f"  (Calculated from {valid_beats} valid beats)")

        # Plotting logic for the first segment only
        if segment_idx == 0:
            plt.figure(figsize=(12, 6))
            # Plot 5 seconds of data
            plot_samples = int(5 * fs)
            start_plot = 1000  # arbitrary offset to avoid edge artifacts
            end_plot = start_plot + plot_samples

            time_axis = np.arange(start_plot, end_plot) / fs
            plt.plot(
                time_axis,
                lp_filtered[start_plot:end_plot],
                label="Filtered PPG",
                color="blue",
            )

            # Plot P1 peaks in this range
            p1_in_range = [p for p in systolic_peaks if start_plot <= p < end_plot]
            plt.plot(
                [p / fs for p in p1_in_range],
                [lp_filtered[p] for p in p1_in_range],
                "ro",
                label="Systolic (P1)",
            )

            # Re-detect P2 for plotting (since we didn't store them all)
            p2_indices = []
            p2_values = []

            for p1 in p1_in_range:
                # Find the corresponding P2
                # This duplicates logic but is safest for visualization without refactoring everything
                try:
                    idx_in_peaks = systolic_peaks.index(p1)
                    if idx_in_peaks < len(systolic_peaks) - 1:
                        next_p1 = systolic_peaks[idx_in_peaks + 1]
                        search_start = p1 + int(0.1 * fs)
                        search_end = next_p1 - int(0.1 * fs)
                        if search_start < search_end:
                            interval = lp_filtered[search_start:search_end]
                            if len(interval) > 0:
                                cands = find_peaks_simple(interval)
                                if cands:
                                    # Find best cand (max)
                                    best_cand = -1
                                    max_val = -float("inf")
                                    for cand in cands:
                                        val = interval[cand]
                                        if val > max_val:
                                            max_val = val
                                            best_cand = cand
                                    if best_cand != -1:
                                        p2_abs = search_start + best_cand
                                        p2_indices.append(p2_abs)
                                        p2_values.append(lp_filtered[p2_abs])
                except ValueError:
                    pass

            plt.plot(
                [p / fs for p in p2_indices], p2_values, "go", label="Diastolic (P2)"
            )

            plt.title("PPG Stiffness Analysis (Reflection Index)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.savefig("stiffness_plot.png")
            print("  Saved visualization to 'stiffness_plot.png'")
            plt.close()

    else:
        print("  Could not detect enough valid P1-P2 pairs.")


def main():
    filepath = "data_180s.txt"
    raw_data = load_data(filepath)

    # Calculate fs
    # 17342 samples in 180s (approx) -> check exact if possible, but user said 180s.
    # 17342 / 180 = 96.34 Hz
    fs = len(raw_data) / 180.0
    print(f"Estimated Sampling Rate: {fs:.2f} Hz")

    # Split into 3 segments of 60s
    samples_per_60s = int(60 * fs)

    seg1 = raw_data[0:samples_per_60s]
    seg2 = raw_data[samples_per_60s : 2 * samples_per_60s]
    seg3 = raw_data[2 * samples_per_60s :]  # Rest

    segments = [seg1, seg2, seg3]

    for i, seg in enumerate(segments):
        if len(seg) < samples_per_60s * 0.5:  # Skip if too short
            continue
        analyze_segment(seg, fs, i)


if __name__ == "__main__":
    main()
