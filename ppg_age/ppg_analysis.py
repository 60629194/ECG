import numpy as np
import filters


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


def calculate_stiffness_index(data, fs):
    """
    Calculates Vascular Stiffness Indices (Reflection Index).

    Args:
        data (list/array): Raw PPG data.
        fs (float): Sampling rate in Hz.

    Returns:
        dict: {'ri': float, 'valid_beats': int}
    """
    # 1. Preprocessing
    hp_filtered = filters.highpass_filter(data, fs, 0.5)
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 8.0)

    # 2. Detect Systolic Peaks (P1)
    min_dist = int(0.5 * fs)
    threshold = np.mean(lp_filtered)
    systolic_peaks = find_peaks_simple(lp_filtered, distance=min_dist, height=threshold)

    ris = []
    valid_beats = 0

    for i in range(len(systolic_peaks) - 1):
        p1_idx = systolic_peaks[i]
        next_p1_idx = systolic_peaks[i + 1]

        # Search for P2 between P1 and next P1
        search_start = p1_idx + int(0.1 * fs)
        search_end = next_p1_idx - int(0.1 * fs)

        if search_start >= search_end:
            continue

        interval = lp_filtered[search_start:search_end]
        if len(interval) == 0:
            continue

        # Find local maxima in interval (P2)
        diastolic_candidates = find_peaks_simple(interval)

        p2_idx = -1
        p2_val = -9999

        if diastolic_candidates:
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

            # Find foot (baseline)
            foot_val = 0
            if i > 0:
                prev_p1 = systolic_peaks[i - 1]
                window_min_idx = np.argmin(
                    lp_filtered[max(prev_p1, p1_idx - int(0.4 * fs)) : p1_idx]
                )
                foot_idx = max(prev_p1, p1_idx - int(0.4 * fs)) + window_min_idx
                foot_val = lp_filtered[foot_idx]
            else:
                if p1_idx > 10:
                    window_min_idx = np.argmin(
                        lp_filtered[max(0, p1_idx - int(0.4 * fs)) : p1_idx]
                    )
                    foot_idx = max(0, p1_idx - int(0.4 * fs)) + window_min_idx
                    foot_val = lp_filtered[foot_idx]

            amp_p1 = p1_val - foot_val
            amp_p2 = p2_val - foot_val

            if amp_p1 > 0:
                ri = amp_p2 / amp_p1
                ris.append(ri)
                valid_beats += 1

    avg_ri = np.mean(ris) if valid_beats > 0 else None

    return {"ri": avg_ri, "valid_beats": valid_beats}


def calculate_sdppg_index(data, fs):
    """
    Calculates Second Derivative PPG (SDPPG) indices.

    Args:
        data (list/array): Raw PPG data.
        fs (float): Sampling rate in Hz.

    Returns:
        dict: {'b_a': float, 'c_a': float, 'd_a': float, 'e_a': float, 'aging_index': float, 'valid_beats': int}
    """
    # 1. Preprocessing
    hp_filtered = filters.highpass_filter(data, fs, 0.5)
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 10.0)

    # 2. Derivatives
    d1 = np.gradient(lp_filtered)
    sdppg = np.gradient(d1)

    # 3. Detect 'a' waves
    threshold_a = np.mean(sdppg) + 1.0 * np.std(sdppg)
    min_dist = int(0.5 * fs)
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
        if not minima_indices:
            continue
        b_idx = a_idx + minima_indices[0]

        # Find c (max)
        c_window = sdppg[b_idx:end_search]
        c_cands = find_peaks_simple(c_window)
        if not c_cands:
            continue
        c_idx = b_idx + c_cands[0]

        # Find d (min)
        d_window = sdppg[c_idx:end_search]
        d_cands = find_peaks_simple(-d_window)
        if not d_cands:
            continue
        d_idx = c_idx + d_cands[0]

        # Find e (max)
        e_window = sdppg[d_idx:end_search]
        e_cands = find_peaks_simple(e_window)
        if not e_cands:
            continue
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
        return {
            "b_a": None,
            "c_a": None,
            "d_a": None,
            "e_a": None,
            "aging_index": None,
            "valid_beats": 0,
        }


def calculate_hrv(data, fs):
    """
    Calculates Heart Rate Variability (HRV) metrics.

    Args:
        data (list/array): Raw PPG data.
        fs (float): Sampling rate in Hz.

    Returns:
        dict: {'mean_nn': float, 'sdnn': float, 'rmssd': float, 'pnn50': float, 'bpm': float}
    """
    # 1. Preprocessing
    # Bandpass 0.5-5Hz for peak detection
    hp_filtered = filters.highpass_filter(data, fs, 0.5)
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 5.0)

    # 2. Detect Peaks
    min_dist = int(0.5 * fs)  # 120 bpm max
    threshold = np.mean(lp_filtered)
    peaks = find_peaks_simple(lp_filtered, distance=min_dist, height=threshold)

    if len(peaks) < 2:
        return {
            "mean_nn": None,
            "sdnn": None,
            "rmssd": None,
            "pnn50": None,
            "bpm": None,
        }

    # 3. Calculate NN intervals (ms)
    # NN = Normal-to-Normal intervals
    nn_intervals = np.diff(peaks) / fs * 1000.0  # Convert to ms

    # Filter outliers (optional but good practice)
    # Simple range check: 300ms (200bpm) to 1500ms (40bpm)
    nn_intervals = [nn for nn in nn_intervals if 300 <= nn <= 1500]

    if len(nn_intervals) < 2:
        return {
            "mean_nn": None,
            "sdnn": None,
            "rmssd": None,
            "pnn50": None,
            "bpm": None,
        }

    # 4. Calculate Metrics
    mean_nn = np.mean(nn_intervals)
    sdnn = np.std(nn_intervals, ddof=1)  # Standard deviation

    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))

    nn50 = np.sum(np.abs(diff_nn) > 50)
    pnn50 = (nn50 / len(diff_nn)) * 100.0 if len(diff_nn) > 0 else 0.0

    bpm = 60000.0 / mean_nn

    return {
        "mean_nn": mean_nn,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "bpm": bpm,
    }


def calculate_ipad_index(data, fs):
    """
    Calculates IPAD (Inflection Point Area and d-peak) index.
    IPAD = (Area_S2 / Area_S1) + (d/a)

    Args:
        data (list/array): Raw PPG data.
        fs (float): Sampling rate in Hz.

    Returns:
        dict: {'ipad': float, 'ipa': float, 's2_s1_ratio': float, 'valid_beats': int}
    """
    # 1. Preprocessing
    hp_filtered = filters.highpass_filter(data, fs, 0.5)
    lp_filtered = filters.lowpass_filter(hp_filtered, fs, 10.0)

    # Robust Normalization to [0.8, 1.8]
    # Use 5th and 95th percentiles to avoid outliers
    p5 = np.percentile(lp_filtered, 5)
    p95 = np.percentile(lp_filtered, 95)

    if p95 > p5:
        # Scale to [0, 1] first based on percentiles
        normalized = (np.array(lp_filtered) - p5) / (p95 - p5)
        # Scale to [0.8, 1.8] (range = 1.0)
        # Note: Values outside p5-p95 will be <0.8 or >1.8
        lp_filtered = 0.8 + normalized * 1.0

        # Clip to ensure strict range [0.8, 1.8] as per user request/paper visual
        lp_filtered = np.clip(lp_filtered, 0.8, 1.8)

    # 2. Derivatives
    d1 = np.gradient(lp_filtered)
    sdppg = np.gradient(d1)

    # 3. Detect 'a' waves
    threshold_a = np.mean(sdppg) + 1.0 * np.std(sdppg)
    min_dist = int(0.5 * fs)
    a_peaks = find_peaks_simple(sdppg, distance=min_dist, height=threshold_a)

    ipad_values = []
    ipa_values = []
    s2_s1_ratios = []
    valid_beats = 0

    for i in range(len(a_peaks) - 1):
        a_idx = a_peaks[i]
        next_a_idx = a_peaks[i + 1]

        # Define beat window
        search_window = int(0.6 * fs)  # Slightly longer to capture full wave
        end_search = min(a_idx + search_window, next_a_idx)

        if end_search <= a_idx:
            continue

        window = sdppg[a_idx:end_search]

        # Find zero crossings in this window
        # Zero crossing: sign change
        zero_crossings = np.where(np.diff(np.signbit(window)))[0]

        # We expect at least 4 zero crossings for a, b, c, d, e waves
        # S1: Area under 'a' (start to 1st ZC)
        # S2: Area under 'b' (1st ZC to 2nd ZC)
        # S3: Area under 'c' (2nd ZC to 3rd ZC)
        # S4: Area under 'd' (3rd ZC to 4th ZC)

        if len(zero_crossings) < 4:
            continue

        # Indices relative to window start
        z1 = zero_crossings[0]
        z2 = zero_crossings[1]
        z3 = zero_crossings[2]
        z4 = zero_crossings[3]

        # Calculate Areas using Trapezoidal rule
        # S1: 0 to z1
        s1_area = np.trapz(np.abs(window[0 : z1 + 1]))

        # S2: z1 to z2
        s2_area = np.trapz(np.abs(window[z1 : z2 + 1]))

        # Find 'd' peak value for (d/a)
        # 'd' is usually in the S4 region (z3 to z4) or around there
        # Let's find 'd' as min in z3 to z4 (or end of window)
        d_search_end = min(z4 + 10, len(window))  # Extend a bit
        d_region = window[z3:d_search_end]
        if len(d_region) == 0:
            continue

        d_val = np.min(d_region)
        a_val = sdppg[a_idx]

        if a_val == 0 or s1_area == 0:
            continue

        d_a_ratio = d_val / a_val

        # IPA = S2 / S1 (Common definition, sometimes S3/S1 etc, but user mentioned S2/S1)
        ipa = s2_area / s1_area

        # IPAD = IPA + (d/a)
        # Note: d/a is usually negative.
        # User formula: IPAD = (Area ratio) + d/a
        ipad = ipa + d_a_ratio

        ipad_values.append(ipad)
        ipa_values.append(ipa)
        s2_s1_ratios.append(ipa)
        valid_beats += 1

    if valid_beats > 0:
        return {
            "ipad": np.mean(ipad_values),
            "ipa": np.mean(ipa_values),
            "s2_s1_ratio": np.mean(s2_s1_ratios),
            "valid_beats": valid_beats,
        }
    else:
        return {"ipad": None, "ipa": None, "s2_s1_ratio": None, "valid_beats": 0}


def estimate_age(ipad_index, sdppg_indices):
    """
    Estimates age using multiple methods:
    1. IPAD (Bae et al.): Age = (IPAD - 0.325) / -0.00748
    2. Takazawa et al.: Age = (Aging_Index + 1.515) / 0.023
    3. Vessel Age (IJBEM): Age = 36.89 + 6.62(b/a) - 27.05(c/a) - 24.68(d/a) + 2.44(e/a)

    Args:
        ipad_index (float): Result from calculate_ipad_index.
        sdppg_indices (dict): Result from calculate_sdppg_index.

    Returns:
        dict: {'age_ipad': float, 'age_takazawa': float, 'age_vessel': float}
    """
    estimates = {"age_ipad": None, "age_takazawa": None, "age_vessel": None}

    # 1. IPAD Method
    if ipad_index is not None:
        estimates["age_ipad"] = (ipad_index - 0.325) / -0.00748

    # 2. Takazawa Method (Aging Index)
    ai = sdppg_indices.get("aging_index")
    if ai is not None:
        # Formula: AI = 0.023 * Age - 1.515
        estimates["age_takazawa"] = (ai + 1.515) / 0.023

    # 3. Vessel Age Method (IJBEM)
    b_a = sdppg_indices.get("b_a")
    c_a = sdppg_indices.get("c_a")
    d_a = sdppg_indices.get("d_a")
    e_a = sdppg_indices.get("e_a")

    if all(v is not None for v in [b_a, c_a, d_a, e_a]):
        # Formula: Age = 36.89 + 6.62(b/a) - 27.05(c/a) - 24.68(d/a) + 2.44(e/a)
        estimates["age_vessel"] = (
            36.89 + (6.62 * b_a) - (27.05 * c_a) - (24.68 * d_a) + (2.44 * e_a)
        )

    return estimates
