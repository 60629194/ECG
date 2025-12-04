import ppg_analysis
import numpy as np


def load_data(filepath):
    with open(filepath, "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]
    return data


def main():
    filepath = "data_180s.txt"
    print(f"Loading data from {filepath}...")
    raw_data = load_data(filepath)

    # Calculate fs
    fs = len(raw_data) / 180.0
    print(f"Estimated Sampling Rate: {fs:.2f} Hz")

    # Split into 3 segments of 60s
    samples_per_60s = int(60 * fs)

    seg1 = raw_data[0:samples_per_60s]
    seg2 = raw_data[samples_per_60s : 2 * samples_per_60s]
    seg3 = raw_data[2 * samples_per_60s :]

    segments = [seg1, seg2, seg3]

    print("\n" + "=" * 50)
    print("PPG Analysis Results")
    print("=" * 50)

    for i, seg in enumerate(segments):
        if len(seg) < samples_per_60s * 0.5:
            continue

        print(f"\n--- Segment {i + 1} (0-{len(seg) / fs:.1f}s) ---")

        # 1. Stiffness
        stiffness_res = ppg_analysis.calculate_stiffness_index(seg, fs)
        print(f"Vascular Stiffness:")
        if stiffness_res["ri"] is not None:
            print(f"  Reflection Index (RI): {stiffness_res['ri']:.4f}")
            print(f"  (Valid beats: {stiffness_res['valid_beats']})")
        else:
            print("  Could not calculate RI.")

        # 2. SDPPG
        sdppg_res = ppg_analysis.calculate_sdppg_index(seg, fs)
        print(f"SDPPG Indices:")
        if sdppg_res["aging_index"] is not None:
            print(f"  Aging Index: {sdppg_res['aging_index']:.4f}")
            print(f"  b/a: {sdppg_res['b_a']:.4f}")
            print(f"  c/a: {sdppg_res['c_a']:.4f}")
            print(f"  d/a: {sdppg_res['d_a']:.4f}")
            print(f"  e/a: {sdppg_res['e_a']:.4f}")
        else:
            print("  Could not calculate SDPPG indices.")

        # 3. HRV
        hrv_res = ppg_analysis.calculate_hrv(seg, fs)
        print(f"Heart Rate Variability (HRV):")
        if hrv_res["mean_nn"] is not None:
            print(f"  BPM: {hrv_res['bpm']:.2f}")
            print(f"  Mean NN: {hrv_res['mean_nn']:.2f} ms")
            print(f"  SDNN: {hrv_res['sdnn']:.2f} ms")
            print(f"  RMSSD: {hrv_res['rmssd']:.2f} ms")
            print(f"  pNN50: {hrv_res['pnn50']:.2f} %")
        else:
            print("  Could not calculate HRV metrics.")

        # 4. Age Estimation & IPAD
        ipad_res = ppg_analysis.calculate_ipad_index(seg, fs)
        age_estimates = ppg_analysis.estimate_age(ipad_res["ipad"], sdppg_res)

        print(f"Age Estimation & Advanced Indices:")
        if age_estimates["age_ipad"] is not None:
            print(
                f"  Estimated Age (IPAD - Bae et al.): {age_estimates['age_ipad']:.1f} years"
            )
        if age_estimates["age_takazawa"] is not None:
            print(
                f"  Estimated Age (Takazawa et al.): {age_estimates['age_takazawa']:.1f} years"
            )
        if age_estimates["age_vessel"] is not None:
            print(
                f"  Estimated Age (Vessel Age - IJBEM): {age_estimates['age_vessel']:.1f} years"
            )

        if ipad_res["ipad"] is not None:
            print(f"  IPAD Index: {ipad_res['ipad']:.4f}")
            print(f"  IPA (S2/S1): {ipad_res['ipa']:.4f}")
        else:
            print("  Could not calculate IPAD.")


if __name__ == "__main__":
    main()
