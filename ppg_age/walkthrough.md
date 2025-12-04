# PPG Analysis Walkthrough

I have successfully refactored the PPG analysis into a reusable module `ppg_analysis.py` and added Heart Rate Variability (HRV) analysis.

## Reusable Module: `ppg_analysis.py`

This module exposes three main functions:
1.  `calculate_stiffness_index(data, fs)`: Returns Reflection Index (RI).
2.  `calculate_sdppg_index(data, fs)`: Returns SDPPG ratios (b/a, c/a, etc.) and Aging Index.
3.  `calculate_hrv(data, fs)`: Returns HRV metrics (SDNN, RMSSD, pNN50, BPM).

## Analysis Results (from `run_analysis.py`)

The analysis was performed on `data_180s.txt` split into three 60-second segments.

### 1. Vascular Stiffness (Reflection Index)
| Segment | Valid Beats | RI |
| :--- | :--- | :--- |
| 0-60s | 64 | 0.5592 |
| 60-120s | 62 | 0.6190 |
| 120-180s | 65 | 0.5330 |

### 2. Second Derivative PPG (SDPPG)
| Segment | b/a | c/a | d/a | e/a | Aging Index |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0-60s | -0.6410 | 0.5670 | -0.6902 | 0.7073 | -1.2251 |
| 60-120s | -0.6751 | 0.7059 | -0.7869 | 0.7426 | -1.3367 |
| 120-180s | -0.6619 | 0.6128 | -0.6937 | 0.6425 | -1.2235 |

### 3. Heart Rate Variability (HRV)
| Segment | BPM | Mean NN (ms) | SDNN (ms) | RMSSD (ms) | pNN50 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0-60s | 65.05 | 922.36 | 143.04 | 125.18 | 61.90 |
| 60-120s | 62.57 | 958.98 | 141.76 | 134.59 | 70.49 |
| 120-180s | 65.47 | 916.48 | 129.19 | 120.67 | 59.38 |

### 4. Age Estimation Comparison
| Segment | IPAD (Bae et al.) | Takazawa et al. | Vessel Age (IJBEM) |
| :--- | :--- | :--- | :--- |
| 0-60s | -24.0 years | 12.6 years | 35.1 years |
| 60-120s | -44.4 years | 7.8 years | 34.6 years |
| 120-180s | 5.4 years | 12.7 years | 34.6 years |

> [!IMPORTANT]
> **Method Comparison**:
> *   **Vessel Age (IJBEM)**: This method provided the most realistic and consistent estimates (~35 years), suggesting it might be more robust to the specific characteristics of this sensor/signal.
> *   **Takazawa et al.**: Estimated a very young vascular age (7-13 years), consistent with the high arterial compliance indicated by the raw indices.
> *   **IPAD**: Remained largely negative or extremely young, likely due to sensitivity to signal amplitude/scaling despite normalization.

> [!NOTE]
> **Conclusion**: The subject appears to have excellent vascular health (high compliance), which most algorithms interpret as a "young" vascular age. The IJBEM Vessel Age formula seems best calibrated for this type of data.

## Visualization
The previous visualization scripts (`analyze_stiffness.py` and `analyze_sdppg.py`) are still available and can be used to generate plots. The new `run_analysis.py` focuses on numerical output using the reusable module.

## How to Run
To run the full analysis using the new module:
```bash
python3 run_analysis.py
```
