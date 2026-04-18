#!/usr/bin/env python3
"""
Refine benchmark data extracted from Kokkoris et al. (2009) figures.

Pipeline:
1. Sort by ascending x, remove duplicates
2. Remove outlier spikes
3. Trim known extraction artifacts (low-pressure Te dip, etc.)
4. Savitzky-Golay smoothing (wider window for noisy curves)
5. Enforce physical constraints (monotonicity where expected)
"""
import os
import numpy as np
from scipy.signal import savgol_filter

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'refined')

# Pre-processed files: pass through with sort+dedup only (no smoothing/trim/despike)
PASSTHROUGH = {
    'kokkoris_Te_vs_poff.csv',
    'kokkoris_Te_vs_power.csv',
    'kokkoris_alpha_vs_poff.csv',
    'kokkoris_ne_vs_poff.csv',
    'kokkoris_ne_vs_power.csv',
    'kokkoris_nFm_vs_poff.csv',
    'kokkoris_nSF3_vs_power.csv',
    'kokkoris_nSF3p_vs_power.csv',
    'kokkoris_nSF4p_vs_power.csv',
    'kokkoris_nF_model_vs_power.csv',
    'kokkoris_dp_model_vs_poff.csv',
    'kokkoris_dp_model_vs_power.csv',
    'kokkoris_nF_vs_poff.csv',
    'kokkoris_nSF3p_vs_poff.csv',
    'kokkoris_nSF4p_vs_poff.csv',
    'kokkoris_nSF5p_vs_poff.csv',
}

# Trim extraction artifacts below this x value
TRIM_LOW_X = {
}

# Files needing wider smoothing window (non-passthrough only)
WIDER_SMOOTH = {
    'kokkoris_nF_model_vs_poff.csv': 11,
    'kokkoris_alpha_vs_power.csv': 13,
}

# Experimental data files: snap x-values to exact known experimental power/pressure values
# These are sparse points (3-4 pts) where the exact x is known from the paper
SNAP_X = {
    'kokkoris_dp_exp_vs_power.csv': [1000, 2000, 3000],
    'kokkoris_dp_exp_vs_poff.csv': [0.921, 1.5, 2.5, 3.5],
    'kokkoris_nF_exp_056_vs_power.csv': [1000, 2000, 3000],
    'kokkoris_nF_exp_076_vs_power.csv': [1000, 2000, 3000],
    'kokkoris_nF_exp_056_vs_poff.csv': [0.921, 1.5, 2.5, 3.5],
    'kokkoris_nF_exp_076_vs_poff.csv': [0.921, 1.5, 2.5, 3.5],
}

def snap_x_values(x, y, exact_x):
    """Snap digitized x-values to nearest known exact values."""
    exact = np.array(exact_x, dtype=float)
    x_new = x.copy()
    for i in range(len(x)):
        closest = exact[np.argmin(np.abs(exact - x[i]))]
        x_new[i] = closest
    return x_new, y

def load_raw(path):
    x, y = [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try: x.append(float(parts[0])); y.append(float(parts[1]))
                except: pass
    return np.array(x), np.array(y)

def remove_spikes(x, y, threshold=3.0):
    if len(x) < 5:
        return x, y
    keep = np.ones(len(x), dtype=bool)
    for i in range(1, len(x) - 1):
        if y[i-1] > 0 and y[i+1] > 0 and y[i] > 0:
            geo = np.sqrt(y[i-1] * y[i+1])
            ratio = y[i] / geo
            if ratio > threshold or ratio < 1.0/threshold:
                keep[i] = False
    return x[keep], y[keep]

def smooth_savgol(y, window=7):
    n = len(y)
    if n < 7: return y.copy()
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 5: return y.copy()
    order = min(3, win - 1)
    if np.all(y > 0) and np.max(y) / np.min(y) > 5:
        return 10**savgol_filter(np.log10(y), win, polyorder=order)
    else:
        return savgol_filter(y, win, polyorder=order)

def refine_file(fname, raw_dir, ref_dir):
    x, y = load_raw(os.path.join(raw_dir, fname))
    if len(x) == 0:
        return 0, 0, 'empty'

    order = np.argsort(x); x, y = x[order], y[order]
    mask = np.concatenate([[True], np.diff(x) > 0]); x, y = x[mask], y[mask]

    # Snap experimental x-values to exact known positions
    if fname in SNAP_X:
        x, y = snap_x_values(x, y, SNAP_X[fname])

    if len(x) < 3:
        np.savetxt(os.path.join(ref_dir, fname),
                   np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
        return len(x), 0, 'copied'

    if fname in PASSTHROUGH:
        # Pre-processed data: sort+dedup only
        np.savetxt(os.path.join(ref_dir, fname),
                   np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
        return len(x), 0, 'passthrough'

    n_before = len(x)

    if fname in TRIM_LOW_X:
        m = x >= TRIM_LOW_X[fname]; x, y = x[m], y[m]

    x, y = remove_spikes(x, y)
    n_removed = n_before - len(x)

    if len(x) >= 5:
        y = smooth_savgol(y, window=WIDER_SMOOTH.get(fname, 7))

    if 'n' in fname or 'alpha' in fname:
        y = np.maximum(y, 0)

    np.savetxt(os.path.join(ref_dir, fname),
               np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
    return len(x), n_removed, 'refined'

if __name__ == '__main__':
    os.makedirs(REF_DIR, exist_ok=True)
    exclude = {'nodeposition', 'nosurface'}
    print(f"{'File':<45} {'Points':>7} {'Removed':>8} {'Status':>10}")
    print("-" * 75)
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith('.csv') or not fname.startswith('kokkoris_'): continue
        if any(ex in fname for ex in exclude): continue
        npts, nrem, status = refine_file(fname, RAW_DIR, REF_DIR)
        flag = " *" if fname in TRIM_LOW_X or fname in WIDER_SMOOTH else (" P" if fname in PASSTHROUGH else (" S" if fname in SNAP_X else ""))
        print(f"  {fname:<43} {npts:>7} {nrem:>8} {status:>10}{flag}")
    print(f"\n* = wider smoothing or low-pressure trim applied")
    print(f"P = pre-processed passthrough (sort+dedup only)")
    print(f"S = experimental x-values snapped to exact positions")
