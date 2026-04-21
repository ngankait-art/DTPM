#!/usr/bin/env python3
"""R_coil discovery sweep at Mettler's 1000 W / 10 mTorr / 90% SF6 / bias-on.

Supervisor 2026-04-19: exact R_coil is unknown; sweep over candidate values
and document how Vpeak, Ipeak, eta, P_abs, and centre [F] depend on R_coil.

Output: results/rcoil_sweep/R{value}ohm/summary.json
        results/rcoil_sweep/rcoil_summary.json

Parallelism: multiprocessing.Pool(N_WORKERS) — 8 points run concurrently on
an 8-core machine. Set N_WORKERS=1 via env var RCOIL_SERIAL=1 for debugging.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


R_COIL_VALUES = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'results', 'rcoil_sweep')


def run_one(r_coil):
    """Worker: one Picard simulation at a single R_coil value."""
    from run_parameter_sweeps import run_simulation, save_sweep_point
    label = f"R{r_coil:.1f}ohm"
    out_dir = os.path.join(OUTPUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': r_coil,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,         # 90% SF6
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
    s['R_coil'] = r_coil
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['F_drop_pct'] = float(s.get('F_drop_pct', 0.0))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return {
        'R_coil': r_coil,
        'eta': s['eta_computed'],
        'I_peak_A': s['I_peak_final'],
        'V_peak_V': s['V_peak_final'],
        'V_rms_V': s['V_rms_final'],
        'R_plasma_ohm': s['R_plasma_final'],
        'P_abs_W': s['P_abs_final'],
        'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
        'F_drop_pct': s['F_drop_pct'],
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    n_workers = 1 if os.environ.get('RCOIL_SERIAL') == '1' else min(8, len(R_COIL_VALUES))
    print(f"# R_coil sweep: {len(R_COIL_VALUES)} points, {n_workers} workers")
    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            summary = pool.map(run_one, R_COIL_VALUES)
    else:
        summary = [run_one(r) for r in R_COIL_VALUES]
    # Sort by R_coil ascending (Pool.map preserves order but be explicit)
    summary = sorted(summary, key=lambda d: d['R_coil'])
    with open(os.path.join(OUTPUT_BASE, 'rcoil_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"# Wall-clock: {time.time() - t0:.1f}s")
    print(f"\n{'#' * 70}\n# R_coil sweep complete — {len(summary)} points\n{'#' * 70}")
    print(f"  {'R_coil':>8s}  {'eta':>6s}  {'Ipeak(A)':>9s}  {'Vpk(V)':>8s}  "
          f"{'Rplas(Ω)':>9s}  {'Pabs(W)':>8s}  {'[F]c(cm^-3)':>12s}  {'Fdrop(%)':>9s}")
    for row in summary:
        print(f"  {row['R_coil']:>8.2f}  {row['eta']:>6.3f}  {row['I_peak_A']:>9.2f}  "
              f"{row['V_peak_V']:>8.1f}  {row['R_plasma_ohm']:>9.2f}  "
              f"{row['P_abs_W']:>8.1f}  {row['nF_centre_wafer_cm3']:>12.2e}  "
              f"{row['F_drop_pct']:>9.2f}")


if __name__ == '__main__':
    main()
