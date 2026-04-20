#!/usr/bin/env python3
"""R_coil discovery sweep at Mettler's 1000 W / 10 mTorr / 90% SF6 / bias-on.

Supervisor 2026-04-19: exact R_coil is unknown; sweep over candidate values
and document how Vpeak, Ipeak, eta, P_abs, and centre [F] depend on R_coil.

Output: results/rcoil_sweep/R{value}ohm/summary.json
        results/rcoil_sweep/rcoil_summary.json
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point  # noqa: E402

R_COIL_VALUES = [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
METTLER_CENTRE_90PCT_BIAS_ON_CM3 = None  # to be documented in memo from raw Mettler digitisation


def run_one(r_coil, out_dir, config_path):
    label = f"R{r_coil:.1f}ohm"
    print(f"\n{'#' * 70}\n# R_coil = {r_coil} Ω  (Mettler 1000W / 10mTorr / 90% SF6 / bias-on)"
          f"\n{'#' * 70}")
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': r_coil,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,         # 90% SF6
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, config_path)
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
    s['R_coil'] = r_coil
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', 0.0))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['F_drop_pct'] = float(s.get('F_drop_pct', 0.0))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return s


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    base = os.path.join(PROJECT_ROOT, 'results', 'rcoil_sweep')
    os.makedirs(base, exist_ok=True)
    summary = []
    for r_coil in R_COIL_VALUES:
        out_dir = os.path.join(base, f"R{r_coil:.1f}ohm")
        os.makedirs(out_dir, exist_ok=True)
        s = run_one(r_coil, out_dir, config_path)
        summary.append({
            'R_coil': r_coil,
            'eta': s['eta_computed'],
            'I_peak_A': s['I_peak_final'],
            'V_peak_V': s['V_peak_final'],
            'V_rms_V': s['V_rms_final'],
            'R_plasma_ohm': s['R_plasma_final'],
            'P_abs_W': s['P_abs_final'],
            'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
            'F_drop_pct': s['F_drop_pct'],
        })
    with open(os.path.join(base, 'rcoil_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
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
