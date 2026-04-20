#!/usr/bin/env python3
"""Run the Mettler Fig 4.17 composition pair at 1000 W / 10 mTorr / 200 W bias.

Two benchmarks:
  (a) 90% SF6 / 10% Ar   (frac_Ar = 0.1)  — calibration reference
  (b) 30% SF6 / 70% Ar   (frac_Ar = 0.7)  — composition blind test

For each composition we also run a BIAS-OFF control to measure the
enhancement factor that the model predicts.  Mettler's enhancement
factors at these conditions are x1.6 (90%) and x2.15 (30% centre).

Output: results/mettler_composition/{90pct_SF6, 30pct_SF6}/
                                      {bias_off, bias_on}/
"""
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point  # noqa: E402


def run_one(label, frac_Ar, bias_on, out_dir, config_path):
    print(f"\n{'#' * 70}\n# {label}  (frac_Ar={frac_Ar}, bias={'ON' if bias_on else 'OFF'})"
          f"\n{'#' * 70}")
    overrides = {
        'circuit.source_power': 1000,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': frac_Ar,
        'bias.enabled': bias_on,
        'bias.P_bias_W': 200 if bias_on else 0,
        'bias.lambda_exp': 3.20,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(
        overrides, config_path)
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
    s['bias_V_dc'] = float(state.get('bias_V_dc', 0.0))
    s['bias_enabled'] = bool(state.get('bias_enabled', False))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return state, s


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    base = os.path.join(PROJECT_ROOT, 'results', 'mettler_composition')

    cases = [
        ('90pct_SF6', 0.1, 1.60),   # (label, frac_Ar, Mettler target enhancement)
        ('30pct_SF6', 0.7, 2.15),
    ]
    summary = []
    for label, frac_Ar, mettler_target in cases:
        case_dir = os.path.join(base, label)
        out_off = os.path.join(case_dir, 'bias_off')
        out_on = os.path.join(case_dir, 'bias_on')
        os.makedirs(out_off, exist_ok=True)
        os.makedirs(out_on, exist_ok=True)
        _, s_off = run_one(f"{label}_off", frac_Ar, False, out_off, config_path)
        _, s_on = run_one(f"{label}_on", frac_Ar, True, out_on, config_path)
        enh = s_on['nF_centre_wafer_cm3'] / s_off['nF_centre_wafer_cm3']
        summary.append({
            'label': label,
            'frac_Ar': frac_Ar,
            'nF_centre_off_cm3': s_off['nF_centre_wafer_cm3'],
            'nF_centre_on_cm3': s_on['nF_centre_wafer_cm3'],
            'enhancement': enh,
            'mettler_target': mettler_target,
            'dev_pct': (enh / mettler_target - 1) * 100,
        })
        print(f"\n[{label}] bias-off [F]c = {s_off['nF_centre_wafer_cm3']:.3e}, "
              f"bias-on = {s_on['nF_centre_wafer_cm3']:.3e}, "
              f"enh = x{enh:.3f} (target x{mettler_target:.2f}, "
              f"dev {(enh/mettler_target-1)*100:+.1f}%)")

    with open(os.path.join(base, 'composition_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'#' * 70}\n# Composition pair complete\n{'#' * 70}")
    for row in summary:
        print(f"  {row['label']}: enhancement = x{row['enhancement']:.3f} "
              f"(Mettler x{row['mettler_target']:.2f}, dev {row['dev_pct']:+.1f}%)")


if __name__ == '__main__':
    main()
