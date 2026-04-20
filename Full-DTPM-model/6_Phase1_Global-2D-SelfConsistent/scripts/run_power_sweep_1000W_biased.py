#!/usr/bin/env python3
"""Run the full power sweep at Mettler conditions (10 mTorr, 70% SF6, 200W bias).

Produces 11 sweep points covering 200 - 1200 W in 100 W increments.
Each point runs the self-consistent circuit + m12 bias sheath + 9-species chemistry.

Output directory: results/sweeps/power_1000W_biased/P####W/
"""
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point  # noqa: E402


POWERS_W = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    base_out = os.path.join(PROJECT_ROOT, 'results', 'sweeps', 'power_1000W_biased')
    os.makedirs(base_out, exist_ok=True)

    summary_index = []
    for p in POWERS_W:
        label = f"P{p:04d}W"
        out_dir = os.path.join(base_out, label)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'#' * 70}\n# {label}: {p} W, 10 mTorr, 70% SF6, 200 W bias\n{'#' * 70}")
        overrides = {
            'circuit.source_power': p,
            'operating.pressure_mTorr': 10,
            'operating.frac_Ar': 0.3,   # 70% SF6
            'bias.enabled': True,
            'bias.P_bias_W': 200,
            'bias.lambda_exp': 3.20,
        }
        state, mesh, inside, tel, config, r0D = run_simulation(
            overrides, config_path)
        s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
        # Augment with circuit + bias diagnostics
        s['eta_computed'] = float(state.get('eta_computed', 0.0))
        s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
        s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
        s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
        s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
        s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
        s['bias_V_dc'] = float(state.get('bias_V_dc', 0.0))
        s['bias_enabled'] = bool(state.get('bias_enabled', False))
        s['bias_lambda_exp'] = float(state.get('bias_lambda_exp', 0.0))
        s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
        # save augmented summary
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(s, f, indent=2)
        summary_index.append({
            'label': label,
            'power': p,
            **{k: s[k] for k in (
                'eta_computed', 'I_peak_final', 'R_plasma_final',
                'F_drop_pct', 'nF_centre_wafer_cm3',
                'ne_avg_icp', 'bias_V_dc',
            )}
        })
        print(f"  [{label}] eta={s['eta_computed']:.3f}, "
              f"I_peak={s['I_peak_final']:.2f} A, "
              f"R_p={s['R_plasma_final']:.2f} Ohm, "
              f"F_drop={s['F_drop_pct']:.1f} %, "
              f"[F]_centre={s['nF_centre_wafer_cm3']:.2e} cm^-3, "
              f"V_dc={s['bias_V_dc']:.0f} V")

    # index file
    with open(os.path.join(base_out, 'index.json'), 'w') as f:
        json.dump(summary_index, f, indent=2)

    print(f"\n{'#' * 70}")
    print(f"# Sweep complete: {len(POWERS_W)} points saved to {base_out}")
    print(f"{'#' * 70}")


if __name__ == '__main__':
    main()
