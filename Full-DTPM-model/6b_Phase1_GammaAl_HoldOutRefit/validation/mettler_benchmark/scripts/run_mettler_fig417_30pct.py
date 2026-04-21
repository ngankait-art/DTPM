#!/usr/bin/env python3
"""Run the 30% SF6 / 70% Ar Mettler-Fig-4.17 benchmark case.

Conditions (Mettler Fig 4.17 top panel, 30% SF6, bias off):
  P_ICP = 1000 W
  p     = 10 mTorr
  flow  = 30 sccm SF6 / 70 sccm Ar  =>  frac_Ar = 0.7

Tests composition-sensitivity of the fixed gamma_Al = 0.18 calibration
(which was tuned to the 90% SF6 branch). Mettler Fig 4.17 measures 67%
centre-to-edge drop for this branch vs 75% for the 90% SF6 branch.
"""
import os
import sys

STEPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE1_ROOT = os.path.join(STEPS_DIR, '5.Phase1_EM_Chemistry_Merged')
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point  # noqa: E402


def main():
    config_path = os.path.join(PHASE1_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'mettler_fig417', '30pct'
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('Running Mettler Fig 4.17 (30% SF6) at 1000 W, 10 mTorr ...')
    overrides = {
        'circuit.source_power': 1000,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.7,
    }
    state, mesh, inside, tel, config, result_0D = run_simulation(
        overrides, config_path
    )
    summary = save_sweep_point(
        state, mesh, inside, result_0D, config, out_dir, '30pct_SF6'
    )
    summary['frac_Ar'] = 0.7
    summary['mettler_reference'] = 'Fig 4.17 top panel, 30% SF6, bias OFF'

    import json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [F] centre-to-edge drop = {summary['F_drop_pct']:.1f}%"
          f"  (Mettler 30% SF6: ~67%)")
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
