#!/usr/bin/env python3
"""Run the 90% SF6 / 10% Ar Mettler-Fig-4.17 benchmark case.

Conditions (Mettler Fig 4.14 / 4.17 top panel, 90% SF6, bias off):
  P_ICP = 1000 W
  p     = 10 mTorr
  flow  = 90 sccm SF6 / 10 sccm Ar  =>  frac_Ar = 0.1

Output: results/mettler_fig417/90pct/ containing 9-species + ne/Te/P_rz fields
for overlay against Mettler's digitised data in
../METTLER_TEL_DATA_DIGITISED.md.

Note: no wafer-bias model in Phase-1; this corresponds to Mettler's bias-OFF
branch. The bias-on enhancement (~x1.6 for 90% SF6) is noted in L5.
"""
import os
import sys

# Anchor to the Phase-1 codebase (5.Phase1_EM_Chemistry_Merged)
STEPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE1_ROOT = os.path.join(STEPS_DIR, '5.Phase1_EM_Chemistry_Merged')
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point  # noqa: E402


def main():
    config_path = os.path.join(PHASE1_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'mettler_fig417', '90pct'
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('Running Mettler Fig 4.17 (90% SF6) at 1000 W, 10 mTorr ...')
    overrides = {
        'circuit.source_power': 1000,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,
    }
    state, mesh, inside, tel, config, result_0D = run_simulation(
        overrides, config_path
    )
    summary = save_sweep_point(
        state, mesh, inside, result_0D, config, out_dir, '90pct_SF6'
    )
    summary['frac_Ar'] = 0.1
    summary['mettler_reference'] = 'Fig 4.17 top panel, 90% SF6, bias OFF'

    import json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [F] centre-to-edge drop = {summary['F_drop_pct']:.1f}%"
          f"  (Mettler 90% SF6: ~75%)")
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
