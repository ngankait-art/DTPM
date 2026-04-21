#!/usr/bin/env python3
"""Run the SF6/Ar 50/50 mixture case (700 W, 10 mTorr) and save for figure 6.20.

Produces one sweep-point directory at results/sweeps/ar_mixture/Ar50/ containing
the usual 9-species + ne/Te/P_rz fields plus an Ar* metastable scalar in the
summary (the 2D Ar* field is later reconstructed from the ne profile since Ar*
is source-limited and shares ne's spatial shape).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from run_parameter_sweeps import run_simulation, save_sweep_point


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'sweeps', 'ar_mixture', 'Ar50')
    os.makedirs(out_dir, exist_ok=True)

    print('Running SF6/Ar 50/50 mixture at 700 W, 10 mTorr ...')
    overrides = {
        'circuit.source_power': 700,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.5,
    }
    state, mesh, inside, tel, config, result_0D = run_simulation(overrides, config_path)
    summary = save_sweep_point(state, mesh, inside, result_0D, config, out_dir, 'Ar50')

    # Augment summary with Ar* metastable density from 0D model (cm^-3)
    summary['n_Ar_star'] = float(result_0D.get('nArm', 0)) * 1e-6
    summary['frac_Ar'] = 0.5

    import json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [F] drop = {summary['F_drop_pct']:.1f}%")
    print(f"  [Ar*] = {summary['n_Ar_star']:.2e} cm^-3")
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    main()
