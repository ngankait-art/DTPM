#!/usr/bin/env python3
"""Power sweeps at Mettler's 90% and 30% SF6 compositions.

The legacy power_1000W_biased sweep is at 70% SF6 (config default), which
makes it apples-to-oranges against Mettler's Fig. 4.17 anchors at 90% or
30% SF6. This script runs composition-matched power sweeps so Fig. 7.21(d)
and any other absolute-[F] comparison plot can be drawn cleanly.

Conditions per run: 10 mTorr, 200 W bias, lambda_exp=3.20, gamma_Al=0.18.
Power grid: 200--1200 W in 100 W steps (same as the 70% sweep).
Output: results/sweeps/power_{90pct|30pct}_SF6_biased/P####W/{summary.json, nF.npy}
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

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
RESULTS_BASE = os.path.join(PROJECT_ROOT, 'results', 'sweeps')

POWERS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

# (label_dir, frac_Ar for 90% = 0.1, for 30% = 0.7)
COMPOSITIONS = [
    ('power_90pct_SF6_biased', 0.1),
    ('power_30pct_SF6_biased', 0.7),
]


def run_one(task):
    comp_dir, frac_Ar, power = task
    out_dir = os.path.join(RESULTS_BASE, comp_dir, f'P{power:04d}W')
    os.makedirs(out_dir, exist_ok=True)
    from run_parameter_sweeps import run_simulation, save_sweep_point
    overrides = {
        'circuit.source_power': power,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': frac_Ar,
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
    }
    t0 = time.time()
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    elapsed = time.time() - t0
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir,
                         f'{comp_dir}_P{power}')
    s['power'] = int(power)
    s['frac_Ar'] = float(frac_Ar)
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['elapsed_sec'] = elapsed
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return {
        'composition': comp_dir,
        'power': power,
        'frac_Ar': frac_Ar,
        'F_drop_pct': float(s['F_drop_pct']),
        'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
        'elapsed_sec': elapsed,
    }


def main():
    tasks = [(comp_dir, frac_Ar, p)
             for comp_dir, frac_Ar in COMPOSITIONS
             for p in POWERS]
    n_workers = 1 if os.environ.get('POWER_SWEEP_SERIAL') == '1' else min(8, len(tasks))
    print(f"# Running {len(tasks)} sims ({len(COMPOSITIONS)} compositions x "
          f"{len(POWERS)} powers), {n_workers} workers")
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_one, tasks)

    # Write an aggregate index per composition
    for comp_dir, frac_Ar in COMPOSITIONS:
        rows = sorted([r for r in results if r['composition'] == comp_dir],
                      key=lambda r: r['power'])
        out_idx = os.path.join(RESULTS_BASE, comp_dir, 'index.json')
        with open(out_idx, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {out_idx}")

    print(f"\n# Both sweeps complete, wall-clock {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
