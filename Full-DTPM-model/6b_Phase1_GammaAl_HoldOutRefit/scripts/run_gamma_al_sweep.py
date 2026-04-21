#!/usr/bin/env python3
"""D6: gamma_Al hold-out refit sweep.

Protocol (per plan, see /Users/muabdelghany/.claude/plans/mutable-swimming-chipmunk.md):

  - Fit point: 90 % SF6 bias-on  (Mettler [F]_c = 3.774e14 cm^-3).
  - Held-out:  90 % SF6 bias-off, 30 % SF6 bias-on, 30 % SF6 bias-off.
  - Grid:      gamma_Al in {0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.25, 0.35}.
  - 9 gamma_Al values x 4 conditions = 36 runs, parallel via Pool(8).
  - All other knobs held fixed (R_coil=0.8, lambda_exp=3.20, D_scale=1, etc.).

Output:
  results/gamma_al_sweep/<condition>/<gamma_al>/summary.json
  results/gamma_al_sweep/gamma_al_sweep_summary.json  (aggregate)

Run serial with `GAMMA_AL_SERIAL=1` env var.
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
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'results', 'gamma_al_sweep')

# Mettler reference values (cm^-3, %) at 1000 W / 10 mTorr
METTLER = {
    '90pct_SF6_bias_on':  {'nF_c_cm3': 3.774e14, 'F_drop_pct': 74.8},
    '90pct_SF6_bias_off': {'nF_c_cm3': 2.327e14, 'F_drop_pct': 74.5},
    '30pct_SF6_bias_on':  {'nF_c_cm3': 1.297e14, 'F_drop_pct': 75.0},
    '30pct_SF6_bias_off': {'nF_c_cm3': 0.616e14, 'F_drop_pct': 67.2},
}

# (label, frac_Ar, bias_enabled, P_bias_W)
CONDITIONS = [
    ('90pct_SF6_bias_on',  0.1, True,  200),
    ('90pct_SF6_bias_off', 0.1, False, 0),
    ('30pct_SF6_bias_on',  0.7, True,  200),
    ('30pct_SF6_bias_off', 0.7, False, 0),
]

GAMMA_AL_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.25, 0.35]


def run_one(task):
    gamma_al, cond = task
    label, frac_Ar, bias_on, P_bias = cond
    tag = f"gamma{gamma_al:.3f}"
    out_dir = os.path.join(OUTPUT_BASE, label, tag)
    os.makedirs(out_dir, exist_ok=True)

    from run_parameter_sweeps import run_simulation, save_sweep_point

    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': 0.8,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': frac_Ar,
        'bias.enabled': bias_on,
        'bias.P_bias_W': P_bias,
        'bias.lambda_exp': 3.20,
        'wall_chemistry.gamma_Al': gamma_al,
    }
    t0 = time.time()
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    elapsed = time.time() - t0

    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, f"{label}_{tag}")
    s['gamma_Al'] = float(gamma_al)
    s['condition'] = label
    s['frac_Ar'] = float(frac_Ar)
    s['bias_enabled'] = bool(bias_on)
    s['P_bias_W'] = int(P_bias)
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['elapsed_sec'] = elapsed

    mettler_c = METTLER[label]['nF_c_cm3']
    mettler_drop = METTLER[label]['F_drop_pct']
    s['mettler_nF_c_cm3'] = mettler_c
    s['mettler_F_drop_pct'] = mettler_drop
    s['residual_F_pct'] = 100.0 * (s['nF_centre_wafer_cm3'] - mettler_c) / mettler_c
    s['residual_drop_pp'] = float(s['F_drop_pct']) - mettler_drop

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return {
        'gamma_Al': float(gamma_al),
        'condition': label,
        'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
        'F_drop_pct': float(s['F_drop_pct']),
        'mettler_nF_c_cm3': mettler_c,
        'mettler_F_drop_pct': mettler_drop,
        'residual_F_pct': s['residual_F_pct'],
        'residual_drop_pp': s['residual_drop_pp'],
        'eta': s['eta_computed'],
        'I_peak_A': s['I_peak_final'],
        'elapsed_sec': elapsed,
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    tasks = [(g, c) for g in GAMMA_AL_VALUES for c in CONDITIONS]
    n_workers = 1 if os.environ.get('GAMMA_AL_SERIAL') == '1' else min(8, len(tasks))
    print(f"# D6 gamma_Al sweep: {len(tasks)} runs ({len(GAMMA_AL_VALUES)} gamma_Al "
          f"x {len(CONDITIONS)} conditions), {n_workers} workers")
    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_one, tasks)
    else:
        results = [run_one(t) for t in tasks]

    results.sort(key=lambda r: (r['condition'], r['gamma_Al']))
    summary = {
        'protocol': 'D6 gamma_Al hold-out refit (Stage A)',
        'fit_point': '90pct_SF6_bias_on',
        'held_out': ['90pct_SF6_bias_off', '30pct_SF6_bias_on', '30pct_SF6_bias_off'],
        'locked_knobs': {
            'R_coil': 0.8, 'lambda_exp': 3.20, 'D_scale': 1.0, 's_F_Al_scale': 1.0,
            'use_boltzmann_rates': False, 'use_2d_alpha': False,
        },
        'gamma_al_grid': GAMMA_AL_VALUES,
        'conditions': [c[0] for c in CONDITIONS],
        'mettler': METTLER,
        'points': results,
        'wall_clock_sec': time.time() - t0,
    }
    with open(os.path.join(OUTPUT_BASE, 'gamma_al_sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n# Sweep complete — {len(results)} points, wall-clock {summary['wall_clock_sec']:.1f}s\n")
    hdr = f"  {'condition':<22s} {'gamma_Al':>9s} {'[F]_c (cm-3)':>14s} {'residF%':>8s} {'F-drop%':>8s} {'drop-pp':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['condition']:<22s} {r['gamma_Al']:>9.3f} "
              f"{r['nF_centre_wafer_cm3']:>14.3e} "
              f"{r['residual_F_pct']:>7.1f}% "
              f"{r['F_drop_pct']:>7.2f} "
              f"{r['residual_drop_pp']:>7.2f}")


if __name__ == '__main__':
    main()
