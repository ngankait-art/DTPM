#!/usr/bin/env python3
"""D6 Stage A': gamma_quartz cross-check sweep.

Purpose: confirm gamma_quartz is a negligible lever at the Mettler operating
point so that the D6 refit conclusion (single-knob gamma_Al) is defensible.

Single condition: 90% SF6 / 200 W bias / R_coil=0.8 / gamma_Al=0.18.
gamma_quartz grid: {0.0001, 0.001, 0.005, 0.02} — spans the Kokkoris and
Chantry quartz-surface literature range.

If [F]_c varies by <= 2 % across the grid, gamma_quartz is confirmed
negligible at Mettler conditions. If > 5 %, it flips the conclusion.
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
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'results', 'gamma_quartz_crosscheck')

GAMMA_QZ_VALUES = [0.0001, 0.001, 0.005, 0.02]


def _patch_WALL_GAMMA_quartz(gamma_qz):
    """Monkey-patch WALL_GAMMA['quartz']['F'] in this worker process.

    The multispecies_transport solver reads the Kokkoris WALL_GAMMA dict
    rather than the config, so a config override is not enough — we need
    to mutate the module-level dict. Safe because each multiprocessing
    worker has its own interpreter state.
    """
    from dtpm.chemistry import sf6_chemistry as sc
    sc.WALL_GAMMA['quartz'] = dict(sc.WALL_GAMMA['quartz'])  # fresh copy
    sc.WALL_GAMMA['quartz']['F'] = gamma_qz


def run_one(gamma_qz):
    out_dir = os.path.join(OUTPUT_BASE, f"gamma_qz_{gamma_qz:.4f}")
    os.makedirs(out_dir, exist_ok=True)
    _patch_WALL_GAMMA_quartz(gamma_qz)
    from run_parameter_sweeps import run_simulation, save_sweep_point
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': 0.8,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
        'wall_chemistry.gamma_Al': 0.18,
    }
    t0 = time.time()
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    elapsed = time.time() - t0
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, f"qz_{gamma_qz:.4f}")
    s['gamma_quartz'] = float(gamma_qz)
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['F_drop_pct'] = float(s['F_drop_pct'])
    s['elapsed_sec'] = elapsed
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return {
        'gamma_quartz': float(gamma_qz),
        'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
        'F_drop_pct': s['F_drop_pct'],
        'elapsed_sec': elapsed,
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    n_workers = min(4, len(GAMMA_QZ_VALUES))
    print(f"# D6 Stage A' gamma_quartz cross-check: {len(GAMMA_QZ_VALUES)} runs, {n_workers} workers")
    print("# Condition: 90% SF6 / 200 W bias / R_coil=0.8 / gamma_Al=0.18 (fixed)")
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_one, GAMMA_QZ_VALUES)
    results.sort(key=lambda r: r['gamma_quartz'])

    # Compute spread
    F_min = min(r['nF_centre_wafer_cm3'] for r in results)
    F_max = max(r['nF_centre_wafer_cm3'] for r in results)
    F_spread_pct = 100.0 * (F_max - F_min) / F_min if F_min > 0 else float('nan')

    summary = {
        'condition': '90% SF6 / 200 W bias / R_coil=0.8 / gamma_Al=0.18',
        'gamma_quartz_grid': GAMMA_QZ_VALUES,
        'points': results,
        'F_centre_spread_pct': F_spread_pct,
        'verdict': (
            'gamma_quartz is NEGLIGIBLE at Mettler conditions — single-knob gamma_Al refit is defensible.'
            if F_spread_pct <= 2.0 else
            'gamma_quartz MATTERS — single-knob gamma_Al refit is marginal.'
            if F_spread_pct <= 5.0 else
            'gamma_quartz FLIPS the conclusion — must be included in the refit.'
        ),
        'wall_clock_sec': time.time() - t0,
    }
    with open(os.path.join(OUTPUT_BASE, 'gamma_quartz_crosscheck_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n# Cross-check complete, wall-clock {summary['wall_clock_sec']:.1f}s\n")
    hdr = f"  {'gamma_quartz':>12s} {'[F]_c (cm-3)':>14s} {'F-drop%':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['gamma_quartz']:>12.4f} {r['nF_centre_wafer_cm3']:>14.3e} {r['F_drop_pct']:>7.2f}")
    print(f"\n# [F]_c spread: {F_spread_pct:.2f}%")
    print(f"# Verdict: {summary['verdict']}")


if __name__ == '__main__':
    main()
