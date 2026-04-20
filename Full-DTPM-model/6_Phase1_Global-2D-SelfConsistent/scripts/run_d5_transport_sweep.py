#!/usr/bin/env python3
"""D5: neutral-transport sensitivity sweep at Mettler 90% SF6 bias-on.

D1 (bias toggle), D3 (R_coil sweep), and D4 (tier-2 PINN rates) all ruled
out the three primary electron-physics knobs as the dominant source of the
~-56% absolute-magnitude residual against Mettler Fig 4.17. D5 tests the
remaining candidate: neutral-transport coefficients and wall-surface kinetics.

The sweep varies two dimensionless scale factors applied on top of the
default transport parameters:

  * s_F_Al_scale — scales the aluminium-surface F-sticking probability
    `SURF_AL['s_F']` (default 0.015). Lower values mean less F loss at
    the chamber wall, which should raise bulk [F]_c.
  * d_scale — scales the neutral-species diffusion coefficients D_s
    returned by `compute_diffusion_coefficients`. Lower D_s increases the
    neutral residence time and concentrates F in the source region.

Gamma_Al is held fixed per the 2026-04-19 supervisor decision.

Sweep grid: 3 x 3 over s_F_Al_scale in {0.25, 1.0, 4.0} and d_scale in
{0.5, 1.0, 2.0}, 9 points total (including the unit-scale control that
reproduces the D3/D4 baseline).

Output: results/d5_transport_sweep/{s_F<scale>_d<scale>}/summary.json
        results/d5_transport_sweep/d5_summary.json

Parallelism: multiprocessing.Pool(8) on an 8-core machine — wall-clock
about 2 minutes vs 12 minutes serial.
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

METTLER_CENTRE_90PCT_BIAS_ON_CM3 = 3.774e14  # m^-3 -> cm^-3 conversion done here
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'results', 'd5_transport_sweep')

# Sweep grid
S_F_AL_SCALES = [0.25, 1.0, 4.0]
D_SCALES = [0.5, 1.0, 2.0]


def _apply_transport_overrides(s_F_Al_scale, d_scale):
    """Patch the neutral-transport module constants for this worker process.

    Each multiprocessing worker has its own interpreter state, so these
    patches are fully isolated between sweep points.
    """
    from dtpm.chemistry import wall_chemistry as wc
    from dtpm.chemistry import sf6_chemistry as sc

    # 1. Scale aluminium surface F sticking probability
    wc.SURF_AL = dict(wc.SURF_AL)  # fresh copy, don't mutate the shared module
    wc.SURF_AL['s_F'] = wc.SURF_AL['s_F'] * s_F_Al_scale

    # 2. Wrap compute_diffusion_coefficients to apply a global scale factor
    _original_compute_D = sc.compute_diffusion_coefficients

    def _scaled_compute_D(Tgas, p_mTorr):
        D = _original_compute_D(Tgas, p_mTorr)
        return {k: v * d_scale for k, v in D.items()}

    sc.compute_diffusion_coefficients = _scaled_compute_D


def run_one(task):
    s_F_Al_scale, d_scale = task
    label = f"s_F{s_F_Al_scale:.2f}_d{d_scale:.2f}"
    out_dir = os.path.join(OUTPUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)
    # Apply patches BEFORE importing the pipeline (so first-import picks them up)
    _apply_transport_overrides(s_F_Al_scale, d_scale)
    from run_parameter_sweeps import run_simulation, save_sweep_point
    t0 = time.time()
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': 0.8,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    elapsed = time.time() - t0
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
    s['s_F_Al_scale'] = float(s_F_Al_scale)
    s['d_scale'] = float(d_scale)
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['mettler_target_cm3'] = METTLER_CENTRE_90PCT_BIAS_ON_CM3
    s['residual_pct'] = 100.0 * (s['nF_centre_wafer_cm3'] - METTLER_CENTRE_90PCT_BIAS_ON_CM3) / METTLER_CENTRE_90PCT_BIAS_ON_CM3
    s['elapsed_sec'] = elapsed
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    return {
        's_F_Al_scale': s_F_Al_scale,
        'd_scale': d_scale,
        'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
        'F_drop_pct': float(s['F_drop_pct']),
        'eta': s['eta_computed'],
        'I_peak_A': s['I_peak_final'],
        'V_peak_V': s['V_peak_final'],
        'residual_pct': s['residual_pct'],
        'elapsed_sec': elapsed,
    }


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    tasks = [(s, d) for s in S_F_AL_SCALES for d in D_SCALES]
    n_workers = 1 if os.environ.get('D5_SERIAL') == '1' else min(8, len(tasks))
    print(f"# D5 transport sweep: {len(tasks)} points, {n_workers} workers")
    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_one, tasks)
    else:
        results = [run_one(t) for t in tasks]
    results.sort(key=lambda r: (r['s_F_Al_scale'], r['d_scale']))
    summary = {
        'mettler_target_cm3': METTLER_CENTRE_90PCT_BIAS_ON_CM3,
        'operating_point': '1000W / 10mTorr / 90% SF6 / 200W bias / R_coil=0.8',
        'sweep_grid': {
            's_F_Al_scale': S_F_AL_SCALES,
            'd_scale': D_SCALES,
        },
        'points': results,
        'wall_clock_sec': time.time() - t0,
    }
    with open(os.path.join(OUTPUT_BASE, 'd5_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n# D5 transport sensitivity sweep complete — {len(results)} points")
    print(f"# Wall-clock: {summary['wall_clock_sec']:.1f}s  "
          f"Mettler target = {METTLER_CENTRE_90PCT_BIAS_ON_CM3:.3e} cm^-3")
    print()
    header = f"  {'s_F_scale':>10s} {'d_scale':>8s} {'[F]_c (cm^-3)':>15s} {'residual':>10s} {'F-drop%':>8s} {'eta':>6s} {'I_pk(A)':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['s_F_Al_scale']:>10.2f} {r['d_scale']:>8.2f} "
              f"{r['nF_centre_wafer_cm3']:>15.3e} "
              f"{r['residual_pct']:>9.1f}% "
              f"{r['F_drop_pct']:>8.2f} "
              f"{r['eta']:>6.3f} {r['I_peak_A']:>8.2f}")


if __name__ == '__main__':
    main()
