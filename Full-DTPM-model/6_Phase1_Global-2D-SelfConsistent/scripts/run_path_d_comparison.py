#!/usr/bin/env python3
"""Path D: scalar alpha baseline vs 2D alpha (renormalised) vs 2D alpha (raw).

Runs three Picard simulations in parallel at Mettler's 1000 W / 10 mTorr / 90%
SF6 / 200 W bias operating point:

  - scalar: use_2d_alpha = False   (alpha = alpha_0D uniform, L1 baseline)
  - renorm: use_2d_alpha = "renorm" (2D shape, volume-avg rescaled to alpha_0D)
  - raw:    use_2d_alpha = "raw"    (2D local shape, no renorm — diagnostic only)

Compares wafer-centre [F], F-drop, eta against the Mettler 3.774e20 m^-3
target. The raw path is included to show why the renormalisation is
necessary — raw local alpha over-estimates electronegativity.

Parallelism: multiprocessing.Pool(3) — ~75 s wall-clock on an 8-core box.
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

METTLER_90PCT_BIAS_ON_CM3 = 3.774e14
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')


def run_one(task):
    label, use_2d_alpha = task
    from run_parameter_sweeps import run_simulation, save_sweep_point
    import numpy as np
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'path_d_comparison', label)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': 0.8,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
        'chemistry.use_2d_alpha': use_2d_alpha,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, CONFIG_PATH)
    elapsed = time.time() - t0
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, label)
    s['mode'] = label
    s['use_2d_alpha'] = str(use_2d_alpha)
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['mettler_target_cm3'] = METTLER_90PCT_BIAS_ON_CM3
    s['residual_pct'] = 100.0 * (s['nF_centre_wafer_cm3'] - METTLER_90PCT_BIAS_ON_CM3) / METTLER_90PCT_BIAS_ON_CM3
    s['elapsed_sec'] = elapsed
    # Also stash alpha stats for interpretation
    ions = state.get('ions', {})
    if 'alpha' in ions:
        import numpy as np
        a = ions['alpha'][inside]
        s['alpha_local_min']  = float(a.min())
        s['alpha_local_mean'] = float(a.mean())
        s['alpha_local_max']  = float(a.max())
    s['alpha_0D'] = float(r0D.get('alpha', 0.0))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    print(f"[{label:7s}] {elapsed:4.0f}s  [F]_c = {s['nF_centre_wafer_cm3']:.3e}  "
          f"F-drop = {s['F_drop_pct']:.2f}%  residual = {s['residual_pct']:.1f}%  "
          f"eta = {s['eta_computed']:.3f}")
    return s


def main():
    out_base = os.path.join(PROJECT_ROOT, 'results', 'path_d_comparison')
    os.makedirs(out_base, exist_ok=True)
    tasks = [
        ('scalar', False),
        ('renorm', 'renorm'),
        ('raw',    'raw'),
    ]
    t0 = time.time()
    with Pool(processes=min(3, len(tasks))) as pool:
        results = pool.map(run_one, tasks)
    elapsed = time.time() - t0
    by_mode = {r['mode']: r for r in results}
    comparison = {
        'mettler_target_cm3': METTLER_90PCT_BIAS_ON_CM3,
        'operating_point': '1000W / 10mTorr / 90% SF6 / 200W bias / R_coil=0.8',
        'modes': {m: {k: by_mode[m].get(k) for k in (
            'nF_centre_wafer_cm3', 'F_drop_pct', 'eta_computed',
            'V_peak_final', 'P_abs_final', 'residual_pct',
            'alpha_local_min', 'alpha_local_mean', 'alpha_local_max',
            'alpha_0D', 'elapsed_sec',
        )} for m in by_mode},
        'wall_clock_parallel_sec': elapsed,
    }
    with open(os.path.join(out_base, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Headline
    print(f"\n{'=' * 74}")
    print(f"# Path D comparison at Mettler 1000W / 10mTorr / 90% SF6 / 200W bias")
    print(f"{'=' * 74}")
    print(f"Mettler target [F]_c = {METTLER_90PCT_BIAS_ON_CM3:.3e} cm^-3, alpha_0D = {by_mode['scalar']['alpha_0D']:.3e}")
    print()
    header = f"  {'mode':>7s}  {'[F]_c (cm^-3)':>15s}  {'F-drop':>8s}  {'residual':>10s}  {'eta':>6s}  {'alpha raw mean':>15s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in ('scalar', 'renorm', 'raw'):
        r = by_mode[m]
        a_mean = r.get('alpha_local_mean', 0.0)
        print(f"  {m:>7s}  {r['nF_centre_wafer_cm3']:>15.3e}  "
              f"{r['F_drop_pct']:>8.2f}  {r['residual_pct']:>9.1f}%  "
              f"{r['eta_computed']:>6.3f}  {a_mean:>15.3e}")
    print(f"\n  Wall-clock (3-process parallel): {elapsed:.1f}s")


if __name__ == '__main__':
    main()
