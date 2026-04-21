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


def _worker(task):
    """Top-level worker for multiprocessing: runs one (label, frac_Ar, bias) point."""
    label, frac_Ar, bias_on, case_dir, config_path = task
    os.makedirs(case_dir, exist_ok=True)
    _, s = run_one(label, frac_Ar, bias_on, case_dir, config_path)
    return label, frac_Ar, bias_on, s


def main():
    import time
    from multiprocessing import Pool
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    base = os.path.join(PROJECT_ROOT, 'results', 'mettler_composition')

    cases = [
        ('90pct_SF6', 0.1, 1.60),
        ('30pct_SF6', 0.7, 2.15),
    ]
    targets = {c[0]: c[2] for c in cases}
    # Build 4 parallel tasks: 2 compositions x 2 bias states
    tasks = []
    for label, frac_Ar, _ in cases:
        case_dir = os.path.join(base, label)
        tasks.append((f"{label}_off", frac_Ar, False,
                      os.path.join(case_dir, 'bias_off'), config_path))
        tasks.append((f"{label}_on",  frac_Ar, True,
                      os.path.join(case_dir, 'bias_on'),  config_path))

    n_workers = 1 if os.environ.get('COMPOSITION_SERIAL') == '1' else min(4, len(tasks))
    print(f"# Mettler composition pair: {len(tasks)} runs, {n_workers} workers")
    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]

    # Assemble summary indexed by composition label
    by_comp = {}
    for label_bias, frac_Ar, bias_on, s in results:
        base_label = label_bias.rsplit('_', 1)[0]
        if base_label not in by_comp:
            by_comp[base_label] = {'frac_Ar': frac_Ar}
        key = 'nF_centre_on_cm3' if bias_on else 'nF_centre_off_cm3'
        by_comp[base_label][key] = s['nF_centre_wafer_cm3']

    summary = []
    for label, d in by_comp.items():
        enh = d['nF_centre_on_cm3'] / d['nF_centre_off_cm3']
        summary.append({
            'label': label,
            'frac_Ar': d['frac_Ar'],
            'nF_centre_off_cm3': d['nF_centre_off_cm3'],
            'nF_centre_on_cm3': d['nF_centre_on_cm3'],
            'enhancement': enh,
            'mettler_target': targets[label],
            'dev_pct': (enh / targets[label] - 1) * 100,
        })
    with open(os.path.join(base, 'composition_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"# Wall-clock: {time.time() - t0:.1f}s")

    print(f"\n{'#' * 70}\n# Composition pair complete\n{'#' * 70}")
    for row in summary:
        print(f"  {row['label']}: enhancement = x{row['enhancement']:.3f} "
              f"(Mettler x{row['mettler_target']:.2f}, dev {row['dev_pct']:+.1f}%)")


if __name__ == '__main__':
    main()
