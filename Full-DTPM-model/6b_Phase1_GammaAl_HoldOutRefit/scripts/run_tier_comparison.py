#!/usr/bin/env python3
"""D4: tier-1 Arrhenius vs tier-2 Boltzmann-PINN side-by-side at Mettler.

Runs two Picard simulations in parallel processes at the Mettler Fig 4.17
90% SF6 bias-on operating point (1000 W / 10 mTorr / 200 W bias / R_coil 0.8 Ω):

  - tier-1: use_boltzmann_rates = False  (Maxwellian Arrhenius, baseline)
  - tier-2: use_boltzmann_rates = True   (PINN-derived Boltzmann rates)

Compares wafer-centre [F], centre-to-edge drop, Te, η, and residuals vs the
Mettler Fig 4.17 reference value (3.774 × 10^20 m^-3 at r=0, bias-on).

Output: results/tier_comparison/{tier1, tier2}/summary.json + a top-level
comparison.json with the delta analysis.

Parallelism: multiprocessing.Pool(2) → two CPU cores active, ~75 s wall.
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


# Mettler Fig 4.17 90% SF6 bias-on centre [F], from the digitised CSV
METTLER_90PCT_BIAS_ON_CM3 = 3.774e14


def run_one_tier(args):
    """Worker: run one Picard simulation with the specified tier flag."""
    tier_name, use_boltzmann = args
    # Import inside the worker so each process has its own state
    from run_parameter_sweeps import run_simulation, save_sweep_point
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'tier_comparison', tier_name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    overrides = {
        'circuit.source_power': 1000,
        'circuit.R_coil': 0.8,
        'operating.pressure_mTorr': 10,
        'operating.frac_Ar': 0.1,  # 90% SF6
        'bias.enabled': True,
        'bias.P_bias_W': 200,
        'bias.lambda_exp': 3.20,
        'chemistry.use_boltzmann_rates': use_boltzmann,
    }
    state, mesh, inside, tel, config, r0D = run_simulation(overrides, config_path)
    elapsed = time.time() - t0
    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, tier_name)
    s['tier'] = tier_name
    s['use_boltzmann_rates'] = bool(state.get('use_boltzmann_rates', False))
    s['eta_computed'] = float(state.get('eta_computed', 0.0))
    s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
    s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
    s['V_rms_final'] = float(state.get('V_rms_final', 0.0))
    s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
    s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
    s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
    s['mettler_target_cm3'] = METTLER_90PCT_BIAS_ON_CM3
    s['residual_pct'] = 100.0 * (s['nF_centre_wafer_cm3'] - METTLER_90PCT_BIAS_ON_CM3) / METTLER_90PCT_BIAS_ON_CM3
    s['elapsed_sec'] = elapsed
    # Include the tier-2 cache snapshot if present
    t2c = state.get('tier2_boltzmann_cache')
    if t2c is not None:
        s['tier2_cache'] = t2c
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(s, f, indent=2)
    print(f"[{tier_name}] done in {elapsed:.1f}s — "
          f"[F]_c = {s['nF_centre_wafer_cm3']:.3e} cm^-3, "
          f"F-drop = {s['F_drop_pct']:.1f}%, residual = {s['residual_pct']:.1f}%")
    return s


def main():
    base = os.path.join(PROJECT_ROOT, 'results', 'tier_comparison')
    os.makedirs(base, exist_ok=True)
    t_start = time.time()
    with Pool(processes=2) as pool:
        # Parallel: both tiers run concurrently on separate cores.
        results = pool.map(run_one_tier, [('tier1', False), ('tier2', True)])
    t_total = time.time() - t_start

    t1 = next(r for r in results if r['tier'] == 'tier1')
    t2 = next(r for r in results if r['tier'] == 'tier2')
    comparison = {
        'mettler_target_cm3': METTLER_90PCT_BIAS_ON_CM3,
        'tier1': {
            'nF_centre_wafer_cm3': t1['nF_centre_wafer_cm3'],
            'F_drop_pct': t1['F_drop_pct'],
            'eta': t1['eta_computed'],
            'I_peak': t1['I_peak_final'],
            'V_peak': t1['V_peak_final'],
            'residual_pct': t1['residual_pct'],
            'elapsed_sec': t1['elapsed_sec'],
        },
        'tier2': {
            'nF_centre_wafer_cm3': t2['nF_centre_wafer_cm3'],
            'F_drop_pct': t2['F_drop_pct'],
            'eta': t2['eta_computed'],
            'I_peak': t2['I_peak_final'],
            'V_peak': t2['V_peak_final'],
            'residual_pct': t2['residual_pct'],
            'elapsed_sec': t2['elapsed_sec'],
            'tier2_cache': t2.get('tier2_cache'),
        },
        'delta': {
            'nF_ratio_t2_over_t1': (t2['nF_centre_wafer_cm3'] / t1['nF_centre_wafer_cm3']
                                    if t1['nF_centre_wafer_cm3'] > 0 else None),
            'residual_improvement_pp': t1['residual_pct'] - t2['residual_pct'],
            'F_drop_change_pp': t2['F_drop_pct'] - t1['F_drop_pct'],
        },
        'wall_clock_parallel_sec': t_total,
    }
    with open(os.path.join(base, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Headline
    print(f"\n{'=' * 70}")
    print(f"# D4 tier-1 vs tier-2 comparison at Mettler 90% SF6 bias-on")
    print(f"{'=' * 70}")
    print(f"Mettler target [F]_c = {METTLER_90PCT_BIAS_ON_CM3:.3e} cm^-3 (Fig 4.17)")
    print()
    print(f"  {'':20s} {'tier-1 Arrh':>14s} {'tier-2 PINN':>14s} {'Δ':>10s}")
    print(f"  {'[F]_c (cm^-3)':20s} {t1['nF_centre_wafer_cm3']:>14.3e} "
          f"{t2['nF_centre_wafer_cm3']:>14.3e} "
          f"{comparison['delta']['nF_ratio_t2_over_t1']:>10.3f}×")
    print(f"  {'residual vs Mettler':20s} {t1['residual_pct']:>13.1f}% "
          f"{t2['residual_pct']:>13.1f}% "
          f"{comparison['delta']['residual_improvement_pp']:>9.1f} pp")
    print(f"  {'F-drop (%)':20s} {t1['F_drop_pct']:>14.2f} {t2['F_drop_pct']:>14.2f} "
          f"{comparison['delta']['F_drop_change_pp']:>9.2f} pp")
    print(f"  {'η':20s} {t1['eta_computed']:>14.3f} {t2['eta_computed']:>14.3f}")
    print(f"  {'V_peak (V)':20s} {t1['V_peak_final']:>14.2f} {t2['V_peak_final']:>14.2f}")
    if t2.get('tier2_cache'):
        c = t2['tier2_cache']
        print()
        print(f"  tier-2 PINN cache: E/N = {c['E_over_N_Td']:.2f} Td, "
              f"Te_eff = {c['Te_eff']:.2f} eV, k_iz = {c['k_iz']:.2e}, "
              f"k_att = {c['k_att']:.2e}, k_diss = {c['k_diss']:.2e}")
    print()
    print(f"  wall-clock (2-process parallel): {t_total:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
