#!/usr/bin/env python3
"""
SF6 Global Plasma Model — Run all sweeps and generate results.

Usage:
    python run_model.py [v1|v3|both]

Default: both versions.
Results are saved to output/ as CSV files.
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sf6_global_model import N_SP, EL, SF6m, Fm, N_NEU, kB, T_gas, NAMES

def run_sweep(solver_module, sweep_type, fixed_val, sweep_vals, label):
    results = []
    for val in sweep_vals:
        p_OFF, P = (fixed_val, val) if sweep_type == 'power' else (val, fixed_val)
        try:
            r = solver_module.run_to_steady_state(p_OFF, P, t_max=1.0, verbose=False)
            n = r['n']; Te = r['Te']; th = r['th']
            ne = n[EL]; alpha = (n[SF6m] + n[Fm]) / max(ne, 1)
            dp = sum(n[:N_NEU]) * kB * T_gas - p_OFF
            row = {
                'Power_W': P, 'pOFF_Pa': p_OFF, 'Te_eV': Te,
                'ne_m3': ne, 'alpha': alpha, 'dp_Pa': dp,
            }
            for i in range(N_SP):
                row[f'n{NAMES[i]}_m3'] = n[i]
            for i, name in enumerate(['thF', 'thSF3', 'thSF4', 'thSF5']):
                row[name] = th[i]
            results.append(row)
            print(f"  {label} {sweep_type} {val}: Te={Te:.3f} ne={ne:.2e} alpha={alpha:.1f}")
        except Exception as e:
            print(f"  {label} {sweep_type} {val}: FAILED - {e}")
    return results

def write_csv(results, path):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {path} ({len(results)} rows)")

# Sweep parameters
POWERS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700,
          800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]
POFFS = [0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.25,
         1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50]

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode in ('v1', 'both'):
        import sf6_solver_v1 as v1
        print("=" * 60)
        print("V1: Lee & Lieberman h-factor, eta=0.50, s_dep=0.03")
        print("=" * 60)
        write_csv(run_sweep(v1, 'power', 0.921, POWERS, 'V1'), 'output/v1_power_sweep.csv')
        write_csv(run_sweep(v1, 'pressure', 2000, POFFS, 'V1'), 'output/v1_pressure_sweep.csv')

    if mode in ('v3', 'both'):
        import sf6_solver_v3 as v3
        print("=" * 60)
        print("V3: Kim h-factor, eta=0.70, s_dep=0.10")
        print("=" * 60)
        write_csv(run_sweep(v3, 'power', 0.921, POWERS, 'V3'), 'output/v3_power_sweep.csv')
        write_csv(run_sweep(v3, 'pressure', 2000, POFFS, 'V3'), 'output/v3_pressure_sweep.csv')

    print("\nDone. Results in output/")
