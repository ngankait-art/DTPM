#!/usr/bin/env python3
"""
Run power and pressure sweeps and save results to CSV.

Usage: python run_sweeps.py
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

from sf6_model import (solve_steady_state, N_SPECIES, N_NEUTRALS,
                        SPECIES_NAMES, kB, T_GAS, SF6m, Fm, EL)

POWERS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500,
          600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500,
          2750, 3000, 3250, 3500]

PRESSURES = [0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10,
             1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50,
             3.75, 4.00, 4.25, 4.50]

FIELDS = ['Power_W', 'pOFF_Pa', 'Te_eV', 'ne_m3', 'alpha', 'dp_Pa',
          'nSF6_m3', 'nSF5_m3', 'nSF4_m3', 'nSF3_m3', 'nF2_m3', 'nF_m3',
          'nSF5p_m3', 'nSF4p_m3', 'nSF3p_m3', 'nF2p_m3', 'nSF6m_m3', 'nFm_m3']

def run_sweep(sweep_type, fixed, values):
    results = []
    for val in values:
        p, P = (fixed, val) if sweep_type == 'power' else (val, fixed)
        try:
            r = solve_steady_state(p, P)
            row = {'Power_W': P, 'pOFF_Pa': p, 'Te_eV': r['Te'],
                   'ne_m3': r['ne'], 'alpha': r['alpha'], 'dp_Pa': r['dp']}
            for i in range(min(N_SPECIES - 1, 12)):
                row[FIELDS[6 + i]] = r['n'][i]
            results.append(row)
            print(f"  {sweep_type} {val}: Te={r['Te']:.3f} ne={r['ne']:.2e} α={r['alpha']:.1f}")
        except Exception as e:
            print(f"  {sweep_type} {val}: FAILED - {e}")
    return results

def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)
    print(f"Saved {path} ({len(results)} rows)")

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(out, exist_ok=True)

    print("=" * 50)
    print("Power sweep (pOFF = 0.921 Pa)")
    print("=" * 50)
    r = run_sweep('power', 0.921, POWERS)
    save_csv(r, os.path.join(out, 'model_power_sweep.csv'))

    print("\n" + "=" * 50)
    print("Pressure sweep (P = 2000 W)")
    print("=" * 50)
    r = run_sweep('pressure', 2000, PRESSURES)
    save_csv(r, os.path.join(out, 'model_pressure_sweep.csv'))
