#!/usr/bin/env python3
"""
Generate CSV data files for all plots.
Run this after sf6_global_model_final.py to export numerical data.
"""

import numpy as np
import csv
import os
try:
    from sf6_global_model_final import solve_model, sweep_with_continuation
except ImportError:
    from sf6_global_model_final import solve_model, sweep_with_continuation

OUTPUT_DIR = 'csv_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_csv(filename, headers, rows):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"  Wrote {path} ({len(rows)} rows)")

# ═══════════════════════════════════════════════════════════════
print("Generating model data for CSV export...")
print("="*60)

base = dict(p_mTorr=10, Q_sccm=40, eta=0.12)
powers = np.linspace(200, 2000, 37)

# Power sweeps at 4 Ar fractions
sweeps = {}
for fAr in [0.0, 0.2, 0.5, 0.8]:
    label = f"{int(fAr*100)}pct_Ar"
    print(f"\n▶ Power sweep: {int(fAr*100)}% Ar")
    sweeps[label] = sweep_with_continuation(
        'P_rf', powers, {**base, 'frac_Ar': fAr}, verbose=False)
    print(f"  Done — {len(sweeps[label])} points")

# Ar fraction sweep at 1500W
print("\n▶ Ar fraction sweep: 0–100% at 1500W")
fracs = np.linspace(0, 1.0, 21)
res_Ar = sweep_with_continuation('frac_Ar', fracs, {**base, 'P_rf': 1500}, verbose=False)
print(f"  Done — {len(res_Ar)} points")

# Alpha vs Ar at 3 pressures
res_alpha = {}
for p in [5, 10, 20]:
    print(f"\n▶ Alpha vs Ar: {p} mTorr, 1500W")
    b = base.copy(); b['p_mTorr'] = p; b['P_rf'] = 1500
    res_alpha[p] = sweep_with_continuation(
        'frac_Ar', np.linspace(0, 0.8, 17), b, verbose=False)
    print(f"  Done — {len(res_alpha[p])} points")

# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Writing CSV files...")
print("="*60)

# --- 1. Power sweep data (one file per Ar fraction) ---
for label, res in sweeps.items():
    headers = [
        'Power_W', 'Te_eV', 'ne_cm3', 'alpha',
        'nF_cm3', 'nF2_cm3', 'nSF6_cm3', 'nSF5_cm3', 'nSF4_cm3',
        'nSF3_cm3', 'nSF2_cm3', 'nSF_cm3', 'nS_cm3',
        'nAr_ground_cm3', 'nAr_metastable_cm3',
        'Ec_eV', 'eps_T_eV', 'SF6_dissoc_pct',
        'R_Penning_m3s', 'converged'
    ]
    rows = []
    for r in res:
        rows.append([
            f"{r['P_rf']:.1f}",
            f"{r['Te']:.4f}",
            f"{r['ne']*1e-6:.6e}",
            f"{r['alpha']:.4f}",
            f"{r['n_F']*1e-6:.6e}",
            f"{r['n_F2']*1e-6:.6e}",
            f"{r['n_SF6']*1e-6:.6e}",
            f"{r['n_SF5']*1e-6:.6e}",
            f"{r['n_SF4']*1e-6:.6e}",
            f"{r['n_SF3']*1e-6:.6e}",
            f"{r['n_SF2']*1e-6:.6e}",
            f"{r['n_SF']*1e-6:.6e}",
            f"{r['n_S']*1e-6:.6e}",
            f"{r['nAr0']*1e-6:.6e}",
            f"{r['nArm']*1e-6:.6e}",
            f"{r['Ec']:.2f}",
            f"{r['eps_T']:.2f}",
            f"{r['dissoc_frac']*100:.2f}",
            f"{r['R_Penning']:.4e}",
            str(r['converged']),
        ])
    write_csv(f"power_sweep_{label}.csv", headers, rows)

# --- 2. Ar fraction sweep at 1500W ---
headers = [
    'Ar_fraction', 'Ar_percent', 'Power_W',
    'Te_eV', 'ne_cm3', 'alpha',
    'nF_cm3', 'nSF6_cm3', 'nAr_ground_cm3', 'nAr_metastable_cm3',
    'Ec_eV', 'SF6_dissoc_pct', 'R_Penning_m3s'
]
rows = []
for r in res_Ar:
    rows.append([
        f"{r['frac_Ar']:.4f}",
        f"{r['frac_Ar']*100:.1f}",
        "1500.0",
        f"{r['Te']:.4f}",
        f"{r['ne']*1e-6:.6e}",
        f"{r['alpha']:.4f}",
        f"{r['n_F']*1e-6:.6e}",
        f"{r['n_SF6']*1e-6:.6e}",
        f"{r['nAr0']*1e-6:.6e}",
        f"{r['nArm']*1e-6:.6e}",
        f"{r['Ec']:.2f}",
        f"{r['dissoc_frac']*100:.2f}",
        f"{r['R_Penning']:.4e}",
    ])
write_csv("ar_fraction_sweep_1500W.csv", headers, rows)

# --- 3. Alpha vs Ar at 3 pressures ---
for p, res in res_alpha.items():
    headers = ['Ar_fraction', 'Ar_percent', 'alpha', 'Te_eV', 'ne_cm3',
               'nSF6_cm3', 'nF_cm3', 'nAr_metastable_cm3']
    rows = []
    for r in res:
        rows.append([
            f"{r['frac_Ar']:.4f}",
            f"{r['frac_Ar']*100:.1f}",
            f"{r['alpha']:.4f}",
            f"{r['Te']:.4f}",
            f"{r['ne']*1e-6:.6e}",
            f"{r['n_SF6']*1e-6:.6e}",
            f"{r['n_F']*1e-6:.6e}",
            f"{r['nArm']*1e-6:.6e}",
        ])
    write_csv(f"alpha_vs_Ar_{p}mTorr.csv", headers, rows)

# --- 4. Summary table ---
headers = ['Quantity', 'Units', 'Pure_SF6', '20pct_Ar', '50pct_Ar', '80pct_Ar']
idx = np.argmin(np.abs(powers - 1500))
rows = []
for label, key, unit, scale in [
    ('ne', 'ne', 'cm-3', 1e-6),
    ('Te', 'Te', 'eV', 1),
    ('alpha', 'alpha', '-', 1),
    ('nF', 'n_F', 'cm-3', 1e-6),
    ('nF2', 'n_F2', 'cm-3', 1e-6),
    ('nSF6', 'n_SF6', 'cm-3', 1e-6),
    ('nSF5', 'n_SF5', 'cm-3', 1e-6),
    ('nSF4', 'n_SF4', 'cm-3', 1e-6),
    ('nSF3', 'n_SF3', 'cm-3', 1e-6),
    ('nAr_ground', 'nAr0', 'cm-3', 1e-6),
    ('nAr_metastable', 'nArm', 'cm-3', 1e-6),
    ('Ec', 'Ec', 'eV', 1),
    ('eps_T', 'eps_T', 'eV', 1),
    ('SF6_dissociation', 'dissoc_frac', '%', 100),
]:
    vals = []
    for sweep_label in ['0pct_Ar', '20pct_Ar', '50pct_Ar', '80pct_Ar']:
        v = sweeps[sweep_label][idx][key] * scale
        vals.append(f"{v:.6e}" if abs(v) > 100 or abs(v) < 0.01 else f"{v:.4f}")
    rows.append([label, unit] + vals)
write_csv("summary_at_1500W_10mTorr.csv", headers, rows)

print(f"\n✓ All CSV files written to {OUTPUT_DIR}/")
print(f"  Total files: {len(os.listdir(OUTPUT_DIR))}")
