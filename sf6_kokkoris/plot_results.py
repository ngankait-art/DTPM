"""
SF6 Global Plasma Model — Publication-Grade Plot Generator
============================================================
Generates all figures and CSV data files for model results.

Figures produced:
  pressure_rise_and_F_density_vs_power.png
  pressure_rise_and_F_density_vs_pOFF.png
  all_species_and_Te_vs_power.png
  all_species_and_Te_vs_pOFF.png
  species_densities_vs_power.png
  species_densities_vs_Te_power_sweep.png
  species_densities_vs_Te_pressure_sweep.png

CSV data files:
  power_sweep_results.csv
  pressure_sweep_results.csv

Usage:
  python plot_results.py [--outdir OUTPUT_DIRECTORY]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf6_ode_solver import run_to_steady_state
from sf6_global_model import (
    SF6, SF5, SF4, SF3, F2, F_, SF5p, SF4p, SF3p, F2p, SF6m, Fm, EL, N_NEU
)

# =====================================================================
# Publication-grade sweep points
# =====================================================================

POWERS_PUB = [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1250,
              1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]

PRESSURES_PUB = [0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00,
                 2.50, 3.00, 3.50, 4.00, 4.50]

P_OFF_FIXED = 0.921
POWER_FIXED = 2000.0

# =====================================================================
# Run sweeps
# =====================================================================

def run_power_sweep(p_OFF=P_OFF_FIXED, powers=None):
    if powers is None:
        powers = POWERS_PUB
    results = []
    for P in powers:
        r = run_to_steady_state(p_OFF, P, t_max=1.0, verbose=True)
        if r is not None:
            results.append(r)
    return results

def run_pressure_sweep(P_abs=POWER_FIXED, pressures=None):
    if pressures is None:
        pressures = PRESSURES_PUB
    results = []
    for p in pressures:
        r = run_to_steady_state(p, P_abs, t_max=1.0, verbose=True)
        if r is not None:
            results.append(r)
    return results

# =====================================================================
# CSV export
# =====================================================================

CSV_COLUMNS = [
    ('Power_W',    lambda r: r['P']),
    ('pOFF_Pa',    lambda r: r['p_OFF']),
    ('Te_eV',      lambda r: r['Te']),
    ('ne_m3',      lambda r: r['n'][EL]),
    ('alpha',      lambda r: r['alpha']),
    ('dp_Pa',      lambda r: r['dp']),
    ('nSF6_m3',    lambda r: r['n'][SF6]),
    ('nSF5_m3',    lambda r: r['n'][SF5]),
    ('nSF4_m3',    lambda r: r['n'][SF4]),
    ('nSF3_m3',    lambda r: r['n'][SF3]),
    ('nF2_m3',     lambda r: r['n'][F2]),
    ('nF_m3',      lambda r: r['n'][F_]),
    ('nSF5p_m3',   lambda r: r['n'][SF5p]),
    ('nSF4p_m3',   lambda r: r['n'][SF4p]),
    ('nSF3p_m3',   lambda r: r['n'][SF3p]),
    ('nF2p_m3',    lambda r: r['n'][F2p]),
    ('nSF6m_m3',   lambda r: r['n'][SF6m]),
    ('nFm_m3',     lambda r: r['n'][Fm]),
    ('theta_F',    lambda r: r['th'][0]),
    ('theta_SF3',  lambda r: r['th'][1]),
    ('theta_SF4',  lambda r: r['th'][2]),
    ('theta_SF5',  lambda r: r['th'][3]),
]

def write_csv(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([col[0] for col in CSV_COLUMNS])
        for r in results:
            row = []
            for _, extractor in CSV_COLUMNS:
                val = extractor(r)
                if isinstance(val, float) and (abs(val) > 1e4 or (abs(val) < 1e-2 and val != 0)):
                    row.append(f'{val:.6e}')
                else:
                    row.append(f'{val:.6f}')
            writer.writerow(row)
    print(f'  Saved {path}')

# =====================================================================
# Plotting helpers
# =====================================================================

NEU_SPEC = [
    (SF6,  'SF$_6$',  'tab:blue',   '-',  'o'),
    (SF5,  'SF$_5$',  'tab:green',  '-',  's'),
    (SF4,  'SF$_4$',  'tab:red',    '-',  '^'),
    (SF3,  'SF$_3$',  'tab:orange', '-',  'v'),
    (F2,   'F$_2$',   'tab:purple', '-',  'D'),
    (F_,   'F',       'tab:brown',  '-',  'p'),
]

ION_SPEC = [
    (SF5p, 'SF$_5^+$', 'tab:blue',   '-',  'o'),
    (SF4p, 'SF$_4^+$', 'tab:green',  '-',  's'),
    (SF3p, 'SF$_3^+$', 'tab:red',    '-',  '^'),
    (F2p,  'F$_2^+$',  'tab:orange', '-',  'v'),
    (SF6m, 'SF$_6^-$', 'tab:cyan',   '--', 'D'),
    (Fm,   'F$^-$',    'tab:purple', '--', 'p'),
    (EL,   'e$^-$',    'black',      '-.', 'x'),
]

def _extract(results, key):
    return [r[key] for r in results]

def _extract_n(results, idx):
    return [r['n'][idx] for r in results]

# =====================================================================
# Figure generators
# =====================================================================

def plot_fig5(results, path):
    P = _extract(results, 'P'); dp = _extract(results, 'dp'); nF = _extract_n(results, F_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(P, dp, 'b-o', ms=4, lw=1.2)
    ax1.set_xlabel('Power (W)', fontsize=12); ax1.set_ylabel('Pressure rise (Pa)', fontsize=12)
    ax1.set_title(f'Pressure rise vs Power (p$_{{OFF}}$={results[0]["p_OFF"]:.3f} Pa)'); ax1.grid(True, alpha=0.3)
    ax2.plot(P, nF, 'r-o', ms=4, lw=1.2)
    ax2.set_xlabel('Power (W)', fontsize=12); ax2.set_ylabel('F density (m$^{-3}$)', fontsize=12)
    ax2.set_title('F density vs Power'); ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

def plot_fig6(results, path):
    poff = _extract(results, 'p_OFF'); dp = _extract(results, 'dp'); nF = _extract_n(results, F_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(poff, dp, 'b-o', ms=4, lw=1.2)
    ax1.set_xlabel('p$_{OFF}$ (Pa)', fontsize=12); ax1.set_ylabel('Pressure rise (Pa)', fontsize=12)
    ax1.set_title(f'Pressure rise vs p$_{{OFF}}$ (P={results[0]["P"]:.0f} W)'); ax1.grid(True, alpha=0.3)
    ax2.plot(poff, nF, 'r-o', ms=4, lw=1.2)
    ax2.set_xlabel('p$_{OFF}$ (Pa)', fontsize=12); ax2.set_ylabel('F density (m$^{-3}$)', fontsize=12)
    ax2.set_title('F density vs p$_{OFF}$'); ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

def plot_fig7(results, path):
    P = _extract(results, 'P'); fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, label, color, ls, mk in NEU_SPEC:
        ax1.semilogy(P, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=3, lw=1.2, label=label)
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('Density (m$^{-3}$)'); ax1.set_title('Neutral species vs Power')
    ax1.set_ylim(1e16, 1e21); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    for idx, label, color, ls, mk in ION_SPEC:
        ax2.semilogy(P, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=3, lw=1.2, label=label)
    ax2.set_xlabel('Power (W)'); ax2.set_ylabel('Density (m$^{-3}$)'); ax2.set_title('Charged species vs Power')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    alpha = _extract(results, 'alpha'); Te = _extract(results, 'Te')
    ax3.plot(P, alpha, 'b-o', ms=3, lw=1.2); ax3.set_xlabel('Power (W)')
    ax3.set_ylabel('n$^-$/n$_e$', color='tab:blue'); ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3r = ax3.twinx(); ax3r.plot(P, Te, 'r-s', ms=3, lw=1.2)
    ax3r.set_ylabel('T$_e$ (eV)', color='tab:red'); ax3r.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_title('Electronegativity & T$_e$'); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

def plot_fig8(results, path):
    poff = _extract(results, 'p_OFF'); fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, label, color, ls, mk in NEU_SPEC:
        ax1.semilogy(poff, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=3, lw=1.2, label=label)
    ax1.set_xlabel('p$_{OFF}$ (Pa)'); ax1.set_ylabel('Density (m$^{-3}$)'); ax1.set_title('Neutral species vs p$_{OFF}$')
    ax1.set_ylim(1e16, 1e21); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    for idx, label, color, ls, mk in ION_SPEC:
        ax2.semilogy(poff, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=3, lw=1.2, label=label)
    ax2.set_xlabel('p$_{OFF}$ (Pa)'); ax2.set_ylabel('Density (m$^{-3}$)'); ax2.set_title('Charged species vs p$_{OFF}$')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    alpha = _extract(results, 'alpha'); Te = _extract(results, 'Te')
    ax3.plot(poff, alpha, 'b-o', ms=3, lw=1.2); ax3.set_xlabel('p$_{OFF}$ (Pa)')
    ax3.set_ylabel('n$^-$/n$_e$', color='tab:blue'); ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3r = ax3.twinx(); ax3r.plot(poff, Te, 'r-s', ms=3, lw=1.2)
    ax3r.set_ylabel('T$_e$ (eV)', color='tab:red'); ax3r.tick_params(axis='y', labelcolor='tab:red')
    ax3.set_title('Electronegativity & T$_e$'); ax3.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

def plot_densities_vs_Te(results, path, sweep_label='Power sweep'):
    Te = _extract(results, 'Te'); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for idx, label, color, ls, mk in NEU_SPEC:
        ax1.semilogy(Te, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=5, lw=1.5, label=label)
    ax1.set_xlabel('Electron temperature, T$_e$ (eV)', fontsize=13)
    ax1.set_ylabel('Number density (m$^{-3}$)', fontsize=13); ax1.set_title('Neutral species vs T$_e$', fontsize=14)
    ax1.set_ylim(1e16, 1e21); ax1.legend(fontsize=10, loc='best'); ax1.grid(True, alpha=0.3)
    for idx, label, color, ls, mk in ION_SPEC:
        ax2.semilogy(Te, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=5, lw=1.5, label=label)
    ax2.set_xlabel('Electron temperature, T$_e$ (eV)', fontsize=13)
    ax2.set_ylabel('Number density (m$^{-3}$)', fontsize=13); ax2.set_title('Charged species vs T$_e$', fontsize=14)
    ax2.legend(fontsize=9, loc='best'); ax2.grid(True, alpha=0.3)
    fig.suptitle(f'Species densities vs electron temperature ({sweep_label})', fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

def plot_densities_vs_power(results, path):
    P = _extract(results, 'P'); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for idx, label, color, ls, mk in NEU_SPEC:
        ax1.semilogy(P, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=5, lw=1.5, label=label)
    ax1.set_xlabel('Power (W)', fontsize=13); ax1.set_ylabel('Number density (m$^{-3}$)', fontsize=13)
    ax1.set_title('Neutral species vs Power', fontsize=14); ax1.set_ylim(1e16, 1e21)
    ax1.legend(fontsize=10, loc='best'); ax1.grid(True, alpha=0.3)
    for idx, label, color, ls, mk in ION_SPEC:
        ax2.semilogy(P, _extract_n(results, idx), color=color, ls=ls, marker=mk, ms=5, lw=1.5, label=label)
    ax2.set_xlabel('Power (W)', fontsize=13); ax2.set_ylabel('Number density (m$^{-3}$)', fontsize=13)
    ax2.set_title('Charged species vs Power', fontsize=14)
    ax2.legend(fontsize=9, loc='best'); ax2.grid(True, alpha=0.3)
    fig.suptitle(f'Species densities vs Power (p$_{{OFF}}$={results[0]["p_OFF"]:.3f} Pa)', fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close(); print(f'  Saved {path}')

# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SF6 Global Model — Publication-Grade Plot Generator')
    parser.add_argument('--outdir', default='output', help='Output directory for plots and CSV')
    args = parser.parse_args()

    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)

    print('='*70)
    print('SF6 Global Plasma Model — Publication-Grade Figures & Data')
    print('='*70)
    print(f'Power sweep:    {len(POWERS_PUB)} points, {POWERS_PUB[0]}–{POWERS_PUB[-1]} W at p_OFF={P_OFF_FIXED} Pa')
    print(f'Pressure sweep: {len(PRESSURES_PUB)} points, {PRESSURES_PUB[0]}–{PRESSURES_PUB[-1]} Pa at P={POWER_FIXED:.0f} W')

    print(f'\n--- Power sweep ({len(POWERS_PUB)} points, p_OFF = {P_OFF_FIXED} Pa) ---')
    rP = run_power_sweep()
    if len(rP) >= 2:
        write_csv(rP, os.path.join(OUT, 'power_sweep_results.csv'))
        plot_fig5(rP, os.path.join(OUT, 'pressure_rise_and_F_density_vs_power.png'))
        plot_fig7(rP, os.path.join(OUT, 'all_species_and_Te_vs_power.png'))
        plot_densities_vs_power(rP, os.path.join(OUT, 'species_densities_vs_power.png'))
        plot_densities_vs_Te(rP, os.path.join(OUT, 'species_densities_vs_Te_power_sweep.png'),
                            sweep_label='power sweep, p$_{OFF}$=0.921 Pa')

    print(f'\n--- Pressure sweep ({len(PRESSURES_PUB)} points, P = {POWER_FIXED:.0f} W) ---')
    rp = run_pressure_sweep()
    if len(rp) >= 2:
        write_csv(rp, os.path.join(OUT, 'pressure_sweep_results.csv'))
        plot_fig6(rp, os.path.join(OUT, 'pressure_rise_and_F_density_vs_pOFF.png'))
        plot_fig8(rp, os.path.join(OUT, 'all_species_and_Te_vs_pOFF.png'))
        plot_densities_vs_Te(rp, os.path.join(OUT, 'species_densities_vs_Te_pressure_sweep.png'),
                            sweep_label='pressure sweep, P=2000 W')

    print(f'\nAll outputs saved to {OUT}/')
    print('Done.')
