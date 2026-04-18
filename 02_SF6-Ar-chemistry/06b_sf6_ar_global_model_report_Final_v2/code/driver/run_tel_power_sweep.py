#!/usr/bin/env python3
"""
Phase A.4 — Replacement TEL power sweep figure.

This figure REPLACES the incorrectly cited Mettler Fig 4.5 (Helicon/PMIC
reference) that previously appeared in the 06a report as
`mettler_benchmark_fig4p5.png` and `fig7_tel_F_vs_power.png`. Both original
figures used PMIC helicon data mis-attributed as TEL benchmarks (E2).

This new figure uses the wall-chemistry-extended 0D model with the TEL
processing-region geometry, at Mettler's Fig 4.17 30%-SF6 bias-off
operating condition (1000 W / 10 mTorr / 30% SF6 / 100 sccm total).

Model: wall-chemistry extended, TEL geometry, eta = 0.16, WC = Kokkoris x 0.007.
Overlay: single Mettler Fig 4.17 30%-SF6 bias-off centre marker at 1000 W.

Produces:
  figures/tel_power_sweep_1000W_30pctAr.png
  output/tel_power_sweep_1000W_30pctAr.csv
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
WALL_CHEM_CODE = os.path.join(REPO, 'code', 'wall_chemistry')
DATA_METTLER = os.path.join(REPO, 'data', 'mettler')
FIGURES = os.path.join(REPO, 'figures')
OUTPUT = os.path.join(REPO, 'output')

sys.path.insert(0, WALL_CHEM_CODE)
from engine import solve_model, sweep_with_continuation, Reactor  # noqa: E402

WC = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
      'p_wallrec': 0.007, 'p_FF': 0.0035,
      'p_F_SF3': 0.5, 'p_F_SF4': 0.2}

TEL = Reactor(R=0.105, L=0.0535)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    powers = np.linspace(400, 1500, 12)  # W
    base = dict(p_mTorr=10, frac_Ar=0.70,  # 30% SF6 = 70% Ar
                Q_sccm=100, eta=0.16, wall_chem=WC, reactor=TEL)

    print(f"Running wall-chem TEL power sweep 400-1500 W at 10 mTorr, 30% SF6...")
    results = sweep_with_continuation('P_rf', powers, base, verbose=True)

    # Mettler anchor: 30%-SF6 bias-off, 1000 W, r=0 centre
    csv_path = os.path.join(DATA_METTLER, 'mettler_fig4p17_30pctSF6_biasoff_density.csv')
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                _, y_mett = (float(v) for v in line.split(','))
                break
    print(f"\nMettler Fig 4.17 30% SF6 bias-off centre = {y_mett:.2e} m^-3 at 1000 W")

    # Traceability CSV
    csv_out = os.path.join(OUTPUT, 'tel_power_sweep_1000W_30pctAr.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['P_rf_W', 'nF_model_volavg_m3', 'ne_m3', 'Te_eV',
                    'alpha', 'dissoc_frac', 'converged'])
        for r in results:
            w.writerow([r['P_rf'], f"{r['n_F']:.3e}", f"{r['ne']:.3e}",
                        f"{r['Te']:.3f}", f"{r['alpha']:.2f}",
                        f"{r['dissoc_frac']:.3f}", r['converged']])
    print(f"[OK] CSV: {csv_out}")

    # Figure
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot([r['P_rf'] for r in results],
            [r['n_F'] for r in results],
            '-o', color='C0', lw=2, ms=5,
            label='Model (wall-chem, TEL geometry)')
    ax.plot([1000], [y_mett], 'D', color='C3', ms=12, mfc='C3',
            label='Mettler Fig 4.17 (30% SF$_6$ bias off, centre)')
    ax.set_xlabel('ICP power (W)')
    ax.set_ylabel(r'[F]  (m$^{-3}$)')
    ax.set_title(r'TEL [F] vs ICP power at 10 mTorr, 30% SF$_6$, bias off')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    # Annotation
    fig.text(0.5, 0.02,
             r'Note: model is volume-averaged; Mettler marker is centre-of-wafer '
             r'(W/Al radical probe). Ratio $n_{\mathrm{centre}}/n_{\mathrm{vol-avg}} \sim 1.6$ expected.',
             ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png_out = os.path.join(FIGURES, 'tel_power_sweep_1000W_30pctAr.png')
    fig.savefig(png_out, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {png_out}")


if __name__ == '__main__':
    main()
