#!/usr/bin/env python3
"""
Phase A.3 — Mettler Fig 4.17 overlay figure.

Compares model volume-averaged [F] to Mettler's centre-of-wafer radical-probe
measurements at the radial-wafer operating condition:
  P_ICP = 1000 W, p = 10 mTorr, 100 sccm total flow, bias off
Two compositions:
  - 90% SF6 (10 sccm Ar)
  - 30% SF6 (70 sccm Ar)

Uses the wall-chemistry-extended solver with the TEL processing-region
geometry, matching the 06a report's Chapter 4 calibration
(WC = Kokkoris × 0.007, eta = 0.16).

Produces:
  figures/mettler_fig417_overlay.png
  output/mettler_fig417_overlay.csv

Caption caveat: the 0D model produces a volume-averaged [F]; Mettler's
markers are centre-of-wafer (r=0) probe measurements. The ~1.62×
actinometry-to-centre ratio (Mettler Eq 4.2) is annotated.
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
from engine import solve_model, Reactor  # noqa: E402

# Kokkoris × 0.007 calibration (06a report wall-chemistry chapter baseline)
WC = {'s_F': 0.00105, 's_SFx': 0.00056, 'p_fluor': 0.025,
      'p_wallrec': 0.007, 'p_FF': 0.0035,
      'p_F_SF3': 0.5, 'p_F_SF4': 0.2}

# TEL processing-region geometry (same as 06a report Chapter 4)
TEL = Reactor(R=0.105, L=0.0535)


def load_mettler_fig417_r0(csv_path):
    """First row (r=0, centre) of the digitised density file."""
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = (float(v) for v in line.split(','))
                return x, y
    return None, None


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    # Mettler centre markers (bias-off)
    _, met_90 = load_mettler_fig417_r0(
        os.path.join(DATA_METTLER, 'mettler_fig4p17_90pctSF6_biasoff_density.csv'))
    _, met_30 = load_mettler_fig417_r0(
        os.path.join(DATA_METTLER, 'mettler_fig4p17_30pctSF6_biasoff_density.csv'))

    print(f"Mettler Fig 4.17 centre markers (bias off):")
    print(f"  90% SF6: [F]_centre = {met_90:.2e} m^-3")
    print(f"  30% SF6: [F]_centre = {met_30:.2e} m^-3")

    # Model: wall-chemistry extended, TEL geometry, eta=0.16 (matching Ch 4 calibration)
    common = dict(P_rf=1000, p_mTorr=10, Q_sccm=100, eta=0.16,
                  wall_chem=WC, reactor=TEL)

    print("\nRunning wall-chem extended model...")
    print("  90% SF6 (frac_Ar=0.10):")
    r_90 = solve_model(frac_Ar=0.10, **common)
    print(f"    nF(vol-avg) = {r_90['n_F']:.2e} m^-3  ne={r_90['ne']:.2e}  Te={r_90['Te']:.2f}")

    print("  30% SF6 (frac_Ar=0.70):")
    r_30 = solve_model(frac_Ar=0.70, **common)
    print(f"    nF(vol-avg) = {r_30['n_F']:.2e} m^-3  ne={r_30['ne']:.2e}  Te={r_30['Te']:.2f}")

    # Traceability CSV
    csv_out = os.path.join(OUTPUT, 'mettler_fig417_overlay.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['composition', 'frac_Ar', 'nF_model_volavg_m3',
                    'nF_Mettler_centre_m3', 'ratio_model_over_centre',
                    'Te_eV', 'ne_m3', 'alpha', 'converged'])
        for comp, r, met in [('90% SF6', r_90, met_90), ('30% SF6', r_30, met_30)]:
            w.writerow([comp, r.get('frac_Ar', '-'),
                        f"{r['n_F']:.3e}", f"{met:.3e}",
                        f"{r['n_F']/met:.3f}",
                        f"{r['Te']:.3f}", f"{r['ne']:.3e}",
                        f"{r['alpha']:.2f}", r['converged']])
    print(f"\n[OK] CSV: {csv_out}")

    # Bar chart: model vs Mettler centre
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = np.arange(2)
    w = 0.35
    model_vals = [r_90['n_F'], r_30['n_F']]
    mett_vals = [met_90, met_30]
    ax.bar(x - w/2, model_vals, w, color='C0', alpha=0.8,
           label='Model (vol-averaged, wall-chem, TEL geom)')
    ax.bar(x + w/2, mett_vals, w, color='C3', alpha=0.8,
           label='Mettler Fig 4.17 (centre, W/Al probe, bias off)')
    ax.set_xticks(x)
    ax.set_xticklabels(['90% SF6\n(10 sccm Ar)', '30% SF6\n(70 sccm Ar)'])
    ax.set_ylabel(r'[F]  (m$^{-3}$)')
    ax.set_title(r'Mettler Fig 4.17 overlay: centre [F] at 1000 W / 10 mTorr, bias off')
    ax.set_yscale('log')
    ax.set_ylim(1e19, 1e21)
    ax.grid(True, which='major', axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    # Annotate ratios on bars
    for i, (m, d) in enumerate(zip(model_vals, mett_vals)):
        ax.annotate(f'{m:.2e}', (i - w/2, m), ha='center', va='bottom', fontsize=8)
        ax.annotate(f'{d:.2e}', (i + w/2, d), ha='center', va='bottom', fontsize=8)
        ax.annotate(f'ratio = {m/d:.2f}',
                    (i, min(m, d) * 0.1), ha='center', va='top', fontsize=8, color='gray')

    # Footnote about the actinometry-to-centre ratio
    fig.text(0.5, 0.02,
             r'Note: Mettler Eq 4.2 reports $n_{\mathrm{centre}}/n_{\mathrm{act}} \approx 1.62$; '
             r'the 0D model output is volume-averaged.',
             ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png_out = os.path.join(FIGURES, 'mettler_fig417_overlay.png')
    fig.savefig(png_out, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {png_out}")


if __name__ == '__main__':
    main()
