#!/usr/bin/env python3
"""
Phase A.2 — Mettler Fig 4.9 overlay figure.

Two SF6-flow sweeps (10-90 sccm, 100 sccm total, balance Ar) matching
Mettler's Fig 4.9 two TEL operating branches:
  Branch 1: 600 W / 20 mTorr
  Branch 2: 700 W / 40 mTorr

Produces:
  figures/mettler_fig49_overlay.png
  output/mettler_fig49_overlay.csv

Model curves are produced by the Lallement bare 0D SF6/Ar global model
(volume-averaged [F], m^-3). Mettler markers are read from the bundled
digitised CSVs. A horizontal dashed line at 1.1e21 m^-3 marks the
kinetic/diffusion-limited threshold (Mettler Fig 4.9 + Table 4.4).

Note on the cross-reactor gap: the bare Lallement model (eta=0.12, no wall
chemistry) systematically underpredicts the TEL-measured absolute [F]
because the TEL has different eta and geometry. The V3 comparison exists
to document this gap, not to match exactly.
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
LALLEMENT = os.path.join(REPO, 'code', 'lallement_sf6ar')
DATA_METTLER = os.path.join(REPO, 'data', 'mettler')
FIGURES = os.path.join(REPO, 'figures')
OUTPUT = os.path.join(REPO, 'output')

sys.path.insert(0, LALLEMENT)
from sf6_global_model_final import solve_model  # noqa: E402


def load_mettler_fig49(csv_path):
    """Read Fig 4.9b digitised CSV (x=sccm SF6, y=nF_m3)."""
    pts = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = (float(v) for v in line.split(','))
            pts.append((x, y))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def run_flow_sweep(P_rf, p_mTorr, sccm_points):
    """Sweep SF6 flow rate at fixed (P_rf, p, 100 sccm total)."""
    results = []
    for q_SF6 in sccm_points:
        frac_Ar = 1.0 - q_SF6 / 100.0  # balance Ar to reach 100 sccm total
        r = solve_model(P_rf=P_rf, p_mTorr=p_mTorr,
                        frac_Ar=frac_Ar, Q_sccm=100, eta=0.12)
        results.append({
            'q_SF6': q_SF6, 'frac_Ar': frac_Ar, 'P_rf': P_rf, 'p_mTorr': p_mTorr,
            'ne': r['ne'], 'Te': r['Te'], 'alpha': r['alpha'],
            'n_F': r['n_F'], 'converged': r['converged'],
        })
    return results


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    # Mettler markers
    m_lo_x, m_lo_y = load_mettler_fig49(os.path.join(DATA_METTLER, 'mettler_fig4p9b_20mTorr.csv'))
    m_hi_x, m_hi_y = load_mettler_fig49(os.path.join(DATA_METTLER, 'mettler_fig4p9b_40mTorr.csv'))

    # Model sweeps at the 5 Mettler flow points (matches the digitised x-axis positions).
    sccm_points = np.array([10, 30, 50, 70, 90])
    print("Running 600W / 20 mTorr flow sweep...")
    sweep_lo = run_flow_sweep(600, 20, sccm_points)
    for r in sweep_lo:
        print(f"  q_SF6={r['q_SF6']:2.0f} sccm: nF={r['n_F']:.2e} m^-3  "
              f"ne={r['ne']:.2e}  Te={r['Te']:.2f}  conv={r['converged']}")

    print("\nRunning 700W / 40 mTorr flow sweep...")
    sweep_hi = run_flow_sweep(700, 40, sccm_points)
    for r in sweep_hi:
        print(f"  q_SF6={r['q_SF6']:2.0f} sccm: nF={r['n_F']:.2e} m^-3  "
              f"ne={r['ne']:.2e}  Te={r['Te']:.2f}  conv={r['converged']}")

    # CSV for traceability
    csv_out = os.path.join(OUTPUT, 'mettler_fig49_overlay.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['branch', 'q_SF6_sccm', 'P_rf_W', 'p_mTorr',
                    'nF_model_m3', 'ne_m3', 'Te_eV', 'alpha', 'converged'])
        for r in sweep_lo:
            w.writerow(['600W/20mTorr', r['q_SF6'], r['P_rf'], r['p_mTorr'],
                        f"{r['n_F']:.3e}", f"{r['ne']:.3e}", f"{r['Te']:.3f}",
                        f"{r['alpha']:.2f}", r['converged']])
        for r in sweep_hi:
            w.writerow(['700W/40mTorr', r['q_SF6'], r['P_rf'], r['p_mTorr'],
                        f"{r['n_F']:.3e}", f"{r['ne']:.3e}", f"{r['Te']:.3f}",
                        f"{r['alpha']:.2f}", r['converged']])
    print(f"\n[OK] CSV: {csv_out}")

    # Figure
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # Model curves
    ax.plot([r['q_SF6'] for r in sweep_lo],
            [r['n_F'] for r in sweep_lo],
            '-', color='C0', lw=2, label='Model: 600 W / 20 mTorr')
    ax.plot([r['q_SF6'] for r in sweep_hi],
            [r['n_F'] for r in sweep_hi],
            '-', color='C3', lw=2, label='Model: 700 W / 40 mTorr')
    # Mettler markers
    ax.plot(m_lo_x, m_lo_y, 'o', color='C0', ms=8, mfc='white',
            mew=1.5, label='Mettler Fig 4.9 (600 W / 20 mTorr)')
    ax.plot(m_hi_x, m_hi_y, 's', color='C3', ms=8, mfc='white',
            mew=1.5, label='Mettler Fig 4.9 (700 W / 40 mTorr)')
    # Kinetic/diffusion threshold
    ax.axhline(1.1e21, color='gray', ls='--', lw=1,
               label=r'Kinetic/diffusion limit $1.1\times10^{21}$ m$^{-3}$')
    ax.set_yscale('log')
    ax.set_xlabel(r'SF$_6$ flow (sccm, total 100 sccm balance Ar)')
    ax.set_ylabel(r'[F]$_{\mathrm{ICP}}$  (m$^{-3}$)')
    ax.set_title(r'Mettler Fig 4.9 overlay: ICP-region [F] vs SF$_6$ flow')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    png_out = os.path.join(FIGURES, 'mettler_fig49_overlay.png')
    fig.savefig(png_out, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {png_out}")


if __name__ == '__main__':
    main()
