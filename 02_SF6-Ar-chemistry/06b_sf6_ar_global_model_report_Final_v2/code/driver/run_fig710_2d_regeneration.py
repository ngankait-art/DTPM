#!/usr/bin/env python3
"""
Phase A.5 — Fig 7.10 radial profile regeneration from cached stage-7 2D solver.

REPLACES the 06a report's Fig 7.10 which was captioned as a 700 W / 10 mTorr
radial benchmark but is actually Mettler's 1000 W / 10 mTorr wafer-radial data
(METTLER_gap_report_CHATGPT_FEEDBACK.md Section B.1).

This driver uses pre-computed 2D solver output from the Stage-7 hybrid
framework (folder 6: Phase1_A Self-Consistent Global-2D Hybrid) at Mettler's
exact Fig 4.14/4.17 operating point:
  P_ICP = 1000 W, p = 10 mTorr, gamma_Al = 0.18, bias OFF
  Two compositions:
    90:10 SF6:Ar (matches Mettler Fig 4.17 top panel, 90%-SF6 branch)
    70:30 SF6:Ar (30% SF6 = 70% Ar) — matches Fig 4.17 top panel 30%-SF6 branch
                  and is close-but-not-identical to Fig 4.14 (70:30 SF6:Ar,
                  which is the SAME gas composition, but bias ON)

The cached NPY arrays live in 06b/data/stage7_cached/. This driver is
self-contained: it reconstructs the stretched-mesh radial coordinates
inline (no dependency on folder 6 source code).

Caveat: the cached stage-7 runs are bias-OFF; Mettler Fig 4.14's cubic
fit is bias-ON (200 W rf wafer bias). Bias scales absolute density by
~1.6-2.15x uniformly, so the NORMALISED radial shape is approximately
bias-invariant; comparison against the cubic fit shape is defensible
for validation of the spatial profile mechanism even though absolute
magnitudes differ.

Produces:
  figures/fig710_radial_F_1000W_regenerated.png
  output/fig710_radial_profile_1000W.csv

Source of cached arrays (must remain discoverable for reproducibility):
  06b/data/stage7_cached/30pct_SF6_bias_off/nF.npy  (50, 80)  -- r=(0..R), z=(wafer..top)
  06b/data/stage7_cached/90pct_SF6_bias_off/nF.npy  (50, 80)
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHED = os.path.join(REPO, 'data', 'stage7_cached')
DATA_METTLER = os.path.join(REPO, 'data', 'mettler')
FIGURES = os.path.join(REPO, 'figures')
OUTPUT = os.path.join(REPO, 'output')

# Stage-7 mesh parameters (reproduced inline from
# 6_Phase1_A .../src/dtpm/core/mesh.py for standalone reproducibility)
R_DOMAIN = 0.105   # m  (TEL processing region radius)
L_DOMAIN = 0.220   # m  (ICP + processing total axial length)
NR = 50
NZ = 80
BETA_R = 1.2       # tanh stretching toward r=R (clusters cells near wall)


def stretched_faces(L, N, beta, two_sided=False):
    """Reproduce Mesh2D._stretched_faces() from folder 6."""
    xi = np.linspace(0, 1, N + 1)
    if beta <= 1.0 + 1e-10:
        return L * xi
    if two_sided:
        return L * 0.5 * (1.0 + np.tanh(beta * (2 * xi - 1)) / np.tanh(beta))
    return L * (np.tanh(beta * xi) / np.tanh(beta))


def radial_cell_centres():
    """Cell-centre radial coordinates matching folder 6's Mesh2D."""
    rf = stretched_faces(R_DOMAIN, NR, BETA_R, two_sided=False)
    rc = 0.5 * (rf[:-1] + rf[1:])
    return rc  # shape (NR,) in metres


def load_stage7(composition):
    """Load nF.npy + summary.json from cached stage-7 run."""
    d = os.path.join(CACHED, f'{composition}_bias_off')
    nF = np.load(os.path.join(d, 'nF.npy'))
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    return nF, summary


def mettler_cubic_fit(r_cm):
    """Mettler Fig 4.14 cubic fit: y(r) = 1.01032 - 0.01847 r^2 + 7.139e-4 r^3."""
    return 1.01032 - 0.01847 * r_cm**2 + 7.139e-4 * r_cm**3


def load_mettler_fig414_data():
    """Return r_cm, y_normalised for Mettler Fig 4.14 5-point data."""
    path = os.path.join(DATA_METTLER, 'mettler_fig414_F_normalized_data.csv')
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = (float(v) for v in line.split(','))
                pts.append((x, y))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def load_mettler_fig417(composition_csv):
    """Read Mettler Fig 4.17 bias-off density CSV (r_cm, nF_m3) and return normalised."""
    path = os.path.join(DATA_METTLER, composition_csv)
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = (float(v) for v in line.split(','))
                pts.append((x, y))
    pts.sort()
    r = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return r, y / y[0]  # normalise to r=0


def main():
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    # Radial coordinates (m -> cm for plotting)
    rc_m = radial_cell_centres()
    rc_cm = rc_m * 100.0

    # Load cached stage-7 results at Mettler's two composition conditions
    nF_30, sum_30 = load_stage7('30pct_SF6')
    nF_90, sum_90 = load_stage7('90pct_SF6')

    print("Loaded stage-7 cached 2D results:")
    print(f"  30% SF6 bias-off: F_drop = {sum_30['F_drop_pct']:.2f}%, eta = {sum_30['eta_computed']:.3f}")
    print(f"  90% SF6 bias-off: F_drop = {sum_90['F_drop_pct']:.2f}%, eta = {sum_90['eta_computed']:.3f}")

    # Extract wafer-slice (z-index 0) radial profile
    nF_30_wafer = nF_30[:, 0]
    nF_90_wafer = nF_90[:, 0]

    # Normalise to centre
    nF_30_norm = nF_30_wafer / nF_30_wafer[0]
    nF_90_norm = nF_90_wafer / nF_90_wafer[0]

    # Model centre-to-edge drops (use r ≈ 8 cm point for comparison to Mettler who reports 0-8cm)
    # Find the index closest to r = 8 cm
    idx_8cm = int(np.argmin(np.abs(rc_cm - 8.0)))
    drop_30_at_8 = (1 - nF_30_norm[idx_8cm]) * 100
    drop_90_at_8 = (1 - nF_90_norm[idx_8cm]) * 100
    print(f"\nCentre-to-r=8cm drop from model:")
    print(f"  30% SF6: {drop_30_at_8:.1f}%  (Mettler: 67%)")
    print(f"  90% SF6: {drop_90_at_8:.1f}%  (Mettler: 76%)")

    # Load Mettler reference data
    r_f414, y_f414 = load_mettler_fig414_data()
    r_f417_30, y_f417_30_norm = load_mettler_fig417(
        'mettler_fig4p17_30pctSF6_biasoff_density.csv')
    r_f417_90, y_f417_90_norm = load_mettler_fig417(
        'mettler_fig4p17_90pctSF6_biasoff_density.csv')

    # Cubic-fit curve for overlay
    r_fine = np.linspace(0, 10.5, 200)
    y_cubic = mettler_cubic_fit(r_fine)

    # Write traceability CSV
    csv_out = os.path.join(OUTPUT, 'fig710_radial_profile_1000W.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['r_cm', 'nF_30pctSF6_m3', 'nF_30pctSF6_normalised',
                    'nF_90pctSF6_m3', 'nF_90pctSF6_normalised'])
        for i, r in enumerate(rc_cm):
            w.writerow([f"{r:.4f}", f"{nF_30_wafer[i]:.3e}",
                        f"{nF_30_norm[i]:.4f}", f"{nF_90_wafer[i]:.3e}",
                        f"{nF_90_norm[i]:.4f}"])
    print(f"\n[OK] CSV: {csv_out}")

    # Figure: two-panel normalised radial profile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # --- Left panel: 30% SF6 vs Mettler Fig 4.17 30%-SF6 bias-off ---
    ax1.plot(rc_cm, nF_30_norm, '-', color='C0', lw=2,
             label=f'Model (Stage-7 2D, drop={drop_30_at_8:.0f}%)')
    ax1.plot(r_f417_30, y_f417_30_norm, 'o', color='C3', ms=8, mfc='white',
             mew=1.5, label='Mettler Fig 4.17 (30% SF$_6$ bias off, 67%)')
    ax1.set_xlabel('r (cm)')
    ax1.set_ylabel(r'[F] / [F]$_{r=0}$')
    ax1.set_title(r'30% SF$_6$: 1000 W / 10 mTorr / bias off')
    ax1.set_xlim(0, 10.5)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')

    # --- Right panel: 90% SF6 with Mettler Fig 4.14 (70:30 bias-on cubic fit)
    # and Fig 4.17 (90% bias-off) markers ---
    ax2.plot(rc_cm, nF_90_norm, '-', color='C0', lw=2,
             label=f'Model (Stage-7 2D, drop={drop_90_at_8:.0f}%)')
    ax2.plot(r_f417_90, y_f417_90_norm, 'o', color='C3', ms=8, mfc='white',
             mew=1.5, label='Mettler Fig 4.17 (90% SF$_6$ bias off, 76%)')
    ax2.plot(r_fine, y_cubic, '--', color='C2', lw=1.5,
             label='Mettler Fig 4.14 cubic fit (70:30 bias ON, 74%)')
    ax2.plot(r_f414, y_f414, 's', color='C2', ms=7, mfc='white', mew=1.5,
             label='Mettler Fig 4.14 data (5 points)')
    ax2.set_xlabel('r (cm)')
    ax2.set_ylabel(r'[F] / [F]$_{r=0}$')
    ax2.set_title(r'90% SF$_6$: 1000 W / 10 mTorr / bias off')
    ax2.set_xlim(0, 10.5)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='lower left')

    fig.suptitle(r'Normalised radial [F] profile at the wafer (Mettler Fig 4.14/4.17 conditions, '
                 r'$\gamma_{\mathrm{Al}}=0.18$)',
                 fontsize=11, y=1.0)
    fig.text(0.5, 0.015,
             r'Model is the Stage-7 (Phase-1_A) 2D self-consistent hybrid solver at bias OFF; Mettler '
             r'Fig 4.14 is bias ON (shape is approximately bias-invariant).',
             ha='center', va='bottom', fontsize=8, style='italic', color='gray')
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    png_out = os.path.join(FIGURES, 'fig710_radial_F_1000W_regenerated.png')
    fig.savefig(png_out, dpi=150)
    plt.close(fig)
    print(f"[OK] Figure: {png_out}")


if __name__ == '__main__':
    main()
