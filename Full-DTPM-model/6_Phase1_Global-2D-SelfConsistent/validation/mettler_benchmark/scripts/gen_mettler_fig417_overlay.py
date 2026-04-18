#!/usr/bin/env python3
"""Three-panel direct benchmark against Mettler Fig 4.14 / 4.17.

Panel (a): Normalised radial [F] at the wafer (1000 W, 90% SF6) overlaid with
           Mettler's Fig 4.14 cubic fit and the 5 digitised data points.
Panel (b): Absolute [F] at the wafer for both 30% SF6 and 90% SF6 runs,
           with Mettler Fig 4.17 bias-off data points overlaid.
Panel (c): Residual = 100*(model - Mettler)/Mettler at the five Mettler radii
           for both compositions. Honestly shows the outer-wafer 15-27% over-
           prediction (see METTLER_gap_report_CHATGPT_FEEDBACK.md).

Inputs:
  5a_Phase1_Mettler_Validation_Correction/results/mettler_fig417/90pct/nF.npy
  5a_Phase1_Mettler_Validation_Correction/results/mettler_fig417/30pct/nF.npy
  (plus mesh + inside from the shared Phase-1 default config)

Output:
  5a_Phase1_Mettler_Validation_Correction/figures/fig_mettler_fig417_overlay.{png,pdf}
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE5A_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
STEPS_DIR = os.path.abspath(os.path.join(STAGE5A_ROOT, '..'))
PHASE1_ROOT = os.path.join(STEPS_DIR, '5.Phase1_EM_Chemistry_Merged')

sys.path.insert(0, os.path.join(PHASE1_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'src'))

# ---------------------------------------------------------------------------
# Mettler digitised data (see METTLER_TEL_DATA_DIGITISED.md)
# ---------------------------------------------------------------------------
# Fig 4.14 cubic fit, r in cm.
def mettler_fig414_cubic(r_cm):
    return 1.01032 - 0.01847 * r_cm**2 + 7.139e-4 * r_cm**3


METTLER_R_CM = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

# Fig 4.14 fit values at the 5 Mettler radii (normalised).
METTLER_FIG414_NORM = mettler_fig414_cubic(METTLER_R_CM)

# Fig 4.17 top panel (bias OFF), absolute n_F in cm^-3.
# 90% SF6 (90 sccm SF6 / 10 sccm Ar).
METTLER_FIG417_90PCT_NF_CM3 = np.array([2.5, 2.3, 1.6, 1.0, 0.6]) * 1e14
# 30% SF6 (30 sccm SF6 / 70 sccm Ar).
METTLER_FIG417_30PCT_NF_CM3 = np.array([0.60, 0.55, 0.42, 0.28, 0.20]) * 1e14


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.2,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
    })


def _build_mesh_and_inside():
    """Rebuild (mesh, inside) from the Phase-1 default config."""
    from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask
    cfg_path = os.path.join(PHASE1_ROOT, 'config', 'default_config.yaml')
    config = SimulationConfig(cfg_path)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total,
                  Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _bc = build_geometry_mask(
        mesh, tel['R_icp'], tel['R_proc'],
        tel['L_proc'], tel['L_proc'] + tel['L_apt'], L_total,
    )
    return mesh, inside


def load_nF(result_dir):
    """Return nF field saved by save_sweep_point."""
    return np.load(os.path.join(result_dir, 'nF.npy'))


def wafer_radial(nF, mesh, inside):
    """Return (r_cm, n_F_cm3) along the wafer surface (z index 0)."""
    r_cm = mesh.rc[inside[:, 0]] * 1e2  # m -> cm
    F_w = nF[:, 0][inside[:, 0]] * 1e-6  # 1/m^3 -> cm^-3
    return r_cm, F_w


def interp_model_at_mettler(r_cm_model, F_cm3_model):
    """Interpolate the model wafer profile onto Mettler's 5 radii (0-8 cm)."""
    return np.interp(METTLER_R_CM, r_cm_model, F_cm3_model)


def make_figure(nF_90, nF_30, mesh, inside, out_dir):
    setup_style()

    r_cm_90, F_cm3_90 = wafer_radial(nF_90, mesh, inside)
    r_cm_30, F_cm3_30 = wafer_radial(nF_30, mesh, inside)

    F_model_90 = interp_model_at_mettler(r_cm_90, F_cm3_90)
    F_model_30 = interp_model_at_mettler(r_cm_30, F_cm3_30)

    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --------- Panel (a): normalised radial + cubic fit ---------
    ax = axes[0]
    r_fit = np.linspace(0, 8, 200)
    ax.plot(r_fit, mettler_fig414_cubic(r_fit), '--', color='tab:green', lw=2.2,
            label='Mettler Fig 4.14 cubic fit')
    ax.plot(METTLER_R_CM, METTLER_FIG414_NORM, 'o', color='black', ms=10,
            mfc='white', mew=2.0, label='Mettler 5-pt data')

    if F_cm3_90[0] > 0:
        ax.plot(r_cm_90, F_cm3_90 / F_cm3_90[0], '-', color='tab:red', lw=2.5,
                label=f'Model 1000 W, 90% SF$_6$ ({(1 - F_cm3_90[-1] / F_cm3_90[0]) * 100:.0f}%)')
    if F_cm3_30[0] > 0:
        ax.plot(r_cm_30, F_cm3_30 / F_cm3_30[0], '-', color='tab:blue', lw=2.5,
                label=f'Model 1000 W, 30% SF$_6$ ({(1 - F_cm3_30[-1] / F_cm3_30[0]) * 100:.0f}%)')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('$r$ (cm)')
    ax.set_ylabel('[F] / [F]$_\\mathrm{centre}$')
    ax.set_title('(a) Normalised radial [F]', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True)

    # --------- Panel (b): absolute radial [F] ---------
    ax = axes[1]
    ax.plot(METTLER_R_CM, METTLER_FIG417_90PCT_NF_CM3, 'o', color='tab:red',
            ms=10, mfc='white', mew=2.0, label='Mettler 90% SF$_6$ (bias off)')
    ax.plot(METTLER_R_CM, METTLER_FIG417_30PCT_NF_CM3, 's', color='tab:blue',
            ms=10, mfc='white', mew=2.0, label='Mettler 30% SF$_6$ (bias off)')
    ax.plot(r_cm_90, F_cm3_90, '-', color='tab:red', lw=2.5,
            label='Model 90% SF$_6$')
    ax.plot(r_cm_30, F_cm3_30, '-', color='tab:blue', lw=2.5,
            label='Model 30% SF$_6$')
    ax.set_xlim(0, 10)
    ax.set_xlabel('$r$ (cm)')
    ax.set_ylabel('[F] at wafer (cm$^{-3}$)')
    ax.set_title('(b) Absolute radial [F]', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, which='both')

    # --------- Panel (c): pointwise residuals ---------
    ax = axes[2]
    resid_90 = 100.0 * (F_model_90 - METTLER_FIG417_90PCT_NF_CM3) / METTLER_FIG417_90PCT_NF_CM3
    resid_30 = 100.0 * (F_model_30 - METTLER_FIG417_30PCT_NF_CM3) / METTLER_FIG417_30PCT_NF_CM3
    width = 0.35
    x = METTLER_R_CM
    ax.bar(x - width / 2, resid_90, width, color='tab:red', alpha=0.85,
           label='90% SF$_6$')
    ax.bar(x + width / 2, resid_30, width, color='tab:blue', alpha=0.85,
           label='30% SF$_6$')
    ax.axhspan(-10, 10, color='tab:green', alpha=0.15, label='$\\pm 10$% band')
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xlabel('$r$ (cm)')
    ax.set_ylabel('Model error vs Mettler (%)')
    ax.set_title('(c) Residuals at Mettler radii', fontweight='bold')
    ax.set_xticks(x)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y')

    fig.suptitle('Direct benchmark vs Mettler Fig 4.14 / 4.17 '
                 '(1000 W, 10 mTorr, bias off)',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'fig_mettler_fig417_overlay.png')
    pdf_path = os.path.join(out_dir, 'fig_mettler_fig417_overlay.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'  wrote {png_path}')
    print(f'  wrote {pdf_path}')

    # Dump numeric residuals for the log
    import json
    log = {
        'r_cm': METTLER_R_CM.tolist(),
        'mettler_90pct_cm3': METTLER_FIG417_90PCT_NF_CM3.tolist(),
        'model_90pct_cm3': F_model_90.tolist(),
        'residual_90pct_pct': resid_90.tolist(),
        'mettler_30pct_cm3': METTLER_FIG417_30PCT_NF_CM3.tolist(),
        'model_30pct_cm3': F_model_30.tolist(),
        'residual_30pct_pct': resid_30.tolist(),
    }
    with open(os.path.join(out_dir, 'fig_mettler_fig417_overlay_data.json'),
              'w') as f:
        json.dump(log, f, indent=2)
    print(f'  residuals 90% SF6: {[f"{v:+.1f}%" for v in resid_90]}')
    print(f'  residuals 30% SF6: {[f"{v:+.1f}%" for v in resid_30]}')


def main():
    results_base = os.path.abspath(os.path.join(STAGE5A_ROOT, 'results',
                                                'mettler_fig417'))
    nF_90_path = os.path.join(results_base, '90pct', 'nF.npy')
    nF_30_path = os.path.join(results_base, '30pct', 'nF.npy')

    if not os.path.exists(nF_90_path):
        sys.exit(f'Missing {nF_90_path}. Run run_mettler_fig417_90pct.py first.')
    if not os.path.exists(nF_30_path):
        sys.exit(f'Missing {nF_30_path}. Run run_mettler_fig417_30pct.py first.')

    mesh, inside = _build_mesh_and_inside()
    nF_90 = load_nF(os.path.join(results_base, '90pct'))
    nF_30 = load_nF(os.path.join(results_base, '30pct'))

    out_dir = os.path.abspath(os.path.join(STAGE5A_ROOT, 'figures'))
    make_figure(nF_90, nF_30, mesh, inside, out_dir)


if __name__ == '__main__':
    main()
