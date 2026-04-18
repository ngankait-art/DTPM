#!/usr/bin/env python3
"""Regenerate the Phase-1 Global–2D figures from the new self-consistent runs.

Produces:
  docs/report/figures/fig_eta_sweep.{png,pdf}            — eta emergent vs P_rf
  docs/report/figures/fig_I_peak_sweep.{png,pdf}         — I_peak vs P_rf
  docs/report/figures/fig_composition_blind_test.{png,pdf}
                                                         — 30% vs 90% SF6 comparison
  docs/report/figures/fig_mettler_fig417_v2.{png,pdf}    — 3-panel absolute benchmark
  docs/report/figures/fig_radial_F_mettler_bias_on.{png,pdf}
                                                         — radial [F] at 1000W, bias on, both compositions

Reads the sweep directories produced by run_power_sweep_1000W_biased.py
and run_mettler_composition_pair.py.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

FIG_DIR = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')

POWER_SWEEP_DIR = os.path.join(PROJECT_ROOT, 'results', 'sweeps',
                               'power_1000W_biased')
COMP_DIR = os.path.join(PROJECT_ROOT, 'results', 'mettler_composition')


# Mettler Fig 4.14 cubic fit: y = 1.01032 - 0.01847 r^2 + 7.139e-4 r^3, r in cm
def mettler_cubic(r_cm):
    return 1.01032 - 0.01847 * r_cm**2 + 7.139e-4 * r_cm**3


# Mettler Fig 4.17 digitised (bias off, centre line cm^-3) from 5a work:
METTLER_F_90PCT_OFF_CM3 = np.array([2.5, 2.3, 1.6, 1.0, 0.6]) * 1e14
METTLER_F_30PCT_OFF_CM3 = np.array([0.60, 0.55, 0.42, 0.28, 0.20]) * 1e14
METTLER_F_90PCT_ON_CM3 = np.array([4.0, 3.6, 2.6, 1.7, 1.0]) * 1e14
METTLER_F_30PCT_ON_CM3 = np.array([1.3, 1.2, 0.75, 0.45, 0.32]) * 1e14
METTLER_R_CM = np.array([0.0, 2.0, 4.0, 6.0, 8.0])


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


def load_power_sweep_index():
    idx_path = os.path.join(POWER_SWEEP_DIR, 'index.json')
    if not os.path.exists(idx_path):
        raise SystemExit(f"Missing {idx_path}. Run run_power_sweep_1000W_biased.py first.")
    with open(idx_path) as f:
        return json.load(f)


def load_sweep_point(path):
    """Load nF and ne arrays + mesh reconstruction."""
    nF = np.load(os.path.join(path, 'nF.npy'))
    ne = np.load(os.path.join(path, 'ne.npy')) if os.path.exists(
        os.path.join(path, 'ne.npy')) else None
    with open(os.path.join(path, 'summary.json')) as f:
        meta = json.load(f)
    return nF, ne, meta


def _build_mesh_and_inside():
    from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask
    cfg_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    config = SimulationConfig(cfg_path)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total,
                  Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _ = build_geometry_mask(
        mesh, tel['R_icp'], tel['R_proc'],
        tel['L_proc'], tel['L_proc'] + tel['L_apt'], L_total,
    )
    return mesh, inside


def wafer_radial(nF, mesh, inside):
    r_cm = mesh.rc[inside[:, 0]] * 100.0
    F_w_cm3 = nF[:, 0][inside[:, 0]] * 1e-6
    return r_cm, F_w_cm3


# ─────────────────────────────────────────────────────────────────────
def gen_eta_sweep(power_index):
    powers = [row['power'] for row in power_index]
    etas = [row['eta_computed'] for row in power_index]
    Ipk = [row['I_peak_final'] for row in power_index]
    Rp = [row['R_plasma_final'] for row in power_index]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(powers, etas, 'o-', color='tab:red', lw=2.2, ms=8)
    ax.set_xlabel('$P_\\mathrm{rf}$ (W)')
    ax.set_ylabel('Coupling efficiency $\\eta$')
    ax.set_ylim(0, 1.0)
    ax.set_title('(a) Emergent $\\eta(P_\\mathrm{rf})$', fontweight='bold')
    ax.grid(True)

    ax = axes[1]
    ax.plot(powers, Ipk, 's-', color='tab:blue', lw=2.2, ms=8)
    ax.set_xlabel('$P_\\mathrm{rf}$ (W)')
    ax.set_ylabel('$I_\\mathrm{peak}$ (A)')
    ax.set_title('(b) Coil current', fontweight='bold')
    ax.grid(True)

    ax = axes[2]
    ax.plot(powers, Rp, '^-', color='tab:green', lw=2.2, ms=8)
    ax.set_xlabel('$P_\\mathrm{rf}$ (W)')
    ax.set_ylabel('$R_\\mathrm{plasma}$ ($\\Omega$)')
    ax.set_title('(c) Plasma loading', fontweight='bold')
    ax.grid(True)

    fig.suptitle(
        'Self-consistent ICP circuit vs $P_\\mathrm{rf}$ '
        '(10 mTorr, 70% SF$_6$, 200 W bias)',
        fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = os.path.join(FIG_DIR, 'fig_eta_sweep.png')
    out_pdf = os.path.join(FIG_DIR, 'fig_eta_sweep.pdf')
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f'  wrote {out_png}')


def gen_Fdrop_vs_power(power_index):
    powers = [row['power'] for row in power_index]
    Fd = [row['F_drop_pct'] for row in power_index]
    Fc = [row['nF_centre_wafer_cm3'] for row in power_index]
    ne_icp = [row['ne_avg_icp'] for row in power_index]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(powers, Fd, 'o-', color='tab:red', lw=2.2, ms=7)
    axes[0].axhline(74, color='black', linestyle='--', lw=1, label='Mettler 74% (90% SF$_6$)')
    axes[0].set_xlabel('$P_\\mathrm{rf}$ (W)')
    axes[0].set_ylabel('[F] centre-to-edge drop (%)')
    axes[0].set_ylim(50, 80)
    axes[0].legend()
    axes[0].set_title('(a) [F] drop vs power', fontweight='bold')
    axes[0].grid(True)

    axes[1].semilogy(powers, Fc, 's-', color='tab:blue', lw=2.2, ms=7)
    axes[1].set_xlabel('$P_\\mathrm{rf}$ (W)')
    axes[1].set_ylabel('[F] centre wafer (cm$^{-3}$)')
    axes[1].set_title('(b) Absolute [F] at wafer centre', fontweight='bold')
    axes[1].grid(True, which='both')

    axes[2].semilogy(powers, ne_icp, '^-', color='tab:green', lw=2.2, ms=7)
    axes[2].set_xlabel('$P_\\mathrm{rf}$ (W)')
    axes[2].set_ylabel('$n_e$ volume-averaged (cm$^{-3}$)')
    axes[2].set_title('(c) ICP $n_e$', fontweight='bold')
    axes[2].grid(True, which='both')

    fig.suptitle('Phase-1 Global–2D outputs vs power '
                 '(10 mTorr, 70% SF$_6$, 200 W bias)',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG_DIR, 'fig_Fdrop_vs_power.png')
    fig.savefig(out); fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig); print(f'  wrote {out}')


def gen_composition_blind_test():
    """30% vs 90% SF6 bias enhancement, blind test."""
    fp = os.path.join(COMP_DIR, 'composition_summary.json')
    if not os.path.exists(fp):
        print(f'  SKIP: {fp} missing. Run run_mettler_composition_pair.py first.')
        return
    with open(fp) as f:
        summary = json.load(f)

    labels = [row['label'].replace('pct_', '% ').replace('SF6', 'SF$_6$')
              for row in summary]
    model = [row['enhancement'] for row in summary]
    target = [row['mettler_target'] for row in summary]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, model, width, label='Model (this work)',
           color='tab:red', alpha=0.85)
    ax.bar(x + width/2, target, width, label='Mettler 2025',
           color='tab:blue', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Bias-on / bias-off [F] centre enhancement')
    ax.set_title('Wafer-bias [F] enhancement: model vs Mettler Fig 4.17 '
                 '(1000 W, 10 mTorr, 200 W bias)', fontweight='bold')
    ax.axhline(1, color='black', lw=1)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y')

    for i, (m, t) in enumerate(zip(model, target)):
        dev = 100 * (m / t - 1)
        ax.text(i, max(m, t) + 0.08,
                f"model ${m:.2f}$, dev ${dev:+.1f}\\%$",
                ha='center', fontsize=10)

    out = os.path.join(FIG_DIR, 'fig_composition_blind_test.png')
    fig.tight_layout()
    fig.savefig(out); fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig); print(f'  wrote {out}')


def gen_mettler_fig417_v2():
    """Three-panel update of the 5a figure with self-consistent + bias model."""
    mesh, inside = _build_mesh_and_inside()

    # Load sweep points for 1000 W 90% SF6 bias-on and 30% SF6 bias-on
    p90 = os.path.join(COMP_DIR, '90pct_SF6', 'bias_on')
    p30 = os.path.join(COMP_DIR, '30pct_SF6', 'bias_on')
    if not all(os.path.isfile(os.path.join(p, 'nF.npy')) for p in (p90, p30)):
        print('  SKIP fig_mettler_fig417_v2: composition pair data missing.')
        return

    nF_90 = np.load(os.path.join(p90, 'nF.npy'))
    nF_30 = np.load(os.path.join(p30, 'nF.npy'))
    r_cm_90, F_90 = wafer_radial(nF_90, mesh, inside)
    r_cm_30, F_30 = wafer_radial(nF_30, mesh, inside)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) normalised shape with cubic fit
    ax = axes[0]
    r_fit = np.linspace(0, 8, 200)
    ax.plot(r_fit, mettler_cubic(r_fit), '--', color='tab:green', lw=2.2,
            label='Mettler Fig 4.14 cubic fit')
    ax.plot(METTLER_R_CM, mettler_cubic(METTLER_R_CM), 'o', color='black',
            ms=10, mfc='white', mew=2.0, label='Mettler 5-pt data')
    if F_90[0] > 0:
        ax.plot(r_cm_90, F_90 / F_90[0], '-', color='tab:red', lw=2.5,
                label=f'Model 90% SF$_6$ ({(1 - F_90[-1]/F_90[0])*100:.0f}%)')
    if F_30[0] > 0:
        ax.plot(r_cm_30, F_30 / F_30[0], '-', color='tab:blue', lw=2.5,
                label=f'Model 30% SF$_6$ ({(1 - F_30[-1]/F_30[0])*100:.0f}%)')
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.15)
    ax.set_xlabel('$r$ (cm)'); ax.set_ylabel('[F] / [F]$_\\mathrm{centre}$')
    ax.set_title('(a) Normalised radial [F]', fontweight='bold')
    ax.legend(loc='lower left'); ax.grid(True)

    # (b) absolute with Mettler bias-on
    ax = axes[1]
    ax.plot(METTLER_R_CM, METTLER_F_90PCT_ON_CM3, 'o', color='tab:red',
            ms=10, mfc='white', mew=2.0, label='Mettler 90% SF$_6$ (bias on)')
    ax.plot(METTLER_R_CM, METTLER_F_30PCT_ON_CM3, 's', color='tab:blue',
            ms=10, mfc='white', mew=2.0, label='Mettler 30% SF$_6$ (bias on)')
    ax.plot(r_cm_90, F_90, '-', color='tab:red', lw=2.5, label='Model 90% SF$_6$')
    ax.plot(r_cm_30, F_30, '-', color='tab:blue', lw=2.5, label='Model 30% SF$_6$')
    ax.set_yscale('log'); ax.set_xlim(0, 10)
    ax.set_xlabel('$r$ (cm)'); ax.set_ylabel('[F] at wafer (cm$^{-3}$)')
    ax.set_title('(b) Absolute radial [F] vs Mettler bias-on', fontweight='bold')
    ax.legend(loc='lower left', fontsize=9); ax.grid(True, which='both')

    # (c) residuals
    ax = axes[2]
    F_model_90 = np.interp(METTLER_R_CM, r_cm_90, F_90)
    F_model_30 = np.interp(METTLER_R_CM, r_cm_30, F_30)
    resid_90 = 100.0 * (F_model_90 - METTLER_F_90PCT_ON_CM3) / METTLER_F_90PCT_ON_CM3
    resid_30 = 100.0 * (F_model_30 - METTLER_F_30PCT_ON_CM3) / METTLER_F_30PCT_ON_CM3
    width = 0.35
    ax.bar(METTLER_R_CM - width/2, resid_90, width, color='tab:red',
           alpha=0.85, label='90% SF$_6$')
    ax.bar(METTLER_R_CM + width/2, resid_30, width, color='tab:blue',
           alpha=0.85, label='30% SF$_6$')
    ax.axhspan(-10, 10, color='tab:green', alpha=0.15, label='$\\pm 10$% band')
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xlabel('$r$ (cm)'); ax.set_ylabel('Model error vs Mettler (%)')
    ax.set_title('(c) Residuals at Mettler radii', fontweight='bold')
    ax.set_xticks(METTLER_R_CM); ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y')

    fig.suptitle('Direct benchmark v2: Self-consistent Global–2D + m12 bias '
                 '(1000 W, 10 mTorr, 200 W bias)',
                 fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, 'fig_mettler_fig417_v2.png')
    fig.savefig(out); fig.savefig(out.replace('.png', '.pdf'))
    plt.close(fig); print(f'  wrote {out}')


def main():
    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'Output directory: {FIG_DIR}')
    power_index = load_power_sweep_index()
    print(f'Power sweep: {len(power_index)} points')
    gen_eta_sweep(power_index)
    gen_Fdrop_vs_power(power_index)
    gen_composition_blind_test()
    gen_mettler_fig417_v2()
    print('All figures generated.')


if __name__ == '__main__':
    main()
