#!/usr/bin/env python3
"""
Generate consolidated species overlay plots matching Stage 10 style:
  1. All 9 species on one wafer radial plot (overlay)
  2. ICP source vs Processing region bar chart
  3. Sequential dissociation chain
  4. All species on one axial centreline plot (overlay)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}

SP_COLOR = {'SF6': '#1f77b4', 'SF5': '#2ca02c', 'SF4': '#ff7f0e',
            'SF3': '#9467bd', 'SF2': '#17becf', 'SF': '#e377c2',
            'S': '#7f7f7f', 'F': '#d62728', 'F2': '#8c564b'}

SP_MARKER = {'SF6': 'o', 'SF5': 's', 'SF4': '^', 'SF3': 'D',
             'SF2': 'v', 'SF': '<', 'S': '>', 'F': 'o', 'F2': 'p'}

SPECIES_ORDER = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 13, 'axes.titlesize': 14,
        'legend.fontsize': 9, 'figure.dpi': 200,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


def fig_all_species_radial_overlay(fields, mesh, inside, out_dir):
    """All 9 species on one wafer radial plot (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    r_mm = mesh.rc[inside[:, 0]] * 1e3

    for sp in SPECIES_ORDER:
        if sp not in fields:
            continue
        profile = fields[sp][:, 0][inside[:, 0]] * 1e-6  # cm^-3
        if np.any(profile > 0):
            ax.semilogy(r_mm, np.maximum(profile, 1e6),
                       f'-{SP_MARKER[sp]}', color=SP_COLOR[sp],
                       lw=2, ms=4, markevery=3, label=SP_LABEL[sp])

    ax.set_xlabel('$r$ (mm)')
    ax.set_ylabel('Density (cm$^{-3}$)')
    ax.set_title('All Neutral Species — Wafer Radial Profiles\n'
                 '700 W, 10 mTorr, pure SF$_6$')
    ax.legend(ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, r_mm[-1] + 2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_all_species_radial_overlay.png'))
    fig.savefig(os.path.join(out_dir, 'fig_all_species_radial_overlay.pdf'))
    plt.close(fig)
    print("  Fig: All species radial overlay")


def fig_all_species_axial_overlay(fields, mesh, inside, out_dir):
    """All 9 species on one axial centreline plot (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    z_mm = mesh.zc * 1e3

    for sp in SPECIES_ORDER:
        if sp not in fields:
            continue
        profile = fields[sp][0, :] * 1e-6  # axis profile, cm^-3
        if np.any(profile > 0):
            ax.semilogy(z_mm, np.maximum(profile, 1e6),
                       f'-{SP_MARKER[sp]}', color=SP_COLOR[sp],
                       lw=2, ms=4, markevery=5, label=SP_LABEL[sp])

    ax.set_xlabel('$z$ (mm)')
    ax.set_ylabel('Density (cm$^{-3}$)')
    ax.set_title('All Neutral Species — Axial Centreline Profiles ($r = 0$)\n'
                 '700 W, 10 mTorr, pure SF$_6$')
    ax.legend(ncol=3, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axvline(50, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(51, ax.get_ylim()[1] * 0.5, 'Aperture', fontsize=8, color='gray', rotation=90, va='center')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_all_species_axial_overlay.png'))
    fig.savefig(os.path.join(out_dir, 'fig_all_species_axial_overlay.pdf'))
    plt.close(fig)
    print("  Fig: All species axial overlay")


def fig_species_bar_chart(fields, mesh, inside, tel, out_dir):
    """ICP source vs Processing region bar chart for all 9 species."""
    R_icp = tel.get('R_icp', 0.038)
    L_proc = tel.get('L_proc', 0.050)
    L_apt = tel.get('L_apt', 0.002)
    z_apt_top = L_proc + L_apt
    Nr, Nz = mesh.Nr, mesh.Nz

    icp_mask = (inside &
                (np.outer(mesh.rc, np.ones(Nz)) <= R_icp) &
                (np.outer(np.ones(Nr), mesh.zc) >= z_apt_top))
    proc_mask = inside & (np.outer(np.ones(Nr), mesh.zc) < L_proc)

    fig, ax = plt.subplots(figsize=(12, 7))

    species_names = SPECIES_ORDER
    icp_avgs = []
    proc_avgs = []

    for sp in species_names:
        field = fields.get(sp, np.zeros((Nr, Nz)))
        icp_avg = np.sum(field[icp_mask] * mesh.vol[icp_mask]) / max(np.sum(mesh.vol[icp_mask]), 1e-30) * 1e-6
        proc_avg = np.sum(field[proc_mask] * mesh.vol[proc_mask]) / max(np.sum(mesh.vol[proc_mask]), 1e-30) * 1e-6
        icp_avgs.append(max(icp_avg, 1e6))
        proc_avgs.append(max(proc_avg, 1e6))

    x = np.arange(len(species_names))
    width = 0.35

    labels = [SP_LABEL[sp] for sp in species_names]

    bars1 = ax.bar(x - width/2, icp_avgs, width, label='ICP source', color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, proc_avgs, width, label='Processing', color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_ylabel('Volume-averaged density (cm$^{-3}$)')
    ax.set_title('9-Species Densities: ICP Source vs Processing Region\n'
                 '700 W, 10 mTorr, pure SF$_6$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_species_bar_chart.png'))
    fig.savefig(os.path.join(out_dir, 'fig_species_bar_chart.pdf'))
    plt.close(fig)
    print("  Fig: Species bar chart (ICP vs Processing)")


def fig_dissociation_chain(fields, mesh, inside, tel, out_dir):
    """Sequential dissociation chain: SF6 -> SF5 -> ... -> S."""
    R_icp = tel.get('R_icp', 0.038)
    L_proc = tel.get('L_proc', 0.050)
    L_apt = tel.get('L_apt', 0.002)
    z_apt_top = L_proc + L_apt
    Nr, Nz = mesh.Nr, mesh.Nz

    icp_mask = (inside &
                (np.outer(mesh.rc, np.ones(Nz)) <= R_icp) &
                (np.outer(np.ones(Nr), mesh.zc) >= z_apt_top))
    proc_mask = inside & (np.outer(np.ones(Nr), mesh.zc) < L_proc)

    chain = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S']

    fig, ax = plt.subplots(figsize=(10, 7))

    icp_vals = []
    proc_vals = []
    for sp in chain:
        field = fields.get(sp, np.zeros((Nr, Nz)))
        icp_avg = np.sum(field[icp_mask] * mesh.vol[icp_mask]) / max(np.sum(mesh.vol[icp_mask]), 1e-30) * 1e-6
        proc_avg = np.sum(field[proc_mask] * mesh.vol[proc_mask]) / max(np.sum(mesh.vol[proc_mask]), 1e-30) * 1e-6
        icp_vals.append(max(icp_avg, 1e6))
        proc_vals.append(max(proc_avg, 1e6))

    x = np.arange(len(chain))
    labels = [SP_LABEL[sp] for sp in chain]

    ax.semilogy(x, icp_vals, 'ro-', lw=2.5, ms=10, label='ICP source', zorder=5)
    ax.semilogy(x, proc_vals, 'bs-', lw=2.5, ms=10, label='Processing', zorder=5)

    # Annotate values
    for i, (iv, pv) in enumerate(zip(icp_vals, proc_vals)):
        ax.annotate(f'{iv:.1e}', (i, iv), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=7, color='red')
        ax.annotate(f'{pv:.1e}', (i, pv), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=7, color='blue')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Volume-averaged density (cm$^{-3}$)')
    ax.set_title('Sequential Dissociation Chain: SF$_6$ $\\to$ S\n'
                 '700 W, 10 mTorr, pure SF$_6$')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Draw arrows between species
    for i in range(len(chain) - 1):
        ax.annotate('', xy=(i+0.7, icp_vals[i+1]), xytext=(i+0.3, icp_vals[i]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.3, lw=1))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_dissociation_chain.png'))
    fig.savefig(os.path.join(out_dir, 'fig_dissociation_chain.pdf'))
    plt.close(fig)
    print("  Fig: Dissociation chain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    args = parser.parse_args()

    setup_style()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = SimulationConfig(config_path)
    tel = config.tel_geometry

    if args.run_dir:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    else:
        runs_base = os.path.join(PROJECT_ROOT, 'results', 'runs')
        runs = sorted([d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))])
        run_dir = os.path.join(runs_base, runs[-1])

    data_dir = os.path.join(run_dir, 'data')
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")

    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, bc_type = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                          tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)

    # Load species
    fields = {}
    for sp in SPECIES_ORDER:
        path = os.path.join(data_dir, f'n{sp}.npy')
        if os.path.exists(path):
            fields[sp] = np.load(path)

    print(f"Loaded {len(fields)} species\n")

    fig_all_species_radial_overlay(fields, mesh, inside, out_dir)
    fig_all_species_axial_overlay(fields, mesh, inside, out_dir)
    fig_species_bar_chart(fields, mesh, inside, tel, out_dir)
    fig_dissociation_chain(fields, mesh, inside, tel, out_dir)

    # Copy to presentation
    import shutil
    pres_dir = os.path.join(PROJECT_ROOT, 'docs', 'presentation', 'figures')
    os.makedirs(pres_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(out_dir, f), os.path.join(pres_dir, f))

    print("\nAll consolidated plots complete")


if __name__ == '__main__':
    main()
