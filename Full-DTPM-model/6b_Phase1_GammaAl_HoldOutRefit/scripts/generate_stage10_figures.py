#!/usr/bin/env python3
"""
Generate Stage 10-style comprehensive figures from parameter sweep data.

Produces ~20 additional figures matching Stage 10's format:
  - Power sweep 2D grids (neutrals + charged)
  - Pressure sweep 2D grids (neutrals + charged)
  - All species overlay (power + pressure)
  - Charged species overlay (power + pressure)
  - Individual species benchmarks (2D vs 0D)
  - Sensitivity analysis
  - Aperture zoom
  - Absolute [F] validation vs power

Requires: results/sweeps/ data from run_parameter_sweeps.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}
SP_COLOR = {'SF6': '#e41a1c', 'SF5': '#ff7f00', 'SF4': '#ffd700',
            'SF3': '#4daf4a', 'SF2': '#377eb8', 'SF': '#984ea3',
            'S': '#999999', 'F': '#e41a1c', 'F2': '#a65628'}
SPECIES_ORDER = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']


def setup_style():
    """Canonical figure style.  Single source of truth for font sizes and DPI.

    Designed to match the visual weight of the well-received fig_nF_contour /
    fig_nSF6_contour style: serif fonts, ≥12pt body text, ≥11pt tick labels so
    that labels remain readable after LaTeX scales a 16-inch figure to the
    ~6-inch printed text width.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 15,
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.2,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
    })


def interp_mirror(field, mesh, inside, Nr=200, Nz=280):
    f = field.copy().astype(float); f[~inside] = np.nan
    r_ext = np.concatenate([[0.0], mesh.rc * 1e3])
    f_ext = np.concatenate([f[0:1, :], f], axis=0)
    r_half = np.linspace(0, mesh.R * 1e3, Nr)
    z_fine = np.linspace(0, mesh.L * 1e3, Nz)
    interp = RegularGridInterpolator((r_ext, mesh.zc * 1e3), f_ext,
                                      method='linear', bounds_error=False, fill_value=np.nan)
    rr, zz = np.meshgrid(r_half, z_fine, indexing='ij')
    f_fine = interp(np.column_stack([rr.ravel(), zz.ravel()])).reshape(Nr, Nz)
    r_full = np.concatenate([-r_half[1:][::-1], r_half])
    f_full = np.concatenate([f_fine[::-1, :][1:, :], f_fine], axis=0)
    return f_full.T, r_full, z_fine


def load_sweep_data(sweep_dir, points, param_prefix, param_format):
    """Load sweep data from saved numpy files."""
    data = []
    for p in points:
        dirname = param_format.format(p)
        d = os.path.join(sweep_dir, dirname)
        if not os.path.exists(d):
            continue
        entry = {'value': p, 'fields': {}, 'ions': {}}
        for sp in SPECIES_ORDER:
            path = os.path.join(d, f'n{sp}.npy')
            if os.path.exists(path):
                entry['fields'][sp] = np.load(path)
        for name in ['ne', 'Te', 'P_rz', 'E_theta_rms']:
            path = os.path.join(d, f'{name}.npy')
            if os.path.exists(path):
                entry[name] = np.load(path)
        for ion in ['ne', 'n+', 'n-', 'alpha', 'SF5+', 'SF3+', 'SF4+', 'F+', 'F-', 'SF6-', 'SF5-', 'SF4-']:
            path = os.path.join(d, f'ion_{ion}.npy')
            if os.path.exists(path):
                entry['ions'][ion] = np.load(path)
        summary_path = os.path.join(d, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                entry['summary'] = json.load(f)
        data.append(entry)
    return data


def _pick_representative_indices(sweep_data, n_keep=5):
    """Select evenly spaced indices including endpoints, deterministically."""
    n = len(sweep_data)
    if n <= n_keep:
        return list(range(n))
    return [int(round(i * (n - 1) / (n_keep - 1))) for i in range(n_keep)]


def gen_neutral_sweep_2D(sweep_data, mesh, inside, tel, param_name, param_unit,
                         out_dir, filename, n_representative=5):
    """Publication-quality figure: 4 species x N representative points + full-range trend panel.

    Layout: 2D density maps for a subset of N (=n_representative) sweep points, followed by
    a wide trend panel spanning all columns that shows volume-averaged densities for EVERY
    sweep point (not just the representative ones). This keeps both visual detail per panel
    and the full sweep trend information.
    """
    show_species = ['F', 'SF6', 'SF5', 'SF4']
    if len(sweep_data) == 0:
        return

    rep_idx = _pick_representative_indices(sweep_data, n_representative)
    rep_data = [sweep_data[i] for i in rep_idx]
    n_cols = len(rep_data)
    n_rows = len(show_species)

    R_icp = tel['R_icp'] * 1e3
    R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    # Taller figure, tall trend panel to avoid squeezed curves.
    # Reactor bounding box is ~210 mm wide x 202 mm tall — each 2D panel ~3.0" x 4.0".
    # Trend panel height_ratio = 1.8 gives ~7.2" of vertical space for the
    # volume-averaged curves, enough for legible markers and legend.
    panel_w, panel_h = 3.0, 4.0
    fig_w = n_cols * panel_w + 1.0
    fig_h = n_rows * panel_h + 5.4  # room for 1.8-ratio trend panel + title
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(n_rows + 1, n_cols + 1,
                           width_ratios=[1] * n_cols + [0.05],
                           height_ratios=[1] * n_rows + [1.8],
                           hspace=0.22, wspace=0.08,
                           left=0.07, right=0.95, top=0.94, bottom=0.06)

    cmap = plt.cm.inferno.copy()

    # Compute per-species vmin/vmax globally across representative points for consistent colour scale
    species_ranges = {}
    for sp in show_species:
        all_vals = []
        for entry in rep_data:
            field = entry['fields'].get(sp, np.zeros((mesh.Nr, mesh.Nz))) * 1e-6
            mask = (field > 0) & inside
            if mask.any():
                all_vals.append(np.log10(field[mask]))
        if all_vals:
            concat = np.concatenate(all_vals)
            species_ranges[sp] = (np.nanpercentile(concat, 2), np.nanpercentile(concat, 99.5))
        else:
            species_ranges[sp] = (9.0, 15.0)

    # 2D maps
    for row, sp in enumerate(show_species):
        vmin, vmax = species_ranges[sp]
        norm = Normalize(vmin=vmin, vmax=vmax)
        last_im = None
        for col, entry in enumerate(rep_data):
            ax = fig.add_subplot(gs[row, col])
            val = entry['value']
            fields = entry['fields']
            nF = fields.get('F', np.zeros((mesh.Nr, mesh.Nz)))
            Fw = nF[:, 0][inside[:, 0]]
            drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0

            field = fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
            F_cm3 = field * 1e-6
            log_f = np.where((F_cm3 > 0) & inside, np.log10(F_cm3), np.nan)
            f_disp, r_mm, z_mm = interp_mirror(log_f, mesh, inside, 160, 220)

            rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
            rgba[np.isnan(f_disp), 3] = 0

            ax.set_facecolor('#ececec')
            last_im = ax.imshow(rgba, origin='lower', aspect='equal',
                     extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                     interpolation='bilinear')
            # reactor outline
            for s in [-1, 1]:
                ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='k', lw=0.8)
                ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='k', lw=0.8)
                ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='k', lw=0.8)
            # top cap + bottom cap
            ax.plot([-R_icp, R_icp], [z_top, z_top], color='k', lw=0.8)
            ax.plot([-R_proc, R_proc], [0, 0], color='k', lw=0.8)

            if row == 0:
                ax.set_title(f'{val} {param_unit}   (drop={drop:.0f}%)',
                             fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'[{SP_LABEL[sp]}]\n$z$ (mm)', fontsize=13, fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row == n_rows - 1:
                ax.set_xlabel('$r$ (mm)', fontsize=12)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=10)
            ax.set_xlim(-R_proc - 5, R_proc + 5)

        # colourbar for this species row
        cax = fig.add_subplot(gs[row, -1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('$\\log_{10}$ (cm$^{-3}$)', fontsize=11)
        cb.ax.tick_params(labelsize=10)

    # Trend panel (full width) — taller, cleaner, bigger fonts
    ax_b = fig.add_subplot(gs[-1, :-1])
    x_vals = [e['value'] for e in sweep_data]
    for sp in SPECIES_ORDER:
        avgs = []
        for entry in sweep_data:
            s = entry.get('summary', {})
            avgs.append(s.get(f'{sp}_icp', 1e6))
        if any(a > 1e6 for a in avgs):
            ax_b.semilogy(x_vals, avgs, '-o', color=SP_COLOR.get(sp, 'gray'),
                          lw=2.2, ms=8, label=SP_LABEL[sp])

    ax_b.set_xlabel(f'{param_name} ({param_unit})', fontsize=14)
    ax_b.set_ylabel('Volume-averaged density (cm$^{-3}$)', fontsize=14)
    ax_b.set_title(f'Volume-averaged neutral densities — all {len(sweep_data)} sweep points',
                   fontweight='bold', fontsize=14)
    ax_b.legend(ncol=5, fontsize=12, loc='lower right', frameon=True, framealpha=0.95)
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.tick_params(labelsize=12)

    fig.suptitle(f'Neutral species 2D distributions — {param_name} sweep',
                 fontsize=16, fontweight='bold', y=0.985)

    fig.savefig(os.path.join(out_dir, f'{filename}.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {filename}  ({fig_w:.1f}x{fig_h:.1f} in, {n_cols} rep points of {len(sweep_data)})")


def gen_charged_sweep_2D(sweep_data, mesh, inside, tel, param_name, param_unit,
                         out_dir, filename, n_representative=5):
    """Charged species 2D grid: ne, n+, n- x N representative points + trend panel.

    Uses the ionisation-source-driven electron density field (entry['ne']) and derives
    n+ and n- fields from alpha = n-/ne stored in the summary (spatially uniform scaling).
    """
    show_species = ['ne', 'n+', 'n-']
    labels = {'ne': '$n_e$', 'n+': '$n_+$', 'n-': '$n_-$'}
    colors = {'ne': '#1b9e77', 'n+': '#d95f02', 'n-': '#7570b3'}
    if len(sweep_data) == 0:
        return

    rep_idx = _pick_representative_indices(sweep_data, n_representative)
    rep_data = [sweep_data[i] for i in rep_idx]
    n_cols = len(rep_data)
    n_rows = len(show_species)

    R_icp = tel['R_icp'] * 1e3
    R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    panel_w, panel_h = 3.0, 4.0
    fig_w = n_cols * panel_w + 1.0
    fig_h = n_rows * panel_h + 5.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows + 1, n_cols + 1,
                           width_ratios=[1] * n_cols + [0.05],
                           height_ratios=[1] * n_rows + [1.8],
                           hspace=0.22, wspace=0.08,
                           left=0.07, right=0.95, top=0.94, bottom=0.06)

    cmap = plt.cm.viridis.copy()

    # Build ion fields per entry from ne and alpha (quasi-neutrality: n+ = ne*(1+alpha), n- = alpha*ne)
    def build_ion_field(entry, kind):
        ne = entry.get('ne', None)
        alpha = entry.get('summary', {}).get('0D_alpha', 0.0) or 0.0
        if ne is None:
            return np.zeros((mesh.Nr, mesh.Nz))
        if kind == 'ne':
            return ne
        if kind == 'n+':
            return ne * (1.0 + alpha)
        if kind == 'n-':
            return ne * alpha
        return np.zeros_like(ne)

    # global vmin/vmax per species row
    species_ranges = {}
    for sp in show_species:
        all_vals = []
        for entry in rep_data:
            fld = build_ion_field(entry, sp) * 1e-6
            mask = (fld > 0) & inside
            if mask.any():
                all_vals.append(np.log10(fld[mask]))
        if all_vals:
            concat = np.concatenate(all_vals)
            species_ranges[sp] = (np.nanpercentile(concat, 2), np.nanpercentile(concat, 99.5))
        else:
            species_ranges[sp] = (9.0, 12.0)

    for row, sp in enumerate(show_species):
        vmin, vmax = species_ranges[sp]
        norm = Normalize(vmin=vmin, vmax=vmax)
        for col, entry in enumerate(rep_data):
            ax = fig.add_subplot(gs[row, col])
            val = entry['value']
            fld = build_ion_field(entry, sp) * 1e-6
            log_f = np.where((fld > 0) & inside, np.log10(fld), np.nan)
            f_disp, r_mm, z_mm = interp_mirror(log_f, mesh, inside, 160, 220)

            rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
            rgba[np.isnan(f_disp), 3] = 0

            ax.set_facecolor('#ececec')
            ax.imshow(rgba, origin='lower', aspect='equal',
                     extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                     interpolation='bilinear')
            for s in [-1, 1]:
                ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='k', lw=0.8)
                ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='k', lw=0.8)
                ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='k', lw=0.8)
            ax.plot([-R_icp, R_icp], [z_top, z_top], color='k', lw=0.8)
            ax.plot([-R_proc, R_proc], [0, 0], color='k', lw=0.8)

            if row == 0:
                ax.set_title(f'{val} {param_unit}', fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{labels[sp]}\n$z$ (mm)', fontsize=13, fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row == n_rows - 1:
                ax.set_xlabel('$r$ (mm)', fontsize=12)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=10)
            ax.set_xlim(-R_proc - 5, R_proc + 5)

        cax = fig.add_subplot(gs[row, -1])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('$\\log_{10}$ (cm$^{-3}$)', fontsize=11)
        cb.ax.tick_params(labelsize=10)

    # Trend panel: ne, n+, n- volume-averaged + alpha on twin axis
    ax_b = fig.add_subplot(gs[-1, :-1])
    x_vals = [e['value'] for e in sweep_data]
    ne_vals = [e.get('summary', {}).get('ne_avg_icp', 0) for e in sweep_data]
    alpha_vals = [e.get('summary', {}).get('0D_alpha', 0) for e in sweep_data]
    np_vals = [n * (1 + a) for n, a in zip(ne_vals, alpha_vals)]
    nm_vals = [n * a for n, a in zip(ne_vals, alpha_vals)]

    ax_b.semilogy(x_vals, ne_vals, '-o', color=colors['ne'], lw=2.2, ms=8, label=labels['ne'])
    ax_b.semilogy(x_vals, np_vals, '-s', color=colors['n+'], lw=2.2, ms=8, label=labels['n+'])
    if any(v > 0 for v in nm_vals):
        ax_b.semilogy(x_vals, nm_vals, '-^', color=colors['n-'], lw=2.2, ms=8, label=labels['n-'])

    ax_b.set_xlabel(f'{param_name} ({param_unit})', fontsize=14)
    ax_b.set_ylabel('Volume-averaged density (cm$^{-3}$)', fontsize=14)
    ax_b.set_title(f'Charged-species volume averages — all {len(sweep_data)} sweep points',
                   fontweight='bold', fontsize=14)
    ax_b.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.95)
    ax_b.grid(True, alpha=0.3, which='both')
    ax_b.tick_params(labelsize=12)

    # twin axis for alpha
    ax_t = ax_b.twinx()
    ax_t.plot(x_vals, alpha_vals, '--d', color='k', lw=1.8, ms=7, label='$\\alpha$')
    ax_t.set_ylabel('Electronegativity $\\alpha = n_-/n_e$', fontsize=13)
    ax_t.tick_params(labelsize=12)
    ax_t.legend(loc='upper right', fontsize=12)

    fig.suptitle(f'Charged species 2D distributions — {param_name} sweep',
                 fontsize=16, fontweight='bold', y=0.985)

    fig.savefig(os.path.join(out_dir, f'{filename}.png'), dpi=180, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  {filename}  ({fig_w:.1f}x{fig_h:.1f} in, {n_cols} rep points of {len(sweep_data)})")


def gen_all_species_sweep(sweep_data, param_name, param_unit, out_dir, filename):
    """All 9 neutrals overlay plot: (a) sweep + (b) alternative."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_vals = [e['value'] for e in sweep_data]

    for sp in SPECIES_ORDER:
        avgs_icp = [e.get('summary', {}).get(f'{sp}_icp', 1e6) for e in sweep_data]
        avgs_0d = [e.get('summary', {}).get(f'0D_{sp}', 1e6) for e in sweep_data]

        if any(a > 1e6 for a in avgs_icp):
            ax1.semilogy(x_vals, avgs_icp, '-o', color=SP_COLOR.get(sp, 'gray'),
                        lw=2, ms=5, label=SP_LABEL[sp])

        if any(a > 1e6 for a in avgs_0d):
            ax2.semilogy(x_vals, avgs_0d, '--s', color=SP_COLOR.get(sp, 'gray'),
                        lw=1.5, ms=4, alpha=0.7, label=f'{SP_LABEL[sp]} (0D)')

    ax1.set_xlabel(f'{param_name} ({param_unit})')
    ax1.set_ylabel('ICP Volume-Averaged Density (cm$^{-3}$)')
    ax1.set_title(f'(a) 2D Coupled Model')
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(f'{param_name} ({param_unit})')
    ax2.set_ylabel('0D Lallement Density (cm$^{-3}$)')
    ax2.set_title(f'(b) 0D Reference Model')
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'All 9 Neutral Species — {param_name} Sweep', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{filename}.png'))
    fig.savefig(os.path.join(out_dir, f'{filename}.pdf'))
    plt.close(fig)
    print(f"  {filename}")


def gen_benchmark_neutrals(power_data, out_dir):
    """Benchmark: 2D coupled (solid) vs 0D Lallement (dashed) for all species."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    x_vals = [e['value'] for e in power_data]

    for idx, sp in enumerate(SPECIES_ORDER):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        icp_2d = [e.get('summary', {}).get(f'{sp}_icp', 1e6) for e in power_data]
        val_0d = [e.get('summary', {}).get(f'0D_{sp}', 1e6) for e in power_data]

        ax.semilogy(x_vals, icp_2d, '-o', color=SP_COLOR.get(sp, 'gray'),
                   lw=2.5, ms=6, label='2D coupled')
        ax.semilogy(x_vals, val_0d, '--s', color=SP_COLOR.get(sp, 'gray'),
                   lw=1.5, ms=5, alpha=0.6, label='0D Lallement')

        # Ratio annotation
        if len(icp_2d) > 0 and len(val_0d) > 0:
            i700 = x_vals.index(700) if 700 in x_vals else len(x_vals) // 2
            if icp_2d[i700] > 0 and val_0d[i700] > 0:
                ratio = icp_2d[i700] / val_0d[i700]
                ax.text(0.95, 0.95, f'700W: {ratio:.1f}$\\times$',
                       transform=ax.transAxes, ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(f'[{SP_LABEL[sp]}]', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 2:
            ax.set_xlabel('RF Power (W)')
        if col == 0:
            ax.set_ylabel('Density (cm$^{-3}$)')

    fig.suptitle('Individual Neutral Species: 2D Coupled vs 0D Lallement',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals_individual.png'))
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals_individual.pdf'))
    plt.close(fig)
    print("  fig_benchmark_neutrals_individual")


def gen_sensitivity(sensitivity_data, out_dir):
    """Sensitivity analysis: [F] drop response to parameter variation."""
    param_labels = {
        'gamma_Al': '$\\gamma_\\mathrm{Al}$',
        'eta': '$\\eta$ (coupling)',
        'pressure': 'Pressure (mTorr)',
        'gamma_wafer': '$\\gamma_\\mathrm{wafer}$',
    }
    param_colors = {'gamma_Al': '#2ca02c', 'eta': '#d62728',
                    'pressure': '#1f77b4', 'gamma_wafer': '#9467bd'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (param, results) in enumerate(sensitivity_data.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        x = [r['value'] for r in results]
        y = [r['F_drop'] for r in results]

        ax.plot(x, y, '-o', color=param_colors.get(param, 'gray'),
               lw=2.5, ms=8)
        ax.axhline(74, ls='--', color='red', lw=1.5, alpha=0.5, label='Mettler (74%)')

        label = param_labels.get(param, param)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('[F] Drop (%)')
        ax.set_title(f'({chr(97+idx)}) {label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Sensitivity Analysis: [F] Drop Response to Parameter Variation',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sensitivity.png'))
    fig.savefig(os.path.join(out_dir, 'fig_sensitivity.pdf'))
    plt.close(fig)
    print("  fig_sensitivity")


def gen_absolute_validation(power_data, out_dir):
    """Absolute [F] vs power with Mettler data overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x_vals = [e['value'] for e in power_data]
    F_icp = [e.get('summary', {}).get('F_icp', 0) for e in power_data]
    ne_icp = [e.get('summary', {}).get('ne_avg_icp', 0) for e in power_data]
    F_drop = [e.get('summary', {}).get('F_drop_pct', 0) for e in power_data]
    Te_vals = [e.get('summary', {}).get('0D_Te', 0) for e in power_data]

    # (a) Absolute [F] vs power
    axes[0, 0].semilogy(x_vals, F_icp, 'r-o', lw=2.5, ms=6, label='Model (ICP avg)')
    axes[0, 0].set_xlabel('RF Power (W)')
    axes[0, 0].set_ylabel('[F] (cm$^{-3}$)')
    axes[0, 0].set_title('(a) Absolute [F] vs Power')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) ne vs power
    axes[0, 1].semilogy(x_vals, ne_icp, 'b-o', lw=2.5, ms=6, label='$n_e$ (ICP avg)')
    axes[0, 1].set_xlabel('RF Power (W)')
    axes[0, 1].set_ylabel('$n_e$ (cm$^{-3}$)')
    axes[0, 1].set_title('(b) Electron Density vs Power')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (c) [F] drop vs power
    axes[1, 0].plot(x_vals, F_drop, 'g-o', lw=2.5, ms=6, label='Model')
    axes[1, 0].axhline(74, ls='--', color='red', lw=1.5, label='Mettler (74%)')
    axes[1, 0].set_xlabel('RF Power (W)')
    axes[1, 0].set_ylabel('[F] Drop (%)')
    axes[1, 0].set_title('(c) [F] Drop vs Power')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (d) Te vs power
    axes[1, 1].plot(x_vals, Te_vals, 'm-o', lw=2.5, ms=6, label='$T_e$ (0D)')
    axes[1, 1].set_xlabel('RF Power (W)')
    axes[1, 1].set_ylabel('$T_e$ (eV)')
    axes[1, 1].set_title('(d) Electron Temperature vs Power')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Absolute Validation and Power Dependence',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_absolute_validation.png'))
    fig.savefig(os.path.join(out_dir, 'fig_absolute_validation.pdf'))
    plt.close(fig)
    print("  fig_absolute_validation")


def gen_charged_sweep(power_data, pressure_data, out_dir):
    """All charged species: ne, n+, n-, alpha vs power and pressure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) ne, n+, n- vs power
    x_p = [e['value'] for e in power_data]
    for name, color, label in [('ne_avg_icp', 'green', '$n_e$'),
                                ('0D_ne', 'green', '$n_e$ (0D)')]:
        vals = [e.get('summary', {}).get(name, 0) for e in power_data]
        ls = '-o' if '0D' not in name else '--s'
        if any(v > 0 for v in vals):
            axes[0, 0].semilogy(x_p, vals, ls, color=color, lw=2, ms=5,
                               label=label, alpha=0.7 if '0D' in name else 1)
    axes[0, 0].set_xlabel('RF Power (W)')
    axes[0, 0].set_ylabel('Density (cm$^{-3}$)')
    axes[0, 0].set_title('(a) $n_e$ vs Power')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (b) Te vs power
    Te_2d = [e.get('summary', {}).get('Te_avg', 0) for e in power_data]
    Te_0d = [e.get('summary', {}).get('0D_Te', 0) for e in power_data]
    axes[0, 1].plot(x_p, Te_0d, 'r-o', lw=2.5, ms=6, label='$T_e$ (0D)')
    axes[0, 1].set_xlabel('RF Power (W)')
    axes[0, 1].set_ylabel('$T_e$ (eV)')
    axes[0, 1].set_title('(b) Electron Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (c) ne vs pressure
    if pressure_data:
        x_pr = [e['value'] for e in pressure_data]
        ne_pr = [e.get('summary', {}).get('ne_avg_icp', 0) for e in pressure_data]
        axes[1, 0].semilogy(x_pr, ne_pr, 'b-o', lw=2.5, ms=6, label='$n_e$ (ICP)')
        axes[1, 0].set_xlabel('Pressure (mTorr)')
        axes[1, 0].set_ylabel('$n_e$ (cm$^{-3}$)')
        axes[1, 0].set_title('(c) $n_e$ vs Pressure')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # (d) alpha vs power
    alpha_vals = [e.get('summary', {}).get('0D_alpha', 0) for e in power_data]
    axes[1, 1].plot(x_p, alpha_vals, 'k-o', lw=2.5, ms=6, label='$\\alpha$ (0D)')
    axes[1, 1].set_xlabel('RF Power (W)')
    axes[1, 1].set_ylabel('$\\alpha = n_-/n_e$')
    axes[1, 1].set_title('(d) Electronegativity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Charged Species Overview', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_all_charged_sweep.png'))
    fig.savefig(os.path.join(out_dir, 'fig_all_charged_sweep.pdf'))
    plt.close(fig)
    print("  fig_all_charged_sweep")


# ═══════════════════════════════════════════════════════════════════════════
# Stage-10-matching figures (radial [F], axial [F], benchmarks, Ar mixture)
# ═══════════════════════════════════════════════════════════════════════════

# Mettler (2025) radical-probe normalised [F]/[F]_centre at the wafer.
# Corrected provenance: Mettler Fig. 4.14 (not Fig. 4.5, which is his Helicon /
# PMIC data). Values below are evaluated from Mettler's published cubic fit
# y(r) = 1.01032 - 0.01847 r^2 + 7.139e-4 r^3  (r in cm), R^2 = 0.997.
# Conditions: P_ICP = 1000 W, p = 10 mTorr, 70 sccm SF6 / 30 sccm Ar, 200 W rf
# wafer bias (which the present Phase-1 model does not simulate).
METTLER_RADIAL_R_MM = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
METTLER_RADIAL_NORM = np.array([1.010, 0.942, 0.761, 0.500, 0.194])
METTLER_FDROP_PCT = 74.0  # 90% SF6 / bias-off branch of Fig. 4.17


def mettler_fig414_cubic(r_cm):
    """Mettler Fig 4.14 published cubic fit. r in cm; returns normalised y."""
    return 1.01032 - 0.01847 * r_cm**2 + 7.139e-4 * r_cm**3


# Mettler Fig 4.17 — absolute [F] (units: 1e20 /m^3 = 1e14 cm^-3)
# Top panel, wafer BIAS OFF, 90% SF6 (90 sccm SF6 / 10 sccm Ar).
METTLER_F417_90PCT_R_CM = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
METTLER_F417_90PCT_NF_CM3 = np.array([2.5, 2.3, 1.6, 1.0, 0.6]) * 1e14
# 30% SF6 (30 sccm SF6 / 70 sccm Ar).
METTLER_F417_30PCT_NF_CM3 = np.array([0.60, 0.55, 0.42, 0.28, 0.20]) * 1e14


def gen_radial_F_wafer_mettler(power_data, mesh, inside, out_dir):
    """Two-panel: absolute [F] at wafer + normalised with Mettler overlay."""
    ref = next((e for e in power_data if e['value'] == 700), None)
    if ref is None or 'F' not in ref['fields']:
        return
    nF = ref['fields']['F']
    r_mm = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6  # cm^-3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(r_mm, F_w / 1e14, 'r-', lw=2.5, label='2D hybrid model')
    ax1.set_xlabel('$r$ (mm)', fontsize=11)
    ax1.set_ylabel('[F] at wafer  ($\\times 10^{14}$ cm$^{-3}$)', fontsize=11)
    ax1.set_title('(a) [F] at wafer', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    if len(F_w) > 0 and F_w[0] > 0:
        F_norm = F_w / F_w[0]
        drop_pct = (1 - F_norm[-1]) * 100
        # Mettler cubic fit across the wafer (0 to 8 cm = 80 mm)
        r_fit_cm = np.linspace(0, 8, 100)
        y_fit = mettler_fig414_cubic(r_fit_cm)
        ax2.plot(r_mm / r_mm[-1], F_norm, 'r-', lw=2.5,
                 label=f'Model 700 W ({drop_pct:.0f}%)')
        ax2.plot(r_fit_cm / 10.0, y_fit, '--', color='tab:green', lw=2.0,
                 label='Mettler Fig 4.14 cubic fit')
        ax2.plot(METTLER_RADIAL_R_MM / 100.0, METTLER_RADIAL_NORM,
                 'o', color='black', ms=9, mfc='white', mew=1.8,
                 label='Mettler probe data (1000 W, 90% SF$_6$, bias on)')
        ax2.set_xlabel('$r / R$', fontsize=11)
        ax2.set_ylabel('[F] / [F]$_{\\mathrm{centre}}$', fontsize=11)
        ax2.set_title('(b) Normalised', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.12)
        ax2.legend(fontsize=9, loc='lower left')
        ax2.grid(True, alpha=0.3)

    fig.suptitle('Normalised radial [F] at wafer — model (700 W) vs Mettler Fig 4.14 (1000 W)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_radial_F_wafer_mettler.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_radial_F_wafer_mettler.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_radial_F_wafer_mettler")


def gen_axial_F_profile(power_data, mesh, inside, tel, out_dir):
    """Axial [F] at r=0 through the reactor with aperture + wafer markers."""
    ref = next((e for e in power_data if e['value'] == 700), None)
    if ref is None or 'F' not in ref['fields']:
        return
    nF = ref['fields']['F']
    j_axis = 0  # first radial cell = axis
    z_all = mesh.zc * 1e3
    valid = inside[j_axis, :]
    z_mm = z_all[valid]
    F_axis = nF[j_axis, :][valid] * 1e-6

    z_apt_low = tel['L_proc'] * 1e3
    z_apt_high = (tel['L_proc'] + tel['L_apt']) * 1e3
    z_wafer = 0.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.semilogx(F_axis, z_mm, 'r-', lw=2.5)
    ax.axhspan(z_apt_low, z_apt_high, color='lightgray', alpha=0.6, label='Aperture')
    ax.axhline(z_wafer, ls='--', color='k', lw=1.0, label='Wafer')
    ax.set_xlabel('[F]  (cm$^{-3}$)', fontsize=11)
    ax.set_ylabel('$z$ (mm)', fontsize=11)
    ax.set_title('Axial [F] at $r = 0$ (700 W, 10 mTorr)', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    # Preserve cylindrical axis orientation (wafer at bottom)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_axial_F_profile.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_axial_F_profile.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_axial_F_profile")


def gen_absolute_validation_4panel(power_data, out_dir):
    """4-panel validation matching Stage 10 Fig 7.

    (a) Absolute [F] — Mettler Fig 4.17 (1000 W) overlay
    (b) Source-sink balance (normalised to 1000 W — Mettler operating point)
    (c) n_e and SF_6 depletion vs power (twin-axis)
    (d) [F] drop vs power — Mettler reference line
    """
    if not power_data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = [e['value'] for e in power_data]
    F_icp = [e.get('summary', {}).get('F_icp', 0) for e in power_data]
    ne_icp = [e.get('summary', {}).get('ne_avg_icp', 0) for e in power_data]
    SF6_icp = [e.get('summary', {}).get('SF6_icp', 0) for e in power_data]
    F_drop = [e.get('summary', {}).get('F_drop_pct', 0) for e in power_data]
    SF6_0 = 3.24e14 * 10 / 10  # baseline SF6 at 10 mTorr, 313 K ~ 3.24e14 cm-3
    sf6_depl = [(1 - s / SF6_0) * 100 if s > 0 else 0 for s in SF6_icp]

    # Mettler Fig 4.17 reference values at 1000 W / 10 mTorr / 200 W bias
    # (wafer-centre [F], probe-direct, Mettler 2025 Ch 4.3.2):
    METTLER_1000W_90 = 3.774e14   # 90 % SF6, bias on
    METTLER_1000W_30 = 1.297e14   # 30 % SF6, bias on

    # (a) Absolute [F] — 1000 W Mettler anchor
    axes[0, 0].semilogy(x, F_icp, 'g-o', lw=2.5, ms=7, label='Model — ICP avg')
    axes[0, 0].plot(1000, METTLER_1000W_90, '^', ms=12, color='red',
                    label=f'Mettler Fig 4.17 (1000 W, 90 % SF$_6$): {METTLER_1000W_90:.2e}')
    axes[0, 0].plot(1000, METTLER_1000W_30, 'v', ms=12, color='royalblue',
                    label=f'Mettler Fig 4.17 (1000 W, 30 % SF$_6$): {METTLER_1000W_30:.2e}')
    axes[0, 0].set_xlabel('RF Power (W)')
    axes[0, 0].set_ylabel('[F] (cm$^{-3}$)')
    axes[0, 0].set_title('(a) Absolute [F] — Mettler Fig 4.17 anchor at 1000 W',
                         fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 0].grid(True, alpha=0.3, which='both')

    # (b) Source-sink balance (normalised to 1000 W, Mettler operating point)
    i_ref = x.index(1000) if 1000 in x else len(x) // 2
    P_ref = x[i_ref]
    src_norm = [ne_icp[i] / max(ne_icp[i_ref], 1) for i in range(len(x))]
    F_norm = [F_icp[i] / max(F_icp[i_ref], 1) for i in range(len(x))]
    axes[0, 1].plot(x, src_norm, 'k-o', lw=2.5, ms=7,
                    label=f'$n_e / n_{{e,{P_ref}}}$')
    axes[0, 1].plot(x, F_norm, 'r-s', lw=2.5, ms=7,
                    label=f'[F] / [F]$_{{{P_ref}}}$')
    axes[0, 1].axhline(1.0, ls=':', color='gray', lw=1)
    axes[0, 1].set_xlabel('RF Power (W)')
    axes[0, 1].set_ylabel(f'Normalised to {P_ref} W')
    axes[0, 1].set_title('(b) Source-sink balance (ref: Mettler 1000 W)',
                         fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # (c) n_e and SF_6 depletion (twin-axis)
    ax_c = axes[1, 0]
    ax_c.semilogy(x, ne_icp, 'b-o', lw=2.5, ms=7, label='$n_e$')
    ax_c.set_xlabel('RF Power (W)')
    ax_c.set_ylabel('$n_e$ (cm$^{-3}$)', color='blue')
    ax_c.tick_params(axis='y', labelcolor='blue')
    ax_c.grid(True, alpha=0.3, which='both')
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x, sf6_depl, 'r-s', lw=2.5, ms=7, label='SF$_6$ depletion')
    ax_c2.set_ylabel('SF$_6$ depletion (%)', color='red')
    ax_c2.tick_params(axis='y', labelcolor='red')
    ax_c.set_title('(c) $n_e$ and SF$_6$ depletion', fontsize=11, fontweight='bold')

    # (d) [F] drop vs power.  Mettler's F-drop is composition-dependent
    # across 67--75 % (Ch. 4.3.2, explicit range), NOT a single 74 % value:
    # 90 % SF6 bias-on  74.8%;  90 % SF6 bias-off 74.5%;
    # 30 % SF6 bias-on  75.0%;  30 % SF6 bias-off 67.2%  (all at 1000 W).
    # Show the full composition-dependent band and mark each of the four
    # direct-probe anchors individually so the single-flat-line error that
    # appeared in prior revisions is not repeated.
    METTLER_FDROP_BAND = (67.2, 75.0)  # (min, max) across Mettler's 4 conds
    METTLER_FDROP_POINTS = [
        (1000, 74.8, '90% SF$_6$ on'),
        (1000, 74.5, '90% SF$_6$ off'),
        (1000, 75.0, '30% SF$_6$ on'),
        (1000, 67.2, '30% SF$_6$ off'),
    ]
    axes[1, 1].axhspan(*METTLER_FDROP_BAND, color='tab:green', alpha=0.18,
                       label=f'Mettler range ({METTLER_FDROP_BAND[0]:.1f}'
                             f'--{METTLER_FDROP_BAND[1]:.1f}\\%, comp.-dep.)')
    markers = ['^', 'v', 's', 'D']
    for (p, d, lab), m in zip(METTLER_FDROP_POINTS, markers):
        axes[1, 1].plot(p, d, m, color='darkgreen', ms=9, mew=1.5, mfc='white',
                        label=f'Mettler 1000 W: {lab} ({d:.1f}\\%)')
    axes[1, 1].plot(x, F_drop, 'r-o', lw=2.5, ms=7, label='Model (fixed-BC, $\\gamma_\\mathrm{Al}=0.18$)')
    axes[1, 1].set_xlabel('RF Power (W)')
    axes[1, 1].set_ylabel('[F] drop (%)')
    axes[1, 1].set_title('(d) [F] drop vs power — Mettler comp.-dep. range',
                         fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=7, loc='lower right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(55, 85)

    fig.suptitle('Absolute [F] validation + source-sink analysis',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_absolute_validation_4panel.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_absolute_validation_4panel.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_absolute_validation_4panel")


def gen_neutral_benchmark_2panel(power_data, pressure_data, out_dir):
    """Two-panel: (a) power sweep @ 10 mTorr, (b) pressure sweep @ 700 W.

    All 9 species, solid+filled-marker = 2D TEL model, dashed+open = 0D Lallement.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def _panel(ax, data, xlabel):
        x = [e['value'] for e in data]
        for sp in SPECIES_ORDER:
            y2 = [e.get('summary', {}).get(f'{sp}_icp', 0) for e in data]
            y0 = [e.get('summary', {}).get(f'0D_{sp}', 0) for e in data]
            color = SP_COLOR.get(sp, 'gray')
            if any(v > 1e6 for v in y2):
                ax.semilogy(x, y2, '-s', color=color, lw=2.0, ms=6,
                            mfc=color, mec=color, label=f'{SP_LABEL[sp]}')
            if any(v > 1e6 for v in y0):
                ax.semilogy(x, y0, '--o', color=color, lw=1.2, ms=5,
                            mfc='none', mec=color, alpha=0.8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('ICP volume-averaged density (cm$^{-3}$)', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

    _panel(ax1, power_data, 'RF Power (W)')
    ax1.set_title('(a) Power sweep — 10 mTorr', fontsize=12, fontweight='bold')

    if pressure_data:
        _panel(ax2, pressure_data, 'Pressure (mTorr)')
        ax2.set_title('(b) Pressure sweep — 700 W', fontsize=12, fontweight='bold')

    # Shared legend: species colours + solid/dashed meaning
    handles = []
    from matplotlib.lines import Line2D
    for sp in SPECIES_ORDER:
        handles.append(Line2D([0], [0], color=SP_COLOR.get(sp, 'gray'),
                              lw=2, marker='s', label=SP_LABEL[sp]))
    handles.append(Line2D([0], [0], color='k', lw=2, marker='s', label='2D TEL model'))
    handles.append(Line2D([0], [0], color='k', lw=1.2, ls='--',
                          marker='o', mfc='none', label='0D Lallement'))
    ax2.legend(handles=handles, fontsize=8, ncol=2, loc='upper right')

    fig.suptitle('Neutral species benchmark: 2D TEL model vs 0D Lallement (TEL geometry)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_neutrals  (2-panel)")


def gen_charged_benchmark_4panel(power_data, pressure_data, out_dir):
    """4-panel charged benchmark matching Stage 10 Fig 17.

    (a) n_e, n+, n- vs power
    (b) n_e, n+, n- vs pressure
    (c) T_e vs power (2D and 0D)
    (d) Electronegativity alpha vs power and pressure
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def _charged_panel(ax, data, xlabel):
        x = [e['value'] for e in data]
        ne = [e.get('summary', {}).get('ne_avg_icp', 0) for e in data]
        a = [e.get('summary', {}).get('0D_alpha', 0) for e in data]
        np_ = [n * (1 + aa) for n, aa in zip(ne, a)]
        nm_ = [n * aa for n, aa in zip(ne, a)]
        ne0 = [e.get('summary', {}).get('0D_ne', 0) for e in data]
        np0 = [n * (1 + aa) for n, aa in zip(ne0, a)]
        nm0 = [n * aa for n, aa in zip(ne0, a)]

        ax.semilogy(x, ne, '-s', color='red', lw=2, ms=6, label='$n_e$ (2D)')
        ax.semilogy(x, ne0, '--o', color='red', lw=1.2, ms=5, mfc='none', label='$n_e$ (0D)')
        ax.semilogy(x, np_, '-s', color='green', lw=2, ms=6, label='$n_+$ (2D)')
        ax.semilogy(x, np0, '--o', color='green', lw=1.2, ms=5, mfc='none', label='$n_+$ (0D)')
        if any(v > 1e6 for v in nm_):
            ax.semilogy(x, nm_, '-s', color='royalblue', lw=2, ms=6, label='$n_-$ (2D)')
            ax.semilogy(x, nm0, '--o', color='royalblue', lw=1.2, ms=5, mfc='none', label='$n_-$ (0D)')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density (cm$^{-3}$)', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

    _charged_panel(axes[0, 0], power_data, 'RF Power (W)')
    axes[0, 0].set_title('(a) $n_e$, $n_+$, $n_-$ vs Power', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8, ncol=2)
    if pressure_data:
        _charged_panel(axes[0, 1], pressure_data, 'Pressure (mTorr)')
        axes[0, 1].set_title('(b) $n_e$, $n_+$, $n_-$ vs Pressure', fontsize=11, fontweight='bold')
        axes[0, 1].legend(fontsize=8, ncol=2)

    # (c) T_e
    xp = [e['value'] for e in power_data]
    Te_2d = [e.get('summary', {}).get('Te_avg', 0) for e in power_data]
    Te_0d = [e.get('summary', {}).get('0D_Te', 0) for e in power_data]
    axes[1, 0].plot(xp, Te_2d, '-s', color='darkblue', lw=2.5, ms=7, label='$T_e$ (2D) vs Power')
    axes[1, 0].plot(xp, Te_0d, '--o', color='darkblue', lw=1.5, ms=5, mfc='none',
                    label='$T_e$ (0D) vs Power')
    if pressure_data:
        xpr = [e['value'] for e in pressure_data]
        Te_2d_p = [e.get('summary', {}).get('Te_avg', 0) for e in pressure_data]
        Te_0d_p = [e.get('summary', {}).get('0D_Te', 0) for e in pressure_data]
        ax_t = axes[1, 0].twiny()
        ax_t.plot(xpr, Te_2d_p, '-s', color='lightblue', lw=2.5, ms=7,
                  label='$T_e$ (2D) vs Pressure')
        ax_t.plot(xpr, Te_0d_p, '--o', color='lightblue', lw=1.5, ms=5, mfc='none',
                  label='$T_e$ (0D) vs Pressure')
        ax_t.set_xlabel('Pressure (mTorr)', fontsize=10, color='lightblue')
        ax_t.tick_params(axis='x', colors='lightblue')
        ax_t.legend(loc='upper right', fontsize=8)
    axes[1, 0].set_xlabel('RF Power (W)', fontsize=11)
    axes[1, 0].set_ylabel('$T_e$ (eV)', fontsize=11)
    axes[1, 0].set_title('(c) Electron temperature', fontsize=11, fontweight='bold')
    axes[1, 0].legend(loc='upper left', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # (d) alpha
    a_pow_0d = [e.get('summary', {}).get('0D_alpha', 0) for e in power_data]
    a_pow_2d = a_pow_0d  # we derive alpha from same quasi-neutrality balance
    axes[1, 1].plot(xp, a_pow_2d, '-s', color='red', lw=2.5, ms=7, label='$\\alpha$ (2D) vs Power')
    axes[1, 1].plot(xp, a_pow_0d, '--o', color='red', lw=1.5, ms=5, mfc='none',
                    label='$\\alpha$ (0D) vs Power')
    if pressure_data:
        a_pr_0d = [e.get('summary', {}).get('0D_alpha', 0) for e in pressure_data]
        ax_t2 = axes[1, 1].twiny()
        ax_t2.plot(xpr, a_pr_0d, '-s', color='gray', lw=2.5, ms=7,
                   label='$\\alpha$ (2D) vs Pressure')
        ax_t2.plot(xpr, a_pr_0d, '--o', color='gray', lw=1.5, ms=5, mfc='none',
                   label='$\\alpha$ (0D) vs Pressure')
        ax_t2.set_xlabel('Pressure (mTorr)', fontsize=10, color='gray')
        ax_t2.tick_params(axis='x', colors='gray')
        ax_t2.legend(loc='lower right', fontsize=8)
    axes[1, 1].set_xlabel('RF Power (W)', fontsize=11)
    axes[1, 1].set_ylabel('$\\alpha = n_- / n_e$', fontsize=11)
    axes[1, 1].set_title('(d) Electronegativity $\\alpha$', fontsize=11, fontweight='bold')
    axes[1, 1].legend(loc='upper left', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Charged species benchmark: 2D TEL model vs 0D Lallement (TEL geometry)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_charged.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_charged.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_charged  (4-panel)")


def gen_individual_neutrals_with_insets(power_data, pressure_data, out_dir):
    """3x3 individual neutral benchmark with pressure inset at 700 W + ratio annotation."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    xp = [e['value'] for e in power_data]

    for idx, sp in enumerate(SPECIES_ORDER):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        y2 = [e.get('summary', {}).get(f'{sp}_icp', 0) for e in power_data]
        y0 = [e.get('summary', {}).get(f'0D_{sp}', 0) for e in power_data]
        color = SP_COLOR.get(sp, 'gray')
        ax.semilogy(xp, y2, '-s', color=color, lw=2.2, ms=7, label='2D TEL')
        ax.semilogy(xp, y0, '--o', color=color, lw=1.3, ms=5, mfc='none',
                    alpha=0.85, label='0D Lallement')

        # Ratio at 700 W
        if 700 in xp:
            i = xp.index(700)
            ratio = y2[i] / max(y0[i], 1) if y0[i] > 0 else 0
        else:
            ratio = 0
        ax.set_title(f'[{SP_LABEL[sp]}] — 700 W: {ratio:.1f}$\\times$',
                     fontsize=11, fontweight='bold')

        if row == 2:
            ax.set_xlabel('RF Power (W)', fontsize=10)
        if col == 0:
            ax.set_ylabel('Density (cm$^{-3}$)', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='upper left')

        # Pressure-sweep inset — lifted to y0=0.16 so its x-axis clears the main plot's;
        # xlabel dropped in favour of a self-describing title.
        if pressure_data:
            axin = ax.inset_axes([0.54, 0.16, 0.43, 0.36])
            xpr = [e['value'] for e in pressure_data]
            y2p = [e.get('summary', {}).get(f'{sp}_icp', 0) for e in pressure_data]
            y0p = [e.get('summary', {}).get(f'0D_{sp}', 0) for e in pressure_data]
            axin.semilogy(xpr, y2p, '-s', color=color, lw=1.5, ms=4)
            axin.semilogy(xpr, y0p, '--o', color=color, lw=1, ms=3, mfc='none', alpha=0.8)
            axin.set_title('p (mTorr) — 700 W', fontsize=8)
            axin.tick_params(labelsize=7)
            axin.grid(True, alpha=0.3, which='both')

    fig.suptitle('Individual neutral species: 2D TEL (solid) vs 0D Lallement (dashed)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals_individual.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_neutrals_individual.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_neutrals_individual  (3x3 with insets)")


def gen_individual_charged_with_ratios(power_data, pressure_data, out_dir):
    """Individual charged benchmark (n_e, n+, n-, T_e, alpha) with ratio table.

    Each of the 5 quantity panels has a pressure-at-700-W inset, mirroring the
    individual-neutral figure so the two benchmarks read as a matched pair.
    """
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    xp = [e['value'] for e in power_data]
    xpr = [e['value'] for e in pressure_data] if pressure_data else None

    ne2 = [e.get('summary', {}).get('ne_avg_icp', 0) for e in power_data]
    ne0 = [e.get('summary', {}).get('0D_ne', 0) for e in power_data]
    a = [e.get('summary', {}).get('0D_alpha', 0) for e in power_data]
    np2 = [n * (1 + aa) for n, aa in zip(ne2, a)]
    np0 = [n * (1 + aa) for n, aa in zip(ne0, a)]
    nm2 = [n * aa for n, aa in zip(ne2, a)]
    nm0 = [n * aa for n, aa in zip(ne0, a)]

    # Pressure-sweep versions for the insets
    if pressure_data:
        ne2p = [e.get('summary', {}).get('ne_avg_icp', 0) for e in pressure_data]
        ne0p = [e.get('summary', {}).get('0D_ne', 0) for e in pressure_data]
        ap   = [e.get('summary', {}).get('0D_alpha', 0) for e in pressure_data]
        np2p = [n * (1 + aa) for n, aa in zip(ne2p, ap)]
        np0p = [n * (1 + aa) for n, aa in zip(ne0p, ap)]
        nm2p = [n * aa for n, aa in zip(ne2p, ap)]
        nm0p = [n * aa for n, aa in zip(ne0p, ap)]

    i700 = xp.index(700) if 700 in xp else len(xp) // 2

    def _add_inset(ax, x, y2, y0, color):
        if not pressure_data:
            return
        axin = ax.inset_axes([0.54, 0.16, 0.43, 0.36])
        axin.semilogy(x, y2, '-s', color=color, lw=1.5, ms=4)
        axin.semilogy(x, y0, '--o', color=color, lw=1, ms=3, mfc='none', alpha=0.8)
        axin.set_title('p (mTorr) — 700 W', fontsize=8)
        axin.tick_params(labelsize=7)
        axin.grid(True, alpha=0.3, which='both')

    def _panel(ax, x, y2, y0, color, title, ylabel, x_inset=None, y2_inset=None, y0_inset=None):
        ax.semilogy(x, y2, '-s', color=color, lw=2.2, ms=7, label='2D')
        ax.semilogy(x, y0, '--o', color=color, lw=1.3, ms=5, mfc='none',
                    alpha=0.85, label='0D')
        if y0[i700] > 0:
            ratio = y2[i700] / y0[i700]
        else:
            ratio = 0
        ax.set_title(f'{title} — 700 W: {ratio:.1f}$\\times$',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('RF Power (W)', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='upper left')
        if x_inset is not None:
            _add_inset(ax, x_inset, y2_inset, y0_inset, color)
        return ratio

    # Row 0: ne, n+, n-
    r_ne = _panel(fig.add_subplot(gs[0, 0]), xp, ne2, ne0, 'red', '$n_e$', '$n_e$ (cm$^{-3}$)',
                  xpr, ne2p if pressure_data else None, ne0p if pressure_data else None)
    r_np = _panel(fig.add_subplot(gs[0, 1]), xp, np2, np0, 'green', '$n_+$', '$n_+$ (cm$^{-3}$)',
                  xpr, np2p if pressure_data else None, np0p if pressure_data else None)
    r_nm = _panel(fig.add_subplot(gs[0, 2]), xp, nm2, nm0, 'royalblue', '$n_-$', '$n_-$ (cm$^{-3}$)',
                  xpr, nm2p if pressure_data else None, nm0p if pressure_data else None)

    # T_e (linear, not semilogy)
    ax_t = fig.add_subplot(gs[1, 0])
    Te_2d = [e.get('summary', {}).get('Te_avg', 0) for e in power_data]
    Te_0d = [e.get('summary', {}).get('0D_Te', 0) for e in power_data]
    ax_t.plot(xp, Te_2d, '-s', color='darkblue', lw=2.2, ms=7, label='2D')
    ax_t.plot(xp, Te_0d, '--o', color='darkblue', lw=1.3, ms=5, mfc='none', label='0D')
    dT = Te_2d[i700] - Te_0d[i700]
    ax_t.set_title(f'$T_e$ — $\\Delta$: {dT:+.2f} eV', fontsize=12, fontweight='bold')
    ax_t.set_xlabel('RF Power (W)', fontsize=11)
    ax_t.set_ylabel('$T_e$ (eV)', fontsize=12)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=10, loc='upper right')
    if pressure_data:
        Te_2d_p = [e.get('summary', {}).get('Te_avg', 0) for e in pressure_data]
        Te_0d_p = [e.get('summary', {}).get('0D_Te', 0) for e in pressure_data]
        axin = ax_t.inset_axes([0.54, 0.16, 0.43, 0.36])
        axin.plot(xpr, Te_2d_p, '-s', color='darkblue', lw=1.5, ms=4)
        axin.plot(xpr, Te_0d_p, '--o', color='darkblue', lw=1, ms=3, mfc='none', alpha=0.8)
        axin.set_title('p (mTorr) — 700 W', fontsize=8)
        axin.tick_params(labelsize=7)
        axin.grid(True, alpha=0.3)

    # alpha (linear) — use same value for 2D and 0D (alpha is from 0D closure)
    ax_a = fig.add_subplot(gs[1, 1])
    ax_a.plot(xp, a, '-s', color='red', lw=2.2, ms=7, label='2D')
    ax_a.plot(xp, a, '--o', color='red', lw=1.3, ms=5, mfc='none', label='0D')
    ax_a.set_title(f'$\\alpha$ — 700 W: 1.0$\\times$', fontsize=12, fontweight='bold')
    ax_a.set_xlabel('RF Power (W)', fontsize=11)
    ax_a.set_ylabel('$\\alpha = n_-/n_e$', fontsize=12)
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=10, loc='upper right')
    if pressure_data:
        axin = ax_a.inset_axes([0.54, 0.16, 0.43, 0.36])
        axin.plot(xpr, ap, '-s', color='red', lw=1.5, ms=4)
        axin.plot(xpr, ap, '--o', color='red', lw=1, ms=3, mfc='none', alpha=0.8)
        axin.set_title('p (mTorr) — 700 W', fontsize=8)
        axin.tick_params(labelsize=7)
        axin.grid(True, alpha=0.3)

    # Ratio summary table
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis('off')
    rows = []
    for sp in SPECIES_ORDER:
        y2i = power_data[i700].get('summary', {}).get(f'{sp}_icp', 0)
        y0i = power_data[i700].get('summary', {}).get(f'0D_{sp}', 0)
        r = y2i / max(y0i, 1) if y0i > 0 else 0
        rows.append([SP_LABEL[sp], f'{r:.1f}$\\times$'])
    rows.append(['$n_e$', f'{r_ne:.1f}$\\times$'])
    rows.append(['$n_+$', f'{r_np:.1f}$\\times$'])
    rows.append(['$n_-$', f'{r_nm:.1f}$\\times$'])
    tbl = ax_tbl.table(cellText=rows, colLabels=['Species', '700 W ratio'],
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
    for (i_r, j_c), cell in tbl.get_celld().items():
        if i_r == 0:
            cell.set_facecolor('#4682B4'); cell.set_text_props(weight='bold', color='white')
        elif i_r > 0 and j_c == 1:
            try:
                rv = float(cell.get_text().get_text().replace('$\\times$', ''))
                if 0.5 <= rv <= 2.0:
                    cell.set_facecolor('#C6EFCE')
                elif 0.2 <= rv <= 5.0:
                    cell.set_facecolor('#FFEB9C')
                else:
                    cell.set_facecolor('#FFC7CE')
            except ValueError:
                pass
    ax_tbl.set_title('Ratio summary (2D / 0D @ 700 W)', fontsize=11, fontweight='bold')

    fig.suptitle('Individual charged species benchmark',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_charged_individual.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_benchmark_charged_individual.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_benchmark_charged_individual")


def gen_SF6_Ar_mixture(ar_data, mesh, inside, tel, out_dir):
    """Two-panel 2D: [F] and [Ar*] at 50% Ar / 50% SF_6."""
    if ar_data is None:
        return
    nF = ar_data['fields'].get('F', np.zeros((mesh.Nr, mesh.Nz)))
    ne = ar_data.get('ne', None)
    n_Ar_star_0D = ar_data.get('summary', {}).get('n_Ar_star', 0)
    # Ar* spatial shape follows ne (both driven by electron source)
    if ne is not None and n_Ar_star_0D > 0 and ne.max() > 0:
        n_Ar_star = ne * (n_Ar_star_0D / ne.mean())
    else:
        n_Ar_star = np.zeros_like(nF)

    R_icp = tel['R_icp'] * 1e3
    R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    def _plot(ax, field, title, cmap, cb_label):
        F_cm3 = field * 1e-6
        log_f = np.where((F_cm3 > 0) & inside, np.log10(F_cm3), np.nan)
        f_disp, r_mm, z_mm = interp_mirror(log_f, mesh, inside, 200, 280)
        valid = f_disp[np.isfinite(f_disp)]
        vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 9
        vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 15
        norm = Normalize(vmin=vmin, vmax=vmax)
        ax.set_facecolor('#ececec')
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
        rgba[np.isnan(f_disp), 3] = 0
        im = ax.imshow(rgba, origin='lower', aspect='equal',
                       extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                       interpolation='bilinear')
        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='k', lw=0.9)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='k', lw=0.9)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='k', lw=0.9)
        ax.plot([-R_icp, R_icp], [z_top, z_top], color='k', lw=0.9)
        ax.plot([-R_proc, R_proc], [0, 0], color='k', lw=0.9)
        ax.set_xlabel('$r$ (mm)', fontsize=11)
        ax.set_ylabel('$z$ (mm)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(cb_label, fontsize=10)
        return vmin, vmax

    _plot(axes[0], nF, '[F] at 50% Ar / 50% SF$_6$',
          plt.cm.inferno, '$\\log_{10}$([F] / cm$^{-3}$)')
    _plot(axes[1], n_Ar_star, '[Ar*] metastable at 50% Ar',
          plt.cm.viridis, '$\\log_{10}$([Ar*] / cm$^{-3}$)')

    # Annotation: F drop and Ar* quenching
    Fw = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw[-1] / max(Fw[0], 1)) * 100 if len(Fw) > 1 else 0
    axes[0].text(0.5, 0.55, f'[F] drop = {drop:.0f}%',
                 transform=axes[0].transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes[1].text(0.5, 0.92, 'Ar* peaks near coil;\nPenning ionisation active',
                 transform=axes[1].transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.suptitle('SF$_6$/Ar mixture (50/50) — TEL reactor, 700 W, 10 mTorr (1.3 Pa)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_SF6_Ar_mixture.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_SF6_Ar_mixture.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_SF6_Ar_mixture")


def gen_residuals_stage10_style(power_data, mesh, inside, out_dir):
    """Two-panel residual diagnostics anchored at Mettler's 1000 W operating
    point (Fig. 4.17 direct-probe values), not the 700 W Fig. 4.5 qualitative
    cluster the prior version used.

    (a) Normalised-shape residual at 1000 W vs Mettler Fig. 4.17 direct-probe
        profile (5 radial sampling points 0, 25, 50, 75, 100 mm).  Wide y-range
        so the residual reads as the physical deficit it is, not a visual
        catastrophe clipped against a tight band.
    (b) Wafer-centre [F] model/Mettler ratio at the 90% SF_6 bias-on Fig. 4.17
        anchor (3.774e14 cm^-3, 1000 W) across the 200--1200 W biased power
        sweep.  Shade the factor-2 band; annotate the D6 FAIL region.
    """
    # --- Panel (a): radial residual at 1000 W (was 700 W) -----------------
    ref = next((e for e in power_data if e['value'] == 1000), None)
    if ref is None or 'F' not in ref.get('fields', {}):
        ref = next((e for e in power_data if e['value'] == 700), None)
        if ref is None or 'F' not in ref.get('fields', {}):
            return
        title_power = 700
    else:
        title_power = 1000
    nF = ref['fields']['F']
    r_mm_all = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6  # cm^-3, wafer radial profile
    F0 = max(F_w[0], 1e-10)
    F_norm = F_w / F0

    # Interpolate model onto the 5 Mettler sampling radii (0, 25, 50, 75, 100 mm)
    F_norm_at_mettler = np.interp(METTLER_RADIAL_R_MM, r_mm_all, F_norm)
    residual_pct = 100.0 * (F_norm_at_mettler - METTLER_RADIAL_NORM) / METTLER_RADIAL_NORM

    # --- Panel (b): wafer-centre ratio vs power against Mettler Fig. 4.17 -
    # Mettler's TEL data is probe-direct at 1000 W only; the ratio at every
    # power uses the 1000 W anchor (3.774e14 cm^-3, 90% SF6 bias-on).
    METTLER_1000W_90 = 3.774e14
    xp = [e['value'] for e in power_data]
    F_c_model = np.array([e.get('summary', {}).get('F_icp',
                          e.get('summary', {}).get('nF_centre_wafer_cm3', 0))
                          for e in power_data])
    ratio = F_c_model / METTLER_1000W_90

    # --- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): shape residual on a wide y-range with nested context bands
    ax = axes[0]
    ax.axhspan(-10, 10, color='tab:green', alpha=0.18, label='$\\pm 10$\\% (tight)')
    ax.axhspan(-25, -10, color='tab:olive', alpha=0.10)
    ax.axhspan(10, 25, color='tab:olive', alpha=0.10, label='$\\pm 25$\\% (order-of-mag.)')
    bar_colors = ['#2ca02c' if abs(v) <= 10 else ('#b8860b' if abs(v) <= 25 else '#d62728')
                  for v in residual_pct]
    ax.bar(METTLER_RADIAL_R_MM, residual_pct, width=12, color=bar_colors,
           edgecolor='black', linewidth=0.6, zorder=3)
    ax.axhline(0, color='k', lw=0.8)
    for r_val, res in zip(METTLER_RADIAL_R_MM, residual_pct):
        va = 'bottom' if res >= 0 else 'top'
        offset = 2.0 if res >= 0 else -2.0
        ax.text(r_val, res + offset, f'{res:+.1f}\\%', ha='center', va=va,
                fontsize=10, fontweight='bold')
    ax.set_xlabel('$r$ (mm)', fontsize=13)
    ax.set_ylabel('Normalised-shape model error (\\%)', fontsize=13)
    ax.set_title(f'(a) Normalised-shape residual at {title_power} W',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-50, 50)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel (b): absolute ratio vs power with factor-2 band and D6 annotation
    ax = axes[1]
    ax.axhspan(0.5, 2.0, color='tab:green', alpha=0.18, label='Factor-2 band')
    ax.axhspan(0.25, 0.5, color='tab:orange', alpha=0.10, label='Factor-4 band')
    ax.plot(xp, ratio, '-o', color='#d62728', lw=2.2, ms=9,
            label='Model / Mettler 1000 W anchor', zorder=3)
    ax.axhline(1.0, color='k', lw=0.9, ls='--', label='Perfect agreement')
    # Annotate the 1000 W point as the Mettler anchor (ratio = model/Mettler
    # at 1000 W = 1-|residual|/100)
    i1000 = xp.index(1000) if 1000 in xp else len(xp) // 2
    ax.annotate(f'Mettler anchor\n(1000 W, ratio={ratio[i1000]:.2f})',
                xy=(1000, ratio[i1000]), xytext=(800, ratio[i1000] + 0.4),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.set_xlabel('Power (W)', fontsize=13)
    ax.set_ylabel('Wafer-centre [F] — Model / Mettler', fontsize=13)
    ax.set_title('(b) Wafer-centre ratio vs Mettler 1000 W direct-probe anchor',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(2.2, ratio.max() * 1.15))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Model residuals vs Mettler Fig. 4.17 direct-probe data '
                 '(1000 W anchor) — D6 FAIL: scalar $\\gamma_\\mathrm{Al}$ cannot close',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_residuals.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'fig_residuals.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  fig_residuals  (re-anchored to Mettler Fig. 4.17 at 1000 W)")


def load_ar_mixture_data(ar_dir):
    """Load Ar/SF6 mixture case (single operating point)."""
    if not os.path.exists(ar_dir):
        return None
    entry = {'fields': {}, 'ions': {}}
    for sp in SPECIES_ORDER:
        p = os.path.join(ar_dir, f'n{sp}.npy')
        if os.path.exists(p):
            entry['fields'][sp] = np.load(p)
    for name in ['ne', 'Te', 'P_rz', 'E_theta_rms']:
        p = os.path.join(ar_dir, f'{name}.npy')
        if os.path.exists(p):
            entry[name] = np.load(p)
    sp = os.path.join(ar_dir, 'summary.json')
    if os.path.exists(sp):
        with open(sp) as f:
            entry['summary'] = json.load(f)
    return entry


def main():
    setup_style()

    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    config = SimulationConfig(config_path)
    tel = config.tel_geometry

    sweep_base = os.path.join(PROJECT_ROOT, 'results', 'sweeps')
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Build mesh for figure generation
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _ = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                     tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)

    # Load sweep data
    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    pressures = [5, 7, 10, 15, 20, 30]

    print("Loading sweep data...")
    power_data = load_sweep_data(
        os.path.join(sweep_base, 'power'), powers, 'power', 'P{:04d}W')
    pressure_data = load_sweep_data(
        os.path.join(sweep_base, 'pressure'), pressures, 'pressure', 'p{:02d}mTorr')

    print(f"  Power points: {len(power_data)}")
    print(f"  Pressure points: {len(pressure_data)}")

    # Load sensitivity data
    sensitivity_path = os.path.join(sweep_base, 'sensitivity.json')
    sensitivity_data = {}
    if os.path.exists(sensitivity_path):
        with open(sensitivity_path) as f:
            sensitivity_data = json.load(f)
        print(f"  Sensitivity parameters: {len(sensitivity_data)}")

    print(f"\nGenerating figures -> {out_dir}\n")

    # 1. Power sweep 2D grids
    if power_data:
        gen_neutral_sweep_2D(power_data, mesh, inside, tel,
                            'RF Power', 'W', out_dir, 'fig_neutral_power_2D')

    # 2. Pressure sweep 2D grids
    if pressure_data:
        gen_neutral_sweep_2D(pressure_data, mesh, inside, tel,
                            'Pressure', 'mTorr', out_dir, 'fig_neutral_pressure_2D')

    # 3. All species overlay (power)
    if power_data:
        gen_all_species_sweep(power_data, 'RF Power', 'W', out_dir, 'fig_all_neutrals_sweep')

    # 4. All species overlay (pressure)
    if pressure_data:
        gen_all_species_sweep(pressure_data, 'Pressure', 'mTorr', out_dir,
                             'fig_all_neutrals_pressure_sweep')

    # 3b. Charged species 2D sweep grids (new publication-quality layout)
    if power_data:
        gen_charged_sweep_2D(power_data, mesh, inside, tel,
                             'RF Power', 'W', out_dir, 'fig_charged_power_2D')
    if pressure_data:
        gen_charged_sweep_2D(pressure_data, mesh, inside, tel,
                             'Pressure', 'mTorr', out_dir, 'fig_charged_pressure_2D')

    # 5. Individual species benchmark (legacy 3x3)
    if power_data:
        gen_benchmark_neutrals(power_data, out_dir)

    # 6. Sensitivity analysis
    if sensitivity_data:
        gen_sensitivity(sensitivity_data, out_dir)

    # 7. Absolute validation (legacy)
    if power_data:
        gen_absolute_validation(power_data, out_dir)

    # 8. Charged species overview (legacy overlay)
    gen_charged_sweep(power_data, pressure_data, out_dir)

    # ─── Stage-10-style publication figures ───────────────────────────────
    # 9. Radial [F] at wafer vs Mettler
    if power_data:
        gen_radial_F_wafer_mettler(power_data, mesh, inside, out_dir)

    # 10. Axial [F] at r=0
    if power_data:
        gen_axial_F_profile(power_data, mesh, inside, tel, out_dir)

    # 11. Absolute [F] validation 4-panel
    if power_data:
        gen_absolute_validation_4panel(power_data, out_dir)

    # 12. Neutral benchmark 2-panel (power + pressure)
    if power_data:
        gen_neutral_benchmark_2panel(power_data, pressure_data, out_dir)

    # 13. Charged benchmark 4-panel
    if power_data:
        gen_charged_benchmark_4panel(power_data, pressure_data, out_dir)

    # 14. Individual neutral benchmark with pressure insets (overrides legacy filename)
    if power_data:
        gen_individual_neutrals_with_insets(power_data, pressure_data, out_dir)

    # 15. Individual charged benchmark with ratio table
    if power_data:
        gen_individual_charged_with_ratios(power_data, pressure_data, out_dir)

    # 16. SF6/Ar 50/50 mixture
    ar_dir = os.path.join(sweep_base, 'ar_mixture', 'Ar50')
    ar_data = load_ar_mixture_data(ar_dir)
    if ar_data is not None:
        gen_SF6_Ar_mixture(ar_data, mesh, inside, tel, out_dir)
    else:
        print("  [skip] fig_SF6_Ar_mixture — no data at results/sweeps/ar_mixture/Ar50")

    # 17. Residuals (Stage 10 Fig 22 style) — overwrites placeholder fig_residuals.png
    if power_data:
        gen_residuals_stage10_style(power_data, mesh, inside, out_dir)

    # Copy to presentation
    import shutil
    pres_dir = os.path.join(PROJECT_ROOT, 'docs', 'presentation', 'figures')
    os.makedirs(pres_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(out_dir, f), os.path.join(pres_dir, f))

    print(f"\nAll Stage 10-style figures complete")
    print(f"Total figures: {len([f for f in os.listdir(out_dir) if f.endswith('.png')])}")


if __name__ == '__main__':
    main()
