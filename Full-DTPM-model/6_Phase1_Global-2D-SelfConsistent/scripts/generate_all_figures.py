#!/usr/bin/env python3
"""
Generate ALL publication-quality figures for the comprehensive report.

Matches Stage 10 quality: 500x700 interpolated fine grid, RGBA rendering,
full-reactor mirroring, geometry overlay with material annotations.

Produces ~40 figures covering:
  - Reactor geometry and schematic
  - EM fields (E_theta, B_r, B_z, |B|)
  - Power deposition and plasma conductivity
  - Electron density and temperature
  - All 9 neutral species (SF6, SF5, SF4, SF3, SF2, SF, S, F, F2)
  - Charged species (ne, n+, n-, alpha, ion fractions)
  - Wafer radial profiles (all species)
  - Axial profiles
  - Multi-species 2D maps (4-panel)
  - Summary figures

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --run-dir results/runs/20260413_104959
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from matplotlib import patheffects
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}


# ══════════════════════════════════════════════════════════════
# Global style matching Stage 10
# ══════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


# ══════════════════════════════════════════════════════════════
# High-resolution interpolation engine (matching Stage 10)
# ══════════════════════════════════════════════════════════════

def interpolate_to_fine_grid(field, mesh, inside, Nr_fine=500, Nz_fine=700):
    """Interpolate a 2D field to a fine display grid and mirror for full reactor.

    Returns (field_display, r_full_mm, z_fine_mm):
        field_display: (Nz_fine, 2*Nr_fine-1) — ready for imshow
        r_full_mm: 1D array from -R to +R in mm
        z_fine_mm: 1D array from 0 to L in mm
    """
    # Prepare field with NaN for inactive cells
    f = field.copy().astype(float)
    f[~inside] = np.nan

    # Extend to r=0 axis (avoid NaN at boundary)
    r_ext = np.concatenate([[0.0], mesh.rc * 1e3])
    f_ext = np.concatenate([f[0:1, :], f], axis=0)

    # Fine grid
    r_half = np.linspace(0, mesh.R * 1e3, Nr_fine)
    z_fine = np.linspace(0, mesh.L * 1e3, Nz_fine)

    interp = RegularGridInterpolator(
        (r_ext, mesh.zc * 1e3), f_ext, method='linear',
        bounds_error=False, fill_value=np.nan)

    # Evaluate on fine grid
    rr, zz = np.meshgrid(r_half, z_fine, indexing='ij')
    pts = np.column_stack([rr.ravel(), zz.ravel()])
    f_fine = interp(pts).reshape(Nr_fine, Nz_fine)

    # Mirror for full reactor view
    r_left = -r_half[1:][::-1]
    r_full = np.concatenate([r_left, r_half])
    f_full = np.concatenate([f_fine[::-1, :][1:, :], f_fine], axis=0)

    # Transpose to (Nz, Nr) for imshow (origin='lower')
    return f_full.T, r_full, z_fine


def hires_contour(field, mesh, inside, tel, title, cmap_name, units,
                  vmin=None, vmax=None, log_scale=False,
                  symmetric=False, figsize=(8, 12), dark_bg=False):
    """Create a high-resolution full-reactor contour figure.

    Matches Stage 10 fig1_cross_section quality.
    """
    f_disp, r_mm, z_mm = interpolate_to_fine_grid(field, mesh, inside)

    cmap = plt.cm.get_cmap(cmap_name).copy()

    if log_scale:
        f_disp_clean = np.where(np.isfinite(f_disp) & (f_disp > 0), f_disp, np.nan)
        if vmin is None:
            valid = f_disp_clean[np.isfinite(f_disp_clean)]
            vmin = np.nanpercentile(valid, 1) if len(valid) > 0 else 1
        if vmax is None:
            valid = f_disp_clean[np.isfinite(f_disp_clean)]
            vmax = np.nanpercentile(valid, 99.5) if len(valid) > 0 else 10
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(np.log10(np.maximum(f_disp_clean, 10**vmin)), nan=vmin)))
        norm_for_cb = Normalize(vmin=vmin, vmax=vmax)
    elif symmetric:
        abs_max = max(abs(np.nanmin(f_disp)), abs(np.nanmax(f_disp)), 1e-10)
        if vmin is None: vmin = -abs_max
        if vmax is None: vmax = abs_max
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=0)))
        norm_for_cb = norm
    else:
        if vmin is None: vmin = np.nanmin(f_disp)
        if vmax is None: vmax = np.nanmax(f_disp)
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
        norm_for_cb = norm

    # Transparent where NaN (inactive)
    rgba[np.isnan(f_disp), 3] = 0.0

    fig, ax = plt.subplots(figsize=figsize)

    R_icp = tel.get('R_icp', 0.038) * 1e3
    R_proc = tel.get('R_proc', 0.105) * 1e3
    L_proc = tel.get('L_proc', 0.050) * 1e3
    L_apt = tel.get('L_apt', 0.002) * 1e3
    L_icp = tel.get('L_icp', 0.1815) * 1e3
    z_apt_bot = L_proc
    z_apt_top = L_proc + L_apt
    z_top = L_proc + L_apt + L_icp

    bg_color = 'black' if dark_bg else '#e8e8e8'
    ax.set_facecolor(bg_color)

    ax.imshow(rgba, origin='lower', aspect='auto',
              extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
              interpolation='bilinear')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_for_cb)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label=units, shrink=0.70, pad=0.03, aspect=30)

    # Reactor outline
    lc = 'white' if dark_bg else '#333333'
    lw = 1.8
    for s in [-1, 1]:
        # ICP tube walls
        ax.plot([s*R_icp, s*R_icp], [z_apt_top, z_top], color=lc, lw=lw, zorder=6)
        # Processing walls
        ax.plot([s*R_proc, s*R_proc], [0, z_apt_bot], color=lc, lw=lw, zorder=6)
        # Shoulder
        ax.plot([s*R_icp, s*R_proc], [z_apt_bot, z_apt_bot], color=lc, lw=lw, zorder=6)
        ax.plot([s*R_icp, s*R_proc], [z_apt_top, z_apt_top], color=lc, lw=0.8, ls='--', zorder=6, alpha=0.5)
        ax.plot([s*R_proc, s*R_proc], [z_apt_bot, z_apt_top], color=lc, lw=lw, zorder=6)
        # Aperture fill
        ax.fill_betweenx([z_apt_bot, z_apt_top], s*R_icp, s*R_proc,
                         color='#1a1a1a', alpha=0.85, zorder=5)
    # Window
    ax.plot([-R_icp, R_icp], [z_top, z_top], color='cyan', lw=2.5, zorder=6)
    # Wafer
    ax.plot([-75, 75], [0, 0], color='#dddddd' if dark_bg else '#666666', lw=2.5, zorder=6)
    # Floor
    ax.fill_betweenx([-3, 0], -R_proc, R_proc, color='#111111', alpha=0.8, zorder=5)

    # Labels
    txt_color = 'white' if dark_bg else '#333333'
    txt_props = dict(ha='center', va='center', fontweight='bold', zorder=7,
                     path_effects=[patheffects.withStroke(linewidth=2,
                     foreground='black' if not dark_bg else 'white')])
    ax.text(0, (z_apt_top + z_top) / 2, 'ICP SOURCE\n(quartz walls)',
            fontsize=11, color='darkorange', **txt_props)
    ax.text(0, z_apt_bot / 2, 'PROCESSING\n(Al walls)',
            fontsize=10, color='mediumpurple', **txt_props)

    # Annotations
    ann_color = 'white' if dark_bg else 'dodgerblue'
    ax.text(0, z_top + 3, f'Dielectric window ({2*R_icp:.0f} mm)',
            ha='center', va='bottom', fontsize=8, color=ann_color, fontweight='bold')
    ax.text(0, z_apt_bot + L_apt/2, f'Aperture ({L_apt:.0f} mm gap)',
            ha='center', va='center', fontsize=8, color=ann_color, fontweight='bold')
    ax.text(0, -5, f'Si wafer ({2*R_proc:.0f} mm)', ha='center', va='top',
            fontsize=8, color='gray')

    # Dimension arrows
    ax.annotate('', xy=(-R_icp-8, z_top), xytext=(-R_icp-8, z_apt_top),
                arrowprops=dict(arrowstyle='<->', color=ann_color, lw=0.8), zorder=7)
    ax.text(-R_icp-10, (z_apt_top+z_top)/2, f'{L_icp:.1f} mm',
            ha='right', va='center', fontsize=7, color=ann_color)

    ax.set_xlabel('$r$ (mm)')
    ax.set_ylabel('$z$ (mm)')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(-R_proc - 15, R_proc + 5)
    ax.set_ylim(-8, z_top + 10)
    ax.set_aspect('equal')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Figure generators
# ══════════════════════════════════════════════════════════════

def gen_species_cross_section(field, mesh, inside, tel, species_name, out_dir,
                               log_vmin=None, log_vmax=None):
    """Generate a cross-section figure for any species (log scale in cm^-3)."""
    field_cm3 = field * 1e-6  # m^-3 to cm^-3
    log_field = np.where((field_cm3 > 0) & inside, np.log10(field_cm3), np.nan)

    if log_vmin is None:
        valid = log_field[np.isfinite(log_field)]
        log_vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 8
    if log_vmax is None:
        valid = log_field[np.isfinite(log_field)]
        log_vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 15

    sp_map = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
              'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
              'F': 'F', 'F2': 'F$_2$', 'S': 'S'}
    sp_label = sp_map.get(species_name, species_name)

    title = f'[{sp_label}]$(r,z)$ — 700 W, 10 mTorr, pure SF$_6$'

    fig = hires_contour(
        log_field, mesh, inside, tel, title,
        'inferno', f'log$_{{10}}$ [{sp_label}] / cm$^{{-3}}$',
        vmin=log_vmin, vmax=log_vmax, dark_bg=True, figsize=(7, 10))

    fname = f'fig_{species_name}_cross_section'
    fig.savefig(os.path.join(out_dir, f'{fname}.png'), facecolor='black')
    fig.savefig(os.path.join(out_dir, f'{fname}.pdf'), facecolor='black')
    plt.close(fig)
    return fname


def gen_multispecies_4panel(fields, mesh, inside, tel, species_list, out_dir, fname):
    """4-panel multi-species contour map (matching Stage 10 fig_multispecies_contours)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 18))

    for idx, sp in enumerate(species_list[:4]):
        ax = axes[idx // 2, idx % 2]
        field = fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
        field_cm3 = field * 1e-6
        log_f = np.where((field_cm3 > 0) & inside, np.log10(field_cm3), np.nan)

        f_disp, r_mm, z_mm = interpolate_to_fine_grid(log_f, mesh, inside, Nr_fine=300, Nz_fine=400)

        valid = f_disp[np.isfinite(f_disp)]
        vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 8
        vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 15

        cmap = plt.cm.inferno.copy()
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
        rgba[np.isnan(f_disp), 3] = 0.0

        ax.set_facecolor('#e8e8e8')
        ax.imshow(rgba, origin='lower', aspect='auto',
                  extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                  interpolation='bilinear')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        sp_label = SP_LABEL.get(sp, sp)

        cb = plt.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(f'log$_{{10}}$ [{sp_label}] / cm$^{{-3}}$', fontsize=9)

        # Geometry outline
        R_icp = tel.get('R_icp', 0.038) * 1e3
        R_proc = tel.get('R_proc', 0.105) * 1e3
        z_apt = tel.get('L_proc', 0.050) * 1e3
        z_top = (tel.get('L_proc', 0.050) + tel.get('L_apt', 0.002) + tel.get('L_icp', 0.1815)) * 1e3
        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='gray', lw=1, zorder=6)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='gray', lw=1, zorder=6)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='gray', lw=1, zorder=6)

        ax.set_xlabel('$r$ (mm)', fontsize=10)
        ax.set_ylabel('$z$ (mm)', fontsize=10)
        ax.set_title(f'[{sp_label}]$(r,z)$', fontsize=12)
        ax.set_aspect('equal')

    fig.suptitle('Multi-Species 2D Density Maps — 700 W, 10 mTorr, pure SF$_6$',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{fname}.png'))
    fig.savefig(os.path.join(out_dir, f'{fname}.pdf'))
    plt.close(fig)


def gen_radial_profiles(fields, mesh, inside, tel, out_dir):
    """Radial profiles at wafer for all 9 species."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    r_mm = mesh.rc[inside[:, 0]] * 1e3

    for idx, sp in enumerate(['F', 'SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F2']):
        ax = axes[idx // 3, idx % 3]
        field = fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
        profile = field[:, 0][inside[:, 0]] * 1e-6  # cm^-3

        color = 'red' if sp == 'F' else 'blue' if sp == 'SF6' else 'green'
        ax.plot(r_mm, profile, '-o', lw=2, ms=2, color=color)

        sp_label = SP_LABEL.get(sp, sp)

        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel(f'[{sp_label}] (cm$^{{-3}}$)')
        ax.set_title(f'[{sp_label}] at wafer')
        ax.grid(True, alpha=0.3)

        if sp == 'F' and len(profile) > 1:
            drop = (1 - profile[-1] / max(profile[0], 1)) * 100 if profile[0] > 0 else 0
            ax.text(0.95, 0.95, f'Drop: {drop:.1f}%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Wafer-Level Radial Profiles — All Neutral Species', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_multispecies_radial.png'))
    fig.savefig(os.path.join(out_dir, 'fig_multispecies_radial.pdf'))
    plt.close(fig)


def gen_axial_profiles(fields, mesh, inside, out_dir):
    """Axial profiles along reactor centreline for all species."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    z_mm = mesh.zc * 1e3

    # Neutrals
    for sp, color, ls in [('F', 'red', '-'), ('SF6', 'blue', '-'), ('SF5', 'green', '-'),
                           ('SF4', 'orange', '-'), ('SF3', 'purple', '-'), ('F2', 'brown', '--'),
                           ('SF2', 'cyan', '--'), ('SF', 'magenta', '--'), ('S', 'gray', '--')]:
        field = fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
        profile = field[0, :] * 1e-6  # axis profile, cm^-3
        if np.any(profile > 0):
            ax1.semilogy(z_mm, np.maximum(profile, 1e6), ls, lw=1.5, color=color, label=sp)

    ax1.set_xlabel('$z$ (mm)')
    ax1.set_ylabel('Density (cm$^{-3}$)')
    ax1.set_title('(a) Neutral Species — Axial Profile ($r = 0$)')
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Volume-averaged bar chart
    species_names = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']
    avgs = []
    for sp in species_names:
        field = fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
        avg = np.sum(field * mesh.vol * inside) / np.sum(mesh.vol * inside) * 1e-6
        avgs.append(avg)

    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'gray', 'red', 'brown']
    ax2.bar(species_names, avgs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('Volume-averaged density (cm$^{-3}$)')
    ax2.set_title('(b) Volume-Averaged Species Densities')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Species Profiles — 700 W, 10 mTorr, pure SF$_6$', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_multispecies_axial.png'))
    fig.savefig(os.path.join(out_dir, 'fig_multispecies_axial.pdf'))
    plt.close(fig)


def gen_charged_species(ions, mesh, inside, tel, out_dir):
    """Charged species 2D maps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 18))

    for idx, (name, cmap_name) in enumerate([
        ('ne', 'viridis'), ('n+', 'Reds'), ('n-', 'Blues'), ('alpha', 'plasma')
    ]):
        ax = axes[idx // 2, idx % 2]
        field = ions.get(name, np.zeros((mesh.Nr, mesh.Nz)))

        f_disp, r_mm, z_mm = interpolate_to_fine_grid(field, mesh, inside, 300, 400)

        if name == 'alpha':
            vmin, vmax = 0, 10
        else:
            valid = f_disp[np.isfinite(f_disp) & (f_disp > 0)]
            vmin = np.nanpercentile(valid, 1) if len(valid) > 0 else 0
            vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 1

        im = ax.imshow(np.nan_to_num(f_disp, nan=0), origin='lower', aspect='auto',
                       extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                       cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation='bilinear')
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

        titles = {'ne': '$n_e$ (m$^{-3}$)', 'n+': '$n_+$ (m$^{-3}$)',
                  'n-': '$n_-$ (m$^{-3}$)', 'alpha': '$\\alpha = n_-/n_e$'}
        ax.set_title(titles.get(name, name))
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('$z$ (mm)')
        ax.set_aspect('equal')

    fig.suptitle('Charged Species — 700 W, 10 mTorr, pure SF$_6$', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_charged_species.png'))
    fig.savefig(os.path.join(out_dir, 'fig_charged_species.pdf'))
    plt.close(fig)


def gen_em_summary(E_theta, E_rms, B_r, B_z, mesh, inside, tel, out_dir):
    """EM field summary: E_theta, E_rms, B_r, B_z, |B|."""
    Nr, Nz = mesh.Nr, mesh.Nz
    Br = B_r[:Nr, :Nz] if B_r.shape[0] >= Nr else B_r
    Bz = B_z[:Nr, :Nz] if B_z.shape[0] >= Nr else B_z

    # E_theta RMS
    fig = hires_contour(E_rms, mesh, inside, tel,
                        '$E_{\\theta,\\mathrm{rms}}(r,z)$ — 40 MHz, 700 W',
                        'hot', 'V/m', dark_bg=True, figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_rms.png'), facecolor='black')
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_rms.pdf'), facecolor='black')
    plt.close(fig)

    # |B| magnitude
    B_mag = np.sqrt(Br**2 + Bz**2)
    fig = hires_contour(B_mag, mesh, inside, tel,
                        '$|\\mathbf{B}|(r,z)$ — Magnetic Field Magnitude',
                        'magma', 'T', figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_B_magnitude.png'))
    fig.savefig(os.path.join(out_dir, 'fig_B_magnitude.pdf'))
    plt.close(fig)

    # E_theta instantaneous (symmetric cmap)
    fig = hires_contour(E_theta, mesh, inside, tel,
                        '$E_\\theta(r,z)$ — Instantaneous',
                        'RdBu_r', 'V/m', symmetric=True, figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_instant.png'))
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_instant.pdf'))
    plt.close(fig)


def gen_power_ne_Te(P_rz, ne, Te, mesh, inside, tel, out_dir):
    """Power deposition, ne, and Te contours."""
    # Power (log)
    P_log = np.where((P_rz > 0) & inside, np.log10(P_rz), np.nan)
    fig = hires_contour(P_log, mesh, inside, tel,
                        'Power Deposition $P(r,z)$',
                        'hot', 'log$_{10}$ $P$ / W m$^{-3}$',
                        dark_bg=True, figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_power_deposition.png'), facecolor='black')
    fig.savefig(os.path.join(out_dir, 'fig_power_deposition.pdf'), facecolor='black')
    plt.close(fig)

    # ne (log)
    ne_log = np.where((ne > 0) & inside, np.log10(ne), np.nan)
    fig = hires_contour(ne_log, mesh, inside, tel,
                        'Electron Density $n_e(r,z)$',
                        'viridis', 'log$_{10}$ $n_e$ / m$^{-3}$',
                        figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_ne_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_ne_contour.pdf'))
    plt.close(fig)

    # Te
    fig = hires_contour(Te, mesh, inside, tel,
                        'Electron Temperature $T_e(r,z)$',
                        'plasma', 'eV', figsize=(7, 10))
    fig.savefig(os.path.join(out_dir, 'fig_Te_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_Te_contour.pdf'))
    plt.close(fig)


def gen_F_validation(nF, mesh, inside, tel, out_dir):
    """F cross-section with validation info (matching Stage 10 fig1)."""
    F_cm3 = nF * 1e-6
    F_log = np.where((F_cm3 > 0) & inside, np.log10(F_cm3), np.nan)

    Fw = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0

    title = (f'Fluorine Atom Density [F]$(r,z)$\n'
             f'$P_{{rf}}$ = 700 W,  $p$ = 10 mTorr,  pure SF$_6$,  '
             f'[F] drop = {drop:.0f}%')

    fig = hires_contour(F_log, mesh, inside, tel, title,
                        'inferno', 'log$_{10}$ [F] / cm$^{-3}$',
                        vmin=12.0, vmax=15.0, dark_bg=True, figsize=(7, 10.5))
    fig.savefig(os.path.join(out_dir, 'fig_cross_section_F.png'), facecolor='black')
    fig.savefig(os.path.join(out_dir, 'fig_cross_section_F.pdf'), facecolor='black')
    plt.close(fig)


def gen_wafer_F_validation(nF, mesh, inside, out_dir):
    """Wafer [F] profile with Mettler comparison (matching Stage 10 fig2)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    r_mm = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6

    ax1.plot(r_mm, F_w, 'r-o', lw=2, ms=3)
    ax1.set_xlabel('$r$ (mm)')
    ax1.set_ylabel('[F] (cm$^{-3}$)')
    ax1.set_title('(a) [F] Radial Profile at Wafer')
    ax1.grid(True, alpha=0.3)
    if len(F_w) > 1:
        drop = (1 - F_w[-1] / max(F_w[0], 1)) * 100
        ax1.text(0.95, 0.95, f'Drop: {drop:.1f}%',
                transform=ax1.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if len(F_w) > 0 and F_w[0] > 0:
        ax2.plot(r_mm, F_w / F_w[0], 'r-o', lw=2, ms=3, label='This work')
        ax2.axhline(0.26, color='green', ls='--', lw=1.5, label='74% drop (Mettler 2025)')
        ax2.set_xlabel('$r$ (mm)')
        ax2.set_ylabel('[F] / [F]$_{\\mathrm{center}}$')
        ax2.set_title('(b) Normalised [F] vs Experiment')
        ax2.set_ylim(0, 1.15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle('Fluorine Density at Wafer Surface — Validation', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_wafer_F_validation.png'))
    fig.savefig(os.path.join(out_dir, 'fig_wafer_F_validation.pdf'))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def _gen_geometry_hires(mesh, inside, bc_type, tel, coil_pos, out_dir):
    """Hi-res geometry figure with mirrored mask and coils."""
    inside_full = np.concatenate([inside[::-1, :], inside[1:, :]], axis=0)
    bc_full = np.concatenate([bc_type[::-1, :], bc_type[1:, :]], axis=0)
    r_full = np.concatenate([-mesh.rc[::-1] * 1e3, mesh.rc[1:] * 1e3])
    z_mm = mesh.zc * 1e3
    R, Z = np.meshgrid(r_full, z_mm, indexing='ij')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Panel a: active cells + coil positions
    ax1.pcolormesh(R, Z, inside_full.astype(float), cmap='Blues',
                   vmin=0, vmax=1, shading='auto', rasterized=True)
    for (ci, cj) in coil_pos:
        ax1.plot(mesh.rc[ci] * 1e3, mesh.zc[cj] * 1e3, 'ro', ms=7, zorder=5)
        ax1.plot(-mesh.rc[ci] * 1e3, mesh.zc[cj] * 1e3, 'ro', ms=7, zorder=5)

    R_icp = tel.get('R_icp', 0.038) * 1e3
    R_proc = tel.get('R_proc', 0.105) * 1e3
    z_apt = tel.get('L_proc', 0.050) * 1e3
    z_top = (tel.get('L_proc', 0.050) + tel.get('L_apt', 0.002) + tel.get('L_icp', 0.150)) * 1e3

    for s in [-1, 1]:
        ax1.plot([s*R_icp, s*R_icp], [z_apt, z_top], 'b-', lw=1.5)
        ax1.plot([s*R_proc, s*R_proc], [0, z_apt], 'gray', lw=1.5)
        ax1.plot([s*R_icp, s*R_proc], [z_apt, z_apt], 'gray', lw=1.5)

    ax1.set_xlabel('$r$ (mm)')
    ax1.set_ylabel('$z$ (mm)')
    ax1.set_title('(a) Reactor Geometry — Active Cells & Coil Positions')
    ax1.set_aspect('equal')

    # Panel b: boundary types
    bc_names = {-1: 'Inactive', 0: 'Interior', 1: 'Axis', 2: 'Quartz',
                3: 'Window', 4: 'Al side', 5: 'Al top', 6: 'Wafer', 7: 'Shoulder'}
    im2 = ax2.pcolormesh(R, Z, bc_full.astype(float), cmap='tab10',
                         vmin=-1.5, vmax=7.5, shading='auto', rasterized=True)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.82, pad=0.02, ticks=list(bc_names.keys()))
    cb2.set_ticklabels(list(bc_names.values()))
    ax2.set_xlabel('$r$ (mm)')
    ax2.set_ylabel('$z$ (mm)')
    ax2.set_title('(b) Boundary Classification')
    ax2.set_aspect('equal')

    fig.suptitle('TEL ICP Reactor Geometry', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig01_geometry.png'))
    fig.savefig(os.path.join(out_dir, 'fig01_geometry.pdf'))
    plt.close(fig)


def _gen_wafer_profiles_hires(nF, nSF6, ne, Te, mesh, inside, out_dir):
    """4-panel wafer profiles (hi-res version)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    r_mm = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6

    axes[0, 0].plot(r_mm, F_w, 'r-o', lw=2, ms=3)
    axes[0, 0].set_xlabel('$r$ (mm)')
    axes[0, 0].set_ylabel('[F] (cm$^{-3}$)')
    axes[0, 0].set_title('(a) [F] Radial Profile at Wafer')
    axes[0, 0].grid(True, alpha=0.3)
    if len(F_w) > 1:
        drop = (1 - F_w[-1] / max(F_w[0], 1)) * 100 if F_w[0] > 0 else 0
        axes[0, 0].text(0.95, 0.95, f'Drop: {drop:.1f}%',
                        transform=axes[0, 0].transAxes, ha='right', va='top',
                        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if len(F_w) > 0 and F_w[0] > 0:
        axes[0, 1].plot(r_mm, F_w / F_w[0], 'r-o', lw=2, ms=3, label='This work')
        axes[0, 1].axhline(0.26, color='green', ls='--', lw=1.5, label='74% drop (Mettler 2025)')
        axes[0, 1].set_xlabel('$r$ (mm)')
        axes[0, 1].set_ylabel('[F] / [F]$_{\\mathrm{center}}$')
        axes[0, 1].set_title('(b) Normalised [F] at Wafer')
        axes[0, 1].set_ylim(0, 1.15)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

    j_mid = np.argmin(np.abs(mesh.zc - mesh.L * 0.6))
    ne_mid = ne[:, j_mid] * 1e-6
    axes[1, 0].plot(mesh.rc * 1e3, ne_mid, 'b-o', lw=2, ms=2)
    axes[1, 0].set_xlabel('$r$ (mm)')
    axes[1, 0].set_ylabel('$n_e$ (cm$^{-3}$)')
    axes[1, 0].set_title(f'(c) $n_e$ Radial Profile at $z$ = {mesh.zc[j_mid]*1e3:.0f} mm')
    axes[1, 0].grid(True, alpha=0.3)

    SF6_w = nSF6[:, 0][inside[:, 0]] * 1e-6
    axes[1, 1].plot(r_mm, SF6_w, 'b-o', lw=2, ms=3)
    axes[1, 1].set_xlabel('$r$ (mm)')
    axes[1, 1].set_ylabel('[SF$_6$] (cm$^{-3}$)')
    axes[1, 1].set_title('(d) SF$_6$ Radial Profile at Wafer')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Wafer-Level Radial Profiles', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_wafer_profiles.png'))
    fig.savefig(os.path.join(out_dir, 'fig_wafer_profiles.pdf'))
    plt.close(fig)


def _gen_summary_6panel_hires(E_rms, P_rz, ne, Te, nF, mesh, inside, tel, out_dir):
    """6-panel summary using high-res interpolation."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fields_info = [
        (E_rms, '$E_{\\theta,\\mathrm{rms}}$ [V/m]', 'hot', False),
        (np.where((P_rz > 0) & inside, np.log10(P_rz), np.nan),
         'Power Deposition [log$_{10}$ W/m$^3$]', 'hot', False),
        (np.where((ne > 0) & inside, np.log10(ne), np.nan),
         '$n_e$ [log$_{10}$ m$^{-3}$]', 'viridis', False),
        (Te, '$T_e$ [eV]', 'plasma', False),
        (np.where((nF > 0) & inside, np.log10(nF * 1e-6), np.nan),
         '[F] [log$_{10}$ cm$^{-3}$]', 'inferno', False),
    ]

    for idx, (field, title, cmap_name, _) in enumerate(fields_info):
        ax = axes[idx // 3, idx % 3]
        f_disp, r_mm, z_mm = interpolate_to_fine_grid(field, mesh, inside, 250, 350)
        valid = f_disp[np.isfinite(f_disp)]
        vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 0
        vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 1
        norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(np.nan_to_num(f_disp, nan=vmin), origin='lower', aspect='auto',
                       extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                       cmap=cmap_name, norm=norm, interpolation='bilinear')
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        R_icp = tel.get('R_icp', 0.038) * 1e3
        R_proc = tel.get('R_proc', 0.105) * 1e3
        z_apt = tel.get('L_proc', 0.050) * 1e3
        z_top = (tel.get('L_proc', 0.050) + tel.get('L_apt', 0.002) + tel.get('L_icp', 0.150)) * 1e3
        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='gray', lw=0.8)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='gray', lw=0.8)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='gray', lw=0.8)

        ax.set_xlabel('$r$ (mm)', fontsize=9)
        ax.set_ylabel('$z$ (mm)', fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')

    # Bottom-right: wafer F profile
    ax = axes[1, 2]
    r_mm_w = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6
    ax.plot(r_mm_w, F_w, 'r-o', lw=2, ms=2)
    if len(F_w) > 0 and F_w[0] > 0:
        ax.axhline(F_w[0] * 0.26, color='green', ls='--', lw=1.5, label='74% drop level')
    ax.set_xlabel('$r$ (mm)')
    ax.set_ylabel('[F] at wafer (cm$^{-3}$)')
    ax.set_title('Wafer [F] Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Fw_active = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw_active[-1] / max(Fw_active[0], 1e-10)) * 100 if len(Fw_active) > 1 else 0
    ne_avg = np.sum(ne * mesh.vol * inside) / np.sum(mesh.vol * inside)
    fig.suptitle(
        f'Summary: $P_{{rf}}$ = 700 W, $f$ = 40 MHz, '
        f'$\\eta$ = 0.430, [F] drop = {drop:.1f}%, '
        f'$n_{{e,avg}}$ = {ne_avg:.1e} m$^{{-3}}$',
        fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_summary_6panel.png'))
    fig.savefig(os.path.join(out_dir, 'fig_summary_6panel.pdf'))
    plt.close(fig)


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
    pres_dir = os.path.join(PROJECT_ROOT, 'docs', 'presentation', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pres_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")

    # Build mesh
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, bc_type = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                          tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)

    def load(name):
        path = os.path.join(data_dir, f'{name}.npy')
        return np.load(path) if os.path.exists(path) else None

    # Load all data
    E_theta = load('E_theta')
    E_rms = load('E_theta_rms')
    B_r = load('B_r')
    B_z = load('B_z')
    P_rz = load('P_rz')
    ne = load('ne')
    Te = load('Te')

    # Load all 9 neutral species
    species_fields = {}
    for sp in ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']:
        data = load(f'n{sp}')
        if data is not None:
            species_fields[sp] = data

    # Load charged species
    ions = {}
    for name in ['ne', 'n+', 'n-', 'alpha', 'SF5+', 'SF3+', 'SF4+', 'F+', 'F-', 'SF6-', 'SF5-', 'SF4-']:
        data = load(f'ion_{name}')
        if data is not None:
            ions[name] = data

    nF = species_fields.get('F', load('nF'))
    nSF6 = species_fields.get('SF6', load('nSF6'))

    print(f"Loaded {len(species_fields)} neutral species, {len(ions)} charged species")
    print(f"\nGenerating figures -> {out_dir}\n")

    # ── 1. EM fields ──
    if E_theta is not None and E_rms is not None and B_r is not None:
        gen_em_summary(E_theta, E_rms, B_r, B_z, mesh, inside, tel, out_dir)
        print("  [1/8] EM fields (E_theta, |B|)")

    # ── 2. Power, ne, Te ──
    if P_rz is not None and ne is not None:
        gen_power_ne_Te(P_rz, ne, Te, mesh, inside, tel, out_dir)
        print("  [2/8] Power, ne, Te contours")

    # ── 3. F cross-section (main validation figure) ──
    if nF is not None:
        gen_F_validation(nF, mesh, inside, tel, out_dir)
        gen_wafer_F_validation(nF, mesh, inside, out_dir)
        print("  [3/8] F cross-section + wafer validation")

    # ── 4. Individual species cross-sections ──
    for sp in ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']:
        if sp in species_fields:
            gen_species_cross_section(species_fields[sp], mesh, inside, tel, sp, out_dir)
    print("  [4/8] Individual species cross-sections (9 figures)")

    # ── 5. Multi-species 4-panel maps ──
    if len(species_fields) >= 4:
        gen_multispecies_4panel(species_fields, mesh, inside, tel,
                               ['F', 'SF6', 'SF5', 'SF4'], out_dir,
                               'fig_multispecies_contours')
        gen_multispecies_4panel(species_fields, mesh, inside, tel,
                               ['SF3', 'SF2', 'SF', 'S'], out_dir,
                               'fig_multispecies_contours_2')
        print("  [5/8] Multi-species 4-panel maps (2 figures)")

    # ── 6. Radial profiles (all species at wafer) ──
    if len(species_fields) >= 4:
        gen_radial_profiles(species_fields, mesh, inside, tel, out_dir)
        print("  [6/8] Radial profiles (all species)")

    # ── 7. Axial profiles + bar chart ──
    if len(species_fields) >= 4:
        gen_axial_profiles(species_fields, mesh, inside, out_dir)
        print("  [7/8] Axial profiles + bar chart")

    # ── 8. Charged species ──
    if len(ions) >= 2:
        gen_charged_species(ions, mesh, inside, tel, out_dir)
        print("  [8/8] Charged species contours")

    # ── 9. Additional figures referenced by report (hi-res versions) ──
    # Geometry figure (mirrored mask + boundary types)
    from dtpm.core.geometry import compute_coil_positions_cylindrical, count_boundary_cells
    coil_pos = compute_coil_positions_cylindrical(config, mesh)
    _gen_geometry_hires(mesh, inside, bc_type, tel, coil_pos, out_dir)

    # nF and nSF6 linear contours (report references these names)
    if nF is not None:
        fig = hires_contour(nF, mesh, inside, tel,
                            'Fluorine Density [F]$(r,z)$',
                            'Reds', 'm$^{-3}$', figsize=(7, 10))
        fig.savefig(os.path.join(out_dir, 'fig_nF_contour.png'))
        fig.savefig(os.path.join(out_dir, 'fig_nF_contour.pdf'))
        plt.close(fig)

    if nSF6 is not None:
        fig = hires_contour(nSF6, mesh, inside, tel,
                            'SF$_6$ Density $(r,z)$',
                            'Blues', 'm$^{-3}$', figsize=(7, 10))
        fig.savefig(os.path.join(out_dir, 'fig_nSF6_contour.png'))
        fig.savefig(os.path.join(out_dir, 'fig_nSF6_contour.pdf'))
        plt.close(fig)

    # Wafer profiles (4-panel)
    if nF is not None and nSF6 is not None and ne is not None:
        _gen_wafer_profiles_hires(nF, nSF6, ne, Te, mesh, inside, out_dir)

    # Summary 6-panel
    if E_rms is not None and P_rz is not None and ne is not None and nF is not None:
        _gen_summary_6panel_hires(E_rms, P_rz, ne, Te, nF, mesh, inside, tel, out_dir)

    print("  [9/9] Report-referenced figures (geometry, nF/SF6, wafer, summary)")

    # Copy PNGs to presentation
    import shutil
    for f in os.listdir(out_dir):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(out_dir, f), os.path.join(pres_dir, f))

    total = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"\nAll figures complete: {total} PNG + {total} PDF files")
    print(f"Saved to: {out_dir}")


if __name__ == '__main__':
    main()
