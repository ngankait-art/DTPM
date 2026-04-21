#!/usr/bin/env python3
"""
Generate all publication-quality figures for the Phase 1 report.

All 2D contour plots show the FULL REACTOR (mirrored at r=0) with
geometry overlay, matching the Stage 10 cross-section figure style.

Usage:
    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --run-dir results/runs/20260412_225203
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import patheffects
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask, build_index_maps


# ══════════════════════════════════════════════════════════════
# Global style
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
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
    })


# ══════════════════════════════════════════════════════════════
# Core mirroring and geometry overlay helpers
# ══════════════════════════════════════════════════════════════

def mirror_field(field, mesh):
    """Mirror a 2D (Nr, Nz) field to create full-reactor view.

    Returns (field_full, r_full_mm):
        field_full: (2*Nr-1, Nz)
        r_full_mm: 1D array from -R to +R in mm
    """
    # Flip radially and concatenate: [-R..0..+R]
    field_full = np.concatenate([field[::-1, :], field[1:, :]], axis=0)
    r_full_mm = np.concatenate([-mesh.rc[::-1] * 1e3, mesh.rc[1:] * 1e3])
    return field_full, r_full_mm


def mirror_mask(inside, mesh):
    """Mirror the geometry mask."""
    inside_full = np.concatenate([inside[::-1, :], inside[1:, :]], axis=0)
    return inside_full


def add_geometry_overlay(ax, tel, show_labels=True, show_dims=True,
                         linecolor='k', linewidth=1.5):
    """Draw the T-shaped reactor outline and annotations on a full-reactor plot."""
    R_icp = tel.get('R_icp', 0.038) * 1e3
    R_proc = tel.get('R_proc', 0.105) * 1e3
    L_proc = tel.get('L_proc', 0.050) * 1e3
    L_apt = tel.get('L_apt', 0.002) * 1e3
    L_icp = tel.get('L_icp', 0.1815) * 1e3
    L_total = L_proc + L_apt + L_icp

    lw = linewidth
    lc = linecolor

    # ICP source outline (narrow top)
    ax.plot([-R_icp, -R_icp], [L_proc + L_apt, L_total], color=lc, lw=lw)
    ax.plot([R_icp, R_icp], [L_proc + L_apt, L_total], color=lc, lw=lw)
    ax.plot([-R_icp, R_icp], [L_total, L_total], color=lc, lw=lw)

    # Processing chamber outline (wide bottom)
    ax.plot([-R_proc, -R_proc], [0, L_proc], color=lc, lw=lw)
    ax.plot([R_proc, R_proc], [0, L_proc], color=lc, lw=lw)
    ax.plot([-R_proc, R_proc], [0, 0], color=lc, lw=lw)

    # Aperture plate (shoulder)
    ax.plot([-R_proc, -R_icp], [L_proc, L_proc], color=lc, lw=lw)
    ax.plot([R_icp, R_proc], [L_proc, L_proc], color=lc, lw=lw)
    ax.plot([-R_proc, -R_icp], [L_proc + L_apt, L_proc + L_apt], color=lc, lw=lw, ls='--', alpha=0.5)
    ax.plot([R_icp, R_proc], [L_proc + L_apt, L_proc + L_apt], color=lc, lw=lw, ls='--', alpha=0.5)

    # Vertical connections at shoulder
    ax.plot([-R_proc, -R_proc], [L_proc, L_proc + L_apt], color=lc, lw=lw)
    ax.plot([R_proc, R_proc], [L_proc, L_proc + L_apt], color=lc, lw=lw)

    if show_labels:
        txt_props = dict(fontsize=9, color='dodgerblue', fontweight='bold',
                         path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        ax.text(0, L_total + 3, 'Dielectric window (76 mm)', ha='center', va='bottom', **txt_props)
        ax.text(0, L_proc + L_apt/2, 'Aperture (2 mm gap)', ha='center', va='center', **txt_props)

        side_props = dict(fontsize=8, color='0.3', style='italic', rotation=90,
                          path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        ax.text(-R_icp - 3, (L_proc + L_apt + L_total) / 2, 'SiO$_2$', ha='right', va='center', **side_props)
        ax.text(-R_proc - 3, L_proc / 2, 'Al', ha='right', va='center', **side_props)

        region_props = dict(fontsize=10, ha='center', va='center', alpha=0.7,
                            path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        ax.text(0, (L_proc + L_apt + L_total) / 2, 'ICP SOURCE\n(quartz walls)',
                color='darkorange', **region_props)
        ax.text(0, L_proc / 2, 'PROCESSING\n(Al walls)',
                color='purple', **region_props)

    if show_dims:
        dim_props = dict(fontsize=8, color='0.4',
                         path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])
        # ICP height
        ax.annotate('', xy=(-R_icp - 8, L_total), xytext=(-R_icp - 8, L_proc + L_apt),
                     arrowprops=dict(arrowstyle='<->', color='0.4', lw=0.8))
        ax.text(-R_icp - 10, (L_proc + L_apt + L_total) / 2, f'{L_icp:.1f} mm',
                ha='right', va='center', **dim_props)
        # Processing height
        ax.annotate('', xy=(-R_proc - 5, L_proc), xytext=(-R_proc - 5, 0),
                     arrowprops=dict(arrowstyle='<->', color='0.4', lw=0.8))
        ax.text(-R_proc - 7, L_proc / 2, f'{L_proc:.0f} mm',
                ha='right', va='center', **dim_props)
        # Widths
        ax.text(0, -5, f'Si wafer ({2*R_proc:.0f} mm)', ha='center', va='top',
                fontsize=8, color='0.4')
        ax.annotate('', xy=(-R_icp, L_total + 1.5), xytext=(R_icp, L_total + 1.5),
                     arrowprops=dict(arrowstyle='<->', color='dodgerblue', lw=0.7))
        ax.text(0, L_total + 2, f'2R$_{{icp}}$ = {2*R_icp:.0f} mm',
                ha='center', va='bottom', fontsize=8, color='dodgerblue')


def make_full_reactor_contour(field, mesh, inside, tel, title, cmap, units,
                               log_scale=False, vmin=None, vmax=None,
                               show_labels=True, show_dims=True,
                               figsize=(7, 9), symmetric_cmap=False):
    """Create a single full-reactor contour figure matching Stage 10 style."""
    field_full, r_mm = mirror_field(field, mesh)
    inside_full = mirror_mask(inside, mesh)
    z_mm = mesh.zc * 1e3

    # Mask inactive cells
    data = field_full.copy().astype(float)
    data[~inside_full] = np.nan

    R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        data_pos = np.where(data > 0, data, np.nan)
        if vmin is None:
            valid = data_pos[np.isfinite(data_pos)]
            vmin = valid.min() if len(valid) > 0 else 1e10
        if vmax is None:
            valid = data_pos[np.isfinite(data_pos)]
            vmax = valid.max() if len(valid) > 0 else 1e20
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(R, Z, data_pos, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
    elif symmetric_cmap:
        abs_max = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(R, Z, data, cmap=cmap, norm=norm,
                           shading='auto', rasterized=True)
    else:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        im = ax.pcolormesh(R, Z, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)

    # Fill inactive region with light gray
    inactive_full = ~inside_full
    ax.pcolormesh(R, Z, np.where(inactive_full, 1.0, np.nan),
                  cmap='Greys', vmin=0, vmax=2, shading='auto',
                  rasterized=True, alpha=0.3)

    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, aspect=30)
    cb.set_label(units, fontsize=11)

    add_geometry_overlay(ax, tel, show_labels=show_labels, show_dims=show_dims)

    ax.set_xlabel('$r$ (mm)', fontsize=13)
    ax.set_ylabel('$z$ (mm)', fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect('equal')

    R_proc = tel.get('R_proc', 0.105) * 1e3
    L_total = (tel.get('L_proc', 0.050) + tel.get('L_apt', 0.002) + tel.get('L_icp', 0.1815)) * 1e3
    ax.set_xlim(-R_proc - 12, R_proc + 5)
    ax.set_ylim(-8, L_total + 10)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# Individual figure generators
# ══════════════════════════════════════════════════════════════

def fig_geometry(mesh, inside, bc_type, tel, coil_pos, out_dir):
    """Fig 1: Reactor geometry with boundary types (full reactor)."""
    inside_full = mirror_mask(inside, mesh)
    bc_full = np.concatenate([bc_type[::-1, :], bc_type[1:, :]], axis=0)
    _, r_mm = mirror_field(inside.astype(float), mesh)
    z_mm = mesh.zc * 1e3
    R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Panel (a): Active cells with coil positions
    ax1.pcolormesh(R, Z, inside_full.astype(float), cmap='Blues',
                   vmin=0, vmax=1, shading='auto', rasterized=True)
    for (ci, cj) in coil_pos:
        ax1.plot(mesh.rc[ci] * 1e3, mesh.zc[cj] * 1e3, 'ro', ms=7, zorder=5)
        ax1.plot(-mesh.rc[ci] * 1e3, mesh.zc[cj] * 1e3, 'ro', ms=7, zorder=5)
    add_geometry_overlay(ax1, tel)
    ax1.set_xlabel('$r$ (mm)')
    ax1.set_ylabel('$z$ (mm)')
    ax1.set_title('(a) Reactor Geometry — Active Cells & Coil Positions')
    ax1.set_aspect('equal')

    # Panel (b): Boundary types
    bc_labels = {-1: 'Inactive', 0: 'Interior', 1: 'Axis', 2: 'Quartz',
                 3: 'Window', 4: 'Al side', 5: 'Al top', 6: 'Wafer', 7: 'Shoulder'}
    im2 = ax2.pcolormesh(R, Z, bc_full.astype(float), cmap='tab10',
                         vmin=-1.5, vmax=7.5, shading='auto', rasterized=True)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.82, pad=0.02, ticks=list(bc_labels.keys()))
    cb2.set_ticklabels(list(bc_labels.values()))
    add_geometry_overlay(ax2, tel, show_labels=False, show_dims=False)
    ax2.set_xlabel('$r$ (mm)')
    ax2.set_ylabel('$z$ (mm)')
    ax2.set_title('(b) Boundary Classification')
    ax2.set_aspect('equal')

    fig.suptitle('TEL ICP Reactor Geometry', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig01_geometry.png'))
    fig.savefig(os.path.join(out_dir, 'fig01_geometry.pdf'))
    plt.close(fig)
    print("  Fig 01: Geometry")


def fig_cross_section_F(nF, mesh, inside, tel, out_dir):
    """Fig: Full cross-section [F] density (matching Stage 10 fig1_cross_section)."""
    F_drop = _compute_F_drop(nF, inside)
    P_rf = 700
    title = (f'Fluorine Atom Density [F]$(r,z)$\n'
             f'$P_{{rf}}$ = {P_rf} W,  $p$ = 10 mTorr,  pure SF$_6$,  '
             f'[F] drop = {F_drop:.0f}%')

    fig = make_full_reactor_contour(
        np.log10(np.maximum(nF * 1e-6, 1e6)),  # log10([F] in cm^-3)
        mesh, inside, tel, title,
        cmap='inferno', units='log$_{10}$ [F] / cm$^{-3}$',
        vmin=12.0, vmax=15.0,
        show_labels=True, show_dims=True, figsize=(7, 9.5))
    fig.savefig(os.path.join(out_dir, 'fig_cross_section_F.png'))
    fig.savefig(os.path.join(out_dir, 'fig_cross_section_F.pdf'))
    plt.close(fig)
    print("  Fig: Cross-section [F]")


def fig_E_theta_fields(E_theta, E_rms, mesh, inside, tel, out_dir):
    """Fig: E_theta instantaneous and RMS (full reactor)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    for ax, field, title, cmap, sym in [
        (ax1, E_theta, '(a) $E_\\theta$ (instantaneous)', 'RdBu_r', True),
        (ax2, E_rms, '(b) $E_\\theta$ RMS', 'hot', False),
    ]:
        f_full, r_mm = mirror_field(field, mesh)
        i_full = mirror_mask(inside, mesh)
        data = f_full.copy().astype(float)
        data[~i_full] = np.nan
        z_mm = mesh.zc * 1e3
        R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

        if sym:
            abs_max = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-10)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.pcolormesh(R, Z, data, cmap=cmap, norm=norm,
                               shading='auto', rasterized=True)
        else:
            im = ax.pcolormesh(R, Z, data, cmap=cmap, shading='auto', rasterized=True)

        ax.pcolormesh(R, Z, np.where(~i_full, 1.0, np.nan),
                      cmap='Greys', vmin=0, vmax=2, shading='auto',
                      rasterized=True, alpha=0.3)
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label='V/m')
        add_geometry_overlay(ax, tel, show_labels=False, show_dims=False)
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('$z$ (mm)')
        ax.set_title(title)
        ax.set_aspect('equal')

    fig.suptitle('FDTD Electromagnetic Fields (Cylindrical TE Mode, 40 MHz)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_fields.png'))
    fig.savefig(os.path.join(out_dir, 'fig_E_theta_fields.pdf'))
    plt.close(fig)
    print("  Fig: E_theta fields")


def fig_B_fields(B_r, B_z, mesh, inside, tel, out_dir):
    """Fig: Magnetic field components (full reactor)."""
    Nr, Nz = mesh.Nr, mesh.Nz
    Br = B_r[:Nr, :Nz] if B_r.shape[0] >= Nr else B_r
    Bz = B_z[:Nr, :Nz] if B_z.shape[0] >= Nr else B_z
    B_mag = np.sqrt(Br**2 + Bz**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    for ax, field, title, cmap, sym in [
        (axes[0], Br, '(a) $B_r$', 'RdBu_r', True),
        (axes[1], Bz, '(b) $B_z$', 'RdBu_r', True),
        (axes[2], B_mag, '(c) $|\\mathbf{B}|$', 'magma', False),
    ]:
        f_full, r_mm = mirror_field(field, mesh)
        i_full = mirror_mask(inside, mesh)
        data = f_full.copy().astype(float)
        data[~i_full] = np.nan
        z_mm = mesh.zc * 1e3
        R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

        if sym:
            abs_max = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-10)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im = ax.pcolormesh(R, Z, data, cmap=cmap, norm=norm,
                               shading='auto', rasterized=True)
        else:
            im = ax.pcolormesh(R, Z, data, cmap=cmap, shading='auto', rasterized=True)

        ax.pcolormesh(R, Z, np.where(~i_full, 1.0, np.nan),
                      cmap='Greys', vmin=0, vmax=2, shading='auto',
                      rasterized=True, alpha=0.3)
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label='T')
        add_geometry_overlay(ax, tel, show_labels=False, show_dims=False)
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('$z$ (mm)')
        ax.set_title(title)
        ax.set_aspect('equal')

    fig.suptitle('Magnetic Field Components', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_B_fields.png'))
    fig.savefig(os.path.join(out_dir, 'fig_B_fields.pdf'))
    plt.close(fig)
    print("  Fig: B fields")


def fig_power_deposition(P_rz, mesh, inside, tel, out_dir):
    """Fig: Power deposition linear and log scale (full reactor)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Linear
    f1 = make_full_reactor_contour(
        P_rz, mesh, inside, tel,
        '(a) Power Deposition $P(r,z)$', 'hot', 'W/m$^3$',
        show_labels=True, show_dims=False, figsize=(7, 9))
    plt.close(f1)

    for ax, log in [(ax1, False), (ax2, True)]:
        f_full, r_mm = mirror_field(P_rz, mesh)
        i_full = mirror_mask(inside, mesh)
        data = f_full.copy().astype(float)
        data[~i_full] = np.nan
        z_mm = mesh.zc * 1e3
        R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

        if log:
            data_pos = np.where(data > 0, data, np.nan)
            valid = data_pos[np.isfinite(data_pos)]
            vmin = max(valid.min(), 1e-2) if len(valid) > 0 else 1e-2
            vmax = valid.max() if len(valid) > 0 else 1e6
            norm = LogNorm(vmin=vmin, vmax=vmax)
            im = ax.pcolormesh(R, Z, data_pos, cmap='hot', norm=norm,
                               shading='auto', rasterized=True)
            plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label='W/m$^3$')
            ax.set_title('(b) $P(r,z)$ — log scale')
        else:
            im = ax.pcolormesh(R, Z, data, cmap='hot', shading='auto', rasterized=True)
            plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label='W/m$^3$')
            ax.set_title('(a) $P(r,z)$ — linear scale')

        ax.pcolormesh(R, Z, np.where(~i_full, 1.0, np.nan),
                      cmap='Greys', vmin=0, vmax=2, shading='auto',
                      rasterized=True, alpha=0.3)
        add_geometry_overlay(ax, tel, show_labels=False, show_dims=False)
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('$z$ (mm)')
        ax.set_aspect('equal')

    P_abs = np.sum(P_rz * mesh.vol * inside)
    eta = P_abs / 700 if P_abs > 0 else 0
    fig.suptitle(f'Power Deposition — $P_{{abs}}$ = {P_abs:.0f} W, $\\eta$ = {eta:.3f}',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_power_deposition.png'))
    fig.savefig(os.path.join(out_dir, 'fig_power_deposition.pdf'))
    plt.close(fig)
    print("  Fig: Power deposition")


def fig_plasma_state(ne, Te, mesh, inside, tel, out_dir):
    """Fig: ne and Te contours (full reactor)."""
    fig = make_full_reactor_contour(
        np.maximum(ne, 1e8), mesh, inside, tel,
        'Electron Density $n_e(r,z)$\n$n_{e,\\mathrm{avg}}$ = '
        f'{np.sum(ne * mesh.vol * inside) / np.sum(mesh.vol * inside):.2e} m$^{{-3}}$',
        cmap='viridis', units='m$^{-3}$', log_scale=True,
        show_labels=True, show_dims=True, figsize=(7, 9.5))
    fig.savefig(os.path.join(out_dir, 'fig_ne_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_ne_contour.pdf'))
    plt.close(fig)

    fig = make_full_reactor_contour(
        Te, mesh, inside, tel,
        'Electron Temperature $T_e(r,z)$',
        cmap='plasma', units='eV',
        show_labels=True, show_dims=False, figsize=(7, 9.5))
    fig.savefig(os.path.join(out_dir, 'fig_Te_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_Te_contour.pdf'))
    plt.close(fig)
    print("  Fig: Plasma state (ne, Te)")


def fig_species_contours(nF, nSF6, mesh, inside, tel, out_dir):
    """Fig: nF and nSF6 full-reactor contours."""
    F_drop = _compute_F_drop(nF, inside)

    fig = make_full_reactor_contour(
        nF, mesh, inside, tel,
        f'Fluorine Density [F]$(r,z)$, [F] drop = {F_drop:.0f}%',
        cmap='Reds', units='m$^{-3}$',
        show_labels=True, show_dims=False, figsize=(7, 9.5))
    fig.savefig(os.path.join(out_dir, 'fig_nF_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_nF_contour.pdf'))
    plt.close(fig)

    fig = make_full_reactor_contour(
        nSF6, mesh, inside, tel,
        'SF$_6$ Density $(r,z)$',
        cmap='Blues', units='m$^{-3}$',
        show_labels=True, show_dims=False, figsize=(7, 9.5))
    fig.savefig(os.path.join(out_dir, 'fig_nSF6_contour.png'))
    fig.savefig(os.path.join(out_dir, 'fig_nSF6_contour.pdf'))
    plt.close(fig)
    print("  Fig: Species contours (nF, nSF6)")


def fig_wafer_profiles(nF, nSF6, ne, Te, mesh, inside, out_dir):
    """Fig: Radial profiles at wafer surface (matching Stage 10 fig2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    r_mm = mesh.rc[inside[:, 0]] * 1e3
    F_w = nF[:, 0][inside[:, 0]] * 1e-6  # cm^-3

    # (a) [F] radial at wafer
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

    # (b) Normalised [F]
    if len(F_w) > 0 and F_w[0] > 0:
        axes[0, 1].plot(r_mm, F_w / F_w[0], 'r-o', lw=2, ms=3, label='Phase 1 model')
        axes[0, 1].axhline(0.26, color='green', ls='--', lw=1.5, label='74% drop (Mettler 2025)')
        axes[0, 1].set_xlabel('$r$ (mm)')
        axes[0, 1].set_ylabel('[F] / [F]$_{\\mathrm{center}}$')
        axes[0, 1].set_title('(b) Normalised [F] at Wafer')
        axes[0, 1].set_ylim(0, 1.15)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

    # (c) ne radial in ICP midplane
    j_mid = np.argmin(np.abs(mesh.zc - mesh.L * 0.6))
    ne_mid = ne[:, j_mid] * 1e-6
    axes[1, 0].plot(mesh.rc * 1e3, ne_mid, 'b-o', lw=2, ms=2)
    axes[1, 0].set_xlabel('$r$ (mm)')
    axes[1, 0].set_ylabel('$n_e$ (cm$^{-3}$)')
    axes[1, 0].set_title(f'(c) $n_e$ Radial Profile at $z$ = {mesh.zc[j_mid]*1e3:.0f} mm')
    axes[1, 0].grid(True, alpha=0.3)

    # (d) SF6 at wafer
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
    print("  Fig: Wafer profiles")


def fig_summary_6panel(E_rms, P_rz, ne, Te, nF, mesh, inside, tel, out_dir):
    """Fig: 6-panel summary (matching fig00_summary style but full-reactor)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fields_info = [
        (E_rms, '$E_\\theta$ RMS [V/m]', 'hot', False, False),
        (P_rz, 'Power Deposition [W/m$^3$]', 'hot', False, False),
        (np.maximum(ne, 1e8), '$n_e$ [m$^{-3}$]', 'viridis', True, False),
        (Te, '$T_e$ [eV]', 'plasma', False, False),
        (nF, '[F] [m$^{-3}$]', 'Reds', False, False),
        (None, None, None, None, None),  # Placeholder for wafer profile
    ]

    for idx, (field, title, cmap, log, _) in enumerate(fields_info):
        if field is None:
            break
        ax = axes[idx // 3, idx % 3]
        f_full, r_mm = mirror_field(field, mesh)
        i_full = mirror_mask(inside, mesh)
        data = f_full.copy().astype(float)
        data[~i_full] = np.nan
        z_mm = mesh.zc * 1e3
        R, Z = np.meshgrid(r_mm, z_mm, indexing='ij')

        if log:
            valid = data[np.isfinite(data) & (data > 0)]
            norm = LogNorm(vmin=valid.min() if len(valid) > 0 else 1e8,
                          vmax=valid.max() if len(valid) > 0 else 1e18)
            im = ax.pcolormesh(R, Z, np.where(data > 0, data, np.nan),
                               cmap=cmap, norm=norm, shading='auto', rasterized=True)
        else:
            im = ax.pcolormesh(R, Z, data, cmap=cmap, shading='auto', rasterized=True)

        ax.pcolormesh(R, Z, np.where(~i_full, 1.0, np.nan),
                      cmap='Greys', vmin=0, vmax=2, shading='auto',
                      rasterized=True, alpha=0.3)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        add_geometry_overlay(ax, tel, show_labels=False, show_dims=False, linewidth=0.8)
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('$z$ (mm)')
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

    F_drop = _compute_F_drop(nF, inside)
    ne_avg = np.sum(ne * mesh.vol * inside) / np.sum(mesh.vol * inside)
    fig.suptitle(
        f'Phase 1 Summary: $P_{{rf}}$ = 700 W, $f$ = 40 MHz, '
        f'$\\eta$ = 0.430, [F] drop = {F_drop:.1f}%, '
        f'$n_{{e,avg}}$ = {ne_avg:.1e} m$^{{-3}}$',
        fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_summary_6panel.png'))
    fig.savefig(os.path.join(out_dir, 'fig_summary_6panel.pdf'))
    plt.close(fig)
    print("  Fig: Summary 6-panel")


def _compute_F_drop(nF, inside):
    Fw = nF[:, 0][inside[:, 0]]
    if len(Fw) < 2 or Fw[0] < 1e6:
        return 0.0
    return (1 - Fw[-1] / Fw[0]) * 100


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate report figures')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Path to run directory with data/ subfolder')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    args = parser.parse_args()

    setup_style()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = SimulationConfig(config_path)
    tel = config.tel_geometry

    # Find latest run dir
    if args.run_dir:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    else:
        runs_base = os.path.join(PROJECT_ROOT, 'results', 'runs')
        runs = sorted([d for d in os.listdir(runs_base) if os.path.isdir(os.path.join(runs_base, d))])
        run_dir = os.path.join(runs_base, runs[-1]) if runs else None
        if not run_dir:
            print("ERROR: No run directories found"); return

    data_dir = os.path.join(run_dir, 'data')
    print(f"Loading data from: {data_dir}")

    # Output directories
    report_fig_dir = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')
    pres_fig_dir = os.path.join(PROJECT_ROOT, 'docs', 'presentation', 'figures')
    os.makedirs(report_fig_dir, exist_ok=True)
    os.makedirs(pres_fig_dir, exist_ok=True)

    # Build mesh and geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, bc_type = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                          tel['L_proc'], tel['L_proc'] + tel['L_apt'], L_total)

    # Load data
    def load(name):
        path = os.path.join(data_dir, f'{name}.npy')
        if os.path.exists(path):
            return np.load(path)
        print(f"  WARNING: {name}.npy not found")
        return None

    E_theta = load('E_theta')
    E_rms = load('E_theta_rms')
    B_r = load('B_r')
    B_z = load('B_z')
    P_rz = load('P_rz')
    ne = load('ne')
    Te = load('Te')
    nF = load('nF')
    nSF6 = load('nSF6')

    # Coil positions
    from dtpm.core.geometry import compute_coil_positions_cylindrical
    coil_pos = compute_coil_positions_cylindrical(config, mesh)

    print(f"\nGenerating figures -> {report_fig_dir}")

    # Generate all figures
    fig_geometry(mesh, inside, bc_type, tel, coil_pos, report_fig_dir)

    if nF is not None:
        fig_cross_section_F(nF, mesh, inside, tel, report_fig_dir)

    if E_theta is not None and E_rms is not None:
        fig_E_theta_fields(E_theta, E_rms, mesh, inside, tel, report_fig_dir)

    if B_r is not None and B_z is not None:
        fig_B_fields(B_r, B_z, mesh, inside, tel, report_fig_dir)

    if P_rz is not None:
        fig_power_deposition(P_rz, mesh, inside, tel, report_fig_dir)

    if ne is not None and Te is not None:
        fig_plasma_state(ne, Te, mesh, inside, tel, report_fig_dir)

    if nF is not None and nSF6 is not None:
        fig_species_contours(nF, nSF6, mesh, inside, tel, report_fig_dir)

    if nF is not None and nSF6 is not None and ne is not None:
        fig_wafer_profiles(nF, nSF6, ne, Te, mesh, inside, report_fig_dir)

    if E_rms is not None and P_rz is not None and ne is not None and nF is not None:
        fig_summary_6panel(E_rms, P_rz, ne, Te, nF, mesh, inside, tel, report_fig_dir)

    # Copy PNGs to presentation/figures
    import shutil
    for f in os.listdir(report_fig_dir):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(report_fig_dir, f), os.path.join(pres_fig_dir, f))

    print(f"\nAll figures saved to {report_fig_dir}")
    print(f"PNG copies in {pres_fig_dir}")


if __name__ == '__main__':
    main()
