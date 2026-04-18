"""
TEL Reactor Publication Figure Generation

All figures are generated directly from the unified solver's masked field.
No stitching, no separate interpolation of ICP and processing regions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
import os


def plot_cross_section(result, solver, savepath, title=None):
    """Full cross-section [F] contour using imshow RGBA rendering.
    
    The field comes from one unified solve on the masked domain.
    The display is mirrored for visual clarity (solver is half-domain only).
    """
    m = result['mesh']
    nF = result['nF'] * 1e-6  # convert to cm^-3
    ins = result['inside']

    # Log-scale the active cells
    F_log = np.full_like(nF, np.nan)
    F_log[ins & (nF > 0)] = np.log10(nF[ins & (nF > 0)])

    # Extend to r=0 using symmetry (avoids NaN at axis)
    r_ext = np.concatenate([[0.0], m.rc * 1000])
    F_ext = np.concatenate([F_log[0:1, :], F_log], axis=0)

    # Fine display grid
    Nr2, Nz2 = 500, 700
    r_half = np.linspace(0, m.R * 1000, Nr2)
    z_fine = np.linspace(0, m.L * 1000, Nz2)

    interp = RegularGridInterpolator(
        (r_ext, m.zc * 1000), F_ext, method='linear',
        bounds_error=False, fill_value=np.nan)

    F_disp = np.full((Nz2, Nr2), np.nan)
    for i in range(Nr2):
        for j in range(Nz2):
            F_disp[j, i] = interp((r_half[i], z_fine[j]))

    # Mirror without duplicating r=0
    r_left = -r_half[1:][::-1]
    r_full = np.concatenate([r_left, r_half])
    F_full = np.concatenate([F_disp[:, 1:][:, ::-1], F_disp], axis=1)

    # RGBA rendering
    cmap = plt.cm.inferno.copy()
    norm = Normalize(vmin=12.0, vmax=14.8)
    rgba = cmap(norm(np.nan_to_num(F_full, nan=0.0)))
    rgba[np.isnan(F_full), 3] = 0.0

    fig, ax = plt.subplots(figsize=(8, 12))

    Ri = solver.R_icp * 1000
    Rp = solver.R_proc * 1000
    zAb = solver.z_apt_bot * 1000
    zAt = solver.z_apt_top * 1000
    zT = solver.z_top * 1000

    ax.imshow(rgba, origin='lower', aspect='auto',
              extent=[r_full[0], r_full[-1], z_fine[0], z_fine[-1]],
              interpolation='bilinear')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='$\\log_{10}$([F] / cm$^{-3}$)',
                 shrink=0.65, pad=0.03, aspect=30)

    # Reactor outline
    for s in [-1, 1]:
        ax.plot([s*Rp, s*Ri, s*Ri, s*Ri, s*Rp, s*Rp],
                [zT, zT, zAt, zAb, zAb, 0],
                color='#bbbbbb', lw=1.8, solid_joinstyle='miter', zorder=6)
        ax.fill_betweenx([zAb, zAt], s*Ri, s*Rp,
                         color='#1a1a1a', alpha=0.85, zorder=5)
        ax.fill_betweenx([zT, zT+3], s*Ri, s*Rp,
                         color='#1a1a1a', alpha=0.7, zorder=5)
    ax.fill_betweenx([-3, 0], -Rp, Rp, color='#111111', alpha=0.8, zorder=5)
    ax.plot([-75, 75], [0, 0], color='#dddddd', lw=2.5, zorder=6)
    ax.plot([-Ri, Ri], [zT, zT], color='cyan', lw=2.5, zorder=6)

    # Labels
    ax.text(0, (zAt+zT)/2, 'ICP SOURCE', ha='center', va='center',
            fontsize=13, color='white', fontweight='bold', zorder=7)
    ax.text(0, zAb/2, 'PROCESSING', ha='center', va='center',
            fontsize=12, color='white', fontweight='bold', zorder=7)
    ax.annotate('', xy=(Ri+6, zAt), xytext=(Ri+6, zT),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.2), zorder=7)
    ax.text(Ri+9, (zAt+zT)/2, '181.5 mm', color='white', fontsize=9, va='center')
    ax.annotate('', xy=(Rp+4, 0), xytext=(Rp+4, zAb),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.2), zorder=7)
    ax.text(Rp+7, zAb/2, '50 mm', color='white', fontsize=9, va='center')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'TEL Reactor: [F] Density, drop = {result["F_drop_pct"]:.0f}%')
    ax.set_xlabel('$r$ (mm)')
    ax.set_ylabel('$z$ (mm)')
    ax.set_xlim(-Rp-15, Rp+15)
    ax.set_ylim(-6, zT+5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    fig.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {savepath}")


def plot_radial_profile(result, solver, mettler_data, savepath):
    """Radial [F] at wafer vs Mettler Fig 4.14."""
    fig, ax = plt.subplots(figsize=(7, 5))
    Fn = result['F_wafer'] / result['F_wafer'][0]
    rn = result['r_wafer'] * 100 / (solver.R_proc * 100)

    if mettler_data is not None:
        ax.plot(mettler_data[:, 0] / (solver.R_proc * 100), mettler_data[:, 1],
                'ko', ms=10, mfc='none', mew=2, zorder=5, label='Mettler (Fig 4.14)')
    ax.plot(rn, Fn, 'r-', lw=2.5,
            label=f'Model ({result["F_drop_pct"]:.0f}% drop)')

    ax.set_xlabel('$r / R_{\\mathrm{proc}}$')
    ax.set_ylabel('[F] / [F]$_{r=0}$')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.text(0.5, 0.3,
            f'Model: {result["F_drop_pct"]:.0f}%\nMettler: 74%\n'
            f'$\\gamma_{{Al}}$ = {solver.gamma_Al:.2f}',
            transform=ax.transAxes, fontsize=10, ha='center', va='center',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath}")


def plot_geometry_mask(solver, savepath):
    """Visualize the geometry mask and boundary classification."""
    m = solver.mesh
    bc = solver.bc_type.copy().astype(float)
    bc[~solver.inside] = -1

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # (a) Active/inactive
    ax = axes[0]
    ax.pcolormesh(m.rc*1000, m.zc*1000, solver.inside.T, cmap='RdYlGn', shading='auto')
    ax.set_xlabel('$r$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title('(a) Geometry mask (green=active)')
    ax.set_aspect('equal')

    # (b) Boundary classification
    ax = axes[1]
    c = ax.pcolormesh(m.rc*1000, m.zc*1000, bc.T, cmap='tab10', shading='auto',
                      vmin=-1, vmax=8)
    plt.colorbar(c, ax=ax, label='BC type')
    ax.set_xlabel('$r$ (mm)'); ax.set_ylabel('$z$ (mm)')
    ax.set_title('(b) Boundary classification')
    ax.set_aspect('equal')

    plt.suptitle('TEL Geometry: Masked Domain', fontsize=14)
    plt.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath}")
