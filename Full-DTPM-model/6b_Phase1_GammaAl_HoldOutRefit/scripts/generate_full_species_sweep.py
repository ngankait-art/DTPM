#!/usr/bin/env python3
"""
Generate Stage 10-style FULL species transport power sweep GIF.

Each frame: 4×5 grid showing ALL neutrals + charged species + ne + alpha.
  Row 1: SF6, SF5, SF4, SF3, SF2
  Row 2: SF, S, F, F2, ne
  Row 3: SF5+, SF3+, SF4+, F+, n+ (total)
  Row 4: F-, SF6-, SF5-, SF4-, alpha

Title: Full Species Transport — P=XXX W, p=10 mTorr, [F] drop=XX%, ne=X.Xe+XX cm^-3

Matches Stage 10's power_sweep_9species.gif exactly.
"""

import os, sys, numpy as np, yaml, tempfile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask, build_index_maps
from dtpm.core.grid import Grid
from dtpm.core.geometry import compute_coil_positions_cylindrical
from dtpm.modules import m01_circuit, m06_fdtd_cylindrical, m11_plasma_chemistry


def interp_mirror(field, mesh, inside, Nr=180, Nz=250):
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


def run_at_power(power, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['circuit']['source_power'] = power
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp, default_flow_style=False); tmp.close()
    config = SimulationConfig(tmp.name)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, bc = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                     tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)
    ijf, fij, na = build_index_maps(inside)
    grid = Grid(config)
    state = {'mesh': mesh, 'inside': inside, 'bc_type': bc,
             'ij_to_flat': ijf, 'flat_to_ij': fij, 'n_active': na,
             'nx': grid.nx, 'ny': grid.ny, 'dx': grid.dx, 'dy': grid.dy,
             'coil_centers_idx': compute_coil_positions_cylindrical(config, mesh),
             'coil_radius': config.geometry.get('coil_radius', 1e-3), 'coil_radius_idx': 1}
    import logging; logging.getLogger('dtpm').setLevel(logging.WARNING)
    state.update(m01_circuit.run(state, config))
    state.update(m06_fdtd_cylindrical.run(state, config))
    state.update(m11_plasma_chemistry.run(state, config))
    os.unlink(tmp.name)
    return state, mesh, inside, tel


def plot_panel(ax, field, mesh, inside, tel, title, cmap_name='inferno', log=True):
    """Plot one species panel in the grid."""
    R_icp = tel['R_icp'] * 1e3; R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    if log:
        plot_f = np.where((field > 0) & inside, np.log10(field), np.nan)
    else:
        plot_f = np.where(inside, field, np.nan)

    f_disp, r_mm, z_mm = interp_mirror(plot_f, mesh, inside)

    valid = f_disp[np.isfinite(f_disp)]
    if len(valid) == 0:
        ax.set_facecolor('#e8e8e8')
        ax.set_title(title, fontsize=8, fontweight='bold')
        return

    vmin = np.nanpercentile(valid, 2)
    vmax = np.nanpercentile(valid, 99)
    if vmax <= vmin:
        vmax = vmin + 1

    cmap = plt.cm.get_cmap(cmap_name).copy()
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
    rgba[np.isnan(f_disp), 3] = 0

    ax.set_facecolor('#e8e8e8')
    ax.imshow(rgba, origin='lower', aspect='auto',
             extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
             interpolation='bilinear')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=15)
    cb.ax.tick_params(labelsize=6)

    for s in [-1, 1]:
        ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='gray', lw=0.6)
        ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='gray', lw=0.6)
        ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='gray', lw=0.6)

    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)


def make_full_frame(state, mesh, inside, tel, power, frame_path):
    """Create one frame: 4×5 grid of ALL species (neutrals + charged)."""
    species = state.get('species_fields', {})
    ions = state.get('ions', {})
    ne = state.get('ne', np.zeros((mesh.Nr, mesh.Nz)))

    # F drop and ne
    nF = species.get('F', state.get('nF', np.zeros((mesh.Nr, mesh.Nz))))
    Fw = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0
    ne_avg = np.sum(ne[inside] * mesh.vol[inside]) / np.sum(mesh.vol[inside]) * 1e-6  # cm^-3

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))

    # Row 1: SF6, SF5, SF4, SF3, SF2
    row1 = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2']
    labels1 = ['[SF$_6$]', '[SF$_5$]', '[SF$_4$]', '[SF$_3$]', '[SF$_2$]']
    for i, (sp, lab) in enumerate(zip(row1, labels1)):
        f = species.get(sp, np.zeros((mesh.Nr, mesh.Nz))) * 1e-6
        plot_panel(axes[0, i], f, mesh, inside, tel, lab, 'inferno')

    # Row 2: SF, S, F, F2, ne
    row2_sp = ['SF', 'S', 'F', 'F2']
    labels2 = ['[SF]', '[S]', '[F]', '[F$_2$]']
    for i, (sp, lab) in enumerate(zip(row2_sp, labels2)):
        f = species.get(sp, np.zeros((mesh.Nr, mesh.Nz))) * 1e-6
        plot_panel(axes[1, i], f, mesh, inside, tel, lab, 'inferno')
    # ne
    plot_panel(axes[1, 4], ne * 1e-6, mesh, inside, tel, '$n_e$', 'viridis')

    # Row 3: positive ions: SF5+, SF3+, SF4+, F+, n+ total
    pos_ions = ['SF5+', 'SF3+', 'SF4+', 'F+', 'n+']
    pos_labels = ['SF$_5^+$', 'SF$_3^+$', 'SF$_4^+$', 'F$^+$', '$n_+$ total']
    for i, (ion, lab) in enumerate(zip(pos_ions, pos_labels)):
        f = ions.get(ion, np.zeros((mesh.Nr, mesh.Nz))) * 1e-6
        plot_panel(axes[2, i], f, mesh, inside, tel, lab, 'Reds')

    # Row 4: negative ions: F-, SF6-, SF5-, SF4-, alpha
    neg_ions = ['F-', 'SF6-', 'SF5-', 'SF4-']
    neg_labels = ['F$^-$', 'SF$_6^-$', 'SF$_5^-$', 'SF$_4^-$']
    for i, (ion, lab) in enumerate(zip(neg_ions, neg_labels)):
        f = ions.get(ion, np.zeros((mesh.Nr, mesh.Nz))) * 1e-6
        plot_panel(axes[3, i], f, mesh, inside, tel, lab, 'Blues')
    # alpha
    alpha = ions.get('alpha', np.zeros((mesh.Nr, mesh.Nz)))
    plot_panel(axes[3, 4], alpha, mesh, inside, tel, '$\\alpha = n_-/n_e$', 'plasma', log=False)

    # Remove axis labels except edges
    for row in range(4):
        for col in range(5):
            if col > 0:
                axes[row, col].set_yticklabels([])
            else:
                axes[row, col].set_ylabel('$z$ (mm)', fontsize=7)
            if row < 3:
                axes[row, col].set_xticklabels([])
            else:
                axes[row, col].set_xlabel('$r$ (mm)', fontsize=7)

    # Row labels on left
    row_labels = [
        'Row 1-2: 9 neutrals + $n_e$',
        '',
        'Row 3: positive ions + $n_+$',
        'Row 4: negative ions + $\\alpha$'
    ]

    fig.suptitle(
        f'Full Species Transport — $P_{{rf}}$ = {power} W, $p$ = 10 mTorr, '
        f'[F] drop = {drop:.0f}%, $n_e$ = {ne_avg:.1e} cm$^{{-3}}$\n'
        f'Row 1-2: 9 neutrals + $n_e$  |  Row 3: positive ions + $n_+$  |  '
        f'Row 4: negative ions + $\\alpha$',
        fontsize=12, fontweight='bold', y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(frame_path, dpi=110, facecolor='white')
    plt.close(fig)


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'animations')
    fig_dir = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    powers = [200, 300, 500, 700, 900, 1200]
    print(f"Full species power sweep: {powers} W")
    print(f"~55s per point × {len(powers)} = ~{len(powers)*55//60} min\n")

    temp_dir = os.path.join(out_dir, '_temp_full')
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, p in enumerate(powers):
        print(f"  [{i+1}/{len(powers)}] P = {p} W...")
        state, mesh, inside, tel = run_at_power(p, config_path)
        frame_path = os.path.join(temp_dir, f'frame_{i:02d}.png')
        make_full_frame(state, mesh, inside, tel, p, frame_path)
        frame_paths.append(frame_path)

    # Create GIF
    frames = [Image.open(fp) for fp in frame_paths]
    gif_path = os.path.join(out_dir, 'power_sweep_full_species.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=2500, loop=0)
    print(f"\n  GIF saved: {gif_path} ({len(frames)} frames)")

    # Also create charged-only GIF (rows 3-4 extracted)
    # And save the 700W frame as static figure
    import shutil
    idx_700 = powers.index(700) if 700 in powers else len(powers) // 2
    shutil.copy2(frame_paths[idx_700],
                 os.path.join(fig_dir, 'fig_full_species_700W.png'))
    print(f"  Static figure: fig_full_species_700W.png")

    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == '__main__':
    main()
