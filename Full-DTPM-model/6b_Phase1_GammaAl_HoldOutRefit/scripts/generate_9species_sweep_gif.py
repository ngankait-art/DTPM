#!/usr/bin/env python3
"""
Generate Stage 10-style 9-species power sweep GIF.

Each frame: 3×3 grid of ALL 9 neutral species at one power level.
Animation sweeps through power levels.

Matches Stage 10's neutrals_2D_power_sweep.gif exactly:
  - 3×3 grid: SF6, SF5, SF4 / SF3, SF2, SF / S, F, F2
  - Full reactor cross-section (mirrored)
  - Log scale colorbar per species
  - Title shows power and [F] drop
"""

import os
import sys
import numpy as np
import yaml
import tempfile

import matplotlib
matplotlib.use('Agg')
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

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}
GRID_ORDER = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']


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


def make_9species_frame(state, mesh, inside, tel, power, frame_path):
    """Create one frame: 3×3 grid of all 9 species at given power."""
    species_fields = state.get('species_fields', {})

    R_icp = tel['R_icp'] * 1e3
    R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    # Compute F drop
    nF = species_fields.get('F', state.get('nF', np.zeros((mesh.Nr, mesh.Nz))))
    Fw = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0

    cmap = plt.cm.inferno.copy()

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    for idx, sp in enumerate(GRID_ORDER):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        field = species_fields.get(sp, np.zeros((mesh.Nr, mesh.Nz)))
        F_cm3 = field * 1e-6
        log_f = np.where((F_cm3 > 0) & inside, np.log10(F_cm3), np.nan)

        f_disp, r_mm, z_mm = interp_mirror(log_f, mesh, inside)

        valid = f_disp[np.isfinite(f_disp)]
        vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 9
        vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 15

        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
        rgba[np.isnan(f_disp), 3] = 0

        ax.set_facecolor('#e8e8e8')
        ax.imshow(rgba, origin='lower', aspect='auto',
                 extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                 interpolation='bilinear')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.85, pad=0.02, aspect=20)
        cb.ax.tick_params(labelsize=7)

        # Geometry outline
        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='gray', lw=0.8)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='gray', lw=0.8)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='gray', lw=0.8)

        ax.set_title(f'[{SP_LABEL[sp]}]', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

        if col == 0:
            ax.set_ylabel('$z$ (mm)', fontsize=9)
        else:
            ax.set_yticklabels([])
        if row == 2:
            ax.set_xlabel('$r$ (mm)', fontsize=9)
        else:
            ax.set_xticklabels([])

    fig.suptitle(f'Neutrals — P={power}W, p=10mTorr, drop={drop:.0f}%',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(frame_path, dpi=120, facecolor='white')
    plt.close(fig)


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'animations')
    os.makedirs(out_dir, exist_ok=True)

    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
    print(f"Running 9-species power sweep: {powers} W")
    print(f"~55s per point × {len(powers)} = ~{len(powers)*55//60} min\n")

    temp_dir = os.path.join(out_dir, '_temp_9sp')
    os.makedirs(temp_dir, exist_ok=True)

    frame_paths = []
    for i, p in enumerate(powers):
        print(f"  [{i+1}/{len(powers)}] P = {p} W...")
        state, mesh, inside, tel = run_at_power(p, config_path)
        frame_path = os.path.join(temp_dir, f'frame_{i:02d}.png')
        make_9species_frame(state, mesh, inside, tel, p, frame_path)
        frame_paths.append(frame_path)

    # Create GIF
    frames = [Image.open(fp) for fp in frame_paths]
    gif_path = os.path.join(out_dir, 'neutrals_2D_power_sweep.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=2000, loop=0)
    print(f"\n  GIF saved: {gif_path} ({len(frames)} frames)")

    # Also save the 700W frame as the static figure
    import shutil
    idx_700 = powers.index(700) if 700 in powers else len(powers) // 2
    shutil.copy2(frame_paths[idx_700],
                 os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures',
                              'fig_9species_700W.png'))

    # Cleanup
    shutil.rmtree(temp_dir)
    print("Done!")


if __name__ == '__main__':
    main()
