#!/usr/bin/env python3
"""
Generate power sweep and pressure sweep GIF animations.

Runs the full 9-species simulation at multiple operating points and
creates animated GIF files showing how species distributions change.

Similar to Stage 10's neutrals_2D_power_sweep.gif and power_sweep_9species.gif.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import (SimulationConfig, Mesh2D, build_geometry_mask,
                        build_index_maps, compute_coil_positions_cylindrical,
                        build_em_active_mask)
from dtpm.core.grid import Grid
from dtpm.modules import m01_circuit, m06_fdtd_cylindrical, m11_plasma_chemistry

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}
SPECIES_ORDER = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']


def interpolate_and_mirror(field, mesh, inside, Nr_fine=300, Nz_fine=400):
    """Interpolate to fine grid and mirror for full reactor."""
    f = field.copy().astype(float)
    f[~inside] = np.nan
    r_ext = np.concatenate([[0.0], mesh.rc * 1e3])
    f_ext = np.concatenate([f[0:1, :], f], axis=0)
    r_half = np.linspace(0, mesh.R * 1e3, Nr_fine)
    z_fine = np.linspace(0, mesh.L * 1e3, Nz_fine)
    interp = RegularGridInterpolator(
        (r_ext, mesh.zc * 1e3), f_ext, method='linear',
        bounds_error=False, fill_value=np.nan)
    rr, zz = np.meshgrid(r_half, z_fine, indexing='ij')
    f_fine = interp(np.column_stack([rr.ravel(), zz.ravel()])).reshape(Nr_fine, Nz_fine)
    r_left = -r_half[1:][::-1]
    r_full = np.concatenate([r_left, r_half])
    f_full = np.concatenate([f_fine[::-1, :][1:, :], f_fine], axis=0)
    return f_full.T, r_full, z_fine


def run_at_power(power, config_path):
    """Run the simulation at a specific power level and return results."""
    import yaml
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Override power
    cfg_dict['circuit']['source_power'] = power

    # Write temp config
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg_dict, tmp, default_flow_style=False)
    tmp.close()

    config = SimulationConfig(tmp.name)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']

    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, bc_type = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                          tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)
    ij_to_flat, flat_to_ij, n_active = build_index_maps(inside)

    grid = Grid(config)
    state = {
        'mesh': mesh, 'inside': inside, 'bc_type': bc_type,
        'ij_to_flat': ij_to_flat, 'flat_to_ij': flat_to_ij,
        'n_active': n_active,
        'nx': grid.nx, 'ny': grid.ny, 'dx': grid.dx, 'dy': grid.dy,
        'coil_centers_idx': compute_coil_positions_cylindrical(config, mesh),
        'coil_radius': config.geometry.get('coil_radius', 1e-3),
        'coil_radius_idx': 1,
    }

    # M01
    m01_result = m01_circuit.run(state, config)
    state.update(m01_result)

    # M06c
    m06c_result = m06_fdtd_cylindrical.run(state, config)
    state.update(m06c_result)

    # M11
    import logging
    logging.getLogger('dtpm').setLevel(logging.WARNING)
    m11_result = m11_plasma_chemistry.run(state, config)
    state.update(m11_result)

    os.unlink(tmp.name)
    return state, mesh, inside, tel


def gen_power_sweep_gif(powers, config_path, out_dir):
    """Generate a GIF showing F, SF6, ne cross-sections at different powers."""
    frames = []
    temp_dir = os.path.join(out_dir, '_temp_sweep')
    os.makedirs(temp_dir, exist_ok=True)

    all_results = []
    for p in powers:
        print(f"  Running P = {p} W...")
        state, mesh, inside, tel = run_at_power(p, config_path)
        all_results.append((p, state, mesh, inside, tel))

    R_icp = tel['R_icp'] * 1e3
    R_proc = tel['R_proc'] * 1e3
    z_apt = tel['L_proc'] * 1e3
    z_top = (tel['L_proc'] + tel['L_apt'] + tel['L_icp']) * 1e3

    for frame_idx, (p, state, mesh, inside, tel) in enumerate(all_results):
        species_fields = state.get('species_fields', {})
        nF = species_fields.get('F', state.get('nF', np.zeros((mesh.Nr, mesh.Nz))))

        # [F] cross-section
        F_cm3 = nF * 1e-6
        log_F = np.where((F_cm3 > 0) & inside, np.log10(F_cm3), np.nan)
        f_disp, r_mm, z_mm = interpolate_and_mirror(log_F, mesh, inside)

        cmap = plt.cm.inferno.copy()
        norm = Normalize(vmin=11.5, vmax=15.0)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=11.5)))
        rgba[np.isnan(f_disp), 3] = 0.0

        fig, ax = plt.subplots(figsize=(7, 9))
        ax.set_facecolor('black')
        ax.imshow(rgba, origin='lower', aspect='auto',
                  extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                  interpolation='bilinear')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='log$_{10}$ [F] / cm$^{-3}$',
                     shrink=0.7, pad=0.03)

        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='white', lw=1.5)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='white', lw=1.5)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='white', lw=1.5)
        ax.plot([-R_icp, R_icp], [z_top, z_top], color='cyan', lw=2)

        # F drop
        Fw = nF[:, 0][inside[:, 0]]
        drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0

        ax.set_title(f'[F]$(r,z)$ — $P_{{rf}}$ = {p} W, 10 mTorr\n'
                     f'[F] drop = {drop:.0f}%',
                     fontsize=14, color='white', pad=10)
        ax.set_xlabel('$r$ (mm)', color='white')
        ax.set_ylabel('$z$ (mm)', color='white')
        ax.tick_params(colors='white')
        ax.set_xlim(-R_proc - 10, R_proc + 5)
        ax.set_ylim(-5, z_top + 8)
        ax.set_aspect('equal')

        fname = os.path.join(temp_dir, f'frame_{frame_idx:02d}.png')
        fig.savefig(fname, facecolor='black', dpi=150)
        plt.close(fig)
        frames.append(imageio.imread(fname))

    gif_path = os.path.join(out_dir, 'power_sweep_F_density.gif')
    imageio.mimsave(gif_path, frames, duration=2.0, loop=0)
    print(f"  GIF: {gif_path} ({len(frames)} frames)")

    # Also create a 9-species power sweep GIF
    species_frames = []
    for frame_idx, (p, state, mesh, inside, tel) in enumerate(all_results):
        species_fields = state.get('species_fields', {})

        fig, ax = plt.subplots(figsize=(10, 7))
        r_mm = mesh.rc[inside[:, 0]] * 1e3

        for sp in SPECIES_ORDER:
            field = species_fields.get(sp)
            if field is None:
                continue
            profile = field[:, 0][inside[:, 0]] * 1e-6
            if np.any(profile > 0):
                ax.semilogy(r_mm, np.maximum(profile, 1e6), '-o', lw=2, ms=3,
                           markevery=3, label=SP_LABEL[sp])

        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('Density (cm$^{-3}$)')
        ax.set_title(f'All Species at Wafer — $P_{{rf}}$ = {p} W, 10 mTorr')
        ax.legend(ncol=3, loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, r_mm[-1] + 2)
        ax.set_ylim(1e10, 1e15)

        fname = os.path.join(temp_dir, f'species_frame_{frame_idx:02d}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        species_frames.append(imageio.imread(fname))

    gif_path2 = os.path.join(out_dir, 'power_sweep_9species.gif')
    imageio.mimsave(gif_path2, species_frames, duration=2.0, loop=0)
    print(f"  GIF: {gif_path2} ({len(species_frames)} frames)")

    import shutil
    shutil.rmtree(temp_dir)


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'animations')
    os.makedirs(out_dir, exist_ok=True)

    powers = [300, 500, 700, 900, 1200]
    print(f"Power sweep: {powers} W")
    print(f"Each run takes ~55s, total ~{len(powers)*55//60} min\n")

    gen_power_sweep_gif(powers, config_path, out_dir)

    print(f"\nAll sweep animations saved to {out_dir}")


if __name__ == '__main__':
    main()
