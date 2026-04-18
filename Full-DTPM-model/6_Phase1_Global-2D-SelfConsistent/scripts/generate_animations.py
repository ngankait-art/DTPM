#!/usr/bin/env python3
"""
Generate GIF animations for the documentation package.

1. Species tour: animate through all 9 neutral species as frames
2. Z-slice tour: animate radial profiles at different z heights
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from matplotlib import patheffects
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask

SP_LABEL = {'SF6': 'SF$_6$', 'SF5': 'SF$_5$', 'SF4': 'SF$_4$',
            'SF3': 'SF$_3$', 'SF2': 'SF$_2$', 'SF': 'SF',
            'F': 'F', 'F2': 'F$_2$', 'S': 'S'}

SPECIES_ORDER = ['F', 'SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F2']


def interpolate_field(field, mesh, inside, Nr_fine=400, Nz_fine=600):
    """Interpolate to fine grid and mirror."""
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


def gen_species_tour_gif(fields, mesh, inside, tel, out_dir):
    """Animate through all 9 species as frames in a GIF."""
    frames = []
    temp_dir = os.path.join(out_dir, '_temp_frames')
    os.makedirs(temp_dir, exist_ok=True)

    R_icp = tel.get('R_icp', 0.038) * 1e3
    R_proc = tel.get('R_proc', 0.105) * 1e3
    z_apt = tel.get('L_proc', 0.050) * 1e3
    z_top = (tel.get('L_proc', 0.050) + tel.get('L_apt', 0.002) + tel.get('L_icp', 0.150)) * 1e3

    for i, sp in enumerate(SPECIES_ORDER):
        if sp not in fields:
            continue

        field_cm3 = fields[sp] * 1e-6
        log_f = np.where((field_cm3 > 0) & inside, np.log10(field_cm3), np.nan)

        f_disp, r_mm, z_mm = interpolate_field(log_f, mesh, inside)

        valid = f_disp[np.isfinite(f_disp)]
        vmin = np.nanpercentile(valid, 2) if len(valid) > 0 else 8
        vmax = np.nanpercentile(valid, 99) if len(valid) > 0 else 15

        cmap = plt.cm.inferno.copy()
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(np.nan_to_num(f_disp, nan=vmin)))
        rgba[np.isnan(f_disp), 3] = 0.0

        fig, ax = plt.subplots(figsize=(7, 9))
        ax.set_facecolor('black')
        ax.imshow(rgba, origin='lower', aspect='auto',
                  extent=[r_mm[0], r_mm[-1], z_mm[0], z_mm[-1]],
                  interpolation='bilinear')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=f'log$_{{10}}$ [{SP_LABEL[sp]}] / cm$^{{-3}}$',
                     shrink=0.7, pad=0.03)

        # Geometry outline
        for s in [-1, 1]:
            ax.plot([s*R_icp, s*R_icp], [z_apt, z_top], color='white', lw=1.5)
            ax.plot([s*R_proc, s*R_proc], [0, z_apt], color='white', lw=1.5)
            ax.plot([s*R_icp, s*R_proc], [z_apt, z_apt], color='white', lw=1.5)
        ax.plot([-R_icp, R_icp], [z_top, z_top], color='cyan', lw=2)

        label = SP_LABEL[sp]
        ax.set_title(f'[{label}]$(r,z)$ — 700 W, 10 mTorr, pure SF$_6$\n'
                     f'Species {i+1} of {len(SPECIES_ORDER)}',
                     fontsize=13, color='white', pad=10)
        ax.set_xlabel('$r$ (mm)', color='white')
        ax.set_ylabel('$z$ (mm)', color='white')
        ax.tick_params(colors='white')
        ax.set_xlim(-R_proc - 10, R_proc + 5)
        ax.set_ylim(-5, z_top + 8)
        ax.set_aspect('equal')

        fname = os.path.join(temp_dir, f'frame_{i:02d}.png')
        fig.savefig(fname, facecolor='black', dpi=150)
        plt.close(fig)
        frames.append(imageio.imread(fname))

    # Create GIF
    gif_path = os.path.join(out_dir, 'species_tour_9species.gif')
    imageio.mimsave(gif_path, frames, duration=1.5, loop=0)
    print(f"  GIF: {gif_path} ({len(frames)} frames)")

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir)


def gen_z_slice_animation(nF, mesh, inside, out_dir):
    """Animate radial [F] profiles at different z heights."""
    frames = []
    temp_dir = os.path.join(out_dir, '_temp_zslice')
    os.makedirs(temp_dir, exist_ok=True)

    z_indices = list(range(0, mesh.Nz, 2))  # every 2nd z index

    for frame_idx, j in enumerate(z_indices):
        fig, ax = plt.subplots(figsize=(8, 5))

        r_mm = mesh.rc * 1e3
        F_profile = nF[:, j] * 1e-6  # cm^-3

        ax.semilogy(r_mm, np.maximum(F_profile, 1e6), 'r-o', lw=2, ms=3)
        ax.set_xlabel('$r$ (mm)')
        ax.set_ylabel('[F] (cm$^{-3}$)')
        ax.set_title(f'[F] Radial Profile at $z$ = {mesh.zc[j]*1e3:.1f} mm')
        ax.set_xlim(0, mesh.R * 1e3 + 2)
        ax.set_ylim(1e10, 1e15)
        ax.grid(True, alpha=0.3)

        # Progress bar
        progress = frame_idx / max(len(z_indices) - 1, 1)
        ax.axhline(y=1e10, xmin=0, xmax=progress, color='blue', lw=4, alpha=0.5)

        fname = os.path.join(temp_dir, f'frame_{frame_idx:03d}.png')
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        frames.append(imageio.imread(fname))

    gif_path = os.path.join(out_dir, 'F_radial_z_sweep.gif')
    imageio.mimsave(gif_path, frames, duration=0.15, loop=0)
    print(f"  GIF: {gif_path} ({len(frames)} frames)")

    import shutil
    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    args = parser.parse_args()

    import argparse as _  # already imported above

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
    anim_dir = os.path.join(PROJECT_ROOT, 'animations')
    os.makedirs(anim_dir, exist_ok=True)

    print(f"Loading data from: {data_dir}")

    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _ = build_geometry_mask(mesh, tel['R_icp'], tel['R_proc'],
                                     tel['L_proc'], tel['L_proc']+tel['L_apt'], L_total)

    fields = {}
    for sp in SPECIES_ORDER:
        path = os.path.join(data_dir, f'n{sp}.npy')
        if os.path.exists(path):
            fields[sp] = np.load(path)

    nF = fields.get('F', np.load(os.path.join(data_dir, 'nF.npy')))

    print(f"Loaded {len(fields)} species\n")

    gen_species_tour_gif(fields, mesh, inside, tel, anim_dir)
    gen_z_slice_animation(nF, mesh, inside, anim_dir)

    print(f"\nAll animations saved to {anim_dir}")


if __name__ == '__main__':
    import argparse
    main()
