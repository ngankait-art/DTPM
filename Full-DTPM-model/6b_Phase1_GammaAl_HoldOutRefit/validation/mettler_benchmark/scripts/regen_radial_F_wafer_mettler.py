#!/usr/bin/env python3
"""Regenerate the existing fig_radial_F_wafer_mettler figure with the
corrected Mettler Fig 4.14 data (0.942, 0.761, 0.500, 0.194) and cubic fit,
after the fix applied to generate_stage10_figures.py. Picks up the cached
power-sweep results at 700 W from the Phase-1 sweep directory.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE5A_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
STEPS_DIR = os.path.abspath(os.path.join(STAGE5A_ROOT, '..'))
PHASE1_ROOT = os.path.join(STEPS_DIR, '5.Phase1_EM_Chemistry_Merged')

sys.path.insert(0, os.path.join(PHASE1_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PHASE1_ROOT, 'src'))

import numpy as np  # noqa: E402
from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask  # noqa: E402


def _build_mesh_and_inside():
    cfg_path = os.path.join(PHASE1_ROOT, 'config', 'default_config.yaml')
    config = SimulationConfig(cfg_path)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _ = build_geometry_mask(
        mesh, tel['R_icp'], tel['R_proc'],
        tel['L_proc'], tel['L_proc'] + tel['L_apt'], L_total,
    )
    return mesh, inside


def _load_power_sweep():
    """Load every power sweep point as a list of dicts compatible with the
    generate_stage10_figures.py gen_radial_F_wafer_mettler signature."""
    sweep_dir = os.path.join(PHASE1_ROOT, 'results', 'sweeps', 'power')
    out = []
    for entry in sorted(os.listdir(sweep_dir)):
        p_dir = os.path.join(sweep_dir, entry)
        summary_path = os.path.join(p_dir, 'summary.json')
        nF_path = os.path.join(p_dir, 'nF.npy')
        if not (os.path.isfile(summary_path) and os.path.isfile(nF_path)):
            continue
        import json
        with open(summary_path) as f:
            s = json.load(f)
        nF = np.load(nF_path)
        out.append({'value': int(s['power']), 'fields': {'F': nF}})
    return out


def main():
    import generate_stage10_figures as gsf
    gsf.setup_style()
    mesh, inside = _build_mesh_and_inside()
    power_data = _load_power_sweep()
    if not power_data:
        sys.exit(f'No power sweep data found in {PHASE1_ROOT}/results/sweeps/power')
    out_dir = os.path.join(PHASE1_ROOT, 'docs', 'report', 'figures')
    gsf.gen_radial_F_wafer_mettler(power_data, mesh, inside, out_dir)
    print(f'Regenerated fig_radial_F_wafer_mettler.{{png,pdf}} in {out_dir}')


if __name__ == '__main__':
    main()
