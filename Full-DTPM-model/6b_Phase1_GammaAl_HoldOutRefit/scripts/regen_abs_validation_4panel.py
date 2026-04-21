#!/usr/bin/env python3
"""Regenerate fig_absolute_validation_4panel AND fig_residuals anchored at
Mettler's 1000 W operating point.

The legacy driver (generate_stage10_figures.py) looks for
`results/sweeps/power/P{:04d}W`, which does not exist in 6b. The current
campaign lives at `results/sweeps/power_1000W_biased/` with per-point
`summary.json` files (every key the generators need, including SF6_icp)
and per-point `nF.npy` field arrays (needed by gen_residuals for the 1000 W
radial profile).
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from generate_stage10_figures import (  # noqa: E402
    gen_absolute_validation_4panel,
    gen_residuals_stage10_style,
    setup_style,
)
from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask  # noqa: E402


def _load_point(point_dir):
    sj = os.path.join(point_dir, 'summary.json')
    if not os.path.isfile(sj):
        return None
    with open(sj) as f:
        summary = json.load(f)
    fields = {}
    nF_path = os.path.join(point_dir, 'nF.npy')
    if os.path.isfile(nF_path):
        fields['F'] = np.load(nF_path)
    return summary, fields


def main():
    setup_style()
    out_dir = os.path.join(PROJECT_ROOT, 'docs', 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Build mesh + inside mask for gen_residuals_stage10_style (which slices
    # the 2D nF field onto the wafer radial profile)
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    config = SimulationConfig(config_path)
    tel = config.tel_geometry
    L_total = tel['L_proc'] + tel['L_apt'] + tel['L_icp']
    mesh = Mesh2D(R=tel['R_proc'], L=L_total, Nr=tel['Nr'], Nz=tel['Nz'],
                  beta_r=tel['beta_r'], beta_z=tel['beta_z'])
    inside, _ = build_geometry_mask(
        mesh, tel['R_icp'], tel['R_proc'],
        tel['L_proc'], tel['L_proc'] + tel['L_apt'], L_total,
        R_wafer=tel.get('R_wafer', 0.075))

    # Iterate P0200W..P1200W; each subdir has summary.json + nF.npy
    sweep_base = os.path.join(PROJECT_ROOT, 'results', 'sweeps',
                              'power_1000W_biased')
    power_data = []
    for entry in sorted(os.listdir(sweep_base)):
        if not (entry.startswith('P') and entry.endswith('W')):
            continue
        point_dir = os.path.join(sweep_base, entry)
        if not os.path.isdir(point_dir):
            continue
        try:
            power = int(entry[1:-1])
        except ValueError:
            continue
        loaded = _load_point(point_dir)
        if loaded is None:
            continue
        summary, fields = loaded
        power_data.append({
            'value': power,
            'summary': summary,
            'fields': fields,
        })
    power_data.sort(key=lambda r: r['value'])
    print(f"  Loaded {len(power_data)} power points "
          f"({power_data[0]['value']}--{power_data[-1]['value']} W); "
          f"nF fields loaded for "
          f"{sum(1 for r in power_data if 'F' in r['fields'])} points")

    # (1) Absolute [F] + source-sink 4-panel — 1000 W anchor
    gen_absolute_validation_4panel(power_data, out_dir)

    # (2) Residuals 2-panel — 1000 W anchor against Mettler Fig. 4.17
    gen_residuals_stage10_style(power_data, mesh, inside, out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
