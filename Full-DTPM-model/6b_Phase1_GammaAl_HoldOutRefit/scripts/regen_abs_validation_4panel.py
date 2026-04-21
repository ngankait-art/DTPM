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


def _load_sweep_dir(sweep_dir):
    """Load all P####W points in a sweep directory into the power_data list
    shape expected by gen_absolute_validation_4panel / gen_residuals."""
    if not os.path.isdir(sweep_dir):
        return []
    power_data = []
    for entry in sorted(os.listdir(sweep_dir)):
        if not (entry.startswith('P') and entry.endswith('W')):
            continue
        point_dir = os.path.join(sweep_dir, entry)
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
    return sorted(power_data, key=lambda r: r['value'])


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

    sweep_base = os.path.join(PROJECT_ROOT, 'results', 'sweeps')

    # Primary dataset: 90% SF6 (matches the main Mettler Fig 4.17 anchor
    # at 3.774e14 cm^-3). Panels (b, c, residuals-a) use this.
    power_data_90 = _load_sweep_dir(os.path.join(sweep_base, 'power_90pct_SF6_biased'))
    # Secondary dataset: 30% SF6 (second Mettler Fig 4.17 anchor at 1.297e14).
    power_data_30 = _load_sweep_dir(os.path.join(sweep_base, 'power_30pct_SF6_biased'))
    # Legacy 70% SF6 sweep — only used as a fallback if neither matched
    # composition is available.
    power_data_70 = _load_sweep_dir(os.path.join(sweep_base, 'power_1000W_biased'))

    if power_data_90:
        primary = power_data_90
        primary_label = '90% SF6'
    elif power_data_30:
        primary = power_data_30
        primary_label = '30% SF6'
    else:
        primary = power_data_70
        primary_label = '70% SF6 (legacy)'
    print(f"  Primary (panels b/c/residuals-a): {primary_label} "
          f"({len(primary)} points)")
    if power_data_90:
        print(f"  90% SF6 sweep (panels a/d): {len(power_data_90)} points")
    if power_data_30:
        print(f"  30% SF6 sweep (panels a/d): {len(power_data_30)} points")

    # (1) Absolute [F] + source-sink 4-panel — composition-matched
    gen_absolute_validation_4panel(
        primary, out_dir,
        power_data_90=power_data_90 if power_data_90 else None,
        power_data_30=power_data_30 if power_data_30 else None)

    # (2) Residuals 2-panel — uses primary for the radial shape at 1000 W
    gen_residuals_stage10_style(primary, mesh, inside, out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
