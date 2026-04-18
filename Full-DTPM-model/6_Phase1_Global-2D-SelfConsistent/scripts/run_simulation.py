#!/usr/bin/env python3
"""
Phase 1 Merged Simulation: EM + SF6 Chemistry on TEL Geometry
==============================================================
Entry point for the unified DTPM simulation that computes EM fields
on the TEL ICP reactor and couples them to the 0D-2D plasma chemistry.

Pipeline:
  1. Build TEL mesh + geometry mask
  2. Run M01 (circuit) -> V_peak, I_peak, omega
  3. Run M06c (cylindrical FDTD) -> E_theta, H_r, H_z
  4. Run M11 (Picard-coupled chemistry) -> ne, Te, nF, nSF6, P_rz, eta
  5. Post-process, validate, plot

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --config config/test_config.yaml
    python scripts/run_simulation.py --em-only          # EM pipeline only
    python scripts/run_simulation.py --no-picard         # Use prescribed ne/Te
    python scripts/run_simulation.py --no-plots
"""

import os
import sys
import argparse
import logging
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import (
    SimulationConfig, Mesh2D, PhysicalConstants,
    compute_coil_positions_cylindrical,
    build_geometry_mask, build_index_maps, count_boundary_cells,
)
from dtpm.core.grid import Grid
from dtpm.core.pipeline import Pipeline
from dtpm.core.data_manager import DataManager
from dtpm.modules import m01_circuit
from dtpm.modules import m06_fdtd_cylindrical
from dtpm.modules import m11_plasma_chemistry


def setup_logging(log_dir=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, 'simulation.log')))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers,
    )


def build_tel_state(config):
    """Build the initial pipeline state for the TEL reactor.

    Creates the cylindrical mesh, geometry mask, and index maps.
    Also creates a Grid wrapper for M01-M04 compatibility.
    """
    tel = config.tel_geometry
    R_icp = tel.get('R_icp', 0.038)
    R_proc = tel.get('R_proc', 0.105)
    L_icp = tel.get('L_icp', 0.1815)
    L_proc = tel.get('L_proc', 0.050)
    L_apt = tel.get('L_apt', 0.002)
    Nr = tel.get('Nr', 50)
    Nz = tel.get('Nz', 80)
    beta_r = tel.get('beta_r', 1.2)
    beta_z = tel.get('beta_z', 1.0)

    L_total = L_proc + L_apt + L_icp

    # Build cylindrical mesh
    mesh = Mesh2D(R=R_proc, L=L_total, Nr=Nr, Nz=Nz,
                  beta_r=beta_r, beta_z=beta_z)

    # Build geometry mask
    z_apt_bot = L_proc
    z_apt_top = L_proc + L_apt
    z_top = L_total
    inside, bc_type = build_geometry_mask(mesh, R_icp, R_proc, z_apt_bot, z_apt_top, z_top)
    ij_to_flat, flat_to_ij, n_active = build_index_maps(inside)

    # Coil positions on the cylindrical mesh
    coil_positions = compute_coil_positions_cylindrical(config, mesh)

    # Grid wrapper for M01 compatibility
    grid = Grid(config)

    state = {
        # Mesh and geometry
        'mesh': mesh,
        'inside': inside,
        'bc_type': bc_type,
        'ij_to_flat': ij_to_flat,
        'flat_to_ij': flat_to_ij,
        'n_active': n_active,
        'coil_positions_cyl': coil_positions,
        # Grid-compatible keys for M01-M04
        'nx': grid.nx,
        'ny': grid.ny,
        'dx': grid.dx,
        'dy': grid.dy,
        'r_grid': grid.r_grid if hasattr(grid, 'r_grid') else None,
        'z_grid': grid.z_grid if hasattr(grid, 'z_grid') else None,
        # Coil geometry for EM modules
        'coil_centers_idx': coil_positions,
        'coil_radius': config.geometry.get('coil_radius', 2e-3),
        'coil_radius_idx': max(1, int(config.geometry.get('coil_radius', 2e-3) / mesh.dr.mean())),
    }

    return state


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Merged EM + Chemistry Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--em-only', action='store_true',
                        help='Run only EM pipeline (M01 + M06c), skip chemistry')
    parser.add_argument('--no-picard', action='store_true',
                        help='Use prescribed ne/Te (baseline, no Picard coupling)')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output-dir', '-o', type=str, default='results')
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    config = SimulationConfig(config_path)

    # Set up output and logging
    dm = DataManager(base_dir=os.path.join(PROJECT_ROOT, args.output_dir))
    run_dir = dm.create_run_directory(config)
    setup_logging(log_dir=os.path.join(run_dir, 'logs'))

    logger = logging.getLogger('dtpm.main')
    logger.info("Phase 1 Merged Simulation Starting")
    logger.info(f"  Config: {config}")
    logger.info(f"  Run directory: {run_dir}")

    # Build initial state
    t0 = time.time()
    state = build_tel_state(config)
    mesh = state['mesh']
    bc_counts = count_boundary_cells(state['bc_type'])

    logger.info(f"  Mesh: {mesh}")
    logger.info(f"  Active cells: {state['n_active']} / {mesh.Nr * mesh.Nz}")
    logger.info(f"  Boundary cells: {bc_counts}")
    logger.info(f"  Coil positions: {len(state['coil_positions_cyl'])} turns")

    # ── Stage 1: EM Pipeline ──
    print(f"\n{'='*60}")
    print(f"  Stage 1: EM Pipeline (M01 + M06c)")
    print(f"{'='*60}")

    # M01: RF Circuit
    m01_result = m01_circuit.run(state, config)
    state.update(m01_result)
    logger.info(f"  M01: V_peak={state['V_peak']:.2f}V, I_peak={state['I_peak']:.4f}A")

    # M06c: Cylindrical FDTD
    m06c_result = m06_fdtd_cylindrical.run(state, config)
    state.update(m06c_result)

    if args.em_only:
        elapsed = time.time() - t0
        print(f"\n  EM-only mode: done in {elapsed:.2f}s")
        _print_summary(state, elapsed)
        return state

    # ── Stage 2: Picard-Coupled Chemistry ──
    print(f"\n{'='*60}")
    print(f"  Stage 2: Picard-Coupled Chemistry (M11)")
    print(f"{'='*60}")

    if args.no_picard:
        # Baseline mode: use prescribed ne/Te from Stage 10
        logger.info("  Running baseline chemistry (no Picard coupling)...")
        from dtpm.chemistry.global_model import solve_0D
        from dtpm.solvers.ambipolar_diffusion import prescribe_bessel_cosine
        from dtpm.solvers.species_transport import solve_species_transport

        oper = config.operating
        # Use 2D's self-consistent eta if available; else fall back to a
        # literature value of 0.7 rather than the old 0.43 (see
        # docs/CODE_REVIEW_ULTRAREVIEW.md).
        eta_2D = float(state.get('eta_computed',
                                 state.get('eta_circuit', 0.70)))
        r0D = solve_0D(
            P_rf=config.circuit['source_power'],
            p_mTorr=oper.get('pressure_mTorr', 10),
            eta=eta_2D,
            R_icp=config.tel_geometry.get('R_icp', 0.038),
            L_icp=config.tel_geometry.get('L_icp', 0.1815),
        )
        tel = config.tel_geometry
        ne = prescribe_bessel_cosine(r0D['ne'], mesh, state['inside'],
                                      tel['R_icp'], tel['L_proc'], tel['L_apt'], tel['L_icp'])
        Te = np.where(state['inside'], r0D['Te'], 0.0)

        chem = solve_species_transport(
            mesh, state['inside'], state['bc_type'],
            state['ij_to_flat'], state['flat_to_ij'], state['n_active'],
            ne, Te, config)

        state.update({'ne': ne, 'Te': Te, **chem, 'eta_computed': oper.get('eta_initial', 0.43)})
    else:
        # Full Picard coupling
        m11_result = m11_plasma_chemistry.run(state, config)
        state.update(m11_result)

    elapsed = time.time() - t0
    _print_summary(state, elapsed)

    # Save results
    _save_results(state, dm)
    logger.info(f"Results saved to {run_dir}")

    # Generate plots
    if not args.no_plots:
        from dtpm.utils.phase1_plots import generate_all_plots
        plots_dir = os.path.join(run_dir, 'plots')
        generate_all_plots(state, mesh, state['inside'], state['bc_type'],
                          config, plots_dir)

    return state


def _print_summary(state, elapsed):
    """Print simulation summary."""
    print(f"\n{'='*60}")
    print(f"  Phase 1 Simulation Complete")
    print(f"{'='*60}")
    if 'V_peak' in state:
        print(f"  V_peak = {state['V_peak']:.2f} V")
    if 'I_peak' in state:
        print(f"  I_peak = {state['I_peak']:.4f} A")
    if 'E_theta_rms' in state:
        print(f"  |E_theta_rms|_max = {state['E_theta_rms'].max():.4e} V/m")
    if 'eta_computed' in state:
        print(f"  eta = {state['eta_computed']:.3f}")
    if 'P_abs' in state:
        print(f"  P_abs = {state['P_abs']:.1f} W")
    if 'ne_avg' in state:
        print(f"  ne_avg = {state['ne_avg']:.2e} m^-3")
    if 'F_drop_pct' in state:
        print(f"  [F] drop = {state['F_drop_pct']:.1f}% (target: 73-75%)")
    if 'picard_iterations' in state:
        print(f"  Picard iterations: {state['picard_iterations']}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"{'='*60}\n")


def _save_results(state, dm):
    """Save key result arrays including all 9 species."""
    # EM fields
    for name in ['ne', 'Te', 'P_rz', 'E_theta', 'E_theta_rms', 'B_r', 'B_z', 'sigma_rz']:
        if name in state and hasattr(state[name], 'shape'):
            dm.save_array(state[name], name)
    # All 9 neutral species
    species_fields = state.get('species_fields', {})
    for sp_name, sp_field in species_fields.items():
        if hasattr(sp_field, 'shape'):
            dm.save_array(sp_field, f'n{sp_name}')
    # Also save F and SF6 at top level if not in species_fields
    for name in ['nF', 'nSF6']:
        if name in state and hasattr(state[name], 'shape') and name not in species_fields:
            dm.save_array(state[name], name)
    # Charged species
    ions = state.get('ions', {})
    for ion_name, ion_field in ions.items():
        if hasattr(ion_field, 'shape'):
            dm.save_array(ion_field, f'ion_{ion_name}')


if __name__ == '__main__':
    main()
