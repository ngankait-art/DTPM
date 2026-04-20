"""
Module 11 — Plasma Chemistry Coupling (Self-Consistent Picard Iteration)
=========================================================================
Implements the outer Picard iteration loop coupling:

  Circuit (m01)
    -> I_peak from P_rf, R_coil, R_plasma (Lieberman transformer coupling)
  FDTD (m06c)
    -> E_theta_rms(r,z) from I_peak and current sigma(r,z)
  Power deposition (m10)
    -> P(r,z) = 0.5 * sigma * |E_theta|^2  (no rescaling; physical magnitude)
    -> P_abs  = integral P(r,z) dV
  Circuit update (m01)
    -> R_plasma = 2 P_abs / I_peak^2
    -> I_peak   = sqrt(2 P_rf / (R_coil + R_plasma))
    -> eta      = R_plasma / (R_coil + R_plasma)
  Energy & transport
    -> Te(r,z) from local power balance
    -> ne(r,z) from ambipolar diffusion
    -> Neutral species transport (9-species SF6 chemistry)

Key architectural change from the prior Phase-1 version
--------------------------------------------------------
The FDTD E-field magnitude is NO LONGER rescaled to match an eta_initial
target.  Instead, I_peak is the physical input and eta emerges as an
observable.  Maxwell's equations are linear in the coil current J, so
when the circuit update changes I_peak by a factor alpha, the stored
E_theta_rms is rescaled by the same alpha without re-running the FDTD.
The FDTD is re-run only when the plasma conductivity sigma(r,z) drifts
significantly (governed by config parameter `rerun_fdtd_every`).

See docs/CODE_REVIEW_ULTRAREVIEW.md for the full legacy diagnosis.
"""

import numpy as np
import logging
from scipy.optimize import brentq
from scipy.constants import e as eC

from . import m06_fdtd_cylindrical
from . import m01_circuit
from . import m12_ccp_bias_sheath
from .m10_power_deposition import compute_power_deposition
from ..solvers.ambipolar_diffusion import solve_ne_ambipolar, prescribe_bessel_cosine
from ..solvers.multispecies_transport import solve_multispecies_transport
from ..chemistry.global_model import solve_0D
from ..chemistry.sf6_rates import rates

logger = logging.getLogger(__name__)


def solve_Te_local_power_balance(P_rz, ne, nSF6_field, mesh, inside, config):
    """Solve for Te(r,z) from local power balance at each active cell.

    P(r,z) = ne(r,z) * nSF6(r,z) * nu_iz(Te) * eps_T(Te) * e
    """
    from scipy.constants import m_e, pi

    Nr, Nz = mesh.Nr, mesh.Nz
    Te = np.full((Nr, Nz), 3.0)

    for i in range(Nr):
        for j in range(Nz):
            if not inside[i, j] or ne[i, j] < 1e10 or P_rz[i, j] < 1e-3:
                continue

            ne_local = ne[i, j]
            nSF6_local = max(nSF6_field[i, j], 1e10)
            P_local = P_rz[i, j]

            def balance(T):
                k = rates(T)
                nu_iz = nSF6_local * k['iz_SF6_total']
                Riz = k['iz_SF6_total']
                Eloss = (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']
                         + 9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3']
                         + 18.6*k['d4'] + 22.7*k['d5'])
                Ec = np.clip(Eloss / max(Riz, 1e-30), 80, 400)
                eiw = 0.5 * T * np.log(max(127 * 1.66e-27 / (2*pi*m_e), 1))
                eps_T = Ec + eiw + 2 * T
                return ne_local * nu_iz * eps_T * eC - P_local

            try:
                Te[i, j] = brentq(balance, 0.5, 15.0, xtol=0.01)
            except (ValueError, RuntimeError):
                Te[i, j] = 3.0

    return Te


def _run_fdtd(state, config, I_peak, sigma_plasma):
    """Invoke m06 FDTD with the given I_peak and plasma conductivity.

    Returns the updated E_theta_rms(r,z) on the full mesh.
    """
    state['I_peak'] = I_peak
    state['sigma_plasma'] = sigma_plasma
    m06_out = m06_fdtd_cylindrical.run(state, config)
    # m06 writes E_theta_rms into state via its return dict
    state.update(m06_out)
    return state['E_theta_rms']


def run(state, config):
    """Run the Picard-coupled EM + plasma chemistry loop with self-consistent eta.

    On each Picard iteration:
      1. Power deposition from current E_theta_rms and sigma -> P_abs.
      2. Close the coil circuit: R_plasma = 2 P_abs / I_peak^2,
         I_peak_new = sqrt(2 P_rf / (R_coil + R_plasma)),
         eta = R_plasma / (R_coil + R_plasma).
      3. Rescale E_theta_rms by I_peak_new / I_peak (Maxwell linearity).
      4. Solve for Te(r,z) from local power balance.
      5. Solve for ne(r,z) from ambipolar diffusion.
      6. Solve the 9-species SF6 chemistry.
      7. Optionally re-run FDTD if sigma has drifted (rerun_fdtd_every).
      8. Check convergence on ne.

    Returns dict with ne, Te, nF, nSF6, P_rz, eta_computed, R_plasma,
    I_peak, F_drop_pct, etc.
    """
    mesh = state['mesh']
    inside = state['inside']
    bc_type = state['bc_type']
    ij_to_flat = state['ij_to_flat']
    flat_to_ij = state['flat_to_ij']
    n_active = state['n_active']

    tel = config.tel_geometry if hasattr(config, 'tel_geometry') else config.get('tel_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    coup = config.coupling if hasattr(config, 'coupling') else config.get('coupling', {})
    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})

    R_icp = tel.get('R_icp', 0.038)
    L_proc = tel.get('L_proc', 0.050)
    L_apt = tel.get('L_apt', 0.002)
    L_icp = tel.get('L_icp', 0.1815)

    max_picard = coup.get('max_picard_iter', 20)
    picard_tol = coup.get('picard_tol', 0.02)
    w_ne = coup.get('under_relax_ne', 0.3)
    w_Te = coup.get('under_relax_Te', 0.3)
    inner_iter = coup.get('inner_chem_iter', 60)
    inner_relax = coup.get('inner_chem_relax', 0.12)
    rerun_fdtd_every = coup.get('rerun_fdtd_every', 1)  # every iter by default

    P_rf = circ['source_power']
    R_coil = state.get('R_coil', circ.get('R_coil', 0.8))
    # m01 already seeded state['R_plasma'] and state['I_peak'] with initial guess
    R_plasma = state.get('R_plasma', m01_circuit.R_PLASMA_INITIAL_GUESS)
    I_peak, eta = m01_circuit.compute_coil_current(P_rf, R_coil, R_plasma)

    Nr, Nz = mesh.Nr, mesh.Nz

    # ── Initial guess from 0D global model ──
    # The 0D model internally prescribes eta (no spatial E-field); use the
    # m01 circuit seed eta here so the 0D guess is consistent with the 2D
    # circuit model at iteration 0.
    logger.info("M11: Computing initial guess from 0D global model...")
    result_0D = solve_0D(
        P_rf=P_rf,
        p_mTorr=oper.get('pressure_mTorr', 10),
        frac_Ar=oper.get('frac_Ar', 0.0),
        Q_sccm=oper.get('Q_sccm', 100),
        Tgas=oper.get('Tgas', 313),
        eta=eta,  # from the self-consistent circuit seed
        R_icp=R_icp, L_icp=L_icp,
    )
    alpha_0D = float(result_0D.get('alpha', 0.0))
    logger.info(f"  0D: Te={result_0D['Te']:.2f}eV, ne={result_0D['ne']:.2e}m^-3, "
                f"alpha={alpha_0D:.2f}")
    # L1 correction: electronegative ambipolar diffusion (Lieberman 2005 §10.3).
    # The scalar alpha from the 0D closure is threaded uniformly into the 2D
    # ambipolar solver.  A 2D alpha(r,z) field would be a Phase-3 upgrade.
    alpha_2D = alpha_0D

    ne = prescribe_bessel_cosine(result_0D['ne'], mesh, inside,
                                 R_icp, L_proc, L_apt, L_icp)
    Te = np.where(inside, result_0D['Te'], 0.0)

    from scipy.constants import k as kB
    p_Pa = oper.get('pressure_mTorr', 10) * 0.133322
    ng = p_Pa / (kB * oper.get('Tgas', 313))
    nSF6_feed = ng * (1 - oper.get('frac_Ar', 0.0))
    nSF6 = np.where(inside, nSF6_feed * 0.5, 0.0)
    nF = np.where(inside, 1e18, 0.0)

    # ── Initial FDTD solve with seed I_peak and sigma(r,z) ──
    # Compute initial sigma from the 0D ne and Te guess via m10 (no rescale).
    E_theta_rms = state.get('E_theta_rms')
    if E_theta_rms is None or not np.any(E_theta_rms > 0):
        # Must invoke FDTD first; initial sigma will be built by m10 inside
        # the first power_deposition call.  We pass sigma_plasma=None so
        # m06 uses the default free-space inside-mask behaviour.
        logger.info("M11: Running initial FDTD with seed I_peak...")
        E_theta_rms = _run_fdtd(state, config, I_peak, sigma_plasma=None)
    else:
        logger.info("M11: Using pre-existing FDTD E-field from state")

    convergence_history = []
    P_abs = 0.0
    P_rz = np.zeros((Nr, Nz))
    F_drop = 0.0

    logger.info(f"M11: Starting Picard iteration (max {max_picard} iter, "
                f"tol={picard_tol}, rerun_fdtd_every={rerun_fdtd_every})")
    print(f"\n{'='*78}")
    print("  Self-Consistent Picard Iteration: Circuit + EM + Chemistry")
    print(f"{'='*78}")
    print(f"  {'It':>3s} {'ne_avg':>10s} {'Te_avg':>6s} {'eta':>6s} "
          f"{'I_peak':>7s} {'Rp':>6s} {'P_abs':>7s} {'Fdrop':>6s} "
          f"{'sigma_max':>10s} {'ne_chg':>8s}")

    for k_iter in range(max_picard):
        ne_old = ne.copy()

        # ────────────────────────────────────────────────────────────────
        # Step 1: Compute P_abs at CURRENT E_theta_rms (physical magnitude)
        # ────────────────────────────────────────────────────────────────
        pd = compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config)
        P_abs = pd['P_abs']
        sigma_rz = pd['sigma_rz']

        # ────────────────────────────────────────────────────────────────
        # Step 2: Close the coil circuit (Lieberman transformer coupling)
        # ────────────────────────────────────────────────────────────────
        if P_abs > 1e-3:
            circuit_upd = m01_circuit.update_circuit_from_Pabs(
                P_rf=P_rf, R_coil=R_coil,
                I_peak_prev=I_peak, P_abs=P_abs,
            )
            R_plasma_new = circuit_upd['R_plasma']
            I_peak_new = circuit_upd['I_peak']
            eta_new = circuit_upd['eta']
        else:
            # Plasma hasn't ignited yet; retain the initial guess
            R_plasma_new, I_peak_new, eta_new = R_plasma, I_peak, eta

        # ────────────────────────────────────────────────────────────────
        # Step 3: Rescale E_theta_rms linearly (Maxwell linearity in J)
        # ────────────────────────────────────────────────────────────────
        # For FIXED sigma(r,z), the FDTD output scales linearly in I_peak.
        # So updating I_peak without re-running FDTD is valid as long as
        # sigma hasn't drifted too much; step 6 handles re-runs.
        if I_peak > 0:
            linear_scale = I_peak_new / I_peak
            E_theta_rms = E_theta_rms * linear_scale

        I_peak = I_peak_new
        R_plasma = R_plasma_new
        eta = eta_new

        # Recompute P_abs after the linear scale (should equal eta*P_rf)
        pd = compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config)
        P_rz = pd['P_rz']
        P_abs = pd['P_abs']
        sigma_rz = pd['sigma_rz']

        # ────────────────────────────────────────────────────────────────
        # Step 4: Te(r,z) from local power balance
        # ────────────────────────────────────────────────────────────────
        Te_new = solve_Te_local_power_balance(P_rz, ne, nSF6, mesh, inside, config)

        # ────────────────────────────────────────────────────────────────
        # Step 4b: Wafer-bias sheath (m12) — adds an expansion source in
        # the process chamber when bias is enabled.  Bypassed entirely
        # when config.bias.enabled = False or P_bias_W = 0.
        # ────────────────────────────────────────────────────────────────
        bias = m12_ccp_bias_sheath.compute_bias_sheath(
            ne, Te_new, mesh, inside, config)
        if bias['enabled']:
            P_rz_total = P_rz + bias['P_rz_bias']
            P_abs_total = P_abs + float(np.sum(
                bias['P_rz_bias'] * mesh.vol * inside))
        else:
            P_rz_total = P_rz
            P_abs_total = P_abs

        # ────────────────────────────────────────────────────────────────
        # Step 5: ne(r,z) from ionization-source diffusion
        # ────────────────────────────────────────────────────────────────
        ne_new, ne_avg = solve_ne_ambipolar(
            Te_new, P_abs_total, mesh, inside, bc_type,
            ij_to_flat, flat_to_ij, n_active, config, P_rz=P_rz_total,
            alpha=alpha_2D,
        )

        # ────────────────────────────────────────────────────────────────
        # Step 6: Full 9-species neutral chemistry
        # ────────────────────────────────────────────────────────────────
        chem_result = solve_multispecies_transport(
            mesh, inside, bc_type, ij_to_flat, flat_to_ij, n_active,
            ne_new, Te_new, config,
            n_iter=inner_iter, w=inner_relax, verbose=(k_iter == 0),
        )
        nF = chem_result['nF']
        nSF6 = chem_result['nSF6']
        F_drop = chem_result['F_drop_pct']

        # ────────────────────────────────────────────────────────────────
        # Step 7: Re-run FDTD periodically if sigma has drifted
        # ────────────────────────────────────────────────────────────────
        if rerun_fdtd_every > 0 and ((k_iter + 1) % rerun_fdtd_every == 0):
            # Pass the latest sigma_rz into the FDTD so dispersion is consistent.
            # I_peak stays the same — circuit closure happens at the top of the
            # next iteration.
            logger.info(f"  Re-running FDTD with updated sigma (iter {k_iter+1})")
            E_theta_rms = _run_fdtd(state, config, I_peak, sigma_plasma=sigma_rz)

        # ────────────────────────────────────────────────────────────────
        # Step 8: Convergence
        # ────────────────────────────────────────────────────────────────
        ne_inside = ne_old[inside]
        ne_new_inside = ne_new[inside]
        rel_change = (np.linalg.norm(ne_new_inside - ne_inside)
                      / max(np.linalg.norm(ne_inside), 1e-30))

        Te_avg = np.mean(Te_new[inside]) if np.any(inside) else 3.0

        convergence_history.append({
            'iter': k_iter,
            'ne_avg': float(ne_avg),
            'Te_avg': float(Te_avg),
            'eta': float(eta),
            'I_peak': float(I_peak),
            'R_plasma': float(R_plasma),
            'P_abs': float(P_abs),
            'F_drop_pct': float(F_drop),
            'rel_change': float(rel_change),
        })

        print(f"  {k_iter:3d} {ne_avg*1e-6:10.2e} {Te_avg:6.2f} {eta:6.3f} "
              f"{I_peak:7.2f} {R_plasma:6.2f} {P_abs:7.1f} {F_drop:5.1f}% "
              f"{sigma_rz.max():10.2e} {rel_change:8.4f}")

        if rel_change < picard_tol and k_iter > 0:
            logger.info(f"  Picard converged at iteration {k_iter+1} "
                        f"(rel_change={rel_change:.4e})")
            break

        # Under-relax ne, Te to stabilise the outer iteration
        ne = (1 - w_ne) * ne + w_ne * ne_new
        Te = (1 - w_Te) * Te + w_Te * Te_new

    print(f"{'='*78}")
    print(f"  Picard complete: {k_iter+1} iterations, "
          f"eta={eta:.3f}, I_peak={I_peak:.2f} A, R_plasma={R_plasma:.2f} Ohm")
    print(f"  [F] drop={F_drop:.1f}%, P_abs={P_abs:.1f} W (P_rf={P_rf} W)")
    print(f"{'='*78}\n")

    species_fields = chem_result.get('fields', {})
    ions = chem_result.get('ions', {})

    # Operating voltages at the coil driven port (Lieberman Eq 12.2.19
    # at matched-resonance condition: reactive part cancelled by the
    # matching network, so V across coil is I * (R_coil + R_plasma)).
    V_peak_final = float(I_peak * (R_coil + R_plasma))
    V_rms_final = V_peak_final / np.sqrt(2.0)

    result = {
        'ne': ne,
        'Te': Te,
        'nF': nF,
        'nSF6': nSF6,
        'P_rz': P_rz,
        'P_abs': float(P_abs),
        'P_abs_final': float(P_abs),
        'eta_computed': float(eta),
        'I_peak_final': float(I_peak),
        'R_plasma_final': float(R_plasma),
        'V_peak_final': V_peak_final,
        'V_rms_final': V_rms_final,
        'R_coil': float(R_coil),
        'E_theta_rms': E_theta_rms,
        'F_drop_pct': float(F_drop),
        'ne_avg': float(ne_avg),
        'picard_iterations': k_iter + 1,
        'convergence_history': convergence_history,
        'result_0D': result_0D,
        'species_fields': species_fields,
        'ions': ions,
        # Bias diagnostics (zero-valued when bias disabled)
        'bias_enabled': bool(bias.get('enabled', False)),
        'bias_V_dc': float(bias.get('V_dc', 0.0)),
        'bias_Gamma_i': float(bias.get('Gamma_i', 0.0)),
        'bias_P_ion_to_gas': float(bias.get('P_ion_to_gas', 0.0)),
        'bias_lambda_exp': float(bias.get('lambda_exp', 0.0)),
        'bias_P_bias_total': float(np.sum(
            bias.get('P_rz_bias', np.zeros((Nr, Nz))) * mesh.vol * inside
        )) if bias.get('enabled', False) else 0.0,
    }
    for sp_name, sp_field in species_fields.items():
        result[f'n{sp_name}'] = sp_field
    return result
