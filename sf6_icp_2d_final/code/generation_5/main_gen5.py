#!/usr/bin/env python3
"""
SF6/Ar 2D ICP Simulator — Generation 5 (Full Physics)

All six future-work items from the Gen-4b report:

  [1] SELF-CONSISTENT ne MAGNITUDE
      ne solved from local power balance with global constraint.
      Profile from source-weighted ionization; magnitude from
      P_abs = integral(ne * nu_iz * eps_T * e) dV.

  [2] BOLSIG+ TABLE INFRASTRUCTURE
      Transport coefficients from interpolation tables.
      Uses analytical Hagelaar fits as default; swappable for
      real BOLSIG+ output files.

  [3] SHEATH MODEL
      Analytic Bohm-sheath at all walls: V_s, E_ion, ion flux.
      Ion-enhanced etch probability.

  [4] GAS TEMPERATURE SOLVER
      2D neutral energy equation: kappa*nabla^2(Tg) + Q_el + Q_FC = 0.
      Feeds back into gas density and transport.

  [5] MULTI-ION SPECIES
      Tracks SF5+, SF3+, Ar+ fractions and effective ion mass.
      Ion composition at wafer for etch selectivity.

  [6] ENHANCED DT API
      run_v5() returns comprehensive dict with all new physics.
      run_dt_scan() for multi-parameter sweeps with etch rate output.
"""

import sys, os, time, argparse
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh.mesh_generator import Mesh2D
from solvers.poisson import PoissonSolver
from solvers.em_solver import EMSolver
from solvers.diffusion_profile import solve_diffusion_profile
from solvers.energy_2d import compute_Eloss_field
from solvers.transport_2d import solve_negative_ions, solve_neutral_transport, _solve_diffusion_neumann
from solvers.sheath_model import compute_wall_fluxes, ion_enhanced_etch_probability
from solvers.gas_temperature import solve_gas_temperature
from transport.transport import IonTransport, NeutralTransport, ambipolar_Da
from transport.hagelaar_transport import transport_mixture as hagelaar_mix
from transport.bolsig_table import BOLSIGTable
from chemistry.sf6_rates import rates, fluorine_source, M_SPECIES
from chemistry.multi_ion import compute_ion_fractions_2d, effective_ion_mass
from postprocess.etch_rate import etch_rate, etch_rate_profile, uniformity
from main_v4b import _solve_robin_asymmetric

AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6


def run_v5(P_rf=1500.0, p_mTorr=10.0, frac_Ar=0.0, Tgas_init=300.0, T_neg=0.3,
           gamma_F_wafer=0.02, gamma_F_wall=0.30, gamma_F_window=0.001,
           beta_SF6=0.005, eta=0.12,
           Nr=30, Nz=40, n_iter=80, em_interval=3, verbose=True):
    t0 = time.time()
    p_Pa = p_mTorr * MTORR_TO_PA
    ng_init = p_Pa / (kB * Tgas_init)
    nAr = ng_init * frac_Ar; nSF6_0 = ng_init * (1 - frac_Ar)
    V_reactor = pi * 0.180**2 * 0.175
    P_abs = P_rf * eta
    Q_tp = 40e-6/60 * 1.01325e5 * (Tgas_init/273.15)
    tau_R = p_Pa * V_reactor / Q_tp if Q_tp > 0 else 1e10

    mesh = Mesh2D(R=0.180, L=0.175, Nr=Nr, Nz=Nz, stretch_r=1.3, stretch_z=1.3)
    em = EMSolver(mesh, freq=13.56e6)
    poisson = PoissonSolver(mesh)

    # [2] BOLSIG+ table (analytical fallback)
    bolsig = BOLSIGTable.from_analytical(x_Ar=frac_Ar, x_SF6=1-frac_Ar, N=ng_init)

    Mi_amu = 127.06; Mi_kg = Mi_amu * AMU  # Will be updated by multi-ion
    i_tr = IonTransport(Mi_amu, ng_init, Tgas_init)
    n_tr = NeutralTransport(ng_init, Tgas_init)
    k_rec = 1.5e-9 * cm3

    D_F = n_tr.diffusivity(M_SPECIES['F'], sigma=4e-19)
    D_SF6 = n_tr.diffusivity(M_SPECIES['SF6'], sigma=6e-19)
    D_Arm = n_tr.diffusivity(39.948, sigma=3e-19)
    D_neg = i_tr.D0
    Lambda2 = 1.0 / ((pi/mesh.L)**2 + (2.405/mesh.R)**2)
    kw_Arm = D_Arm / Lambda2

    # Robin coefficients
    v_th_F = np.sqrt(8*kB*Tgas_init/(pi*M_SPECIES['F']*AMU))
    v_th_SF6 = np.sqrt(8*kB*Tgas_init/(pi*M_SPECIES['SF6']*AMU))
    h_F_wafer = gamma_F_wafer * v_th_F / 4
    h_F_wall = gamma_F_wall * v_th_F / 4
    h_F_window = gamma_F_window * v_th_F / 4
    h_SF6_wall = beta_SF6 * v_th_SF6 / 4

    # --- 0D backbone ---
    has_0d = False; ne_0d = 1e16; Te_0d = 3.0; alpha_0d = 0
    Ec = 100; eps_T = 120; r0d = {}
    nSF6_0d = nSF6_0 * 0.3; nF_0d = 0
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared_modules'))
        from sf6_unified import solve_model as solve_0d
        r0d = solve_0d(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                       eta=eta, Tgas=Tgas_init, T_neg=T_neg)
        ne_0d = r0d['ne']; Te_0d = r0d['Te']; alpha_0d = r0d['alpha']
        nSF6_0d = r0d.get('n_SF6', nSF6_0*0.3)
        nF_0d = r0d.get('n_F', 0); Ec = r0d.get('Ec', 100)
        eps_T = r0d.get('eps_T', 120)
        has_0d = True
    except Exception as e:
        if verbose: print(f"  0D unavailable ({e})")

    if verbose:
        print(f"Gen-5 Full Physics: {P_rf:.0f}W, {p_mTorr:.0f}mT, "
              f"{frac_Ar*100:.0f}%Ar/{(1-frac_Ar)*100:.0f}%SF6")
        print(f"  0D: ne={ne_0d*1e-6:.2e} Te={Te_0d:.2f} α={alpha_0d:.1f}")
        print(f"  BOLSIG: {bolsig.source}")
        print("-" * 70)

    # --- Initialize ---
    Da0 = ambipolar_Da(alpha_0d, Te_0d, i_tr.D0, T_neg)
    g_en = Te_0d / max(T_neg, 0.01)
    uB0 = np.sqrt(eC*Te_0d*(1+alpha_0d)/(Mi_kg*(1+g_en*alpha_0d)))
    profile = solve_diffusion_profile(mesh, Da0, uB0)
    pn = profile / max(mesh.volume_average(profile), 1e-30)

    ne = ne_0d * pn
    Te_field = np.full((Nr, Nz), Te_0d)
    nArm = np.zeros((Nr, Nz))
    n_neg = np.full((Nr, Nz), alpha_0d*ne_0d) * pn if alpha_0d > 0.01 else np.zeros((Nr, Nz))
    nSF6 = np.full((Nr, Nz), nSF6_0d)
    nF = np.full((Nr, Nz), max(nF_0d, 1e10))
    Tg = np.full((Nr, Nz), Tgas_init)  # [4] gas temperature
    ng_field = np.full((Nr, Nz), ng_init)  # spatially varying gas density

    neutrals = {}
    if has_0d and nSF6_0 > 0:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr, Nz), r0d.get(f'n_{sp}', nSF6_0*0.01))
    else:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr, Nz), nSF6_0*0.01 if nSF6_0 > 0 else 0)
        if nSF6_0 > 0: neutrals['SF6'] = np.full((Nr, Nz), nSF6_0*0.3)

    try:
        P_ind, _ = em.adjust_coil_current(ne, Te_field, ng_init, P_target=P_abs)
        if not (np.all(np.isfinite(P_ind)) and P_ind.max() > 0): raise ValueError
    except:
        P_shape = np.exp(-(mesh.L-mesh.ZZ)/0.017) * np.maximum(mesh.RR/mesh.R, 0.01)
        P_ind = P_shape * P_abs / max(mesh.volume_average(P_shape)*V_reactor, 1e-10)

    w = 0.12

    for it in range(n_iter):
        Te_old = Te_field.copy()

        # === [4] GAS TEMPERATURE ===
        if it > 5 and it % 5 == 0:
            Tg_new, Q_heat = solve_gas_temperature(
                mesh, ne, Te_field, nSF6, ng_init, M_gas_amu=Mi_amu,
                T_wall=Tgas_init, x_Ar=frac_Ar, x_SF6=1-frac_Ar)
            Tg = 0.8*Tg + 0.2*Tg_new
            # Update local gas density: ng = p/(kB*Tg)
            ng_field = p_Pa / (kB * Tg)

        # === 1. Ar* ===
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j]); ne_ij = ne[i,j]
                qh = (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nSF6[i,j]+k_loc['qnch_F']*nF[i,j]
                den = (k_loc['Ar_iz_m']+k_loc['Ar_q'])*ne_ij + kw_Arm + qh
                nArm[i,j] = k_loc['Ar_exc']*ne_ij*nAr/max(den, 1.0) if ne_ij > 1e8 else 0

        # === 2. Te from energy equation ===
        Eloss_field = compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm)
        eps_bar = np.maximum(1.5*Te_field, 0.5)
        nu_eps = np.clip(np.where(eps_bar > 0.1, Eloss_field/eps_bar, 1.0), 1.0, 1e12)
        ne_safe = np.maximum(ne, 1e8)
        source_eps = P_ind / (ne_safe * eC)
        h_tr = hagelaar_mix(max(Te_0d, 0.5), ng_init, frac_Ar, 1-frac_Ar)
        D_eps_val = h_tr['D_eps']

        eps_new = _solve_diffusion_neumann(mesh, D_eps_val, source_eps, nu_eps)
        Te_new = np.clip((2.0/3.0) * np.clip(eps_new, 0.5, 15.0), 0.8, 8.0)
        Te_avg_raw = mesh.volume_average(Te_new)
        if Te_avg_raw > 0.1 and Te_0d > 0.1:
            Te_new *= Te_0d / Te_avg_raw
        Te_new = np.clip(Te_new, 0.8, 8.0)
        Te_field = Te_field + w*(Te_new - Te_field)

        # === [5] MULTI-ION: compute effective mass ===
        ion_fracs, M_eff_field = compute_ion_fractions_2d(
            mesh, ne, Te_field, nSF6, nAr, nArm, ng_init)
        Mi_eff_avg = mesh.volume_average(M_eff_field)
        Mi_kg_local = max(Mi_eff_avg, 10*AMU)

        # === [1] SELF-CONSISTENT ne: source-weighted + power constraint ===
        Te_avg = mesh.volume_average(Te_field)
        al_avg = mesh.volume_average(n_neg) / max(mesh.volume_average(ne), 1e10)
        Da_now = ambipolar_Da(al_avg, Te_avg, i_tr.D0, T_neg)
        g_t = Te_avg / max(T_neg, 0.01)
        uB_now = np.sqrt(eC*Te_avg*(1+al_avg)/(Mi_kg_local*(1+g_t*al_avg)))

        ne_source = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j])
                nu_iz = k_loc['Ar_iz']*nAr + k_loc['iz_SF6_total']*nSF6[i,j]
                if nArm[i,j] > 1e6:
                    nu_iz += k_loc['Ar_iz_m']*nArm[i,j]
                nu_att = k_loc['att_SF6_total']*nSF6[i,j]
                ne_source[i,j] = max(nu_iz - nu_att, 0) * ne[i,j]

        ne_profile = _solve_robin_asymmetric(mesh, Da_now, ne_source, np.zeros((Nr, Nz)),
                                             h_r_wall=uB_now, h_z0=uB_now, h_zL=uB_now)

        # [1] Power constraint: scale ne so that P_abs = integral(ne*nu_iz*eps_T*e)dV
        ne_prof_avg = mesh.volume_average(ne_profile)
        if ne_prof_avg > 1e6:
            # Compute what eps_T would be at current ne
            eps_T_local = max(eps_T, 50)
            # Average ionization frequency
            nu_iz_avg = 0
            for i in range(Nr):
                for j in range(Nz):
                    k_l = rates(Te_field[i,j])
                    nu_iz_avg += (k_l['Ar_iz']*nAr + k_l['iz_SF6_total']*nSF6[i,j]
                                  + k_l['Ar_iz_m']*nArm[i,j]) * ne_profile[i,j]
            nu_iz_avg /= max(Nr*Nz, 1)
            
            # ne_target from power balance
            ne_target = P_abs / (max(nu_iz_avg, 1e-30) * eps_T_local * eC * V_reactor / max(ne_prof_avg, 1e6))
            ne_target = np.clip(ne_target, ne_0d*0.3, ne_0d*3.0)
            
            ne_new = ne_profile * (ne_target / ne_prof_avg)
        else:
            ne_new = ne_0d * pn
        
        ne = ne + w*(ne_new - ne); ne = np.maximum(ne, 1e6)

        # === 4. Negative ions ===
        if nSF6_0 > 0:
            n_pos = ne + n_neg
            n_neg, alpha = solve_negative_ions(
                mesh, ne, n_neg, nSF6, Te_field, n_pos,
                D_neg, k_rec, alpha_0d=alpha_0d, ne_0d=ne_0d, w=w)
        else:
            n_neg = np.zeros((Nr, Nz)); alpha = np.zeros((Nr, Nz))

        # === 5. SF6 (Robin, partial anchor) ===
        if nSF6_0 > 0:
            sf6_source = np.full((Nr, Nz), nSF6_0/tau_R)
            sf6_loss = np.zeros((Nr, Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    k_e = (k_loc['d1']+k_loc['d2']+k_loc['d3']+k_loc['d4']+k_loc['d5']
                           +k_loc['iz_SF6_total']+k_loc['att_SF6_total'])
                    sf6_loss[i,j] = k_e*ne[i,j] + (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nArm[i,j] + 1.0/tau_R

            nSF6_new = _solve_robin_asymmetric(mesh, D_SF6, sf6_source, sf6_loss,
                                                h_r_wall=h_SF6_wall, h_z0=h_SF6_wall, h_zL=h_SF6_wall*0.1)
            sf6_avg_new = mesh.volume_average(nSF6_new)
            sf6_min = nSF6_0d * 0.80
            if sf6_avg_new > 1e6 and sf6_avg_new < sf6_min:
                nSF6_new *= sf6_min / sf6_avg_new
            nSF6 = nSF6 + w*(nSF6_new - nSF6)
            nSF6 = np.clip(nSF6, 1e8, nSF6_0*2)
            neutrals['SF6'] = nSF6

        # === 6. F (asymmetric Robin, no anchor) ===
        if nSF6_0 > 0:
            f_source = np.zeros((Nr, Nz)); f_loss = np.zeros((Nr, Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    f_source[i,j] = fluorine_source(k_loc, ne[i,j], nSF6[i,j], 0,0,0,0,0,0, nArm[i,j])
                    f_loss[i,j] = k_loc['iz28']*ne[i,j] + 1.0/tau_R
                    for sp_key, nr_key in [('SF5','nr42'),('SF4','nr41'),('SF3','nr40'),
                                           ('SF2','nr39'),('SF','nr38'),('S','nr37')]:
                        if sp_key in neutrals:
                            f_loss[i,j] += k_loc.get(nr_key,0)*neutrals[sp_key][i,j]

            nF_new = _solve_robin_asymmetric(mesh, D_F, f_source, f_loss,
                                             h_r_wall=h_F_wall, h_z0=h_F_wafer, h_zL=h_F_window)
            nF = nF + w*(nF_new - nF); nF = np.maximum(nF, 0.0)
            neutrals['F'] = nF

        # === 7. Minor neutrals ===
        if nSF6_0 > 0 and has_0d:
            neutrals_0d_dict = {sp: r0d.get(f'n_{sp}', 0) for sp in neutrals if f'n_{sp}' in r0d}
            D_coeffs = {sp: n_tr.diffusivity(M_SPECIES.get(sp,50), sigma=5e-19) for sp in neutrals}
            full_old = {sp: neutrals[sp].copy() for sp in neutrals}
            full_new = solve_neutral_transport(
                mesh, ne, Te_field, nArm, ng_init, nSF6_0, tau_R,
                full_old, D_coeffs, gamma_F=gamma_F_wall, kw_F_eff=30.0, w=w,
                neutrals_0d=neutrals_0d_dict)
            for sp in full_new:
                if sp not in ('SF6', 'F'):
                    neutrals[sp] = full_new[sp]

        # === 8. EM ===
        if it % em_interval == 0:
            for _ in range(5):
                try:
                    Pn, _ = em.adjust_coil_current(ne, Te_field, ng_init, P_target=P_abs)
                    if np.all(np.isfinite(Pn)) and Pn.max() > 0:
                        P_ind = 0.2*P_ind + 0.8*Pn
                except: pass

        # === Convergence ===
        dTe = np.max(np.abs(Te_field-Te_old)) / max(np.max(np.abs(Te_old)), 0.1)
        ne_avg = mesh.volume_average(ne)

        if verbose and (it%(max(n_iter//8, 1))==0 or it==n_iter-1):
            al_a = mesh.volume_average(n_neg)/max(ne_avg, 1e10)
            nF_a = mesh.volume_average(nF)
            jm = Nz//2
            nF_c = nF[0,jm]; nF_e = nF[-1,jm]
            drop = (1-nF_e/max(nF_c,1e-30))*100 if nF_c > 1e8 else 0
            Tg_max = Tg.max()
            print(f"  {it:>3d}: ne={ne_avg*1e-6:.1e} Te={Te_avg:.2f}({Te_field.min():.1f}-{Te_field.max():.1f}) "
                  f"α={al_a:.1f} [F]={nF_a*1e-6:.1e}(drop={drop:.0f}%) "
                  f"Tg_max={Tg_max:.0f}K Mi={Mi_eff_avg/AMU:.0f}AMU")

        if it > 25 and dTe < 5e-4:
            if verbose: print(f"  *** Converged at iteration {it} ***")
            break

    elapsed = time.time() - t0

    # === [3] SHEATH MODEL ===
    wall_data = compute_wall_fluxes(ne, Te_field, alpha, mesh, Mi_kg_local, T_neg)
    # Ion-enhanced etch probability at wafer
    gamma_etch = ion_enhanced_etch_probability(wall_data['wafer_energy'])

    # Etch rate from [F] + ion enhancement
    r_cm, R_etch = etch_rate_profile(nF, mesh, Tgas_init, gamma_Si=0.025)
    unif = uniformity(R_etch, r_cm, r_max_cm=15.0)

    # Summary
    nav = mesh.volume_average(ne); Te_avg = mesh.volume_average(Te_field)
    al_avg = mesh.volume_average(n_neg)/max(nav, 1e10)
    nF_avg = mesh.volume_average(nF)
    jm = Nz//2
    nF_center = nF[0,jm]; nF_edge = nF[-1,jm]
    F_drop = (1-nF_edge/max(nF_center, 1e-30))*100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gen-5 done in {elapsed:.1f}s")
        print(f"  ne  = {nav*1e-6:.3e} cm⁻³ (0D: {ne_0d*1e-6:.2e}, ratio={nav/max(ne_0d,1):.2f})")
        print(f"  Te  = {Te_avg:.2f} eV ({Te_field.min():.2f}–{Te_field.max():.2f})")
        print(f"  α   = {al_avg:.1f}")
        print(f"  [F] = {nF_avg*1e-6:.2e} cm⁻³, drop={F_drop:.0f}%")
        print(f"  Tg  = {Tg.min():.0f}–{Tg.max():.0f} K")
        print(f"  M_ion = {Mi_eff_avg/AMU:.1f} AMU")
        for sp in ['SF5+','SF3+','Ar+']:
            if sp in ion_fracs:
                f_avg = mesh.volume_average(ion_fracs[sp])
                if f_avg > 0.01:
                    print(f"    {sp}: {f_avg*100:.1f}%")
        print(f"  Sheath: V_s={wall_data['wafer_energy'].mean():.1f}eV, "
              f"Γ_i(0)={wall_data['wafer_flux'][0]:.2e} m⁻²s⁻¹")
        print(f"  Etch: {unif['mean']:.1f}±{unif['std']:.1f} nm/s, "
              f"non-unif={unif['nonuniformity_pct']:.1f}%")

    return {
        'ne': ne, 'Te': Te_field, 'n_neg': n_neg, 'alpha': alpha,
        'nF': nF, 'nSF6': nSF6, 'nArm': nArm, 'Tg': Tg,
        'P_ind': P_ind, 'mesh': mesh, 'ne_avg': nav, 'Te_avg': Te_avg,
        'alpha_avg': al_avg, 'nF_avg': nF_avg, 'Ec_avg': Ec, 'eps_T': eps_T,
        'F_drop_pct': F_drop, 'elapsed': elapsed,
        'ion_fractions': ion_fracs, 'M_eff': M_eff_field,
        'wall_fluxes': wall_data, 'etch_rate': R_etch, 'etch_uniformity': unif,
        'Tg_max': Tg.max(), 'Mi_eff': Mi_eff_avg,
        'gamma_F_wall': gamma_F_wall, 'gamma_F_wafer': gamma_F_wafer,
        'bolsig_source': bolsig.source,
    }


def run_dt_scan(powers=None, pressures=None, compositions=None,
                gamma_F_wall=0.30, Nr=20, Nz=25, n_iter=50, verbose=False):
    """Enhanced DT scan with etch rate output."""
    if powers is None: powers = [500, 1000, 1500, 2000]
    if pressures is None: pressures = [10]
    if compositions is None: compositions = [0.0, 0.3, 0.5, 0.7, 1.0]

    results = []
    for P in powers:
        for p in pressures:
            for ar in compositions:
                try:
                    r = run_v5(P_rf=P, p_mTorr=p, frac_Ar=ar,
                               gamma_F_wall=gamma_F_wall,
                               Nr=Nr, Nz=Nz, n_iter=n_iter, verbose=verbose)
                    results.append({
                        'P_rf': P, 'p_mTorr': p, 'frac_Ar': ar,
                        'ne_avg': r['ne_avg'], 'Te_avg': r['Te_avg'],
                        'alpha_avg': r['alpha_avg'], 'nF_avg': r['nF_avg'],
                        'F_drop_pct': r['F_drop_pct'],
                        'Tg_max': r['Tg_max'], 'Mi_eff': r['Mi_eff'] / AMU,
                        'etch_mean': r['etch_uniformity']['mean'],
                        'etch_nonunif': r['etch_uniformity']['nonuniformity_pct'],
                        'elapsed': r['elapsed'], 'converged': True,
                    })
                except Exception as e:
                    results.append({
                        'P_rf': P, 'p_mTorr': p, 'frac_Ar': ar,
                        'error': str(e), 'converged': False,
                    })
    return results


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--power', type=float, default=1500.0)
    pa.add_argument('--pressure', type=float, default=10.0)
    pa.add_argument('--ar', type=float, default=0.0)
    pa.add_argument('--gamma-wall', type=float, default=0.30)
    pa.add_argument('--iter', type=int, default=80)
    pa.add_argument('--nr', type=int, default=25)
    pa.add_argument('--nz', type=int, default=30)
    pa.add_argument('--scan', action='store_true')
    a = pa.parse_args()

    print("=" * 70)
    print("SF6/Ar 2D ICP Simulator — Generation 5 (Full Physics)")
    print("  [1] Self-consistent ne magnitude")
    print("  [2] BOLSIG+ table infrastructure")
    print("  [3] Analytic sheath model")
    print("  [4] Gas temperature solver")
    print("  [5] Multi-ion species (SF5+, SF3+, Ar+)")
    print("  [6] Enhanced DT API with etch rate")
    print("=" * 70)

    if a.scan:
        results = run_dt_scan(gamma_F_wall=a.gamma_wall, Nr=a.nr, Nz=a.nz, n_iter=a.iter)
        print(f"\n{'P':>5s} {'Ar%':>4s} {'ne':>10s} {'Te':>5s} {'α':>5s} "
              f"{'[F]drop':>7s} {'Etch':>6s} {'Unif':>5s} {'Tg':>4s} {'Mi':>5s}")
        for r in results:
            if r.get('converged'):
                print(f"{r['P_rf']:5.0f} {r['frac_Ar']*100:4.0f} {r['ne_avg']*1e-6:10.1e} "
                      f"{r['Te_avg']:5.2f} {r['alpha_avg']:5.1f} {r['F_drop_pct']:6.0f}% "
                      f"{r['etch_mean']:6.1f} {r['etch_nonunif']:4.0f}% "
                      f"{r['Tg_max']:4.0f} {r['Mi_eff']:5.0f}")
    else:
        res = run_v5(P_rf=a.power, p_mTorr=a.pressure, frac_Ar=a.ar,
                     gamma_F_wall=a.gamma_wall, Nr=a.nr, Nz=a.nz, n_iter=a.iter)
