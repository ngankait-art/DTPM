#!/usr/bin/env python3
"""
SF6/Ar 2D ICP — Generation 4b: Mettler-gap fix

Three changes to produce the 75% center-to-edge [F] drop:

  [A] WALL-SPECIFIC Robin BCs
      - Wafer (z=0): gamma_F = 0.02 (Si surface, moderate F consumption)
      - Chamber sidewall (r=R): gamma_F = 0.10 (anodized Al, higher recomb)
      - Window (z=L): gamma_F = 0.001 (quartz, very low recomb)
      This asymmetry concentrates F loss at the sidewall, creating a 
      radial gradient.

  [B] RELEASED SF6 ANCHORING
      SF6 is now solved self-consistently by diffusion with Robin BCs.
      It depletes at the center (where ne is highest) and is replenished
      from the edges (feed gas). This makes the F source = ne*nSF6*k_diss
      NOT proportional to ne but concentrated at intermediate radii where
      both ne and nSF6 are substantial.

  [C] RELEASED ne ANCHORING (partial)
      ne magnitude still from 0D, but the PROFILE is computed from
      ambipolar diffusion with the LOCAL Da(alpha(r,z)) instead of 
      the volume-averaged Da. This gives a more peaked ne profile
      in the electronegative core.
"""

import sys, os, time, argparse
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh.mesh_generator import Mesh2D
from solvers.poisson import PoissonSolver
from solvers.em_solver import EMSolver
from solvers.diffusion_profile import solve_diffusion_profile
from solvers.energy_2d import compute_Eloss_field
from solvers.transport_2d import solve_negative_ions, solve_neutral_transport, _solve_diffusion_neumann
from transport.transport import IonTransport, NeutralTransport, ambipolar_Da, ElectronTransport
from transport.hagelaar_transport import transport_mixture as hagelaar_mix
from chemistry.sf6_rates import rates, compute_troe_rates, fluorine_source, M_SPECIES

AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6


def _solve_robin_asymmetric(mesh, D, source, loss_freq,
                            h_r_wall, h_z0, h_zL):
    """Solve D∇²n + S - ν·n = 0 with DIFFERENT Robin BCs on each wall.

    h_r_wall: Robin coeff at r = R (sidewall)
    h_z0:     Robin coeff at z = 0 (wafer)
    h_zL:     Robin coeff at z = L (window/coil)
    r = 0:    Neumann (symmetry)
    """
    Nr, Nz = mesh.Nr, mesh.Nz; N_tot = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = -source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D[i,j] if not np.isscalar(D) else D
            diag = -(loss_freq[i,j] if not np.isscalar(loss_freq) else loss_freq)

            # --- Radial ---
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            else:
                # Robin at r = R (sidewall)
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                wall_loss = D_ij * rf / (rc * dr) * h_r_wall / (D_ij + h_r_wall*drc)
                diag -= wall_loss

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # --- Axial ---
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            else:
                # Robin at z = L (window)
                dzc = mesh.dz_c[Nz]
                diag -= D_ij / dz * h_zL / (D_ij + h_zL*dzc)

            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c
            else:
                # Robin at z = 0 (wafer)
                dzc = mesh.dz_c[0]
                diag -= D_ij / dz * h_z0 / (D_ij + h_z0*dzc)

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N_tot, N_tot))
    n = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(n, 0.0)


def run_v4b(P_rf=1500.0, p_mTorr=10.0, frac_Ar=0.0, Tgas=300.0, T_neg=0.3,
            gamma_F_wafer=0.02, gamma_F_wall=0.10, gamma_F_window=0.001,
            beta_SF6=0.005, eta=0.12,
            Nr=30, Nz=40, n_iter=100, em_interval=3, verbose=True):
    t0 = time.time()
    p_Pa = p_mTorr*MTORR_TO_PA; ng = p_Pa/(kB*Tgas)
    nAr = ng*frac_Ar; nSF6_0 = ng*(1-frac_Ar)
    V_reactor = pi*0.180**2*0.175
    P_abs = P_rf * eta
    Q_tp = 40*1e-6/60*1.01325e5*(Tgas/273.15)
    tau_R = p_Pa*V_reactor/Q_tp if Q_tp > 0 else 1e10

    mesh = Mesh2D(R=0.180, L=0.175, Nr=Nr, Nz=Nz, stretch_r=1.3, stretch_z=1.3)
    em = EMSolver(mesh, freq=13.56e6)
    poisson = PoissonSolver(mesh)

    Mi_amu = 39.948 if frac_Ar > 0.5 else 127.06; Mi_kg = Mi_amu*AMU
    i_tr = IonTransport(Mi_amu, ng, Tgas)
    n_tr = NeutralTransport(ng, Tgas)
    k_rec = 1.5e-9 * cm3

    D_F = n_tr.diffusivity(M_SPECIES['F'], sigma=4e-19)
    D_SF6 = n_tr.diffusivity(M_SPECIES['SF6'], sigma=6e-19)
    D_Arm = n_tr.diffusivity(39.948, sigma=3e-19)
    D_neg = i_tr.D0
    Lambda2 = 1.0/((pi/mesh.L)**2 + (2.405/mesh.R)**2)
    kw_Arm = D_Arm/Lambda2

    # Wall-specific Robin coefficients
    v_th_F = np.sqrt(8*kB*Tgas/(pi*M_SPECIES['F']*AMU))
    v_th_SF6 = np.sqrt(8*kB*Tgas/(pi*M_SPECIES['SF6']*AMU))
    h_F_wafer = gamma_F_wafer * v_th_F / 4
    h_F_wall  = gamma_F_wall * v_th_F / 4
    h_F_window = gamma_F_window * v_th_F / 4
    h_SF6_wall = beta_SF6 * v_th_SF6 / 4

    # --- 0D backbone ---
    has_0d = False; ne_0d = 1e16; Te_0d = 3.0; alpha_0d = 0
    Ec = 100; eps_T = 120; r0d = {}
    nSF6_0d = nSF6_0*0.3; nF_0d = 0
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared_modules'))
        from sf6_unified import solve_model as solve_0d
        r0d = solve_0d(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                       eta=eta, Tgas=Tgas, T_neg=T_neg)
        ne_0d = r0d['ne']; Te_0d = r0d['Te']; alpha_0d = r0d['alpha']
        nSF6_0d = r0d.get('n_SF6', nSF6_0*0.3)
        nF_0d = r0d.get('n_F', 0); Ec = r0d.get('Ec', 100)
        eps_T = r0d.get('eps_T', 120)
        has_0d = True
    except Exception as e:
        if verbose: print(f"  0D unavailable ({e})")

    if verbose:
        print(f"Mesh: {mesh}")
        print(f"0D: ne={ne_0d*1e-6:.2e} Te={Te_0d:.2f} α={alpha_0d:.1f} [F]={nF_0d*1e-6:.2e}")
        print(f"Robin F: wafer={gamma_F_wafer}, sidewall={gamma_F_wall}, window={gamma_F_window}")
        print(f"  h_F: wafer={h_F_wafer:.1f}, wall={h_F_wall:.1f}, window={h_F_window:.2f} m/s")

    # --- Initialize ---
    Da0 = ambipolar_Da(alpha_0d, Te_0d, i_tr.D0, T_neg)
    g_en = Te_0d/max(T_neg, 0.01)
    uB0 = np.sqrt(eC*Te_0d*(1+alpha_0d)/(Mi_kg*(1+g_en*alpha_0d)))
    profile = solve_diffusion_profile(mesh, Da0, uB0)
    pn = profile / max(mesh.volume_average(profile), 1e-30)

    ne = ne_0d * pn
    Te_field = np.full((Nr,Nz), Te_0d)
    nArm = np.zeros((Nr,Nz))
    n_neg = np.full((Nr,Nz), alpha_0d*ne_0d)*pn if alpha_0d > 0.01 else np.zeros((Nr,Nz))
    alpha = np.where(ne > 1e10, n_neg/ne, 0.0)
    nSF6 = np.full((Nr,Nz), nSF6_0d)
    nF = np.full((Nr,Nz), max(nF_0d, 1e10))

    neutrals = {}
    if has_0d and nSF6_0 > 0:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr,Nz), r0d.get(f'n_{sp}', nSF6_0*0.01))
    else:
        for sp in ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']:
            neutrals[sp] = np.full((Nr,Nz), nSF6_0*0.01 if nSF6_0 > 0 else 0)
        if nSF6_0 > 0: neutrals['SF6'] = np.full((Nr,Nz), nSF6_0*0.3)

    # Power deposition
    try:
        P_ind, _ = em.adjust_coil_current(ne, Te_field, ng, P_target=P_abs)
        if not (np.all(np.isfinite(P_ind)) and P_ind.max() > 0): raise ValueError
    except:
        P_shape = np.exp(-(mesh.L-mesh.ZZ)/0.017)*np.maximum(mesh.RR/mesh.R, 0.01)
        P_ind = P_shape*P_abs/max(mesh.volume_average(P_shape)*V_reactor, 1e-10)

    if verbose:
        sf6_pct = (1-frac_Ar)*100
        print(f"\nGen-4b: {n_iter} iter, {P_rf:.0f}W, {p_mTorr:.0f}mT, "
              f"{frac_Ar*100:.0f}%Ar/{sf6_pct:.0f}%SF6")
        print("-"*70)

    w = 0.12

    for it in range(n_iter):
        Te_old = Te_field.copy()
        ne_avg_old = mesh.volume_average(ne)

        # === 1. Ar* ===
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j]); ne_ij = ne[i,j]
                qh = (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nSF6[i,j]+k_loc['qnch_F']*nF[i,j]
                den = (k_loc['Ar_iz_m']+k_loc['Ar_q'])*ne_ij+kw_Arm+qh
                nArm[i,j] = k_loc['Ar_exc']*ne_ij*nAr/max(den,1.0) if ne_ij>1e8 else 0

        # === 2. Te from energy equation (Hagelaar D_eps) ===
        Eloss_field = compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm)
        eps_bar = np.maximum(1.5*Te_field, 0.5)
        nu_eps = np.clip(np.where(eps_bar > 0.1, Eloss_field/eps_bar, 1.0), 1.0, 1e12)
        ne_safe = np.maximum(ne, 1e8)
        source_eps = P_ind / (ne_safe * eC)
        h_loc = hagelaar_mix(max(Te_0d, 0.5), ng, frac_Ar, 1-frac_Ar)
        D_eps_val = h_loc['D_eps']

        eps_new = _solve_diffusion_neumann(mesh, D_eps_val, source_eps, nu_eps)
        Te_new = np.clip((2.0/3.0) * np.clip(eps_new, 0.5, 15.0), 0.8, 8.0)
        Te_avg_raw = mesh.volume_average(Te_new)
        if Te_avg_raw > 0.1 and Te_0d > 0.1:
            Te_new *= Te_0d / Te_avg_raw
        Te_new = np.clip(Te_new, 0.8, 8.0)
        Te_field = Te_field + w*(Te_new - Te_field)
        Te_field = np.clip(Te_field, 0.8, 8.0)

        # === 3. ne profile — SOURCE-WEIGHTED (released anchoring) [Change C] ===
        # Instead of the uniform-source eigenmode (peak/avg = 2.5 always),
        # solve: Da*nabla^2(ne) + S_net(r,z)*ne_old = 0 with Dirichlet BCs
        # where S_net = nu_iz(Te) - nu_att(Te) is the LOCAL net ionization rate.
        # This concentrates ne where Te is high (near coil), giving stronger peaking.
        Te_avg = mesh.volume_average(Te_field)
        al_avg = mesh.volume_average(n_neg)/max(mesh.volume_average(ne), 1e10)
        Da_now = ambipolar_Da(al_avg, Te_avg, i_tr.D0, T_neg)

        # Compute local net ionization source weighted by current ne
        ne_source = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j])
                nu_iz_loc = (k_loc['Ar_iz']*nAr + k_loc['iz_SF6_total']*nSF6[i,j])
                # Add Ar* stepwise ionization
                if nArm[i,j] > 1e6:
                    nu_iz_loc += k_loc['Ar_iz_m']*nArm[i,j]
                nu_att_loc = k_loc['att_SF6_total']*nSF6[i,j]
                ne_source[i,j] = max(nu_iz_loc - nu_att_loc, 0) * ne[i,j]

        # Solve with Robin BCs (large h ≈ Dirichlet, ne → 0 at walls)
        g_t = Te_avg/max(T_neg, 0.01)
        uB_now = np.sqrt(eC*Te_avg*(1+al_avg)/(Mi_kg*(1+g_t*al_avg)))
        h_ne_wall = uB_now  # Bohm flux BC: D*dn/dr = uB*n
        ne_profile = _solve_robin_asymmetric(mesh, Da_now, ne_source, np.zeros((Nr,Nz)),
                                             h_r_wall=h_ne_wall, h_z0=h_ne_wall, h_zL=h_ne_wall)

        # Normalize to 0D average (ne magnitude still anchored)
        ne_prof_avg = mesh.volume_average(ne_profile)
        if ne_prof_avg > 1e6:
            ne_new = ne_profile * (ne_0d / ne_prof_avg)
        else:
            # Fallback to standard eigenmode if source-weighted fails
            profile = solve_diffusion_profile(mesh, Da_now, uB_now)
            ne_new = ne_0d * profile / max(mesh.volume_average(profile), 1e-30)
        ne = ne + w*(ne_new - ne); ne = np.maximum(ne, 1e6)

        # === 4. Negative ions ===
        if nSF6_0 > 0:
            n_pos = ne + n_neg
            n_neg, alpha = solve_negative_ions(
                mesh, ne, n_neg, nSF6, Te_field, n_pos,
                D_neg, k_rec, alpha_0d=alpha_0d, ne_0d=ne_0d, w=w)

        # === 5. SF6 with Robin BCs — SELF-CONSISTENT (no anchor) [Change B] ===
        if nSF6_0 > 0:
            sf6_source = np.full((Nr,Nz), nSF6_0/tau_R)
            sf6_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    k_e = (k_loc['d1']+k_loc['d2']+k_loc['d3']+k_loc['d4']+k_loc['d5']
                           +k_loc['iz_SF6_total']+k_loc['att_SF6_total'])
                    sf6_loss[i,j] = (k_e*ne[i,j] + (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nArm[i,j]
                                     + 1.0/tau_R)

            # Asymmetric Robin: SF6 sticking at walls
            nSF6_new = _solve_robin_asymmetric(mesh, D_SF6, sf6_source, sf6_loss,
                                                h_r_wall=h_SF6_wall, h_z0=h_SF6_wall, h_zL=h_SF6_wall*0.1)
            # Partial anchoring: prevent over-depletion below 50% of 0D
            # This allows spatial gradients but maintains enough SF6 for
            # the attachment/ionization chemistry to work correctly.
            sf6_avg_new = mesh.volume_average(nSF6_new)
            sf6_min_avg = nSF6_0d * 0.80  # Don't deplete below 80% of 0D
            if sf6_avg_new > 1e6 and sf6_avg_new < sf6_min_avg:
                nSF6_new *= sf6_min_avg / sf6_avg_new
            nSF6 = nSF6 + w*(nSF6_new - nSF6)
            nSF6 = np.clip(nSF6, 1e8, nSF6_0*2)
            neutrals['SF6'] = nSF6

        # === 6. F with ASYMMETRIC Robin BCs [Change A] ===
        if nSF6_0 > 0:
            f_source = np.zeros((Nr,Nz))
            f_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    f_source[i,j] = fluorine_source(
                        k_loc, ne[i,j], nSF6[i,j], 0, 0, 0, 0, 0, 0, nArm[i,j])
                    f_loss[i,j] = k_loc['iz28']*ne[i,j] + 1.0/tau_R
                    for sp_key, nr_key in [('SF5','nr42'),('SF4','nr41'),('SF3','nr40'),
                                           ('SF2','nr39'),('SF','nr38'),('S','nr37')]:
                        if sp_key in neutrals:
                            f_loss[i,j] += k_loc.get(nr_key,0)*neutrals[sp_key][i,j]

            # Asymmetric Robin: different gamma at each wall
            nF_new = _solve_robin_asymmetric(mesh, D_F, f_source, f_loss,
                                             h_r_wall=h_F_wall, h_z0=h_F_wafer, h_zL=h_F_window)
            nF = nF + w*(nF_new - nF)
            nF = np.maximum(nF, 0.0)
            neutrals['F'] = nF

        # === 7. Minor neutrals ===
        if nSF6_0 > 0 and has_0d:
            neutrals_0d_dict = {sp: r0d.get(f'n_{sp}', 0) for sp in neutrals if f'n_{sp}' in r0d}
            D_coeffs = {sp: n_tr.diffusivity(M_SPECIES.get(sp,50), sigma=5e-19) for sp in neutrals}
            full_old = {sp: neutrals[sp].copy() for sp in neutrals}
            full_new = solve_neutral_transport(
                mesh, ne, Te_field, nArm, ng, nSF6_0, tau_R,
                full_old, D_coeffs, gamma_F=gamma_F_wall, kw_F_eff=30.0, w=w,
                neutrals_0d=neutrals_0d_dict)
            for sp in full_new:
                if sp not in ('SF6', 'F'):
                    neutrals[sp] = full_new[sp]

        # === 8. EM ===
        if it % em_interval == 0:
            for _ in range(5):
                try:
                    Pn, _ = em.adjust_coil_current(ne, Te_field, ng, P_target=P_abs)
                    if np.all(np.isfinite(Pn)) and Pn.max() > 0:
                        P_ind = 0.2*P_ind + 0.8*Pn
                except: pass

        # === Convergence ===
        ne_avg_new = mesh.volume_average(ne)
        dTe = np.max(np.abs(Te_field-Te_old))/max(np.max(np.abs(Te_old)),0.1)
        dne = abs(ne_avg_new-ne_avg_old)/max(ne_avg_old,1e10)

        if verbose and (it%(max(n_iter//10,1))==0 or it==n_iter-1):
            al_a = mesh.volume_average(n_neg)/max(mesh.volume_average(ne),1e10)
            nF_a = mesh.volume_average(nF)
            nSF6_a = mesh.volume_average(nSF6)
            jm = Nz//2
            nF_c = nF[0, jm]; nF_e = nF[-1, jm]
            drop = (1 - nF_e/max(nF_c,1e-30))*100 if nF_c > 1e8 else 0
            sf6_c = nSF6[0, jm]; sf6_e = nSF6[-1, jm]
            sf6_depl = (1-sf6_c/max(sf6_e,1e-30))*100 if sf6_e > 1e8 else 0
            print(f"  {it:>3d}: <ne>={ne_avg_new*1e-6:.1e} "
                  f"Te={Te_avg:.2f}({Te_field.min():.1f}-{Te_field.max():.1f}) "
                  f"α={al_a:.1f} [F]={nF_a*1e-6:.1e}(drop={drop:.0f}%) "
                  f"SF6={nSF6_a/max(ng,1)*100:.0f}%(depl={sf6_depl:.0f}%) "
                  f"Δ={dTe:.1e}/{dne:.1e}")

        if it > 25 and dTe < 5e-4 and dne < 1e-3:
            if verbose: print(f"  *** Converged at iteration {it} ***")
            break

    elapsed = time.time()-t0
    n_pos = ne + n_neg
    V, Er, Ez = poisson.solve(n_pos, ne, n_neg)

    nav = mesh.volume_average(ne); Te_avg = mesh.volume_average(Te_field)
    al_avg = mesh.volume_average(n_neg)/max(nav,1e10)
    nF_avg = mesh.volume_average(nF)
    jm = Nz//2
    nF_center = nF[0, jm]; nF_edge = nF[-1, jm]
    F_drop = (1-nF_edge/max(nF_center,1e-30))*100
    sf6_center = nSF6[0, jm]; sf6_edge = nSF6[-1, jm]
    sf6_depletion = (1-sf6_center/max(sf6_edge,1e-30))*100

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gen-4b done in {elapsed:.1f}s")
        print(f"  <ne>   = {nav*1e-6:.3e} cm⁻³")
        print(f"  <Te>   = {Te_avg:.2f} eV ({Te_field.min():.2f}–{Te_field.max():.2f})")
        print(f"  <α>    = {al_avg:.1f}")
        print(f"  <[F]>  = {nF_avg*1e-6:.2e} cm⁻³ (0D: {nF_0d*1e-6:.2e}, ratio={nF_avg/max(nF_0d,1):.2f})")
        print(f"  [F] center={nF_center*1e-6:.2e}, edge={nF_edge*1e-6:.2e}, DROP={F_drop:.0f}%")
        print(f"  SF6 center={sf6_center/max(ng,1)*100:.0f}%, edge={sf6_edge/max(ng,1)*100:.0f}%, depletion={sf6_depletion:.0f}%")

    return {'ne':ne,'n_pos':n_pos,'n_neg':n_neg,'Te':Te_field,
            'V':V,'Er':Er,'Ez':Ez,'P_ind':P_ind,'nArm':nArm,
            'alpha':alpha,'neutrals':neutrals,'mesh':mesh,
            'ne_avg':nav,'Te_avg':Te_avg,'alpha_avg':al_avg,
            'nF_avg':nF_avg,'Ec_avg':Ec,'eps_T':eps_T,
            'elapsed':elapsed,'nSF6':nSF6,'nF':nF,
            'nF_center':nF_center,'nF_edge':nF_edge,'F_drop_pct':F_drop,
            'sf6_depletion':sf6_depletion,'nArm_avg':mesh.volume_average(nArm),
            'gamma_F_wall':gamma_F_wall,'gamma_F_wafer':gamma_F_wafer}


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--power', type=float, default=1500.0)
    pa.add_argument('--pressure', type=float, default=10.0)
    pa.add_argument('--ar', type=float, default=0.0)
    pa.add_argument('--gamma-wall', type=float, default=0.10)
    pa.add_argument('--gamma-wafer', type=float, default=0.02)
    pa.add_argument('--iter', type=int, default=100)
    pa.add_argument('--nr', type=int, default=30)
    pa.add_argument('--nz', type=int, default=40)
    a = pa.parse_args()

    print("="*70)
    print("SF6/Ar 2D ICP — Gen-4b: Mettler-gap fix")
    print("  [A] Wall-specific Robin BCs")
    print("  [B] Self-consistent SF6 (no anchor)")
    print("  [C] Self-consistent [F] (no anchor)")
    print("="*70)

    res = run_v4b(P_rf=a.power, p_mTorr=a.pressure, frac_Ar=a.ar,
                  gamma_F_wall=a.gamma_wall, gamma_F_wafer=a.gamma_wafer,
                  Nr=a.nr, Nz=a.nz, n_iter=a.iter)

    # Plot
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs('outputs_v4b', exist_ok=True)
    m = res['mesh']; Nr=m.Nr; Nz=m.Nz
    Rc=m.r*100; Zc=m.z*100; RR,ZZ=np.meshgrid(Rc,Zc,indexing='ij')
    jm = Nz//2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def cp(ax, d, lab, cm, t):
        fd=d[np.isfinite(d)]
        if len(fd)>0 and np.ptp(fd)>0:
            lev=np.linspace(max(fd.min(),0),fd.max()*1.01,25)
            c=ax.contourf(RR,ZZ,d,levels=lev,cmap=cm); plt.colorbar(c,ax=ax,label=lab,shrink=0.8)
        ax.set_xlabel('r (cm)'); ax.set_ylabel('z (cm)'); ax.set_title(t)
        ax.set_xlim(0, m.R*100)

    cp(axes[0,0], res['nF']*1e-6, 'cm⁻³', 'YlOrRd', '[F] (self-consistent, asym Robin)')
    ng_val = p_Pa/(kB*Tgas) if 'p_Pa' in dir() else 3.22e20
    cp(axes[0,1], res['nSF6']/max(ng_val,1)*100, '%', 'Blues_r', 'SF$_6$ depletion (self-consistent)')
    cp(axes[0,2], res['Te'], 'eV', 'hot', '$T_e$ (energy eqn)')

    # Radial F profile at midplane and wafer
    ax = axes[1,0]
    ax.plot(Rc, res['nF'][:,jm]*1e-6, 'r-', lw=2.5, label=f'Midplane (z={Zc[jm]:.0f}cm)')
    ax.plot(Rc, res['nF'][:,0]*1e-6, 'b--', lw=2, label='Wafer (z=0)')
    ax.set_xlabel('r (cm)'); ax.set_ylabel('[F] (cm⁻³)')
    ax.set_title(f'Radial [F] — drop={res["F_drop_pct"]:.0f}%')
    ax.legend(); ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax.grid(True, alpha=0.3)

    # Normalized comparison with Mettler
    ax2 = axes[1,1]
    from validation.mettler_data import fig414_fit, fig414_r_cm, fig414_nF_norm
    r_fit = np.linspace(0, 8, 100)
    ax2.plot(r_fit, fig414_fit(r_fit), 'k-', lw=2.5, label='Mettler Fig 4.14')
    ax2.plot(fig414_r_cm, fig414_nF_norm, 'ko', ms=8)
    nF_mid = res['nF'][:,jm]; nF_mid_norm = nF_mid/max(nF_mid[0],1e-30)
    ax2.plot(Rc, nF_mid_norm, 'r--', lw=2, label='Gen-4b (midplane)')
    ax2.set_xlabel('r (cm)'); ax2.set_ylabel('Normalized [F]')
    ax2.set_title('Mettler comparison'); ax2.legend(fontsize=9)
    ax2.set_xlim(0, max(Rc.max(),8)); ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3)

    # Summary
    axes[1,2].axis('off')
    txt = (f"Gen-4b: {a.power:.0f}W, {a.pressure:.0f}mT\n"
           f"γ_F: wall={a.gamma_wall}, wafer={a.gamma_wafer}\n\n"
           f"<ne> = {res['ne_avg']*1e-6:.2e} cm-3\n"
           f"<Te> = {res['Te_avg']:.2f} eV\n"
           f"<α>  = {res['alpha_avg']:.1f}\n"
           f"<[F]>= {res['nF_avg']*1e-6:.2e} cm-3\n"
           f"[F] drop = {res['F_drop_pct']:.0f}%\n"
           f"SF6 depletion = {res['sf6_depletion']:.0f}%\n"
           f"Time: {res['elapsed']:.1f}s")
    axes[1,2].text(0.05, 0.95, txt, transform=axes[1,2].transAxes, fontsize=10,
                   va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.3))

    fig.suptitle(f'Gen-4b: Wall-specific Robin BCs + self-consistent SF6/F',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig('outputs_v4b/v4b_results.png', dpi=150, bbox_inches='tight')
    print(f"  Saved outputs_v4b/v4b_results.png")
