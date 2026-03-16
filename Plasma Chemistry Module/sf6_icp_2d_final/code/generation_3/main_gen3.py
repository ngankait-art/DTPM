#!/usr/bin/env python3
"""
SF6/Ar 2D ICP Simulator — Generation 3

Advances over Gen-2:
  [1] Self-consistent 2D energy equation for Te(r,z)
      — solves D_ε∇²(ne·ε̄) + P_ind/e - ne·ν_ε·ε̄ = 0
      — replaces the empirical Te ∝ P_ind^0.15
  [2] Full neutral diffusion transport for SF6 and F
      — SF6: Neumann BCs (no wall consumption), source = feed
      — F:   Dirichlet BCs (wall recombination γ_F = 0.01)
      — produces the radial [F](r) gradient for Mettler validation
  [3] Stronger EM—plasma feedback
      — 5 sub-iterations, 80% coupling, every 3 outer iterations
  [4] Partial anchoring release
      — ne: from 0D (stable backbone)
      — Te: from 2D energy equation, shape renormalized to 0D average
      — SF6: from 2D diffusion, anchored to 0D average
      — F:   SELF-CONSISTENT from 2D diffusion (no 0D anchor)
      — n_neg: from 2D diffusion, anchored to 0D alpha
"""

import sys, os, time, argparse
import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh.mesh_generator import Mesh2D
from solvers.poisson import PoissonSolver
from solvers.em_solver import EMSolver
from solvers.diffusion_profile import solve_diffusion_profile
from solvers.energy_2d import _solve_energy_diffusion, compute_Eloss_field
from solvers.transport_2d import solve_negative_ions, solve_neutral_transport, _solve_diffusion_neumann
from transport.transport import IonTransport, NeutralTransport, ambipolar_Da, ElectronTransport
from chemistry.sf6_rates import rates, compute_troe_rates, fluorine_source, M_SPECIES

AMU = 1.66054e-27; MTORR_TO_PA = 0.133322; cm3 = 1e-6


def _solve_diffusion_dirichlet(mesh, D, source, loss_freq):
    """D∇²n + S - ν·n = 0 with Dirichlet n=0 at walls.
    Wrapper around the energy solver's infrastructure."""
    return _solve_energy_diffusion(mesh, D, source, loss_freq)


def run_v3(P_rf=1500.0, p_mTorr=10.0, frac_Ar=0.0, Tgas=300.0, T_neg=0.3,
           gamma_F=0.01, eta=0.12,
           Nr=30, Nz=40, n_iter=120, em_interval=3, verbose=True):
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
    e_tr = ElectronTransport(ng, x_SF6=1-frac_Ar, x_Ar=frac_Ar)
    k_rec = 1.5e-9 * cm3

    D_Arm = n_tr.diffusivity(39.948, sigma=3e-19)
    D_F = n_tr.diffusivity(M_SPECIES['F'], sigma=4e-19)
    D_SF6 = n_tr.diffusivity(M_SPECIES['SF6'], sigma=6e-19)
    D_neg = i_tr.D0
    Lambda2 = 1.0/((pi/mesh.L)**2 + (2.405/mesh.R)**2)
    kw_Arm = D_Arm/Lambda2
    v_F = np.sqrt(8*kB*Tgas/(pi*M_SPECIES['F']*AMU))
    wall_A = 2*pi*mesh.R**2 + 2*pi*mesh.R*mesh.L
    kw_F_eff = 1.0/(Lambda2/D_F + 2*V_reactor*(2-gamma_F)/(wall_A*v_F*gamma_F+1e-30))

    # --- 0D backbone ---
    has_0d = False; ne_0d = 1e16; Te_0d = 3.0; alpha_0d = 0; Ec = 100; eps_T = 120
    nSF6_0d = nSF6_0*0.3; nF_0d = 0; nArm_0d = 0; r0d = {}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared_modules'))
        from sf6_unified import solve_model as solve_0d
        r0d = solve_0d(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                       eta=eta, Tgas=Tgas, T_neg=T_neg)
        ne_0d = r0d['ne']; Te_0d = r0d['Te']; alpha_0d = r0d['alpha']
        nSF6_0d = r0d.get('n_SF6', nSF6_0*0.3)
        nF_0d = r0d.get('n_F', 0); Ec = r0d.get('Ec', 100)
        eps_T = r0d.get('eps_T', 120); nArm_0d = r0d.get('nArm', 0)
        has_0d = True
    except Exception as e:
        if verbose: print(f"  0D unavailable ({e}), using Ar-only fallback")

    if verbose:
        print(f"Mesh: {mesh}")
        print(f"0D: ne={ne_0d*1e-6:.2e}cm-3 Te={Te_0d:.2f}eV α={alpha_0d:.1f} [F]={nF_0d*1e-6:.2e}cm-3")

    # --- Initialize 2D fields ---
    Da0 = ambipolar_Da(alpha_0d, Te_0d, i_tr.D0, T_neg)
    gamma_en = Te_0d/max(T_neg, 0.01)
    uB0 = np.sqrt(eC*Te_0d*(1+alpha_0d)/(Mi_kg*(1+gamma_en*alpha_0d)))
    profile = solve_diffusion_profile(mesh, Da0, uB0)
    pn = profile / max(mesh.volume_average(profile), 1e-30)

    ne = ne_0d * pn
    Te_field = np.full((Nr,Nz), Te_0d)
    nArm = np.zeros((Nr,Nz))
    n_neg = np.full((Nr,Nz), alpha_0d*ne_0d) * pn if alpha_0d > 0.01 else np.zeros((Nr,Nz))
    alpha = np.where(ne > 1e10, n_neg/ne, 0.0)

    # Neutrals: SF6 and F from diffusion; others from 0D
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

    D_eps = e_tr.energy_diffusivity(Te_0d)

    if verbose:
        sf6_pct = (1-frac_Ar)*100
        print(f"\nGen-3: {n_iter} iter, {P_rf:.0f}W(abs={P_abs:.0f}), "
              f"{p_mTorr:.0f}mT, {frac_Ar*100:.0f}%Ar/{sf6_pct:.0f}%SF6")
        print(f"  D_eps={D_eps:.2e}, D_F={D_F:.2e}, D_SF6={D_SF6:.2e}")
        print("-"*70)

    w = 0.12

    for it in range(n_iter):
        Te_old = Te_field.copy()
        ne_avg_old = mesh.volume_average(ne)

        # === 1. Ar* from local algebraic balance ===
        for i in range(Nr):
            for j in range(Nz):
                k_loc = rates(Te_field[i,j]); ne_ij = ne[i,j]
                qh = (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nSF6[i,j]+k_loc['qnch_F']*nF[i,j]
                den = (k_loc['Ar_iz_m']+k_loc['Ar_q'])*ne_ij+kw_Arm+qh
                nArm[i,j] = k_loc['Ar_exc']*ne_ij*nAr/max(den,1.0) if ne_ij>1e8 else 0

        # === 2. Te(r,z) FROM SELF-CONSISTENT ENERGY EQUATION [Step 1] ===
        Eloss_field = compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm)

        eps_bar = np.maximum(1.5 * Te_field, 0.5)
        nu_eps = np.where(eps_bar > 0.1, Eloss_field / eps_bar, 1.0)
        nu_eps = np.clip(nu_eps, 1.0, 1e12)

        ne_safe = np.maximum(ne, 1e8)
        source_per_ne = P_ind / (ne_safe * eC)  # [eV/s] per electron

        # D_eff: use a reduced energy diffusion coefficient
        # The (5/3)De overestimates at low pressure. Use De itself (Hagelaar 2005).
        D_eff = max(D_eps / 5.0, 0.1)  # Reduce to get physically reasonable Te gradients

        # Solve for ε̄ with Neumann BCs (energy escapes via collisional loss, not wall flux)
        # Using Neumann prevents Te→0 at walls which is unphysical for energy
        eps_new = _solve_diffusion_neumann(mesh, D_eff, source_per_ne, nu_eps)
        eps_new = np.clip(eps_new, 0.5, 30.0)

        Te_new = (2.0/3.0) * eps_new
        Te_new = np.clip(Te_new, 0.5, 12.0)

        # Anchor: renormalize so <Te> matches 0D particle balance
        Te_avg_raw = mesh.volume_average(Te_new)
        if Te_avg_raw > 0.1 and Te_0d > 0.1:
            Te_new *= Te_0d / Te_avg_raw

        Te_new = np.clip(Te_new, 0.8, 12.0)
        Te_field = Te_field + w*(Te_new - Te_field)
        Te_field = np.clip(Te_field, 0.8, 12.0)

        # === 3. ne PROFILE from ambipolar diffusion + 0D magnitude ===
        Te_avg = mesh.volume_average(Te_field)
        al_avg = mesh.volume_average(n_neg)/max(mesh.volume_average(ne), 1e10)
        Da_now = ambipolar_Da(al_avg, Te_avg, i_tr.D0, T_neg)
        gamma_t = Te_avg/max(T_neg, 0.01)
        uB_now = np.sqrt(eC*Te_avg*(1+al_avg)/(Mi_kg*(1+gamma_t*al_avg)))
        profile = solve_diffusion_profile(mesh, Da_now, uB_now)
        pn = profile/max(mesh.volume_average(profile), 1e-30)

        ne_new = ne_0d * pn
        ne = ne + w*(ne_new - ne); ne = np.maximum(ne, 1e6)

        # === 4. NEGATIVE ION TRANSPORT ===
        if nSF6_0 > 0:
            n_pos = ne + n_neg
            n_neg, alpha = solve_negative_ions(
                mesh, ne, n_neg, nSF6, Te_field, n_pos,
                D_neg, k_rec, alpha_0d=alpha_0d, ne_0d=ne_0d, w=w)
        else:
            n_neg = np.zeros((Nr,Nz)); alpha = np.zeros((Nr,Nz))

        # === 5. SF6 DIFFUSION TRANSPORT [Step 2] ===
        if nSF6_0 > 0:
            # Source: feed gas supply; Loss: electron-impact + Penning + pumping
            sf6_source = np.full((Nr,Nz), nSF6_0/tau_R)
            sf6_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    k_e = (k_loc['d1']+k_loc['d2']+k_loc['d3']+k_loc['d4']+k_loc['d5']
                           +k_loc['iz_SF6_total']+k_loc['att_SF6_total'])
                    sf6_loss[i,j] = k_e*ne[i,j] + (k_loc['Penn_SF6']+k_loc['qnch_SF6'])*nArm[i,j] + 1.0/tau_R
                    # Recombination source
                    sf6_source[i,j] += k_loc.get('nr42',0)*neutrals.get('SF5',np.zeros((Nr,Nz)))[i,j]*nF[i,j]

            nSF6_new = _solve_diffusion_neumann(mesh, D_SF6, sf6_source, sf6_loss)
            # Anchor to 0D average
            avg_new = mesh.volume_average(nSF6_new)
            if avg_new > 1e6 and nSF6_0d > 0:
                nSF6_new *= nSF6_0d / avg_new
            nSF6 = nSF6 + w*(nSF6_new - nSF6)
            nSF6 = np.clip(nSF6, 1e8, nSF6_0*2)
            neutrals['SF6'] = nSF6

        # === 6. F ATOM DIFFUSION TRANSPORT [Step 2, SELF-CONSISTENT] ===
        if nSF6_0 > 0:
            f_source = np.zeros((Nr,Nz))
            f_loss = np.zeros((Nr,Nz))
            for i in range(Nr):
                for j in range(Nz):
                    k_loc = rates(Te_field[i,j])
                    f_source[i,j] = fluorine_source(
                        k_loc, ne[i,j], nSF6[i,j], 0, 0, 0, 0, 0, 0, nArm[i,j])
                    # Wall loss as effective volumetric loss (same as 0D kw_F)
                    f_loss[i,j] = kw_F_eff + k_loc['iz28']*ne[i,j] + 1.0/tau_R
                    for sp_key, nr_key in [('SF5','nr42'),('SF4','nr41'),('SF3','nr40'),
                                           ('SF2','nr39'),('SF','nr38'),('S','nr37')]:
                        if sp_key in neutrals:
                            f_loss[i,j] += k_loc.get(nr_key,0)*neutrals[sp_key][i,j]

            # Neumann BCs (wall loss handled by kw_F_eff volumetric term)
            # This gives the same <[F]> as the 0D model when kw_F_eff matches,
            # but produces a spatial profile peaked at the centre (where source is highest)
            nF_new = _solve_diffusion_neumann(mesh, D_F, f_source, f_loss)
            # NO ANCHORING for F — this is the self-consistent quantity [Step 4]
            nF = nF + w*(nF_new - nF)
            nF = np.maximum(nF, 0.0)
            neutrals['F'] = nF

        # === 7. OTHER NEUTRALS: algebraic from transport_2d ===
        if nSF6_0 > 0 and has_0d:
            neutrals_0d_dict = {}
            for sp in neutrals:
                key = f'n_{sp}'
                if key in r0d: neutrals_0d_dict[sp] = r0d[key]
            D_coeffs = {sp: n_tr.diffusivity(M_SPECIES.get(sp,50), sigma=5e-19) for sp in neutrals}
            # Pass full neutrals dict (needed for cross-references in chemistry)
            # but only take back the minor species
            full_old = {sp: neutrals[sp].copy() for sp in neutrals}
            full_new = solve_neutral_transport(
                mesh, ne, Te_field, nArm, ng, nSF6_0, tau_R,
                full_old, D_coeffs, gamma_F=gamma_F, kw_F_eff=kw_F_eff, w=w,
                neutrals_0d=neutrals_0d_dict)
            for sp in full_new:
                if sp not in ('SF6', 'F'):  # Don't overwrite diffusion-solved species
                    neutrals[sp] = full_new[sp]

        # === 8. EM—PLASMA FEEDBACK [Step 3] ===
        if it % em_interval == 0:
            for em_sub in range(5):
                try:
                    Pn, _ = em.adjust_coil_current(ne, Te_field, ng, P_target=P_abs)
                    if np.all(np.isfinite(Pn)) and Pn.max() > 0:
                        P_ind = 0.2*P_ind + 0.8*Pn
                except: pass

        # === CONVERGENCE ===
        ne_avg_new = mesh.volume_average(ne)
        dTe = np.max(np.abs(Te_field-Te_old))/max(np.max(np.abs(Te_old)),0.1)
        dne = abs(ne_avg_new-ne_avg_old)/max(ne_avg_old,1e10)

        if verbose and (it%(max(n_iter//12,1))==0 or it==n_iter-1):
            al_a = mesh.volume_average(n_neg)/max(mesh.volume_average(ne),1e10)
            nF_a = mesh.volume_average(nF)
            nSF6_a = mesh.volume_average(nSF6)
            P_tot = mesh.volume_average(P_ind)*V_reactor
            print(f"  {it:>4d}: <ne>={ne_avg_new:.2e}({ne_avg_new*1e-6:.1e}cm⁻³) "
                  f"Te={mesh.volume_average(Te_field):.2f}({Te_field.min():.1f}-{Te_field.max():.1f}) "
                  f"α={al_a:.1f} [F]={nF_a*1e-6:.1e} SF6={nSF6_a/max(ng,1)*100:.0f}% "
                  f"P={P_tot:.0f}W Δ={dTe:.1e}/{dne:.1e}")

        if it > 20 and dTe < 5e-4 and dne < 1e-3:
            if verbose: print(f"  *** Converged at iteration {it} ***")
            break

    elapsed = time.time()-t0
    n_pos = ne + n_neg
    V, Er, Ez = poisson.solve(n_pos, ne, n_neg)

    nav = mesh.volume_average(ne); Te_avg = mesh.volume_average(Te_field)
    al_avg = mesh.volume_average(n_neg)/max(nav,1e10)
    nF_avg = mesh.volume_average(nF)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gen-3 done in {elapsed:.1f}s")
        print(f"  <ne>   = {nav:.3e} m⁻³ ({nav*1e-6:.3e} cm⁻³)")
        print(f"  ne_pk  = {ne.max():.3e} m⁻³")
        print(f"  <Te>   = {Te_avg:.2f} eV (range {Te_field.min():.2f}–{Te_field.max():.2f})")
        print(f"  <α>    = {al_avg:.1f} (range {alpha.min():.1f}–{alpha.max():.1f})")
        print(f"  <[F]>  = {nF_avg:.2e} m⁻³ ({nF_avg*1e-6:.2e} cm⁻³) [SELF-CONSISTENT]")
        print(f"    0D [F] = {nF_0d:.2e} m⁻³ ({nF_0d*1e-6:.2e} cm⁻³)")
        print(f"    Ratio 2D/0D = {nF_avg/max(nF_0d,1):.2f}")
        print(f"  [F] peak = {nF.max()*1e-6:.2e} cm⁻³")
        print(f"  [F] at wafer (z=0, r=0) = {nF[0,0]*1e-6:.2e} cm⁻³")
        print(f"  <SF6> = {mesh.volume_average(nSF6)/max(ng,1)*100:.1f}% of gas")
        if np.ptp(V) > 0.01: print(f"  V: {V.min():.1f}–{V.max():.1f} V")
        for sp in ['SF5','SF4','SF3','F2']:
            if sp in neutrals:
                print(f"  <{sp:>3s}> = {mesh.volume_average(neutrals[sp]):.2e} m⁻³")

    return {'ne':ne,'n_pos':n_pos,'n_neg':n_neg,'Te':Te_field,
            'V':V,'Er':Er,'Ez':Ez,'P_ind':P_ind,'nArm':nArm,
            'alpha':alpha,'neutrals':neutrals,'mesh':mesh,
            'ne_avg':nav,'Te_avg':Te_avg,'nArm_avg':mesh.volume_average(nArm),
            'alpha_avg':al_avg,'nF_avg':nF_avg,'Ec_avg':Ec,'eps_T':eps_T,
            'elapsed':elapsed,'nSF6':nSF6,'nF':nF}


def plot_v3(res, out='outputs_v3', p_mTorr=10, frac_Ar=0.0, P_rf=1500):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    m = res['mesh']; Nr=m.Nr; Nz=m.Nz
    Rc=m.r*100; Zc=m.z*100; RR,ZZ=np.meshgrid(Rc,Zc,indexing='ij')

    fig = plt.figure(figsize=(18, 16))

    def cplot(pos, data, label, cmap, title):
        ax = fig.add_subplot(4,3,pos)
        fd = data[np.isfinite(data)]
        if len(fd)>0 and np.ptp(fd)>0:
            lev = np.linspace(max(fd.min(),0), fd.max()*1.01, 25)
            c = ax.contourf(RR, ZZ, data, levels=lev, cmap=cmap)
            plt.colorbar(c, ax=ax, label=label, shrink=0.8)
        ax.set_xlabel('r (cm)'); ax.set_ylabel('z (cm)'); ax.set_title(title, fontsize=11)
        ax.set_xlim(0, m.R*100); return ax

    # Row 1: ne, Te, P_ind
    cplot(1, res['ne']*1e-6, 'cm⁻³', 'plasma', '$n_e$')
    cplot(2, res['Te'], 'eV', 'hot', '$T_e$ (energy eqn)')
    cplot(3, res['P_ind']*1e-3, 'kW/m³', 'inferno', '$P_{ind}$')

    # Row 2: nArm / alpha / n_neg
    if res['nArm'].max() > 1e8:
        cplot(4, res['nArm']*1e-6, 'cm⁻³', 'viridis', 'Ar*')
    else:
        fig.add_subplot(4,3,4).axis('off')
    if res['alpha'].max() > 0.01:
        cplot(5, np.clip(res['alpha'],0,500), '', 'coolwarm', '$\\alpha$ (spatially resolved)')
    else:
        fig.add_subplot(4,3,5).axis('off')
    if res['n_neg'].max() > 1e8:
        cplot(6, res['n_neg']*1e-6, 'cm⁻³', 'PuBu', '$n_-$ (trapped)')
    else:
        fig.add_subplot(4,3,6).axis('off')

    # Row 3: SF6 depletion, [F] profile, potential
    if res['nSF6'].max() > 1e10:
        ng_loc = p_mTorr*MTORR_TO_PA/(kB*300)
        cplot(7, res['nSF6']/max(ng_loc,1)*100, '% of gas', 'Blues_r', 'SF$_6$ depletion')
    else:
        fig.add_subplot(4,3,7).axis('off')
    if res['nF'].max() > 1e10:
        cplot(8, res['nF']*1e-6, 'cm⁻³', 'YlOrRd', '[F] (self-consistent)')
    else:
        fig.add_subplot(4,3,8).axis('off')
    if np.ptp(res['V']) > 0.01:
        cplot(9, res['V'], 'V', 'RdBu_r', 'Potential')
    else:
        fig.add_subplot(4,3,9).axis('off')

    # Row 4: radial profiles, Te profile, summary
    ax10 = fig.add_subplot(4,3,10); jm = Nz//2
    ax10.plot(Rc, res['ne'][:,jm]*1e-6, 'b-', lw=2.5, label='$n_e$')
    if res['nF'].max() > 1e10:
        ax10.plot(Rc, res['nF'][:,jm]*1e-6, 'r:', lw=2.5, label='[F]')
    if res['n_neg'].max() > 1e10:
        ax10.plot(Rc, res['n_neg'][:,jm]*1e-6, 'm-.', lw=2, label='$n_-$')
    if res['nArm'].max() > 1e8:
        ax10.plot(Rc, res['nArm'][:,jm]*1e-6, 'g--', lw=2, label='Ar*')
    ax10.set_xlabel('r (cm)'); ax10.set_ylabel('cm⁻³')
    ax10.set_title('Radial (midplane)'); ax10.legend(fontsize=8)
    ax10.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    ax11 = fig.add_subplot(4,3,11)
    ax11.plot(Zc, res['Te'][0,:], 'r-', lw=2.5, label='Axis')
    ax11.plot(Zc, res['Te'][Nr//2,:], 'r--', lw=1.5, alpha=0.6, label=f'r={Rc[Nr//2]:.0f}cm')
    ax11.set_xlabel('z (cm)'); ax11.set_ylabel('$T_e$ (eV)')
    ax11.set_title('$T_e$ (energy equation)'); ax11.legend(fontsize=8)

    ax12 = fig.add_subplot(4,3,12); ax12.axis('off')
    txt = (f"Gen-3 Results\n"
           f"  P={P_rf:.0f}W(abs={P_rf*0.12:.0f}), {p_mTorr:.0f}mT\n"
           f"  {frac_Ar*100:.0f}%Ar/{(1-frac_Ar)*100:.0f}%SF6\n\n"
           f"  <ne>  = {res['ne_avg']*1e-6:.2e} cm-3\n"
           f"  <Te>  = {res['Te_avg']:.2f} eV ({res['Te'].min():.1f}-{res['Te'].max():.1f})\n"
           f"  <alpha> = {res['alpha_avg']:.1f}\n"
           f"  <[F]> = {res['nF_avg']*1e-6:.2e} cm-3 [self-consist.]\n"
           f"  [F]pk = {res['nF'].max()*1e-6:.2e} cm-3\n"
           f"  [F](r=0,z=0) = {res['nF'][0,0]*1e-6:.2e}\n"
           f"  Ec={res['Ec_avg']:.0f}, eT={res['eps_T']:.0f}\n"
           f"  Time: {res['elapsed']:.1f}s")
    ax12.text(0.05, 0.95, txt, transform=ax12.transAxes, fontsize=9.5,
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.3))

    gas = f"{frac_Ar*100:.0f}%Ar/{(1-frac_Ar)*100:.0f}%SF6"
    fig.suptitle(f'Gen-3 ICP 2D: {P_rf:.0f}W, {p_mTorr:.0f}mTorr, {gas}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(f'{out}/v3_profiles.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {out}/v3_profiles.png"); plt.close()

    # --- Radial [F] profile at wafer plane ---
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Rc, res['nF'][:,0]*1e-6, 'r-', lw=2.5, label='[F] at wafer (z=0)')
    ax.plot(Rc, res['nF'][:,jm]*1e-6, 'r--', lw=2, alpha=0.6, label=f'[F] at midplane (z={Zc[jm]:.0f}cm)')
    ax.set_xlabel('Radial position r (cm)', fontsize=12)
    ax.set_ylabel('[F] (cm$^{-3}$)', fontsize=12)
    ax.set_title(f'Radial fluorine profile — {P_rf:.0f}W, {p_mTorr:.0f}mTorr, {gas}', fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_xlim(0, m.R*100)
    fig2.tight_layout()
    fig2.savefig(f'{out}/v3_F_radial.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {out}/v3_F_radial.png"); plt.close()


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--power',type=float,default=1500.0)
    pa.add_argument('--pressure',type=float,default=10.0)
    pa.add_argument('--ar',type=float,default=0.0)
    pa.add_argument('--iter',type=int,default=120)
    pa.add_argument('--nr',type=int,default=30)
    pa.add_argument('--nz',type=int,default=40)
    pa.add_argument('--no-plot',action='store_true')
    a = pa.parse_args()
    print("="*70)
    print("SF6/Ar 2D ICP Simulator — Generation 3")
    print("  [1] Self-consistent Te from energy equation")
    print("  [2] SF6 + F diffusion transport")
    print("  [3] Strong EM-plasma feedback")
    print("  [4] Partial anchoring release ([F] self-consistent)")
    print("="*70)
    res = run_v3(P_rf=a.power, p_mTorr=a.pressure, frac_Ar=a.ar,
                 Nr=a.nr, Nz=a.nz, n_iter=a.iter)
    if not a.no_plot:
        plot_v3(res, p_mTorr=a.pressure, frac_Ar=a.ar, P_rf=a.power)
