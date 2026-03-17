"""
SF6 Global Model — ODE time-integration solver
Integrates dn_i/dt forward from pre-discharge initial conditions until steady state.
This inherently respects conservation and avoids spurious roots.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Import all constants, species, rate coefficients from the main model
from sf6_global_model import (
    kB, eV, e_q, m_e, amu, pi, 
    SF6, SF5, SF4, SF3, F2, F_, SF5p, SF4p, SF3p, F2p, SF6m, Fm, EL,
    N_SP, N_NEU, N_SURF, TH_F, TH_SF3, TH_SF4, TH_SF5,
    M, V_re, A_ax, A_rad, A_re, LAMBDA, R_re, L_re,
    T_gas, Q_SF6_mol, Q_Ar_mol, Q_total, f_SF6, f_Ar, NAMES,
    krate, G1p, G2p, G3p, G4p, G5p, G6p, G7p,
    G8p, G9p, G10p, G11p, G12p, G13p, G14p, G15p, G16p,
    G17p, G18p, G19p, G22p, G26p,
    G28p, G29p, G30p, G31p, G32p, G33p, G34p,
    k_detach_Fm, k_detach_SF6m,
    k_recomb_G35, k_recomb_G36, k_recomb_G37,
    k_G38, k_G39, k_G40, k_G41,
    k_ii, k_G50,
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s32,
    v_th, u_bohm, D_free, h_L, h_R, ion_wall_freq,
    chantry_wall_freq, sheath_V, sigma_coll, SCCM
)

# ================================================================
# V3: Kim et al. (2006) / Thorsteinsson & Gudmundsson (2009) h-factors
# Replaces the Lee & Lieberman (1995) prefactor with the quadratic
# ansatz: h^2 = h_a^2 + h_c^2
# ================================================================

# Ion-ion mutual neutralization rate [m^3/s]
# Use k_ii from the model (1.1e-13 m^3/s) as mutual neutralization rate
K_RECOMB_MUTUAL = 1.1e-13  # same as k_ii in the paper

def h_L_kim(lam_i, alpha_avg=0.0, Te_eV=5.0, p_Pa=1.0, ne=1e16):
    """Axial h-factor — Kim et al. (2006) quadratic ansatz.
    h^2 = h_a^2 + h_c^2 (Thorsteinsson two-term version)."""
    p_mT = p_Pa * 7.50062
    Ti_eV = (0.5 - T_gas/11605.0)/max(p_mT, 0.1) + T_gas/11605.0 if p_mT > 1.0 else 0.5
    gamma = Te_eV / max(Ti_eV, 0.01)
    
    alpha0 = 1.5 * alpha_avg  # center value from parabolic profile
    
    # h_a: electropositive structure with 1/(1+alpha0) correction
    uB0 = np.sqrt(e_q * Te_eV / (127*amu))  # EP Bohm velocity
    v_thi = np.sqrt(kB * Ti_eV * 11605.0 / (127*amu))
    nu_i = v_thi / max(lam_i, 1e-6)
    Da = e_q * Te_eV / (127*amu * nu_i) if nu_i > 0 else 1e10
    hp_term = (0.86 * L_re * uB0 / (pi * Da))**2
    ha = 0.86 / np.sqrt(3.0 + L_re/(2.0*lam_i) + hp_term) / (1.0 + alpha0)
    
    # h_c: flat-topped EN profile floor
    hc = 0.0
    if K_RECOMB_MUTUAL > 0 and lam_i > 0 and alpha_avg > 0.01 and ne > 1e6:
        v_i = np.sqrt(8.0 * e_q * Ti_eV / (pi * 127*amu))
        n_star = (15.0/56.0) / (K_RECOMB_MUTUAL * lam_i) * v_i
        n_neg = alpha_avg * ne
        n_pos = ne + n_neg
        sqrt_gam_sum = 2.0 * np.sqrt(gamma)
        if n_neg > 1e6:
            ratio = np.sqrt(n_star) * n_pos / max(n_neg**1.5, 1e-30)
            hc = 1.0 / max(sqrt_gam_sum * ratio, 1e-30)
            hc = min(hc, 1.0)
    
    return np.clip(np.sqrt(ha**2 + hc**2), 1e-5, 1.0)

def h_R_kim(lam_i, alpha_avg=0.0, Te_eV=5.0, p_Pa=1.0, ne=1e16):
    """Radial h-factor — Kim et al. (2006) quadratic ansatz."""
    p_mT = p_Pa * 7.50062
    Ti_eV = (0.5 - T_gas/11605.0)/max(p_mT, 0.1) + T_gas/11605.0 if p_mT > 1.0 else 0.5
    gamma = Te_eV / max(Ti_eV, 0.01)
    
    alpha0 = 1.5 * alpha_avg
    
    uB0 = np.sqrt(e_q * Te_eV / (127*amu))
    v_thi = np.sqrt(kB * Ti_eV * 11605.0 / (127*amu))
    nu_i = v_thi / max(lam_i, 1e-6)
    Da = e_q * Te_eV / (127*amu * nu_i) if nu_i > 0 else 1e10
    J1_2405 = 0.5191
    hp_term = (0.80 * R_re * uB0 / (2.405 * J1_2405 * Da))**2
    ha = 0.80 / np.sqrt(4.0 + R_re/lam_i + hp_term) / (1.0 + alpha0)
    
    # Same h_c as axial (isotropic flat-topped profile)
    hc = 0.0
    if K_RECOMB_MUTUAL > 0 and lam_i > 0 and alpha_avg > 0.01 and ne > 1e6:
        v_i = np.sqrt(8.0 * e_q * Ti_eV / (pi * 127*amu))
        n_star = (15.0/56.0) / (K_RECOMB_MUTUAL * lam_i) * v_i
        n_neg = alpha_avg * ne
        n_pos = ne + n_neg
        sqrt_gam_sum = 2.0 * np.sqrt(gamma)
        if n_neg > 1e6:
            ratio = np.sqrt(n_star) * n_pos / max(n_neg**1.5, 1e-30)
            hc = 1.0 / max(sqrt_gam_sum * ratio, 1e-30)
            hc = min(hc, 1.0)
    
    return np.clip(np.sqrt(ha**2 + hc**2), 1e-5, 1.0)

def ion_wall_freq_kim(idx, Te, n_gas, alpha_avg=0.0, p_Pa=1.0, ne=1e16):
    """Ion wall loss frequency using Kim/Thorsteinsson h-factors.
    Uses ELECTROPOSITIVE Bohm velocity — h-factors carry the EN correction."""
    lam_i = 1.0 / max(n_gas * sigma_coll(idx), 1e-10)
    Aeff = (h_L_kim(lam_i, alpha_avg, Te, p_Pa, ne) * A_ax + 
            h_R_kim(lam_i, alpha_avg, Te, p_Pa, ne) * A_rad)
    uB0 = np.sqrt(e_q * Te / (M[idx]))  # EP Bohm velocity
    return uB0 * Aeff / V_re


def ode_rhs(t, y, p_OFF, P_abs):
    """
    dy/dt for the time-dependent global model.
    V3: Kim h-factors, eta=0.70, s_dep=0.10.
    """
    # V3 overrides
    global s32
    s32 = 0.10  # V3: increased deposition probability
    # --- Unpack with safety clamps ---
    n = np.maximum(y[:N_SP], 1e6)
    Te = max(y[N_SP], 1.0)
    th = np.clip(y[N_SP+1:N_SP+1+N_SURF], 1e-10, 0.999)
    
    ne = n[EL]
    thF, thSF3, thSF4, thSF5 = th
    thSFx  = thSF3 + thSF4 + thSF5
    thTot  = thF + thSFx
    thBare = max(1.0 - thTot, 1e-10)
    
    nn = sum(n[i] for i in range(N_NEU))
    nAr = f_Ar * p_OFF / (kB * T_gas)
    ng = max(nn + nAr, 1e15)
    
    # --- Pumping ---
    n0 = p_OFF / (kB * T_gas)
    k_pump = Q_total / (n0 * V_re)
    
    # --- Rate coefficients ---
    k1=krate(G1p,Te); k2=krate(G2p,Te); k3=krate(G3p,Te)
    k4=krate(G4p,Te); k5=krate(G5p,Te)
    k6=krate(G6p,Te); k7=krate(G7p,Te)
    k8=krate(G8p,Te); k9=krate(G9p,Te); k10=krate(G10p,Te)
    k11=krate(G11p,Te); k12=krate(G12p,Te)
    k13=krate(G13p,Te); k14=krate(G14p,Te)
    k15=krate(G15p,Te); k16=krate(G16p,Te)
    k17=krate(G17p,Te); k18=krate(G18p,Te); k19=krate(G19p,Te)
    
    # --- Pressure-dependent recombination rates (Ryan & Plumb Troe formula) ---
    kG35 = k_recomb_G35(ng)
    kG36 = k_recomb_G36(ng)
    kG37 = k_recomb_G37(ng)
    
    # --- Surface rates (Chantry + branching, audit A2) ---
    sF_tot = s1*thBare + s5*thSF3 + s6*thSF4 + s7*thSF5 + s8*thF
    nuF = chantry_wall_freq(F_, sF_tot, ng)
    fS1=s1*thBare; fS5=s5*thSF3; fS6=s6*thSF4; fS7=s7*thSF5; fS8=s8*thF
    denom = max(sF_tot, 1e-30)
    RS1  = nuF * fS1/denom * n[F_]
    RS5  = nuF * fS5/denom * n[F_]
    RS6  = nuF * fS6/denom * n[F_]
    RS7  = nuF * fS7/denom * n[F_]
    RS8  = nuF * fS8/denom * n[F_]
    
    sSF3_tot = s2*thBare + s9*thF + s32*thSFx
    nuSF3 = chantry_wall_freq(SF3, sSF3_tot, ng)
    d3 = max(sSF3_tot, 1e-30)
    RS2 = nuSF3 * s2*thBare/d3 * n[SF3]
    RS9 = nuSF3 * s9*thF/d3 * n[SF3]
    RD3 = nuSF3 * s32*thSFx/d3 * n[SF3]
    
    sSF4_tot = s3*thBare + s10*thF + s32*thSFx
    nuSF4 = chantry_wall_freq(SF4, sSF4_tot, ng)
    d4 = max(sSF4_tot, 1e-30)
    RS3  = nuSF4 * s3*thBare/d4 * n[SF4]
    RS10 = nuSF4 * s10*thF/d4 * n[SF4]
    RD4  = nuSF4 * s32*thSFx/d4 * n[SF4]
    
    sSF5_tot = s4*thBare + s11*thF + s32*thSFx
    nuSF5 = chantry_wall_freq(SF5, sSF5_tot, ng)
    d5 = max(sSF5_tot, 1e-30)
    RS4  = nuSF5 * s4*thBare/d5 * n[SF5]
    RS11 = nuSF5 * s11*thF/d5 * n[SF5]
    RD5  = nuSF5 * s32*thSFx/d5 * n[SF5]
    
    # --- Ion wall losses (V3: Kim/Thorsteinsson h-factors) ---
    alpha_avg = (n[Fm] + n[SF6m]) / max(ne, 1e6)
    wSF5p = ion_wall_freq_kim(SF5p, Te, ng, alpha_avg, p_OFF, ne)
    wSF4p = ion_wall_freq_kim(SF4p, Te, ng, alpha_avg, p_OFF, ne)
    wSF3p = ion_wall_freq_kim(SF3p, Te, ng, alpha_avg, p_OFF, ne)
    wF2p  = ion_wall_freq_kim(F2p,  Te, ng, alpha_avg, p_OFF, ne)
    GwSF5p=wSF5p*n[SF5p]; GwSF4p=wSF4p*n[SF4p]
    GwSF3p=wSF3p*n[SF3p]; GwF2p=wF2p*n[F2p]
    
    n_neg = n[Fm] + n[SF6m]
    n_pos = n[SF5p]+n[SF4p]+n[SF3p]+n[F2p]
    RiiSF5p=k_ii*n[SF5p]*n_neg; RiiSF4p=k_ii*n[SF4p]*n_neg
    RiiSF3p=k_ii*n[SF3p]*n_neg; RiiF2p=k_ii*n[F2p]*n_neg
    RiiSF6m=k_ii*n[SF6m]*n_pos; RiiFm=k_ii*n[Fm]*n_pos
    
    # F returned from ion sputtering (A4)
    Fret = (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thF
    
    # ================================================================
    # dn/dt for each species
    # ================================================================
    dndt = np.zeros(N_SP + 1 + N_SURF)
    
    # SF6
    dndt[SF6] = (kG35*n[F_]*n[SF5] + k_G38*n[F2]*n[SF5] + k_G41*n[SF5]**2
                 + RS7 + RS11 + k_detach_SF6m*n[SF6m]*nn + RiiSF6m
                 + Q_SF6_mol/V_re
                 - (k1+k2+k3+k8+k9+k10+k17+k18)*ne*n[SF6] - k_pump*n[SF6])
    
    # SF5
    dndt[SF5] = (k1*ne*n[SF6] + k17*ne*n[SF6]
                 + kG36*n[F_]*n[SF4] + k_G39*n[F2]*n[SF4]
                 + RS10 + RiiSF5p
                 - (k4+k11+k12)*ne*n[SF5] - kG35*n[F_]*n[SF5]
                 - k_G38*n[F2]*n[SF5] - 2*k_G41*n[SF5]**2
                 - nuSF5*n[SF5] - k_pump*n[SF5])
    
    # SF4
    dndt[SF4] = (k2*ne*n[SF6] + k4*ne*n[SF5]
                 + kG37*n[F_]*n[SF3] + k_G40*n[F2]*n[SF3]
                 + k_G41*n[SF5]**2 + RS9 + RiiSF4p
                 - (k5+k13+k14)*ne*n[SF4] - kG36*n[F_]*n[SF4]
                 - k_G39*n[F2]*n[SF4] - nuSF4*n[SF4] - k_pump*n[SF4])
    
    # SF3
    dndt[SF3] = (k3*ne*n[SF6] + k5*ne*n[SF4] + RiiSF3p
                 - k15*ne*n[SF3] - kG37*n[F_]*n[SF3] - k_G40*n[F2]*n[SF3]
                 - nuSF3*n[SF3] - k_pump*n[SF3])
    
    # F2
    dndt[F2] = (RS8 + k_G50*n[SF6]*n[SF5p] + RiiF2p
                - (k6+k7+k16+k19)*ne*n[F2]
                - k_G38*n[SF5]*n[F2] - k_G39*n[SF4]*n[F2] - k_G40*n[SF3]*n[F2]
                - k_pump*n[F2])
    
    # F
    dndt[F_] = ((k1+2*k2+3*k3)*ne*n[SF6] + k4*ne*n[SF5] + k5*ne*n[SF4]
                + 2*(k6+k7)*ne*n[F2]
                + (k8+2*k9+3*k10)*ne*n[SF6] + k12*ne*n[SF5] + k14*ne*n[SF4]
                + k19*ne*n[F2] + k_detach_Fm*n[Fm]*nn
                + k_G38*n[F2]*n[SF5] + k_G39*n[F2]*n[SF4] + k_G40*n[F2]*n[SF3]
                + RiiFm + Fret
                - kG35*n[SF5]*n[F_] - kG36*n[SF4]*n[F_] - kG37*n[SF3]*n[F_]
                - nuF*n[F_] - k_pump*n[F_])
    
    # SF5+
    dndt[SF5p] = (k8*ne*n[SF6] + k11*ne*n[SF5]
                  - RiiSF5p - GwSF5p - k_G50*n[SF6]*n[SF5p] - k_pump*n[SF5p])
    
    # SF4+
    dndt[SF4p] = (k9*ne*n[SF6] + k12*ne*n[SF5] + k13*ne*n[SF4]
                  - RiiSF4p - GwSF4p - k_pump*n[SF4p])
    
    # SF3+
    dndt[SF3p] = (k10*ne*n[SF6] + k14*ne*n[SF4] + k15*ne*n[SF3]
                  + k_G50*n[SF6]*n[SF5p]
                  - RiiSF3p - GwSF3p - k_pump*n[SF3p])
    
    # F2+
    dndt[F2p] = (k16*ne*n[F2] - RiiF2p - GwF2p - k_pump*n[F2p])
    
    # SF6-  (no pump, no wall loss)
    dndt[SF6m] = k18*ne*n[SF6] - k_detach_SF6m*n[SF6m]*nn - RiiSF6m
    
    # F-  (no pump, no wall loss)
    dndt[Fm] = k17*ne*n[SF6] + k19*ne*n[F2] - k_detach_Fm*n[Fm]*nn - RiiFm
    
    # Electron: from charge neutrality ne = n_pos - n_neg
    # dne/dt = d(n_pos)/dt - d(n_neg)/dt
    dndt[EL] = (dndt[SF5p]+dndt[SF4p]+dndt[SF3p]+dndt[F2p]
                - dndt[SF6m] - dndt[Fm])
    
    # ================================================================
    # dTe/dt from electron energy balance
    # ================================================================
    # d(3/2 ne Te)/dt = P_abs/V - P_coll - P_wall
    # → dTe/dt = (2/3ne) × [P_abs/V/e - ne×Σ(k×n×ε) - Γ_wall×E_wall] - Te×dne/dt/ne
    
    Pcoll_eV = 0.0  # [eV s⁻¹ per electron]
    for p_, nt in [(G1p,n[SF6]),(G2p,n[SF6]),(G3p,n[SF6]),
                   (G4p,n[SF5]),(G5p,n[SF4]),
                   (G6p,n[F2]),(G7p,n[F2]),
                   (G8p,n[SF6]),(G9p,n[SF6]),(G10p,n[SF6]),
                   (G11p,n[SF5]),(G12p,n[SF5]),
                   (G13p,n[SF4]),(G14p,n[SF4]),
                   (G15p,n[SF3]),(G16p,n[F2]),
                   (G17p,n[SF6]),(G18p,n[SF6]),(G19p,n[F2]),
                   (G28p,n[SF6]),
                   (G29p,n[F2]),(G30p,n[F2]),(G31p,n[F2]),
                   (G32p,n[F2]),(G33p,n[F2]),(G34p,n[F2])]:
        Pcoll_eV += krate(p_,Te) * nt * p_[5]
    
    for p_,nt,mass in [(G22p,n[SF6],M[SF6]),(G22p,n[SF5],M[SF5]),
                       (G22p,n[SF4],M[SF4]),(G22p,n[SF3],M[SF3]),
                       (G26p,n[F2],M[F2]),  (G26p,n[F_],M[F_])]:
        Pcoll_eV += krate(p_,Te) * nt * 3*m_e/mass * Te
    
    # Ionization/attachment 3/2 Te correction (A3)
    # REMOVED per paper Section 4.1: "energy loss = threshold energy"
    # In electronegative plasmas, net_att > net_iz, making this term negative,
    # which reduces total collisional loss and inflates ne.
    # net_iz = ((k8+k9+k10)*n[SF6] + (k11+k12)*n[SF5]
    #           + (k13+k14)*n[SF4] + k15*n[SF3] + k16*n[F2])
    # net_att = (k17+k18)*n[SF6] + k19*n[F2]
    # Pcoll_eV += (net_iz - net_att) * 1.5 * Te
    
    # Wall power per Ref [36] Eq. 4:
    # ε_e,w = 2kBTe, ε_ion,w = 6.5kBTe (mean for high-density sources)
    # For SF6 heavier ions, use per-species Vs: ε_ion,w = 0.5Te + Vs
    Gion_w = GwSF5p + GwSF4p + GwSF3p + GwF2p
    Ewall = 0.0
    for idx, Gw in [(SF5p,GwSF5p),(SF4p,GwSF4p),(SF3p,GwSF3p),(F2p,GwF2p)]:
        Ewall += Gw * (0.5*Te + sheath_V(idx, Te))
    Ewall += Gion_w * 2.0 * Te  # electron energy to wall
    
    # Pumping loss of charged species energy (Ref [36] Eq. 4, last term)
    n0 = p_OFF / (kB * T_gas)
    k_pump_local = Q_total / (n0 * V_re)
    Tion_eV = ((0.5 - T_gas/11605.0) * 0.00750062 / max(p_OFF*7.50062, 0.001) 
               + T_gas/11605.0) if p_OFF > 0.0133 else 0.5  # Eq.5, p in mTorr
    # Actually simpler: use the formula from Eq 5
    p_mTorr = p_OFF * 7.50062  # Pa to mTorr
    if p_mTorr > 1.0:
        Tion_eV = (0.5 - T_gas/11605.0)/p_mTorr + T_gas/11605.0
    else:
        Tion_eV = 0.5
    E_pump = k_pump_local * (1.5*ne*Te + 1.5*(n[SF5p]+n[SF4p]+n[SF3p]+n[F2p])*Tion_eV)
    
    # Power coupling efficiency (not stated in paper but physically < 1)
    # Paper assumes P_abs = P_source but Ryan & Plumb needed factor 2 reduction
    eta_power = 0.70  # V3: increased from 0.50
    
    # Energy balance: d(3/2 ne Te)/dt = η×P_abs/(V×e) - ne×Pcoll_eV - Ewall - E_pump
    # Using product rule: ne × dTe/dt + Te × dne/dt = (2/3) × [...]
    # → dTe/dt = (2/(3 ne)) × [P_abs/(V×e) - ne×Pcoll_eV - Ewall] - Te/ne × dne/dt
    
    ne_safe = max(ne, 1e10)
    dTedt = ((2.0/(3.0*ne_safe)) * (eta_power*P_abs/(V_re*e_q) - ne*Pcoll_eV - Ewall - E_pump)
             - Te/ne_safe * dndt[EL])
    # Clamp dTe/dt to prevent runaway when ne is very small
    dTedt = np.clip(dTedt, -1e8, 1e8)
    
    dndt[N_SP] = dTedt
    
    # ================================================================
    # Surface coverages: dθ/dt
    # ================================================================
    # Convert volumetric rates to surface areal rates: × V/A
    # But θ is dimensionless: dθ/dt has units [s⁻¹]
    # Surface site density: assume ~10^19 sites/m² (typical)
    n_sites = 1e19  # [m⁻²]
    
    # Flux = volumetric rate × V / A gives [m⁻² s⁻¹]
    # dθ/dt = flux / n_sites
    VoA = V_re / A_re
    
    dthF = (RS1 - RS8 - RS9 - RS10 - RS11 
            - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thF
            + 2*GwF2p*thBare) * VoA / n_sites
    
    dthSF3 = (RS2 + RD3 - RS5
              + GwSF3p*thBare
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF3) * VoA / n_sites
    # Note: RD3 adds SF3(s) (gas SF3 deposits), RS5 removes SF3(s) (fluorination)
    
    dthSF4 = (RS3 + RS5 + RD4 - RS6
              + GwSF4p*thBare
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF4) * VoA / n_sites
    
    dthSF5 = (RS4 + RS6 + RD5 - RS7
              + GwSF5p*thBare
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF5) * VoA / n_sites
    
    dndt[N_SP+1+TH_F]   = dthF
    dndt[N_SP+1+TH_SF3] = dthSF3
    dndt[N_SP+1+TH_SF4] = dthSF4
    dndt[N_SP+1+TH_SF5] = dthSF5
    
    return dndt


def run_to_steady_state(p_OFF, P_abs, t_max=0.5, verbose=True):
    """
    Integrate ODE from pre-discharge initial conditions to steady state.
    """
    # Initial conditions: pure SF6 (no plasma)
    y0 = np.zeros(N_SP + 1 + N_SURF)
    n_SF6_0 = f_SF6 * p_OFF / (kB * T_gas)
    
    y0[SF6] = n_SF6_0
    y0[SF5] = 1e12; y0[SF4] = 1e12; y0[SF3] = 1e11
    y0[F2]  = 1e12; y0[F_]  = 1e13
    # Seed ions and electrons high enough for ignition
    y0[SF5p] = 1e14; y0[SF4p] = 1e13; y0[SF3p] = 1e13
    y0[F2p] = 1e11; y0[SF6m] = 1e13; y0[Fm] = 1e14
    y0[EL] = y0[SF5p]+y0[SF4p]+y0[SF3p]+y0[F2p]-y0[SF6m]-y0[Fm]
    y0[EL] = max(y0[EL], 1e13)
    
    y0[N_SP] = 5.0  # Te initial guess [eV]
    y0[N_SP+1:] = [0.05, 0.02, 0.02, 0.02]  # initial coverages
    
    if verbose:
        print(f"  Integrating p_OFF={p_OFF:.2f} Pa, P={P_abs:.0f} W, t_max={t_max} s...")
    
    def rhs(t, y):
        return ode_rhs(t, y, p_OFF, P_abs)
    
    sol = solve_ivp(rhs, [0, t_max], y0, method='BDF',
                    rtol=1e-8, atol=1e-6,
                    max_step=1e-3,
                    dense_output=True)
    
    if not sol.success:
        if verbose:
            print(f"  Integration failed: {sol.message}")
        return None
    
    # Extract final state
    yf = sol.y[:, -1]
    n = np.maximum(yf[:N_SP], 0)
    Te = max(yf[N_SP], 0.5)
    th = np.clip(yf[N_SP+1:N_SP+1+N_SURF], 0, 1)
    
    nn = sum(n[i] for i in range(N_NEU))
    nAr = f_Ar * p_OFF / (kB * T_gas)
    p_dis = (nn + nAr) * kB * T_gas
    dp = p_dis - p_OFF
    alpha = (n[Fm]+n[SF6m]) / max(n[EL], 1e6)
    
    if verbose:
        print(f"  → Te={Te:.2f}eV ne={n[EL]:.2e} nF={n[F_]:.2e} "
              f"dp={dp:.4f}Pa α={alpha:.1f} (t_final={sol.t[-1]:.4f}s, "
              f"steps={len(sol.t)})")
    
    return dict(n=n, Te=Te, th=th, p_OFF=p_OFF, P=P_abs, dp=dp,
                alpha=alpha, sol=sol, ok=sol.success)


if __name__ == '__main__':
    OUT = '/home/claude/sf6_model/output'
    os.makedirs(OUT, exist_ok=True)
    
    print("="*70)
    print("SF6 Global Model — ODE time integration")
    print("="*70)
    
    # Single point test
    print("\n--- Single point: p_OFF=0.921 Pa, P=2000 W ---")
    r = run_to_steady_state(0.921, 2000, t_max=1.0)
    if r:
        for i in range(N_SP):
            print(f"  {NAMES[i]:6s} = {r['n'][i]:.3e}")
        print(f"  Te = {r['Te']:.3f} eV")
        print(f"  dp = {r['dp']:.4f} Pa")
        print(f"  α  = {r['alpha']:.2f}")
    
    # Power sweep
    print("\n--- Power sweep ---")
    results_P = []
    for P in [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500]:
        r = run_to_steady_state(0.921, P, t_max=1.0)
        if r:
            results_P.append(r)
    
    if len(results_P) >= 2:
        from sf6_global_model import plot_fig5, plot_fig7
        plot_fig5(results_P, f'{OUT}/fig5_ode.png')
        plot_fig7(results_P, f'{OUT}/fig7_ode.png')
    
    # Pressure sweep
    print("\n--- Pressure sweep ---")
    results_p = []
    for p in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        r = run_to_steady_state(p, 2000, t_max=1.0)
        if r:
            results_p.append(r)
    
    if len(results_p) >= 2:
        from sf6_global_model import plot_fig6, plot_fig8
        plot_fig6(results_p, f'{OUT}/fig6_ode.png')
        plot_fig8(results_p, f'{OUT}/fig8_ode.png')
    
    print(f"\nDone. Output in {OUT}/")
