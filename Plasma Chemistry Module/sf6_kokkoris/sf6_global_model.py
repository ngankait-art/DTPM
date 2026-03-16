"""
SF6 Global Plasma Model
========================
Reproduces: Kokkoris et al., J. Phys. D: Appl. Phys. 42 (2009) 055209

Implementation choices documented in SF6_implementation_audit.md.
All parameters traced to literature sources in SF6_FINAL_resolved_parameters.md.
"""

import numpy as np
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

# ================================================================
# PHYSICAL CONSTANTS
# ================================================================
kB   = 1.3807e-23    # Boltzmann constant [J/K]
eV   = 1.6022e-19    # eV to Joules
e_q  = 1.6022e-19    # elementary charge [C]
m_e  = 9.1095e-31    # electron mass [kg]
amu  = 1.6606e-27    # atomic mass unit [kg]
pi   = np.pi

# ================================================================
# SPECIES INDICES
# ================================================================
# Neutrals
SF6, SF5, SF4, SF3, F2, F_ = 0, 1, 2, 3, 4, 5
N_NEU = 6
# Positive ions
SF5p, SF4p, SF3p, F2p = 6, 7, 8, 9
# Negative ions
SF6m, Fm = 10, 11
# Electrons
EL = 12
N_SP = 13  # total gas-phase species

# Surface coverages: θ_F, θ_SF3, θ_SF4, θ_SF5
TH_F, TH_SF3, TH_SF4, TH_SF5 = 0, 1, 2, 3
N_SURF = 4

# Unknown vector: [log10(n_0) .. log10(n_12), Te, θ_F, θ_SF3, θ_SF4, θ_SF5]
IDX_TE = N_SP        # index 13
IDX_TH = N_SP + 1    # indices 14..17
N_UNK  = N_SP + 1 + N_SURF  # 18

NAMES = ['SF6','SF5','SF4','SF3','F2','F',
         'SF5+','SF4+','SF3+','F2+','SF6-','F-','e']

# ================================================================
# MASSES [kg]
# ================================================================
M = np.array([146,127,108,89,38,19, 127,108,89,38, 146,19, 5.486e-4]) * amu

# ================================================================
# REACTOR GEOMETRY (Paper Section 2)
# ================================================================
R_re  = 0.19                         # radius [m]
L_re  = 0.21 + 0.17                  # total length [m] (ICP + main)
V_re  = pi * R_re**2 * L_re          # volume [m³]
A_ax  = 2 * pi * R_re**2             # top + bottom area [m²]
A_rad = 2 * pi * R_re * L_re         # side area [m²]
A_re  = A_ax + A_rad                  # total wall area [m²]
# Diffusion length: 1/Λ² = (π/L)² + (2.405/R)²
LAMBDA = 1.0 / np.sqrt((pi/L_re)**2 + (2.405/R_re)**2)  # ≈ 0.0661 m

# ================================================================
# OPERATING CONDITIONS
# ================================================================
T_gas = 315.0  # [K]
SCCM  = 1.01325e5 / (kB * 273.15) * 1e-6 / 60.0  # 1 sccm → molecules/s
Q_SF6_mol = 100.0 * SCCM   # SF6 feed [molecules/s]
Q_Ar_mol  = 10.0  * SCCM   # Ar feed [molecules/s]
Q_total   = Q_SF6_mol + Q_Ar_mol
f_SF6     = Q_SF6_mol / Q_total   # SF6 fraction in feed
f_Ar      = Q_Ar_mol  / Q_total   # Ar fraction in feed

# ================================================================
# LENNARD-JONES σ [m] — BSL for SF6, F2; interpolated for SFx
# ================================================================
sigma_LJ = {SF6: 5.51e-10, SF5: 5.20e-10, SF4: 4.90e-10,
            SF3: 4.50e-10, F2: 3.36e-10, F_: 2.90e-10}

def sigma_coll(i):
    """Hard-sphere collision cross section of species i with SF6 [m²]."""
    d = (sigma_LJ.get(i, 5.0e-10) + sigma_LJ[SF6]) / 2.0
    return pi * d**2

# ================================================================
# RATE COEFFICIENT PARAMETRIC FORM (Table 1, Druyvesteyn EEDF)
# k = exp(A + B ln(Te) + C/Te + D/Te² + E/Te³)  [m³/s], Te in eV
# Stored as (A, B, C, D, E, E_threshold_eV)
# ================================================================
# Neutral dissociations
G1p = (-29.35,-0.2379,-14.11,-15.25,-1.204,  9.6)
G2p = (-31.61,-0.2592,-10.00,-31.24,-0.7126, 12.1)
G3p = (-40.26, 3.135,  5.895,-64.68, 0.2607, 16.0)
G4p = (-29.36,-0.2379,-14.11,-15.25,-1.204,  9.6)   # same as G1
G5p = (-29.36,-0.2379,-14.11,-15.25,-1.204,  9.6)   # same as G1
G6p = (-31.44,-0.6986,-5.170,-1.389,-0.0650, 3.16)
G7p = (-33.44,-0.2761,-3.564,-3.946,-0.0393, 4.34)
# Ionizations
G8p  = (-33.66, 1.212, -4.594,-56.66,-0.3226, 15.5)
G9p  = (-37.14, 1.515, -4.829,-80.42,-0.7924, 18.5)
G10p = (-36.82, 1.740, -0.1047,-98.18,0.1060, 20.0)
G11p = (-34.92, 1.487, -2.377,-29.71,-0.1449, 11.2)
G12p = (-36.27, 1.892, -1.387,-50.87,-0.0758, 14.5)
G13p = (-32.95, 0.8763,-10.19,-31.21,-3.989,  13.0)
G14p = (-32.75, 0.8222,-10.82,-40.59,-4.274,  14.5)
G15p = (-35.55, 1.750, -2.086,-28.70,-0.1357, 11.0)
G16p = (-35.60, 1.467, -6.140,-57.14,-0.4860, 15.69)
# Attachments
G17p = (-33.43,-1.173,-0.5614, 0.1798,-0.0145, 0.1)
G18p = (-33.46,-1.500, 0.0002,-0.0023, 0.0,    0.0)
G19p = (-33.31,-1.487,-0.2795, 0.0109,-0.0004, 0.0)
# Momentum transfer (elastic)
G22p = (-29.15, 0.2126,-1.455, 0.2456,-0.0141, 0.0)   # SF6
G26p = (-29.04,-0.0987,-0.4897,-0.0319,0.0055, 0.0)   # F2, F
# Excitations
G28p = (-24.81,-2.174,-13.47, 12.45,-4.400, 0.09)
G29p = (-33.85,-1.549,-0.6197,0.0306,-0.0012,0.1108)
G30p = (-33.57,-1.552,-0.6555,0.0224,-0.0012,0.2188)
G31p = (-33.90,-1.554,-0.6757,0.0041,-0.0008,0.3237)
G32p = (-35.01,-1.546,-0.6177,-0.0295,0.0001,0.4205)
G33p = (-33.89, 0.7953,-6.732,-29.19,-0.5969,11.57)
G34p = (-39.03, 1.740,-3.465,-40.04,-0.2956, 13.08)

def krate(p, Te):
    """Electron-impact rate coefficient [m³/s]. Te in eV."""
    Te = np.clip(Te, 0.5, 50.0)
    A,B,C,D,E,_ = p
    return np.exp(np.clip(A + B*np.log(Te) + C/Te + D/Te**2 + E/Te**3, -120, 0))

# Constant rate coefficients [m³/s]
k_detach_Fm   = np.exp(-44.39)   # G20: F⁻+N→F+N+e
k_detach_SF6m = np.exp(-44.98)   # G21: SF6⁻+N→SF6+N+e
# G35-G37: Troe fall-off from Ryan & Plumb (1990) Table III
# k = (k0[M]/(1+k0[M]/kinf)) × Fc^{(1+[log10(k0[M]/kinf)]^2)^-1}
# Parameters for SF6 as third body:
#   G35 (SF5+F→SF6): k0=3.4e-23 cm6/s, kinf=1e-11 cm3/s, Fc=0.43
#   G36 (SF4+F→SF5): k0=3.7e-28 cm6/s, kinf=5e-12 cm3/s, Fc=0.46
#   G37 (SF3+F→SF4): k0=2.8e-26 cm6/s, kinf=2e-11 cm3/s, Fc=0.47
# G41 (SF5+SF5→SF6+SF4): bimolecular, 2.5e-11 cm3/s = 2.5e-17 m3/s
RP_G35 = (3.4e-23, 1.0e-11, 0.43)  # (k0_cgs, kinf_cgs, Fc)
RP_G36 = (3.7e-28, 5.0e-12, 0.46)
RP_G37 = (2.8e-26, 2.0e-11, 0.47)

def troe_falloff_SI(k0_cgs, kinf_cgs, Fc, M_SI):
    """Troe fall-off rate [m3/s]. M_SI = total gas density [m-3]."""
    M_cgs = M_SI * 1e-6  # m-3 → cm-3
    ratio = k0_cgs * M_cgs / kinf_cgs
    if ratio <= 0:
        return 0.0
    log_r = np.log10(ratio)
    Fc_exp = 1.0 / (1.0 + log_r**2)
    k_cgs = (k0_cgs * M_cgs / (1.0 + ratio)) * Fc**Fc_exp
    return k_cgs * 1e-6  # cm3/s → m3/s

def k_recomb_G35(M_SI):
    return troe_falloff_SI(*RP_G35, M_SI)
def k_recomb_G36(M_SI):
    return troe_falloff_SI(*RP_G36, M_SI)
def k_recomb_G37(M_SI):
    return troe_falloff_SI(*RP_G37, M_SI)

k_G38 = np.exp(-46.41)       # F2+SF5→SF6+F (Anderson et al.)
k_G39 = np.exp(-46.41)       # F2+SF4→SF5+F
k_G40 = np.exp(-46.41)       # F2+SF3→SF4+F
k_G41 = np.exp(-41.50)        # SF5+SF5→SF6+SF4 (Kokkoris Table 1, from literature)
k_ii  = np.exp(-29.93)           # G42-G49: ion-ion recombination
k_G50 = np.exp(-39.65)           # SF6+SF5+→SF6+SF3++F2

# Surface reaction probabilities (Table 2)
s1  = 0.150   # F+bare→F(s)
s2  = 0.080   # SF3+bare→SF3(s)
s3  = 0.080   # SF4+bare→SF4(s)
s4  = 0.080   # SF5+bare→SF5(s)
s5  = 0.500   # F+SF3(s)→SF4(s)
s6  = 0.200   # F+SF4(s)→SF5(s)
s7  = 0.025   # F+SF5(s)→SF6(gas)
s8  = 0.500   # F+F(s)→F2(gas)
s9  = 1.000   # SF3+F(s)→SF4(gas)
s10 = 1.000   # SF4+F(s)→SF5(gas)
s11 = 1.000   # SF5+F(s)→SF6(gas)
s32 = 0.030   # SFx+SFy(s)→SFx(s)+P(s) deposition

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def v_th(idx):
    """Thermal velocity [m/s]."""
    return np.sqrt(8*kB*T_gas / (pi*M[idx]))

def u_bohm(idx, Te):
    """Bohm velocity [m/s]. Te in eV."""
    return np.sqrt(e_q*Te / M[idx])

def D_free(idx, n_gas):
    """Free diffusion coefficient [m²/s]."""
    lam = 1.0 / max(n_gas * sigma_coll(idx), 1e-10)
    return pi/8.0 * lam * v_th(idx)

def h_L(lam_i, alpha_avg=0.0, Te_eV=5.0, p_Pa=1.0):
    """Axial h-factor — full Lee & Lieberman 1995 Eq. A9.
    Includes electronegative prefactor and high-pressure diffusion term."""
    # Ion temperature (Eq. 5 of Kokkoris Ref [36])
    p_mT = p_Pa * 7.50062
    Ti_eV = (0.5 - T_gas/11605.0)/max(p_mT, 0.1) + T_gas/11605.0 if p_mT > 1.0 else 0.5
    gamma = Te_eV / max(Ti_eV, 0.01)
    # Electronegative prefactor
    prefactor = (1.0 + 3.0*alpha_avg/gamma) / (1.0 + alpha_avg)
    # High-pressure ambipolar diffusion term
    # Da ≈ e*Te/(M*nu_i), nu_i = v_thi/lambda_i
    # uB²/Da² = uB²*M²*nu_i²/(e²*Te²) = M*nu_i²/(e*Te) [since uB²=eTe/M]
    # = nu_i² / (e²Te²/M²) ... simpler to just compute directly
    uB = np.sqrt(e_q*Te_eV / (127*amu))  # use SF5+ mass as representative
    v_thi = np.sqrt(kB * Ti_eV*11605.0 / (127*amu))
    nu_i = v_thi / max(lam_i, 1e-6)
    Da = e_q * Te_eV / (127*amu * nu_i) if nu_i > 0 else 1e10
    hp_term = (0.86 * L_re * uB / (pi * Da))**2
    return prefactor * 0.86 / np.sqrt(3.0 + L_re/(2.0*lam_i) + hp_term)

def h_R(lam_i, alpha_avg=0.0, Te_eV=5.0, p_Pa=1.0):
    """Radial h-factor — full Lee & Lieberman 1995 Eq. A10."""
    p_mT = p_Pa * 7.50062
    Ti_eV = (0.5 - T_gas/11605.0)/max(p_mT, 0.1) + T_gas/11605.0 if p_mT > 1.0 else 0.5
    gamma = Te_eV / max(Ti_eV, 0.01)
    prefactor = (1.0 + 3.0*alpha_avg/gamma) / (1.0 + alpha_avg)
    uB = np.sqrt(e_q*Te_eV / (127*amu))
    v_thi = np.sqrt(kB * Ti_eV*11605.0 / (127*amu))
    nu_i = v_thi / max(lam_i, 1e-6)
    Da = e_q * Te_eV / (127*amu * nu_i) if nu_i > 0 else 1e10
    J1_2405 = 0.5191
    hp_term = (0.80 * R_re * uB / (2.405 * J1_2405 * Da))**2
    return prefactor * 0.80 / np.sqrt(4.0 + R_re/lam_i + hp_term)

def ion_wall_freq(idx, Te, n_gas, alpha_avg=0.0, p_Pa=1.0):
    """Ion wall loss frequency [s⁻¹]. Full Lee & Lieberman."""
    lam_i = 1.0 / max(n_gas * sigma_coll(idx), 1e-10)
    Aeff = h_L(lam_i, alpha_avg, Te, p_Pa)*A_ax + h_R(lam_i, alpha_avg, Te, p_Pa)*A_rad
    return u_bohm(idx, Te) * Aeff / V_re

def chantry_wall_freq(sp_idx, s_eff, n_gas):
    """
    Neutral wall loss frequency [s⁻¹] using Chantry formula.
    
    ν = [1/ν_surf + 1/ν_diff]⁻¹
    ν_surf = s_eff/(2-s_eff) × v_th/4 × A/V
    ν_diff = D/Λ²
    
    Audit choice A2: s_eff is the TOTAL effective sticking for this species.
    """
    if s_eff <= 0:
        return 0.0
    vt = v_th(sp_idx)
    nu_surf = s_eff / (2.0 - s_eff) * vt / 4.0 * A_re / V_re
    D = D_free(sp_idx, n_gas)
    nu_diff = D / LAMBDA**2
    return 1.0 / (1.0/max(nu_surf,1e-30) + 1.0/max(nu_diff,1e-30))

def sheath_V(idx, Te):
    """Sheath voltage [eV] for ion species idx. Te in eV."""
    return 0.5 * Te * np.log(M[idx] / (2*pi*m_e))

# ================================================================
# EQUATION SYSTEM  (18 equations, 18 unknowns)
# ================================================================
def residuals(x, p_OFF, P_abs):
    """
    x = [log10(n_0)..log10(n_12), Te, θ_F, θ_SF3, θ_SF4, θ_SF5]
    Returns residual vector F(x) = 0.
    """
    # --- Unpack ---
    n = 10.0 ** np.clip(x[:N_SP], 5, 25)
    Te = np.clip(x[IDX_TE], 1.0, 20.0)
    th = np.clip(x[IDX_TH:IDX_TH+N_SURF], 1e-8, 0.999)
    
    ne = n[EL]
    thF, thSF3, thSF4, thSF5 = th
    thSFx  = thSF3 + thSF4 + thSF5
    thTot  = thF + thSFx
    thBare = max(1.0 - thTot, 1e-10)
    
    # Total neutral density (including Ar)
    nn = sum(n[i] for i in range(N_NEU))
    n_Ar = f_Ar * p_OFF / (kB * T_gas)
    ng = nn + n_Ar   # total gas density for diffusion/mfp
    ng = max(ng, 1e15)
    
    # --- Pumping ---
    n0_total = p_OFF / (kB * T_gas)
    k_pump = Q_total / (n0_total * V_re)
    
    # --- Electron-impact rate coefficients ---
    k1=krate(G1p,Te);  k2=krate(G2p,Te);  k3=krate(G3p,Te)
    k4=krate(G4p,Te);  k5=krate(G5p,Te)
    k6=krate(G6p,Te);  k7=krate(G7p,Te)
    k8=krate(G8p,Te);  k9=krate(G9p,Te);  k10=krate(G10p,Te)
    k11=krate(G11p,Te); k12=krate(G12p,Te)
    k13=krate(G13p,Te); k14=krate(G14p,Te)
    k15=krate(G15p,Te); k16=krate(G16p,Te)
    k17=krate(G17p,Te); k18=krate(G18p,Te); k19=krate(G19p,Te)
    
    # Pressure-dependent recombination rates (Troe fall-off)
    kG35 = k_recomb_G35(ng)
    kG36 = k_recomb_G36(ng)
    kG37 = k_recomb_G37(ng)
    
    # --- Total effective sticking and Chantry wall rates (Audit A2) ---
    # F: competes in S1,S5,S6,S7,S8
    sF_tot = s1*thBare + s5*thSF3 + s6*thSF4 + s7*thSF5 + s8*thF
    nuF_wall = chantry_wall_freq(F_, sF_tot, ng)
    # Branch fractions for F
    bF = {}
    if sF_tot > 0:
        bF['S1'] = s1*thBare/sF_tot;  bF['S5'] = s5*thSF3/sF_tot
        bF['S6'] = s6*thSF4/sF_tot;   bF['S7'] = s7*thSF5/sF_tot
        bF['S8'] = s8*thF/sF_tot
    else:
        bF = {k:0 for k in ['S1','S5','S6','S7','S8']}
    
    # SF3: competes in S2 (bare), S9 (F(s)), deposition on SFx(s)
    sSF3_tot = s2*thBare + s9*thF + s32*thSFx
    nuSF3_wall = chantry_wall_freq(SF3, sSF3_tot, ng)
    bSF3 = {}
    if sSF3_tot > 0:
        bSF3['S2']=s2*thBare/sSF3_tot; bSF3['S9']=s9*thF/sSF3_tot
        bSF3['dep']=s32*thSFx/sSF3_tot
    else:
        bSF3 = {k:0 for k in ['S2','S9','dep']}
    
    # SF4: competes in S3, S10, deposition
    sSF4_tot = s3*thBare + s10*thF + s32*thSFx
    nuSF4_wall = chantry_wall_freq(SF4, sSF4_tot, ng)
    bSF4 = {}
    if sSF4_tot > 0:
        bSF4['S3']=s3*thBare/sSF4_tot; bSF4['S10']=s10*thF/sSF4_tot
        bSF4['dep']=s32*thSFx/sSF4_tot
    else:
        bSF4 = {k:0 for k in ['S3','S10','dep']}
    
    # SF5: competes in S4, S11, deposition
    sSF5_tot = s4*thBare + s11*thF + s32*thSFx
    nuSF5_wall = chantry_wall_freq(SF5, sSF5_tot, ng)
    bSF5 = {}
    if sSF5_tot > 0:
        bSF5['S4']=s4*thBare/sSF5_tot; bSF5['S11']=s11*thF/sSF5_tot
        bSF5['dep']=s32*thSFx/sSF5_tot
    else:
        bSF5 = {k:0 for k in ['S4','S11','dep']}
    
    # Individual surface reaction RATES [m⁻³ s⁻¹]
    RS1  = nuF_wall * bF['S1'] * n[F_]
    RS5  = nuF_wall * bF['S5'] * n[F_]
    RS6  = nuF_wall * bF['S6'] * n[F_]
    RS7  = nuF_wall * bF['S7'] * n[F_]   # produces SF6(gas)
    RS8  = nuF_wall * bF['S8'] * n[F_]   # produces F2(gas)
    
    RS2  = nuSF3_wall * bSF3['S2'] * n[SF3]
    RS9  = nuSF3_wall * bSF3['S9'] * n[SF3]   # produces SF4(gas)
    RD3  = nuSF3_wall * bSF3['dep'] * n[SF3]  # deposition consuming SF3(gas)
    
    RS3  = nuSF4_wall * bSF4['S3'] * n[SF4]
    RS10 = nuSF4_wall * bSF4['S10'] * n[SF4]  # produces SF5(gas)
    RD4  = nuSF4_wall * bSF4['dep'] * n[SF4]
    
    RS4  = nuSF5_wall * bSF5['S4'] * n[SF5]
    RS11 = nuSF5_wall * bSF5['S11'] * n[SF5]  # produces SF6(gas)
    RD5  = nuSF5_wall * bSF5['dep'] * n[SF5]
    
    # --- Ion wall loss frequencies [s⁻¹] ---
    wSF5p = ion_wall_freq(SF5p, Te, ng)
    wSF4p = ion_wall_freq(SF4p, Te, ng)
    wSF3p = ion_wall_freq(SF3p, Te, ng)
    wF2p  = ion_wall_freq(F2p,  Te, ng)
    
    # Ion wall loss RATES [m⁻³ s⁻¹]
    GwSF5p = wSF5p * n[SF5p]
    GwSF4p = wSF4p * n[SF4p]
    GwSF3p = wSF3p * n[SF3p]
    GwF2p  = wF2p  * n[F2p]
    
    # Ion-ion recombination rates
    n_neg = n[Fm] + n[SF6m]
    n_pos = n[SF5p] + n[SF4p] + n[SF3p] + n[F2p]
    RiiSF5p = k_ii * n[SF5p] * n_neg
    RiiSF4p = k_ii * n[SF4p] * n_neg
    RiiSF3p = k_ii * n[SF3p] * n_neg
    RiiF2p  = k_ii * n[F2p]  * n_neg
    RiiSF6m = k_ii * n[SF6m] * n_pos
    RiiFm   = k_ii * n[Fm]   * n_pos
    
    # Ion-surface sputtering: F returned to gas (Audit A4)
    # SFx+ hitting F(s) → SFx(s) + F(gas): rate = Γ_ion_wall × θ_F
    Fret_from_ions = (GwSF5p + GwSF4p + GwSF3p) * thF  # F returned
    # F2+ hitting F(s) → 2F(s) + F(gas): also returns F
    Fret_from_F2p = GwF2p * thF
    
    # ================================================================
    # GAS-PHASE SPECIES BALANCES: (prod - loss)/n_i = 0
    # ================================================================
    R = np.zeros(N_UNK)
    
    # --- SF6 ---
    prod = (kG35*n[F_]*n[SF5] + k_G38*n[F2]*n[SF5] + k_G41*n[SF5]**2
            + RS7 + RS11                          # surface: F+SF5(s)→SF6, SF5+F(s)→SF6
            + k_detach_SF6m*n[SF6m]*nn            # SF6⁻ detachment
            + RiiSF6m                             # ion-ion recomb of SF6⁻
            + Q_SF6_mol/V_re)                     # feed
    loss = (k1+k2+k3+k8+k9+k10+k17+k18)*ne*n[SF6] + k_pump*n[SF6]
    R[SF6] = (prod - loss) / max(n[SF6], 1e10)
    
    # --- SF5 ---
    prod = (k1*ne*n[SF6] + k17*ne*n[SF6]         # dissociation + attachment product
            + kG36*n[F_]*n[SF4] + k_G39*n[F2]*n[SF4]
            + RS10                                 # SF4+F(s)→SF5
            + RiiSF5p)                             # ion-ion recomb
    loss = ((k4+k11+k12)*ne + kG35*n[F_] + k_G38*n[F2] + 2*k_G41*n[SF5]
            + nuSF5_wall + k_pump) * n[SF5]
    R[SF5] = (prod - loss) / max(n[SF5], 1e10)
    
    # --- SF4 ---
    prod = (k2*ne*n[SF6] + k4*ne*n[SF5]
            + kG37*n[F_]*n[SF3] + k_G40*n[F2]*n[SF3]
            + k_G41*n[SF5]**2
            + RS9                                  # SF3+F(s)→SF4
            + RiiSF4p)
    loss = ((k5+k13+k14)*ne + kG36*n[F_] + k_G39*n[F2]
            + nuSF4_wall + k_pump) * n[SF4]
    R[SF4] = (prod - loss) / max(n[SF4], 1e10)
    
    # --- SF3 ---
    prod = (k3*ne*n[SF6] + k5*ne*n[SF4] + RiiSF3p)
    loss = (k15*ne + kG37*n[F_] + k_G40*n[F2]
            + nuSF3_wall + k_pump) * n[SF3]
    R[SF3] = (prod - loss) / max(n[SF3], 1e10)
    
    # --- F2 ---
    prod = (RS8                                    # F+F(s)→F2
            + k_G50*n[SF6]*n[SF5p]                 # ion-molecule
            + RiiF2p)
    loss = ((k6+k7+k16+k19)*ne
            + k_G38*n[SF5] + k_G39*n[SF4] + k_G40*n[SF3]
            + k_pump) * n[F2]
    R[F2] = (prod - loss) / max(n[F2], 1e10)
    
    # --- F ---
    prod = ((k1 + 2*k2 + 3*k3)*ne*n[SF6]         # neutral dissociations
            + k4*ne*n[SF5] + k5*ne*n[SF4]
            + 2*(k6+k7)*ne*n[F2]
            + (k8 + 2*k9 + 3*k10)*ne*n[SF6]       # ionization F products
            + k12*ne*n[SF5] + k14*ne*n[SF4]
            + k19*ne*n[F2]                          # F2+e→F+F⁻
            + k_detach_Fm*n[Fm]*nn                  # F⁻ detachment
            + k_G38*n[F2]*n[SF5] + k_G39*n[F2]*n[SF4] + k_G40*n[F2]*n[SF3]
            + RiiFm                                 # ion-ion recomb of F⁻
            + Fret_from_ions + Fret_from_F2p)       # Audit A4: F from ion sputtering
    loss = (kG35*n[SF5] + kG36*n[SF4] + kG37*n[SF3]
            + nuF_wall + k_pump) * n[F_]
    R[F_] = (prod - loss) / max(n[F_], 1e10)
    
    # --- SF5+ ---
    prod = k8*ne*n[SF6] + k11*ne*n[SF5]
    loss = (RiiSF5p + GwSF5p + k_G50*n[SF6]*n[SF5p] + k_pump*n[SF5p])
    R[SF5p] = (prod - loss) / max(n[SF5p], 1e8)
    
    # --- SF4+ ---
    prod = k9*ne*n[SF6] + k12*ne*n[SF5] + k13*ne*n[SF4]
    loss = RiiSF4p + GwSF4p + k_pump*n[SF4p]
    R[SF4p] = (prod - loss) / max(n[SF4p], 1e8)
    
    # --- SF3+ ---
    prod = k10*ne*n[SF6] + k14*ne*n[SF4] + k15*ne*n[SF3] + k_G50*n[SF6]*n[SF5p]
    loss = RiiSF3p + GwSF3p + k_pump*n[SF3p]
    R[SF3p] = (prod - loss) / max(n[SF3p], 1e8)
    
    # --- F2+ ---
    prod = k16*ne*n[F2]
    loss = RiiF2p + GwF2p + k_pump*n[F2p]
    R[F2p] = (prod - loss) / max(n[F2p], 1e6)
    
    # --- SF6⁻ ---  (NOT pumped, NOT lost to walls)
    prod = k18*ne*n[SF6]
    loss = k_detach_SF6m*n[SF6m]*nn + RiiSF6m
    R[SF6m] = (prod - loss) / max(n[SF6m], 1e6)
    
    # --- F⁻ ---  (NOT pumped, NOT lost to walls)
    prod = k17*ne*n[SF6] + k19*ne*n[F2]
    loss = k_detach_Fm*n[Fm]*nn + RiiFm
    R[Fm] = (prod - loss) / max(n[Fm], 1e6)
    
    # --- Charge neutrality (replaces electron balance) ---
    R[EL] = (n[SF5p]+n[SF4p]+n[SF3p]+n[F2p] - ne - n[Fm] - n[SF6m]) / max(ne, 1e10)
    
    # ================================================================
    # POWER BALANCE (Audit A3: include 3/2 Te correction)
    # ================================================================
    # Collisional losses: Σ k × n_target × ε_threshold
    Pcoll = 0.0
    # Inelastic
    for p_,nt in [(G1p,n[SF6]),(G2p,n[SF6]),(G3p,n[SF6]),
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
        Pcoll += krate(p_,Te) * nt * p_[5]   # k × n × Eth [eV/s per electron]
    
    # Elastic: 3(m_e/M)Te per collision
    for p_,nt,mass in [(G22p,n[SF6],M[SF6]),(G22p,n[SF5],M[SF5]),
                       (G22p,n[SF4],M[SF4]),(G22p,n[SF3],M[SF3]),
                       (G26p,n[F2],M[F2]),  (G26p,n[F_],M[F_])]:
        Pcoll += krate(p_,Te) * nt * 3.0*m_e/mass * Te
    
    # Ionization/attachment energy: (3/2)Te per net electron gained/lost (Audit A3)
    net_ionization = ((k8+k9+k10)*n[SF6] + (k11+k12)*n[SF5] 
                      + (k13+k14)*n[SF4] + k15*n[SF3] + k16*n[F2])
    net_attachment = (k17+k18)*n[SF6] + k19*n[F2]
    Pcoll += (net_ionization - net_attachment) * 1.5 * Te
    
    Pcoll_W = ne * Pcoll * e_q   # [W/m³]
    
    # Wall losses
    Gion_total = GwSF5p + GwSF4p + GwSF3p + GwF2p   # total ion wall flux [m⁻³ s⁻¹]
    
    # Per-species sheath voltage (Audit A6)
    Ewall = 0.0
    for idx, Gw in [(SF5p,GwSF5p),(SF4p,GwSF4p),(SF3p,GwSF3p),(F2p,GwF2p)]:
        Vs = sheath_V(idx, Te)
        Ewall += Gw * (0.5*Te + Vs)   # ion energy to wall [eV m⁻³ s⁻¹]
    Ewall += Gion_total * 2.0 * Te     # electron energy to wall (ambipolarity)
    Pwall_W = Ewall * e_q              # [W/m³]
    
    R[IDX_TE] = (P_abs/V_re - Pcoll_W - Pwall_W) / max(P_abs/V_re, 1.0)
    
    # ================================================================
    # SURFACE COVERAGE BALANCES
    # ================================================================
    # All rates below are [m⁻³ s⁻¹]; we normalize by a typical rate.
    
    # θ_F: production = adsorption(S1) + F2⁺→2F(s)
    #       loss = S8(F recomb) + S9,S10,S11(SFx recomb) + ion sputtering
    dthF = (RS1                                              # S1: F+bare→F(s)
            + 2*GwF2p*thBare                                 # F2+ on bare → 2F(s)
            - RS8                                            # S8: F+F(s)→F2
            - nuSF3_wall*bSF3.get('S9',0)*n[SF3]            # S9: SF3+F(s)→SF4
            - nuSF4_wall*bSF4.get('S10',0)*n[SF4]           # S10: SF4+F(s)→SF5
            - nuSF5_wall*bSF5.get('S11',0)*n[SF5]           # S11: SF5+F(s)→SF6
            - (GwSF5p+GwSF4p+GwSF3p)*thF                    # ions sputter F(s)
            - GwF2p*thF)                                     # F2+ on F(s)
    
    # θ_SF3: production = S2(adsorption) + SF3⁺ adsorption + deposition gain
    #        loss = S5(fluorination) + deposition loss + ion sputtering
    # Deposition (Audit A5): gain from SF3(gas) depositing on SFy(s): already in RD3
    # But RD3 removes SF3 from gas AND converts SFy(s)→P(s) while creating SF3(s)
    # Loss from SFx depositing on SF3(s): rate ∝ θ_SF3 × total SFx gas flux
    dep_loss_3 = s32 * thSF3 * sum(chantry_wall_freq(i, s32*thSFx, ng)*n[i] 
                                    for i in [SF3,SF4,SF5]) if thSFx > 0 else 0
    dthSF3 = (RS2                                # S2: SF3+bare→SF3(s)
              + GwSF3p*thBare                    # SF3⁺ adsorbs on bare
              + RD3                              # SF3 deposits, creating SF3(s)
              - RS5                              # S5: F+SF3(s)→SF4(s)
              - dep_loss_3                       # other SFx deposit on SF3(s)
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF3)  # ion sputtering
    
    # θ_SF4
    dep_loss_4 = s32 * thSF4 * sum(chantry_wall_freq(i, s32*thSFx, ng)*n[i]
                                    for i in [SF3,SF4,SF5]) if thSFx > 0 else 0
    dthSF4 = (RS3 + RS5                         # adsorption + fluorination of SF3(s)
              + GwSF4p*thBare
              + RD4
              - RS6                              # S6: F+SF4(s)→SF5(s)
              - dep_loss_4
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF4)
    
    # θ_SF5
    dep_loss_5 = s32 * thSF5 * sum(chantry_wall_freq(i, s32*thSFx, ng)*n[i]
                                    for i in [SF3,SF4,SF5]) if thSFx > 0 else 0
    dthSF5 = (RS4 + RS6                         # adsorption + fluorination of SF4(s)
              + GwSF5p*thBare
              + RD5
              - RS7                              # S7: F+SF5(s)→SF6(gas)
              - dep_loss_5
              - (GwSF5p+GwSF4p+GwSF3p+GwF2p)*thSF5)
    
    # Normalize surface equations
    surf_norm = max(RS1, RS8, RS2, GwSF5p*0.1, 1e12)
    R[IDX_TH+TH_F]   = dthF   / surf_norm
    R[IDX_TH+TH_SF3] = dthSF3 / surf_norm
    R[IDX_TH+TH_SF4] = dthSF4 / surf_norm
    R[IDX_TH+TH_SF5] = dthSF5 / surf_norm
    
    return R

# ================================================================
# INITIAL GUESS
# ================================================================
def guess(p_OFF, P_abs):
    x = np.zeros(N_UNK)
    n0 = f_SF6 * p_OFF / (kB * T_gas)
    d = min(0.3, P_abs/5000)
    dens = np.array([
        n0*(1-d),           # SF6
        n0*d*0.25,          # SF5
        n0*d*0.10,          # SF4
        n0*d*0.02,          # SF3
        n0*d*0.08,          # F2
        n0*d*0.30,          # F
        n0*1e-4*max(P_abs/1000,0.1)*0.60,  # SF5+
        n0*1e-4*max(P_abs/1000,0.1)*0.10,  # SF4+
        n0*1e-4*max(P_abs/1000,0.1)*0.15,  # SF3+
        n0*1e-4*max(P_abs/1000,0.1)*0.005, # F2+
        n0*1e-4*max(P_abs/1000,0.1)*0.03,  # SF6-
        n0*1e-4*max(P_abs/1000,0.1)*0.40,  # F-
        0  # electron — will be set by charge neutrality
    ])
    dens[EL] = max(sum(dens[SF5p:F2p+1]) - dens[SF6m] - dens[Fm], 1e14)
    dens = np.maximum(dens, 1e8)
    x[:N_SP] = np.log10(dens)
    x[IDX_TE] = 5.0
    x[IDX_TH:] = [0.10, 0.03, 0.03, 0.03]
    return x

# ================================================================
# SOLVER
# ================================================================
def solve(p_OFF, P_abs, x0=None, verbose=False):
    if x0 is None:
        x0 = guess(p_OFF, P_abs)
    sol = root(residuals, x0, args=(p_OFF, P_abs), method='hybr',
               options={'maxfev':80000, 'xtol':1e-10})
    if not sol.success:
        sol = root(residuals, x0, args=(p_OFF, P_abs), method='lm',
                   options={'maxiter':80000, 'ftol':1e-10})
    n = 10.0**np.clip(sol.x[:N_SP], 5, 25)
    Te = sol.x[IDX_TE]
    th = sol.x[IDX_TH:]
    nn = sum(n[i] for i in range(N_NEU))
    nAr = f_Ar * p_OFF / (kB * T_gas)
    p_dis = (nn + nAr) * kB * T_gas
    dp = p_dis - p_OFF
    alpha = (n[Fm]+n[SF6m]) / max(n[EL],1e10)
    res_norm = np.linalg.norm(sol.fun)
    if verbose:
        tag = "OK" if sol.success else f"WARN(res={res_norm:.1e})"
        print(f"  p={p_OFF:.2f}Pa P={P_abs:.0f}W → Te={Te:.2f}eV "
              f"ne={n[EL]:.2e} nF={n[F_]:.2e} dp={dp:.3f}Pa α={alpha:.1f} [{tag}]")
    return dict(n=n, Te=Te, th=th, p_OFF=p_OFF, P=P_abs, dp=dp,
                alpha=alpha, ok=sol.success, res=res_norm, x=sol.x)

def sweep_power(p_OFF=0.921, powers=None, verbose=True):
    if powers is None:
        powers = np.concatenate([np.arange(100,500,100), np.arange(500,3600,250)])
    results, xp = [], None
    for P in powers:
        r = solve(p_OFF, P, x0=xp, verbose=verbose)
        results.append(r)
        if r['ok']: xp = r['x'].copy()
    return results

def sweep_pressure(P_abs=2000, pressures=None, verbose=True):
    if pressures is None:
        pressures = np.arange(0.4, 4.5, 0.3)
    results, xp = [], None
    for p in pressures:
        r = solve(p, P_abs, x0=xp, verbose=verbose)
        results.append(r)
        if r['ok']: xp = r['x'].copy()
    return results

# ================================================================
# PLOTTING
# ================================================================
def plot_fig5(res, path):
    ok = [r for r in res if r['ok']]
    if len(ok)<2: print("  Fig5: insufficient converged points"); return
    P = [r['P'] for r in ok]; dp = [r['dp'] for r in ok]; nF = [r['n'][F_] for r in ok]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
    a1.plot(P,dp,'bo-',ms=4); a1.set(xlabel='Power (W)',ylabel='Pressure rise (Pa)',
            title=f'Fig 5a: Pressure rise vs Power, pOFF={ok[0]["p_OFF"]:.3f} Pa')
    a1.grid(True,alpha=.3)
    a2.plot(P,nF,'ro-',ms=4); a2.set(xlabel='Power (W)',ylabel='F density (m⁻³)',
            title=f'Fig 5b: F density vs Power'); a2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    a2.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"  Saved {path}")

def plot_fig7(res, path):
    ok = [r for r in res if r['ok']]
    if len(ok)<2: return
    P = [r['P'] for r in ok]
    fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(17,5))
    for i,nm,c in zip(range(N_NEU),['SF₆','SF₅','SF₄','SF₃','F₂','F'],
                      ['b','g','r','orange','purple','brown']):
        a1.semilogy(P,[r['n'][i] for r in ok],c=c,marker='.',ms=3,label=nm)
    a1.set(xlabel='Power (W)',ylabel='Density (m⁻³)',title='Fig 7a: Neutrals',ylim=(1e16,1e21))
    a1.legend(fontsize=8); a1.grid(True,alpha=.3)
    for i,nm,c,ls in [(SF5p,'SF₅⁺','b','-'),(SF4p,'SF₄⁺','g','-'),(SF3p,'SF₃⁺','r','-'),
                       (F2p,'F₂⁺','orange','-'),(SF6m,'SF₆⁻','cyan','--'),(Fm,'F⁻','m','--'),
                       (EL,'e⁻','k','-.')]:
        a2.semilogy(P,[r['n'][i] for r in ok],c=c,ls=ls,marker='.',ms=3,label=nm)
    a2.set(xlabel='Power (W)',ylabel='Density (m⁻³)',title='Fig 7b: Charged'); a2.legend(fontsize=7); a2.grid(True,alpha=.3)
    ax3r=a3.twinx()
    a3.plot(P,[r['alpha'] for r in ok],'b-o',ms=3,label='n⁻/nₑ')
    ax3r.plot(P,[r['Te'] for r in ok],'r-s',ms=3,label='Tₑ')
    a3.set(xlabel='Power (W)',ylabel='n⁻/nₑ',title='Fig 7c: α & Tₑ'); a3.tick_params(axis='y',labelcolor='b')
    ax3r.set_ylabel('Tₑ (eV)',color='r'); ax3r.tick_params(axis='y',labelcolor='r'); a3.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"  Saved {path}")

def plot_fig6(res, path):
    ok = [r for r in res if r['ok']]
    if len(ok)<2: return
    P = [r['p_OFF'] for r in ok]; dp = [r['dp'] for r in ok]; nF = [r['n'][F_] for r in ok]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
    a1.plot(P,dp,'bo-',ms=4); a1.set(xlabel='pOFF (Pa)',ylabel='Pressure rise (Pa)',
            title=f'Fig 6a: Pressure rise vs pOFF, P={ok[0]["P"]:.0f} W'); a1.grid(True,alpha=.3)
    a2.plot(P,nF,'ro-',ms=4); a2.set(xlabel='pOFF (Pa)',ylabel='F density (m⁻³)',
            title=f'Fig 6b: F density vs pOFF'); a2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    a2.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"  Saved {path}")

def plot_fig8(res, path):
    ok = [r for r in res if r['ok']]
    if len(ok)<2: return
    P = [r['p_OFF'] for r in ok]
    fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(17,5))
    for i,nm,c in zip(range(N_NEU),['SF₆','SF₅','SF₄','SF₃','F₂','F'],
                      ['b','g','r','orange','purple','brown']):
        a1.semilogy(P,[r['n'][i] for r in ok],c=c,marker='.',ms=3,label=nm)
    a1.set(xlabel='pOFF (Pa)',ylabel='Density (m⁻³)',title='Fig 8a: Neutrals',ylim=(1e16,1e21))
    a1.legend(fontsize=8); a1.grid(True,alpha=.3)
    for i,nm,c,ls in [(SF5p,'SF₅⁺','b','-'),(SF4p,'SF₄⁺','g','-'),(SF3p,'SF₃⁺','r','-'),
                       (F2p,'F₂⁺','orange','-'),(SF6m,'SF₆⁻','cyan','--'),(Fm,'F⁻','m','--'),
                       (EL,'e⁻','k','-.')]:
        a2.semilogy(P,[r['n'][i] for r in ok],c=c,ls=ls,marker='.',ms=3,label=nm)
    a2.set(xlabel='pOFF (Pa)',ylabel='Density (m⁻³)',title='Fig 8b: Charged'); a2.legend(fontsize=7); a2.grid(True,alpha=.3)
    ax3r=a3.twinx()
    a3.plot(P,[r['alpha'] for r in ok],'b-o',ms=3); ax3r.plot(P,[r['Te'] for r in ok],'r-s',ms=3)
    a3.set(xlabel='pOFF (Pa)',ylabel='n⁻/nₑ',title='Fig 8c: α & Tₑ'); a3.tick_params(axis='y',labelcolor='b')
    ax3r.set_ylabel('Tₑ (eV)',color='r'); ax3r.tick_params(axis='y',labelcolor='r'); a3.grid(True,alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150); plt.close(); print(f"  Saved {path}")

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    OUT = '/mnt/user-data/outputs'
    os.makedirs(OUT, exist_ok=True)
    
    print("="*70)
    print("SF6 Global Plasma Model — Kokkoris et al. (2009)")
    print("="*70)
    
    # Single point test
    print("\n--- Single point: p_OFF=0.921 Pa, P=2000 W ---")
    r = solve(0.921, 2000, verbose=True)
    for i in range(N_SP):
        print(f"  {NAMES[i]:6s} = {r['n'][i]:.3e} m⁻³")
    print(f"  Te = {r['Te']:.3f} eV")
    print(f"  θ = F:{r['th'][0]:.3f} SF3:{r['th'][1]:.3f} SF4:{r['th'][2]:.3f} SF5:{r['th'][3]:.3f}")
    print(f"  Pressure rise = {r['dp']:.4f} Pa")
    print(f"  Residual norm = {r['res']:.2e}")
    
    # Power sweep
    print("\n--- Power sweep (pOFF = 0.921 Pa) ---")
    rP = sweep_power(0.921, verbose=True)
    plot_fig5(rP, f'{OUT}/fig5.png')
    plot_fig7(rP, f'{OUT}/fig7.png')
    
    # Pressure sweep
    print("\n--- Pressure sweep (P = 2000 W) ---")
    rp = sweep_pressure(2000, verbose=True)
    plot_fig6(rp, f'{OUT}/fig6.png')
    plot_fig8(rp, f'{OUT}/fig8.png')
    
    print(f"\nDone. Outputs in {OUT}/")
