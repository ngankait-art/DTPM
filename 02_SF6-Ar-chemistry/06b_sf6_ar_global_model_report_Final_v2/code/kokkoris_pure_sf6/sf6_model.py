"""
SF6 ICP Global Plasma Model
============================
Zero-dimensional (volume-averaged) model for an SF6 inductively coupled
plasma discharge. Implements the reaction set and transport physics from:

    Kokkoris et al., J. Phys. D: Appl. Phys. 42 (2009) 055209

with the Kim et al. (2006) h-factor formulation for electronegative
ion wall loss, as recommended by Thorsteinsson & Gudmundsson (2010)
and evaluated by Monahan & Turner (2008).

Rate coefficients use the Druyvesteyn EEDF parameterisation.
Pressure-dependent recombination uses Troe fall-off (Ryan & Plumb 1990).
Neutral wall loss uses the Chantry (1987) formulation.
"""

import numpy as np
from scipy.integrate import solve_ivp

# =====================================================================
# Physical Constants
# =====================================================================
kB  = 1.3807e-23     # Boltzmann constant [J/K]
eV  = 1.6022e-19     # eV to Joules
e_q = 1.6022e-19     # elementary charge [C]
m_e = 9.1095e-31     # electron mass [kg]
amu = 1.6606e-27     # atomic mass unit [kg]
pi  = np.pi

# =====================================================================
# Species
# =====================================================================
# Indices into the state vector
SF6, SF5, SF4, SF3, F2, F_ = 0, 1, 2, 3, 4, 5     # Neutrals
SF5p, SF4p, SF3p, F2p      = 6, 7, 8, 9            # Positive ions
SF6m, Fm                    = 10, 11                 # Negative ions
EL                          = 12                     # Electrons
N_SPECIES = 13
N_NEUTRALS = 6

# Surface coverages
TH_F, TH_SF3, TH_SF4, TH_SF5 = 0, 1, 2, 3
N_SURFACE = 4
N_STATE = N_SPECIES + 1 + N_SURFACE  # 18 ODEs

SPECIES_NAMES = [
    'SF6', 'SF5', 'SF4', 'SF3', 'F2', 'F',
    'SF5+', 'SF4+', 'SF3+', 'F2+', 'SF6-', 'F-', 'e'
]

# Masses [kg]
MASS = np.array([146, 127, 108, 89, 38, 19,
                 127, 108, 89, 38, 146, 19, 5.486e-4]) * amu

# =====================================================================
# Reactor Geometry
# =====================================================================
RADIUS = 0.19                                    # [m]
LENGTH = 0.38                                    # [m] (ICP source + diffusion chamber)
VOLUME = pi * RADIUS**2 * LENGTH                 # [m^3]
AREA_AXIAL  = 2 * pi * RADIUS**2                # top + bottom [m^2]
AREA_RADIAL = 2 * pi * RADIUS * LENGTH          # cylindrical wall [m^2]
AREA_TOTAL  = AREA_AXIAL + AREA_RADIAL           # total wall area [m^2]
DIFF_LENGTH = 1.0 / np.sqrt((pi/LENGTH)**2 + (2.405/RADIUS)**2)  # [m]

# =====================================================================
# Operating Conditions
# =====================================================================
T_GAS = 315.0                                     # gas temperature [K]
SCCM_CONV = 1.01325e5 / (kB * 273.15) * 1e-6 / 60.0  # 1 sccm -> molecules/s
Q_SF6 = 100.0 * SCCM_CONV                        # SF6 feed rate [molecules/s]
Q_AR  = 10.0  * SCCM_CONV                        # Ar actinometer [molecules/s]
Q_TOTAL = Q_SF6 + Q_AR
F_SF6 = Q_SF6 / Q_TOTAL                          # SF6 mole fraction in feed
F_AR  = Q_AR  / Q_TOTAL

# Model parameters
ETA = 0.70           # power coupling efficiency
S_DEP = 0.10         # SFx deposition probability
K_RECOMB_MUTUAL = 1.1e-13  # mutual neutralisation rate [m^3/s] for h_c term

# =====================================================================
# Lennard-Jones Diameters [m] — for neutral-neutral cross sections
# =====================================================================
SIGMA_LJ = {SF6: 5.51e-10, SF5: 5.20e-10, SF4: 4.90e-10,
             SF3: 4.50e-10, F2: 3.36e-10, F_: 2.90e-10}

def cross_section(species_idx):
    """Hard-sphere collision cross section with SF6 background [m^2]."""
    d = (SIGMA_LJ.get(species_idx, 5.0e-10) + SIGMA_LJ[SF6]) / 2.0
    return pi * d**2

# =====================================================================
# Electron-Impact Rate Coefficients (Druyvesteyn EEDF)
# =====================================================================
# k(Te) = exp(A + B*ln(Te) + C/Te + D/Te^2 + E/Te^3)  [m^3/s]
# Format: (A, B, C, D, E, threshold_eV)

# Dissociations (G1-G7)
_G1  = (-29.35, -0.2379, -14.11,  -15.25,  -1.204,   9.6)
_G2  = (-31.61, -0.2592, -10.00,  -31.24,  -0.7126, 12.1)
_G3  = (-40.26,  3.135,    5.895, -64.68,   0.2607, 16.0)
_G4  = (-29.36, -0.2379, -14.11,  -15.25,  -1.204,   9.6)
_G5  = (-29.36, -0.2379, -14.11,  -15.25,  -1.204,   9.6)
_G6  = (-31.44, -0.6986,  -5.170,  -1.389, -0.0650,  3.16)
_G7  = (-33.44, -0.2761,  -3.564,  -3.946, -0.0393,  4.34)

# Ionisations (G8-G16)
_G8  = (-33.66,  1.212,  -4.594, -56.66, -0.3226, 15.5)
_G9  = (-37.14,  1.515,  -4.829, -80.42, -0.7924, 18.5)
_G10 = (-36.82,  1.740,  -0.1047,-98.18,  0.1060, 20.0)
_G11 = (-34.92,  1.487,  -2.377, -29.71, -0.1449, 11.2)
_G12 = (-36.27,  1.892,  -1.387, -50.87, -0.0758, 14.5)
_G13 = (-32.95,  0.8763,-10.19,  -31.21, -3.989,  13.0)
_G14 = (-32.75,  0.8222,-10.82,  -40.59, -4.274,  14.5)
_G15 = (-35.55,  1.750,  -2.086, -28.70, -0.1357, 11.0)
_G16 = (-35.60,  1.467,  -6.140, -57.14, -0.4860, 15.69)

# Attachments (G17-G19)
_G17 = (-33.43, -1.173, -0.5614,  0.1798, -0.0145, 0.1)
_G18 = (-33.46, -1.500,  0.0002, -0.0023,  0.0,    0.0)
_G19 = (-33.31, -1.487, -0.2795,  0.0109, -0.0004, 0.0)

# Elastic momentum transfer (G22, G26)
_G22 = (-29.15,  0.2126, -1.455,  0.2456, -0.0141, 0.0)
_G26 = (-29.04, -0.0987, -0.4897,-0.0319,  0.0055, 0.0)

# Excitations (G28-G34)
_G28 = (-24.81, -2.174, -13.47,  12.45, -4.400,  0.09)
_G29 = (-33.85, -1.549,  -0.6197, 0.0306,-0.0012, 0.1108)
_G30 = (-33.57, -1.552,  -0.6555, 0.0224,-0.0012, 0.2188)
_G31 = (-33.90, -1.554,  -0.6757, 0.0041,-0.0008, 0.3237)
_G32 = (-35.01, -1.546,  -0.6177,-0.0295, 0.0001, 0.4205)
_G33 = (-33.89,  0.7953, -6.732, -29.19, -0.5969, 11.57)
_G34 = (-39.03,  1.740,  -3.465, -40.04, -0.2956, 13.08)

def rate_coeff(params, Te):
    """Electron-impact rate coefficient [m^3/s]. Te in eV."""
    Te = np.clip(Te, 0.5, 50.0)
    A, B, C, D, E, _ = params
    return np.exp(np.clip(A + B*np.log(Te) + C/Te + D/Te**2 + E/Te**3, -120, 0))

# =====================================================================
# Heavy-Particle Rate Coefficients
# =====================================================================
# Detachment
K_DETACH_FM   = np.exp(-44.39)  # G20: F- + N -> F + N + e
K_DETACH_SF6M = np.exp(-44.98)  # G21: SF6- + N -> SF6 + N + e

# Troe fall-off parameters (Ryan & Plumb 1990, Table III)
_TROE_G35 = (3.4e-23, 1.0e-11, 0.43)  # F + SF5 -> SF6
_TROE_G36 = (3.7e-28, 5.0e-12, 0.46)  # F + SF4 -> SF5
_TROE_G37 = (2.8e-26, 2.0e-11, 0.47)  # F + SF3 -> SF4

def troe_rate(k0_cgs, kinf_cgs, Fc, n_total_SI):
    """Troe fall-off rate [m^3/s]. n_total_SI in m^-3."""
    M_cgs = n_total_SI * 1e-6
    ratio = k0_cgs * M_cgs / kinf_cgs
    if ratio <= 0:
        return 0.0
    Fc_exp = 1.0 / (1.0 + np.log10(ratio)**2)
    return (k0_cgs * M_cgs / (1.0 + ratio)) * Fc**Fc_exp * 1e-6

# Neutral-neutral
K_G38 = np.exp(-46.41)  # F2 + SF5 -> SF6 + F
K_G39 = np.exp(-46.41)  # F2 + SF4 -> SF5 + F
K_G40 = np.exp(-46.41)  # F2 + SF3 -> SF4 + F
K_G41 = np.exp(-41.50)  # SF5 + SF5 -> SF6 + SF4

# Ion-ion mutual neutralisation
K_II = np.exp(-29.93)   # G42-G49

# Ion-molecule
K_G50 = np.exp(-39.65)  # SF6 + SF5+ -> SF6 + SF3+ + F2

# Surface reaction probabilities
S1  = 0.150   # F + bare -> F(s)
S2  = 0.080   # SF3 + bare -> SF3(s)
S3  = 0.080   # SF4 + bare -> SF4(s)
S4  = 0.080   # SF5 + bare -> SF5(s)
S5  = 0.500   # F + SF3(s) -> SF4(s)
S6  = 0.200   # F + SF4(s) -> SF5(s)
S7  = 0.025   # F + SF5(s) -> SF6(gas)
S8  = 0.500   # F + F(s) -> F2(gas)
S9  = 1.000   # SF3 + F(s) -> SF4(gas)
S10 = 1.000   # SF4 + F(s) -> SF5(gas)
S11 = 1.000   # SF5 + F(s) -> SF6(gas)

# =====================================================================
# Transport Functions
# =====================================================================
def thermal_velocity(species_idx):
    """Mean thermal velocity [m/s]."""
    return np.sqrt(8 * kB * T_GAS / (pi * MASS[species_idx]))

def bohm_velocity(species_idx, Te):
    """Electropositive Bohm velocity [m/s]. Te in eV."""
    return np.sqrt(e_q * Te / MASS[species_idx])

def diffusion_coeff(species_idx, n_gas):
    """Free diffusion coefficient [m^2/s]."""
    lam = 1.0 / max(n_gas * cross_section(species_idx), 1e-10)
    return pi / 8.0 * lam * thermal_velocity(species_idx)

def sheath_voltage(species_idx, Te):
    """Sheath voltage [eV]."""
    return Te / 2.0 * np.log(MASS[species_idx] / (2 * pi * m_e))

def ion_temperature(p_Pa):
    """Ion temperature [eV] from Lee & Lieberman (1995)."""
    p_mT = p_Pa * 7.50062
    if p_mT > 1.0:
        return (0.5 - T_GAS / 11605.0) / p_mT + T_GAS / 11605.0
    return 0.5

def chantry_wall_loss(species_idx, s_eff, n_gas):
    """Neutral wall loss frequency [s^-1] using Chantry (1987)."""
    if s_eff <= 0:
        return 0.0
    vth = thermal_velocity(species_idx)
    nu_surf = s_eff / (2.0 - s_eff) * vth / 4.0 * AREA_TOTAL / VOLUME
    D = diffusion_coeff(species_idx, n_gas)
    nu_diff = D / DIFF_LENGTH**2
    return nu_surf * nu_diff / (nu_surf + nu_diff)

# =====================================================================
# Kim / Thorsteinsson h-Factor (2006/2010)
# =====================================================================
def _h_factor(geometry, lam_i, alpha_avg, Te_eV, p_Pa, ne):
    """Edge-to-centre density ratio using quadratic two-term ansatz.
    
    h^2 = h_a^2 + h_c^2
    
    h_a: parabolic profile component (reduces to Godyak when alpha->0)
    h_c: flat-topped EN profile floor (prevents zero wall loss at high alpha)
    """
    Ti_eV = ion_temperature(p_Pa)
    gamma = Te_eV / max(Ti_eV, 0.01)
    alpha0 = 1.5 * alpha_avg  # centre value from parabolic profile

    uB0 = np.sqrt(e_q * Te_eV / (127 * amu))
    v_thi = np.sqrt(kB * Ti_eV * 11605.0 / (127 * amu))
    nu_i = v_thi / max(lam_i, 1e-6)
    Da = e_q * Te_eV / (127 * amu * nu_i) if nu_i > 0 else 1e10

    if geometry == 'axial':
        hp = (0.86 * LENGTH * uB0 / (pi * Da))**2
        ha = 0.86 / np.sqrt(3.0 + LENGTH / (2 * lam_i) + hp) / (1.0 + alpha0)
    else:
        hp = (0.80 * RADIUS * uB0 / (2.405 * 0.5191 * Da))**2
        ha = 0.80 / np.sqrt(4.0 + RADIUS / lam_i + hp) / (1.0 + alpha0)

    hc = 0.0
    if K_RECOMB_MUTUAL > 0 and lam_i > 0 and alpha_avg > 0.01 and ne > 1e6:
        v_i = np.sqrt(8.0 * e_q * Ti_eV / (pi * 127 * amu))
        n_star = (15.0 / 56.0) / (K_RECOMB_MUTUAL * lam_i) * v_i
        n_neg = alpha_avg * ne
        n_pos = ne + n_neg
        if n_neg > 1e6:
            sqrt_gam_sum = 2.0 * np.sqrt(gamma)
            ratio = np.sqrt(n_star) * n_pos / max(n_neg**1.5, 1e-30)
            hc = 1.0 / max(sqrt_gam_sum * ratio, 1e-30)
            hc = min(hc, 1.0)

    return np.clip(np.sqrt(ha**2 + hc**2), 1e-5, 1.0)

def ion_wall_loss_freq(species_idx, Te, n_gas, alpha_avg, p_Pa, ne):
    """Ion wall loss frequency [s^-1] using Kim h-factors."""
    lam_i = 1.0 / max(n_gas * cross_section(species_idx), 1e-10)
    h_L = _h_factor('axial', lam_i, alpha_avg, Te, p_Pa, ne)
    h_R = _h_factor('radial', lam_i, alpha_avg, Te, p_Pa, ne)
    Aeff = h_L * AREA_AXIAL + h_R * AREA_RADIAL
    uB0 = np.sqrt(e_q * Te / MASS[species_idx])
    return uB0 * Aeff / VOLUME

# =====================================================================
# ODE Right-Hand Side
# =====================================================================
def ode_system(t, y, p_OFF, P_abs):
    """Time derivative of the 18-component state vector."""
    n = np.maximum(y[:N_SPECIES], 1e6)
    Te = max(y[N_SPECIES], 1.0)
    th = np.clip(y[N_SPECIES + 1:N_SPECIES + 1 + N_SURFACE], 1e-10, 0.999)

    ne = n[EL]
    thF, thSF3, thSF4, thSF5 = th
    thSFx = thSF3 + thSF4 + thSF5
    thBare = max(1.0 - thF - thSFx, 1e-10)

    nn = sum(n[i] for i in range(N_NEUTRALS))
    nAr = F_AR * p_OFF / (kB * T_GAS)
    ng = max(nn + nAr, 1e15)
    n0 = p_OFF / (kB * T_GAS)
    k_pump = Q_TOTAL / (n0 * VOLUME)

    # --- Electron-impact rate coefficients ---
    k1  = rate_coeff(_G1, Te);  k2  = rate_coeff(_G2, Te)
    k3  = rate_coeff(_G3, Te);  k4  = rate_coeff(_G4, Te)
    k5  = rate_coeff(_G5, Te);  k6  = rate_coeff(_G6, Te)
    k7  = rate_coeff(_G7, Te);  k8  = rate_coeff(_G8, Te)
    k9  = rate_coeff(_G9, Te);  k10 = rate_coeff(_G10, Te)
    k11 = rate_coeff(_G11, Te); k12 = rate_coeff(_G12, Te)
    k13 = rate_coeff(_G13, Te); k14 = rate_coeff(_G14, Te)
    k15 = rate_coeff(_G15, Te); k16 = rate_coeff(_G16, Te)
    k17 = rate_coeff(_G17, Te); k18 = rate_coeff(_G18, Te)
    k19 = rate_coeff(_G19, Te)

    # --- Pressure-dependent recombinations ---
    kG35 = troe_rate(*_TROE_G35, ng)
    kG36 = troe_rate(*_TROE_G36, ng)
    kG37 = troe_rate(*_TROE_G37, ng)

    # --- Neutral surface rates (Chantry) ---
    sF_tot = S1*thBare + S5*thSF3 + S6*thSF4 + S7*thSF5 + S8*thF
    nuF = chantry_wall_loss(F_, sF_tot, ng)
    dm = max(sF_tot, 1e-30)
    RS1 = nuF * S1*thBare/dm * n[F_]
    RS5 = nuF * S5*thSF3/dm * n[F_]
    RS6 = nuF * S6*thSF4/dm * n[F_]
    RS7 = nuF * S7*thSF5/dm * n[F_]
    RS8 = nuF * S8*thF/dm * n[F_]

    sSF3_tot = S2*thBare + S9*thF + S_DEP*thSFx
    nuSF3 = chantry_wall_loss(SF3, sSF3_tot, ng)
    d3 = max(sSF3_tot, 1e-30)
    RS2 = nuSF3 * S2*thBare/d3 * n[SF3]
    RS9 = nuSF3 * S9*thF/d3 * n[SF3]
    RD3 = nuSF3 * S_DEP*thSFx/d3 * n[SF3]

    sSF4_tot = S3*thBare + S10*thF + S_DEP*thSFx
    nuSF4 = chantry_wall_loss(SF4, sSF4_tot, ng)
    d4 = max(sSF4_tot, 1e-30)
    RS3  = nuSF4 * S3*thBare/d4 * n[SF4]
    RS10 = nuSF4 * S10*thF/d4 * n[SF4]
    RD4  = nuSF4 * S_DEP*thSFx/d4 * n[SF4]

    sSF5_tot = S4*thBare + S11*thF + S_DEP*thSFx
    nuSF5 = chantry_wall_loss(SF5, sSF5_tot, ng)
    d5 = max(sSF5_tot, 1e-30)
    RS4  = nuSF5 * S4*thBare/d5 * n[SF5]
    RS11 = nuSF5 * S11*thF/d5 * n[SF5]
    RD5  = nuSF5 * S_DEP*thSFx/d5 * n[SF5]

    # --- Ion wall losses ---
    alpha = (n[Fm] + n[SF6m]) / max(ne, 1e6)
    wSF5p = ion_wall_loss_freq(SF5p, Te, ng, alpha, p_OFF, ne)
    wSF4p = ion_wall_loss_freq(SF4p, Te, ng, alpha, p_OFF, ne)
    wSF3p = ion_wall_loss_freq(SF3p, Te, ng, alpha, p_OFF, ne)
    wF2p  = ion_wall_loss_freq(F2p,  Te, ng, alpha, p_OFF, ne)
    GwSF5p = wSF5p * n[SF5p]; GwSF4p = wSF4p * n[SF4p]
    GwSF3p = wSF3p * n[SF3p]; GwF2p  = wF2p  * n[F2p]
    Gw_total = GwSF5p + GwSF4p + GwSF3p + GwF2p

    n_neg = n[Fm] + n[SF6m]
    n_pos = n[SF5p] + n[SF4p] + n[SF3p] + n[F2p]
    RiiSF5p = K_II * n[SF5p] * n_neg; RiiSF4p = K_II * n[SF4p] * n_neg
    RiiSF3p = K_II * n[SF3p] * n_neg; RiiF2p  = K_II * n[F2p]  * n_neg
    RiiSF6m = K_II * n[SF6m] * n_pos; RiiFm   = K_II * n[Fm]   * n_pos

    Fret = Gw_total * thF  # F returned from ion sputtering

    # --- Species balance equations ---
    dydt = np.zeros(N_STATE)

    dydt[SF6] = (kG35*n[F_]*n[SF5] + K_G38*n[F2]*n[SF5] + K_G41*n[SF5]**2
                 + RS7 + RS11 + K_DETACH_SF6M*n[SF6m]*nn + RiiSF6m
                 + Q_SF6/VOLUME
                 - (k1+k2+k3+k8+k9+k10+k17+k18)*ne*n[SF6] - k_pump*n[SF6])

    dydt[SF5] = (k1*ne*n[SF6] + k17*ne*n[SF6]
                 + kG36*n[F_]*n[SF4] + K_G39*n[F2]*n[SF4]
                 + RS10 + RiiSF5p
                 - (k4+k11+k12)*ne*n[SF5] - kG35*n[F_]*n[SF5]
                 - K_G38*n[F2]*n[SF5] - 2*K_G41*n[SF5]**2
                 - nuSF5*n[SF5] - k_pump*n[SF5])

    dydt[SF4] = (k2*ne*n[SF6] + k4*ne*n[SF5]
                 + kG37*n[F_]*n[SF3] + K_G40*n[F2]*n[SF3]
                 + K_G41*n[SF5]**2 + RS9 + RiiSF4p
                 - (k5+k13+k14)*ne*n[SF4] - kG36*n[F_]*n[SF4]
                 - K_G39*n[F2]*n[SF4] - nuSF4*n[SF4] - k_pump*n[SF4])

    dydt[SF3] = (k3*ne*n[SF6] + k5*ne*n[SF4] + RiiSF3p
                 - k15*ne*n[SF3] - kG37*n[F_]*n[SF3] - K_G40*n[F2]*n[SF3]
                 - nuSF3*n[SF3] - k_pump*n[SF3])

    dydt[F2] = (RS8 + K_G50*n[SF6]*n[SF5p] + RiiF2p
                - (k6+k7+k16+k19)*ne*n[F2]
                - K_G38*n[SF5]*n[F2] - K_G39*n[SF4]*n[F2] - K_G40*n[SF3]*n[F2]
                - k_pump*n[F2])

    dydt[F_] = ((k1+2*k2+3*k3)*ne*n[SF6] + k4*ne*n[SF5] + k5*ne*n[SF4]
                + 2*(k6+k7)*ne*n[F2]
                + (k8+2*k9+3*k10)*ne*n[SF6] + k12*ne*n[SF5] + k14*ne*n[SF4]
                + k19*ne*n[F2] + K_DETACH_FM*n[Fm]*nn
                + K_G38*n[F2]*n[SF5] + K_G39*n[F2]*n[SF4] + K_G40*n[F2]*n[SF3]
                + RiiFm + Fret
                - kG35*n[SF5]*n[F_] - kG36*n[SF4]*n[F_] - kG37*n[SF3]*n[F_]
                - nuF*n[F_] - k_pump*n[F_])

    dydt[SF5p] = (k8*ne*n[SF6] + k11*ne*n[SF5]
                  - RiiSF5p - GwSF5p - K_G50*n[SF6]*n[SF5p] - k_pump*n[SF5p])

    dydt[SF4p] = (k9*ne*n[SF6] + k12*ne*n[SF5] + k13*ne*n[SF4]
                  - RiiSF4p - GwSF4p - k_pump*n[SF4p])

    dydt[SF3p] = (k10*ne*n[SF6] + k14*ne*n[SF4] + k15*ne*n[SF3]
                  + K_G50*n[SF6]*n[SF5p]
                  - RiiSF3p - GwSF3p - k_pump*n[SF3p])

    dydt[F2p] = k16*ne*n[F2] - RiiF2p - GwF2p - k_pump*n[F2p]

    dydt[SF6m] = k18*ne*n[SF6] - K_DETACH_SF6M*n[SF6m]*nn - RiiSF6m
    dydt[Fm]   = k17*ne*n[SF6] + k19*ne*n[F2] - K_DETACH_FM*n[Fm]*nn - RiiFm

    dydt[EL] = (dydt[SF5p] + dydt[SF4p] + dydt[SF3p] + dydt[F2p]
                - dydt[SF6m] - dydt[Fm])

    # --- Electron energy balance ---
    Pcoll = 0.0
    for params, nt in [(_G1,n[SF6]),(_G2,n[SF6]),(_G3,n[SF6]),
                        (_G4,n[SF5]),(_G5,n[SF4]),(_G6,n[F2]),(_G7,n[F2]),
                        (_G8,n[SF6]),(_G9,n[SF6]),(_G10,n[SF6]),
                        (_G11,n[SF5]),(_G12,n[SF5]),
                        (_G13,n[SF4]),(_G14,n[SF4]),
                        (_G15,n[SF3]),(_G16,n[F2]),
                        (_G17,n[SF6]),(_G18,n[SF6]),(_G19,n[F2]),
                        (_G28,n[SF6]),
                        (_G29,n[F2]),(_G30,n[F2]),(_G31,n[F2]),
                        (_G32,n[F2]),(_G33,n[F2]),(_G34,n[F2])]:
        Pcoll += rate_coeff(params, Te) * nt * params[5]

    for params, nt, mass in [(_G22,n[SF6],MASS[SF6]),(_G22,n[SF5],MASS[SF5]),
                              (_G22,n[SF4],MASS[SF4]),(_G22,n[SF3],MASS[SF3]),
                              (_G26,n[F2],MASS[F2]),  (_G26,n[F_],MASS[F_])]:
        Pcoll += rate_coeff(params, Te) * nt * 3*m_e/mass * Te

    Ewall = sum(Gw * (0.5*Te + sheath_voltage(idx, Te))
                for idx, Gw in [(SF5p,GwSF5p),(SF4p,GwSF4p),
                                 (SF3p,GwSF3p),(F2p,GwF2p)])
    Ewall += Gw_total * 2.0 * Te

    Ti_eV = ion_temperature(p_OFF)
    E_pump = k_pump * (1.5*ne*Te + 1.5*n_pos*Ti_eV)

    ne_safe = max(ne, 1e10)
    dTedt = ((2.0 / (3.0 * ne_safe)) *
             (ETA * P_abs / (VOLUME * e_q) - ne * Pcoll - Ewall - E_pump)
             - Te / ne_safe * dydt[EL])
    dydt[N_SPECIES] = np.clip(dTedt, -1e8, 1e8)

    # --- Surface coverages ---
    n_sites = 1e19
    VoA = VOLUME / AREA_TOTAL

    dydt[N_SPECIES+1+TH_F] = (RS1 - RS8 - RS9 - RS10 - RS11
                                - Gw_total*thF + 2*GwF2p*thBare) * VoA / n_sites
    dydt[N_SPECIES+1+TH_SF3] = (RS2 + RD3 - RS5 + GwSF3p*thBare
                                  - Gw_total*thSF3) * VoA / n_sites
    dydt[N_SPECIES+1+TH_SF4] = (RS3 + RS5 + RD4 - RS6 + GwSF4p*thBare
                                  - Gw_total*thSF4) * VoA / n_sites
    dydt[N_SPECIES+1+TH_SF5] = (RS4 + RS6 + RD5 - RS7 + GwSF5p*thBare
                                  - Gw_total*thSF5) * VoA / n_sites
    return dydt

# =====================================================================
# Solver
# =====================================================================
def solve_steady_state(p_OFF, P_abs, t_max=1.0):
    """Integrate from pre-discharge conditions to steady state.
    
    Returns dict with keys: 'n' (densities), 'Te', 'theta', 'dp', 'alpha'
    """
    y0 = np.zeros(N_STATE)
    n_SF6_0 = F_SF6 * p_OFF / (kB * T_GAS)
    y0[SF6] = n_SF6_0
    y0[SF5] = 1e12; y0[SF4] = 1e12; y0[SF3] = 1e11
    y0[F2] = 1e12;  y0[F_] = 1e13
    y0[SF5p] = 1e14; y0[SF4p] = 1e13; y0[SF3p] = 1e13
    y0[F2p] = 1e11;  y0[SF6m] = 1e13; y0[Fm] = 1e14
    y0[EL] = max(y0[SF5p]+y0[SF4p]+y0[SF3p]+y0[F2p]-y0[SF6m]-y0[Fm], 1e13)
    y0[N_SPECIES] = 5.0  # Te [eV]
    y0[N_SPECIES+1:] = [0.05, 0.02, 0.02, 0.02]

    sol = solve_ivp(lambda t, y: ode_system(t, y, p_OFF, P_abs),
                    [0, t_max], y0, method='BDF',
                    rtol=1e-8, atol=1e-6, max_step=0.01)

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    y_final = sol.y[:, -1]
    n = np.maximum(y_final[:N_SPECIES], 0)
    Te = y_final[N_SPECIES]
    theta = y_final[N_SPECIES+1:]
    ne = n[EL]
    alpha = (n[Fm] + n[SF6m]) / max(ne, 1)
    n_gas = sum(n[:N_NEUTRALS])
    dp = n_gas * kB * T_GAS - p_OFF

    return {
        'n': n, 'Te': Te, 'theta': theta,
        'dp': dp, 'alpha': alpha, 'ne': ne,
        'p_OFF': p_OFF, 'P_abs': P_abs,
    }
