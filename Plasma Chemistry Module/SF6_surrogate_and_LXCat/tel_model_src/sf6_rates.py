"""
SF6/Ar chemistry module.

Contains the complete 54-reaction set from Lallement et al. (2009).
Imported directly from the validated 0D code.
"""
import numpy as np

cm3 = 1e-6  # cm^3 to m^3 conversion

M_SPECIES = {'S':32.06,'F':19.0,'F2':38.0,'SF':51.06,'SF2':70.06,
             'SF3':89.06,'SF4':108.06,'SF5':127.06,'SF6':146.06,'Ar':39.948}


def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990).

    Parameters
    ----------
    k0 : float
        Low-pressure termolecular rate constant [cm⁶/s].
    kinf : float
        High-pressure bimolecular limit [cm³/s].
    Fc : float
        Troe broadening factor.
    M_cm3 : float
        Total gas number density [cm⁻³].

    Returns
    -------
    float
        Effective bimolecular rate coefficient [cm³/s].
    """
    if M_cm3 <= 0 or k0 <= 0:
        return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F


# ═══════════════════════════════════════════════════════════
# Rate coefficients (from sf6_unified.py lines 81–134, UNCHANGED)
# ═══════════════════════════════════════════════════════════

def rates(Te):
    """Compute all 54+ rate coefficients given electron temperature.

    Parameters
    ----------
    Te : float
        Electron temperature [eV]. In the 2D code, Te = (2/3) * mean_energy.

    Returns
    -------
    dict
        All rate coefficients in m³/s (SI). Keys match the 0D code exactly.
    """
    Te = max(Te, 0.3)
    k = {}

    # --- SF6 dissociation (reactions d1–d5) ---
    k['d1']  = 1.5e-7  * np.exp(-8.1 / Te)  * cm3   # SF6 + e → SF5 + F + e
    k['d2']  = 9e-9    * np.exp(-13.4 / Te)  * cm3   # SF6 + e → SF4 + 2F + e
    k['d3']  = 2.5e-8  * np.exp(-33.5 / Te)  * cm3   # SF6 + e → SF3 + 3F + e
    k['d4']  = 2.3e-8  * np.exp(-23.9 / Te)  * cm3   # SF6 + e → SF2 + 2F + F2 + e
    k['d5']  = 1.5e-9  * np.exp(-26.0 / Te)  * cm3   # SF6 + e → SF + 3F + F2 + e

    # --- F2 dissociation and sequential SFx dissociation (d6–d11) ---
    k['d6']  = 1.2e-8  * np.exp(-5.8 / Te)   * cm3   # F2 + e → 2F + e
    k['d7']  = 1.5e-7  * np.exp(-9.0 / Te)   * cm3   # SF5 + e → SF4 + F + e
    k['d8']  = 6.2e-8  * np.exp(-9.0 / Te)   * cm3   # SF4 + e → SF3 + F + e
    k['d9']  = 8.6e-8  * np.exp(-9.0 / Te)   * cm3   # SF3 + e → SF2 + F + e
    k['d10'] = 4.5e-8  * np.exp(-9.0 / Te)   * cm3   # SF2 + e → SF + F + e
    k['d11'] = 6.2e-8  * np.exp(-9.0 / Te)   * cm3   # SF + e → S + F + e

    # --- Excitation and elastic ---
    k['vib_SF6'] = 7.9e-8 * np.exp(-0.1 * Te + 0.002 * Te**2) * cm3
    k['el_SF6']  = 2.8e-7 * np.exp(-1.5 / Te) * cm3
    k['exc_F']   = 9.2e-9 * np.exp(-14.3 / Te) * cm3
    k['el_F']    = 1.1e-7 * np.exp(-1.93 / Te) * cm3
    k['vib_F2']  = 1.8e-10 * Te**1.72 * np.exp(-1.55 / Te) * cm3
    k['el_F2']   = 2.5e-7 * np.exp(-0.48 / Te) * cm3

    # --- Ionization from SF6 (reactions iz18–iz24) ---
    k['iz18'] = 1.2e-7  * np.exp(-18.1 / Te) * cm3  # SF6 + e → SF5+ + F + 2e
    k['iz19'] = 8.4e-9  * np.exp(-19.9 / Te) * cm3  # SF6 + e → SF4+ + 2F + 2e
    k['iz20'] = 3.2e-8  * np.exp(-20.7 / Te) * cm3  # SF6 + e → SF3+ + 3F + 2e
    k['iz21'] = 7.6e-9  * np.exp(-24.4 / Te) * cm3  # SF6 + e → SF2+ + 2F + F2 + 2e
    k['iz22'] = 1.2e-8  * np.exp(-26.0 / Te) * cm3  # SF6 + e → SF+ + 3F + F2 + 2e
    k['iz23'] = 1.4e-8  * np.exp(-39.9 / Te) * cm3  # SF6 + e → S+ + 4F + F2 + 2e
    k['iz24'] = 1.2e-8  * np.exp(-31.7 / Te) * cm3  # SF6 + e → F+ + SF5 + 2e

    # --- Ionization from fragments (iz25–iz29) ---
    k['iz25'] = 1.0e-7  * np.exp(-17.8 / Te) * cm3  # SF5 + e → SF5+ + 2e
    k['iz26'] = 9.4e-8  * np.exp(-22.8 / Te) * cm3  # SF5 + e → SF4+ + F + 2e
    k['iz27'] = 1.0e-7  * np.exp(-18.9 / Te) * cm3  # SF3 + e → SF3+ + 2e
    k['iz28'] = 1.3e-8  * np.exp(-16.5 / Te) * cm3  # F + e → F+ + 2e
    k['iz29'] = 1.6e-7  * np.exp(-13.3 / Te) * cm3  # S + e → S+ + 2e

    # --- Dissociative attachment (at30–at36) ---
    k['at30'] = 2.4e-10 / Te**1.49 * cm3                           # SF6 + e → SF6-
    k['at31'] = 2.0e-11 / Te**1.46 * cm3                           # SF6 + e → SF5- + F
    k['at32'] = 3.9e-12 * np.exp(0.45*Te - 0.04*Te**2) * cm3      # SF6 + e → SF4- + 2F
    k['at33'] = 1.2e-13 * np.exp(0.70*Te - 0.05*Te**2) * cm3      # SF6 + e → SF3- + 3F
    k['at34'] = 5.4e-15 * np.exp(0.77*Te - 0.05*Te**2) * cm3      # SF6 + e → SF2- + 4F
    k['at35'] = 3.4e-11 * np.exp(0.46*Te - 0.04*Te**2) * cm3      # SF6 + e → F- + SF5
    k['at36'] = 2.2e-13 * np.exp(0.71*Te - 0.05*Te**2) * cm3      # SF6 + e → F2- + SF4

    # --- Neutral recombination Troe parameters (Ryan & Plumb 1990) ---
    k['nr42_k0'] = 3.4e-23;  k['nr42_kinf'] = 1.0e-11; k['nr42_Fc'] = 0.43  # SF5+F→SF6
    k['nr41_k0'] = 3.7e-28;  k['nr41_kinf'] = 5.0e-12; k['nr41_Fc'] = 0.46  # SF4+F→SF5
    k['nr40_k0'] = 2.8e-26;  k['nr40_kinf'] = 2.0e-11; k['nr40_Fc'] = 0.47  # SF3+F→SF4
    k['nr39_k0'] = 1.7e-28;  k['nr39_kinf'] = 2.0e-11; k['nr39_Fc'] = 0.56  # SF2+F→SF3
    k['nr38_k0'] = 1.0e-30;  k['nr38_kinf'] = 2.0e-11; k['nr38_Fc'] = 0.67  # SF+F→SF2
    k['nr37_k0'] = 7.5e-33;  k['nr37_kinf'] = 2.0e-11; k['nr37_Fc'] = 0.73  # S+F→SF

    # --- Disproportionation ---
    k['nr45'] = 2.5e-11 * cm3   # SF5 + SF5 → SF4 + SF6
    k['nr44'] = 2.5e-11 * cm3   # SF4 + SF4 → SF3 + SF5
    k['nr43'] = 2.5e-11 * cm3   # SF3 + SF3 → SF2 + SF4

    # --- Ion-ion recombination ---
    k['rec'] = 1.5e-9 * cm3     # All positive + negative ion pairs

    # --- Ar electron-impact reactions ---
    k['Ar_iz']   = 1.2e-10  * np.exp(-21.7 / Te)  * cm3   # Ar + e → Ar+ + 2e
    k['Ar_exc']  = 4.2e-9   * np.exp(-8.0 / Te)   * cm3   # Ar + e → Ar* + e
    k['Ar_iz_m'] = 2.05e-7  * np.exp(-4.95 / Te)  * cm3   # Ar* + e → Ar+ + 2e
    k['Ar_q']    = 2.0e-7   * cm3                          # Ar* + e → Ar + e
    k['Ar_el']   = max((-1.1e-8 + 3.9e-8*Te - 1.3e-8*Te**2
                        + 2e-9*Te**3 - 1.4e-10*Te**4 + 3.9e-12*Te**5) * cm3, 1e-20)

    # --- Penning ionization and Ar* quenching ---
    k['Penn_SF6']  = 2.0e-10 * cm3   # Ar* + SF6 → SF5+ + F + Ar + e
    k['qnch_SF6']  = 3.0e-10 * cm3   # Ar* + SF6 → Ar + SF5 + F
    k['qnch_SFx']  = 1.0e-10 * cm3   # Ar* + SFx → Ar + SFx
    k['qnch_F2']   = 5.0e-11 * cm3   # Ar* + F2 → Ar + 2F
    k['qnch_F']    = 5.0e-12 * cm3   # Ar* + F → Ar + F

    # --- Totals ---
    k['iz_SF6_total']  = sum(k[f'iz{i}'] for i in [18, 19, 20, 21, 22, 23, 24])
    k['att_SF6_total'] = sum(k[f'at{i}'] for i in [30, 31, 32, 33, 34, 35, 36])

    return k


def compute_troe_rates(k, ng_m3):
    """Evaluate Troe fall-off neutral recombination rates at local gas density.

    Parameters
    ----------
    k : dict
        Rate coefficient dictionary from rates().
    ng_m3 : float
        Local total gas number density [m⁻³].

    Returns
    -------
    dict
        Updated k with keys 'nr37' through 'nr42' added [m³/s].
    """
    ng_cm3 = ng_m3 * 1e-6  # Convert to cm⁻³ for Troe formula
    for idx in [42, 41, 40, 39, 38, 37]:
        key = f'nr{idx}'
        k[key] = troe_rate(k[f'{key}_k0'], k[f'{key}_kinf'], k[f'{key}_Fc'], ng_cm3) * cm3
    return k


# ═══════════════════════════════════════════════════════════
# Source term evaluators for the 2D solver
# ═══════════════════════════════════════════════════════════

def electron_source(k, ne, nSF6, nSF5, nSF3, nF, nS, nAr, nArm):
    """Net electron production rate at a single grid point [m⁻³ s⁻¹].

    This is the RHS of the electron continuity equation:
    Sₑ = ionization - attachment + Penning
    """
    # Electron-impact ionization (all channels)
    Riz = ne * (k['iz_SF6_total'] * nSF6
                + (k['iz25'] + k['iz26']) * nSF5
                + k['iz27'] * nSF3
                + k['iz28'] * nF
                + k['iz29'] * nS
                + k['Ar_iz'] * nAr
                + k['Ar_iz_m'] * nArm)

    # Penning ionization
    R_Penn = k['Penn_SF6'] * nArm * nSF6

    # Attachment
    Ratt = ne * k['att_SF6_total'] * nSF6

    return Riz + R_Penn - Ratt


def ion_ion_recombination(k, n_pos, n_neg):
    """Ion-ion recombination rate [m⁻³ s⁻¹]."""
    return k['rec'] * n_pos * n_neg


def energy_loss_density(k, ne, Te, nSF6, nSF5, nSF3, nF, nF2, nS, nAr, nArm):
    """Electron energy loss rate per unit volume [eV m⁻³ s⁻¹].

    This is the collision term in the electron energy equation.
    Transcribed from sf6_unified.py lines 253–261.

    Parameters
    ----------
    All densities in m⁻³, Te in eV.

    Returns
    -------
    P_loss : float
        Energy loss rate [eV m⁻³ s⁻¹]. Multiply by eC (1.6e-19) for Watts/m³.
    """
    from scipy.constants import m_e

    # Fraction of Ar excitation energy that leads to stepwise ionization
    # (the rest is quenched and must be counted as energy loss)
    R_quench = ((k['Ar_iz_m'] + k['Ar_q']) * ne + 1e10)  # approximate
    frac_stepwise = k['Ar_iz_m'] * ne / max(R_quench, 1e-30)
    frac_stepwise = min(frac_stepwise, 1.0)
    Ar_exc_loss = 12.0 * k['Ar_exc'] * nAr * (1.0 - frac_stepwise)

    Eloss = (
        # SF6 ionization energy losses
        (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']) * nSF6
        # SF6 dissociation energy losses
        + (9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3'] + 18.6*k['d4'] + 22.7*k['d5']) * nSF6
        # SF6 vibrational + elastic
        + 0.09 * k['vib_SF6'] * nSF6
        + k['el_SF6'] * nSF6 * 3 * m_e / (M_SPECIES['SF6'] * AMU) * Te
        # SF5 losses
        + (11*k['iz25'] + 15*k['iz26']) * nSF5 + 5*k['d7'] * nSF5
        # SF3 losses
        + 11*k['iz27'] * nSF3 + 5*k['d9'] * nSF3
        # F losses
        + 15*k['iz28'] * nF + 14.4*k['exc_F'] * nF
        + k['el_F'] * nF * 3 * m_e / (M_SPECIES['F'] * AMU) * Te
        # F2 losses
        + 3.2*k['d6'] * nF2 + 0.11*k['vib_F2'] * nF2
        # S losses
        + 10*k['iz29'] * nS
        # Ar losses
        + 16*k['Ar_iz'] * nAr
        + Ar_exc_loss
        + k['Ar_el'] * nAr * 3 * m_e / (M_SPECIES['Ar'] * AMU) * Te
        # Ar* stepwise ionization energy
        + (12 + 4.95) * k['Ar_iz_m'] * nArm
    )

    return ne * Eloss  # [eV m⁻³ s⁻¹]


def fluorine_source(k, ne, nSF6, nSF5, nSF4, nSF3, nSF2, nSF, nF2, nArm):
    """Net F atom volumetric production rate [m⁻³ s⁻¹].

    Transcribed from sf6_unified.py lines 217–224.
    """
    RF = ne * nSF6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5']
                       + k['iz18'] + 2*k['iz19'] + 3*k['iz20'] + 2*k['iz21']
                       + 3*k['iz22'] + 4*k['iz23'] + k['at31'])
    RF += ne * (nSF5 * (k['d7'] + k['iz26']) + nSF4 * k['d8']
                + nSF3 * k['d9'] + nSF2 * k['d10'] + nSF * k['d11'])
    RF += k['Penn_SF6'] * nArm * nSF6
    RF += k['qnch_SF6'] * nArm * nSF6
    RF += 2 * k['qnch_F2'] * nArm * nF2
    RF += 2 * k['d6'] * ne * nF2
    return RF
