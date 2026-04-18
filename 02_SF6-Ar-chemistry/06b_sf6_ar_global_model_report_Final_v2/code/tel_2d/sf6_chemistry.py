"""
SF6/Ar Chemistry Module for 2D Transport
=========================================
Contains all rate coefficients and source/sink terms from the 
validated 0D Lallement model (sf6_global_model_final.py).

Species solved (9 neutrals):
  SF6, SF5, SF4, SF3, SF2, SF, S, F, F2

Electron parameters (ne, Te) are prescribed from 0D balance.
Ar* is in local steady-state (short lifetime).
"""

import numpy as np
from scipy.constants import k as kB, e as eC, pi, atomic_mass as AMU

# Molar masses (kg)
M = {'S':32.06,'F':19.0,'F2':38.0,'SF':51.06,'SF2':70.06,'SF3':89.06,
     'SF4':108.06,'SF5':127.06,'SF6':146.06,'Ar':39.948}

SPECIES = ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']

def compute_diffusion_coefficients(p_mTorr, Tgas):
    """Compute D for each species via Chapman-Enskog at given pressure."""
    p_Pa = p_mTorr * 0.1333
    ng = p_Pa / (kB * Tgas)
    sigma = 3.0e-10  # m, LJ collision diameter (F-SF6 combining rule)
    D = {}
    for sp in SPECIES:
        m_kg = M[sp] * AMU
        v_th = np.sqrt(8 * kB * Tgas / (pi * m_kg))
        mfp = 1.0 / (ng * pi * sigma**2)  # simplified
        D[sp] = v_th * mfp / 3.0  # kinetic theory
    return D

def compute_thermal_speeds(Tgas):
    """Thermal speeds for Robin BC."""
    v_th = {}
    for sp in SPECIES:
        m_kg = M[sp] * AMU
        v_th[sp] = np.sqrt(8 * kB * Tgas / (pi * m_kg))
    return v_th

def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990)."""
    if M_cm3 <= 0 or k0 <= 0: return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F

def compute_rates(Te, ng_cm3, frac_Ar=0.0):
    """Compute all rate coefficients at given Te (eV) and gas density."""
    cm3 = 1e-6  # cm³ → m³
    k = {}
    
    # === Electron-impact dissociation of SF6 ===
    k['d1'] = 1.5e-7 * np.exp(-8.1/Te) * cm3   # SF6 → SF5 + F
    k['d2'] = 9e-9 * np.exp(-13.4/Te) * cm3     # SF6 → SF4 + 2F
    k['d3'] = 2.5e-8 * np.exp(-33.5/Te) * cm3   # SF6 → SF3 + 3F
    k['d4'] = 2.3e-8 * np.exp(-23.9/Te) * cm3   # SF6 → SF2 + 2F + F2
    k['d5'] = 1.5e-9 * np.exp(-26.0/Te) * cm3   # SF6 → SF + 3F + F2
    k['d6'] = 1.2e-8 * np.exp(-5.8/Te) * cm3    # F2 → 2F
    
    # === Electron-impact dissociation of fragments ===
    k['d7'] = 1.5e-7 * np.exp(-9.0/Te) * cm3    # SF5 → SF4 + F
    k['d8'] = 6.2e-8 * np.exp(-9.0/Te) * cm3    # SF4 → SF3 + F
    k['d9'] = 8.6e-8 * np.exp(-9.0/Te) * cm3    # SF3 → SF2 + F
    k['d10'] = 4.5e-8 * np.exp(-9.0/Te) * cm3   # SF2 → SF + F
    k['d11'] = 6.2e-8 * np.exp(-9.0/Te) * cm3   # SF → S + F
    
    # === Ionisation from SF6 ===
    k['iz18'] = 1.2e-7 * np.exp(-18.1/Te) * cm3
    k['iz19'] = 8.4e-9 * np.exp(-19.9/Te) * cm3
    k['iz20'] = 3.2e-8 * np.exp(-20.7/Te) * cm3
    k['iz21'] = 7.6e-9 * np.exp(-24.4/Te) * cm3
    k['iz22'] = 1.2e-8 * np.exp(-26.0/Te) * cm3
    k['iz23'] = 1.4e-8 * np.exp(-39.9/Te) * cm3
    k['iz24'] = 1.2e-8 * np.exp(-31.7/Te) * cm3
    
    # === Ionisation from fragments ===
    k['iz25'] = 1.0e-7 * np.exp(-17.8/Te) * cm3  # SF5
    k['iz26'] = 9.4e-8 * np.exp(-22.8/Te) * cm3   # SF5
    k['iz27'] = 1.0e-7 * np.exp(-18.9/Te) * cm3   # SF3
    k['iz28'] = 1.3e-8 * np.exp(-16.5/Te) * cm3   # F
    k['iz29'] = 1.6e-7 * np.exp(-13.3/Te) * cm3   # S
    
    # === Attachment ===
    k['at30'] = 2.4e-10 / Te**1.49 * cm3
    k['at31'] = 2.0e-11 / Te**1.46 * cm3
    k['at_total'] = k['at30'] + k['at31']
    
    # === Neutral recombination (Troe fall-off) ===
    k['nr42'] = troe_rate(3.4e-23, 1.0e-11, 0.43, ng_cm3) * cm3  # SF5+F→SF6
    k['nr41'] = troe_rate(3.7e-28, 5.0e-12, 0.46, ng_cm3) * cm3  # SF4+F→SF5
    k['nr40'] = troe_rate(2.8e-26, 2.0e-11, 0.47, ng_cm3) * cm3  # SF3+F→SF4
    k['nr39'] = troe_rate(1.7e-28, 2.0e-11, 0.56, ng_cm3) * cm3  # SF2+F→SF3
    k['nr38'] = troe_rate(1.0e-30, 2.0e-11, 0.67, ng_cm3) * cm3  # SF+F→SF2
    k['nr37'] = troe_rate(7.5e-33, 2.0e-11, 0.73, ng_cm3) * cm3  # S+F→SF
    k['nr45'] = 2.5e-11 * cm3  # SF5+SF5→SF4+SF6
    
    # === Penning / Ar* chemistry ===
    k['Penn_SF6'] = 2.0e-10 * cm3
    k['qnch_SF6'] = 3.0e-10 * cm3
    k['qnch_SFx'] = 1.0e-10 * cm3
    k['qnch_F2'] = 5.0e-11 * cm3
    k['qnch_F'] = 5.0e-12 * cm3
    k['Ar_exc'] = 4.2e-9 * np.exp(-8.0/Te) * cm3
    k['Ar_iz_m'] = 2.05e-7 * np.exp(-4.95/Te) * cm3
    k['Ar_q'] = 2.0e-7 * cm3
    
    # Totals
    k['iz_SF6_total'] = k['iz18']+k['iz19']+k['iz20']+k['iz21']+k['iz22']+k['iz23']+k['iz24']
    k['k_e_SF6'] = k['d1']+k['d2']+k['d3']+k['d4']+k['d5']+k['iz_SF6_total']+k['at_total']
    
    return k


def compute_source_sink(species_fields, ne, Te, ng_cm3, frac_Ar, Tgas, tau_R, nSF6_feed):
    """
    Compute volumetric source/sink for each species at each grid point.
    
    species_fields: dict of 2D arrays {name: field[Nr,Nz]} in m^-3
    ne, Te: 2D arrays
    Returns: dict of 2D arrays {name: source[Nr,Nz]} in m^-3 s^-1
    """
    nSF6 = species_fields['SF6']
    nSF5 = species_fields['SF5']
    nSF4 = species_fields['SF4']
    nSF3 = species_fields['SF3']
    nSF2 = species_fields['SF2']
    nSF  = species_fields['SF']
    nS   = species_fields['S']
    nF   = species_fields['F']
    nF2  = species_fields['F2']
    
    k = compute_rates(np.mean(Te[Te > 0]) if np.any(Te > 0) else 2.5, ng_cm3, frac_Ar)
    
    # Ar* local balance
    nAr0 = frac_Ar * ng_cm3 * 1e6  # m^-3
    if frac_Ar > 0:
        R_quench = (k['Penn_SF6']*nSF6 + k['qnch_SF6']*nSF6 + 
                    k['qnch_SFx']*(nSF5+nSF4+nSF3+nSF2+nSF) +
                    k['qnch_F2']*nF2 + k['qnch_F']*nF)
        nArm = k['Ar_exc'] * ne * nAr0 / (
            (k['Ar_iz_m'] + k['Ar_q']) * ne + R_quench + 1e-30)
    else:
        nArm = np.zeros_like(ne)
    
    S = {}
    
    # SF6: feed replenishment - electron destruction - Penning
    S['SF6'] = ((nSF6_feed - nSF6) / tau_R 
                - k['k_e_SF6'] * ne * nSF6
                - (k['Penn_SF6'] + k['qnch_SF6']) * nArm * nSF6
                + k['nr42'] * nSF5 * nF
                + k['nr45'] * nSF5**2)
    
    # SF5: from SF6 dissociation, lost by further dissociation and recombination
    S['SF5'] = (k['d1'] * ne * nSF6
                + k['nr41'] * nSF4 * nF
                + k['qnch_SF6'] * nArm * nSF6
                - (k['d7'] + k['iz25'] + k['iz26']) * ne * nSF5
                - k['nr42'] * nSF5 * nF
                - 2 * k['nr45'] * nSF5**2
                - k['qnch_SFx'] * nArm * nSF5
                - nSF5 / tau_R)
    
    # SF4
    S['SF4'] = (k['d2'] * ne * nSF6
                + k['d7'] * ne * nSF5
                + k['nr45'] * nSF5**2
                + k['nr40'] * nSF3 * nF
                - k['d8'] * ne * nSF4
                - k['nr41'] * nSF4 * nF
                - k['qnch_SFx'] * nArm * nSF4
                - nSF4 / tau_R)
    
    # SF3
    S['SF3'] = (k['d3'] * ne * nSF6
                + k['d8'] * ne * nSF4
                + k['nr39'] * nSF2 * nF
                - (k['d9'] + k['iz27']) * ne * nSF3
                - k['nr40'] * nSF3 * nF
                - k['qnch_SFx'] * nArm * nSF3
                - nSF3 / tau_R)
    
    # SF2
    S['SF2'] = (k['d4'] * ne * nSF6
                + k['d9'] * ne * nSF3
                + k['nr38'] * nSF * nF
                - k['d10'] * ne * nSF2
                - k['nr39'] * nSF2 * nF
                - nSF2 / tau_R)
    
    # SF
    S['SF'] = (k['d5'] * ne * nSF6
               + k['d10'] * ne * nSF2
               + k['nr37'] * nS * nF
               - k['d11'] * ne * nSF
               - k['nr38'] * nSF * nF
               - nSF / tau_R)
    
    # S
    S['S'] = (k['d11'] * ne * nSF
              - k['iz29'] * ne * nS
              - k['nr37'] * nS * nF
              - nS / tau_R)
    
    # F: produced by all dissociation channels
    RF = ne * nSF6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5']
                       + k['iz18'] + 2*k['iz19'] + 3*k['iz20'] + 2*k['iz21']
                       + 3*k['iz22'] + 4*k['iz23'] + k['at31'])
    RF += ne * (nSF5*(k['d7']+k['iz26']) + nSF4*k['d8'] + nSF3*k['d9']
                + nSF2*k['d10'] + nSF*k['d11'])
    RF += k['Penn_SF6'] * nArm * nSF6  # Penning
    RF += k['qnch_SF6'] * nArm * nSF6  # non-ionising quench
    RF += 2 * k['qnch_F2'] * nArm * nF2
    RF += 2 * k['d6'] * ne * nF2
    
    # F losses: wall (handled by Robin BC), pump, ionisation, neutral recombination
    F_loss = (nF / tau_R
              + k['iz28'] * ne * nF
              + k['nr42'] * nSF5 * nF
              + k['nr41'] * nSF4 * nF
              + k['nr40'] * nSF3 * nF
              + k['nr39'] * nSF2 * nF
              + k['nr38'] * nSF * nF
              + k['nr37'] * nS * nF)
    
    S['F'] = RF - F_loss
    
    # F2
    S['F2'] = (ne * nSF6 * (k['d4'] + k['d5'] + k['iz21'] + k['iz22'] + k['iz23'])
               - k['d6'] * ne * nF2
               - k['qnch_F2'] * nArm * nF2
               - nF2 / tau_R)
    
    return S


# Wall recombination probabilities (Kokkoris 2009) — per species, per material
# gamma_eff = s_F for F, beta_SFx for SFx fragments, ~0 for F2
WALL_GAMMA = {
    'quartz': {'SF6':0, 'SF5':0.0005, 'SF4':0.0005, 'SF3':0.0005, 
               'SF2':0.0005, 'SF':0.0005, 'S':0.001, 'F':0.001, 'F2':0.0001},
    'aluminium': {'SF6':0, 'SF5':0.008, 'SF4':0.008, 'SF3':0.008,
                  'SF2':0.008, 'SF':0.008, 'S':0.01, 'F':0.015, 'F2':0.001},
    'silicon': {'SF6':0, 'SF5':0.001, 'SF4':0.001, 'SF3':0.001,
                'SF2':0.001, 'SF':0.001, 'S':0.001, 'F':0.025, 'F2':0.0001},
    'window': {'SF6':0, 'SF5':0.0003, 'SF4':0.0003, 'SF3':0.0003,
               'SF2':0.0003, 'SF':0.0003, 'S':0.001, 'F':0.001, 'F2':0.0001},
}
