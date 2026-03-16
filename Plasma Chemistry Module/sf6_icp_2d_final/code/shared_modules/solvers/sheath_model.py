"""
Analytic sheath model for ICP plasma-wall interface.

Provides:
  - Sheath voltage V_s from Bohm criterion + floating potential
  - Ion energy at the wall E_ion = e*|V_s| + Te/2
  - Ion flux to each wall surface: Gamma_i = n_e,edge * u_B
  - Ion-enhanced etch probability correction

Physics:
  At the sheath edge, ions enter with Bohm velocity u_B.
  The sheath potential drop accelerates ions and repels electrons.
  For a floating wall: V_s = -(Te/2)*ln(Mi/(2*pi*me))
  For a biased wafer: V_s = V_float + V_bias

References:
  Lieberman & Lichtenberg, "Principles of Plasma Discharges" Ch. 6
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi

AMU = 1.66054e-27


def sheath_voltage(Te_eV, Mi_kg):
    """Floating sheath voltage (negative, repels electrons).
    
    V_s = -(Te/2) * ln(Mi / (2*pi*me))
    
    Parameters
    ----------
    Te_eV : float or array — electron temperature [eV]
    Mi_kg : float — ion mass [kg]
    
    Returns
    -------
    V_s : float or array — sheath voltage [V] (negative)
    """
    return -0.5 * Te_eV * np.log(Mi_kg / (2 * pi * m_e))


def bohm_velocity(Te_eV, Mi_kg, alpha=0.0, gamma=None, T_neg=0.3):
    """Modified Bohm velocity for electronegative plasma.
    
    u_B = sqrt(e*Te*(1+alpha) / (Mi*(1+gamma*alpha)))
    """
    if gamma is None:
        gamma = Te_eV / max(T_neg, 0.01)
    return np.sqrt(eC * Te_eV * (1 + alpha) / (Mi_kg * (1 + gamma * alpha)))


def ion_flux_to_wall(ne_edge, Te_eV, Mi_kg, alpha=0.0, T_neg=0.3):
    """Ion flux at the wall [m^-2 s^-1].
    
    Gamma_i = n_e,edge * u_B
    """
    uB = bohm_velocity(Te_eV, Mi_kg, alpha, T_neg=T_neg)
    return ne_edge * uB


def ion_energy_at_wall(Te_eV, Mi_kg, V_bias=0.0):
    """Mean ion energy arriving at the wall [eV].
    
    E_ion = e*|V_s| + Te/2 + e*|V_bias|
    The Te/2 is the directed kinetic energy at the sheath edge (Bohm).
    """
    V_s = sheath_voltage(Te_eV, Mi_kg)
    return abs(V_s) + 0.5 * Te_eV + abs(V_bias)


def compute_wall_fluxes(ne_2d, Te_2d, alpha_2d, mesh, Mi_kg, T_neg=0.3):
    """Compute ion flux profiles at all four walls.
    
    Returns dict with:
      'wafer':   Gamma_i(r) at z=0
      'window':  Gamma_i(r) at z=L
      'sidewall': Gamma_i(z) at r=R
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    
    # Wafer (z=0): use j=0
    ne_wafer = ne_2d[:, 0]
    Te_wafer = Te_2d[:, 0]
    al_wafer = alpha_2d[:, 0]
    flux_wafer = np.array([
        ion_flux_to_wall(ne_wafer[i], Te_wafer[i], Mi_kg, al_wafer[i], T_neg)
        for i in range(Nr)])
    
    # Window (z=L): use j=Nz-1
    ne_window = ne_2d[:, -1]
    Te_window = Te_2d[:, -1]
    al_window = alpha_2d[:, -1]
    flux_window = np.array([
        ion_flux_to_wall(ne_window[i], Te_window[i], Mi_kg, al_window[i], T_neg)
        for i in range(Nr)])
    
    # Sidewall (r=R): use i=Nr-1
    ne_side = ne_2d[-1, :]
    Te_side = Te_2d[-1, :]
    al_side = alpha_2d[-1, :]
    flux_side = np.array([
        ion_flux_to_wall(ne_side[j], Te_side[j], Mi_kg, al_side[j], T_neg)
        for j in range(Nz)])
    
    # Energy at wafer
    E_wafer = np.array([ion_energy_at_wall(Te_wafer[i], Mi_kg) for i in range(Nr)])
    
    return {
        'wafer_flux': flux_wafer,       # m^-2 s^-1, vs r
        'window_flux': flux_window,
        'sidewall_flux': flux_side,     # vs z
        'wafer_energy': E_wafer,        # eV, vs r
        'wafer_r_cm': mesh.r * 100,
        'sidewall_z_cm': mesh.z * 100,
    }


def ion_enhanced_etch_probability(E_ion_eV, gamma_chem=0.025, E_th=20.0, Y0=0.5):
    """Ion-enhanced etch probability.
    
    gamma_eff = gamma_chem + Y0 * sqrt(max(E_ion - E_th, 0)) / E_ion
    
    Parameters
    ----------
    E_ion_eV : ion energy at the wafer [eV]
    gamma_chem : chemical (radical-only) etch probability
    E_th : ion sputtering threshold [eV]
    Y0 : ion sputtering yield coefficient
    """
    E = np.asarray(E_ion_eV, dtype=float)
    phys = np.maximum(E - E_th, 0)
    ion_yield = Y0 * np.sqrt(phys) / np.maximum(E, 1.0)
    return gamma_chem + ion_yield


if __name__ == '__main__':
    Mi_SF5 = 127.06 * AMU
    Mi_Ar = 39.948 * AMU
    
    print("Sheath Model Validation")
    print("=" * 50)
    
    for Te in [2.0, 3.0, 5.0]:
        Vs = sheath_voltage(Te, Mi_SF5)
        uB = bohm_velocity(Te, Mi_SF5, alpha=35)
        Ei = ion_energy_at_wall(Te, Mi_SF5)
        print(f"\nTe={Te:.1f} eV (SF5+, alpha=35):")
        print(f"  V_sheath = {Vs:.1f} V")
        print(f"  u_Bohm   = {uB:.0f} m/s")
        print(f"  E_ion    = {Ei:.1f} eV")
        print(f"  gamma_eff(E_ion) = {ion_enhanced_etch_probability(Ei):.4f}")
    
    print(f"\nAr+ at Te=3 eV:")
    Vs = sheath_voltage(3.0, Mi_Ar)
    Ei = ion_energy_at_wall(3.0, Mi_Ar)
    print(f"  V_sheath = {Vs:.1f} V, E_ion = {Ei:.1f} eV")
