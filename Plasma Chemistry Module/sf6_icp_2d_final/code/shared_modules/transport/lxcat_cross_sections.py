"""
LXCat cross-section data for Ar (Phelps) and SF6 (Biagi v10.6).

Parsed from LXCat downloads, March 16, 2026.
References:
  - Phelps database: Yamabe, Buckman, and Phelps, Phys. Rev. 27, 1345 (1983)
  - Biagi database: S.F. Biagi, Magboltz v10.6
"""

import numpy as np
from scipy.interpolate import interp1d

# ═══════════════════════════════════════════════════════════════
# ARGON (Phelps)
# ═══════════════════════════════════════════════════════════════

AR_EFFECTIVE_E = np.array([
    0, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 8.5e-3, 1e-2, 1.5e-2, 2e-2,
    3e-2, 4e-2, 5e-2, 7e-2, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25,
    0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 1.7,
    1.9, 2.1, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6, 4.0, 4.5,
    5.0, 6.0, 7.0, 8.0, 10, 12, 15, 17, 20, 25,
    30, 50, 75, 100, 150, 200, 300, 500, 700, 1000,
    1500, 2000, 3000, 5000, 7000, 10000])
AR_EFFECTIVE_S = np.array([
    7.5e-20, 7.5e-20, 7.1e-20, 6.7e-20, 6.1e-20, 5.4e-20, 5.05e-20, 4.6e-20, 3.75e-20, 3.25e-20,
    2.5e-20, 2.05e-20, 1.73e-20, 1.13e-20, 5.9e-21, 4.0e-21, 2.3e-21, 1.6e-21, 1.03e-21, 9.1e-22,
    1.53e-21, 2.35e-21, 3.3e-21, 5.1e-21, 8.6e-21, 1.38e-20, 1.66e-20, 1.82e-20, 2.1e-20, 2.3e-20,
    2.5e-20, 2.8e-20, 2.9e-20, 3.3e-20, 3.8e-20, 4.1e-20, 4.5e-20, 4.9e-20, 5.4e-20, 6.1e-20,
    6.7e-20, 8.1e-20, 9.6e-20, 1.17e-19, 1.5e-19, 1.52e-19, 1.41e-19, 1.31e-19, 1.1e-19, 9.45e-20,
    8.74e-20, 6.9e-20, 5.85e-20, 5.25e-20, 4.24e-20, 3.76e-20, 3.02e-20, 2.1e-20, 1.64e-20, 1.21e-20,
    8.8e-21, 6.6e-21, 4.5e-21, 3.1e-21, 2.3e-21, 1.75e-21])
AR_MASS_RATIO = 1.36e-5

AR_EXC_E = np.array([
    11.5, 12.7, 13.7, 14.7, 15.9, 16.5, 17.5, 18.5, 19.9, 22.2,
    24.7, 27.0, 30.0, 33.0, 35.3, 42.0, 48.0, 52.0, 70.0, 100,
    150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000])
AR_EXC_S = np.array([
    0, 7e-22, 1.41e-21, 2.28e-21, 3.8e-21, 4.8e-21, 6.1e-21, 7.5e-21, 9.2e-21, 1.17e-20,
    1.33e-20, 1.42e-20, 1.44e-20, 1.41e-20, 1.34e-20, 1.25e-20, 1.16e-20, 1.11e-20, 9.4e-21, 7.6e-21,
    6.0e-21, 5.05e-21, 3.95e-21, 2.8e-21, 2.25e-21, 1.77e-21, 1.36e-21, 1.1e-21, 8.3e-22, 5.8e-22,
    4.5e-22, 3.5e-22])
AR_EXC_THRESHOLD = 11.5

AR_ION_E = np.array([
    15.8, 16.0, 17.0, 18.0, 20.0, 22.0, 23.75, 25.0, 26.5, 30.0,
    32.5, 35.0, 37.5, 40.0, 50.0, 55.0, 100, 150, 200, 300,
    500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000])
AR_ION_S = np.array([
    0, 2.02e-22, 1.34e-21, 2.94e-21, 6.3e-21, 9.3e-21, 1.15e-20, 1.3e-20, 1.45e-20, 1.8e-20,
    1.99e-20, 2.17e-20, 2.31e-20, 2.39e-20, 2.53e-20, 2.6e-20, 2.85e-20, 2.52e-20, 2.39e-20, 2.0e-20,
    1.45e-20, 1.15e-20, 8.6e-21, 6.4e-21, 5.2e-21, 3.6e-21, 2.4e-21, 1.8e-21, 1.35e-21])
AR_ION_THRESHOLD = 15.8

# ═══════════════════════════════════════════════════════════════
# SF6 (Biagi v10.6)
# ═══════════════════════════════════════════════════════════════

SF6_ELASTIC_E = np.array([
    1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5,
    3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 12,
    15, 20, 25, 30, 40, 50, 70, 100, 150, 200,
    300, 500, 1000])
SF6_ELASTIC_S = np.array([
    1.434e-17, 8.9e-18, 6.2e-18, 3.86e-18, 1.99e-18, 1.22e-18, 6.15e-19, 3.6e-19, 2.8e-19, 2.08e-19,
    1.5e-19, 1.2e-19, 1.03e-19, 9.21e-20, 8.57e-20, 8.25e-20, 8.75e-20, 1.05e-19, 1.45e-19, 1.63e-19,
    1.51e-19, 1.43e-19, 1.36e-19, 1.39e-19, 1.51e-19, 1.55e-19, 1.48e-19, 1.47e-19, 1.53e-19, 1.76e-19,
    1.44e-19, 1.57e-19, 1.54e-19, 1.34e-19, 1.03e-19, 8.73e-20, 7.24e-20, 5.49e-20, 3.95e-20, 2.84e-20,
    1.66e-20, 9.6e-21, 4.47e-21])
SF6_MASS_RATIO = 3.7e-6

# SF6 attachment → SF5- (peaks at ~0.5 eV)
SF6_ATT_SF5_E = np.array([
    1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0])
SF6_ATT_SF5_S = np.array([
    0, 4.77e-21, 4.13e-21, 3.42e-21, 2.59e-21, 1.71e-21, 1.13e-21, 9.9e-22, 1.23e-21, 2.65e-21,
    5.13e-21, 7.43e-21, 7.1e-21, 4.79e-21, 2.99e-21, 1.9e-21, 1.3e-21, 5.73e-22, 1.89e-22, 0])

# SF6 attachment → SF6- (thermal, HUGE at 0 eV)
SF6_ATT_SF6_E = np.array([
    1e-6, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2,
    2e-2, 5e-2, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55])
SF6_ATT_SF6_S = np.array([
    1.9845e-16, 1.9845e-16, 1.2346e-16, 8.571e-17, 5.906e-17, 3.551e-17, 2.374e-17, 1.54e-17, 8.654e-18, 5.346e-18,
    3.083e-18, 1.306e-18, 5.047e-19, 1.466e-19, 3.99e-20, 1.0e-20, 1.26e-21, 2.9e-22, 0])

# SF6 vibrational excitations (summed for energy loss)
SF6_VIB_V4_THRESHOLD = 0.076253
SF6_VIB_V1_THRESHOLD = 0.096032
SF6_VIB_V3_THRESHOLD = 0.11754
SF6_VIB_2V1_THRESHOLD = 0.192064
SF6_VIB_3V1_THRESHOLD = 0.288096

# Triplet dissociation at 9.6 eV
SF6_DISS_TRIP_E = np.array([
    9.6, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 25, 30, 40, 50, 70, 100, 200, 500, 1000])
SF6_DISS_TRIP_S = np.array([
    0, 3.6e-23, 1.5e-22, 3.0e-22, 4.5e-22, 5.0e-22, 5.4e-22, 5.65e-22, 5.9e-22, 5.95e-22,
    5.9e-22, 5.8e-22, 4.5e-22, 3.07e-22, 1.66e-22, 1.06e-22, 5.43e-23, 2.66e-23, 6.66e-24, 1.06e-24, 2.66e-25])
SF6_DISS_TRIP_THRESHOLD = 9.6

# Singlet dissociation at 12 eV
SF6_DISS_SING12_E = np.array([
    12, 13, 14, 15, 16, 17, 18, 19, 20, 22,
    25, 30, 40, 50, 70, 100, 200, 500, 1000])
SF6_DISS_SING12_S = np.array([
    0, 9.59e-23, 1.80e-22, 2.55e-22, 3.21e-22, 3.80e-22, 4.33e-22, 4.80e-22, 5.23e-22, 5.95e-22,
    6.80e-22, 7.75e-22, 8.68e-22, 8.96e-22, 8.81e-22, 8.10e-22, 6.03e-22, 3.45e-22, 2.10e-22])
SF6_DISS_SING12_THRESHOLD = 12.0

# Singlet dissociation at 16 eV
SF6_DISS_SING16_E = np.array([
    16, 17, 18, 19, 20, 22, 25, 30, 40, 50,
    70, 100, 200, 500, 1000])
SF6_DISS_SING16_S = np.array([
    0, 7.79e-23, 1.48e-22, 2.12e-22, 2.70e-22, 3.72e-22, 4.93e-22, 6.38e-22, 8.01e-22, 8.74e-22,
    9.09e-22, 8.72e-22, 6.83e-22, 4.06e-22, 2.52e-22])
SF6_DISS_SING16_THRESHOLD = 16.0

# SF6 ionization → SF5+ at 15.67 eV
SF6_ION_E = np.array([
    15.67, 16.5, 17, 18, 19, 20, 22, 25, 28, 30,
    35, 40, 50, 60, 70, 80, 90, 100, 120, 150,
    200, 300, 500, 1000, 2000, 4000])
SF6_ION_S = np.array([
    0, 2.0e-22, 3.5e-22, 8.4e-22, 1.54e-21, 3.14e-21, 4.34e-21, 1.13e-20, 1.36e-20, 1.61e-20,
    2.08e-20, 2.46e-20, 2.81e-20, 3.02e-20, 3.16e-20, 3.25e-20, 3.27e-20, 3.30e-20, 3.33e-20, 3.30e-20,
    3.19e-20, 2.91e-20, 2.40e-20, 1.63e-20, 9.38e-21, 5.32e-21])
SF6_ION_THRESHOLD = 15.67


def get_interpolator(E_arr, S_arr, fill_value=0.0):
    """Create a log-interpolated cross-section function."""
    mask = S_arr > 0
    if mask.sum() < 2:
        return lambda e: np.zeros_like(np.atleast_1d(e))
    return interp1d(np.log10(E_arr[mask] + 1e-30), np.log10(S_arr[mask] + 1e-30),
                    kind='linear', bounds_error=False,
                    fill_value=(np.log10(S_arr[mask][0]+1e-30), -40))


# Build interpolators
def sigma_Ar_eff(e):
    """Ar effective momentum transfer [m²]."""
    f = interp1d(AR_EFFECTIVE_E, AR_EFFECTIVE_S, kind='linear',
                 bounds_error=False, fill_value=(AR_EFFECTIVE_S[0], AR_EFFECTIVE_S[-1]))
    return f(np.atleast_1d(e))

def sigma_Ar_exc(e):
    f = interp1d(AR_EXC_E, AR_EXC_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_Ar_ion(e):
    f = interp1d(AR_ION_E, AR_ION_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_elastic(e):
    f = interp1d(SF6_ELASTIC_E, SF6_ELASTIC_S, kind='linear',
                 bounds_error=False, fill_value=(SF6_ELASTIC_S[0], SF6_ELASTIC_S[-1]))
    return f(np.atleast_1d(e))

def sigma_SF6_att_SF5(e):
    f = interp1d(SF6_ATT_SF5_E, SF6_ATT_SF5_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_att_SF6(e):
    f = interp1d(SF6_ATT_SF6_E, SF6_ATT_SF6_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_trip(e):
    f = interp1d(SF6_DISS_TRIP_E, SF6_DISS_TRIP_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_sing12(e):
    f = interp1d(SF6_DISS_SING12_E, SF6_DISS_SING12_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_diss_sing16(e):
    f = interp1d(SF6_DISS_SING16_E, SF6_DISS_SING16_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)

def sigma_SF6_ion(e):
    f = interp1d(SF6_ION_E, SF6_ION_S, kind='linear', bounds_error=False, fill_value=0)
    return np.maximum(f(np.atleast_1d(e)), 0)


if __name__ == '__main__':
    print("Cross-section data loaded successfully.")
    print(f"  Ar effective: {len(AR_EFFECTIVE_E)} points, Ramsauer min at 0.25eV = {sigma_Ar_eff(0.25)[0]:.2e} m²")
    print(f"  SF6 elastic: {len(SF6_ELASTIC_E)} points, at 1eV = {sigma_SF6_elastic(1.0)[0]:.2e} m²")
    print(f"  SF6 att(SF6-): at 0.001eV = {sigma_SF6_att_SF6(0.001)[0]:.2e} m² (thermal)")
    print(f"  SF6 att(SF5-): at 0.5eV = {sigma_SF6_att_SF5(0.5)[0]:.2e} m² (resonance)")
    print(f"  SF6 ionization: at 30eV = {sigma_SF6_ion(30)[0]:.2e} m²")
