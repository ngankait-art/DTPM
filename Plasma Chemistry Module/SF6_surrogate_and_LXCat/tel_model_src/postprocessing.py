"""
Post-processing utilities for TEL simulation results.
"""
import numpy as np
from scipy.constants import k as kB


def etch_rate_profile(F_wafer_cm3, Tgas=313, gamma_Si=0.025):
    """Compute Si etch rate from F density at wafer.
    
    Parameters
    ----------
    F_wafer_cm3 : array
        F density at wafer (cm^-3).
    Tgas : float
        Gas temperature (K).
    gamma_Si : float
        Si etch probability.
        
    Returns
    -------
    R_etch : array
        Etch rate (nm/s).
    """
    v_th_F = np.sqrt(8 * kB * Tgas / (np.pi * 19.0 * 1.66054e-27))
    n_Si = 5e28  # Si atom density (m^-3)
    return gamma_Si * v_th_F / 4 * F_wafer_cm3 * 1e6 / n_Si * 1e9


def zone_averages(result, solver):
    """Compute volume-averaged quantities in each zone.
    
    Returns dict with ICP and processing averages.
    """
    m = result['mesh']
    Nr, Nz = m.Nr, m.Nz
    ins = result['inside']
    nF = result['nF']

    icp_mask = (ins &
                (np.outer(m.rc, np.ones(Nz)) <= solver.R_icp) &
                (np.outer(np.ones(Nr), m.zc) >= solver.z_apt_top))
    proc_mask = ins & (np.outer(np.ones(Nr), m.zc) < solver.z_apt_bot)

    def vol_avg(field, mask):
        return np.sum(field[mask] * m.vol[mask]) / max(np.sum(m.vol[mask]), 1e-30)

    return {
        'F_icp': vol_avg(nF, icp_mask) * 1e-6,    # cm^-3
        'F_proc': vol_avg(nF, proc_mask) * 1e-6,
        'ne_icp': vol_avg(result['ne'], icp_mask) * 1e-6,
        'Te_icp': np.mean(result['Te'][icp_mask]) if np.any(icp_mask) else 0,
    }
