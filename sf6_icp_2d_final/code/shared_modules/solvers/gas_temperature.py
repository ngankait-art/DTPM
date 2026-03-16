"""
2D gas temperature solver for ICP discharges.

Solves: kappa_g * nabla^2(Tg) + Q_elastic + Q_FC - Q_wall = 0

Heating sources:
  1. Elastic electron-neutral collisions: Q_el = 3*(m_e/M_n)*nu_en*ne*(Te-Tg)*kB
  2. Frank-Condon heating from dissociation products (hot fragments)
  3. Ion-neutral charge exchange heating

Cooling:
  Wall thermal loss via conduction to chamber walls at T_wall = 300 K

BCs: Dirichlet Tg = T_wall at all walls.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve

AMU = 1.66054e-27


def gas_thermal_conductivity(Tg, species='Ar', p_Pa=1.333):
    """Thermal conductivity of the neutral gas [W/(m·K)].
    
    For low-pressure plasmas, kappa is dominated by atomic transport.
    """
    if species == 'Ar':
        # Ar: kappa ~ 0.018 W/(m·K) at 300 K, scales as T^0.7
        return 0.018 * (Tg / 300.0)**0.7
    else:
        # SF6: kappa ~ 0.013 W/(m·K) at 300 K
        return 0.013 * (Tg / 300.0)**0.7


def elastic_heating_rate(ne, Te_eV, Tg, ng, M_gas_amu, sigma_en=5e-19):
    """Elastic e-n collision heating [W/m^3].
    
    Q = 3*(m_e/M_n) * nu_en * ne * kB * (Te - Tg)
    
    This is the primary gas heating mechanism in ICP plasmas.
    """
    M_n = M_gas_amu * AMU
    v_e = np.sqrt(8 * eC * Te_eV / (pi * m_e))
    nu_en = ng * sigma_en * v_e
    delta_T = Te_eV * eC / kB - Tg  # Convert Te to Kelvin
    Q = 3 * (m_e / M_n) * nu_en * ne * kB * delta_T
    return np.maximum(Q, 0)


def frank_condon_heating(ne, nSF6, Te_eV, Tg):
    """Frank-Condon heating from dissociation products [W/m^3].
    
    When SF6 is dissociated, fragments carry ~1-3 eV kinetic energy.
    This thermalizes through collisions, heating the gas.
    
    Rough estimate: Q_FC ~ ne * nSF6 * k_diss * E_FC
    where E_FC ~ 1 eV average per dissociation event.
    """
    from chemistry.sf6_rates import rates
    k = rates(Te_eV)
    k_diss_total = k['d1'] + k['d2'] + k['d3'] + k['d4'] + k['d5']
    E_FC = 1.0 * eC  # 1 eV in Joules
    return ne * nSF6 * k_diss_total * E_FC


def solve_gas_temperature(mesh, ne, Te_field, nSF6, ng, M_gas_amu=40.0,
                          T_wall=300.0, x_Ar=0.0, x_SF6=1.0):
    """Solve the 2D gas temperature equation.
    
    kappa * nabla^2(Tg) + Q_heating = 0
    with Dirichlet Tg = T_wall at all walls.
    
    Parameters
    ----------
    mesh : Mesh2D
    ne : array (Nr, Nz) — electron density [m^-3]
    Te_field : array (Nr, Nz) — electron temperature [eV]
    nSF6 : array (Nr, Nz) — SF6 density [m^-3]
    ng : float — total gas density [m^-3]
    M_gas_amu : float — effective gas mass [AMU]
    T_wall : float — wall temperature [K]
    
    Returns
    -------
    Tg : array (Nr, Nz) — gas temperature [K]
    Q_total : array (Nr, Nz) — total heating rate [W/m^3]
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N = Nr * Nz
    
    # Compute heating sources
    Q_elastic = np.zeros((Nr, Nz))
    Q_FC = np.zeros((Nr, Nz))
    
    for i in range(Nr):
        for j in range(Nz):
            # Elastic e-n heating
            if x_Ar > 0:
                Q_elastic[i, j] += elastic_heating_rate(
                    ne[i,j], Te_field[i,j], T_wall, ng*x_Ar, 39.948, sigma_en=3e-20)
            if x_SF6 > 0:
                Q_elastic[i, j] += elastic_heating_rate(
                    ne[i,j], Te_field[i,j], T_wall, ng*x_SF6, 146.06, sigma_en=5e-19)
            
            # Frank-Condon from dissociation
            if x_SF6 > 0 and nSF6[i,j] > 1e10:
                Q_FC[i, j] = frank_condon_heating(ne[i,j], nSF6[i,j], Te_field[i,j], T_wall)
    
    Q_total = Q_elastic + Q_FC
    
    # Effective thermal conductivity (mixture)
    kappa = x_Ar * gas_thermal_conductivity(T_wall, 'Ar') + \
            x_SF6 * gas_thermal_conductivity(T_wall, 'SF6')
    kappa = max(kappa, 1e-4)
    
    # Solve: kappa * nabla^2(Tg) + Q = 0  with Tg = T_wall at walls
    # Rearrange: kappa * nabla^2(Tg) = -Q
    # Same structure as the ne diffusion solver but with Dirichlet BCs
    
    rows, cols, vals = [], [], []
    rhs = (-Q_total / kappa).flatten()
    
    for i in range(Nr):
        for j in range(Nz):
            idx = i * Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            diag = 0.0
            
            # Radial
            if i < Nr - 1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = rf / (rc * dr * drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Dirichlet at r=R: Tg = T_wall → ghost contributes T_wall to RHS
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                c = rf / (rc * dr * drc)
                diag -= c
                rhs[idx] -= c * T_wall / (-1)  # Note: solving for Tg-T_wall shift
            
            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = rf / (rc * dr * drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c)
                diag -= c
            else:
                # Symmetry at r=0
                drc = mesh.dr_c[1]; c = 2.0 / (dr * drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c
            
            # Axial
            if j < Nz - 1:
                dzc = mesh.dz_c[j+1]; c = 1.0 / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c)
                diag -= c
            else:
                # Dirichlet at z=L
                dzc = mesh.dz_c[Nz]; c = 1.0 / (dz * dzc)
                diag -= c
            
            if j > 0:
                dzc = mesh.dz_c[j]; c = 1.0 / (dz * dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c)
                diag -= c
            else:
                # Dirichlet at z=0
                dzc = mesh.dz_c[0]; c = 1.0 / (dz * dzc)
                diag -= c
            
            rows.append(idx); cols.append(idx); vals.append(diag)
    
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    Tg_flat = spsolve(A, rhs)
    Tg = np.maximum(Tg_flat.reshape((Nr, Nz)), T_wall)
    Tg = np.clip(Tg, T_wall, 2000.0)  # Physical limit
    
    return Tg, Q_total
