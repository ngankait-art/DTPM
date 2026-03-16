"""
Self-consistent 2D electron energy equation solver.

Solves the steady-state energy balance:
    D_eps ∇²(n_e ε̄) + P_ind(r,z)/e - n_e Σ E_k k_kj n_j = 0

where ε̄ = (3/2)Te is the mean electron energy.

The collisional loss term is linearized as ν_loss * (n_e ε̄):
    ν_loss = Σ E_k k_kj n_j / ε̄

This makes the equation:
    D_eps ∇²(n_e ε̄) + P_ind/e - ν_loss * (n_e ε̄) = 0

which is the same form as solve_steady_diffusion_with_source in main.py.

Boundary conditions: n_e ε̄ = 0 at walls (all energy lost at walls).

Returns Te(r,z) = (2/3) ε̄(r,z) = (2/3) (n_e ε̄)(r,z) / n_e(r,z).
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_Te_2d(mesh, ne, P_ind, Eloss_per_e, D_eps, Te_floor=0.5, Te_cap=15.0):
    """Solve the 2D energy equation for Te(r,z).

    Formulation: at each point, steady-state energy balance gives
        P_ind(r,z) / e = ne(r,z) * Eloss_per_e(Te)

    This is an algebraic equation for Te at each point (no diffusion needed
    for the leading-order solution). Diffusion provides a smoothing correction.

    For points where ne is substantial: Te = Te such that Eloss(Te) = P_ind/(ne*e)
    For points where ne is small: Te is smoothed from neighbours by diffusion.

    We solve this as: D∇²Te + (P_local - Eloss_local)/(ne*(3/2)*e) = 0
    where the source/loss are expressed as a forcing on Te.

    In practice, the simplest robust approach is:
    1. Compute Te_local from local energy balance (algebraic)
    2. Smooth with a diffusion step
    """
    from scipy.optimize import brentq as _brentq
    from chemistry.sf6_rates import rates, M_SPECIES

    Nr, Nz = mesh.Nr, mesh.Nz
    AMU = 1.66054e-27

    # Step 1: Algebraic Te at each point from P_ind = ne * Eloss(Te) * e
    Te_raw = np.full((Nr, Nz), 3.0)
    for i in range(Nr):
        for j in range(Nz):
            ne_ij = ne[i,j]; P_ij = P_ind[i,j]
            if ne_ij < 1e12 or P_ij < 1.0:
                Te_raw[i,j] = Te_floor
                continue

            target_Eloss = P_ij / (ne_ij * eC)  # eV/s per electron

            if Eloss_per_e[i,j] > 0:
                # Scale current Te by the ratio
                ratio = target_Eloss / Eloss_per_e[i,j]
                # Eloss scales roughly as exp(-const/Te)*Te, so Te ~ Te_old * ratio^0.3
                Te_est = np.clip(3.0 * ratio**0.3, Te_floor, Te_cap)
                Te_raw[i,j] = Te_est
            else:
                Te_raw[i,j] = Te_floor

    Te_raw = np.clip(Te_raw, Te_floor, Te_cap)

    # Step 2: Smooth with one implicit diffusion step
    # This removes noise from the algebraic step and enforces Te → Te_wall at boundaries
    from solvers.diffusion_profile import solve_diffusion_profile
    # Use the diffusion solver with Te_raw as source and a damping term
    # D∇²Te + (Te_raw - Te)/τ = 0  → steady state gives Te smoothed toward Te_raw
    # Reformulate: D∇²Te - Te/τ = -Te_raw/τ
    tau_smooth = 1e-4  # s — smoothing timescale (small = less smoothing)
    loss_freq = np.full((Nr,Nz), 1.0/tau_smooth)
    source = Te_raw / tau_smooth

    # Solve with Dirichlet BCs (Te = 0 at walls → will be floored)
    Te_smooth = _solve_diffusion_for_Te(mesh, D_eps/1e4, source, loss_freq)
    Te_smooth = np.clip(Te_smooth, Te_floor, Te_cap)

    ne_eps = ne * 1.5 * Te_smooth
    return Te_smooth, ne_eps


def _solve_diffusion_for_Te(mesh, D, source, loss_freq):
    """Solve D∇²Te + S - ν*Te = 0 with Neumann BCs (insulated walls for Te)."""
    Nr, Nz = mesh.Nr, mesh.Nz; N = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz+j; rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            diag = -loss_freq[i,j] if not np.isscalar(loss_freq) else -loss_freq

            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            # Neumann at r=R

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            if j > 0:
                dzc = mesh.dz_c[j]; c = D/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    Te = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(Te, 0.0)


def _solve_energy_diffusion(mesh, D, source, loss_freq):
    """Solve D∇²u + S - ν·u = 0 with Dirichlet u=0 at walls.

    Same structure as solve_steady_diffusion_with_source in main.py
    but always uses Dirichlet BCs (energy is lost at walls).
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    N = Nr * Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz + j
            rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D[i,j] if not np.isscalar(D) else D
            diag = -loss_freq[i,j] if not np.isscalar(loss_freq) else -loss_freq

            # Radial
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c)
                diag -= c
            else:
                rf = mesh.r_faces[Nr]; drc = mesh.dr_c[Nr]
                diag -= D_ij*rf/(rc*dr*drc)  # Dirichlet: u=0 at wall

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c)
                diag -= c
            else:
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # Axial
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[Nz]
                diag -= D_ij/(dz*dzc)

            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c)
                diag -= c
            else:
                dzc = mesh.dz_c[0]
                diag -= D_ij/(dz*dzc)

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u = spsolve(A, rhs).reshape((Nr, Nz))
    return np.maximum(u, 0.0)


def compute_Eloss_field(mesh, ne, Te_field, nAr, nSF6, nF, nArm, kw_Arm):
    """Compute the collisional energy loss rate per electron at each grid point.

    Returns Eloss_per_e(r,z) in [eV/s].
    """
    from chemistry.sf6_rates import rates, M_SPECIES
    AMU = 1.66054e-27
    Nr, Nz = mesh.Nr, mesh.Nz
    Eloss = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz):
            Te_ij = Te_field[i,j]; ne_ij = ne[i,j]
            k = rates(Te_ij)

            # Ar* at this point
            qh = 0.0
            if nSF6[i,j] > 1e10:
                qh = (k['Penn_SF6']+k['qnch_SF6'])*nSF6[i,j]
            den = (k['Ar_iz_m']+k['Ar_q'])*ne_ij + kw_Arm + qh
            nArm_ij = k['Ar_exc']*ne_ij*nAr/max(den,1.0) if ne_ij > 1e8 else 0

            # Fraction of excitation that leads to stepwise ionization
            fstep = k['Ar_iz_m']*ne_ij/max(den, 1.0)
            fstep = min(fstep, 1.0)

            E = (15.7*k['Ar_iz']*nAr
                 + 11.56*k['Ar_exc']*nAr*(1-fstep)
                 + (11.56+4.14)*k['Ar_iz_m']*nArm_ij
                 + k['Ar_el']*nAr*3*m_e/(39.948*AMU)*Te_ij)

            if nSF6[i,j] > 1e10:
                E += ((9.6*k['d1']+12.1*k['d2']+16*k['d3']+18.6*k['d4']+22.7*k['d5']
                      +16*k['iz18']+20*k['iz19']+20.5*k['iz20']+28*k['iz21']
                      +37.5*k['iz22']+18*k['iz23']+29*k['iz24']
                      +0.09*k['vib_SF6']
                      +k['el_SF6']*3*m_e/(M_SPECIES['SF6']*AMU)*Te_ij)*nSF6[i,j]
                      + (15*k['iz28']+14.4*k['exc_F']
                         +k['el_F']*3*m_e/(M_SPECIES['F']*AMU)*Te_ij)*nF[i,j])

            Eloss[i,j] = E

    return Eloss
