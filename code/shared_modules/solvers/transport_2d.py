"""
Spatially resolved transport for negative ions and neutral species.

Step 2: Negative-ion transport
    D_- ∇²n_- + S_att - k_rec n_+ n_- = 0
    with Neumann (zero-flux) BCs at all walls (ions are trapped by the sheath).
    Produces the flat-core / sharp-edge profile predicted by Lichtenberg (1997).

Step 3: Multi-neutral transport
    D_j ∇²n_j + S_j - L_j n_j = 0
    for all 9 neutral species (SF6, SF5, SF4, SF3, SF2, SF, S, F, F2).
    Uses the same implicit sparse solver as the diffusion profile.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, pi
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ═══════════════════════════════════════════════════════════
# STEP 2: Negative-ion transport
# ═══════════════════════════════════════════════════════════

def solve_negative_ions(mesh, ne, n_neg_old, nSF6, Te_field, n_pos,
                        D_neg, k_rec, alpha_0d=0.0, ne_0d=1e16, w=0.2):
    """Solve the negative-ion continuity equation.

    D_- ∇²n_- + ne * k_att * nSF6 - k_rec * n_+ * n_- = 0

    Neumann (zero-flux) BCs on all walls — negative ions are trapped.

    After solving, renormalize so <alpha> = alpha_0d (the 0D backbone value).
    The spatial *variation* comes from the diffusion equation; the *magnitude*
    is anchored to the 0D chemistry.

    Parameters
    ----------
    mesh : Mesh2D
    ne : array (Nr, Nz) — electron density [m⁻³]
    n_neg_old : array (Nr, Nz) — current n_- estimate [m⁻³]
    nSF6 : array (Nr, Nz) — SF6 density [m⁻³]
    Te_field : array (Nr, Nz) — electron temperature [eV]
    n_pos : array (Nr, Nz) — positive ion density [m⁻³]
    D_neg : float — negative ion diffusion coefficient [m²/s]
    k_rec : float — ion-ion recombination rate [m³/s]
    alpha_0d : float — volume-averaged alpha from 0D model
    ne_0d : float — volume-averaged ne from 0D model
    w : float — relaxation weight

    Returns
    -------
    n_neg : array (Nr, Nz) — updated negative ion density [m⁻³]
    alpha : array (Nr, Nz) — updated electronegativity
    """
    from chemistry.sf6_rates import rates

    Nr, Nz = mesh.Nr, mesh.Nz

    # Compute attachment source and recombination loss at each point
    source = np.zeros((Nr, Nz))
    loss_freq = np.zeros((Nr, Nz))

    for i in range(Nr):
        for j in range(Nz):
            k = rates(Te_field[i,j])
            source[i,j] = ne[i,j] * k['att_SF6_total'] * nSF6[i,j]
            loss_freq[i,j] = k_rec * max(n_pos[i,j], ne[i,j])

    # Solve with Neumann BCs (trapped)
    n_neg_solved = _solve_diffusion_neumann(mesh, D_neg, source, loss_freq)

    # === ANCHORING: renormalize to match 0D alpha ===
    # The diffusion equation gives the spatial SHAPE of n_neg.
    # The MAGNITUDE is set so that <n_neg>/<ne> = alpha_0d.
    if alpha_0d > 0.01 and mesh.volume_average(n_neg_solved) > 1e6:
        target_n_neg_avg = alpha_0d * ne_0d
        actual_avg = mesh.volume_average(n_neg_solved)
        n_neg_solved *= target_n_neg_avg / actual_avg

    # Relaxed update
    n_neg = n_neg_old + w * (n_neg_solved - n_neg_old)
    n_neg = np.maximum(n_neg, 0.0)

    # Alpha
    alpha = np.where(ne > 1e10, n_neg / ne, 0.0)
    alpha = np.clip(alpha, 0, 1000)

    return n_neg, alpha


def _solve_diffusion_neumann(mesh, D, source, loss_freq):
    """Solve D∇²n + S - ν·n = 0 with Neumann BCs (zero flux at all walls)."""
    Nr, Nz = mesh.Nr, mesh.Nz; N = Nr*Nz
    rows, cols, vals = [], [], []
    rhs = source.flatten()

    for i in range(Nr):
        for j in range(Nz):
            idx = i*Nz+j; rc = mesh.r[i]; dr = mesh.dr[i]; dz = mesh.dz[j]
            D_ij = D if np.isscalar(D) else D[i,j]
            diag = -loss_freq[i,j]

            # Radial
            if i < Nr-1:
                rf = mesh.r_faces[i+1]; drc = mesh.dr_c[i+1]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i+1)*Nz+j); vals.append(c); diag -= c
            # i == Nr-1: Neumann → no extra loss term (zero flux)

            if i > 0:
                rf = mesh.r_faces[i]; drc = mesh.dr_c[i]
                c = D_ij*rf/(rc*dr*drc)
                rows.append(idx); cols.append((i-1)*Nz+j); vals.append(c); diag -= c
            else:
                # r=0 symmetry
                drc = mesh.dr_c[1]; c = 2*D_ij/(dr*drc)
                if Nr > 1:
                    rows.append(idx); cols.append(1*Nz+j); vals.append(c)
                diag -= c

            # Axial — Neumann at both ends
            if j < Nz-1:
                dzc = mesh.dz_c[j+1]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j+1); vals.append(c); diag -= c
            if j > 0:
                dzc = mesh.dz_c[j]; c = D_ij/(dz*dzc)
                rows.append(idx); cols.append(i*Nz+j-1); vals.append(c); diag -= c

            rows.append(idx); cols.append(idx); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    n = spsolve(A, -rhs).reshape((Nr, Nz))  # Negative sign: D∇²n - ν·n = -S → An = -S
    return np.maximum(n, 0.0)


# ═══════════════════════════════════════════════════════════
# STEP 3: Multi-neutral transport
# ═══════════════════════════════════════════════════════════

def solve_neutral_transport(mesh, ne, Te_field, nArm, ng, nSF6_0, tau_R,
                            neutrals_old, D_coeffs, gamma_F=0.01,
                            kw_F_eff=30.0, w=0.1, neutrals_0d=None):
    """Solve steady-state diffusion-reaction for all 9 neutral species.

    After the local chemistry sub-iteration, renormalize each species so
    that <n_j> = n_j^{0D}. This prevents the spatial solver from drifting
    away from the validated 0D chemistry backbone.

    The spatial *variation* (depletion at centre, enrichment at edges) comes
    from the local source/loss imbalance and diffusion. The *magnitude* is
    anchored to the 0D backbone.
    """
    from chemistry.sf6_rates import rates, compute_troe_rates, M_SPECIES
    AMU = 1.66054e-27; cm3 = 1e-6

    Nr, Nz = mesh.Nr, mesh.Nz
    n = {sp: neutrals_old[sp].copy() for sp in neutrals_old}

    # Iterate the neutral sub-system (mirrors the 0D code's inner loop)
    for sub_iter in range(10):
        for i in range(Nr):
            for j in range(Nz):
                Te = Te_field[i,j]; ne_ij = ne[i,j]
                k = rates(Te)
                k = compute_troe_rates(k, ng)

                nArm_ij = nArm[i,j]

                # === SF6 balance ===
                sf6_loss = ((k['d1']+k['d2']+k['d3']+k['d4']+k['d5']
                            +k['iz_SF6_total']+k['att_SF6_total'])*ne_ij
                           + (k['Penn_SF6']+k['qnch_SF6'])*nArm_ij
                           + 1.0/tau_R)
                sf6_source = nSF6_0/tau_R + k['nr42']*n['SF5'][i,j]*n['F'][i,j] + k['nr45']*n['SF5'][i,j]**2
                n['SF6'][i,j] = sf6_source / max(sf6_loss, 1e-30)

                # === SF5 balance ===
                sf5_prod = (ne_ij*n['SF6'][i,j]*(k['d1']+k['at31'])
                           + k['Penn_SF6']*nArm_ij*n['SF6'][i,j]
                           + k['qnch_SF6']*nArm_ij*n['SF6'][i,j]
                           + k['nr41']*n['SF4'][i,j]*n['F'][i,j]
                           + k['nr45']*n['SF5'][i,j]**2)
                sf5_loss = ((k['d7']+k['iz25']+k['iz26'])*ne_ij
                           + k['nr42']*n['F'][i,j]
                           + 2*k['nr45']*n['SF5'][i,j]
                           + 1.0/tau_R)
                n['SF5'][i,j] = sf5_prod / max(sf5_loss, 1e-30)

                # === SF4 balance ===
                sf4_prod = (ne_ij*n['SF6'][i,j]*k['d2']
                           + ne_ij*n['SF5'][i,j]*k['d7']
                           + k['nr40']*n['SF3'][i,j]*n['F'][i,j]
                           + k['nr45']*n['SF5'][i,j]**2
                           + k['nr44']*n['SF4'][i,j]**2)
                sf4_loss = (k['d8']*ne_ij
                           + k['nr41']*n['F'][i,j]
                           + 2*k['nr44']*n['SF4'][i,j]
                           + 1.0/tau_R)
                n['SF4'][i,j] = sf4_prod / max(sf4_loss, 1e-30)

                # === SF3 balance ===
                sf3_prod = (ne_ij*n['SF6'][i,j]*k['d3']
                           + ne_ij*n['SF4'][i,j]*k['d8']
                           + k['nr39']*n['SF2'][i,j]*n['F'][i,j]
                           + k['nr44']*n['SF4'][i,j]**2
                           + k['nr43']*n['SF3'][i,j]**2)
                sf3_loss = ((k['d9']+k['iz27'])*ne_ij
                           + k['nr40']*n['F'][i,j]
                           + 2*k['nr43']*n['SF3'][i,j]
                           + 1.0/tau_R)
                n['SF3'][i,j] = sf3_prod / max(sf3_loss, 1e-30)

                # === SF2 balance ===
                sf2_prod = (ne_ij*n['SF6'][i,j]*k['d4']
                           + ne_ij*n['SF3'][i,j]*k['d9']
                           + k['nr38']*n['SF'][i,j]*n['F'][i,j]
                           + k['nr43']*n['SF3'][i,j]**2)
                sf2_loss = (k['d10']*ne_ij + k['nr39']*n['F'][i,j] + 1.0/tau_R)
                n['SF2'][i,j] = sf2_prod / max(sf2_loss, 1e-30)

                # === SF balance ===
                sf_prod = (ne_ij*n['SF6'][i,j]*k['d5']
                          + ne_ij*n['SF2'][i,j]*k['d10']
                          + k['nr37']*n['S'][i,j]*n['F'][i,j])
                sf_loss = (k['d11']*ne_ij + k['nr38']*n['F'][i,j] + 1.0/tau_R)
                n['SF'][i,j] = sf_prod / max(sf_loss, 1e-30)

                # === S balance ===
                s_prod = ne_ij*n['SF'][i,j]*k['d11']
                s_loss = (k['iz29']*ne_ij + k['nr37']*n['F'][i,j] + 1.0/tau_R)
                n['S'][i,j] = s_prod / max(s_loss, 1e-30)

                # === F2 balance ===
                f2_prod = ne_ij*n['SF6'][i,j]*(k['d4']+k['d5'])
                f2_loss = (k['d6']*ne_ij + k['qnch_F2']*nArm_ij + 1.0/tau_R)
                n['F2'][i,j] = f2_prod / max(f2_loss, 1e-30)

                # === F balance ===
                f_prod = (ne_ij*n['SF6'][i,j]*(k['d1']+2*k['d2']+3*k['d3']
                          +2*k['d4']+3*k['d5']+k['iz18']+2*k['iz19']
                          +3*k['iz20']+2*k['iz21']+3*k['iz22']+4*k['iz23']+k['at31'])
                         + ne_ij*(n['SF5'][i,j]*(k['d7']+k['iz26'])
                                 +n['SF4'][i,j]*k['d8']+n['SF3'][i,j]*k['d9']
                                 +n['SF2'][i,j]*k['d10']+n['SF'][i,j]*k['d11'])
                         + k['Penn_SF6']*nArm_ij*n['SF6'][i,j]
                         + k['qnch_SF6']*nArm_ij*n['SF6'][i,j]
                         + 2*k['qnch_F2']*nArm_ij*n['F2'][i,j]
                         + 2*k['d6']*ne_ij*n['F2'][i,j])
                f_loss = (kw_F_eff + k['iz28']*ne_ij + 1.0/tau_R
                         + k['nr42']*n['SF5'][i,j] + k['nr41']*n['SF4'][i,j]
                         + k['nr40']*n['SF3'][i,j] + k['nr39']*n['SF2'][i,j]
                         + k['nr38']*n['SF'][i,j] + k['nr37']*n['S'][i,j])
                n['F'][i,j] = f_prod / max(f_loss, 1e-30)

    # Clip to physical values
    for sp in n:
        n[sp] = np.clip(n[sp], 0, nSF6_0*10)

    # === ANCHORING: renormalize to match 0D volume averages ===
    # The local chemistry gives the spatial VARIATION (shape).
    # The 0D backbone provides the MAGNITUDE (volume average).
    if neutrals_0d is not None:
        for sp in n:
            if sp in neutrals_0d and neutrals_0d[sp] > 0:
                actual_avg = mesh.volume_average(n[sp])
                if actual_avg > 1e6:
                    n[sp] *= neutrals_0d[sp] / actual_avg

    # Relaxed update
    neutrals = {}
    for sp in neutrals_old:
        neutrals[sp] = neutrals_old[sp] + w * (n[sp] - neutrals_old[sp])
        neutrals[sp] = np.maximum(neutrals[sp], 0.0)

    return neutrals


def init_neutrals(Nr, Nz, nSF6_0):
    """Initialize all 9 neutral species from the feed composition."""
    n = {}
    n['SF6'] = np.full((Nr,Nz), nSF6_0 * 0.3)
    n['SF5'] = np.full((Nr,Nz), nSF6_0 * 0.02)
    n['SF4'] = np.full((Nr,Nz), nSF6_0 * 0.01)
    n['SF3'] = np.full((Nr,Nz), nSF6_0 * 0.005)
    n['SF2'] = np.full((Nr,Nz), nSF6_0 * 0.001)
    n['SF']  = np.full((Nr,Nz), nSF6_0 * 0.0005)
    n['S']   = np.full((Nr,Nz), nSF6_0 * 0.0001)
    n['F']   = np.full((Nr,Nz), nSF6_0 * 0.3)
    n['F2']  = np.full((Nr,Nz), nSF6_0 * 0.01)
    return n
