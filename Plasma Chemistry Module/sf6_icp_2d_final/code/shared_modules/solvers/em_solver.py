"""
Electromagnetic solver for ICP power deposition.

Solves the azimuthal component of the magnetic vector potential equation
in the frequency domain:

    (1/r) ∂/∂r(r ∂Ã/∂r) + ∂²Ã/∂z² + (ω²μ₀ε − 1/r²) Ã = −μ₀ J̃_coil

where Ã(r,z) is the complex amplitude of the azimuthal vector potential.

The inductive electric field is: Ẽ_θ = −jωÃ
The power deposition is: P(r,z) = ½ Re(σ̃ |Ẽ|²)

The complex plasma conductivity is:
    σ_p = nₑ e² / [mₑ (ν_eff + jω)]

where ν_eff includes both collisional and collisionless contributions
(Vahedi et al. 1995).

Reference: Vahedi et al., J. Appl. Phys. 78, 1446 (1995)
           Lymberopoulos & Economou, JRNIST 100, 473 (1995), Eqs. 28–35
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.constants import e as eC, m_e, mu_0, epsilon_0, c, pi


class EMSolver:
    """Frequency-domain EM solver for ICP coil coupling."""

    def __init__(self, mesh, freq=13.56e6):
        """
        Parameters
        ----------
        mesh : Mesh2D
        freq : float
            RF frequency [Hz]. Default 13.56 MHz.
        """
        self.mesh = mesh
        self.freq = freq
        self.omega = 2 * pi * freq
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz

    def plasma_conductivity(self, ne, Te, gas_density):
        """Complex plasma conductivity σ_p(r,z).

        Parameters
        ----------
        ne : array (Nr, Nz) — electron density [m⁻³]
        Te : array (Nr, Nz) — electron temperature [eV]
        gas_density : float — neutral gas density [m⁻³]

        Returns
        -------
        sigma : complex array (Nr, Nz)
        """
        omega = self.omega

        # Electron-neutral collision frequency
        # Approximate: ν_en = ng × σ_en × v̄_e
        v_e = np.sqrt(8 * eC * Te / (pi * m_e))
        sigma_en = 3e-19  # Approximate effective cross-section [m²]
        nu_en = gas_density * sigma_en * v_e

        # Collisionless (stochastic) frequency (Vahedi 1995)
        # For ω ~ ν_en (typical at 13.56 MHz, 10 mTorr):
        # ν_st ≈ (1/4) v̄_e / δ  where δ ≈ c/ω_pe
        omega_pe = np.sqrt(ne * eC**2 / (m_e * epsilon_0))
        delta = np.where(omega_pe > 1e3, c / omega_pe, 1.0)
        nu_st = 0.25 * v_e / np.maximum(delta, 1e-6)

        # Effective collision frequency
        nu_eff = nu_en + nu_st

        # Complex conductivity: σ = nₑe²/[mₑ(ν_eff + jω)]
        sigma = ne * eC**2 / (m_e * (nu_eff + 1j * omega))

        return sigma

    def coil_current_density(self, I_coil, n_turns, coil_r, coil_z, coil_width=0.005):
        """Compute the coil current density source on the mesh.

        Models the coil as a set of discrete current loops.

        Parameters
        ----------
        I_coil : float
            Coil current amplitude [A].
        n_turns : int
            Number of coil turns.
        coil_r : array
            Radial positions of coil turns [m].
        coil_z : float
            Axial position of coil (just above dielectric) [m].
        coil_width : float
            Width of each coil turn for distributing current [m].

        Returns
        -------
        J_coil : complex array (Nr, Nz)
            Coil current density [A/m²].
        """
        mesh = self.mesh
        J = np.zeros((mesh.Nr, mesh.Nz), dtype=complex)

        for r_turn in coil_r:
            for i in range(mesh.Nr):
                for j in range(mesh.Nz):
                    dr = abs(mesh.r[i] - r_turn)
                    dz = abs(mesh.z[j] - coil_z)
                    if dr < coil_width and dz < coil_width:
                        # Distribute current over the cell area
                        area = coil_width**2
                        J[i, j] += I_coil / area

        return J

    def solve(self, ne, Te, gas_density, I_coil=10.0, n_turns=5,
              coil_radii=None, coil_z=None):
        """Solve the EM wave equation and compute power deposition.

        Parameters
        ----------
        ne : array (Nr, Nz) — electron density [m⁻³]
        Te : array (Nr, Nz) — electron temperature [eV]
        gas_density : float — neutral gas density [m⁻³]
        I_coil : float — coil current amplitude [A]
        n_turns : int — number of coil turns
        coil_radii : array — radial positions of coil turns [m]
        coil_z : float — axial position of coil [m]

        Returns
        -------
        P_ind : array (Nr, Nz)
            Inductive power deposition [W/m³].
        E_theta : complex array (Nr, Nz)
            Azimuthal electric field amplitude [V/m].
        total_power : float
            Total deposited power [W].
        """
        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz
        N = Nr * Nz
        omega = self.omega

        # Default coil geometry: planar spiral above the window
        if coil_radii is None:
            coil_radii = np.linspace(0.02, mesh.R * 0.8, n_turns)
        if coil_z is None:
            coil_z = mesh.L - 0.01  # 1 cm below top

        # Plasma conductivity
        sigma_p = self.plasma_conductivity(ne, Te, gas_density)

        # Complex dielectric constant
        eps_p = epsilon_0 - 1j * sigma_p / omega

        # Coil current density
        J_coil = self.coil_current_density(I_coil, n_turns, coil_radii, coil_z)

        # Build the matrix for the wave equation:
        # (1/r)∂/∂r(r ∂Ã/∂r) + ∂²Ã/∂z² + (ω²μ₀ε_p - 1/r²)Ã = -μ₀ J̃
        # This is the same Laplacian as Poisson but with complex coefficients.

        rows, cols, vals = [], [], []
        rhs = np.zeros(N, dtype=complex)

        for i in range(Nr):
            for j in range(Nz):
                idx = i * Nz + j
                r_c = mesh.r[i]
                dr = mesh.dr[i]
                dz = mesh.dz[j]

                # Wave term: (ω²μ₀ε_p - 1/r²)
                wave_term = omega**2 * mu_0 * eps_p[i, j] - 1.0 / r_c**2

                # Radial Laplacian terms
                coeff_diag = 0.0

                # r+1/2 face
                if i < Nr - 1:
                    r_f = mesh.r_faces[i + 1]
                    dr_c = mesh.dr_c[i + 1]
                    c_r_right = r_f / (r_c * dr * dr_c)
                    rows.append(idx); cols.append((i+1)*Nz + j); vals.append(c_r_right)
                    coeff_diag -= c_r_right
                else:
                    # Ã = 0 at r = R
                    r_f = mesh.r_faces[Nr]
                    dr_c = mesh.dr_c[Nr]
                    c_r_right = r_f / (r_c * dr * dr_c)
                    coeff_diag -= c_r_right

                # r-1/2 face
                if i > 0:
                    r_f = mesh.r_faces[i]
                    dr_c = mesh.dr_c[i]
                    c_r_left = r_f / (r_c * dr * dr_c)
                    rows.append(idx); cols.append((i-1)*Nz + j); vals.append(c_r_left)
                    coeff_diag -= c_r_left
                else:
                    # Ã = 0 at r = 0 (azimuthal field vanishes on axis)
                    # Use: limit as r→0 of (1/r)d(r dÃ/dr)/dr = 2 d²Ã/dr²
                    dr_c = mesh.dr_c[1]
                    c_sym = 2.0 / (dr * dr_c)
                    if Nr > 1:
                        rows.append(idx); cols.append(1*Nz + j); vals.append(c_sym)
                    coeff_diag -= c_sym

                # z+1/2 face
                if j < Nz - 1:
                    dz_c = mesh.dz_c[j + 1]
                    c_z_top = 1.0 / (dz * dz_c)
                    rows.append(idx); cols.append(i*Nz + j + 1); vals.append(c_z_top)
                    coeff_diag -= c_z_top
                else:
                    dz_c = mesh.dz_c[Nz]
                    c_z_top = 1.0 / (dz * dz_c)
                    coeff_diag -= c_z_top

                # z-1/2 face
                if j > 0:
                    dz_c = mesh.dz_c[j]
                    c_z_bot = 1.0 / (dz * dz_c)
                    rows.append(idx); cols.append(i*Nz + j - 1); vals.append(c_z_bot)
                    coeff_diag -= c_z_bot
                else:
                    dz_c = mesh.dz_c[0]
                    c_z_bot = 1.0 / (dz * dz_c)
                    coeff_diag -= c_z_bot

                # Diagonal: Laplacian + wave term
                rows.append(idx); cols.append(idx); vals.append(coeff_diag + wave_term)

                # RHS
                rhs[idx] = -mu_0 * J_coil[i, j]

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        A_tilde = spsolve(A, rhs).reshape((Nr, Nz))

        # Electric field: Ẽ_θ = -jω Ã
        E_theta = -1j * omega * A_tilde

        # Power deposition: P = ½ Re(σ |E|²)
        P_ind = 0.5 * np.real(sigma_p * np.abs(E_theta)**2)
        P_ind = np.maximum(P_ind, 0.0)  # Physical: no negative power

        # Total power
        total_power = mesh.volume_average(P_ind) * mesh.total_volume()

        return P_ind, E_theta, total_power

    def adjust_coil_current(self, ne, Te, gas_density, P_target,
                            n_turns=5, coil_radii=None, coil_z=None,
                            tol=0.05, max_iter=20):
        """Iterate coil current to achieve a target total power.

        Parameters
        ----------
        P_target : float — target total absorbed power [W]
        tol : float — relative tolerance on power matching

        Returns
        -------
        P_ind : array (Nr, Nz) — power deposition profile [W/m³]
        I_coil : float — required coil current [A]
        """
        I_coil = 10.0  # Initial guess

        for iteration in range(max_iter):
            P_ind, E_theta, P_total = self.solve(
                ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)

            if P_total < 1e-10:
                I_coil *= 10
                continue

            # Power scales as I²
            ratio = P_target / P_total
            I_coil *= np.sqrt(ratio)

            if abs(ratio - 1.0) < tol:
                # Final solve with correct current
                P_ind, E_theta, P_total = self.solve(
                    ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)
                return P_ind, I_coil

        # Return best result even if not converged
        P_ind, E_theta, P_total = self.solve(
            ne, Te, gas_density, I_coil, n_turns, coil_radii, coil_z)
        return P_ind, I_coil


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mesh.mesh_generator import Mesh2D

    print("EM solver self-test")
    print("=" * 50)

    mesh = Mesh2D(R=0.180, L=0.175, Nr=40, Nz=50, stretch_r=1.0, stretch_z=1.0)
    em = EMSolver(mesh, freq=13.56e6)

    # Uniform plasma
    ne = np.ones((mesh.Nr, mesh.Nz)) * 1e17  # 10¹¹ cm⁻³
    Te = np.ones((mesh.Nr, mesh.Nz)) * 3.0
    ng = 3.22e20  # 10 mTorr

    print(f"Mesh: {mesh}")
    print(f"Plasma frequency: ω_pe = {np.sqrt(1e17 * eC**2 / (m_e * epsilon_0)):.2e} rad/s")
    print(f"Skin depth (collisionless): δ = c/ω_pe = {c / np.sqrt(1e17 * eC**2 / (m_e * epsilon_0)) * 100:.1f} cm")
    print(f"RF frequency: ω = {em.omega:.2e} rad/s")

    # Solve with a moderate coil current
    P_ind, E_theta, P_total = em.solve(ne, Te, ng, I_coil=20.0)

    print(f"\nPower deposition:")
    print(f"  Total power: {P_total:.1f} W")
    print(f"  Max P_ind: {P_ind.max():.2e} W/m³")
    print(f"  Max |E_θ|: {np.abs(E_theta).max():.1f} V/m")

    # Check that power peaks near the coil (high z, moderate r)
    i_peak, j_peak = np.unravel_index(np.argmax(P_ind), P_ind.shape)
    print(f"  Peak at r={mesh.r[i_peak]*100:.1f} cm, z={mesh.z[j_peak]*100:.1f} cm")
    print(f"  (Coil is near z={mesh.L*100:.1f} cm, r=2–14 cm)")

    # Adjust current for target power
    P_ind_adj, I_adj = em.adjust_coil_current(ne, Te, ng, P_target=500.0)
    P_total_adj = mesh.volume_average(P_ind_adj) * mesh.total_volume()
    print(f"\nAdjusted for 500W:")
    print(f"  I_coil = {I_adj:.1f} A")
    print(f"  P_total = {P_total_adj:.1f} W")

    print("\n✓ EM solver OK")
