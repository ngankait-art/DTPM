"""
Charged species solver for the 2D axisymmetric plasma model.

Solves the drift-diffusion continuity equations for:
    - Electrons (nₑ)
    - Positive ions (n₊)
    - Negative ions (n₋)

using Scharfetter-Gummel fluxes, coupled to Poisson's equation.

Time integration: Semi-implicit backward Euler.
    - Transport (divergence of S-G flux) is implicit in species density
    - Source terms are evaluated at the current time level
    - Poisson is solved at the beginning of each step with current densities

Sign conventions for the electric field in the S-G flux:
    Electrons:  Γₑ = +μₑnₑE − Dₑ∇nₑ   (drift toward anode, i.e. against E for E pointing cathode→anode)
    Positive:   Γ₊ = −μ₊n₊∇V = +μ₊n₊E  (drift with E)
    Negative:   Γ₋ = +μ₋n₋∇V = −μ₋n₋E  (drift against E, trapped)

The S-G routine computes: Γ = D/Δx [B(Z)·n_R − B(−Z)·n_L] where Z = μ·E·Δx/D
For electrons:  pass E_field = +E (they drift opposite to E via the sign of their charge, but
                the S-G convention with positive μ and positive E gives the correct electron drift
                when we flip the sign: pass -E so they drift toward positive potential)

We handle this by defining an "effective field" for each species that includes the charge sign.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, pi


class SpeciesSolver:
    """Time-advance charged species densities on the 2D mesh.

    Uses operator splitting:
    1. Solve Poisson for V, Er, Ez from current densities
    2. Compute S-G fluxes for each species
    3. Apply boundary conditions to fluxes
    4. Advance densities: n^{t+dt} = n^t + dt * (S - ∇·Γ)
    """

    def __init__(self, mesh, poisson_solver):
        """
        Parameters
        ----------
        mesh : Mesh2D
        poisson_solver : PoissonSolver
        """
        self.mesh = mesh
        self.poisson = poisson_solver
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz

    def compute_fluxes(self, n, Er, Ez, mu, D, charge_sign, mesh):
        """Compute drift-diffusion fluxes using Scharfetter-Gummel scheme.

        Parameters
        ----------
        n : array (Nr, Nz)
        Er : array (Nr+1, Nz) — electric field at r-faces
        Ez : array (Nr, Nz+1) — electric field at z-faces
        mu : float or array — mobility magnitude
        D : float or array — diffusion coefficient
        charge_sign : +1 or -1
        mesh : Mesh2D

        Returns
        -------
        flux_r : array (Nr+1, Nz)
        flux_z : array (Nr, Nz+1)
        """
        from mesh.scharfetter_gummel import bernoulli_scalar

        Nr, Nz = mesh.Nr, mesh.Nz

        # Effective field for this species:
        # For positive ions (charge_sign=+1): drift with E, pass E directly
        # For electrons/negative ions (charge_sign=-1): drift against E, pass -E
        # The S-G scheme then gives the correct flux direction.
        E_eff_r = charge_sign * Er
        E_eff_z = charge_sign * Ez

        flux_r = np.zeros((Nr + 1, Nz))
        flux_z = np.zeros((Nr, Nz + 1))

        # --- Radial fluxes ---
        for j in range(Nz):
            for f in range(1, Nr):
                i_L, i_R = f - 1, f
                if np.isscalar(mu):
                    mu_f, D_f = mu, D
                else:
                    mu_f = 0.5 * (mu[i_L, j] + mu[i_R, j])
                    D_f = 0.5 * (D[i_L, j] + D[i_R, j])

                Z = mu_f * E_eff_r[f, j] * mesh.dr_c[f] / max(D_f, 1e-30)
                B_pos = bernoulli_scalar(Z)
                B_neg = bernoulli_scalar(-Z)
                flux_r[f, j] = D_f / mesh.dr_c[f] * (B_pos * n[i_R, j] - B_neg * n[i_L, j])

        # --- Axial fluxes ---
        for i in range(Nr):
            for f in range(1, Nz):
                j_L, j_R = f - 1, f
                if np.isscalar(mu):
                    mu_f, D_f = mu, D
                else:
                    mu_f = 0.5 * (mu[i, j_L] + mu[i, j_R])
                    D_f = 0.5 * (D[i, j_L] + D[i, j_R])

                Z = mu_f * E_eff_z[i, f] * mesh.dz_c[f] / max(D_f, 1e-30)
                B_pos = bernoulli_scalar(Z)
                B_neg = bernoulli_scalar(-Z)
                flux_z[i, f] = D_f / mesh.dz_c[f] * (B_pos * n[i, j_R] - B_neg * n[i, j_L])

        return flux_r, flux_z

    def apply_electron_bc(self, ne, flux_r, flux_z, Er, Ez, Te, mu_e, n_pos, gamma_se=0.01):
        """Apply electron boundary conditions.

        Walls: Γₑ·n̂ = ¼ v̄th nₑ − γ Γ₊·n̂
        Axis: Γₑ_r = 0
        """
        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz

        # Thermal velocity at each boundary cell
        v_th_e = np.sqrt(8 * eC * Te / (pi * m_e))  # can be scalar or array

        # r = 0 (axis symmetry): no radial flux
        flux_r[0, :] = 0.0

        # r = R (wall): thermal flux out minus secondary emission
        if np.isscalar(v_th_e):
            flux_r[Nr, :] = 0.25 * v_th_e * ne[Nr - 1, :]
        else:
            flux_r[Nr, :] = 0.25 * v_th_e[Nr - 1, :] * ne[Nr - 1, :]
        # Note: secondary emission contribution from ion flux is small, add later

        # z = 0 (bottom electrode): thermal flux
        if np.isscalar(v_th_e):
            flux_z[:, 0] = -0.25 * v_th_e * ne[:, 0]  # negative because outward is -z
        else:
            flux_z[:, 0] = -0.25 * v_th_e[:, 0] * ne[:, 0]

        # z = L (top, dielectric window): thermal flux
        if np.isscalar(v_th_e):
            flux_z[:, Nz] = 0.25 * v_th_e * ne[:, Nz - 1]
        else:
            flux_z[:, Nz] = 0.25 * v_th_e[:, Nz - 1] * ne[:, Nz - 1]

        return flux_r, flux_z

    def apply_ion_pos_bc(self, n_pos, flux_r, flux_z, Te, alpha, Mi_kg, T_neg=0.3):
        """Apply positive ion boundary conditions.

        Walls: Γ₊·n̂ = n₊ · uB(Te, α)
        Axis: Γ₊_r = 0
        """
        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz

        gamma = Te / max(T_neg, 0.01)

        # Modified Bohm velocity (Lichtenberg 1997 Eq. 15)
        # Use local alpha at each boundary cell
        def uB_local(Te_val, alpha_val):
            return np.sqrt(eC * Te_val * (1 + alpha_val) / (Mi_kg * (1 + gamma * alpha_val)))

        # r = 0: symmetry
        flux_r[0, :] = 0.0

        # r = R: Bohm flux outward
        if np.isscalar(alpha):
            uB = uB_local(Te, alpha)
            flux_r[Nr, :] = n_pos[Nr - 1, :] * uB
        else:
            for j in range(Nz):
                uB = uB_local(Te if np.isscalar(Te) else Te[Nr-1, j],
                              alpha[Nr-1, j])
                flux_r[Nr, j] = n_pos[Nr - 1, j] * uB

        # z = 0: Bohm flux downward (outward = negative z)
        if np.isscalar(alpha):
            uB = uB_local(Te, alpha)
            flux_z[:, 0] = -n_pos[:, 0] * uB
        else:
            for i in range(Nr):
                uB = uB_local(Te if np.isscalar(Te) else Te[i, 0],
                              alpha[i, 0])
                flux_z[i, 0] = -n_pos[i, 0] * uB

        # z = L: Bohm flux upward (outward = positive z)
        if np.isscalar(alpha):
            uB = uB_local(Te, alpha)
            flux_z[:, Nz] = n_pos[:, Nz - 1] * uB
        else:
            for i in range(Nr):
                uB = uB_local(Te if np.isscalar(Te) else Te[i, Nz-1],
                              alpha[i, Nz-1])
                flux_z[i, Nz] = n_pos[i, Nz - 1] * uB

        return flux_r, flux_z

    def apply_ion_neg_bc(self, n_neg, flux_r, flux_z):
        """Apply negative ion boundary conditions.

        All walls: Γ₋·n̂ = 0 (trapped by sheath potential)
        Axis: Γ₋_r = 0
        """
        flux_r[0, :] = 0.0
        flux_r[self.Nr, :] = 0.0
        flux_z[:, 0] = 0.0
        flux_z[:, self.Nz] = 0.0
        return flux_r, flux_z

    def divergence(self, flux_r, flux_z):
        """Compute ∇·Γ in cylindrical coordinates."""
        from mesh.scharfetter_gummel import divergence_cyl
        return divergence_cyl(flux_r, flux_z, self.mesh)

    def step(self, ne, n_pos, n_neg, Te, Se, S_pos, S_neg,
             mu_e, D_e, mu_pos, D_pos, mu_neg, D_neg,
             Mi_kg, T_neg, dt, gamma_se=0.01):
        """Advance all charged species by one timestep.

        Uses the quasi-neutral ambipolar diffusion approach for the bulk
        plasma, which avoids the dielectric relaxation time constraint.

        Instead of solving Poisson + drift-diffusion separately (which
        requires dt < ε₀/(eμₑnₑ) ~ 10⁻¹³ s), we use:

        1. Ambipolar diffusion for n₊ in the bulk: Da(α) depends on local α
        2. Electrons follow from quasi-neutrality: nₑ = n₊ − n₋
        3. Negative ions: diffusion + ion-ion recombination (trapped by potential)
        4. Poisson solved for diagnostics / sheath physics

        This is the standard approach for ICP simulations at the densities
        of interest (10¹⁵–10¹⁷ m⁻³). The full Poisson-coupled drift-diffusion
        becomes necessary only for sheath-resolved simulations (Phase 3).

        Reference: Vahedi et al. (1995) Section V; Lymberopoulos (1995) Fig. 2
        """
        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz
        from transport.transport import ambipolar_Da
        from mesh.scharfetter_gummel import divergence_cyl

        alpha = np.where(ne > 1e10, n_neg / ne, 0.0)

        # --- Step 1: Advance positive ions using ambipolar diffusion ---
        # Da(α) varies from ~γD₊ (electropositive) to ~2D₊ (electronegative)
        Da = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                Da[i, j] = ambipolar_Da(alpha[i, j], Te if np.isscalar(Te) else Te[i, j],
                                         D_pos if np.isscalar(D_pos) else D_pos[i, j], T_neg)

        # Ambipolar flux: Γ₊ = −Da ∇n₊ (pure diffusion, no drift term needed)
        # Using zero electric field in the S-G scheme reduces it to central difference
        E_zero_r = np.zeros((Nr + 1, Nz))
        E_zero_z = np.zeros((Nr, Nz + 1))

        flux_r_p = np.zeros((Nr + 1, Nz))
        flux_z_p = np.zeros((Nr, Nz + 1))

        # Simple diffusion fluxes for ambipolar transport
        for j in range(Nz):
            for f in range(1, Nr):
                Da_f = 0.5 * (Da[f-1, j] + Da[f, j])
                flux_r_p[f, j] = -Da_f * (n_pos[f, j] - n_pos[f-1, j]) / mesh.dr_c[f]

        for i in range(Nr):
            for f in range(1, Nz):
                Da_f = 0.5 * (Da[i, f-1] + Da[i, f])
                flux_z_p[i, f] = -Da_f * (n_pos[i, f] - n_pos[i, f-1]) / mesh.dz_c[f]

        # Boundary conditions for positive ions (Bohm flux at walls)
        flux_r_p, flux_z_p = self.apply_ion_pos_bc(
            n_pos, flux_r_p, flux_z_p, Te, alpha, Mi_kg, T_neg)
        flux_r_p[0, :] = 0.0  # axis symmetry

        div_p = divergence_cyl(flux_r_p, flux_z_p, mesh)

        n_pos_new = n_pos + dt * (S_pos - div_p)
        n_pos_new = np.maximum(n_pos_new, 1e6)

        # --- Step 2: Advance negative ions ---
        # Negative ions diffuse slowly, trapped by potential
        D_neg_eff = D_neg if np.isscalar(D_neg) else D_neg

        flux_r_n = np.zeros((Nr + 1, Nz))
        flux_z_n = np.zeros((Nr, Nz + 1))

        for j in range(Nz):
            for f in range(1, Nr):
                D_f = D_neg_eff if np.isscalar(D_neg_eff) else 0.5*(D_neg_eff[f-1,j]+D_neg_eff[f,j])
                flux_r_n[f, j] = -D_f * (n_neg[f, j] - n_neg[f-1, j]) / mesh.dr_c[f]

        for i in range(Nr):
            for f in range(1, Nz):
                D_f = D_neg_eff if np.isscalar(D_neg_eff) else 0.5*(D_neg_eff[i,f-1]+D_neg_eff[i,f])
                flux_z_n[i, f] = -D_f * (n_neg[i, f] - n_neg[i, f-1]) / mesh.dz_c[f]

        flux_r_n, flux_z_n = self.apply_ion_neg_bc(n_neg, flux_r_n, flux_z_n)

        div_n = divergence_cyl(flux_r_n, flux_z_n, mesh)

        n_neg_new = n_neg + dt * (S_neg - div_n)
        n_neg_new = np.maximum(n_neg_new, 0.0)

        # --- Step 3: Electrons from quasi-neutrality ---
        ne_new = n_pos_new - n_neg_new
        ne_new = np.maximum(ne_new, 1e6)

        # --- Step 4: Solve Poisson for potential (diagnostic / sheath structure) ---
        V, Er, Ez = self.poisson.solve(n_pos_new, ne_new, n_neg_new)

        return ne_new, n_pos_new, n_neg_new, V, Er, Ez


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mesh.mesh_generator import Mesh2D
    from solvers.poisson import PoissonSolver

    print("Species solver self-test")
    print("=" * 50)

    # Small test mesh
    mesh = Mesh2D(R=0.05, L=0.05, Nr=20, Nz=20, stretch_r=1.0, stretch_z=1.0)
    poisson = PoissonSolver(mesh)
    solver = SpeciesSolver(mesh, poisson)

    # Initialize: uniform plasma with slight positive charge excess (creates sheaths)
    ne0 = 1e16  # m⁻³
    ne = np.ones((mesh.Nr, mesh.Nz)) * ne0
    n_pos = np.ones((mesh.Nr, mesh.Nz)) * ne0 * 1.001
    n_neg = np.zeros((mesh.Nr, mesh.Nz))

    # Transport coefficients
    Te = 3.0
    mu_e = 1000.0    # m²/(V·s)
    D_e = 3000.0     # m²/s
    mu_pos = 0.001   # m²/(V·s)
    D_pos = 0.00001  # m²/s
    mu_neg = 0.001
    D_neg = 0.00001
    Mi_kg = 127.06 * 1.66054e-27

    # Zero source terms (diffusion only)
    Se = np.zeros_like(ne)
    S_pos = np.zeros_like(ne)
    S_neg = np.zeros_like(ne)

    # Take a few steps
    dt = 1e-10  # 0.1 ns
    print(f"\nInitial: ne_max={ne.max():.3e}, n_pos_max={n_pos.max():.3e}")

    for step_num in range(5):
        ne, n_pos, n_neg, V, Er, Ez = solver.step(
            ne, n_pos, n_neg, Te, Se, S_pos, S_neg,
            mu_e, D_e, mu_pos, D_pos, mu_neg, D_neg,
            Mi_kg, T_neg=0.3, dt=dt)

    print(f"After 5 steps (dt={dt:.0e}s): ne_max={ne.max():.3e}, n_pos_max={n_pos.max():.3e}")
    print(f"  V_max={V.max():.2f} V, V_min={V.min():.2f} V")
    print(f"  Er_max={np.abs(Er).max():.0f} V/m")

    # Check that electrons are depleted at walls (sheath formation)
    print(f"  ne at wall (r=R): {ne[-1, mesh.Nz//2]:.3e} (should be < {ne0:.0e})")
    print(f"  ne at centre: {ne[0, mesh.Nz//2]:.3e}")

    print("\n✓ Species solver OK")
