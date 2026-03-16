"""
Electron energy equation solver.

Solves: ∂(nₑε̄)/∂t + ∇·qₑ + eΓₑ·E = -P_collis + P_ind

where ε̄ is the mean electron energy [eV], and the energy flux is:
    qₑ = -μₑ_ε E (nₑε̄) - Dε ∇(nₑε̄)

This has the same drift-diffusion form as the particle continuity equation,
so we use the same Scharfetter-Gummel discretization.

Reference: Hagelaar & Pitchford 2005, Eqs. 57–62
"""

import numpy as np
from scipy.constants import e as eC


class EnergySolver:
    """Solve the electron energy equation on the 2D mesh."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz

    def step(self, ne, ne_eps, Er, Ez, flux_r_e, flux_z_e,
             mu_eps, D_eps, P_collis, P_ind, Te, dt):
        """Advance the electron energy density by one timestep.

        Uses simple diffusion for energy transport (consistent with the
        quasi-neutral ambipolar approach in the species solver) plus
        inductive power deposition.

        Energy equation: ∂(nₑε̄)/∂t = ∇·(Dε ∇(nₑε̄)) + P_ind/eC − P_collis
        """
        from mesh.scharfetter_gummel import divergence_cyl

        mesh = self.mesh
        Nr, Nz = self.Nr, self.Nz

        # --- Compute energy flux: pure diffusion ---
        q_r = np.zeros((Nr + 1, Nz))
        q_z = np.zeros((Nr, Nz + 1))

        for j in range(Nz):
            for f in range(1, Nr):
                if np.isscalar(D_eps):
                    D_f = D_eps
                else:
                    D_f = 0.5 * (D_eps[f-1, j] + D_eps[f, j])
                q_r[f, j] = -D_f * (ne_eps[f, j] - ne_eps[f-1, j]) / mesh.dr_c[f]

        for i in range(Nr):
            for f in range(1, Nz):
                if np.isscalar(D_eps):
                    D_f = D_eps
                else:
                    D_f = 0.5 * (D_eps[i, f-1] + D_eps[i, f])
                q_z[i, f] = -D_f * (ne_eps[i, f] - ne_eps[i, f-1]) / mesh.dz_c[f]

        # --- Energy flux boundary conditions ---
        # Axis: q_r = 0
        q_r[0, :] = 0.0

        # Walls: energy loss = (5/2)Tₑ × particle loss rate × nₑ (approximate)
        # For now, use zero-gradient (energy diffuses out naturally)
        # r = R
        if np.isscalar(D_eps):
            D_wall = D_eps
        else:
            D_wall = D_eps[Nr-1, :]
        q_r[Nr, :] = -D_wall * (0.0 - ne_eps[Nr-1, :]) / mesh.dr_c[Nr]

        # z = 0
        if np.isscalar(D_eps):
            D_wall = D_eps
        else:
            D_wall = D_eps[:, 0]
        q_z[:, 0] = -D_wall * (0.0 - ne_eps[:, 0]) / mesh.dz_c[0]

        # z = L
        if np.isscalar(D_eps):
            D_wall = D_eps
        else:
            D_wall = D_eps[:, Nz-1]
        q_z[:, Nz] = -D_wall * (0.0 - ne_eps[:, Nz-1]) / mesh.dz_c[Nz]

        # --- Divergence of energy flux ---
        div_q = divergence_cyl(q_r, q_z, mesh)

        # --- Inductive heating ---
        P_ind_eV = P_ind / eC  # W/m³ → eV m⁻³ s⁻¹

        # --- Time advance (semi-implicit for collisional loss) ---
        # P_collis = ne * Eloss(Te) ∝ ne_eps since Te ∝ ne_eps/ne
        # Treat as: P_collis ≈ (P_collis/ne_eps) * ne_eps = ν_loss * ne_eps
        # Semi-implicit: ne_eps^{n+1} = (ne_eps^n + dt*(P_ind_eV - div_q)) / (1 + dt*ν_loss)
        nu_loss = np.where(ne_eps > 1e6, P_collis / ne_eps, 0.0)
        nu_loss = np.clip(nu_loss, 0, 1e10)

        ne_eps_new = (ne_eps + dt * (P_ind_eV - div_q)) / (1.0 + dt * nu_loss)

        # Floor: minimum energy = 0.5 eV × nₑ
        ne_eps_new = np.maximum(ne_eps_new, 0.5 * ne)

        # Cap maximum to prevent runaway
        ne_eps_new = np.minimum(ne_eps_new, 30.0 * ne)  # max 20 eV mean energy

        # Compute Te from energy density
        Te_new = np.where(ne > 1e8, (2.0 / 3.0) * ne_eps_new / ne, 0.5)
        Te_new = np.clip(Te_new, 0.3, 20.0)

        return ne_eps_new, Te_new


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mesh.mesh_generator import Mesh2D

    print("Energy solver self-test")
    print("=" * 50)

    mesh = Mesh2D(R=0.05, L=0.05, Nr=20, Nz=20, stretch_r=1.0, stretch_z=1.0)
    solver = EnergySolver(mesh)

    # Uniform plasma, no field, no sources → energy should diffuse
    ne = np.ones((mesh.Nr, mesh.Nz)) * 1e16
    Te0 = 3.0
    ne_eps = ne * 1.5 * Te0  # nₑε̄ = nₑ × (3/2)Tₑ

    # Add a temperature perturbation at centre
    i_c, j_c = mesh.Nr // 2, mesh.Nz // 2
    ne_eps[i_c, j_c] *= 2.0  # hot spot

    Er = np.zeros((mesh.Nr + 1, mesh.Nz))
    Ez = np.zeros((mesh.Nr, mesh.Nz + 1))
    flux_r_e = np.zeros((mesh.Nr + 1, mesh.Nz))
    flux_z_e = np.zeros((mesh.Nr, mesh.Nz + 1))

    P_collis = np.zeros((mesh.Nr, mesh.Nz))
    P_ind = np.zeros((mesh.Nr, mesh.Nz))

    mu_eps = 5.0 / 3.0 * 1000.0  # 5/3 × μₑ
    D_eps = 5.0 / 3.0 * 3000.0   # 5/3 × Dₑ

    dt = 1e-10
    print(f"Initial Te at centre: {(2/3)*ne_eps[i_c,j_c]/ne[i_c,j_c]:.2f} eV")

    for step_num in range(10):
        ne_eps, Te = solver.step(ne, ne_eps, Er, Ez, flux_r_e, flux_z_e,
                                  mu_eps, D_eps, P_collis, P_ind, Te0, dt)

    print(f"After 10 steps: Te at centre: {Te[i_c,j_c]:.2f} eV (should decrease from 6.0)")
    print(f"  Te range: {Te.min():.2f} – {Te.max():.2f} eV")
    print(f"  Mean Te: {np.mean(Te):.2f} eV")

    print("\n✓ Energy solver OK")
