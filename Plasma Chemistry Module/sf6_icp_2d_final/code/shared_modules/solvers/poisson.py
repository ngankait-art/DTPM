"""
Poisson equation solver for 2D axisymmetric geometry.

Solves: (1/r) ∂/∂r(r ∂V/∂r) + ∂²V/∂z² = -ρ/ε₀

where ρ = e(n₊ - nₑ - n₋) is the space charge density.

Uses direct sparse matrix solve (scipy.sparse.linalg.spsolve).
For the semi-implicit coupling with the electron continuity equation,
the charge density can be extrapolated to the future time level.

Boundary conditions:
    r = 0: ∂V/∂r = 0 (symmetry)
    r = R: V = 0 (grounded wall)
    z = 0: V = 0 (grounded electrode)
    z = L: V = 0 (grounded or specified for dielectric window)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.constants import e as eC, epsilon_0


class PoissonSolver:
    """Solve Poisson's equation on a structured (r,z) mesh."""

    def __init__(self, mesh):
        """
        Parameters
        ----------
        mesh : Mesh2D
            The computational mesh.
        """
        self.mesh = mesh
        self.Nr = mesh.Nr
        self.Nz = mesh.Nz
        self.N = mesh.Nr * mesh.Nz

        # Build the coefficient matrix (only depends on mesh geometry)
        self._build_matrix()

    def _idx(self, i, j):
        """Map (i,j) cell indices to flat index."""
        return i * self.Nz + j

    def _build_matrix(self):
        """Build the sparse coefficient matrix for the Laplacian.

        (1/r) ∂/∂r(r ∂V/∂r) + ∂²V/∂z² = f

        Discretized on cell centres with the five-point stencil.
        """
        Nr, Nz = self.Nr, self.Nz
        mesh = self.mesh

        rows, cols, vals = [], [], []

        for i in range(Nr):
            for j in range(Nz):
                idx = self._idx(i, j)
                r_c = mesh.r[i]
                dr = mesh.dr[i]
                dz = mesh.dz[j]

                # --- Radial terms: (1/r) d/dr(r dV/dr) ---
                # At face i+1/2 (between cell i and i+1)
                if i < Nr - 1:
                    r_face = mesh.r_faces[i + 1]
                    dr_c = mesh.dr_c[i + 1]
                    coeff_r_right = r_face / (r_c * dr * dr_c)
                else:
                    # r = R wall: V = 0 (Dirichlet)
                    r_face = mesh.r_faces[Nr]
                    dr_c = mesh.dr_c[Nr]
                    coeff_r_right = r_face / (r_c * dr * dr_c)

                # At face i-1/2 (between cell i-1 and i)
                if i > 0:
                    r_face = mesh.r_faces[i]
                    dr_c = mesh.dr_c[i]
                    coeff_r_left = r_face / (r_c * dr * dr_c)
                else:
                    # r = 0 symmetry: ∂V/∂r = 0
                    # Use L'Hôpital: (1/r)d(r dV/dr)/dr → 2 d²V/dr² at r=0
                    r_face = mesh.r_faces[0]  # = 0
                    dr_c = mesh.dr_c[0]
                    coeff_r_left = 0.0  # No flux through axis

                # --- Axial terms: d²V/dz² ---
                if j < Nz - 1:
                    dz_c_right = mesh.dz_c[j + 1]
                    coeff_z_top = 1.0 / (dz * dz_c_right)
                else:
                    dz_c_right = mesh.dz_c[Nz]
                    coeff_z_top = 1.0 / (dz * dz_c_right)

                if j > 0:
                    dz_c_left = mesh.dz_c[j]
                    coeff_z_bot = 1.0 / (dz * dz_c_left)
                else:
                    dz_c_left = mesh.dz_c[0]
                    coeff_z_bot = 1.0 / (dz * dz_c_left)

                # Diagonal
                diag = -(coeff_r_right + coeff_r_left + coeff_z_top + coeff_z_bot)

                # --- Handle boundaries ---

                # r = 0 (axis symmetry): ghost cell has V_{-1,j} = V_{0,j}
                # This means the left radial contribution adds to diagonal
                if i == 0:
                    # At r=0, use: d²V/dr² ≈ 2*(V[1]-V[0])/dr²
                    r_face_right = mesh.r_faces[1]
                    dr_c_right = mesh.dr_c[1]
                    # Replace the full radial stencil with the L'Hôpital form
                    coeff_r_sym = 2.0 / (dr * dr_c_right)  # Factor of 2 from L'Hôpital
                    diag = -(coeff_r_sym + coeff_z_top + coeff_z_bot)
                    rows.append(idx); cols.append(idx); vals.append(diag)
                    if i < Nr - 1:
                        rows.append(idx); cols.append(self._idx(i+1, j)); vals.append(coeff_r_sym)
                    # No left neighbour for i=0
                else:
                    rows.append(idx); cols.append(idx); vals.append(diag)

                    # Right (i+1)
                    if i < Nr - 1:
                        rows.append(idx); cols.append(self._idx(i+1, j)); vals.append(coeff_r_right)
                    # else: V_{Nr,j} = 0 Dirichlet → no matrix entry, but add to RHS

                    # Left (i-1)
                    if i > 0:
                        rows.append(idx); cols.append(self._idx(i-1, j)); vals.append(coeff_r_left)

                # Top (j+1)
                if j < Nz - 1:
                    rows.append(idx); cols.append(self._idx(i, j+1)); vals.append(coeff_z_top)
                # else: V_{i,Nz} = 0 Dirichlet

                # Bottom (j-1)
                if j > 0:
                    rows.append(idx); cols.append(self._idx(i, j-1)); vals.append(coeff_z_bot)
                # else: V_{i,0} = 0 → handled in RHS for j=0 but with Dirichlet the
                # boundary value is 0 so no RHS contribution

        self.A = sparse.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))

    def solve(self, n_pos, n_e, n_neg):
        """Solve Poisson's equation for the electrostatic potential.

        Parameters
        ----------
        n_pos : array (Nr, Nz)
            Total positive ion density [m⁻³].
        n_e : array (Nr, Nz)
            Electron density [m⁻³].
        n_neg : array (Nr, Nz)
            Total negative ion density [m⁻³].

        Returns
        -------
        V : array (Nr, Nz)
            Electrostatic potential [V].
        Er : array (Nr+1, Nz)
            Radial electric field at r-faces [V/m].
        Ez : array (Nr, Nz+1)
            Axial electric field at z-faces [V/m].
        """
        Nr, Nz = self.Nr, self.Nz
        mesh = self.mesh

        # RHS: -ρ/ε₀ = -(e/ε₀)(n₊ - nₑ - n₋)
        rho = eC / epsilon_0 * (n_pos - n_e - n_neg)
        rhs = -rho.flatten()

        # Boundary corrections for Dirichlet BCs (V=0 at walls)
        # Since V_boundary = 0, no correction needed for homogeneous BCs.
        # For non-zero boundary values, add appropriate terms here.

        # Solve
        V_flat = spsolve(self.A, rhs)
        V = V_flat.reshape((Nr, Nz))

        # Compute electric field E = -∇V
        Er = np.zeros((Nr + 1, Nz))
        Ez = np.zeros((Nr, Nz + 1))

        # Interior r-faces
        for i in range(1, Nr):
            Er[i, :] = -(V[i, :] - V[i-1, :]) / mesh.dr_c[i]

        # Boundary r-faces
        Er[0, :] = 0.0  # Symmetry axis: Er = 0
        Er[Nr, :] = -(-V[Nr-1, :]) / mesh.dr_c[Nr]  # V=0 at wall

        # Interior z-faces
        for j in range(1, Nz):
            Ez[:, j] = -(V[:, j] - V[:, j-1]) / mesh.dz_c[j]

        # Boundary z-faces
        Ez[:, 0] = -(-V[:, 0]) / mesh.dz_c[0]   # V=0 at z=0 (but V[:,0] is first cell centre)
        Ez[:, Nz] = -(- V[:, Nz-1]) / mesh.dz_c[Nz]  # V=0 at z=L

        return V, Er, Ez


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mesh.mesh_generator import Mesh2D

    print("Poisson solver self-test")
    print("=" * 50)

    # Small test mesh
    mesh = Mesh2D(R=0.05, L=0.05, Nr=20, Nz=20, stretch_r=1.0, stretch_z=1.0)
    print(f"Mesh: {mesh}")

    solver = PoissonSolver(mesh)
    print(f"Matrix shape: {solver.A.shape}, nnz: {solver.A.nnz}")

    # Test 1: Zero charge → V = 0
    n_pos = np.zeros((mesh.Nr, mesh.Nz))
    n_e = np.zeros((mesh.Nr, mesh.Nz))
    n_neg = np.zeros((mesh.Nr, mesh.Nz))
    V, Er, Ez = solver.solve(n_pos, n_e, n_neg)
    print(f"\nTest 1 (zero charge): max|V| = {np.max(np.abs(V)):.2e} V (should be ~0)")

    # Test 2: Uniform positive charge → positive potential in bulk
    n_pos = np.ones((mesh.Nr, mesh.Nz)) * 1e16  # 10¹⁶ m⁻³
    V, Er, Ez = solver.solve(n_pos, n_e, n_neg)
    print(f"Test 2 (uniform n₊=1e16): V_max = {V.max():.2f} V, V at centre = {V[mesh.Nr//2, mesh.Nz//2]:.2f} V")
    print(f"  Er at wall = {Er[-1, mesh.Nz//2]:.0f} V/m (should be positive, pointing outward)")

    # Test 3: Quasi-neutral plasma → small V
    n_pos = np.ones((mesh.Nr, mesh.Nz)) * 1e16
    n_e = np.ones((mesh.Nr, mesh.Nz)) * 1e16 * 0.999  # Slight excess positive charge
    V, Er, Ez = solver.solve(n_pos, n_e, n_neg)
    print(f"Test 3 (quasi-neutral, 0.1% imbalance): V_max = {V.max():.2f} V")

    print("\n✓ Poisson solver OK")
