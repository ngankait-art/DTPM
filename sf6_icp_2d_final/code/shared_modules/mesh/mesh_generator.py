"""
Structured (r,z) mesh for 2D axisymmetric ICP reactor.

Provides non-uniform spacing with algebraic stretching near walls
to resolve sheaths and boundary layers.

Coordinate system:
    r: radial, 0 ≤ r ≤ R   (r=0 is symmetry axis)
    z: axial,  0 ≤ z ≤ L   (z=0 is grounded bottom, z=L is dielectric window / coil)
"""

import numpy as np


class Mesh2D:
    """Structured axisymmetric (r,z) mesh with optional wall stretching."""

    def __init__(self, R, L, Nr, Nz, stretch_r=1.0, stretch_z=1.0):
        """
        Parameters
        ----------
        R : float
            Reactor radius [m].
        L : float
            Reactor height [m].
        Nr : int
            Number of cells in r direction.
        Nz : int
            Number of cells in z direction.
        stretch_r : float
            Stretching factor for r (1.0 = uniform; >1 concentrates points near r=R).
        stretch_z : float
            Stretching factor for z (1.0 = uniform; >1 concentrates points near z=0 and z=L).
        """
        self.R = R
        self.L = L
        self.Nr = Nr
        self.Nz = Nz

        # --- Build 1D node coordinates ---
        # r: stretch toward r=R (wall)
        self.r_nodes = self._stretch_one_side(R, Nr + 1, stretch_r)
        # z: stretch toward both z=0 and z=L (electrodes / window)
        self.z_nodes = self._stretch_two_sides(L, Nz + 1, stretch_z)

        # Cell-centre coordinates
        self.r = 0.5 * (self.r_nodes[:-1] + self.r_nodes[1:])  # (Nr,)
        self.z = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])  # (Nz,)

        # Cell widths
        self.dr = np.diff(self.r_nodes)  # (Nr,)
        self.dz = np.diff(self.z_nodes)  # (Nz,)

        # Distance between cell centres (for gradient computation)
        self.dr_c = np.zeros(Nr + 1)  # at r-faces (Nr+1 faces: 0..Nr)
        self.dr_c[1:-1] = 0.5 * (self.dr[:-1] + self.dr[1:])
        self.dr_c[0] = 0.5 * self.dr[0]      # axis face
        self.dr_c[-1] = 0.5 * self.dr[-1]     # wall face

        self.dz_c = np.zeros(Nz + 1)
        self.dz_c[1:-1] = 0.5 * (self.dz[:-1] + self.dz[1:])
        self.dz_c[0] = 0.5 * self.dz[0]
        self.dz_c[-1] = 0.5 * self.dz[-1]

        # Face positions (cell boundaries)
        self.r_faces = self.r_nodes  # (Nr+1,)
        self.z_faces = self.z_nodes  # (Nz+1,)

        # Face areas for flux computation in cylindrical coordinates
        # Radial face area: A_r = 2π r_face * Δz  (for each z-cell)
        # Axial face area:  A_z = π(r²_right - r²_left)  (annular ring for each r-cell)
        self.cell_volume = np.zeros((Nr, Nz))
        for i in range(Nr):
            for j in range(Nz):
                r_lo, r_hi = self.r_nodes[i], self.r_nodes[i + 1]
                self.cell_volume[i, j] = np.pi * (r_hi**2 - r_lo**2) * self.dz[j]

        # 2D coordinate grids (cell centres)
        self.RR, self.ZZ = np.meshgrid(self.r, self.z, indexing='ij')  # (Nr, Nz)

        # Summary
        self.total_cells = Nr * Nz
        self.dr_min = self.dr.min()
        self.dz_min = self.dz.min()

    # --- Stretching functions ---

    @staticmethod
    def _stretch_one_side(L, N, beta):
        """Stretch grid toward x = L. beta=1 gives uniform."""
        if abs(beta - 1.0) < 1e-10:
            return np.linspace(0, L, N)
        xi = np.linspace(0, 1, N)
        # Hyperbolic tangent stretching
        x = L * (1.0 + np.tanh(beta * (xi - 1.0)) / np.tanh(beta))
        x[0] = 0.0
        x[-1] = L
        return x

    @staticmethod
    def _stretch_two_sides(L, N, beta):
        """Stretch grid toward both x = 0 and x = L. beta=1 gives uniform."""
        if abs(beta - 1.0) < 1e-10:
            return np.linspace(0, L, N)
        xi = np.linspace(0, 1, N)
        # Symmetric hyperbolic tangent stretching
        x = L * 0.5 * (1.0 + np.tanh(beta * (2.0 * xi - 1.0)) / np.tanh(beta))
        x[0] = 0.0
        x[-1] = L
        return x

    def total_volume(self):
        """Total reactor volume [m³]."""
        return np.pi * self.R**2 * self.L

    def volume_average(self, field):
        """Compute volume-weighted average of a 2D field (Nr, Nz)."""
        return np.sum(field * self.cell_volume) / np.sum(self.cell_volume)

    def radial_profile_at_midplane(self, field):
        """Extract radial profile at the axial midplane."""
        j_mid = self.Nz // 2
        return self.r, field[:, j_mid]

    def axial_profile_on_axis(self, field):
        """Extract axial profile at r = 0 (first cell centre)."""
        return self.z, field[0, :]

    def __repr__(self):
        return (f"Mesh2D(R={self.R:.3f}m, L={self.L:.3f}m, "
                f"{self.Nr}×{self.Nz}={self.total_cells} cells, "
                f"Δr_min={self.dr_min*1e3:.2f}mm, Δz_min={self.dz_min*1e3:.2f}mm)")


# === Convenience constructor ===

def make_icp_mesh(R=0.180, L=0.175, Nr=80, Nz=100, stretch_r=1.5, stretch_z=1.5):
    """Create a mesh suitable for ICP reactor simulation.

    Default: Lallement reactor (R=18cm, L=17.5cm) with 80×100 cells
    and moderate stretching near walls.
    """
    return Mesh2D(R, L, Nr, Nz, stretch_r, stretch_z)


# === Self-test ===

if __name__ == '__main__':
    mesh = make_icp_mesh()
    print(mesh)
    print(f"  Total volume: {mesh.total_volume()*1e6:.1f} cm³")
    print(f"  Cell volume range: {mesh.cell_volume.min()*1e9:.3f} – {mesh.cell_volume.max()*1e9:.1f} mm³")
    print(f"  r range: {mesh.r[0]*1e3:.2f} – {mesh.r[-1]*1e3:.1f} mm")
    print(f"  z range: {mesh.z[0]*1e3:.2f} – {mesh.z[-1]*1e3:.1f} mm")

    # Quick visual check
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(np.arange(len(mesh.dr)), mesh.dr * 1e3, 'o-', ms=3)
        ax1.set_xlabel('Cell index (r)')
        ax1.set_ylabel('Δr (mm)')
        ax1.set_title('Radial cell widths')
        ax2.plot(np.arange(len(mesh.dz)), mesh.dz * 1e3, 'o-', ms=3)
        ax2.set_xlabel('Cell index (z)')
        ax2.set_ylabel('Δz (mm)')
        ax2.set_title('Axial cell widths')
        plt.tight_layout()
        plt.savefig('mesh_test.png', dpi=120)
        print("  Saved mesh_test.png")
    except Exception as e:
        print(f"  (plotting skipped: {e})")
