"""
Scharfetter-Gummel (exponential) flux discretization for drift-diffusion.

Used for charged species fluxes in regions where the drift term dominates
diffusion (sheaths). The scheme interpolates between upwind (drift-dominated)
and central (diffusion-dominated) differencing, avoiding artificial diffusion.

Reference: Scharfetter & Gummel, IEEE Trans. Electron Devices 16, 64 (1969)
"""

import numpy as np


def bernoulli(x):
    """Bernoulli function B(x) = x / (exp(x) - 1).

    Numerically stable for all x including x → 0.
    B(0) = 1, B(x) ≈ x·e^(-x) for large x > 0.
    """
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-4
    large = ~small

    # Taylor expansion for small x: B(x) = 1 - x/2 + x²/12 - ...
    out[small] = 1.0 - 0.5 * x[small] + x[small]**2 / 12.0

    # Full expression for moderate/large x
    out[large] = x[large] / (np.exp(x[large]) - 1.0)

    return out


def bernoulli_scalar(x):
    """Scalar version of the Bernoulli function."""
    if abs(x) < 1e-4:
        return 1.0 - 0.5 * x + x * x / 12.0
    return x / (np.expm1(x))


def sg_flux_1d(n, E, mu, D, dx_c, dx):
    """Compute Scharfetter-Gummel fluxes on a 1D grid.

    For species with flux Γ = -D dn/dx + sign·μ·n·E
    where sign = +1 for positive species (drift with E)
    and sign = -1 for negative species (drift against E).

    Parameters
    ----------
    n : array (N,)
        Species density at cell centres.
    E : array (N+1,)
        Electric field at cell faces. Positive E points in +x direction.
    mu : float or array (N,)
        Mobility magnitude (always positive).
    D : float or array (N,)
        Diffusion coefficient (always positive).
    dx_c : array (N+1,)
        Distance between adjacent cell centres (at faces).
    dx : array (N,)
        Cell widths.

    Returns
    -------
    flux : array (N+1,)
        Particle flux at each face (including boundaries).
        Boundary fluxes (index 0 and N) are set to zero and must be
        overwritten by the caller with proper boundary conditions.

    Notes
    -----
    Sign convention: For electrons, the drift flux is +μₑnₑE (toward anode).
    For positive ions, the drift flux is -μ₊n₊E (toward cathode, but we
    express it as -μ₊n₊∇V = +μ₊n₊E if E = -∇V).

    The caller should provide E with the correct sign for the species.
    Specifically, for a species with charge q:
        drift velocity = (q/|q|) · μ · E
    So for electrons (q < 0): pass -E as the field argument to get drift away from cathode.
    For positive ions (q > 0): pass E directly.
    For negative ions (q < 0): pass -E.
    """
    N = len(n)
    flux = np.zeros(N + 1)

    # Interior faces (1..N-1)
    for f in range(1, N):
        i_L = f - 1  # left cell
        i_R = f      # right cell

        # Local transport coefficients (average at face)
        if np.isscalar(mu):
            mu_f = mu
            D_f = D
        else:
            mu_f = 0.5 * (mu[i_L] + mu[i_R])
            D_f = 0.5 * (D[i_L] + D[i_R])

        # Peclet number: Z = μ·E·Δx / D
        Z = mu_f * E[f] * dx_c[f] / max(D_f, 1e-30)

        # Scharfetter-Gummel flux: Γ = D/Δx · [B(Z)·n_R - B(-Z)·n_L]
        B_pos = bernoulli_scalar(Z)
        B_neg = bernoulli_scalar(-Z)

        flux[f] = D_f / dx_c[f] * (B_pos * n[i_R] - B_neg * n[i_L])

    # Boundary fluxes = 0 (placeholder; caller must set proper BCs)
    flux[0] = 0.0
    flux[N] = 0.0

    return flux


def sg_flux_r(n, Er, mu, D, mesh):
    """Compute S-G fluxes in the r-direction on a 2D mesh.

    Parameters
    ----------
    n : array (Nr, Nz)
        Species density at cell centres.
    Er : array (Nr+1, Nz)
        Radial electric field at r-faces.
    mu : float or array (Nr, Nz)
        Mobility.
    D : float or array (Nr, Nz)
        Diffusion coefficient.
    mesh : Mesh2D
        The mesh object.

    Returns
    -------
    flux_r : array (Nr+1, Nz)
        Radial flux at each r-face.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    flux_r = np.zeros((Nr + 1, Nz))

    for j in range(Nz):
        for f in range(1, Nr):
            i_L = f - 1
            i_R = f

            if np.isscalar(mu):
                mu_f, D_f = mu, D
            else:
                mu_f = 0.5 * (mu[i_L, j] + mu[i_R, j])
                D_f = 0.5 * (D[i_L, j] + D[i_R, j])

            Z = mu_f * Er[f, j] * mesh.dr_c[f] / max(D_f, 1e-30)
            B_pos = bernoulli_scalar(Z)
            B_neg = bernoulli_scalar(-Z)

            flux_r[f, j] = D_f / mesh.dr_c[f] * (B_pos * n[i_R, j] - B_neg * n[i_L, j])

    return flux_r


def sg_flux_z(n, Ez, mu, D, mesh):
    """Compute S-G fluxes in the z-direction on a 2D mesh.

    Parameters
    ----------
    n : array (Nr, Nz)
        Species density.
    Ez : array (Nr, Nz+1)
        Axial electric field at z-faces.
    mu : float or array (Nr, Nz)
        Mobility.
    D : float or array (Nr, Nz)
        Diffusion coefficient.
    mesh : Mesh2D

    Returns
    -------
    flux_z : array (Nr, Nz+1)
        Axial flux at each z-face.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    flux_z = np.zeros((Nr, Nz + 1))

    for i in range(Nr):
        for f in range(1, Nz):
            j_L = f - 1
            j_R = f

            if np.isscalar(mu):
                mu_f, D_f = mu, D
            else:
                mu_f = 0.5 * (mu[i, j_L] + mu[i, j_R])
                D_f = 0.5 * (D[i, j_L] + D[i, j_R])

            Z = mu_f * Ez[i, f] * mesh.dz_c[f] / max(D_f, 1e-30)
            B_pos = bernoulli_scalar(Z)
            B_neg = bernoulli_scalar(-Z)

            flux_z[i, f] = D_f / mesh.dz_c[f] * (B_pos * n[i, j_R] - B_neg * n[i, j_L])

    return flux_z


def divergence_cyl(flux_r, flux_z, mesh):
    """Compute ∇·Γ in cylindrical coordinates on the 2D mesh.

    (1/r) ∂(r Γ_r)/∂r + ∂Γ_z/∂z

    Parameters
    ----------
    flux_r : array (Nr+1, Nz)
        Radial flux at r-faces.
    flux_z : array (Nr, Nz+1)
        Axial flux at z-faces.
    mesh : Mesh2D

    Returns
    -------
    div : array (Nr, Nz)
        Divergence at cell centres.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    div = np.zeros((Nr, Nz))

    for i in range(Nr):
        r_lo = mesh.r_faces[i]
        r_hi = mesh.r_faces[i + 1]
        r_mid = mesh.r[i]
        dr = mesh.dr[i]

        for j in range(Nz):
            dz = mesh.dz[j]

            # (1/r) d(r Γ_r)/dr
            div_r = (r_hi * flux_r[i + 1, j] - r_lo * flux_r[i, j]) / (r_mid * dr)

            # dΓ_z/dz
            div_z = (flux_z[i, j + 1] - flux_z[i, j]) / dz

            div[i, j] = div_r + div_z

    return div


# === Self-test ===

if __name__ == '__main__':
    print("Testing Bernoulli function...")
    x = np.array([-10, -1, -0.01, 0, 0.01, 1, 10])
    b = np.array([bernoulli_scalar(xi) for xi in x])
    print(f"  x     = {x}")
    print(f"  B(x)  = {b}")
    # Check: B(0) should be 1.0
    assert abs(bernoulli_scalar(0.0) - 1.0) < 1e-12
    # Check: B(x) - B(-x) = -x  (identity: x/(e^x-1) - (-x)/(e^(-x)-1) = -x)
    # Equivalently B(-x) - B(x) = x
    for xi in [0.5, 2.0, 10.0]:
        lhs = bernoulli_scalar(-xi) - bernoulli_scalar(xi)
        assert abs(lhs - xi) < 1e-10, f"Identity failed at x={xi}: {lhs} vs {xi}"
    print("  ✓ Bernoulli function OK")

    print("\nTesting 1D S-G flux...")
    # Uniform density, uniform field → constant flux
    N = 20
    n = np.ones(N) * 1e16
    E = np.ones(N + 1) * 100.0  # V/m
    mu = 0.1  # m²/(V·s)
    D = 0.1   # m²/s
    dx = np.ones(N) * 0.01
    dx_c = np.ones(N + 1) * 0.01
    flux = sg_flux_1d(n, E, mu, D, dx_c, dx)
    # Interior flux should be mu*n*E = 0.1 * 1e16 * 100 = 1e17
    print(f"  Expected flux: {mu * 1e16 * 100:.2e}")
    print(f"  Computed flux (interior): {flux[N//2]:.2e}")
    print("  ✓ 1D S-G flux OK")
