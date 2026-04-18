"""
Physics-Informed Neural Network (PINN) solver for the TEL ICP reactor.

Replaces the finite-difference sparse-matrix solver in solver.py with a
mesh-free PINN that learns n_F(r,z), n_SF6(r,z), and T_e(r,z) by
minimising PDE residuals via automatic differentiation.

Governing equations (from solver.py, cylindrical axisymmetric):

  Species k ∈ {F, SF6}:
    (1/r) ∂/∂r(r D_k ∂n_k/∂r) + ∂/∂z(D_k ∂n_k/∂z) + S_k - L_k n_k = 0

  Electron energy (from TELSolverWithEnergy):
    (1/r) ∂/∂r(r κ_e ∂Te/∂r) + ∂/∂z(κ_e ∂Te/∂z) + P_abs/(3/2 ne) - L_e(Te) = 0

Boundary conditions (material-specific Robin from boundary_conditions.py):
    -D_k ∂n_k/∂n_wall = γ_k (v_th/4) n_k     (wall recombination)
    ∂n_k/∂r |_{r=0}   = 0                      (axis symmetry)

This module provides:
    TELPINNModel      — PyTorch neural network (r,z) → (n_F, n_SF6, Te)
    TELPINNTrainer    — Training loop with PDE + BC + data loss terms
    train_pinn()      — End-to-end training entry point
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.constants import k as kB, e as eC, pi

# ═══════════════════════════════════════════════════════════════════
# 1. CONSTANTS AND GEOMETRY (matching solver.py exactly)
# ═══════════════════════════════════════════════════════════════════

# TEL reactor dimensions (m)
R_ICP  = 0.038
R_PROC = 0.105
L_ICP  = 0.1815
L_PROC = 0.050
L_APT  = 0.002
Z_APT_BOT = L_PROC
Z_APT_TOP = L_PROC + L_APT
Z_TOP     = L_PROC + L_APT + L_ICP

# Reference scales for non-dimensionalisation
N_REF  = 1e19       # m^-3  (density scale)
TE_REF = 3.0        # eV    (temperature scale)
R_REF  = R_PROC     # m     (radial length scale)
Z_REF  = Z_TOP      # m     (axial length scale)


# ═══════════════════════════════════════════════════════════════════
# 2. GEOMETRY MASK (differentiable smooth approximation)
# ═══════════════════════════════════════════════════════════════════

def inside_domain(r: torch.Tensor, z: torch.Tensor,
                  sharpness: float = 200.0) -> torch.Tensor:
    """Smooth approximation of the TEL T-shaped domain indicator.

    Returns a value in [0, 1] that is ~1 inside the gas volume and
    ~0 in the solid. Uses sigmoid transitions so gradients can flow.

    Geometry (same as geometry.py):
        ICP:        r <= R_icp,  z >= z_apt_top
        Aperture:   r <= R_icp,  z_apt_bot <= z < z_apt_top
        Processing: r <= R_proc, z < z_apt_bot
    """
    s = sharpness
    # Processing region: r <= R_proc AND z < z_apt_bot
    proc = torch.sigmoid(s * (R_PROC - r)) * torch.sigmoid(s * (Z_APT_BOT - z))
    # ICP + aperture: r <= R_icp AND z >= z_apt_bot
    icp = torch.sigmoid(s * (R_ICP - r)) * torch.sigmoid(s * (z - Z_APT_BOT))
    return torch.clamp(proc + icp, 0.0, 1.0)


def is_wall(r: torch.Tensor, z: torch.Tensor,
            tol: float = 0.002) -> Dict[str, torch.Tensor]:
    """Identify wall proximity for each boundary type.

    Returns dict of soft indicators (0 to 1) for each wall.
    tol is the wall layer thickness in metres.
    """
    s = 1.0 / tol
    walls = {}
    omega = inside_domain(r, z)

    # Axis: r ≈ 0
    walls['axis'] = torch.sigmoid(s * (tol - r)) * omega

    # Wafer: z ≈ 0, r <= R_proc
    walls['wafer'] = torch.sigmoid(s * (tol - z)) * omega

    # Quartz: r ≈ R_icp, z >= z_apt_top (ICP side wall)
    walls['quartz'] = (torch.sigmoid(s * (r - (R_ICP - tol)))
                       * torch.sigmoid(s * (z - Z_APT_TOP))
                       * omega)

    # Window: z ≈ z_top (top of ICP, dielectric)
    walls['window'] = torch.sigmoid(s * (z - (Z_TOP - tol))) * omega

    # Al side: r ≈ R_proc, z < z_apt_bot (processing side wall)
    walls['al_side'] = (torch.sigmoid(s * (r - (R_PROC - tol)))
                        * torch.sigmoid(s * (Z_APT_BOT - z))
                        * omega)

    # Al top (aperture underside): z ≈ z_apt_bot, r > R_icp
    walls['al_top'] = (torch.sigmoid(s * (r - R_ICP))
                       * torch.sigmoid(s * (z - (Z_APT_BOT - tol)))
                       * torch.sigmoid(s * (Z_APT_TOP - z))
                       * omega)

    # Shoulder: r ≈ R_icp, z_apt_bot <= z < z_apt_top
    walls['shoulder'] = (torch.sigmoid(s * (r - (R_ICP - tol)))
                         * torch.sigmoid(s * (z - Z_APT_BOT))
                         * torch.sigmoid(s * (Z_APT_TOP - z))
                         * omega)

    return walls


# ═══════════════════════════════════════════════════════════════════
# 3. NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════

class FourierFeatures(nn.Module):
    """Random Fourier feature embedding for (r, z) inputs.

    Helps the network learn high-frequency spatial structure
    (Tancik et al., NeurIPS 2020).
    """
    def __init__(self, n_input: int = 2, n_features: int = 64,
                 scale: float = 4.0):
        super().__init__()
        B = torch.randn(n_input, n_features) * scale
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B  # (N, n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TELPINNModel(nn.Module):
    """Neural network mapping (r, z) → (n_F, n_SF6, Te).

    Architecture
    ------------
    Input:  (r/R_ref, z/Z_ref) → Fourier features → [sin, cos] embedding
    Hidden: 6 layers × 128 units, GELU activation, residual connections
    Output: 3 channels (n_F_hat, n_SF6_hat, Te_hat) in normalised units

    Physical constraints on outputs:
        n_F, n_SF6 > 0  (enforced via softplus)
        Te > 0.3 eV     (enforced via softplus + offset)
    """
    def __init__(self, n_hidden: int = 128, n_layers: int = 6,
                 n_fourier: int = 64, fourier_scale: float = 4.0):
        super().__init__()
        self.ff = FourierFeatures(2, n_fourier, fourier_scale)
        n_in = 2 * n_fourier  # sin + cos

        layers = []
        layers.append(nn.Linear(n_in, n_hidden))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.GELU())
        self.backbone = nn.Sequential(*layers)

        # Residual projection (if input dim != hidden dim)
        self.proj_in = nn.Linear(n_in, n_hidden)

        # Output heads
        self.head_nF   = nn.Linear(n_hidden, 1)
        self.head_nSF6 = nn.Linear(n_hidden, 1)
        self.head_Te   = nn.Linear(n_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rz: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        rz : (N, 2) tensor
            Columns: (r / R_ref, z / Z_ref), normalised coordinates.

        Returns
        -------
        out : (N, 3) tensor
            Columns: (n_F / N_ref, n_SF6 / N_ref, Te / Te_ref)
            All positive by construction.
        """
        x = self.ff(rz)
        h = self.backbone(x)
        # Residual connection
        h = h + self.proj_in(x)

        nF_raw   = self.head_nF(h)
        nSF6_raw = self.head_nSF6(h)
        Te_raw   = self.head_Te(h)

        # Enforce positivity
        nF   = nn.functional.softplus(nF_raw)
        nSF6 = nn.functional.softplus(nSF6_raw)
        Te   = nn.functional.softplus(Te_raw) + 0.1  # Te > 0.3 eV (0.1 * TE_REF)

        return torch.cat([nF, nSF6, Te], dim=-1)


# ═══════════════════════════════════════════════════════════════════
# 4. PDE RESIDUALS (automatic differentiation)
# ═══════════════════════════════════════════════════════════════════

def _compute_derivatives(model: TELPINNModel,
                         r: torch.Tensor,
                         z: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
    """Compute all first and second derivatives via autograd.

    For a scalar field u(r,z), computes:
        ∂u/∂r, ∂u/∂z, ∂²u/∂r², ∂²u/∂z²

    For the cylindrical Laplacian of species diffusion:
        (1/r) ∂/∂r(r D ∂n/∂r) + ∂/∂z(D ∂n/∂z)
      = D [∂²n/∂r² + (1/r)∂n/∂r + ∂²n/∂z²]
    when D is spatially uniform per species.
    """
    r = r.requires_grad_(True)
    z = z.requires_grad_(True)

    rz = torch.stack([r / R_REF, z / Z_REF], dim=-1)
    out = model(rz)  # (N, 3) normalised

    # Un-normalise
    nF   = out[:, 0] * N_REF
    nSF6 = out[:, 1] * N_REF
    Te   = out[:, 2] * TE_REF

    derivs = {}

    for name, field in [('nF', nF), ('nSF6', nSF6), ('Te', Te)]:
        # First derivatives
        g = torch.autograd.grad(field.sum(), [r, z], create_graph=True)
        df_dr = g[0]
        df_dz = g[1]

        # Second derivatives
        g2r = torch.autograd.grad(df_dr.sum(), r, create_graph=True)[0]
        g2z = torch.autograd.grad(df_dz.sum(), z, create_graph=True)[0]

        derivs[f'd{name}_dr']  = df_dr
        derivs[f'd{name}_dz']  = df_dz
        derivs[f'd2{name}_dr2'] = g2r
        derivs[f'd2{name}_dz2'] = g2z

    derivs['nF']   = nF
    derivs['nSF6'] = nSF6
    derivs['Te']   = Te
    derivs['r']    = r
    derivs['z']    = z

    return derivs


def species_residual(derivs: Dict[str, torch.Tensor],
                     D: float, S: torch.Tensor, L: torch.Tensor,
                     species: str) -> torch.Tensor:
    """Compute the diffusion–reaction residual for species `species`.

    PDE: D [(1/r) ∂/∂r(r ∂n/∂r) + ∂²n/∂z²] + S - L·n = 0

    Expanding the cylindrical radial term:
        (1/r) ∂/∂r(r ∂n/∂r) = ∂²n/∂r² + (1/r) ∂n/∂r

    Parameters
    ----------
    derivs : dict from _compute_derivatives
    D : float, diffusion coefficient (m²/s)
    S : (N,) tensor, volumetric source term (m⁻³ s⁻¹)
    L : (N,) tensor, loss frequency (s⁻¹), positive
    species : 'nF' or 'nSF6'
    """
    n     = derivs[species]
    dn_dr = derivs[f'd{species}_dr']
    d2n_dr2 = derivs[f'd2{species}_dr2']
    d2n_dz2 = derivs[f'd2{species}_dz2']
    r = derivs['r']

    # Cylindrical Laplacian: ∂²n/∂r² + (1/r)∂n/∂r + ∂²n/∂z²
    # Handle axis (r → 0) with L'Hôpital: lim_{r→0} (1/r)∂n/∂r = ∂²n/∂r²
    # so the full radial operator becomes 2·∂²n/∂r² at r=0.
    on_axis = (r < 1e-5).float()
    r_safe = torch.clamp(r, min=1e-4)
    radial_term = d2n_dr2 + (1.0 - on_axis) * dn_dr / r_safe + on_axis * d2n_dr2
    laplacian = radial_term + d2n_dz2

    residual = D * laplacian + S - L * n
    return residual


def energy_residual(derivs: Dict[str, torch.Tensor],
                    ne: torch.Tensor,
                    P_abs: torch.Tensor,
                    kappa_e: float = 100.0 * 5.0 / 3.0
                    ) -> torch.Tensor:
    """Compute the electron energy PDE residual.

    PDE: κ_e [(1/r)∂/∂r(r ∂Te/∂r) + ∂²Te/∂z²] + P_abs/(3/2 e ne) - L_e = 0

    where κ_e ∝ D_e · ne (but we use a reference value for stability,
    matching solver.py TELSolverWithEnergy).
    """
    Te = derivs['Te']
    dTe_dr = derivs['dTe_dr']
    d2Te_dr2 = derivs['d2Te_dr2']
    d2Te_dz2 = derivs['d2Te_dz2']
    r = derivs['r']

    on_axis = (r < 1e-5).float()
    r_safe = torch.clamp(r, min=1e-4)
    ne_safe = torch.clamp(ne, min=1e10)

    laplacian_Te = (d2Te_dr2
                    + (1.0 - on_axis) * dTe_dr / r_safe
                    + on_axis * d2Te_dr2
                    + d2Te_dz2)

    # Heating: P_abs / (3/2 · e · ne)  [eV/s]
    heating = P_abs / (1.5 * eC * ne_safe)

    # Collisional loss (simplified Arrhenius from sf6_rates.py)
    nSF6 = derivs['nSF6']
    k_iz  = 1.2e-7 * torch.exp(-18.1 / Te) * 1e-6
    k_att = 2.4e-10 / torch.clamp(Te, min=0.3)**1.49 * 1e-6
    k_d1  = 1.5e-7 * torch.exp(-8.1 / Te) * 1e-6
    L_e = nSF6 * (15.7 * k_iz + 3.5 * k_att + 8.1 * k_d1)

    residual = kappa_e * laplacian_Te + heating - L_e
    return residual


# ═══════════════════════════════════════════════════════════════════
# 5. BOUNDARY CONDITION LOSSES
# ═══════════════════════════════════════════════════════════════════

def bc_loss_axis(derivs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Neumann symmetry: ∂n/∂r = 0 at r = 0 for all fields.

    Normalised by reference scales to keep loss O(1).
    """
    loss = ((derivs['dnF_dr'] / (N_REF / R_REF))**2
            + (derivs['dnSF6_dr'] / (N_REF / R_REF))**2
            + (derivs['dTe_dr'] / (TE_REF / R_REF))**2)
    return loss


def bc_loss_wall(derivs: Dict[str, torch.Tensor],
                 gamma: float,
                 v_th: float,
                 D: float,
                 species: str,
                 normal_dir: str  # 'r' or 'z'
                 ) -> torch.Tensor:
    """Robin BC: -D ∂n/∂n_wall = γ (v_th/4) n.

    Rearranged as residual:
        D ∂n/∂n + γ (v_th/4) n = 0

    The sign of ∂n/∂n depends on the outward normal direction.
    """
    n = derivs[species]
    if normal_dir == 'r':
        dn = derivs[f'd{species}_dr']   # outward normal is +r
    elif normal_dir == '-r':
        dn = -derivs[f'd{species}_dr']  # inward wall (shouldn't occur)
    elif normal_dir == 'z':
        dn = derivs[f'd{species}_dz']   # outward normal is +z (top)
    elif normal_dir == '-z':
        dn = -derivs[f'd{species}_dz']  # outward normal is -z (wafer)
    else:
        raise ValueError(f"Unknown normal_dir: {normal_dir}")

    residual = D * dn + gamma * (v_th / 4.0) * n
    # Normalise by characteristic flux scale D * N_ref / R_ref
    scale = D * N_REF / R_REF
    return (residual / scale)**2


def bc_loss_Te_wall(derivs: Dict[str, torch.Tensor],
                    Te_wall: float = 1.0) -> torch.Tensor:
    """Dirichlet BC for electron temperature at walls: Te = Te_wall."""
    Te = derivs['Te']
    return ((Te - Te_wall) / TE_REF)**2


# ═══════════════════════════════════════════════════════════════════
# 6. COLLOCATION SAMPLING
# ═══════════════════════════════════════════════════════════════════

def sample_interior(n_points: int, device: torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample collocation points inside the TEL domain.

    Uses rejection sampling on the T-shaped geometry:
        ICP:        r ∈ [0, R_icp],  z ∈ [z_apt_top, z_top]
        Aperture:   r ∈ [0, R_icp],  z ∈ [z_apt_bot, z_apt_top]
        Processing: r ∈ [0, R_proc], z ∈ [0, z_apt_bot]
    """
    points_r, points_z = [], []

    # Processing region (largest volume)
    n_proc = int(n_points * 0.5)
    r = torch.rand(n_proc, device=device) * R_PROC
    z = torch.rand(n_proc, device=device) * Z_APT_BOT
    points_r.append(r)
    points_z.append(z)

    # ICP region
    n_icp = int(n_points * 0.4)
    r = torch.rand(n_icp, device=device) * R_ICP
    z = torch.rand(n_icp, device=device) * L_ICP + Z_APT_TOP
    points_r.append(r)
    points_z.append(z)

    # Aperture channel (small but important)
    n_apt = n_points - n_proc - n_icp
    r = torch.rand(n_apt, device=device) * R_ICP
    z = torch.rand(n_apt, device=device) * L_APT + Z_APT_BOT
    points_r.append(r)
    points_z.append(z)

    return torch.cat(points_r), torch.cat(points_z)


def sample_boundary(n_per_wall: int, device: torch.device
                    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, str]]:
    """Sample points on each boundary segment.

    Returns dict: name → (r, z, outward_normal_direction).
    """
    bc = {}
    n = n_per_wall

    # Axis: r = 0, z ∈ [0, z_top] (only where inside)
    z = torch.rand(n, device=device) * Z_TOP
    bc['axis'] = (torch.zeros(n, device=device), z, 'r')

    # Wafer: z = 0, r ∈ [0, R_proc]
    r = torch.rand(n, device=device) * R_PROC
    bc['wafer'] = (r, torch.zeros(n, device=device), '-z')

    # Quartz: r = R_icp, z ∈ [z_apt_top, z_top]
    z = torch.rand(n, device=device) * L_ICP + Z_APT_TOP
    bc['quartz'] = (torch.full((n,), R_ICP, device=device), z, 'r')

    # Window: z = z_top, r ∈ [0, R_icp]
    r = torch.rand(n, device=device) * R_ICP
    bc['window'] = (r, torch.full((n,), Z_TOP, device=device), 'z')

    # Al side: r = R_proc, z ∈ [0, z_apt_bot]
    z = torch.rand(n, device=device) * Z_APT_BOT
    bc['al_side'] = (torch.full((n,), R_PROC, device=device), z, 'r')

    # Al top (aperture underside): z = z_apt_bot, r ∈ [R_icp, R_proc]
    r = torch.rand(n, device=device) * (R_PROC - R_ICP) + R_ICP
    bc['al_top'] = (r, torch.full((n,), Z_APT_BOT, device=device), '-z')

    return bc


# ═══════════════════════════════════════════════════════════════════
# 7. CHEMISTRY SOURCE/SINK TERMS (from sf6_rates.py)
# ═══════════════════════════════════════════════════════════════════

def compute_chemistry(nF: torch.Tensor, nSF6: torch.Tensor,
                      Te: torch.Tensor, ne: torch.Tensor,
                      Tgas: float = 313.0, p_mTorr: float = 10.0
                      ) -> Dict[str, torch.Tensor]:
    """Compute source and loss terms for F and SF6 transport.

    Implements the same reaction set as solver.py lines 396–444.
    All rate coefficients from sf6_rates.py (Arrhenius, in m³/s).
    """
    Te_safe = torch.clamp(Te, min=0.3, max=15.0)

    # Key rates (m³/s) — clamp exponents to prevent overflow
    k_d1  = 1.5e-7  * torch.exp(torch.clamp(-8.1  / Te_safe, min=-40)) * 1e-6
    k_d2  = 9e-9    * torch.exp(torch.clamp(-13.4 / Te_safe, min=-40)) * 1e-6
    k_d3  = 2.5e-8  * torch.exp(torch.clamp(-33.5 / Te_safe, min=-40)) * 1e-6
    k_d4  = 2.3e-8  * torch.exp(torch.clamp(-23.9 / Te_safe, min=-40)) * 1e-6
    k_d5  = 1.5e-9  * torch.exp(torch.clamp(-26.0 / Te_safe, min=-40)) * 1e-6
    k_iz  = 1.2e-7  * torch.exp(torch.clamp(-18.1 / Te_safe, min=-40)) * 1e-6
    k_att = 2.4e-10 / Te_safe**1.49 * 1e-6

    # --- SF6 loss rate (dissociation + ionisation + attachment) ---
    k_total_SF6 = k_d1 + k_d2 + k_d3 + k_d4 + k_d5 + k_iz + k_att
    # Residence time loss
    ng = p_mTorr * 0.133322 / (kB * Tgas)
    nSF6_feed = ng
    V_total = pi * R_ICP**2 * L_ICP + pi * R_PROC**2 * L_PROC
    tau_R = p_mTorr * 0.133322 * V_total / (100e-6/60 * 1.01325e5 * Tgas/273.15)
    tau_R = max(tau_R, 1e-3)

    L_SF6 = ne * k_total_SF6 + 1.0 / tau_R          # loss frequency (s⁻¹)
    S_SF6 = nSF6_feed / tau_R                        # feed replenishment (m⁻³ s⁻¹)

    # --- Fluorine source (electron-impact dissociation of SF6) ---
    S_F = ne * nSF6 * (k_d1 + 2*k_d2 + 3*k_d3 + 2*k_d4 + 3*k_d5)
    # F loss: pump + wall (wall loss is handled via BC, pump here)
    L_F = 1.0 / tau_R                               # loss frequency (s⁻¹)

    return {
        'S_F': S_F, 'L_F': L_F,
        'S_SF6': S_SF6, 'L_SF6': L_SF6,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. POWER DEPOSITION PROFILE (from solver.py _compute_power_deposition)
# ═══════════════════════════════════════════════════════════════════

def power_deposition(r: torch.Tensor, z: torch.Tensor,
                     P_abs: float = 301.0) -> torch.Tensor:
    """EM power deposition profile P_abs(r,z) [W/m³].

    Gaussian skin-depth ansatz inside the ICP region,
    zero in the processing region. Normalised to integrate to P_abs.
    """
    # Only in ICP region
    in_icp = ((r <= R_ICP) & (z >= Z_APT_TOP)).float()
    z_local = (z - Z_APT_TOP) / L_ICP  # 0 at bottom, 1 at top
    r_norm = r / R_ICP
    # Radial: peaks near wall at r/R ≈ 0.7 (skin depth)
    p_r = torch.exp(-((r_norm - 0.7) / 0.25)**2)
    # Axial: peaks near coil (top third)
    p_z = torch.exp(-((z_local - 0.85) / 0.3)**2)
    p = p_r * p_z * in_icp

    # Normalise (approximate — exact normalisation requires integration)
    V_icp = pi * R_ICP**2 * L_ICP
    p_avg = p.mean()
    if p_avg > 0:
        p = p * (P_abs / (p_avg * V_icp + 1e-30))
    return p


# ═══════════════════════════════════════════════════════════════════
# 9. TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PINNConfig:
    """Training hyperparameters."""
    # Collocation
    n_interior: int = 4000
    n_boundary: int = 200      # per wall segment

    # Loss weights
    w_pde_F:    float = 1.0
    w_pde_SF6:  float = 1.0
    w_pde_Te:   float = 1e-6   # energy PDE is extremely stiff — ramp up later
    w_bc_robin: float = 10.0   # BCs are critical
    w_bc_axis:  float = 10.0
    w_bc_Te:    float = 10.0
    w_data:     float = 1.0    # supervised data from FD solver (if available)

    # Physics
    D_F:   float = 1.6         # m²/s (fluorine diffusion)
    D_SF6: float = 0.5         # m²/s (SF6 diffusion)
    P_rf:  float = 700.0       # W
    eta:   float = 0.43
    p_mTorr: float = 10.0
    Tgas:  float = 313.0

    # Robin BC gammas (from boundary_conditions.py)
    gamma_quartz: float = 0.001
    gamma_Al:     float = 0.18
    gamma_wafer:  float = 0.025
    gamma_window: float = 0.001

    # Optimiser
    lr: float = 1e-3
    n_epochs: int = 20000
    scheduler_step: int = 5000
    scheduler_gamma: float = 0.5

    # Device
    device: str = 'cpu'


# ═══════════════════════════════════════════════════════════════════
# 10. TRAINER
# ═══════════════════════════════════════════════════════════════════

class TELPINNTrainer:
    """Training loop for the TEL PINN solver."""

    def __init__(self, model: TELPINNModel, cfg: PINNConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.P_abs = cfg.eta * cfg.P_rf

        # Thermal speed of F atoms
        self.v_th_F = float(np.sqrt(8 * kB * cfg.Tgas / (pi * 19.0 * 1.66054e-27)))

        # Robin BC gamma for each wall
        self.gamma_walls = {
            'wafer':   cfg.gamma_wafer,
            'quartz':  cfg.gamma_quartz,
            'window':  cfg.gamma_window,
            'al_side': cfg.gamma_Al,
            'al_top':  cfg.gamma_Al,
            'shoulder': cfg.gamma_Al,
        }

        self.optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser, step_size=cfg.scheduler_step,
            gamma=cfg.scheduler_gamma)

        # Optional: reference FD solution for data loss
        self.ref_data = None

    def load_reference(self, fd_result: dict):
        """Load reference solution from the FD solver for data loss.

        fd_result is the dict returned by TELSolver.solve().
        """
        m = fd_result['mesh']
        inside = fd_result['inside']
        r_pts, z_pts, nF_ref, nSF6_ref, Te_ref = [], [], [], [], []

        for i in range(m.Nr):
            for j in range(m.Nz):
                if inside[i, j]:
                    r_pts.append(m.rc[i])
                    z_pts.append(m.zc[j])
                    nF_ref.append(fd_result['nF'][i, j])
                    nSF6_ref.append(fd_result['nSF6'][i, j])
                    Te_ref.append(fd_result['Te'][i, j])

        self.ref_data = {
            'r': torch.tensor(r_pts, dtype=torch.float32, device=self.device),
            'z': torch.tensor(z_pts, dtype=torch.float32, device=self.device),
            'nF': torch.tensor(nF_ref, dtype=torch.float32, device=self.device),
            'nSF6': torch.tensor(nSF6_ref, dtype=torch.float32, device=self.device),
            'Te': torch.tensor(Te_ref, dtype=torch.float32, device=self.device),
        }

    def _compute_ne_from_Te(self, Te: torch.Tensor,
                            r: torch.Tensor, z: torch.Tensor
                            ) -> torch.Tensor:
        """Approximate ne(r,z) from Te using the same power-balance
        and eigenmode shape as solver.py _compute_ne.

        This is NOT a PDE solve for ne — it is the prescribed shape
        from the FD solver, consistent with the 0D power balance.
        """
        # Simplified: ne ∝ Bessel-cosine in ICP, exponential decay below
        in_icp = ((r <= R_ICP) & (z >= Z_APT_TOP)).float()
        in_proc = (z < Z_APT_BOT).float()

        # ICP eigenmode (approximate J0 · cos)
        ne_icp = 0.5 * torch.cos(pi * (z - Z_APT_TOP) / (2 * L_ICP)).clamp(min=0)
        # Processing: exponential decay from aperture
        ne_proc = 0.1 * torch.exp(-(Z_APT_BOT - z) / 0.012)
        ne_shape = in_icp * ne_icp + in_proc * ne_proc + 0.01

        # Scale from 0D power balance (rough estimate)
        Te_avg = Te.mean().item()
        k_iz = 1.2e-7 * np.exp(-18.1 / max(Te_avg, 0.5)) * 1e-6
        nSF6_feed = self.cfg.p_mTorr * 0.133322 / (kB * self.cfg.Tgas)
        eps_T = 200.0  # eV (energy cost per ion pair, approximate)
        V_icp = pi * R_ICP**2 * L_ICP
        ne_avg = self.P_abs / max(nSF6_feed * k_iz * eps_T * eC * V_icp, 1e-30)

        return ne_shape * ne_avg

    def train_step(self) -> Dict[str, float]:
        """One training iteration. Returns loss components."""
        self.model.train()
        cfg = self.cfg

        # ── Sample collocation points ──
        r_int, z_int = sample_interior(cfg.n_interior, self.device)
        bc_pts = sample_boundary(cfg.n_boundary, self.device)

        # ── Interior PDE residuals ──
        derivs = _compute_derivatives(self.model, r_int, z_int)
        nF   = derivs['nF']
        nSF6 = derivs['nSF6']
        Te   = derivs['Te']

        # Electron density (prescribed shape)
        ne = self._compute_ne_from_Te(Te, r_int, z_int)

        # Chemistry source/sink
        chem = compute_chemistry(nF, nSF6, Te, ne,
                                 cfg.Tgas, cfg.p_mTorr)

        # Domain mask (suppress residuals outside gas volume)
        omega = inside_domain(r_int, z_int)

        # F diffusion residual (normalised by D_F * N_ref / R_ref^2)
        res_F = species_residual(derivs, cfg.D_F,
                                 chem['S_F'], chem['L_F'], 'nF')
        scale_F = cfg.D_F * N_REF / R_REF**2
        L_pde_F = (omega * (res_F / scale_F)**2).mean()

        # SF6 diffusion residual
        res_SF6 = species_residual(derivs, cfg.D_SF6,
                                   chem['S_SF6'], chem['L_SF6'], 'nSF6')
        scale_SF6 = cfg.D_SF6 * N_REF / R_REF**2
        L_pde_SF6 = (omega * (res_SF6 / scale_SF6)**2).mean()

        # Electron energy residual
        P_local = power_deposition(r_int, z_int, self.P_abs)
        res_Te = energy_residual(derivs, ne, P_local)
        scale_Te = TE_REF / R_REF**2  # characteristic scale
        L_pde_Te = (omega * (res_Te / max(scale_Te, 1e-10))**2).mean()

        # ── Boundary condition losses ──
        L_bc = torch.tensor(0.0, device=self.device)

        # Axis symmetry (r = 0)
        r_ax, z_ax, _ = bc_pts['axis']
        derivs_ax = _compute_derivatives(self.model, r_ax, z_ax)
        L_bc = L_bc + cfg.w_bc_axis * bc_loss_axis(derivs_ax).mean()

        # Robin BCs at material walls
        for wall_name, gamma in self.gamma_walls.items():
            if wall_name not in bc_pts:
                continue
            r_w, z_w, normal = bc_pts[wall_name]
            derivs_w = _compute_derivatives(self.model, r_w, z_w)
            # F Robin BC
            L_bc = L_bc + cfg.w_bc_robin * bc_loss_wall(
                derivs_w, gamma, self.v_th_F, cfg.D_F, 'nF', normal).mean()
            # Te Dirichlet at walls
            L_bc = L_bc + cfg.w_bc_Te * bc_loss_Te_wall(derivs_w, 1.0).mean()

        # ── Data loss (if FD reference available) ──
        L_data = torch.tensor(0.0, device=self.device)
        if self.ref_data is not None:
            derivs_d = _compute_derivatives(
                self.model, self.ref_data['r'], self.ref_data['z'])
            L_data = (
                ((derivs_d['nF'] - self.ref_data['nF']) / N_REF)**2
                + ((derivs_d['nSF6'] - self.ref_data['nSF6']) / N_REF)**2
                + ((derivs_d['Te'] - self.ref_data['Te']) / TE_REF)**2
            ).mean()

        # ── Total loss ──
        loss = (cfg.w_pde_F * L_pde_F
                + cfg.w_pde_SF6 * L_pde_SF6
                + cfg.w_pde_Te * L_pde_Te
                + L_bc
                + cfg.w_data * L_data)

        # Guard against NaN (stiff chemistry can produce it)
        if not torch.isfinite(loss):
            return {k: float('nan') for k in
                    ['loss', 'pde_F', 'pde_SF6', 'pde_Te', 'bc', 'data']}

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimiser.step()
        self.scheduler.step()

        return {
            'loss': loss.item(),
            'pde_F': L_pde_F.item(),
            'pde_SF6': L_pde_SF6.item(),
            'pde_Te': L_pde_Te.item(),
            'bc': L_bc.item(),
            'data': L_data.item(),
        }

    def train(self, verbose: bool = True) -> list:
        """Full training loop."""
        history = []
        for epoch in range(self.cfg.n_epochs):
            metrics = self.train_step()
            history.append(metrics)

            if verbose and epoch % max(self.cfg.n_epochs // 20, 1) == 0:
                lr = self.optimiser.param_groups[0]['lr']
                print(f"  [{epoch:5d}/{self.cfg.n_epochs}] "
                      f"loss={metrics['loss']:.3e}  "
                      f"F={metrics['pde_F']:.2e}  "
                      f"SF6={metrics['pde_SF6']:.2e}  "
                      f"Te={metrics['pde_Te']:.2e}  "
                      f"bc={metrics['bc']:.2e}  "
                      f"data={metrics['data']:.2e}  "
                      f"lr={lr:.1e}")
        return history

    def predict(self, r: np.ndarray, z: np.ndarray
                ) -> Dict[str, np.ndarray]:
        """Evaluate the trained PINN on arbitrary (r, z) points.

        Returns physical (un-normalised) fields.
        """
        self.model.eval()
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        rz = torch.stack([r_t / R_REF, z_t / Z_REF], dim=-1)

        with torch.no_grad():
            out = self.model(rz).cpu().numpy()

        return {
            'nF':   out[:, 0] * N_REF,
            'nSF6': out[:, 1] * N_REF,
            'Te':   out[:, 2] * TE_REF,
        }


# ═══════════════════════════════════════════════════════════════════
# 11. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def train_pinn(fd_result: Optional[dict] = None,
               cfg: Optional[PINNConfig] = None,
               verbose: bool = True) -> Tuple[TELPINNModel, list]:
    """End-to-end training entry point.

    Parameters
    ----------
    fd_result : dict, optional
        Reference solution from TELSolver.solve() for hybrid
        PINN training (physics + data losses). If None, trains
        purely from PDE residuals (harder, slower convergence).
    cfg : PINNConfig, optional
        Training configuration. Defaults to standard settings.
    verbose : bool
        Print progress.

    Returns
    -------
    model : TELPINNModel
        Trained neural network.
    history : list of dicts
        Per-epoch loss components.

    Example
    -------
    >>> from solver import TELSolver
    >>> fd = TELSolver(P_rf=700, p_mTorr=10).solve(verbose=False)
    >>> model, hist = train_pinn(fd_result=fd)
    >>> # Evaluate anywhere in the domain:
    >>> r = np.linspace(0, 0.105, 100)
    >>> z = np.full(100, 0.025)
    >>> fields = trainer.predict(r, z)
    >>> print(fields['nF'])   # F density profile at z = 25mm
    """
    if cfg is None:
        cfg = PINNConfig()

    if verbose:
        dev = cfg.device
        print(f"{'='*60}")
        print(f"  TEL PINN Solver — Physics-Informed Training")
        print(f"{'='*60}")
        print(f"  Device: {dev}")
        print(f"  Collocation: {cfg.n_interior} interior, "
              f"{cfg.n_boundary} per wall")
        print(f"  Epochs: {cfg.n_epochs}, lr={cfg.lr}")
        print(f"  Data loss: {'YES (hybrid)' if fd_result else 'NO (pure PINN)'}")
        print()

    model = TELPINNModel()
    trainer = TELPINNTrainer(model, cfg)

    if fd_result is not None:
        trainer.load_reference(fd_result)

    history = trainer.train(verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training complete. Final loss: {history[-1]['loss']:.3e}")
        print(f"{'='*60}")

    return model, history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TEL PINN Solver')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--hybrid', action='store_true',
                        help='Use FD reference for hybrid training')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    cfg = PINNConfig(n_epochs=args.epochs, device=args.device)

    fd_result = None
    if args.hybrid:
        print("Running FD solver for reference data...")
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from solver import TELSolver
        fd_result = TELSolver(P_rf=700, p_mTorr=10).solve(verbose=True)

    model, history = train_pinn(fd_result=fd_result, cfg=cfg)

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg),
        'history': history,
    }, os.path.join(os.path.dirname(__file__), '..', 'results',
                    'pinn_solver.pt'))
    print("Saved results/pinn_solver.pt")
