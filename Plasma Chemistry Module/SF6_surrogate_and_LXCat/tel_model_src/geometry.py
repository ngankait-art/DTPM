"""
TEL Reactor Geometry Definition

Defines the T-shaped axisymmetric gas volume for the TEL ICP etcher
using a boolean mask on a structured (r,z) mesh.

Physical layout (half-domain, r >= 0):

    r=0        R_icp=38mm   R_proc=105mm
     |            |              |
     |  ICP       | (quartz      |
     |  SOURCE    |  wall)       |  z = z_top (dielectric window)
     |  (active)  |              |
     |            |              |
     |............|..............|  z = z_apt_top
     |  aperture  | SHOULDER     |
     |  channel   | (inactive)   |  z = z_apt_bot
     |            |              |
     |  PROCESSING REGION        |
     |  (active)                 |  z = z_wafer (Si wafer)
     |___________________________|

Boundary types:
    AXIS      — r = 0 (Neumann symmetry)
    QUARTZ    — r = R_icp, ICP region (SiO2)
    WINDOW    — z = z_top (dielectric)
    AL_SIDE   — r = R_proc, processing (aluminium)
    AL_TOP    — z = z_apt_bot, r > R_icp (aperture plate underside)
    WAFER     — z = z_wafer (silicon)
    SHOULDER  — r = R_icp, aperture height (Al)
"""
import numpy as np

# Boundary type constants
BC_INTERIOR  = 0
BC_AXIS      = 1
BC_QUARTZ    = 2
BC_WINDOW    = 3
BC_AL_SIDE   = 4
BC_AL_TOP    = 5
BC_WAFER     = 6
BC_SHOULDER  = 7
BC_INACTIVE  = -1


def build_geometry_mask(mesh, R_icp, R_proc, z_apt_bot, z_apt_top, z_top):
    """Build boolean mask and boundary classification for the TEL geometry.
    
    Parameters
    ----------
    mesh : Mesh2D
        Structured (r,z) mesh covering [0, R_proc] x [0, z_top].
    R_icp : float
        ICP source radius (m).
    R_proc : float
        Processing region radius (m).
    z_apt_bot : float
        Bottom of aperture plate = top of processing region (m).
    z_apt_top : float
        Top of aperture plate = bottom of ICP source (m).
    z_top : float
        Top of ICP source = dielectric window (m).
    
    Returns
    -------
    inside : ndarray (Nr, Nz), bool
        True for cells inside the gas volume.
    bc_type : ndarray (Nr, Nz), int
        Boundary classification for each cell.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    inside = np.zeros((Nr, Nz), dtype=bool)
    bc_type = np.full((Nr, Nz), BC_INACTIVE, dtype=int)

    # Step 1: Mark active cells
    for i in range(Nr):
        r = mesh.rc[i]
        for j in range(Nz):
            z = mesh.zc[j]
            in_icp = (r <= R_icp) and (z >= z_apt_top)
            in_apt = (r <= R_icp) and (z_apt_bot <= z < z_apt_top)
            in_proc = (r <= R_proc) and (z < z_apt_bot)
            if in_icp or in_apt or in_proc:
                inside[i, j] = True
                bc_type[i, j] = BC_INTERIOR

    # Step 2: Classify boundary cells
    for i in range(Nr):
        for j in range(Nz):
            if not inside[i, j]:
                continue
            r, z = mesh.rc[i], mesh.zc[j]

            # Axis of symmetry
            if i == 0:
                bc_type[i, j] = BC_AXIS

            # Radial outer boundary
            if i == Nr - 1 and z < z_apt_bot:
                bc_type[i, j] = BC_AL_SIDE
            elif i < Nr - 1 and not inside[i + 1, j]:
                if z >= z_apt_top:
                    bc_type[i, j] = BC_QUARTZ
                elif z >= z_apt_bot:
                    bc_type[i, j] = BC_SHOULDER
                else:
                    bc_type[i, j] = BC_AL_SIDE

            # Top boundary
            if j == Nz - 1:
                bc_type[i, j] = BC_WINDOW

            # Bottom boundary  
            if j == 0:
                bc_type[i, j] = BC_WAFER

            # Aperture plate underside
            if z < z_apt_bot and r > R_icp:
                if j < Nz - 1 and not inside[i, j + 1]:
                    bc_type[i, j] = BC_AL_TOP

    return inside, bc_type


def build_index_maps(inside):
    """Build mapping between (i,j) grid positions and active-cell flat indices.
    
    Returns
    -------
    ij_to_flat : ndarray (Nr, Nz), int
        Maps (i,j) -> flat index. -1 for inactive cells.
    flat_to_ij : list of (i, j) tuples
        Maps flat index -> (i,j).
    n_active : int
        Number of active cells.
    """
    Nr, Nz = inside.shape
    ij_to_flat = np.full((Nr, Nz), -1, dtype=int)
    flat_to_ij = []
    k = 0
    for i in range(Nr):
        for j in range(Nz):
            if inside[i, j]:
                ij_to_flat[i, j] = k
                flat_to_ij.append((i, j))
                k += 1
    return ij_to_flat, flat_to_ij, k


def count_boundary_cells(bc_type):
    """Count cells of each boundary type for diagnostics."""
    names = {
        BC_INTERIOR: 'Interior', BC_AXIS: 'Axis', BC_QUARTZ: 'Quartz',
        BC_WINDOW: 'Window', BC_AL_SIDE: 'Al side', BC_AL_TOP: 'Al top',
        BC_WAFER: 'Wafer', BC_SHOULDER: 'Shoulder', BC_INACTIVE: 'Inactive'
    }
    counts = {}
    for bc_val, name in names.items():
        n = int(np.sum(bc_type == bc_val))
        if n > 0:
            counts[name] = n
    return counts
