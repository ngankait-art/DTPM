"""Unit tests for the BC-map floor-annulus fix (Stage A0 of the D6 diagnostic).

The prior build_geometry_mask() labelled the entire j=0 row as BC_WAFER, which
mis-attributed the Al pedestal-top / chamber-floor annulus (R_wafer < r <=
R_proc) as silicon. These tests pin down the corrected behaviour:

1. r <= R_wafer  -> BC_WAFER (silicon)
2. r >  R_wafer  -> BC_AL_TOP (aluminium)
3. Default R_wafer = 0.075 m matches A_wafer_eff = pi*0.075^2 = 0.01767 m^2.
4. Total bottom-boundary cell count equals Nr (one cell per radial index at j=0).
5. The fix never touches BC labels at j > 0.
"""
import numpy as np
import pytest

from dtpm.core.geometry import (
    build_geometry_mask,
    BC_WAFER, BC_AL_TOP, BC_QUARTZ, BC_AL_SIDE, BC_INTERIOR, BC_INACTIVE,
)
from dtpm.core.mesh import Mesh2D


def _build_mesh():
    """Standard TEL mesh used by the production pipeline."""
    R_proc = 0.105
    L_total = 0.050 + 0.002 + 0.150   # L_proc + L_apt + L_icp
    return Mesh2D(R=R_proc, L=L_total, Nr=50, Nz=80, beta_r=1.2, beta_z=1.0)


def test_floor_annulus_is_aluminium_by_default():
    """With default R_wafer=0.075, r > 0.075 at j=0 must be BC_AL_TOP."""
    mesh = _build_mesh()
    _, bc = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202)
    j = 0
    for i in range(mesh.Nr):
        r = mesh.rc[i]
        if r <= 0.075:
            assert bc[i, j] == BC_WAFER, (
                f"r={r:.4f} (<=0.075) should be BC_WAFER, got {bc[i, j]}")
        else:
            assert bc[i, j] == BC_AL_TOP, (
                f"r={r:.4f} (>0.075) should be BC_AL_TOP, got {bc[i, j]}")


def test_custom_R_wafer_splits_correctly():
    """Explicit R_wafer=0.05 must move the split radius to r=0.05."""
    mesh = _build_mesh()
    _, bc = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202,
                                R_wafer=0.05)
    j = 0
    wafer_max_r = max(mesh.rc[i] for i in range(mesh.Nr) if bc[i, j] == BC_WAFER)
    al_min_r = min(mesh.rc[i] for i in range(mesh.Nr) if bc[i, j] == BC_AL_TOP)
    assert wafer_max_r <= 0.05 + 1e-9
    assert al_min_r > 0.05


def test_bottom_row_is_fully_classified():
    """Every cell at j=0 must be either BC_WAFER or BC_AL_TOP (no BC_INACTIVE)."""
    mesh = _build_mesh()
    _, bc = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202)
    labels = set(int(bc[i, 0]) for i in range(mesh.Nr))
    assert labels.issubset({BC_WAFER, BC_AL_TOP}), (
        f"j=0 row has unexpected labels: {labels}")


def test_floor_annulus_cell_count_is_nontrivial():
    """With R_wafer=0.075 and R_proc=0.105 on a 50-cell radial grid, the
    annulus must account for a substantial fraction of the bottom row."""
    mesh = _build_mesh()
    _, bc = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202)
    n_wafer = int(np.sum(bc[:, 0] == BC_WAFER))
    n_al_floor = int(np.sum(bc[:, 0] == BC_AL_TOP))
    assert n_wafer + n_al_floor == mesh.Nr
    # Annulus area fraction ~ (R_proc^2 - R_wafer^2) / R_proc^2 ~ 0.49.
    # With beta_r=1.2 mesh stretching the cell count isn't a linear map, but
    # the annulus must have at least 5 cells out of 50 to be physically
    # meaningful.
    assert n_al_floor >= 5, f"Annulus cell count suspiciously small: {n_al_floor}"


def test_fix_does_not_affect_j_gt_0():
    """The floor-annulus fix only touches j=0; j>0 labels must be unchanged
    from the old map produced by a call with the legacy default."""
    mesh = _build_mesh()
    _, bc_new = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                    z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202,
                                    R_wafer=0.075)
    # Emulate old behaviour: R_wafer = R_proc forces the entire j=0 row to
    # BC_WAFER (since r <= R_proc always holds for active cells).
    _, bc_old = build_geometry_mask(mesh, R_icp=0.038, R_proc=0.105,
                                    z_apt_bot=0.050, z_apt_top=0.052, z_top=0.202,
                                    R_wafer=0.105)
    assert np.array_equal(bc_new[:, 1:], bc_old[:, 1:])
    assert not np.array_equal(bc_new[:, 0], bc_old[:, 0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
