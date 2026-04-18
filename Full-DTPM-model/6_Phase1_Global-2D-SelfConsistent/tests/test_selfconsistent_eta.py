"""Unit tests for self-consistent coupling efficiency (m01 circuit model).

These tests target the invariants that an absolute-magnitude benchmark
depends on:

1. eta is NOT a free parameter — it is derived from R_coil and R_plasma.
2. In the limit R_coil = 0, eta -> 1 (all RF power coupled to plasma).
3. In the limit R_plasma = 0, eta -> 0 (no plasma -> no coupling).
4. I_peak and R_plasma are consistent: for a given P_rf, scaling R_plasma
   up by a factor f scales I_peak down by sqrt(1 + f * R_plasma / R_coil).
5. update_circuit_from_Pabs is idempotent at the fixed point.
"""
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'src')))

import pytest
from dtpm.modules.m01_circuit import (
    compute_coil_current,
    update_circuit_from_Pabs,
    R_PLASMA_INITIAL_GUESS,
)


class TestLimitCases:
    def test_eta_unity_when_coil_lossless(self):
        """R_coil = 0 should give eta = 1 (no coil losses)."""
        I, eta = compute_coil_current(P_rf=1000.0, R_coil=0.0, R_plasma=5.0)
        assert eta == pytest.approx(1.0, rel=1e-12)
        assert I == pytest.approx((2 * 1000.0 / 5.0) ** 0.5, rel=1e-12)

    def test_eta_zero_when_no_plasma_loading(self):
        """R_plasma = 0 should give eta = 0 (all power dissipated in coil)."""
        I, eta = compute_coil_current(P_rf=1000.0, R_coil=0.8, R_plasma=0.0)
        assert eta == pytest.approx(0.0, abs=1e-12)
        assert I == pytest.approx((2 * 1000.0 / 0.8) ** 0.5, rel=1e-12)

    def test_raise_on_zero_total_resistance(self):
        with pytest.raises(ValueError):
            compute_coil_current(P_rf=1000.0, R_coil=0.0, R_plasma=0.0)


class TestSelfConsistency:
    def test_eta_in_physical_window(self):
        """TEL-class conditions (R_coil=0.8, R_plasma=5) should give eta in [0.6, 0.95]."""
        _, eta = compute_coil_current(
            P_rf=1000.0, R_coil=0.8, R_plasma=R_PLASMA_INITIAL_GUESS)
        assert 0.6 <= eta <= 0.95, f"eta={eta} outside physical window"

    def test_update_from_pabs_fixed_point(self):
        """If P_abs = eta * P_rf exactly, the update is a fixed point."""
        P_rf = 1000.0
        R_coil = 0.8
        R_plasma_true = 5.0
        # Correct I_peak for these values
        I_peak = (2 * P_rf / (R_coil + R_plasma_true)) ** 0.5
        P_abs = 0.5 * I_peak**2 * R_plasma_true  # = eta * P_rf exactly

        upd = update_circuit_from_Pabs(
            P_rf=P_rf, R_coil=R_coil, I_peak_prev=I_peak, P_abs=P_abs)

        assert upd['R_plasma'] == pytest.approx(R_plasma_true, rel=1e-12)
        assert upd['I_peak'] == pytest.approx(I_peak, rel=1e-12)
        expected_eta = R_plasma_true / (R_coil + R_plasma_true)
        assert upd['eta'] == pytest.approx(expected_eta, rel=1e-12)

    def test_update_from_pabs_converges(self):
        """Start with a seed R_plasma; one update should move toward truth."""
        P_rf = 1000.0
        R_coil = 0.8
        R_plasma_true = 7.5
        # Seed: off by factor 2
        R_plasma_seed = 3.75
        I_seed = (2 * P_rf / (R_coil + R_plasma_seed)) ** 0.5

        # True P_abs at the seed I_peak if the real R_plasma is 7.5
        P_abs = 0.5 * I_seed**2 * R_plasma_true

        upd = update_circuit_from_Pabs(
            P_rf=P_rf, R_coil=R_coil, I_peak_prev=I_seed, P_abs=P_abs)
        assert upd['R_plasma'] == pytest.approx(R_plasma_true, rel=1e-10)


class TestEtaIsNotADial:
    """Verify eta cannot be prescribed via any back-door."""

    def test_eta_depends_on_R_plasma(self):
        """Same P_rf + R_coil, different R_plasma -> different eta."""
        _, eta_1 = compute_coil_current(1000.0, 0.8, 2.0)
        _, eta_2 = compute_coil_current(1000.0, 0.8, 8.0)
        assert eta_1 < eta_2
        assert eta_2 - eta_1 > 0.1  # materially different

    def test_eta_invariant_to_P_rf(self):
        """Scaling P_rf at fixed R_coil, R_plasma must NOT change eta."""
        _, eta_1 = compute_coil_current(500.0, 0.8, 5.0)
        _, eta_2 = compute_coil_current(1000.0, 0.8, 5.0)
        _, eta_3 = compute_coil_current(2000.0, 0.8, 5.0)
        assert eta_1 == pytest.approx(eta_2, rel=1e-12)
        assert eta_2 == pytest.approx(eta_3, rel=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
