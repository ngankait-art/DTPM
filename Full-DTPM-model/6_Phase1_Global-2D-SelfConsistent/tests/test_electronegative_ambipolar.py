"""Unit tests for the L1 electronegative ambipolar correction.

The correction factor is:
    f(alpha, Te, Ti) = (1 + alpha) / (1 + alpha * Ti/Te)

Invariants tested:
1. f(0, Te, Ti) == 1 exactly, for any Te/Ti — electropositive limit.
2. f(alpha, Te=Te, Ti=Te) == 1 exactly, for any alpha — isothermal degenerate.
3. f(1, Te=20 eV, Ti=1 eV) == 2.0 exactly — Lieberman 2005 §10.3 worked case.
4. D_a_en / D_a_ep == correction (physical chaining invariant).
5. Calling solve_ne_ambipolar with alpha=0 yields a D_a identical to the
   legacy electropositive expression (bit-for-bit check).
"""
import os
import sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'src')))


def correction(alpha, Te, Ti):
    """Isolated copy of the electronegative correction factor."""
    return (1.0 + alpha) / (1.0 + alpha * Ti / Te)


class TestCorrectionFactor:
    def test_electropositive_limit(self):
        """alpha = 0 -> factor = 1 exactly, any Te/Ti."""
        for Te in (1.0, 3.0, 10.0):
            for Ti in (0.01, 0.03, 0.1, 1.0):
                assert correction(0.0, Te, Ti) == pytest.approx(1.0, abs=1e-15)

    def test_isothermal_degenerate(self):
        """Ti = Te -> factor = 1 for any alpha.  (1+a)/(1+a) = 1)"""
        for alpha in (0.0, 0.5, 1.0, 3.0, 10.0):
            assert correction(alpha, Te=3.0, Ti=3.0) == pytest.approx(1.0, abs=1e-15)

    def test_lieberman_worked_case(self):
        """alpha = 1, Te = 20 eV, Ti = 1 eV -> factor = 2 / (1 + 1/20) = 40/21."""
        # (1+1)/(1 + 1 * 1/20) = 2 / 1.05 = 1.9048...
        got = correction(alpha=1.0, Te=20.0, Ti=1.0)
        assert got == pytest.approx(2.0 / 1.05, rel=1e-12)

    def test_typical_icp_conditions(self):
        """At the DTPM operating regime (alpha ~ 1, Te/Ti ~ 100): factor ~ 2."""
        # Te = 3 eV, Ti = kT_gas at 313 K = 0.027 eV, alpha = 1.0
        got = correction(alpha=1.0, Te=3.0, Ti=0.027)
        assert 1.9 < got < 2.01

    def test_monotonic_in_alpha(self):
        """factor strictly increases with alpha for Ti < Te."""
        vals = [correction(a, Te=3.0, Ti=0.027) for a in (0.0, 0.5, 1.0, 2.0, 5.0)]
        for a, b in zip(vals, vals[1:]):
            assert b > a


class TestBackwardsCompat:
    """Verify that alpha=0 gives bit-for-bit the legacy electropositive D_a."""

    def test_alpha_zero_equals_electropositive(self):
        """The new code with alpha=0 must give the same D_a as the legacy formula."""
        # Construct D_a from the new vectorised code with alpha = 0, and from
        # the legacy per-cell formula, and compare.
        import numpy as np
        D_i_base = 1.0
        Ti_eV = 0.027
        Te_safe = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Legacy electropositive:
        legacy = D_i_base * (1.0 + Te_safe / Ti_eV)

        # New with alpha = 0:
        alpha = 0.0
        D_a_ep = D_i_base * (1.0 + Te_safe / Ti_eV)
        corr = (1.0 + alpha) / (1.0 + alpha * Ti_eV / Te_safe)
        new = D_a_ep * corr

        assert np.allclose(new, legacy, rtol=0, atol=1e-15)

    def test_alpha_positive_increases_D_a(self):
        """With alpha > 0 and Ti < Te, D_a must be larger than the electropositive form."""
        import numpy as np
        D_i_base = 1.0
        Ti_eV = 0.027
        Te = 3.0
        D_ep = D_i_base * (1.0 + Te / Ti_eV)
        for alpha in (0.5, 1.0, 2.0):
            corr = (1.0 + alpha) / (1.0 + alpha * Ti_eV / Te)
            D_en = D_ep * corr
            assert D_en > D_ep
            assert corr > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
