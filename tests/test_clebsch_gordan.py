import pytest
import numpy as np
from PySymmetry.abstract_phys.utils.clebsch_gordan import (
    triangle_condition,
    is_valid_jm,
    double_factorial,
    cached_factorial,
    ClebschGordan,
    Wigner3j,
    Wigner6j,
    Wigner9j,
    RacahCoefficient,
)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_triangle_condition_valid(self):
        """Test valid triangle conditions."""
        assert triangle_condition(1, 1, 0) is True
        assert triangle_condition(1, 1, 1) is True
        assert triangle_condition(1, 1, 2) is True
        assert triangle_condition(0.5, 0.5, 1) is True

    def test_triangle_condition_invalid(self):
        """Test invalid triangle conditions."""
        assert triangle_condition(1, 1, 3) is False
        assert triangle_condition(2, 2, 0.5) is False

    def test_is_valid_jm_integer(self):
        """Test valid integer j and m."""
        assert is_valid_jm(1, 0) is True
        assert is_valid_jm(1, 1) is True
        assert is_valid_jm(1, -1) is True

    def test_is_valid_jm_half_integer(self):
        """Test valid half-integer j and m."""
        assert is_valid_jm(0.5, 0.5) is True
        assert is_valid_jm(0.5, -0.5) is True
        assert is_valid_jm(1.5, 0.5) is True

    def test_is_valid_jm_invalid_negative_j(self):
        """Test invalid negative j."""
        assert is_valid_jm(-1, 0) is False

    def test_is_valid_jm_invalid_m_out_of_range(self):
        """Test invalid m out of range."""
        assert is_valid_jm(1, 2) is False
        assert is_valid_jm(1, -2) is False

    def test_double_factorial(self):
        """Test double factorial."""
        assert double_factorial(0) == 1
        assert double_factorial(1) == 1
        assert double_factorial(2) == 2
        assert double_factorial(3) == 3
        assert double_factorial(4) == 8
        assert double_factorial(5) == 15
        assert double_factorial(6) == 48

    def test_cached_factorial(self):
        """Test cached factorial."""
        assert cached_factorial(0) == 1
        assert cached_factorial(5) == 120
        assert cached_factorial(10) == 3628800


class TestClebschGordan:
    """Test suite for ClebschGordan class."""

    def test_creation(self):
        """Test basic creation."""
        cg = ClebschGordan()
        assert cg is not None

    def test_compute_spin_half_coupling(self):
        """Test spin-1/2 coupling to spin-1."""
        cg = ClebschGordan()

        c = cg.compute(0.5, 0.5, 0.5, 0.5, 1.0, 1.0)
        assert np.isclose(c, 1.0)

        c = cg.compute(0.5, 0.5, 0.5, -0.5, 0.0, 0.0)
        assert abs(c) > 0

    def test_compute_invalid_triangle(self):
        """Test invalid triangle condition returns 0."""
        cg = ClebschGordan()
        c = cg.compute(1, 0, 1, 0, 3, 0)
        assert c == 0.0

    def test_compute_invalid_m(self):
        """Test invalid m returns 0."""
        cg = ClebschGordan()
        c = cg.compute(0.5, 0.5, 0.5, 0.5, 1.0, 2.0)
        assert c == 0.0

    def test_coupling_basic(self):
        """Test basic coupling functionality."""
        cg = ClebschGordan()
        c = cg.compute(1.0, 0, 1.0, 0, 1.0, 0)
        assert isinstance(c, float)


class TestWigner3j:
    """Test suite for Wigner 3j symbols."""

    def test_creation(self):
        """Test basic creation."""
        w3j = Wigner3j()
        assert w3j is not None

    def test_selection_rules(self):
        """Test selection rules."""
        w3j = Wigner3j()
        result = w3j.compute(1, 1, 1, 0, 0, 0)
        assert isinstance(result, float)

    def test_symmetry(self):
        """Test 3j symbol symmetry."""
        w3j = Wigner3j()
        r1 = w3j.compute(1, 0, 1, 1, 2, -1)
        r2 = w3j.compute(1, 1, 1, 0, 2, -1)
        assert abs(r1 - r2) < 1e-10 or abs(r1 + r2) < 1e-10


class TestWigner6j:
    """Test suite for Wigner 6j symbols."""

    def test_creation(self):
        """Test basic creation."""
        w6j = Wigner6j()
        assert w6j is not None

    def test_compute(self):
        """Test 6j symbol computation."""
        w6j = Wigner6j()
        result = w6j.compute(1, 1, 1, 0.5, 0.5, 0)
        assert isinstance(result, float)

    def test_selection_rules(self):
        """Test selection rules."""
        w6j = Wigner6j()
        result = w6j.compute(0.5, 0.5, 1, 0.5, 0.5, 1)
        assert isinstance(result, float)


class TestWigner9j:
    """Test suite for Wigner 9j symbols."""

    def test_creation(self):
        """Test basic creation."""
        w9j = Wigner9j()
        assert w9j is not None

    def test_compute(self):
        """Test 9j symbol computation."""
        w9j = Wigner9j()
        result = w9j.compute(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0)
        assert isinstance(result, float)


class TestRacahCoefficient:
    """Test suite for Racah coefficients."""

    def test_creation(self):
        """Test basic creation."""
        racah = RacahCoefficient()
        assert racah is not None

    def test_compute(self):
        """Test Racah coefficient computation."""
        racah = RacahCoefficient()
        result = racah.compute(1, 1, 1, 0.5, 1, 0.5)
        assert isinstance(result, float)
