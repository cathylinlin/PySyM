import pytest
import numpy as np
from PySymmetry.abstract_phys.utils.spherical_harmonics import (
    SphericalHarmonics,
    GauntCoefficient,
)


class TestSphericalHarmonics:
    """Test suite for SphericalHarmonics class."""

    def test_creation(self):
        """Test basic creation."""
        sh = SphericalHarmonics()
        assert sh is not None

    def test_compute_l0_m0(self):
        """Test Y_0^0 (constant)."""
        result = SphericalHarmonics.compute(0, 0, np.pi / 4, np.pi / 3)
        expected = np.sqrt(1 / (4 * np.pi))
        assert np.isclose(result, expected)

    def test_compute_l1_m0(self):
        """Test Y_1^0 (cosθ dependent)."""
        theta = np.pi / 2
        result = SphericalHarmonics.compute(1, 0, theta, 0)
        expected = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        assert np.isclose(result, expected)

    def test_compute_invalid_l(self):
        """Test negative l raises error."""
        with pytest.raises(ValueError):
            SphericalHarmonics.compute(-1, 0, np.pi / 4, 0)

    def test_compute_invalid_m(self):
        """Test |m| > l raises error."""
        with pytest.raises(ValueError):
            SphericalHarmonics.compute(1, 2, np.pi / 4, 0)

    def test_real_spherical_harmonic_l0_m0(self):
        """Test real Y_0^0."""
        result = SphericalHarmonics.real_spherical_harmonic(0, 0, np.pi / 4, 0)
        expected = np.sqrt(1 / (4 * np.pi))
        assert np.isclose(result, expected)

    def test_real_spherical_harmonic_l1_m0(self):
        """Test real Y_1^0."""
        theta = np.pi / 2
        result = SphericalHarmonics.real_spherical_harmonic(1, 0, theta, 0)
        expected = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        assert np.isclose(result, expected)

    def test_real_spherical_harmonic_l1_m1(self):
        """Test real Y_1^1."""
        theta = np.pi / 2
        phi = 0
        result = SphericalHarmonics.real_spherical_harmonic(1, 1, theta, phi)
        assert isinstance(result, float)


class TestGauntCoefficient:
    """Test Gaunt coefficient computation."""

    def test_creation(self):
        """Test basic creation."""
        gc = GauntCoefficient()
        assert gc is not None

    def test_gaunt_basic(self):
        """Test basic Gaunt coefficient."""
        gc = GauntCoefficient()
        result = gc.compute(0, 0, 0, 0, 0, 0)
        assert isinstance(result, (int, float))

    def test_gaunt_selection_rules(self):
        """Test Gaunt coefficient selection rules."""
        gc = GauntCoefficient()
        result = gc.compute(1, 1, 1, 1, 1, -2)
        assert result == 0


class TestRealSphericalHarmonics:
    """Test real spherical harmonics properties."""

    def test_real_values_are_real(self):
        """Test real spherical harmonics return real values."""
        for l in range(3):
            for m in range(-l, l + 1):
                result = SphericalHarmonics.real_spherical_harmonic(
                    l, m, np.pi / 4, np.pi / 4
                )
                assert np.isreal(result) or isinstance(result, float)

    def test_m0_real(self):
        """Test m=0 harmonics are real."""
        for l in range(3):
            result = SphericalHarmonics.real_spherical_harmonic(l, 0, np.pi / 4, 0)
            imag = np.imag(result)
            assert np.isclose(imag, 0, atol=1e-10)
