import pytest
import numpy as np
from PySymmetry.abstract_phys.utils.wigner_Dmatrix import (
    WignerSmallD,
)


class TestWignerSmallD:
    """Test suite for WignerSmallD class."""

    def test_creation(self):
        """Test basic creation."""
        wigner_d = WignerSmallD()
        assert wigner_d is not None

    def test_compute_j0(self):
        """Test j=0 returns 1."""
        result = WignerSmallD.compute(0, 0, 0, np.pi / 4)
        assert result == 1.0

    def test_compute_j_half(self):
        """Test spin-1/2 computation."""
        result = WignerSmallD.compute(0.5, 0.5, 0.5, np.pi / 2)
        cos_val = np.cos(np.pi / 4)
        assert np.isclose(result, cos_val)

    def test_compute_invalid_m(self):
        """Test invalid m returns 0."""
        result = WignerSmallD.compute(0.5, 1.5, 0.5, np.pi / 4)
        assert result == 0.0

    def test_compute_j_one(self):
        """Test spin-1 computation."""
        result = WignerSmallD.compute(1, 0, 0, np.pi / 2)
        assert isinstance(result, float)
