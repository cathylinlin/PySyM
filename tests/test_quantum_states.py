import pytest
import numpy as np
from PySymmetry.phys.quantum.states import (
    Ket,
    Bra,
    StateVector,
    DensityMatrix,
    basis_state,
    bell_state,
    w_state,
    ghz_state,
    tensor_product,
    superposition,
)


class TestKet:
    def test_ket_from_array(self):
        vec = np.array([1.0, 0.0], dtype=complex)
        ket = Ket(vec)
        assert np.allclose(ket.vector, vec)

    def test_ket_from_int(self):
        ket = Ket(2)
        assert ket.dimension == 2
        assert ket[0] == 1.0

    def test_ket_labels(self):
        ket = Ket("0")
        assert np.allclose(ket.vector, [1, 0])
        ket = Ket("1")
        assert np.allclose(ket.vector, [0, 1])

    def test_ket_norm(self):
        vec = np.array([1, 1], dtype=complex) / np.sqrt(2)
        ket = Ket(vec)
        assert np.isclose(ket.norm(), 1.0)

    def test_ket_normalize(self):
        vec = np.array([2.0, 0.0], dtype=complex)
        ket = Ket(vec)
        normalized = ket.normalize()
        assert np.isclose(normalized.norm(), 1.0)

    def test_ket_copy(self):
        vec = np.array([1.0, 0.0], dtype=complex)
        ket = Ket(vec)
        copied = ket.copy()
        assert np.allclose(ket.vector, copied.vector)

    def test_ket_add(self):
        ket0 = Ket("0")
        ket1 = Ket("1")
        result = ket0 + ket1
        assert result.dimension == 2

    def test_ket_scalar_mul(self):
        ket = Ket("0")
        result = ket * 2
        assert np.allclose(result.vector, [2, 0])

    def test_ket_neg(self):
        ket = Ket("0")
        result = -ket
        assert np.allclose(result.vector, [-1, 0])


class TestBra:
    def test_bra_from_array(self):
        vec = np.array([1.0, 0.0], dtype=complex)
        bra = Bra(vec)
        assert np.allclose(bra.vector, vec)

    def test_bra_from_ket(self):
        ket = Ket(np.array([1.0, 0.0], dtype=complex))
        bra = Bra(ket)
        assert np.allclose(bra.vector, [1, 0])

    def test_bra_ket_conversion(self):
        vec = np.array([1.0, 0.0], dtype=complex)
        ket = Ket(vec)
        bra = Bra(ket)
        assert np.allclose(bra.ket.vector, vec)


class TestStateVector:
    def test_statevector_creation(self):
        vec = np.array([1, 0], dtype=complex)
        sv = StateVector(vec)
        assert sv.dimension == 2

    def test_statevector_invalid_dim(self):
        vec = np.array([[1, 0], [0, 1]], dtype=complex)
        with pytest.raises(ValueError):
            StateVector(vec)

    def test_statevector_probabilities(self):
        vec = np.array([1, 0], dtype=complex)
        sv = StateVector(vec)
        probs = sv.probabilities
        assert np.allclose(probs, [1, 0])

    def test_statevector_normalize(self):
        vec = np.array([2.0, 0.0], dtype=complex)
        sv = StateVector(vec)
        normalized = sv.normalize()
        assert np.isclose(normalized.norm(), 1.0)

    def test_statevector_measure(self):
        vec = np.array([1, 0], dtype=complex)
        sv = StateVector(vec)
        idx, prob = sv.measure()
        assert idx == 0
        assert prob == 1.0


class TestDensityMatrix:
    def test_density_matrix_from_ket(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        dm = DensityMatrix(ket)
        assert dm.dimension == 2
        assert dm.is_pure == True

    def test_density_matrix_from_vector(self):
        vec = np.array([1, 0], dtype=complex)
        dm = DensityMatrix(vec)
        assert dm.dimension == 2

    def test_density_matrix_from_array(self):
        matrix = np.array([[1, 0], [0, 0]], dtype=complex)
        dm = DensityMatrix(matrix)
        assert dm.dimension == 2

    def test_density_matrix_purity(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        dm = DensityMatrix(ket)
        assert np.isclose(dm.purity, 1.0)

    def test_density_matrix_entropy_pure(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        dm = DensityMatrix(ket)
        assert np.isclose(dm.entropy(), 0.0)

    def test_density_matrix_normalize(self):
        matrix = np.array([[2, 0], [0, 2]], dtype=complex)
        dm = DensityMatrix(matrix)
        normalized = dm.normalize()
        assert np.isclose(np.trace(normalized.matrix), 1.0)

    def test_density_matrix_expectation(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        dm = DensityMatrix(ket)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        exp = dm.expectation(sigma_z)
        assert np.isclose(exp, 1.0)

    def test_density_matrix_variance(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        dm = DensityMatrix(ket)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        var = dm.variance(sigma_z)
        assert np.isclose(var, 0.0)

    def test_density_matrix_fidelity(self):
        ket0 = Ket(np.array([1, 0], dtype=complex))
        ket1 = Ket(np.array([0, 1], dtype=complex))
        dm0 = DensityMatrix(ket0)
        dm1 = DensityMatrix(ket1)
        fidelity = dm0.fidelity(dm1)
        assert fidelity < 0.01


class TestHelperFunctions:
    def test_basis_state(self):
        ket = basis_state(0, 2)
        assert np.allclose(ket.vector, [1, 0])
        ket = basis_state(1, 2)
        assert np.allclose(ket.vector, [0, 1])

    def test_bell_state_0(self):
        ket = bell_state(0)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(ket.vector, expected)

    def test_bell_state_1(self):
        ket = bell_state(1)
        expected = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
        assert np.allclose(ket.vector, expected)

    def test_w_state(self):
        ket = w_state(2)
        vec = ket.vector
        probs = np.abs(vec) ** 2
        assert np.isclose(np.sum(probs), 1.0)
        nonzero_indices = np.where(probs > 0)[0]
        assert len(nonzero_indices) == 2

    def test_ghz_state(self):
        ket = ghz_state(2)
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        assert np.allclose(ket.vector, expected)

    def test_tensor_product(self):
        ket0 = Ket(np.array([1, 0], dtype=complex))
        ket1 = Ket(np.array([0, 1], dtype=complex))
        result = tensor_product(ket0, ket1)
        assert result.dimension == 4

    def test_superposition(self):
        ket0 = Ket(np.array([1, 0], dtype=complex))
        ket1 = Ket(np.array([0, 1], dtype=complex))
        result = superposition((ket0, 1 / np.sqrt(2)), (ket1, 1 / np.sqrt(2)))
        assert np.isclose(result.norm(), 1.0)
