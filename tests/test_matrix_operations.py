import pytest
import numpy as np
from PySymmetry.core.matrix.operations import MatrixOperations


class TestMatrixOperations:
    def test_add(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = MatrixOperations.add(a, b)
        expected = np.array([[6, 8], [10, 12]])
        assert np.allclose(result, expected)

    def test_add_shape_mismatch(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            MatrixOperations.add(a, b)

    def test_subtract(self):
        a = np.array([[5, 6], [7, 8]])
        b = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.subtract(a, b)
        expected = np.array([[4, 4], [4, 4]])
        assert np.allclose(result, expected)

    def test_multiply(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = MatrixOperations.multiply(a, b)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result, expected)

    def test_multiply_dimension_mismatch(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2, 3], [4, 5, 6]])
        result = MatrixOperations.multiply(a, b)
        assert result.shape == (2, 3)

    def test_elementwise_multiply(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = MatrixOperations.elementwise_multiply(a, b)
        expected = np.array([[5, 12], [21, 32]])
        assert np.allclose(result, expected)

    def test_elementwise_divide(self):
        a = np.array([[6, 8], [10, 12]])
        b = np.array([[2, 4], [5, 6]])
        result = MatrixOperations.elementwise_divide(a, b)
        expected = np.array([[3, 2], [2, 2]])
        assert np.allclose(result, expected)

    def test_transpose(self):
        m = np.array([[1, 2, 3], [4, 5, 6]])
        result = MatrixOperations.transpose(m)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        assert np.allclose(result, expected)

    def test_conjugate_transpose(self):
        m = np.array([[1 + 1j, 2], [3, 4 + 1j]])
        result = MatrixOperations.conjugate_transpose(m)
        expected = np.array([[1 - 1j, 3], [2, 4 - 1j]])
        assert np.allclose(result, expected)

    def test_kronecker_product(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[0, 1], [1, 0]])
        result = MatrixOperations.kronecker_product(a, b)
        expected = np.array([[0, 1, 0, 2], [1, 0, 2, 0], [0, 3, 0, 4], [3, 0, 4, 0]])
        assert np.allclose(result, expected)

    def test_hadamard_product(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = MatrixOperations.hadamard_product(a, b)
        expected = np.array([[5, 12], [21, 32]])
        assert np.allclose(result, expected)

    def test_outer_product(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = MatrixOperations.outer_product(a, b)
        expected = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])
        assert np.allclose(result, expected)

    def test_inner_product(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = MatrixOperations.inner_product(a, b)
        assert result == 32

    def test_dot_product(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = MatrixOperations.dot_product(a, b)
        assert result == 32

    def test_power(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.power(m, 3)
        expected = m @ m @ m
        assert np.allclose(result, expected)

    def test_power_not_square(self):
        m = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            MatrixOperations.power(m, 2)

    def test_inverse(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.inverse(m)
        expected = np.linalg.inv(m)
        assert np.allclose(result, expected)

    def test_inverse_singular(self):
        m = np.array([[1, 2], [2, 4]])
        with pytest.raises(ValueError):
            MatrixOperations.inverse(m)

    def test_pseudo_inverse(self):
        m = np.array([[1, 2], [3, 4], [5, 6]])
        result = MatrixOperations.pseudo_inverse(m)
        expected = np.linalg.pinv(m)
        assert np.allclose(result, expected)

    def test_solve(self):
        A = np.array([[1, 1], [3, 1]])
        b = np.array([3, 7])
        x = MatrixOperations.solve(A, b)
        assert np.allclose(A @ x, b)

    def test_solve_singular(self):
        A = np.array([[1, 2], [2, 4]])
        b = np.array([3, 6])
        with pytest.raises(ValueError):
            MatrixOperations.solve(A, b)

    def test_least_squares(self):
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 3])
        x = MatrixOperations.least_squares(A, b)
        assert np.allclose(A @ x, b, atol=1e-10)

    def test_matrix_exponential(self):
        m = np.array([[0, -np.pi], [np.pi, 0]])
        result = MatrixOperations.matrix_exponential(m)
        assert result.shape == (2, 2)

    def test_matrix_logarithm(self):
        m = np.array([[1, 0], [0, 2]])
        result = MatrixOperations.matrix_logarithm(m)
        assert result.shape == (2, 2)

    def test_matrix_sqrt(self):
        m = np.array([[4, 0], [0, 9]])
        result = MatrixOperations.matrix_sqrt(m)
        expected = np.array([[2, 0], [0, 3]])
        assert np.allclose(result, expected)

    def test_trace(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = MatrixOperations.trace(m)
        assert result == 15

    def test_determinant(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.determinant(m)
        assert np.isclose(result, -2)

    def test_rank(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = MatrixOperations.rank(m)
        assert result == 2

    def test_norm_fro(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.norm(m, "fro")
        expected = np.sqrt(30)
        assert np.allclose(result, expected)

    def test_norm_1(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.norm(m, 1)
        assert result == 6

    def test_condition_number(self):
        m = np.array([[1, 0], [0, 2]])
        result = MatrixOperations.condition_number(m)
        assert result == 2

    def test_concatenate_axis_0(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6]])
        result = MatrixOperations.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        assert np.allclose(result, expected)

    def test_concatenate_axis_1(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5], [6]])
        result = MatrixOperations.concatenate([a, b], axis=1)
        expected = np.array([[1, 2, 5], [3, 4, 6]])
        assert np.allclose(result, expected)

    def test_concatenate_empty(self):
        with pytest.raises(ValueError):
            MatrixOperations.concatenate([], axis=0)

    def test_stack(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = MatrixOperations.stack([a, b], axis=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(result, expected)

    def test_split(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        result = MatrixOperations.split(m, 2, axis=0)
        assert len(result) == 2

    def test_tile(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.tile(m, (2, 3))
        assert result.shape == (4, 6)

    def test_repeat(self):
        m = np.array([[1, 2], [3, 4]])
        result = MatrixOperations.repeat(m, 2, axis=0)
        assert result.shape == (4, 2)
