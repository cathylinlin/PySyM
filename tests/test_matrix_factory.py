import pytest
import numpy as np
from PySymmetry.core.matrix.factory import MatrixFactory


class TestMatrixFactory:
    def test_zeros(self):
        m = MatrixFactory.zeros(3, 4)
        assert m.shape == (3, 4)
        assert np.allclose(m, 0)

    def test_zeros_invalid(self):
        with pytest.raises(ValueError):
            MatrixFactory.zeros(0, 3)
        with pytest.raises(ValueError):
            MatrixFactory.zeros(3, -1)

    def test_ones(self):
        m = MatrixFactory.ones(2, 3)
        assert m.shape == (2, 3)
        assert np.allclose(m, 1)

    def test_identity(self):
        m = MatrixFactory.identity(3)
        assert m.shape == (3, 3)
        assert np.allclose(m, np.eye(3))

    def test_identity_invalid(self):
        with pytest.raises(ValueError):
            MatrixFactory.identity(0)

    def test_random(self):
        m = MatrixFactory.random(3, 3)
        assert m.shape == (3, 3)
        assert np.all((m >= 0) & (m <= 1))

    def test_random_normal(self):
        m = MatrixFactory.random_normal(3, 3, mean=5.0, std=2.0)
        assert m.shape == (3, 3)

    def test_random_normal_invalid_std(self):
        with pytest.raises(ValueError):
            MatrixFactory.random_normal(3, 3, std=-1)

    def test_from_diagonal(self):
        d = np.array([1, 2, 3])
        m = MatrixFactory.from_diagonal(d)
        expected = np.diag(d)
        assert np.allclose(m, expected)

    def test_from_diagonal_invalid(self):
        with pytest.raises(ValueError):
            MatrixFactory.from_diagonal(np.array([[1, 2], [3, 4]]))

    def test_from_list(self):
        data = [[1, 2], [3, 4]]
        m = MatrixFactory.from_list(data)
        expected = np.array(data)
        assert np.allclose(m, expected)

    def test_from_list_empty(self):
        with pytest.raises(ValueError):
            MatrixFactory.from_list([])

    def test_from_function(self):
        m = MatrixFactory.from_function(2, 2, lambda i, j: i + j)
        expected = np.array([[0, 1], [1, 2]])
        assert np.allclose(m, expected)

    def test_toeplitz_invalid(self):
        with pytest.raises(ValueError):
            MatrixFactory.toeplitz(np.array([[1, 2], [3, 4]]))

    def test_circulant(self):
        row = np.array([1, 2, 3])
        m = MatrixFactory.circulant(row)
        assert m.shape == (3, 3)

    def test_vandermonde(self):
        x = np.array([1, 2, 3])
        m = MatrixFactory.vandermonde(x)
        expected = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
        assert np.allclose(m, expected)

    def test_hilbert(self):
        m = MatrixFactory.hilbert(3)
        expected = np.array(
            [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4], [1 / 3, 1 / 4, 1 / 5]]
        )
        assert np.allclose(m, expected)

    def test_invhilbert(self):
        m = MatrixFactory.invhilbert(3)
        assert m.shape == (3, 3)

    def test_pascal_symmetric(self):
        m = MatrixFactory.pascal(3, kind="symmetric")
        expected = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]])
        assert np.allclose(m, expected)

    def test_pascal_lower(self):
        m = MatrixFactory.pascal(3, kind="lower")
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 2, 1]])
        assert np.allclose(m, expected)

    def test_dft(self):
        m = MatrixFactory.dft(4)
        assert m.shape == (4, 4)
        assert np.allclose(m @ m.conj().T, np.eye(4), atol=1e-10)

    def test_rotation_2d(self):
        theta = np.pi / 2
        m = MatrixFactory.rotation_2d(theta)
        expected = np.array([[0, -1], [1, 0]])
        assert np.allclose(m, expected)

    def test_rotation_3d_x(self):
        theta = np.pi
        m = MatrixFactory.rotation_3d("x", theta)
        expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        assert np.allclose(m, expected)

    def test_rotation_3d_y(self):
        theta = np.pi
        m = MatrixFactory.rotation_3d("y", theta)
        expected = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert np.allclose(m, expected)

    def test_rotation_3d_z(self):
        theta = np.pi
        m = MatrixFactory.rotation_3d("z", theta)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        assert np.allclose(m, expected)

    def test_rotation_3d_invalid_axis(self):
        with pytest.raises(ValueError):
            MatrixFactory.rotation_3d("w", np.pi)

    def test_reflection_2d(self):
        m = MatrixFactory.reflection_2d(0)
        expected = np.array([[1, 0], [0, -1]])
        assert np.allclose(m, expected)

    def test_shear_2d_x(self):
        m = MatrixFactory.shear_2d("x", 2)
        expected = np.array([[1, 2], [0, 1]])
        assert np.allclose(m, expected)

    def test_shear_2d_y(self):
        m = MatrixFactory.shear_2d("y", 2)
        expected = np.array([[1, 0], [2, 1]])
        assert np.allclose(m, expected)

    def test_shear_2d_invalid_axis(self):
        with pytest.raises(ValueError):
            MatrixFactory.shear_2d("z", 1)

    def test_scaling_2d(self):
        m = MatrixFactory.scaling_2d(2, 3)
        expected = np.diag([2, 3])
        assert np.allclose(m, expected)

    def test_scaling_3d(self):
        m = MatrixFactory.scaling_3d(1, 2, 3)
        expected = np.diag([1, 2, 3])
        assert np.allclose(m, expected)

    def test_permutation(self):
        m = MatrixFactory.permutation(3, [2, 0, 1])
        expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        assert np.allclose(m, expected)

    def test_permutation_invalid_length(self):
        with pytest.raises(ValueError):
            MatrixFactory.permutation(3, [0, 1])

    def test_permutation_invalid_elements(self):
        with pytest.raises(ValueError):
            MatrixFactory.permutation(3, [0, 0, 1])

    def test_block_diagonal(self):
        blocks = [np.eye(2), np.eye(3)]
        m = MatrixFactory.block_diagonal(blocks)
        expected = np.block(
            [[np.eye(2), np.zeros((2, 3))], [np.zeros((3, 2)), np.eye(3)]]
        )
        assert np.allclose(m, expected)

    def test_block_diagonal_empty(self):
        with pytest.raises(ValueError):
            MatrixFactory.block_diagonal([])

    def test_tridiagonal(self):
        d = np.array([1, 2, 3, 4])
        u = np.array([1, 2, 3])
        l = np.array([1, 2, 3])
        m = MatrixFactory.tridiagonal(d, u, l)
        expected = np.array([[1, 1, 0, 0], [1, 2, 2, 0], [0, 2, 3, 3], [0, 0, 3, 4]])
        assert np.allclose(m, expected)

    def test_symmetric(self):
        m = MatrixFactory.symmetric(3, 3)
        assert m.shape == (3, 3)
        assert np.allclose(m, m.T)

    def test_positive_definite(self):
        m = MatrixFactory.positive_definite(3)
        assert m.shape == (3, 3)
        assert np.all(np.linalg.eigvals(m) > 0)

    def test_orthogonal(self):
        m = MatrixFactory.orthogonal(3)
        assert m.shape == (3, 3)
        assert np.allclose(m @ m.T, np.eye(3))

    def test_unitary(self):
        m = MatrixFactory.unitary(3)
        assert m.shape == (3, 3)
        assert np.allclose(m @ m.conj().T, np.eye(3))

    def test_sparse(self):
        m = MatrixFactory.sparse(10, 10, density=0.3)
        assert m.shape == (10, 10)

    def test_sparse_invalid_density(self):
        with pytest.raises(ValueError):
            MatrixFactory.sparse(3, 3, density=1.5)

    def test_magic_invalid(self):
        with pytest.raises(ValueError):
            MatrixFactory.magic(2)
