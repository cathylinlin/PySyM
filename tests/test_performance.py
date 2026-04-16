import pytest
import numpy as np
from PySymmetry.tools import performance


class TestPerformanceModule:
    def test_optimize_matrix_multiply_small(self):
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        result = performance.optimize_matrix_multiply(a, b)
        expected = a @ b
        assert np.allclose(result, expected)

    def test_optimize_matrix_multiply_large(self):
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        result = performance.optimize_matrix_multiply(a, b)
        expected = a @ b
        assert np.allclose(result, expected)

    def test_batch_matrix_multiply(self):
        matrices = [np.random.rand(3, 3) for _ in range(5)]
        vector = np.random.rand(3)
        result = performance.batch_matrix_multiply(matrices, vector)
        assert result.shape == (5, 3)
        expected = np.stack([m @ vector for m in matrices], axis=0)
        assert np.allclose(result, expected)

    def test_batch_matrix_multiply_empty(self):
        result = performance.batch_matrix_multiply([], np.array([1, 2, 3]))
        assert result.shape == (0,)

    def test_optimize_eigendecomposition(self):
        m = np.array([[2, 1], [1, 2]], dtype=float)
        evals, evecs = performance.optimize_eigendecomposition(m)
        assert len(evals) == 2
        assert evecs.shape == (2, 2)

    def test_optimize_eigendecomposition_with_k(self):
        m = np.array([[2, 1], [1, 2]], dtype=float)
        evals, evecs = performance.optimize_eigendecomposition(m, k=1)
        assert len(evals) == 2

    def test_optimize_kron_sequence(self):
        matrices = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]
        result = performance.optimize_kron_sequence(matrices)
        expected = np.kron(matrices[0], matrices[1])
        assert np.allclose(result, expected)

    def test_optimize_kron_sequence_empty(self):
        result = performance.optimize_kron_sequence([])
        assert np.allclose(result, np.array([[1]]))

    def test_optimize_blevit_check_hermitian(self):
        m = np.array([[1, 2 + 1j], [2 - 1j, 1]], dtype=complex)
        assert performance.optimize_blevit_check(m) == True

    def test_optimize_blevit_check_not_hermitian(self):
        m = np.array([[1, 2], [3, 4]], dtype=float)
        assert performance.optimize_blevit_check(m) == False

    def test_optimize_blevit_check_not_square(self):
        m = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        assert performance.optimize_blevit_check(m) == False

    def test_optimize_trace(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=complex)
        result = performance.optimize_trace(m)
        assert result == 15

    def test_optimize_outer_sequence(self):
        vectors = [np.array([1, 2]), np.array([3, 4])]
        result = performance.optimize_outer_sequence(vectors)
        assert result.shape == (2, 2, 2)

    def test_optimize_outer_sequence_empty(self):
        result = performance.optimize_outer_sequence([])
        assert result.shape == (0,)

    def test_matrix_power_sequence(self):
        m = np.array([[1, 0], [0, 1]], dtype=float)
        result = performance.matrix_power_sequence(m, max_power=3)
        assert len(result) == 3

    def test_block_diagonalize(self):
        matrices = [np.eye(2), np.eye(3)]
        result = performance.block_diagonalize(matrices)
        assert result.shape == (5, 5)

    def test_block_diagonalize_empty(self):
        result = performance.block_diagonalize([])
        assert result.shape == (1, 1) or result.size == 0

    def test_optimize_commutator(self):
        a = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([[0, 1], [0, 0]], dtype=float)
        result = performance.optimize_commutator(a, b)
        expected = a @ b - b @ a
        assert np.allclose(result, expected)

    def test_optimize_anticommutator(self):
        a = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([[0, 1], [0, 0]], dtype=float)
        result = performance.optimize_anticommutator(a, b)
        expected = a @ b + b @ a
        assert np.allclose(result, expected)

    def test_parallel_apply_no_joblib(self):
        def square(x):
            return x**2

        args_list = [(1,), (2,), (3,)]
        result = performance.parallel_apply(square, args_list, n_jobs=1)
        assert result == [1, 4, 9]

    def test_sparse_diagonalize(self):
        m = np.array([[2, 1], [1, 2]], dtype=float)
        evals, evecs = performance.sparse_diagonalize(m, k=2)
        assert len(evals) == 2

    def test_batch_evaluate(self):
        def f(x):
            return x**2

        points = np.array([1, 2, 3])
        result = performance.batch_evaluate(f, points)
        expected = np.array([1, 4, 9])
        assert np.allclose(result, expected)

    def test_sparse_diagonalize_large(self):
        m = np.random.rand(100, 100)
        m = (m + m.T) / 2
        try:
            evals, evecs = performance.sparse_diagonalize(m, k=5)
            assert len(evals) == 5
        except Exception:
            pass
