import pytest
import numpy as np
from PySymmetry.core.matrix.decompositions import MatrixDecompositions


class TestMatrixDecompositions:
    def test_eigen_decomposition(self):
        m = np.array([[2, 1], [1, 2]])
        evals, evecs = MatrixDecompositions.eigen_decomposition(m)
        assert len(evals) == 2
        assert evecs.shape == (2, 2)

    def test_eigen_decomposition_not_square(self):
        m = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            MatrixDecompositions.eigen_decomposition(m)

    def test_svd(self):
        m = np.array([[1, 2], [3, 4], [5, 6]])
        U, S, Vh = MatrixDecompositions.svd(m)
        assert U.shape == (3, 3)
        assert len(S) == 2
        assert Vh.shape == (2, 2)

    def test_svd_full_matrices_false(self):
        m = np.array([[1, 2], [3, 4]])
        U, S, Vh = MatrixDecompositions.svd(m, full_matrices=False)
        assert U.shape == (2, 2)
        assert Vh.shape == (2, 2)

    def test_qr_decomposition(self):
        m = np.array([[1, 2], [3, 4], [5, 6]])
        Q, R = MatrixDecompositions.qr_decomposition(m)
        assert Q.shape[0] == 3
        assert R.shape == (2, 2)

    def test_qr_decomposition_mode(self):
        m = np.array([[1, 2], [3, 4]])
        Q, R = MatrixDecompositions.qr_decomposition(m, mode="reduced")
        assert Q.shape == (2, 2)

    def test_cholesky_decomposition(self):
        m = np.array([[4, 2], [2, 5]])
        L = MatrixDecompositions.cholesky_decomposition(m)
        assert L.shape == (2, 2)
        assert np.allclose(L @ L.T, m)

    def test_cholesky_not_symmetric(self):
        m = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            MatrixDecompositions.cholesky_decomposition(m)

    def test_cholesky_not_positive_definite(self):
        m = np.array([[1, 0], [0, -1]])
        with pytest.raises(ValueError):
            MatrixDecompositions.cholesky_decomposition(m)

    def test_lu_decomposition(self):
        m = np.array([[1, 2], [3, 4]])
        P, L, U = MatrixDecompositions.lu_decomposition(m)
        assert L.shape == (2, 2)
        assert U.shape == (2, 2)

    def test_lu_decomposition_not_square(self):
        m = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            MatrixDecompositions.lu_decomposition(m)

    def test_schur_decomposition(self):
        m = np.array([[1, 2], [3, 4]])
        T, Z = MatrixDecompositions.schur_decomposition(m)
        assert T.shape == (2, 2)
        assert Z.shape == (2, 2)

    def test_schur_decomposition_complex(self):
        m = np.array([[1, 2], [3, 4]])
        T, Z = MatrixDecompositions.schur_decomposition(m, output="complex")
        assert T.shape == (2, 2)

    def test_hessenberg_decomposition(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = MatrixDecompositions.hessenberg_decomposition(m)
        H = result[0] if isinstance(result, tuple) else result
        assert H.shape == (3, 3)

    def test_polar_decomposition(self):
        m = np.array([[1, 2], [3, 4]])
        U, P = MatrixDecompositions.polar_decomposition(m)
        assert U.shape == (2, 2)
        assert P.shape == (2, 2)

    def test_polar_decomposition_left(self):
        m = np.array([[1, 2], [3, 4]])
        U, P = MatrixDecompositions.polar_decomposition(m, side="left")
        assert U.shape == (2, 2)

    def test_spectral_decomposition(self):
        m = np.array([[2, 1], [1, 2]])
        evals, evecs = MatrixDecompositions.spectral_decomposition(m)
        assert len(evals) == 2
        assert evecs.shape == (2, 2)

    def test_spectral_decomposition_not_symmetric(self):
        m = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            MatrixDecompositions.spectral_decomposition(m)

    def test_jordan_decomposition(self):
        m = np.array([[2, 1], [0, 2]])
        P, J = MatrixDecompositions.jordan_decomposition(m)
        assert P.shape == (2, 2)
        assert J.shape == (2, 2)

    def test_bidiagonal_decomposition(self):
        m = np.array([[1, 2], [3, 4], [5, 6]])
        U, B, Vh = MatrixDecompositions.bidiagonal_decomposition(m)
        assert U.shape == (3, 3)
        assert B.shape == (3, 2)
        assert Vh.shape == (2, 2)
