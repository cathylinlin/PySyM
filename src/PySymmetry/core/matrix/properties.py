"""矩阵性质模块"""

import numpy as np
from scipy.linalg import null_space as scipy_null_space
from scipy.linalg import orth


class MatrixProperties:
    @staticmethod
    def is_square(matrix: np.ndarray) -> bool:
        if matrix.ndim != 2:
            return False
        return matrix.shape[0] == matrix.shape[1]

    @staticmethod
    def is_symmetric(
        matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        if matrix.ndim != 2:
            return False
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_skew_symmetric(
        matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        if matrix.ndim != 2:
            return False
        return np.allclose(matrix, -matrix.T, rtol=rtol, atol=atol)

    @staticmethod
    def is_hermitian(
        matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        if matrix.ndim != 2:
            return False
        return np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol)

    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.all(eigenvalues > 0))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def is_positive_semidefinite(matrix: np.ndarray) -> bool:
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.all(eigenvalues >= 0))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def is_negative_definite(matrix: np.ndarray) -> bool:
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.all(eigenvalues < 0))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def is_negative_semidefinite(matrix: np.ndarray) -> bool:
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.all(eigenvalues <= 0))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def is_indefinite(matrix: np.ndarray) -> bool:
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return bool(np.any(eigenvalues > 0) and np.any(eigenvalues < 0))
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def is_singular(matrix: np.ndarray) -> bool:
        return np.linalg.matrix_rank(matrix) < min(matrix.shape)

    @staticmethod
    def is_invertible(matrix: np.ndarray) -> bool:
        return np.linalg.matrix_rank(matrix) == min(matrix.shape)

    @staticmethod
    def is_idempotent(
        matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        try:
            return np.allclose(matrix @ matrix, matrix, rtol=rtol, atol=atol)
        except Exception:
            return False

    @staticmethod
    def is_normal(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        try:
            return np.allclose(
                matrix @ matrix.conj().T, matrix.conj().T @ matrix, rtol=rtol, atol=atol
            )
        except Exception:
            return False

    @staticmethod
    def is_unitary(
        matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool:
        try:
            return np.allclose(
                matrix @ matrix.conj().T, np.eye(matrix.shape[0]), rtol=rtol, atol=atol
            )
        except Exception:
            return False

    @staticmethod
    def is_sparse(matrix: np.ndarray, threshold: float = 0.5) -> bool:
        if matrix.ndim != 2:
            return False
        zero_ratio = np.sum(np.abs(matrix) < 1e-10) / matrix.size
        return bool(zero_ratio > threshold)

    @staticmethod
    def null_space(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        return scipy_null_space(matrix, rcond=tol)

    @staticmethod
    def column_space(matrix: np.ndarray) -> np.ndarray:
        return orth(matrix.T).T

    @staticmethod
    def row_space(matrix: np.ndarray) -> np.ndarray:
        return orth(matrix)
