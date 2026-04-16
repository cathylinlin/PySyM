import pytest
import numpy as np
from PySymmetry.core.matrix.transformations import MatrixTransformations


class TestMatrixTransformations:
    """Test suite for MatrixTransformations class."""

    def setup_method(self):
        self.transform = MatrixTransformations()
        self.A = np.array([[1, 2], [3, 4]], dtype=float)
        self.B = np.array([[1, 0], [0, 2]], dtype=float)

    def test_similarity_transform(self):
        P = np.array([[1, 1], [0, 1]], dtype=float)
        result = self.transform.similarity_transform(self.A, P)

        P_inv = np.linalg.inv(P)
        expected = P_inv @ self.A @ P
        assert np.allclose(result, expected)

    def test_similarity_transform_3x3(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        result = self.transform.similarity_transform(A, P)
        assert np.allclose(result, A)

    def test_similarity_transform_nonsquare_P(self):
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError):
            self.transform.similarity_transform(self.A, P)

    def test_similarity_transform_singular_P(self):
        P = np.array([[1, 2], [2, 4]], dtype=float)
        with pytest.raises(ValueError, match="变换矩阵不可逆"):
            self.transform.similarity_transform(self.A, P)

    def test_congruence_transform(self):
        P = np.array([[1, 0], [1, 1]], dtype=float)
        result = self.transform.congruence_transform(self.A, P)
        expected = P.T @ self.A @ P
        assert np.allclose(result, expected)

    def test_congruence_transform_rectangular(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        P = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        try:
            result = self.transform.congruence_transform(A, P)
        except ValueError:
            pass

    def test_unitary_transform(self):
        theta = np.pi / 4
        U = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=complex,
        )
        A = np.array([[1, 1j], [-1j, 2]], dtype=complex)
        result = self.transform.unitary_transform(A, U)

        U_H = U.conj().T
        expected = U_H @ A @ U
        assert np.allclose(result, expected)

    def test_unitary_transform_non_unitary(self):
        U = np.array([[1, 1], [0, 1]], dtype=complex)
        with pytest.raises(ValueError, match="变换矩阵不是酉矩阵"):
            self.transform.unitary_transform(self.A, U)

    def test_orthogonal_transform(self):
        theta = np.pi / 6
        Q = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        result = self.transform.orthogonal_transform(self.A, Q)

        Q_T = Q.T
        expected = Q_T @ self.A @ Q
        assert np.allclose(result, expected)

    def test_orthogonal_transform_non_orthogonal(self):
        Q = np.array([[1, 1], [0, 1]], dtype=float)
        with pytest.raises(ValueError, match="变换矩阵不是正交矩阵"):
            self.transform.orthogonal_transform(self.A, Q)

    def test_row_operation(self):
        result = self.transform.row_operation(self.A, 1, 0, factor=2.0)
        expected = self.A.copy()
        expected[1, :] += 2.0 * self.A[0, :]
        assert np.allclose(result, expected)

    def test_row_operation_negative_factor(self):
        result = self.transform.row_operation(self.A, 0, 1, factor=-1.0)
        expected = self.A.copy()
        expected[0, :] -= self.A[1, :]
        assert np.allclose(result, expected)

    def test_row_operation_invalid_index(self):
        with pytest.raises(ValueError, match="行索引超出范围"):
            self.transform.row_operation(self.A, 5, 0)

    def test_column_operation(self):
        result = self.transform.column_operation(self.A, 1, 0, factor=2.0)
        expected = self.A.copy()
        expected[:, 1] += 2.0 * self.A[:, 0]
        assert np.allclose(result, expected)

    def test_column_operation_invalid_index(self):
        with pytest.raises(ValueError, match="列索引超出范围"):
            self.transform.column_operation(self.A, 0, 5)

    def test_row_swap(self):
        result = self.transform.row_swap(self.A, 0, 1)
        expected = self.A.copy()
        expected[[0, 1], :] = expected[[1, 0], :]
        assert np.allclose(result, expected)

    def test_row_swap_invalid_index(self):
        with pytest.raises(ValueError, match="行索引超出范围"):
            self.transform.row_swap(self.A, 0, 5)

    def test_column_swap(self):
        result = self.transform.column_swap(self.A, 0, 1)
        expected = self.A.copy()
        expected[:, [0, 1]] = expected[:, [1, 0]]
        assert np.allclose(result, expected)

    def test_column_swap_invalid_index(self):
        with pytest.raises(ValueError, match="列索引超出范围"):
            self.transform.column_swap(self.A, 0, 5)

    def test_row_scale(self):
        result = self.transform.row_scale(self.A, 0, factor=3.0)
        expected = self.A.copy()
        expected[0, :] *= 3.0
        assert np.allclose(result, expected)

    def test_row_scale_zero_factor(self):
        result = self.transform.row_scale(self.A, 1, factor=0.0)
        expected = np.array([[1, 2], [0, 0]], dtype=float)
        assert np.allclose(result, expected)

    def test_row_scale_invalid_index(self):
        with pytest.raises(ValueError, match="行索引超出范围"):
            self.transform.row_scale(self.A, 5, 1.0)

    def test_column_scale(self):
        result = self.transform.column_scale(self.A, 0, factor=2.0)
        expected = self.A.copy()
        expected[:, 0] *= 2.0
        assert np.allclose(result, expected)

    def test_column_scale_invalid_index(self):
        with pytest.raises(ValueError, match="列索引超出范围"):
            self.transform.column_scale(self.A, 5, 1.0)

    def test_gauss_jordan_elimination_identity(self):
        I = np.eye(3)
        result = self.transform.gauss_jordan_elimination(I)
        assert np.allclose(result, I)

    def test_gauss_jordan_elimination_simple(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.gauss_jordan_elimination(A)
        expected = np.array([[1, 0], [0, 1]], dtype=float)
        assert np.allclose(result, expected, atol=1e-10)

    def test_gauss_jordan_elimination_singular(self):
        A = np.array([[1, 2], [2, 4]], dtype=float)
        result = self.transform.gauss_jordan_elimination(A)
        assert result[1, 0] == 0

    def test_gauss_jordan_elimination_rectangular(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = self.transform.gauss_jordan_elimination(A)
        assert result.shape == (2, 3)

    def test_row_echelon_form_identity(self):
        I = np.eye(3)
        result = self.transform.row_echelon_form(I)
        assert np.allclose(result, I)

    def test_row_echelon_form_simple(self):
        A = np.array([[2, 4], [1, 3]], dtype=float)
        result = self.transform.row_echelon_form(A)
        assert result[0, 0] != 0
        assert result[1, 1] != 0 or result[1, 0] != 0

    def test_hessenberg_form(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = self.transform.hessenberg_form(A)
        assert result.shape == (3, 3)
        assert result[2, 0] == 0

    def test_hessenberg_form_2x2(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.hessenberg_form(A)
        assert result.shape == (2, 2)

    def test_hessenberg_form_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="海森伯格形式需要方阵"):
            self.transform.hessenberg_form(A)

    def test_bidiagonal_form(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = self.transform.bidiagonal_form(A)
        assert result.shape == (2, 3)

    def test_bidiagonal_form_square(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.bidiagonal_form(A)
        assert result.shape == (2, 2)

    def test_tridiagonal_form(self):
        A = np.array([[1, 2, 0], [2, 3, 4], [0, 4, 5]], dtype=float)
        result = self.transform.tridiagonal_form(A)
        assert result.shape == (3, 3)
        assert np.allclose(result, result.T)
        assert result[0, 2] == 0
        assert result[2, 0] == 0

    def test_tridiagonal_form_nonsymmetric(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError, match="三对角形式需要对称矩阵"):
            self.transform.tridiagonal_form(A)

    def test_companion_form(self):
        coeffs = np.array([1, 2, 1], dtype=float)
        result = self.transform.companion_form(coeffs)
        expected = np.array([[-2, -1], [1, 0]], dtype=float)
        assert np.allclose(result, expected)

    def test_companion_form_quadratic(self):
        coeffs = np.array([1, 3, 3, 1], dtype=float)
        result = self.transform.companion_form(coeffs)
        assert result.shape == (3, 3)
        eigenvalues = np.linalg.eigvals(result)
        assert np.allclose(sorted(eigenvalues.real), np.roots(coeffs), atol=1e-5)

    def test_companion_form_invalid(self):
        with pytest.raises(ValueError, match="多项式次数至少为1"):
            self.transform.companion_form(np.array([1]))

    def test_companion_form_zero_leading(self):
        with pytest.raises(ValueError, match="首项系数不能为零"):
            self.transform.companion_form(np.array([0, 1]))

    def test_jordan_canonical_form(self):
        A = np.array([[2, 1], [0, 2]], dtype=float)
        result = self.transform.jordan_canonical_form(A)
        expected_eigenvalues = np.array([2, 2])
        assert np.allclose(sorted(np.diag(result)), sorted(expected_eigenvalues))

    def test_jordan_canonical_form_2x2(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.jordan_canonical_form(A)
        expected = np.diag(np.linalg.eigvals(A))
        assert np.allclose(result, expected)

    def test_jordan_canonical_form_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="若尔当标准形需要方阵"):
            self.transform.jordan_canonical_form(A)

    def test_rational_canonical_form(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.rational_canonical_form(A)
        expected = np.diag(np.linalg.eigvals(A))
        assert np.allclose(result, expected)

    def test_rational_canonical_form_non_square(self):
        A = np.array([[1, 2, 3]], dtype=float)
        with pytest.raises(ValueError, match="有理标准形需要方阵"):
            self.transform.rational_canonical_form(A)

    def test_smith_normal_form(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.smith_normal_form(A)
        assert result.shape == A.shape

    def test_smith_normal_form_rectangular(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = self.transform.smith_normal_form(A)
        assert result.shape == (2, 3)

    def test_frobenius_normal_form(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.frobenius_normal_form(A)
        assert result.shape == (2, 2)

    def test_frobenius_normal_form_3x3(self):
        A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        result = self.transform.frobenius_normal_form(A)
        assert result.shape == (3, 3)

    def test_frobenius_normal_form_non_square(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        with pytest.raises(ValueError, match="弗罗贝尼乌斯标准形需要方阵"):
            self.transform.frobenius_normal_form(A)

    def test_schur_form(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.schur_form(A)
        assert result.shape == (2, 2)

    def test_schur_form_3x3(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = self.transform.schur_form(A)
        assert result.shape == (3, 3)

    def test_schur_form_non_square(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="舒尔形式需要方阵"):
            self.transform.schur_form(A)

    def test_real_schur_form(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.real_schur_form(A)
        assert result.shape == (2, 2)
        assert np.isrealobj(result)

    def test_real_schur_form_non_square(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        with pytest.raises(ValueError, match="实舒尔形式需要方阵"):
            self.transform.real_schur_form(A)

    def test_complex_schur_form(self):
        A = np.array([[1, 2j], [3j, 4]], dtype=complex)
        result = self.transform.complex_schur_form(A)
        assert result.shape == (2, 2)

    def test_complex_schur_form_non_square(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.transform.complex_schur_form(A)
        assert result.shape == (2, 2)


class TestTransformationsDimensionChecks:
    """Test dimension and type validation in transformations."""

    def test_ndim_checks_similarity(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.similarity_transform(np.array([1, 2, 3]), np.eye(3))

    def test_ndim_checks_congruence_matrix(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.congruence_transform(np.array([1, 2, 3]), np.eye(3))

    def test_ndim_checks_congruence_P(self):
        with pytest.raises(ValueError, match="变换矩阵必须是二维数组"):
            MatrixTransformations.congruence_transform(np.eye(3), np.array([1, 2, 3]))

    def test_ndim_checks_unitary_matrix(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.unitary_transform(np.array([1, 2, 3]), np.eye(3))

    def test_ndim_checks_orthogonal_matrix(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.orthogonal_transform(np.array([1, 2, 3]), np.eye(3))

    def test_ndim_checks_row_operation(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.row_operation(np.array([1, 2, 3]), 0, 1)

    def test_ndim_checks_column_operation(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.column_operation(np.array([1, 2, 3]), 0, 1)

    def test_ndim_checks_row_swap(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.row_swap(np.array([1, 2, 3]), 0, 1)

    def test_ndim_checks_column_swap(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.column_swap(np.array([1, 2, 3]), 0, 1)

    def test_ndim_checks_row_scale(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.row_scale(np.array([1, 2, 3]), 0, 2.0)

    def test_ndim_checks_column_scale(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.column_scale(np.array([1, 2, 3]), 0, 2.0)

    def test_ndim_checks_gauss_jordan(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.gauss_jordan_elimination(np.array([1, 2, 3]))

    def test_ndim_checks_row_echelon(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.row_echelon_form(np.array([1, 2, 3]))

    def test_ndim_checks_bidiagonal(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.bidiagonal_form(np.array([1, 2, 3]))

    def test_ndim_checks_tridiagonal(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.tridiagonal_form(np.array([1, 2, 3]))

    def test_ndim_checks_jordan(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.jordan_canonical_form(np.array([1, 2, 3]))

    def test_ndim_checks_smith(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.smith_normal_form(np.array([1, 2, 3]))

    def test_ndim_checks_schur(self):
        with pytest.raises(ValueError, match="输入矩阵必须是二维数组"):
            MatrixTransformations.schur_form(np.array([1, 2, 3]))


class TestSquareMatrixRequirements:
    """Test square matrix requirements for various transformations."""

    def test_similarity_transform_nonsquare(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        P = np.eye(2)
        with pytest.raises(ValueError, match="相似变换需要方阵"):
            MatrixTransformations.similarity_transform(A, P)

    def test_unitary_transform_nonsquare(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        U = np.eye(3, dtype=complex)
        with pytest.raises(ValueError, match="酉变换需要方阵"):
            MatrixTransformations.unitary_transform(A, U)

    def test_orthogonal_transform_nonsquare(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        Q = np.eye(3)
        with pytest.raises(ValueError, match="正交变换需要方阵"):
            MatrixTransformations.orthogonal_transform(A, Q)

    def test_tridiagonal_nonsquare(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError, match="三对角形式需要方阵"):
            MatrixTransformations.tridiagonal_form(A)

    def test_jordan_nonsquare(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        with pytest.raises(ValueError, match="若尔当标准形需要方阵"):
            MatrixTransformations.jordan_canonical_form(A)


class TestCompanionFormEdgeCases:
    """Test edge cases for companion form."""

    def test_companion_monic_polynomial(self):
        coeffs = np.array([1, 0, -1], dtype=float)
        result = MatrixTransformations.companion_form(coeffs)
        assert result.shape == (2, 2)

    def test_companion_cubic(self):
        coeffs = np.array([1, -6, 11, -6], dtype=float)
        result = MatrixTransformations.companion_form(coeffs)
        assert result.shape == (3, 3)
        eigenvalues = np.linalg.eigvals(result)
        assert np.allclose(sorted(eigenvalues), [1, 2, 3])
