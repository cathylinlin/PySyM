import pytest
import numpy as np
from src.PySymmetry.core.matrix.special_matrices import (
    DiagonalMatrix,
    SymmetricMatrix,
    HermitianMatrix,
    OrthogonalMatrix,
    UnitaryMatrix,
    UpperTriangularMatrix,
    LowerTriangularMatrix,
    TridiagonalMatrix,
    ToeplitzMatrix,
    CirculantMatrix,
    HankelMatrix,
    PermutationMatrix,
    PositiveDefiniteMatrix,
    PositiveSemidefiniteMatrix,
    RotationMatrix,
    ReflectionMatrix,
    ProjectionMatrix,
)


class TestDiagonalMatrix:
    def test_init_from_array(self):
        diag = [1, 2, 3]
        dm = DiagonalMatrix(diag)
        assert np.allclose(dm.data, np.diag(diag))
        assert np.allclose(dm.diagonal, diag)

    def test_init_from_1d_array(self):
        arr = np.array([1, 2, 3])
        dm = DiagonalMatrix(arr)
        assert np.allclose(dm.data, np.diag(arr))

    def test_init_from_matrix(self):
        mat = np.diag([1, 2, 3])
        dm = DiagonalMatrix(mat)
        assert np.allclose(dm.data, mat)

    def test_init_invalid_non_diagonal(self):
        mat = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="不是对角矩阵"):
            DiagonalMatrix(mat)

    def test_init_invalid_non_square(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="必须是方阵"):
            DiagonalMatrix(mat)

    def test_inverse(self):
        diag = [1, 2, 3]
        dm = DiagonalMatrix(diag)
        inv = dm.inverse()
        assert np.allclose(inv.data, np.diag([1.0, 0.5, 1.0 / 3.0]))

    def test_inverse_singular(self):
        diag = [1, 0, 3]
        dm = DiagonalMatrix(diag)
        with pytest.raises(ValueError, match="不可逆"):
            dm.inverse()

    def test_power(self):
        diag = [1, 2, 3]
        dm = DiagonalMatrix(diag)
        p2 = dm.power(2)
        assert np.allclose(p2.data, np.diag([1, 4, 9]))

    def test_power_negative(self):
        diag = [1, 2, 4]
        dm = DiagonalMatrix(diag)
        p_neg1 = dm.power(-1)
        assert np.allclose(p_neg1.data, np.diag([1, 0.5, 0.25]))


class TestSymmetricMatrix:
    def test_init_valid(self):
        data = np.array([[1, 2], [2, 3]])
        sm = SymmetricMatrix(data)
        assert np.allclose(sm.data, data)

    def test_init_invalid_not_square(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="必须是方阵"):
            SymmetricMatrix(data)

    def test_init_invalid_not_symmetric(self):
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="不对称"):
            SymmetricMatrix(data)


class TestHermitianMatrix:
    def test_init_valid_real(self):
        data = np.array([[1, 2], [2, 3]], dtype=complex)
        hm = HermitianMatrix(data)
        assert np.allclose(hm.data, data)

    def test_init_valid_complex(self):
        data = np.array([[1, 2j], [-2j, 3]])
        hm = HermitianMatrix(data)
        assert np.allclose(hm.data, data)

    def test_init_invalid_not_hermitian(self):
        data = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises(ValueError, match="不是埃尔米特矩阵"):
            HermitianMatrix(data)


class TestOrthogonalMatrix:
    def test_init_2d_rotation(self):
        theta = np.pi / 4
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        om = OrthogonalMatrix(Q)
        assert np.allclose(om.data, Q)

    def test_init_identity(self):
        I = np.eye(3)
        om = OrthogonalMatrix(I)
        assert np.allclose(om.data, I)

    def test_init_invalid_not_orthogonal(self):
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="不是正交矩阵"):
            OrthogonalMatrix(data)

    def test_inverse(self):
        theta = np.pi / 6
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        om = OrthogonalMatrix(Q)
        inv = om.inverse()
        assert np.allclose(inv.data, Q.T)


class TestUnitaryMatrix:
    def test_init_identity(self):
        I = np.eye(3, dtype=complex)
        um = UnitaryMatrix(I)
        assert np.allclose(um.data, I)

    def test_init_hadamard(self):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        um = UnitaryMatrix(H)
        assert np.allclose(um.data, H)

    def test_init_invalid_not_unitary(self):
        data = np.array([[1, 2], [3, 4]], dtype=complex)
        with pytest.raises(ValueError, match="不是酉矩阵"):
            UnitaryMatrix(data)

    def test_inverse(self):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        um = UnitaryMatrix(H)
        inv = um.inverse()
        assert np.allclose(inv.data, H.conj().T)


class TestUpperTriangularMatrix:
    def test_init_valid(self):
        data = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
        utm = UpperTriangularMatrix(data)
        assert np.allclose(utm.data, data)

    def test_init_invalid_not_upper_triangular(self):
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="不是上三角矩阵"):
            UpperTriangularMatrix(data)


class TestLowerTriangularMatrix:
    def test_init_valid(self):
        data = np.array([[1, 0, 0], [2, 4, 0], [3, 5, 6]], dtype=float)
        ltm = LowerTriangularMatrix(data)
        assert np.allclose(ltm.data, data)

    def test_init_invalid_not_lower_triangular(self):
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="不是下三角矩阵"):
            LowerTriangularMatrix(data)


class TestTridiagonalMatrix:
    def test_init_valid(self):
        data = np.diag([1, 2, 3]) + np.diag([4, 5], k=1) + np.diag([6, 7], k=-1)
        tdm = TridiagonalMatrix(data)
        assert np.allclose(tdm.data, data)

    def test_init_invalid_not_tridiagonal(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError, match="不是三对角矩阵"):
            TridiagonalMatrix(data)


class TestToeplitzMatrix:
    def test_init_valid(self):
        first_row = [1, 2, 3, 4]
        first_col = [1, 5, 6, 7]
        data = np.array([[1, 2, 3, 4], [5, 1, 2, 3], [6, 5, 1, 2], [7, 6, 5, 1]])
        tpm = ToeplitzMatrix(data)
        assert np.allclose(tpm.data, data)

    def test_init_invalid_not_toeplitz(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError, match="不是托普利茨矩阵"):
            ToeplitzMatrix(data)


class TestCirculantMatrix:
    def test_init_invalid_not_circulant(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError):
            CirculantMatrix(data)


class TestHankelMatrix:
    def test_init_valid(self):
        data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        hm = HankelMatrix(data)
        assert np.allclose(hm.data, data)

    def test_init_invalid_not_hankel(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError, match="不是汉克尔矩阵"):
            HankelMatrix(data)


class TestPermutationMatrix:
    def test_init_valid_swap(self):
        P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        pm = PermutationMatrix(P)
        assert np.allclose(pm.data, P)

    def test_init_identity(self):
        I = np.eye(3)
        pm = PermutationMatrix(I)
        assert np.allclose(pm.data, I)

    def test_init_invalid_row_sum(self):
        P = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="置换矩阵"):
            PermutationMatrix(P)

    def test_init_invalid_col_sum(self):
        P = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]])
        with pytest.raises(ValueError, match="置换矩阵"):
            PermutationMatrix(P)

    def test_inverse(self):
        P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        pm = PermutationMatrix(P)
        inv = pm.inverse()
        assert np.allclose(inv.data, P.T)


class TestPositiveDefiniteMatrix:
    def test_init_valid(self):
        A = np.array([[2, 1], [1, 2]])
        pdm = PositiveDefiniteMatrix(A)
        assert np.allclose(pdm.data, A)

    def test_init_invalid_not_symmetric(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        with pytest.raises(ValueError, match="必须是对称的"):
            PositiveDefiniteMatrix(A)

    def test_init_invalid_not_positive_definite(self):
        A = np.array([[-1, 0], [0, -1]])
        with pytest.raises(ValueError, match="不是正定矩阵"):
            PositiveDefiniteMatrix(A)


class TestPositiveSemidefiniteMatrix:
    def test_init_valid(self):
        A = np.array([[1, 0], [0, 1]])
        psdm = PositiveSemidefiniteMatrix(A)
        assert np.allclose(psdm.data, A)

    def test_init_valid_singular(self):
        A = np.array([[1, 0], [0, 0]])
        psdm = PositiveSemidefiniteMatrix(A)
        assert np.allclose(psdm.data, A)

    def test_init_invalid_negative_eigenvalue(self):
        A = np.array([[-1, 0], [0, 1]])
        with pytest.raises(ValueError, match="不是半正定矩阵"):
            PositiveSemidefiniteMatrix(A)


class TestRotationMatrix:
    def test_init_2d_rotation(self):
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rm = RotationMatrix(R)
        assert np.allclose(rm.data, R)

    def test_init_3d_rotation(self):
        theta = np.pi / 3
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        rm = RotationMatrix(R)
        assert np.allclose(rm.data, R)

    def test_init_invalid_wrong_size(self):
        R = np.eye(4)
        with pytest.raises(ValueError, match="必须是2D或3D"):
            RotationMatrix(R)

    def test_init_invalid_det_not_1(self):
        R = np.eye(2) * 2
        with pytest.raises(ValueError, match="行列式必须为1"):
            RotationMatrix(R)


class TestReflectionMatrix:
    def test_init_valid_2d(self):
        n = np.array([1, 0])
        R = np.array([[-1, 0], [0, 1]])
        rm = ReflectionMatrix(R)
        assert np.allclose(rm.data, R)

    def test_init_invalid_det_not_minus_1(self):
        R = np.eye(2)
        with pytest.raises(ValueError, match="行列式必须为-1"):
            ReflectionMatrix(R)


class TestProjectionMatrix:
    def test_init_valid(self):
        v = np.array([[1], [0]], dtype=float)
        P = v @ v.T / (v.T @ v)
        pm = ProjectionMatrix(P)
        assert np.allclose(pm.data, P)

    def test_init_invalid_not_idempotent(self):
        A = np.array([[2, 0], [0, 2]])
        with pytest.raises(ValueError):
            ProjectionMatrix(A)

    def test_init_invalid_not_symmetric(self):
        A = np.array([[0.5, 0.6], [0.4, 0.5]])
        with pytest.raises(ValueError):
            ProjectionMatrix(A)
