from __future__ import annotations

import numpy as np

from .base import AbstractMatrix


class DiagonalMatrix(AbstractMatrix):
    """对角矩阵"""

    def __init__(self, diagonal_or_matrix):
        """初始化对角矩阵

        Args:
            diagonal_or_matrix: 对角线元素或对角矩阵
        """
        if isinstance(diagonal_or_matrix, np.ndarray):
            if diagonal_or_matrix.ndim == 2:
                if not (diagonal_or_matrix.shape[0] == diagonal_or_matrix.shape[1]):
                    raise ValueError("输入矩阵必须是方阵")
                if np.allclose(
                    diagonal_or_matrix, np.diag(np.diag(diagonal_or_matrix))
                ):
                    self._diagonal = np.diag(diagonal_or_matrix)
                    super().__init__(diagonal_or_matrix)
                else:
                    raise ValueError("输入矩阵不是对角矩阵")
            elif diagonal_or_matrix.ndim == 1:
                self._diagonal = np.array(
                    diagonal_or_matrix,
                    dtype=np.complex128
                    if np.iscomplexobj(diagonal_or_matrix)
                    else np.float64,
                )
                super().__init__(np.diag(diagonal_or_matrix))
            else:
                raise ValueError("输入必须是一维或二维数组")
        else:
            try:
                diagonal = np.array(
                    diagonal_or_matrix,
                    dtype=np.complex128
                    if any(isinstance(x, complex) for x in diagonal_or_matrix)
                    else np.float64,
                )
                if diagonal.ndim != 1:
                    raise ValueError("输入必须是一维序列")
                self._diagonal = diagonal
                super().__init__(np.diag(diagonal))
            except Exception:
                raise ValueError("输入必须是可转换为一维数组的序列")

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("对角矩阵必须是方阵")
        if not np.allclose(self._data, np.diag(np.diag(self._data))):
            raise ValueError("矩阵不是对角矩阵")

    @property
    def diagonal(self) -> np.ndarray:
        return self._diagonal

    def inverse(self) -> DiagonalMatrix:
        if np.any(np.abs(self._diagonal) < 1e-10):
            raise ValueError("对角矩阵不可逆")
        return DiagonalMatrix(1.0 / self._diagonal)

    def power(self, n: int) -> DiagonalMatrix:
        return DiagonalMatrix(self._diagonal**n)


class SymmetricMatrix(AbstractMatrix):
    """对称矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化对称矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("对称矩阵必须是方阵")
        if not np.allclose(self._data, self._data.T):
            raise ValueError("矩阵不对称")


class HermitianMatrix(AbstractMatrix):
    """埃尔米特矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化埃尔米特矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("埃尔米特矩阵必须是方阵")
        if not np.allclose(self._data, self._data.conj().T):
            raise ValueError("矩阵不是埃尔米特矩阵")


class OrthogonalMatrix(AbstractMatrix):
    """正交矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化正交矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("正交矩阵必须是方阵")
        product = self._data @ self._data.T
        identity = np.eye(self.shape[0])
        if not np.allclose(product, identity):
            raise ValueError("矩阵不是正交矩阵")

    def inverse(self) -> OrthogonalMatrix:
        return OrthogonalMatrix(self._data.T)


class UnitaryMatrix(AbstractMatrix):
    """酉矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化酉矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("酉矩阵必须是方阵")
        product = self._data @ self._data.conj().T
        identity = np.eye(self.shape[0])
        if not np.allclose(product, identity):
            raise ValueError("矩阵不是酉矩阵")

    def inverse(self) -> UnitaryMatrix:
        return UnitaryMatrix(self._data.conj().T)


class UpperTriangularMatrix(AbstractMatrix):
    """上三角矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化上三角矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("三角矩阵必须是方阵")
        n = self.shape[0]
        lower_part = self._data[np.tril_indices(n, k=-1)]
        if not np.allclose(lower_part, 0):
            raise ValueError("矩阵不是上三角矩阵")


class LowerTriangularMatrix(AbstractMatrix):
    """下三角矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化下三角矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("三角矩阵必须是方阵")
        n = self.shape[0]
        upper_part = self._data[np.triu_indices(n, k=1)]
        if not np.allclose(upper_part, 0):
            raise ValueError("矩阵不是下三角矩阵")


class TridiagonalMatrix(AbstractMatrix):
    """三对角矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化三对角矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("三对角矩阵必须是方阵")
        n = self.shape[0]
        # 创建一个掩码，只保留对角线、上对角线和下对角线
        mask = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(mask, True)
        if n > 1:
            np.fill_diagonal(mask[1:], True)
            np.fill_diagonal(mask[:, 1:], True)
        # 检查掩码外的元素是否为零
        if not np.allclose(self._data[~mask], 0):
            raise ValueError("矩阵不是三对角矩阵")


class ToeplitzMatrix(AbstractMatrix):
    """托普利茨矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化托普利茨矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("托普利茨矩阵必须是方阵")
        n = self.shape[0]
        # 检查每一条对角线是否相等
        for k in range(-n + 1, n):
            diagonal = np.diag(self._data, k)
            if not np.allclose(diagonal, diagonal[0]):
                raise ValueError("矩阵不是托普利茨矩阵")


class CirculantMatrix(AbstractMatrix):
    """循环矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化循环矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("循环矩阵必须是方阵")
        n = self.shape[0]
        # 检查每一行是否是前一行的循环移位
        for i in range(1, n):
            expected_row = np.roll(self._data[0], -i)
            if not np.allclose(self._data[i], expected_row):
                raise ValueError("矩阵不是循环矩阵")


class HankelMatrix(AbstractMatrix):
    """汉克尔矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化汉克尔矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("汉克尔矩阵必须是方阵")
        n = self.shape[0]
        # 检查每一条反对角线是否相等
        for k in range(2 * n - 1):
            # 提取第k条反对角线
            diagonal = []
            for i in range(max(0, k - n + 1), min(n, k + 1)):
                j = k - i
                if j < n:
                    diagonal.append(self._data[i, j])
            if len(diagonal) > 1 and not np.allclose(diagonal, diagonal[0]):
                raise ValueError("矩阵不是汉克尔矩阵")


class PermutationMatrix(AbstractMatrix):
    """置换矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化置换矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("置换矩阵必须是方阵")
        # 检查每行每列是否只有一个1，其余为0
        row_sums = np.sum(np.abs(self._data), axis=1)
        col_sums = np.sum(np.abs(self._data), axis=0)
        if not (np.allclose(row_sums, 1) and np.allclose(col_sums, 1)):
            raise ValueError("矩阵不是置换矩阵")
        # 检查所有元素是否为0或1
        if not np.all((self._data == 0) | (self._data == 1)):
            raise ValueError("置换矩阵只能包含0和1")

    def inverse(self) -> PermutationMatrix:
        return PermutationMatrix(self._data.T)


class PositiveDefiniteMatrix(AbstractMatrix):
    """正定矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化正定矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("正定矩阵必须是方阵")
        if not np.allclose(self._data, self._data.T):
            raise ValueError("正定矩阵必须是对称的")
        try:
            eigenvalues = np.linalg.eigvalsh(self._data)
            if not np.all(eigenvalues > 0):
                raise ValueError("矩阵不是正定矩阵")
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不是正定矩阵")


class PositiveSemidefiniteMatrix(AbstractMatrix):
    """半正定矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化半正定矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("半正定矩阵必须是方阵")
        if not np.allclose(self._data, self._data.T):
            raise ValueError("半正定矩阵必须是对称的")
        try:
            eigenvalues = np.linalg.eigvalsh(self._data)
            if not np.all(eigenvalues >= 0):
                raise ValueError("矩阵不是半正定矩阵")
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不是半正定矩阵")


class RotationMatrix(AbstractMatrix):
    """旋转矩阵（2D或3D）"""

    def __init__(self, data: np.ndarray):
        """初始化旋转矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("旋转矩阵必须是方阵")
        if self.shape[0] not in [2, 3]:
            raise ValueError("旋转矩阵必须是2D或3D")
        if not np.allclose(np.linalg.det(self._data), 1):
            raise ValueError("旋转矩阵的行列式必须为1")
        if not np.allclose(self._data @ self._data.T, np.eye(self.shape[0])):
            raise ValueError("旋转矩阵必须是正交矩阵")


class ReflectionMatrix(AbstractMatrix):
    """反射矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化反射矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("反射矩阵必须是方阵")
        if not np.allclose(np.linalg.det(self._data), -1):
            raise ValueError("反射矩阵的行列式必须为-1")
        if not np.allclose(self._data @ self._data.T, np.eye(self.shape[0])):
            raise ValueError("反射矩阵必须是正交矩阵")


class ProjectionMatrix(AbstractMatrix):
    """投影矩阵"""

    def __init__(self, data: np.ndarray):
        """初始化投影矩阵

        Args:
            data: 输入矩阵
        """
        if data.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        super().__init__(data)

    def _validate_shape(self) -> None:
        if not self.is_square():
            raise ValueError("投影矩阵必须是方阵")
        if not np.allclose(self._data @ self._data, self._data):
            raise ValueError("投影矩阵必须满足 A^2 = A")
        if not np.allclose(self._data, self._data.T):
            raise ValueError("投影矩阵必须是对称的")
