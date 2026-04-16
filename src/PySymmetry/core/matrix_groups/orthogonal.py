"""正交群 O(n) 和特殊正交群 SO(n)"""

import numpy as np

from .base import MatrixGroup, MatrixGroupElement


class OnElement(MatrixGroupElement):
    """正交群元素"""

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not self.is_orthogonal():
            raise ValueError("矩阵不是正交矩阵，不能属于正交群")

    def __mul__(self, other: "OnElement") -> "OnElement":
        """矩阵乘法"""
        return OnElement(self.matrix @ other.matrix)

    def __pow__(self, n: int) -> "OnElement":
        """矩阵幂"""
        return OnElement(np.linalg.matrix_power(self.matrix, n))

    def inverse(self) -> "OnElement":
        """逆元（正交矩阵的逆等于其转置）"""
        return OnElement(self.matrix.T)


class OrthogonalGroup(MatrixGroup[OnElement]):
    """正交群 O(n)

    由所有 n×n 正交矩阵组成的群。
    正交矩阵满足 Q^T Q = I。
    """

    def __init__(self, n: int):
        """
        初始化正交群
        :param n: 矩阵维度
        """
        super().__init__(f"O({n})", "real", n)
        self.n = n

    def identity(self) -> OnElement:
        """单位矩阵"""
        return OnElement(np.eye(self.n))

    def __contains__(self, element: OnElement) -> bool:
        """判断元素是否属于该群"""
        if not isinstance(element, OnElement):
            return False
        return element.matrix.shape == (self.n, self.n) and element.is_orthogonal()


class SOnElement(MatrixGroupElement):
    """特殊正交群元素"""

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not self.is_orthogonal():
            raise ValueError("矩阵不是正交矩阵，不能属于特殊正交群")
        if not np.isclose(self.determinant(), 1):
            raise ValueError("矩阵行列式不为1，不能属于特殊正交群")

    def __mul__(self, other: "SOnElement") -> "SOnElement":
        """矩阵乘法"""
        return SOnElement(self.matrix @ other.matrix)

    def __pow__(self, n: int) -> "SOnElement":
        """矩阵幂"""
        return SOnElement(np.linalg.matrix_power(self.matrix, n))

    def inverse(self) -> "SOnElement":
        """逆元（正交矩阵的逆等于其转置）"""
        return SOnElement(self.matrix.T)


class SpecialOrthogonalGroup(MatrixGroup[SOnElement]):
    """特殊正交群 SO(n)

    由所有 n×n 行列式为1的正交矩阵组成的群。
    SO(n) 是 O(n) 的连通分支，包含单位元。
    """

    def __init__(self, n: int):
        """
        初始化特殊正交群
        :param n: 矩阵维度
        """
        super().__init__(f"SO({n})", "real", n)
        self.n = n

    def identity(self) -> SOnElement:
        """单位矩阵"""
        return SOnElement(np.eye(self.n))

    def __contains__(self, element: SOnElement) -> bool:
        """判断元素是否属于该群"""
        if not isinstance(element, SOnElement):
            return False
        return (
            element.matrix.shape == (self.n, self.n)
            and element.is_orthogonal()
            and np.isclose(element.determinant(), 1)
        )
