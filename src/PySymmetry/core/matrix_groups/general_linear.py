"""一般线性群 GL(n, F)"""

import numpy as np

from ..utils.matrix_utils import is_invertible
from .base import MatrixGroup, MatrixGroupElement


class GLnElement(MatrixGroupElement):
    """一般线性群元素"""

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not is_invertible(matrix):
            raise ValueError("矩阵不是可逆的，不能属于一般线性群")

    def __mul__(self, other: "GLnElement") -> "GLnElement":
        """矩阵乘法"""
        return GLnElement(self.matrix @ other.matrix)

    def __pow__(self, n: int) -> "GLnElement":
        """矩阵幂"""
        return GLnElement(np.linalg.matrix_power(self.matrix, n))

    def inverse(self) -> "GLnElement":
        """逆元"""
        return GLnElement(np.linalg.inv(self.matrix))


class GeneralLinearGroup(MatrixGroup[GLnElement]):
    """一般线性群 GL(n, F)

    由所有 n×n 可逆矩阵组成的群。
    """

    def __init__(self, n: int, field: str = "real"):
        """
        初始化一般线性群
        :param n: 矩阵维度
        :param field: 域（real, complex等）
        """
        super().__init__(f"GL({n}, {field})", field, n)
        self.n = n
        self.field = field

    def identity(self) -> GLnElement:
        """单位矩阵"""
        return GLnElement(np.eye(self.n))

    def __contains__(self, element: GLnElement) -> bool:
        """判断元素是否属于该群"""
        if not isinstance(element, GLnElement):
            return False
        return element.matrix.shape == (self.n, self.n) and is_invertible(
            element.matrix
        )
