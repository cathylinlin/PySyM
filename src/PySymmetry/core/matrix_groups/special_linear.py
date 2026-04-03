"""特殊线性群 SL(n, F)"""
import numpy as np
from .base import MatrixGroupElement, MatrixGroup
from ..utils.matrix_utils import is_invertible


class SLnElement(MatrixGroupElement):
    """特殊线性群元素"""
    
    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix)
        if not is_invertible(matrix):
            raise ValueError("矩阵不是可逆的，不能属于特殊线性群")
        if not np.isclose(np.linalg.det(matrix), 1):
            raise ValueError("矩阵行列式不为1，不能属于特殊线性群")
    
    def __mul__(self, other: 'SLnElement') -> 'SLnElement':
        """矩阵乘法"""
        return SLnElement(self.matrix @ other.matrix)
    
    def __pow__(self, n: int) -> 'SLnElement':
        """矩阵幂"""
        return SLnElement(np.linalg.matrix_power(self.matrix, n))
    
    def inverse(self) -> 'SLnElement':
        """逆元"""
        return SLnElement(np.linalg.inv(self.matrix))


class SpecialLinearGroup(MatrixGroup[SLnElement]):
    """特殊线性群 SL(n, F)
    
    由所有 n×n 行列式为1的矩阵组成的群。
    """
    
    def __init__(self, n: int, field: str = "real"):
        """
        初始化特殊线性群
        :param n: 矩阵维度
        :param field: 域（real, complex等）
        """
        super().__init__(f"SL({n}, {field})", field, n)
        self.n = n
        self.field = field
    
    def identity(self) -> SLnElement:
        """单位矩阵"""
        return SLnElement(np.eye(self.n))
    
    def __contains__(self, element: SLnElement) -> bool:
        """判断元素是否属于该群"""
        if not isinstance(element, SLnElement):
            return False
        return (element.matrix.shape == (self.n, self.n) and 
                is_invertible(element.matrix) and 
                np.isclose(element.determinant(), 1))
