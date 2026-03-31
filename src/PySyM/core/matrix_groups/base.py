"""矩阵群基类模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
import numpy as np

T = TypeVar('T', bound='MatrixGroupElement')


class MatrixGroupElement(ABC):
    """矩阵群元素抽象基类"""
    
    def __init__(self, matrix: np.ndarray):
        # 自动选择数据类型，保留复数信息
        self.matrix = np.array(matrix)
        self.dimension = matrix.shape[0]
        # 使用矩阵的字节表示作为哈希值
        self._hash = hash(self.matrix.tobytes())
    
    def __hash__(self) -> int:
        """哈希值，用于集合和字典"""
        return self._hash
    
    def __eq__(self, other: object) -> bool:
        """判断两个矩阵元素是否相等"""
        if not isinstance(other, MatrixGroupElement):
            return NotImplemented
        return np.allclose(self.matrix, other.matrix)
    
    def determinant(self) -> float:
        """计算行列式"""
        return float(np.linalg.det(self.matrix))
    
    def is_orthogonal(self) -> bool:
        """检查是否为正交矩阵"""
        return np.allclose(self.matrix @ self.matrix.T, np.eye(self.dimension))
    
    def is_unitary(self) -> bool:
        """检查是否为酉矩阵"""
        return np.allclose(self.matrix @ self.matrix.conj().T, np.eye(self.dimension))
    
    def is_identity(self) -> bool:
        """是否为单位矩阵"""
        return np.allclose(self.matrix, np.eye(self.dimension))
    
    @abstractmethod
    def __mul__(self, other: 'MatrixGroupElement') -> 'MatrixGroupElement':
        """矩阵乘法"""
        pass
    
    @abstractmethod
    def __pow__(self, n: int) -> 'MatrixGroupElement':
        """矩阵幂"""
        pass
    
    @abstractmethod
    def inverse(self) -> 'MatrixGroupElement':
        """逆元"""
        pass
    
    def order(self) -> int:
        """元素阶数（对于连续群返回-1）"""
        return -1
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.matrix})"


class MatrixGroup(ABC, Generic[T]):
    """矩阵群抽象基类"""
    
    def __init__(self, name: str, field: str = "real", dimension: int = 2):
        """
        初始化矩阵群
        :param name: 群的名称
        :param field: 域（real, complex, finite_field等）
        :param dimension: 矩阵维度
        """
        self.name = name
        self.field = field
        self.dimension = dimension
        self._generators: Optional[List[T]] = None
        self._properties = {
            "order": -1,  # -1表示无限群
            "is_finite": False,
            "is_abelian": False,
            "is_simple": False,
            "center_order": 0,
            "conjugacy_classes": 0
        }
        self._cache = {}
    
    @abstractmethod
    def identity(self) -> T:
        """单位矩阵"""
        pass
    
    def is_abelian(self) -> bool:
        """检查是否为阿贝尔群（矩阵乘法一般不可交换）"""
        return self._properties.get("is_abelian", False)
    
    def order(self) -> int:
        """群的阶（无限群返回-1）"""
        return self._properties.get("order", -1)
    
    def is_finite(self) -> bool:
        """检查是否为有限群"""
        return self._properties.get("is_finite", False)
