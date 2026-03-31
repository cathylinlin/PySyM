"""
物理状态和空间实现
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Callable, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .abstract_physical_objects import PhysicalSpace

from .abstract_physical_objects import PhysicalSpace

class HilbertSpace(PhysicalSpace):
    """
    希尔伯特空间
    
    量子系统的状态空间。与 core.matrix 模块有良好的集成。
    """
    
    def __init__(self, dimension: int):
        """初始化希尔伯特空间
        
        Args:
            dimension: 空间的维度
        """
        self._dimension = dimension
        self._basis: Optional[np.ndarray] = None
        self._metric: Optional[np.ndarray] = None
    
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension
    
    def inner_product(self, x: np.ndarray, y: np.ndarray) -> complex:
        """内积：<x|y>"""
        if self._metric is not None:
            result = complex(np.vdot(x, np.dot(self._metric, y)))
        else:
            result = complex(np.vdot(x, y))
        return result
    
    def norm(self, x: np.ndarray) -> float:
        """范数：||x||"""
        inner = self.inner_product(x, x)
        if isinstance(inner, complex):
            return float(np.sqrt(np.abs(inner.real)))
        return float(np.sqrt(np.abs(inner)))
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化向量
        
        Args:
            x: 输入向量
            
        Returns:
            归一化后的向量
        """
        n = self.norm(x)
        if n < 1e-10:
            return x
        return x / n
    
    def orthogonalize(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """Gram-Schmidt 正交化
        
        Args:
            vectors: 输入向量列表
            
        Returns:
            正交化后的向量列表
        """
        result = []
        for v in vectors:
            for u in result:
                proj = self.inner_product(v, u) / self.inner_product(u, u)
                v = v - proj * u
            if self.norm(v) > 1e-10:
                result.append(self.normalize(v))
        return result
    
    def project(self, v: np.ndarray, subspace: List[np.ndarray]) -> np.ndarray:
        """投影到子空间
        
        Args:
            v: 被投影的向量
            subspace: 子空间基向量
            
        Returns:
            投影向量
        """
        result = np.zeros_like(v)
        for u in subspace:
            coeff = self.inner_product(v, u) / self.inner_product(u, u)
            result += coeff * u
        return result
    
    def basis_vector(self, index: int) -> np.ndarray:
        """获取第 index 个基向量
        
        Args:
            index: 基向量索引
            
        Returns:
            基向量
        """
        if self._basis is not None:
            return self._basis[index]
        
        basis = np.zeros(self._dimension)
        basis[index] = 1.0
        return basis
    
    @property
    def metric(self) -> Optional[np.ndarray]:
        """获取度规算子"""
        return self._metric
    
    @metric.setter
    def metric(self, m: np.ndarray) -> None:
        """设置度规算子
        
        Args:
            m: 度规矩阵
        """
        if m.shape != (self._dimension, self._dimension):
            raise ValueError(f"度规矩阵维度 {m.shape} 与空间维度不匹配")
        self._metric = np.array(m)

class EuclideanSpace(PhysicalSpace):
    """
    欧几里得空间
    
    经典系统的构型空间
    """
    
    def __init__(self, dimension):
        self._dimension = dimension
    
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension
    
    def inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """内积：x·y"""
        return np.dot(x, y)
    
    def norm(self, x: np.ndarray) -> float:
        """范数：||x||"""
        return np.linalg.norm(x)

class SymplecticSpace(PhysicalSpace):
    """
    辛空间
    
    哈密顿系统的相空间
    """
    
    def __init__(self, dimension):
        """
        初始化辛空间
        
        Args:
            dimension: 构形空间的维度，相空间维度为 2*dimension
        """
        self._dimension = 2 * dimension
        self._n = dimension
        # 辛矩阵
        self._omega = np.block([
            [np.zeros((self._n, self._n)), np.eye(self._n)],
            [-np.eye(self._n), np.zeros((self._n, self._n))]
        ])
    
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension
    
    def inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """辛内积：ω(x, y)"""
        return np.dot(x, np.dot(self._omega, y))
    
    def norm(self, x: np.ndarray) -> float:
        """范数：||x||"""
        return np.linalg.norm(x)
    
    def symplectic_matrix(self) -> np.ndarray:
        """返回辛矩阵"""
        return self._omega

class TangentBundle(PhysicalSpace):
    """
    切丛
    
    拉格朗日系统的状态空间
    """
    
    def __init__(self, dimension):
        """
        初始化切丛
        
        Args:
            dimension: 构形空间的维度，切丛维度为 2*dimension
        """
        self._dimension = 2 * dimension
        self._n = dimension
    
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension
    
    def inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """内积：x·y"""
        return np.dot(x, y)
    
    def norm(self, x: np.ndarray) -> float:
        """范数：||x||"""
        return np.linalg.norm(x)
