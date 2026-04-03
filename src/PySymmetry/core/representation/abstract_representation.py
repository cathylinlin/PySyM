import numpy as np
from abc import ABC,abstractmethod
from typing import Generic, TypeVar, Callable, List
from ..group_theory.abstract_group import Group, GroupElement
from ..matrix_groups.base import MatrixGroup, MatrixGroupElement

T = TypeVar('T', bound=GroupElement)

class GroupRepresentation(ABC, Generic[T]):
    """群表示抽象基类。

    数学上为群同态 ``ρ: G → GL(V)``。本库约定 **ρ(ab) = ρ(a)ρ(b)**（矩阵乘），
    与具体群实现中 ``multiply(a,b)`` 的含义一致；上层物理模块构造拉格朗日量或荷载表示时请保持同一约定。
    """
    
    def __init__(self, group: Group[T], dimension: int):
        """
        Args:
            group: 被表示的群
            dimension: 表示的维度(向量空间V的维度)
        """
        if dimension <= 0:
            raise ValueError("表示维度必须为正整数")
        self._group = group
        self._dimension = dimension
    
    @property
    def group(self) -> Group[T]:
        """获取被表示的群"""
        return self._group
    
    @property
    def dimension(self) -> int:
        """获取表示的维度"""
        return self._dimension
    
    @abstractmethod
    def __call__(self, element: T) -> MatrixGroupElement:
        """将群元素映射为矩阵
        
        Args:
            element: 群元素
            
        Returns:
            表示该群元素的矩阵
        """
        ...
    
    @abstractmethod
    def is_homomorphism(self) -> bool:
        """验证是否为群同态
        
        验证是否满足 ρ(ab) = ρ(a)ρ(b) 对所有 a,b∈G
        """
        ...
    
    def is_faithful(self) -> bool:
        """检查是否为忠实表示(单射)"""
        for element in self._group.elements():
            if element != self._group.identity() and np.allclose(self(element).matrix, np.eye(self._dimension)):
                return False
        return True
    
    def kernel(self) -> List[T]:
        """计算表示的核"""
        kernel_elements = []
        for element in self._group.elements():
            if np.allclose(self(element).matrix, np.eye(self._dimension)):
                kernel_elements.append(element)
        return kernel_elements
    
    def character(self) -> Callable[[T], complex]:
        """返回该表示的特征标函数"""
        def char(element: T) -> complex:
            return np.trace(self(element).matrix)
        return char

    def is_equivalent(self, other: 'GroupRepresentation') -> bool:
        """检查两个表示是否等价"""
        # 等价表示：存在一个可逆矩阵P，使得对于所有g，P^{-1}ρ(g)P = σ(g)
        # 简化实现：只检查维度和特征表是否相同
        if self.group != other.group:
            return False
        if self.dimension != other.dimension:
            return False
        
        # 检查特征表是否相同
        for element in self.group.elements():
            if not np.isclose(self.character()(element), other.character()(element)):
                return False
        return True