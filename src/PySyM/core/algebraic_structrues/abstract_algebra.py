"""抽象代数结构基类模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from dataclasses import dataclass

T = TypeVar('T')  # 代数结构元素类型


@dataclass
class AlgebraicProperties:
    """代数结构属性数据类"""
    is_finite: bool              # 是否有限
    characteristic: Optional[int]  # 特征
    order: Optional[int]         # 阶（如果有限）


class SemigroupElement(ABC):
    """半群元素抽象基类"""
    
    @abstractmethod
    def __mul__(self, other: 'SemigroupElement') -> 'SemigroupElement':
        """半群乘法"""
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        """必须可哈希，以便存入集合或作为字典键"""
        pass
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemigroupElement):
            return NotImplemented
        return hash(self) == hash(other)


class Semigroup(ABC, Generic[T]):
    """半群抽象基类"""
    
    def __init__(self, name: str = ""):
        """
        初始化半群
        
        Args:
            name: 半群的名称
        """
        self.name = name
        self._cache: Dict[str, Any] = {}
        self._properties: Optional[AlgebraicProperties] = None
    
    @abstractmethod
    def multiply(self, a: T, b: T) -> T:
        """半群乘法"""
        pass
    
    @abstractmethod
    def __contains__(self, element: T) -> bool:
        """判断元素是否属于该半群"""
        pass
    
    def is_finite(self) -> bool:
        """检查是否为有限半群"""
        if self._properties and self._properties.is_finite is not None:
            return self._properties.is_finite
        
        # 默认假设有限，除非有特殊说明
        self._update_properties()
        return True
    
    def order(self) -> Optional[int]:
        """半群的阶（无限半群返回None）"""
        if self._properties and self._properties.order is not None:
            return self._properties.order
        return None
    
    def _update_properties(self, **kwargs) -> None:
        """更新代数结构属性"""
        if self._properties is None:
            self._properties = AlgebraicProperties(
                is_finite=kwargs.get('is_finite', True),
                characteristic=kwargs.get('characteristic'),
                order=kwargs.get('order')
            )
        else:
            for key, value in kwargs.items():
                if hasattr(self._properties, key):
                    setattr(self._properties, key, value)


class MonoidElement(SemigroupElement):
    """幺半群元素抽象基类"""
    
    @abstractmethod
    def is_identity(self) -> bool:
        """是否为单位元"""
        pass


class Monoid(Semigroup[T]):
    """幺半群抽象基类"""
    
    @abstractmethod
    def identity(self) -> T:
        """单位元"""
        pass
    
    def is_idempotent(self, element: T) -> bool:
        """检查元素是否为幂等元"""
        return self.multiply(element, element) == element


class GroupElement(MonoidElement):
    """群元素抽象基类"""
    
    @abstractmethod
    def inverse(self) -> 'GroupElement':
        """逆元"""
        pass
    
    @abstractmethod
    def __pow__(self, n: int) -> 'GroupElement':
        """幂运算"""
        pass
    
    @abstractmethod
    def order(self) -> int:
        """元素阶数"""
        pass
    
    def __truediv__(self, other: 'GroupElement') -> 'GroupElement':
        """a/b = a * b^(-1)"""
        return self * other.inverse()


class Group(Monoid[T]):
    """群抽象基类"""
    
    @abstractmethod
    def inverse(self, a: T) -> T:
        """逆元"""
        pass
    
    @abstractmethod
    def order(self) -> int:
        """群的阶（无限群返回-1）"""
        pass
    
    @abstractmethod
    def elements(self) -> List[T]:
        """所有群元素（有限群）"""
        raise NotImplementedError("子类必须实现elements方法")
