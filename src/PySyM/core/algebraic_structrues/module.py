"""模论和向量空间模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Union
from dataclasses import dataclass
from .ring import Ring, RingElement
from .field import Field, FieldElement

M = TypeVar('M')  # 模元素类型
R = TypeVar('R', bound=RingElement)  # 环元素类型
F = TypeVar('F', bound=FieldElement)  # 域元素类型


class ModuleElement(ABC):
    """模元素抽象基类"""
    
    @abstractmethod
    def __add__(self, other: 'ModuleElement') -> 'ModuleElement':
        """加法"""
        pass
    
    @abstractmethod
    def __sub__(self, other: 'ModuleElement') -> 'ModuleElement':
        """减法"""
        pass
    
    @abstractmethod
    def __mul__(self, scalar: R) -> 'ModuleElement':
        """标量乘法"""
        pass
    
    @abstractmethod
    def is_zero(self) -> bool:
        """是否为零元"""
        pass


class Module(ABC, Generic[M, R]):
    """模抽象基类"""
    
    def __init__(self, ring: Ring[R], name: str = ""):
        """
        初始化模
        
        Args:
            ring: 基础环
            name: 模的名称
        """
        self.ring = ring
        self.name = name
    
    @abstractmethod
    def add(self, a: M, b: M) -> M:
        """加法"""
        pass
    
    @abstractmethod
    def scalar_multiply(self, a: M, scalar: R) -> M:
        """标量乘法"""
        pass
    
    @abstractmethod
    def zero(self) -> M:
        """零元"""
        pass
    
    def subtract(self, a: M, b: M) -> M:
        """减法"""
        return self.add(a, self.additive_inverse(b))
    
    def additive_inverse(self, a: M) -> M:
        """加法逆元"""
        return self.scalar_multiply(a, self.ring.inverse(self.ring.one()))
    
    def __contains__(self, element: M) -> bool:
        """判断元素是否属于该模"""
        raise NotImplementedError("子类必须实现__contains__方法")
    
    def is_finite_dimensional(self) -> bool:
        """检查是否为有限维模"""
        return False
    
    def dimension(self) -> Optional[int]:
        """模的维数"""
        return None


class VectorSpaceElement(ModuleElement):
    """向量空间元素抽象基类"""
    
    @abstractmethod
    def __mul__(self, scalar: F) -> 'VectorSpaceElement':
        """标量乘法"""
        pass


class VectorSpace(Module[M, F]):
    """向量空间抽象基类"""
    
    def __init__(self, field: Field[F], name: str = ""):
        """
        初始化向量空间
        
        Args:
            field: 基础域
            name: 向量空间的名称
        """
        super().__init__(field, name)
        self.field = field
    
    def scalar_multiply(self, a: M, scalar: F) -> M:
        """标量乘法"""
        return a * scalar
    
    def is_vector_space(self) -> bool:
        """检查是否为向量空间"""
        return True
    
    def basis(self) -> Optional[List[M]]:
        """向量空间的基"""
        return None


@dataclass
class FiniteDimensionalVectorSpaceElement(VectorSpaceElement):
    """有限维向量空间元素"""
    components: List[F]
    field: Field[F]
    
    def __add__(self, other: 'FiniteDimensionalVectorSpaceElement') -> 'FiniteDimensionalVectorSpaceElement':
        if len(self.components) != len(other.components):
            raise ValueError("向量维度不匹配")
        if self.field != other.field:
            raise ValueError("向量空间基础域不匹配")
        new_components = [self.field.add(a, b) for a, b in zip(self.components, other.components)]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def __sub__(self, other: 'FiniteDimensionalVectorSpaceElement') -> 'FiniteDimensionalVectorSpaceElement':
        if len(self.components) != len(other.components):
            raise ValueError("向量维度不匹配")
        if self.field != other.field:
            raise ValueError("向量空间基础域不匹配")
        new_components = [self.field.subtract(a, b) for a, b in zip(self.components, other.components)]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def __mul__(self, scalar: F) -> 'FiniteDimensionalVectorSpaceElement':
        new_components = [self.field.multiply(a, scalar) for a in self.components]
        return FiniteDimensionalVectorSpaceElement(new_components, self.field)
    
    def is_zero(self) -> bool:
        return all(c.is_zero() for c in self.components)
    
    def __str__(self) -> str:
        components_str = [str(c) for c in self.components]
        return f"({', '.join(components_str)})"


class FiniteDimensionalVectorSpace(VectorSpace[FiniteDimensionalVectorSpaceElement, F]):
    """有限维向量空间"""
    
    def __init__(self, field: Field[F], dimension: int, name: str = ""):
        """
        初始化有限维向量空间
        
        Args:
            field: 基础域
            dimension: 向量空间的维度
            name: 向量空间的名称
        """
        super().__init__(field, name or f"{field.name} Vector Space of dimension {dimension}")
        self._dimension = dimension
    
    def add(self, a: FiniteDimensionalVectorSpaceElement, b: FiniteDimensionalVectorSpaceElement) -> FiniteDimensionalVectorSpaceElement:
        return a + b
    
    def zero(self) -> FiniteDimensionalVectorSpaceElement:
        zero_components = [self.field.zero() for _ in range(self._dimension)]
        return FiniteDimensionalVectorSpaceElement(zero_components, self.field)
    
    def __contains__(self, element: FiniteDimensionalVectorSpaceElement) -> bool:
        return (isinstance(element, FiniteDimensionalVectorSpaceElement) and 
                len(element.components) == self._dimension and 
                element.field == self.field)
    
    def is_finite_dimensional(self) -> bool:
        return True
    
    def dimension(self) -> int:
        return self._dimension
    
    def basis(self) -> List[FiniteDimensionalVectorSpaceElement]:
        """返回标准基"""
        basis = []
        for i in range(self._dimension):
            components = [self.field.zero() for _ in range(self._dimension)]
            components[i] = self.field.one()
            basis.append(FiniteDimensionalVectorSpaceElement(components, self.field))
        return basis
