"""抽象李代数基类

该模块定义了李代数的抽象基类，包括：
- LieAlgebraElement: 李代数元素抽象基类
- LieAlgebra: 李代数抽象基类
- LieAlgebraProperties: 李代数属性数据类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, TypeVar, Generic

T = TypeVar('T', bound='LieAlgebraElement')


class LieAlgebraElement(ABC):
    """李代数元素抽象基类"""
    
    @abstractmethod
    def __add__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """加法运算"""
        pass
    
    @abstractmethod
    def __sub__(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """减法运算"""
        pass
    
    @abstractmethod
    def __mul__(self, scalar: float) -> 'LieAlgebraElement':
        """标量乘法"""
        pass
    
    @abstractmethod
    def bracket(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """李括号运算"""
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """相等性判断"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """字符串表示"""
        pass


class LieAlgebra(ABC, Generic[T]):
    """李代数抽象基类"""
    
    @abstractmethod
    def __init__(self, dimension: int):
        """初始化李代数"""
        self.dimension = dimension
    
    @abstractmethod
    def zero(self) -> T:
        """返回零元素"""
        pass
    
    @abstractmethod
    def bracket(self, x: T, y: T) -> T:
        """李括号运算"""
        pass
    
    @abstractmethod
    def add(self, x: T, y: T) -> T:
        """加法运算"""
        pass
    
    @abstractmethod
    def scalar_multiply(self, x: T, scalar: float) -> T:
        """标量乘法"""
        pass
    
    @abstractmethod
    def basis(self) -> List[T]:
        """返回李代数的一组基"""
        pass
    
    @abstractmethod
    def from_vector(self, vector: List[float]) -> T:
        """从向量创建李代数元素"""
        pass
    
    @abstractmethod
    def to_vector(self, element: T) -> List[float]:
        """将李代数元素转换为向量"""
        pass
    
    @abstractmethod
    def properties(self) -> 'LieAlgebraProperties':
        """返回李代数的属性"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """字符串表示"""
        pass


@dataclass
class LieAlgebraProperties:
    """李代数属性数据类"""
    name: str  # 李代数名称
    dimension: int  # 李代数维度
    is_semisimple: bool  # 是否半单
    is_simple: bool  # 是否单
    is_abelian: bool  # 是否交换
    root_system_type: Optional[str] = None  # 根系类型
    rank: Optional[int] = None  # 秩
