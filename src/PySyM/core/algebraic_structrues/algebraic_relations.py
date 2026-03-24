"""代数结构之间的关系和转换模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Callable, Tuple
from .abstract_algebra import Group, GroupElement, Monoid, Semigroup
from .ring import Ring, RingElement
from .field import Field, FieldElement
from .module import Module, ModuleElement, VectorSpace, VectorSpaceElement

# 类型变量
G = TypeVar('G', bound=GroupElement)
H = TypeVar('H', bound=GroupElement)
R1 = TypeVar('R1', bound=RingElement)
R2 = TypeVar('R2', bound=RingElement)
F1 = TypeVar('F1', bound=FieldElement)
F2 = TypeVar('F2', bound=FieldElement)
M1 = TypeVar('M1', bound=ModuleElement)
M2 = TypeVar('M2', bound=ModuleElement)


class GroupHomomorphism(ABC, Generic[G, H]):
    """群同态抽象基类"""
    
    def __init__(self, domain: Group[G], codomain: Group[H]):
        """
        初始化群同态
        
        Args:
            domain: 定义域群
            codomain: 陪域群
        """
        self.domain = domain
        self.codomain = codomain
    
    @abstractmethod
    def __call__(self, element: G) -> H:
        """同态映射"""
        pass
    
    def is_homomorphism(self) -> bool:
        """检查是否为群同态"""
        if not self.domain.is_finite():
            # 对于无限群，默认假设是同态
            return True
        
        elements = self.domain.elements()
        for a in elements:
            for b in elements:
                ab = self.domain.multiply(a, b)  # 使用乘法，因为 Group 类中定义的是 multiply 方法
                f_ab = self(ab)
                f_a = self(a)
                f_b = self(b)
                f_a_f_b = self.codomain.multiply(f_a, f_b)  # 使用乘法
                if f_ab != f_a_f_b:
                    return False
        return True
    
    def kernel(self) -> List[G]:
        """计算同态的核"""
        kernel = []
        if self.domain.is_finite():
            elements = self.domain.elements()
            for elem in elements:
                if self(elem) == self.codomain.identity():
                    kernel.append(elem)
        return kernel
    
    def image(self) -> List[H]:
        """计算同态的像"""
        image = []
        if self.domain.is_finite():
            elements = self.domain.elements()
            for elem in elements:
                image_elem = self(elem)
                if image_elem not in image:
                    image.append(image_elem)
        return image
    
    def is_isomorphism(self) -> bool:
        """检查是否为群同构"""
        if not self.is_homomorphism():
            return False
        
        # 检查是否为单射
        if self.domain.is_finite():
            elements = self.domain.elements()
            image = self.image()
            return len(elements) == len(image)
        
        # 对于无限群，默认假设是同构
        return True


class RingHomomorphism(ABC, Generic[R1, R2]):
    """环同态抽象基类"""
    
    def __init__(self, domain: Ring[R1], codomain: Ring[R2]):
        """
        初始化环同态
        
        Args:
            domain: 定义域环
            codomain: 陪域环
        """
        self.domain = domain
        self.codomain = codomain
    
    @abstractmethod
    def __call__(self, element: R1) -> R2:
        """同态映射"""
        pass
    
    def is_homomorphism(self) -> bool:
        """检查是否为环同态"""
        if not self.domain.is_finite():
            # 对于无限环，默认假设是同态
            return True
        
        elements = self.domain.elements()
        for a in elements:
            for b in elements:
                # 检查加法同态
                a_plus_b = self.domain.add(a, b)
                f_a_plus_b = self(a_plus_b)
                f_a = self(a)
                f_b = self(b)
                f_a_plus_f_b = self.codomain.add(f_a, f_b)
                if f_a_plus_b != f_a_plus_f_b:
                    return False
                
                # 检查乘法同态
                a_times_b = self.domain.multiply(a, b)
                f_a_times_b = self(a_times_b)
                f_a_times_f_b = self.codomain.multiply(f_a, f_b)
                if f_a_times_b != f_a_times_f_b:
                    return False
        
        # 检查单位元映射
        if self(self.domain.one()) != self.codomain.one():
            return False
        
        return True
    
    def kernel(self) -> List[R1]:
        """计算同态的核"""
        kernel = []
        if self.domain.is_finite():
            elements = self.domain.elements()
            for elem in elements:
                if self(elem) == self.codomain.zero():
                    kernel.append(elem)
        return kernel
    
    def image(self) -> List[R2]:
        """计算同态的像"""
        image = []
        if self.domain.is_finite():
            elements = self.domain.elements()
            for elem in elements:
                image_elem = self(elem)
                if image_elem not in image:
                    image.append(image_elem)
        return image
    
    def is_isomorphism(self) -> bool:
        """检查是否为环同构"""
        if not self.is_homomorphism():
            return False
        
        # 检查是否为单射
        if self.domain.is_finite():
            elements = self.domain.elements()
            image = self.image()
            return len(elements) == len(image)
        
        # 对于无限环，默认假设是同构
        return True


class FieldHomomorphism(RingHomomorphism[F1, F2]):
    """域同态类"""
    
    def __init__(self, domain: Field[F1], codomain: Field[F2]):
        """
        初始化域同态
        
        Args:
            domain: 定义域域
            codomain: 陪域域
        """
        super().__init__(domain, codomain)
        self.domain = domain
        self.codomain = codomain
    
    def is_homomorphism(self) -> bool:
        """检查是否为域同态"""
        if not super().is_homomorphism():
            return False
        
        # 检查非零元素的逆元映射
        if self.domain.is_finite():
            elements = self.domain.elements()
            for elem in elements:
                if not self.domain.is_zero(elem):
                    f_elem = self(elem)
                    if self.codomain.is_zero(f_elem):
                        return False
                    f_elem_inv = self.codomain.multiplicative_inverse(f_elem)
                    elem_inv = self.domain.multiplicative_inverse(elem)
                    f_elem_inv_expected = self(elem_inv)
                    if f_elem_inv != f_elem_inv_expected:
                        return False
        
        return True


class ModuleHomomorphism(ABC, Generic[M1, M2, R1]):
    """模同态抽象基类"""
    
    def __init__(self, domain: Module[M1, R1], codomain: Module[M2, R1]):
        """
        初始化模同态
        
        Args:
            domain: 定义域模
            codomain: 陪域模
        """
        self.domain = domain
        self.codomain = codomain
    
    @abstractmethod
    def __call__(self, element: M1) -> M2:
        """同态映射"""
        pass
    
    def is_homomorphism(self) -> bool:
        """检查是否为模同态"""
        # 对于模同态，默认假设是同态
        return True


class VectorSpaceHomomorphism(ModuleHomomorphism[M1, M2, F1]):
    """向量空间同态类"""
    
    def __init__(self, domain: VectorSpace[M1, F1], codomain: VectorSpace[M2, F1]):
        """
        初始化向量空间同态
        
        Args:
            domain: 定义域向量空间
            codomain: 陪域向量空间
        """
        super().__init__(domain, codomain)
        self.domain = domain
        self.codomain = codomain


# 代数结构构造函数
def direct_sum_groups(groups: List[Group[G]]) -> Group[Tuple[G, ...]]:
    """构造群的直和
    
    Args:
        groups: 群列表
    
    Returns:
        群的直和
    """
    class DirectSumGroupElement(GroupElement):
        def __init__(self, elements: Tuple[G, ...]):
            self.elements = elements
        
        def __mul__(self, other: 'DirectSumGroupElement') -> 'DirectSumGroupElement':
            new_elements = tuple(groups[i].multiply(a, b) for i, (a, b) in enumerate(zip(self.elements, other.elements)))
            return DirectSumGroupElement(new_elements)
        
        def __pow__(self, n: int) -> 'DirectSumGroupElement':
            new_elements = tuple(groups[i].multiply(a, a) for i, a in enumerate(self.elements))
            return DirectSumGroupElement(new_elements)
        
        def inverse(self) -> 'DirectSumGroupElement':
            new_elements = tuple(groups[i].inverse(a) for i, a in enumerate(self.elements))
            return DirectSumGroupElement(new_elements)
        
        def __hash__(self) -> int:
            return hash(self.elements)
        
        def is_identity(self) -> bool:
            return all(groups[i].identity() == a for i, a in enumerate(self.elements))
        
        def order(self) -> int:
            orders = [a.order() for a in self.elements]
            # 计算最小公倍数
            def lcm(a, b):
                from math import gcd
                return a * b // gcd(a, b)
            result = 1
            for o in orders:
                result = lcm(result, o)
            return result
    
    class DirectSumGroup(Group[DirectSumGroupElement]):
        def __init__(self, groups: List[Group[G]]):
            super().__init__(f"Direct Sum of {[g.name for g in groups]}")
            self.groups = groups
        
        def identity(self) -> DirectSumGroupElement:
            identities = tuple(g.identity() for g in self.groups)
            return DirectSumGroupElement(identities)
        
        def multiply(self, a: DirectSumGroupElement, b: DirectSumGroupElement) -> DirectSumGroupElement:
            return a * b
        
        def inverse(self, a: DirectSumGroupElement) -> DirectSumGroupElement:
            return a.inverse()
        
        def __contains__(self, element: DirectSumGroupElement) -> bool:
            return isinstance(element, DirectSumGroupElement) and len(element.elements) == len(self.groups)
        
        def order(self) -> int:
            orders = [g.order() for g in self.groups]
            result = 1
            for o in orders:
                result *= o
            return result
        
        def elements(self) -> List[DirectSumGroupElement]:
            # 生成所有可能的元素组合
            from itertools import product
            elements_lists = [g.elements() for g in self.groups]
            return [DirectSumGroupElement(elem_tuple) for elem_tuple in product(*elements_lists)]
    
    return DirectSumGroup(groups)


def polynomial_ring(ring: Ring[R1]) -> Ring:
    """构造多项式环
    
    Args:
        ring: 基础环
    
    Returns:
        多项式环 R[x]
    """
    from .ring import PolynomialRing
    return PolynomialRing()


def finite_field(characteristic: int) -> Field:
    """构造有限域
    
    Args:
        characteristic: 有限域的特征，必须是素数
    
    Returns:
        有限域 GF(characteristic)
    """
    from .field import FiniteField
    return FiniteField(characteristic)


def vector_space(field: Field[F1], dimension: int) -> VectorSpace:
    """构造有限维向量空间
    
    Args:
        field: 基础域
        dimension: 向量空间的维度
    
    Returns:
        有限维向量空间
    """
    from .module import FiniteDimensionalVectorSpace
    return FiniteDimensionalVectorSpace(field, dimension)
