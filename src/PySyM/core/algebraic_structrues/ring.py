"""环论模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Set
from dataclasses import dataclass
from .abstract_algebra import Group, GroupElement, Monoid

T = TypeVar('T')  # 环元素类型


class RingElement(GroupElement):
    """环元素抽象基类"""
    
    @abstractmethod
    def __add__(self, other: 'RingElement') -> 'RingElement':
        """加法"""
        pass
    
    @abstractmethod
    def __sub__(self, other: 'RingElement') -> 'RingElement':
        """减法"""
        pass
    
    @abstractmethod
    def __mul__(self, other: 'RingElement') -> 'RingElement':
        """乘法"""
        pass
    
    @abstractmethod
    def is_zero(self) -> bool:
        """是否为零元"""
        pass


class Ring(Group[T], Monoid[T]):
    """环抽象基类"""
    
    @abstractmethod
    def add(self, a: T, b: T) -> T:
        """加法"""
        pass
    
    @abstractmethod
    def zero(self) -> T:
        """零元"""
        pass
    
    @abstractmethod
    def multiply(self, a: T, b: T) -> T:
        """乘法"""
        pass
    
    @abstractmethod
    def one(self) -> T:
        """乘法单位元"""
        pass
    
    def subtract(self, a: T, b: T) -> T:
        """减法"""
        return self.add(a, self.inverse(b))
    
    def is_commutative(self) -> bool:
        """检查是否为交换环"""
        if hasattr(self, '_is_commutative'):
            return self._is_commutative
        
        # 默认假设是交换环，除非有特殊说明
        self._is_commutative = True
        return True
    
    def is_integral_domain(self) -> bool:
        """检查是否为整环"""
        if not self.is_commutative():
            return False
        
        # 检查是否有零因子
        if not self.is_finite():
            # 对于无限环，默认假设是整环
            return True
        
        elements = self.elements()
        for a in elements:
            if a == self.zero():
                continue
            for b in elements:
                if b == self.zero():
                    continue
                if self.multiply(a, b) == self.zero():
                    return False
        return True
    
    def is_field(self) -> bool:
        """检查是否为域"""
        if not self.is_integral_domain():
            return False
        
        # 检查非零元素是否都有乘法逆元
        if not self.is_finite():
            # 对于无限环，默认假设不是域（除了特殊情况）
            return False
        
        elements = self.elements()
        for a in elements:
            if a == self.zero():
                continue
            has_inverse = False
            for b in elements:
                if self.multiply(a, b) == self.one():
                    has_inverse = True
                    break
            if not has_inverse:
                return False
        return True


@dataclass
class IntegerRingElement(RingElement):
    """整数环元素"""
    value: int
    
    def __add__(self, other: 'IntegerRingElement') -> 'IntegerRingElement':
        return IntegerRingElement(self.value + other.value)
    
    def __sub__(self, other: 'IntegerRingElement') -> 'IntegerRingElement':
        return IntegerRingElement(self.value - other.value)
    
    def __mul__(self, other: 'IntegerRingElement') -> 'IntegerRingElement':
        return IntegerRingElement(self.value * other.value)
    
    def __pow__(self, n: int) -> 'IntegerRingElement':
        return IntegerRingElement(self.value ** n)
    
    def inverse(self) -> 'IntegerRingElement':
        # 整数加法逆元
        return IntegerRingElement(-self.value)
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def is_identity(self) -> bool:
        return self.value == 0  # 加法单位元
    
    def is_zero(self) -> bool:
        return self.value == 0
    
    def order(self) -> int:
        # 整数的加法阶，只有0的阶是1
        if self.value == 0:
            return 1
        return -1  # 无限阶
    
    def __str__(self) -> str:
        return str(self.value)


class IntegerRing(Ring[IntegerRingElement]):
    """整数环"""
    
    def __init__(self):
        super().__init__("Integer Ring")
        self._is_commutative = True
    
    def add(self, a: IntegerRingElement, b: IntegerRingElement) -> IntegerRingElement:
        return a + b
    
    def multiply(self, a: IntegerRingElement, b: IntegerRingElement) -> IntegerRingElement:
        return a * b
    
    def inverse(self, a: IntegerRingElement) -> IntegerRingElement:
        return a.inverse()
    
    def identity(self) -> IntegerRingElement:
        return IntegerRingElement(0)  # 加法单位元
    
    def zero(self) -> IntegerRingElement:
        return IntegerRingElement(0)
    
    def one(self) -> IntegerRingElement:
        return IntegerRingElement(1)  # 乘法单位元
    
    def __contains__(self, element: IntegerRingElement) -> bool:
        return isinstance(element, IntegerRingElement)
    
    def order(self) -> int:
        return -1  # 无限环
    
    def elements(self) -> List[IntegerRingElement]:
        raise ValueError("整数环是无限的，无法列出所有元素")
    
    def is_finite(self) -> bool:
        return False


@dataclass
class PolynomialRingElement(RingElement):
    """多项式环元素"""
    coefficients: List[int]  # 系数列表，从常数项开始
    
    def __post_init__(self):
        # 移除最高次项的零系数
        while len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            self.coefficients.pop()
    
    def __add__(self, other: 'PolynomialRingElement') -> 'PolynomialRingElement':
        max_len = max(len(self.coefficients), len(other.coefficients))
        new_coeffs = []
        for i in range(max_len):
            a = self.coefficients[i] if i < len(self.coefficients) else 0
            b = other.coefficients[i] if i < len(other.coefficients) else 0
            new_coeffs.append(a + b)
        return PolynomialRingElement(new_coeffs)
    
    def __sub__(self, other: 'PolynomialRingElement') -> 'PolynomialRingElement':
        max_len = max(len(self.coefficients), len(other.coefficients))
        new_coeffs = []
        for i in range(max_len):
            a = self.coefficients[i] if i < len(self.coefficients) else 0
            b = other.coefficients[i] if i < len(other.coefficients) else 0
            new_coeffs.append(a - b)
        return PolynomialRingElement(new_coeffs)
    
    def __mul__(self, other: 'PolynomialRingElement') -> 'PolynomialRingElement':
        new_coeffs = [0] * (len(self.coefficients) + len(other.coefficients) - 1)
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                new_coeffs[i + j] += a * b
        return PolynomialRingElement(new_coeffs)
    
    def __pow__(self, n: int) -> 'PolynomialRingElement':
        if n == 0:
            return PolynomialRingElement([1])
        result = self
        for _ in range(1, n):
            result = result * self
        return result
    
    def inverse(self) -> 'PolynomialRingElement':
        # 多项式加法逆元
        return PolynomialRingElement([-c for c in self.coefficients])
    
    def __hash__(self) -> int:
        return hash(tuple(self.coefficients))
    
    def is_identity(self) -> bool:
        return self.coefficients == [0]  # 加法单位元
    
    def is_zero(self) -> bool:
        return self.coefficients == [0]
    
    def order(self) -> int:
        # 多项式的加法阶，只有零多项式的阶是1
        if self.is_zero():
            return 1
        return -1  # 无限阶
    
    def degree(self) -> int:
        """多项式的次数"""
        if self.is_zero():
            return -1
        return len(self.coefficients) - 1
    
    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        terms = []
        for i, coeff in enumerate(self.coefficients):
            if coeff == 0:
                continue
            if i == 0:
                terms.append(str(coeff))
            elif i == 1:
                terms.append(f"{coeff}x")
            else:
                terms.append(f"{coeff}x^{i}")
        # 处理负号，将 " + -" 替换为 " - "
        result = " + ".join(terms)
        return result.replace(" + -", " - ")


class PolynomialRing(Ring[PolynomialRingElement]):
    """多项式环 Z[x]"""
    
    def __init__(self):
        super().__init__("Polynomial Ring Z[x]")
        self._is_commutative = True
    
    def add(self, a: PolynomialRingElement, b: PolynomialRingElement) -> PolynomialRingElement:
        return a + b
    
    def multiply(self, a: PolynomialRingElement, b: PolynomialRingElement) -> PolynomialRingElement:
        return a * b
    
    def inverse(self, a: PolynomialRingElement) -> PolynomialRingElement:
        return a.inverse()
    
    def identity(self) -> PolynomialRingElement:
        return PolynomialRingElement([0])  # 加法单位元
    
    def zero(self) -> PolynomialRingElement:
        return PolynomialRingElement([0])
    
    def one(self) -> PolynomialRingElement:
        return PolynomialRingElement([1])  # 乘法单位元
    
    def __contains__(self, element: PolynomialRingElement) -> bool:
        return isinstance(element, PolynomialRingElement)
    
    def order(self) -> int:
        return -1  # 无限环
    
    def elements(self) -> List[PolynomialRingElement]:
        raise ValueError("多项式环是无限的，无法列出所有元素")
    
    def is_finite(self) -> bool:
        return False
