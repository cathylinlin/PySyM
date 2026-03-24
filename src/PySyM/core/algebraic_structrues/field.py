"""域论模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
from dataclasses import dataclass
from .ring import Ring, RingElement

T = TypeVar('T')  # 域元素类型


class FieldElement(RingElement):
    """域元素抽象基类"""
    
    @abstractmethod
    def __truediv__(self, other: 'FieldElement') -> 'FieldElement':
        """除法"""
        pass
    
    @abstractmethod
    def multiplicative_inverse(self) -> 'FieldElement':
        """乘法逆元"""
        pass


class Field(Ring[T]):
    """域抽象基类"""
    
    @abstractmethod
    def divide(self, a: T, b: T) -> T:
        """除法"""
        pass
    
    @abstractmethod
    def multiplicative_inverse(self, a: T) -> T:
        """乘法逆元"""
        pass
    
    def is_field(self) -> bool:
        """检查是否为域"""
        return True


@dataclass
class RationalFieldElement(FieldElement):
    """有理数域元素"""
    numerator: int  # 分子
    denominator: int  # 分母
    
    def __post_init__(self):
        # 化简分数
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
        gcd = self._gcd(abs(self.numerator), abs(self.denominator))
        if gcd > 0:
            self.numerator //= gcd
            self.denominator //= gcd
    
    def _gcd(self, a: int, b: int) -> int:
        """计算最大公约数"""
        while b:
            a, b = b, a % b
        return a
    
    def __add__(self, other: 'RationalFieldElement') -> 'RationalFieldElement':
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)
    
    def __sub__(self, other: 'RationalFieldElement') -> 'RationalFieldElement':
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)
    
    def __mul__(self, other: 'RationalFieldElement') -> 'RationalFieldElement':
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)
    
    def __truediv__(self, other: 'RationalFieldElement') -> 'RationalFieldElement':
        if other.numerator == 0:
            raise ZeroDivisionError("除数不能为零")
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return RationalFieldElement(new_numerator, new_denominator)
    
    def __pow__(self, n: int) -> 'RationalFieldElement':
        if n >= 0:
            return RationalFieldElement(self.numerator ** n, self.denominator ** n)
        else:
            return RationalFieldElement(self.denominator ** (-n), self.numerator ** (-n))
    
    def inverse(self) -> 'RationalFieldElement':
        # 加法逆元
        return RationalFieldElement(-self.numerator, self.denominator)
    
    def multiplicative_inverse(self) -> 'RationalFieldElement':
        # 乘法逆元
        if self.numerator == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return RationalFieldElement(self.denominator, self.numerator)
    
    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))
    
    def is_identity(self) -> bool:
        return self.numerator == 0 and self.denominator == 1  # 加法单位元
    
    def is_zero(self) -> bool:
        return self.numerator == 0
    
    def order(self) -> int:
        # 有理数的加法阶，只有0的阶是1
        if self.is_zero():
            return 1
        return -1  # 无限阶
    
    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


class RationalField(Field[RationalFieldElement]):
    """有理数域"""
    
    def __init__(self):
        super().__init__("Rational Field")
        self._is_commutative = True
    
    def add(self, a: RationalFieldElement, b: RationalFieldElement) -> RationalFieldElement:
        return a + b
    
    def multiply(self, a: RationalFieldElement, b: RationalFieldElement) -> RationalFieldElement:
        return a * b
    
    def divide(self, a: RationalFieldElement, b: RationalFieldElement) -> RationalFieldElement:
        return a / b
    
    def inverse(self, a: RationalFieldElement) -> RationalFieldElement:
        return a.inverse()
    
    def multiplicative_inverse(self, a: RationalFieldElement) -> RationalFieldElement:
        return a.multiplicative_inverse()
    
    def identity(self) -> RationalFieldElement:
        return RationalFieldElement(0, 1)  # 加法单位元
    
    def zero(self) -> RationalFieldElement:
        return RationalFieldElement(0, 1)
    
    def one(self) -> RationalFieldElement:
        return RationalFieldElement(1, 1)  # 乘法单位元
    
    def __contains__(self, element: RationalFieldElement) -> bool:
        return isinstance(element, RationalFieldElement)
    
    def order(self) -> int:
        return -1  # 无限域
    
    def elements(self) -> List[RationalFieldElement]:
        raise ValueError("有理数域是无限的，无法列出所有元素")
    
    def is_finite(self) -> bool:
        return False


@dataclass
class RealFieldElement(FieldElement):
    """实数域元素"""
    value: float
    
    def __add__(self, other: 'RealFieldElement') -> 'RealFieldElement':
        return RealFieldElement(self.value + other.value)
    
    def __sub__(self, other: 'RealFieldElement') -> 'RealFieldElement':
        return RealFieldElement(self.value - other.value)
    
    def __mul__(self, other: 'RealFieldElement') -> 'RealFieldElement':
        return RealFieldElement(self.value * other.value)
    
    def __truediv__(self, other: 'RealFieldElement') -> 'RealFieldElement':
        if other.value == 0:
            raise ZeroDivisionError("除数不能为零")
        return RealFieldElement(self.value / other.value)
    
    def __pow__(self, n: int) -> 'RealFieldElement':
        return RealFieldElement(self.value ** n)
    
    def inverse(self) -> 'RealFieldElement':
        # 加法逆元
        return RealFieldElement(-self.value)
    
    def multiplicative_inverse(self) -> 'RealFieldElement':
        # 乘法逆元
        if self.value == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return RealFieldElement(1.0 / self.value)
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def is_identity(self) -> bool:
        return abs(self.value) < 1e-10  # 加法单位元
    
    def is_zero(self) -> bool:
        return abs(self.value) < 1e-10
    
    def order(self) -> int:
        # 实数的加法阶，只有0的阶是1
        if self.is_zero():
            return 1
        return -1  # 无限阶
    
    def __str__(self) -> str:
        return str(self.value)


class RealField(Field[RealFieldElement]):
    """实数域"""
    
    def __init__(self):
        super().__init__("Real Field")
        self._is_commutative = True
    
    def add(self, a: RealFieldElement, b: RealFieldElement) -> RealFieldElement:
        return a + b
    
    def multiply(self, a: RealFieldElement, b: RealFieldElement) -> RealFieldElement:
        return a * b
    
    def divide(self, a: RealFieldElement, b: RealFieldElement) -> RealFieldElement:
        return a / b
    
    def inverse(self, a: RealFieldElement) -> RealFieldElement:
        return a.inverse()
    
    def multiplicative_inverse(self, a: RealFieldElement) -> RealFieldElement:
        return a.multiplicative_inverse()
    
    def identity(self) -> RealFieldElement:
        return RealFieldElement(0.0)  # 加法单位元
    
    def zero(self) -> RealFieldElement:
        return RealFieldElement(0.0)
    
    def one(self) -> RealFieldElement:
        return RealFieldElement(1.0)  # 乘法单位元
    
    def __contains__(self, element: RealFieldElement) -> bool:
        return isinstance(element, RealFieldElement)
    
    def order(self) -> int:
        return -1  # 无限域
    
    def elements(self) -> List[RealFieldElement]:
        raise ValueError("实数域是无限的，无法列出所有元素")
    
    def is_finite(self) -> bool:
        return False


@dataclass
class FiniteFieldElement(FieldElement):
    """有限域元素"""
    value: int
    characteristic: int  # 有限域的特征
    
    def __post_init__(self):
        # 确保值在有限域的范围内
        self.value %= self.characteristic
    
    def __add__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement((self.value + other.value) % self.characteristic, self.characteristic)
    
    def __sub__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement((self.value - other.value) % self.characteristic, self.characteristic)
    
    def __mul__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement((self.value * other.value) % self.characteristic, self.characteristic)
    
    def __truediv__(self, other: 'FiniteFieldElement') -> 'FiniteFieldElement':
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if other.value == 0:
            raise ZeroDivisionError("除数不能为零")
        # 使用扩展欧几里得算法求乘法逆元
        inverse = self._mod_inverse(other.value, self.characteristic)
        return FiniteFieldElement((self.value * inverse) % self.characteristic, self.characteristic)
    
    def __pow__(self, n: int) -> 'FiniteFieldElement':
        # 使用费马小定理优化幂运算
        if n < 0:
            n = -n
            return self.multiplicative_inverse() ** n
        result = 1
        base = self.value
        while n > 0:
            if n % 2 == 1:
                result = (result * base) % self.characteristic
            base = (base * base) % self.characteristic
            n //= 2
        return FiniteFieldElement(result, self.characteristic)
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """使用扩展欧几里得算法求模逆元"""
        g, x, _ = self._extended_gcd(a, m)
        if g != 1:
            raise ValueError(f"{a} 和 {m} 不互质，没有模逆元")
        return x % m
    
    def _extended_gcd(self, a: int, b: int) -> tuple:
        """扩展欧几里得算法"""
        if b == 0:
            return (a, 1, 0)
        else:
            g, x, y = self._extended_gcd(b, a % b)
            return (g, y, x - (a // b) * y)
    
    def inverse(self) -> 'FiniteFieldElement':
        # 加法逆元
        return FiniteFieldElement((-self.value) % self.characteristic, self.characteristic)
    
    def multiplicative_inverse(self) -> 'FiniteFieldElement':
        # 乘法逆元
        if self.value == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        inverse = self._mod_inverse(self.value, self.characteristic)
        return FiniteFieldElement(inverse, self.characteristic)
    
    def __hash__(self) -> int:
        return hash((self.value, self.characteristic))
    
    def is_identity(self) -> bool:
        return self.value == 0  # 加法单位元
    
    def is_zero(self) -> bool:
        return self.value == 0
    
    def order(self) -> int:
        # 有限域元素的加法阶
        if self.value == 0:
            return 1
        # 加法阶是特征的因数
        for d in range(1, self.characteristic + 1):
            if (self.value * d) % self.characteristic == 0:
                return d
        return self.characteristic
    
    def __str__(self) -> str:
        return f"{self.value} mod {self.characteristic}"


class FiniteField(Field[FiniteFieldElement]):
    """有限域 GF(p)，其中 p 是素数"""
    
    def __init__(self, characteristic: int):
        """
        初始化有限域
        
        Args:
            characteristic: 有限域的特征，必须是素数
        """
        if not self._is_prime(characteristic):
            raise ValueError("有限域的特征必须是素数")
        super().__init__(f"Finite Field GF({characteristic})")
        self.characteristic = characteristic
        self._is_commutative = True
    
    def _is_prime(self, n: int) -> bool:
        """检查是否为素数"""
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def add(self, a: FiniteFieldElement, b: FiniteFieldElement) -> FiniteFieldElement:
        return a + b
    
    def multiply(self, a: FiniteFieldElement, b: FiniteFieldElement) -> FiniteFieldElement:
        return a * b
    
    def divide(self, a: FiniteFieldElement, b: FiniteFieldElement) -> FiniteFieldElement:
        return a / b
    
    def inverse(self, a: FiniteFieldElement) -> FiniteFieldElement:
        return a.inverse()
    
    def multiplicative_inverse(self, a: FiniteFieldElement) -> FiniteFieldElement:
        return a.multiplicative_inverse()
    
    def identity(self) -> FiniteFieldElement:
        return FiniteFieldElement(0, self.characteristic)  # 加法单位元
    
    def zero(self) -> FiniteFieldElement:
        return FiniteFieldElement(0, self.characteristic)
    
    def one(self) -> FiniteFieldElement:
        return FiniteFieldElement(1, self.characteristic)  # 乘法单位元
    
    def __contains__(self, element: FiniteFieldElement) -> bool:
        return isinstance(element, FiniteFieldElement) and element.characteristic == self.characteristic
    
    def order(self) -> int:
        return self.characteristic  # 有限域的阶
    
    def elements(self) -> List[FiniteFieldElement]:
        """列出有限域的所有元素"""
        return [FiniteFieldElement(i, self.characteristic) for i in range(self.characteristic)]
    
    def is_finite(self) -> bool:
        return True
