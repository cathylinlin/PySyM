"""环论模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Set
from dataclasses import dataclass
from .abstract_algebra import Group, GroupElement, Monoid

T = TypeVar('T')  # 环元素类型
R1 = TypeVar('R1', bound='RingElement')  # 环元素类型，用于矩阵环


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
            # 对于无限环，需要具体判断，这里默认返回False
            # 子类应该重写此方法来提供准确的判断
            return False
        
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
    
    def is_integral_domain(self) -> bool:
        """整数环是整环"""
        return True


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
    
    def is_integral_domain(self) -> bool:
        """多项式环Z[x]是整环"""
        return True


@dataclass
class MatrixRingElement(RingElement):
    """矩阵环元素"""
    entries: List[List[R1]]  # 矩阵元素，二维列表
    ring: Ring[R1]  # 基础环
    rows: int  # 行数
    cols: int  # 列数
    
    def __post_init__(self):
        # 验证矩阵维度
        if len(self.entries) != self.rows:
            raise ValueError(f"矩阵行数不匹配，期望{self.rows}行，实际{len(self.entries)}行")
        for row in self.entries:
            if len(row) != self.cols:
                raise ValueError(f"矩阵列数不匹配，期望{self.cols}列，实际{len(row)}列")
    
    def __add__(self, other: 'MatrixRingElement') -> 'MatrixRingElement':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩阵维度不匹配，无法相加")
        if self.ring != other.ring:
            raise ValueError("基础环不匹配，无法相加")
        
        new_entries = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                new_entry = self.ring.add(self.entries[i][j], other.entries[i][j])
                new_row.append(new_entry)
            new_entries.append(new_row)
        return MatrixRingElement(new_entries, self.ring, self.rows, self.cols)
    
    def __sub__(self, other: 'MatrixRingElement') -> 'MatrixRingElement':
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("矩阵维度不匹配，无法相减")
        if self.ring != other.ring:
            raise ValueError("基础环不匹配，无法相减")
        
        new_entries = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                new_entry = self.ring.subtract(self.entries[i][j], other.entries[i][j])
                new_row.append(new_entry)
            new_entries.append(new_row)
        return MatrixRingElement(new_entries, self.ring, self.rows, self.cols)
    
    def __mul__(self, other: 'MatrixRingElement') -> 'MatrixRingElement':
        if self.cols != other.rows:
            raise ValueError("矩阵维度不匹配，无法相乘")
        if self.ring != other.ring:
            raise ValueError("基础环不匹配，无法相乘")
        
        new_entries = []
        for i in range(self.rows):
            new_row = []
            for j in range(other.cols):
                entry = self.ring.zero()
                for k in range(self.cols):
                    entry = self.ring.add(entry, self.ring.multiply(self.entries[i][k], other.entries[k][j]))
                new_row.append(entry)
            new_entries.append(new_row)
        return MatrixRingElement(new_entries, self.ring, self.rows, other.cols)
    
    def __pow__(self, n: int) -> 'MatrixRingElement':
        if self.rows != self.cols:
            raise ValueError("只有方阵才能进行幂运算")
        
        # 初始化为单位矩阵
        result = MatrixRingElement([[self.ring.one() if i == j else self.ring.zero() for j in range(self.rows)] for i in range(self.rows)], 
                                self.ring, self.rows, self.cols)
        
        if n == 0:
            return result
        
        base = self
        exponent = n
        
        if n < 0:
            # 对于负幂次，需要求矩阵的逆
            # 这里简化处理，实际实现中需要根据具体的环来计算逆矩阵
            raise NotImplementedError("负幂次需要矩阵的逆，尚未实现")
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * base
            base = base * base
            exponent //= 2
        return result
    
    def inverse(self) -> 'MatrixRingElement':
        # 加法逆元
        new_entries = []
        for row in self.entries:
            new_row = [self.ring.inverse(entry) for entry in row]
            new_entries.append(new_row)
        return MatrixRingElement(new_entries, self.ring, self.rows, self.cols)
    
    def __hash__(self) -> int:
        # 将矩阵转换为元组以计算哈希值
        flat_entries = tuple(tuple(row) for row in self.entries)
        return hash((flat_entries, self.ring, self.rows, self.cols))
    
    def is_identity(self) -> bool:
        # 加法单位元（零矩阵）
        for row in self.entries:
            for entry in row:
                if not entry.is_zero():
                    return False
        return True
    
    def is_zero(self) -> bool:
        # 零矩阵
        for row in self.entries:
            for entry in row:
                if not entry.is_zero():
                    return False
        return True
    
    def order(self) -> int:
        # 矩阵的加法阶，只有零矩阵的阶是1
        if self.is_zero():
            return 1
        return -1  # 无限阶
    
    def __str__(self) -> str:
        if self.rows == 0 or self.cols == 0:
            return "[]"
        
        # 计算每列的最大宽度
        col_widths = [0] * self.cols
        for row in self.entries:
            for j, entry in enumerate(row):
                col_widths[j] = max(col_widths[j], len(str(entry)))
        
        # 构建矩阵字符串
        lines = []
        for row in self.entries:
            line = "[".ljust(2)
            for j, entry in enumerate(row):
                entry_str = str(entry)
                line += entry_str.ljust(col_widths[j] + 2)
            line += "]"
            lines.append(line)
        
        return "\n".join(lines)


class MatrixRing(Ring[MatrixRingElement]):
    """矩阵环 M_n(R)，其中 R 是任意环"""
    
    def __init__(self, ring: Ring[R1], n: int, name: str = ""):
        """
        初始化矩阵环
        
        Args:
            ring: 基础环
            n: 矩阵的阶数（行数和列数）
            name: 矩阵环的名称
        """
        super().__init__(name or f"Matrix Ring M_{n}({ring.name})")
        self.ring = ring
        self.n = n
        self._is_commutative = ring.is_commutative() and n == 1  # 只有1x1矩阵环是交换的
    
    def add(self, a: MatrixRingElement, b: MatrixRingElement) -> MatrixRingElement:
        return a + b
    
    def multiply(self, a: MatrixRingElement, b: MatrixRingElement) -> MatrixRingElement:
        return a * b
    
    def inverse(self, a: MatrixRingElement) -> MatrixRingElement:
        return a.inverse()
    
    def identity(self) -> MatrixRingElement:
        # 加法单位元（零矩阵）
        zero_entries = [[self.ring.zero() for _ in range(self.n)] for _ in range(self.n)]
        return MatrixRingElement(zero_entries, self.ring, self.n, self.n)
    
    def zero(self) -> MatrixRingElement:
        # 零元（零矩阵）
        zero_entries = [[self.ring.zero() for _ in range(self.n)] for _ in range(self.n)]
        return MatrixRingElement(zero_entries, self.ring, self.n, self.n)
    
    def one(self) -> MatrixRingElement:
        # 乘法单位元（单位矩阵）
        identity_entries = [[self.ring.one() if i == j else self.ring.zero() for j in range(self.n)] for i in range(self.n)]
        return MatrixRingElement(identity_entries, self.ring, self.n, self.n)
    
    def __contains__(self, element: MatrixRingElement) -> bool:
        return (isinstance(element, MatrixRingElement) and 
                element.ring == self.ring and 
                element.rows == self.n and 
                element.cols == self.n)
    
    def order(self) -> int:
        if self.ring.is_finite():
            # 有限环上的矩阵环的阶是 |R|^(n^2)
            ring_order = self.ring.order()
            if ring_order > 0:
                return ring_order ** (self.n ** 2)
        return -1  # 无限环
    
    def elements(self) -> List[MatrixRingElement]:
        if not self.ring.is_finite():
            raise ValueError("基础环是无限的，无法列出所有矩阵元素")
        
        # 生成所有可能的矩阵元素
        from itertools import product
        ring_elements = self.ring.elements()
        # 生成所有可能的行
        row_product = product(ring_elements, repeat=self.n)
        # 生成所有可能的矩阵
        matrix_product = product(row_product, repeat=self.n)
        
        elements = []
        for matrix_tuple in matrix_product:
            entries = [list(row) for row in matrix_tuple]
            elements.append(MatrixRingElement(entries, self.ring, self.n, self.n))
        
        return elements
    
    def is_finite(self) -> bool:
        return self.ring.is_finite()