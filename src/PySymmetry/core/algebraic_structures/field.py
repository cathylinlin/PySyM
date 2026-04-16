"""域论模块"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from .ring import Ring, RingElement

T = TypeVar("T")  # 域元素类型


class FieldElement(RingElement):
    """域元素抽象基类"""

    @abstractmethod
    def __truediv__(self, other: "FieldElement") -> "FieldElement":
        """除法"""
        pass

    @abstractmethod
    def multiplicative_inverse(self) -> "FieldElement":
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

    def __add__(self, other: "RationalFieldElement") -> "RationalFieldElement":
        new_numerator = (
            self.numerator * other.denominator + other.numerator * self.denominator
        )
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)

    def __sub__(self, other: "RationalFieldElement") -> "RationalFieldElement":
        new_numerator = (
            self.numerator * other.denominator - other.numerator * self.denominator
        )
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)

    def __mul__(self, other: "RationalFieldElement") -> "RationalFieldElement":
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return RationalFieldElement(new_numerator, new_denominator)

    def __truediv__(self, other: "RationalFieldElement") -> "RationalFieldElement":
        if other.numerator == 0:
            raise ZeroDivisionError("除数不能为零")
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return RationalFieldElement(new_numerator, new_denominator)

    def __pow__(self, n: int) -> "RationalFieldElement":
        if n >= 0:
            return RationalFieldElement(self.numerator**n, self.denominator**n)
        # 负次幂需要乘法逆元；零元素没有逆元。
        if self.numerator == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return (self.multiplicative_inverse()) ** (-n)

    def inverse(self) -> "RationalFieldElement":
        # 加法逆元
        return RationalFieldElement(-self.numerator, self.denominator)

    def multiplicative_inverse(self) -> "RationalFieldElement":
        # 乘法逆元
        if self.numerator == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return RationalFieldElement(self.denominator, self.numerator)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RationalFieldElement):
            return NotImplemented
        # 比较化简后的分数
        return (
            self.numerator == other.numerator and self.denominator == other.denominator
        )

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

    def add(
        self, a: RationalFieldElement, b: RationalFieldElement
    ) -> RationalFieldElement:
        return a + b

    def multiply(
        self, a: RationalFieldElement, b: RationalFieldElement
    ) -> RationalFieldElement:
        return a * b

    def divide(
        self, a: RationalFieldElement, b: RationalFieldElement
    ) -> RationalFieldElement:
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

    def elements(self) -> list[RationalFieldElement]:
        raise ValueError("有理数域是无限的，无法列出所有元素")

    def is_finite(self) -> bool:
        return False


@dataclass
class RealFieldElement(FieldElement):
    """实数域元素"""

    value: float

    def __add__(self, other: "RealFieldElement") -> "RealFieldElement":
        return RealFieldElement(self.value + other.value)

    def __sub__(self, other: "RealFieldElement") -> "RealFieldElement":
        return RealFieldElement(self.value - other.value)

    def __mul__(self, other: "RealFieldElement") -> "RealFieldElement":
        return RealFieldElement(self.value * other.value)

    def __truediv__(self, other: "RealFieldElement") -> "RealFieldElement":
        if other.value == 0:
            raise ZeroDivisionError("除数不能为零")
        return RealFieldElement(self.value / other.value)

    def __pow__(self, n: int) -> "RealFieldElement":
        return RealFieldElement(self.value**n)

    def inverse(self) -> "RealFieldElement":
        # 加法逆元
        return RealFieldElement(-self.value)

    def multiplicative_inverse(self) -> "RealFieldElement":
        # 乘法逆元
        if self.value == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return RealFieldElement(1.0 / self.value)

    def __hash__(self) -> int:
        # 为了避免浮点数精度问题，使用四舍五入到一定小数位后再计算哈希
        return hash(round(self.value, 10))

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

    def elements(self) -> list[RealFieldElement]:
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

    def __add__(self, other: "FiniteFieldElement") -> "FiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement(
            (self.value + other.value) % self.characteristic, self.characteristic
        )

    def __sub__(self, other: "FiniteFieldElement") -> "FiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement(
            (self.value - other.value) % self.characteristic, self.characteristic
        )

    def __mul__(self, other: "FiniteFieldElement") -> "FiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        return FiniteFieldElement(
            (self.value * other.value) % self.characteristic, self.characteristic
        )

    def __truediv__(self, other: "FiniteFieldElement") -> "FiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if other.value == 0:
            raise ZeroDivisionError("除数不能为零")
        # 使用扩展欧几里得算法求乘法逆元
        inverse = self._mod_inverse(other.value, self.characteristic)
        return FiniteFieldElement(
            (self.value * inverse) % self.characteristic, self.characteristic
        )

    def __pow__(self, n: int) -> "FiniteFieldElement":
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

    def inverse(self) -> "FiniteFieldElement":
        # 加法逆元
        return FiniteFieldElement(
            (-self.value) % self.characteristic, self.characteristic
        )

    def multiplicative_inverse(self) -> "FiniteFieldElement":
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
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def add(self, a: FiniteFieldElement, b: FiniteFieldElement) -> FiniteFieldElement:
        return a + b

    def multiply(
        self, a: FiniteFieldElement, b: FiniteFieldElement
    ) -> FiniteFieldElement:
        return a * b

    def divide(
        self, a: FiniteFieldElement, b: FiniteFieldElement
    ) -> FiniteFieldElement:
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
        return (
            isinstance(element, FiniteFieldElement)
            and element.characteristic == self.characteristic
        )

    def order(self) -> int:
        return self.characteristic  # 有限域的阶

    def elements(self) -> list[FiniteFieldElement]:
        """列出有限域的所有元素"""
        return [
            FiniteFieldElement(i, self.characteristic)
            for i in range(self.characteristic)
        ]

    def is_finite(self) -> bool:
        return True


@dataclass
class ComplexFieldElement(FieldElement):
    """复数域元素"""

    real: float  # 实部
    imag: float  # 虚部

    def __add__(self, other: "ComplexFieldElement") -> "ComplexFieldElement":
        return ComplexFieldElement(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "ComplexFieldElement") -> "ComplexFieldElement":
        return ComplexFieldElement(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "ComplexFieldElement") -> "ComplexFieldElement":
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return ComplexFieldElement(real_part, imag_part)

    def __truediv__(self, other: "ComplexFieldElement") -> "ComplexFieldElement":
        denominator = other.real**2 + other.imag**2
        if denominator == 0:
            raise ZeroDivisionError("除数不能为零")
        real_part = (self.real * other.real + self.imag * other.imag) / denominator
        imag_part = (self.imag * other.real - self.real * other.imag) / denominator
        return ComplexFieldElement(real_part, imag_part)

    def __pow__(self, n: int) -> "ComplexFieldElement":
        if n == 0:
            return ComplexFieldElement(1.0, 0.0)
        result = self
        for _ in range(1, n):
            result = result * self
        return result

    def inverse(self) -> "ComplexFieldElement":
        # 加法逆元
        return ComplexFieldElement(-self.real, -self.imag)

    def multiplicative_inverse(self) -> "ComplexFieldElement":
        # 乘法逆元
        denominator = self.real**2 + self.imag**2
        if denominator == 0:
            raise ZeroDivisionError("零元素没有乘法逆元")
        return ComplexFieldElement(self.real / denominator, -self.imag / denominator)

    def __hash__(self) -> int:
        # 为了避免浮点数精度问题，使用四舍五入到一定小数位后再计算哈希
        return hash((round(self.real, 10), round(self.imag, 10)))

    def is_identity(self) -> bool:
        return abs(self.real) < 1e-10 and abs(self.imag) < 1e-10  # 加法单位元

    def is_zero(self) -> bool:
        return abs(self.real) < 1e-10 and abs(self.imag) < 1e-10

    def order(self) -> int:
        # 复数的加法阶，只有0的阶是1
        if self.is_zero():
            return 1
        return -1  # 无限阶

    def __str__(self) -> str:
        if abs(self.imag) < 1e-10:
            return str(self.real)
        elif abs(self.real) < 1e-10:
            return f"{self.imag}i"
        elif self.imag > 0:
            return f"{self.real} + {self.imag}i"
        else:
            return f"{self.real} - {abs(self.imag)}i"


class ComplexField(Field[ComplexFieldElement]):
    """复数域"""

    def __init__(self):
        super().__init__("Complex Field")
        self._is_commutative = True

    def add(
        self, a: ComplexFieldElement, b: ComplexFieldElement
    ) -> ComplexFieldElement:
        return a + b

    def multiply(
        self, a: ComplexFieldElement, b: ComplexFieldElement
    ) -> ComplexFieldElement:
        return a * b

    def divide(
        self, a: ComplexFieldElement, b: ComplexFieldElement
    ) -> ComplexFieldElement:
        return a / b

    def inverse(self, a: ComplexFieldElement) -> ComplexFieldElement:
        return a.inverse()

    def multiplicative_inverse(self, a: ComplexFieldElement) -> ComplexFieldElement:
        return a.multiplicative_inverse()

    def identity(self) -> ComplexFieldElement:
        return ComplexFieldElement(0.0, 0.0)  # 加法单位元

    def zero(self) -> ComplexFieldElement:
        return ComplexFieldElement(0.0, 0.0)

    def one(self) -> ComplexFieldElement:
        return ComplexFieldElement(1.0, 0.0)  # 乘法单位元

    def __contains__(self, element: ComplexFieldElement) -> bool:
        return isinstance(element, ComplexFieldElement)

    def order(self) -> int:
        return -1  # 无限域

    def elements(self) -> list[ComplexFieldElement]:
        raise ValueError("复数域是无限的，无法列出所有元素")

    def is_finite(self) -> bool:
        return False


@dataclass
class ExtendedFiniteFieldElement(FieldElement):
    """扩展有限域元素 GF(p^n)"""

    coefficients: list[int]  # 多项式系数，从常数项开始
    characteristic: int  # 基础域的特征
    irreducible_poly: list[int]  # 不可约多项式系数

    def __post_init__(self):
        # 确保系数在基础域的范围内
        for i in range(len(self.coefficients)):
            self.coefficients[i] %= self.characteristic
        # 移除最高次项的零系数
        while len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            self.coefficients.pop()

    def _add_poly(self, a: list[int], b: list[int]) -> list[int]:
        """多项式加法"""
        max_len = max(len(a), len(b))
        result = [0] * max_len
        for i in range(max_len):
            if i < len(a):
                result[i] += a[i]
            if i < len(b):
                result[i] += b[i]
            result[i] %= self.characteristic
        # 移除最高次项的零系数
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def _multiply_poly(self, a: list[int], b: list[int]) -> list[int]:
        """多项式乘法"""
        result = [0] * (len(a) + len(b) - 1)
        for i, coeff_a in enumerate(a):
            for j, coeff_b in enumerate(b):
                result[i + j] = (
                    result[i + j] + coeff_a * coeff_b
                ) % self.characteristic
        # 对不可约多项式取模
        return self._mod_poly(result, self.irreducible_poly)

    def _mod_poly(self, dividend: list[int], divisor: list[int]) -> list[int]:
        """多项式取模"""
        if len(dividend) < len(divisor):
            return dividend

        degree_diff = len(dividend) - len(divisor)
        while degree_diff >= 0:
            # 计算商的首项系数
            leading_coeff = (
                dividend[-1]
                * self._mod_inverse(divisor[-1], self.characteristic)
                % self.characteristic
            )
            # 生成商多项式
            quotient = [0] * (degree_diff + 1)
            quotient[-1] = leading_coeff
            # 计算商与除数的乘积
            product = [0] * (len(divisor) + degree_diff)
            for i, coeff in enumerate(divisor):
                product[i + degree_diff] = (coeff * leading_coeff) % self.characteristic
            # 从被除数中减去乘积
            for i in range(len(dividend)):
                if i < len(product):
                    dividend[i] = (dividend[i] - product[i]) % self.characteristic
            # 移除最高次项的零系数
            while len(dividend) > 1 and dividend[-1] == 0:
                dividend.pop()
            degree_diff = len(dividend) - len(divisor)

        return dividend

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

    def __add__(
        self, other: "ExtendedFiniteFieldElement"
    ) -> "ExtendedFiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if self.irreducible_poly != other.irreducible_poly:
            raise ValueError("不可约多项式不同，无法运算")
        new_coeffs = self._add_poly(self.coefficients, other.coefficients)
        return ExtendedFiniteFieldElement(
            new_coeffs, self.characteristic, self.irreducible_poly
        )

    def __sub__(
        self, other: "ExtendedFiniteFieldElement"
    ) -> "ExtendedFiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if self.irreducible_poly != other.irreducible_poly:
            raise ValueError("不可约多项式不同，无法运算")
        # 计算other的加法逆元
        neg_other_coeffs = [(-c) % self.characteristic for c in other.coefficients]
        neg_other = ExtendedFiniteFieldElement(
            neg_other_coeffs, self.characteristic, self.irreducible_poly
        )
        return self + neg_other

    def __mul__(
        self, other: "ExtendedFiniteFieldElement"
    ) -> "ExtendedFiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if self.irreducible_poly != other.irreducible_poly:
            raise ValueError("不可约多项式不同，无法运算")
        new_coeffs = self._multiply_poly(self.coefficients, other.coefficients)
        return ExtendedFiniteFieldElement(
            new_coeffs, self.characteristic, self.irreducible_poly
        )

    def __truediv__(
        self, other: "ExtendedFiniteFieldElement"
    ) -> "ExtendedFiniteFieldElement":
        if self.characteristic != other.characteristic:
            raise ValueError("有限域特征不同，无法运算")
        if self.irreducible_poly != other.irreducible_poly:
            raise ValueError("不可约多项式不同，无法运算")
        if other.is_zero():
            raise ZeroDivisionError("除数不能为零")
        return self * other.multiplicative_inverse()

    def __pow__(self, n: int) -> "ExtendedFiniteFieldElement":
        if n < 0:
            n = -n
            return self.multiplicative_inverse() ** n
        result = ExtendedFiniteFieldElement(
            [1], self.characteristic, self.irreducible_poly
        )  # 乘法单位元
        base = self
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        return result

    def inverse(self) -> "ExtendedFiniteFieldElement":
        # 加法逆元
        neg_coeffs = [(-c) % self.characteristic for c in self.coefficients]
        return ExtendedFiniteFieldElement(
            neg_coeffs, self.characteristic, self.irreducible_poly
        )

    def multiplicative_inverse(self) -> "ExtendedFiniteFieldElement":
        # 乘法逆元，使用扩展欧几里得算法
        if self.is_zero():
            raise ZeroDivisionError("零元素没有乘法逆元")

        # 扩展欧几里得算法求多项式逆元
        a = self.coefficients
        b = self.irreducible_poly
        x = [1]  # 初始化为1
        y = [0]  # 初始化为0

        while len(b) > 1 or (len(b) == 1 and b[0] != 0):
            if len(a) < len(b):
                a, b = b, a
                x, y = y, x

            degree_diff = len(a) - len(b)
            # 计算商的首项系数
            leading_coeff = (
                a[-1]
                * self._mod_inverse(b[-1], self.characteristic)
                % self.characteristic
            )
            # 生成商多项式
            quotient = [0] * (degree_diff + 1)
            quotient[-1] = leading_coeff
            # 计算商与除数的乘积
            product = [0] * (len(b) + degree_diff)
            for i, coeff in enumerate(b):
                product[i + degree_diff] = (coeff * leading_coeff) % self.characteristic
            # 更新a和x
            new_a = [0] * len(a)
            for i in range(len(a)):
                if i < len(product):
                    new_a[i] = (a[i] - product[i]) % self.characteristic
            # 移除最高次项的零系数
            while len(new_a) > 1 and new_a[-1] == 0:
                new_a.pop()

            new_x = [0] * max(len(x), len(quotient))
            for i in range(len(x)):
                new_x[i] = x[i]
            for i in range(len(quotient)):
                if i < len(y):
                    new_x[i + (len(new_x) - len(quotient))] = (
                        new_x[i + (len(new_x) - len(quotient))] - quotient[i] * y[i]
                    ) % self.characteristic
            # 移除最高次项的零系数
            while len(new_x) > 1 and new_x[-1] == 0:
                new_x.pop()

            a, b = new_a, b
            x, y = new_x, y

        # 归一化逆元
        inv_leading = self._mod_inverse(a[0], self.characteristic)
        inv_coeffs = [(c * inv_leading) % self.characteristic for c in x]
        return ExtendedFiniteFieldElement(
            inv_coeffs, self.characteristic, self.irreducible_poly
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.coefficients),
                self.characteristic,
                tuple(self.irreducible_poly),
            )
        )

    def is_identity(self) -> bool:
        return self.coefficients == [0]  # 加法单位元

    def is_zero(self) -> bool:
        return self.coefficients == [0]

    def multiplicative_order(self) -> int:
        # 有限域元素的乘法阶
        if self.is_zero():
            return 0
        # 乘法群是循环群，阶数整除 p^n - 1
        order = 1
        temp = self
        while not temp.is_one():
            temp = temp * self
            order += 1
        return order

    def additive_order(self) -> int:
        # 有限域元素的加法阶
        if self.is_zero():
            return 1
        # 加法阶是特征的因数
        for d in range(1, self.characteristic + 1):
            # 计算 d * self
            result = ExtendedFiniteFieldElement(
                [0], self.characteristic, self.irreducible_poly
            )
            temp = self
            for _ in range(d):
                result = result + temp
            if result.is_zero():
                return d
        return self.characteristic

    def order(self) -> int:
        # 默认返回乘法阶
        return self.multiplicative_order()

    def is_one(self) -> bool:
        return self.coefficients == [1]

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


class ExtendedFiniteField(Field[ExtendedFiniteFieldElement]):
    """扩展有限域 GF(p^n)"""

    def __init__(self, characteristic: int, degree: int, irreducible_poly: list[int]):
        """
        初始化扩展有限域

        Args:
            characteristic: 基础域的特征，必须是素数
            degree: 扩展次数
            irreducible_poly: 不可约多项式系数，从常数项开始
        """
        if not self._is_prime(characteristic):
            raise ValueError("有限域的特征必须是素数")
        if len(irreducible_poly) != degree + 1:
            raise ValueError(f"不可约多项式的次数必须为{degree}")
        super().__init__(f"Extended Finite Field GF({characteristic}^{degree})")
        self.characteristic = characteristic
        self.degree = degree
        self.irreducible_poly = irreducible_poly
        self._is_commutative = True

    def _is_prime(self, n: int) -> bool:
        """检查是否为素数"""
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def add(
        self, a: ExtendedFiniteFieldElement, b: ExtendedFiniteFieldElement
    ) -> ExtendedFiniteFieldElement:
        return a + b

    def multiply(
        self, a: ExtendedFiniteFieldElement, b: ExtendedFiniteFieldElement
    ) -> ExtendedFiniteFieldElement:
        return a * b

    def divide(
        self, a: ExtendedFiniteFieldElement, b: ExtendedFiniteFieldElement
    ) -> ExtendedFiniteFieldElement:
        return a / b

    def inverse(self, a: ExtendedFiniteFieldElement) -> ExtendedFiniteFieldElement:
        return a.inverse()

    def multiplicative_inverse(
        self, a: ExtendedFiniteFieldElement
    ) -> ExtendedFiniteFieldElement:
        return a.multiplicative_inverse()

    def identity(self) -> ExtendedFiniteFieldElement:
        return ExtendedFiniteFieldElement(
            [0], self.characteristic, self.irreducible_poly
        )  # 加法单位元

    def zero(self) -> ExtendedFiniteFieldElement:
        return ExtendedFiniteFieldElement(
            [0], self.characteristic, self.irreducible_poly
        )

    def one(self) -> ExtendedFiniteFieldElement:
        return ExtendedFiniteFieldElement(
            [1], self.characteristic, self.irreducible_poly
        )  # 乘法单位元

    def __contains__(self, element: ExtendedFiniteFieldElement) -> bool:
        return (
            isinstance(element, ExtendedFiniteFieldElement)
            and element.characteristic == self.characteristic
            and element.irreducible_poly == self.irreducible_poly
        )

    def order(self) -> int:
        return self.characteristic**self.degree  # 有限域的阶

    def elements(self) -> list[ExtendedFiniteFieldElement]:
        """列出有限域的所有元素"""
        elements = []

        # 生成所有次数小于n的多项式
        def generate_polynomials(degree, current):
            if current is None:
                current = []
            if degree == 0:
                elements.append(
                    ExtendedFiniteFieldElement(
                        current.copy(), self.characteristic, self.irreducible_poly
                    )
                )
                return
            for i in range(self.characteristic):
                current.append(i)
                generate_polynomials(degree - 1, current)
                current.pop()

        generate_polynomials(self.degree, [])
        return elements

    def is_finite(self) -> bool:
        return True
