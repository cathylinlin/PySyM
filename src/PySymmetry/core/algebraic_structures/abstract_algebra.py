"""抽象代数结构基类模块"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")  # 代数结构元素类型


@dataclass
class AlgebraicProperties:
    """代数结构属性数据类"""

    is_finite: bool  # 是否有限
    characteristic: int | None  # 特征
    order: int | None  # 阶（如果有限）


class SemigroupElement(ABC):
    """半群元素抽象基类"""

    @abstractmethod
    def __mul__(self, other: "SemigroupElement") -> "SemigroupElement":
        """半群乘法"""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """必须可哈希，以便存入集合或作为字典键"""
        pass

    def __eq__(self, other: object) -> bool:
        # 默认情况下尽量做结构性相等比较，避免仅靠 hash() 可能出现的碰撞。
        # 如果子类已经实现了 __eq__，这里通常不会被调用（会被子类覆盖）。
        if type(self) is not type(other):
            return NotImplemented
        if hasattr(self, "__dict__") and hasattr(other, "__dict__"):
            return self.__dict__ == other.__dict__
        return hash(self) == hash(other)


class Semigroup(ABC, Generic[T]):
    """半群抽象基类

    半群是一个带有结合二元运算的集合。
    """

    def __init__(self, name: str = ""):
        """
        初始化半群

        Args:
            name: 半群的名称
        """
        self.name = name
        self._cache: dict[str, Any] = {}
        self._properties: AlgebraicProperties | None = None

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

        # 子类应该重写此方法来提供准确的判断
        raise NotImplementedError("子类必须实现is_finite方法")

    def order(self) -> int | None:
        """半群的阶（无限半群返回None）"""
        if self._properties and self._properties.order is not None:
            return self._properties.order
        return None

    def _update_properties(self, **kwargs) -> None:
        """更新代数结构属性"""
        if self._properties is None:
            self._properties = AlgebraicProperties(
                is_finite=kwargs.get("is_finite", True),
                characteristic=kwargs.get("characteristic"),
                order=kwargs.get("order"),
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
    """幺半群抽象基类

    幺半群是一个带有单位元的半群。
    """

    @abstractmethod
    def identity(self) -> T:
        """单位元"""
        pass

    def is_idempotent(self, element: T) -> bool:
        """检查元素是否为幂等元

        幂等元是指满足 a * a = a 的元素。

        Args:
            element: 要检查的元素

        Returns:
            如果元素是幂等元，返回True，否则返回False
        """
        return self.multiply(element, element) == element


class GroupElement(MonoidElement):
    """群元素抽象基类"""

    @abstractmethod
    def inverse(self) -> "GroupElement":
        """逆元"""
        pass

    @abstractmethod
    def __pow__(self, n: int) -> "GroupElement":
        """幂运算"""
        pass

    @abstractmethod
    def order(self) -> int:
        """元素阶数"""
        pass

    def __truediv__(self, other: "GroupElement") -> "GroupElement":
        """a/b = a * b^(-1)"""
        if not isinstance(other, GroupElement):
            return NotImplemented
        return self * other.inverse()


class Group(Monoid[T]):
    """群抽象基类

    群是一个带有逆元的幺半群。
    """

    @abstractmethod
    def inverse(self, a: T) -> T:
        """逆元

        返回元素的逆元。

        Args:
            a: 要计算逆元的元素

        Returns:
            元素的逆元
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """群的阶（无限群返回-1）

        Returns:
            群的阶数，无限群返回-1
        """
        pass

    @abstractmethod
    def elements(self) -> list[T]:
        """所有群元素（有限群）

        Returns:
            有限群的所有元素列表

        Raises:
            NotImplementedError: 如果群是无限的
        """
        raise NotImplementedError("子类必须实现elements方法")
