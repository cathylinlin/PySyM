"""群同态和群作用实现模块"""

from collections.abc import Callable
from typing import Generic, TypeVar

from .abstract_group import FiniteGroup, GroupElement

T = TypeVar("T")  # 目标集合元素类型


class GroupHomomorphism:
    """群同态表示

    群同态是保持群结构的映射，即满足 f(ab) = f(a)f(b) 对所有 a, b 属于群。
    """

    def __init__(
        self,
        domain: FiniteGroup,
        codomain: FiniteGroup,
        homomorphism: Callable[[GroupElement], GroupElement],
    ):
        """
        初始化群同态

        Args:
            domain: 同态的定义域（群H）
            codomain: 同态的值域（群Aut(N)）
            homomorphism: 同态函数
        """
        self.domain = domain
        self.codomain = codomain
        self.homomorphism = homomorphism

    def __call__(self, element: GroupElement) -> GroupElement:
        """应用同态

        Args:
            element: 定义域中的元素

        Returns:
            元素在同态下的像
        """
        return self.homomorphism(element)

    def is_homomorphism(self) -> bool:
        """检查是否为同态

        验证映射是否满足同态条件：f(ab) = f(a)f(b) 对所有 a, b 属于群。

        Returns:
            如果是同态返回True，否则返回False
        """
        for a in self.domain.elements():
            for b in self.domain.elements():
                if self(a * b) != self(a) * self(b):
                    return False
        return True

    def is_injective(self) -> bool:
        """检查是否为单射

        检查同态的核是否为平凡群（只包含单位元）。

        Returns:
            如果是单射返回True，否则返回False
        """
        # 检查核是否为平凡群
        kernel = [
            a for a in self.domain.elements() if self(a) == self.codomain.identity()
        ]
        return len(kernel) == 1

    def is_surjective(self) -> bool:
        """检查是否为满射

        检查同态的像是否等于值域。

        Returns:
            如果是满射返回True，否则返回False
        """
        # 检查像是否等于值域
        image = set(self(a) for a in self.domain.elements())
        return len(image) == self.codomain.order()

    def is_isomorphism(self) -> bool:
        """检查是否为同构

        检查同态是否同时是单射和满射。

        Returns:
            如果是同构返回True，否则返回False
        """
        return self.is_injective() and self.is_surjective()


class GroupAction(Generic[T]):
    """群作用实现

    群作用是群到集合自同构群的同态，描述了群对集合的对称操作。
    """

    def __init__(
        self,
        group: FiniteGroup,
        target_set: list[T],
        action: Callable[[GroupElement, T], T],
    ):
        """
        初始化群作用

        Args:
            group: 作用的群
            target_set: 目标集合
            action: 群作用函数，接受群元素和集合元素，返回作用结果
        """
        self.group = group
        self.target_set = target_set
        self.action = action

    def apply(self, g: GroupElement, x: T) -> T:
        """应用群作用

        Args:
            g: 群元素
            x: 集合元素

        Returns:
            群元素g作用于x的结果
        """
        return self.action(g, x)

    def orbit(self, x: T) -> list[T]:
        """计算元素x的轨道

        轨道是集合中所有可以通过群作用从x得到的元素。

        Args:
            x: 集合中的元素

        Returns:
            x的轨道
        """
        orbit = []
        for g in self.group.elements():
            orbit.append(self.apply(g, x))
        return orbit

    def stabilizer(self, x: T) -> list[GroupElement]:
        """计算元素x的稳定子群

        稳定子群是群中保持x不变的元素集合。

        Args:
            x: 集合中的元素

        Returns:
            x的稳定子群
        """
        stabilizer = []
        for g in self.group.elements():
            if self.apply(g, x) == x:
                stabilizer.append(g)
        return stabilizer

    def orbit_stabilizer_theorem(self, x: T) -> int:
        """应用轨道-稳定子定理

        轨道-稳定子定理：轨道的大小乘以稳定子群的大小等于群的阶。

        Args:
            x: 集合中的元素

        Returns:
            轨道大小乘以稳定子群大小的结果，应该等于群的阶
        """
        orbit_size = len(self.orbit(x))
        stabilizer_size = len(self.stabilizer(x))
        return orbit_size * stabilizer_size
