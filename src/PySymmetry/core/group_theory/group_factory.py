from .abstract_group import Group


class GroupFactory:
    """用于生成抽象群的工厂类"""

    @staticmethod
    def cyclic_group(n: int) -> Group[int]:
        """生成n阶循环群 C_n

        Args:
            n: 群的阶

        Returns:
            n阶循环群
        """
        from .specific_group import CyclicGroup

        return CyclicGroup(n)

    @staticmethod
    def symmetric_group(n: int) -> Group[tuple[int, ...]]:
        """生成n次对称群 S_n

        Args:
            n: 对称群的次数

        Returns:
            n次对称群
        """
        from .specific_group import SymmetricGroup

        return SymmetricGroup(n)

    @staticmethod
    def dihedral_group(n: int) -> Group[tuple[int, int]]:
        """生成2n阶二面体群 D_n

        Args:
            n: 正多边形的边数

        Returns:
            2n阶二面体群
        """
        from .specific_group import DihedralGroup

        return DihedralGroup(n)

    @staticmethod
    def quaternion_group() -> Group[str]:
        """生成四元数群 Q_8

        Returns:
            8阶四元数群
        """
        from .specific_group import QuaternionGroup

        return QuaternionGroup()

    @staticmethod
    def klein_four_group() -> Group[tuple[int, int]]:
        """生成克莱因四元群 V_4

        Returns:
            克莱因四元群
        """
        from .specific_group import KleinGroup

        return KleinGroup()

    @staticmethod
    def alternating_group(n: int) -> Group[tuple[int, ...]]:
        """生成n次交错群 A_n

        Args:
            n: 交错群的次数

        Returns:
            n次交错群
        """
        from .specific_group import AlternatingGroup

        return AlternatingGroup(n)


class FreeGroupElement:
    """自由群元素"""

    def __init__(self, word: list[str], group: "FreeGroup"):
        """
        初始化自由群元素
        :param word: 生成元及其逆元的序列
        :param group: 所属的自由群
        """
        self.word = self._reduce(word)
        self.group = group

    def _reduce(self, word: list[str]) -> list[str]:
        """化简字"""
        reduced = []
        for letter in word:
            if reduced and self._inverse(letter) == reduced[-1]:
                reduced.pop()
            else:
                reduced.append(letter)
        return reduced

    def _inverse(self, letter: str) -> str:
        """获取生成元的逆元"""
        if letter.startswith("-"):
            return letter[1:]
        else:
            return "-" + letter

    def __mul__(self, other: "FreeGroupElement") -> "FreeGroupElement":
        """群乘法"""
        if self.group != other.group:
            raise ValueError("两个元素不属于同一个自由群")
        return FreeGroupElement(self.word + other.word, self.group)

    def __pow__(self, n: int) -> "FreeGroupElement":
        """幂运算"""
        if n == 0:
            return self.group.identity()
        elif n > 0:
            result = self
            for _ in range(n - 1):
                result *= self
            return result
        else:
            return self.inverse() ** (-n)

    def inverse(self) -> "FreeGroupElement":
        """逆元"""
        inverse_word = [self._inverse(letter) for letter in reversed(self.word)]
        return FreeGroupElement(inverse_word, self.group)

    def __hash__(self) -> int:
        """哈希值"""
        return hash(tuple(self.word))

    def is_identity(self) -> bool:
        """是否为单位元"""
        return len(self.word) == 0

    def order(self) -> int:
        """元素阶数（自由群中除单位元外，所有元素的阶都是无限的）"""
        if self.is_identity():
            return 1
        else:
            return -1  # 表示无限阶

    def __eq__(self, other: object) -> bool:
        """相等性比较"""
        if not isinstance(other, FreeGroupElement):
            return NotImplemented
        return self.word == other.word and self.group == other.group

    def __repr__(self) -> str:
        """字符串表示"""
        if not self.word:
            return "e"
        return "".join(self.word)


class FreeGroup(Group[FreeGroupElement]):
    """自由群"""

    def __init__(self, generators: list[str]):
        """
        初始化自由群
        :param generators: 生成元列表
        """
        super().__init__(f"F({len(generators)})")
        self.generators = generators

    def identity(self) -> FreeGroupElement:
        """单位元"""
        return FreeGroupElement([], self)

    def multiply(self, a: FreeGroupElement, b: FreeGroupElement) -> FreeGroupElement:
        """群乘法"""
        return a * b

    def inverse(self, a: FreeGroupElement) -> FreeGroupElement:
        """逆元"""
        return a.inverse()

    def __contains__(self, element: FreeGroupElement) -> bool:
        """判断元素是否属于该群"""
        return isinstance(element, FreeGroupElement) and element.group == self

    def order(self) -> int:
        """群的阶（自由群是无限群）"""
        return -1

    def elements(self) -> list[FreeGroupElement]:
        """所有群元素（自由群是无限群，此方法不适用）"""
        raise ValueError("自由群是无限群，无法生成所有元素")

    def is_abelian(self) -> bool:
        """检查是否为阿贝尔群（自由群只有秩为1时是阿贝尔群）"""
        return len(self.generators) <= 1

    def is_simple(self) -> bool:
        """检查是否为单群（自由群不是单群）"""
        return False

    def is_solvable(self) -> bool:
        """检查是否为可解群（自由群不是可解群，除非秩为1）"""
        return len(self.generators) <= 1
