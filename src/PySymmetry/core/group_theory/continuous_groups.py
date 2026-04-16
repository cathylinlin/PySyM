"""连续群实现

该模块提供连续群的实现，包括：
- TranslationGroup: 平移群
- RotationGroup: 旋转群
- TimeTranslationGroup: 时间平移群
"""

import numpy as np

from .abstract_group import Group


class TranslationGroup(Group):
    """平移群 R^n"""

    def __init__(self, dimension: int):
        """初始化平移群

        Args:
            dimension: 空间维度
        """
        super().__init__(f"TranslationGroup({dimension})")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension

    def identity(self):
        """单位元：零平移"""
        return np.zeros(self._dimension)

    def multiply(self, a, b):
        """群乘法：平移的叠加"""
        return a + b

    def inverse(self, element):
        """逆元：相反方向的平移"""
        return -element

    def __contains__(self, element):
        """判断元素是否属于该群"""
        return isinstance(element, np.ndarray) and element.shape == (self._dimension,)

    def order(self):
        """群的阶：无限群返回-1"""
        return -1

    def elements(self):
        """所有群元素：无限群抛出异常"""
        raise ValueError("无限群没有元素列表")

    def is_continuous(self) -> bool:
        """是否为连续群"""
        return True


class RotationGroup(Group):
    """旋转群 SO(n)"""

    def __init__(self, dimension: int):
        """初始化旋转群

        Args:
            dimension: 空间维度
        """
        super().__init__(f"RotationGroup({dimension})")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension

    def identity(self):
        """单位元：单位矩阵"""
        return np.eye(self._dimension)

    def multiply(self, a, b):
        """群乘法：矩阵乘法"""
        return np.dot(a, b)

    def inverse(self, element):
        """逆元：矩阵转置"""
        return element.T

    def __contains__(self, element):
        """判断元素是否属于该群"""
        if not isinstance(element, np.ndarray):
            return False
        if element.shape != (self._dimension, self._dimension):
            return False
        # 检查是否为正交矩阵且行列式为1
        if not np.allclose(np.dot(element.T, element), np.eye(self._dimension)):
            return False
        if not np.isclose(np.linalg.det(element), 1):
            return False
        return True

    def order(self):
        """群的阶：无限群返回-1"""
        return -1

    def elements(self):
        """所有群元素：无限群抛出异常"""
        raise ValueError("无限群没有元素列表")

    def is_continuous(self) -> bool:
        """是否为连续群"""
        return True


class TimeTranslationGroup(Group):
    """时间平移群 R"""

    def __init__(self):
        """初始化时间平移群"""
        super().__init__("TimeTranslationGroup")

    def identity(self):
        """单位元：零时间平移"""
        return 0.0

    def multiply(self, a, b):
        """群乘法：时间平移的叠加"""
        return a + b

    def inverse(self, element):
        """逆元：相反方向的时间平移"""
        return -element

    def __contains__(self, element):
        """判断元素是否属于该群"""
        return isinstance(element, (int, float))

    def order(self):
        """群的阶：无限群返回-1"""
        return -1

    def elements(self):
        """所有群元素：无限群抛出异常"""
        raise ValueError("无限群没有元素列表")

    def is_continuous(self) -> bool:
        """是否为连续群"""
        return True
