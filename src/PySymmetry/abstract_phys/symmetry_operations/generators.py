"""对称操作生成元

该模块提供各种对称操作的生成元实现，包括：
- MomentumGenerator: 动量生成元
- AngularMomentumGenerator: 角动量生成元
- HamiltonianGenerator: 哈密顿量生成元
- ParityOperator: 宇称算符
- TimeReversalOperator: 时间反演算符
"""


class MomentumGenerator:
    """动量生成元"""

    def __init__(self, index: int):
        """初始化动量生成元

        Args:
            index: 动量分量索引
        """
        self._index = index

    @property
    def index(self) -> int:
        """动量分量索引"""
        return self._index

    def __repr__(self):
        return f"MomentumGenerator({self._index})"


class AngularMomentumGenerator:
    """角动量生成元"""

    def __init__(self, i: int, j: int):
        """初始化角动量生成元

        Args:
            i: 第一个分量索引
            j: 第二个分量索引
        """
        self._i = i
        self._j = j

    @property
    def i(self) -> int:
        """第一个分量索引"""
        return self._i

    @property
    def j(self) -> int:
        """第二个分量索引"""
        return self._j

    def __repr__(self):
        return f"AngularMomentumGenerator({self._i}, {self._j})"


class HamiltonianGenerator:
    """哈密顿量生成元"""

    def __repr__(self):
        return "HamiltonianGenerator()"


class ParityOperator:
    """宇称算符"""

    def __repr__(self):
        return "ParityOperator()"


class TimeReversalOperator:
    """时间反演算符"""

    def __repr__(self):
        return "TimeReversalOperator()"
