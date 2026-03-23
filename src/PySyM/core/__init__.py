"""PySyM 核心模块

该模块提供群论计算的核心功能，包括：
- group_theory: 抽象群论基础
- matrix_groups: 矩阵群实现
- utils: 工具函数
"""

from . import group_theory
from . import matrix_groups
from . import utils

__all__ = [
    'group_theory',
    'matrix_groups',
    'utils',
]
