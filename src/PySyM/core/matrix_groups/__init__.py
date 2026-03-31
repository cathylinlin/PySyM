"""矩阵群模块

该模块提供各种矩阵群的实现，包括：
- 一般线性群 GL(n, F)
- 特殊线性群 SL(n, F)
- 正交群 O(n)
- 特殊正交群 SO(n)
"""

from .base import MatrixGroupElement, MatrixGroup
from .general_linear import GLnElement, GeneralLinearGroup
from .special_linear import SLnElement, SpecialLinearGroup
from .orthogonal import OnElement, OrthogonalGroup, SOnElement, SpecialOrthogonalGroup

__all__ = [
    'MatrixGroupElement',
    'MatrixGroup',
    'GLnElement',
    'GeneralLinearGroup',
    'SLnElement',
    'SpecialLinearGroup',
    'OnElement',
    'OrthogonalGroup',
    'SOnElement',
    'SpecialOrthogonalGroup',
]
