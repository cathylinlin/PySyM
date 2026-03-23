"""群论基础模块

该模块提供抽象群论的基础实现，包括：
- 群元素抽象基类 GroupElement
- 群抽象基类 Group
- 子群类 Subgroup
- 群工厂类 GroupFactory
- 群属性数据类 GroupProperties

具体群实现：
- CyclicGroup: 循环群
- SymmetricGroup: 对称群
- DihedralGroup: 二面群
- QuaternionGroup: 四元群
- KleinGroup: Klein群
- AlternatingGroup: 交错群
"""

from .abstract_group import (
    GroupElement,
    Group,
    GroupProperties,
    FiniteGroup
)
from .subgroup import Subgroup
from .coset import Coset, CosetSpace, QuotientGroup
from .product_group import ProductGroup, DirectProductGroup, SemidirectProductGroup
from .group_factory import GroupFactory, FreeGroup, FreeGroupElement
from .group_func import GroupHomomorphism, GroupAction
from .specific_group import (
    CyclicGroup,
    SymmetricGroup,
    DihedralGroup,
    QuaternionGroup,
    KleinGroup,
    AlternatingGroup
)

__all__ = [
    # 抽象基类
    'GroupElement',
    'Group',
    'GroupProperties',
    'FiniteGroup',
    'Subgroup',
    'Coset',
    'CosetSpace',
    'QuotientGroup',
    'ProductGroup',
    'DirectProductGroup',
    'SemidirectProductGroup',
    'GroupFactory',
    'FreeGroup',
    'FreeGroupElement',
    'GroupHomomorphism',
    'GroupAction',
    # 具体群
    'CyclicGroup',
    'SymmetricGroup',
    'DihedralGroup',
    'QuaternionGroup',
    'KleinGroup',
    'AlternatingGroup',
]
