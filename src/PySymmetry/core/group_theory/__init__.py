"""群论基础模块

**约定**：具体群（如 ``SymmetricGroup``）的 ``multiply(a,b)`` 一般为「先 ``b`` 后 ``a``」，与矩阵表示 ``ρ(ab)=ρ(a)ρ(b)`` 一致。

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

连续群实现：
- TranslationGroup: 平移群
- RotationGroup: 旋转群
- TimeTranslationGroup: 时间平移群

离散群实现：
- ParityGroup: 宇称群
- TimeReversalGroup: 时间反演群

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
from .continuous_groups import TranslationGroup, RotationGroup, TimeTranslationGroup
from .discrete_groups import ParityGroup, TimeReversalGroup
# 注意：避免从 abstract_phys 导入，以防止循环导入
# 这些类应该在 abstract_phys 层使用，而不是在 core 层导出
# from ...abstract_phys.symmetry_operations.generators import (
#     MomentumGenerator,
#     AngularMomentumGenerator,
#     HamiltonianGenerator,
#     ParityOperator,
#     TimeReversalOperator
# )
# from ...abstract_phys.symmetry_enviroments.gauge_groups import (
#     GaugeGroup,
#     U1GaugeGroup,
#     SU2GaugeGroup,
#     SU3GaugeGroup,
#     GaugeGroupFactory
# )

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
    # 连续群
    'TranslationGroup',
    'RotationGroup',
    'TimeTranslationGroup',
    # 离散群
    'ParityGroup',
    'TimeReversalGroup',
    # 注意：以下类已从 core 层移除，请在 abstract_phys 层直接使用
    # 'MomentumGenerator', 'AngularMomentumGenerator', 'HamiltonianGenerator',
    # 'ParityOperator', 'TimeReversalOperator',
    # 'GaugeGroup', 'U1GaugeGroup', 'SU2GaugeGroup', 'SU3GaugeGroup', 'GaugeGroupFactory',
]
