"""
对称操作实现

本模块提供对称操作的基础设施和具体实现。
"""
from .base import (
    SymmetryOperation,
    Symmetric,
    Transformable,
    Observable,
)
from .specific_operations import (
    TranslationOperation,
    RotationOperation,
    TimeTranslationOperation,
    ParityOperation,
    TimeReversalOperation,
    GaugeOperation,
    CompositeOperation,
    IdentityOperation,
    TranslationSymmetry,
    RotationSymmetry,
    TimeTranslationSymmetry,
    ParitySymmetry,
    TimeReversalSymmetry,
    GaugeSymmetry,
    CompositeSymmetry,
)
from .generators import (
    MomentumGenerator,
    AngularMomentumGenerator,
    HamiltonianGenerator,
    ParityOperator,
    TimeReversalOperator,
)
from .analyzer import SymmetryAnalyzer
from .invariants import (
    CasimirOperator,
    TopologicalInvariant,
    WindingNumber,
    ChernNumber,
    SymmetryInvariant,
    PontryaginIndex,
    SecondChernNumber,
    TopologicalCharge,
)
from .breaking import (
    SymmetryBreaking,
    ExplicitBreaking,
    SpontaneousBreaking,
    GoldstoneMode,
    HiggsMechanism,
    DynamicalBreaking,
    AnomalyMatching,
)

__all__ = [
    # 基础
    'SymmetryOperation',
    'Symmetric',
    'Transformable',
    'Observable',
    # 具体操作
    'TranslationOperation',
    'RotationOperation',
    'TimeTranslationOperation',
    'ParityOperation',
    'TimeReversalOperation',
    'GaugeOperation',
    'CompositeOperation',
    'IdentityOperation',
    # 对称性类
    'TranslationSymmetry',
    'RotationSymmetry',
    'TimeTranslationSymmetry',
    'ParitySymmetry',
    'TimeReversalSymmetry',
    'GaugeSymmetry',
    'CompositeSymmetry',
    # 生成元
    'MomentumGenerator',
    'AngularMomentumGenerator',
    'HamiltonianGenerator',
    'ParityOperator',
    'TimeReversalOperator',
    # 分析器
    'SymmetryAnalyzer',
    # 不变量
    'CasimirOperator',
    'TopologicalInvariant',
    'WindingNumber',
    'ChernNumber',
    'SymmetryInvariant',
    'PontryaginIndex',
    'SecondChernNumber',
    'TopologicalCharge',
    # 对称性破缺
    'SymmetryBreaking',
    'ExplicitBreaking',
    'SpontaneousBreaking',
    'GoldstoneMode',
    'HiggsMechanism',
    'DynamicalBreaking',
    'AnomalyMatching',
]