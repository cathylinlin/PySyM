"""
对称操作实现

本模块提供对称操作的基础设施和具体实现。
"""

from .analyzer import SymmetryAnalyzer
from .base import (
    Observable,
    Symmetric,
    SymmetryOperation,
    Transformable,
)
from .breaking import (
    AnomalyMatching,
    DynamicalBreaking,
    ExplicitBreaking,
    GoldstoneMode,
    HiggsMechanism,
    SpontaneousBreaking,
    SymmetryBreaking,
)
from .generators import (
    AngularMomentumGenerator,
    HamiltonianGenerator,
    MomentumGenerator,
    ParityOperator,
    TimeReversalOperator,
)
from .invariants import (
    CasimirOperator,
    ChernNumber,
    PontryaginIndex,
    SecondChernNumber,
    SymmetryInvariant,
    TopologicalCharge,
    TopologicalInvariant,
    WindingNumber,
)
from .specific_operations import (
    CompositeOperation,
    CompositeSymmetry,
    GaugeOperation,
    GaugeSymmetry,
    IdentityOperation,
    ParityOperation,
    ParitySymmetry,
    RotationOperation,
    RotationSymmetry,
    TimeReversalOperation,
    TimeReversalSymmetry,
    TimeTranslationOperation,
    TimeTranslationSymmetry,
    TranslationOperation,
    TranslationSymmetry,
)

__all__ = [
    # 基础
    "SymmetryOperation",
    "Symmetric",
    "Transformable",
    "Observable",
    # 具体操作
    "TranslationOperation",
    "RotationOperation",
    "TimeTranslationOperation",
    "ParityOperation",
    "TimeReversalOperation",
    "GaugeOperation",
    "CompositeOperation",
    "IdentityOperation",
    # 对称性类
    "TranslationSymmetry",
    "RotationSymmetry",
    "TimeTranslationSymmetry",
    "ParitySymmetry",
    "TimeReversalSymmetry",
    "GaugeSymmetry",
    "CompositeSymmetry",
    # 生成元
    "MomentumGenerator",
    "AngularMomentumGenerator",
    "HamiltonianGenerator",
    "ParityOperator",
    "TimeReversalOperator",
    # 分析器
    "SymmetryAnalyzer",
    # 不变量
    "CasimirOperator",
    "TopologicalInvariant",
    "WindingNumber",
    "ChernNumber",
    "SymmetryInvariant",
    "PontryaginIndex",
    "SecondChernNumber",
    "TopologicalCharge",
    # 对称性破缺
    "SymmetryBreaking",
    "ExplicitBreaking",
    "SpontaneousBreaking",
    "GoldstoneMode",
    "HiggsMechanism",
    "DynamicalBreaking",
    "AnomalyMatching",
]
