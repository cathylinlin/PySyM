"""
抽象物理层工具模块

提供各种物理计算工具，包括：
- Wigner D 矩阵和球谐函数旋转
- Clebsch-Gordan 系数
- 特征标表
- 张量运算
"""

from .character_tables import (
    CharacterTable,
    CharacterTableDatabase,
    DirectProductCalculator,
    find_allowed_transitions,
    print_character_table,
)
from .clebsch_gordan import (
    ClebschGordan,
    RacahCoefficient,
    Wigner3j,
    Wigner6j,
    Wigner9j,
)
from .spglib_integration import (
    CrystalStructure,
    SpglibAdapter,
    SpglibSpaceGroup,
    analyze_crystal,
    quick_spacegroup,
)
from .spherical_harmonics import (
    GauntCoefficient,
    SphericalHarmonics,
    SphericalHarmonicsAddition,
    SphericalHarmonicsIntegral,
    SphericalHarmonicsOperators,
)
from .tensor_operations import (
    AngularMomentumOperators,
    IrreducibleTensorOperator,
    QuadraticTensorOperator,
    ScalarOperator,
    SixJApplications,
    TensorProduct,
    VectorOperator,
    WignerEckartTheorem,
)
from .wigner_Dmatrix import (
    SpecialRotations,
    SphericalHarmonicsRotation,
    WignerBigD,
    WignerDProperties,
    WignerSmallD,
)

__all__ = [
    # Wigner D 矩阵
    "WignerSmallD",
    "WignerBigD",
    "SpecialRotations",
    "WignerDProperties",
    "SphericalHarmonicsRotation",
    # Clebsch-Gordan
    "ClebschGordan",
    "Wigner3j",
    "Wigner6j",
    "Wigner9j",
    "RacahCoefficient",
    # 球谐函数
    "SphericalHarmonics",
    "SphericalHarmonicsAddition",
    "GauntCoefficient",
    "SphericalHarmonicsIntegral",
    "SphericalHarmonicsOperators",
    # 张量运算
    "IrreducibleTensorOperator",
    "WignerEckartTheorem",
    "ScalarOperator",
    "VectorOperator",
    "QuadraticTensorOperator",
    "AngularMomentumOperators",
    "TensorProduct",
    "SixJApplications",
    # 特征标表
    "CharacterTable",
    "CharacterTableDatabase",
    "DirectProductCalculator",
    "print_character_table",
    "find_allowed_transitions",
    # spglib
    "SpglibAdapter",
    "CrystalStructure",
    "SpglibSpaceGroup",
    "analyze_crystal",
    "quick_spacegroup",
]
