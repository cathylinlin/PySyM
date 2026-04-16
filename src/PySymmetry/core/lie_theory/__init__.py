"""李代数模块

该模块提供李代数的基础实现，包括：
- 抽象李代数基类 LieAlgebra
- 李代数元素类 LieAlgebraElement
- 具体李代数实现
- 李代数表示
- 李代数结构相关功能
- 李代数操作相关功能
- 李代数工厂类

具体李代数实现：
- GeneralLinearLieAlgebra: 一般线性李代数 gl(n)
- SpecialLinearLieAlgebra: 特殊线性李代数 sl(n)
- OrthogonalLieAlgebra: 正交李代数 so(n)
- SymplecticLieAlgebra: 辛李代数 sp(2n)（参数 n 为半数维数）
- UnitaryLieAlgebra: 酉李代数 u(n)
- SpecialUnitaryLieAlgebra: 特殊酉李代数 su(n)
"""

from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement, LieAlgebraProperties
from .lie_algebra_factory import LieAlgebraFactory
from .lie_algebra_operations import (
    LieAlgebraAction,
    LieAlgebraHomomorphism,
    LieBracket,
    LinearLieAlgebraAction,
    LinearLieAlgebraHomomorphism,
    StandardLieBracket,
)
from .lie_algebra_representation import (
    AdjointRepresentation,
    FundamentalRepresentation,
    LieAlgebraRepresentation,
    TensorProductRepresentation,
)
from .lie_algebra_structure import CartanSubalgebra, KillingForm, RootSystem, WeylGroup
from .specific_lie_algebra import (
    GeneralLinearLieAlgebra,
    MatrixLieAlgebraElement,
    OrthogonalLieAlgebra,
    SpecialLinearLieAlgebra,
    SpecialUnitaryLieAlgebra,
    SymplecticLieAlgebra,
    UnitaryLieAlgebra,
)

__all__ = [
    # 抽象基类
    "LieAlgebraElement",
    "LieAlgebra",
    "LieAlgebraProperties",
    "MatrixLieAlgebraElement",
    # 具体李代数
    "GeneralLinearLieAlgebra",
    "SpecialLinearLieAlgebra",
    "OrthogonalLieAlgebra",
    "SymplecticLieAlgebra",
    "UnitaryLieAlgebra",
    "SpecialUnitaryLieAlgebra",
    # 表示
    "LieAlgebraRepresentation",
    "AdjointRepresentation",
    "FundamentalRepresentation",
    "TensorProductRepresentation",
    # 结构
    "CartanSubalgebra",
    "RootSystem",
    "WeylGroup",
    "KillingForm",
    # 操作
    "LieBracket",
    "StandardLieBracket",
    "LieAlgebraHomomorphism",
    "LinearLieAlgebraHomomorphism",
    "LieAlgebraAction",
    "LinearLieAlgebraAction",
    # 工厂
    "LieAlgebraFactory",
]
