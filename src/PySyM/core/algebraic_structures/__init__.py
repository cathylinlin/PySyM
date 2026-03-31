"""代数结构模块"""

# 抽象代数结构基类
from .abstract_algebra import (
    AlgebraicProperties,
    SemigroupElement,
    Semigroup,
    MonoidElement,
    Monoid,
    GroupElement,
    Group
)

# 环论
from .ring import (
    RingElement,
    Ring,
    IntegerRingElement,
    IntegerRing,
    PolynomialRingElement,
    PolynomialRing,
    MatrixRingElement,
    MatrixRing
)

# 域论
from .field import (
    FieldElement,
    Field,
    RationalFieldElement,
    RationalField,
    RealFieldElement,
    RealField,
    FiniteFieldElement,
    FiniteField,
    ComplexFieldElement,
    ComplexField,
    ExtendedFiniteFieldElement,
    ExtendedFiniteField
)

# 模论和向量空间
from .module import (
    ModuleElement,
    Module,
    VectorSpaceElement,
    VectorSpace,
    FiniteDimensionalVectorSpaceElement,
    FiniteDimensionalVectorSpace,
    LinearTransformation,
    TensorProductElement,
    TensorProduct
)

# 代数结构之间的关系和转换
from .algebraic_relations import (
    GroupHomomorphism,
    RingHomomorphism,
    FieldHomomorphism,
    ModuleHomomorphism,
    VectorSpaceHomomorphism,
    Isomorphism,
    Automorphism,
    direct_sum_groups,
    polynomial_ring,
    finite_field,
    vector_space
)

__all__ = [
    # 抽象代数结构基类
    'AlgebraicProperties',
    'SemigroupElement',
    'Semigroup',
    'MonoidElement',
    'Monoid',
    'GroupElement',
    'Group',
    
    # 环论
    'RingElement',
    'Ring',
    'IntegerRingElement',
    'IntegerRing',
    'PolynomialRingElement',
    'PolynomialRing',
    'MatrixRingElement',
    'MatrixRing',
    
    # 域论
    'FieldElement',
    'Field',
    'RationalFieldElement',
    'RationalField',
    'RealFieldElement',
    'RealField',
    'FiniteFieldElement',
    'FiniteField',
    'ComplexFieldElement',
    'ComplexField',
    'ExtendedFiniteFieldElement',
    'ExtendedFiniteField',
    
    # 模论和向量空间
    'ModuleElement',
    'Module',
    'VectorSpaceElement',
    'VectorSpace',
    'FiniteDimensionalVectorSpaceElement',
    'FiniteDimensionalVectorSpace',
    'LinearTransformation',
    'TensorProductElement',
    'TensorProduct',
    
    # 代数结构之间的关系和转换
    'GroupHomomorphism',
    'RingHomomorphism',
    'FieldHomomorphism',
    'ModuleHomomorphism',
    'VectorSpaceHomomorphism',
    'Isomorphism',
    'Automorphism',
    'direct_sum_groups',
    'polynomial_ring',
    'finite_field',
    'vector_space'
]
