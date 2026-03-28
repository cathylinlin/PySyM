"""李代数工厂

该模块实现了李代数工厂类，用于创建各种李代数实例。
"""

from .specific_lie_algebra import (
    GeneralLinearLieAlgebra,
    SpecialLinearLieAlgebra,
    OrthogonalLieAlgebra,
    SymplecticLieAlgebra,
    UnitaryLieAlgebra,
    SpecialUnitaryLieAlgebra
)
from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement
from typing import Optional, Type


class LieAlgebraFactory:
    """李代数工厂类"""
    
    @staticmethod
    def create_general_linear(n: int) -> GeneralLinearLieAlgebra:
        """创建一般线性李代数 gl(n)"""
        return GeneralLinearLieAlgebra(n)
    
    @staticmethod
    def create_special_linear(n: int) -> SpecialLinearLieAlgebra:
        """创建特殊线性李代数 sl(n)"""
        return SpecialLinearLieAlgebra(n)
    
    @staticmethod
    def create_orthogonal(n: int) -> OrthogonalLieAlgebra:
        """创建正交李代数 so(n)"""
        return OrthogonalLieAlgebra(n)
    
    @staticmethod
    def create_symplectic(n: int) -> SymplecticLieAlgebra:
        """创建辛李代数 sp(2n, R)（2n×2n 矩阵；参数 n 为辛群的“半数”）。"""
        return SymplecticLieAlgebra(n)
    
    @staticmethod
    def create_unitary(n: int) -> UnitaryLieAlgebra:
        """创建酉李代数 u(n)"""
        return UnitaryLieAlgebra(n)
    
    @staticmethod
    def create_special_unitary(n: int) -> SpecialUnitaryLieAlgebra:
        """创建特殊酉李代数 su(n)"""
        return SpecialUnitaryLieAlgebra(n)
    
    @staticmethod
    def create_lie_algebra(name: str, n: int) -> LieAlgebra:
        """根据名称创建李代数
        
        参数:
            name: 李代数名称，可选值包括: 'gl', 'sl', 'so', 'sp', 'u', 'su'
            n: 阶数；对 'sp' 表示 sp(2n) 中的 n（矩阵大小为 2n）
            
        返回:
            李代数实例
        """
        name = name.lower()
        if name == 'gl':
            return GeneralLinearLieAlgebra(n)
        elif name == 'sl':
            return SpecialLinearLieAlgebra(n)
        elif name == 'so':
            return OrthogonalLieAlgebra(n)
        elif name == 'sp':
            return SymplecticLieAlgebra(n)
        elif name == 'u':
            return UnitaryLieAlgebra(n)
        elif name == 'su':
            return SpecialUnitaryLieAlgebra(n)
        else:
            raise ValueError(f"未知的李代数名称: {name}")
    
    @staticmethod
    def get_lie_algebra_class(name: str) -> Type[LieAlgebra]:
        """根据名称获取李代数类
        
        参数:
            name: 李代数名称，可选值包括: 'gl', 'sl', 'so', 'sp', 'u', 'su'
            
        返回:
            李代数类
        """
        name = name.lower()
        if name == 'gl':
            return GeneralLinearLieAlgebra
        elif name == 'sl':
            return SpecialLinearLieAlgebra
        elif name == 'so':
            return OrthogonalLieAlgebra
        elif name == 'sp':
            return SymplecticLieAlgebra
        elif name == 'u':
            return UnitaryLieAlgebra
        elif name == 'su':
            return SpecialUnitaryLieAlgebra
        else:
            raise ValueError(f"未知的李代数名称: {name}")
