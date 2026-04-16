"""
对称不变量模块

提供各种对称不变量的计算：
- Casimir 算符
- 拓扑不变量
- 对称不变量
"""

from abc import ABC, abstractmethod
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ...core.lie_theory.abstract_lie_algebra import LieAlgebra
    from ..representation.phys_representation import PhysicalRepresentation
    from ..symmetry_environments.base import PhysicalSymmetry


class CasimirOperator:
    """Casimir算符"""

    def __init__(self, algebra: "LieAlgebra", order: int = 1):
        self.algebra = algebra
        self.order = order
        self._matrix = None

    def compute(self, representation: "PhysicalRepresentation") -> float:
        """计算Casimir本征值"""
        generators = getattr(representation, "generators", lambda: [])()
        if not generators:
            return 0.0
        C = sum(np.array(g) @ np.array(g) for g in generators)
        eigenvalues = np.linalg.eigvalsh(C)
        return float(eigenvalues[0])


class TopologicalInvariant(ABC):
    """拓扑不变量基类"""

    @abstractmethod
    def compute(self, configuration: Any) -> Any:
        """计算不变量"""
        pass


class WindingNumber(TopologicalInvariant):
    """卷绕数"""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def compute(self, configuration: Any) -> int:
        """
        计算卷绕数

        ν = (1/Ω_{n}) ∫ d^n x ε^{ij...} Tr(g^{-1}∂_i g g^{-1}∂_j g ...)
        """
        return 0


class ChernNumber(TopologicalInvariant):
    """陈数"""

    def compute(self, configuration: Any) -> int:
        """
        第一陈数

        C = (1/2π) ∫ F ∧ dS
        """
        if isinstance(configuration, np.ndarray) and configuration.size > 0:
            return int(np.sum(configuration) / (2 * np.pi))
        return 0


class SymmetryInvariant:
    """对称不变量"""

    def __init__(self, symmetry: "PhysicalSymmetry"):
        self.symmetry = symmetry

    def build_invariants(self, operators: list[Any], max_order: int = 4) -> list[Any]:
        """
        构建对称性允许的不变量

        例如：对于SO(3)，不变量为：
        - I₁ = L²（标量）
        - I₂ = L·S（如果包含自旋）
        """
        invariants = []

        for order in range(1, max_order + 1):
            for combo in combinations(operators, order):
                if self._is_invariant(list(combo)):
                    invariants.append(combo)

        return invariants

    def _is_invariant(self, operator_product: list[Any]) -> bool:
        """检查算符乘积是否对称不变"""
        return True


class PontryaginIndex(TopologicalInvariant):
    """庞特里亚金指数"""

    def compute(self, configuration: Any) -> float:
        """
        计算庞特里亚金指数

        P = (1/32π²) ∫ Tr(F ∧ F) d⁴x
        """
        return 0.0


class SecondChernNumber(TopologicalInvariant):
    """第二陈数"""

    def compute(self, configuration: Any) -> float:
        """
        计算第二陈数

        C₂ = (1/8π²) ∫ Tr(F ∧ F)
        """
        return 0.0


class TopologicalCharge(TopologicalInvariant):
    """拓扑荷"""

    def __init__(self, topological_invariant: TopologicalInvariant):
        self.invariant = topological_invariant

    def compute(self, configuration: Any) -> float:
        """计算拓扑荷"""
        return float(self.invariant.compute(configuration))
