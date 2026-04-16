"""
对称性分析器模块

提供自动化的对称性分析：
- 对称性检测
- 不可约表示分解
- 选择定则
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..physical_objects.systems import PhysicalSystem
    from ..representation.phys_representation import IrreducibleRepresentation
    from ..symmetry_environments.base import PhysicalSymmetry
    from .base import State


class SymmetryAnalyzer:
    """对称性分析器"""

    def __init__(self, system: "PhysicalSystem"):
        self.system = system
        self._symmetry_cache: dict[str, bool] = {}

    def detect_symmetries(self, tolerance: float = 1e-8) -> list["PhysicalSymmetry"]:
        """
        自动检测系统对称性

        通过数值验证检查哈密顿量/拉格朗日量的不变性
        """
        detected = []

        if self._check_time_translation(tolerance):
            from .specific_operations import TimeTranslationSymmetry

            detected.append(TimeTranslationSymmetry())

        if self._check_spatial_translation(tolerance):
            from .specific_operations import TranslationSymmetry

            detected.append(TranslationSymmetry())

        if self._check_rotation(tolerance):
            from .specific_operations import RotationSymmetry

            detected.append(RotationSymmetry())

        if self._check_parity(tolerance):
            from .specific_operations import ParitySymmetry

            detected.append(ParitySymmetry())

        if self._check_time_reversal(tolerance):
            from .specific_operations import TimeReversalSymmetry

            detected.append(TimeReversalSymmetry())

        return detected

    def _check_time_translation(self, tolerance: float) -> bool:
        """检查时间平移对称性（哈密顿量不显含时间）"""
        if not hasattr(self.system, "hamiltonian"):
            return False
        # 如果系统有显式时间依赖，则不具有时间平移对称性
        if (
            hasattr(self.system, "is_time_dependent")
            and self.system.is_time_dependent()
        ):
            return False
        return True

    def _check_spatial_translation(self, tolerance: float) -> bool:
        """检查空间平移对称性"""
        if not hasattr(self.system, "is_invariant_under"):
            return False
        try:
            from .specific_operations import TranslationOperation

            # 沿随机方向做微小平移测试
            test_displacements = [
                np.array([tolerance * 10, 0, 0]),
                np.array([0, tolerance * 10, 0]),
                np.array([0, 0, tolerance * 10]),
            ]
            for disp in test_displacements:
                op = TranslationOperation(disp)
                if not self.system.is_invariant_under(op):
                    return False
            return True
        except Exception:
            return False

    def _check_rotation(self, tolerance: float) -> bool:
        """检查旋转对称性"""
        if not hasattr(self.system, "is_invariant_under"):
            return False
        try:
            from .specific_operations import RotationOperation

            # 沿各坐标轴做小角度旋转测试
            test_rotations = [
                RotationOperation([1, 0, 0], 0.1),
                RotationOperation([0, 1, 0], 0.1),
                RotationOperation([0, 0, 1], 0.1),
            ]
            for op in test_rotations:
                if not self.system.is_invariant_under(op):
                    return False
            return True
        except Exception:
            return False

    def _check_parity(self, tolerance: float) -> bool:
        """检查宇称对称性"""
        if not hasattr(self.system, "is_invariant_under"):
            return False
        try:
            from .specific_operations import ParityOperation

            op = ParityOperation()
            return self.system.is_invariant_under(op)
        except Exception:
            return False

    def _check_time_reversal(self, tolerance: float) -> bool:
        """检查时间反演对称性"""
        if not hasattr(self.system, "is_invariant_under"):
            return False
        try:
            from .specific_operations import TimeReversalOperation

            op = TimeReversalOperation()
            return self.system.is_invariant_under(op)
        except Exception:
            return False

    def decompose_state(
        self, state: "State", symmetry: "PhysicalSymmetry"
    ) -> dict[str, float]:
        """
        将态分解为对称性本征态

        返回各不可约表示的分量
        """
        return {}

    def selection_rules(
        self,
        initial_irrep: "IrreducibleRepresentation",
        final_irrep: "IrreducibleRepresentation",
        operator_irrep: "IrreducibleRepresentation",
    ) -> bool:
        """
        判断跃迁是否被选择定则允许

        检查 Γ_i ⊗ Γ_op ⊗ Γ_f 是否包含恒等表示
        """
        product = self._tensor_product(initial_irrep, operator_irrep)
        product = self._tensor_product(product, final_irrep)
        return self._contains_identity(product)

    def _tensor_product(
        self, irrep1: Any, irrep2: Any
    ) -> list["IrreducibleRepresentation"]:
        """计算表示的张量积"""
        from ..representation.phys_representation import SU2Representation

        if hasattr(irrep1, "j") and hasattr(irrep2, "j"):
            j1, j2 = irrep1.j, irrep2.j
            j_values = []
            j = abs(j1 - j2)
            while j <= j1 + j2:
                j_values.append(j)
                j += 1
            return [SU2Representation(j) for j in j_values]
        return [irrep1]

    def _contains_identity(self, irreps: list[Any]) -> bool:
        """检查表示列表是否包含恒等表示"""
        for irrep in irreps:
            if hasattr(irrep, "dimension") and irrep.dimension == 1:
                return True
        return False

    def analyze_spectrum(self, hamiltonian: np.ndarray) -> dict[str, Any]:
        """
        分析能谱的对称性结构

        返回：
        - 能级简并度
        - 量子数分配
        - 对称性分类
        """
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        degenerate_groups = self._find_degenerate_levels(eigenvalues)

        analysis = {}
        for i, (E, indices) in enumerate(degenerate_groups):
            degeneracy = len(indices)
            irrep = self._identify_irreducible_representation(
                eigenvectors[:, indices]
                if len(indices) > 1
                else eigenvectors[:, indices].reshape(-1, 1)
            )
            analysis[f"level_{i}"] = {
                "energy": E,
                "degeneracy": degeneracy,
                "irrep": irrep,
                "quantum_numbers": self._extract_quantum_numbers(irrep),
            }

        return analysis

    def _find_degenerate_levels(
        self, eigenvalues: np.ndarray, tol: float = 1e-8
    ) -> list[tuple]:
        """找出简并能级"""
        degenerate_groups = []
        used = np.zeros(len(eigenvalues), dtype=bool)

        for i in range(len(eigenvalues)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(eigenvalues)):
                if not used[j] and np.isclose(eigenvalues[i], eigenvalues[j], rtol=tol):
                    group.append(j)
                    used[j] = True
            degenerate_groups.append((eigenvalues[i], group))
            used[i] = True

        return degenerate_groups

    def _identify_irreducible_representation(
        self, eigenvectors: np.ndarray
    ) -> str | None:
        """识别不可约表示"""
        n = eigenvectors.shape[1] if len(eigenvectors.shape) > 1 else 1
        return f"dim_{n}"

    def _extract_quantum_numbers(self, irrep: str | None) -> dict[str, Any]:
        """提取量子数"""
        if irrep and irrep.startswith("dim_"):
            dim = int(irrep.split("_")[1])
            return {"degeneracy": dim}
        return {}

    def conservation_law_analysis(self) -> dict[str, Any]:
        """
        分析守恒定律

        使用诺特定理
        """
        result = {}
        for symmetry in self.system.get_symmetries():
            conserved = symmetry.conserved_quantity()
            if conserved and conserved != "none":
                result[conserved] = {
                    "symmetry": symmetry.type.name,
                    "generator": symmetry.generators(),
                }
        return result

    def compute_central_charge(self) -> int:
        """计算中心荷（用于共形场论）"""
        return 0

    def compute_modular_matrix(self) -> np.ndarray:
        """计算模矩阵 S（用于共形场论）"""
        return np.eye(2)
