"""
选择定则模块

包含：
- 光学跃迁选择定则
- 振动选择定则
- 宇称选择定则
- 群论选择定则
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..symmetry_environments.discrete_symmetries.point_groups import PointGroup

# -----------------------------------------------------------------------------
# 1. 选择定则基类
# -----------------------------------------------------------------------------


class SelectionRule(ABC):
    """选择定则基类"""

    @abstractmethod
    def is_allowed(
        self, initial_state: Any, final_state: Any, operator: Any = None
    ) -> bool:
        """判断跃迁是否允许"""
        pass

    @abstractmethod
    def get_allowed_transitions(self, states: list[Any]) -> list[tuple[int, int]]:
        """获取所有允许跃迁"""
        pass


# -----------------------------------------------------------------------------
# 2. 电偶极跃迁选择定则
# -----------------------------------------------------------------------------


class ElectricDipoleSelectionRule(SelectionRule):
    """
    电偶极跃迁选择定则

    原子光谱：
    Δl = ±1
    Δm = 0, ±1
    Δs = 0
    Δj = 0, ±1 (j=0 → j'=0禁阻)
    """

    def __init__(self, include_spin_orbit: bool = True):
        self.include_spin_orbit = include_spin_orbit

    def is_allowed(
        self, initial_state: dict, final_state: dict, operator: Any = None
    ) -> bool:
        """判断电偶极跃迁是否允许"""

        # 轨道角动量选择定则
        l_i = initial_state.get("l", 0)
        l_f = final_state.get("l", 0)

        if abs(l_f - l_i) != 1:
            return False

        # 自旋选择定则
        s_i = initial_state.get("s", 0)
        s_f = final_state.get("s", 0)

        if s_i != s_f:
            return False

        if self.include_spin_orbit:
            # 总角动量选择定则
            j_i = initial_state.get("j", 0)
            j_f = final_state.get("j", 0)

            delta_j = abs(j_f - j_i)
            if delta_j > 1:
                return False
            if j_i == 0 and j_f == 0:
                return False

        return True

    def get_allowed_transitions(self, states: list[dict]) -> list[tuple[int, int]]:
        """获取所有允许跃迁"""
        transitions = []

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if i >= j:
                    continue
                if self.is_allowed(state_i, state_j):
                    transitions.append((i, j))

        return transitions


# -----------------------------------------------------------------------------
# 3. 宇称选择定则
# -----------------------------------------------------------------------------


class ParitySelectionRule(SelectionRule):
    """
    宇称选择定则

    电偶极：g ↔ u
    磁偶极：g ↔ g, u ↔ u
    电四极：g ↔ g, u ↔ u
    """

    def __init__(self, transition_type: str = "electric_dipole"):
        """
        Parameters:
            transition_type: 'electric_dipole', 'magnetic_dipole', 'electric_quadrupole'
        """
        self.transition_type = transition_type

    def is_allowed(
        self, initial_state: dict, final_state: dict, operator: Any = None
    ) -> bool:
        """判断宇称选择定则"""

        parity_i = initial_state.get("parity", 1)  # +1 for g, -1 for u
        parity_f = final_state.get("parity", 1)

        # 算符的宇称
        if self.transition_type == "electric_dipole":
            operator_parity = -1  # 奇宇称
        elif self.transition_type in ["magnetic_dipole", "electric_quadrupole"]:
            operator_parity = 1  # 偶宇称
        else:
            operator_parity = 1

        # 积分非零条件：总宇称为偶
        total_parity = parity_i * operator_parity * parity_f

        return total_parity == 1

    def get_allowed_transitions(self, states: list[dict]) -> list[tuple[int, int]]:
        """获取所有允许跃迁"""
        transitions = []

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if i >= j:
                    continue
                if self.is_allowed(state_i, state_j):
                    transitions.append((i, j))

        return transitions


# -----------------------------------------------------------------------------
# 4. 群论选择定则
# -----------------------------------------------------------------------------


class GroupTheorySelectionRule(SelectionRule):
    """
    群论选择定则

    跃迁积分：<ψ_f|O|ψ_i> 非零的条件：
    Γ_f ⊗ Γ_O ⊗ Γ_i 必须包含全对称表示
    """

    def __init__(self, point_group: PointGroup):
        self.point_group = point_group
        self.character_table, self.irrep_names, self.class_names = (
            point_group.character_table_full()
        )

    def is_allowed(
        self, initial_irrep: str, final_irrep: str, operator_irrep: str
    ) -> bool:
        """
        判断跃迁是否允许

        Parameters:
            initial_irrep: 初态不可约表示
            final_irrep: 末态不可约表示
            operator_irrep: 算符不可约表示
        """

        # 计算直积
        product = self._direct_product(initial_irrep, operator_irrep)
        product = self._direct_product_contain(product, final_irrep)

        # 检查是否包含全对称表示 A1
        if "A1" in self.irrep_names:
            return self._contains_irrep(product, "A1")
        elif "A1g" in self.irrep_names:
            return self._contains_irrep(product, "A1g")

        return False

    def _direct_product(self, irrep1: str, irrep2: str) -> dict[str, int]:
        """
        两个不可约表示的直积分解

        返回包含的不可约表示及其重数
        """
        idx1 = self.irrep_names.index(irrep1)
        idx2 = self.irrep_names.index(irrep2)

        # 特征标相乘
        product_chars = self.character_table[idx1] * self.character_table[idx2]

        # 投影分解
        decomposition = {}
        for i, irrep in enumerate(self.irrep_names):
            # 内积计算重数
            multiplicity = self._inner_product(product_chars, self.character_table[i])
            if multiplicity > 0:
                decomposition[irrep] = int(multiplicity)

        return decomposition

    def _direct_product_contain(
        self, product: dict[str, int], irrep: str
    ) -> dict[str, int]:
        """直积与第三个表示的乘积"""
        result = {}
        for irrep_p, mult_p in product.items():
            sub_product = self._direct_product(irrep_p, irrep)
            for irrep_s, mult_s in sub_product.items():
                if irrep_s in result:
                    result[irrep_s] += mult_p * mult_s
                else:
                    result[irrep_s] = mult_p * mult_s
        return result

    def _inner_product(self, chars1: np.ndarray, chars2: np.ndarray) -> float:
        """
        特征标的内积

        (1/|G|) Σ_g χ1(g)* χ2(g)
        """
        # 简化：假设类的大小已经考虑
        return np.dot(chars1, chars2) / self.character_table.shape[0]

    def _contains_irrep(self, decomposition: dict[str, int], irrep: str) -> bool:
        """检查分解中是否包含某不可约表示"""
        return irrep in decomposition and decomposition[irrep] > 0

    def get_allowed_transitions(
        self,
        states: list[tuple[str, str]],  # [(irrep, state_label)]
        operator_irrep: str,
    ) -> list[tuple[int, int]]:
        """
        获取所有允许跃迁

        Parameters:
            states: [(不可约表示名称, 状态标签)]
            operator_irrep: 算符的不可约表示
        """
        transitions = []

        for i, (irrep_i, _) in enumerate(states):
            for j, (irrep_f, _) in enumerate(states):
                if i >= j:
                    continue
                if self.is_allowed(irrep_i, irrep_f, operator_irrep):
                    transitions.append((i, j))

        return transitions

    def transition_intensity(
        self, initial_irrep: str, final_irrep: str, operator_components: list[str]
    ) -> float:
        """
        估计跃迁强度

        通过直积分解的重数估计
        """
        total_strength = 0.0

        for op_irrep in operator_components:
            if self.is_allowed(initial_irrep, final_irrep, op_irrep):
                # 归一化强度
                total_strength += 1.0

        return total_strength / len(operator_components)


# -----------------------------------------------------------------------------
# 5. 振动选择定则
# -----------------------------------------------------------------------------


class VibrationSelectionRule(SelectionRule):
    """
    振动选择定则

    红外活性：振动模式必须与偶极矩分量（x, y, z）同属一个不可约表示
    拉曼活性：振动模式必须与极化率分量（xy, yz, zx, x², y², z²）同属一个不可约表示
    """

    def __init__(self, point_group: PointGroup):
        self.point_group = point_group
        self.group_rule = GroupTheorySelectionRule(point_group)

        # 偶极矩算符的不可约表示
        self.dipole_irreps = self._get_dipole_irreps()

        # 极化率算符的不可约表示
        self.polarizability_irreps = self._get_polarizability_irreps()

    def _get_dipole_irreps(self) -> list[str]:
        """获取x, y, z对应的不可约表示"""
        # 根据点群查表
        # 例如对于Oh: T1u
        # 对于Td: T2
        # 对于D4h: A2u + Eu

        pg_name = self.point_group.name

        dipole_table = {
            "Oh": ["T1u"],
            "Td": ["T2"],
            "D4h": ["A2u", "Eu"],
            "C2v": ["A1", "B1", "B2"],
            # 更多...
        }

        return dipole_table.get(pg_name, [])

    def _get_polarizability_irreps(self) -> list[str]:
        """获取极化率张量对应的不可约表示"""
        # 对称二阶张量的表示

        pg_name = self.point_group.name

        raman_table = {
            "Oh": ["A1g", "Eg", "T2g"],
            "Td": ["A1", "E", "T2"],
            "D4h": ["A1g", "B1g", "B2g", "Eg"],
            # 更多...
        }

        return raman_table.get(pg_name, [])

    def is_infrared_active(self, vibration_irrep: str) -> bool:
        """判断振动模式是否红外活性"""
        for dipole_irrep in self.dipole_irreps:
            # 检查振动模式是否与偶极矩同表示
            if vibration_irrep == dipole_irrep:
                return True
        return False

    def is_raman_active(self, vibration_irrep: str) -> bool:
        """判断振动模式是否拉曼活性"""
        for pol_irrep in self.polarizability_irreps:
            if vibration_irrep == pol_irrep:
                return True
        return False

    def is_allowed(
        self, initial_state: dict, final_state: dict, operator: Any = None
    ) -> bool:
        """判断振动跃迁是否允许"""
        # 获取振动模式的不可约表示
        vib_irrep = final_state.get("vibration_irrep", "")

        transition_type = initial_state.get("transition_type", "infrared")

        if transition_type == "infrared":
            return self.is_infrared_active(vib_irrep)
        elif transition_type == "raman":
            return self.is_raman_active(vib_irrep)

        return False

    def get_allowed_transitions(self, states: list[dict]) -> list[tuple[int, int]]:
        """获取所有允许跃迁"""
        transitions = []

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if i >= j:
                    continue

                # 检查红外和拉曼
                if self.is_infrared_active(state_j.get("vibration_irrep", "")):
                    transitions.append((i, j, "infrared"))
                if self.is_raman_active(state_j.get("vibration_irrep", "")):
                    transitions.append((i, j, "raman"))

        return transitions

    def analyze_vibrational_modes(self, mode_irreps: list[str]) -> dict[str, list[int]]:
        """
        分析振动模式的光谱活性

        Parameters:
            mode_irreps: 振动模式的不可约表示列表

        Returns:
            {'infrared': [活性模式索引], 'raman': [活性模式索引], 'silent': [禁阻模式索引]}
        """
        infrared_active = []
        raman_active = []
        silent = []

        for i, irrep in enumerate(mode_irreps):
            is_ir = self.is_infrared_active(irrep)
            is_ram = self.is_raman_active(irrep)

            if is_ir:
                infrared_active.append(i)
            if is_ram:
                raman_active.append(i)
            if not is_ir and not is_ram:
                silent.append(i)

        return {"infrared": infrared_active, "raman": raman_active, "silent": silent}


# -----------------------------------------------------------------------------
# 6. 综合选择定则
# -----------------------------------------------------------------------------


class ComprehensiveSelectionRules:
    """综合选择定则"""

    def __init__(self, point_group: PointGroup | None = None):
        self.parity_rule = ParitySelectionRule()
        self.electric_dipole = ElectricDipoleSelectionRule()

        if point_group:
            self.group_rule = GroupTheorySelectionRule(point_group)
            self.vibration_rule = VibrationSelectionRule(point_group)
        else:
            self.group_rule = None
            self.vibration_rule = None

    def analyze_transition(
        self,
        initial_state: dict,
        final_state: dict,
        transition_type: str = "electric_dipole",
    ) -> dict[str, Any]:
        """
        综合分析跃迁

        Returns:
            {
                'allowed': bool,
                'forbidden_reasons': List[str],
                'allowed_factors': List[str],
                'intensity_estimate': float
            }
        """
        result = {
            "allowed": True,
            "forbidden_reasons": [],
            "allowed_factors": [],
            "intensity_estimate": 1.0,
        }

        # 宇称选择定则
        if not self.parity_rule.is_allowed(initial_state, final_state):
            result["allowed"] = False
            result["forbidden_reasons"].append("Parity forbidden")
            result["intensity_estimate"] *= 0.01
        else:
            result["allowed_factors"].append("Parity allowed")

        # 电偶极选择定则
        if transition_type == "electric_dipole":
            if not self.electric_dipole.is_allowed(initial_state, final_state):
                result["allowed"] = False
                result["forbidden_reasons"].append("Electric dipole forbidden")
                result["intensity_estimate"] *= 0.01
            else:
                result["allowed_factors"].append("Electric dipole allowed")

        # 群论选择定则
        if self.group_rule:
            irrep_i = initial_state.get("irrep", "A1")
            irrep_f = final_state.get("irrep", "A1")
            operator_irrep = initial_state.get("operator_irrep", "T1u")

            if not self.group_rule.is_allowed(irrep_i, irrep_f, operator_irrep):
                result["allowed"] = False
                result["forbidden_reasons"].append("Group theory forbidden")
            else:
                result["allowed_factors"].append("Group theory allowed")

        return result
