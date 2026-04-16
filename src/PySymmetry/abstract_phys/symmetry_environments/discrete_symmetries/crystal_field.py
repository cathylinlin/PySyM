"""
晶体场理论模块

包含：
- 晶体场势能
- d轨道分裂
- 配位场理论
- Tanabe-Sugano图
- 晶体场参数
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.special import sph_harm

from .point_groups import PointGroup

# -----------------------------------------------------------------------------
# 1. 晶体场参数
# -----------------------------------------------------------------------------


@dataclass
class CrystalFieldParameters:
    """晶体场参数"""

    # Racah参数（电子间排斥）
    B: float = 1000.0  # cm^-1
    C: float = 4000.0  # cm^-1

    # 晶体场参数（Wybourne符号）
    B20: float = 0.0
    B40: float = 0.0
    B44: float = 0.0
    B60: float = 0.0
    B64: float = 0.0

    # 10Dq (Δ)
    Dq: float = 0.0  # cm^-1

    # 低对称性参数
    B22: float = 0.0
    B42: float = 0.0
    B43: float = 0.0

    # 自旋-轨道耦合
    zeta: float = 0.0  # cm^-1

    def to_dict(self) -> dict[str, float]:
        return {
            "B": self.B,
            "C": self.C,
            "B20": self.B20,
            "B40": self.B40,
            "B44": self.B44,
            "B60": self.B60,
            "B64": self.B64,
            "Dq": self.Dq,
            "B22": self.B22,
            "B42": self.B42,
            "B43": self.B43,
            "zeta": self.zeta,
        }


class CoordinationGeometry(Enum):
    """配位几何"""

    OCTAHEDRAL = "Oh"  # 八面体
    TETRAHEDRAL = "Td"  # 四面体
    SQUARE_PLANAR = "D4h"  # 平面四方
    SQUARE_PYRAMIDAL = "C4v"  # 四方锥
    TRIGONAL_BIPYRAMIDAL = "D3h"  # 三方双锥
    CUBIC = "Oh"  # 立方体
    DODECAHEDRAL = "D2d"  # 十二面体


# -----------------------------------------------------------------------------
# 2. d轨道
# -----------------------------------------------------------------------------


class DOrbital:
    """d轨道集合"""

    # 五个d轨道的球谐函数表示
    ORBITALS = ["d_z2", "d_x2-y2", "d_xy", "d_xz", "d_yz"]

    # 惯例索引
    m_values = {"d_z2": 0, "d_x2-y2": 2, "d_xy": -2, "d_xz": 1, "d_yz": -1}

    @staticmethod
    def spherical_harmonic(m: int, theta: float, phi: float) -> complex:
        """实球谐函数 Y_{2,m}(θ, φ)"""
        if m == 0:
            # d_z2
            return np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(theta) ** 2 - 1)
        elif m == 2:
            # d_x2-y2
            return np.sqrt(15 / (16 * np.pi)) * np.sin(theta) ** 2 * np.cos(2 * phi)
        elif m == -2:
            # d_xy
            return np.sqrt(15 / (16 * np.pi)) * np.sin(theta) ** 2 * np.sin(2 * phi)
        elif m == 1:
            # d_xz
            return (
                np.sqrt(15 / (4 * np.pi)) * np.sin(theta) * np.cos(theta) * np.cos(phi)
            )
        elif m == -1:
            # d_yz
            return (
                np.sqrt(15 / (4 * np.pi)) * np.sin(theta) * np.cos(theta) * np.sin(phi)
            )

    @staticmethod
    def basis_matrices() -> dict[str, np.ndarray]:
        """d轨道的实空间表示矩阵（用于计算矩阵元）"""
        return {
            "d_z2": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "d_x2-y2": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / np.sqrt(2),
            "d_xy": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / np.sqrt(2),
            "d_xz": np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / np.sqrt(2),
            "d_yz": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / np.sqrt(2),
        }


# -----------------------------------------------------------------------------
# 3. 晶体场势能
# -----------------------------------------------------------------------------


class CrystalFieldPotential:
    """晶体场势能"""

    def __init__(self, point_group: PointGroup, parameters: CrystalFieldParameters):
        self.point_group = point_group
        self.params = parameters

    def potential_operator(self, position: np.ndarray) -> float:
        """
        晶体场势能

        V_CF = Σ_k Σ_q B_kq C_kq

        其中 C_kq 是重新归一化的球谐函数
        """
        r = np.linalg.norm(position)
        theta = np.arccos(position[2] / r) if r > 0 else 0
        phi = np.arctan2(position[1], position[0])

        V = 0.0

        # 二阶项
        V += self.params.B20 * self._C(2, 0, theta, phi)

        # 四阶项
        V += self.params.B40 * self._C(4, 0, theta, phi)
        V += self.params.B44 * self._C(4, 4, theta, phi)

        # 六阶项
        V += self.params.B60 * self._C(6, 0, theta, phi)
        V += self.params.B64 * self._C(6, 4, theta, phi)

        return V

    def _C(self, k: int, q: int, theta: float, phi: float) -> float:
        """重新归一化的球谐函数"""
        if q == 0:
            return np.sqrt(4 * np.pi / (2 * k + 1)) * sph_harm(0, k, phi, theta).real
        else:
            # 实组合
            C_real = (
                sph_harm(q, k, phi, theta) + sph_harm(-q, k, phi, theta)
            ) / np.sqrt(2)
            return np.sqrt(4 * np.pi / (2 * k + 1)) * C_real.real

    def hamiltonian_matrix(self, orbital_set: str = "d") -> np.ndarray:
        """
        晶体场哈密顿矩阵

        在给定轨道基下的矩阵表示
        """
        if orbital_set == "d":
            return self._d_orbital_hamiltonian()
        elif orbital_set == "f":
            return self._f_orbital_hamiltonian()

    def _d_orbital_hamiltonian(self) -> np.ndarray:
        """5×5 d轨道晶体场矩阵"""
        H = np.zeros((5, 5), dtype=complex)

        # 根据点群对称性构造矩阵
        if self.point_group.name == "Oh":
            # 八面体场：d轨道分裂为 t2g (低) 和 eg (高)
            H = self._octahedral_field()
        elif self.point_group.name == "Td":
            # 四面体场：d轨道分裂为 e (低) 和 t2 (高)
            H = self._tetrahedral_field()
        elif self.point_group.name == "D4h":
            # 四方场
            H = self._tetragonal_field()

        return H

    def _octahedral_field(self) -> np.ndarray:
        """
        八面体晶体场

        d轨道分裂：t2g (d_xy, d_xz, d_yz) 和 eg (d_z2, d_x2-y2)
        能量差 = 10Dq
        """
        Dq = self.params.Dq

        # eg轨道能量 = +6Dq
        # t2g轨道能量 = -4Dq
        # 质心保持为零

        H = np.diag(
            [
                6 * Dq,  # d_z2 (eg)
                6 * Dq,  # d_x2-y2 (eg)
                -4 * Dq,  # d_xy (t2g)
                -4 * Dq,  # d_xz (t2g)
                -4 * Dq,  # d_yz (t2g)
            ]
        )

        return H

    def _tetrahedral_field(self) -> np.ndarray:
        """
        四面体晶体场

        分裂与八面体相反，且 Δ_tet = (4/9)Δ_oct
        """
        Dq = self.params.Dq * (4 / 9)  # 四面体场较弱

        # e轨道能量 = -6Dq
        # t2轨道能量 = +4Dq

        H = np.diag(
            [
                -6 * Dq,  # d_z2 (e)
                -6 * Dq,  # d_x2-y2 (e)
                4 * Dq,  # d_xy (t2)
                4 * Dq,  # d_xz (t2)
                4 * Dq,  # d_yz (t2)
            ]
        )

        return H

    def _tetragonal_field(self) -> np.ndarray:
        """
        四方晶体场（D4h）

        进一步分裂 t2g 和 eg
        """
        Dq = self.params.Dq
        Ds = self.params.B20  # 四方畸变参数
        Dt = self.params.B40  # 四方畸变参数

        # 更复杂的分裂模式
        H = np.zeros((5, 5))

        # d_z2: 6Dq - 2Ds - 6Dt
        H[0, 0] = 6 * Dq - 2 * Ds - 6 * Dt

        # d_x2-y2: 6Dq + 2Ds - Dt
        H[1, 1] = 6 * Dq + 2 * Ds - Dt

        # d_xy: -4Dq + 2Ds - Dt
        H[2, 2] = -4 * Dq + 2 * Ds - Dt

        # d_xz, d_yz: -4Dq - Ds + 4Dt
        H[3, 3] = -4 * Dq - Ds + 4 * Dt
        H[4, 4] = -4 * Dq - Ds + 4 * Dt

        return H

    def _f_orbital_hamiltonian(self) -> np.ndarray:
        """7×7 f轨道晶体场矩阵"""
        # f轨道需要更高阶的晶体场参数
        H = np.zeros((7, 7))
        # 实现略
        return H


# -----------------------------------------------------------------------------
# 4. 电子间排斥
# -----------------------------------------------------------------------------


class ElectronRepulsion:
    """电子间排斥（Racah参数化）"""

    def __init__(self, n_electrons: int, B: float, C: float):
        self.n = n_electrons
        self.B = B
        self.C = C

    def slater_integrals(self) -> tuple[float, float, float]:
        """
        Slater积分 F0, F2, F4

        与Racah参数关系：
        B = F2/49 - 5F4/441
        C = 35F4/63
        """
        # 从Racah参数反推Slater积分
        F4 = 63 * self.C / 35
        F2 = 49 * (self.B + 5 * F4 / 441)
        F0 = 0  # 常数项，通常吸收到能量零点

        return F0, F2, F4

    def coulomb_matrix_element(self, m1: int, m2: int, m3: int, m4: int) -> float:
        """
        Coulomb积分 <m1 m2|V|m3 m4>

        使用Slater-Condon规则
        """
        # 简化实现
        pass


# -----------------------------------------------------------------------------
# 5. 能级计算
# -----------------------------------------------------------------------------


class CrystalFieldLevelCalculator:
    """晶体场能级计算器"""

    def __init__(
        self, point_group: PointGroup, d_electrons: int, spin_state: str = "high_spin"
    ):
        self.point_group = point_group
        self.n_d = d_electrons
        self.spin_state = spin_state
        self.S = self._determine_spin()

    def _determine_spin(self) -> int:
        """确定总自旋"""
        if self.spin_state == "high_spin":
            # 高自旋：按照Hund规则填充
            if self.n_d <= 5:
                return self.n_d // 2
            else:
                return (10 - self.n_d) // 2
        else:
            # 低自旋：配对优先
            if self.n_d <= 3:
                return 0
            elif self.n_d <= 6:
                return (self.n_d - 3) // 2
            else:
                return (10 - self.n_d) // 2

    def calculate_levels(self, params: CrystalFieldParameters) -> dict[str, Any]:
        """
        计算晶体场能级

        包括：
        - 基态和激发态能量
        - 简并度
        - 波函数
        """
        # 构造哈密顿量
        H_cf = CrystalFieldPotential(self.point_group, params)
        H = H_cf.hamiltonian_matrix("d")

        # 加入电子间排斥
        # 这需要构造完整的d^n组态空间
        # 简化：仅对单电子情况

        if self.n_d == 1:
            # d1：简单情况
            eigenvalues, eigenvectors = np.linalg.eigh(H)

            return {
                "energies": eigenvalues.tolist(),
                "wavefunctions": eigenvectors,
                "terms": self._assign_terms(eigenvalues, eigenvectors),
            }
        else:
            # 多电子：需要组态相互作用
            return self._multi_electron_levels(params)

    def _multi_electron_levels(self, params: CrystalFieldParameters) -> dict:
        """多电子能级计算"""
        # 构造完整的d^n组态
        # 使用配位场理论方法
        # 这里简化处理

        pass

    def _assign_terms(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray
    ) -> list[str]:
        """指派光谱项"""
        terms = []

        # 根据点群和简并度指派
        if self.point_group.name == "Oh":
            for i, E in enumerate(eigenvalues):
                # 检查简并度
                degeneracy = sum(abs(E - e) < 1e-6 for e in eigenvalues)

                if degeneracy == 1:
                    terms.append("A1g" if i < 2 else "A2g")
                elif degeneracy == 2:
                    terms.append("Eg")
                elif degeneracy == 3:
                    terms.append("T2g")

        return terms


# -----------------------------------------------------------------------------
# 6. Tanabe-Sugano图
# -----------------------------------------------------------------------------


class TanabeSuganoDiagram:
    """
    Tanabe-Sugano图

    能量 vs Δ/B 图，用于预测d-d跃迁
    """

    def __init__(self, d_electrons: int):
        self.n_d = d_electrons
        self._data = None

    def generate(
        self,
        delta_range: tuple[float, float] = (0, 50),
        n_points: int = 100,
        B: float = 1000.0,
        C_B_ratio: float = 4.0,
    ) -> dict[str, np.ndarray]:
        """
        生成Tanabe-Sugano图数据

        Parameters:
            delta_range: Δ/B的范围
            n_points: 点数
            B: Racah参数B
            C_B_ratio: C/B比值
        """
        delta_B = np.linspace(delta_range[0], delta_range[1], n_points)
        C = B * C_B_ratio

        # 计算每个Δ/B值下的能级
        energies = {}

        # 根据d电子数选择合适的方法
        if self.n_d in [1, 9]:
            energies = self._d1_d9(delta_B)
        elif self.n_d in [2, 8]:
            energies = self._d2_d8(delta_B, B, C)
        elif self.n_d in [3, 7]:
            energies = self._d3_d7(delta_B, B, C)
        elif self.n_d in [4, 6]:
            energies = self._d4_d6(delta_B, B, C)
        elif self.n_d == 5:
            energies = self._d5(delta_B, B, C)

        self._data = {"delta_B": delta_B, "energies": energies, "B": B, "C": C}

        return self._data

    def _d1_d9(self, delta_B: np.ndarray) -> dict[str, np.ndarray]:
        """d1和d9组态（单空穴）"""
        # d1: 2D → 2T2g (基态)
        # 激发态: 2Eg
        # E(T2g) = -4Dq, E(Eg) = +6Dq

        energies = {
            "2T2g": -4 * delta_B,  # 基态
            "2Eg": 6 * delta_B,  # 激发态
        }
        return energies

    def _d2_d8(self, delta_B: np.ndarray, B: float, C: float) -> dict[str, np.ndarray]:
        """d2和d8组态"""
        # 简化：仅考虑主要能级
        energies = {
            "3T1g(F)": np.zeros_like(delta_B),  # 基态设为零
            "3T2g": 10 * delta_B,
            "3T1g(P)": 15 * delta_B + 15 * B,
            # 更多能级...
        }
        return energies

    def _d3_d7(self, delta_B: np.ndarray, B: float, C: float) -> dict[str, np.ndarray]:
        """d3和d7组态"""
        pass

    def _d4_d6(self, delta_B: np.ndarray, B: float, C: float) -> dict[str, np.ndarray]:
        """d4和d6组态（可能有高低自旋交叉）"""
        pass

    def _d5(self, delta_B: np.ndarray, B: float, C: float) -> dict[str, np.ndarray]:
        """d5组态"""
        pass

    def find_spin_crossover(self) -> float | None:
        """寻找自旋交叉点（如果存在）"""
        if self.n_d not in [4, 5, 6, 7]:
            return None

        # 需要具体计算
        pass

    def plot_data(self) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
        """返回绘图数据"""
        if self._data is None:
            raise ValueError("请先调用 generate() 方法")

        x = self._data["delta_B"]
        curves = [(name, E) for name, E in self._data["energies"].items()]

        return x, curves


# -----------------------------------------------------------------------------
# 7. 配位场理论
# -----------------------------------------------------------------------------


class LigandFieldTheory:
    """
    配位场理论

    考虑配体轨道与金属轨道的相互作用
    """

    def __init__(
        self,
        metal_orbitals: list[str],
        ligand_positions: list[np.ndarray],
        point_group: PointGroup,
    ):
        self.metal_orbitals = metal_orbitals
        self.ligand_positions = ligand_positions
        self.point_group = point_group

    def construct_symmetry_adapted_orbitals(self) -> dict[str, np.ndarray]:
        """
        构造对称性匹配的配体轨道

        将配体轨道投影到点群的不可约表示
        """
        # 投影算符方法
        irreps = self.point_group._irrep_names
        symmetry_orbitals = {}

        for irrep in irreps:
            # 构造属于该不可约表示的对称性匹配线性组合
            orbital = self._project_to_irrep(irrep)
            if orbital is not None:
                symmetry_orbitals[irrep] = orbital

        return symmetry_orbitals

    def _project_to_irrep(self, irrep: str) -> np.ndarray | None:
        """投影到不可约表示"""
        char_table, irrep_names, _ = self.point_group.character_table_full()
        _ = irrep_names.index(irrep)
        return None

    def overlap_integrals(self, ligand_orbital: str) -> np.ndarray:
        """
        计算重叠积分

        <metal_orbital | ligand_orbital>
        """
        # 使用Slater型轨道或高斯型轨道
        pass

    def bonding_analysis(self) -> dict[str, Any]:
        """
        成键分析

        返回：
        - σ, π, δ键的强度
        - 反馈键效应
        - 分子轨道能级图
        """
        pass


# -----------------------------------------------------------------------------
# 8. 光谱预测
# -----------------------------------------------------------------------------


class CrystalFieldSpectrum:
    """晶体场光谱预测"""

    def __init__(self, calculator: CrystalFieldLevelCalculator):
        self.calculator = calculator

    def dd_transitions(self, params: CrystalFieldParameters) -> list[dict]:
        """
        d-d跃迁预测

        返回允许的跃迁及其能量
        """
        levels = self.calculator.calculate_levels(params)

        transitions = []
        energies = levels["energies"]
        terms = levels.get("terms", [""] * len(energies))

        # 假设基态是最低能级
        ground_energy = energies[0]
        ground_term = terms[0] if terms else ""

        for i, (E, term) in enumerate(zip(energies, terms)):
            if i == 0:
                continue

            transition_energy = E - ground_energy

            # 检查选择定则
            # 自旋选择：ΔS = 0
            # Laporte选择：g ↔ u（d-d跃迁禁阻，但可振动允许）

            transition = {
                "from": ground_term,
                "to": term,
                "energy": transition_energy,
                "wavelength": 1e7 / transition_energy
                if transition_energy > 0
                else np.inf,  # cm^-1 to nm
                "allowed": self._check_spin_selection(ground_term, term),
            }
            transitions.append(transition)

        return transitions

    def _check_spin_selection(self, term1: str, term2: str) -> bool:
        """检查自旋选择定则"""
        # 从光谱项提取自旋多重度
        # 例如: 3T2g → 多重度 = 3, S = 1

        mult1 = self._extract_multiplicity(term1)
        mult2 = self._extract_multiplicity(term2)

        return mult1 == mult2

    def _extract_multiplicity(self, term: str) -> int:
        """从光谱项提取多重度"""
        # 例如: "3T2g" → 3
        if term and term[0].isdigit():
            return int(term[0])
        return 1

    def intensity_estimate(self, transition: dict) -> float:
        """
        跃迁强度估计

        考虑：
        - 自旋禁阻（强度降低~100倍）
        - Laporte禁阻（强度降低~100倍）
        """
        intensity = 1.0

        if not transition["allowed"]:
            intensity *= 0.01  # 自旋禁阻

        # d-d跃迁都是Laporte禁阻
        intensity *= 0.01

        return intensity
