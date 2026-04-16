"""
物理表示实现

本模块提供物理表示的实现，与 core 模块的表示论集成：
- PhysicalRepresentation: 物理表示基类
- IrreducibleRepresentation: 不可约表示
- SU2Representation: SU(2) 表示（角动量理论）
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ...core.group_theory.abstract_group import Group
    from ..physical_objects.state import HilbertSpace


class PhysicalRepresentation:
    """物理表示

    将群元素映射到物理算符的表示。
    """

    def __init__(self, group: "Group", representation_space: "HilbertSpace"):
        self.group = group
        self.space = representation_space
        self._matrices: dict[Any, np.ndarray] = {}

    def get_matrix(self, group_element: Any) -> np.ndarray:
        """获取群元的表示矩阵"""
        if group_element not in self._matrices:
            dim = self.space.dimension() if hasattr(self.space, "dimension") else 2
            self._matrices[group_element] = np.eye(dim, dtype=complex)
        return self._matrices[group_element]

    def set_matrix(self, group_element: Any, matrix: np.ndarray) -> None:
        """设置群元的表示矩阵"""
        self._matrices[group_element] = np.asarray(matrix)

    def is_unitary(self) -> bool:
        """检查是否酉表示"""
        for elem, mat in self._matrices.items():
            if not np.allclose(mat.conj().T @ mat, np.eye(len(mat))):
                return False
        return True

    def character(self, group_element: Any) -> complex:
        """特征标"""
        return complex(np.trace(self.get_matrix(group_element)))

    def decompose(self) -> list["IrreducibleRepresentation"]:
        """分解为不可约表示"""
        return [self]

    def tensor_product(
        self, other: "PhysicalRepresentation"
    ) -> "PhysicalRepresentation":
        """张量积表示"""
        new_space_dim = self.space.dimension() * other.space.dimension()
        from ..physical_objects.state import HilbertSpace

        new_space = HilbertSpace(new_space_dim)
        result = PhysicalRepresentation(self.group, new_space)
        return result


class IrreducibleRepresentation(PhysicalRepresentation, ABC):
    """不可约表示

    不能进一步分解的表示。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """表示维度"""
        pass

    @abstractmethod
    def highest_weight(self) -> list[int]:
        """最高权"""
        pass

    @abstractmethod
    def weight_diagram(self) -> list[tuple]:
        """权图"""
        pass

    def is_irreducible(self) -> bool:
        """总是返回 True"""
        return True


class SU2Representation(IrreducibleRepresentation):
    """SU(2)表示 - 角动量理论

    用于描述自旋和轨道角动量。
    """

    def __init__(self, j: float):
        """
        Args:
            j: 角动量量子数 j = 0, 1/2, 1, 3/2, ...
        """
        if j < 0:
            raise ValueError("角动量量子数不能为负数")
        if not isinstance(j, (int, float)):
            raise TypeError("角动量量子数必须是数字")
        if not (2 * j).is_integer() and j != int(j):
            raise ValueError("角动量量子数必须是整数或半整数")

        self.j = j
        self._dim = int(2 * j + 1)
        self._cache: dict[str, np.ndarray] = {}

        from ..physical_objects.state import HilbertSpace

        space = HilbertSpace(self._dim)
        # 使用抽象_phys层的规范群
        from ..symmetry_environments.gauge_groups import SU2GaugeGroup

        group = SU2GaugeGroup()
        super().__init__(group, space)

    @property
    def dimension(self) -> int:
        return self._dim

    def highest_weight(self) -> list[int]:
        """最高权"""
        return [int(2 * self.j)]

    def weight_diagram(self) -> list[tuple]:
        """权图"""
        weights = []
        for m in np.arange(-self.j, self.j + 1, 1):
            weights.append((m,))
        return weights

    def generators(self) -> list[np.ndarray]:
        """返回 SU(2) 生成元 J_x, J_y, J_z"""
        # 对于 SU(2) 表示，生成元是角动量算符
        return list(self._angular_momentum_matrices())

    def _angular_momentum_matrices(self) -> tuple:
        """获取角动量算符矩阵"""
        if "angular_momentum" in self._cache:
            return tuple(self._cache["angular_momentum"])

        j = self.j
        dim = self._dim
        Jx = np.zeros((dim, dim), dtype=complex)
        Jy = np.zeros((dim, dim), dtype=complex)
        Jz = np.zeros((dim, dim), dtype=complex)

        # 生成 m 值，从 -j 到 j，步长为 1
        m_values = np.arange(-j, j + 1, 1)

        for i, m in enumerate(m_values):
            Jz[i, i] = m

            if i < dim - 1:
                sqrt_factor = np.sqrt((j - m) * (j + m + 1))
                Jx[i, i + 1] = 0.5 * sqrt_factor
                Jx[i + 1, i] = 0.5 * sqrt_factor
                Jy[i, i + 1] = -0.5j * sqrt_factor
                Jy[i + 1, i] = 0.5j * sqrt_factor

        self._cache["angular_momentum"] = [Jx, Jy, Jz]
        return Jx, Jy, Jz

    def rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Wigner D矩阵 D^j(轴, 角)"""
        Jx, Jy, Jz = self._angular_momentum_matrices()
        n = np.array(axis) / np.linalg.norm(axis)
        Jn = n[0] * Jx + n[1] * Jy + n[2] * Jz

        if "scipy" not in self._cache:
            try:
                from scipy.linalg import expm

                self._cache["scipy"] = expm
            except ImportError:
                self._cache["scipy"] = None

        if self._cache.get("scipy"):
            return self._cache["scipy"](-1j * angle * Jn)
        else:
            return self._matrix_exp(-1j * angle * Jn)

    def _matrix_exp(self, matrix: np.ndarray, n_terms: int = 20) -> np.ndarray:
        """矩阵指数的数值计算"""
        result = np.eye(len(matrix), dtype=complex)
        term = np.eye(len(matrix), dtype=complex)
        for k in range(1, n_terms):
            term = term @ matrix / k
            result = result + term
        return result

    def clebsch_gordan(self, other: "SU2Representation") -> dict:
        """Clebsch-Gordan系数 C^{j1,j2;j3}_{m1,m2;m3}

        使用 utils 模块中的 ClebschGordan 类计算准确的 CG 系数
        """
        from ..utils.clebsch_gordan import ClebschGordan

        j1, j2 = self.j, other.j
        cg_coeffs = {}

        for j3 in np.arange(abs(j1 - j2), j1 + j2 + 1, 1):
            cg_coeffs[j3] = {}
            for m1 in np.arange(-j1, j1 + 1, 1):
                for m2 in np.arange(-j2, j2 + 1, 1):
                    for m3 in np.arange(-j3, j3 + 1, 1):
                        if abs(m1 + m2 - m3) < 1e-10:
                            cg = ClebschGordan.compute(j1, m1, j2, m2, j3, m3)
                            if abs(cg) > 1e-10:
                                cg_coeffs[j3][(m1, m2, m3)] = float(cg)

        return cg_coeffs

    def tensor_product_decomposition(
        self, other: "SU2Representation"
    ) -> list["SU2Representation"]:
        """张量积分解 j1 ⊗ j2 = |j1-j2| ⊕ ... ⊕ (j1+j2)"""
        j1, j2 = self.j, other.j
        j_values = []
        j = abs(j1 - j2)
        while j <= j1 + j2:
            j_values.append(j)
            j += 1
        return [SU2Representation(j) for j in j_values]

    def wigner_3j_symbol(
        self,
        j2: float,
        m1: float,
        m2: float,
        j3: float,
        m1_prime: float,
        m2_prime: float,
    ) -> float:
        """Wigner 3j 符号 (j1 j2 j3; m1 m2 m3)"""
        from ..utils.clebsch_gordan import Wigner3j

        return Wigner3j.compute(self.j, j2, j3, m1, m2, m1_prime)

    def wigner_6j_symbol(
        self, j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
    ) -> float:
        """Wigner 6j 符号 {j1 j2 j3; j4 j5 j6}"""
        from ..utils.clebsch_gordan import Wigner6j

        return Wigner6j.compute(j1, j2, j3, j4, j5, j6)


class SU3Representation(IrreducibleRepresentation):
    """SU(3)表示 - 强相互作用

    用于描述夸克的色自由度和味道。
    """

    def __init__(self, p: int, q: int):
        """
        Args:
            p: 基础表示权重分量（第一主权）
            q: 基础表示权重分量（第二主权）
        """
        if p < 0 or q < 0:
            raise ValueError("表示参数必须非负")

        self.p = p
        self.q = q
        self._dim = int((p + 1) * (q + 1) * (p + q + 2) // 2)
        self._cache: dict[str, Any] = {}

        from ..physical_objects.state import HilbertSpace

        space = HilbertSpace(self._dim)
        # 使用抽象_phys层的规范群
        from ..symmetry_environments.gauge_groups import SU3GaugeGroup

        group = SU3GaugeGroup()
        super().__init__(group, space)

    @property
    def dimension(self) -> int:
        return self._dim

    def highest_weight(self) -> list[int]:
        """最高权 (p, q)"""
        return [self.p, self.q]

    def weight_diagram(self) -> list[tuple]:
        """权图"""
        weights = []
        # SU(3) 权重形式为 (m1, m2)，满足 m1 + m2 ≤ p + q
        for m1 in range(self.p + 1):
            for m2 in range(self.q + 1):
                weights.append((m1, m2))
        return weights

    def generators(self) -> list[np.ndarray]:
        """返回 SU(3) 生成元（盖尔曼矩阵）"""
        # 使用 core 模块的 SpecialUnitaryLieAlgebra 获取生成元
        from ...core.lie_theory.specific_lie_algebra import SpecialUnitaryLieAlgebra

        su3 = SpecialUnitaryLieAlgebra(3)
        return [elem.matrix for elem in su3.basis()]

    def decomposition(self) -> list["SU3Representation"]:
        """分解为 SU(2) × U(1) 子群表示"""
        return [self]


class LorentzRepresentation(PhysicalRepresentation):
    """洛伦兹群表示

    用于描述相对论性场。
    """

    SCALAR = (0, 0)
    SPINOR = (1 / 2, 0)
    CONJUGATE_SPINOR = (0, 1 / 2)
    VECTOR = (1 / 2, 1 / 2)
    DIRAC_SPINOR = ((1 / 2, 0), (0, 1 / 2))  # 直和表示

    def __init__(self, left_chirality: float, right_chirality: float):
        self.left_chirality = left_chirality
        self.right_chirality = right_chirality
        dim = int(2 * left_chirality + 1) * int(2 * right_chirality + 1)

        from ..physical_objects.state import HilbertSpace

        space = HilbertSpace(dim)
        # 使用已实现的 LorentzGroup
        from ..symmetry_environments.lorentz_group import LorentzGroup

        group = LorentzGroup()
        super().__init__(group, space)

    @property
    def dimension(self) -> int:
        return int(2 * self.left_chirality + 1) * int(2 * self.right_chirality + 1)

    def is_unitary(self) -> bool:
        """洛伦兹群非紧致，有限维表示一般不是幺正的"""
        if self.left_chirality == 0 and self.right_chirality == 0:
            return True
        return False

    def get_representation_type(self) -> str:
        """获取表示类型"""
        if (self.left_chirality, self.right_chirality) == self.SCALAR:
            return "标量"
        elif (self.left_chirality, self.right_chirality) == self.SPINOR:
            return "左手旋量"
        elif (self.left_chirality, self.right_chirality) == self.CONJUGATE_SPINOR:
            return "右手旋量"
        elif (self.left_chirality, self.right_chirality) == self.VECTOR:
            return "矢量"
        else:
            return f"({self.left_chirality}, {self.right_chirality}) 表示"

    def tensor_product(
        self, other: "LorentzRepresentation"
    ) -> list["LorentzRepresentation"]:
        """张量积表示

        洛伦兹群表示的张量积需要分解为不可约表示的直和
        """
        # 洛伦兹群表示 (a,b) ⊗ (c,d) = (a+c, b+d) ⊕ (a+c-1, b+d) ⊕ ... ⊕ (|a-c|, |b-d|)
        # 这里简化实现，仅返回最高权重表示
        new_left = self.left_chirality + other.left_chirality
        new_right = self.right_chirality + other.right_chirality
        return [LorentzRepresentation(new_left, new_right)]


class RepresentationFactory:
    """表示工厂类"""

    _registry: dict[str, type] = {
        "SU(2)": SU2Representation,
        "SU(3)": SU3Representation,
        "Lorentz": LorentzRepresentation,
    }

    @classmethod
    def create(cls, group_name: str, **kwargs) -> IrreducibleRepresentation:
        """创建表示

        Args:
            group_name: 群名称
            **kwargs: 额外的群特定参数
        """
        if group_name not in cls._registry:
            raise ValueError(f"不支持的群: {group_name}")
        return cls._registry[group_name](**kwargs)

    @classmethod
    def register(cls, group_name: str, rep_class: type):
        """注册新的表示类"""
        cls._registry[group_name] = rep_class

    @classmethod
    def list_supported(cls) -> list[str]:
        """列出所有支持的群"""
        return list(cls._registry.keys())
