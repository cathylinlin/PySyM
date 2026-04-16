"""
点群对称性模块

包含：
- 32个晶体学点群
- 分子点群
- 点群操作
- 特征标表
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any

import numpy as np

from ..base import PhysicalSymmetry, SymmetryCategory, SymmetryOperation, SymmetryType

# -----------------------------------------------------------------------------
# 1. 点群类型枚举
# -----------------------------------------------------------------------------


class PointGroupType(Enum):
    """点群类型"""

    # 非轴向群
    C1 = "C1"  # 无对称性
    Cs = "Cs"  # 仅镜面
    Ci = "Ci"  # 仅反演中心

    # 单轴群
    Cn = "Cn"  # n重旋转轴
    Cnv = "Cnv"  # Cn + n个垂直镜面
    Cnh = "Cnh"  # Cn + 水平镜面
    Sn = "Sn"  # n重旋转反射轴

    # 双面群
    Dn = "Dn"  # Cn + n个垂直C2轴
    Dnh = "Dnh"  # Dn + 水平镜面
    Dnd = "Dnd"  # Dn + n个对角镜面

    # 立方群
    T = "T"  # 四面体群
    Th = "Th"  # T + 反演中心
    Td = "Td"  # 完全四面体群
    O = "O"  # 八面体群
    Oh = "Oh"  # 完全八面体群

    # 二十面体群
    I = "I"  # 二十面体群
    Ih = "Ih"  # 完全二十面体群

    # 线性群（分子）
    Cinfv = "C∞v"  # 线性分子（无反演中心）
    Dinfh = "D∞h"  # 线性分子（有反演中心）


class CrystalSystem(Enum):
    """晶系"""

    TRICLINIC = "triclinic"  # 三斜
    MONOCLINIC = "monoclinic"  # 单斜
    ORTHORHOMBIC = "orthorhombic"  # 正交
    TETRAGONAL = "tetragonal"  # 四方
    TRIGONAL = "trigonal"  # 三方
    HEXAGONAL = "hexagonal"  # 六方
    CUBIC = "cubic"  # 立方


# -----------------------------------------------------------------------------
# 2. 点群操作
# -----------------------------------------------------------------------------


@dataclass
class PointGroupOperation(SymmetryOperation):
    """点群操作"""

    operation_type: str  # 'E', 'Cn', 'sigma', 'i', 'Sn'
    axis: np.ndarray  # 旋转轴或镜面法向
    order: int = 1  # 旋转阶数
    power: int = 1  # 幂次

    def __post_init__(self):
        if isinstance(self.axis, list):
            self.axis = np.array(self.axis, dtype=float)
        if np.linalg.norm(self.axis) > 0:
            self.axis = self.axis / np.linalg.norm(self.axis)
        self._group = None  # 在被添加到具体点群时设置

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @property
    def is_continuous(self) -> bool:
        return False

    def compose(self, other: "PointGroupOperation") -> "PointGroupOperation":
        """组合两个操作"""
        # 矩阵乘法
        M1 = self.matrix_representation()
        M2 = other.matrix_representation()
        M_combined = M1 @ M2
        return self._matrix_to_operation(M_combined)

    def inverse(self) -> "PointGroupOperation":
        """逆操作"""
        if self.operation_type == "E":
            return self
        elif self.operation_type == "Cn":
            return PointGroupOperation(
                "Cn", self.axis, self.order, self.order - self.power
            )
        elif self.operation_type == "sigma":
            return self  # 镜面自逆
        elif self.operation_type == "i":
            return self  # 反演自逆
        elif self.operation_type == "Sn":
            return PointGroupOperation(
                "Sn", self.axis, self.order, self.order - self.power
            )

    def act_on(self, obj: Any) -> Any:
        """作用于对象"""
        M = self.matrix_representation()
        if hasattr(obj, "position"):
            obj.position = M @ obj.position
        return obj

    def matrix_representation(self, dimension: int = 3) -> np.ndarray:
        """3D矩阵表示"""
        if self.operation_type == "E":
            return np.eye(3)

        elif self.operation_type == "i":
            return -np.eye(3)

        elif self.operation_type == "sigma":
            # 反射矩阵: R = I - 2*n*n^T
            n = self.axis.reshape(3, 1)
            return np.eye(3) - 2 * n @ n.T

        elif self.operation_type == "Cn":
            # 旋转矩阵: R = exp(-i*theta*n·J)
            theta = 2 * np.pi * self.power / self.order
            return self._rotation_matrix(self.axis, theta)

        elif self.operation_type == "Sn":
            # 旋转反射 = 旋转 + 反射
            R = self._rotation_matrix(self.axis, 2 * np.pi * self.power / self.order)
            n = self.axis.reshape(3, 1)
            sigma = np.eye(3) - 2 * n @ n.T
            return R @ sigma

    def _rotation_matrix(self, axis: np.ndarray, theta: float) -> np.ndarray:
        """绕轴旋转矩阵"""
        n = axis / np.linalg.norm(axis)
        c, s = np.cos(theta), np.sin(theta)
        nx, ny, nz = n

        R = np.array(
            [
                [
                    c + nx * nx * (1 - c),
                    nx * ny * (1 - c) - nz * s,
                    nx * nz * (1 - c) + ny * s,
                ],
                [
                    ny * nx * (1 - c) + nz * s,
                    c + ny * ny * (1 - c),
                    ny * nz * (1 - c) - nx * s,
                ],
                [
                    nz * nx * (1 - c) - ny * s,
                    nz * ny * (1 - c) + nx * s,
                    c + nz * nz * (1 - c),
                ],
            ]
        )
        return R

    def _matrix_to_operation(self, M: np.ndarray) -> "PointGroupOperation":
        """从矩阵反推操作类型"""
        det = np.linalg.det(M)
        trace = np.trace(M)

        if np.allclose(M, np.eye(3)):
            return PointGroupOperation("E", [0, 0, 1], 1, 1)
        elif np.allclose(M, -np.eye(3)):
            return PointGroupOperation("i", [0, 0, 1], 1, 1)
        elif np.isclose(det, -1):
            # 镜面或旋转反射
            # 找镜面法向
            eigenvalues, eigenvectors = np.linalg.eig(M)
            # 实本征值-1对应的特征向量是镜面法向
            for i, ev in enumerate(eigenvalues):
                if np.isclose(ev, -1) and np.isclose(np.imag(ev), 0):
                    normal = np.real(eigenvectors[:, i])
                    # 检查是否纯反射
                    M_check = np.eye(3) - 2 * np.outer(normal, normal)
                    if np.allclose(M, M_check):
                        return PointGroupOperation("sigma", normal, 1, 1)
        elif np.isclose(det, 1):
            # 旋转
            # 旋转轴是本征值1对应的特征向量
            eigenvalues, eigenvectors = np.linalg.eig(M)
            for i, ev in enumerate(eigenvalues):
                if np.isclose(ev, 1) and np.isclose(np.imag(ev), 0):
                    axis = np.real(eigenvectors[:, i])
                    # 确定旋转角
                    theta = np.arccos((trace - 1) / 2)
                    n = int(2 * np.pi / theta) if theta > 0 else 1
                    return PointGroupOperation("Cn", axis, n, 1)

        # 默认返回单位元
        return PointGroupOperation("E", [0, 0, 1], 1, 1)

    def character(self) -> int:
        """特征标（迹）"""
        return int(np.round(np.trace(self.matrix_representation())))

    def __str__(self) -> str:
        if self.operation_type == "E":
            return "E"
        elif self.operation_type == "i":
            return "i"
        elif self.operation_type == "sigma":
            return "σ"
        elif self.operation_type == "Cn":
            if self.power == 1:
                return f"C{self.order}"
            return f"C{self.order}^{self.power}"
        elif self.operation_type == "Sn":
            if self.power == 1:
                return f"S{self.order}"
            return f"S{self.order}^{self.power}"
        return f"{self.operation_type}"


# -----------------------------------------------------------------------------
# 3. 点群基类
# -----------------------------------------------------------------------------


class PointGroup(PhysicalSymmetry):
    """
    点群基类

    所有具体点群的基类
    """

    # 晶体学点群列表
    CRYSTALLOGRAPHIC_GROUPS = {
        CrystalSystem.TRICLINIC: ["C1", "Ci"],
        CrystalSystem.MONOCLINIC: ["C2", "Cs", "C2h"],
        CrystalSystem.ORTHORHOMBIC: ["D2", "C2v", "D2h"],
        CrystalSystem.TETRAGONAL: ["C4", "S4", "C4h", "D4", "C4v", "D2d", "D4h"],
        CrystalSystem.TRIGONAL: ["C3", "C3i", "D3", "C3v", "D3d"],
        CrystalSystem.HEXAGONAL: ["C6", "C3h", "C6h", "D6", "C6v", "D3h", "D6h"],
        CrystalSystem.CUBIC: ["T", "Th", "Td", "O", "Oh"],
    }

    def __init__(
        self, name: str, schoenflies: str, order: int, crystal_system: CrystalSystem
    ):
        super().__init__(
            symmetry_type=SymmetryType.SPATIAL_ROTATION,  # 点群是空间旋转的离散子群
            group=None,  # 稍后设置
            category=SymmetryCategory.DISCRETE,
        )

        self._name = name
        self._schoenflies = schoenflies
        self._order = order
        self._crystal_system = crystal_system
        self._elements: list[PointGroupOperation] = []
        self._classes: list[list[PointGroupOperation]] = []
        self._character_table: np.ndarray | None = None
        self._irrep_names: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def schoenflies(self) -> str:
        """熊夫利符号"""
        return self._schoenflies

    @property
    def order(self) -> int:
        """群阶"""
        return self._order

    @property
    def crystal_system(self) -> CrystalSystem:
        return self._crystal_system

    @property
    def elements(self) -> list[PointGroupOperation]:
        """所有群元素"""
        return self._elements

    @property
    def classes(self) -> list[list[PointGroupOperation]]:
        """共轭类"""
        return self._classes

    @abstractmethod
    def _generate_elements(self):
        """生成所有群元素"""
        pass

    @abstractmethod
    def _compute_conjugacy_classes(self):
        """计算共轭类"""
        pass

    def generators(self) -> list[Any]:
        """生成元"""
        return []

    def conserved_quantity(self) -> str:
        """点群对应的守恒量"""
        return "symmetry_quantum_numbers"

    def create_operation(self, params) -> PointGroupOperation:
        """创建点群操作"""
        pass

    def contains_operation(self, operation: PointGroupOperation) -> bool:
        """检查是否包含某操作"""
        for elem in self._elements:
            if np.allclose(
                elem.matrix_representation(), operation.matrix_representation()
            ):
                return True
        return False

    def multiplication_table(self) -> np.ndarray:
        """乘法表"""
        n = len(self._elements)
        table = np.zeros((n, n), dtype=int)

        for i, g1 in enumerate(self._elements):
            for j, g2 in enumerate(self._elements):
                g_product = g1.compose(g2)
                # 找到乘积元素在群中的索引
                for k, g in enumerate(self._elements):
                    if np.allclose(
                        g.matrix_representation(), g_product.matrix_representation()
                    ):
                        table[i, j] = k
                        break

        return table

    def character_table_full(self) -> tuple[np.ndarray, list[str], list[str]]:
        """
        完整特征标表

        Returns:
            table: 特征标矩阵
            irrep_names: 不可约表示名称
            class_names: 共轭类名称
        """
        return self._character_table, self._irrep_names, self._class_names


# -----------------------------------------------------------------------------
# 4. 具体点群实现
# -----------------------------------------------------------------------------


class CyclicGroup(PointGroup):
    """循环群 Cn"""

    def __init__(self, n: int):
        self._n = n

        # 确定晶系
        if n == 1:
            crystal_sys = CrystalSystem.TRICLINIC
        elif n == 2:
            crystal_sys = CrystalSystem.MONOCLINIC
        elif n in [3, 4, 6]:
            if n == 3:
                crystal_sys = CrystalSystem.TRIGONAL
            elif n == 4:
                crystal_sys = CrystalSystem.TETRAGONAL
            else:
                crystal_sys = CrystalSystem.HEXAGONAL
        else:
            crystal_sys = None  # 非晶体学点群

        super().__init__(
            name=f"C{n}", schoenflies=f"C{n}", order=n, crystal_system=crystal_sys
        )

        self._generate_elements()
        self._compute_conjugacy_classes()
        self._compute_character_table()

    def _generate_elements(self):
        """生成元素: E, Cn, Cn², ..., Cn^(n-1)"""
        # 单位元
        E = PointGroupOperation("E", [0, 0, 1], 1, 1)
        self._elements.append(E)

        # Cn轴沿z方向
        for k in range(1, self._n):
            C = PointGroupOperation("Cn", [0, 0, 1], self._n, k)
            self._elements.append(C)

    def _compute_conjugacy_classes(self):
        """循环群每个元素自成一类（阿贝尔群）"""
        self._classes = [[elem] for elem in self._elements]
        self._class_names = [str(elem) for elem in self._elements]

    def _compute_character_table(self):
        """计算特征标表"""
        n = self._n
        # Cn群有n个一维不可约表示
        self._character_table = np.zeros((n, n), dtype=complex)

        # 不可约表示名称: A, E₁, E₁*, E₂, E₂*, ..., B(偶数n)
        self._irrep_names = ["A"]
        if n > 1:
            for i in range(1, (n + 1) // 2):
                self._irrep_names.append(f"E{i}")
                if n - i != i:  # 共轭对
                    self._irrep_names.append(f"E{i}*")
            if n % 2 == 0:
                self._irrep_names.append("B")

        # 特征标: χ_j(C^k) = exp(2πijk/n)
        for j_idx in range(n):
            for k in range(n):
                self._character_table[j_idx, k] = np.exp(2 * 1j * np.pi * j_idx * k / n)

        # 对于偶数n，B表示的特征标
        if n % 2 == 0:
            self._character_table[n // 2, :] = np.array([(-1) ** k for k in range(n)])


class DihedralGroup(PointGroup):
    """双面群 Dn"""

    def __init__(self, n: int):
        self._n = n

        # 确定晶系
        if n == 2:
            crystal_sys = CrystalSystem.ORTHORHOMBIC
        elif n in [3, 4, 6]:
            if n == 3:
                crystal_sys = CrystalSystem.TRIGONAL
            elif n == 4:
                crystal_sys = CrystalSystem.TETRAGONAL
            else:
                crystal_sys = CrystalSystem.HEXAGONAL
        else:
            crystal_sys = None

        super().__init__(
            name=f"D{n}", schoenflies=f"D{n}", order=2 * n, crystal_system=crystal_sys
        )

        self._generate_elements()
        self._compute_conjugacy_classes()
        self._compute_character_table()

    def _generate_elements(self):
        """生成元素: E, n个Cn旋转, n个C2轴"""
        # 单位元
        E = PointGroupOperation("E", [0, 0, 1], 1, 1)
        self._elements.append(E)

        # n个Cn旋转
        for k in range(1, self._n):
            C = PointGroupOperation("Cn", [0, 0, 1], self._n, k)
            self._elements.append(C)

        # n个垂直C2轴（在xy平面内）
        for i in range(self._n):
            angle = np.pi * i / self._n
            axis = [np.cos(angle), np.sin(angle), 0]
            C2 = PointGroupOperation("Cn", axis, 2, 1)
            self._elements.append(C2)

    def _compute_conjugacy_classes(self):
        """计算共轭类"""
        self._classes = []
        self._class_names = []

        # E自成一类
        self._classes.append([self._elements[0]])
        self._class_names.append("E")

        # Cn^k 和 Cn^(n-k) 在同一类
        for k in range(1, (self._n + 1) // 2):
            if k < self._n - k:
                class_k = [self._elements[k], self._elements[self._n - k]]
                self._classes.append(class_k)
                self._class_names.append(f"2C{self._n}^{k}")

        # C2轴的类
        if self._n % 2 == 0:
            # 偶数n: 两组C2轴
            class1 = [self._elements[self._n + 2 * i] for i in range(self._n // 2)]
            class2 = [self._elements[self._n + 2 * i + 1] for i in range(self._n // 2)]
            self._classes.append(class1)
            self._classes.append(class2)
            self._class_names.append(f"{self._n // 2}C2'")
            self._class_names.append(f'{self._n // 2}C2"')
        else:
            # 奇数n: 所有C2轴在一类
            class_c2 = [self._elements[self._n + i] for i in range(self._n)]
            self._classes.append(class_c2)
            self._class_names.append(f"{self._n}C2")

    def _compute_character_table(self):
        """计算特征标表"""
        n_classes = len(self._classes)
        # Dn有4个不可约表示（n为奇数）或5个（n为偶数）
        n_irreps = 4 if self._n % 2 == 1 else 5

        self._character_table = np.zeros((n_irreps, n_classes))

        # A1表示（全对称）
        self._character_table[0, :] = 1

        # A2表示
        if self._n % 2 == 1:
            self._character_table[1, :] = [1, 1, -1]
        else:
            self._character_table[1, :] = [1, 1, 1, -1, -1]

        # E表示（二维）
        # 需要计算每个类的特征标
        # 这部分较复杂，简化处理
        pass


class CnvGroup(PointGroup):
    """Cnv群"""

    def __init__(self, n: int):
        self._n = n

        # 确定晶系
        if n == 2:
            crystal_sys = CrystalSystem.ORTHORHOMBIC
        elif n == 3:
            crystal_sys = CrystalSystem.TRIGONAL
        elif n == 4:
            crystal_sys = CrystalSystem.TETRAGONAL
        elif n == 6:
            crystal_sys = CrystalSystem.HEXAGONAL
        else:
            crystal_sys = None

        super().__init__(
            name=f"C{n}v", schoenflies=f"C{n}v", order=2 * n, crystal_system=crystal_sys
        )

        self._generate_elements()
        self._compute_conjugacy_classes()
        self._compute_character_table()

    def _generate_elements(self):
        """生成元素: Cn操作 + n个垂直镜面"""
        # 单位元
        E = PointGroupOperation("E", [0, 0, 1], 1, 1)
        self._elements.append(E)

        # Cn旋转
        for k in range(1, self._n):
            C = PointGroupOperation("Cn", [0, 0, 1], self._n, k)
            self._elements.append(C)

        # n个垂直镜面（包含z轴）
        for i in range(self._n):
            angle = np.pi * i / self._n
            normal = [np.sin(angle), -np.cos(angle), 0]  # 镜面法向
            sigma = PointGroupOperation("sigma", normal, 1, 1)
            self._elements.append(sigma)

    def _compute_conjugacy_classes(self):
        """计算共轭类"""
        # 类似Dn，但用镜面代替C2轴
        self._classes = [[self._elements[0]]]
        self._class_names = ["E"]

        # Cn的类
        for k in range(1, (self._n + 1) // 2):
            if k < self._n - k:
                self._classes.append([self._elements[k], self._elements[self._n - k]])
                self._class_names.append(f"2C{self._n}^{k}")

        # 镜面的类
        if self._n % 2 == 0:
            self._classes.append(
                [self._elements[self._n + 2 * i] for i in range(self._n // 2)]
            )
            self._classes.append(
                [self._elements[self._n + 2 * i + 1] for i in range(self._n // 2)]
            )
            self._class_names.append(f"{self._n // 2}σv")
            self._class_names.append(f"{self._n // 2}σd")
        else:
            self._classes.append([self._elements[self._n + i] for i in range(self._n)])
            self._class_names.append(f"{self._n}σv")

    def _compute_character_table(self):
        """计算特征标表"""
        pass


class OhGroup(PointGroup):
    """Oh群 - 完全八面体群"""

    def __init__(self):
        super().__init__(
            name="Oh", schoenflies="Oh", order=48, crystal_system=CrystalSystem.CUBIC
        )

        self._generate_elements()
        self._compute_conjugacy_classes()
        self._compute_character_table()

    def _generate_elements(self):
        """生成48个元素"""
        # 单位元
        E = PointGroupOperation("E", [0, 0, 1], 1, 1)
        self._elements.append(E)

        # 反演
        i = PointGroupOperation("i", [0, 0, 1], 1, 1)

        # 8个C3轴（体对角线）
        for signs in product([1, -1], repeat=2):
            for perm in [(0, 1, 2), (0, 2, 1)]:
                axis = [0, 0, 0]
                axis[perm[0]] = 1
                axis[perm[1]] = signs[0]
                axis[perm[2]] = signs[1]
                axis = np.array(axis) / np.sqrt(3)
                self._elements.append(PointGroupOperation("Cn", axis, 3, 1))

        # 6个C4轴（坐标轴）
        for axis in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]:
            self._elements.append(PointGroupOperation("Cn", axis, 4, 1))
            self._elements.append(PointGroupOperation("Cn", axis, 4, 3))

        # 6个C2轴（坐标轴方向）
        for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            self._elements.append(PointGroupOperation("Cn", axis, 2, 1))

        # 更多C2轴（面对角线）
        # ... 需要完整枚举

        # 镜面和旋转反射
        # ...

    def _compute_conjugacy_classes(self):
        """Oh群有10个共轭类"""
        self._class_names = [
            "E",
            "8C3",
            "6C2",
            "6C4",
            "3C2(=C4²)",
            "i",
            "6S4",
            "8S6",
            "3σh",
            "6σd",
        ]

    def _compute_character_table(self):
        """Oh群特征标表（10个不可约表示）"""
        # 不可约表示名称
        self._irrep_names = [
            "A1g",
            "A2g",
            "Eg",
            "T1g",
            "T2g",
            "A1u",
            "A2u",
            "Eu",
            "T1u",
            "T2u",
        ]

        # 完整的特征标表
        self._character_table = np.array(
            [
                # A1g
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                # A2g
                [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
                # Eg (二维)
                [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
                # T1g (三维)
                [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
                # T2g (三维)
                [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
                # A1u
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                # A2u
                [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
                # Eu (二维)
                [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
                # T1u (三维)
                [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
                # T2u (三维)
                [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],
            ]
        )


class TdGroup(PointGroup):
    """Td群 - 完全四面体群"""

    def __init__(self):
        super().__init__(
            name="Td", schoenflies="Td", order=24, crystal_system=CrystalSystem.CUBIC
        )

        self._generate_elements()
        self._compute_conjugacy_classes()
        self._compute_character_table()

    def _generate_elements(self):
        """生成24个元素"""
        # 单位元
        E = PointGroupOperation("E", [0, 0, 1], 1, 1)
        self._elements.append(E)

        # 8个C3轴
        # 3个C2轴
        # 6个镜面
        # 6个S4轴
        # ... 完整枚举
        pass

    def _compute_conjugacy_classes(self):
        """Td群有5个共轭类"""
        self._class_names = ["E", "8C3", "3C2", "6S4", "6σd"]

    def _compute_character_table(self):
        """Td群特征标表"""
        self._irrep_names = ["A1", "A2", "E", "T1", "T2"]

        self._character_table = np.array(
            [
                # A1
                [1, 1, 1, 1, 1],
                # A2
                [1, 1, 1, -1, -1],
                # E (二维)
                [2, -1, 2, 0, 0],
                # T1 (三维)
                [3, 0, -1, 1, -1],
                # T2 (三维)
                [3, 0, -1, -1, 1],
            ]
        )
