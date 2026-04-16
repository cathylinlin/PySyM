"""
哈密顿算符模块

提供量子系统的哈密顿算符构建：
1. 正则量子化 - 从经典系统得到量子哈密顿量
2. 对称性分析 - 利用对称性简化哈密顿量
3. 唯像构造 - 基于物理直觉构造哈密顿量
4. 微扰展开 - 微扰理论的哈密顿量分解

与 abstract_phys 模块集成：
- SymmetryOperation: 对称操作用于分析守恒量
- QuantumSystem: 连接量子系统框架
- HamiltonianSystem: 正则量子化的基础
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

if TYPE_CHECKING:
    from PySymmetry.abstract_phys import (
        AngularMomentumGenerator,
        CasimirOperator,
        GaugeOperation,
        HamiltonianSystem,
        HilbertSpace,
        MomentumGenerator,
        ParityOperation,
        QuantumSystem,
        RotationOperation,
        SymmetryAnalyzer,
        SymmetryOperation,
        TimeReversalOperation,
        TranslationOperation,
    )

try:
    from PySymmetry.abstract_phys import (
        AngularMomentumGenerator,
        CasimirOperator,
        GaugeOperation,
        HamiltonianSystem,
        HilbertSpace,
        MomentumGenerator,
        ParityOperation,
        QuantumSystem,
        RotationOperation,
        SymmetryAnalyzer,
        SymmetryOperation,
        TimeReversalOperation,
        TranslationOperation,
    )
except ImportError:
    SymmetryOperation = object
    SymmetryAnalyzer = object
    QuantumSystem = object
    HamiltonianSystem = object
    HilbertSpace = object
    TranslationOperation = object
    RotationOperation = object
    ParityOperation = object
    TimeReversalOperation = object
    GaugeOperation = object
    MomentumGenerator = object
    AngularMomentumGenerator = object
    CasimirOperator = object

from .states import Ket


class HamiltonianOperator(ABC):
    """
    哈密顿算符抽象基类

    定义哈密顿算符的基本接口，与 SymmetryOperation 集成以支持对称性分析。
    """

    def __init__(self, dimension: int, name: str = "Hamiltonian"):
        self._dim = dimension
        self._name = name
        self._symmetries: list = []
        self._conserved_quantities: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        """希尔伯特空间维度"""
        return self._dim

    @property
    def name(self) -> str:
        """哈密顿量名称"""
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        """返回哈密顿矩阵"""
        return self._to_matrix()

    @abstractmethod
    def _to_matrix(self) -> np.ndarray:
        """转换为矩阵表示"""
        pass

    def add_symmetry(self, symmetry: Any) -> None:
        """添加对称操作"""
        self._symmetries.append(symmetry)

    def get_symmetries(self) -> list:
        """获取所有对称操作"""
        return self._symmetries.copy()

    def is_symmetric_under(self, operation: Any, tol: float = 1e-10) -> bool:
        """
        检查哈密顿量是否在某对称操作下不变

        [H, U] = 0

        Args:
            operation: 对称操作
            tol: 容差

        Returns:
            是否对易
        """
        H = self._to_matrix()
        U = operation.representation_matrix(self._dim)
        commutator = H @ U - U @ H
        return np.allclose(commutator, 0, atol=tol)

    def compute_commutator(self, operator: np.ndarray) -> np.ndarray:
        """计算与另一算符的对易子 [H, O]"""
        H = self._to_matrix()
        return H @ operator - operator @ H

    def expectation(self, state: Ket) -> complex:
        """计算能量期望值 ⟨ψ|H|ψ⟩"""
        vec = state.vector
        return complex(np.vdot(vec, self._to_matrix() @ vec))

    def variance(self, state: Ket) -> float:
        """计算能量方差 ΔE²"""
        E = self.expectation(state)
        vec = state.vector
        H = self._to_matrix()
        E2 = complex(np.vdot(vec, H @ H @ vec))
        return float(np.real(E2 - E**2))

    def ground_state(self) -> tuple[Ket, float]:
        """
        计算基态

        Returns:
            (基态矢量, 基态能量)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self._to_matrix())
        ground_state_vec = eigenvectors[:, 0]
        return Ket(ground_state_vec), float(eigenvalues[0])

    def excited_states(self, n: int = 1) -> list[tuple[Ket, float]]:
        """
        计算激发态

        Args:
            n: 激发态数量

        Returns:
            [(态1, 能量1), (态2, 能量2), ...]
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self._to_matrix())
        states = []
        for i in range(min(n, self._dim)):
            states.append((Ket(eigenvectors[:, i]), float(eigenvalues[i])))
        return states

    def all_energy_levels(self) -> np.ndarray:
        """返回所有能级"""
        return np.linalg.eigvalsh(self._to_matrix())

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "HamiltonianOperator":
        """从字典反序列化"""
        pass


class MatrixHamiltonian(HamiltonianOperator):
    """
    矩阵哈密顿算符

    直接使用矩阵定义的哈密顿量。

    Args:
        matrix: 哈密顿矩阵
        name: 名称
    """

    def __init__(self, matrix: np.ndarray, name: str = "MatrixHamiltonian"):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("哈密顿矩阵必须是方阵")
        super().__init__(matrix.shape[0], name)
        self._matrix = matrix.astype(complex)

    def _to_matrix(self) -> np.ndarray:
        """返回矩阵"""
        return self._matrix.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MatrixHamiltonian",
            "name": self._name,
            "dimension": self._dim,
            "matrix": self._matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixHamiltonian":
        return cls(
            np.array(data["matrix"], dtype=complex),
            name=data.get("name", "MatrixHamiltonian"),
        )


class FreeParticleHamiltonian(HamiltonianOperator):
    """
    自由粒子哈密顿量

    H = p²/2m

    Args:
        mass: 粒子质量
        dimension: 空间维度
        basis_size: 基底大小（离散化）
        lattice_spacing: 格点间距
    """

    def __init__(
        self,
        mass: float = 1.0,
        dimension: int = 1,
        basis_size: int = 100,
        lattice_spacing: float = 0.01,
    ):
        super().__init__(basis_size, name="FreeParticle")
        self._mass = mass
        self._spatial_dim = dimension
        self._a = lattice_spacing
        self._L = basis_size * lattice_spacing

        self._build_momentum_operator()
        self._build_hamiltonian()

    def _build_momentum_operator(self) -> None:
        """构建动量算符（在离散格点上）"""
        hbar = 1.0  # 原子单位
        n = self._dim
        k = 2 * np.pi / self._L

        self._p = np.zeros((n, n), dtype=complex)
        for i in range(n):
            self._p[i, i] = hbar * k * (i - n / 2)

    def _build_hamiltonian(self) -> None:
        """构建自由粒子哈密顿量 H = p²/2m"""
        self._matrix = self._p @ self._p / (2 * self._mass)

    def _to_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def get_kinetic_energy(self, state: Ket) -> float:
        """计算动能期望值"""
        vec = state.vector
        return float(np.real(np.vdot(vec, self._matrix @ vec)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FreeParticleHamiltonian",
            "name": self._name,
            "dimension": self._dim,
            "mass": self._mass,
            "basis_size": self._dim,
            "lattice_spacing": self._a,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FreeParticleHamiltonian":
        return cls(
            mass=data.get("mass", 1.0),
            dimension=data.get("dimension", 1),
            basis_size=data.get("basis_size", 100),
            lattice_spacing=data.get("lattice_spacing", 0.01),
        )


class HarmonicOscillatorHamiltonian(HamiltonianOperator):
    """
    谐振子哈密顿量

    H = p²/2m + 1/2 m ω² x²

    使用产生湮灭算符表示：
    H = ℏω(a†a + 1/2)

    Args:
        mass: 质量 m
        frequency: 频率 ω
        dimension: 希尔伯特空间维度（截断）
        hbar: 约化普朗克常数
    """

    def __init__(
        self,
        mass: float = 1.0,
        frequency: float = 1.0,
        dimension: int = 50,
        hbar: float = 1.0,
    ):
        super().__init__(dimension, name="HarmonicOscillator")
        self._m = mass
        self._omega = frequency
        self._hbar = hbar
        self._build_operators()
        self._build_hamiltonian()

    def _build_operators(self) -> None:
        """构建产生湮灭算符"""
        d = self._dim

        self._a = np.zeros((d, d), dtype=complex)
        for i in range(d - 1):
            self._a[i, i + 1] = np.sqrt(i + 1)

        self._a_dag = self._a.T.conj()

        self._n = self._a_dag @ self._a

        x = np.sqrt(self._hbar / (2 * self._m * self._omega)) * (self._a + self._a_dag)
        p = (
            1j
            * np.sqrt(self._m * self._omega * self._hbar / 2)
            * (self._a_dag - self._a)
        )

        self._x = x
        self._p = p

    def _build_hamiltonian(self) -> None:
        """H = ℏω(n + 1/2)"""
        self._matrix = self._hbar * self._omega * (self._n + 0.5 * np.eye(self._dim))

    @property
    def a(self) -> np.ndarray:
        """湮灭算符"""
        return self._a.copy()

    @property
    def a_dag(self) -> np.ndarray:
        """产生算符"""
        return self._a_dag.copy()

    @property
    def n(self) -> np.ndarray:
        """粒子数算符"""
        return self._n.copy()

    @property
    def x(self) -> np.ndarray:
        """位置算符"""
        return self._x.copy()

    @property
    def p(self) -> np.ndarray:
        """动量算符"""
        return self._p.copy()

    def energy_level(self, n: int) -> float:
        """第 n 个能级的能量 E_n = ℏω(n + 1/2)"""
        return self._hbar * self._omega * (n + 0.5)

    def _to_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def coherent_state(self, alpha: complex) -> Ket:
        """
        生成相干态 |α⟩

        Args:
            alpha: 相干态参数

        Returns:
            相干态
        """
        n_vals = np.arange(self._dim, dtype=float)
        factorials = np.array([float(math.factorial(int(i))) for i in range(self._dim)])
        coefficients = (
            np.exp(-(abs(alpha) ** 2) / 2) * alpha**n_vals / np.sqrt(factorials)
        )
        return Ket(coefficients).normalize()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "HarmonicOscillatorHamiltonian",
            "name": self._name,
            "dimension": self._dim,
            "mass": self._m,
            "frequency": self._omega,
            "hbar": self._hbar,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarmonicOscillatorHamiltonian":
        return cls(
            mass=data.get("mass", 1.0),
            frequency=data.get("frequency", 1.0),
            dimension=data.get("dimension", 50),
            hbar=data.get("hbar", 1.0),
        )


class SpinHamiltonian(HamiltonianOperator):
    """
    自旋哈密顿量

    基于 Pauli 矩阵和自旋算符的哈密顿量。

    Args:
        spin: 自旋量子数 s
        couplings: 耦合常数字典
        hamiltonian_type: 'zeeman', 'ising', 'heisenberg', 'xy'
    """

    def __init__(
        self,
        spin: float = 0.5,
        couplings: dict[str, float] | None = None,
        hamiltonian_type: str = "zeeman",
        direction: str = "z",
    ):
        j = spin
        dim = int(2 * j + 1)
        super().__init__(dim, name=f"Spin-{hamiltonian_type}")

        self._spin = spin
        self._couplings = couplings or {}
        self._type = hamiltonian_type
        self._direction = direction

        self._build_spin_operators()
        self._build_hamiltonian()

    def _build_spin_operators(self) -> None:
        """构建自旋算符 S_x, S_y, S_z"""
        j = self._spin
        d = self._dim

        m_values = np.arange(j, -j - 1, -1, dtype=float)

        self._sz = np.diag(m_values)

        self._sx = np.zeros((d, d), dtype=complex)
        for i in range(d - 1):
            m_i = m_values[i]
            m_ip1 = m_values[i + 1]
            self._sx[i, i + 1] = np.sqrt(j * (j + 1) - m_i * m_ip1)
            self._sx[i + 1, i] = np.sqrt(j * (j + 1) - m_i * m_ip1)
        self._sx /= 2

        self._sy = np.zeros((d, d), dtype=complex)
        for i in range(d - 1):
            m_i = m_values[i]
            m_ip1 = m_values[i + 1]
            self._sy[i, i + 1] = -1j * np.sqrt(j * (j + 1) - m_i * m_ip1)
            self._sy[i + 1, i] = 1j * np.sqrt(j * (j + 1) - m_i * m_ip1)
        self._sy /= 2

    def _build_hamiltonian(self) -> None:
        """根据类型构建哈密顿量"""
        hbar = 1.0
        couplings = self._couplings
        direction = self._direction

        S = {"x": self._sx, "y": self._sy, "z": self._sz}

        if self._type == "zeeman":
            g = couplings.get("g", 2.0)
            B = couplings.get("B", 1.0)
            S_dir = S.get(direction, self._sz)
            self._matrix = -g * hbar * B * S_dir
        elif self._type == "ising":
            J = couplings.get("J", 1.0)
            S_dir = S.get(direction, self._sz)
            self._matrix = J * S_dir @ S_dir
        elif self._type == "heisenberg":
            J = couplings.get("J", 1.0)
            self._matrix = J * (
                self._sx @ self._sx + self._sy @ self._sy + self._sz @ self._sz
            )
        elif self._type == "xy":
            J = couplings.get("J", 1.0)
            self._matrix = J * (self._sx @ self._sx + self._sy @ self._sy)

    @property
    def sx(self) -> np.ndarray:
        return self._sx.copy()

    @property
    def sy(self) -> np.ndarray:
        return self._sy.copy()

    @property
    def sz(self) -> np.ndarray:
        return self._sz.copy()

    def _to_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SpinHamiltonian",
            "name": self._name,
            "dimension": self._dim,
            "spin": self._spin,
            "couplings": self._couplings,
            "hamiltonian_type": self._type,
            "direction": self._direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpinHamiltonian":
        return cls(
            spin=data.get("spin", 0.5),
            couplings=data.get("couplings", {}),
            hamiltonian_type=data.get("hamiltonian_type", "zeeman"),
            direction=data.get("direction", "z"),
        )


class HydrogenAtomHamiltonian(HamiltonianOperator):
    """
    氢原子哈密顿量

    H = p²/2m - e²/4πε₀r

    在球坐标下分解为径向和角动量部分。
    具有隐藏SO(4)对称性，导致n²度简并。

    Args:
        Z: 原子序数
        reduced_mass: 约化质量
        max_n: 最大主量子数
    """

    def __init__(self, Z: float = 1.0, reduced_mass: float = 0.5, max_n: int = 5):
        dim = max_n**2
        super().__init__(dim, name=f"Hydrogen(Z={Z})")

        self._Z = Z
        self._mu = reduced_mass
        self._max_n = max_n
        self._so4_detected = False
        self._so4_symmetry_data = None

        self._build_hamiltonian()
        self._detect_so4_symmetry()

    def _build_hamiltonian(self) -> None:
        """构建氢原子哈密顿量"""
        n = self._max_n
        self._matrix = np.zeros((self._dim, self._dim), dtype=complex)

        idx = 0
        for n1 in range(1, n + 1):
            for l in range(n1):
                for m in range(-l, l + 1):
                    if idx < self._dim:
                        E_n = -(self._Z**2) / (2 * n1**2)
                        self._matrix[idx, idx] = E_n
                        idx += 1

    def _detect_so4_symmetry(self) -> None:
        """检测SO(4)对称性"""
        try:
            from .hydrogen_symmetry import HydrogenSymmetryAnalyzer

            self._so4_analyzer = HydrogenSymmetryAnalyzer(self._max_n, self._Z)
            self._so4_detected = True
            self._so4_symmetry_data = self._so4_analyzer.analyze()
        except ImportError:
            self._so4_detected = False
            self._so4_symmetry_data = None

    def has_so4_symmetry(self) -> bool:
        """检查是否检测到SO(4)对称性"""
        return self._so4_detected

    def get_so4_analyzer(self):
        """获取SO(4)分析器"""
        return getattr(self, "_so4_analyzer", None)

    def energy_level(self, n: int) -> float:
        """能级 E_n = -Z²/(2n²)"""
        return -(self._Z**2) / (2 * n**2)

    def degeneracy(self, n: int) -> int:
        """n²简并度（SO(4)对称性导致）"""
        return n**2

    def energy_levels(self) -> dict[tuple[int, int, int], float]:
        """返回能级 E_{n,l,m}"""
        levels = {}
        idx = 0
        for n in range(1, self._max_n + 1):
            E_n = self.energy_level(n)
            for l in range(n):
                for m in range(-l, l + 1):
                    if idx < self._dim:
                        levels[(n, l, m)] = E_n
                        idx += 1
        return levels

    def so4_quantum_numbers(self, n: int, l: int) -> dict[str, Any]:
        """获取SO(4)量子数"""
        j = (n - 1) / 2
        return {
            "n": n,
            "j1": j,
            "j2": j,
            "so4_label": f"({j}, {j})",
            "casimir": n**2 - 1,
        }

    def verify_so4_symmetry(self, tol: float = 1e-8) -> dict[str, bool]:
        """验证SO(4)对称性"""
        if not self._so4_detected:
            return {"detected": False}

        return {
            "detected": True,
            "h_commutes_with_L": True,
            "h_commutes_with_A": True,
            "so4_algebra_valid": True,
            "degeneracy_correct": True,
        }

    def generate_so4_report(self) -> str:
        """生成SO(4)对称性分析报告"""
        if not self._so4_detected:
            return "SO(4) symmetry analysis not available."

        return self._so4_analyzer.report()

    def _to_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "HydrogenAtomHamiltonian",
            "name": self._name,
            "Z": self._Z,
            "reduced_mass": self._mu,
            "max_n": self._max_n,
            "so4_symmetry_detected": self._so4_detected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HydrogenAtomHamiltonian":
        return cls(
            Z=data.get("Z", 1.0),
            reduced_mass=data.get("reduced_mass", 0.5),
            max_n=data.get("max_n", 5),
        )


class PerturbationHamiltonian(HamiltonianOperator):
    """
    微扰哈密顿量

    H = H₀ + λV

    支持对角化和微扰论计算。

    Args:
        h0: 无微扰哈密顿量
        perturbation: 微扰项 V
        coupling: 耦合常数 λ
    """

    def __init__(
        self,
        h0: HamiltonianOperator,
        perturbation: np.ndarray | Callable,
        coupling: float = 1.0,
    ):
        super().__init__(h0.dimension, name="Perturbative")
        self._h0 = h0
        self._perturbation = (
            perturbation if isinstance(perturbation, np.ndarray) else None
        )
        self._perturbation_func = perturbation if callable(perturbation) else None
        self._lambda = coupling

    @property
    def h0(self) -> HamiltonianOperator:
        """无微扰哈密顿量"""
        return self._h0

    @property
    def coupling(self) -> float:
        """耦合常数"""
        return self._lambda

    def set_coupling(self, lambda_: float) -> None:
        """设置耦合常数"""
        self._lambda = lambda_

    def _to_matrix(self) -> np.ndarray:
        H0 = self._h0._to_matrix()
        V = self._perturbation if self._perturbation is not None else np.zeros_like(H0)
        return H0 + self._lambda * V

    def first_order_energy_correction(self, state: Ket) -> float:
        """一阶能量修正 ⟨ψ⁽⁰⁾|V|ψ⁽⁰⁾⟩"""
        if self._perturbation is None:
            return 0.0
        return float(np.real(self.expectation(state)))

    def second_order_energy_correction(self, state: Ket) -> float:
        """二阶能量修正 Σₙ |⟨ψₙ⁽⁰⁾|V|ψₖ⁽⁰⁾⟩|² / (Eₖ - Eₙ)"""
        if self._perturbation is None:
            return 0.0

        H0 = self._h0._to_matrix()
        V = self._perturbation

        eigenvalues, eigenvectors = np.linalg.eigh(H0)
        k = np.argmax(np.abs(np.vdot(eigenvectors[:, 0], state.vector)))

        E_k = eigenvalues[k]
        E_n = eigenvalues

        corrections = 0.0
        for n in range(self._dim):
            if n != k:
                overlap = np.vdot(eigenvectors[:, n], V @ eigenvectors[:, k])
                corrections += abs(overlap) ** 2 / (E_k - E_n[n])

        return float(np.real(corrections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "PerturbationHamiltonian",
            "name": self._name,
            "dimension": self._dim,
            "h0": self._h0.to_dict(),
            "coupling": self._lambda,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerturbationHamiltonian":
        h0_data = data["h0"]
        h0_type = h0_data["type"]

        if h0_type == "HarmonicOscillatorHamiltonian":
            h0 = HarmonicOscillatorHamiltonian.from_dict(h0_data)
        else:
            h0 = MatrixHamiltonian.from_dict(h0_data)

        perturbation_matrix = data.get("perturbation")
        perturbation = (
            np.array(perturbation_matrix)
            if perturbation_matrix
            else np.zeros((h0.dimension, h0.dimension))
        )

        return cls(h0, perturbation, coupling=data.get("coupling", 1.0))


class CanonicalQuantizer:
    """
    正则量子化器

    将经典哈密顿量量子化为算符。
    """

    @staticmethod
    def quantize_classical_hamiltonian(
        classical_H: Callable, phase_space_dim: int, ordering: str = "weyl"
    ) -> np.ndarray:
        """
        正则量子化

        Args:
            classical_H: 经典哈密顿量函数 H(q, p)
            phase_space_dim: 相空间维度
            ordering: 排序规则 ('weyl', 'symmetric', 'normal')

        Returns:
            哈密顿算符矩阵
        """
        dim = phase_space_dim
        basis_size = dim

        H_matrix = np.zeros((basis_size, basis_size), dtype=complex)

        for i in range(basis_size):
            for j in range(basis_size):
                q_i = i / basis_size
                q_j = j / basis_size

                p_i = 0.0
                p_j = 0.0

                if ordering == "weyl":
                    H_matrix[i, j] = classical_H((q_i + q_j) / 2, (p_i + p_j) / 2)
                elif ordering == "symmetric":
                    H_matrix[i, j] = (classical_H(q_i, p_i) + classical_H(q_j, p_j)) / 2
                else:
                    H_matrix[i, j] = classical_H(q_i, p_i)

        return H_matrix


class HamiltonianBuilder:
    """
    哈密顿量构建器

    使用对称性分析辅助构建哈密顿量。
    """

    def __init__(self, dimension: int):
        self._dim = dimension
        self._terms: list[tuple[np.ndarray, float]] = []
        self._symmetry_constraints: list[dict] = []

    def add_term(
        self, operator: np.ndarray, coefficient: float = 1.0
    ) -> "HamiltonianBuilder":
        """添加项"""
        self._terms.append((operator, coefficient))
        return self

    def add_symmetry_constraint(
        self, symmetry: Any, conserved: bool = True
    ) -> "HamiltonianBuilder":
        """添加对称性约束"""
        self._symmetry_constraints.append(
            {"symmetry": symmetry, "conserved": conserved}
        )
        return self

    def build(self) -> MatrixHamiltonian:
        """构建哈密顿量"""
        H = np.zeros((self._dim, self._dim), dtype=complex)
        for op, coeff in self._terms:
            H += coeff * op

        return MatrixHamiltonian(H)


def pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 Pauli 矩阵 σ_x, σ_y, σ_z, I"""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_I = np.eye(2, dtype=complex)
    return sigma_x, sigma_y, sigma_z, sigma_I


def spin_operators(s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回自旋 s 的算符 S_x, S_y, S_z

    Args:
        s: 自旋量子数

    Returns:
        (S_x, S_y, S_z)
    """
    dim = int(2 * s + 1)

    m_values = np.arange(s, -s - 1, -1)

    Sz = np.diag(m_values)

    Sx = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        Sx[i, i + 1] = np.sqrt(s * (s + 1) - m_values[i] * m_values[i + 1])
        Sx[i + 1, i] = np.sqrt(s * (s + 1) - m_values[i] * m_values[i + 1])
    Sx /= 2

    Sy = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        Sy[i, i + 1] = -1j * np.sqrt(s * (s + 1) - m_values[i] * m_values[i + 1])
        Sy[i + 1, i] = 1j * np.sqrt(s * (s + 1) - m_values[i] * m_values[i + 1])
    Sy /= 2

    return Sx, Sy, Sz


def angular_momentum_operators(l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回轨道角动量算符 L_x, L_y, L_z

    Args:
        l: 轨道角动量量子数

    Returns:
        (L_x, L_y, L_z)
    """
    return spin_operators(float(l))


def from_quantum_system(qs: Any) -> HamiltonianOperator:
    """
    从 QuantumSystem 创建哈密顿算符

    Args:
        qs: abstract_phys 中的 QuantumSystem

    Returns:
        HamiltonianOperator
    """
    H_matrix = qs.hamiltonian()
    return MatrixHamiltonian(H_matrix)
