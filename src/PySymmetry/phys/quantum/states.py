"""
量子态表示模块

提供量子态的抽象表示，包括：
- Ket: 右矢（态矢量）
- Bra: 左矢（对偶态矢量）
- StateVector: 态向量
- DensityMatrix: 密度矩阵

与 abstract_phys.HilbertSpace 集成以支持任意维度的量子态空间。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from PySymmetry.abstract_phys import HilbertSpace


class QuantumState(ABC):
    """
    量子态抽象基类

    定义量子态的基本接口。
    """

    @abstractmethod
    def inner_product(self, other: "QuantumState") -> complex:
        """内积 <self|other>"""
        pass

    @abstractmethod
    def norm(self) -> float:
        """态的范数"""
        pass

    @abstractmethod
    def normalize(self) -> "QuantumState":
        """归一化态"""
        pass

    @abstractmethod
    def copy(self) -> "QuantumState":
        """复制态"""
        pass

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """序列化为字典"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "QuantumState":
        """从字典反序列化"""
        pass


class Ket(QuantumState):
    """
    右矢（Ket）

    表示量子态的 Dirac 符号 |ψ⟩。

    Args:
        data: 态向量系数或标签
        basis: 基底标签（可选）
        hilbert_space: 关联的希尔伯特空间（可选）
    """

    def __init__(
        self,
        data: np.ndarray | str | int,
        basis: str | None = None,
        hilbert_space: Optional["HilbertSpace"] = None,
    ):
        if isinstance(data, np.ndarray):
            self._vector = data.astype(complex)
        elif isinstance(data, str):
            self._vector = self._label_to_vector(data)
        elif isinstance(data, int):
            dim = data
            vec = np.zeros(dim, dtype=complex)
            vec[0] = 1.0
            self._vector = vec
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        self._basis = basis
        self._hilbert_space = hilbert_space

    def _label_to_vector(self, label: str) -> np.ndarray:
        """将标签转换为向量"""
        if label == "0" or label.lower() == "zero":
            return np.array([1.0, 0.0], dtype=complex)
        elif label == "1" or label.lower() == "one":
            return np.array([0.0, 1.0], dtype=complex)
        elif label.startswith("|+"):
            s = float(label[2:])
            theta = np.arccos(1 / np.sqrt(1 + np.exp(-2 * s)))
            phi = 0.0
            vec = np.array(
                [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex
            )
            return vec
        elif label.startswith("|-"):
            s = float(label[2:])
            theta = np.arccos(-1 / np.sqrt(1 + np.exp(-2 * s)))
            phi = 0.0
            vec = np.array(
                [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex
            )
            return vec
        else:
            raise ValueError(f"未知标签: {label}")

    @property
    def vector(self) -> np.ndarray:
        """返回态向量"""
        return self._vector.copy()

    @property
    def dimension(self) -> int:
        """返回希尔伯特空间维度"""
        return len(self._vector)

    @property
    def hilbert_space(self) -> Optional["HilbertSpace"]:
        """返回关联的希尔伯特空间"""
        return self._hilbert_space

    def inner_product(self, bra: "Bra") -> complex:
        """内积 <ψ|φ> = ⟨ψ|φ⟩"""
        if not isinstance(bra, Bra):
            raise TypeError("右矢只能与左矢做内积")
        return complex(np.vdot(self._vector, bra.vector))

    def norm(self) -> float:
        """范数 ⟨ψ|ψ⟩^{1/2}"""
        return float(np.sqrt(np.abs(np.vdot(self._vector, self._vector))))

    def normalize(self) -> "Ket":
        """归一化态"""
        n = self.norm()
        if n > 1e-10:
            return Ket(self._vector / n, self._basis, self._hilbert_space)
        return self.copy()

    def copy(self) -> "Ket":
        """复制态"""
        return Ket(self._vector.copy(), self._basis, self._hilbert_space)

    def to_vector(self) -> np.ndarray:
        """转换为复向量"""
        return self._vector.copy()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "type": "Ket",
            "vector": self._vector.tolist(),
            "basis": self._basis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Ket":
        """从字典反序列化"""
        return cls(np.array(data["vector"], dtype=complex), basis=data.get("basis"))

    def __getitem__(self, index: int) -> complex:
        """获取第 index 个分量"""
        return self._vector[index]

    def __add__(self, other: "Ket") -> "Ket":
        """态的叠加"""
        if self.dimension != other.dimension:
            raise ValueError(f"维度不匹配: {self.dimension} vs {other.dimension}")
        return Ket(self._vector + other._vector, hilbert_space=self._hilbert_space)

    def __mul__(self, scalar: complex) -> "Ket":
        """标量乘法"""
        return Ket(self._vector * scalar, hilbert_space=self._hilbert_space)

    def __rmul__(self, scalar: complex) -> "Ket":
        """标量乘法 (左侧)"""
        return Ket(scalar * self._vector, hilbert_space=self._hilbert_space)

    def __neg__(self) -> "Ket":
        """取负"""
        return Ket(-self._vector, hilbert_space=self._hilbert_space)

    def __repr__(self) -> str:
        return f"Ket(dim={self.dimension}, norm={self.norm():.4f})"


class Bra(QuantumState):
    """
    左矢（Bra）

    表示量子态的 Dirac 符号 ⟨ψ|。
    是 Ket 的对偶空间元素。

    Args:
        data: 态向量系数或标签
        basis: 基底标签（可选）
        hilbert_space: 关联的希尔伯特空间（可选）
    """

    def __init__(
        self,
        data: np.ndarray | str | int | Ket,
        basis: str | None = None,
        hilbert_space: Optional["HilbertSpace"] = None,
    ):
        if isinstance(data, Ket):
            self._vector = np.conj(data.vector)
        elif isinstance(data, np.ndarray):
            self._vector = np.conj(data.astype(complex))
        elif isinstance(data, str):
            ket = Ket(data)
            self._vector = np.conj(ket.vector)
        elif isinstance(data, int):
            dim = data
            vec = np.zeros(dim, dtype=complex)
            vec[0] = 1.0
            self._vector = np.conj(vec)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        self._basis = basis
        self._hilbert_space = hilbert_space

    @property
    def vector(self) -> np.ndarray:
        """返回态向量（共轭转置）"""
        return self._vector.copy()

    @property
    def dimension(self) -> int:
        """返回希尔伯特空间维度"""
        return len(self._vector)

    @property
    def ket(self) -> Ket:
        """返回对应的右矢"""
        return Ket(np.conj(self._vector), self._basis, self._hilbert_space)

    def inner_product(self, ket: Ket) -> complex:
        """内积 ⟨ψ|φ⟩"""
        if not isinstance(ket, Ket):
            raise TypeError("左矢只能与右矢做内积")
        return complex(np.vdot(self._vector, ket.vector))

    def norm(self) -> float:
        """范数 ⟨ψ|ψ⟩^{1/2}"""
        return float(np.sqrt(np.abs(np.vdot(self._vector, self._vector))))

    def normalize(self) -> "Bra":
        """归一化态"""
        n = self.norm()
        if n > 1e-10:
            return Bra(self._vector / n, self._basis, self._hilbert_space)
        return self.copy()

    def copy(self) -> "Bra":
        """复制态"""
        return Bra(self._vector.copy(), self._basis, self._hilbert_space)

    def to_vector(self) -> np.ndarray:
        """转换为复向量（共轭）"""
        return self._vector.copy()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "type": "Bra",
            "vector": self._vector.tolist(),
            "basis": self._basis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Bra":
        """从字典反序列化"""
        return cls(np.array(data["vector"], dtype=complex), basis=data.get("basis"))

    def __getitem__(self, index: int) -> complex:
        """获取第 index 个分量（共轭）"""
        return self._vector[index]

    def __add__(self, other: "Bra") -> "Bra":
        """态的叠加"""
        if self.dimension != other.dimension:
            raise ValueError(f"维度不匹配: {self.dimension} vs {other.dimension}")
        return Bra(self._vector + other._vector, hilbert_space=self._hilbert_space)

    def __mul__(self, scalar: complex) -> "Bra":
        """标量乘法"""
        return Bra(self._vector * scalar, hilbert_space=self._hilbert_space)

    def __rmul__(self, scalar: complex) -> "Bra":
        """标量乘法 (左侧)"""
        return Bra(scalar * self._vector, hilbert_space=self._hilbert_space)

    def __neg__(self) -> "Bra":
        """取负"""
        return Bra(-self._vector, hilbert_space=self._hilbert_space)

    def __repr__(self) -> str:
        return f"Bra(dim={self.dimension}, norm={self.norm():.4f})"


class StateVector(QuantumState):
    """
    态向量

    封装 numpy 数组的量子态表示，支持更多操作。

    Args:
        data: 态向量数据
        labels: 标签列表（可选）
        hilbert_space: 关联的希尔伯特空间
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: list[str] | None = None,
        hilbert_space: Optional["HilbertSpace"] = None,
    ):
        if data.ndim != 1:
            raise ValueError("态向量必须是一维数组")
        self._data = data.astype(complex)
        self._labels = labels
        self._hilbert_space = hilbert_space

    @property
    def data(self) -> np.ndarray:
        """返回原始数据"""
        return self._data.copy()

    @property
    def dimension(self) -> int:
        """维度"""
        return len(self._data)

    @property
    def labels(self) -> list[str] | None:
        """标签"""
        return self._labels.copy() if self._labels else None

    @property
    def probabilities(self) -> np.ndarray:
        """测量概率分布"""
        return np.abs(self._data) ** 2

    @property
    def phases(self) -> np.ndarray:
        """相对相位"""
        return np.angle(self._data)

    def entropy(self) -> float:
        """von Neumann 熵（对于纯态为零）"""
        return 0.0

    def inner_product(self, other: "StateVector") -> complex:
        """内积"""
        if isinstance(other, Ket):
            return complex(np.vdot(self._data, other.vector))
        return complex(np.vdot(self._data, other._data))

    def norm(self) -> float:
        """范数"""
        return float(np.linalg.norm(self._data))

    def normalize(self) -> "StateVector":
        """归一化"""
        n = self.norm()
        if n > 1e-10:
            return StateVector(self._data / n, self._labels, self._hilbert_space)
        return self.copy()

    def copy(self) -> "StateVector":
        """复制"""
        return StateVector(self._data.copy(), self._labels, self._hilbert_space)

    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        return self._data.copy()

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "type": "StateVector",
            "data": self._data.tolist(),
            "labels": self._labels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateVector":
        """反序列化"""
        return cls(np.array(data["data"], dtype=complex), labels=data.get("labels"))

    def measure(self, basis: np.ndarray = None) -> tuple[int, float]:
        """
        在给定基底上测量

        Args:
            basis: 测量基底，默认为计算基底

        Returns:
            (测量结果索引, 概率)
        """
        probs = self.probabilities
        probs /= probs.sum()
        idx = np.random.choice(self.dimension, p=probs)
        return idx, probs[idx]

    def partial_trace(self, subsystems: list[int], dims: list[int]) -> "DensityMatrix":
        """
        对子系统求迹

        Args:
            subsystems: 要求迹的子系统索引
            dims: 各子系统维度

        Returns:
            约化密度矩阵
        """
        rho = DensityMatrix(self)
        return rho.partial_trace(subsystems, dims)

    def __repr__(self) -> str:
        return f"StateVector(dim={self.dimension})"


class DensityMatrix(QuantumState):
    """
    密度矩阵

    描述混合态的密度算符 ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|。

    Args:
        data: 密度矩阵数据或纯态向量
        is_pure: 是否为纯态
    """

    def __init__(self, data: np.ndarray | StateVector | Ket, is_pure: bool = False):
        if isinstance(data, (StateVector, Ket)):
            vec = data.to_vector() if isinstance(data, StateVector) else data.vector
            self._matrix = np.outer(vec, np.conj(vec))
            self._is_pure = True
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                self._matrix = np.outer(data, np.conj(data))
                self._is_pure = True
            elif data.ndim == 2:
                self._matrix = data.astype(complex)
                self._is_pure = is_pure
            else:
                raise ValueError("数据必须是一维向量或二维矩阵")
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        self._traced_out_dims: list[int] = []

    @property
    def matrix(self) -> np.ndarray:
        """返回密度矩阵"""
        return self._matrix.copy()

    @property
    def dimension(self) -> int:
        """希尔伯特空间维度"""
        return self._matrix.shape[0]

    @property
    def is_pure(self) -> bool:
        """是否为纯态"""
        return self._is_pure

    @property
    def purity(self) -> float:
        """纯度 Tr(ρ²)"""
        return float(np.real(np.trace(self._matrix @ self._matrix)))

    @property
    def probabilities(self) -> np.ndarray:
        """对角元素（测量概率）"""
        return np.real(np.diagonal(self._matrix))

    def entropy(self) -> float:
        """von Neumann 熵 S = -Tr(ρ log ρ)"""
        eigenvalues = np.linalg.eigvalsh(self._matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def inner_product(self, other: "DensityMatrix") -> complex:
        """内积 Tr(ρ₁ ρ₂)"""
        return complex(np.trace(self._matrix @ other._matrix))

    def norm(self) -> float:
        """Hilbert-Schmidt 范数"""
        return float(np.linalg.norm(self._matrix, "fro"))

    def normalize(self) -> "DensityMatrix":
        """归一化（迹为1）"""
        tr = np.trace(self._matrix)
        if abs(tr) > 1e-10:
            return DensityMatrix(self._matrix / tr)
        return self.copy()

    def copy(self) -> "DensityMatrix":
        """复制"""
        dm = DensityMatrix(self._matrix.copy(), self._is_pure)
        dm._traced_out_dims = self._traced_out_dims.copy()
        return dm

    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        return self._matrix.flatten()

    def to_dict(self) -> dict:
        """序列化"""
        return {
            "type": "DensityMatrix",
            "matrix": self._matrix.tolist(),
            "is_pure": self._is_pure,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DensityMatrix":
        """反序列化"""
        dm = cls(np.array(data["matrix"], dtype=complex))
        dm._is_pure = data.get("is_pure", False)
        return dm

    def expectation(self, operator: np.ndarray) -> complex:
        """算符期望值 Tr(ρ O)"""
        return complex(np.trace(self._matrix @ operator))

    def variance(self, operator: np.ndarray) -> float:
        """算符方差 Var(O) = Tr(ρ O²) - Tr(ρ O)²"""
        E = self.expectation(operator)
        E2 = self.expectation(operator @ operator)
        return float(np.real(E2 - E**2))

    def partial_trace(self, subsystems: list[int], dims: list[int]) -> "DensityMatrix":
        """
        对子系统求迹

        使用合并-分割方法计算约化密度矩阵。

        Args:
            subsystems: 要求迹的子系统索引
            dims: 各子系统维度列表

        Returns:
            约化密度矩阵
        """
        if len(dims) == 0:
            return self.copy()

        total_dim = np.prod(dims)
        if total_dim != self.dimension:
            raise ValueError(f"维度不匹配: {total_dim} vs {self.dimension}")

        rho = self._matrix.reshape(*dims, *dims)
        kept_dims = [i for i in range(len(dims)) if i not in subsystems]

        for idx in subsystems:
            rho = np.trace(rho, axis1=idx, axis2=idx + len(dims))

        new_dims = [dims[i] for i in kept_dims]
        return DensityMatrix(rho.reshape(np.prod(new_dims), np.prod(new_dims)))

    def fidelity(self, other: "DensityMatrix") -> float:
        """
        与另一个密度矩阵的保真度

        F(ρ, σ) = (Tr √(√ρ σ √ρ))²
        """
        if self.dimension != other.dimension:
            return 0.0

        epsilon = 1e-10

        rho = self._matrix + epsilon * np.eye(self.dimension, dtype=complex)
        sigma = other._matrix + epsilon * np.eye(other.dimension, dtype=complex)

        sqrt_rho = self._sqrt_matrix(rho)

        product = sqrt_rho @ sigma @ sqrt_rho
        sqrt_product = self._sqrt_matrix(product)

        eigenvalues = np.linalg.eigvalsh(sqrt_product)
        return float(np.sum(np.sqrt(np.maximum(eigenvalues, 0))) ** 2)

    def _sqrt_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """计算矩阵的平方根"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    def __repr__(self) -> str:
        purity = self.purity
        return f"DensityMatrix(dim={self.dimension}, pure={self._is_pure}, purity={purity:.4f})"


def basis_state(index: int, dimension: int) -> Ket:
    """创建计算基底 |n⟩"""
    vec = np.zeros(dimension, dtype=complex)
    vec[index] = 1.0
    return Ket(vec)


def bell_state(index: int = 0) -> Ket:
    """
    创建 Bell 态

    Args:
        index: 0-3，分别对应 |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩
    """
    states = [
        np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
        np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
        np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
    ]
    return Ket(states[index])


def w_state(n: int) -> Ket:
    """创建 W 态（n qubits）"""
    vec = np.zeros(2**n, dtype=complex)
    for i in range(n):
        idx = 1 << i
        vec[idx] = 1.0 / np.sqrt(n)
    return Ket(vec)


def ghz_state(n: int) -> Ket:
    """创建 GHZ 态（n qubits）"""
    vec = np.zeros(2**n, dtype=complex)
    vec[0] = 1.0 / np.sqrt(2)
    vec[-1] = 1.0 / np.sqrt(2)
    return Ket(vec)


def tensor_product(*states: Ket) -> Ket:
    """计算态的张量积 |ψ⟩ ⊗ |φ⟩"""
    result = states[0].vector
    for state in states[1:]:
        result = np.kron(result, state.vector)
    return Ket(result)


def superposition(*states: tuple[Ket, complex]) -> Ket:
    """
    创建态的叠加

    Args:
        *states: (Ket, coefficient) 元组列表

    Returns:
        归一化的叠加态
    """
    result = np.zeros(states[0][0].dimension, dtype=complex)
    for ket, coeff in states:
        result += coeff * ket.vector
    return Ket(result).normalize()
