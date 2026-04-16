"""
场景模拟器模块

提供量子系统的场景模拟功能：
1. 粒子模拟 - 量子粒子的运动和相互作用
2. 场模拟 - 量子场的时空演化
3. 测量模拟 - 量子测量的统计特性
4. 退相干模拟 - 环境导致的退相干效应

与 abstract_phys 模块集成，支持粒子、场和系统的模拟。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    from PySymmetry.abstract_phys import (
        ClassicalSystem,
        FieldSystem,
        PhysicalSystem,
        QuantumSystem,
        ScalarField,
        SpinorField,
        VectorField,
        YangMillsField,
    )
except ImportError:
    PhysicalSystem = object
    ClassicalSystem = object
    QuantumSystem = object
    FieldSystem = object
    ScalarField = object
    VectorField = object
    SpinorField = object
    YangMillsField = object

from .hamiltonian import HamiltonianOperator, MatrixHamiltonian
from .solver import TimeEvolutionSolver
from .states import DensityMatrix, Ket


class Simulator(ABC):
    """
    模拟器抽象基类

    定义量子系统模拟的基本接口。
    """

    def __init__(self, name: str = "Simulator"):
        self._name = name
        self._time: float = 0.0
        self._results: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def time(self) -> float:
        return self._time

    @abstractmethod
    def step(self, dt: float) -> None:
        """执行一步模拟"""
        pass

    @abstractmethod
    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行完整模拟"""
        pass

    def get_results(self) -> dict[str, Any]:
        """获取模拟结果"""
        return self._results.copy()


class QuantumSimulator(Simulator):
    """
    量子系统模拟器

    模拟量子系统的动力学演化。

    Args:
        hamiltonian: 哈密顿算符
        initial_state: 初始态
    """

    def __init__(
        self,
        hamiltonian: HamiltonianOperator,
        initial_state: Ket,
        name: str = "QuantumSimulator",
    ):
        super().__init__(name)
        self._H = hamiltonian
        self._state = initial_state.copy()
        self._evolution_solver = TimeEvolutionSolver(hamiltonian)

        self._state_history: list[Ket] = [self._state]
        self._time_history: list[float] = [0.0]
        self._energy_history: list[float] = [
            float(self._H.expectation(self._state).real)
        ]

    @property
    def state(self) -> Ket:
        """当前态"""
        return self._state.copy()

    @property
    def state_history(self) -> list[Ket]:
        """态历史"""
        return self._state_history.copy()

    def step(self, dt: float) -> None:
        """执行一步时间演化"""
        states, times = self._evolution_solver.evolve(
            self._state, t0=self._time, num_steps=1
        )
        self._state = states[-1]
        self._time = times[-1]

        self._state_history.append(self._state)
        self._time_history.append(self._time)
        self._energy_history.append(float(self._H.expectation(self._state).real))

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行模拟"""
        num_steps = int(duration / dt)

        for _ in range(num_steps):
            self.step(dt)

        self._results = {
            "times": np.array(self._time_history),
            "final_state": self._state,
            "energy_history": np.array(self._energy_history),
            "final_energy": self._energy_history[-1],
        }

        return self._results


class MeasurementSimulator(Simulator):
    """
    量子测量模拟器

    模拟量子测量过程，包括：
    - 计算基底测量
    - 不对易算符的顺序测量
    - 弱测量

    Args:
        initial_state: 初始态
        measurement_basis: 测量基底
    """

    def __init__(
        self,
        initial_state: Ket | DensityMatrix,
        measurement_basis: np.ndarray | None = None,
        name: str = "MeasurementSimulator",
    ):
        super().__init__(name)
        self._state = initial_state
        self._basis = measurement_basis
        self._measurement_results: list[int] = []
        self._measurement_probs: list[np.ndarray] = []

        if measurement_basis is None:
            self._basis = np.eye(initial_state.dimension)

    def measure(self) -> tuple[int, float]:
        """
        执行一次测量

        Returns:
            (测量结果索引, 对应概率)
        """
        if isinstance(self._state, Ket):
            probs = np.abs(self._basis @ self._state.vector) ** 2
        else:
            basis_mat = (
                self._basis
                if self._basis is not None
                else np.eye(self._state.dimension)
            )
            probs = np.real(
                np.diagonal(basis_mat @ self._state.matrix @ basis_mat.conj().T)
            )

        probs = probs / probs.sum()

        result_idx = np.random.choice(len(probs), p=probs)

        self._measurement_results.append(result_idx)
        self._measurement_probs.append(probs)

        return result_idx, float(probs[result_idx])

    def ensemble_measure(self, num_samples: int) -> dict[str, Any]:
        """
        集合测量

        Args:
            num_samples: 样本数量

        Returns:
            测量统计结果
        """
        results = []
        for _ in range(num_samples):
            results.append(self.measure())

        result_indices = [r[0] for r in results]
        minlength = (
            len(self._basis) if self._basis is not None else len(results[0][0]) + 1
        )
        counts = np.bincount(result_indices, minlength=minlength)

        return {
            "counts": counts,
            "frequencies": counts / num_samples,
            "expected_probs": self._measurement_probs[-1]
            if self._measurement_probs
            else None,
        }

    def step(self, dt: float) -> None:
        """模拟步骤（测量不改变时间）"""
        pass

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行测量模拟"""
        return self.ensemble_measure(int(duration / dt))


class DecoherenceSimulator(Simulator):
    """
    退相干模拟器

    模拟环境导致的退相干过程。

    Args:
        density_matrix: 初始密度矩阵
        decoherence_rate: 退相干率
        dephasing_basis: 退相干基底
    """

    def __init__(
        self,
        initial_state: DensityMatrix,
        decoherence_rate: float = 0.1,
        dephasing_basis: np.ndarray | None = None,
        name: str = "DecoherenceSimulator",
    ):
        super().__init__(name)
        self._rho = initial_state.copy()
        self._gamma = decoherence_rate
        self._basis = dephasing_basis or np.eye(initial_state.dimension)

        self._purity_history: list[float] = [self._rho.purity]
        self._entropy_history: list[float] = [self._rho.entropy()]

    @property
    def state(self) -> DensityMatrix:
        """当前态"""
        return self._rho.copy()

    def step(self, dt: float) -> None:
        """执行退相干步骤（相位阻尼）"""
        gamma = self._gamma
        decay = np.exp(-gamma * dt)
        rho = self._rho.matrix.copy()

        eigvals, eigvecs = np.linalg.eigh(rho)

        rho_diag = eigvecs.conj().T @ rho @ eigvecs

        n = len(eigvals)
        for i in range(n):
            for j in range(n):
                if i != j:
                    rho_diag[i, j] *= decay

        self._rho._matrix = eigvecs @ rho_diag @ eigvecs.conj().T

        trace = np.trace(self._rho._matrix)
        if abs(trace) > 1e-10:
            self._rho._matrix /= trace

        self._time += dt

        self._purity_history.append(self._rho.purity)
        self._entropy_history.append(self._rho.entropy())

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行退相干模拟"""
        num_steps = int(duration / dt)

        for _ in range(num_steps):
            self.step(dt)

        self._results = {
            "times": np.linspace(0, duration, len(self._purity_history)),
            "purity_history": np.array(self._purity_history),
            "entropy_history": np.array(self._entropy_history),
            "final_state": self._rho,
        }

        return self._results


class ParticleFieldSimulator(Simulator):
    """
    粒子-场相互作用模拟器

    模拟带电粒子与电磁场的相互作用。

    Args:
        particles: 粒子列表（位置、动量）
        field: 电磁场
        interaction_strength: 耦合强度
    """

    def __init__(
        self,
        particle_positions: np.ndarray,
        particle_momenta: np.ndarray,
        charges: np.ndarray,
        masses: np.ndarray,
        field: Any | None = None,
        interaction_strength: float = 1.0,
        name: str = "ParticleFieldSimulator",
    ):
        super().__init__(name)
        self._positions = particle_positions.astype(float)
        self._momenta = particle_momenta.astype(float)
        self._charges = charges.astype(float)
        self._masses = masses.astype(float)
        self._field = field
        self._g = interaction_strength

        self._num_particles = len(particle_positions)
        self._position_history: list[np.ndarray] = [self._positions.copy()]
        self._momentum_history: list[np.ndarray] = [self._momenta.copy()]

    def _compute_force(self) -> np.ndarray:
        """计算粒子受力"""
        forces = np.zeros_like(self._positions)

        if self._field is not None:
            for i in range(self._num_particles):
                E = self._field.get_electric_field(self._positions[i])
                forces[i] += self._charges[i] * E

        return forces

    def step(self, dt: float) -> None:
        """执行一步模拟"""
        forces = self._compute_force()

        old_momenta = self._momenta.copy()

        accelerations = forces / self._masses[:, np.newaxis]

        self._momenta += accelerations * dt

        avg_momenta = (old_momenta + self._momenta) / 2
        self._positions += avg_momenta / self._masses[:, np.newaxis] * dt

        self._time += dt

        self._position_history.append(self._positions.copy())
        self._momentum_history.append(self._momenta.copy())

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行模拟"""
        num_steps = int(duration / dt)

        for _ in range(num_steps):
            self.step(dt)

        self._results = {
            "times": np.linspace(0, duration, len(self._position_history)),
            "position_history": np.array(self._position_history),
            "momentum_history": np.array(self._momentum_history),
            "final_positions": self._positions,
            "final_momenta": self._momenta,
        }

        return self._results


class ScatteringSimulator(Simulator):
    """
    散射模拟器

    模拟粒子散射过程。

    Args:
        initial_state: 初始态
        interaction_potential: 相互作用势
        impact_parameter_range: 碰撞参数范围
    """

    def __init__(
        self,
        initial_wavefunction: np.ndarray,
        interaction_potential: Callable[[np.ndarray], float],
        impact_parameter: float = 0.0,
        name: str = "ScatteringSimulator",
    ):
        super().__init__(name)
        self._psi0 = initial_wavefunction.copy()
        self._V = interaction_potential
        self._b = impact_parameter

        self._psi_history: list[np.ndarray] = [self._psi0]
        self._transmission_probability: float | None = None
        self._reflection_probability: float | None = None

    @property
    def wavefunction(self) -> np.ndarray:
        """当前波函数"""
        return self._psi_history[-1].copy()

    def compute_observables(self) -> dict[str, float]:
        """计算散射可观测量"""
        psi = self.wavefunction

        probability = np.sum(np.abs(psi) ** 2)

        reflection = np.sum(np.abs(psi[: len(psi) // 3]) ** 2)
        transmission = np.sum(np.abs(psi[2 * len(psi) // 3 :]) ** 2)

        self._reflection_probability = (
            float(reflection / probability) if probability > 0 else 0.0
        )
        self._transmission_probability = (
            float(transmission / probability) if probability > 0 else 0.0
        )

        return {
            "reflection_probability": self._reflection_probability,
            "transmission_probability": self._transmission_probability,
        }

    def step(self, dt: float) -> None:
        """执行一步模拟"""
        pass

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行模拟"""
        self._results = self.compute_observables()
        return self._results


class SpinChainSimulator(Simulator):
    """
    自旋链模拟器

    模拟一维自旋链系统。

    Args:
        num_sites: 格点数
        couplings: 耦合常数
        external_field: 外磁场
        initial_state: 初始态
    """

    def __init__(
        self,
        num_sites: int,
        couplings: dict[str, float],
        external_field: float = 0.0,
        initial_state: Ket | None = None,
        name: str = "SpinChainSimulator",
    ):
        super().__init__(name)
        self._n = num_sites
        self._J = couplings
        self._h = external_field
        self._dim = 2**num_sites

        if initial_state is None:
            self._state = Ket(np.zeros(self._dim, dtype=complex))
            self._state._vector[0] = 1.0
        else:
            self._state = initial_state

        self._build_hamiltonian()
        self._solver = TimeEvolutionSolver(self._H)

        self._magnetization_history: list[float] = []
        self._energy_history: list[float] = []

    def _build_hamiltonian(self) -> None:
        """构建自旋链哈密顿量"""
        from .hamiltonian import pauli_matrices

        sx, sy, sz, I = pauli_matrices()
        sx /= 2
        sy /= 2
        sz /= 2

        H = np.zeros((self._dim, self._dim), dtype=complex)

        for i in range(self._n - 1):
            term = np.zeros((self._dim, self._dim), dtype=complex)

            for state in range(self._dim):
                bit_state = format(state, f"0{self._n}b")
                bits = [int(b) for b in bit_state]

                si = bits[i]
                sip1 = bits[i + 1]

                term[state, state] = self._J.get("z", 0.0) * (si - 0.5) * (sip1 - 0.5)
                term[state, state] += self._J.get("x", 0.0) * 0.25 if si != sip1 else 0
                term[state, state] += self._J.get("y", 0.0) * 0.25 if si != sip1 else 0

            for j in range(self._n - 1):
                if j == i:
                    continue

                for state in range(self._dim):
                    new_state = state

                    bit_i = (state >> (self._n - 1 - i)) & 1
                    bit_j = (state >> (self._n - 1 - j)) & 1

                    if bit_i != bit_j:
                        flipped = (
                            state ^ (1 << (self._n - 1 - i)) ^ (1 << (self._n - 1 - j))
                        )
                        new_state = flipped

                    term[new_state, state] += self._J.get("x", 0.0) * 0.5
                    term[new_state, state] += self._J.get("y", 0.0) * 0.5 * 1j
                    term[new_state, state] += self._J.get("z", 0.0) * 0.25

            H += term

        if self._h != 0:
            for state in range(self._dim):
                bit_state = format(state, f"0{self._n}b")
                m_total = sum((int(b) - 0.5) for b in bit_state)
                H[state, state] -= self._h * m_total

        self._H = MatrixHamiltonian(H)

    def compute_magnetization(self) -> float:
        """计算总磁化"""
        sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        I = np.eye(2)

        total_mag = 0.0
        for i in range(self._n):
            term = 1
            for k in range(self._n):
                if k == i:
                    term = np.kron(term, sz)
                else:
                    term = np.kron(term, I)
            total_mag += float(
                np.vdot(self._state.vector, term @ self._state.vector).real
            )

        return total_mag / self._n

    def step(self, dt: float) -> None:
        """执行一步模拟"""
        states, times = self._solver.evolve(self._state, t0=self._time, num_steps=1)
        self._state = states[-1]
        self._time = times[-1]

        self._magnetization_history.append(self.compute_magnetization())
        self._energy_history.append(float(self._H.expectation(self._state).real))

    def run(self, duration: float, dt: float) -> dict[str, Any]:
        """运行模拟"""
        num_steps = int(duration / dt)

        for _ in range(num_steps):
            self.step(dt)

        self._results = {
            "times": np.linspace(0, duration, len(self._magnetization_history)),
            "magnetization": np.array(self._magnetization_history),
            "energy": np.array(self._energy_history),
            "final_state": self._state,
        }

        return self._results


def create_simulation(system_type: str, **kwargs) -> Simulator:
    """
    创建模拟器便捷函数

    Args:
        system_type: 系统类型 ('quantum', 'measurement', 'decoherence', 'spin_chain')
        **kwargs: 其他参数

    Returns:
        Simulator 实例
    """
    if system_type == "quantum":
        from .hamiltonian import HarmonicOscillatorHamiltonian

        H = kwargs.get("hamiltonian", HarmonicOscillatorHamiltonian())
        initial = kwargs.get("initial_state", Ket(np.array([1.0, 0.0], dtype=complex)))
        return QuantumSimulator(H, initial)

    elif system_type == "decoherence":
        rho = kwargs.get("density_matrix")
        rate = kwargs.get("decoherence_rate", 0.1)
        if rho is None:
            from .states import DensityMatrix

            rho = DensityMatrix(np.array([1.0, 0.0], dtype=complex))
        return DecoherenceSimulator(rho, rate)

    elif system_type == "spin_chain":
        n = kwargs.get("num_sites", 4)
        couplings = kwargs.get("couplings", {"z": 1.0})
        h = kwargs.get("external_field", 0.0)
        return SpinChainSimulator(n, couplings, h)

    else:
        raise ValueError(f"未知系统类型: {system_type}")
