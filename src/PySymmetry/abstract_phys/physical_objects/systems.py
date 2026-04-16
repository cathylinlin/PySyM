"""
物理系统实现
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from PySymmetry.core.group_theory.product_group import DirectProductGroup

from ..symmetry_environments import PhysicalSymmetry, SymmetryType
from ..symmetry_operations.base import SymmetryOperation
from .abstract_physical_objects import PhysicalSpace


class PhysicalSystem(ABC):
    """物理系统抽象基类"""

    def __init__(self, name: str = "PhysicalSystem"):
        self._name = name
        self._symmetries = []
        self._parameters: dict[str, Any] = {}
        self._constraints: list[Callable] = []

    @property
    def name(self) -> str:
        """系统名称"""
        return self._name

    @property
    @abstractmethod
    def degrees_of_freedom(self) -> int:
        """自由度数目"""
        pass

    @property
    @abstractmethod
    def configuration_space(self) -> PhysicalSpace:
        """构型空间"""
        pass

    @property
    @abstractmethod
    def phase_space(self) -> PhysicalSpace:
        """相空间"""
        pass

    def add_symmetry(self, symmetry: PhysicalSymmetry):
        """添加对称性"""
        self._symmetries.append(symmetry)

    def remove_symmetry(self, symmetry_type: SymmetryType):
        """移除对称性"""
        self._symmetries = [s for s in self._symmetries if s.type != symmetry_type]

    def get_symmetries(self) -> list[PhysicalSymmetry]:
        """获取所有对称性"""
        return self._symmetries.copy()

    def has_symmetry(self, symmetry_type: SymmetryType) -> bool:
        """检查是否具有某类对称性"""
        return any(s.type == symmetry_type for s in self._symmetries)

    @abstractmethod
    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查系统在某对称操作下是否不变"""
        pass

    def symmetry_group(self):
        """获取完整对称群"""
        if len(self._symmetries) == 0:
            return None
        elif len(self._symmetries) == 1:
            return self._symmetries[0].group
        else:
            groups = [s.group for s in self._symmetries]
            return DirectProductGroup(*groups)

    @abstractmethod
    def equations_of_motion(self, *args, **kwargs) -> Any:
        """运动方程"""
        pass

    @abstractmethod
    def energy(self, *args, **kwargs) -> float:
        """系统能量"""
        pass

    def set_parameter(self, name: str, value: Any):
        """设置系统参数"""
        self._parameters[name] = value

    def get_parameter(self, name: str) -> Any | None:
        """获取系统参数"""
        return self._parameters.get(name)

    def add_constraint(self, constraint: Callable):
        """添加约束"""
        self._constraints.append(constraint)

    def apply_constraints(self, *args, **kwargs) -> bool:
        """应用约束"""
        return all(constraint(*args, **kwargs) for constraint in self._constraints)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystem":
        """从字典反序列化"""
        pass

    @abstractmethod
    def add_particle(self, particle):
        """添加粒子到系统"""
        pass

    @abstractmethod
    def remove_particle(self, particle):
        """从系统中移除粒子"""
        pass

    @abstractmethod
    def get_particles(self):
        """获取系统中的所有粒子"""
        pass

    @abstractmethod
    def get_total_energy(self):
        """获取系统总能量"""
        pass

    @abstractmethod
    def get_total_momentum(self):
        """获取系统总动量"""
        pass

    @abstractmethod
    def evolve(self, dt):
        """演化系统状态"""
        pass


class ClassicalSystem(PhysicalSystem):
    """经典物理系统"""

    def __init__(self, interactions=None, name: str = "ClassicalSystem"):
        super().__init__(name)
        self._particles = []
        self._interactions = interactions if interactions is not None else []

    @property
    def degrees_of_freedom(self) -> int:
        """自由度数目"""
        return len(self._particles) * 3  # 每个粒子3个自由度

    @property
    def configuration_space(self) -> PhysicalSpace:
        """构型空间"""
        from .state import EuclideanSpace

        return EuclideanSpace(self.degrees_of_freedom)

    @property
    def phase_space(self) -> PhysicalSpace:
        """相空间"""
        from .state import SymplecticSpace

        return SymplecticSpace(len(self._particles) * 3)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查系统在某对称操作下是否不变"""
        # 简化实现：检查所有粒子变换后能量是否不变
        original_energy = self.get_total_energy()

        # 复制系统并变换所有粒子
        import copy

        transformed_system = copy.deepcopy(self)
        for i, particle in enumerate(transformed_system._particles):
            transformed_system._particles[i] = operation.act_on(particle)

        transformed_energy = transformed_system.get_total_energy()
        return np.isclose(original_energy, transformed_energy, rtol=1e-6)

    def equations_of_motion(self, *args, **kwargs) -> Any:
        """运动方程"""
        # 返回每个粒子的加速度
        accelerations = []
        for particle in self._particles:
            if hasattr(particle, "velocity"):
                total_force = np.zeros_like(particle.velocity)
                for other_particle in self._particles:
                    if particle != other_particle:
                        for interaction in self._interactions:
                            total_force += interaction.calculate_force(
                                particle, other_particle
                            )
                acceleration = total_force / particle.get_mass()
                accelerations.append(acceleration)
            else:
                accelerations.append(np.zeros(3))
        return accelerations

    def energy(self, *args, **kwargs) -> float:
        """系统能量"""
        return self.get_total_energy()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self._name,
            "particles": [
                {
                    "mass": p.get_mass(),
                    "charge": p.get_charge(),
                    "spin": p.get_spin(),
                    "position": p.position.tolist(),
                    "velocity": p.velocity.tolist(),
                }
                for p in self._particles
            ],
            "interactions": [type(inter).__name__ for inter in self._interactions],
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystem":
        """从字典反序列化"""
        name = data.get("name", "ClassicalSystem")
        system = cls(name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复对称性（需要根据对称性类型字符串创建实际的对称性对象）
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                # 这里需要根据 symmetry_name 创建对应的 PhysicalSymmetry 对象
                pass

        return system

    def add_particle(self, particle):
        self._particles.append(particle)

    def remove_particle(self, particle):
        if particle in self._particles:
            self._particles.remove(particle)

    def get_particles(self):
        return self._particles.copy()

    def get_total_energy(self):
        kinetic_energy = 0.0
        potential_energy = 0.0

        for i, particle in enumerate(self._particles):
            if hasattr(particle, "velocity"):
                v = particle.velocity
                m = particle.get_mass()
                kinetic_energy += 0.5 * m * np.sum(v**2)

            for j in range(i + 1, len(self._particles)):
                for interaction in self._interactions:
                    potential_energy += interaction.calculate_potential(
                        particle, self._particles[j]
                    )

        return kinetic_energy + potential_energy

    def get_total_momentum(self):
        total_momentum = np.zeros(3)
        for particle in self._particles:
            if hasattr(particle, "velocity"):
                total_momentum += particle.get_mass() * particle.velocity
        return total_momentum

    def evolve(self, dt):
        if dt < 0:
            raise ValueError("时间步长不能为负数")

        for particle in self._particles:
            if hasattr(particle, "velocity") and hasattr(particle, "position"):
                total_force = np.zeros_like(particle.velocity)
                for other_particle in self._particles:
                    if particle != other_particle:
                        for interaction in self._interactions:
                            total_force += interaction.calculate_force(
                                particle, other_particle
                            )

                mass = particle.get_mass()
                if mass <= 0:
                    raise ValueError(f"粒子质量必须为正数，当前值: {mass}")

                acceleration = total_force / mass
                particle.velocity += acceleration * dt
                particle.position += particle.velocity * dt


class QuantumSystem(PhysicalSystem):
    """
    量子系统

    由哈密顿算符定义的量子系统
    """

    def __init__(
        self,
        hamiltonian_operator: Any,  # 算符对象
        hilbert_space_dim: int,
        name: str = "QuantumSystem",
    ):
        super().__init__(name)
        self._H = hamiltonian_operator
        self._dim = hilbert_space_dim

    @property
    def degrees_of_freedom(self) -> int:
        return self._dim

    @property
    def configuration_space(self) -> PhysicalSpace:
        """希尔伯特空间"""
        from .state import HilbertSpace

        return HilbertSpace(self._dim)

    @property
    def phase_space(self) -> PhysicalSpace:
        """量子系统没有经典相空间"""
        return None

    def hamiltonian(self):
        """哈密顿算符"""
        return self._H

    def equations_of_motion(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        薛定谔方程

        iℏ d|ψ>/dt = H|ψ>
        """
        # 薛定谔方程：dψ/dt = -i*H*ψ/ℏ
        hbar = 1.0  # 自然单位制
        return -1j * self._H @ state / hbar

    def energy(self, state: np.ndarray) -> float:
        """能量期望值 <ψ|H|ψ>"""
        return np.real(np.conj(state) @ self._H @ state)

    def ground_state(self) -> np.ndarray:
        """基态"""
        # 对角化哈密顿量
        eigenvalues, eigenvectors = np.linalg.eigh(self._H)
        return eigenvectors[:, 0]

    def energy_levels(self, n: int = None) -> np.ndarray:
        """能谱"""
        eigenvalues, _ = np.linalg.eigh(self._H)
        if n is None:
            return eigenvalues
        return eigenvalues[:n]

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查哈密顿量与对称算符对易"""
        # [H, U] = 0
        U = operation.representation_matrix(self._dim)
        commutator = self._H @ U - U @ self._H
        return np.allclose(commutator, 0, atol=1e-10)

    def symmetry_quantum_numbers(self) -> dict[str, list[Any]]:
        """
        根据对称性确定量子数

        例如：角动量 J, J_z，宇称等
        """
        quantum_numbers = {}
        for symmetry in self._symmetries:
            # 根据对称性类型确定量子数
            pass
        return quantum_numbers

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "dim": self._dim,
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantumSystem":
        """从字典反序列化"""
        name = data.get("name", "QuantumSystem")
        dim = data.get("dim", 2)
        # 这里需要创建一个哈密顿算符，实际应用中可能需要更复杂的逻辑
        # 暂时创建一个简单的对角哈密顿量
        hamiltonian = np.eye(dim)
        system = cls(hamiltonian_operator=hamiltonian, hilbert_space_dim=dim, name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复对称性（需要根据对称性类型字符串创建实际的对称性对象）
        # 这里简化处理，实际应用中需要更复杂的逻辑
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                # 这里需要根据 symmetry_name 创建对应的 PhysicalSymmetry 对象
                pass

        return system


class FieldSystem(PhysicalSystem):
    """场系统"""

    def __init__(self, fields=None, name: str = "FieldSystem"):
        super().__init__(name)
        self._fields = fields if fields is not None else []
        self._grid_points = []

    @property
    def degrees_of_freedom(self) -> int:
        """自由度数目"""
        return len(self._fields) * len(self._grid_points) if self._grid_points else 0

    @property
    def configuration_space(self) -> PhysicalSpace:
        """构型空间"""
        from .state import EuclideanSpace

        return EuclideanSpace(self.degrees_of_freedom)

    @property
    def phase_space(self) -> PhysicalSpace:
        """相空间"""
        from .state import SymplecticSpace

        return SymplecticSpace(self.degrees_of_freedom)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查系统在某对称操作下是否不变"""
        return True

    def equations_of_motion(self, *args, **kwargs) -> Any:
        """运动方程"""
        # 场的运动方程，如Klein-Gordon方程、Maxwell方程等
        return []

    def energy(self, *args, **kwargs) -> float:
        """系统能量"""
        return self.get_total_energy()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self._name,
            "fields": [type(field).__name__ for field in self._fields],
            "grid_points": [p.tolist() for p in self._grid_points],
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystem":
        """从字典反序列化"""
        name = data.get("name", "FieldSystem")
        system = cls(name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复网格点
        if "grid_points" in data:
            system._grid_points = [np.array(p) for p in data["grid_points"]]

        # 恢复对称性
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                pass

        return system

    def add_particle(self, field):
        self._fields.append(field)

    def remove_particle(self, field):
        if field in self._fields:
            self._fields.remove(field)

    def get_particles(self):
        return self._fields.copy()

    def get_total_energy(self):
        total_energy = 0.0
        for field in self._fields:
            for point in self._grid_points:
                total_energy += field.get_energy_density(point)
        return total_energy

    def get_total_momentum(self):
        return np.zeros(3)

    def evolve(self, dt):
        pass

    def set_grid(self, grid_points):
        if not isinstance(grid_points, list):
            raise TypeError("网格点必须是列表")
        if not grid_points:
            raise ValueError("网格点列表不能为空")

        # 检查网格点是否为有效的numpy数组
        for point in grid_points:
            if not isinstance(point, np.ndarray):
                raise TypeError("每个网格点必须是numpy数组")
            if len(point.shape) != 1:
                raise ValueError("每个网格点必须是一维数组")

        self._grid_points = grid_points

    def get_grid(self):
        return self._grid_points


class RelativisticSystem(PhysicalSystem):
    """相对论系统"""

    def __init__(self, spacetime, interactions=None, name: str = "RelativisticSystem"):
        super().__init__(name)
        self._spacetime = spacetime
        self._interactions = interactions if interactions is not None else []
        self._particles = []

    @property
    def degrees_of_freedom(self) -> int:
        """自由度数目"""
        return len(self._particles) * 4  # 每个粒子4个自由度（四维时空）

    @property
    def configuration_space(self) -> PhysicalSpace:
        """构型空间"""
        from .state import EuclideanSpace

        return EuclideanSpace(self.degrees_of_freedom)

    @property
    def phase_space(self) -> PhysicalSpace:
        """相空间"""
        from .state import SymplecticSpace

        return SymplecticSpace(self.degrees_of_freedom)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查系统在某对称操作下是否不变"""
        return True

    def equations_of_motion(self, *args, **kwargs) -> Any:
        """运动方程"""
        # 返回每个粒子的四维加速度
        accelerations = []
        for particle in self._particles:
            if hasattr(particle, "four_position") and hasattr(
                particle, "four_velocity"
            ):
                acceleration = self._spacetime.geodesic_equation(
                    particle.four_position, particle.four_velocity
                )
                accelerations.append(acceleration)
            else:
                accelerations.append(np.zeros(4))
        return accelerations

    def energy(self, *args, **kwargs) -> float:
        """系统能量"""
        return self.get_total_energy()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self._name,
            "spacetime": type(self._spacetime).__name__,
            "particles": [
                {"mass": p.get_mass(), "charge": p.get_charge(), "spin": p.get_spin()}
                for p in self._particles
            ],
            "interactions": [type(inter).__name__ for inter in self._interactions],
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystem":
        """从字典反序列化"""
        name = data.get("name", "RelativisticSystem")
        # 这里需要创建spacetime对象，暂时简化处理
        from .spacetime import MinkowskiSpacetime

        spacetime = MinkowskiSpacetime()
        system = cls(spacetime=spacetime, name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复对称性
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                pass

        return system

    def add_particle(self, particle):
        self._particles.append(particle)

    def remove_particle(self, particle):
        if particle in self._particles:
            self._particles.remove(particle)

    def get_particles(self):
        return self._particles.copy()

    def get_total_energy(self):
        total_energy = 0.0
        for particle in self._particles:
            if hasattr(particle, "four_velocity"):
                total_energy += particle.get_mass() * particle.four_velocity[0]
        return total_energy

    def get_total_momentum(self):
        total_momentum = np.zeros(3)
        for particle in self._particles:
            if hasattr(particle, "four_velocity"):
                total_momentum += particle.get_mass() * particle.four_velocity[1:]
        return total_momentum

    def evolve(self, dt):
        for particle in self._particles:
            if hasattr(particle, "four_position") and hasattr(
                particle, "four_velocity"
            ):
                acceleration = self._spacetime.geodesic_equation(
                    particle.four_position, particle.four_velocity
                )
                particle.four_velocity += acceleration * dt
                particle.four_position += particle.four_velocity * dt


class CompositeSystem(PhysicalSystem):
    """复合系统"""

    def __init__(self, subsystems=None, name: str = "CompositeSystem"):
        super().__init__(name)
        self._subsystems = subsystems if subsystems is not None else []

    @property
    def degrees_of_freedom(self) -> int:
        """自由度数目"""
        return sum(subsystem.degrees_of_freedom for subsystem in self._subsystems)

    @property
    def configuration_space(self) -> PhysicalSpace:
        """构型空间"""
        from .state import EuclideanSpace

        return EuclideanSpace(self.degrees_of_freedom)

    @property
    def phase_space(self) -> PhysicalSpace:
        """相空间"""
        from .state import SymplecticSpace

        return SymplecticSpace(self.degrees_of_freedom)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查系统在某对称操作下是否不变"""
        # 复合系统在对称操作下不变当且仅当所有子系统都不变
        return all(
            subsystem.is_invariant_under(operation) for subsystem in self._subsystems
        )

    def equations_of_motion(self, *args, **kwargs) -> Any:
        """运动方程"""
        # 返回所有子系统的运动方程
        return [
            subsystem.equations_of_motion(*args, **kwargs)
            for subsystem in self._subsystems
        ]

    def energy(self, *args, **kwargs) -> float:
        """系统能量"""
        return self.get_total_energy()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self._name,
            "subsystems": [subsystem.to_dict() for subsystem in self._subsystems],
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalSystem":
        """从字典反序列化"""
        name = data.get("name", "CompositeSystem")
        system = cls(name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复子系统（需要根据子系统类型字符串创建实际的子系统对象）
        if "subsystems" in data:
            for subsystem_data in data["subsystems"]:
                # 这里需要根据 subsystem_data 创建对应的子系统对象
                pass

        # 恢复对称性
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                pass

        return system

    def add_particle(self, particle):
        for subsystem in self._subsystems:
            subsystem.add_particle(particle)

    def remove_particle(self, particle):
        for subsystem in self._subsystems:
            subsystem.remove_particle(particle)

    def get_particles(self):
        all_particles = []
        for subsystem in self._subsystems:
            all_particles.extend(subsystem.get_particles())
        return all_particles

    def get_total_energy(self):
        return sum(subsystem.get_total_energy() for subsystem in self._subsystems)

    def get_total_momentum(self):
        total_momentum = np.zeros(3)
        for subsystem in self._subsystems:
            total_momentum += subsystem.get_total_momentum()
        return total_momentum

    def evolve(self, dt):
        for subsystem in self._subsystems:
            subsystem.evolve(dt)

    def add_subsystem(self, subsystem):
        self._subsystems.append(subsystem)

    def remove_subsystem(self, subsystem):
        if subsystem in self._subsystems:
            self._subsystems.remove(subsystem)

    def get_subsystems(self):
        return self._subsystems.copy()


class HamiltonianSystem(PhysicalSystem):
    """
    哈密顿系统

    由哈密顿量定义的系统
    """

    def __init__(
        self, hamiltonian: Callable, dim: int, name: str = "HamiltonianSystem"
    ):
        super().__init__(name)
        self._hamiltonian = hamiltonian
        self._dim = dim

    @property
    def degrees_of_freedom(self) -> int:
        """自由度"""
        return self._dim

    @property
    def configuration_space(self) -> PhysicalSpace:
        """构型空间：坐标空间"""
        from .state import EuclideanSpace

        return EuclideanSpace(self._dim)

    @property
    def phase_space(self) -> PhysicalSpace:
        """相空间：2n维"""
        from .state import SymplecticSpace

        return SymplecticSpace(self._dim)

    def hamiltonian(self, q: np.ndarray, p: np.ndarray, t: float = 0) -> float:
        """计算哈密顿量 H(q, p, t)"""
        return self._hamiltonian(q, p, t)

    def equations_of_motion(
        self, q: np.ndarray, p: np.ndarray, t: float = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        哈密顿方程

        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        """
        # 数值计算偏导数
        epsilon = 1e-8

        dq_dt = np.zeros_like(q)
        dp_dt = np.zeros_like(p)

        # ∂H/∂p
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += epsilon
            p_minus = p.copy()
            p_minus[i] -= epsilon
            dq_dt[i] = (
                self._hamiltonian(q, p_plus, t) - self._hamiltonian(q, p_minus, t)
            ) / (2 * epsilon)

        # -∂H/∂q
        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += epsilon
            q_minus = q.copy()
            q_minus[i] -= epsilon
            dp_dt[i] = -(
                self._hamiltonian(q_plus, p, t) - self._hamiltonian(q_minus, p, t)
            ) / (2 * epsilon)

        return dq_dt, dp_dt

    def energy(self, q: np.ndarray, p: np.ndarray, t: float = 0) -> float:
        """能量 = 哈密顿量"""
        return self._hamiltonian(q, p, t)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查哈密顿量是否在对称操作下不变"""
        # 需要具体实现
        # 简化版：数值验证
        test_points = [
            (np.random.randn(self._dim), np.random.randn(self._dim)) for _ in range(10)
        ]

        for q, p in test_points:
            H_original = self._hamiltonian(q, p, 0)
            q_transformed, p_transformed = operation.act_on_state(q, p)
            H_transformed = self._hamiltonian(q_transformed, p_transformed, 0)

            if not np.isclose(H_original, H_transformed, rtol=1e-6):
                return False

        return True

    def poisson_bracket(
        self, f: Callable, g: Callable, q: np.ndarray, p: np.ndarray
    ) -> float:
        """
        泊松括号 {f, g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
        """
        epsilon = 1e-8

        result = 0.0
        for i in range(len(q)):
            # ∂f/∂q_i
            q_plus = q.copy()
            q_plus[i] += epsilon
            q_minus = q.copy()
            q_minus[i] -= epsilon
            df_dq = (f(q_plus, p) - f(q_minus, p)) / (2 * epsilon)

            # ∂g/∂p_i
            p_plus = p.copy()
            p_plus[i] += epsilon
            p_minus = p.copy()
            p_minus[i] -= epsilon
            dg_dp = (g(q, p_plus) - g(q, p_minus)) / (2 * epsilon)

            # ∂f/∂p_i
            df_dp = (f(q, p_plus) - f(q, p_minus)) / (2 * epsilon)

            # ∂g/∂q_i
            dg_dq = (g(q_plus, p) - g(q_minus, p)) / (2 * epsilon)

            result += df_dq * dg_dp - df_dp * dg_dq

        return result

    def conserved_quantities(self) -> list[Callable]:
        """根据对称性找出守恒量"""
        conserved = []
        for symmetry in self._symmetries:
            # 利用诺特定理找到守恒量
            # 具体实现依赖于对称性类型
            pass
        return conserved

    def to_dict(self) -> dict[str, Any]:
        """序列化"""
        return {
            "name": self._name,
            "dim": self._dim,
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HamiltonianSystem":
        """反序列化"""
        name = data.get("name", "HamiltonianSystem")
        dim = data.get("dim", 1)

        # 创建一个简单的哈密顿量函数（实际应用中需要更复杂的逻辑）
        def simple_hamiltonian(q, p, t=0):
            return 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)  # 谐振子哈密顿量

        system = cls(hamiltonian=simple_hamiltonian, dim=dim, name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复对称性
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                pass

        return system


class LagrangianSystem(PhysicalSystem):
    """
    拉格朗日系统

    由拉格朗日量定义的系统
    """

    def __init__(self, lagrangian: Callable, dim: int, name: str = "LagrangianSystem"):
        super().__init__(name)
        self._lagrangian = lagrangian
        self._dim = dim

    @property
    def degrees_of_freedom(self) -> int:
        return self._dim

    @property
    def configuration_space(self) -> PhysicalSpace:
        from .state import EuclideanSpace

        return EuclideanSpace(self._dim)

    @property
    def phase_space(self) -> PhysicalSpace:
        """切丛"""
        from .state import TangentBundle

        return TangentBundle(self._dim)

    def lagrangian(self, q: np.ndarray, dq: np.ndarray, t: float = 0) -> float:
        """计算拉格朗日量 L(q, dq/dt, t)"""
        return self._lagrangian(q, dq, t)

    def equations_of_motion(
        self, q: np.ndarray, dq: np.ndarray, t: float = 0
    ) -> np.ndarray:
        """
        欧拉-拉格朗日方程

        d/dt(∂L/∂dq_i) - ∂L/∂q_i = 0

        返回：d^2q/dt^2（加速度）
        """
        epsilon = 1e-8

        # 计算 ∂L/∂q_i 和 ∂L/∂dq_i
        dL_dq = np.zeros(self._dim)
        dL_ddq = np.zeros(self._dim)

        for i in range(self._dim):
            # ∂L/∂q_i
            q_plus = q.copy()
            q_plus[i] += epsilon
            q_minus = q.copy()
            q_minus[i] -= epsilon
            dL_dq[i] = (
                self._lagrangian(q_plus, dq, t) - self._lagrangian(q_minus, dq, t)
            ) / (2 * epsilon)

            # ∂L/∂dq_i
            dq_plus = dq.copy()
            dq_plus[i] += epsilon
            dq_minus = dq.copy()
            dq_minus[i] -= epsilon
            dL_ddq[i] = (
                self._lagrangian(q, dq_plus, t) - self._lagrangian(q, dq_minus, t)
            ) / (2 * epsilon)

        # 计算质量矩阵 M_ij = ∂²L/∂dq_i∂dq_j
        M = np.zeros((self._dim, self._dim))
        for i in range(self._dim):
            for j in range(self._dim):
                dq_plus = dq.copy()
                dq_plus[j] += epsilon
                dq_minus = dq.copy()
                dq_minus[j] -= epsilon

                dq_plus[i] += epsilon
                dq_minus[i] += epsilon
                dL_ddq_plus_i = (
                    self._lagrangian(q, dq_plus, t) - self._lagrangian(q, dq_minus, t)
                ) / (2 * epsilon)

                dq_plus[i] -= 2 * epsilon
                dq_minus[i] -= 2 * epsilon
                dL_ddq_minus_i = (
                    self._lagrangian(q, dq_plus, t) - self._lagrangian(q, dq_minus, t)
                ) / (2 * epsilon)

                M[i, j] = (dL_ddq_plus_i - dL_ddq_minus_i) / (2 * epsilon)

        # 计算 d/dt(∂L/∂dq_i) - ∂L/∂q_i
        # 假设质量矩阵是常数，简化计算
        # 对于 L = T - V，T = 1/2 m dq²，V = V(q)
        # 则 M = m I，方程变为 m d²q/dt² = -∇V

        # 尝试求解线性方程组 M a = -dL_dq
        try:
            M_inv = np.linalg.inv(M)
            acceleration = -np.dot(M_inv, dL_dq)
        except np.linalg.LinAlgError:
            # 如果质量矩阵不可逆，使用简化方法
            acceleration = -dL_dq

        return acceleration

    def energy(self, q: np.ndarray, dq: np.ndarray, t: float = 0) -> float:
        """能量 = 哈密顿量（需要勒让德变换）"""
        # 计算共轭动量
        epsilon = 1e-8
        p = np.zeros(self._dim)

        for i in range(self._dim):
            dq_plus = dq.copy()
            dq_plus[i] += epsilon
            dq_minus = dq.copy()
            dq_minus[i] -= epsilon
            p[i] = (
                self._lagrangian(q, dq_plus, t) - self._lagrangian(q, dq_minus, t)
            ) / (2 * epsilon)

        # H = p*dq - L
        return np.dot(p, dq) - self._lagrangian(q, dq, t)

    def is_invariant_under(self, operation: SymmetryOperation) -> bool:
        """检查拉格朗日量是否在对称操作下不变"""
        # 类似于哈密顿系统
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "dim": self._dim,
            "symmetries": [s.type.name for s in self._symmetries],
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LagrangianSystem":
        """反序列化"""
        name = data.get("name", "LagrangianSystem")
        dim = data.get("dim", 1)

        # 创建一个简单的拉格朗日量函数（实际应用中需要更复杂的逻辑）
        def simple_lagrangian(q, dq, t=0):
            return 0.5 * np.sum(dq**2) - 0.5 * np.sum(q**2)  # 谐振子拉格朗日量

        system = cls(lagrangian=simple_lagrangian, dim=dim, name=name)

        # 恢复参数
        if "parameters" in data:
            for key, value in data["parameters"].items():
                system.set_parameter(key, value)

        # 恢复对称性
        if "symmetries" in data:
            for symmetry_name in data["symmetries"]:
                pass

        return system
