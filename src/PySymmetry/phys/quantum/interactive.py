"""
通用量子模拟器框架

提供灵活的量子系统配置和模拟能力。

核心功能:
1. SceneBuilder - 场景构建器，支持任意粒子和势场配置
2. InteractiveSimulator - 交互式模拟器
3. Visualizer - 结果可视化

使用示例:
    from PySymmetry.phys.quantum.interactive import SceneBuilder, simulate
    
    # 创建氢原子
    scene = SceneBuilder().add_electron(position=[0,0,0]).add_potential('coulomb', center=[0,0,0], strength=-1).build()
    result = simulate(scene)
    result.plot()
    
    # 自定义势阱
    scene = SceneBuilder().set_potential(lambda r: 0 if 0<r.x<10 else 1e10).build()
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
import numpy as np

try:
    from PySymmetry.abstract_phys import (
        ElementaryParticle,
        Field,
        ScalarField,
        VectorField,
        SpinorField,
    )
except ImportError:
    ElementaryParticle = object
    Field = object
    ScalarField = object
    VectorField = object
    SpinorField = object


@dataclass
class Particle:
    """粒子配置"""
    name: str
    mass: float
    charge: float
    spin: float
    position: np.ndarray
    momentum: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.momentum is not None:
            self.momentum = np.asarray(self.momentum, dtype=float)


@dataclass
class Potential:
    """势能配置"""
    name: str
    potential_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    function: Optional[Callable] = None
    
    @classmethod
    def coulomb(cls, center: np.ndarray, strength: float, Z: float = 1.0):
        """库仑势 V(r) = -Z*strength/r"""
        def V(x):
            r = np.linalg.norm(x - center)
            if r < 1e-10:
                return -1e10
            return -Z * strength / r
        return cls(name='Coulomb', potential_type='coulomb', 
                   parameters={'center': center, 'strength': strength, 'Z': Z},
                   function=V)
    
    @classmethod
    def harmonic(cls, center: np.ndarray, k: float):
        """谐振子势 V(r) = 1/2 * k * r^2"""
        def V(x):
            r_sq = np.sum((x - center)**2)
            return 0.5 * k * r_sq
        return cls(name='Harmonic', potential_type='harmonic',
                   parameters={'center': center, 'k': k},
                   function=V)
    
    @classmethod
    def square_well(cls, center: np.ndarray, radius: float, depth: float):
        """方势阱"""
        def V(x):
            r = np.linalg.norm(x - center)
            return -depth if r < radius else 0.0
        return cls(name='SquareWell', potential_type='square_well',
                   parameters={'center': center, 'radius': radius, 'depth': depth},
                   function=V)
    
    @classmethod
    def step(cls, position: float, height: float):
        """阶梯势"""
        def V(x):
            x_val = x[0] if hasattr(x, '__len__') and len(x) > 0 else x
            return height if x_val > position else 0.0
        return cls(name='Step', potential_type='step',
                   parameters={'position': position, 'height': height},
                   function=V)
    
    @classmethod
    def custom(cls, func: Callable, name: str = 'Custom'):
        """自定义势能"""
        return cls(name=name, potential_type='custom', function=func)
    
    def evaluate(self, x: np.ndarray) -> float:
        """计算势能值"""
        if self.function is not None:
            return self.function(x)
        return 0.0


@dataclass 
class QuantumScene:
    """量子场景配置"""
    name: str
    particles: List[Particle] = field(default_factory=list)
    potentials: List[Potential] = field(default_factory=list)
    spatial_range: Tuple[float, float] = (-10.0, 10.0)
    dimension: int = 1
    grid_points: int = 100
    boundary_condition: str = 'infinite'
    spin_coupling: bool = False
    external_field: Optional[Dict[str, float]] = None


class SceneBuilder:
    """
    场景构建器
    
    链式调用构建任意量子系统。
    
    使用示例:
        scene = (SceneBuilder("我的模拟")
                 .add_particle('electron', mass=1.0, charge=-1, spin=0.5, position=[0,0,0])
                 .add_potential(Potential.coulomb(center=[0,0,0], strength=1.0))
                 .set_spatial_range(-5, 5)
                 .set_grid_points(200)
                 .build())
    """
    
    def __init__(self, name: str = "QuantumScene"):
        self._name = name
        self._particles: List[Particle] = []
        self._potentials: List[Potential] = []
        self._spatial_range = (-10.0, 10.0)
        self._dimension = 1
        self._grid_points = 100
        self._boundary = 'infinite'
        self._spin_coupling = False
        self._external_field = None
    
    def add_particle(self, 
                     name: str,
                     mass: float,
                     charge: float,
                     spin: float = 0.5,
                     position: List[float] = None,
                     momentum: List[float] = None) -> 'SceneBuilder':
        """添加粒子"""
        if position is None:
            position = [0.0] * self._dimension
        if momentum is None:
            momentum = [0.0] * self._dimension
        
        particle = Particle(
            name=name,
            mass=mass,
            charge=charge,
            spin=spin,
            position=np.array(position),
            momentum=np.array(momentum)
        )
        self._particles.append(particle)
        return self
    
    def add_electron(self, position: List[float] = None, momentum: List[float] = None) -> 'SceneBuilder':
        """添加电子 (便捷方法)"""
        return self.add_particle(
            name=f"electron_{len([p for p in self._particles if 'electron' in p.name])}",
            mass=1.0,
            charge=-1.0,
            spin=0.5,
            position=position,
            momentum=momentum
        )
    
    def add_proton(self, position: List[float] = None, momentum: List[float] = None) -> 'SceneBuilder':
        """添加质子 (便捷方法)"""
        return self.add_particle(
            name=f"proton_{len([p for p in self._particles if 'proton' in p.name])}",
            mass=1836.0,
            charge=1.0,
            spin=0.5,
            position=position,
            momentum=momentum
        )
    
    def add_neutron(self, position: List[float] = None) -> 'SceneBuilder':
        """添加中子"""
        return self.add_particle(
            name=f"neutron_{len([p for p in self._particles if 'neutron' in p.name])}",
            mass=1839.0,
            charge=0.0,
            spin=0.5,
            position=position
        )
    
    def add_potential(self, potential: Potential) -> 'SceneBuilder':
        """添加势能"""
        self._potentials.append(potential)
        return self
    
    def add_coulomb_potential(self, 
                              center: List[float],
                              strength: float,
                              Z: float = 1.0) -> 'SceneBuilder':
        """添加库仑势"""
        self._potentials.append(Potential.coulomb(np.array(center), strength, Z))
        return self
    
    def add_harmonic_potential(self,
                              center: List[float],
                              k: float) -> 'SceneBuilder':
        """添加谐振子势"""
        self._potentials.append(Potential.harmonic(np.array(center), k))
        return self
    
    def add_square_well(self,
                         center: List[float],
                         radius: float,
                         depth: float) -> 'SceneBuilder':
        """添加方势阱"""
        self._potentials.append(Potential.square_well(np.array(center), radius, depth))
        return self
    
    def add_custom_potential(self, 
                              func: Callable,
                              name: str = 'Custom') -> 'SceneBuilder':
        """添加自定义势能"""
        self._potentials.append(Potential.custom(func, name))
        return self
    
    def set_spatial_range(self, xmin: float, xmax: float) -> 'SceneBuilder':
        """设置空间范围"""
        self._spatial_range = (xmin, xmax)
        return self
    
    def set_dimension(self, dim: int) -> 'SceneBuilder':
        """设置空间维度"""
        self._dimension = dim
        return self
    
    def set_grid_points(self, n: int) -> 'SceneBuilder':
        """设置网格点数"""
        self._grid_points = n
        return self
    
    def set_boundary_condition(self, bc: str) -> 'SceneBuilder':
        """设置边界条件 ('infinite', 'periodic', 'zero')"""
        self._boundary = bc
        return self
    
    def enable_spin_coupling(self, enable: bool = True) -> 'SceneBuilder':
        """启用自旋耦合"""
        self._spin_coupling = enable
        return self
    
    def set_external_field(self, field_type: str, **params) -> 'SceneBuilder':
        """设置外场"""
        self._external_field = {'type': field_type, 'params': params}
        return self
    
    def build(self) -> QuantumScene:
        """构建场景"""
        return QuantumScene(
            name=self._name,
            particles=self._particles.copy(),
            potentials=self._potentials.copy(),
            spatial_range=self._spatial_range,
            dimension=self._dimension,
            grid_points=self._grid_points,
            boundary_condition=self._boundary,
            spin_coupling=self._spin_coupling,
            external_field=self._external_field
        )
    
    def summary(self) -> str:
        """场景摘要"""
        lines = [f"Scene: {self._name}"]
        lines.append(f"Particles: {len(self._particles)}")
        for p in self._particles:
            lines.append(f"  - {p.name}: m={p.mass}, q={p.charge}, spin={p.spin}, pos={p.position.tolist()}")
        lines.append(f"Potentials: {len(self._potentials)}")
        for v in self._potentials:
            lines.append(f"  - {v.name} ({v.potential_type})")
        lines.append(f"Dimension: {self._dimension}D")
        lines.append(f"Grid: {self._grid_points} points")
        return "\n".join(lines)


class HamiltonianBuilder:
    """
    哈密顿量构建器
    
    从场景配置构建哈密顿算符。
    """
    
    def __init__(self, scene: QuantumScene):
        self._scene = scene
        self._grid = self._setup_grid()
    
    def _setup_grid(self) -> np.ndarray:
        """设置空间网格"""
        xmin, xmax = self._scene.spatial_range
        n = self._scene.grid_points
        return np.linspace(xmin, xmax, n)
    
    def build_kinetic_term(self) -> np.ndarray:
        """构建动能项 T = -h²/(2m) * d²/dx²
        
        Standard three-point finite difference:
        T[i,i] = h²/(m*dx²)
        T[i,i±1] = -h²/(2m*dx²)
        """
        n = len(self._grid)
        dx = self._grid[1] - self._grid[0]
        
        hbar = 1.0  # hbar = 1 in natural units
        total_mass = sum(p.mass for p in self._scene.particles) if self._scene.particles else 1.0
        
        coeff = hbar**2 / (2 * total_mass * dx**2)
        
        T = np.zeros((n, n))
        for i in range(n):
            T[i, i] = 2.0 * coeff  # +h²/(m*dx²)
            if i > 0:
                T[i, i-1] = -coeff  # -h²/(2m*dx²)
            if i < n-1:
                T[i, i+1] = -coeff  # -h²/(2m*dx²)
        
        return T
    
    def build_potential_term(self) -> np.ndarray:
        """构建势能项"""
        n = len(self._grid)
        V = np.zeros(n)
        
        for i, x in enumerate(self._grid):
            pos = np.array([x] + [0.0] * (self._scene.dimension - 1))
            for pot in self._scene.potentials:
                V[i] += pot.evaluate(pos)
        
        return np.diag(V)
    
    def build_interaction_term(self) -> np.ndarray:
        """构建粒子间相互作用项"""
        n = len(self._grid)
        V_int = np.zeros(n)
        
        for i, p1 in enumerate(self._scene.particles):
            for j, p2 in enumerate(self._scene.particles):
                if i >= j:
                    continue
                if p1.charge != 0 and p2.charge != 0:
                    for k, x in enumerate(self._grid):
                        pos = np.array([x] + [0.0] * (self._scene.dimension - 1))
                        r = np.linalg.norm(pos - p1.position)
                        if r > 1e-10:
                            V_int[k] += p1.charge * p2.charge / r
        
        return np.diag(V_int)
    
    def build(self) -> np.ndarray:
        """构建完整哈密顿量"""
        T = self.build_kinetic_term()
        V = self.build_potential_term()
        
        H = T + V
        
        if len(self._scene.particles) > 1:
            V_int = self.build_interaction_term()
            H += V_int
        
        return H


def simulate(scene: QuantumScene, num_states: int = 5) -> 'SimulationResult':
    """
    模拟量子场景
    
    Args:
        scene: 量子场景配置
        num_states: 计算的状态数量
        
    Returns:
        SimulationResult: 模拟结果
    """
    from .solver import ExactDiagonalizationSolver
    from .hamiltonian import MatrixHamiltonian
    
    builder = HamiltonianBuilder(scene)
    H_matrix = builder.build()
    
    H = MatrixHamiltonian(H_matrix, name=scene.name)
    solver = ExactDiagonalizationSolver(H)
    states, energies = solver.solve()
    
    return SimulationResult(
        scene=scene,
        hamiltonian=H,
        states=states[:num_states],
        energies=energies[:num_states],
        grid=builder._grid
    )


@dataclass
class SimulationResult:
    """模拟结果"""
    scene: QuantumScene
    hamiltonian: Any
    states: List
    energies: np.ndarray
    grid: np.ndarray
    
    def summary(self) -> str:
        """结果摘要"""
        lines = [f"=== {self.scene.name} ==="]
        lines.append(f"Particles: {len(self.scene.particles)}")
        lines.append(f"Potentials: {len(self.scene.potentials)}")
        lines.append(f"Grid points: {len(self.grid)}")
        lines.append(f"")
        lines.append("Energy levels:")
        for i, E in enumerate(self.energies):
            lines.append(f"  State {i}: E = {E:.6f}")
        return "\n".join(lines)
    
    def get_wavefunction(self, state_index: int) -> np.ndarray:
        """获取波函数"""
        return self.states[state_index].to_vector()
    
    def get_probability_density(self, state_index: int) -> np.ndarray:
        """获取概率密度 |ψ|²"""
        psi = self.get_wavefunction(state_index)
        return np.abs(psi)**2
    
    def get_position_expectation(self, state_index: int) -> float:
        """计算位置期望值 <x>"""
        psi = self.get_wavefunction(state_index)
        x = self.grid
        return float(np.sum(x * np.abs(psi)**2) * (x[1] - x[0]))
    
    def get_energy_uncertainty(self, state_index: int) -> float:
        """计算能量不确定度 ΔE"""
        E = self.energies[state_index]
        H = self.hamiltonian
        return self.hamiltonian.variance(self.states[state_index])


class Visualizer:
    """结果可视化器"""
    
    def __init__(self, result: SimulationResult):
        self._result = result
    
    def plot_potential(self, ax=None, **kwargs):
        """绘制势能曲线"""
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots()
            
            V = np.zeros(len(self._result.grid))
            for pot in self._result.scene.potentials:
                for i, x in enumerate(self._result.grid):
                    pos = np.array([x] + [0.0] * (self._result.scene.dimension - 1))
                    V[i] += pot.evaluate(pos)
            
            ax.plot(self._result.grid, V, **kwargs)
            ax.set_xlabel('Position x')
            ax.set_ylabel('Potential V(x)')
            ax.set_title(f'Potential Energy - {self._result.scene.name}')
            ax.grid(True, alpha=0.3)
            
            return ax
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
            return None
    
    def plot_wavefunctions(self, num_states: int = 3, ax=None, **kwargs):
        """绘制波函数"""
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for i in range(min(num_states, len(self._result.states))):
                psi = self._result.get_wavefunction(i)
                E = self._result.energies[i]
                ax.plot(self._result.grid, psi.real + E, label=f'n={i}, E={E:.4f}', **kwargs)
            
            ax.set_xlabel('Position x')
            ax.set_ylabel('Wavefunction + Energy')
            ax.set_title(f'Wavefunctions - {self._result.scene.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return ax
        except ImportError:
            print("matplotlib not available")
            return None
    
    def plot_probability_density(self, num_states: int = 3, ax=None, **kwargs):
        """绘制概率密度"""
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for i in range(min(num_states, len(self._result.states))):
                prob = self._result.get_probability_density(i)
                E = self._result.energies[i]
                ax.plot(self._result.grid, prob + E, label=f'n={i}, E={E:.4f}', **kwargs)
            
            ax.set_xlabel('Position x')
            ax.set_ylabel('|ψ|² + Energy')
            ax.set_title(f'Probability Density - {self._result.scene.name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return ax
        except ImportError:
            print("matplotlib not available")
            return None
    
    def plot_spectrum(self, ax=None, **kwargs):
        """绘制能谱"""
        try:
            import matplotlib.pyplot as plt
            
            if ax is None:
                fig, ax = plt.subplots()
            
            n = len(self._result.energies)
            ax.bar(range(n), self._result.energies, **kwargs)
            ax.set_xlabel('State Index')
            ax.set_ylabel('Energy')
            ax.set_title(f'Energy Spectrum - {self._result.scene.name}')
            ax.grid(True, alpha=0.3, axis='y')
            
            return ax
        except ImportError:
            print("matplotlib not available")
            return None
    
    def plot_all(self):
        """绘制所有图形"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            self.plot_potential(ax=axes[0, 0])
            self.plot_wavefunctions(ax=axes[0, 1])
            self.plot_probability_density(ax=axes[1, 0])
            self.plot_spectrum(ax=axes[1, 1])
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available")


def quick_simulate(particles: List[Dict] = None,
                   potentials: List[Dict] = None,
                   x_range: Tuple[float, float] = (-10, 10),
                   n_points: int = 200,
                   **kwargs) -> SimulationResult:
    """
    快速模拟接口
    
    Args:
        particles: 粒子列表 [{'type': 'electron', 'position': [0]}]
        potentials: 势能列表 [{'type': 'coulomb', 'center': [0], 'strength': 1}]
        x_range: 空间范围
        n_points: 网格点数
        **kwargs: 其他参数
        
    Returns:
        SimulationResult
    """
    scene = SceneBuilder("QuickSim")
    scene.set_spatial_range(*x_range)
    scene.set_grid_points(n_points)
    
    if particles:
        for p in particles:
            ptype = p.get('type', 'electron')
            pos = p.get('position', [0])
            if ptype == 'electron':
                scene.add_electron(position=pos)
            elif ptype == 'proton':
                scene.add_proton(position=pos)
            elif ptype == 'neutron':
                scene.add_neutron(position=pos)
            else:
                scene.add_particle(ptype, 
                                 mass=p.get('mass', 1),
                                 charge=p.get('charge', 0),
                                 position=pos)
    
    if potentials:
        for v in potentials:
            vtype = v.get('type')
            if vtype == 'coulomb':
                scene.add_coulomb_potential(v['center'], v.get('strength', 1), v.get('Z', 1))
            elif vtype == 'harmonic':
                scene.add_harmonic_potential(v['center'], v.get('k', 1))
            elif vtype == 'square_well':
                scene.add_square_well(v['center'], v.get('radius', 1), v.get('depth', 1))
            elif vtype == 'custom' and 'func' in v:
                scene.add_custom_potential(v['func'], v.get('name', 'Custom'))
    
    qs = scene.build()
    return simulate(qs, **kwargs)


__all__ = [
    'Particle',
    'Potential', 
    'QuantumScene',
    'SceneBuilder',
    'HamiltonianBuilder',
    'simulate',
    'SimulationResult',
    'Visualizer',
    'quick_simulate',
]
