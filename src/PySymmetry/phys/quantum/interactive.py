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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from PySymmetry.abstract_phys import (
        ElementaryParticle,
        Field,
        ScalarField,
        SpinorField,
        VectorField,
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
    momentum: np.ndarray | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.momentum is not None:
            self.momentum = np.asarray(self.momentum, dtype=float)


@dataclass
class Potential:
    """势能配置"""

    name: str
    potential_type: str
    dimension: int = 3
    parameters: dict[str, Any] = field(default_factory=dict)
    function: Callable | None = None

    @classmethod
    def coulomb(cls, center: np.ndarray, strength: float, Z: float = 1.0):
        """库仑势 V(r) = -Z*strength/r"""
        dim = len(center) if hasattr(center, "__len__") else 3

        def V(x):
            r = np.linalg.norm(x - center)
            epsilon = 1e-6
            if r < epsilon:
                r = epsilon
            return -Z * strength / r

        return cls(
            name="Coulomb",
            potential_type="coulomb",
            dimension=dim,
            parameters={"center": center, "strength": strength, "Z": Z},
            function=V,
        )

    @classmethod
    def harmonic(cls, center: np.ndarray, k: float):
        """谐振子势 V(r) = 1/2 * k * r^2"""
        dim = len(center) if hasattr(center, "__len__") else 3

        def V(x):
            r_sq = np.sum((x - center) ** 2)
            return 0.5 * k * r_sq

        return cls(
            name="Harmonic",
            potential_type="harmonic",
            dimension=dim,
            parameters={"center": center, "k": k},
            function=V,
        )

    @classmethod
    def square_well(cls, center: np.ndarray, radius: float, depth: float):
        """方势阱"""
        dim = len(center) if hasattr(center, "__len__") else 3

        def V(x):
            r = np.linalg.norm(x - center)
            return -depth if r < radius else 0.0

        return cls(
            name="SquareWell",
            potential_type="square_well",
            dimension=dim,
            parameters={"center": center, "radius": radius, "depth": depth},
            function=V,
        )

    @classmethod
    def harmonic_3d(
        cls,
        center: np.ndarray = None,
        kx: float = 1.0,
        ky: float = None,
        kz: float = None,
    ):
        """3D各向异性谐振子势"""
        if center is None:
            center = np.array([0.0, 0.0, 0.0])
        if ky is None:
            ky = kx
        if kz is None:
            kz = kx

        def V(x):
            dx, dy, dz = x - center
            return 0.5 * (kx * dx**2 + ky * dy**2 + kz * dz**2)

        return cls(
            name="Harmonic3D",
            potential_type="harmonic_3d",
            dimension=3,
            parameters={"center": center, "kx": kx, "ky": ky, "kz": kz},
            function=V,
        )

    @classmethod
    def spherical_well(cls, radius: float, depth: float, center: np.ndarray = None):
        """球方势阱"""
        if center is None:
            center = np.array([0.0, 0.0, 0.0])

        def V(x):
            r = np.linalg.norm(x - center)
            return -depth if r < radius else 0.0

        return cls(
            name="SphericalWell",
            potential_type="spherical_well",
            dimension=3,
            parameters={"center": center, "radius": radius, "depth": depth},
            function=V,
        )

    @classmethod
    def gaussian_well(cls, center: np.ndarray, depth: float, width: float = 1.0):
        """高斯势阱 V(r) = -depth * exp(-r^2 / (2*width^2))"""
        dim = len(center) if hasattr(center, "__len__") else 3

        def V(x):
            r_sq = np.sum((x - center) ** 2)
            return -depth * np.exp(-r_sq / (2 * width**2))

        return cls(
            name="GaussianWell",
            potential_type="gaussian_well",
            dimension=dim,
            parameters={"center": center, "depth": depth, "width": width},
            function=V,
        )

    @classmethod
    def cylindrical_potential(
        cls, axis: str = "z", radius: float = 1.0, depth: float = 1.0
    ):
        """柱对称势能"""

        def V(x):
            if axis == "z":
                rho = np.sqrt(x[0] ** 2 + x[1] ** 2)
            elif axis == "y":
                rho = np.sqrt(x[0] ** 2 + x[2] ** 2)
            else:
                rho = np.sqrt(x[1] ** 2 + x[2] ** 2)
            return -depth if rho < radius else 0.0

        return cls(
            name="Cylindrical",
            potential_type="cylindrical",
            dimension=3,
            parameters={"axis": axis, "radius": radius, "depth": depth},
            function=V,
        )

    @classmethod
    def step(cls, position: float, height: float):
        """阶梯势"""

        def V(x):
            x_val = x[0] if hasattr(x, "__len__") and len(x) > 0 else x
            return height if x_val > position else 0.0

        return cls(
            name="Step",
            potential_type="step",
            dimension=1,
            parameters={"position": position, "height": height},
            function=V,
        )

    @classmethod
    def custom(cls, func: Callable, name: str = "Custom", dimension: int = 3):
        """自定义势能"""
        return cls(
            name=name, potential_type="custom", dimension=dimension, function=func
        )

    def evaluate(self, x: np.ndarray) -> float:
        """计算势能值"""
        if self.function is not None:
            return self.function(x)
        return 0.0


@dataclass
class QuantumScene:
    """量子场景配置"""

    name: str
    particles: list[Particle] = field(default_factory=list)
    potentials: list[Potential] = field(default_factory=list)
    spatial_range: tuple[float, float] = (-10.0, 10.0)
    spatial_range_3d: list[tuple[float, float]] = field(
        default_factory=lambda: [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    )
    dimension: int = 1
    grid_points: int = 100
    grid_points_3d: list[int] = field(default_factory=lambda: [50, 50, 50])
    boundary_condition: str = "infinite"
    spin_coupling: bool = False
    external_field: dict[str, float] | None = None
    symmetry_info: dict[str, Any] | None = field(default_factory=dict, repr=False)

    def has_symmetry(self, symmetry_type: str) -> bool:
        """检查是否具有指定类型的对称性"""
        return symmetry_type in self.symmetry_info.get("detected", [])

    def get_conserved_quantities(self) -> list[str]:
        """获取守恒量列表"""
        return self.symmetry_info.get("conserved", [])

    def get_grid_3d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取3D网格"""
        grids = []
        for i in range(3):
            xmin, xmax = self.spatial_range_3d[i]
            n = self.grid_points_3d[i]
            grids.append(np.linspace(xmin, xmax, n))
        return tuple(grids)


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
        self._particles: list[Particle] = []
        self._potentials: list[Potential] = []
        self._spatial_range = (-10.0, 10.0)
        self._spatial_range_3d = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
        self._dimension = 1
        self._grid_points = 100
        self._grid_points_3d = [50, 50, 50]
        self._boundary = "infinite"
        self._spin_coupling = False
        self._external_field = None

    def add_particle(
        self,
        name: str,
        mass: float,
        charge: float,
        spin: float = 0.5,
        position: list[float] = None,
        momentum: list[float] = None,
    ) -> "SceneBuilder":
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
            momentum=np.array(momentum),
        )
        self._particles.append(particle)
        return self

    def add_electron(
        self, position: list[float] = None, momentum: list[float] = None
    ) -> "SceneBuilder":
        """添加电子 (便捷方法)"""
        return self.add_particle(
            name=f"electron_{len([p for p in self._particles if 'electron' in p.name])}",
            mass=1.0,
            charge=-1.0,
            spin=0.5,
            position=position,
            momentum=momentum,
        )

    def add_proton(
        self, position: list[float] = None, momentum: list[float] = None
    ) -> "SceneBuilder":
        """添加质子 (便捷方法)"""
        return self.add_particle(
            name=f"proton_{len([p for p in self._particles if 'proton' in p.name])}",
            mass=1836.0,
            charge=1.0,
            spin=0.5,
            position=position,
            momentum=momentum,
        )

    def add_neutron(self, position: list[float] = None) -> "SceneBuilder":
        """添加中子"""
        return self.add_particle(
            name=f"neutron_{len([p for p in self._particles if 'neutron' in p.name])}",
            mass=1839.0,
            charge=0.0,
            spin=0.5,
            position=position,
        )

    def add_potential(self, potential: Potential) -> "SceneBuilder":
        """添加势能"""
        self._potentials.append(potential)
        return self

    def add_coulomb_potential(
        self, center: list[float], strength: float, Z: float = 1.0
    ) -> "SceneBuilder":
        """添加库仑势"""
        self._potentials.append(Potential.coulomb(np.array(center), strength, Z))
        return self

    def add_harmonic_potential(self, center: list[float], k: float) -> "SceneBuilder":
        """添加谐振子势"""
        self._potentials.append(Potential.harmonic(np.array(center), k))
        return self

    def add_square_well(
        self, center: list[float], radius: float, depth: float
    ) -> "SceneBuilder":
        """添加方势阱"""
        self._potentials.append(Potential.square_well(np.array(center), radius, depth))
        return self

    def add_harmonic_3d(
        self,
        center: list[float] = None,
        kx: float = 1.0,
        ky: float = None,
        kz: float = None,
    ) -> "SceneBuilder":
        """添加3D各向异性谐振子势"""
        c = np.array(center) if center else np.array([0.0, 0.0, 0.0])
        self._potentials.append(Potential.harmonic_3d(c, kx, ky, kz))
        self._dimension = 3
        return self

    def add_spherical_well(
        self, radius: float, depth: float, center: list[float] = None
    ) -> "SceneBuilder":
        """添加球方势阱"""
        c = np.array(center) if center else np.array([0.0, 0.0, 0.0])
        self._potentials.append(Potential.spherical_well(radius, depth, c))
        self._dimension = 3
        return self

    def add_gaussian_well(
        self, center: list[float], depth: float, width: float = 1.0
    ) -> "SceneBuilder":
        """添加高斯势阱"""
        self._potentials.append(Potential.gaussian_well(np.array(center), depth, width))
        self._dimension = 3
        return self

    def add_custom_potential(
        self, func: Callable, name: str = "Custom"
    ) -> "SceneBuilder":
        """添加自定义势能"""
        self._potentials.append(Potential.custom(func, name, self._dimension))
        return self

    def set_spatial_range(self, xmin: float, xmax: float) -> "SceneBuilder":
        """设置1D空间范围"""
        self._spatial_range = (xmin, xmax)
        self._spatial_range_3d = [(xmin, xmax), (xmin, xmax), (xmin, xmax)]
        return self

    def set_spatial_range_3d(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float] = None,
        z_range: tuple[float, float] = None,
    ) -> "SceneBuilder":
        """设置3D空间范围"""
        if y_range is None:
            y_range = x_range
        if z_range is None:
            z_range = x_range
        self._spatial_range_3d = [x_range, y_range, z_range]
        self._spatial_range = x_range
        self._dimension = 3
        return self

    def set_dimension(self, dim: int) -> "SceneBuilder":
        """设置空间维度"""
        self._dimension = dim
        return self

    def set_grid_points(self, n: int) -> "SceneBuilder":
        """设置1D网格点数"""
        self._grid_points = n
        self._grid_points_3d = [n, n, n]
        return self

    def set_grid_points_3d(
        self, nx: int, ny: int = None, nz: int = None
    ) -> "SceneBuilder":
        """设置3D网格点数"""
        if ny is None:
            ny = nx
        if nz is None:
            nz = nx
        self._grid_points_3d = [nx, ny, nz]
        self._grid_points = nx * ny * nz
        self._dimension = 3
        return self

    def set_boundary_condition(self, bc: str) -> "SceneBuilder":
        """设置边界条件 ('infinite', 'periodic', 'zero')"""
        self._boundary = bc
        return self

    def enable_spin_coupling(self, enable: bool = True) -> "SceneBuilder":
        """启用自旋耦合"""
        self._spin_coupling = enable
        return self

    def set_external_field(self, field_type: str, **params) -> "SceneBuilder":
        """设置外场"""
        self._external_field = {"type": field_type, "params": params}
        return self

    def build(self) -> QuantumScene:
        """构建场景"""
        return QuantumScene(
            name=self._name,
            particles=self._particles.copy(),
            potentials=self._potentials.copy(),
            spatial_range=self._spatial_range,
            spatial_range_3d=self._spatial_range_3d.copy(),
            dimension=self._dimension,
            grid_points=self._grid_points,
            grid_points_3d=self._grid_points_3d.copy(),
            boundary_condition=self._boundary,
            spin_coupling=self._spin_coupling,
            external_field=self._external_field,
        )

    def summary(self) -> str:
        """场景摘要"""
        lines = [f"Scene: {self._name}"]
        lines.append(f"Particles: {len(self._particles)}")
        for p in self._particles:
            lines.append(
                f"  - {p.name}: m={p.mass}, q={p.charge}, spin={p.spin}, pos={p.position.tolist()}"
            )
        lines.append(f"Potentials: {len(self._potentials)}")
        for v in self._potentials:
            lines.append(f"  - {v.name} ({v.potential_type})")
        lines.append(f"Dimension: {self._dimension}D")
        lines.append(f"Grid: {self._grid_points} points")
        return "\n".join(lines)


class SceneHamiltonianBuilder:
    """
    哈密顿量构建器

    从场景配置构建哈密顿算符。支持1D和3D情况。
    """

    def __init__(self, scene: QuantumScene):
        self._scene = scene
        self._grid = self._setup_grid()
        self._grid_3d = self._setup_grid_3d() if scene.dimension >= 3 else None

    def _setup_grid(self) -> np.ndarray:
        """设置1D空间网格"""
        xmin, xmax = self._scene.spatial_range
        n = self._scene.grid_points
        return np.linspace(xmin, xmax, n)

    def _setup_grid_3d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """设置3D空间网格"""
        gx, gy, gz = self._scene.get_grid_3d()
        return gx, gy, gz

    def build_kinetic_term(self) -> np.ndarray:
        """构建动能项 T = -h²/(2m) * ∇²

        Standard three-point finite difference:
        T[i,i] = h²/(m*dx²)
        T[i,i±1] = -h²/(2m*dx²)
        """
        if self._scene.dimension >= 3:
            return self._build_kinetic_term_3d()
        return self._build_kinetic_term_1d()

    def _build_kinetic_term_1d(self) -> np.ndarray:
        """1D动能项"""
        n = len(self._grid)
        dx = self._grid[1] - self._grid[0]

        hbar = 1.0
        total_mass = (
            sum(p.mass for p in self._scene.particles) if self._scene.particles else 1.0
        )

        coeff = hbar**2 / (2 * total_mass * dx**2)

        T = np.zeros((n, n))
        for i in range(n):
            T[i, i] = 2.0 * coeff
            if i > 0:
                T[i, i - 1] = -coeff
            if i < n - 1:
                T[i, i + 1] = -coeff

        return T

    def _build_kinetic_term_3d(self) -> np.ndarray:
        """3D动能项使用Kronecker积"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        dx = gx[1] - gx[0]
        dy = gy[1] - gy[0]
        dz = gz[1] - gz[0]

        hbar = 1.0
        total_mass = (
            sum(p.mass for p in self._scene.particles) if self._scene.particles else 1.0
        )

        Tx = np.zeros((nx, nx))
        coeff_x = hbar**2 / (2 * total_mass * dx**2)
        for i in range(nx):
            Tx[i, i] = 2.0 * coeff_x
            if i > 0:
                Tx[i, i - 1] = -coeff_x
            if i < nx - 1:
                Tx[i, i + 1] = -coeff_x

        Ty = np.zeros((ny, ny))
        coeff_y = hbar**2 / (2 * total_mass * dy**2)
        for i in range(ny):
            Ty[i, i] = 2.0 * coeff_y
            if i > 0:
                Ty[i, i - 1] = -coeff_y
            if i < ny - 1:
                Ty[i, i + 1] = -coeff_y

        Tz = np.zeros((nz, nz))
        coeff_z = hbar**2 / (2 * total_mass * dz**2)
        for i in range(nz):
            Tz[i, i] = 2.0 * coeff_z
            if i > 0:
                Tz[i, i - 1] = -coeff_z
            if i < nz - 1:
                Tz[i, i + 1] = -coeff_z

        T = np.kron(np.kron(Tx, np.eye(ny)), np.eye(nz))
        T += np.kron(np.eye(nx), np.kron(Ty, np.eye(nz)))
        T += np.kron(np.eye(nx), np.kron(np.eye(ny), Tz))

        return T

    def build_potential_term(self) -> np.ndarray:
        """构建势能项"""
        if self._scene.dimension >= 3:
            return self._build_potential_term_3d()
        return self._build_potential_term_1d()

    def _build_potential_term_1d(self) -> np.ndarray:
        """1D势能项"""
        n = len(self._grid)
        V = np.zeros(n)

        for i, x in enumerate(self._grid):
            pos = np.array([x] + [0.0] * (self._scene.dimension - 1))
            for pot in self._scene.potentials:
                V[i] += pot.evaluate(pos)

        return np.diag(V)

    def _build_potential_term_3d(self) -> np.ndarray:
        """3D势能项"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        N = nx * ny * nz

        V = np.zeros(N)

        for ix, x in enumerate(gx):
            for iy, y in enumerate(gy):
                for iz, z in enumerate(gz):
                    idx = ix * ny * nz + iy * nz + iz
                    pos = np.array([x, y, z])
                    for pot in self._scene.potentials:
                        V[idx] += pot.evaluate(pos)

        return np.diag(V)

    def build_interaction_term(self) -> np.ndarray:
        """构建粒子间相互作用项"""
        if self._scene.dimension >= 3:
            return self._build_interaction_term_3d()
        return self._build_interaction_term_1d()

    def _build_interaction_term_1d(self) -> np.ndarray:
        """1D相互作用项"""
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

    def _build_interaction_term_3d(self) -> np.ndarray:
        """3D相互作用项"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        N = nx * ny * nz

        V_int = np.zeros(N)

        for i, p1 in enumerate(self._scene.particles):
            for j, p2 in enumerate(self._scene.particles):
                if i >= j:
                    continue
                if p1.charge != 0 and p2.charge != 0:
                    for ix, x in enumerate(gx):
                        for iy, y in enumerate(gy):
                            for iz, z in enumerate(gz):
                                idx = ix * ny * nz + iy * nz + iz
                                pos = np.array([x, y, z])
                                r = np.linalg.norm(pos - p1.position)
                                if r > 1e-10:
                                    V_int[idx] += p1.charge * p2.charge / r

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

    @property
    def grid(self) -> np.ndarray:
        """获取1D网格"""
        return self._grid

    @property
    def grid_3d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取3D网格"""
        return self._grid_3d

    @property
    def dimension(self) -> int:
        """获取空间维度"""
        return self._scene.dimension

    @property
    def num_states(self) -> int:
        """获取希尔伯特空间维度"""
        if self._scene.dimension >= 3:
            nx, ny, nz = self._grid_3d
            return len(nx) * len(ny) * len(nz)
        return len(self._grid)


class SceneSymmetryAnalyzer:
    """
    场景对称性分析器

    自动检测量子场景中的对称性并返回守恒量信息。

    检测的对称性类型:
    - parity: 空间反演对称性
    - translation: 空间平移对称性
    - rotation: 旋转对称性
    - time_translation: 时间平移对称性（能量守恒）
    """

    def __init__(self, scene: QuantumScene):
        self._scene = scene
        self._grid = self._setup_grid()
        self._grid_3d = None
        self._detected_symmetries: list[str] = []
        self._conserved_quantities: list[str] = []
        self._symmetry_matrices: dict[str, np.ndarray] = {}

        if scene.dimension >= 3:
            self._grid_3d = self._setup_grid_3d()

    def _setup_grid(self) -> np.ndarray:
        """设置网格"""
        xmin, xmax = self._scene.spatial_range
        n = self._scene.grid_points
        return np.linspace(xmin, xmax, n)

    def _setup_grid_3d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """设置3D网格"""
        return self._scene.get_grid_3d()

    def analyze(self) -> dict[str, Any]:
        """
        执行完整的对称性分析

        Returns:
            包含检测到的对称性和守恒量的字典
        """
        self._detected_symmetries = []
        self._conserved_quantities = []
        self._symmetry_matrices = {}

        dim = self._scene.dimension
        grid_3d_none = self._grid_3d is None
        if dim >= 3 and not grid_3d_none:
            self._check_parity_symmetry_3d()
            self._check_translation_symmetry_3d()
            self._check_spherical_symmetry()
            self._check_cylindrical_symmetry()
        elif dim == 1:
            self._check_parity_symmetry()
            self._check_translation_symmetry()
            self._check_central_potential()

        self._check_periodic_boundary()

        self._conserved_quantities.append("energy")

        return {
            "detected": self._detected_symmetries,
            "conserved": self._conserved_quantities,
            "matrices": self._symmetry_matrices,
            "description": self._generate_description(),
            "dimension": dim,
        }

    def _check_parity_symmetry_3d(self) -> bool:
        """检查3D空间反演对称性"""
        gx, gy, gz = self._grid_3d

        V = self._compute_potential_on_grid_3d()

        nx, ny, nz = len(gx), len(gy), len(gz)

        V_3d = V.reshape(nx, ny, nz)

        V_x_sym = np.allclose(V_3d, V_3d[::-1, :, :], rtol=1e-5)
        V_y_sym = np.allclose(V_3d, V_3d[:, ::-1, :], rtol=1e-5)
        V_z_sym = np.allclose(V_3d, V_3d[:, :, ::-1], rtol=1e-5)

        if V_x_sym and V_y_sym and V_z_sym:
            self._detected_symmetries.append("parity")
            self._conserved_quantities.append("parity")
            self._symmetry_matrices["parity_x"] = self._build_parity_matrix_3d("x")
            self._symmetry_matrices["parity_y"] = self._build_parity_matrix_3d("y")
            self._symmetry_matrices["parity_z"] = self._build_parity_matrix_3d("z")
            return True
        return False

    def _check_translation_symmetry_3d(self) -> bool:
        """检查3D平移对称性"""
        if self._scene.boundary_condition == "periodic":
            self._detected_symmetries.append("translation_3d")
            self._conserved_quantities.append("crystal_momentum")
            self._symmetry_matrices["translation_x"] = (
                self._build_translation_matrix_3d(1, 0, 0)
            )
            self._symmetry_matrices["translation_y"] = (
                self._build_translation_matrix_3d(0, 1, 0)
            )
            self._symmetry_matrices["translation_z"] = (
                self._build_translation_matrix_3d(0, 0, 1)
            )
            return True
        return False

    def _check_spherical_symmetry(self) -> bool:
        """检查球对称性"""
        for pot in self._scene.potentials:
            if pot.potential_type in (
                "coulomb",
                "spherical_well",
                "harmonic_3d",
                "gaussian_well",
            ):
                self._detected_symmetries.append("spherical")
                self._conserved_quantities.append("angular_momentum")
                self._conserved_quantities.append("L_z")
                return True
        return False

    def _check_cylindrical_symmetry(self) -> bool:
        """检查柱对称性"""
        for pot in self._scene.potentials:
            if pot.potential_type == "cylindrical":
                self._detected_symmetries.append("cylindrical")
                self._conserved_quantities.append("L_z")
                return True
        return False

    def _compute_potential_on_grid_3d(self) -> np.ndarray:
        """计算3D网格上的势能"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        N = nx * ny * nz
        V = np.zeros(N)

        for ix, x in enumerate(gx):
            for iy, y in enumerate(gy):
                for iz, z in enumerate(gz):
                    idx = ix * ny * nz + iy * nz + iz
                    pos = np.array([x, y, z])
                    for pot in self._scene.potentials:
                        V[idx] += pot.evaluate(pos)

        return V

    def _build_parity_matrix_3d(self, axis: str = "x") -> np.ndarray:
        """构建3D反演矩阵"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        N = nx * ny * nz
        P = np.zeros((N, N))

        if axis == "x":
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        i = ix * ny * nz + iy * nz + iz
                        j = (nx - 1 - ix) * ny * nz + iy * nz + iz
                        P[i, j] = 1.0
        elif axis == "y":
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        i = ix * ny * nz + iy * nz + iz
                        j = ix * ny * nz + (ny - 1 - iy) * nz + iz
                        P[i, j] = 1.0
        else:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        i = ix * ny * nz + iy * nz + iz
                        j = ix * ny * nz + iy * nz + (nz - 1 - iz)
                        P[i, j] = 1.0

        return P

    def _build_translation_matrix_3d(
        self, shift_x: int = 0, shift_y: int = 0, shift_z: int = 0
    ) -> np.ndarray:
        """构建3D平移矩阵"""
        gx, gy, gz = self._grid_3d
        nx, ny, nz = len(gx), len(gy), len(gz)
        N = nx * ny * nz
        T = np.zeros((N, N))

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    i = ix * ny * nz + iy * nz + iz
                    jx = (ix + shift_x) % nx
                    jy = (iy + shift_y) % ny
                    jz = (iz + shift_z) % nz
                    j = jx * ny * nz + jy * nz + jz
                    T[i, j] = 1.0

        return T

    def _check_parity_symmetry(self) -> bool:
        """检查空间反演对称性 (V(x) = V(-x))"""
        if self._scene.dimension != 1:
            return False

        V = self._compute_potential_on_grid()

        n = len(V)
        mid = n // 2

        if n % 2 == 0:
            left = V[:mid]
            right = V[mid:][::-1]
        else:
            left = V[:mid]
            right = V[mid + 1 :][::-1]

        is_symmetric = np.allclose(left, right, rtol=1e-5)

        if is_symmetric:
            self._detected_symmetries.append("parity")
            self._conserved_quantities.append("parity")
            parity_matrix = self._build_parity_matrix()
            self._symmetry_matrices["parity"] = parity_matrix

        return is_symmetric

    def _check_translation_symmetry(self) -> bool:
        """检查平移对称性（均匀势能）"""
        V = self._compute_potential_on_grid()

        spacing = np.diff(self._grid)
        is_uniform = np.allclose(spacing, spacing[0], rtol=1e-5)

        if is_uniform and self._scene.boundary_condition == "periodic":
            self._detected_symmetries.append("translation")
            self._conserved_quantities.append("crystal_momentum")
            trans_matrix = self._build_translation_matrix()
            self._symmetry_matrices["translation"] = trans_matrix
            return True

        return False

    def _check_central_potential(self) -> bool:
        """检查中心势能（球对称）"""
        for pot in self._scene.potentials:
            if pot.potential_type == "coulomb":
                self._detected_symmetries.append("central")
                self._conserved_quantities.append("angular_momentum")
                return True
        return False

    def _check_periodic_boundary(self) -> bool:
        """检查周期性边界条件"""
        if self._scene.boundary_condition == "periodic":
            self._detected_symmetries.append("translation")
            self._conserved_quantities.append("crystal_momentum")
            return True
        return False

    def _compute_potential_on_grid(self) -> np.ndarray:
        """计算网格上的势能"""
        n = len(self._grid)
        V = np.zeros(n)

        for i, x in enumerate(self._grid):
            pos = np.array([x] + [0.0] * (self._scene.dimension - 1))
            for pot in self._scene.potentials:
                V[i] += pot.evaluate(pos)

        return V

    def _build_parity_matrix(self) -> np.ndarray:
        """构建反演矩阵"""
        n = len(self._grid)
        P = np.zeros((n, n))
        for i in range(n):
            P[i, n - 1 - i] = 1.0
        return P

    def _build_translation_matrix(self, shift: int = 1) -> np.ndarray:
        """构建平移矩阵"""
        n = len(self._grid)
        T = np.zeros((n, n))
        for i in range(n):
            T[i, (i + shift) % n] = 1.0
        return T

    def _generate_description(self) -> str:
        """生成对称性描述"""
        lines = []
        if "parity" in self._detected_symmetries:
            lines.append("空间反演对称 (Parity): V(x) = V(-x)")
        if "translation" in self._detected_symmetries:
            lines.append("平移对称 (Translation): 周期边界或均匀势")
        if "central" in self._detected_symmetries:
            lines.append("中心对称 (Central): 球对称势")
        if "energy" in self._conserved_quantities:
            lines.append("能量守恒 (Energy): 时间平移对称")
        return "; ".join(lines) if lines else "无明显对称性"

    def get_parity_eigenvalues(self, states: list[Any]) -> list[int]:
        """计算态的宇称本征值"""
        if "parity" not in self._symmetry_matrices:
            return []

        P = self._symmetry_matrices["parity"]
        eigenvalues = []

        for state in states:
            psi = state.to_vector()
            P_psi = P @ psi
            overlap = np.vdot(psi, P_psi).real
            eigenvalue = 1 if overlap > 0 else -1
            eigenvalues.append(eigenvalue)

        return eigenvalues

    def classify_states_by_symmetry(
        self, states: list[Any], energies: np.ndarray
    ) -> dict[str, Any]:
        """根据对称性对态进行分类"""
        classifications = {"parity": {}, "quantum_numbers": {}}

        if "parity" in self._detected_symmetries:
            parities = self.get_parity_eigenvalues(states)
            for i, (E, p) in enumerate(zip(energies, parities)):
                classifications["parity"][i] = {"energy": E, "parity": p}

        return classifications


def analyze_scene_symmetry(scene: QuantumScene) -> dict[str, Any]:
    """
    便捷函数：对场景进行对称性分析

    Args:
        scene: 量子场景

    Returns:
        对称性分析结果
    """
    analyzer = SceneSymmetryAnalyzer(scene)
    return analyzer.analyze()


def simulate(
    scene: QuantumScene, num_states: int = 5, analyze_symmetry: bool = True
) -> "SimulationResult":
    """
    模拟量子场景

    Args:
        scene: 量子场景配置
        num_states: 计算的状态数量
        analyze_symmetry: 是否进行对称性分析

    Returns:
        SimulationResult: 模拟结果
    """
    from .hamiltonian import MatrixHamiltonian
    from .solver import (
        ExactDiagonalizationSolver,
        HarmonicOscillatorSolver,
        HydrogenAtomSolver,
        QuantumSolverFactory,
    )

    sym_classifications = None
    sym_analyzer = None

    if analyze_symmetry:
        sym_analyzer = SceneSymmetryAnalyzer(scene)
        scene.symmetry_info = sym_analyzer.analyze()

    # 使用工厂自动选择求解器
    solver = QuantumSolverFactory.create_from_scene(scene)

    # 如果是专用解析求解器，直接使用
    if isinstance(solver, (HydrogenAtomSolver, HarmonicOscillatorSolver)):
        states, energies = solver.solve()
        if scene.dimension >= 3:
            grid = scene.get_grid_3d()
        else:
            n_points = 100
            grid = np.linspace(-5, 5, n_points)
        # 解析求解器不进行对称性分类(波函数基底不同)
        sym_classifications = None
    else:
        # 数值求解
        builder = SceneHamiltonianBuilder(scene)
        H_matrix = builder.build()
        H = MatrixHamiltonian(H_matrix, name=scene.name)
        exact_solver = ExactDiagonalizationSolver(H)
        states, energies = exact_solver.solve()
        grid = builder.grid if scene.dimension < 3 else builder.grid_3d

        if analyze_symmetry and sym_analyzer is not None:
            try:
                sym_classifications = sym_analyzer.classify_states_by_symmetry(
                    states, energies
                )
            except Exception:
                sym_classifications = None

    return SimulationResult(
        scene=scene,
        hamiltonian=None,
        states=states[:num_states],
        energies=energies[:num_states],
        grid=grid,
        symmetry_analysis=sym_classifications,
    )


@dataclass
class SimulationResult:
    """模拟结果"""

    scene: QuantumScene
    hamiltonian: Any
    states: list
    energies: np.ndarray
    grid: np.ndarray
    symmetry_analysis: dict[str, Any] | None = None

    def summary(self) -> str:
        """结果摘要"""
        lines = [f"=== {self.scene.name} ==="]
        lines.append(f"Particles: {len(self.scene.particles)}")
        lines.append(f"Potentials: {len(self.scene.potentials)}")
        lines.append(f"Grid points: {len(self.grid)}")

        if self.scene.symmetry_info:
            lines.append("")
            lines.append("Symmetries:")
            for sym in self.scene.symmetry_info.get("detected", []):
                lines.append(f"  - {sym}")

        lines.append("")
        lines.append("Energy levels:")
        for i, E in enumerate(self.energies):
            parity_info = ""
            if self.symmetry_analysis and "parity" in self.symmetry_analysis:
                parity_data = self.symmetry_analysis.get("parity", {})
                if i in parity_data:
                    p = parity_data[i].get("parity", 0)
                    parity_info = f" (P={'+' if p > 0 else '-'})"
            lines.append(f"  State {i}: E = {E:.6f}{parity_info}")
        return "\n".join(lines)

    def get_wavefunction(self, state_index: int) -> np.ndarray:
        """获取波函数"""
        return self.states[state_index].to_vector()

    def get_probability_density(self, state_index: int) -> np.ndarray:
        """获取概率密度 |ψ|²"""
        psi = self.get_wavefunction(state_index)
        return np.abs(psi) ** 2

    def get_position_expectation(self, state_index: int) -> float:
        """计算位置期望值 <x>"""
        psi = self.get_wavefunction(state_index)
        x = self.grid
        return float(np.sum(x * np.abs(psi) ** 2) * (x[1] - x[0]))

    def get_energy_uncertainty(self, state_index: int) -> float:
        """计算能量不确定度 ΔE"""
        E = self.energies[state_index]
        H = self.hamiltonian
        return self.hamiltonian.variance(self.states[state_index])

    def get_parity(self, state_index: int) -> int | None:
        """获取态的宇称"""
        if self.symmetry_analysis and "parity" in self.symmetry_analysis:
            parity_data = self.symmetry_analysis["parity"]
            if state_index in parity_data:
                return parity_data[state_index].get("parity")
        return None

    def get_conserved_quantities(self) -> list[str]:
        """获取守恒量列表"""
        return self.scene.get_conserved_quantities()


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
            ax.set_xlabel("Position x")
            ax.set_ylabel("Potential V(x)")
            ax.set_title(f"Potential Energy - {self._result.scene.name}")
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
                ax.plot(
                    self._result.grid, psi.real + E, label=f"n={i}, E={E:.4f}", **kwargs
                )

            ax.set_xlabel("Position x")
            ax.set_ylabel("Wavefunction + Energy")
            ax.set_title(f"Wavefunctions - {self._result.scene.name}")
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
                ax.plot(
                    self._result.grid, prob + E, label=f"n={i}, E={E:.4f}", **kwargs
                )

            ax.set_xlabel("Position x")
            ax.set_ylabel("|ψ|² + Energy")
            ax.set_title(f"Probability Density - {self._result.scene.name}")
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
            ax.set_xlabel("State Index")
            ax.set_ylabel("Energy")
            ax.set_title(f"Energy Spectrum - {self._result.scene.name}")
            ax.grid(True, alpha=0.3, axis="y")

            return ax
        except ImportError:
            print("matplotlib not available")
            return None

    def plot_3d_probability_density(self, state_index: int = 0, ax=None, **kwargs):
        """绘制3D概率密度等值面"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            grid = self._result.grid
            if not isinstance(grid, tuple) or len(grid) != 3:
                print("3D grid required for 3D visualization")
                return None

            gx, gy, gz = grid
            psi = self._result.get_wavefunction(state_index)
            prob = np.abs(psi) ** 2

            nx, ny, nz = len(gx), len(gy), len(gz)
            prob_3d = prob.reshape(nx, ny, nz)

            if ax is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")

            X, Y = np.meshgrid(gx, gy)
            slice_idx = nz // 2
            Z = prob_3d[:, :, slice_idx]

            ax.contourf(X, Y, Z, levels=15, alpha=0.8, **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Probability Density (z-slice) - State {state_index}")

            return ax
        except ImportError:
            print("matplotlib not available")
            return None
        except Exception as e:
            print(f"3D plot error: {e}")
            return None

    def plot_3d_slices(self, state_index: int = 0, ax=None, **kwargs):
        """绘制3D波函数的xy, yz, xz切片"""
        try:
            import matplotlib.pyplot as plt

            grid = self._result.grid
            if not isinstance(grid, tuple) or len(grid) != 3:
                print("3D grid required for 3D visualization")
                return None

            gx, gy, gz = grid
            psi = self._result.get_wavefunction(state_index)
            prob = np.abs(psi) ** 2

            nx, ny, nz = len(gx), len(gy), len(gz)
            prob_3d = prob.reshape(nx, ny, nz)

            if ax is None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            else:
                axes = ax

            slice_x, slice_y, slice_z = nx // 2, ny // 2, nz // 2

            axes[0].contourf(gy, gz, prob_3d[slice_x, :, :], levels=15)
            axes[0].set_xlabel("Y")
            axes[0].set_ylabel("Z")
            axes[0].set_title(f"X = {gx[slice_x]:.2f}")
            axes[0].set_aspect("equal")

            axes[1].contourf(gx, gz, prob_3d[:, slice_y, :], levels=15)
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Z")
            axes[1].set_title(f"Y = {gy[slice_y]:.2f}")
            axes[1].set_aspect("equal")

            axes[2].contourf(gx, gy, prob_3d[:, :, slice_z], levels=15)
            axes[2].set_xlabel("X")
            axes[2].set_ylabel("Y")
            axes[2].set_title(f"Z = {gz[slice_z]:.2f}")
            axes[2].set_aspect("equal")

            plt.suptitle(f"Probability Density Slices - State {state_index}")

            return axes
        except ImportError:
            print("matplotlib not available")
            return None
        except Exception as e:
            print(f"3D slice plot error: {e}")
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


def quick_simulate(
    particles: list[dict] = None,
    potentials: list[dict] = None,
    x_range: tuple[float, float] = (-10, 10),
    n_points: int = 200,
    **kwargs,
) -> SimulationResult:
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
            ptype = p.get("type", "electron")
            pos = p.get("position", [0])
            if ptype == "electron":
                scene.add_electron(position=pos)
            elif ptype == "proton":
                scene.add_proton(position=pos)
            elif ptype == "neutron":
                scene.add_neutron(position=pos)
            else:
                scene.add_particle(
                    ptype,
                    mass=p.get("mass", 1),
                    charge=p.get("charge", 0),
                    position=pos,
                )

    if potentials:
        for v in potentials:
            vtype = v.get("type")
            if vtype == "coulomb":
                scene.add_coulomb_potential(
                    v["center"], v.get("strength", 1), v.get("Z", 1)
                )
            elif vtype == "harmonic":
                scene.add_harmonic_potential(v["center"], v.get("k", 1))
            elif vtype == "square_well":
                scene.add_square_well(
                    v["center"], v.get("radius", 1), v.get("depth", 1)
                )
            elif vtype == "custom" and "func" in v:
                scene.add_custom_potential(v["func"], v.get("name", "Custom"))

    qs = scene.build()
    return simulate(qs, **kwargs)


__all__ = [
    "Particle",
    "Potential",
    "QuantumScene",
    "SceneBuilder",
    "SceneHamiltonianBuilder",
    "SceneSymmetryAnalyzer",
    "simulate",
    "SimulationResult",
    "Visualizer",
    "quick_simulate",
    "analyze_scene_symmetry",
]
