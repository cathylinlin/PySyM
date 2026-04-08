# PySymmetry 量子模块使用文档

量子物理模块提供量子系统的建模、模拟和对称性分析功能。

## 目录

- [模块概览](#模块概览)
- [快速开始](#快速开始)
- [量子态表示](#量子态表示)
- [哈密顿算符](#哈密顿算符)
- [求解器](#求解器)
- [对称性分析](#对称性分析)
- [氢原子SO4对称性](#氢原子so4对称性)
- [李代数与群表示](#李代数与群表示)
- [集成与桥接](#集成与桥接)
- [交互式模拟](#交互式模拟)

---

## 模块概览

```
phys.quantum
├── states         # 量子态 (Ket, Bra, DensityMatrix)
├── hamiltonian    # 哈密顿算符构建
├── solver         # 本征问题求解器
├── simulator      # 动力学模拟
├── interactive    # 场景构建与模拟
├── analysis       # 对称性分析
├── hydrogen_symmetry  # 氢原子SO(4)对称性
├── integration    # 与core/abstract_phys桥接
└── explainer      # 结果解释
```

---

## 快速开始

```python
from PySymmetry.phys.quantum import *

# 1. 构建量子场景
scene = SceneBuilder("氢原子")
scene.add_electron(position=[0, 0, 0])
scene.add_coulomb_potential(center=[0, 0, 0], strength=1.0, Z=1.0)
scene.set_spatial_range(-10, 10)
scene.set_grid_points(200)
qs = scene.build()

# 2. 运行模拟
result = simulate(qs, num_states=5)

# 3. 对称性分析
sym_result = analyze_scene_symmetry(qs)
print(f"检测到的对称性: {sym_result['detected']}")
print(f"守恒量: {sym_result['conserved']}")
```

---

## 量子态表示

### Ket 与 Bra

```python
from PySymmetry.phys.quantum import Ket, Bra, DensityMatrix
import numpy as np

# 从向量创建
vec = np.array([1, 0], dtype=complex)
ket = Ket(vec)

# 标签创建 ( qubit )
ket0 = Ket('0')  # |0⟩
ket1 = Ket('1')  # |1⟩
ket_plus = Ket('|+')  # |+⟩ = (|0⟩ + |1⟩)/√2

# 内积 <ψ|φ>
bra = Bra(ket0)
inner = ket0.inner_product(bra)

# 归一化
ket = ket.normalize()

# 维度
print(f"维度: {ket.dimension}")
print(f"范数: {ket.norm()}")
```

### 常用量子态

```python
from PySymmetry.phys.quantum import (
    bell_state, w_state, ghz_state, 
    tensor_product, superposition
)

# Bell 态
bell = bell_state(0)  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
bell = bell_state(1)  # |Φ⁻⟩
bell = bell_state(2)  # |Ψ⁺⟩
bell = bell_state(3)  # |Ψ⁻⟩

# W态 (3量子比特)
w = w_state(3)

# GHZ态
ghz = ghz_state(3)

# 张量积 |0⟩ ⊗ |1⟩
state = tensor_product(Ket('0'), Ket('1'))

# 叠加态
state = superposition((Ket('0'), 1/np.sqrt(2)), (Ket('1'), 1/np.sqrt(2)))
```

### 密度矩阵

```python
# 从纯态创建
ket = Ket(np.array([1, 0], dtype=complex))
rho = DensityMatrix(ket)

# 从向量直接创建
rho = DensityMatrix(np.array([1, 0], dtype=complex))

# 性质
print(f"纯态: {rho.is_pure}")
print(f"纯度: {rho.purity}")
print(f"熵: {rho.entropy()}")

# 保真度
rho2 = DensityMatrix(np.array([0, 1], dtype=complex))
f = rho.fidelity(rho2)
```

---

## 哈密顿算符

### 内置哈密顿量

```python
from PySymmetry.phys.quantum import (
    MatrixHamiltonian,
    FreeParticleHamiltonian,
    HarmonicOscillatorHamiltonian,
    HydrogenAtomHamiltonian,
    SpinHamiltonian,
    PerturbationHamiltonian,
)

# 矩阵形式
H_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
H = MatrixHamiltonian(H_matrix)

# 自由粒子 H = p²/2m
H_free = FreeParticleHamiltonian(mass=1.0, basis_size=100)

# 谐振子 H = p²/2m + ½mω²x²
H_HO = HarmonicOscillatorHamiltonian(mass=1.0, frequency=1.0, dimension=50)

# 氢原子 (具有SO(4)隐藏对称性)
H_H = HydrogenAtomHamiltonian(Z=1.0, max_n=5)

# 自旋系统
H_spin = SpinHamiltonian(spin=1, hamiltonian_type='heisenberg', 
                         couplings={'J': 1.0})

# 微扰哈密顿量
H0 = HarmonicOscillatorHamiltonian(dimension=20)
V = np.diag(np.random.randn(20))
H_pert = PerturbationHamiltonian(H0, V, coupling=0.1)
```

### 哈密顿量操作

```python
# 期望值
E = H.expectation(ket)

# 方差
var = H.variance(ket)

# 基态与激发态
ground_state, E0 = H.ground_state()
excited = H.excited_states(n=3)

# 能量本征值
energies = H.all_energy_levels()
```

### 自旋与Pauli算符

```python
from PySymmetry.phys.quantum import pauli_matrices, spin_operators

# Pauli矩阵 σx, σy, σz, I
sx, sy, sz, I = pauli_matrices()

# 自旋算符 s=1/2, 1, 3/2...
Sx, Sy, Sz = spin_operators(s=0.5)

# 角动量算符
Lx, Ly, Lz = angular_momentum_operators(l=1)
```

---

## 求解器

### 求解器选择

```python
from PySymmetry.phys.quantum import (
    ExactDiagonalizationSolver,
    SparseMatrixSolver,
    LanczosSolver,
    TimeEvolutionSolver,
    NumerovSolver,
    QuantumSolverFactory,
    solve_schrodinger,
)

# 小矩阵 (<1000维)
solver = ExactDiagonalizationSolver(H)
states, energies = solver.solve()

# 中等矩阵 (1000-10000维)
solver = SparseMatrixSolver(H, num_eigenvalues=20, which='SA')
states, energies = solver.solve()

# 大矩阵 (>10000维)
solver = LanczosSolver(H, num_eigenvalues=10)
states, energies = solver.solve()

# 自动选择
solver = QuantumSolverFactory.create(hamiltonian=H)
states, energies = solver.solve()
```

### 特殊系统求解器

```python
from PySymmetry.phys.quantum import (
    HydrogenAtomSolver,
    HarmonicOscillatorSolver,
    ParticleInBoxSolver,
)

# 氢原子 (解析解)
hydrogen_solver = HydrogenAtomSolver(Z=1.0, max_n=10)
states, energies = hydrogen_solver.solve()

# 谐振子 (解析解)
ho_solver = HarmonicOscillatorSolver(mass=1.0, omega=1.0, max_n=50)
states, energies = ho_solver.solve()

# 方势阱
pib_solver = ParticleInBoxSolver(L=1.0)
states, energies = pib_solver.solve(num_states=10)
```

### 时间演化

```python
# 含时演化
演化器 = TimeEvolutionSolver(H, dt=0.01)
states, times = 演化器.evolve(initial_state, t0=0, tf=10, num_steps=100)

# 分步傅里叶 (含势场)
from PySymmetry.phys.quantum import SplitStepFourierSolver

def V(x):
    return 0.5 * x**2

solver = SplitStepFourierSolver(potential=V, x_range=10, num_points=512)
snapshots, times = solver.evolve(initial_wavefunction, num_steps=100)
```

---

## 对称性分析

### 量子Analyzer

```python
from PySymmetry.phys.quantum import (
    QuantumAnalyzer,
    QuantumParityOperation,
    analyze,
    quick_report,
    check_parity,
)

# 创建分析器
analyzer = QuantumAnalyzer(hamiltonian=H, result=simulation_result)

# 检测对称性
symmetries = analyzer.detect_symmetries()
print(f"检测到 {len(symmetries)} 种对称性")

# 宇称分析
parity, desc = analyzer.analyze_parity(ket)
print(f"宇称: {parity} ({desc})")

# 态分类
classifications = analyzer.classify_states()

# 选择规则
rules = analyzer.compute_selection_rules()

# 生成报告
report = analyzer.generate_report()
print(report)
```

### 对称操作

```python
# 宇称操作 (空间反演)
parity_op = QuantumParityOperation(dimension=100)
P_matrix = parity_op.representation_matrix()
eigenvalue = parity_op.eigenvalue(psi)

# 平移操作
from PySymmetry.phys.quantum import QuantumTranslationOperation
translation_op = QuantumTranslationOperation(displacement=0.1, dimension=100)
T_matrix = translation_op.representation_matrix()
```

### 便捷函数

```python
# 完整分析
result = analyze(hamiltonian, simulation_result)

# 快速报告
report = quick_report(hamiltonian, simulation_result)

# 检查宇称
parity = check_parity(ket, dim=100)
```

---

## 氢原子SO4对称性

氢原子具有隐藏的SO(4)对称性，由Runge-Lenz向量与角动量共同生成。

### 基本分析

```python
from PySymmetry.phys.quantum import (
    HydrogenSymmetryAnalyzer,
    SO4HydrogenAnalyzer,
    detect_hydrogen_so4_symmetry,
    analyze_hydrogen_symmetry,
)

# 使用氢原子哈密顿量
H = HydrogenAtomHamiltonian(Z=1.0, max_n=5)

# SO(4)分析器
so4_analyzer = SO4HydrogenAnalyzer(H, max_n=5)

# 检测SO(4)对称性
result = so4_analyzer.detect_so4_symmetry()
print(f"检测到: {result['detected']}")
print(f"描述: {result['description']}")

# 生成完整报告
report = so4_analyzer.generate_report()
print(report)
```

### 代数结构分析

```python
# SO(4)代数结构
structure = so4_analyzer.analyze_so4_structure()
print(f"代数: {structure['algebra']}")
print(f"生成元: {structure['generators']}")
print(f"对易关系: {structure['commutation_relations']}")

# 态分类
classifications = so4_analyzer.classify_hydrogen_states()
for c in classifications[:5]:
    print(f"n={c['n']}, l={c['l']}, m={c['m']}, SO4={c['so4_label']}")
```

### 便捷函数

```python
# 检测氢原子SO(4)对称性
result = detect_hydrogen_so4_symmetry(Z=1.0, max_n=5)

# 生成分析报告
report = analyze_hydrogen_symmetry(max_n=5, Z=1.0)
print(report)
```

---

## 李代数与群表示

### 群表示

```python
from PySymmetry.phys.quantum import (
    GroupRepresentation,
    LieAlgebra,
    SO3LieAlgebra,
    SU2LieAlgebra,
)

# 创建群表示
rep = GroupRepresentation('D3', dimension=3)
rep.add_element('E', np.eye(3))
rep.add_element('C3', np.eye(3))  # 示例

# 计算特征标
rep.compute_character(['E', 'C3'])
print(f"不可约: {rep.is_irreducible()}")
```

### 李代数

```python
# SO(3) (角动量代数)
so3 = SO3LieAlgebra()
print(f"生成元: {list(so3.generators.keys())}")

# 计算结构常数
f = so3.compute_structure_constants()

# Casimir算符
C = so3.casimir_operator()

# 角动量算符
L2, Jz, Jx, Jy = so3.angular_momentum_operator(l=1)
```

### 群论分析器

```python
from PySymmetry.phys.quantum import (
    GroupTheoryAnalyzer,
    analyze_group_theory,
)

# 创建分析器
analyzer = GroupTheoryAnalyzer(hamiltonian=H)

# 检测对称群
groups = analyzer.detect_symmetry_group(H_matrix)
print(f"对称群: {groups}")

# 不可约表示分类
irreps = analyzer.classify_irreps(energies, states)

# 选择规则
rules = analyzer.compute_selection_rules(states, states, operator='dipole')

# 生成报告
report = analyzer.generate_group_report()
```

### Wigner-Eckart定理

```python
from PySymmetry.phys.quantum import wigner_eckart_coupling, tensor_operator_matrix_element

# CG系数 (简化版)
cg = wigner_eckart_coupling(j1=1, j2=1, j3=0, m1=1, m2=-1, m3=0, q=0)

# 张量算符矩阵元
matrix_element = tensor_operator_matrix_element(j=1, k=1, q=0, j_prime=1, m=0, m_prime=0)
```

---

## 集成与桥接

### 量子-抽象物理桥接

```python
from PySymmetry.phys.quantum import (
    QuantumToAbstractBridge,
    SymmetryToQuantumBridge,
    LieAlgebraToQuantumBridge,
    GroupRepresentationBridge,
    integrate_with_abstract_phys,
    quick_bridge,
    quick_lie_bridge,
)

# 对称操作桥接
bridge = quick_bridge(dimension=100)
P = bridge.parity_operator()
T = bridge.translation_operator(k=0.5)
R = bridge.rotation_operator(theta=np.pi/2)

# 检查哈密顿量对称性
symmetries = bridge.find_symmetries(H_matrix)

# 李代数桥接
la_bridge = quick_lie_bridge()
su2_ops = la_bridge.su2_generators(s=0.5)
so3_ops = la_bridge.so3_generators(l=1)

# Heisenberg哈密顿量
H_heis = la_bridge.heisenberg_xxx_hamiltonian(J=1.0, n_sites=4)
```

### 完整集成

```python
# 完整集成分析
result = integrate_with_abstract_phys(
    scene=quantum_scene,
    hamiltonian_matrix=H_matrix,
    analyze_symmetry=True,
    compute_lie_algebra=True
)

print(f"对称操作: {result.symmetry_operations}")
print(f"李代数: {result.lie_algebras}")
print(f"守恒量: {result.conserved_quantities}")
```

---

## 交互式模拟

### 场景构建

```python
from PySymmetry.phys.quantum import (
    SceneBuilder,
    Potential,
    QuantumScene,
)

# 氢原子场景
scene = (SceneBuilder("氢原子")
    .add_electron(position=[0, 0, 0])
    .add_coulomb_potential(center=[0, 0, 0], strength=1.0, Z=1.0)
    .set_spatial_range(-10, 10)
    .set_grid_points(200)
    .build())

# 谐振子场景
scene = (SceneBuilder("3D谐振子")
    .add_electron()
    .add_harmonic_3d(center=[0, 0, 0], kx=1.0, ky=1.0, kz=1.0)
    .set_spatial_range_3d((-5, 5))
    .set_grid_points_3d(50)
    .build())

# 方势阱
scene = (SceneBuilder("方势阱")
    .add_square_well(center=[0], radius=5, depth=10)
    .set_spatial_range(-10, 10)
    .set_grid_points(200)
    .build())
```

### 运行模拟

```python
from PySymmetry.phys.quantum import simulate, quick_simulate, analyze_scene_symmetry

# 完整模拟
result = simulate(scene, num_states=5, analyze_symmetry=True)

# 快速模拟
result = quick_simulate(
    potentials=[{'type': 'coulomb', 'center': [0], 'strength': 1}],
    x_range=(-10, 10),
    n_points=200
)

# 对称性分析
sym_info = analyze_scene_symmetry(scene)
print(f"对称性: {sym_info['detected']}")
print(f"描述: {sym_info['description']}")
```

### 结果处理

```python
from PySymmetry.phys.quantum import SimulationResult, Visualizer

result = simulate(scene)

# 获取结果
energies = result.energies
states = result.states
grid = result.grid

# 波函数
psi = result.get_wavefunction(0)
prob = result.get_probability_density(0)

# 期望值
x_exp = result.get_position_expectation(0)

# 宇称
parity = result.get_parity(0)

# 摘要
print(result.summary())
```

### 可视化

```python
viz = Visualizer(result)

# 势能
viz.plot_potential()

# 波函数
viz.plot_wavefunctions(num_states=3)

# 概率密度
viz.plot_probability_density(num_states=3)

# 能谱
viz.plot_spectrum()

# 3D (需要3D场景)
viz.plot_3d_probability_density(state_index=0)
viz.plot_3d_slices(state_index=0)

# 全部
viz.plot_all()
```

---

## 高级示例

### 氢原子完整分析

```python
import numpy as np
from PySymmetry.phys.quantum import *

# 1. 创建氢原子场景
scene = (SceneBuilder("氢原子")
    .add_electron()
    .add_coulomb_potential(center=[0, 0, 0], strength=1.0, Z=1.0)
    .set_spatial_range(-20, 20)
    .set_grid_points(400)
    .build())

# 2. 模拟
result = simulate(scene, num_states=10, analyze_symmetry=True)

# 3. SO(4)分析
so4_result = detect_hydrogen_so4_symmetry(max_n=5)
print(f"SO(4)检测: {so4_result['detected']}")

# 4. 对称性分析
sym_info = analyze_scene_symmetry(scene)

# 5. 可视化
viz = Visualizer(result)
viz.plot_probability_density(num_states=4)
viz.plot_spectrum()
```

### 自旋链模拟

```python
from PySymmetry.phys.quantum import SpinChainSimulator

# Heisenberg自旋链
sim = SpinChainSimulator(
    num_sites=4,
    couplings={'x': 1.0, 'y': 1.0, 'z': 1.0},  # XXX模型
    external_field=0.5,  # 外磁场
    initial_state=Ket(np.array([1, 0, 0, 0], dtype=complex))
)

results = sim.run(duration=10, dt=0.01)

# 磁化演化
plt.plot(results['times'], results['magnetization'])
plt.show()
```

### 退相干模拟

```python
from PySymmetry.phys.quantum import DecoherenceSimulator

# 初始Bell态
bell = bell_state(0)
rho = DensityMatrix(bell)

# 退相干模拟
sim = DecoherenceSimulator(
    initial_state=rho,
    decoherence_rate=0.1
)

results = sim.run(duration=10, dt=0.01)

# 纯度和熵的演化
plt.plot(results['times'], results['purity_history'])
plt.plot(results['times'], results['entropy_history'])
plt.show()
```
