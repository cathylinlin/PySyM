"""
量子系统示例脚本

展示 PySymmetry 量子模块的经典场景模拟功能。

运行方式:
    python examples/quantum_examples.py

场景包括:
1. 氢原子 - 能级和光谱
2. 无限深势阱 - 粒子在势阱中的行为
3. 量子谐振子 - 能级和相干态
4. 自旋系统 - Zeeman 效应和 Heisenberg 模型
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySymmetry.phys.quantum import *


def print_header(title):
    """打印标题"""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """打印章节标题"""
    print()
    print("-" * 70)
    print(f"  {title}")
    print("-" * 70)


def example_hydrogen_atom():
    """氢原子示例"""
    print_header("1. 氢原子 (Hydrogen Atom)")

    print("""
    氢原子是量子力学中最基本的系统之一。
    核外电子在库仑势 V(r) = -e^2/r 中运动。
    
    解析能级公式: E_n = -1/(2n^2) (原子单位)
    
    主要量子数:
    - n: 主量子数 (1, 2, 3, ...)
    - l: 轨道量子数 (0, 1, ..., n-1)
    - m: 磁量子数 (-l, ..., l)
    """)

    Z = 1  # 原子序数
    max_n = 5  # 最大主量子数

    print_section("能级计算")

    H = HydrogenAtomHamiltonian(Z=Z, reduced_mass=0.5, max_n=max_n)
    print(f"哈密顿量维度: {H.dimension}")
    print()

    levels = H.energy_levels()
    print("能级结构 (n, l, m) -> E:")
    print("-" * 40)

    for n in range(1, max_n + 1):
        E_n = -1 / (2 * n**2)
        degeneracy = n**2
        print(f"n={n}: E = {E_n:.6f} (简并度 = {degeneracy})")

        if n <= 3:
            for l in range(n):
                for m in range(-l, l + 1):
                    actual_E = levels.get((n, l, m), E_n)
                    print(f"      (n={n}, l={l}, m={m})")

    print()
    print_section("光谱分析")

    transitions = []
    for n1 in range(1, max_n):
        for n2 in range(n1 + 1, max_n + 1):
            E_diff = -1 / (2 * n2**2) - (-1 / (2 * n1**2))
            wavelength = 1 / E_diff if E_diff > 0 else float("inf")  # 简化
            transitions.append((n1, n2, E_diff))

    print("可能的光谱跃迁 (发射):")
    print("-" * 40)
    for n1, n2, E in sorted(transitions, key=lambda x: x[2])[:5]:
        print(f"  n={n2} -> n={n1}: Delta-E = {E:.6f}")

    print()
    return H


def example_infinite_square_well():
    """无限深势阱示例"""
    print_header("2. 无限深势阱 (Infinite Square Well)")

    print("""
    粒子被限制在 [0, L] 的无限深势阱中。
    势能: V(x) = 0 for 0 < x < L, V(x) = infinity otherwise
    
    解析解:
    - 波函数: psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)
    - 能级: E_n = n^2 * pi^2 / (2mL^2)
    
    特点:
    - 能级离散化
    - 基态无节点，第一激发态有一个节点，...
    - 能量与 n^2 成正比
    """)

    L = np.pi  # 势阱宽度
    N = 200  # 格点数

    print_section("离散化哈密顿量")
    print(f"势阱宽度 L = {L:.4f}")
    print(f"内部格点数 N = {N}")
    print(f"格点间距 dx = {L / (N + 1):.6f}")
    print()

    dx = L / (N + 1)

    H_matrix = np.zeros((N, N))
    for i in range(N):
        H_matrix[i, i] = 1.0 / dx**2
        if i > 0:
            H_matrix[i, i - 1] = -0.5 / dx**2
        if i < N - 1:
            H_matrix[i, i + 1] = -0.5 / dx**2

    H = MatrixHamiltonian(H_matrix, name="InfiniteSquareWell")
    solver = ExactDiagonalizationSolver(H)
    states, energies = solver.solve()

    print_section("能级对比")
    print(f"{'n':<5} {'解析解':<15} {'数值解':<15} {'相对误差':<15}")
    print("-" * 50)

    for n in range(5):
        expected = (n + 1) ** 2 * np.pi**2 / (2 * L**2)
        actual = energies[n]
        rel_error = abs(expected - actual) / expected * 100
        print(f"{n + 1:<5} {expected:<15.6f} {actual:<15.6f} {rel_error:<15.4f}%")

    print()
    print_section("波函数分析")

    x_points = np.linspace(0, L, N)

    for n in range(3):
        psi = states[n].to_vector()
        max_val = np.max(np.abs(psi))
        node_count = np.sum(np.abs(np.diff(np.sign(psi.real))) > 0.1)
        print(f"第{n + 1}态: 最大值={max_val:.4f}, 节点数={node_count}")

    print()
    print("观察: 节点数 = n-1, 这符合解析解 sin(n*pi*x/L) 的性质")

    print()
    return H, states


def example_quantum_harmonic_oscillator():
    """量子谐振子示例"""
    print_header("3. 量子谐振子 (Quantum Harmonic Oscillator)")

    print("""
    谐振子势: V(x) = 1/2 * m * omega^2 * x^2
    
    解析解:
    - 能级: E_n = omega * (n + 1/2)
    - 波函数: Hermite 多项式乘以高斯包络
    
    特点:
    - 能级等间距 (Delta-E = omega)
    - 基态为高斯形
    - 所有态在无穷远处趋于零
    """)

    omega = 1.0
    dimension = 30

    print_section("能级结构")

    H = HarmonicOscillatorHamiltonian(
        mass=1.0, frequency=omega, dimension=dimension, hbar=1.0
    )

    solver = ExactDiagonalizationSolver(H)
    states, energies = solver.solve()

    print(f"频率 omega = {omega}")
    print(f"希尔伯特空间维度 = {dimension}")
    print()
    print(f"{'n':<5} {'解析解':<15} {'数值解':<15} {'间距 Delta-E':<15}")
    print("-" * 50)

    for n in range(5):
        expected = omega * (n + 0.5)
        actual = energies[n]
        delta = omega if n > 0 else 0
        print(f"{n:<5} {expected:<15.6f} {actual:<15.6f} {delta:<15.6f}")

    print()
    print(f"能级间距 Delta-E = omega = {omega} (等间距!)")

    print_section("相干态")

    alphas = [0.5, 1.0, 2.0]
    print("相干态 |alpha> 的性质:")
    print(f"{'alpha':<10} {'|<n|alpha>|^2 (n=0)':<25} {'|<n|alpha>|^2 (n=1)':<25}")
    print("-" * 60)

    for alpha in alphas:
        coh_state = H.coherent_state(alpha)
        probs = np.abs(coh_state.to_vector()) ** 2
        print(f"{alpha:<10} {probs[0]:<25.4f} {probs[1]:<25.4f}")

    print()
    print("相干态是湮灭算符的本征态，在时间演化下保持形式不变")

    print()
    return H, states


def example_spin_system():
    """自旋系统示例"""
    print_header("4. 自旋系统 (Spin Systems)")

    print("""
    自旋是粒子的内禀角动量，由 Pauli 矩阵描述。
    
    常见自旋哈密顿量:
    1. Zeeman: H = -g * mu_B * B * S_z
    2. Heisenberg: H = J * S1 · S2
    3. Ising: H = J * S_z^1 * S_z^2
    """)

    print_section("自旋-1/2 粒子")

    print("Pauli 矩阵:")
    sx, sy, sz, I = pauli_matrices()
    print(f"""
    sigma_x = [[0, 1], [1, 0]]
    sigma_y = [[0, -i], [i, 0]]
    sigma_z = [[1, 0], [0, -1]]
    """)

    print_section("Zeeman 效应")

    couplings = {"g": 2.0, "B": 1.0}
    H_zeeman = SpinHamiltonian(spin=0.5, couplings=couplings, hamiltonian_type="zeeman")

    solver = ExactDiagonalizationSolver(H_zeeman)
    states, energies = solver.solve()

    print(f"磁场 B = {couplings['B']}, g因子 = {couplings['g']}")
    print(f"能级分裂: Delta-E = {energies[1] - energies[0]:.4f}")
    print(f"解释: 自旋向上/向下态能量差为 g*mu_B*B")

    print_section("Heisenberg 模型")

    H_heisenberg = SpinHamiltonian(
        spin=0.5, couplings={"J": 1.0}, hamiltonian_type="heisenberg"
    )

    solver2 = ExactDiagonalizationSolver(H_heisenberg)
    states2, energies2 = solver2.solve()

    print(f"交换耦合 J = 1.0")
    print("本征能量:")
    for i, E in enumerate(energies2):
        state_type = ["单态 (singlet)", "三重态 (triplet)"][min(i, 1)]
        print(f"  E_{i} = {E:.4f} ({state_type})")

    print()
    print("注意: 单态 (反平行) 和三重态 (平行) 的能级分裂")

    print()
    return H_zeeman, H_heisenberg


def example_measurement():
    """量子测量示例"""
    print_header("5. 量子测量 (Quantum Measurement)")

    print("""
    量子测量会导致波函数坍缩到被测量的本征态。
    测量结果按概率分布出现。
    """)

    print_section("自旋测量")

    H = SpinHamiltonian(spin=0.5, hamiltonian_type="zeeman", couplings={"B": 1.0})

    state = Ket(np.array([1.0, 1.0], dtype=complex)).normalize()

    print(f"初始态: |psi> = [{state[0]:.4f}, {state[1]:.4f}]")
    print()

    num_measurements = 1000
    results = []

    meas_sim = MeasurementSimulator(state)
    for _ in range(num_measurements):
        result, prob = meas_sim.measure()
        results.append(result)

    print(f"测量 {num_measurements} 次的结果统计:")
    print(f"  |0> (自旋向上): {results.count(0)} 次")
    print(f"  |1> (自旋向下): {results.count(1)} 次")
    print(
        f"  理论概率: |c0|^2 = {np.abs(state[0]) ** 2:.4f}, |c1|^2 = {np.abs(state[1]) ** 2:.4f}"
    )

    print()
    return meas_sim


def example_time_evolution():
    """时间演化示例"""
    print_header("6. 时间演化 (Time Evolution)")

    print("""
    含时薛定谔方程: i * d|psi>/dt = H|psi>
    
    解的形式: |psi(t)> = U(t) |psi(0)>
    其中 U(t) = exp(-i H t) 是时间传播子
    """)

    print_section("自旋系统演化")

    H = SpinHamiltonian(spin=0.5, hamiltonian_type="zeeman", couplings={"B": 1.0})
    initial_state = Ket(np.array([1.0, 0.0], dtype=complex))

    print(f"初始态: |psi(0)> = |0> (自旋向上)")
    print(f"哈密顿量: H = diag({H.matrix[0, 0]:.2f}, {H.matrix[1, 1]:.2f})")
    print()

    solver = TimeEvolutionSolver(H, dt=0.1)
    times = [0.0, 0.25, 0.5, 1.0]

    print("时间演化:")
    print(f"{'t':<10} {'|<0|psi(t)>|^2':<20} {'|<1|psi(t)>|^2':<20}")
    print("-" * 50)

    for t in times:
        states, _ = solver.evolve(initial_state, t0=0.0, num_steps=int(t / 0.1))
        state_t = states[-1]
        probs = np.abs(state_t.to_vector()) ** 2
        print(f"{t:<10.2f} {probs[0]:<20.4f} {probs[1]:<20.4f}")

    print()
    print("观察: 概率在两个态之间振荡 (Rabi 振荡)")

    print()
    return solver


def example_spectrum_analysis():
    """能谱分析示例"""
    print_header("7. 能谱分析 (Spectrum Analysis)")

    print("""
    能谱分析揭示系统的量子特性:
    - 能级间距反映相互作用强度
    - 简并度反映对称性
    - 能隙指示稳定性
    """)

    print_section("谐振子能谱")

    H = HarmonicOscillatorHamiltonian(frequency=1.0, dimension=20)
    explainer = EnergySpectrumExplainer(H)

    print(explainer.explain())

    print_section("态分析")

    ground, _ = H.ground_state()
    state_explainer = QuantumStateExplainer(ground)
    print(state_explainer.explain())

    print()
    return H


def run_all_examples():
    """运行所有示例"""
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  PySymmetry 量子物理模块 - 经典场景演示  ".center(68) + "#")
    print(
        "#"
        + "  PySymmetry Quantum Module - Classical Scenarios Demo  ".center(68)
        + "#"
    )
    print("#" + " " * 68 + "#")
    print("#" * 70)

    examples = [
        ("氢原子", example_hydrogen_atom),
        ("无限深势阱", example_infinite_square_well),
        ("量子谐振子", example_quantum_harmonic_oscillator),
        ("自旋系统", example_spin_system),
        ("量子测量", example_measurement),
        ("时间演化", example_time_evolution),
        ("能谱分析", example_spectrum_analysis),
    ]

    results = []
    for name, func in examples:
        try:
            result = func()
            results.append((name, True, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False, None))

    print()
    print("#" * 70)
    print("#  总结  ".center(68) + "#")
    print("#" * 70)

    print()
    print("场景测试结果:")
    for name, success, _ in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name:<20} {status}")

    print()
    print("所有经典量子场景演示完成!")
    print()

    return results


if __name__ == "__main__":
    run_all_examples()
