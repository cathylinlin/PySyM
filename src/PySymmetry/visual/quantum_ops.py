"""
量子门与操作可视化模块

提供量子门和量子操作的可视化：
- 量子门矩阵
- 量子电路图
- 门操作动画
- 保真度演化

依赖：numpy, matplotlib
"""

import numpy as np

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    patches = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed")


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


GATE_MATRICES = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
    "CNOT": CNOT,
    "SWAP": SWAP,
}


class QuantumGateVisualizer:
    """
    量子门可视化器
    """

    def __init__(self, figsize: tuple[float, float] = (10, 6)):
        _check_matplotlib()
        self.figsize = figsize

    def plot_gate_matrix(
        self, gate: np.ndarray, name: str = "Gate", title: str | None = None, ax=None
    ) -> tuple:
        """
        绘制量子门矩阵

        Args:
            gate: 门矩阵
            name: 门名称
            title: 标题
            ax: 坐标轴（可选）

        Returns:
            (fig, axes)
        """
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig = ax.figure
            axes = ax

        real_part = np.real(gate)
        imag_part = np.imag(gate)

        vmax = max(np.max(np.abs(real_part)), np.max(np.abs(imag_part)))

        im0 = axes[0].imshow(real_part, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title("Real Part")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(imag_part, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("Imaginary Part")
        plt.colorbar(im1, ax=axes[1])

        n = gate.shape[0]
        axes_list = (
            [axes]
            if not isinstance(axes, np.ndarray) and not hasattr(axes, "__len__")
            else axes
        )
        for ax in axes_list:
            if ax is not None:
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xlabel("Column")
                ax.set_ylabel("Row")

        title = title or f"Quantum Gate: {name}"
        if ax is None:
            fig.suptitle(title)
            plt.tight_layout()
        return fig, axes

    def plot_circuit(
        self,
        gates: list[tuple[str, int]],
        n_qubits: int = 2,
        title: str = "Quantum Circuit",
    ) -> tuple:
        """
        绘制量子电路

        Args:
            gates: 门列表 [(gate_name, qubit), ...]
            n_qubits: 量子比特数
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=(max(10, len(gates)), n_qubits * 1.5))

        wire_spacing = 1.0
        gate_width = 0.8

        for i in range(n_qubits):
            y = i * wire_spacing
            ax.plot([0, len(gates) + 1], [y, y], "k-", linewidth=1)
            ax.text(-0.3, y, f"q{i}⟩", fontsize=12, va="center")

        for i, (gate_name, qubit) in enumerate(gates):
            x = i + 1

            if gate_name == "CNOT":
                control = qubit
                target = qubit + 1
                if target < n_qubits:
                    ax.plot(
                        [x, x],
                        [control * wire_spacing, target * wire_spacing],
                        "k-",
                        linewidth=2,
                    )
                    ax.add_patch(
                        plt.Circle(
                            (x, control * wire_spacing), 0.1, fill=False, color="black"
                        )
                    )
                    ax.add_patch(
                        plt.Circle(
                            (x, target * wire_spacing), 0.1, fill=True, color="black"
                        )
                    )
            elif gate_name == "SWAP":
                ax.plot(
                    [x, x],
                    [qubit * wire_spacing, (qubit + 1) * wire_spacing],
                    "k-",
                    linewidth=2,
                )
                ax.add_patch(
                    plt.Circle((x, qubit * wire_spacing), 0.1, fill=True, color="black")
                )
                ax.add_patch(
                    plt.Circle(
                        (x, (qubit + 1) * wire_spacing), 0.1, fill=True, color="black"
                    )
                )
            else:
                rect = patches.FancyBboxPatch(
                    (x - gate_width / 2, qubit * wire_spacing - gate_width / 2),
                    gate_width,
                    gate_width,
                    boxstyle="round,pad=0.05",
                    facecolor="lightblue",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(
                    x,
                    qubit * wire_spacing,
                    gate_name,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_xlim(-1, len(gates) + 2)
        ax.set_ylim(-0.5, n_qubits * wire_spacing - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def plot_gate_sequence(
        self, gates: list[str], title: str = "Gate Sequence"
    ) -> tuple:
        """
        绘制门序列时间线

        Args:
            gates: 门名称列表
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.Set3(np.linspace(0, 1, len(set(gates))))
        gate_colors = {g: colors[i] for i, g in enumerate(set(gates))}

        for i, gate in enumerate(gates):
            rect = patches.FancyBboxPatch(
                (i * 1.2, 0.3),
                1,
                0.4,
                boxstyle="round,pad=0.05",
                facecolor=gate_colors[gate],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(i * 1.2 + 0.5, 0.5, gate, ha="center", va="center", fontsize=12)

        ax.set_xlim(-0.5, len(gates) * 1.2 + 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title)

        plt.tight_layout()
        return fig, ax

    def plot_gate_decomposition(
        self,
        target_gate: np.ndarray,
        decomposed_gates: list[str],
        title: str = "Gate Decomposition",
    ) -> tuple:
        """
        绘制门分解

        Args:
            target_gate: 目标门矩阵
            decomposed_gates: 分解后的门序列
            title: 标题

        Returns:
            (fig, axes)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        self.plot_gate_matrix(target_gate, title="Target Gate", ax=axes[0])
        self.plot_gate_sequence(decomposed_gates, ax=axes[1])

        fig.suptitle(title)
        plt.tight_layout()

        return fig, axes


class FidelityVisualizer:
    """
    保真度可视化器
    """

    def __init__(self, figsize: tuple[float, float] = (10, 6)):
        _check_matplotlib()
        self.figsize = figsize

    def fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        """
        计算保真度 F(ρ, σ) = Tr(√√ρ σ √√ρ)

        Args:
            rho: 密度矩阵
            sigma: 目标密度矩阵

        Returns:
            保真度值
        """
        try:
            from scipy.linalg import sqrtm

            sqrt_rho = sqrtm(rho)
            product = sqrt_rho @ sigma @ sqrt_rho
            sqrt_product = sqrtm(product)
            return float(np.real(np.trace(sqrt_product)))
        except ImportError:
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            eigenvalues = np.maximum(eigenvalues, 0)
            sqrt_rho = (
                eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
            )
            product = sqrt_rho @ sigma @ sqrt_rho
            eigenvalues_p, eigenvectors_p = np.linalg.eigh(product)
            eigenvalues_p = np.maximum(eigenvalues_p, 0)
            sqrt_product = (
                eigenvectors_p
                @ np.diag(np.sqrt(eigenvalues_p))
                @ eigenvectors_p.conj().T
            )
            return float(np.real(np.trace(sqrt_product)))

    def plot_fidelity_evolution(
        self,
        states: list[np.ndarray],
        target: np.ndarray,
        times: list[float] | None = None,
        title: str = "Fidelity Evolution",
    ) -> tuple:
        """
        绘制保真度随时间演化

        Args:
            states: 密度矩阵时间序列
            target: 目标密度矩阵
            times: 时间序列
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if times is None:
            times = list(range(len(states)))

        fidelities = [self.fidelity(rho, target) for rho in states]

        ax.plot(times, fidelities, "o-", color="purple", linewidth=2, markersize=4)
        ax.fill_between(times, fidelities, alpha=0.3, color="purple")

        ax.set_xlabel("Time")
        ax.set_ylabel("Fidelity")
        ax.set_title(title)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        ax.axhline(
            y=1.0, color="red", linestyle="--", alpha=0.5, label="Perfect Fidelity"
        )
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_fidelity_heatmap(
        self,
        states_a: list[np.ndarray],
        states_b: list[np.ndarray],
        labels_a: list[str] | None = None,
        labels_b: list[str] | None = None,
        title: str = "Fidelity Matrix",
    ) -> tuple:
        """
        绘制保真度热力图

        Args:
            states_a: 第一组密度矩阵
            states_b: 第二组密度矩阵
            labels_a, labels_b: 标签

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        fidelity_matrix = np.zeros((len(states_a), len(states_b)))

        for i, rho in enumerate(states_a):
            for j, sigma in enumerate(states_b):
                fidelity_matrix[i, j] = self.fidelity(rho, sigma)

        im = ax.imshow(fidelity_matrix, cmap="YlOrRd", vmin=0, vmax=1)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Fidelity")

        ax.set_xlabel("States B")
        ax.set_ylabel("States A")
        ax.set_title(title)

        if labels_a:
            ax.set_yticks(range(len(states_a)))
            ax.set_yticklabels(labels_a)
        if labels_b:
            ax.set_xticks(range(len(states_b)))
            ax.set_xticklabels(labels_b, rotation=45)

        plt.tight_layout()
        return fig, ax

    def plot_population_dynamics(
        self,
        states: list[np.ndarray],
        labels: list[str] | None = None,
        title: str = "Population Dynamics",
    ) -> tuple:
        """
        绘制布居数演化

        Args:
            states: 密度矩阵时间序列
            labels: 标签
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        n_levels = states[0].shape[0]
        times = list(range(len(states)))

        populations = np.zeros((len(states), n_levels))
        for i, rho in enumerate(states):
            populations[i] = np.real(np.diag(rho))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_levels))

        for i in range(n_levels):
            label = labels[i] if labels and i < len(labels) else f"|{i}⟩"
            ax.plot(
                times,
                populations[:, i],
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=3,
                label=label,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Population")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        return fig, ax


def plot_gate(gate: str, title: str | None = None) -> tuple:
    """
    绘制量子门

    Args:
        gate: 门名称 ('X', 'Y', 'Z', 'H', 'S', 'T', 'CNOT', 'SWAP')
        title: 标题

    Returns:
        (fig, axes)
    """
    viz = QuantumGateVisualizer()
    if gate not in GATE_MATRICES:
        raise ValueError(f"Unknown gate: {gate}")
    return viz.plot_gate_matrix(GATE_MATRICES[gate], gate, title)


def plot_circuit(
    gates: list[tuple[str, int]], n_qubits: int = 2, title: str = "Quantum Circuit"
) -> tuple:
    """
    绘制量子电路

    Args:
        gates: 门列表
        n_qubits: 量子比特数
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = QuantumGateVisualizer()
    return viz.plot_circuit(gates, n_qubits, title)


def plot_fidelity(
    states: list[np.ndarray],
    target: np.ndarray,
    times: list[float] | None = None,
    title: str = "Fidelity Evolution",
) -> tuple:
    """
    绘制保真度演化

    Args:
        states: 密度矩阵列表
        target: 目标态
        times: 时间列表
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = FidelityVisualizer()
    return viz.plot_fidelity_evolution(states, target, times, title)
