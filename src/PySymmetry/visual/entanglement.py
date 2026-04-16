"""
纠缠态与量子关联可视化模块

提供纠缠态和量子关联的可视化：
- 纠缠态表示
- Von Neumann熵
- 量子关联度量
- Schmidt分解可视化

依赖：numpy, matplotlib
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    plt = None
    FancyBboxPatch = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed")


def concurrence(rho: np.ndarray) -> float:
    """
    计算 concurrence（纠缠度量）

    Args:
        rho: 2比特密度矩阵

    Returns:
        concurrence值
    """
    if rho.shape != (4, 4):
        raise ValueError("需要2比特系统的4x4密度矩阵")

    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y2 = np.kron(sigma_y, sigma_y)

    rho_tilde = sigma_y2 @ np.conj(rho) @ sigma_y2

    eigenvalues = np.linalg.eigvals(rho @ rho_tilde)
    eigenvalues = np.sort(np.sqrt(np.abs(eigenvalues)))

    return max(0, eigenvalues[-1] - np.sum(eigenvalues[:-1]))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    计算 Von Neumann 熵

    Args:
        rho: 密度矩阵

    Returns:
        熵值
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def partial_transpose(rho: np.ndarray, subsystem: int = 0) -> np.ndarray:
    """
    计算偏转置

    Args:
        rho: 密度矩阵
        subsystem: 要求偏转置的子系统 (0 or 1)

    Returns:
        偏转置后的矩阵
    """
    d = int(np.sqrt(rho.shape[0]))
    rho_reshaped = rho.reshape(d, d, d, d)

    if subsystem == 0:
        return rho_reshaped.transpose(1, 0, 3, 2).reshape(rho.shape)
    else:
        return rho_reshaped.transpose(0, 1, 3, 2).reshape(rho.shape)


def negativity(rho: np.ndarray, subsystem: int = 0) -> float:
    """
    计算 Negativity（负性度量）

    Args:
        rho: 密度矩阵
        subsystem: 子系统

    Returns:
        Negativity值
    """
    rho_pt = partial_transpose(rho, subsystem)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < 0]))


def schmidt_decomposition(
    state: np.ndarray, dim: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Schmidt分解

    Args:
        state: 复合系统态向量
        dim: 每个子系统的维度

    Returns:
        (schmidt_coeffs, basis_a, basis_b)
    """
    state_reshaped = state.reshape(dim, dim)
    U, s, Vh = np.linalg.svd(state_reshaped)

    return s, U, Vh


class EntanglementVisualizer:
    """
    纠缠态可视化器
    """

    def __init__(self, figsize: tuple[float, float] = (10, 6)):
        _check_matplotlib()
        self.figsize = figsize

    def plot_entanglement_measures(
        self,
        states: list[np.ndarray],
        labels: list[str] | None = None,
        title: str = "Entanglement Measures",
    ) -> tuple:
        """
        绘制多种纠缠度量对比

        Args:
            states: 密度矩阵列表
            labels: 标签列表
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        concurrences = []
        entropies = []
        negativities = []

        for rho in states:
            try:
                concurrences.append(concurrence(rho))
            except:
                concurrences.append(0)
            entropies.append(von_neumann_entropy(rho))
            negativities.append(negativity(rho))

        x = np.arange(len(states))

        axes[0].bar(x, concurrences, color="steelblue", alpha=0.8)
        axes[0].set_ylabel("Concurrence")
        axes[0].set_title("Concurrence")
        axes[0].set_ylim([0, 1])

        axes[1].bar(x, entropies, color="coral", alpha=0.8)
        axes[1].set_ylabel("Von Neumann Entropy")
        axes[1].set_title("Entanglement Entropy")
        axes[1].set_ylim([0, 2])

        axes[2].bar(x, negativities, color="seagreen", alpha=0.8)
        axes[2].set_ylabel("Negativity")
        axes[2].set_title("Negativity")
        axes[2].set_ylim([0, 0.5])

        if labels:
            for ax in axes:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)

        plt.suptitle(title)
        plt.tight_layout()

        return fig, axes

    def plot_entropy_evolution(
        self,
        rho_t: list[np.ndarray],
        times: list[float] | None = None,
        title: str = "Entropy Evolution",
    ) -> tuple:
        """
        绘制熵随时间演化

        Args:
            rho_t: 密度矩阵时间序列
            times: 时间序列
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if times is None:
            times = list(range(len(rho_t)))

        entropies = [von_neumann_entropy(rho) for rho in rho_t]

        ax.plot(times, entropies, "o-", color="coral", linewidth=2, markersize=4)
        ax.fill_between(times, entropies, alpha=0.3, color="coral")

        ax.set_xlabel("Time")
        ax.set_ylabel("Von Neumann Entropy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def plot_schmidt_spectrum(
        self, state: np.ndarray, dim: int = 2, title: str = "Schmidt Spectrum"
    ) -> tuple:
        """
        绘制Schmidt谱

        Args:
            state: 双体态向量
            dim: 子系统维度
            title: 标题

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        coeffs, _, _ = schmidt_decomposition(state, dim)

        x = np.arange(len(coeffs))
        colors = plt.cm.viridis(coeffs / max(coeffs))

        bars = ax.bar(x, coeffs, color=colors, alpha=0.8)

        ax.set_xlabel("Schmidt Index")
        ax.set_ylabel("Schmidt Coefficient")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"λ{i}" for i in range(len(coeffs))])
        ax.grid(True, alpha=0.3, axis="y")

        entropy = -np.sum(coeffs**2 * np.log2(coeffs**2 + 1e-10))
        ax.text(
            0.95,
            0.95,
            f"Entanglement Entropy: {entropy:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig, ax

    def plot_bipartite_state(
        self, state: np.ndarray, title: str = "Bipartite State"
    ) -> tuple:
        """
        绘制双体态的直积结构

        Args:
            state: 态向量

        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        n = int(np.sqrt(len(state)))

        state_matrix = state.reshape(n, n)

        im = ax.imshow(np.abs(state_matrix), cmap="viridis", aspect="auto")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Amplitude|")

        ax.set_xlabel("Subsystem B")
        ax.set_ylabel("Subsystem A")
        ax.set_title(title)

        for i in range(n):
            for j in range(n):
                amp = state_matrix[i, j]
                if np.abs(amp) > 0.1:
                    ax.text(
                        j,
                        i,
                        f"{np.abs(amp):.2f}",
                        ha="center",
                        va="center",
                        color="white" if np.abs(amp) > 0.5 else "black",
                        fontsize=8,
                    )

        plt.tight_layout()
        return fig, ax


def bell_state_visualization(state_idx: int = 0) -> tuple:
    """
    Bell态可视化

    Args:
        state_idx: Bell态索引 (0-3)

    Returns:
        (fig, axes)
    """
    bell_states = [
        (np.array([1, 0, 0, 1]) / np.sqrt(2), "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2"),
        (np.array([1, 0, 0, -1]) / np.sqrt(2), "|Φ⁻⟩ = (|00⟩ - |11⟩)/√2"),
        (np.array([0, 1, 1, 0]) / np.sqrt(2), "|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"),
        (np.array([0, 1, -1, 0]) / np.sqrt(2), "|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"),
    ]

    state, name = bell_states[state_idx]
    rho = np.outer(state, np.conj(state))

    viz = EntanglementVisualizer()
    return viz.plot_bipartite_state(state, title=name)


def plot_entanglement_measures(
    states: list[np.ndarray],
    labels: list[str] | None = None,
    title: str = "Entanglement Measures",
) -> tuple:
    """
    绘制纠缠度量

    Args:
        states: 密度矩阵列表
        labels: 标签列表
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = EntanglementVisualizer()
    return viz.plot_entanglement_measures(states, labels, title)


def plot_entropy_evolution(
    rho_t: list[np.ndarray],
    times: list[float] | None = None,
    title: str = "Entropy Evolution",
) -> tuple:
    """
    绘制熵演化

    Args:
        rho_t: 密度矩阵时间序列
        times: 时间序列
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = EntanglementVisualizer()
    return viz.plot_entropy_evolution(rho_t, times, title)
