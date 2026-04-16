"""
可视化工具模块

提供可视化所需的通用工具：
- 颜色方案
- 样式预设
- 辅助函数
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, hex2color
except ImportError:
    plt = None
    LinearSegmentedColormap = None
    hex2color = None


QUANTUM_COLORSCHEME = {
    "quantum_red": "#E74C3C",
    "quantum_blue": "#3498DB",
    "quantum_green": "#2ECC71",
    "quantum_purple": "#9B59B6",
    "quantum_orange": "#E67E22",
    "quantum_cyan": "#1ABC9C",
    "quantum_pink": "#E91E63",
    "quantum_yellow": "#F1C40F",
    "quantum_gray": "#95A5A6",
    "quantum_dark": "#2C3E50",
}

STATE_COLORS = {
    "|0⟩": "#E74C3C",
    "|1⟩": "#3498DB",
    "|+⟩": "#2ECC71",
    "|-⟩": "#9B59B6",
    "|+i⟩": "#E67E22",
    "|-i⟩": "#1ABC9C",
}

GATE_COLORS = {
    "H": "#E74C3C",
    "X": "#3498DB",
    "Y": "#2ECC71",
    "Z": "#9B59B6",
    "CNOT": "#E67E22",
    "SWAP": "#1ABC9C",
    "T": "#E91E63",
    "S": "#F1C40F",
}


class QuantumColormap:
    """量子色图"""

    @staticmethod
    def probability() -> LinearSegmentedColormap:
        """概率分布色图（蓝到红）"""
        colors = ["#FFFFFF", "#3498DB", "#2980B9", "#1A5276", "#E74C3C"]
        return LinearSegmentedColormap.from_list("probability", colors)

    @staticmethod
    def phase() -> LinearSegmentedColormap:
        """相位色图（彩虹色）"""
        colors = ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]
        return LinearSegmentedColormap.from_list("phase", colors)

    @staticmethod
    def entanglement() -> LinearSegmentedColormap:
        """纠缠度量色图（绿到红）"""
        colors = ["#2ECC71", "#F1C40F", "#E74C3C"]
        return LinearSegmentedColormap.from_list("entanglement", colors)

    @staticmethod
    def energy() -> LinearSegmentedColormap:
        """能级色图"""
        colors = ["#1ABC9C", "#3498DB", "#9B59B6", "#E74C3C"]
        return LinearSegmentedColormap.from_list("energy", colors)


class PlotStyle:
    """绘图样式预设"""

    PAPER = {
        "figure.figsize": (8, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }

    PRESENTATION = {
        "figure.figsize": (12, 8),
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.5,
    }

    POSTER = {
        "figure.figsize": (16, 12),
        "font.size": 24,
        "axes.titlesize": 32,
        "axes.labelsize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "axes.linewidth": 3,
        "axes.grid": True,
        "grid.alpha": 0.7,
    }

    DARK = {
        "figure.facecolor": "#2C3E50",
        "axes.facecolor": "#34495E",
        "text.color": "#ECF0F1",
        "axes.labelcolor": "#ECF0F1",
        "xtick.color": "#ECF0F1",
        "ytick.color": "#ECF0F1",
        "axes.edgecolor": "#BDC3C7",
        "grid.color": "#7F8C8D",
        "grid.alpha": 0.3,
    }

    @staticmethod
    def apply(style: str = "paper") -> None:
        """应用样式预设"""
        if plt is None:
            return

        styles = {
            "paper": PlotStyle.PAPER,
            "presentation": PlotStyle.PRESENTATION,
            "poster": PlotStyle.POSTER,
            "dark": PlotStyle.DARK,
        }

        if style in styles:
            plt.rcParams.update(styles[style])

    @staticmethod
    def reset() -> None:
        """重置为默认样式"""
        if plt is not None:
            plt.rcParams.update(plt.rcParamsDefault)


def setup_axes(
    ax,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    grid: bool = True,
) -> None:
    """
    设置坐标轴属性

    Args:
        ax: matplotlib axes 对象
        xlabel: x轴标签
        ylabel: y轴标签
        zlabel: z轴标签
        title: 标题
        xlim: x轴范围
        ylim: y轴范围
        zlim: z轴范围
        grid: 是否显示网格
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)
    if grid:
        ax.grid(True)


def color_by_probability(probs: np.ndarray) -> list[str]:
    """
    根据概率返回颜色列表

    Args:
        probs: 概率分布

    Returns:
        颜色列表
    """
    cmap = QuantumColormap.probability()
    norm_probs = probs / probs.max() if probs.max() > 0 else probs
    return [cmap(p) for p in norm_probs]


def color_by_phase(phases: np.ndarray) -> list[str]:
    """
    根据相位返回颜色列表

    Args:
        phases: 相位数组

    Returns:
        颜色列表
    """
    cmap = QuantumColormap.phase()
    norm_phases = (phases + np.pi) / (2 * np.pi)
    return [cmap(p % 1) for p in norm_phases]


def legend_outside(ax, loc: str = "right", pad: float = 0.02) -> None:
    """
    将图例放到图形外部

    Args:
        ax: matplotlib axes 对象
        loc: 位置 ('right', 'bottom')
        pad: 内边距
    """
    if loc == "right":
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    elif loc == "bottom":
        ax.legend(
            bbox_to_anchor=(0.5, -0.15),
            loc="upper center",
            ncol=min(4, len(ax.get_legend_handles_labels()[0])),
        )


def save_figure(
    fig,
    filename: str,
    formats: list[str] | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    """
    保存图形到多种格式

    Args:
        fig: matplotlib figure 对象
        filename: 文件名（不含扩展名）
        formats: 格式列表，如 ['png', 'pdf', 'svg']
        dpi: 分辨率
        transparent: 是否透明背景
    """
    if plt is None:
        return

    if formats is None:
        formats = ["png"]

    for fmt in formats:
        filepath = f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches="tight")
        print(f"Saved: {filepath}")


def create_subplots(
    n_plots: int,
    n_cols: int = 2,
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple:
    """
    创建子图网格

    Args:
        n_plots: 子图数量
        n_cols: 列数
        figsize: 图形大小
        sharex: 是否共享x轴
        sharey: 是否共享y轴

    Returns:
        (fig, axes)
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey
    )

    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = np.atleast_1d(axes)

    for i in range(n_plots, len(axes.flat)):
        axes.flat[i].set_visible(False)

    return fig, axes


__all__ = [
    "QUANTUM_COLORSCHEME",
    "STATE_COLORS",
    "GATE_COLORS",
    "QuantumColormap",
    "PlotStyle",
    "setup_axes",
    "color_by_probability",
    "color_by_phase",
    "legend_outside",
    "save_figure",
    "create_subplots",
]
