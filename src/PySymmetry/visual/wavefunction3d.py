"""
3D波函数可视化模块

提供3D波函数和概率密度的可视化：
- 等值面图
- 切片图
- 原子轨道可视化

依赖：numpy, matplotlib, mayavi (可选)
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None
    Axes3D = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


class Wavefunction3DVisualizer:
    """
    3D波函数可视化器

    用于可视化3D空间中的波函数和概率密度。
    """

    def __init__(self, figsize: tuple[float, float] = (10, 8)):
        _check_matplotlib()
        self.figsize = figsize

    def plot_isosurface(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        isolevel: float = 0.1,
        title: str = "3D Isosurface",
        cmap: str = "viridis",
        ax=None,
    ) -> tuple:
        """
        绘制3D等值面

        Args:
            x, y, z: 3D网格
            values: 函数值
            isolevel: 等值面水平
            title: 标题
            cmap: 颜色映射
            ax: 坐标轴

        Returns:
            (fig, ax)
        """
        _check_matplotlib()

        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        values_3d = values.reshape(len(x), len(y), len(z))

        slice_idx = len(z) // 2
        ax.contour3D(x, y, values_3d[:, :, slice_idx], levels=15, cmap=cmap)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        return fig, ax

    def plot_slices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        slice_position: str = "middle",
        title: str = "Wavefunction Slices",
    ) -> tuple:
        """
        绘制3D波函数的正交切片

        Args:
            x, y, z: 3D网格
            values: 波函数值
            slice_position: 'middle', 'all'
            title: 标题

        Returns:
            (fig, axes)
        """
        _check_matplotlib()

        values_3d = values.reshape(len(x), len(y), len(z))

        nx, ny, nz = len(x), len(y), len(z)

        if slice_position == "all":
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            sx, sy, sz = nx // 2, ny // 2, nz // 2

            im0 = axes[0].contourf(y, z, values_3d[sx, :, :], levels=15, cmap="viridis")
            axes[0].set_xlabel("Y")
            axes[0].set_ylabel("Z")
            axes[0].set_title(f"X = {x[sx]:.2f}")
            axes[0].set_aspect("equal")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].contourf(x, z, values_3d[:, sy, :], levels=15, cmap="viridis")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Z")
            axes[1].set_title(f"Y = {y[sy]:.2f}")
            axes[1].set_aspect("equal")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].contourf(x, y, values_3d[:, :, sz], levels=15, cmap="viridis")
            axes[2].set_xlabel("X")
            axes[2].set_ylabel("Y")
            axes[2].set_title(f"Z = {z[sz]:.2f}")
            axes[2].set_aspect("equal")
            plt.colorbar(im2, ax=axes[2])

            plt.suptitle(title)
            plt.tight_layout()

            return fig, axes
        else:
            fig, ax = plt.subplots(figsize=self.figsize)

            sx, sy = nx // 2, ny // 2
            im = ax.contourf(x, y, values_3d[:, :, nz // 2], levels=15, cmap="viridis")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(title)
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax)

            plt.tight_layout()
            return fig, ax

    def plot_surface_3d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        title: str = "3D Surface",
        cmap: str = "coolwarm",
        projection: str | None = None,
    ) -> tuple:
        """
        绘制3D表面图

        Args:
            x, y, z: 3D网格
            values: 函数值
            title: 标题
            cmap: 颜色映射
            projection: 投影类型

        Returns:
            (fig, ax)
        """
        _check_matplotlib()

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        values_3d = values.reshape(len(x), len(y), len(z))

        slice_idx = len(z) // 2
        Z = values_3d[:, :, slice_idx]

        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap, alpha=0.8, linewidth=0, antialiased=True
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Value")
        ax.set_title(title)

        fig.colorbar(surf, ax=ax, shrink=0.5)

        plt.tight_layout()
        return fig, ax

    def plot_volume_render(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        title: str = "Volume Rendering",
    ) -> tuple:
        """
        绘制体绘制（使用alpha通道模拟）

        Args:
            x, y, z: 3D网格
            values: 函数值
            title: 标题

        Returns:
            (fig, ax)
        """
        _check_matplotlib()

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        values_3d = values.reshape(len(x), len(y), len(z))

        nx, ny, nz = len(x), len(y), len(z)

        levels = 5
        for i in range(levels):
            level = np.percentile(values_3d, 20 + i * 15)

            slice_idx = nz // 2
            ax.contourf(
                x,
                y,
                values_3d[:, :, slice_idx],
                levels=[level - 0.01, level + 0.01],
                alpha=0.3,
                colors=[plt.cm.viridis(i / levels)],
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        plt.tight_layout()
        return fig, ax


def hydrogen_orbital(
    n: int, l: int, m: int, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    计算氢原子轨道波函数

    Args:
        n, l, m: 量子数
        x, y, z: 位置网格

    Returns:
        波函数值
    """
    try:
        from scipy.special import sph_harm_y as spherical_harmonic
    except ImportError:
        from scipy.special import sph_harm as spherical_harmonic
    from scipy.special import assoc_laguerre, factorial

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-10))
    phi = np.arctan2(y, x)

    a0 = 1.0

    rho = 2 * r / (n * a0)

    prefactor = np.sqrt(
        (2 / (n * a0)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))
    )

    laguerre = assoc_laguerre(rho, n - l - 1, 2 * l + 1)

    radial = prefactor * np.exp(-rho / 2) * rho**l * laguerre

    try:
        Y_lm = spherical_harmonic(m, l, theta, phi)
    except (NameError, TypeError):
        Y_lm = spherical_harmonic(m, l, phi, theta)

    psi = radial * Y_lm

    return np.abs(psi) ** 2


def plot_3d_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    isolevel: float = 0.1,
    title: str = "3D Wavefunction",
) -> tuple:
    """
    绘制3D波函数

    Args:
        x, y, z: 网格
        psi: 波函数值
        isolevel: 等值面水平
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = Wavefunction3DVisualizer()
    return viz.plot_isosurface(x, y, z, np.abs(psi) ** 2, isolevel, title)


def plot_3d_probability_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    probability: np.ndarray,
    isolevel: float = 0.1,
    title: str = "3D Probability Isosurface",
) -> tuple:
    """
    绘制3D概率密度等值面

    Args:
        x, y, z: 网格
        probability: 概率密度
        isolevel: 等值面水平
        title: 标题

    Returns:
        (fig, ax)
    """
    viz = Wavefunction3DVisualizer()
    return viz.plot_isosurface(x, y, z, probability, isolevel, title)


def plot_3d_slices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    title: str = "3D Wavefunction Slices",
) -> tuple:
    """
    绘制3D切片

    Args:
        x, y, z: 网格
        values: 函数值
        title: 标题

    Returns:
        (fig, axes)
    """
    viz = Wavefunction3DVisualizer()
    return viz.plot_slices(x, y, z, values, slice_position="all", title=title)


def plot_orbital(
    n: int, l: int, m: int, r_max: float = 10.0, num_points: int = 50, view: str = "3d"
) -> tuple:
    """
    绘制氢原子轨道

    Args:
        n, l, m: 量子数
        r_max: 最大半径
        num_points: 网格点数
        view: '3d', 'slices'

    Returns:
        (fig, ax) or (fig, axes)
    """
    x = np.linspace(-r_max, r_max, num_points)
    y = np.linspace(-r_max, r_max, num_points)
    z = np.linspace(-r_max, r_max, num_points)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    prob = hydrogen_orbital(n, l, m, X, Y, Z)

    title = f"Hydrogen Orbital (n={n}, l={l}, m={m})"

    if view == "3d":
        viz = Wavefunction3DVisualizer()
        return viz.plot_slices(x, y, z, prob, slice_position="middle", title=title)
    else:
        viz = Wavefunction3DVisualizer()
        return viz.plot_slices(x, y, z, prob, slice_position="all", title=title)
