"""
概率密度可视化模块

提供概率密度和密度矩阵的可视化功能：
- 1D/2D概率密度图
- 密度矩阵热力图
- 3D等值面图

依赖：numpy, matplotlib
"""

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    plt = None
    LinearSegmentedColormap = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


class ProbabilityDensityPlotter:
    """
    概率密度绘制器
    
    用于绘制1D和2D概率密度分布。
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 6)):
        _check_matplotlib()
        self.figsize = figsize
    
    def plot_1d(self, x: np.ndarray, 
                densities: List[np.ndarray],
                labels: Optional[List[str]] = None,
                title: str = "Probability Density",
                xlabel: str = "Position",
                ylabel: str = "|ψ|²") -> Tuple:
        """
        绘制1D概率密度
        
        Args:
            x: 位置网格
            densities: 概率密度列表
            labels: 标签列表
            title: 标题
            xlabel: x轴标签
            ylabel: y轴标签
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(densities)))
        
        for i, density in enumerate(densities):
            label = labels[i] if labels and i < len(labels) else f"State {i}"
            ax.plot(x, density, color=colors[i], linewidth=2, label=label)
            ax.fill_between(x, density, alpha=0.3, color=colors[i])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_2d_heatmap(self, x: np.ndarray, 
                        y: np.ndarray,
                        density: np.ndarray,
                        title: str = "Probability Density",
                        cmap: str = 'viridis',
                        levels: int = 20) -> Tuple:
        """
        绘制2D概率密度热力图
        
        Args:
            x, y: 网格
            density: 概率密度
            title: 标题
            cmap: 颜色映射
            levels: 等值线级别
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        X, Y = np.meshgrid(x, y)
        Z = density.T if density.shape == (len(y), len(x)) else density
        
        im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.contour(X, Y, Z, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|ψ|²')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_2d_contour(self, x: np.ndarray,
                       y: np.ndarray,
                       density: np.ndarray,
                       num_contours: int = 10,
                       title: str = "Probability Density Contours") -> Tuple:
        """
        绘制2D等值线图
        
        Args:
            x, y: 网格
            density: 概率密度
            num_contours: 等值线数量
            title: 标题
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        X, Y = np.meshgrid(x, y)
        Z = density.T if density.shape == (len(y), len(x)) else density
        
        cs = ax.contour(X, Y, Z, levels=num_contours, cmap='coolwarm')
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax


class DensityMatrixVisualizer:
    """
    密度矩阵可视化器
    
    用于可视化量子态的密度矩阵。
    """
    
    def __init__(self, figsize: Tuple[float, float] = (8, 6)):
        _check_matplotlib()
        self.figsize = figsize
    
    def plot_real(self, rho: np.ndarray,
                 title: str = "Density Matrix (Real Part)",
                 cmap: str = 'RdBu_r') -> Tuple:
        """
        绘制密度矩阵实部
        
        Args:
            rho: 密度矩阵
            title: 标题
            cmap: 颜色映射
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        real_part = np.real(rho)
        
        im = ax.imshow(real_part, cmap=cmap, aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Re(ρ)')
        
        n = rho.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{real_part[i, j]:.2f}',
                              ha='center', va='center', 
                              color='white' if abs(real_part[i, j]) > 0.3 else 'black',
                              fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_imag(self, rho: np.ndarray,
                 title: str = "Density Matrix (Imaginary Part)",
                 cmap: str = 'RdBu_r') -> Tuple:
        """
        绘制密度矩阵虚部
        
        Args:
            rho: 密度矩阵
            title: 标题
            cmap: 颜色映射
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        imag_part = np.imag(rho)
        
        im = ax.imshow(imag_part, cmap=cmap, aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Im(ρ)')
        
        n = rho.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{imag_part[i, j]:.2f}',
                              ha='center', va='center',
                              color='white' if abs(imag_part[i, j]) > 0.2 else 'black',
                              fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_abs(self, rho: np.ndarray,
                title: str = "Density Matrix (Absolute Value)",
                cmap: str = 'viridis') -> Tuple:
        """
        绘制密度矩阵绝对值
        
        Args:
            rho: 密度矩阵
            title: 标题
            cmap: 颜色映射
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        abs_part = np.abs(rho)
        
        im = ax.imshow(abs_part, cmap=cmap, aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|ρ|')
        
        n = rho.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_all(self, rho: np.ndarray) -> Tuple:
        """
        绘制密度矩阵的实部、虚部和绝对值
        
        Args:
            rho: 密度矩阵
        
        Returns:
            (fig, axes)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        real_part = np.real(rho)
        imag_part = np.imag(rho)
        abs_part = np.abs(rho)
        
        im0 = axes[0].imshow(real_part, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_title('Real Part')
        
        im1 = axes[1].imshow(imag_part, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title('Imaginary Part')
        
        im2 = axes[2].imshow(abs_part, cmap='viridis', aspect='auto')
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title('Absolute Value')
        
        for ax in axes:
            n = rho.shape[0]
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
        
        plt.tight_layout()
        return fig, axes
    
    def plot_diagonal(self, rho: np.ndarray,
                     labels: Optional[List[str]] = None,
                     title: str = "Density Matrix Diagonal") -> Tuple:
        """
        绘制密度矩阵对角元（ populations）
        
        Args:
            rho: 密度矩阵
            labels: 标签列表
            title: 标题
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        diagonal = np.real(np.diag(rho))
        n = len(diagonal)
        
        x = range(n)
        colors = plt.cm.viridis(np.linspace(0, 1, n))
        
        ax.bar(x, diagonal, color=colors)
        
        if labels:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([f'|{i}⟩' for i in range(n)])
        
        ax.set_ylabel('Population')
        ax.set_title(title)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax


def plot_probability_density(x: np.ndarray,
                             densities: List[np.ndarray],
                             labels: Optional[List[str]] = None,
                             title: str = "Probability Density") -> Tuple:
    """
    绘制1D概率密度
    
    Args:
        x: 位置网格
        densities: 概率密度列表
        labels: 标签列表
        title: 标题
    
    Returns:
        (fig, ax)
    """
    plotter = ProbabilityDensityPlotter()
    return plotter.plot_1d(x, densities, labels, title)


def plot_density_matrix(rho: np.ndarray,
                        mode: str = 'all',
                        title: Optional[str] = None) -> Tuple:
    """
    绘制密度矩阵
    
    Args:
        rho: 密度矩阵
        mode: 'real', 'imag', 'abs', 'all', 'diagonal'
        title: 标题
    
    Returns:
        (fig, ax) or (fig, axes)
    """
    viz = DensityMatrixVisualizer()
    
    if mode == 'real':
        title = title or "Density Matrix (Real)"
        return viz.plot_real(rho, title)
    elif mode == 'imag':
        title = title or "Density Matrix (Imaginary)"
        return viz.plot_imag(rho, title)
    elif mode == 'abs':
        title = title or "Density Matrix (Absolute)"
        return viz.plot_abs(rho, title)
    elif mode == 'all':
        return viz.plot_all(rho)
    elif mode == 'diagonal':
        title = title or "Density Matrix Diagonal"
        return viz.plot_diagonal(rho, title=title)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def plot_2d_density_heatmap(x: np.ndarray,
                            y: np.ndarray,
                            density: np.ndarray,
                            title: str = "Probability Density",
                            cmap: str = 'viridis') -> Tuple:
    """
    绘制2D概率密度热力图
    
    Args:
        x, y: 网格
        density: 概率密度
        title: 标题
        cmap: 颜色映射
    
    Returns:
        (fig, ax)
    """
    plotter = ProbabilityDensityPlotter()
    return plotter.plot_2d_heatmap(x, y, density, title, cmap)


def plot_3d_density_isosurface(x: np.ndarray,
                              y: np.ndarray,
                              z: np.ndarray,
                              density: np.ndarray,
                              isolevel: float = 0.1,
                              title: str = "3D Probability Isosurface") -> Tuple:
    """
    绘制3D概率密度等值面
    
    Args:
        x, y, z: 3D网格
        density: 概率密度
        isolevel: 等值面水平
        title: 标题
    
    Returns:
        (fig, ax)
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    density_3d = density.reshape(len(x), len(y), len(z))
    
    ax.contour3D(x, y, density_3d[:, :, len(z)//2], levels=20, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig, ax