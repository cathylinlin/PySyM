"""
能量谱可视化模块

提供能量谱和能级图的可视化功能：
- 能级图
- 能量阶梯图
- 简并度可视化

依赖：numpy, matplotlib
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    plt = None
    Rectangle = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


class EnergySpectrumPlotter:
    """
    能量谱绘制器
    
    用于绘制量子系统的能量谱。
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        _check_matplotlib()
        self.figsize = figsize
    
    def plot_levels(self, energies: np.ndarray,
                   labels: Optional[List[str]] = None,
                   title: str = "Energy Spectrum",
                   ylabel: str = "Energy",
                   show_values: bool = True) -> Tuple:
        """
        绘制能级图
        
        Args:
            energies: 能量值数组
            labels: 标签列表
            title: 标题
            ylabel: y轴标签
            show_values: 是否显示能量值
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n = len(energies)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
        
        for i, E in enumerate(energies):
            ymin = min(energies) - 0.1 * abs(min(energies)) if min(energies) != max(energies) else -1
            ymax = max(energies) + 0.1 * abs(max(energies)) if min(energies) != max(energies) else 1
            
            line = ax.hlines(E, 0.3, 0.7, colors=colors[i], linewidth=3)
            
            if show_values:
                label = labels[i] if labels and i < len(labels) else f'n={i}'
                ax.text(0.75, E, f'{E:.4f}', va='center', fontsize=9)
                ax.text(0.05, E, label, va='center', ha='left', fontsize=9)
        
        ax.set_xlim(0, 1)
        y_margin = (max(energies) - min(energies)) * 0.1 if len(energies) > 1 else 1
        ax.set_ylim(min(energies) - y_margin, max(energies) + y_margin)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_ladder(self, energies: np.ndarray,
                   energies_2: Optional[np.ndarray] = None,
                   title: str = "Energy Ladder",
                   labels: Optional[List[str]] = None) -> Tuple:
        """
        绘制能级阶梯图
        
        Args:
            energies: 主能量谱
            energies_2: 可选第二个能量谱（如自旋向下）
            title: 标题
            labels: 标签
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n = len(energies)
        y_positions = np.arange(n)
        
        ax.barh(y_positions, energies, height=0.6, 
                label=labels[0] if labels else 'Energy', 
                color='steelblue', alpha=0.8)
        
        if energies_2 is not None:
            ax.barh(y_positions + 0.3, energies_2, height=0.4,
                   label=labels[1] if labels and len(labels) > 1 else 'Energy 2',
                   color='coral', alpha=0.8)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'n={i}' for i in range(n)])
        ax.set_xlabel('Energy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_spectrum_with_degeneracy(self, energies: np.ndarray,
                                      degeneracies: Optional[List[int]] = None,
                                      title: str = "Energy Spectrum with Degeneracy") -> Tuple:
        """
        绘制带简并度的能谱
        
        Args:
            energies: 能量值
            degeneracies: 每个能级的简并度
            title: 标题
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if degeneracies is None:
            degeneracies = [1] * len(energies)
        
        unique_energies = []
        unique_degeneracies = []
        
        for i, E in enumerate(energies):
            if E not in unique_energies:
                unique_energies.append(E)
                unique_degeneracies.append(degeneracies[i])
            else:
                idx = unique_energies.index(E)
                unique_degeneracies[idx] += degeneracies[i]
        
        x_positions = np.arange(len(unique_energies))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(unique_energies)))
        
        bars = ax.bar(x_positions, unique_energies, color=colors, alpha=0.8)
        
        for i, (bar, deg) in enumerate(zip(bars, unique_degeneracies)):
            if deg > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'g={deg}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'E{i}' for i in range(len(unique_energies))])
        ax.set_ylabel('Energy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_comparison(self, energies_list: List[np.ndarray],
                       labels: Optional[List[str]] = None,
                       title: str = "Energy Spectrum Comparison") -> Tuple:
        """
        绘制多个能量谱对比
        
        Args:
            energies_list: 能量谱列表
            labels: 标签列表
            title: 标题
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(energies_list)))
        
        max_states = max(len(e) for e in energies_list)
        
        for idx, energies in enumerate(energies_list):
            label = labels[idx] if labels and idx < len(labels) else f'Spectrum {idx}'
            
            for i, E in enumerate(energies):
                offset = idx * 0.2
                ax.scatter(i + offset, E, color=colors[idx], s=100, label=label if i == 0 else '', zorder=3)
                
                if i < len(energies) - 1:
                    ax.plot([i + offset, i + offset + 0.8], [E, energies[i+1]], 
                           color=colors[idx], alpha=0.3, linewidth=1)
        
        ax.set_xlabel('State Index')
        ax.set_ylabel('Energy')
        ax.set_title(title)
        
        if labels:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_diagram(self, energies: np.ndarray,
                    transitions: Optional[List[Tuple[int, int, float]]] = None,
                    title: str = "Energy Level Diagram") -> Tuple:
        """
        绘制能级跃迁图
        
        Args:
            energies: 能量值
            transitions: 跃迁列表 [(from, to, wavelength), ...]
            title: 标题
        
        Returns:
            (fig, ax)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sorted_indices = np.argsort(energies)
        
        y_positions = {idx: i for i, idx in enumerate(sorted_indices)}
        
        for i, idx in enumerate(sorted_indices):
            E = energies[idx]
            ax.hlines(E, 0.3, 0.7, colors='steelblue', linewidth=2)
            ax.text(0.05, E, f'n={idx}', va='center', ha='left', fontsize=9)
            ax.text(0.75, E, f'{E:.4f}', va='center', ha='left', fontsize=9)
        
        if transitions:
            for from_idx, to_idx, wavelength in transitions:
                from_E = energies[from_idx]
                to_E = energies[to_idx]
                
                if abs(from_E - to_E) > 0.001:
                    ax.annotate('', xy=(0.85, to_E), xytext=(0.85, from_E),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
                    ax.text(0.9, (from_E + to_E)/2, f'λ={wavelength:.2f}', 
                           va='center', fontsize=8, color='red')
        
        ax.set_xlim(0, 1.1)
        ax.set_ylim(min(energies) - 0.5, max(energies) + 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Energy')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax


def plot_energy_levels(energies: np.ndarray,
                       labels: Optional[List[str]] = None,
                       title: str = "Energy Levels") -> Tuple:
    """
    绘制能级图
    
    Args:
        energies: 能量值数组
        labels: 标签列表
        title: 标题
    
    Returns:
        (fig, ax)
    """
    plotter = EnergySpectrumPlotter()
    return plotter.plot_levels(energies, labels, title)


def plot_energy_spectrum(energies: np.ndarray,
                         degeneracies: Optional[List[int]] = None,
                         title: str = "Energy Spectrum") -> Tuple:
    """
    绘制能量谱
    
    Args:
        energies: 能量值数组
        degeneracies: 简并度列表
        title: 标题
    
    Returns:
        (fig, ax)
    """
    plotter = EnergySpectrumPlotter()
    return plotter.plot_spectrum_with_degeneracy(energies, degeneracies, title)


def plot_energy_diagram(energies: np.ndarray,
                        transitions: Optional[List[Tuple[int, int, float]]] = None,
                        title: str = "Energy Level Diagram") -> Tuple:
    """
    绘制能级图
    
    Args:
        energies: 能量值数组
        transitions: 跃迁列表
        title: 标题
    
    Returns:
        (fig, ax)
    """
    plotter = EnergySpectrumPlotter()
    return plotter.plot_diagram(energies, transitions, title)