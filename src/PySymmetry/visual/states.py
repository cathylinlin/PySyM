"""
量子态可视化模块

提供量子态的可视化功能：
- Bloch 球表示
- 态矢量图
- 量子态演化轨迹

依赖：numpy, matplotlib
"""

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import to_rgba
except ImportError:
    plt = None
    Axes3D = None
    to_rgba = None


def _check_matplotlib():
    """检查matplotlib是否可用"""
    if plt is None:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")


class BlochSphere:
    """
    Bloch 球可视化器
    
    用于可视化单比特量子态在Bloch球上的表示。
    
    Args:
        figsize: 图形大小，默认为(10, 10)
        title: 标题
    """
    
    def __init__(self, figsize: Tuple[float, float] = (10, 10), title: str = "Bloch Sphere"):
        _check_matplotlib()
        self.figsize = figsize
        self.title = title
        self._figure = None
        self._axes = None
    
    def _setup_axes(self):
        """设置3D坐标轴"""
        self._figure = plt.figure(figsize=self.figsize)
        self._axes = self._figure.add_subplot(111, projection='3d')
        self._axes.set_xlim([-1.5, 1.5])
        self._axes.set_ylim([-1.5, 1.5])
        self._axes.set_zlim([-1.5, 1.5])
        self._axes.set_xlabel('x')
        self._axes.set_ylabel('y')
        self._axes.set_zlabel('z')
        self._axes.set_title(self.title)
    
    def _draw_sphere(self, alpha: float = 0.2, color: str = 'lightblue'):
        """绘制球面"""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self._axes.plot_surface(x, y, z, alpha=alpha, color=color)
    
    def _draw_axes(self):
        """绘制坐标轴"""
        self._axes.plot([-1.5, 1.5], [0, 0], [0, 0], 'k-', linewidth=0.5)
        self._axes.plot([0, 0], [-1.5, 1.5], [0, 0], 'k-', linewidth=0.5)
        self._axes.plot([0, 0], [0, 0], [-1.5, 1.5], 'k-', linewidth=0.5)
        self._axes.text(1.6, 0, 0, '|0⟩')
        self._axes.text(0, 1.6, 0, '+y')
        self._axes.text(0, 0, 1.6, '+z')
    
    def _draw_circle(self, axis: str, alpha: float = 0.3, color: str = 'gray'):
        """绘制大圆"""
        if axis == 'xy':
            theta = np.linspace(0, 2*np.pi, 50)
            self._axes.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), 
                           alpha=alpha, color=color)
        elif axis == 'xz':
            theta = np.linspace(0, 2*np.pi, 50)
            self._axes.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta),
                           alpha=alpha, color=color)
        elif axis == 'yz':
            theta = np.linspace(0, 2*np.pi, 50)
            self._axes.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta),
                           alpha=alpha, color=color)
    
    def add_state(self, state: Union[np.ndarray, List[complex]], 
                  label: Optional[str] = None,
                  color: str = 'red',
                  arrow: bool = True):
        """
        在Bloch球上添加量子态
        
        Args:
            state: 量子态向量 [alpha, beta]，或者Bloch向量 [x, y, z]
            label: 标签
            color: 颜色
            arrow: 是否绘制箭头
        """
        if self._figure is None:
            self._setup_axes()
            self._draw_sphere()
            self._draw_axes()
            self._draw_circle('xy')
            self._draw_circle('xz')
            self._draw_circle('yz')
        
        if isinstance(state, (list, np.ndarray)):
            if len(state) == 2:
                a, b = complex(state[0]), complex(state[1])
                theta = 2 * np.arccos(np.abs(a))
                phi = np.angle(b) - np.angle(a)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
            else:
                x, y, z = state[0], state[1], state[2]
        else:
            return
        
        if arrow:
            self._axes.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.1, linewidth=2)
        
        if label:
            self._axes.text(x*1.2, y*1.2, z*1.2, label)
    
    def show(self):
        """显示图形"""
        if self._figure is None:
            self._setup_axes()
            self._draw_sphere()
            self._draw_axes()
            self._draw_circle('xy')
            self._draw_circle('xz')
            self._draw_circle('yz')
        plt.show()
    
    def save(self, filename: str, dpi: int = 300):
        """保存图形"""
        if self._figure is None:
            self.show()
        self._figure.savefig(filename, dpi=dpi)


def bloch_vector(ket: np.ndarray) -> np.ndarray:
    """
    计算Bloch向量
    
    Args:
        ket: 二比特量子态向量 [alpha, beta]
    
    Returns:
        Bloch向量 [x, y, z]
    """
    if len(ket) != 2:
        raise ValueError("只支持二比特态")
    
    a, b = complex(ket[0]), complex(ket[1])
    rho = np.array([
        [abs(a)**2, a * np.conj(b)],
        [b * np.conj(a), abs(b)**2]
    ])
    
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    
    return np.array([x, y, z])


class StateVectorPlotter:
    """
    态矢量图绘制器
    
    在复平面上绘制量子态的振幅和相位。
    """
    
    def __init__(self, figsize: Tuple[float, float] = (8, 8)):
        _check_matplotlib()
        self.figsize = figsize
    
    def plot(self, states: List[np.ndarray], 
             labels: Optional[List[str]] = None,
             title: str = "Quantum State Vectors"):
        """
        绘制多个量子态的矢量图
        
        Args:
            states: 量子态向量列表
            labels: 标签列表
            title: 标题
        """
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw={'projection': 'polar'})
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
        
        for i, state in enumerate(states):
            label = labels[i] if labels and i < len(labels) else f"|{i}⟩"
            
            for j, amp in enumerate(state):
                r = abs(amp)
                theta = np.angle(amp)
                ax.plot([0, theta], [0, r], color=colors[i], linewidth=2, label=f"{label}: c{j}")
                ax.plot([theta], [r], 'o', color=colors[i], markersize=8)
        
        ax.set_title(title)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(True)
        
        if labels:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        return fig, ax
    
    def show(self):
        plt.show()


def plot_bloch_sphere(states: List[np.ndarray], 
                       labels: Optional[List[str]] = None,
                       title: str = "Bloch Sphere Representation") -> BlochSphere:
    """
    绘制Bloch球表示
    
    Args:
        states: 量子态列表
        labels: 标签列表
        title: 标题
    
    Returns:
        BlochSphere对象
    """
    bs = BlochSphere(title=title)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for i, state in enumerate(states):
        label = None
        if labels and i < len(labels):
            label = labels[i]
        color = colors[i % len(colors)]
        bs.add_state(state, label=label, color=color)
    
    return bs


def plot_state_vectors(states: List[np.ndarray],
                       labels: Optional[List[str]] = None,
                       title: str = "State Vectors") -> Tuple:
    """
    绘制态矢量图
    
    Args:
        states: 量子态列表
        labels: 标签列表
        title: 标题
    
    Returns:
        (fig, ax)
    """
    plotter = StateVectorPlotter()
    return plotter.plot(states, labels, title)


def plot_bloch_trajectory(trajectory: List[np.ndarray],
                          label: str = "Trajectory",
                          color: str = 'blue') -> BlochSphere:
    """
    绘制Bloch球上的演化轨迹
    
    Args:
        trajectory: 量子态轨迹（时间序列）
        label: 标签
        color: 颜色
    
    Returns:
        BlochSphere对象
    """
    bs = BlochSphere()
    
    for i, state in enumerate(trajectory):
        bs.add_state(state, label=f"t={i}" if i % 10 == 0 else None, 
                    color=color, arrow=False)
    
    if len(trajectory) > 1:
        points = np.array([bloch_vector(s) if len(s) == 2 else s for s in trajectory])
        bs._axes.plot(points[:, 0], points[:, 1], points[:, 2], 
                     color=color, linewidth=1, alpha=0.5)
    
    return bs