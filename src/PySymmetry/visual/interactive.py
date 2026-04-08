"""
交互式可视化模块

基于Plotly的交互式可视化：
- 交互式Bloch球
- 交互式能级图
- 交互式3D等值面
- 量子系统仪表盘

依赖：numpy, plotly
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False


def _check_plotly():
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly not installed. Run: pip install plotly")


class InteractivePlotter:
    """
    交互式Plotly绘图器
    """
    
    def __init__(self):
        _check_plotly()
    
    def bloch_sphere(self, states: List[np.ndarray],
                    labels: Optional[List[str]] = None,
                    title: str = "Interactive Bloch Sphere") -> "go.Figure":
        """
        创建交互式Bloch球
        
        Args:
            states: 量子态列表
            labels: 标签列表
            title: 标题
        
        Returns:
            Plotly图形
        """
        _check_plotly()
        
        fig = go.Figure()
        
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, np.pi, 25)
        Theta, Phi = np.meshgrid(theta, phi)
        
        X = np.sin(Phi) * np.cos(Theta)
        Y = np.sin(Phi) * np.sin(Theta)
        Z = np.cos(Phi)
        
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, 
                                opacity=0.3, 
                                colorscale='Gray',
                                showscale=False,
                                name='Sphere',
                                hoverinfo='skip'))
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0, 0], y=[0, 0, 0], z=[1.5, -1.5, 0],
            mode='text',
            text=['|0⟩', '|1⟩', ''],
            textposition='top center',
            showlegend=False
        ))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
        
        for i, state in enumerate(states):
            if len(state) == 2:
                a, b = complex(state[0]), complex(state[1])
                theta_angle = 2 * np.arccos(np.abs(a))
                phi_angle = np.angle(b) - np.angle(a)
                x = np.sin(theta_angle) * np.cos(phi_angle)
                y = np.sin(theta_angle) * np.sin(phi_angle)
                z = np.cos(theta_angle)
            else:
                x, y, z = state[0], state[1], state[2]
            
            label = labels[i] if labels and i < len(labels) else f"State {i}"
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode='lines+markers',
                line=dict(color=color, width=5),
                marker=dict(size=5, color=color),
                name=label
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], title='X'),
                yaxis=dict(range=[-1.5, 1.5], title='Y'),
                zaxis=dict(range=[-1.5, 1.5], title='Z'),
                aspectmode='cube'
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def energy_levels(self, energies: np.ndarray,
                     labels: Optional[List[str]] = None,
                     title: str = "Interactive Energy Levels") -> "go.Figure":
        """
        创建交互式能级图
        
        Args:
            energies: 能量值
            labels: 标签列表
            title: 标题
        
        Returns:
            Plotly图形
        """
        _check_plotly()
        
        fig = go.Figure()
        
        for i, E in enumerate(energies):
            label = labels[i] if labels and i < len(labels) else f"n={i}"
            
            fig.add_hline(y=E, line_dash="dash", line_color="blue", opacity=0.5)
            
            fig.add_trace(go.Scatter(
                x=[0.5], y=[E],
                mode='markers+text',
                marker=dict(size=15, color='blue'),
                text=[f"{E:.4f}"],
                textposition='top right',
                name=label,
                showlegend=False
            ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(range=[0, 1], showticklabels=False),
            yaxis=dict(title='Energy'),
            showlegend=False,
            height=600
        )
        
        return fig
    
    def probability_3d(self, x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      values: np.ndarray,
isolevel: float = 0.1,
                         title: str = "3D Probability Density") -> "go.Figure":
        """
        创建交互式3D概率密度
        
        Args:
            x, y, z: 网格
            values: 概率密度
            isolevel: 等值面水平
            title: 标题
        
        Returns:
            Plotly图形
        """
        _check_plotly()
        
        values_3d = values.reshape(len(x), len(y), len(z))
        
        slice_idx = len(z) // 2
        
        fig = go.Figure(data=[go.Contour(
            x=x, y=y,
            z=values_3d[:, :, slice_idx],
            colorscale='Viridis',
            colorbar=dict(title='|ψ|²'),
            contours=dict(showlabels=True)
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            height=600,
            width=800
        )
        
        return fig
    
    def wavefunction_animation(self, x: np.ndarray,
                               states: List[np.ndarray],
                               times: Optional[List[float]] = None,
                               title: str = "Wavefunction Evolution") -> "go.Figure":
        """
        创建波函数演化动画
        
        Args:
            x: 位置网格
            states: 状态列表
            times: 时间列表
            title: 标题
        
        Returns:
            Plotly图形
        """
        _check_plotly()
        
        fig = go.Figure()
        
        if times is None:
            times = list(range(len(states)))
        
        for i, (t, state) in enumerate(zip(times, states)):
            if isinstance(state, np.ndarray):
                psi = state
            else:
                psi = np.abs(state)**2
            
            fig.add_trace(go.Scatter(
                x=x, y=psi,
                mode='lines',
                name=f't={t:.2f}',
                visible=(i == 0)
            ))
        
        steps = []
        for i in range(len(states)):
            step = dict(
                method="update",
                args=[{"visible": [j == i for j in range(len(states))]}],
                label=f"{times[i]:.2f}"
            )
            steps.append(step)
        
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig.update_layout(
            title=title,
            xaxis_title='Position',
            yaxis_title='|ψ|²',
            sliders=sliders,
            height=500
        )
        
        return fig
    
    def dashboard(self, energies: np.ndarray,
                 states: List[np.ndarray],
                 x: Optional[np.ndarray] = None,
                 densities: Optional[List[np.ndarray]] = None,
                 title: str = "Quantum System Dashboard") -> "go.Figure":
        """
        创建量子系统仪表盘
        
        Args:
            energies: 能量值
            states: 量子态列表
            x: 位置网格（可选）
            densities: 概率密度列表（可选）
            title: 标题
        
        Returns:
            Plotly图形
        """
        _check_plotly()
        
        n_states = min(5, len(energies))
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter", "colspan": 2}, None]],
            subplot_titles=("Bloch Vectors", "Energy Spectrum", "Probability Densities")
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i in range(n_states):
            if len(states[i]) == 2:
                a, b = complex(states[i][0]), complex(states[i][1])
                theta = 2 * np.arccos(np.abs(a))
                phi = np.angle(b) - np.angle(a)
                x_bloch = np.sin(theta) * np.cos(phi)
                y_bloch = np.sin(theta) * np.sin(phi)
                z_bloch = np.cos(theta)
            else:
                x_bloch, y_bloch, z_bloch = states[i][0], states[i][1], states[i][2]
            
            fig.add_trace(
                go.Scatter3d(
                    x=[0, x_bloch], y=[0, y_bloch], z=[0, z_bloch],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=5, color=colors[i]),
                    name=f'State {i}',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Bar(x=list(range(n_states)), y=energies[:n_states],
                  marker_color=colors[:n_states], name='Energy'),
            row=1, col=2
        )
        
        if x is not None and densities is not None:
            for i, density in enumerate(densities[:n_states]):
                fig.add_trace(
                    go.Scatter(x=x, y=density, mode='lines',
                              name=f'State {i}', line=dict(color=colors[i])),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        fig.update_scenes(
            dict(
                xaxis=dict(range=[-1.5, 1.5], title='X'),
                yaxis=dict(range=[-1.5, 1.5], title='Y'),
                zaxis=dict(range=[-1.5, 1.5], title='Z'),
                aspectmode='cube'
            ),
            row=1, col=1
        )
        
        return fig


def plotly_bloch_sphere(states: List[np.ndarray],
                        labels: Optional[List[str]] = None,
                        title: str = "Bloch Sphere") -> "go.Figure":
    """
    Plotly Bloch球
    
    Args:
        states: 量子态列表
        labels: 标签列表
        title: 标题
    
    Returns:
        Plotly图形
    """
    plotter = InteractivePlotter()
    return plotter.bloch_sphere(states, labels, title)


def plotly_energy_levels(energies: np.ndarray,
                        labels: Optional[List[str]] = None,
                        title: str = "Energy Levels") -> "go.Figure":
    """
    Plotly 能级图
    
    Args:
        energies: 能量值
        labels: 标签列表
        title: 标题
    
    Returns:
        Plotly图形
    """
    plotter = InteractivePlotter()
    return plotter.energy_levels(energies, labels, title)


def plotly_3d_isosurface(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         values: np.ndarray,
                         isolevel: float = 0.1,
                         title: str = "3D Isosurface") -> "go.Figure":
    """
    Plotly 3D等值面
    
    Args:
        x, y, z: 网格
        values: 函数值
        isolevel: 等值面水平
        title: 标题
    
    Returns:
        Plotly图形
    """
    plotter = InteractivePlotter()
    return plotter.probability_3d(x, y, z, values, isolevel, title)


def create_quantum_dashboard(energies: np.ndarray,
                              states: List[np.ndarray],
                              x: Optional[np.ndarray] = None,
                              densities: Optional[List[np.ndarray]] = None,
                              title: str = "Quantum Dashboard") -> "go.Figure":
    """
    创建量子系统仪表盘
    
    Args:
        energies: 能量值
        states: 量子态列表
        x: 位置网格
        densities: 概率密度列表
        title: 标题
    
    Returns:
        Plotly图形
    """
    plotter = InteractivePlotter()
    return plotter.dashboard(energies, states, x, densities, title)