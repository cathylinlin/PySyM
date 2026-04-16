"""
量子态演化动画模块

提供量子态时间演化的动画可视化：
- Bloch球轨迹动画
- 概率密度演化动画
- 动画保存功能

依赖：numpy, matplotlib, animation
"""

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError:
    plt = None
    animation = None
    FuncAnimation = None


def _check_matplotlib():
    if plt is None:
        raise ImportError("matplotlib not installed")


class StateEvolutionAnimator:
    """
    量子态演化动画器
    """

    def __init__(self, figsize: tuple[float, float] = (10, 8)):
        _check_matplotlib()
        self.figsize = figsize

    def animate_bloch_sphere(
        self,
        states: list[np.ndarray],
        times: list[float] | None = None,
        labels: list[str] | None = None,
        title: str = "Bloch Sphere Evolution",
        interval: int = 100,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        创建Bloch球演化动画

        Args:
            states: 量子态序列
            times: 时间序列
            labels: 标签
            title: 标题
            interval: 帧间隔(ms)
            save_path: 保存路径

        Returns:
            动画对象
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, alpha=0.1, color="gray")

        ax.plot([-1.5, 1.5], [0, 0], [0, 0], "k-", linewidth=0.5)
        ax.plot([0, 0], [-1.5, 1.5], [0, 0], "k-", linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-1.5, 1.5], "k-", linewidth=0.5)

        (line,) = ax.plot([], [], [], "r-", linewidth=2)
        (point,) = ax.plot([], [], [], "ro", markersize=10)

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

        def get_bloch_vector(state):
            if len(state) == 2:
                a, b = complex(state[0]), complex(state[1])
                theta = 2 * np.arccos(np.abs(a))
                phi = np.angle(b) - np.angle(a)
                return (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                )
            return tuple(state[:3])

        trajectory_x, trajectory_y, trajectory_z = [], [], []

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point

        def update(frame):
            trajectory_x.append(get_bloch_vector(states[frame])[0])
            trajectory_y.append(get_bloch_vector(states[frame])[1])
            trajectory_z.append(get_bloch_vector(states[frame])[2])

            line.set_data(trajectory_x, trajectory_y)
            line.set_3d_properties(trajectory_z)

            point.set_data([trajectory_x[-1]], [trajectory_y[-1]])
            point.set_3d_properties([trajectory_z[-1]])

            if times:
                ax.set_title(f"{title} (t={times[frame]:.2f})")

            return line, point

        ani = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=interval,
            blit=False,
        )

        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))
            print(f"Animation saved to {save_path}")

        plt.close()
        return ani

    def animate_probability_density(
        self,
        x: np.ndarray,
        states: list[np.ndarray],
        times: list[float] | None = None,
        title: str = "Probability Density Evolution",
        interval: int = 100,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        创建概率密度演化动画

        Args:
            x: 位置网格
            states: 状态序列
            times: 时间序列
            title: 标题
            interval: 帧间隔
            save_path: 保存路径

        Returns:
            动画对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        (line,) = ax.plot([], [], "b-", linewidth=2)
        fill = ax.fill_between(x, 0, [], alpha=0.3, color="blue")

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, max(np.max(np.abs(s) ** 2) for s in states) * 1.1)
        ax.set_xlabel("Position")
        ax.set_ylabel("|ψ|²")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        def init():
            line.set_data([], [])
            return (line,)

        def update(frame):
            psi = states[frame]
            if hasattr(psi, "flatten"):
                y = np.abs(psi.flatten()) ** 2
            else:
                y = np.abs(psi) ** 2

            line.set_data(x, y)

            ax.collections.clear()
            ax.fill_between(x, 0, y, alpha=0.3, color="blue")

            if times:
                ax.set_title(f"{title} (t={times[frame]:.2f})")

            return (line,)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=interval,
            blit=False,
        )

        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))
            print(f"Animation saved to {save_path}")

        plt.close()
        return ani

    def animate_density_matrix(
        self,
        states: list[np.ndarray],
        times: list[float] | None = None,
        title: str = "Density Matrix Evolution",
        interval: int = 200,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        创建密度矩阵演化动画

        Args:
            states: 密度矩阵序列
            times: 时间序列
            title: 标题
            interval: 帧间隔
            save_path: 保存路径

        Returns:
            动画对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(
            np.real(states[0]), cmap="RdBu_r", vmin=-1, vmax=1, animated=True
        )
        cbar = plt.colorbar(im, ax=ax)

        ax.set_title(title)

        def init():
            im.set_array(np.real(states[0]))
            return (im,)

        def update(frame):
            im.set_array(np.real(states[frame]))

            if times:
                ax.set_title(f"{title} (t={times[frame]:.2f})")

            return (im,)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(states),
            init_func=init,
            interval=interval,
            blit=False,
        )

        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))
            print(f"Animation saved to {save_path}")

        plt.close()
        return ani

    def animate_energy_levels(
        self,
        energies: list[np.ndarray],
        times: list[float] | None = None,
        title: str = "Energy Spectrum Evolution",
        interval: int = 200,
        save_path: str | None = None,
    ) -> FuncAnimation:
        """
        创建能级演化动画

        Args:
            energies: 能量序列
            times: 时间序列
            title: 标题
            interval: 帧间隔
            save_path: 保存路径

        Returns:
            动画对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        all_energies = np.concatenate(energies)
        y_min, y_max = all_energies.min(), all_energies.max()

        bars = ax.bar(
            range(len(energies[0])), energies[0], color="steelblue", alpha=0.8
        )

        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
        ax.set_xlabel("State Index")
        ax.set_ylabel("Energy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        def init():
            for bar, h in zip(bars, energies[0]):
                bar.set_height(h)
            return (bars,)

        def update(frame):
            for bar, h in zip(bars, energies[frame]):
                bar.set_height(h)

            if times:
                ax.set_title(f"{title} (t={times[frame]:.2f})")

            return (bars,)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(energies),
            init_func=init,
            interval=interval,
            blit=False,
        )

        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))
            print(f"Animation saved to {save_path}")

        plt.close()
        return ani


def animate_bloch(
    states: list[np.ndarray],
    times: list[float] | None = None,
    title: str = "Bloch Sphere Evolution",
    save_path: str | None = None,
) -> FuncAnimation:
    """
    创建Bloch球动画

    Args:
        states: 量子态序列
        times: 时间序列
        title: 标题
        save_path: 保存路径

    Returns:
        动画对象
    """
    animator = StateEvolutionAnimator()
    return animator.animate_bloch_sphere(
        states, times, title=title, save_path=save_path
    )


def animate_probability(
    x: np.ndarray,
    states: list[np.ndarray],
    times: list[float] | None = None,
    title: str = "Probability Evolution",
    save_path: str | None = None,
) -> FuncAnimation:
    """
    创建概率密度动画

    Args:
        x: 位置网格
        states: 状态序列
        times: 时间序列
        title: 标题
        save_path: 保存路径

    Returns:
        动画对象
    """
    animator = StateEvolutionAnimator()
    return animator.animate_probability_density(
        x, states, times, title, save_path=save_path
    )


def create_rabi_oscillation_animation(
    Omega: float = 1.0, T: float = 10.0, n_points: int = 100
) -> FuncAnimation:
    """
    创建Rabi振荡动画

    Args:
        Omega: Rabi频率
        T: 总时间
        n_points: 采样点数

    Returns:
        动画对象
    """
    times = np.linspace(0, T, n_points)

    states = []
    for t in times:
        theta = Omega * t
        state = np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)])
        states.append(state)

    animator = StateEvolutionAnimator()
    return animator.animate_bloch_sphere(
        states, times.tolist(), title="Rabi Oscillation"
    )
