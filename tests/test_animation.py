import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PySymmetry.visual.animation import (
    StateEvolutionAnimator,
    animate_bloch,
    create_rabi_oscillation_animation,
)


class TestAnimator:
    def test_creation(self):
        animator = StateEvolutionAnimator()
        assert animator is not None


class TestBlochAnimation:
    def test_animate_bloch(self):
        states = [
            np.array([np.cos(t / 2), 1j * np.sin(t / 2)])
            for t in np.linspace(0, np.pi, 10)
        ]
        times = list(np.linspace(0, 1, 10))
        ani = animate_bloch(states, times, save_path=None)
        assert ani is not None
        plt.close("all")


class TestRabiAnimation:
    def test_creation(self):
        ani = create_rabi_oscillation_animation(Omega=1.0, T=5.0, n_points=10)
        assert ani is not None
        plt.close("all")
