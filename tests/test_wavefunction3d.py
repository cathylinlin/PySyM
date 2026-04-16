import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PySymmetry.visual.wavefunction3d import (
    Wavefunction3DVisualizer,
    hydrogen_orbital,
    plot_3d_wavefunction,
    plot_3d_probability_isosurface,
    plot_3d_slices,
    plot_orbital,
)


class TestWavefunction3DVisualizer:
    def test_creation(self):
        viz = Wavefunction3DVisualizer()
        assert viz is not None

    def test_plot_isosurface(self):
        viz = Wavefunction3DVisualizer()
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        psi = np.exp(-(X**2 + Y**2 + Z**2))
        fig, ax = viz.plot_isosurface(x, y, z, psi, isolevel=0.1)
        assert fig is not None
        plt.close("all")

    def test_plot_slices(self):
        viz = Wavefunction3DVisualizer()
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        z = np.linspace(-5, 5, 20)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        psi = np.exp(-(X**2 + Y**2 + Z**2))
        fig, axes = viz.plot_slices(x, y, z, psi)
        assert fig is not None
        plt.close("all")

    def test_plot_volume_render(self):
        viz = Wavefunction3DVisualizer()
        x = np.linspace(-5, 5, 8)
        y = np.linspace(-5, 5, 8)
        z = np.linspace(-5, 5, 8)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        psi = np.exp(-(X**2 + Y**2 + Z**2))
        fig, ax = viz.plot_volume_render(x, y, z, psi)
        assert fig is not None
        plt.close("all")


class TestHydrogenOrbital:
    def test_hydrogen_orbital(self):
        x = np.linspace(-5, 5, 5)
        y = np.linspace(-5, 5, 5)
        z = np.linspace(-5, 5, 5)
        result = hydrogen_orbital(1, 0, 0, x, y, z)
        assert result is not None
        assert isinstance(result, np.ndarray)


class TestPlot3DWavefunction:
    def test_plot_3d_wavefunction(self):
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        psi = np.exp(-(X**2 + Y**2 + Z**2))
        fig, ax = plot_3d_wavefunction(x, y, z, psi)
        assert fig is not None
        plt.close("all")


class TestPlot3DProbabilityIsosurface:
    def test_plot_3d_probability_isosurface(self):
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        prob = np.exp(-(X**2 + Y**2 + Z**2))
        fig, ax = plot_3d_probability_isosurface(x, y, z, prob)
        assert fig is not None
        plt.close("all")


class TestPlot3DSlices:
    def test_plot_3d_slices(self):
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        z = np.linspace(-5, 5, 20)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        psi = np.exp(-(X**2 + Y**2 + Z**2))
        fig, axes = plot_3d_slices(x, y, z, psi)
        assert fig is not None
        plt.close("all")


class TestPlotOrbital:
    def test_plot_orbital(self):
        fig = plot_orbital(n=1, l=0, m=0)
        assert fig is not None
        plt.close("all")

    def test_plot_orbital_p(self):
        fig = plot_orbital(n=2, l=1, m=0)
        assert fig is not None
        plt.close("all")
