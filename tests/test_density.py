import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PySymmetry.visual.density import (
    ProbabilityDensityPlotter,
    DensityMatrixVisualizer,
    plot_probability_density,
    plot_density_matrix,
    plot_2d_density_heatmap,
    plot_3d_density_isosurface,
)
from PySymmetry.phys.quantum.states import Ket, DensityMatrix


class TestProbabilityDensityPlotter:
    def test_creation(self):
        plotter = ProbabilityDensityPlotter()
        assert plotter is not None

    def test_plot_1d(self):
        plotter = ProbabilityDensityPlotter()
        x = np.linspace(-5, 5, 100)
        psi = np.exp(-(x**2))
        fig, ax = plotter.plot_1d(x, [psi])
        assert fig is not None
        plt.close("all")

    def test_plot_2d_heatmap(self):
        plotter = ProbabilityDensityPlotter()
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        prob = np.exp(-(X**2 + Y**2))
        fig, ax = plotter.plot_2d_heatmap(x, y, prob)
        assert fig is not None
        plt.close("all")

    def test_plot_2d_contour(self):
        plotter = ProbabilityDensityPlotter()
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        prob = np.exp(-(X**2 + Y**2))
        fig, ax = plotter.plot_2d_contour(x, y, prob)
        assert fig is not None
        plt.close("all")


class TestDensityMatrixVisualizer:
    def test_creation(self):
        viz = DensityMatrixVisualizer()
        assert viz is not None

    def test_plot_real(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_real(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_imag(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_imag(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_abs(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_abs(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_all(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_all(rho)
        assert fig is not None
        plt.close("all")


class TestPlotProbabilityDensity:
    def test_plot_probability_density(self):
        x = np.linspace(-5, 5, 50)
        psi = np.exp(-(x**2))
        fig, ax = plot_probability_density(x, densities=[psi])
        assert fig is not None
        plt.close("all")

    def test_plot_2d_heatmap(self):
        plotter = ProbabilityDensityPlotter()
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        prob = np.exp(-(X**2 + Y**2))
        fig, ax = plotter.plot_2d_heatmap(x, y, prob)
        assert fig is not None
        plt.close("all")

    def test_plot_2d_contour(self):
        plotter = ProbabilityDensityPlotter()
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        prob = np.exp(-(X**2 + Y**2))
        fig, ax = plotter.plot_2d_contour(x, y, prob)
        assert fig is not None
        plt.close("all")


class TestDensityMatrixVisualizer:
    def test_creation(self):
        viz = DensityMatrixVisualizer()
        assert viz is not None

    def test_plot_real(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_real(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_imag(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_imag(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_abs(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_abs(rho)
        assert fig is not None
        plt.close("all")

    def test_plot_all(self):
        viz = DensityMatrixVisualizer()
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = viz.plot_all(rho)
        assert fig is not None
        plt.close("all")


class TestPlotProbabilityDensity:
    def test_plot_probability_density(self):
        x = np.linspace(-5, 5, 50)
        psi = np.exp(-(x**2))
        fig, ax = plot_probability_density(x, [psi], ["test"])
        assert fig is not None
        plt.close("all")


class TestPlotDensityMatrix:
    def test_plot_density_matrix(self):
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        fig, ax = plot_density_matrix(rho)
        assert fig is not None
        plt.close("all")


class TestPlot2DDensityHeatmap:
    def test_plot_2d_density_heatmap(self):
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        prob = np.exp(-(X**2 + Y**2))
        fig, ax = plot_2d_density_heatmap(x, y, prob)
        assert fig is not None
        plt.close("all")


class TestPlot3DDensityIsosurface:
    def test_plot_3d_density_isosurface(self):
        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        prob = np.exp(-(X**2 + Y**2 + Z**2))
        fig, ax = plot_3d_density_isosurface(x, y, z, prob)
        assert fig is not None
        plt.close("all")
