import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestQuantumColormap:
    def test_probability_cmap(self):
        from PySymmetry.visual import QuantumColormap

        cmap = QuantumColormap.probability()
        assert cmap is not None
        colors = cmap([0.0, 0.5, 1.0])
        assert len(colors) == 3

    def test_phase_cmap(self):
        from PySymmetry.visual import QuantumColormap

        cmap = QuantumColormap.phase()
        assert cmap is not None
        colors = cmap([0.0, 0.25, 0.5])
        assert len(colors) == 3

    def test_entanglement_cmap(self):
        from PySymmetry.visual import QuantumColormap

        cmap = QuantumColormap.entanglement()
        assert cmap is not None

    def test_energy_cmap(self):
        from PySymmetry.visual import QuantumColormap

        cmap = QuantumColormap.energy()
        assert cmap is not None


class TestPlotStyle:
    def test_apply_paper(self):
        from PySymmetry.visual import PlotStyle

        PlotStyle.apply("paper")
        PlotStyle.reset()

    def test_apply_presentation(self):
        from PySymmetry.visual import PlotStyle

        PlotStyle.apply("presentation")
        PlotStyle.reset()

    def test_apply_dark(self):
        from PySymmetry.visual import PlotStyle

        PlotStyle.apply("dark")
        PlotStyle.reset()

    def test_reset(self):
        from PySymmetry.visual import PlotStyle

        PlotStyle.apply("dark")
        PlotStyle.reset()


class TestHelperFunctions:
    def test_setup_axes(self):
        from PySymmetry.visual import setup_axes

        fig, ax = plt.subplots()
        setup_axes(ax, xlabel="x", ylabel="y", title="Test")
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        plt.close(fig)

    def test_color_by_probability(self):
        from PySymmetry.visual import color_by_probability

        probs = np.array([0.1, 0.5, 1.0])
        colors = color_by_probability(probs)
        assert len(colors) == 3

    def test_color_by_phase(self):
        from PySymmetry.visual import color_by_phase

        phases = np.array([0.0, np.pi / 2, np.pi])
        colors = color_by_phase(phases)
        assert len(colors) == 3

    def test_create_subplots(self):
        from PySymmetry.visual import create_subplots

        fig, axes = create_subplots(4, n_cols=2)
        assert len(axes.flat) >= 4
        plt.close(fig)

    def test_create_subplots_single(self):
        from PySymmetry.visual import create_subplots

        fig, axes = create_subplots(1)
        plt.close(fig)


class TestColorConstants:
    def test_quantum_colorscheme(self):
        from PySymmetry.visual import QUANTUM_COLORSCHEME

        assert "quantum_red" in QUANTUM_COLORSCHEME
        assert len(QUANTUM_COLORSCHEME) > 0

    def test_state_colors(self):
        from PySymmetry.visual import STATE_COLORS

        assert "|0⟩" in STATE_COLORS
        assert "|1⟩" in STATE_COLORS

    def test_gate_colors(self):
        from PySymmetry.visual import GATE_COLORS

        assert "H" in GATE_COLORS
        assert "X" in GATE_COLORS
        assert "CNOT" in GATE_COLORS
