import pytest
import numpy as np


class TestInteractivePlotterImport:
    """Test InteractivePlotter import behavior."""

    def test_plotly_check_function_exists(self):
        """Test _check_plotly function exists."""
        from PySymmetry.visual.interactive import _check_plotly

        assert callable(_check_plotly)

    def test_interactive_plotter_exists(self):
        """Test InteractivePlotter class exists."""
        from PySymmetry.visual.interactive import InteractivePlotter

        assert InteractivePlotter is not None

    def test_convenience_functions_exist(self):
        """Test convenience functions exist."""
        from PySymmetry.visual.interactive import (
            plotly_bloch_sphere,
            plotly_energy_levels,
            plotly_3d_isosurface,
            create_quantum_dashboard,
        )

        assert callable(plotly_bloch_sphere)
        assert callable(plotly_energy_levels)
        assert callable(plotly_3d_isosurface)
        assert callable(create_quantum_dashboard)


class TestInteractivePlotterCreation:
    """Test InteractivePlotter creation."""

    def test_creation_without_plotly(self, monkeypatch):
        """Test creation behavior when plotly is not available."""
        import sys
        import importlib

        test_module = sys.modules["PySymmetry.visual.interactive"]
        original_plotly = test_module.PLOTLY_AVAILABLE

        monkeypatch.setattr(test_module, "PLOTLY_AVAILABLE", False)

        try:
            from PySymmetry.visual.interactive import InteractivePlotter

            with pytest.raises(ImportError, match="plotly not installed"):
                InteractivePlotter()
        finally:
            monkeypatch.setattr(test_module, "PLOTLY_AVAILABLE", original_plotly)


class TestPlotlyAvailableCheck:
    """Test plotly availability check."""

    def test_check_plotly_raises_when_not_available(self, monkeypatch):
        """Test _check_plotly raises when plotly not available."""
        import sys

        test_module = sys.modules["PySymmetry.visual.interactive"]

        monkeypatch.setattr(test_module, "PLOTLY_AVAILABLE", False)

        from PySymmetry.visual.interactive import _check_plotly

        with pytest.raises(ImportError, match="plotly not installed"):
            _check_plotly()


@pytest.mark.skipif(
    __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
)
class TestInteractivePlotterMethods:
    """Test InteractivePlotter methods when plotly is available."""

    def test_bloch_sphere_basic(self):
        """Test bloch_sphere method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        states = [np.array([1, 0]), np.array([0, 1])]
        fig = plotter.bloch_sphere(states)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_bloch_sphere_with_labels(self):
        """Test bloch_sphere with custom labels."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        states = [np.array([1, 0]), np.array([0, 1])]
        labels = ["|0⟩", "|1⟩"]
        fig = plotter.bloch_sphere(states, labels=labels, title="Test Title")

        assert fig is not None

    def test_bloch_sphere_3d_states(self):
        """Test bloch_sphere with 3D states."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        states = [np.array([1, 0, 0])]
        fig = plotter.bloch_sphere(states)

        assert fig is not None

    def test_energy_levels_basic(self):
        """Test energy_levels method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plotter.energy_levels(energies)

        assert fig is not None
        assert hasattr(fig, "data")

    def test_energy_levels_with_labels(self):
        """Test energy_levels with custom labels."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.5, 1.5, 2.5])
        labels = ["Ground", "1st Excited", "2nd Excited"]
        fig = plotter.energy_levels(energies, labels=labels, title="Energy Levels")

        assert fig is not None

    def test_probability_3d_basic(self):
        """Test probability_3d method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        values = np.random.rand(10, 10, 10)

        fig = plotter.probability_3d(x, y, z, values)

        assert fig is not None

    def test_wavefunction_animation_basic(self):
        """Test wavefunction_animation method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        x = np.linspace(-5, 5, 100)
        states = [np.random.rand(100) for _ in range(5)]
        times = [0.0, 0.1, 0.2, 0.3, 0.4]

        fig = plotter.wavefunction_animation(x, states, times)

        assert fig is not None

    def test_wavefunction_animation_no_times(self):
        """Test wavefunction_animation without times."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        x = np.linspace(-5, 5, 100)
        states = [np.random.rand(100) for _ in range(3)]

        fig = plotter.wavefunction_animation(x, states)

        assert fig is not None

    def test_dashboard_basic(self):
        """Test dashboard method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        states = [np.array([1, 0]) for _ in range(5)]

        fig = plotter.dashboard(energies, states)

        assert fig is not None

    def test_dashboard_with_densities(self):
        """Test dashboard with probability densities."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.0, 1.0, 2.0])
        states = [np.array([1, 0]) for _ in range(3)]
        x = np.linspace(-5, 5, 100)
        densities = [np.random.rand(100) for _ in range(3)]

        fig = plotter.dashboard(energies, states, x, densities)

        assert fig is not None


@pytest.mark.skipif(
    __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
)
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_plotly_bloch_sphere(self):
        """Test plotly_bloch_sphere convenience function."""
        from PySymmetry.visual.interactive import plotly_bloch_sphere

        states = [np.array([1, 0]), np.array([0, 1])]
        fig = plotly_bloch_sphere(states)

        assert fig is not None

    def test_plotly_energy_levels(self):
        """Test plotly_energy_levels convenience function."""
        from PySymmetry.visual.interactive import plotly_energy_levels

        energies = np.array([0.0, 1.0, 2.0])
        fig = plotly_energy_levels(energies)

        assert fig is not None

    def test_plotly_3d_isosurface(self):
        """Test plotly_3d_isosurface convenience function."""
        from PySymmetry.visual.interactive import plotly_3d_isosurface

        x = np.linspace(-5, 5, 5)
        y = np.linspace(-5, 5, 5)
        z = np.linspace(-5, 5, 5)
        values = np.random.rand(5, 5, 5)

        fig = plotly_3d_isosurface(x, y, z, values)

        assert fig is not None

    def test_create_quantum_dashboard(self):
        """Test create_quantum_dashboard convenience function."""
        from PySymmetry.visual.interactive import create_quantum_dashboard

        energies = np.array([0.0, 1.0, 2.0])
        states = [np.array([1, 0]) for _ in range(3)]

        fig = create_quantum_dashboard(energies, states)

        assert fig is not None


class TestPlotlyModuleAttributes:
    """Test module-level attributes."""

    def test_module_has_pltoly_available_flag(self):
        """Test PLOTLY_AVAILABLE flag exists."""
        from PySymmetry.visual import interactive

        assert hasattr(interactive, "PLOTLY_AVAILABLE")

    def test_go_and_make_subplots_exist(self):
        """Test go and make_subplots exist in module."""
        from PySymmetry.visual import interactive

        if interactive.PLOTLY_AVAILABLE:
            assert interactive.go is not None
            assert interactive.make_subplots is not None
