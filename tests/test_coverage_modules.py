import pytest
import numpy as np


class TestIrreducibleRepresentationFinderExists:
    """Test that IrreducibleRepresentationFinder class exists and can be imported."""

    def test_class_exists(self):
        """Test class exists."""
        from PySymmetry.core.representation.irreducible import (
            IrreducibleRepresentationFinder,
        )

        assert IrreducibleRepresentationFinder is not None

    def test_s3_2d_irrep_exists(self):
        """Test S3 2D irrep dictionary exists."""
        from PySymmetry.core.representation.irreducible import _S3_2D_IRREP

        assert _S3_2D_IRREP is not None
        assert len(_S3_2D_IRREP) == 6

    def test_is_standard_s3_method_exists(self):
        """Test _is_standard_s3 static method exists."""
        from PySymmetry.core.representation.irreducible import (
            IrreducibleRepresentationFinder,
        )

        assert hasattr(IrreducibleRepresentationFinder, "_is_standard_s3")
        assert callable(IrreducibleRepresentationFinder._is_standard_s3)

    def test_s3_2d_irrep_dimensions(self):
        """Test S3 2D irrep matrices have correct dimensions."""
        from PySymmetry.core.representation.irreducible import _S3_2D_IRREP

        for key, matrix in _S3_2D_IRREP.items():
            assert matrix.shape == (2, 2), f"Wrong shape for {key}"

    def test_s3_2d_irrep_identity(self):
        """Test identity element has identity matrix."""
        from PySymmetry.core.representation.irreducible import _S3_2D_IRREP

        identity = _S3_2D_IRREP[(0, 1, 2)]
        assert np.allclose(identity, np.eye(2))

    def test_s3_2d_irrep_complex(self):
        """Test S3 2D irrep matrices are complex."""
        from PySymmetry.core.representation.irreducible import _S3_2D_IRREP

        for key, matrix in _S3_2D_IRREP.items():
            assert np.iscomplexobj(matrix)

    def test_s3_2d_irrep_nonzero_determinant(self):
        """Test all matrices have non-zero determinant."""
        from PySymmetry.core.representation.irreducible import _S3_2D_IRREP

        for key, matrix in _S3_2D_IRREP.items():
            det = np.linalg.det(matrix)
            assert abs(det) > 0.01, f"Zero determinant for {key}"


class TestProductGroupModuleExists:
    """Test that product_group module can be imported."""

    def test_module_imports(self):
        """Test module can be imported."""
        from PySymmetry.core.group_theory.product_group import (
            ProductGroupElement,
            ProductGroup,
            DirectProductGroupElement,
            DirectProductGroup,
            SemidirectProductGroupElement,
            SemidirectProductGroup,
        )

        assert ProductGroupElement is not None
        assert ProductGroup is not None
        assert DirectProductGroupElement is not None
        assert DirectProductGroup is not None
        assert SemidirectProductGroupElement is not None
        assert SemidirectProductGroup is not None

    def test_product_group_element_is_abstract(self):
        """Test ProductGroupElement is abstract."""
        from PySymmetry.core.group_theory.product_group import ProductGroupElement

        with pytest.raises(TypeError):
            ProductGroupElement(None, 0)

    def test_product_group_is_abstract(self):
        """Test ProductGroup is abstract."""
        from PySymmetry.core.group_theory.product_group import ProductGroup

        with pytest.raises(TypeError):
            ProductGroup("Test", None, None)


class TestVisualInteractiveModule:
    """Test visual.interactive module."""

    def test_module_imports(self):
        """Test module can be imported."""
        from PySymmetry.visual.interactive import (
            InteractivePlotter,
            _check_plotly,
            plotly_bloch_sphere,
            plotly_energy_levels,
            plotly_3d_isosurface,
            create_quantum_dashboard,
            PLOTLY_AVAILABLE,
        )

        assert InteractivePlotter is not None
        assert callable(_check_plotly)
        assert callable(plotly_bloch_sphere)
        assert callable(plotly_energy_levels)
        assert callable(plotly_3d_isosurface)
        assert callable(create_quantum_dashboard)

    def test_check_plotly_without_plotly(self, monkeypatch):
        """Test _check_plotly raises when plotly not available."""
        import sys

        test_module = sys.modules["PySymmetry.visual.interactive"]
        monkeypatch.setattr(test_module, "PLOTLY_AVAILABLE", False)

        from PySymmetry.visual.interactive import _check_plotly

        with pytest.raises(ImportError, match="plotly not installed"):
            _check_plotly()

    def test_interactive_plotter_requires_plotly(self, monkeypatch):
        """Test InteractivePlotter requires plotly."""
        import sys

        test_module = sys.modules["PySymmetry.visual.interactive"]
        monkeypatch.setattr(test_module, "PLOTLY_AVAILABLE", False)

        from PySymmetry.visual.interactive import InteractivePlotter

        with pytest.raises(ImportError):
            InteractivePlotter()


class TestVisualInteractivePlotly:
    """Test visual.interactive with plotly available."""

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_interactive_plotter_creation(self):
        """Test InteractivePlotter can be created with plotly."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()
        assert plotter is not None

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_bloch_sphere(self):
        """Test bloch_sphere method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        states = [np.array([1, 0]), np.array([0, 1])]
        fig = plotter.bloch_sphere(states)

        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_energy_levels(self):
        """Test energy_levels method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plotter.energy_levels(energies)

        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_probability_3d(self):
        """Test probability_3d method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        x = np.linspace(-5, 5, 10)
        y = np.linspace(-5, 5, 10)
        z = np.linspace(-5, 5, 10)
        values = np.random.rand(10, 10, 10)

        fig = plotter.probability_3d(x, y, z, values)

        assert fig is not None

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_wavefunction_animation(self):
        """Test wavefunction_animation method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        x = np.linspace(-5, 5, 100)
        states = [np.random.rand(100) for _ in range(3)]
        times = [0.0, 0.1, 0.2]

        fig = plotter.wavefunction_animation(x, states, times)

        assert fig is not None

    @pytest.mark.skipif(
        __import__("sys").modules.get("plotly") is None, reason="plotly not installed"
    )
    def test_dashboard(self):
        """Test dashboard method."""
        from PySymmetry.visual.interactive import InteractivePlotter

        plotter = InteractivePlotter()

        energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        states = [np.array([1, 0]) for _ in range(5)]

        fig = plotter.dashboard(energies, states)

        assert fig is not None
