import pytest
import numpy as np
from PySymmetry.phys.quantum.analysis import (
    QuantumAnalyzer,
    SymmetryInfo,
    StateClassification,
    TransitionRule,
    AnalysisResult,
    QuantumParityOperation,
    QuantumTranslationOperation,
    analyze,
    quick_report,
    check_parity,
    GroupRepresentation,
    LieAlgebra,
    SO3LieAlgebra,
    SU2LieAlgebra,
    GroupTheoryAnalyzer,
    analyze_group_theory,
    SO4HydrogenAnalyzer,
    detect_hydrogen_so4_symmetry,
)
from PySymmetry.phys.quantum.hamiltonian import (
    MatrixHamiltonian,
    HarmonicOscillatorHamiltonian,
)
from PySymmetry.phys.quantum.states import Ket, DensityMatrix


class TestQuantumAnalyzer:
    def test_creation_with_hamiltonian(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        analyzer = QuantumAnalyzer(hamiltonian=ham)
        assert analyzer._H is ham

    def test_creation_with_result(self):
        class MockResult:
            grid = np.linspace(-5, 5, 10)
            energies = np.array([0.5, 1.5])
            states = [Ket(np.array([1, 0], dtype=complex))]

        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        analyzer = QuantumAnalyzer(hamiltonian=ham, result=MockResult())
        assert analyzer._H is ham

    def test_has_abstract_phys(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        analyzer = QuantumAnalyzer(hamiltonian=ham)
        assert analyzer.has_abstract_phys is not None

    def test_get_parity_operation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        analyzer = QuantumAnalyzer(hamiltonian=ham)
        parity = analyzer.get_parity_operation(4)
        assert isinstance(parity, QuantumParityOperation)

    def test_detect_symmetries(self):
        H = np.array([[1, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        analyzer = QuantumAnalyzer(hamiltonian=ham)
        symmetries = analyzer.detect_symmetries()
        assert isinstance(symmetries, list)


class TestQuantumParityOperation:
    def test_creation(self):
        op = QuantumParityOperation(4)
        assert op._dimension == 4

    def test_representation_matrix(self):
        op = QuantumParityOperation(4)
        matrix = op.representation_matrix()
        assert matrix.shape == (4, 4)


class TestQuantumTranslationOperation:
    def test_creation(self):
        op = QuantumTranslationOperation(4, 1)
        assert op is not None


class TestSymmetryInfo:
    def test_creation(self):
        info = SymmetryInfo(
            name="TestSym",
            description="Test description",
            symmetry_type="discrete",
            is_exact=True,
            conserved_quantity="Test quantity",
        )
        assert info.name == "TestSym"


class TestAnalyzeFunction:
    def test_analyze(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)

        class MockResult:
            grid = np.array([0, 1])
            energies = np.array([1.0])
            states = [Ket(np.array([1, 0], dtype=complex))]

        result = analyze(ham, MockResult())
        assert result is not None


class TestQuickReport:
    def test_quick_report(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)

        class MockResult:
            grid = np.array([0, 1])
            energies = np.array([1.0])
            states = [Ket(np.array([1, 0], dtype=complex))]

        report = quick_report(ham, MockResult())
        assert isinstance(report, str)


class TestCheckParity:
    def test_check_parity(self):
        ket = Ket(np.array([1, 0], dtype=complex))
        parity, label = check_parity(ket, 2)
        assert parity in [1.0, -1.0]
        assert "even" in label or "odd" in label


class TestGroupRepresentation:
    def test_creation(self):
        rep = GroupRepresentation("SO3", 3)
        assert rep is not None


class TestLieAlgebra:
    def test_creation(self):
        algebra = LieAlgebra(3, "TestAlgebra")
        assert algebra is not None


class TestSO3LieAlgebra:
    def test_creation(self):
        algebra = SO3LieAlgebra()
        assert algebra.name == "SO(3)"


class TestSU2LieAlgebra:
    def test_creation(self):
        algebra = SU2LieAlgebra()
        assert algebra.name == "SU(2)"


class TestGroupTheoryAnalyzer:
    def test_creation(self):
        analyzer = GroupTheoryAnalyzer("SO3")
        assert analyzer is not None


class TestSO4HydrogenAnalyzer:
    def test_creation(self):
        analyzer = SO4HydrogenAnalyzer()
        assert analyzer is not None


class TestDetectHydrogenSO4Symmetry:
    def test_detect_hydrogen_so4_symmetry(self):
        H = np.array([[1, 0], [0, -1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        result = detect_hydrogen_so4_symmetry(ham)
        assert result is not None
