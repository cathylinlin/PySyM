import pytest
import numpy as np
from PySymmetry.phys.quantum.explainer import (
    ResultExplainer,
    EnergySpectrumExplainer,
    QuantumStateExplainer,
)
from PySymmetry.phys.quantum.states import Ket, DensityMatrix
from PySymmetry.phys.quantum.hamiltonian import MatrixHamiltonian


class TestResultExplainer:
    def test_init(self):
        class TestExplainer(ResultExplainer):
            def explain(self) -> str:
                return "test"

        explainer = TestExplainer("Test")
        assert explainer._name == "Test"


class TestEnergySpectrumExplainer:
    def test_init(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        explainer = EnergySpectrumExplainer(ham)
        assert explainer._H is ham

    def test_compute_spectrum_info(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        explainer = EnergySpectrumExplainer(ham)
        info = explainer.compute_spectrum_info()

        assert "ground_energy" in info
        assert "first_excited_energy" in info

    def test_explain(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        explainer = EnergySpectrumExplainer(ham)
        result = explainer.explain()
        assert isinstance(result, str)


class TestQuantumStateExplainer:
    def test_init_with_ket(self):
        vec = np.array([1, 0], dtype=complex)
        ket = Ket(vec)
        explainer = QuantumStateExplainer(ket)
        assert explainer._state is ket

    def test_init_with_density_matrix(self):
        rho = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        explainer = QuantumStateExplainer(rho)
        assert explainer._state is rho

    def test_compute_state_properties(self):
        vec = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        ket = Ket(vec)
        explainer = QuantumStateExplainer(ket)
        props = explainer.compute_state_properties()

        assert props["is_pure"] == True

    def test_explain(self):
        vec = np.array([1, 0], dtype=complex)
        ket = Ket(vec)
        explainer = QuantumStateExplainer(ket)
        result = explainer.explain()
        assert isinstance(result, str)
