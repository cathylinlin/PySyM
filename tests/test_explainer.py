import pytest
import numpy as np
from src.PySymmetry.phys.quantum.explainer import (
    ResultExplainer,
    EnergySpectrumExplainer,
    QuantumStateExplainer,
    MeasurementExplainer,
    DynamicsExplainer,
    DecoherenceExplainer,
    CompositeExplainer,
    SymmetryExplainer,
    explain_quantum_system,
    explain_measurement_results
)
from src.PySymmetry.phys.quantum.states import Ket, DensityMatrix
from src.PySymmetry.phys.quantum.hamiltonian import MatrixHamiltonian


class MockHamiltonian(MatrixHamiltonian):
    def __init__(self, matrix):
        super().__init__(matrix, name="Mock")
    
    def all_energy_levels(self):
        return np.linalg.eigvalsh(self._matrix)


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
        mock_H = MockHamiltonian(H)
        explainer = EnergySpectrumExplainer(mock_H)
        assert explainer._H is mock_H

    def test_compute_spectrum_info(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        mock_H = MockHamiltonian(H)
        explainer = EnergySpectrumExplainer(mock_H)
        info = explainer.compute_spectrum_info()
        
        assert 'ground_energy' in info
        assert 'first_excited_energy' in info
        assert 'energy_gaps' in info
        assert 'all_energies' in info
        assert 'degeneracies' in info

    def test_compute_degeneracies(self):
        H = np.array([[1, 0], [0, 1]], dtype=complex)
        mock_H = MockHamiltonian(H)
        explainer = EnergySpectrumExplainer(mock_H)
        energies = np.array([1.0, 1.0, 2.0])
        deg = explainer._compute_degeneracies(energies)
        assert 1.0 in deg
        assert 2.0 in deg

    def test_explain(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        mock_H = MockHamiltonian(H)
        explainer = EnergySpectrumExplainer(mock_H)
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

    def test_compute_state_properties_pure(self):
        vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        ket = Ket(vec)
        explainer = QuantumStateExplainer(ket)
        props = explainer.compute_state_properties()
        
        assert props['is_pure'] == True
        assert 'dimension' in props
        assert 'norm' in props
        assert 'probabilities' in props

    def test_explain(self):
        vec = np.array([1, 0], dtype=complex)
        ket = Ket(vec)
        explainer = QuantumStateExplainer(ket)
        result = explainer.explain()
        assert isinstance(result, str)
