import pytest
import numpy as np
from PySymmetry.phys.quantum.simulator import (
    Simulator,
    QuantumSimulator,
    MeasurementSimulator,
    DecoherenceSimulator,
    ParticleFieldSimulator,
    ScatteringSimulator,
    SpinChainSimulator,
)
from PySymmetry.phys.quantum.hamiltonian import MatrixHamiltonian
from PySymmetry.phys.quantum.states import Ket, DensityMatrix


class TestQuantumSimulator:
    def test_creation(self):
        H = np.array([[0, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        sim = QuantumSimulator(ham, ket)
        assert sim.name == "QuantumSimulator"

    def test_step(self):
        H = np.array([[0, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        sim = QuantumSimulator(ham, ket)
        sim.step(0.01)
        assert sim.time >= 0

    def test_run(self):
        H = np.array([[0, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        sim = QuantumSimulator(ham, ket)
        results = sim.run(duration=0.01, dt=0.005)
        assert "times" in results

    def test_state_history(self):
        H = np.array([[0, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        sim = QuantumSimulator(ham, ket)
        sim.run(duration=0.01, dt=0.005)
        history = sim.state_history
        assert len(history) > 0


class TestMeasurementSimulator:
    def test_creation(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = MeasurementSimulator(dm)
        assert sim.name == "MeasurementSimulator"

    def test_single_measurement(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = MeasurementSimulator(dm)
        result = sim.measure()
        assert len(result) == 2

    def test_ensemble_measurement(self):
        dm = DensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex))
        sim = MeasurementSimulator(dm)
        result = sim.ensemble_measure(num_samples=10)
        assert "counts" in result
        assert "frequencies" in result

    def test_run(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = MeasurementSimulator(dm)
        result = sim.run(duration=10, dt=1)
        assert "counts" in result


class TestDecoherenceSimulator:
    def test_creation(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = DecoherenceSimulator(dm)
        assert sim.name == "DecoherenceSimulator"

    def test_step(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = DecoherenceSimulator(dm)
        sim.step(0.1)
        assert sim.time > 0

    def test_evolve(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = DecoherenceSimulator(dm)
        sim.step(0.1)
        state = sim.state
        assert state.dimension == 2

    def test_run(self):
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        sim = DecoherenceSimulator(dm)
        result = sim.run(duration=0.1, dt=0.05)
        assert "times" in result
        assert "purity_history" in result

    def test_dephasing_rate(self):
        dm = DensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex))
        sim = DecoherenceSimulator(dm, decoherence_rate=1.0)
        sim.step(0.5)
        state = sim.state
        assert state.dimension == 2


class TestParticleFieldSimulator:
    def test_creation(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        momenta = np.array([[1.0, 0.0, 0.0]])
        charges = np.array([1.0])
        masses = np.array([1.0])
        sim = ParticleFieldSimulator(
            particle_positions=positions,
            particle_momenta=momenta,
            charges=charges,
            masses=masses,
        )
        assert sim.name == "ParticleFieldSimulator"

    def test_run(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        momenta = np.array([[1.0, 0.0, 0.0]])
        charges = np.array([1.0])
        masses = np.array([1.0])
        sim = ParticleFieldSimulator(
            particle_positions=positions,
            particle_momenta=momenta,
            charges=charges,
            masses=masses,
        )
        result = sim.run(duration=0.01, dt=0.005)
        assert "times" in result


class TestScatteringSimulator:
    def test_creation(self):
        psi = np.zeros(64)
        psi[32] = 1.0
        V = lambda x: 0.0
        sim = ScatteringSimulator(initial_wavefunction=psi, interaction_potential=V)
        assert sim.name == "ScatteringSimulator"


class TestSpinChainSimulator:
    def test_creation(self):
        couplings = {"xx": 1.0, "zz": 0.5}
        sim = SpinChainSimulator(num_sites=3, couplings=couplings)
        assert sim.name == "SpinChainSimulator"

    def test_run(self):
        couplings = {"xx": 1.0, "zz": 0.5}
        sim = SpinChainSimulator(num_sites=3, couplings=couplings)
        result = sim.run(duration=0.01, dt=0.005)
        assert result is not None
