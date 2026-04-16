import pytest
import numpy as np
from PySymmetry.phys.quantum.hamiltonian import (
    HamiltonianOperator,
    MatrixHamiltonian,
    FreeParticleHamiltonian,
    HarmonicOscillatorHamiltonian,
)


class TestMatrixHamiltonian:
    def test_creation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        assert ham.dimension == 2
        assert ham.name == "MatrixHamiltonian"

    def test_creation_invalid(self):
        H = np.array([[1, 0, 0], [0, 2, 0]], dtype=complex)
        with pytest.raises(ValueError):
            MatrixHamiltonian(H)

    def test_matrix(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        result = ham.matrix
        assert np.allclose(result, H)

    def test_custom_name(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H, name="Custom")
        assert ham.name == "Custom"

    def test_expectation(self):
        from PySymmetry.phys.quantum.states import Ket

        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        exp = ham.expectation(ket)
        assert np.isclose(exp, 1.0)

    def test_variance(self):
        from PySymmetry.phys.quantum.states import Ket

        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        ket = Ket(np.array([1, 0], dtype=complex))
        var = ham.variance(ket)
        assert np.isclose(var, 0.0)

    def test_ground_state(self):
        from PySymmetry.phys.quantum.states import Ket

        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        state, energy = ham.ground_state()
        assert np.isclose(energy, 1.0)

    def test_excited_states(self):
        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        states = ham.excited_states(n=2)
        assert len(states) == 2

    def test_all_energy_levels(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        levels = ham.all_energy_levels()
        assert len(levels) == 2

    def test_commutator(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        O = np.array([[0, 1], [1, 0]], dtype=complex)
        ham = MatrixHamiltonian(H)
        comm = ham.compute_commutator(O)
        expected = H @ O - O @ H
        assert np.allclose(comm, expected)

    def test_add_symmetry(self):
        class MockSymmetry:
            def representation_matrix(self, dim):
                return np.eye(dim)

        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        sym = MockSymmetry()
        ham.add_symmetry(sym)
        symmetries = ham.get_symmetries()
        assert len(symmetries) == 1

    def test_to_dict(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        d = ham.to_dict()
        assert d["type"] == "MatrixHamiltonian"
        assert d["dimension"] == 2

    def test_from_dict(self):
        data = {"type": "MatrixHamiltonian", "name": "Test", "matrix": [[1, 0], [0, 2]]}
        ham = MatrixHamiltonian.from_dict(data)
        assert ham.dimension == 2
        assert ham.name == "Test"


class TestFreeParticleHamiltonian:
    def test_creation(self):
        ham = FreeParticleHamiltonian(mass=1.0, dimension=1, basis_size=10)
        assert ham.dimension == 10

    def test_kinetic_energy(self):
        from PySymmetry.phys.quantum.states import Ket

        ham = FreeParticleHamiltonian(mass=1.0, basis_size=10)
        ket = Ket(10)
        E = ham.get_kinetic_energy(ket)
        assert E >= 0

    def test_to_dict(self):
        ham = FreeParticleHamiltonian(mass=2.0, basis_size=20)
        d = ham.to_dict()
        assert d["type"] == "FreeParticleHamiltonian"
        assert d["mass"] == 2.0

    def test_from_dict(self):
        data = {
            "type": "FreeParticleHamiltonian",
            "mass": 1.5,
            "dimension": 1,
            "basis_size": 50,
            "lattice_spacing": 0.01,
        }
        ham = FreeParticleHamiltonian.from_dict(data)
        assert ham._mass == 1.5


class TestHarmonicOscillatorHamiltonian:
    def test_creation(self):
        ham = HarmonicOscillatorHamiltonian(mass=1.0, frequency=1.0, dimension=20)
        assert ham.dimension == 20

    def test_operators(self):
        ham = HarmonicOscillatorHamiltonian(dimension=10)
        a = ham.a
        a_dag = ham.a_dag
        n = ham.n
        assert a.shape == (10, 10)
        assert a_dag.shape == (10, 10)
        assert n.shape == (10, 10)

    def test_x_p_operators(self):
        ham = HarmonicOscillatorHamiltonian(dimension=10)
        x = ham.x
        p = ham.p
        assert x.shape == (10, 10)
        assert p.shape == (10, 10)

    def test_energy_level(self):
        ham = HarmonicOscillatorHamiltonian(hbar=1.0, frequency=1.0, dimension=20)
        E0 = ham.energy_level(0)
        E1 = ham.energy_level(1)
        assert np.isclose(E0, 0.5)
        assert np.isclose(E1, 1.5)

    def test_coherent_state(self):
        ham = HarmonicOscillatorHamiltonian(dimension=30)
        ket = ham.coherent_state(alpha=1.0)
        assert ket.dimension == 30
        assert np.isclose(ket.norm(), 1.0)

    def test_to_dict(self):
        ham = HarmonicOscillatorHamiltonian(mass=2.0, frequency=0.5, dimension=25)
        d = ham.to_dict()
        assert d["type"] == "HarmonicOscillatorHamiltonian"
        assert d["mass"] == 2.0
        assert d["frequency"] == 0.5

    def test_from_dict(self):
        data = {
            "type": "HarmonicOscillatorHamiltonian",
            "mass": 1.0,
            "frequency": 1.0,
            "dimension": 20,
            "hbar": 1.0,
        }
        ham = HarmonicOscillatorHamiltonian.from_dict(data)
        assert ham._m == 1.0
        assert hasattr(ham, "_omega")

    def test_ground_state(self):
        ham = HarmonicOscillatorHamiltonian(dimension=30)
        ket, E = ham.ground_state()
        assert np.isclose(E, 0.5, atol=0.01)

    def test_excited_states(self):
        ham = HarmonicOscillatorHamiltonian(dimension=30)
        states = ham.excited_states(n=3)
        assert len(states) == 3
        energies = [e for _, e in states]
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1]
