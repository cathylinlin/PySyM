import pytest
import numpy as np
from PySymmetry.phys.quantum.solver import (
    ExactDiagonalizationSolver,
    SparseMatrixSolver,
    LanczosSolver,
    TimeEvolutionSolver,
    SplitStepFourierSolver,
    NumerovSolver,
    QuantumSolverFactory,
    HydrogenAtomSolver,
    HarmonicOscillatorSolver,
    ParticleInBoxSolver,
)
from PySymmetry.phys.quantum.hamiltonian import MatrixHamiltonian
from PySymmetry.phys.quantum.states import Ket


class TestExactDiagonalizationSolver:
    def test_creation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        assert solver._H is ham

    def test_solve(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states, energies = solver.solve()
        assert len(states) == 2
        assert len(energies) == 2

    def test_solve_non_hermitian_raises(self):
        H = np.array([[1, 1], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        with pytest.raises(ValueError):
            solver.solve()

    def test_solve_skip_hermitian_check(self):
        H = np.array([[1, 1], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham, check_hermitian=False)
        states, energies = solver.solve()
        assert len(states) == 2

    def test_ground_state(self):
        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        ket, E = solver.ground_state()
        assert np.isclose(E, 1.0)

    def test_excited_states(self):
        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states = solver.excited_states(n=1)
        assert len(states) == 2

    def test_eigenvalues_sorted(self):
        H = np.array([[2, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states, energies = solver.solve()
        assert energies[0] <= energies[1]


class TestSparseMatrixSolver:
    def test_creation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = SparseMatrixSolver(ham, num_eigenvalues=1, which="SM")
        assert solver._H is ham

    def test_ground_state(self):
        H = np.array([[5, 2, 0], [2, 4, 2], [0, 2, 3]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = SparseMatrixSolver(ham, num_eigenvalues=1, which="SM")
        ket, E = solver.ground_state()
        assert E is not None


class TestLanczosSolver:
    def test_creation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = LanczosSolver(ham)
        assert solver._H is ham

    def test_solve(self):
        H = np.array([[5, 2, 0], [2, 4, 2], [0, 2, 3]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = LanczosSolver(ham, num_eigenvalues=2)
        states, energies = solver.solve()
        assert len(states) >= 1


class TestTimeEvolutionSolver:
    def test_creation(self):
        H = np.array([[0, 1], [1, 0]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = TimeEvolutionSolver(ham)
        assert solver._H is ham

    def test_evolve(self):
        H = np.array([[0, 1], [1, 0]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = TimeEvolutionSolver(ham)
        initial = Ket(np.array([1, 0], dtype=complex))
        states, times = solver.evolve(initial, t0=0.0, num_steps=10)
        assert len(states) == 11
        assert len(times) == 11


class TestSplitStepFourierSolver:
    def test_creation(self):
        V_func = lambda x: x**2
        solver = SplitStepFourierSolver(potential=V_func)
        assert solver._N == 512

    def test_creation_with_params(self):
        V_func = lambda x: x**2
        solver = SplitStepFourierSolver(potential=V_func, num_points=128, x_range=5.0)
        assert solver._N == 128
        assert solver._L == 5.0

    def test_evolve(self):
        V_func = lambda x: x**2
        solver = SplitStepFourierSolver(potential=V_func, num_points=64, x_range=5.0)
        x = np.linspace(-5, 5, 64, endpoint=False)
        psi = np.exp(-(x**2))
        psi = psi / np.linalg.norm(psi)
        snapshots, times = solver.evolve(psi, num_steps=10, measure_interval=5)
        assert len(snapshots) >= 1


class TestNumerovSolver:
    def test_creation(self):
        solver = NumerovSolver(potential=lambda x: x**2, num_points=100)
        assert solver._N == 100

    def test_solve(self):
        solver = NumerovSolver(potential=lambda x: x**2, num_points=100)
        result = solver.solve()
        assert result is not None


class TestQuantumSolverFactory:
    def test_create_exact(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = QuantumSolverFactory.create(ham)
        assert solver is not None


class TestHydrogenAtomSolver:
    def test_creation(self):
        solver = HydrogenAtomSolver()
        assert solver is not None


class TestHarmonicOscillatorSolver:
    def test_creation(self):
        solver = HarmonicOscillatorSolver()
        assert solver is not None


class TestParticleInBoxSolver:
    def test_creation(self):
        solver = ParticleInBoxSolver(L=1.0)
        assert solver._L == 1.0
