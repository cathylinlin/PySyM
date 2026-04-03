"""
Quantum Physics Module

Provides quantum system modeling, simulation, and analysis.

Core Modules:
- states: Quantum state representations (Ket, Bra, DensityMatrix)
- hamiltonian: Hamiltonian operators and quantization
- solver: Schrodinger equation solvers
- simulator: Simulation frameworks
- interactive: SceneBuilder for arbitrary scenarios
- analysis: Symmetry analysis and result interpretation

Integrates with abstract_phys module for symmetry operations,
generators, and physical system frameworks.
"""

from .states import (
    QuantumState,
    Ket,
    Bra,
    StateVector,
    DensityMatrix,
    basis_state,
    bell_state,
    w_state,
    ghz_state,
    tensor_product,
    superposition,
)

from .hamiltonian import (
    HamiltonianOperator,
    MatrixHamiltonian,
    FreeParticleHamiltonian,
    HarmonicOscillatorHamiltonian,
    SpinHamiltonian,
    HydrogenAtomHamiltonian,
    PerturbationHamiltonian,
    CanonicalQuantizer,
    HamiltonianBuilder,
    pauli_matrices,
    spin_operators,
    angular_momentum_operators,
    from_quantum_system,
)

from .solver import (
    Solver,
    ExactDiagonalizationSolver,
    SparseMatrixSolver,
    LanczosSolver,
    TimeEvolutionSolver,
    SplitStepFourierSolver,
    VariationalSolver,
    NumerovSolver,
    solve_schrodinger,
    time_evolve,
    compute_spectrum,
)

from .simulator import (
    Simulator,
    QuantumSimulator,
    MeasurementSimulator,
    DecoherenceSimulator,
    ParticleFieldSimulator,
    ScatteringSimulator,
    SpinChainSimulator,
    create_simulation,
)

from .interactive import (
    Particle,
    Potential,
    QuantumScene,
    SceneBuilder,
    HamiltonianBuilder,
    simulate,
    SimulationResult,
    Visualizer,
    quick_simulate,
)

from .analysis import (
    QuantumAnalyzer,
    QuantumSymmetryOperation,
    QuantumParityOperation,
    QuantumTranslationOperation,
    SymmetryInfo,
    StateClassification,
    TransitionRule,
    AnalysisResult,
    analyze,
    quick_report,
    check_parity,
)

__all__ = [
    # States
    'QuantumState',
    'Ket',
    'Bra',
    'StateVector',
    'DensityMatrix',
    'basis_state',
    'bell_state',
    'w_state',
    'ghz_state',
    'tensor_product',
    'superposition',
    
    # Hamiltonian
    'HamiltonianOperator',
    'MatrixHamiltonian',
    'FreeParticleHamiltonian',
    'HarmonicOscillatorHamiltonian',
    'SpinHamiltonian',
    'HydrogenAtomHamiltonian',
    'PerturbationHamiltonian',
    'CanonicalQuantizer',
    'HamiltonianBuilder',
    'pauli_matrices',
    'spin_operators',
    'angular_momentum_operators',
    'from_quantum_system',
    
    # Solvers
    'Solver',
    'ExactDiagonalizationSolver',
    'SparseMatrixSolver',
    'LanczosSolver',
    'TimeEvolutionSolver',
    'SplitStepFourierSolver',
    'VariationalSolver',
    'NumerovSolver',
    'solve_schrodinger',
    'time_evolve',
    'compute_spectrum',
    
    # Simulators
    'Simulator',
    'QuantumSimulator',
    'MeasurementSimulator',
    'DecoherenceSimulator',
    'ParticleFieldSimulator',
    'ScatteringSimulator',
    'SpinChainSimulator',
    'create_simulation',
    
    # Interactive
    'Particle',
    'Potential',
    'QuantumScene',
    'SceneBuilder',
    'simulate',
    'SimulationResult',
    'Visualizer',
    'quick_simulate',
    
    # Analysis
    'QuantumAnalyzer',
    'QuantumSymmetryOperation',
    'QuantumParityOperation',
    'QuantumTranslationOperation',
    'SymmetryInfo',
    'StateClassification',
    'TransitionRule',
    'AnalysisResult',
    'analyze',
    'quick_report',
    'check_parity',
]
