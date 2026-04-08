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
    QuantumSolverFactory,
    HydrogenAtomSolver,
    HarmonicOscillatorSolver,
    ParticleInBoxSolver,
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
    SceneHamiltonianBuilder,
    SceneSymmetryAnalyzer,
    simulate,
    SimulationResult,
    Visualizer,
    quick_simulate,
    analyze_scene_symmetry,
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
    GroupRepresentation,
    LieAlgebra,
    SO3LieAlgebra,
    SU2LieAlgebra,
    GroupTheoryAnalyzer,
    analyze_group_theory,
    wigner_eckart_coupling,
    tensor_operator_matrix_element,
    SO4HydrogenAnalyzer,
    detect_hydrogen_so4_symmetry,
)

from .hydrogen_symmetry import (
    HydrogenSymmetryAnalyzer,
    HydrogenSO4Symmetry,
    SO4Generators,
    RungeLenzVector,
    AngularMomentumOperator,
    analyze_hydrogen_symmetry,
    create_hydrogen_so4_analyzer,
)

from .integration import (
    QuantumToAbstractBridge,
    SymmetryToQuantumBridge,
    LieAlgebraToQuantumBridge,
    GroupRepresentationBridge,
    integrate_with_abstract_phys,
    create_quantum_from_abstract,
    export_to_abstract_phys,
    quick_bridge,
    quick_lie_bridge,
)

from .explainer import (
    ResultExplainer,
    EnergySpectrumExplainer,
    QuantumStateExplainer,
    DynamicsExplainer,
    MeasurementExplainer,
    DecoherenceExplainer,
    CompositeExplainer,
    SymmetryExplainer,
    explain_quantum_system,
    explain_measurement_results,
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
    'QuantumSolverFactory',
    'HydrogenAtomSolver',
    'HarmonicOscillatorSolver',
    'ParticleInBoxSolver',
    
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
    'SceneHamiltonianBuilder',
    'SceneSymmetryAnalyzer',
    'simulate',
    'SimulationResult',
    'Visualizer',
    'quick_simulate',
    'analyze_scene_symmetry',
    
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
    'GroupRepresentation',
    'LieAlgebra',
    'SO3LieAlgebra',
    'SU2LieAlgebra',
    'GroupTheoryAnalyzer',
    'analyze_group_theory',
    'wigner_eckart_coupling',
    'tensor_operator_matrix_element',
    'SO4HydrogenAnalyzer',
    'detect_hydrogen_so4_symmetry',
    
    # Hydrogen SO4 Symmetry
    'HydrogenSymmetryAnalyzer',
    'HydrogenSO4Symmetry',
    'SO4Generators',
    'RungeLenzVector',
    'AngularMomentumOperator',
    'analyze_hydrogen_symmetry',
    'create_hydrogen_so4_analyzer',
    
    # Explainer
    'ResultExplainer',
    'EnergySpectrumExplainer',
    'QuantumStateExplainer',
    'DynamicsExplainer',
    'MeasurementExplainer',
    'DecoherenceExplainer',
    'CompositeExplainer',
    'SymmetryExplainer',
    'explain_quantum_system',
    'explain_measurement_results',
    
    # Integration with core and abstract_phys
    'QuantumToAbstractBridge',
    'SymmetryToQuantumBridge',
    'LieAlgebraToQuantumBridge',
    'GroupRepresentationBridge',
    'integrate_with_abstract_phys',
    'create_quantum_from_abstract',
    'export_to_abstract_phys',
    'quick_bridge',
    'quick_lie_bridge',
]
