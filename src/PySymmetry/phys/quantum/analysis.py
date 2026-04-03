"""
Quantum Analysis Module

Provides comprehensive analysis for quantum simulation results:
1. Symmetry analysis - detect symmetries, classify states
2. Spectrum analysis - energy levels, gaps, degeneracies
3. Selection rules - transition rules based on symmetry
4. State properties - parities, quantum numbers, invariants
5. Result explanation - human-readable reports

Integrates with abstract_phys for symmetry operations and generators.
"""
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from .states import Ket, StateVector
from .hamiltonian import HamiltonianOperator
from .interactive import SimulationResult

if TYPE_CHECKING:
    from PySymmetry.abstract_phys.symmetry_operations.base import SymmetryOperation
    from PySymmetry.abstract_phys.symmetry_operations.specific_operations import (
        ParityOperation,
        TranslationOperation,
        TimeReversalOperation,
    )

try:
    from PySymmetry.abstract_phys.symmetry_operations.base import SymmetryOperation
    from PySymmetry.abstract_phys.symmetry_operations.analyzer import SymmetryAnalyzer
    from PySymmetry.abstract_phys.symmetry_operations.generators import (
        MomentumGenerator,
        AngularMomentumGenerator,
        HamiltonianGenerator,
    )
    from PySymmetry.core.group_theory.discrete_groups import ParityGroup
    from PySymmetry.core.group_theory.continuous_groups import TranslationGroup
    _HAS_ABSTRACT_PHYS = True
except ImportError:
    SymmetryOperation = object
    SymmetryAnalyzer = object
    MomentumGenerator = object
    AngularMomentumGenerator = object
    HamiltonianGenerator = object
    _HAS_ABSTRACT_PHYS = False


@dataclass
class SymmetryInfo:
    """Information about a detected symmetry"""
    name: str
    description: str
    symmetry_type: str
    is_exact: bool
    conserved_quantity: Optional[str] = None
    generator: Optional[Any] = None


@dataclass
class StateClassification:
    """Classification of a quantum state by symmetry"""
    index: int
    energy: float
    parity: float
    irrep: str
    quantum_numbers: Dict[str, Any]
    is_degenerate: bool = False
    degeneracy_partners: List[int] = field(default_factory=list)


@dataclass
class TransitionRule:
    """Selection rule for a transition"""
    initial: int
    final: int
    allowed: bool
    energy_gap: float
    parity_change: int


@dataclass 
class AnalysisResult:
    """Complete analysis result"""
    symmetries: List[SymmetryInfo]
    state_classifications: List[StateClassification]
    transition_rules: List[TransitionRule]
    conserved_quantities: Dict[str, Dict[str, Any]]
    invariants: Dict[str, float]


class QuantumSymmetryOperation:
    """
    Base class for quantum-specific symmetry operations.
    
    Extends abstract_phys SymmetryOperation for quantum systems.
    """
    
    def __init__(self, dimension: int = None):
        self._dimension = dimension
        self._cached_matrix: Optional[np.ndarray] = None
    
    def representation_matrix(self, dim: int = None) -> np.ndarray:
        """Get the representation matrix"""
        d = dim or self._dimension
        if self._cached_matrix is not None and self._cached_matrix.shape[0] == d:
            return self._cached_matrix
        matrix = self._compute_matrix(d)
        self._cached_matrix = matrix
        return matrix
    
    def _compute_matrix(self, dim: int) -> np.ndarray:
        """Compute the matrix representation - subclasses override"""
        return np.eye(dim)
    
    def apply_to_state(self, psi: np.ndarray) -> np.ndarray:
        """Apply symmetry operation to wavefunction"""
        return self.representation_matrix(len(psi)) @ psi


class QuantumParityOperation(QuantumSymmetryOperation):
    """
    Quantum parity (spatial inversion) operation: x -> -x
    """
    
    def __init__(self, dimension: int = None):
        super().__init__(dimension)
    
    @property
    def group(self):
        if _HAS_ABSTRACT_PHYS:
            return ParityGroup()
        return None
    
    def _compute_matrix(self, dim: int) -> np.ndarray:
        """Parity matrix reverses basis order"""
        matrix = np.zeros((dim, dim))
        for i in range(dim):
            matrix[i, dim - 1 - i] = 1.0
        return matrix
    
    def eigenvalue(self, psi: np.ndarray) -> float:
        """Compute parity eigenvalue of a state"""
        parity_psi = self.apply_to_state(psi)
        overlap = np.vdot(psi, parity_psi).real
        return 1.0 if overlap > 0 else -1.0


class QuantumTranslationOperation(QuantumSymmetryOperation):
    """
    Translation operation: x -> x + a
    
    Used for periodic systems and crystal momentum.
    """
    
    def __init__(self, displacement: float, dimension: int = None):
        super().__init__(dimension)
        self._displacement = displacement
    
    @property
    def group(self):
        if _HAS_ABSTRACT_PHYS:
            return TranslationGroup(1)
        return None
    
    def _compute_matrix(self, dim: int) -> np.ndarray:
        """Translation matrix in position basis"""
        matrix = np.zeros((dim, dim))
        for i in range(dim):
            matrix[i, (i - 1) % dim] = 1.0
        return matrix


class QuantumAnalyzer:
    """
    Main analyzer for quantum simulation results.
    
    Provides unified interface for:
    - Symmetry detection
    - State classification
    - Selection rules
    - Conservation laws
    - Invariant computation
    
    Integrates with abstract_phys SymmetryAnalyzer when available.
    """
    
    def __init__(
        self,
        hamiltonian: Optional[HamiltonianOperator] = None,
        result: Optional[SimulationResult] = None
    ):
        self._H = hamiltonian
        self._result = result
        
        self._parity_op: Optional[QuantumParityOperation] = None
        self._translation_op: Optional[QuantumTranslationOperation] = None
        
        self._abstract_analyzer: Optional[SymmetryAnalyzer] = None
        if _HAS_ABSTRACT_PHYS and hamiltonian is not None:
            if hasattr(hamiltonian, 'system'):
                self._abstract_analyzer = SymmetryAnalyzer(hamiltonian.system)
        
        self._symmetries: List[SymmetryInfo] = []
        self._classifications: List[StateClassification] = []
    
    @property
    def has_abstract_phys(self) -> bool:
        """Check if abstract_phys is available"""
        return _HAS_ABSTRACT_PHYS
    
    def get_parity_operation(self, dim: int = None) -> QuantumParityOperation:
        """Get the parity operation"""
        if dim is None and self._result is not None:
            dim = len(self._result.grid)
        if self._parity_op is None or (dim and self._parity_op._dimension != dim):
            self._parity_op = QuantumParityOperation(dim)
        return self._parity_op
    
    def detect_symmetries(self, tolerance: float = 1e-8) -> List[SymmetryInfo]:
        """Detect symmetries in the quantum system"""
        self._symmetries = []
        
        if self._H is None:
            return self._symmetries
        
        H_matrix = self._H.matrix
        dim = H_matrix.shape[0]
        
        parity_op = self.get_parity_operation(dim)
        parity_mat = parity_op.representation_matrix()
        
        if self._commutes(H_matrix, parity_mat, tolerance):
            self._symmetries.append(SymmetryInfo(
                name="Parity",
                description="Spatial inversion symmetry x -> -x",
                symmetry_type="discrete",
                is_exact=True,
                conserved_quantity="Parity quantum number (+/- 1)"
            ))
        
        if self._has_uniform_grid():
            self._symmetries.append(SymmetryInfo(
                name="Translation",
                description="Translational symmetry",
                symmetry_type="continuous",
                is_exact=True,
                conserved_quantity="Crystal momentum"
            ))
        
        if self._is_time_independent():
            self._symmetries.append(SymmetryInfo(
                name="TimeTranslation",
                description="Time translation symmetry",
                symmetry_type="continuous",
                is_exact=True,
                conserved_quantity="Energy",
                generator=HamiltonianGenerator() if _HAS_ABSTRACT_PHYS else None
            ))
        
        return self._symmetries
    
    def _commutes(self, A: np.ndarray, B: np.ndarray, tol: float) -> bool:
        """Check if two matrices commute"""
        commutator = A @ B - B @ A
        return bool(np.linalg.norm(commutator) < tol * np.linalg.norm(A))
    
    def _has_uniform_grid(self) -> bool:
        """Check if grid has uniform spacing"""
        if self._result is None:
            return False
        grid = self._result.grid
        if len(grid.shape) > 1:
            grid = grid[:, 0]
        spacing = np.diff(grid)
        return bool(np.allclose(spacing, spacing[0], rtol=1e-5))
    
    def _is_time_independent(self) -> bool:
        """Check if Hamiltonian is time-independent"""
        if self._H is None:
            return True
        return not getattr(self._H, 'is_time_dependent', lambda: False)()
    
    def analyze_parity(self, state: Ket) -> Tuple[float, str]:
        """Analyze parity of a quantum state"""
        psi = state.to_vector()
        parity_op = self.get_parity_operation(len(psi))
        parity = parity_op.eigenvalue(psi)
        desc = "even (symmetric)" if parity > 0 else "odd (antisymmetric)"
        return parity, desc
    
    def classify_states(self) -> List[StateClassification]:
        """Classify all eigenstates by symmetry"""
        self._classifications = []
        
        if self._result is None:
            return self._classifications
        
        symmetries = self.detect_symmetries()
        degenerate_groups = self._find_degenerate_states()
        deg_map = {idx: group for group in degenerate_groups for idx in group}
        
        for i in range(len(self._result.states)):
            psi = self._result.states[i]
            
            parity = 0.0
            if psi is not None:
                parity, _ = self.analyze_parity(psi)
            
            is_degenerate = i in deg_map
            partners = deg_map.get(i, [])
            
            irrep = self._identify_irrep(parity, len(partners))
            
            qn = {
                'n': i,
                'E': self._result.energies[i],
                'parity': '+' if parity > 0 else '-' if parity < 0 else '0'
            }
            
            self._classifications.append(StateClassification(
                index=i,
                energy=self._result.energies[i],
                parity=parity,
                irrep=irrep,
                quantum_numbers=qn,
                is_degenerate=is_degenerate,
                degeneracy_partners=partners
            ))
        
        return self._classifications
    
    def _identify_irrep(self, parity: float, degeneracy: int) -> str:
        """Identify irreducible representation"""
        if degeneracy > 1:
            return f"D_{degeneracy}"
        if abs(parity - 1.0) < 0.1:
            return "A" if degeneracy == 1 else f"A_{degeneracy}"
        if abs(parity + 1.0) < 0.1:
            return "B" if degeneracy == 1 else f"B_{degeneracy}"
        return "?"
    
    def _find_degenerate_states(self, tol: float = 1e-6) -> List[List[int]]:
        """Find groups of degenerate states"""
        if self._result is None:
            return []
        
        energies = self._result.energies
        groups = []
        used = np.zeros(len(energies), dtype=bool)
        
        for i in range(len(energies)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(energies)):
                if not used[j] and abs(energies[i] - energies[j]) < tol:
                    group.append(j)
                    used[j] = True
            groups.append(group)
            used[i] = True
        
        return [g for g in groups if len(g) > 1]
    
    def compute_selection_rules(self) -> List[TransitionRule]:
        """Compute electric dipole selection rules"""
        if self._result is None:
            return []
        
        classifications = self.classify_states()
        rules = []
        
        for i in range(min(len(classifications), 10)):
            for f in range(i + 1, min(len(classifications), 10)):
                p_i = classifications[i].parity
                p_f = classifications[f].parity
                
                allowed = bool(p_i * p_f < 0)
                
                rules.append(TransitionRule(
                    initial=i,
                    final=f,
                    allowed=allowed,
                    energy_gap=self._result.energies[f] - self._result.energies[i],
                    parity_change=int(p_f - p_i)
                ))
        
        return rules
    
    def compute_conserved_quantities(self) -> Dict[str, Dict[str, Any]]:
        """Compute conserved quantities from symmetries"""
        conserved = {}
        
        symmetries = self.detect_symmetries()
        
        for sym in symmetries:
            if sym.name == "Parity":
                conserved['parity'] = {
                    'operator': 'Parity (P)',
                    'conserved': True,
                    'eigenvalues': '+/- 1',
                    'generator': 'Position inversion'
                }
            elif sym.name == "Translation":
                conserved['momentum'] = {
                    'operator': 'Momentum (p)',
                    'conserved': True,
                    'eigenvalues': 'Continuous',
                    'generator': MomentumGenerator(1) if _HAS_ABSTRACT_PHYS else 'Spatial translation'
                }
            elif sym.name == "TimeTranslation":
                conserved['energy'] = {
                    'operator': 'Hamiltonian (H)',
                    'conserved': True,
                    'eigenvalues': 'E_n',
                    'generator': HamiltonianGenerator() if _HAS_ABSTRACT_PHYS else 'Time evolution'
                }
        
        return conserved
    
    def compute_invariants(self) -> Dict[str, float]:
        """Compute quantum invariants"""
        invariants = {}
        
        if self._result is None:
            return invariants
        
        if len(self._result.energies) >= 2:
            invariants['energy_gap'] = float(abs(self._result.energies[1] - self._result.energies[0]))
        
        invariants['degeneracy_count'] = float(len(self._find_degenerate_states()))
        
        if self._result.grid is not None:
            grid = self._result.grid
            if hasattr(grid, 'shape') and len(grid.shape) > 1:
                grid = grid[:, 0]
            invariants['system_size'] = float(np.max(grid) - np.min(grid))
        
        return invariants
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        lines = []
        lines.append("=" * 60)
        lines.append("QUANTUM SYSTEM ANALYSIS REPORT")
        lines.append("=" * 60)
        
        symmetries = self.detect_symmetries()
        lines.append("\n## Symmetries")
        lines.append("-" * 40)
        for sym in symmetries:
            lines.append(f"  {sym.name}: {sym.description}")
            lines.append(f"    Type: {sym.symmetry_type}, Exact: {sym.is_exact}")
            if sym.conserved_quantity:
                lines.append(f"    Conserved: {sym.conserved_quantity}")
        
        classifications = self.classify_states()
        lines.append("\n## State Classifications")
        lines.append("-" * 40)
        if classifications:
            lines.append(f"{'n':<4} {'E':<12} {'Parity':<8} {'Irrep':<6} {'Quantum Numbers'}")
            for cls in classifications[:8]:
                p = f"{cls.parity:+.0f}" if isinstance(cls.parity, (int, float)) else "?"
                lines.append(f"{cls.index:<4} {cls.energy:<12.4f} {p:<8} {cls.irrep:<6} {cls.quantum_numbers}")
        
        rules = self.compute_selection_rules()
        allowed = [r for r in rules if r.allowed]
        lines.append(f"\n## Selection Rules")
        lines.append("-" * 40)
        lines.append(f"  Allowed: {len(allowed)}, Forbidden: {len(rules) - len(allowed)}")
        if allowed[:3]:
            lines.append("  Sample allowed transitions:")
            for r in allowed[:3]:
                lines.append(f"    |{r.initial}> -> |{r.final}>: dE = {r.energy_gap:.4f}")
        
        conserved = self.compute_conserved_quantities()
        lines.append("\n## Conservation Laws")
        lines.append("-" * 40)
        for name, info in conserved.items():
            lines.append(f"  {name}: {info['eigenvalues']}")
        
        invariants = self.compute_invariants()
        lines.append("\n## Invariants")
        lines.append("-" * 40)
        for name, value in invariants.items():
            lines.append(f"  {name}: {value}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
    
    def analyze(self) -> AnalysisResult:
        """Perform complete analysis and return structured result"""
        return AnalysisResult(
            symmetries=self.detect_symmetries(),
            state_classifications=self.classify_states(),
            transition_rules=self.compute_selection_rules(),
            conserved_quantities=self.compute_conserved_quantities(),
            invariants=self.compute_invariants()
        )


def analyze(
    hamiltonian: HamiltonianOperator,
    result: SimulationResult
) -> AnalysisResult:
    """
    Perform complete quantum analysis.
    
    Args:
        hamiltonian: Hamiltonian operator
        result: Simulation result with eigenstates
        
    Returns:
        Structured AnalysisResult
    """
    analyzer = QuantumAnalyzer(hamiltonian, result)
    return analyzer.analyze()


def quick_report(
    hamiltonian: HamiltonianOperator,
    result: SimulationResult
) -> str:
    """
    Generate quick analysis report.
    
    Args:
        hamiltonian: Hamiltonian operator
        result: Simulation result
        
    Returns:
        Report string
    """
    analyzer = QuantumAnalyzer(hamiltonian, result)
    return analyzer.generate_report()


def check_parity(state: Ket, dim: int = None) -> Tuple[float, str]:
    """
    Quick parity check for a state.
    
    Args:
        state: Quantum state
        dim: Optional dimension
        
    Returns:
        (parity eigenvalue, description)
    """
    analyzer = QuantumAnalyzer()
    if dim:
        analyzer.get_parity_operation(dim)
    return analyzer.analyze_parity(state)
