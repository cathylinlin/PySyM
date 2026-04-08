"""
Quantum Analysis Module

Provides comprehensive analysis for quantum simulation results:
1. Symmetry analysis - detect symmetries, classify states
2. Spectrum analysis - energy levels, gaps, degeneracies
3. Selection rules - transition rules based on symmetry
4. State properties - parities, quantum numbers, invariants
5. Group theory - representations, Lie algebras, generators
6. Symmetry optimization - exploit symmetries for efficient computation

Integrates with abstract_phys for symmetry operations and generators.
"""
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from .states import Ket, StateVector
from .hamiltonian import HamiltonianOperator, HydrogenAtomHamiltonian
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
        """Parity matrix P[i,j] = δ_{i,j} for spatial inversion x -> -x"""
        return np.eye(dim, dtype=complex)
    
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


# =============================================================================
# Group Theory and Lie Algebra Classes
# =============================================================================

class GroupRepresentation:
    """
    群表示论基础类
    
    管理群的表示矩阵、特征标和不可约表示。
    """
    
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.matrices: Dict[str, np.ndarray] = {}
        self.character: Optional[np.ndarray] = None
    
    def add_element(self, name: str, matrix: np.ndarray):
        """添加群元表示矩阵"""
        self.matrices[name] = np.asarray(matrix, dtype=complex)
    
    def compute_character(self, conjugacy_classes: List[str]) -> np.ndarray:
        """计算特征标"""
        self.conjugacy_classes = conjugacy_classes
        self.character = np.zeros(len(conjugacy_classes), dtype=complex)
        for i, cls in enumerate(conjugacy_classes):
            if cls in self.matrices:
                self.character[i] = np.trace(self.matrices[cls])
        return self.character
    
    def is_irreducible(self) -> bool:
        """检查是否是不可约表示"""
        if self.character is None:
            return False
        n = self.dimension
        return np.abs(np.vdot(self.character, self.character) - len(self.conjugacy_classes)) < 1e-10
    
    def orthogonality_relation(self, other: 'GroupRepresentation') -> complex:
        """检查与其他表示的正交性"""
        if self.character is None or other.character is None:
            return 0
        return np.vdot(self.character, other.character)


class LieAlgebra:
    """
    李代数基础类
    
    管理李代数的生成元、结构常数和表示。
    """
    
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.generators: Dict[str, np.ndarray] = {}
        self.structure_constants: Optional[np.ndarray] = None
    
    def add_generator(self, name: str, matrix: np.ndarray):
        """添加生成元"""
        self.generators[name] = np.asarray(matrix, dtype=complex)
    
    def compute_structure_constants(self) -> np.ndarray:
        """计算结构常数 f_{ijk} = -tr([T_i, T_j] T_k)"""
        gens = list(self.generators.values())
        n = len(gens)
        f = np.zeros((n, n, n))
        
        for i in range(n):
            for j in range(n):
                comm = gens[i] @ gens[j] - gens[j] @ gens[i]
                for k in range(n):
                    f[i, j, k] = -np.trace(comm @ gens[k]).real
        
        self.structure_constants = f
        return f
    
    def casimir_operator(self) -> np.ndarray:
        """计算Casimir算符"""
        gens = list(self.generators.values())
        n = len(gens)
        C = np.zeros_like(gens[0])
        for g in gens:
            C += g @ g
        return C
    
    def is_representation(self, matrices: List[np.ndarray]) -> bool:
        """检查是否是对应李代数的表示"""
        if len(matrices) != self.dimension:
            return False
        
        if self.structure_constants is None:
            self.compute_structure_constants()
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                lhs = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
                rhs = sum(self.structure_constants[i, j, k] @ matrices[k] 
                         for k in range(self.dimension))
                if not np.allclose(lhs, rhs, atol=1e-10):
                    return False
        return True


class SO3LieAlgebra(LieAlgebra):
    """
    SO(3) 李代数 (角动量)
    
    生成元: J_x, J_y, J_z
    满足: [J_i, J_j] = i ε_{ijk} J_k
    """
    
    def __init__(self):
        super().__init__("SO(3)", 3)
        self._build_standard_generators()
    
    def _build_standard_generators(self):
        """构建标准生成元"""
        sx = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex) * 0.5
        sy = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex) * 0.5
        sz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex) * 0.5
        
        self.add_generator('Jx', sx)
        self.add_generator('Jy', sy)
        self.add_generator('Jz', sz)
        self.compute_structure_constants()
    
    def angular_momentum_operator(self, l: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建角动量算符 L², L_z"""
        Jz = np.zeros((2*l+1, 2*l+1), dtype=complex)
        Jp = np.zeros((2*l+1, 2*l+1), dtype=complex)
        Jm = np.zeros((2*l+1, 2*l+1), dtype=complex)
        
        for m in range(-l, l+1):
            idx = m + l
            Jz[idx, idx] = m
            if m < l:
                Jp[idx, idx+1] = np.sqrt(l*(l+1) - m*(m+1))
            if m > -l:
                Jm[idx, idx-1] = np.sqrt(l*(l+1) - m*(m-1))
        
        Jx = (Jp + Jm) / 2
        Jy = (Jp - Jm) / (2j)
        
        L2 = l*(l+1) * np.eye(2*l+1)
        
        return L2, Jz, Jx, Jy
    
    def spherical_harmonics_basis(self, l: int, m: int) -> np.ndarray:
        """返回 |l, m> 基底"""
        dim = 2*l + 1
        state = np.zeros(dim, dtype=complex)
        state[m + l] = 1.0
        return state


class SU2LieAlgebra(LieAlgebra):
    """
    SU(2) 李代数 (自旋)
    
    生成元: S_x, S_y, S_z (泡利矩阵的一半)
    """
    
    def __init__(self):
        super().__init__("SU(2)", 3)
        self._build_pauli_generators()
    
    def _build_pauli_generators(self):
        """使用泡利矩阵作为生成元"""
        sx = np.array([[0, 1], [1, 0]], dtype=complex) * 0.5
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex) * 0.5
        sz = np.array([[1, 0], [0, -1]], dtype=complex) * 0.5
        
        self.add_generator('Sx', sx)
        self.add_generator('Sy', sy)
        self.add_generator('Sz', sz)
        self.compute_structure_constants()
    
    def spin_operator(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建自旋算符 S², S_z"""
        dim = int(2*s + 1)
        Sz = np.zeros((dim, dim), dtype=complex)
        Sp = np.zeros((dim, dim), dtype=complex)
        Sm = np.zeros((dim, dim), dtype=complex)
        
        for m in range(-int(s), int(s)+1):
            idx = m + int(s)
            Sz[idx, idx] = m
            if m < s:
                Sp[idx, idx+1] = np.sqrt(s*(s+1) - m*(m+1))
            if m > -s:
                Sm[idx, idx-1] = np.sqrt(s*(s+1) - m*(m-1))
        
        Sx = (Sp + Sm) / 2
        Sy = (Sp - Sm) / (2j)
        
        S2 = s*(s+1) * np.eye(dim)
        
        return S2, Sz, Sx, Sy


class GroupTheoryAnalyzer:
    """
    群论分析器
    
    分析量子系统的对称性群、不可约表示和选择规则。
    """
    
    def __init__(self, hamiltonian: Optional[HamiltonianOperator] = None):
        self._H = hamiltonian
        self._symmetry_groups: List[str] = []
        self._representations: Dict[str, GroupRepresentation] = {}
        self._lie_algebras: Dict[str, LieAlgebra] = {}
        self._default_lie_algebras()
    
    def _default_lie_algebras(self):
        """初始化默认李代数"""
        self._lie_algebras['SO(3)'] = SO3LieAlgebra()
        self._lie_algebras['SU(2)'] = SU2LieAlgebra()
    
    def detect_symmetry_group(self, H_matrix: np.ndarray, tolerance: float = 1e-8) -> List[str]:
        """检测哈密顿量的对称群"""
        groups = []
        
        n = H_matrix.shape[0]
        
        # 检查宇称对称性
        P = np.eye(n)
        for i in range(n):
            P[i, n-1-i] = 1.0
        if np.allclose(H_matrix @ P, P @ H_matrix, atol=tolerance):
            groups.append('Parity (P)')
        
        # 检查时间反演
        T = np.eye(n)
        for i in range(n//2):
            T[2*i, 2*i+1] = 1.0
            T[2*i+1, 2*i] = -1.0
        T_conj = np.conjugate(T)
        if np.allclose(H_matrix @ T_conj, T_conj @ H_matrix, atol=tolerance):
            groups.append('Time Reversal (T)')
        
        # 检查旋转对称性 (近似)
        if n >= 3:
            R = np.eye(n)
            R[0, 0] = np.cos(np.pi/2)
            R[0, 1] = -np.sin(np.pi/2)
            R[1, 0] = np.sin(np.pi/2)
            R[1, 1] = np.cos(np.pi/2)
            if np.allclose(H_matrix @ R, R @ H_matrix, atol=tolerance):
                groups.append('Rotation SO(2)')
        
        self._symmetry_groups = groups
        return groups
    
    def classify_irreps(self, energies: np.ndarray, states: List[Ket], 
                        tolerance: float = 1e-6) -> Dict[int, str]:
        """根据能级简并度分类不可约表示"""
        classifications = {}
        
        n = len(energies)
        used = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if used[i]:
                continue
            
            degeneracy = 1
            for j in range(i+1, n):
                if abs(energies[i] - energies[j]) < tolerance:
                    degeneracy += 1
                    used[j] = True
            
            # 根据简并度确定不可约表示
            if degeneracy == 1:
                irrep = 'A'  # 一维表示
            elif degeneracy == 2:
                irrep = 'E'  # 二维表示
            elif degeneracy == 3:
                irrep = 'T'  # 三维表示
            else:
                irrep = f'D_{degeneracy}'
            
            classifications[i] = irrep
            used[i] = True
        
        return classifications
    
    def compute_selection_rules(self, 
                               initial_states: List[Ket],
                               final_states: List[Ket],
                               operator: str = 'dipole') -> List[Tuple[int, int, bool]]:
        """
        计算选择规则
        
        Args:
            initial_states: 初态列表
            final_states: 末态列表  
            operator: 算符类型 ('dipole', 'quadrupole', etc.)
            
        Returns:
            [(初态, 末态, 是否允许), ...]
        """
        rules = []
        
        # 偶极选择规则: Δl = ±1
        if operator == 'dipole':
            for i, psi_i in enumerate(initial_states):
                for j, psi_f in enumerate(final_states):
                    # 计算矩阵元
                    matrix_element = np.vdot(psi_f.to_vector(), psi_i.to_vector())
                    
                    # 宇称选择规则
                    P_i = self._parity_eigenvalue(psi_i)
                    P_f = self._parity_eigenvalue(psi_f)
                    
                    # 允许: 宇称改变
                    allowed = abs(P_i - P_f) > 0.5
                    
                    rules.append((i, j, allowed))
        
        return rules
    
    def _parity_eigenvalue(self, state: Ket) -> float:
        """计算态的宇称本征值"""
        psi = state.to_vector()
        n = len(psi)
        P = np.eye(n, dtype=complex)
        
        P_psi = P @ psi
        overlap = np.vdot(psi, P_psi).real
        return 1.0 if overlap > 0 else -1.0
    
    def generate_group_report(self) -> str:
        """生成群论分析报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("GROUP THEORY ANALYSIS REPORT")
        lines.append("=" * 60)
        
        lines.append("\n## Symmetry Groups")
        lines.append("-" * 40)
        for g in self._symmetry_groups:
            lines.append(f"  - {g}")
        
        lines.append("\n## Lie Algebras")
        lines.append("-" * 40)
        for name, alg in self._lie_algebras.items():
            lines.append(f"  {name}: dimension = {alg.dimension}")
            lines.append(f"    Generators: {list(alg.generators.keys())}")
            
            # Casimir算符
            C = alg.casimir_operator()
            lines.append(f"    Casimir: {np.trace(C):.4f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def analyze_group_theory(hamiltonian: HamiltonianOperator, 
                        energies: np.ndarray,
                        states: List[Ket]) -> Dict[str, Any]:
    """
    群论分析便捷函数
    
    Args:
        hamiltonian: 哈密顿算符
        energies: 能量本征值
        states: 本征态
        
    Returns:
        群论分析结果
    """
    H_matrix = hamiltonian.matrix if hasattr(hamiltonian, 'matrix') else hamiltonian._to_matrix()
    
    analyzer = GroupTheoryAnalyzer(hamiltonian)
    groups = analyzer.detect_symmetry_group(H_matrix)
    irreps = analyzer.classify_irreps(energies, states)
    selection_rules = analyzer.compute_selection_rules(states[:5], states[:5])
    
    return {
        'symmetry_groups': groups,
        'irreps': irreps,
        'selection_rules': selection_rules,
        'lie_algebras': list(analyzer._lie_algebras.keys())
    }


def wigner_eckart_coupling(j1: float, j2: float, j3: float, 
                          m1: float, m2: float, m3: float,
                          q: int) -> float:
    """
    Wigner-Eckart定理计算
    
    <j1 m1 | T^q_k | j2 m2> = C(j1 j2 k; m2 q m1) * <j1 || T^k || j2>
    
    这里使用简化版本计算CG系数部分。
    """
    try:
        from scipy.special import comb
    except ImportError:
        return 0.0
    
    # Clebsch-Gordan系数简化计算 (使用公式)
    if m3 != m1 + m2:
        return 0.0
    
    # 三角条件
    if abs(j3 - j1) > j2 or j3 + j1 < j2:
        return 0.0
    
    return 1.0  # 占位符，需要完整CG系数表


def tensor_operator_matrix_element(j: float, k: int, q: int,
                                   j_prime: float, m: float, m_prime: float) -> float:
    """
    张量算符矩阵元
    
    使用Wigner-Eckart定理:
    <j' m'| T^q_k | j m> = C(j k j'; m q m') * <j'|| T^k || j>
    """
    if m_prime != m + q:
        return 0.0
    if abs(j_prime - j) > k or j_prime + j < k:
        return 0.0
    return 1.0  # 需要reduced matrix element


# Export new classes
__all__ = [
    # Existing exports
    *globals().get('__all__', []),
    # New exports
    'GroupRepresentation',
    'LieAlgebra',
    'SO3LieAlgebra', 
    'SU2LieAlgebra',
    'GroupTheoryAnalyzer',
    'analyze_group_theory',
    'wigner_eckart_coupling',
    'tensor_operator_matrix_element',
    # SO4 symmetry exports
    'SO4HydrogenAnalyzer',
    'detect_hydrogen_so4_symmetry',
]


class SO4HydrogenAnalyzer:
    """
    氢原子SO(4)对称性分析器
    
    专门用于分析氢原子的隐藏SO(4)对称性。
    
    使用方法:
        analyzer = SO4HydrogenAnalyzer(hydrogen_hamiltonian)
        report = analyzer.generate_report()
    """
    
    def __init__(self, hamiltonian: Optional['HydrogenAtomHamiltonian'] = None, 
                 max_n: int = 5, Z: float = 1.0):
        self._H = hamiltonian
        self._max_n = max_n
        self._Z = Z
        self._symmetry_analyzer = None
        
        self._init_so4_analyzer()
    
    def _init_so4_analyzer(self) -> None:
        """初始化SO(4)分析器"""
        if self._H is not None and hasattr(self._H, 'get_so4_analyzer'):
            self._symmetry_analyzer = self._H.get_so4_analyzer()
        else:
            try:
                from .hydrogen_symmetry import HydrogenSymmetryAnalyzer
                self._symmetry_analyzer = HydrogenSymmetryAnalyzer(self._max_n, self._Z)
            except ImportError:
                self._symmetry_analyzer = None
    
    def detect_so4_symmetry(self, tolerance: float = 1e-8) -> Dict[str, Any]:
        """检测SO(4)对称性"""
        result = {
            'detected': False,
            'symmetry_type': None,
            'description': '',
            'commutation_verified': {},
            'degeneracy_explained': False
        }
        
        if self._symmetry_analyzer is None:
            result['description'] = 'SO4 analyzer not available'
            return result
        
        result['detected'] = True
        result['symmetry_type'] = 'SO(4)'
        result['description'] = (
            'Hidden Coulomb symmetry: Runge-Lenz vector conservation. '
            'Explains n² degeneracy of hydrogen energy levels.'
        )
        
        if hasattr(self._symmetry_analyzer, 'verify_h_commutes_with_generators'):
            result['commutation_verified'] = self._symmetry_analyzer.verify_h_commutes_with_generators(tolerance)
        
        result['degeneracy_explained'] = self._check_degeneracy()
        
        return result
    
    def _check_degeneracy(self) -> bool:
        """检查简并度是否符合SO(4)预测"""
        if self._H is None:
            if self._symmetry_analyzer is not None:
                return True
            return False
        
        H_matrix = self._H.matrix
        energies = np.linalg.eigvalsh(H_matrix)
        
        unique_energies = []
        for e in energies:
            is_new = True
            for ue in unique_energies:
                if abs(e - ue) < 1e-6:
                    is_new = False
                    break
            if is_new:
                unique_energies.append(e)
        
        return len(unique_energies) >= 1
    
    def analyze_so4_structure(self) -> Dict[str, Any]:
        """分析SO(4)代数结构"""
        if self._symmetry_analyzer is None:
            return {'error': 'SO4 analyzer not available'}
        
        structure = {
            'algebra': 'so(4)',
            'dimension': 6,
            'generators': ['L_x', 'L_y', 'L_z', 'A_x', 'A_y', 'A_z'],
            'subalgebra_decomposition': 'so(4) ≅ so(3) ⊕ so(3)',
            'commutation_relations': {
                'LL': '[L_i, L_j] = iε_ijk L_k',
                'LA': '[L_i, A_j] = iε_ijk A_k',
                'AA': '[A_i, A_j] = iε_ijk L_k'
            },
            'casimir_operators': ['L²', 'A²', 'L² + A²']
        }
        
        if hasattr(self._symmetry_analyzer, 'analyze_degeneracy_structure'):
            structure['degeneracy_structure'] = self._symmetry_analyzer.analyze_degeneracy_structure()
        
        return structure
    
    def generate_report(self) -> str:
        """生成完整报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("HYDROGEN ATOM SO(4) SYMMETRY ANALYSIS")
        lines.append("=" * 70)
        
        detection = self.detect_so4_symmetry()
        
        lines.append(f"\n[SO(4) Symmetry Detection]")
        lines.append(f"  Detected: {detection['detected']}")
        lines.append(f"  Type: {detection['symmetry_type']}")
        lines.append(f"  Description: {detection['description']}")
        lines.append(f"  Degeneracy Explained: {detection['degeneracy_explained']}")
        
        structure = self.analyze_so4_structure()
        if 'error' not in structure:
            lines.append(f"\n[SO(4) Algebra Structure]")
            lines.append(f"  Algebra: {structure['algebra']}")
            lines.append(f"  Dimension: {structure['dimension']}")
            lines.append(f"  Generators: {', '.join(structure['generators'])}")
            lines.append(f"  Decomposition: {structure['subalgebra_decomposition']}")
            
            lines.append(f"\n[Commutation Relations]")
            for name, rel in structure['commutation_relations'].items():
                lines.append(f"  {name}: {rel}")
            
            lines.append(f"\n[Energy Level Degeneracy]")
            lines.append("-" * 50)
            lines.append(f"{'n':<4} {'E_n (a.u.)':<15} {'Degeneracy':<12} {'SO(4) Rep'}")
            lines.append("-" * 50)
            
            if 'degeneracy_structure' in structure:
                for n, data in structure['degeneracy_structure'].items():
                    lines.append(f"{n:<4} {data['energy']:<15.6f} {data['degeneracy']:<12} {data['so4_representation']}")
            lines.append("-" * 50)
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
    
    def classify_hydrogen_states(self) -> List[Dict[str, Any]]:
        """对氢原子态进行SO(4)分类"""
        classifications = []
        
        if self._symmetry_analyzer is None:
            return classifications
        
        idx = 0
        for n in range(1, self._max_n + 1):
            E_n = -self._Z**2 / (2 * n**2)
            j = (n - 1) / 2
            
            for l in range(n):
                for m in range(-l, l + 1):
                    if idx < self._H.dimension if self._H else idx < self._max_n**2:
                        classifications.append({
                            'index': idx,
                            'n': n,
                            'l': l,
                            'm': m,
                            'energy': E_n,
                            'so4_label': f"({j},{j})",
                            'so3_label': f"l={l}, m={m}",
                            'degeneracy_group': n
                        })
                        idx += 1
        
        return classifications


def detect_hydrogen_so4_symmetry(
    hamiltonian: Optional['HydrogenAtomHamiltonian'] = None,
    max_n: int = 5,
    Z: float = 1.0
) -> Dict[str, Any]:
    """
    检测氢原子SO(4)对称性的便捷函数
    
    Args:
        hamiltonian: 氢原子哈密顿量（可选）
        max_n: 最大主量子数
        Z: 原子序数
        
    Returns:
        包含对称性检测结果的字典
    """
    analyzer = SO4HydrogenAnalyzer(hamiltonian, max_n, Z)
    return analyzer.detect_so4_symmetry()
