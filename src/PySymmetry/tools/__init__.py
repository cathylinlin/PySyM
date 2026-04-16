"""PySymmetry 工具模块

该模块提供各种实用工具，包括：
- 符号计算工具（基于 SymPy）
- 性能优化工具
"""

from .lie_algebra_symbolic import (
    SymbolicKillingForm,
    SymbolicLieAlgebra,
    SymbolicLieAlgebraElement,
    SymbolicLieBracket,
    SymbolicWeylGroup,
    compute_structure_constants,
    generate_weyl_coordinates,
    verify_jacobi_identity,
)
from .performance import (
    JOBLIB_AVAILABLE,
    NUMBA_AVAILABLE,
    batch_evaluate,
    batch_matrix_multiply,
    block_diagonalize,
    cache_result,
    matrix_power_sequence,
    optimize_anticommutator,
    optimize_blevit_check,
    optimize_commutator,
    optimize_eigendecomposition,
    optimize_kron_sequence,
    optimize_matrix_multiply,
    optimize_outer_sequence,
    optimize_trace,
    optimize_wigner_d,
    parallel_apply,
    sparse_diagonalize,
    vectorize_func,
)

__all__ = [
    "SymbolicLieAlgebra",
    "SymbolicLieAlgebraElement",
    "SymbolicLieBracket",
    "SymbolicKillingForm",
    "SymbolicWeylGroup",
    "compute_structure_constants",
    "verify_jacobi_identity",
    "generate_weyl_coordinates",
    "cache_result",
    "vectorize_func",
    "optimize_matrix_multiply",
    "batch_matrix_multiply",
    "optimize_eigendecomposition",
    "parallel_apply",
    "optimize_kron_sequence",
    "optimize_wigner_d",
    "optimize_blevit_check",
    "sparse_diagonalize",
    "optimize_trace",
    "optimize_outer_sequence",
    "matrix_power_sequence",
    "block_diagonalize",
    "optimize_commutator",
    "optimize_anticommutator",
    "batch_evaluate",
    "NUMBA_AVAILABLE",
    "JOBLIB_AVAILABLE",
]
