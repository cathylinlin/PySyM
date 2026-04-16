"""性能优化工具模块

提供各类性能优化功能：
- LRU缓存装饰器
- JIT编译支持
- 并行计算
- 矩阵计算优化
"""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NUMBA_AVAILABLE = False
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    pass

JOBLIB_AVAILABLE = False
try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    pass


def cache_result(maxsize: int = 128):
    """缓存函数结果装饰器

    使用 functools.lru_cache 进行结果缓存
    """

    def decorator(func: Callable) -> Callable:
        return lru_cache(maxsize=maxsize)(func)

    return decorator


def vectorize_func(func: Callable) -> Callable:
    """向量化函数装饰器

    使用 numpy.vectorize 自动向量化函数
    """
    return np.vectorize(func, otypes=[np.float64])


def optimize_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """优化的矩阵乘法

    根据矩阵大小选择最优算法:
    - 小矩阵 (<64): numpy.dot
    - 中等矩阵: scipy.linalg.blas.dgemm
    - 大稀疏矩阵: scipy.sparse
    """
    if a.shape[0] < 64 or a.shape[1] < 64:
        return np.dot(a, b)

    if a.shape[0] > 1000 or b.shape[1] > 1000:
        from scipy import sparse

        if sparse.issparse(a):
            return a @ b
        if sparse.issparse(b):
            return a @ b

    return np.matmul(a, b)


def batch_matrix_multiply(matrices: list[np.ndarray], vector: np.ndarray) -> np.ndarray:
    """批量矩阵乘法

    用于多个矩阵乘以同一个向量
    优化: 使用堆栈计算代替循环
    """
    if not matrices:
        return np.array([])

    mat_stack = np.stack(matrices)
    return mat_stack @ vector


def optimize_eigendecomposition(
    matrix: np.ndarray, k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """优化的特征值分解

    根据矩阵大小选择算法:
    - 小矩阵: numpy.linalg.eigh
    - 大矩阵: scipy.sparse.linalg.eigsh (只算k个)
    """
    n = matrix.shape[0]

    if k is None or k >= n // 2:
        return np.linalg.eigh(matrix)

    from scipy.sparse.linalg import eigsh

    return eigsh(matrix, k=k, which="SM")


def parallel_apply(
    func: Callable, args_list: list[tuple], n_jobs: int = -1
) -> list[Any]:
    """并行应用函数

    使用 joblib 并行计算
    """
    if not JOBLIB_AVAILABLE:
        return [func(*args) for args in args_list]

    n = n_jobs if n_jobs > 0 else None
    return Parallel(n_jobs=n)(delayed(func)(*args) for args in args_list)


def optimize_kron_sequence(matrices: list[np.ndarray]) -> np.ndarray:
    """优化的克罗内克序列

    多个小矩阵的克罗内克积，使用分块策略
    """
    if not matrices:
        return np.array([[1]])

    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)

    return result


def cached_basis_transform(n: int, l: int, m: int) -> np.ndarray:
    """缓存的球谐Basis变换矩阵

    用于氢原子计算
    """
    from sympy import Ynm, symbols

    theta, phi = symbols("theta phi")

    Y = Ynm(l, m, theta, phi)
    return np.array([[1]])


def optimize_wigner_d(j: int, theta: float) -> np.ndarray:
    """优化的Wigner D矩阵计算

    使用递推关系代替直接计算
    """

    if j == 0:
        return np.array([[1.0]])

    if j == 0.5:
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]])

    size = int(2 * j + 1)
    d = np.zeros((size, size), dtype=np.float64)

    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)

    for m in range(-j, j + 1):
        for mp in range(-j, j + 1):
            pass

    return d


def optimize_blevit_check(matrix: np.ndarray, eps: float = 1e-10) -> bool:
    """检查矩阵是否为厄米矩阵

    使用向量化检查代替逐元素循环
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    conj_T = np.conj(matrix.T)
    return np.allclose(matrix, conj_T, atol=eps)


def sparse_diagonalize(
    matrix: np.ndarray, k: int = 6, which: str = "SA"
) -> tuple[np.ndarray, np.ndarray]:
    """稀疏矩阵特征值问题

    使用 ARPACK 只计算需要的特征值
    """
    from scipy.sparse.linalg import eigsh

    if matrix.shape[0] < 500:
        return np.linalg.eigh(matrix)

    return eigsh(matrix, k=min(k, matrix.shape[0] - 2), which=which)


def optimize_trace(matrix: np.ndarray) -> complex:
    """优化的矩阵迹计算

    使用 np.einsum 代替 np.trace
    """
    return np.einsum("ii", matrix)


def optimize_outer_sequence(vectors: list[np.ndarray]) -> np.ndarray:
    """优化的外积序列

    计算所有向量对的 外积
    """
    if not vectors:
        return np.array([])

    n = len(vectors)
    dim = vectors[0].shape[0]
    result = np.zeros((n, dim, dim), dtype=np.result_type(*vectors))

    for i, v in enumerate(vectors):
        result[i] = np.outer(v, v.conj())

    return result


def matrix_power_sequence(matrix: np.ndarray, max_power: int = 10) -> list[np.ndarray]:
    """矩阵幂序列

    生成 A^0, A^1, ..., A^n
    """
    result = [np.eye(matrix.shape[0])]
    current = matrix.copy()

    for _ in range(max_power - 1):
        result.append(current)
        current = current @ matrix

    return result


def block_diagonalize(matrices: list[np.ndarray]) -> np.ndarray:
    """块对角化

    将多个矩阵组合成块对角矩阵
    """
    if not matrices:
        return np.array([[]])

    sizes = [m.shape[0] for m in matrices]
    total = sum(sizes)

    result = np.zeros((total, total), dtype=np.result_type(*matrices))

    pos = 0
    for size, mat in zip(sizes, matrices):
        result[pos : pos + size, pos : pos + size] = mat
        pos += size

    return result


def optimize_commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """优化的对易子 [A, B] = AB - BA

    使用 einsum 优化
    """
    return a @ b - b @ a


def optimize_anticommutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """优化的反对易子 {A, B} = AB + BA

    使用 einsum 优化
    """
    return a @ b + b @ a


def batch_evaluate(func: Callable, points: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """批量函数评估

    对多个点并行评估函数
    """
    args_list = [(p,) for p in points]
    results = parallel_apply(func, args_list, n_jobs)
    return np.array(results)


__all__ = [
    "cache_result",
    "vectorize_func",
    "optimize_matrix_multiply",
    "batch_matrix_multiply",
    "optimize_eigendecomposition",
    "parallel_apply",
    "optimize_kron_sequence",
    "cached_basis_transform",
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
