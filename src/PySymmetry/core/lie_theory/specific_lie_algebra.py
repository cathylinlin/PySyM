"""具体矩阵李代数实现

提供 gl(n)、sl(n)、so(n)、sp(2n)、u(n)、su(n) 等经典实李代数的矩阵模型。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement, LieAlgebraProperties


def _mat_bracket(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


class MatrixLieAlgebraElement(LieAlgebraElement):
    """矩阵李代数元素（实或复矩阵）。"""

    def __init__(self, matrix: np.ndarray, lie_algebra: MatrixLieAlgebraBase):
        self.matrix = np.asarray(matrix)
        self._lie = lie_algebra

    @property
    def lie_algebra(self) -> MatrixLieAlgebraBase:
        return self._lie

    def __add__(self, other: LieAlgebraElement) -> MatrixLieAlgebraElement:
        if (
            not isinstance(other, MatrixLieAlgebraElement)
            or other._lie is not self._lie
        ):
            return NotImplemented
        return MatrixLieAlgebraElement(self.matrix + other.matrix, self._lie)

    def __sub__(self, other: LieAlgebraElement) -> MatrixLieAlgebraElement:
        if (
            not isinstance(other, MatrixLieAlgebraElement)
            or other._lie is not self._lie
        ):
            return NotImplemented
        return MatrixLieAlgebraElement(self.matrix - other.matrix, self._lie)

    def __mul__(self, scalar: float) -> MatrixLieAlgebraElement:
        return MatrixLieAlgebraElement(self.matrix * scalar, self._lie)

    def bracket(self, other: LieAlgebraElement) -> MatrixLieAlgebraElement:
        if (
            not isinstance(other, MatrixLieAlgebraElement)
            or other._lie is not self._lie
        ):
            raise TypeError("李括号仅对同一李代数中的元素定义")
        return MatrixLieAlgebraElement(
            _mat_bracket(self.matrix, other.matrix), self._lie
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatrixLieAlgebraElement):
            return False
        if other._lie is not self._lie:
            return False
        return np.allclose(self.matrix, other.matrix)

    def __str__(self) -> str:
        return str(self.matrix)


class MatrixLieAlgebraBase(LieAlgebra[MatrixLieAlgebraElement]):
    """矩阵李代数公共逻辑。"""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def zero(self) -> MatrixLieAlgebraElement:
        return MatrixLieAlgebraElement(
            np.zeros(self._matrix_shape(), dtype=self._dtype()), self
        )

    def bracket(
        self, x: MatrixLieAlgebraElement, y: MatrixLieAlgebraElement
    ) -> MatrixLieAlgebraElement:
        return x.bracket(y)

    def add(
        self, x: MatrixLieAlgebraElement, y: MatrixLieAlgebraElement
    ) -> MatrixLieAlgebraElement:
        return x + y

    def scalar_multiply(
        self, x: MatrixLieAlgebraElement, scalar: float
    ) -> MatrixLieAlgebraElement:
        return x * scalar

    def _matrix_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _dtype(self) -> np.dtype:
        return np.float64


class GeneralLinearLieAlgebra(MatrixLieAlgebraBase):
    """一般线性李代数 gl(n, R)，即全体 n×n 实矩阵，[A,B]=AB-BA。"""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        super().__init__(n * n)

    def _matrix_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats = []
        for i in range(self.n):
            for j in range(self.n):
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        arr = np.asarray(vector, dtype=np.float64).reshape(self.n, self.n)
        return MatrixLieAlgebraElement(arr, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        return element.matrix.reshape(-1).tolist()

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name=f"gl({self.n})",
            dimension=self.dimension,
            is_semisimple=False,
            is_simple=False,
            is_abelian=self.n == 1,
            root_system_type=None,
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"gl({self.n})"


class SpecialLinearLieAlgebra(MatrixLieAlgebraBase):
    """特殊线性李代数 sl(n, R)：迹为零的 n×n 实矩阵。"""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("sl(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * n - 1)

    def _matrix_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        for k in range(self.n - 1):
            h = np.zeros((self.n, self.n))
            h[k, k] = 1.0
            h[k + 1, k + 1] = -1.0
            mats.append(MatrixLieAlgebraElement(h, self))
        return mats

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self.n, self.n), dtype=np.float64)
        idx = 0
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                m[i, j] = vector[idx]
                idx += 1
        for k in range(self.n - 1):
            m[k, k] += vector[idx]
            m[k + 1, k + 1] -= vector[idx]
            idx += 1
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        m = element.matrix
        out: list[float] = []
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                out.append(float(m[i, j]))
        for k in range(self.n - 1):
            out.append(float(sum(m[i, i] for i in range(k + 1))))
        return out

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name=f"sl({self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=self.n >= 2,
            is_abelian=False,
            root_system_type=f"A{self.n - 1}",
            rank=self.n - 1,
        )

    def __str__(self) -> str:
        return f"sl({self.n})"


class OrthogonalLieAlgebra(MatrixLieAlgebraBase):
    """正交李代数 so(n)：实反对称 n×n 矩阵。"""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("so(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * (n - 1) // 2)

    def _matrix_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = np.zeros((self.n, self.n))
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self.n, self.n), dtype=np.float64)
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                m[i, j] = vector[idx]
                m[j, i] = -vector[idx]
                idx += 1
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        m = element.matrix
        out: list[float] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                out.append(float(m[i, j]))
        return out

    def properties(self) -> LieAlgebraProperties:
        t = f"D{self.n // 2}" if self.n % 2 == 0 else f"B{(self.n - 1) // 2}"
        return LieAlgebraProperties(
            name=f"so({self.n})",
            dimension=self.dimension,
            is_semisimple=self.n >= 3,
            is_simple=self.n >= 3 and self.n != 4,
            is_abelian=False,
            root_system_type=t,
            rank=self.n // 2,
        )

    def __str__(self) -> str:
        return f"so({self.n})"


def _symplectic_J(n: int) -> np.ndarray:
    """标准辛形式 J = [[0, I_n], [-I_n, 0]]（2n×2n）。"""
    d = 2 * n
    j = np.zeros((d, d), dtype=np.float64)
    j[:n, n:] = np.eye(n)
    j[n:, :n] = -np.eye(n)
    return j


def _sp2n_basis_matrices(n: int) -> list[np.ndarray]:
    """通过 X^T J + J X = 0 的零空间得到 sp(2n) 的一组基。"""
    d = 2 * n
    j = _symplectic_J(n)
    dim = d * d
    m = np.zeros((dim, dim), dtype=np.float64)
    for k in range(dim):
        x = np.zeros((d, d))
        x.flat[k] = 1.0
        y = x.T @ j + j @ x
        m[:, k] = y.flatten()
    ns = null_space(m)
    if ns.shape[1] == 0:
        raise RuntimeError("无法构造 sp(2n) 的基")
    basis: list[np.ndarray] = []
    for c in range(ns.shape[1]):
        basis.append(ns[:, c].reshape(d, d))
    return basis


class SymplecticLieAlgebra(MatrixLieAlgebraBase):
    """辛李代数 sp(2n, R)：满足 X^T J + J X = 0 的 2n×2n 实矩阵（J 为标准辛形式）。"""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        self._d = 2 * n
        self._basis_mats = _sp2n_basis_matrices(n)
        super().__init__(len(self._basis_mats))

    def _matrix_shape(self) -> tuple[int, int]:
        return (self._d, self._d)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        return [MatrixLieAlgebraElement(np.array(m), self) for m in self._basis_mats]

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        if len(vector) != self.dimension:
            raise ValueError(f"期望长度为 {self.dimension} 的向量")
        m = np.zeros((self._d, self._d), dtype=np.float64)
        for c, v in enumerate(vector):
            m += v * self._basis_mats[c]
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        # 基一般不正交，用最小二乘在基上展开
        b = element.matrix.flatten()
        a = np.column_stack([bm.flatten() for bm in self._basis_mats])
        coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return coef.tolist()

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name=f"sp({2 * self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=self.n >= 1,
            is_abelian=False,
            root_system_type=f"C{self.n}",
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"sp({2 * self.n})"


def _skew_hermitian_to_vector(m: np.ndarray) -> list[float]:
    n = m.shape[0]
    out: list[float] = []
    for i in range(n):
        out.append(float(np.imag(m[i, i])))
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.real(m[i, j])))
            out.append(float(np.imag(m[i, j])))
    return out


def _vector_to_skew_hermitian(v: list[float], n: int) -> np.ndarray:
    if len(v) != n * n:
        raise ValueError("向量长度与 u(n) 维数不符")
    m = np.zeros((n, n), dtype=np.complex128)
    idx = 0
    for i in range(n):
        m[i, i] = 1j * v[idx]
        idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            m[i, j] = re + 1j * im
            m[j, i] = -re + 1j * im
    return m


def _vector_to_skew_hermitian_su(v: list[float], n: int) -> np.ndarray:
    if len(v) != n * n - 1:
        raise ValueError("向量长度与 su(n) 维数不符")
    m = np.zeros((n, n), dtype=np.complex128)
    idx = 0
    for k in range(n - 1):
        c = v[idx]
        idx += 1
        m[k, k] += 1j * c
        m[n - 1, n - 1] -= 1j * c
    for i in range(n):
        for j in range(i + 1, n):
            re = v[idx]
            im = v[idx + 1]
            idx += 2
            m[i, j] = re + 1j * im
            m[j, i] = -re + 1j * im
    return m


def _skew_hermitian_su_to_vector(m: np.ndarray) -> list[float]:
    n = m.shape[0]
    out: list[float] = []
    for i in range(n - 1):
        out.append(float(np.imag(m[i, i])))
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(np.real(m[i, j])))
            out.append(float(np.imag(m[i, j])))
    return out


class UnitaryLieAlgebra(MatrixLieAlgebraBase):
    """酉李代数 ``u(n)``：``n×n`` 反厄米特矩阵（实维数 ``n²``）。

    与李群 ``U(n)`` 的指数映射一致：``exp: u(n) → U(n)``。若文献使用厄米生成元 ``H``，常见关系为
    ``H = -i X``（``X`` 为本库中的反厄米元素）；结构常数、Casimir 等需与所选约定一致。
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n 必须为正整数")
        self.n = n
        super().__init__(n * n)

    def _matrix_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def _dtype(self) -> np.dtype:
        return np.complex128

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for i in range(self.n):
            e = np.zeros((self.n, self.n), dtype=np.complex128)
            e[i, i] = 1j
            mats.append(MatrixLieAlgebraElement(e, self))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = np.zeros((self.n, self.n), dtype=np.complex128)
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
                f = np.zeros((self.n, self.n), dtype=np.complex128)
                f[i, j] = 1j
                f[j, i] = 1j
                mats.append(MatrixLieAlgebraElement(f, self))
        return mats

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        m = _vector_to_skew_hermitian(vector, self.n)
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        return _skew_hermitian_to_vector(element.matrix)

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name=f"u({self.n})",
            dimension=self.dimension,
            is_semisimple=False,
            is_simple=False,
            is_abelian=self.n == 1,
            root_system_type=None,
            rank=self.n,
        )

    def __str__(self) -> str:
        return f"u({self.n})"


class SpecialUnitaryLieAlgebra(MatrixLieAlgebraBase):
    """特殊酉李代数 ``su(n)``：迹为零的反厄米特矩阵（实维数 ``n²−1``）。

    同样采用反厄米模型；厄米生成元与反厄米李代数元素的关系与 ``u(n)`` 相同（通常 ``H = -i X``）。
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("su(n) 要求 n >= 2")
        self.n = n
        super().__init__(n * n - 1)

    def _matrix_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def _dtype(self) -> np.dtype:
        return np.complex128

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for k in range(self.n - 1):
            h = np.zeros((self.n, self.n), dtype=np.complex128)
            h[k, k] = 1j
            h[self.n - 1, self.n - 1] = -1j
            mats.append(MatrixLieAlgebraElement(h, self))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                e = np.zeros((self.n, self.n), dtype=np.complex128)
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
                f = np.zeros((self.n, self.n), dtype=np.complex128)
                f[i, j] = 1j
                f[j, i] = 1j
                mats.append(MatrixLieAlgebraElement(f, self))
        return mats

    def from_vector(self, vector: list[float]) -> MatrixLieAlgebraElement:
        m = _vector_to_skew_hermitian_su(vector, self.n)
        return MatrixLieAlgebraElement(m, self)

    def to_vector(self, element: MatrixLieAlgebraElement) -> list[float]:
        return _skew_hermitian_su_to_vector(element.matrix)

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name=f"su({self.n})",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=True,
            is_abelian=False,
            root_system_type=f"A{self.n - 1}",
            rank=self.n - 1,
        )

    def __str__(self) -> str:
        return f"su({self.n})"


class SO4LieAlgebra(MatrixLieAlgebraBase):
    """SO(4) 李代数

    so(4) ≅ so(3) ⊕ so(3)，是半单李代数，由6个生成元组成：
    - L_i (i=1,2,3): 角动量算符，对应so(3)的生成元
    - A_i (i=1,2,3): Runge-Lenz向量算符

    对易关系：
    - [L_i, L_j] = iε_ijk L_k
    - [L_i, A_j] = iε_ijk A_k
    - [A_i, A_j] = iε_ijk L_k

    Casimir不变量：C_L = L·L, C_A = A·A, C_total = L·L + A·A

    在氢原子中，L² + A² = n² - 1（原子单位）
    """

    def __init__(self):
        self.n = 4
        super().__init__(6)

    def _matrix_shape(self) -> tuple[int, int]:
        return (4, 4)

    def _dtype(self) -> np.dtype:
        return np.complex128

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for i in range(4):
            for j in range(i + 1, 4):
                e = np.zeros((4, 4), dtype=np.complex128)
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def _levi_civita(self, i: int, j: int, k: int) -> float:
        if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            return 1.0
        if (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            return -1.0
        return 0.0

    def check_bracket_relations(self, verbose: bool = False) -> bool:
        gens = self.basis()
        gens_dict = {
            f"T{i}{j}": gens[k]
            for k, (i, j) in enumerate(
                [(i, j) for i in range(4) for j in range(i + 1, 4)]
            )
        }

        indices = [(i, j) for i in range(4) for j in range(i + 1, 4)]

        for a, (ia, ja) in enumerate(indices):
            for b, (ib, jb) in enumerate(indices):
                if a >= b:
                    continue

                expected = np.zeros((4, 4), dtype=np.complex128)
                for c, (ic, jc) in enumerate(indices):
                    f_abc = self._levi_civita(ia, ja, ic) * (
                        1 if ja == ib else -1 if ja == jb else 0
                    )
                    f_abc += self._levi_civita(ia, ja, jc) * (
                        -1 if ja == ib else 1 if ja == jb else 0
                    )
                    f_abc += self._levi_civita(ib, jb, ic) * (
                        1 if jb == ia else -1 if jb == ja else 0
                    )
                    f_abc += self._levi_civita(ib, jb, jc) * (
                        -1 if jb == ia else 1 if jb == ja else 0
                    )
                    if abs(f_abc) > 0.1:
                        expected += f_abc * gens[c].matrix

                computed = (
                    gens[a].matrix @ gens[b].matrix - gens[b].matrix @ gens[a].matrix
                )

                if not np.allclose(computed, expected, atol=1e-10):
                    if verbose:
                        print(f"Bracket [{a},{b}] mismatch")
                    return False
        return True

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name="so(4)",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=False,
            is_abelian=False,
            root_system_type="D2",
            rank=2,
        )

    def __str__(self) -> str:
        return "so(4)"


class SO31LieAlgebra(MatrixLieAlgebraBase):
    """SO(3) ⊂ SO(4) 李代数（第一分量）

    只包含L_i生成元的so(3)子代数。
    """

    def __init__(self):
        super().__init__(3)

    def _matrix_shape(self) -> tuple[int, int]:
        return (3, 3)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for i in range(3):
            for j in range(i + 1, 3):
                e = np.zeros((3, 3))
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name="so(3)",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=True,
            is_abelian=False,
            root_system_type="B1",
            rank=1,
        )

    def __str__(self) -> str:
        return "so(3)"


class SO32LieAlgebra(MatrixLieAlgebraBase):
    """SO(3) ⊂ SO(4) 李代数（第二分量）

    只包含A_i生成元的so(3)子代数。
    """

    def __init__(self):
        super().__init__(3)

    def _matrix_shape(self) -> tuple[int, int]:
        return (3, 3)

    def basis(self) -> list[MatrixLieAlgebraElement]:
        mats: list[MatrixLieAlgebraElement] = []
        for i in range(3):
            for j in range(i + 1, 3):
                e = np.zeros((3, 3))
                e[i, j] = 1.0
                e[j, i] = -1.0
                mats.append(MatrixLieAlgebraElement(e, self))
        return mats

    def properties(self) -> LieAlgebraProperties:
        return LieAlgebraProperties(
            name="so'(3)",
            dimension=self.dimension,
            is_semisimple=True,
            is_simple=True,
            is_abelian=False,
            root_system_type="B1'",
            rank=1,
        )

    def __str__(self) -> str:
        return "so'(3)"
