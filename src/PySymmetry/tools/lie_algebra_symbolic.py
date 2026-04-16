"""Lie代数符号计算模块

该模块提供基于SymPy的Lie代数符号计算功能：
- SymbolicLieAlgebra: 符号Lie代数
- SymbolicLieAlgebraElement: 符号Lie代数元素
- SymbolicBracket: 符号李括号
- SymbolicKillingForm: 符号基灵型
- SymbolicWeylGroup: 符号外尔群
- 结构常数计算与验证
"""

from typing import Any

from sympy import Matrix, Rational, expand, eye, simplify, symbols, zeros
from sympy.matrices import Matrix as SymMatrix


class SymbolicLieAlgebraElement:
    """符号Lie代数元素

    用符号系数表示的Lie代数元素。
    """

    def __init__(self, coefficients: list[Any], basis_labels: list[str] | None = None):
        """初始化符号Lie代数元素

        Args:
            coefficients: 基的符号系数列表
            basis_labels: 基的标签列表
        """
        self.coefficients = coefficients
        n = len(coefficients)
        if basis_labels is None:
            basis_labels = [f"e{i}" for i in range(n)]
        self.basis_labels = basis_labels
        self.dimension = n

    @classmethod
    def basis_element(
        cls, index: int, dimension: int, basis_labels: list[str] | None = None
    ) -> "SymbolicLieAlgebraElement":
        """创建基元"""
        coeffs = [0] * dimension
        coeffs[index] = 1
        return cls(coeffs, basis_labels)

    def __add__(
        self, other: "SymbolicLieAlgebraElement"
    ) -> "SymbolicLieAlgebraElement":
        """加法"""
        new_coeffs = [a + b for a, b in zip(self.coefficients, other.coefficients)]
        return SymbolicLieAlgebraElement(new_coeffs, self.basis_labels)

    def __sub__(
        self, other: "SymbolicLieAlgebraElement"
    ) -> "SymbolicLieAlgebraElement":
        """减法"""
        new_coeffs = [a - b for a, b in zip(self.coefficients, other.coefficients)]
        return SymbolicLieAlgebraElement(new_coeffs, self.basis_labels)

    def __mul__(self, scalar: Any) -> "SymbolicLieAlgebraElement":
        """数乘"""
        new_coeffs = [c * scalar for c in self.coefficients]
        return SymbolicLieAlgebraElement(new_coeffs, self.basis_labels)

    def __rmul__(self, scalar: Any) -> "SymbolicLieAlgebraElement":
        """右数乘"""
        return self.__mul__(scalar)

    def __neg__(self) -> "SymbolicLieAlgebraElement":
        """负元"""
        return self * (-1)

    def to_symbolic_vector(self) -> Matrix:
        """转换为符号向量"""
        return Matrix(self.coefficients)

    def to_expression(self) -> Any:
        """转换为符号表达式"""
        expr = 0
        for c, label in zip(self.coefficients, self.basis_labels):
            expr += c * symbols(label)
        return expr

    def simplify(self) -> "SymbolicLieAlgebraElement":
        """简化系数"""
        new_coeffs = [simplify(c) for c in self.coefficients]
        return SymbolicLieAlgebraElement(new_coeffs, self.basis_labels)

    def expand(self) -> "SymbolicLieAlgebraElement":
        """展开表达式"""
        new_coeffs = [expand(c) for c in self.coefficients]
        return SymbolicLieAlgebraElement(new_coeffs, self.basis_labels)

    def __str__(self) -> str:
        terms = []
        for c, label in zip(self.coefficients, self.basis_labels):
            if c != 0:
                terms.append(f"{c}*{label}")
        return " + ".join(terms) if terms else "0"

    def __repr__(self) -> str:
        return f"SymbolicLieAlgebraElement({self.coefficients}, {self.basis_labels})"


class SymbolicLieBracket:
    """符号李括号

    基于结构常数的符号李括号计算。
    """

    def __init__(self, structure_constants: dict[tuple[int, int], list[int]]):
        """初始化符号李括号

        Args:
            structure_constants: 结构常数 c_{ij}^k，满足 [e_i, e_j] = sum_k c_{ij}^k e_k
        """
        self.structure_constants = structure_constants

    def __call__(
        self, x: SymbolicLieAlgebraElement, y: SymbolicLieAlgebraElement
    ) -> SymbolicLieAlgebraElement:
        """计算李括号 [x, y]"""
        dim = x.dimension
        result_coeffs = [0] * dim

        for i in range(dim):
            for j in range(dim):
                key = (i, j)
                if key in self.structure_constants:
                    coeffs = self.structure_constants[key]
                    for k, c_ij_k in enumerate(coeffs):
                        if c_ij_k != 0:
                            result_coeffs[k] += (
                                c_ij_k * x.coefficients[i] * y.coefficients[j]
                            )

        result = SymbolicLieAlgebraElement(result_coeffs, x.basis_labels)
        return result.simplify()

    def verify_anticommutative(self, basis: list[SymbolicLieAlgebraElement]) -> bool:
        """验证反对称性 [x, y] = -[y, x]"""
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                bracket_ij = self(basis[i], basis[j])
                bracket_ji = self(basis[j], basis[i])
                combined = bracket_ij + bracket_ji
                if any(expand(c) != 0 for c in combined.coefficients):
                    return False
        return True

    def verify_jacobi_identity(self, basis: list[SymbolicLieAlgebraElement]) -> bool:
        """验证雅可比恒等式 [x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0"""
        for i in range(len(basis)):
            for j in range(len(basis)):
                for k in range(len(basis)):
                    x, y, z = basis[i], basis[j], basis[k]
                    term1 = self(x, self(y, z))
                    term2 = self(y, self(z, x))
                    term3 = self(z, self(x, y))
                    result = term1 + term2 + term3
                    if any(expand(c) != 0 for c in result.coefficients):
                        return False
        return True

    def to_matrix(self) -> SymMatrix:
        """将李括号表示为矩阵形式"""
        dim = (
            max(max(k[1] for k in self.structure_constants.keys())) + 1
            if self.structure_constants
            else 0
        )
        matrices = []

        for i in range(dim):
            ad_matrix = zeros(dim, dim)
            for j in range(dim):
                key = (i, j)
                if key in self.structure_constants:
                    coeffs = self.structure_constants[key]
                    for k, c in enumerate(coeffs):
                        ad_matrix[k, j] = c
            matrices.append(ad_matrix)

        return matrices


class SymbolicLieAlgebra:
    """符号Lie代数

    使用SymPy符号表示的Lie代数，支持结构常数计算、根系统、Weyl群等。
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        structure_constants: dict[tuple[int, int], list[int]] | None = None,
        basis_labels: list[str] | None = None,
        cartan_matrix: Matrix | None = None,
    ):
        """初始化符号Lie代数

        Args:
            name: Lie代数名称
            dimension: 维数
            structure_constants: 结构常数
            basis_labels: 基标签
            cartan_matrix: Cartan矩阵
        """
        self.name = name
        self.dimension = dimension
        self.structure_constants = structure_constants or {}

        if basis_labels is None:
            basis_labels = [f"H_{i}" for i in range(dimension)]
        self.basis_labels = basis_labels

        self.cartan_matrix = cartan_matrix
        self._bracket = SymbolicLieBracket(self.structure_constants)

    @classmethod
    def from_cartan_matrix(
        cls, name: str, cartan_matrix: Matrix
    ) -> "SymbolicLieAlgebra":
        """从Cartan矩阵创建符号Lie代数

        Args:
            name: Lie代数名称
            cartan_matrix: Cartan矩阵
        """
        rank = cartan_matrix.rows
        dim = cls._dimension_from_cartan(name, rank)

        basis_labels = [f"H_{i}" for i in range(rank)]
        basis_labels += [f"E_{i}" for i in range(dim - rank)]

        structure_constants = cls._compute_structure_constants(cartan_matrix, dim)

        return cls(name, dim, structure_constants, basis_labels, cartan_matrix)

    @staticmethod
    def _dimension_from_cartan(name: str, rank: int) -> int:
        """根据Cartan矩阵计算Lie代数维数"""
        name = name.upper()
        if name == "A":
            return rank + 1
        elif name in ("B", "C"):
            return rank * 2 + 1
        elif name == "D":
            return rank * 2
        elif name == "G2":
            return 14
        elif name == "F4":
            return 52
        elif name == "E6":
            return 78
        elif name == "E7":
            return 133
        elif name == "E8":
            return 248
        return rank * rank

    @staticmethod
    def _compute_structure_constants(
        cartan_matrix: Matrix, dim: int
    ) -> dict[tuple[int, int], list[int]]:
        """从Cartan矩阵计算结构常数"""
        rank = cartan_matrix.rows
        n = dim
        structure_constants = {}

        simple_roots = SymbolicLieAlgebra._generate_simple_roots(cartan_matrix)

        positive_roots = SymbolicLieAlgebra._generate_positive_roots(simple_roots, rank)

        root_indices = list(range(rank, n))

        for i in range(rank):
            for j in range(rank, n):
                key = (i, j)
                coeffs = [0] * n

                alpha_i = simple_roots[i]
                alpha_j = positive_roots[j - rank]

                dot_ij = float(alpha_i.dot(alpha_j))
                dot_ii = float(alpha_i.dot(alpha_i))

                if dot_ii != 0:
                    coeff = int(2 * dot_ij / dot_ii)
                    if coeff != 0:
                        coeffs[j] = coeff

                structure_constants[key] = coeffs

        for i in range(rank, n):
            for j in range(rank, n):
                key = (i, j)
                coeffs = [0] * n

                structure_constants[key] = coeffs

        for i in range(rank):
            for j in range(rank):
                key = (i, j)
                coeffs = [0] * n
                structure_constants[key] = coeffs

        for i in range(rank, n):
            for j in range(rank):
                key = (i, j)
                coeffs = [0] * n

                alpha_j = simple_roots[j]
                alpha_i = positive_roots[i - rank]

                dot_ji = float(alpha_j.dot(alpha_i))
                dot_jj = float(alpha_j.dot(alpha_j))

                if dot_jj != 0:
                    coeff = int(2 * dot_ji / dot_jj)
                    if coeff != 0:
                        coeffs[j] = -coeff

                structure_constants[key] = coeffs

        return structure_constants

    @staticmethod
    def _generate_simple_roots(cartan_matrix: Matrix) -> list[Matrix]:
        """生成单根"""
        rank = cartan_matrix.rows
        simple_roots = []

        if rank == 1:
            simple_roots = [Matrix([1])]
        elif rank == 2:
            c = cartan_matrix
            a1 = Matrix([1, 0])
            a2 = (
                Matrix([-c[0, 1], c[0, 0]]) * Rational(1, c[0, 0])
                if c[0, 0] != 0
                else Matrix([0, 1])
            )
            simple_roots = [a1, a2]
        else:
            for i in range(rank):
                root = zeros(rank, 1)
                root[i] = 1
                simple_roots.append(root)

        return simple_roots

    @staticmethod
    def _generate_positive_roots(simple_roots: list[Matrix], rank: int) -> list[Matrix]:
        """生成正根"""
        positive_roots = []

        for i in range(len(simple_roots)):
            positive_roots.append(simple_roots[i])

        max_depth = 10
        for _ in range(max_depth):
            new_roots = []
            for root in positive_roots:
                for simple in simple_roots:
                    new_root = root + simple
                    if all(abs(new_root[i]) <= 10 for i in range(rank)):
                        if new_root not in positive_roots and new_root not in new_roots:
                            new_roots.append(new_root)
            positive_roots.extend(new_roots)
            if not new_roots:
                break

        return positive_roots

    def basis(self) -> list[SymbolicLieAlgebraElement]:
        """返回基元列表"""
        return [
            SymbolicLieAlgebraElement.basis_element(
                i, self.dimension, self.basis_labels
            )
            for i in range(self.dimension)
        ]

    def bracket(
        self, x: SymbolicLieAlgebraElement, y: SymbolicLieAlgebraElement
    ) -> SymbolicLieAlgebraElement:
        """计算李括号 [x, y]"""
        return self._bracket(x, y)

    def killing_form_matrix(self) -> SymMatrix:
        """计算基灵型矩阵"""
        n = self.dimension
        K = zeros(n, n)

        basis = self.basis()

        for i in range(n):
            for j in range(n):
                x = basis[i]
                y = basis[j]

                ad_x = self._adjoint_matrix(x)
                ad_y = self._adjoint_matrix(y)

                K[i, j] = simplify((ad_x * ad_y).trace())

        return K

    def _adjoint_matrix(self, x: SymbolicLieAlgebraElement) -> SymMatrix:
        """计算伴随表示矩阵"""
        n = self.dimension
        ad_x = zeros(n, n)

        basis = self.basis()

        for j in range(n):
            bracket_x_ej = self.bracket(x, basis[j])
            for k in range(n):
                ad_x[k, j] = bracket_x_ej.coefficients[k]

        return ad_x

    def is_semisimple(self) -> bool:
        """判断是否为半单"""
        K = self.killing_form_matrix()
        return K.det() != 0

    def root_system(self) -> dict[str, Any]:
        """获取根系信息"""
        if self.cartan_matrix is None:
            return {"type": self.name, "rank": None, "roots": []}

        rank = self.cartan_matrix.rows
        simple_roots = self._generate_simple_roots(self.cartan_matrix)
        positive_roots = self._generate_positive_roots(simple_roots, rank)

        return {
            "type": self.name,
            "rank": rank,
            "simple_roots": simple_roots,
            "positive_roots": positive_roots,
            "root_number": len(positive_roots) * 2 + rank,
            "cartan_matrix": self.cartan_matrix,
        }

    def __str__(self) -> str:
        return f"SymbolicLieAlgebra({self.name}, dim={self.dimension})"

    def __repr__(self) -> str:
        return f"SymbolicLieAlgebra(name='{self.name}', dimension={self.dimension})"


class SymbolicKillingForm:
    """符号基灵型

    使用SymPy符号计算的基灵型。
    """

    def __init__(self, lie_algebra: SymbolicLieAlgebra):
        """初始化符号基灵型"""
        self.lie_algebra = lie_algebra
        self.matrix = lie_algebra.killing_form_matrix()

    def __call__(
        self, x: SymbolicLieAlgebraElement, y: SymbolicLieAlgebraElement
    ) -> Any:
        """计算K(x, y)"""
        vec_x = Matrix(x.coefficients)
        vec_y = Matrix(y.coefficients)
        return simplify(vec_x.T * self.matrix * vec_y)[0]

    def is_non_degenerate(self) -> bool:
        """判断是否非退化"""
        return simplify(self.matrix.det()) != 0

    def __str__(self) -> str:
        return f"SymbolicKillingForm({self.lie_algebra.name})"


class SymbolicWeylGroup:
    """符号外尔群

    使用SymPy符号计算的外尔群。
    """

    def __init__(self, lie_algebra: SymbolicLieAlgebra):
        """初始化符号外尔群"""
        self.lie_algebra = lie_algebra
        self.root_system_info = lie_algebra.root_system()
        self._generate_reflections()

    def _generate_reflections(self) -> list[Matrix]:
        """生成简单反射"""
        rank = self.root_system_info.get("rank", 0)
        if rank is None or rank == 0:
            self.reflections = []
            return []

        cartan = self.lie_algebra.cartan_matrix
        if cartan is None:
            self.reflections = []
            return []

        self.reflections = []

        for i in range(rank):
            reflection = eye(rank)

            for j in range(rank):
                delta_ij = 1 if i == j else 0
                reflection[i, j] = delta_ij - cartan[i, j]

            self.reflections.append(reflection)

        return self.reflections

    def reflect(self, i: int, vector: Matrix) -> Matrix:
        """对向量应用第i个反射"""
        if i >= len(self.reflections):
            return vector
        return self.reflections[i] * vector

    def order(self) -> int:
        """计算外尔群阶数"""
        root_type = self.root_system_info.get("type", "Unknown")

        order_map = {
            "A1": 2,
            "A2": 6,
            "A3": 24,
            "A4": 120,
            "B2": 8,
            "B3": 48,
            "B4": 384,
            "C2": 8,
            "C3": 48,
            "C4": 384,
            "D2": 4,
            "D3": 24,
            "D4": 192,
            "G2": 12,
            "F4": 1152,
            "E6": 51840,
            "E7": 2903040,
            "E8": 696729600,
        }

        return order_map.get(root_type, 1)

    def __str__(self) -> str:
        return f"SymbolicWeylGroup({self.lie_algebra.name}, order={self.order()})"


def compute_structure_constants(name: str, n: int) -> dict[tuple[int, int], list[int]]:
    """计算Lie代数的结构常数

    Args:
        name: Lie代数类型 ('A', 'B', 'C', 'D', 'G2', 'F4', 'E6', 'E7', 'E8')
        n: rank或维度参数

    Returns:
        结构常数字典
    """
    name = name.upper()

    if name == "A":
        rank = n - 1
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if abs(i - j) == 1 else 0)
        )
    elif name == "B":
        rank = n
        cartan = Matrix(
            rank,
            rank,
            lambda i, j: -2 if i == j else (1 if j == i + 1 or i == j + 1 else 0),
        )
    elif name == "C":
        rank = n
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if j == i + 1 else 0)
        )
    elif name == "D":
        rank = n
        cartan = Matrix(
            rank,
            rank,
            lambda i, j: (
                -2
                if i == j
                else (
                    1
                    if (
                        j == i + 1
                        or (i == rank - 1 and j == rank - 2)
                        or (i == rank - 2 and j == rank - 1)
                    )
                    else 0
                )
            ),
        )
    elif name == "G2":
        rank = 2
        cartan = Matrix([[2, -1], [-3, 2]])
    elif name == "F4":
        rank = 4
        cartan = Matrix([[2, -1, 0, 0], [-1, 2, -2, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
    elif name == "E6":
        rank = 6
        cartan = Matrix(
            [
                [2, -1, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0],
                [0, -1, 2, -1, 0, 0],
                [0, 0, -1, 2, -1, 0],
                [0, 0, 0, -1, 2, -1],
                [0, 0, 0, 0, -1, 2],
            ]
        )
    elif name == "E7":
        rank = 7
        cartan = Matrix(
            [
                [2, -1, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, 0],
                [0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, -1, 2, -1],
                [0, 0, 0, 0, 0, -1, 2],
            ]
        )
    elif name == "E8":
        rank = 8
        cartan = Matrix(
            [
                [2, -1, 0, 0, 0, 0, 0, 0],
                [-1, 2, -1, 0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0, 0, 0, 0],
                [0, 0, -1, 2, -1, 0, 0, 0],
                [0, 0, 0, -1, 2, -1, 0, 0],
                [0, 0, 0, 0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0, -1, 2, -1],
                [0, 0, 0, 0, 0, 0, -1, 2],
            ]
        )
    else:
        raise ValueError(f"未知的Lie代数类型: {name}")

    dim = SymbolicLieAlgebra._dimension_from_cartan(name, rank)
    return SymbolicLieAlgebra._compute_structure_constants(cartan, dim)


def verify_jacobi_identity(name: str, n: int) -> bool:
    """验证Lie代数的雅可比恒等式

    Args:
        name: Lie代数类型
        n: rank或维度参数

    Returns:
        是否满足雅可比恒等式
    """
    structure_constants = compute_structure_constants(name, n)
    bracket = SymbolicLieBracket(structure_constants)

    dim = max(max(k[0], k[1]) for k in structure_constants.keys()) + 1

    basis = [SymbolicLieAlgebraElement.basis_element(i, dim) for i in range(dim)]

    return bracket.verify_jacobi_identity(basis)


def generate_weyl_coordinates(name: str, rank: int) -> list[Matrix]:
    """生成Weyl群坐标

    Args:
        name: Lie代数类型
        rank: rank

    Returns:
        Weyl群元素对应的坐标变换矩阵
    """
    if name.upper() == "A":
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if j == i + 1 else 0)
        )
    elif name.upper() == "B":
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if j == i + 1 else 0)
        )
    elif name.upper() == "C":
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if j == i + 1 else 0)
        )
    elif name.upper() == "D":
        cartan = Matrix(
            rank, rank, lambda i, j: -2 if i == j else (1 if j == i + 1 else 0)
        )
    elif name.upper() == "G2":
        cartan = Matrix([[2, -1], [-3, 2]])
    elif name.upper() == "F4":
        cartan = Matrix([[2, -1, 0, 0], [-1, 2, -2, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
    else:
        return []

    simple_roots = []
    for i in range(rank):
        root = zeros(rank, 1)
        root[i] = 1
        simple_roots.append(root)

    reflections = []
    for i in range(rank):
        reflection = eye(rank)

        for j in range(rank):
            delta_ij = 1 if i == j else 0
            reflection[i, j] = delta_ij - cartan[i, j]

        reflections.append(reflection)

    return reflections
