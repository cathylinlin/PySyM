"""李代数结构

该模块实现了李代数的结构相关功能，包括：
- CartanSubalgebra: 嘉当子代数
- RootSystem: 根系
- WeylGroup: 外尔群
- KillingForm: 基灵型
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from .abstract_lie_algebra import LieAlgebra, LieAlgebraElement
from .lie_algebra_representation import AdjointRepresentation

T = TypeVar("T", bound="LieAlgebraElement")


class CartanSubalgebra(ABC, Generic[T]):
    """嘉当子代数"""

    def __init__(self, lie_algebra: LieAlgebra[T], basis: list[T]):
        """初始化嘉当子代数"""
        self.lie_algebra = lie_algebra
        self.basis = basis
        self.dimension = len(basis)

    @abstractmethod
    def roots(self) -> "RootSystem":
        """计算根系"""
        pass

    @abstractmethod
    def weyl_group(self) -> "WeylGroup":
        """计算外尔群"""
        pass

    @abstractmethod
    def killing_form(self) -> "KillingForm":
        """计算基灵型"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """字符串表示"""
        pass


class RootSystem:
    """根系"""

    def __init__(
        self, roots: list[np.ndarray], coroots: list[np.ndarray] | None = None
    ):
        """初始化根系"""
        self.roots = roots
        self.coroots = coroots or []
        self.rank = len(roots[0]) if roots else 0

    def positive_roots(self) -> list[np.ndarray]:
        """计算正根"""
        # 简单实现：选择第一个非零分量为正的根
        if not self.roots:
            return []
        positive = []
        for root in self.roots:
            for component in root:
                if component != 0:
                    if component > 0:
                        positive.append(root)
                    break
        return positive

    def simple_roots(self) -> list[np.ndarray]:
        """计算单根"""
        # 简单实现：返回正根中不能表示为其他正根和的根
        positive = self.positive_roots()
        simple = []
        for root in positive:
            is_simple = True
            for other_root in positive:
                if not np.array_equal(root, other_root):
                    # 检查root是否可以表示为其他正根的和
                    if np.all(root >= other_root) and np.any(root > other_root):
                        is_simple = False
                        break
            if is_simple:
                simple.append(root)
        return simple

    def root_system_type(self) -> str:
        """确定根系类型"""
        # 简单实现：根据秩和根的数量判断
        rank = self.rank
        root_count = len(self.roots)

        if rank == 1:
            return "A1"
        elif rank == 2:
            if root_count == 6:
                return "A2"
            elif root_count == 8:
                return "B2"
            elif root_count == 10:
                return "G2"
        elif rank == 3:
            if root_count == 12:
                return "A3"
            elif root_count == 18:
                return "B3"
            elif root_count == 24:
                return "C3"
        elif rank == 4:
            if root_count == 24:
                return "A4"
            elif root_count == 36:
                return "B4"
            elif root_count == 48:
                return "C4"
            elif root_count == 60:
                return "D4"

        return f"Unknown type (rank={rank}, roots={root_count})"

    def __str__(self) -> str:
        return f"RootSystem(type={self.root_system_type()}, rank={self.rank}, roots={len(self.roots)})"


class WeylGroup:
    """外尔群"""

    def __init__(self, root_system: RootSystem):
        """初始化外尔群"""
        self.root_system = root_system
        self.generators = self._generate_reflections()

    def _generate_reflections(self) -> list[np.ndarray]:
        """生成外尔群的生成元（反射）"""
        simple_roots = self.root_system.simple_roots()
        generators = []

        for root in simple_roots:
            # 计算反射矩阵
            rank = self.root_system.rank
            reflection = np.eye(rank)
            root_norm = np.dot(root, root)
            if root_norm > 0:
                reflection -= 2 * np.outer(root, root) / root_norm
            generators.append(reflection)

        return generators

    def order(self) -> int:
        """计算外尔群的阶"""
        # 简单实现：根据根系类型返回阶
        root_type = self.root_system.root_system_type()
        order_map = {
            "A1": 2,
            "A2": 6,
            "A3": 24,
            "A4": 120,
            "B2": 8,
            "B3": 48,
            "B4": 384,
            "C3": 48,
            "C4": 384,
            "D4": 192,
            "G2": 12,
        }
        return order_map.get(root_type, 1)

    def act(self, element: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """外尔群元素作用于向量"""
        # 简单实现：矩阵乘法
        return np.dot(element, vector)

    def __str__(self) -> str:
        return f"WeylGroup(type={self.root_system.root_system_type()}, order={self.order()})"


class KillingForm:
    """基灵型"""

    def __init__(self, lie_algebra: LieAlgebra[T]):
        """初始化基灵型"""
        self.lie_algebra = lie_algebra
        self.matrix = self._compute_matrix()

    def _compute_matrix(self) -> np.ndarray:
        """计算基灵型矩阵"""
        basis = self.lie_algebra.basis()
        dim = self.lie_algebra.dimension
        matrix = np.zeros((dim, dim))

        # 计算伴随表示
        adjoint_rep = AdjointRepresentation(self.lie_algebra)

        # 计算基灵型：K(X, Y) = tr(ad_X ad_Y)
        for i, x in enumerate(basis):
            ad_x = np.asarray(adjoint_rep(x))
            for j, y in enumerate(basis):
                ad_y = np.asarray(adjoint_rep(y))
                ad_x_ad_y = ad_x @ ad_y
                matrix[i, j] = float(np.trace(ad_x_ad_y).real)

        return matrix

    def __call__(self, x: T, y: T) -> float:
        """计算两个李代数元素的基灵型"""
        # 将元素转换为基的坐标
        x_coords = self.lie_algebra.to_vector(x)
        y_coords = self.lie_algebra.to_vector(y)

        # 计算基灵型
        return float(np.dot(x_coords, np.dot(self.matrix, y_coords)))

    def is_non_degenerate(self) -> bool:
        """判断基灵型是否非退化（对称双线性型用秩判定，避免浮点行列式误判）。"""
        m = self.matrix
        if m.size == 0:
            return True
        return np.linalg.matrix_rank(m) == m.shape[0]

    def __str__(self) -> str:
        return f"KillingForm(lie_algebra={self.lie_algebra}, non_degenerate={self.is_non_degenerate()})"
