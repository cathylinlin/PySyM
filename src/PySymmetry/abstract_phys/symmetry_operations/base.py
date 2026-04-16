from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    from PySymmetry.abstract_phys.physical_objects.abstract_physical_objects import (
        PhysicalObject,
    )
    from PySymmetry.core.group_theory.abstract_group import Group

T = TypeVar("T")


class Transformation(Protocol[T]):
    """变换协议"""

    def apply(self, obj: T) -> T:
        """应用变换"""
        ...


class State(Protocol):
    """状态协议"""

    pass


class PhysicalQuantity(Protocol):
    """物理量协议"""

    pass


class SymmetryOperation(ABC):
    """
    对称操作基类

    表示物理系统上的对称变换操作。
    与 core 模块的 Group 和 Matrix 有良好的集成。
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        self._cached_matrix: np.ndarray | None = None
        self._representation_dim: int | None = None

    @property
    @abstractmethod
    def group(self) -> "Group":
        """所属对称群"""
        pass

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """是否连续对称性"""
        pass

    @abstractmethod
    def compose(self, other: "SymmetryOperation") -> "SymmetryOperation":
        """组合两个对称操作"""
        pass

    @abstractmethod
    def inverse(self) -> "SymmetryOperation":
        """逆操作"""
        pass

    @abstractmethod
    def act_on(self, obj: "PhysicalObject") -> "PhysicalObject":
        """作用于物理对象

        Args:
            obj: 物理对象

        Returns:
            变换后的物理对象
        """
        pass

    def act_on_state(self, q: Any, p: Any) -> tuple:
        """作用于相空间状态

        Args:
            q: 广义坐标
            p: 广义动量

        Returns:
            变换后的 (q, p)
        """
        return q, p

    def representation_matrix(self, dim: int | None = None) -> np.ndarray:
        """获取该对称操作的表示矩阵

        Args:
            dim: 表示维度，如果不提供则尝试使用缓存的维度

        Returns:
            表示矩阵
        """
        if dim is None:
            dim = self._representation_dim if self._representation_dim else 2

        if self._cached_matrix is not None and self._representation_dim == dim:
            return self._cached_matrix

        matrix = self._compute_representation_matrix(dim)
        self._cached_matrix = matrix
        self._representation_dim = dim
        return matrix

    def _compute_representation_matrix(self, dim: int) -> np.ndarray:
        """计算表示矩阵

        子类应重写此方法以提供具体的表示矩阵

        Args:
            dim: 表示维度

        Returns:
            表示矩阵
        """
        return np.eye(dim)

    def apply_to_vector(self, vec: np.ndarray) -> np.ndarray:
        """将对称操作应用到向量

        Args:
            vec: 输入向量

        Returns:
            变换后的向量
        """
        matrix = self.representation_matrix(len(vec))
        return matrix @ vec

    def apply_to_matrix(self, mat: np.ndarray) -> np.ndarray:
        """将对称操作应用到矩阵

        使用相似变换：M' = R @ M @ R^(-1) 或 M' = R @ M @ R^dagger（酉情况）

        Args:
            mat: 输入矩阵

        Returns:
            变换后的矩阵
        """
        matrix = self.representation_matrix(mat.shape[0])
        return matrix @ mat @ matrix.conj().T

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """检查表示是否为幺正的

        对于幺正表示：U^dagger U = I

        Args:
            tol: 容差

        Returns:
            是否幺正
        """
        if self._cached_matrix is None:
            return True

        matrix = self._cached_matrix
        return bool(
            np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=tol)
        )

    def is_orthogonal(self, tol: float = 1e-10) -> bool:
        """检查表示是否为正交的

        对于正交表示：R^T R = I

        Args:
            tol: 容差

        Returns:
            是否正交
        """
        if self._cached_matrix is None:
            return True

        matrix = self._cached_matrix
        if not np.isrealobj(matrix):
            return False
        return bool(np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0]), atol=tol))


class Symmetric(Protocol):
    """具有对称性的对象协议"""

    def get_symmetry_group(self) -> "Group":
        """获取对称群"""
        ...

    def check_symmetry(self, operation: "SymmetryOperation") -> bool:
        """检查对称性"""
        ...


class Transformable(Protocol):
    """可变换对象协议"""

    def apply_transformation(self, transformation: "Transformation") -> "Transformable":
        """应用变换"""
        ...


class Observable(Protocol):
    """可观测量协议"""

    def expectation_value(self, state: "State") -> "PhysicalQuantity":
        """期望值"""
        ...

    def variance(self, state: "State") -> "PhysicalQuantity":
        """方差"""
        ...
