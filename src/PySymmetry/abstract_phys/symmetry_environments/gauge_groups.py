"""规范群实现

该模块提供规范群的实现，与 core 模块的矩阵群集成：
- GaugeGroupFactory: 规范群工厂类
- U1GaugeGroup: U(1) 规范群
- SU2GaugeGroup: SU(2) 规范群（使用泡利矩阵）
- SU3GaugeGroup: SU(3) 规范群（使用盖尔曼矩阵）

约定说明：
- SU(n) 生成元采用 core 模块的 SpecialUnitaryLieAlgebra 约定
- 生成元为反厄米矩阵（与 core 的 SU(n) 李代数一致）
- 指数映射 exp(θ^a T_a) 生成群元素
"""

from typing import Any, TypeVar

import numpy as np

from ...core.group_theory.abstract_group import Group
from ...core.matrix_groups.special_linear import SLnElement

T = TypeVar("T")


class GaugeGroup(Group[np.ndarray]):
    """规范群基类

    无限李群，不继承 MatrixGroup 以避免方法签名冲突
    """

    def __init__(self, name: str):
        super().__init__(name)

    def generators(self) -> list[np.ndarray]:
        """获取群生成元"""
        raise NotImplementedError

    def exp(self, *args, **kwargs) -> np.ndarray:
        """指数映射"""
        raise NotImplementedError

    def identity(self) -> np.ndarray:
        """单位元"""
        raise NotImplementedError

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """群乘法"""
        raise NotImplementedError

    def inverse(self, a: np.ndarray) -> np.ndarray:
        """逆元"""
        raise NotImplementedError

    def __contains__(self, element: Any) -> bool:
        """检查元素是否在群中"""
        raise NotImplementedError

    def order(self) -> int:
        """群的阶（无限群返回 -1）"""
        return -1

    def elements(self) -> list[np.ndarray]:
        """群元素列表"""
        raise ValueError("无限群没有有限元素列表")


class U1GaugeGroup(GaugeGroup):
    """U(1)规范群

    U(1) 群是阿贝尔群，用于描述电磁相互作用。
    生成元为 i（纯相位）。
    """

    def __init__(self):
        super().__init__("U(1)")

    def identity(self) -> np.ndarray:
        return np.array([[1.0 + 0j]])

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return np.conj(a)

    def __contains__(self, element: Any) -> bool:
        if isinstance(element, (int, float)):
            return bool(abs(element) == 1.0)
        if isinstance(element, complex):
            return bool(np.isclose(abs(element), 1.0))
        if isinstance(element, np.ndarray):
            return bool(np.isclose(np.abs(element), 1.0))
        return False

    def order(self) -> int:
        return -1

    def elements(self) -> list[np.ndarray]:
        raise ValueError("无限群没有有限元素列表")

    def generators(self) -> list[np.ndarray]:
        """U(1)生成元：i"""
        return [np.array(1j)]

    def exp(self, theta: float) -> np.ndarray:
        """指数映射 exp(iθ)"""
        return np.exp(1j * theta)


class SU2GaugeGroup(GaugeGroup):
    """SU(2)规范群

    SU(2) 群是非阿贝尔群，用于描述弱相互作用。
    生成元为反厄米矩阵 τ_i = i σ_i/2，与 core 模块的 SpecialUnitaryLieAlgebra 一致。
    """

    PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    PAULI_MATRICES = [PAULI_X, PAULI_Y, PAULI_Z]

    def __init__(self):
        super().__init__("SU(2)")

    def identity(self) -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return a.conj().T

    def __contains__(self, element: Any) -> bool:
        if isinstance(element, SLnElement):
            element = element.matrix
        if not isinstance(element, np.ndarray):
            return False
        if element.shape != (2, 2):
            return False
        if not np.allclose(element.conj().T @ element, np.eye(2)):
            return False
        if not np.isclose(np.linalg.det(element), 1):
            return False
        return True

    def order(self) -> int:
        return -1

    def elements(self) -> list[np.ndarray]:
        raise ValueError("无限群没有有限元素列表")

    def generators(self) -> list[np.ndarray]:
        """SU(2)生成元：τ_i = i σ_i/2（反厄米，符合 core 约定）"""
        # 使用 core 模块的 SpecialUnitaryLieAlgebra 获取生成元
        from ...core.lie_theory.specific_lie_algebra import SpecialUnitaryLieAlgebra

        su2 = SpecialUnitaryLieAlgebra(2)
        return [elem.matrix for elem in su2.basis()]

    def exp(self, theta: np.ndarray) -> np.ndarray:
        """指数映射 exp(θ·τ)，符合 core 中的反厄米约定"""
        from scipy.linalg import expm

        theta = np.asarray(theta)
        generator = sum(theta[i] * self.generators()[i] for i in range(3))
        return expm(generator)

    def commutation_relations(self) -> list[tuple]:
        """返回对易关系 [τ_i, τ_j] = i ε_ijk τ_k"""
        return [
            (0, 1, 2, 1j),
            (0, 2, 1, -1j),
            (1, 2, 0, 1j),
        ]


class SU3GaugeGroup(GaugeGroup):
    """SU(3)规范群

    SU(3) 群是非阿贝尔群，用于描述强相互作用。
    生成元为反厄米矩阵 λ_i/2，与 core 模块的 SpecialUnitaryLieAlgebra 一致。
    """

    LAMBDA_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128)
    LAMBDA_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128)
    LAMBDA_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    LAMBDA_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128)
    LAMBDA_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128)
    LAMBDA_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128)
    LAMBDA_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128)
    LAMBDA_8 = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=np.complex128
    ) / np.sqrt(3)
    GELL_MANN_MATRICES = [
        LAMBDA_1,
        LAMBDA_2,
        LAMBDA_3,
        LAMBDA_4,
        LAMBDA_5,
        LAMBDA_6,
        LAMBDA_7,
        LAMBDA_8,
    ]

    def __init__(self):
        super().__init__("SU(3)")

    def identity(self) -> np.ndarray:
        return np.eye(3, dtype=np.complex128)

    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return a.conj().T

    def __contains__(self, element: Any) -> bool:
        if isinstance(element, SLnElement):
            element = element.matrix
        if not isinstance(element, np.ndarray):
            return False
        if element.shape != (3, 3):
            return False
        if not np.allclose(element.conj().T @ element, np.eye(3)):
            return False
        if not np.isclose(np.linalg.det(element), 1):
            return False
        return True

    def order(self) -> int:
        return -1

    def elements(self) -> list[np.ndarray]:
        raise ValueError("无限群没有有限元素列表")

    def generators(self) -> list[np.ndarray]:
        """SU(3)生成元：λ_i/2 * i（反厄米，符合 core 约定）"""
        # 使用 core 模块的 SpecialUnitaryLieAlgebra 获取生成元
        from ...core.lie_theory.specific_lie_algebra import SpecialUnitaryLieAlgebra

        su3 = SpecialUnitaryLieAlgebra(3)
        return [elem.matrix for elem in su3.basis()]

    def exp(self, theta: np.ndarray) -> np.ndarray:
        """指数映射 exp(θ^a λ_a/2 * i)，符合 core 中的反厄米约定"""
        from scipy.linalg import expm

        theta = np.asarray(theta)
        generator = sum(theta[i] * self.generators()[i] for i in range(8))
        return expm(generator)

    def structure_constants(self) -> np.ndarray:
        """SU(3) 结构常数 f_abc"""
        f = np.zeros((8, 8, 8))
        for i in range(8):
            for j in range(8):
                commutator = (
                    self.GELL_MANN_MATRICES[i] @ self.GELL_MANN_MATRICES[j]
                    - self.GELL_MANN_MATRICES[j] @ self.GELL_MANN_MATRICES[i]
                )
                for k in range(8):
                    f[i, j, k] = np.imag(
                        np.trace(commutator @ self.GELL_MANN_MATRICES[k])
                    )
        return f


class GaugeGroupFactory:
    """规范群工厂类"""

    _registry = {
        "U(1)": U1GaugeGroup,
        "U1": U1GaugeGroup,
        "SU(2)": SU2GaugeGroup,
        "SU2": SU2GaugeGroup,
        "SU(3)": SU3GaugeGroup,
        "SU3": SU3GaugeGroup,
    }

    @classmethod
    def create(cls, gauge_group: str) -> GaugeGroup:
        """创建规范群

        Args:
            gauge_group: 规范群名称，如 "U(1)"、"SU(2)"、"SU(3)"

        Returns:
            对应的规范群实例

        Raises:
            ValueError: 如果规范群类型不受支持
        """
        if gauge_group not in cls._registry:
            raise ValueError(f"不支持的规范群类型: {gauge_group}")
        return cls._registry[gauge_group]()

    @classmethod
    def _create_su_n_group(cls, n: int) -> GaugeGroup:
        """创建 SU(n) 规范群

        Args:
            n: 群的维度

        Returns:
            SU(n) 规范群实例
        """

        class SUNGaugeGroup(GaugeGroup):
            """SU(n)规范群"""

            def __init__(self):
                super().__init__(f"SU({n})")
                self._dimension = n

            def identity(self) -> np.ndarray:
                return np.eye(n, dtype=np.complex128)

            def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
                return np.dot(a, b)

            def inverse(self, a: np.ndarray) -> np.ndarray:
                return a.conj().T

            def __contains__(self, element: Any) -> bool:
                if isinstance(element, SLnElement):
                    element = element.matrix
                if not isinstance(element, np.ndarray):
                    return False
                if element.shape != (n, n):
                    return False
                if not np.allclose(element.conj().T @ element, np.eye(n)):
                    return False
                if not np.isclose(np.linalg.det(element), 1):
                    return False
                return True

            def order(self) -> int:
                return -1

            def elements(self) -> list[np.ndarray]:
                raise ValueError("无限群没有有限元素列表")

            def generators(self) -> list[np.ndarray]:
                """SU(n)生成元：反厄米矩阵，符合 core 约定"""
                from ...core.lie_theory.specific_lie_algebra import (
                    SpecialUnitaryLieAlgebra,
                )

                su_n = SpecialUnitaryLieAlgebra(n)
                basis = su_n.basis()
                # basis() 返回 MatrixLieAlgebraElement 列表，需要提取 matrix 属性
                return [elem.matrix for elem in basis]

            def exp(self, theta: np.ndarray) -> np.ndarray:
                """指数映射"""
                from scipy.linalg import expm

                theta = np.asarray(theta)
                basis = self.generators()
                generator = sum(theta[i] * basis[i] for i in range(len(basis)))
                return expm(generator)

        return SUNGaugeGroup()

    @classmethod
    def register(cls, name: str, group_class: type):
        """注册新的规范群类型"""
        cls._registry[name] = group_class

    @classmethod
    def list_supported(cls) -> list[str]:
        """列出所有支持的规范群"""
        return list(cls._registry.keys())
