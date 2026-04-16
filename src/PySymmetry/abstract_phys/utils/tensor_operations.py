"""
张量算符模块

实现不可约张量算符理论：
- 不可约张量算符
- Wigner-Eckart定理
- 约化矩阵元
- 张量积
- 3j符号应用
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .clebsch_gordan import ClebschGordan, Wigner3j

# -----------------------------------------------------------------------------
# 1. 不可约张量算符基类
# -----------------------------------------------------------------------------


class IrreducibleTensorOperator(ABC):
    """
    不可约张量算符基类

    T_q^k: 秩为k的不可约张量算符的第q个分量

    满足对易关系：
    [J_z, T_q^k] = q T_q^k
    [J_±, T_q^k] = √[k(k+1) - q(q±1)] T_{q±1}^k
    """

    def __init__(self, rank: int, component: int):
        """
        Parameters:
            rank: 张量秩 k
            component: 分量 q (-k ≤ q ≤ k)
        """
        if abs(component) > rank:
            raise ValueError(f"Component |{component}| must be ≤ rank {rank}")

        self.rank = rank
        self.component = component

    @abstractmethod
    def matrix_element(self, bra_state: Any, ket_state: Any) -> complex:
        """
        计算矩阵元 <bra| T_q^k |ket>
        """
        pass

    def __repr__(self) -> str:
        return f"T^{self.rank}_{self.component}"


# -----------------------------------------------------------------------------
# 2. Wigner-Eckart定理
# -----------------------------------------------------------------------------


class WignerEckartTheorem:
    """
    Wigner-Eckart定理

    <α', j', m' | T_q^k | α, j, m> = <j, m, k, q | j', m'>
                                    × <α', j' || T^k || α, j> / √(2j'+1)

    或用3j符号：
    <α', j', m' | T_q^k | α, j, m> = (-1)^{j'-m'} (j' k j; -m' q m)
                                    × <α', j' || T^k || α, j>

    其中 <α', j' || T^k || α, j> 是约化矩阵元
    """

    @staticmethod
    def matrix_element(
        reduced_me: complex,
        j: float,
        m: float,
        k: int,
        q: int,
        j_prime: float,
        m_prime: float,
    ) -> complex:
        """
        使用Wigner-Eckart定理计算矩阵元

        Parameters:
            reduced_me: 约化矩阵元 <α', j' || T^k || α, j>
            j, m: 初态角动量量子数
            k, q: 张量算符的秩和分量
            j_prime, m_prime: 末态角动量量子数

        Returns:
            完整矩阵元
        """
        # 检查选择定则
        # |j - k| ≤ j' ≤ j + k
        if not (abs(j - k) <= j_prime <= j + k):
            return 0.0

        # m' = m + q
        if abs(m_prime - (m + q)) > 1e-10:
            return 0.0

        # 使用CG系数
        cg = ClebschGordan.compute(j, m, k, q, j_prime, m_prime)

        # Wigner-Eckart定理
        return cg * reduced_me / np.sqrt(2 * j_prime + 1)

    @staticmethod
    def matrix_element_3j(
        reduced_me: complex,
        j: float,
        m: float,
        k: int,
        q: int,
        j_prime: float,
        m_prime: float,
    ) -> complex:
        """
        使用3j符号的版本
        """
        if not (abs(j - k) <= j_prime <= j + k):
            return 0.0

        if abs(m_prime - (m + q)) > 1e-10:
            return 0.0

        # 使用3j符号
        three_j = Wigner3j.compute(j, k, j_prime, m, q, -m_prime)

        # 相位因子
        phase = (-1) ** (j_prime - m_prime)

        return phase * three_j * reduced_me

    @staticmethod
    def reduced_matrix_element(
        full_me: complex,
        j: float,
        m: float,
        k: int,
        q: int,
        j_prime: float,
        m_prime: float,
    ) -> complex:
        """
        从完整矩阵元提取约化矩阵元
        """
        # 检查选择定则
        if not (abs(j - k) <= j_prime <= j + k):
            return 0.0

        if abs(m_prime - (m + q)) > 1e-10:
            return 0.0

        # CG系数
        cg = ClebschGordan.compute(j, m, k, q, j_prime, m_prime)

        if abs(cg) < 1e-10:
            return 0.0

        # 提取约化矩阵元
        return full_me * np.sqrt(2 * j_prime + 1) / cg


# -----------------------------------------------------------------------------
# 3. 标准张量算符
# -----------------------------------------------------------------------------


class ScalarOperator(IrreducibleTensorOperator):
    """
    标量算符 (k=0)

    T_0^0

    与角动量对易：[J_i, T_0^0] = 0
    """

    def __init__(self, operator: np.ndarray):
        super().__init__(rank=0, component=0)
        self._operator = operator

    def matrix_element(self, bra_state: Any, ket_state: Any) -> complex:
        """标量矩阵元"""
        # 对于标量：约化矩阵元 = 完整矩阵元
        if hasattr(bra_state, "j") and hasattr(ket_state, "j"):
            # 检查 j' = j, m' = m
            if bra_state.j != ket_state.j or bra_state.m != ket_state.m:
                return 0.0

        return np.vdot(bra_state.vector, self._operator @ ket_state.vector)


class VectorOperator(IrreducibleTensorOperator):
    """
    矢量算符 (k=1)

    T_q^1, q = -1, 0, 1

    分量与球谐分量的关系：
    T_+1^1 = -(V_x + iV_y)/√2
    T_0^1 = V_z
    T_-1^1 = (V_x - iV_y)/√2
    """

    def __init__(self, component: int, cartesian_operator: np.ndarray = None):
        super().__init__(rank=1, component=component)
        self._cartesian = cartesian_operator

    def matrix_element(self, bra_state: Any, ket_state: Any) -> complex:
        """矢量矩阵元"""
        if self._cartesian is None:
            raise NotImplementedError("需要提供具体的算符")

        # 使用Wigner-Eckart定理
        pass

    @classmethod
    def from_cartesian(
        cls, Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray
    ) -> list["VectorOperator"]:
        """从笛卡尔分量创建球分量"""
        T_plus = cls(1)  # -(Vx + iVy)/√2
        T_zero = cls(0)  # Vz
        T_minus = cls(-1)  # (Vx - iVy)/√2

        return [T_plus, T_zero, T_minus]


class QuadraticTensorOperator(IrreducibleTensorOperator):
    """
    二阶张量算符 (k=2)

    例如四极矩算符 Q_q^2
    """

    def __init__(self, component: int):
        super().__init__(rank=2, component=component)

    def matrix_element(self, bra_state: Any, ket_state: Any) -> complex:
        pass


# -----------------------------------------------------------------------------
# 4. 角动量算符
# -----------------------------------------------------------------------------


class AngularMomentumOperators:
    """角动量算符"""

    @staticmethod
    def J_plus(j: float) -> np.ndarray:
        """
        升算符 J⁺

        <j, m'| J⁺ |j, m> = √[j(j+1) - m(m+1)] δ_{m', m+1}
        """
        dim = int(2 * j + 1)
        J_p = np.zeros((dim, dim), dtype=complex)

        for i, m in enumerate(np.arange(-j, j + 1, 1)):
            if m < j:
                # m' = m + 1
                j_idx = i + 1
                J_p[j_idx, i] = np.sqrt(j * (j + 1) - m * (m + 1))

        return J_p

    @staticmethod
    def J_minus(j: float) -> np.ndarray:
        """
        降算符 J⁻

        <j, m'| J⁻ |j, m> = √[j(j+1) - m(m-1)] δ_{m', m-1}
        """
        dim = int(2 * j + 1)
        J_m = np.zeros((dim, dim), dtype=complex)

        for i, m in enumerate(np.arange(-j, j + 1, 1)):
            if m > -j:
                # m' = m - 1
                j_idx = i - 1
                J_m[j_idx, i] = np.sqrt(j * (j + 1) - m * (m - 1))

        return J_m

    @staticmethod
    def J_x(j: float) -> np.ndarray:
        """J_x = (J⁺ + J⁻)/2"""
        return (
            AngularMomentumOperators.J_plus(j) + AngularMomentumOperators.J_minus(j)
        ) / 2

    @staticmethod
    def J_y(j: float) -> np.ndarray:
        """J_y = (J⁺ - J⁻)/(2i)"""
        return (
            AngularMomentumOperators.J_plus(j) - AngularMomentumOperators.J_minus(j)
        ) / (2 * 1j)

    @staticmethod
    def J_z(j: float) -> np.ndarray:
        """
        J_z

        <j, m'| J_z |j, m> = m δ_{m', m}
        """
        dim = int(2 * j + 1)
        J_z = np.zeros((dim, dim), dtype=complex)

        for i, m in enumerate(np.arange(-j, j + 1, 1)):
            J_z[i, i] = m

        return J_z

    @staticmethod
    def J_squared(j: float) -> np.ndarray:
        """
        J² = J_x² + J_y² + J_z²

        <j, m'| J² |j, m> = j(j+1) δ_{m', m}
        """
        dim = int(2 * j + 1)
        J_sq = j * (j + 1) * np.eye(dim, dtype=complex)
        return J_sq

    @staticmethod
    def spherical_components(j: float) -> list[np.ndarray]:
        """
        J的球分量

        J_+1^1 = -(J_x + iJ_y)/√2
        J_0^1 = J_z
        J_-1^1 = (J_x - iJ_y)/√2
        """
        Jx = AngularMomentumOperators.J_x(j)
        Jy = AngularMomentumOperators.J_y(j)
        Jz = AngularMomentumOperators.J_z(j)

        J_plus1 = -(Jx + 1j * Jy) / np.sqrt(2)
        J_0 = Jz
        J_minus1 = (Jx - 1j * Jy) / np.sqrt(2)

        return [J_plus1, J_0, J_minus1]


# -----------------------------------------------------------------------------
# 5. 张量积
# -----------------------------------------------------------------------------


class TensorProduct:
    """
    张量积

    两个不可约张量的乘积

    [T^{k1} ⊗ U^{k2}]_q^k = Σ_{q1, q2} <k1, q1, k2, q2 | k, q> T_{q1}^{k1} U_{q2}^{k2}
    """

    @staticmethod
    def product(
        T1: IrreducibleTensorOperator,
        T2: IrreducibleTensorOperator,
        result_rank: int,
        result_component: int,
    ) -> IrreducibleTensorOperator:
        """
        计算张量积

        Parameters:
            T1: 第一个张量算符
            T2: 第二个张量算符
            result_rank: 结果张量的秩
            result_component: 结果张量的分量
        """
        # 检查三角条件
        if not (abs(T1.rank - T2.rank) <= result_rank <= T1.rank + T2.rank):
            raise ValueError("张量积的秩不满足三角条件")

        # 创建新张量
        class ProductTensor(IrreducibleTensorOperator):
            def __init__(self, t1, t2, k, q):
                super().__init__(k, q)
                self._t1 = t1
                self._t2 = t2

            def matrix_element(self, bra, ket):
                # 计算张量积的矩阵元
                result = 0.0
                for q1 in range(-self._t1.rank, self._t1.rank + 1):
                    q2 = self.component - q1
                    if abs(q2) <= self._t2.rank:
                        cg = ClebschGordan.compute(
                            self._t1.rank,
                            q1,
                            self._t2.rank,
                            q2,
                            self.rank,
                            self.component,
                        )
                        # 需要中间态求和
                        # 简化版本
                        result += (
                            cg
                            * self._t1.matrix_element(bra, ket)
                            * self._t2.matrix_element(bra, ket)
                        )

                return result

        return ProductTensor(T1, T2, result_rank, result_component)

    @staticmethod
    def scalar_product(
        T1: IrreducibleTensorOperator, T2: IrreducibleTensorOperator
    ) -> "ScalarOperator":
        """
        标量积

        T¹ · U¹ = Σ_q (-1)^q T_q^1 U_{-q}^1
        """
        # 标量积是k=0的张量积
        result = 0.0
        for q in range(-1, 2):
            result += (-1) ** q * T1.component_value(q) * T2.component_value(-q)

        return ScalarOperator(np.array([[result]]))


# -----------------------------------------------------------------------------
# 6. 6j符号应用
# -----------------------------------------------------------------------------


class SixJApplications:
    """6j符号的应用"""

    @staticmethod
    def recoupling_coefficient(
        j1: float, j2: float, j12: float, j3: float, J: float, j23: float
    ) -> float:
        """
        重耦合系数

        |(j1 j2)j12, j3; J> 与 |j1, (j2 j3)j23; J> 之间的变换系数
        """
        from .clebsch_gordan import Wigner6j

        # 重耦合系数与6j符号的关系
        return (
            (-1) ** (j1 + j2 + j3 + J)
            * np.sqrt((2 * j12 + 1) * (2 * j23 + 1))
            * Wigner6j.compute(j1, j2, j12, j3, J, j23)
        )

    @staticmethod
    def reduced_matrix_element_product(
        T1_rank: int,
        T2_rank: int,
        j1: float,
        j: float,
        j2: float,
        reduced_me1: complex,
        reduced_me2: complex,
    ) -> complex:
        """
        计算张量积的约化矩阵元
        """
        from .clebsch_gordan import Wigner6j

        result = 0.0
        for j_prime in np.arange(abs(j1 - T1_rank), j1 + T1_rank + 1, 1):
            six_j = Wigner6j.compute(T1_rank, j1, j_prime, j2, j, T2_rank)

            phase = (-1) ** (j_prime + j2 + j + T1_rank)

            result += phase * (2 * j_prime + 1) * six_j * reduced_me1 * reduced_me2

        return result
