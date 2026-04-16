"""
Clebsch-Gordan系数计算模块

实现角动量耦合的核心计算：
- Clebsch-Gordan系数
- Wigner 3j, 6j, 9j符号
- Racah系数
- 角动量耦合系数
"""

from functools import lru_cache
from math import factorial, sqrt

import numpy as np

# -----------------------------------------------------------------------------
# 1. 辅助函数
# -----------------------------------------------------------------------------


def triangle_condition(j1: float, j2: float, j3: float) -> bool:
    """
    三角条件检验

    |j1 - j2| ≤ j3 ≤ j1 + j2
    """
    return (abs(j1 - j2) <= j3 <= j1 + j2) and (j1 + j2 + j3) == int(j1 + j2 + j3)


def is_valid_jm(j: float, m: float) -> bool:
    """检查j, m是否有效"""
    # j >= 0, m在[-j, j]内
    if j < 0:
        return False
    if abs(m) > j + 1e-10:
        return False
    # j和m都是整数或半整数
    if (abs(2 * j - int(2 * j)) > 1e-10) or (abs(2 * m - int(2 * m)) > 1e-10):
        return False
    return True


def double_factorial(n: int) -> int:
    """双阶乘 n!!"""
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


@lru_cache(maxsize=1024)
def cached_factorial(n: int) -> int:
    """缓存的阶乘"""
    return factorial(n)


# -----------------------------------------------------------------------------
# 2. Clebsch-Gordan系数
# -----------------------------------------------------------------------------


class ClebschGordan:
    """
    Clebsch-Gordan系数计算器

    <j1 m1, j2 m2 | j m>

    用于两个角动量的耦合
    """

    @staticmethod
    def compute(
        j1: float, m1: float, j2: float, m2: float, j: float, m: float
    ) -> float:
        """
        计算Clebsch-Gordan系数

        使用Racah公式：
        C(j1,j2,j; m1,m2,m) = δ(m, m1+m2) × √[(2j+1) × sum] × sqrt_term

        Parameters:
            j1, m1: 第一个角动量及其投影
            j2, m2: 第二个角动量及其投影
            j, m: 耦合后的角动量及其投影

        Returns:
            CG系数值
        """
        # 检查有效性
        if not is_valid_jm(j1, m1) or not is_valid_jm(j2, m2) or not is_valid_jm(j, m):
            return 0.0

        # 检查三角条件
        if not triangle_condition(j1, j2, j):
            return 0.0

        # 检查投影守恒
        if abs(m - (m1 + m2)) > 1e-10:
            return 0.0

        # 特殊情况
        if j1 == 0:
            if j == j2 and m == m2:
                return 1.0
            else:
                return 0.0

        if j2 == 0:
            if j == j1 and m == m1:
                return 1.0
            else:
                return 0.0

        # 使用Racah公式计算
        return ClebschGordan._racah_formula(j1, m1, j2, m2, j, m)

    @staticmethod
    def _racah_formula(
        j1: float, m1: float, j2: float, m2: float, j: float, m: float
    ) -> float:
        """Racah公式实现"""

        # 前置因子
        prefactor = sqrt(
            (2 * j + 1)
            * cached_factorial(int(j1 + j2 - j))
            * cached_factorial(int(j1 - j2 + j))
            * cached_factorial(int(-j1 + j2 + j))
            / cached_factorial(int(j1 + j2 + j + 1))
        )

        prefactor *= sqrt(
            cached_factorial(int(j + m))
            * cached_factorial(int(j - m))
            * cached_factorial(int(j1 + m1))
            * cached_factorial(int(j1 - m1))
            * cached_factorial(int(j2 + m2))
            * cached_factorial(int(j2 - m2))
        )

        # 求和范围
        k_min = max(0, j2 - j - m1, j1 + m2 - j)
        k_max = min(j1 + j2 - j, j1 - m1, j2 + m2)

        # 转换为整数
        k_min = int(k_min)
        k_max = int(k_max)

        # 求和
        sum_term = 0.0
        for k in range(k_min, k_max + 1):
            numerator = (
                cached_factorial(int(j1 + j2 - j - k))
                * cached_factorial(int(j1 - m1 - k))
                * cached_factorial(int(j2 + m2 - k))
                * cached_factorial(int(j - j2 + m1 + k))
                * cached_factorial(int(j - j1 - m2 + k))
            )
            denominator = (
                cached_factorial(k)
                * cached_factorial(int(j + j1 - j2 - k))
                * cached_factorial(int(j - j1 + j2 - k))
            )

            term = ((-1) ** k) * numerator / denominator
            sum_term += term

        return prefactor * sum_term

    @staticmethod
    def coupling_matrix(j1: float, j2: float) -> dict[tuple[float, float], np.ndarray]:
        """
        生成完整的耦合矩阵

        Returns:
            {(j, m): 矩阵} 字典
        """
        matrices = {}

        # 所有可能的j值
        j_min = abs(j1 - j2)
        j_max = j1 + j2

        j_values = [j_min + i for i in range(int(j_max - j_min) + 1)]

        for j in j_values:
            for m in np.arange(-j, j + 1, 1):
                # 构建(j, m)态在(j1, j2)基下的展开系数
                # 矩阵维度: (2*j1+1) × (2*j2+1)

                mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1)))

                for i1, m1 in enumerate(np.arange(-j1, j1 + 1, 1)):
                    for i2, m2 in enumerate(np.arange(-j2, j2 + 1, 1)):
                        cg = ClebschGordan.compute(j1, m1, j2, m2, j, m)
                        mat[i1, i2] = cg

                matrices[(j, m)] = mat

        return matrices

    @staticmethod
    def inverse_clebsch_gordan(
        j: float, m: float, j1: float, j2: float
    ) -> dict[tuple[float, float], float]:
        """
        反向展开：将|j, m>展开为|j1, m1> ⊗ |j2, m2>

        Returns:
            {(m1, m2): 系数}
        """
        expansion = {}

        for m1 in np.arange(-j1, j1 + 1, 1):
            m2 = m - m1
            if abs(m2) <= j2:
                cg = ClebschGordan.compute(j1, m1, j2, m2, j, m)
                if abs(cg) > 1e-10:
                    expansion[(m1, m2)] = cg

        return expansion


# -----------------------------------------------------------------------------
# 3. Wigner 3j符号
# -----------------------------------------------------------------------------


class Wigner3j:
    """
    Wigner 3j符号

    ( j1  j2  j3 )
    ( m1  m2  m3 )

    与CG系数的关系：
    (j1 j2 j3)  = (-1)^(j1-j2-m3) / sqrt(2*j3+1) × C(j1,j2,j3; m1,m2,-m3)
    (m1 m2 m3)
    """

    @staticmethod
    def compute(
        j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
    ) -> float:
        """
        计算3j符号
        """
        # 检查投影守恒
        if abs(m1 + m2 + m3) > 1e-10:
            return 0.0

        # 检查三角条件
        if not triangle_condition(j1, j2, j3):
            return 0.0

        # 与CG系数的关系
        cg = ClebschGordan.compute(j1, m1, j2, m2, j3, -m3)

        # 相位因子
        phase = (-1) ** int(j1 - j2 - m3)

        return phase / sqrt(2 * j3 + 1) * cg

    @staticmethod
    def compute_direct(
        j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
    ) -> float:
        """
        直接计算3j符号（不通过CG系数）

        使用Racah公式
        """
        if abs(m1 + m2 + m3) > 1e-10:
            return 0.0

        if not triangle_condition(j1, j2, j3):
            return 0.0

        # 计算前置因子
        prefactor = (-1) ** int(j1 - j2 - m3) * sqrt(
            cached_factorial(int(j1 + j2 - j3))
            * cached_factorial(int(j1 - j2 + j3))
            * cached_factorial(int(-j1 + j2 + j3))
            / cached_factorial(int(j1 + j2 + j3 + 1))
        )

        t1 = int(j1 - m1)
        t2 = int(j2 + m2)
        t3 = int(j1 + m1)
        t4 = int(j2 - m2)
        t5 = int(j3 - m3)
        t6 = int(j3 + m3)

        prefactor *= sqrt(
            cached_factorial(t1)
            * cached_factorial(t2)
            * cached_factorial(t3)
            * cached_factorial(t4)
            * cached_factorial(t5)
            * cached_factorial(t6)
        )

        # 求和范围
        k_min = max(0, int(j2 - j3 - m1), int(j1 + m2 - j3))
        k_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))

        # 求和
        sum_term = 0.0
        for k in range(k_min, k_max + 1):
            term = ((-1) ** k) / (
                cached_factorial(k)
                * cached_factorial(int(j1 + j2 - j3 - k))
                * cached_factorial(int(j1 - m1 - k))
                * cached_factorial(int(j2 + m2 - k))
                * cached_factorial(int(j3 - j2 + m1 + k))
                * cached_factorial(int(j3 - j1 - m2 + k))
            )
            sum_term += term

        return prefactor * sum_term


# -----------------------------------------------------------------------------
# 4. Wigner 6j符号
# -----------------------------------------------------------------------------


class Wigner6j:
    """
    Wigner 6j符号

    { j1 j2 j3 }
    { j4 j5 j6 }

    用于三个角动量的耦合
    """

    @staticmethod
    def compute(
        j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
    ) -> float:
        """
        计算6j符号

        使用Racah公式
        """
        # 检查四个三角条件
        triads = [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)]

        for a, b, c in triads:
            if not triangle_condition(a, b, c):
                return 0.0

        # Racah公式
        return Wigner6j._racah_formula(j1, j2, j3, j4, j5, j6)

    @staticmethod
    def _racah_formula(
        j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
    ) -> float:
        """Racah公式实现"""

        # 计算三角形乘积
        def tri_prod(a, b, c):
            return (
                cached_factorial(int(a + b - c))
                * cached_factorial(int(a - b + c))
                * cached_factorial(int(-a + b + c))
            )

        # 前置因子
        prefactor = sqrt(
            tri_prod(j1, j2, j3)
            * tri_prod(j1, j5, j6)
            * tri_prod(j4, j2, j6)
            * tri_prod(j4, j5, j3)
            / cached_factorial(int(j1 + j2 + j3 + 1))
        )

        # 求和范围
        k_min = max(j1 + j2 + j3, j1 + j5 + j6, j4 + j2 + j6, j4 + j5 + j3)
        k_max = min(j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6)

        k_min = int(k_min)
        k_max = int(k_max)

        # 求和
        sum_term = 0.0
        for k in range(k_min, k_max + 1):
            numerator = cached_factorial(k + 1)
            denominator = (
                cached_factorial(int(k - j1 - j2 - j3))
                * cached_factorial(int(k - j1 - j5 - j6))
                * cached_factorial(int(k - j4 - j2 - j6))
                * cached_factorial(int(k - j4 - j5 - j3))
                * cached_factorial(int(j1 + j2 + j4 + j5 - k))
                * cached_factorial(int(j1 + j3 + j4 + j6 - k))
                * cached_factorial(int(j2 + j3 + j5 + j6 - k))
            )
            sum_term += (-1) ** k * numerator / denominator

        return prefactor * sum_term


# -----------------------------------------------------------------------------
# 5. Wigner 9j符号
# -----------------------------------------------------------------------------


class Wigner9j:
    """
    Wigner 9j符号

    { j11 j12 j13 }
    { j21 j22 j23 }
    { j31 j32 j33 }

    用于四个角动量的耦合
    """

    @staticmethod
    def compute(
        j11: float,
        j12: float,
        j13: float,
        j21: float,
        j22: float,
        j23: float,
        j31: float,
        j32: float,
        j33: float,
    ) -> float:
        """
        计算9j符号

        通过6j符号展开
        """
        # 检查三角条件（每行每列）
        # 省略详细检查

        # 通过6j符号计算
        sum_val = 0.0

        # 找中间变量的范围
        x_min = max(abs(j11 - j33), abs(j32 - j21), abs(j13 - j22))
        x_max = min(j11 + j33, j32 + j21, j13 + j22)

        for x in np.arange(x_min, x_max + 1, 1):
            sixj1 = Wigner6j.compute(j11, j21, j31, j32, j33, x)
            sixj2 = Wigner6j.compute(j13, j23, j33, x, j21, j12)
            sixj3 = Wigner6j.compute(j11, j12, j13, j23, x, j22)

            sum_val += (2 * x + 1) * sixj1 * sixj2 * sixj3

        return sum_val


# -----------------------------------------------------------------------------
# 6. Racah系数
# -----------------------------------------------------------------------------


class RacahCoefficient:
    """
    Racah系数

    W(j1, j2, J2, j, J1, J)

    用于不同耦合方案的转换
    """

    @staticmethod
    def compute(
        j1: float, j2: float, j3: float, l1: float, l2: float, l3: float
    ) -> float:
        """
        计算Racah系数

        与6j符号的关系：
        W(j1,j2,j3,l1,l2,l3) = (-1)^(j1+j2+l1+l2) × {j1 j2 j3}
                                                    {l2 l1 l3}
        """
        phase = (-1) ** int(j1 + j2 + l1 + l2)
        sixj = Wigner6j.compute(j1, j2, j3, l2, l1, l3)

        return phase * sixj
