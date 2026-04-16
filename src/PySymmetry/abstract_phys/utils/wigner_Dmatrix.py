"""
Wigner D矩阵和D函数计算模块

实现旋转群的表示：
- Wigner小d矩阵 d^j_{m'm}(β)
- Wigner大D矩阵 D^j_{m'm}(α,β,γ)
- Euler角表示
- 旋转矩阵元
"""

from math import cos, factorial, pi, sin, sqrt

import numpy as np

# -----------------------------------------------------------------------------
# 1. Wigner小d矩阵
# -----------------------------------------------------------------------------


class WignerSmallD:
    """
    Wigner小d矩阵 d^j_{m'm}(β)

    定义：
    d^j_{m'm}(β) = <j m' | exp(-iβJ_y) | j m>

    用于绕y轴旋转
    """

    @staticmethod
    def compute(j: float, m_prime: float, m: float, beta: float) -> float:
        """
        计算Wigner小d矩阵元

        使用Wigner公式
        """
        # 检查量子数有效性
        if not (abs(m_prime) <= j and abs(m) <= j):
            return 0.0

        # 特殊情况
        if j == 0:
            return 1.0

        # 使用Wigner公式
        return WignerSmallD._wigner_formula(j, m_prime, m, beta)

    @staticmethod
    def _wigner_formula(j: float, m_prime: float, m: float, beta: float) -> float:
        """Wigner公式实现"""

        # 求和范围
        k_min = max(0, m - m_prime)
        k_max = min(j + m, j - m_prime)

        # 转换为整数
        k_min = int(k_min)
        k_max = int(k_max)

        # 前置因子
        prefactor = sqrt(
            factorial(int(j + m))
            * factorial(int(j - m))
            * factorial(int(j + m_prime))
            * factorial(int(j - m_prime))
        )

        sum_term = 0.0
        for k in range(k_min, k_max + 1):
            numerator = (
                (-1) ** (k + m - m_prime)
                * cos(beta / 2) ** (2 * j + m_prime - m - 2 * k)
                * sin(beta / 2) ** (m - m_prime + 2 * k)
            )

            denominator = (
                factorial(k)
                * factorial(int(j + m - k))
                * factorial(int(j - m_prime - k))
                * factorial(int(k + m_prime - m))
            )

            sum_term += numerator / denominator

        return prefactor * sum_term

    @staticmethod
    def matrix(j: float, beta: float) -> np.ndarray:
        """
        生成完整的d^j矩阵

        Returns:
            (2j+1) × (2j+1) 矩阵
        """
        dim = int(2 * j + 1)
        d_matrix = np.zeros((dim, dim))

        for i, m_prime in enumerate(np.arange(-j, j + 1, 1)):
            for j_idx, m in enumerate(np.arange(-j, j + 1, 1)):
                d_matrix[i, j_idx] = WignerSmallD.compute(j, m_prime, m, beta)

        return d_matrix

    @staticmethod
    def symmetry_relations(
        j: float, m_prime: float, m: float, beta: float
    ) -> dict[str, float]:
        """
        对称性关系

        d^j_{m'm}(β) = (-1)^{m'-m} d^j_{mm'}(β)
        d^j_{m'm}(-β) = d^j_{mm'}(β)
        d^j_{m'm}(π-β) = (-1)^{j-m'} d^j_{-m',m}(β)
        """
        return {
            "original": WignerSmallD.compute(j, m_prime, m, beta),
            "transpose": WignerSmallD.compute(j, m, m_prime, beta),
            "negative_beta": WignerSmallD.compute(j, m_prime, m, -beta),
            "pi_minus_beta": WignerSmallD.compute(j, -m_prime, m, pi - beta),
        }


# -----------------------------------------------------------------------------
# 2. Wigner大D矩阵
# -----------------------------------------------------------------------------


class WignerBigD:
    """
    Wigner大D矩阵 D^j_{m'm}(α,β,γ)

    Euler角表示的旋转算符矩阵元

    D^j_{m'm}(α,β,γ) = e^{-im'α} d^j_{m'm}(β) e^{-imγ}

    对应旋转 R(α,β,γ) = R_z(α) R_y(β) R_z(γ)
    """

    @staticmethod
    def compute(
        j: float, m_prime: float, m: float, alpha: float, beta: float, gamma: float
    ) -> complex:
        """
        计算Wigner大D矩阵元

        Parameters:
            j: 角动量量子数
            m_prime: 末态投影
            m: 初态投影
            alpha, beta, gamma: Euler角
        """
        d = WignerSmallD.compute(j, m_prime, m, beta)

        # 相位因子
        phase = np.exp(-1j * m_prime * alpha) * np.exp(-1j * m * gamma)

        return d * phase

    @staticmethod
    def matrix(j: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        生成完整的D^j矩阵
        """
        dim = int(2 * j + 1)
        D_matrix = np.zeros((dim, dim), dtype=complex)

        for i, m_prime in enumerate(np.arange(-j, j + 1, 1)):
            for j_idx, m in enumerate(np.arange(-j, j + 1, 1)):
                D_matrix[i, j_idx] = WignerBigD.compute(
                    j, m_prime, m, alpha, beta, gamma
                )

        return D_matrix

    @staticmethod
    def composition(D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
        """
        矩阵乘法：两个旋转的组合

        D(R1 ∘ R2) = D(R1) × D(R2)
        """
        return D1 @ D2

    @staticmethod
    def inverse(D: np.ndarray) -> np.ndarray:
        """
        D矩阵的逆

        D(R⁻¹) = D(R)†
        """
        return D.conj().T


# -----------------------------------------------------------------------------
# 3. 特殊旋转的D矩阵
# -----------------------------------------------------------------------------


class SpecialRotations:
    """特殊旋转的D矩阵"""

    @staticmethod
    def rotation_x(j: float, angle: float) -> np.ndarray:
        """
        绕x轴旋转

        Rx(θ) = Rz(-π/2) Ry(θ) Rz(π/2)
        """
        D1 = WignerBigD.matrix(j, -pi / 2, 0, 0)
        D2 = WignerBigD.matrix(j, 0, angle, 0)
        D3 = WignerBigD.matrix(j, pi / 2, 0, 0)

        return D1 @ D2 @ D3

    @staticmethod
    def rotation_y(j: float, angle: float) -> np.ndarray:
        """绕y轴旋转"""
        return WignerBigD.matrix(j, 0, angle, 0)

    @staticmethod
    def rotation_z(j: float, angle: float) -> np.ndarray:
        """绕z轴旋转"""
        return WignerBigD.matrix(j, angle, 0, 0)

    @staticmethod
    def rotation_axis_angle(j: float, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        绕任意轴旋转

        使用Rodrigues公式转换为Euler角
        """
        # 归一化轴
        axis = axis / np.linalg.norm(axis)

        # 转换为Euler角
        # 这是一个复杂的转换，简化实现
        # 实际需要完整的转换公式

        # 简化：使用小d矩阵的组合
        # ...

        pass

    @staticmethod
    def identity(j: float) -> np.ndarray:
        """单位旋转"""
        return WignerBigD.matrix(j, 0, 0, 0)

    @staticmethod
    def inversion(j: float) -> np.ndarray:
        """
        空间反演

        P = (-1)^j I
        """
        dim = int(2 * j + 1)
        return (-1) ** j * np.eye(dim, dtype=complex)


# -----------------------------------------------------------------------------
# 4. D矩阵的性质
# -----------------------------------------------------------------------------


class WignerDProperties:
    """Wigner D矩阵的性质"""

    @staticmethod
    def orthonormality(j: float) -> bool:
        """
        正交归一性

        ∫ D^j_{m1'm1}* D^j_{m2'm2} dΩ = (8π²)/(2j+1) δ_{m1'm2'} δ_{m1m2}
        """
        # 验证性质
        pass

    @staticmethod
    def completeness(j_max: float) -> bool:
        """
        完备性

        Σ_{j=0}^{∞} Σ_{m,m'} (2j+1)/(8π²) D^j_{mm'}*(α,β,γ) D^j_{mm'}(α',β',γ')
        = δ(cosθ-cosθ') δ(φ-φ') δ(ψ-ψ')
        """
        pass

    @staticmethod
    def addition_theorem(j: float, angles1: tuple, angles2: tuple) -> np.ndarray:
        """
        加法定理

        D(R1 ∘ R2) = D(R1) D(R2)
        """
        alpha1, beta1, gamma1 = angles1
        alpha2, beta2, gamma2 = angles2

        D1 = WignerBigD.matrix(j, alpha1, beta1, gamma1)
        D2 = WignerBigD.matrix(j, alpha2, beta2, gamma2)

        return D1 @ D2


# -----------------------------------------------------------------------------
# 5. 球谐函数的旋转
# -----------------------------------------------------------------------------


class SphericalHarmonicsRotation:
    """球谐函数的旋转"""

    @staticmethod
    def rotate_harmonic(
        l: int, m: int, alpha: float, beta: float, gamma: float
    ) -> dict[int, complex]:
        """
        旋转球谐函数

        R(α,β,γ) Y_l^m = Σ_{m'=-l}^{l} D^l_{m'm}(α,β,γ) Y_l^{m'}

        Returns:
            {m': 系数}
        """
        coefficients = {}

        for m_prime in range(-l, l + 1):
            D = WignerBigD.compute(l, m_prime, m, alpha, beta, gamma)
            if abs(D) > 1e-10:
                coefficients[m_prime] = D

        return coefficients

    @staticmethod
    def rotation_matrix_for_l(
        l: int, alpha: float, beta: float, gamma: float
    ) -> np.ndarray:
        """
        对于固定l的旋转矩阵

        作用于Y_l^m的基
        """
        return WignerBigD.matrix(l, alpha, beta, gamma)
