"""
球谐函数计算模块

实现球谐函数及相关计算：
- 球谐函数 Y_lm(θ, φ)
- 实球谐函数
- 球谐函数加法定理
- Gaunt系数
- 球谐函数积分
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Callable
from math import factorial, sqrt, pi, cos, sin, exp
from scipy.special import lpmv
from functools import lru_cache


# -----------------------------------------------------------------------------
# 1. 球谐函数
# -----------------------------------------------------------------------------

class SphericalHarmonics:
    """
    球谐函数
    
    Y_l^m(θ, φ) = (-1)^m √[(2l+1)/(4π) (l-m)!/(l+m)!] P_l^m(cos θ) e^{imφ}
    
    P_l^m: 连带勒让德函数
    """
    
    @staticmethod
    def compute(l: int, m: int, theta: float, phi: float) -> complex:
        """
        计算球谐函数 Y_l^m(θ, φ)
        
        Parameters:
            l: 角量子数 (l ≥ 0)
            m: 磁量子数 (-l ≤ m ≤ l)
            theta: 极角 [0, π]
            phi: 方位角 [0, 2π)
        
        Returns:
            球谐函数值（复数）
        """
        # 检查有效性
        if l < 0:
            raise ValueError(f"l must be non-negative, got {l}")
        if abs(m) > l:
            raise ValueError(f"|m| must be ≤ l, got l={l}, m={m}")
        
        # 使用Condon-Shortley相位约定
        norm = sqrt((2*l + 1) / (4*pi) * 
                   factorial(l - m) / factorial(l + m))
        
        # 连带勒让德函数
        P_lm = lpmv(m, l, cos(theta))
        
        # 球谐函数
        phase = (-1)**m if m >= 0 else (-1)**m
        
        Y = phase * norm * P_lm * exp(1j * m * phi)
        
        return Y
    
    @staticmethod
    def compute_array(l: int, m: int, 
                      theta_array: np.ndarray, 
                      phi_array: np.ndarray) -> np.ndarray:
        """批量计算球谐函数"""
        # 网格化
        theta_grid, phi_grid = np.meshgrid(theta_array, phi_array, indexing='ij')
        
        # 矢量化计算
        norm = sqrt((2*l + 1) / (4*pi) * 
                   factorial(l - abs(m)) / factorial(l + abs(m)))
        
        P_lm = lpmv(abs(m), l, np.cos(theta_grid))
        
        phase = (-1)**m if m >= 0 else (-1)**m
        
        Y = phase * norm * P_lm * np.exp(1j * m * phi_grid)
        
        return Y
    
    @staticmethod
    def real_spherical_harmonic(l: int, m: int, theta: float, phi: float) -> float:
        """
        实球谐函数
        
        m > 0: Y_l^m = √2 (-1)^m [Y_l^m + (-1)^m Y_l^{-m}] / 2
        m = 0: Y_l^0 = Y_l^0
        m < 0: Y_l^m = √2 (-1)^m [Y_l^{-|m|} - (-1)^m Y_l^{|m|}] / (2i)
        """
        if m == 0:
            return SphericalHarmonics.compute(l, 0, theta, phi).real
        elif m > 0:
            Y_plus = SphericalHarmonics.compute(l, m, theta, phi)
            Y_minus = SphericalHarmonics.compute(l, -m, theta, phi)
            return sqrt(2) * (-1)**m * (Y_plus + (-1)**m * Y_minus).real / 2
        else:  # m < 0
            Y_plus = SphericalHarmonics.compute(l, -m, theta, phi)
            Y_minus = SphericalHarmonics.compute(l, m, theta, phi)
            return sqrt(2) * (-1)**m * (Y_plus - (-1)**m * Y_minus).imag / 2
    
    @staticmethod
    def orthonormality_integral(l1: int, m1: int, 
                                l2: int, m2: int) -> float:
        """
        正交归一性积分
        
        ∫∫ Y_{l1}^{m1*} Y_{l2}^{m2} sin θ dθ dφ = δ_{l1,l2} δ_{m1,m2}
        """
        if l1 == l2 and m1 == m2:
            return 1.0
        else:
            return 0.0


# -----------------------------------------------------------------------------
# 2. 球谐函数加法定理
# -----------------------------------------------------------------------------

class SphericalHarmonicsAddition:
    """
    球谐函数加法定理
    """
    
    @staticmethod
    def addition_theorem(l: int, 
                        theta1: float, phi1: float,
                        theta2: float, phi2: float) -> float:
        """
        球谐函数加法定理
        
        Σ_{m=-l}^{l} Y_l^m*(θ1,φ1) Y_l^m(θ2,φ2) = (2l+1)/(4π) P_l(cos γ)
        
        γ: 两点之间的夹角
        """
        # 计算夹角 γ
        cos_gamma = (cos(theta1) * cos(theta2) + 
                    sin(theta1) * sin(theta2) * cos(phi1 - phi2))
        
        # 勒让德多项式
        P_l = lpmv(0, l, cos_gamma)
        
        return (2*l + 1) / (4*pi) * P_l
    
    @staticmethod
    def angle_between_vectors(r1: np.ndarray, r2: np.ndarray) -> float:
        """
        计算两个矢量之间的夹角
        """
        cos_gamma = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
        return np.arccos(np.clip(cos_gamma, -1, 1))


# -----------------------------------------------------------------------------
# 3. Gaunt系数
# -----------------------------------------------------------------------------

class GauntCoefficient:
    """
    Gaunt系数
    
    三个球谐函数的积分：
    
    G(l1,l2,l3; m1,m2,m3) = ∫∫ Y_{l1}^{m1} Y_{l2}^{m2} Y_{l3}^{m3} sin θ dθ dφ
    """
    
    @staticmethod
    def compute(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
        """
        计算Gaunt系数
        
        通过Wigner 3j符号计算
        """
        # 检查投影守恒
        if m1 + m2 + m3 != 0:
            return 0.0
        
        # 检查三角条件
        if not (abs(l1 - l2) <= l3 <= l1 + l2):
            return 0.0
        
        # 使用Wigner 3j符号
        from .clebsch_gordan import Wigner3j
        
        three_j = Wigner3j.compute(l1, l2, l3, m1, m2, m3)
        
        # 另一个3j符号 (l1 l2 l3; 0 0 0)
        three_j_000 = Wigner3j.compute(l1, l2, l3, 0, 0, 0)
        
        # Gaunt系数公式: G = sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) × 3j(0,0,0) × 3j(m1,m2,m3)
        gaunt = sqrt(
            (2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*pi)
        ) * three_j_000 * three_j
        
        return gaunt
    
    @staticmethod
    def compute_numerical(l1: int, m1: int, l2: int, m2: int, 
                         l3: int, m3: int, n_points: int = 100) -> float:
        """
        数值积分计算Gaunt系数
        """
        from scipy.integrate import dblquad
        
        def integrand(theta, phi):
            Y1 = SphericalHarmonics.compute(l1, m1, theta, phi)
            Y2 = SphericalHarmonics.compute(l2, m2, theta, phi)
            Y3 = SphericalHarmonics.compute(l3, m3, theta, phi)
            return (Y1 * Y2 * Y3 * sin(theta)).real
        
        result, _ = dblquad(
            integrand, 0, 2*pi,
            lambda x: 0, lambda x: pi,
            epsabs=1e-10, epsrel=1e-10
        )
        
        return result


# -----------------------------------------------------------------------------
# 4. 球谐函数积分
# -----------------------------------------------------------------------------

class SphericalHarmonicsIntegral:
    """
    球谐函数积分工具
    """
    
    @staticmethod
    def integrate_product(Y1: Callable, Y2: Callable, 
                         n_points: int = 200) -> complex:
        """
        数值积分两个球谐函数的乘积
        """
        from scipy.integrate import dblquad
        
        def integrand(theta, phi):
            y1 = Y1(theta, phi)
            y2 = Y2(theta, phi)
            return (np.conj(y1) * y2 * sin(theta))
        
        result, _ = dblquad(
            lambda th, ph: integrand(th, ph),
            0, 2*pi,
            lambda x: 0, lambda x: pi
        )
        
        return result
    
    @staticmethod
    def expand_in_spherical_harmonics(f: Callable, l_max: int) -> Dict[Tuple[int, int], complex]:
        """
        将函数展开为球谐函数级数
        
        f(θ, φ) = Σ_{l=0}^{l_max} Σ_{m=-l}^{l} a_{lm} Y_l^m(θ, φ)
        
        Returns:
            {(l, m): 展开系数 a_{lm}}
        """
        from scipy.integrate import dblquad
        
        coefficients = {}
        
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                # 计算系数 a_{lm} = ∫∫ f(θ,φ) Y_l^{m*}(θ,φ) sin θ dθ dφ
                
                def integrand(theta, phi):
                    y_lm = SphericalHarmonics.compute(l, m, theta, phi)
                    return (f(theta, phi) * np.conj(y_lm) * sin(theta))
                
                # 数值积分
                result, _ = dblquad(
                    lambda th, ph: integrand(th, ph),
                    0, 2*pi,
                    lambda x: 0, lambda x: pi,
                    epsabs=1e-8, epsrel=1e-8
                )
                
                coefficients[(l, m)] = result
        
        return coefficients


# -----------------------------------------------------------------------------
# 5. 梯度和其他算符
# -----------------------------------------------------------------------------

class SphericalHarmonicsOperators:
    """
    球谐函数上的算符
    """
    
    @staticmethod
    def angular_momentum_l_plus(l: int, m: int) -> Tuple[int, int]:
        """
        升算符 L⁺ 作用
        
        L⁺ Y_l^m = √[l(l+1) - m(m+1)] Y_l^{m+1}
        
        Returns:
            (新l, 新m, 系数)
        """
        if m >= l:
            return (l, l, 0.0)  # 湮灭
        
        coeff = sqrt(l*(l+1) - m*(m+1))
        return (l, m+1, coeff)
    
    @staticmethod
    def angular_momentum_l_minus(l: int, m: int) -> Tuple[int, int]:
        """
        降算符 L⁻ 作用
        
        L⁻ Y_l^m = √[l(l+1) - m(m-1)] Y_l^{m-1}
        """
        if m <= -l:
            return (l, -l, 0.0)  # 湮灭
        
        coeff = sqrt(l*(l+1) - m*(m-1))
        return (l, m-1, coeff)
    
    @staticmethod
    def angular_momentum_l_z(l: int, m: int) -> float:
        """
        L_z 算符
        
        L_z Y_l^m = m Y_l^m
        """
        return float(m)
    
    @staticmethod
    def angular_momentum_l_squared(l: int, m: int) -> float:
        """
        L² 算符
        
        L² Y_l^m = l(l+1) Y_l^m
        """
        return l * (l + 1)



