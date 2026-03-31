"""
庞加莱群模块

实现相对论的完全时空对称性：
- 洛伦兹变换 + 时空平移
- 庞加莱代数
- 不可约表示：质量和自旋
- 单粒子态分类
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod

from .lorentz_group import (
    LorentzTransformation, LorentzGroup, FourVector,
    MetricSignature
)
from .base import PhysicalSymmetry, SymmetryOperation, SymmetryParameters, SymmetryType, SymmetryCategory



# -----------------------------------------------------------------------------
# 1. 庞加莱变换
# -----------------------------------------------------------------------------

@dataclass
class PoincareTransformation(SymmetryOperation):
    """
    庞加莱变换
    
    (Λ, a): xμ → Λμν xν + aμ
    
    包含洛伦兹变换和时空平移
    """
    
    lorentz: LorentzTransformation
    translation: np.ndarray  # 四维平移矢量
    
    def __init__(self, 
                 lorentz: LorentzTransformation,
                 translation: np.ndarray = None):
        self.lorentz = lorentz
        
        if translation is None:
            self.translation = np.zeros(4)
        else:
            self.translation = np.asarray(translation, dtype=float)
            if len(self.translation) != 4:
                raise ValueError("平移必须是四维矢量")
    
    @property
    def group(self):
        return "Poincare"
    
    @property
    def is_continuous(self) -> bool:
        return True
    
    @property
    def matrix(self) -> np.ndarray:
        """5×5齐次坐标矩阵"""
        matrix = np.eye(5)
        matrix[0:4, 0:4] = self.lorentz.matrix
        matrix[0:4, 4] = self.translation
        return matrix
    
    def compose(self, other: 'PoincareTransformation') -> 'PoincareTransformation':
        """
        组合两个庞加莱变换
        
        (Λ₁, a₁)(Λ₂, a₂) = (Λ₁Λ₂, Λ₁a₂ + a₁)
        """
        new_lorentz = self.lorentz.compose(other.lorentz)
        new_translation = (self.lorentz.matrix @ other.translation + 
                          self.translation)
        
        return PoincareTransformation(new_lorentz, new_translation)
    
    def inverse(self) -> 'PoincareTransformation':
        """
        逆变换
        
        (Λ, a)⁻¹ = (Λ⁻¹, -Λ⁻¹a)
        """
        inv_lorentz = self.lorentz.inverse()
        inv_translation = -inv_lorentz.matrix @ self.translation
        
        return PoincareTransformation(inv_lorentz, inv_translation)
    
    def act_on(self, four_vector: FourVector) -> FourVector:
        """作用于四矢量"""
        # x' = Λx + a
        transformed = self.lorentz.act_on(four_vector)
        new_components = transformed.components + self.translation
        
        return FourVector(new_components, four_vector.metric)
    
    def act_on_field(self, field_value: np.ndarray, 
                     position: FourVector) -> np.ndarray:
        """
        作用于场
        
        φ'(x) = φ(Λ⁻¹(x - a))
        """
        # 需要根据场的类型（标量、旋量、矢量）实现
        pass
    
    @classmethod
    def identity(cls) -> 'PoincareTransformation':
        """单位元"""
        return cls(LorentzTransformation.identity(), np.zeros(4))
    
    @classmethod
    def pure_translation(cls, translation: np.ndarray) -> 'PoincareTransformation':
        """纯平移"""
        return cls(LorentzTransformation.identity(), translation)
    
    @classmethod
    def pure_lorentz(cls, lorentz: LorentzTransformation) -> 'PoincareTransformation':
        """纯洛伦兹变换"""
        return cls(lorentz, np.zeros(4))
    
    @classmethod
    def time_translation(cls, dt: float) -> 'PoincareTransformation':
        """时间平移"""
        translation = np.array([dt, 0, 0, 0])
        return cls.pure_translation(translation)
    
    @classmethod
    def spatial_translation(cls, dx: np.ndarray) -> 'PoincareTransformation':
        """空间平移"""
        translation = np.array([0, dx[0], dx[1], dx[2]])
        return cls.pure_translation(translation)
    
    def __repr__(self) -> str:
        return (f"PoincareTransformation(lorentz={self.lorentz.transformation_type()}, "
                f"translation={self.translation})")


# -----------------------------------------------------------------------------
# 2. 庞加莱群
# -----------------------------------------------------------------------------

class PoincareGroup(PhysicalSymmetry):
    """
    庞加莱群（非齐次洛伦兹群）
    
    ISO⁺(1,3) 或 ISL(2,C)
    
    半直积结构：ℝ⁴ ⋊ SO⁺(1,3)
    """
    
    def __init__(self, proper: bool = True, orthochronous: bool = True):
        super().__init__(
            symmetry_type=SymmetryType.LORENTZ,  # 使用相近类型
            group=None,  # 庞加莱群是半直积，暂设为None
            category=SymmetryCategory.SPACETIME
        )
        
        self.proper = proper
        self.orthochronous = orthochronous
        
        # 洛伦兹子群
        self.lorentz_subgroup = LorentzGroup(proper, orthochronous)
        
        # 群名称
        if proper and orthochronous:
            self._name = "ISO⁺(1,3)"
        else:
            self._name = "ISO(1,3)"
    
    @property
    def name(self) -> str:
        return self._name
    
    def create_operation(self, params: SymmetryParameters) -> PoincareTransformation:
        """创建庞加莱变换"""
        cont_params = params.continuous_params
        lorentz_params = SymmetryParameters(continuous_params=cont_params.get('lorentz', {}))
        translation = cont_params.get('translation', np.zeros(4))
        
        lorentz = self.lorentz_subgroup.create_operation(lorentz_params)
        
        return PoincareTransformation(lorentz, translation)
    
    def generators(self) -> List[Tuple[np.ndarray, str]]:
        """
        庞加莱群生成元
        
        10个生成元：
        - 4个平移: P₀, P₁, P₂, P₃
        - 3个旋转: J₁, J₂, J₃
        - 3个boost: K₁, K₂, K₃
        
        Returns:
            List of (generator_matrix, name)
        """
        generators = []
        
        # 平移生成元 Pμ
        for mu in range(4):
            P = np.zeros((5, 5))
            P[mu, 4] = 1
            generators.append((P, f'P{mu}'))
        
        # 洛伦兹生成元（扩展到5×5）
        lorentz_gens = self.lorentz_subgroup.generators()
        
        # Boost生成元 K₁, K₂, K₃
        for i, K in enumerate(lorentz_gens[:3]):
            K_ext = np.zeros((5, 5))
            K_ext[0:4, 0:4] = K
            generators.append((K_ext, f'K{i+1}'))
        
        # 旋转生成元 J₁, J₂, J₃
        for i, J in enumerate(lorentz_gens[3:6]):
            J_ext = np.zeros((5, 5))
            J_ext[0:4, 0:4] = J
            generators.append((J_ext, f'J{i+1}'))
        
        return generators
    
    def lie_algebra(self) -> Dict[str, Any]:
        """
        庞加莱李代数
        
        [Pμ, Pν] = 0
        [Ji, P₀] = 0
        [Ji, Pj] = iεijk Pk
        [Ki, P₀] = -iPi
        [Ki, Pj] = -iδij P₀
        [Ji, Jj] = iεijk Jk
        [Ji, Kj] = iεijk Kk
        [Ki, Kj] = -iεijk Jk
        """
        return {
            'translations': {
                'commutation': '[Pμ, Pν] = 0',
                'abelian': True
            },
            'rotations': {
                'commutation': '[Ji, Jj] = iεijk Jk',
                'so3_subalgebra': True
            },
            'rotation_translation': {
                'commutation': '[Ji, Pj] = iεijk Pk'
            },
            'boost_translation': {
                'commutation': '[Ki, P₀] = -iPi, [Ki, Pj] = -iδij P₀'
            },
            'boosts': {
                'commutation': '[Ki, Kj] = -iεijk Jk'
            }
        }
    
    def conserved_quantity(self) -> str:
        """守恒量"""
        return "4-momentum, angular_momentum"
    
    def casimir_operators(self) -> Dict[str, Any]:
        """
        庞加莱群的Casimir算符
        
        C₁ = PμPμ = m² (质量平方)
        C₂ = WμWμ = -m²s(s+1) (自旋，对于m>0)
        
        Wμ = Pauli-Lubanski矢量:
        Wμ = (1/2)εμναβ Jνλ Pλ
        """
        return {
            'C1': {
                'expression': 'PμPμ',
                'interpretation': 'mass_squared',
                'eigenvalues': 'm² ≥ 0'
            },
            'C2': {
                'expression': 'WμWμ',
                'interpretation': 'spin_squared',
                'eigenvalues': '-m²s(s+1) for m>0, λ² for m=0'
            }
        }


# -----------------------------------------------------------------------------
# 3. Pauli-Lubanski矢量
# -----------------------------------------------------------------------------

class PauliLubanskiVector:
    """
    Pauli-Lubanski伪矢量
    
    Wμ = (1/2)εμναβ Jνλ Pλ
    
    用于定义粒子的自旋
    """
    
    @staticmethod
    def compute(momentum: FourVector, 
                angular_momentum: np.ndarray) -> FourVector:
        """
        计算Pauli-Lubanski矢量
        
        Parameters:
            momentum: 四动量 Pμ
            angular_momentum: 角动量张量 Mμν (4×4反对称)
        """
        # Levi-Civita符号
        epsilon = np.zeros((4, 4, 4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        perm = [i, j, k, l]
                        # 计算排列的符号
                        epsilon[i, j, k, l] = PauliLubanskiVector._levi_civita(perm)
        
        # Wμ = (1/2) εμναβ Mνλ Pλ
        W = np.zeros(4)
        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    for beta in range(4):
                        W[mu] += (1/2) * epsilon[mu, nu, alpha, beta] *                                  angular_momentum[nu, alpha] *                                  momentum.components[beta]
        
        return FourVector(W)
    
    @staticmethod
    def _levi_civita(perm: List[int]) -> int:
        """计算Levi-Civita符号"""
        from itertools import permutations
        
        if len(set(perm)) < 4:
            return 0
        
        # 计算逆序数
        inversions = 0
        for i in range(4):
            for j in range(i+1, 4):
                if perm[i] > perm[j]:
                    inversions += 1
        
        return 1 if inversions % 2 == 0 else -1


# -----------------------------------------------------------------------------
# 4. 单粒子态分类
# -----------------------------------------------------------------------------

class ParticleClassification:
    """
    庞加莱群不可约表示分类
    
    基于Casimir算符本征值分类单粒子态
    """
    
    @staticmethod
    def classify_by_mass_spin(mass: float, spin: float) -> Dict[str, Any]:
        """
        根据质量和自旋分类粒子
        
        Parameters:
            mass: 质量 (m ≥ 0)
            spin: 自旋 (s = 0, 1/2, 1, 3/2, ...)
        
        Returns:
            分类信息
        """
        classification = {
            'mass': mass,
            'spin': spin,
            'casimir_values': {
                'C1': mass**2,
                'C2': -mass**2 * spin * (spin + 1) if mass > 0 else None
            }
        }
        
        # 确定表示类型
        if mass > 0:
            classification['type'] = 'massive'
            classification['little_group'] = 'SO(3)'
            classification['degrees_of_freedom'] = int(2 * spin + 1)
            classification['helicity_range'] = [-spin, spin]
            
            # 物理示例
            if spin == 0:
                classification['examples'] = ['Higgs boson', 'pion']
            elif spin == 0.5:
                classification['examples'] = ['electron', 'quark', 'neutrino (if massive)']
            elif spin == 1:
                classification['examples'] = ['W/Z bosons', 'rho meson']
            elif spin == 1.5:
                classification['examples'] = ['Delta baryon']
            elif spin == 2:
                classification['examples'] = ['graviton (hypothetical)']
        
        elif mass == 0:
            classification['type'] = 'massless'
            classification['little_group'] = 'ISO(2)'
            classification['degrees_of_freedom'] = 2  # 对于非零自旋
            classification['helicity_values'] = [spin, -spin]
            
            # 物理示例
            if spin == 0:
                classification['examples'] = ['massless scalar (hypothetical)']
            elif spin == 0.5:
                classification['examples'] = ['neutrino (if massless)']
            elif spin == 1:
                classification['examples'] = ['photon', 'gluon']
            elif spin == 2:
                classification['examples'] = ['graviton']
        
        else:  # mass < 0 (tachyon)
            classification['type'] = 'tachyonic'
            classification['physical'] = False
            classification['examples'] = ['No physical examples']
        
        return classification
    
    @staticmethod
    def little_group(momentum: FourVector) -> str:
        """
        确定给定动量的小群
        
        小群：保持四动量不变的子群
        
        - p² > 0 (有质量): SO(3)
        - p² = 0 (无质量): ISO(2) (欧几里得群)
        - p² < 0 (快子): SO(1,2)
        """
        p_sq = momentum.minkowski_norm_squared()
        
        if p_sq > 0:
            return "SO(3)"
        elif abs(p_sq) < 1e-10:
            return "ISO(2)"
        else:
            return "SO(1,2)"
    
    @staticmethod
    def standard_momentum(mass: float, spin: float) -> FourVector:
        """
        标准动量（参考系动量）
        
        用于定义小群的表示
        """
        if mass > 0:
            # 有质量粒子：静止系
            return FourVector.from_rest_frame(mass)
        else:
            # 无质量粒子：沿z轴运动
            return FourVector.momentum(1.0, 0, 0, 1.0)


# -----------------------------------------------------------------------------
# 5. Wigner旋转
# -----------------------------------------------------------------------------

class WignerRotation:
    """
    Wigner旋转
    
    在不同参考系观察粒子自旋时的旋转效应
    """
    
    @staticmethod
    def compute(boost1: LorentzTransformation, 
                boost2: LorentzTransformation,
                momentum: FourVector) -> Tuple[np.ndarray, float]:
        """
        计算Wigner旋转
        
        当对粒子进行连续boost时，自旋会额外旋转
        
        R_W = L(p')⁻¹ L₂ L₁ L(p)
        
        其中 L(p) 是将标准动量 boost 到 p 的变换
        
        Returns:
            (旋转矩阵, 旋转角)
        """
        # 计算组合boost
        combined = boost2.compose(boost1)
        
        # 简化：直接计算旋转角
        # 实际需要更复杂的实现
        
        rotation_angle = WignerRotation._compute_angle(boost1, boost2, momentum)
        
        # 旋转轴（假设绕z轴）
        axis = np.array([0, 0, 1])
        
        # Rodrigues公式构造旋转矩阵
        R = LorentzTransformation.rotation(axis, rotation_angle).matrix[1:4, 1:4]
        
        return R, rotation_angle
    
    @staticmethod
    def _compute_angle(boost1: LorentzTransformation,
                      boost2: LorentzTransformation,
                      momentum: FourVector) -> float:
        """计算旋转角"""
        # 提取速度参数
        beta1 = -boost1.matrix[0, 1] / boost1.matrix[0, 0]
        beta2 = -boost2.matrix[0, 1] / boost2.matrix[0, 0]
        
        gamma1 = boost1.matrix[0, 0]
        gamma2 = boost2.matrix[0, 0]
        
        # 简化的Wigner角公式（对于共线boost）
        # tan(ω) = ... 需要完整实现
        
        return 0.0  # 简化返回


# -----------------------------------------------------------------------------
# 6. 场的变换性质
# -----------------------------------------------------------------------------

class FieldTransformation:
    """
    场在庞加莱变换下的性质
    """
    
    @staticmethod
    def scalar_field_transformation(field_value: float,
                                   transformation: PoincareTransformation,
                                   position: FourVector) -> float:
        """
        标量场变换
        
        φ'(x) = φ(Λ⁻¹(x - a))
        """
        inv_transform = transformation.inverse()
        new_position = inv_transform.act_on(position)
        
        # 需要实际场值，这里返回概念
        return field_value
    
    @staticmethod
    def vector_field_transformation(field_value: np.ndarray,
                                   transformation: PoincareTransformation,
                                   position: FourVector) -> np.ndarray:
        """
        矢量场变换
        
        A'μ(x) = Λμν Aν(Λ⁻¹(x - a))
        """
        inv_transform = transformation.inverse()
        new_position = inv_transform.act_on(position)
        
        # 矢量场变换
        transformed_field = transformation.lorentz.matrix @ field_value
        
        return transformed_field
    
    @staticmethod
    def spinor_field_transformation(field_value: np.ndarray,
                                   transformation: PoincareTransformation,
                                   position: FourVector) -> np.ndarray:
        """
        Dirac旋量场变换
        
        ψ'(x) = S(Λ) ψ(Λ⁻¹(x - a))
        
        S(Λ) 是旋量表示
        """
        # 需要构造S(Λ)矩阵
        # 这是Dirac表示
        pass


# -----------------------------------------------------------------------------
# 7. 衰变和散射的运动学
# -----------------------------------------------------------------------------

class RelativisticScattering:
    """相对论散射运动学"""
    
    @staticmethod
    def center_of_mass_energy(p1: FourVector, p2: FourVector) -> float:
        """
        质心系能量 √s
        
        s = (p1 + p2)²
        """
        total = p1 + p2
        return total.minkowski_norm()
    
    @staticmethod
    def mandelstam_variables(p1: FourVector, p2: FourVector,
                           p3: FourVector, p4: FourVector) -> Dict[str, float]:
        """
        Mandelstam变量
        
        s = (p1 + p2)² = (p3 + p4)²
        t = (p1 - p3)² = (p2 - p4)²
        u = (p1 - p4)² = (p2 - p3)²
        
        约束: s + t + u = m1² + m2² + m3² + m4²
        """
        s = (p1 + p2).minkowski_norm_squared()
        t = (p1 - p3).minkowski_norm_squared()
        u = (p1 - p4).minkowski_norm_squared()
        
        return {
            's': s,
            't': t,
            'u': u,
            'check': s + t + u - (p1.minkowski_norm_squared() + 
                                  p2.minkowski_norm_squared() + 
                                  p3.minkowski_norm_squared() + 
                                  p4.minkowski_norm_squared())
        }
    
    @staticmethod
    def two_body_decay(m_parent: float, 
                      m1: float, 
                      m2: float) -> Dict[str, float]:
        """
        两体衰变运动学
        
        在静止系中:
        E1 = (m² + m1² - m2²) / (2m)
        E2 = (m² - m1² + m2²) / (2m)
        |p| = √[λ(m², m1², m2²)] / (2m)
        
        λ(x,y,z) = x² + y² + z² - 2xy - 2xz - 2yz (Källén函数)
        """
        m = m_parent
        m1_sq, m2_sq = m1**2, m2**2
        m_sq = m**2
        
        # 能量
        E1 = (m_sq + m1_sq - m2_sq) / (2 * m)
        E2 = (m_sq - m1_sq + m2_sq) / (2 * m)
        
        # 动量大小（Källén函数）
        lambda_func = m_sq**2 + m1_sq**2 + m2_sq**2 -                       2*m_sq*m1_sq - 2*m_sq*m2_sq - 2*m1_sq*m2_sq
        
        if lambda_func < 0:
            raise ValueError("衰变运动学不允许")
        
        p_mag = np.sqrt(lambda_func) / (2 * m)
        
        return {
            'E1': E1,
            'E2': E2,
            'p_magnitude': p_mag,
            'momentum_frame': 'rest_frame'
        }
    
    @staticmethod
    def cross_section_kinematics(s: float, 
                                m1: float, 
                                m2: float,
                                m3: float,
                                m4: float) -> Dict[str, float]:
        """
        散射截面的运动学因子
        
        质心系中的动量
        """
        sqrt_s = np.sqrt(s)
        
        # 初态动量
        lambda_i = s**2 + m1**4 + m2**4 - 2*s*m1**2 - 2*s*m2**2 - 2*m1**2*m2**2
        
        if lambda_i < 0:
            raise ValueError("初态运动学不允许")
        
        p_i = np.sqrt(lambda_i) / (2 * sqrt_s)
        
        # 末态动量
        lambda_f = s**2 + m3**4 + m4**4 - 2*s*m3**2 - 2*s*m4**2 - 2*m3**2*m4**2
        
        if lambda_f < 0:
            raise ValueError("末态运动学不允许")
        
        p_f = np.sqrt(lambda_f) / (2 * sqrt_s)
        
        # 通量因子
        flux = 4 * p_i * sqrt_s
        
        return {
            'p_initial': p_i,
            'p_final': p_f,
            'flux_factor': flux,
            'phase_space': p_f / (8 * np.pi * sqrt_s)
        }
