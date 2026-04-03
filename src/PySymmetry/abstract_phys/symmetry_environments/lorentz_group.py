"""
洛伦兹群模块

实现狭义相对论的时空对称性：
- 洛伦兹变换（boost + 旋转）
- 洛伦兹群的表示
- 四矢量、张量变换
- 相对论运动学
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod

from .base import PhysicalSymmetry, SymmetryOperation, SymmetryParameters, SymmetryType, SymmetryCategory



# -----------------------------------------------------------------------------
# 1. 时空类型定义
# -----------------------------------------------------------------------------

class MetricSignature(Enum):
    """度规符号约定"""
    PLUS_MINUS = (+1, -1, -1, -1)  # 粒子物理常用 (+---)
    MINUS_PLUS = (-1, +1, +1, +1)  # 相对论常用 (-+++)


@dataclass
class FourVector:
    """
    四矢量
    
    在自然单位制下：c = 1
    """
    components: np.ndarray  # 4维数组
    metric: MetricSignature = MetricSignature.PLUS_MINUS
    
    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=float)
        if len(self.components) != 4:
            raise ValueError("四矢量必须有4个分量")
    
    @property
    def t(self) -> float:
        """时间分量"""
        return self.components[0]
    
    @property
    def spatial(self) -> np.ndarray:
        """空间分量"""
        return self.components[1:4]
    
    def minkowski_norm_squared(self) -> float:
        """
        闵可夫斯基模方
        
        s² = t² - x² - y² - z² (for +---)
        """
        if self.metric == MetricSignature.PLUS_MINUS:
            return self.t**2 - np.dot(self.spatial, self.spatial)
        else:
            return -self.t**2 + np.dot(self.spatial, self.spatial)
    
    def minkowski_norm(self) -> float:
        """闵可夫斯基模"""
        return np.sqrt(abs(self.minkowski_norm_squared()))
    
    def is_timelike(self) -> bool:
        """是否类时"""
        return self.minkowski_norm_squared() > 0
    
    def is_spacelike(self) -> bool:
        """是否类空"""
        return self.minkowski_norm_squared() < 0
    
    def is_lightlike(self) -> bool:
        """是否类光（光锥上）"""
        return abs(self.minkowski_norm_squared()) < 1e-10
    
    def proper_time(self) -> float:
        """固有时间 τ = s/c"""
        if self.is_timelike():
            return self.minkowski_norm()
        else:
            raise ValueError("只有类时矢量有固有时间")
    
    def gamma(self) -> float:
        """
        洛伦兹因子 γ = dt/dτ
        
        对于四速度：γ = 1/√(1-v²)
        """
        if self.t > 0:
            v_squared = np.dot(self.spatial, self.spatial) / self.t**2
            if v_squared < 1:
                return 1.0 / np.sqrt(1 - v_squared)
        return 1.0
    
    @classmethod
    def from_rest_frame(cls, mass: float) -> 'FourVector':
        """从静止系创建四动量"""
        return cls(np.array([mass, 0, 0, 0]))
    
    @classmethod
    def position(cls, t: float, x: float, y: float, z: float) -> 'FourVector':
        """创建时空坐标四矢量"""
        return cls(np.array([t, x, y, z]))
    
    @classmethod
    def momentum(cls, E: float, px: float, py: float, pz: float) -> 'FourVector':
        """创建四动量"""
        return cls(np.array([E, px, py, pz]))
    
    def __add__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(self.components + other.components, self.metric)
    
    def __sub__(self, other: 'FourVector') -> 'FourVector':
        return FourVector(self.components - other.components, self.metric)
    
    def __mul__(self, scalar: float) -> 'FourVector':
        return FourVector(self.components * scalar, self.metric)
    
    def __rmul__(self, scalar: float) -> 'FourVector':
        return self.__mul__(scalar)
    
    def dot(self, other: 'FourVector') -> float:
        """闵可夫斯基内积"""
        if self.metric == MetricSignature.PLUS_MINUS:
            return (self.t * other.t - 
                   self.spatial[0] * other.spatial[0] -
                   self.spatial[1] * other.spatial[1] -
                   self.spatial[2] * other.spatial[2])
        else:
            return (-self.t * other.t + 
                   np.dot(self.spatial, other.spatial))


# -----------------------------------------------------------------------------
# 2. 洛伦兹变换
# -----------------------------------------------------------------------------

class LorentzTransformation(SymmetryOperation):
    """
    洛伦兹变换
    
    Λ: xμ → Λμν xν
    
    保持闵可夫斯基度规不变：ΛᵀηΛ = η
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Parameters:
            matrix: 4x4洛伦兹变换矩阵
        """
        super().__init__()
        if matrix.shape != (4, 4):
            raise ValueError("洛伦兹变换矩阵必须是4x4")
        
        self.matrix = matrix.astype(float)
        self._validate()
    
    def _validate(self):
        """验证是否是合法的洛伦兹变换"""
        eta = self._get_minkowski_metric()
        
        # 检查 ΛᵀηΛ = η
        product = self.matrix.T @ eta @ self.matrix
        
        if not np.allclose(product, eta, atol=1e-10):
            raise ValueError("不是合法的洛伦兹变换")
    
    def _get_minkowski_metric(self) -> np.ndarray:
        """闵可夫斯基度规张量"""
        return np.diag([1, -1, -1, -1])
    
    @property
    def group(self) -> Any:
        """所属对称群
        
        返回字符串标识符，因为 LorentzGroup 不是标准的 Group 实现
        """
        return "Lorentz"
    
    @property
    def is_continuous(self) -> bool:
        return True
    
    @property
    def det(self) -> float:
        """行列式：±1"""
        return np.linalg.det(self.matrix)
    
    @property
    def is_proper(self) -> bool:
        """是否真洛伦兹变换 (det = +1)"""
        return abs(self.det - 1) < 1e-10
    
    @property
    def is_orthochronous(self) -> bool:
        """是否正时 (Λ₀₀ ≥ 1)"""
        return self.matrix[0, 0] >= 1
    
    def compose(self, other: 'LorentzTransformation') -> 'LorentzTransformation':
        """组合两个洛伦兹变换"""
        return LorentzTransformation(self.matrix @ other.matrix)
    
    def inverse(self) -> 'LorentzTransformation':
        """逆变换：Λ⁻¹ = ηΛᵀη"""
        eta = self._get_minkowski_metric()
        inv_matrix = eta @ self.matrix.T @ eta
        return LorentzTransformation(inv_matrix)
    
    def act_on(self, four_vector: FourVector) -> FourVector:
        """作用于四矢量"""
        new_components = self.matrix @ four_vector.components
        return FourVector(new_components, four_vector.metric)
    
    def transformation_type(self) -> str:
        """判断变换类型"""
        if self.is_proper and self.is_orthochronous:
            return "proper orthochronous"  # SO⁺(1,3)
        elif self.is_proper:
            return "proper"
        elif self.is_orthochronous:
            return "orthochronous"
        else:
            return "general"
    
    # -------------------------------------------------------------------------
    # 静态工厂方法：常用洛伦兹变换
    # -------------------------------------------------------------------------
    
    @classmethod
    def identity(cls) -> 'LorentzTransformation':
        """单位元"""
        return cls(np.eye(4))
    
    @classmethod
    def boost_x(cls, beta: float) -> 'LorentzTransformation':
        """
        沿x方向的boost
        
        γ = 1/√(1-β²), β = v/c
        
        Λ = [γ      -γβ    0    0  ]
            [-γβ     γ     0    0  ]
            [0       0     1    0  ]
            [0       0     0    1  ]
        """
        if abs(beta) >= 1:
            raise ValueError(f"β={beta} 必须 < 1 (光速不可达)")
        
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        matrix = np.array([
            [gamma,      -gamma * beta,  0,  0],
            [-gamma * beta,  gamma,      0,  0],
            [0,              0,          1,  0],
            [0,              0,          0,  1]
        ])
        
        return cls(matrix)
    
    @classmethod
    def boost_y(cls, beta: float) -> 'LorentzTransformation':
        """沿y方向的boost"""
        if abs(beta) >= 1:
            raise ValueError(f"β={beta} 必须 < 1")
        
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        matrix = np.array([
            [gamma,  0, -gamma * beta,  0],
            [0,      1,             0,  0],
            [-gamma * beta, 0, gamma,  0],
            [0,      0,             0,  1]
        ])
        
        return cls(matrix)
    
    @classmethod
    def boost_z(cls, beta: float) -> 'LorentzTransformation':
        """沿z方向的boost"""
        if abs(beta) >= 1:
            raise ValueError(f"β={beta} 必须 < 1")
        
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        matrix = np.array([
            [gamma,  0,  0, -gamma * beta],
            [0,      1,  0,             0],
            [0,      0,  1,             0],
            [-gamma * beta, 0, 0, gamma]
        ])
        
        return cls(matrix)
    
    @classmethod
    def boost(cls, velocity: np.ndarray) -> 'LorentzTransformation':
        """
        沿任意方向的boost
        
        使用 (+---) 度规约定：ds² = dt² - dx² - dy² - dz²
        
        沿方向 n 的 boost 矩阵：
        Λ = | γ        -γβ n_i |
            | -γβ n_j  δ_ij + (γ-1) n_i n_j |
        
        Parameters:
            velocity: 三维速度矢量 [vx, vy, vz] (单位: c)
        """
        velocity = np.asarray(velocity, dtype=float)
        v_squared = np.dot(velocity, velocity)
        
        if v_squared >= 1:
            raise ValueError(f"速度大小 {np.sqrt(v_squared)}c 必须 < c")
        
        if v_squared < 1e-10:
            return cls.identity()
        
        gamma = 1.0 / np.sqrt(1 - v_squared)
        beta = np.sqrt(v_squared)
        
        n = velocity / beta
        
        matrix = np.zeros((4, 4))
        matrix[0, 0] = gamma
        matrix[0, 1:4] = -gamma * beta * n
        matrix[1:4, 0] = -gamma * beta * n
        matrix[1:4, 1:4] = np.eye(3) + (gamma - 1) * np.outer(n, n)
        
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, theta: float) -> 'LorentzTransformation':
        """绕x轴旋转"""
        c, s = np.cos(theta), np.sin(theta)
        
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ])
        
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, theta: float) -> 'LorentzTransformation':
        """绕y轴旋转"""
        c, s = np.cos(theta), np.sin(theta)
        
        matrix = np.array([
            [1, 0, 0, 0],
            [0, c, 0, s],
            [0, 0, 1, 0],
            [0, -s, 0, c]
        ])
        
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, theta: float) -> 'LorentzTransformation':
        """绕z轴旋转"""
        c, s = np.cos(theta), np.sin(theta)
        
        matrix = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
        
        return cls(matrix)
    
    @classmethod
    def rotation(cls, axis: np.ndarray, theta: float) -> 'LorentzTransformation':
        """绕任意轴旋转"""
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues公式
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        matrix = np.eye(4)
        matrix[1:4, 1:4] = R
        
        return cls(matrix)
    
    @classmethod
    def parity_inversion(cls) -> 'LorentzTransformation':
        """空间反演 P"""
        matrix = np.diag([1, -1, -1, -1])
        return cls(matrix)
    
    @classmethod
    def time_reversal(cls) -> 'LorentzTransformation':
        """时间反演 T"""
        matrix = np.diag([-1, 1, 1, 1])
        return cls(matrix)


# -----------------------------------------------------------------------------
# 3. 洛伦兹群
# -----------------------------------------------------------------------------

class LorentzGroup(PhysicalSymmetry):
    """
    洛伦兹群 O(1,3)
    
    保持闵可夫斯基度规不变的变换群
    """
    
    def __init__(self, proper: bool = True, orthochronous: bool = True):
        """
        Parameters:
            proper: 是否限制为真洛伦兹变换 (det = +1)
            orthochronous: 是否限制为正时变换 (Λ₀₀ ≥ 1)
        """
        super().__init__(
            symmetry_type=SymmetryType.LORENTZ,
            group=None,  # 稍后设置
            category=SymmetryCategory.SPACETIME
        )
        
        self.proper = proper
        self.orthochronous = orthochronous
        
        # 确定群名称
        if proper and orthochronous:
            self._name = "SO⁺(1,3)"  # 真正时洛伦兹群
        elif proper:
            self._name = "SO(1,3)"   # 真洛伦兹群
        elif orthochronous:
            self._name = "O⁺(1,3)"   # 正时洛伦兹群
        else:
            self._name = "O(1,3)"    # 完全洛伦兹群
    
    @property
    def name(self) -> str:
        return self._name
    
    def create_operation(self, params: SymmetryParameters) -> LorentzTransformation:
        """创建洛伦兹变换"""
        cont_params = params.continuous_params
        if 'boost' in cont_params:
            velocity = cont_params['boost']
            return LorentzTransformation.boost(velocity)
        elif 'boost_x' in cont_params:
            return LorentzTransformation.boost_x(cont_params['boost_x'])
        elif 'rotation' in cont_params:
            axis, theta = cont_params['rotation']
            return LorentzTransformation.rotation(axis, theta)
        elif 'matrix' in cont_params:
            return LorentzTransformation(cont_params['matrix'])
        else:
            return LorentzTransformation.identity()
    
    def generators(self) -> List[np.ndarray]:
        """
        洛伦兹群生成元
        
        6个生成元：
        - 3个boost: K₁, K₂, K₃
        - 3个旋转: J₁, J₂, J₃
        """
        # Boost生成元
        K1 = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        K2 = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        K3 = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ])
        
        # 旋转生成元
        J1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0]
        ])
        
        J2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0]
        ])
        
        J3 = np.array([
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ])
        
        return [K1, K2, K3, J1, J2, J3]
    
    def lie_algebra(self) -> Dict[str, Any]:
        """
        洛伦兹李代数
        
        [Ji, Jj] = iεijk Jk
        [Ji, Kj] = iεijk Kk
        [Ki, Kj] = -iεijk Jk
        """
        return {
            'rotations': {
                'commutation': '[Ji, Jj] = iεijk Jk',
                'structure_constants': 'εijk'
            },
            'boost_rotation': {
                'commutation': '[Ji, Kj] = iεijk Kk'
            },
            'boosts': {
                'commutation': '[Ki, Kj] = -iεijk Jk'
            }
        }
    
    def conserved_quantity(self) -> str:
        """守恒量"""
        return "angular_momentum"  # 相对论角动量


# -----------------------------------------------------------------------------
# 4. 旋量表示
# -----------------------------------------------------------------------------

class LorentzSpinor:
    """
    洛伦兹群的旋量表示
    
    Weyl旋量和Dirac旋量
    """
    
    @staticmethod
    def weyl_spinor_transformation(spinor: np.ndarray, 
                                   transformation: LorentzTransformation) -> np.ndarray:
        """
        Weyl旋量变换
        
        左手: ψL → exp(-θ·σ/2 - iφ·σ/2) ψL
        右手: ψR → exp(-θ·σ/2 + iφ·σ/2) ψR
        
        其中θ是旋转角，φ是boost参数 (tanh φ = β)
        """
        # 需要提取旋转和boost参数
        pass
    
    @staticmethod
    def dirac_matrices() -> Dict[str, np.ndarray]:
        """
        Dirac矩阵（Weyl/chiral表示）
        
        γ⁰ = [0  I]
             [I  0]
        
        γi = [0   σi]
             [-σi  0 ]
        """
        # Pauli矩阵
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        I2 = np.eye(2, dtype=complex)
        
        # Dirac矩阵
        gamma0 = np.block([[np.zeros((2,2)), I2], [I2, np.zeros((2,2))]])
        gamma1 = np.block([[np.zeros((2,2)), sigma_x], [-sigma_x, np.zeros((2,2))]])
        gamma2 = np.block([[np.zeros((2,2)), sigma_y], [-sigma_y, np.zeros((2,2))]])
        gamma3 = np.block([[np.zeros((2,2)), sigma_z], [-sigma_z, np.zeros((2,2))]])
        
        # γ⁵ = iγ⁰γ¹γ²γ³ (Weyl表示)
        gamma5 = 1j * gamma0 @ gamma1 @ gamma2 @ gamma3
        
        # 验证 gamma5² = 1 (Weyl表示)
        if not np.allclose(gamma5 @ gamma5, np.eye(4)):
            # 如果不满足，换用标准约定
            gamma5 = np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=complex)
        
        return {
            'gamma0': gamma0,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma3': gamma3,
            'gamma5': gamma5
        }
    
    @staticmethod
    def dirac_spinor_transformation(spinor: np.ndarray,
                                   transformation: LorentzTransformation) -> np.ndarray:
        """
        Dirac旋量变换
        
        ψ → S(Λ) ψ
        
        S(Λ) = exp(-i/4 ωμν σμν)
        """
        pass


# -----------------------------------------------------------------------------
# 5. 相对论运动学
# -----------------------------------------------------------------------------

class RelativisticKinematics:
    """相对论运动学计算"""
    
    @staticmethod
    def velocity_addition(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        相对论速度叠加
        
        u = (v1 + v2_∥ + γv2_⊥)/(1 + v1·v2)
        
        其中 v2_∥ 和 v2_⊥ 是 v2 平行和垂直于 v1 的分量
        """
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        
        v1_sq = np.dot(v1, v1)
        
        if v1_sq < 1e-10:
            return v2
        
        gamma1 = 1.0 / np.sqrt(1 - v1_sq)
        
        # 平行和垂直分量
        v1_hat = v1 / np.sqrt(v1_sq)
        v2_parallel = np.dot(v2, v1_hat) * v1_hat
        v2_perp = v2 - v2_parallel
        
        # 叠加
        numerator = v1 + v2_parallel / gamma1 + v2_perp
        denominator = 1 + np.dot(v1, v2)
        
        return numerator / denominator
    
    @staticmethod
    def thomas_precession(velocity: np.ndarray, 
                         angular_velocity: np.ndarray,
                         dt: float) -> np.ndarray:
        """
        Thomas进动
        
        旋转角速度 ω_T = (γ²/(γ+1)) v × a
        """
        velocity = np.asarray(velocity)
        v_sq = np.dot(velocity, velocity)
        
        if v_sq < 1e-10:
            return np.zeros(3)
        
        gamma = 1.0 / np.sqrt(1 - v_sq)
        
        # 加速度
        acceleration = angular_velocity  # 简化
        
        # Thomas进动角速度
        omega_T = (gamma**2 / (gamma + 1)) * np.cross(velocity, acceleration)
        
        return omega_T
    
    @staticmethod
    def doppler_effect(frequency: float, 
                      velocity: np.ndarray,
                      direction: np.ndarray) -> float:
        """
        相对论多普勒效应
        
        f' = f * γ(1 - β·n)
        
        n: 光传播方向的单位矢量
        """
        velocity = np.asarray(velocity)
        direction = np.asarray(direction)
        direction = direction / np.linalg.norm(direction)
        
        v_sq = np.dot(velocity, velocity)
        gamma = 1.0 / np.sqrt(1 - v_sq)
        
        beta = velocity  # c=1
        doppler_factor = gamma * (1 - np.dot(beta, direction))
        
        return frequency * doppler_factor
    
    @staticmethod
    def aberration_angle(theta: float, 
                        velocity: np.ndarray,
                        direction: np.ndarray) -> float:
        """
        光行差效应
        
        cos θ' = (cos θ - β)/(1 - β cos θ)
        """
        velocity = np.asarray(velocity)
        beta = np.linalg.norm(velocity)
        
        if beta < 1e-10:
            return theta
        
        cos_theta = np.cos(theta)
        cos_theta_prime = (cos_theta - beta) / (1 - beta * cos_theta)
        
        return np.arccos(cos_theta_prime)
    
    @staticmethod
    def energy_momentum_relation(mass: float, 
                                momentum: np.ndarray) -> Tuple[float, float]:
        """
        能量-动量关系
        
        E² = p²c² + m²c⁴
        
        Returns:
            (energy, gamma)
        """
        p_sq = np.dot(momentum, momentum)
        
        # E = √(p² + m²) in natural units
        energy = np.sqrt(p_sq + mass**2)
        
        gamma = energy / mass if mass > 0 else float('inf')
        
        return energy, gamma
