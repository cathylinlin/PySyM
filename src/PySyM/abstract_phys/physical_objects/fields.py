"""
物理场实现

定义各种物理场的抽象基类和具体实现，包括标量场、矢量场、电磁场和引力场。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..symmetry_operations.base import SymmetryOperation

class Field(ABC):
    """物理场抽象基类
    
    所有物理场的基类，定义了场的基本操作接口。
    """
    
    @abstractmethod
    def evaluate(self, position: np.ndarray) -> Any:
        """在给定位置计算场的值
        
        Args:
            position: 空间位置
            
        Returns:
            场在该位置的值
        """
        pass
    
    @abstractmethod
    def gradient(self, position: np.ndarray) -> Any:
        """计算场的梯度
        
        Args:
            position: 空间位置
            
        Returns:
            场在该位置的梯度
        """
        pass
    
    @abstractmethod
    def get_energy_density(self, position: np.ndarray) -> float:
        """获取能量密度
        
        Args:
            position: 空间位置
            
        Returns:
            场在该位置的能量密度
        """
        pass
    
    def transform(self, operation: 'SymmetryOperation') -> 'Field':
        """在对称操作下变换场"""
        return self
    
    def is_invariant_under(self, operation: 'SymmetryOperation') -> bool:
        """检查场在某对称操作下是否不变"""
        return True


class ScalarField(Field):
    """标量场
    
    描述单一数值随空间时间变化的场，如 Klein-Gordon 场。
    """
    
    def __init__(self, field_function: Callable[[np.ndarray], float]):
        self._field_function = field_function
    
    def evaluate(self, position: np.ndarray) -> float:
        return float(self._field_function(position))
    
    def gradient(self, position: np.ndarray) -> np.ndarray:
        epsilon = 1e-6
        dim = len(position)
        grad = np.zeros(dim)
        for i in range(dim):
            pos_plus = position.copy()
            pos_plus[i] += epsilon
            pos_minus = position.copy()
            pos_minus[i] -= epsilon
            grad[i] = (self.evaluate(pos_plus) - self.evaluate(pos_minus)) / (2 * epsilon)
        return grad
    
    def get_energy_density(self, position: np.ndarray) -> float:
        grad = self.gradient(position)
        return float(0.5 * np.sum(grad**2))
    
    def laplacian(self, position: np.ndarray) -> float:
        """计算拉普拉斯算子 ∇²φ"""
        epsilon = 1e-6
        phi = self.evaluate(position)
        laplacian = 0.0
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_plus[i] += epsilon
            pos_minus = position.copy()
            pos_minus[i] -= epsilon
            laplacian += (self.evaluate(pos_plus) - 2 * phi + self.evaluate(pos_minus)) / (epsilon ** 2)
        return float(laplacian)


class VectorField(Field):
    """矢量场
    
    描述向量随空间时间变化的场，如电磁场、流体速度场。
    """
    
    def __init__(self, field_function: Callable[[np.ndarray], np.ndarray]):
        self._field_function = field_function
    
    def evaluate(self, position: np.ndarray) -> np.ndarray:
        result = self._field_function(position)
        return np.asarray(result)
    
    def gradient(self, position: np.ndarray) -> np.ndarray:
        epsilon = 1e-6
        value = self.evaluate(position)
        if value.ndim == 0:
            value = np.array([value])
        dim = len(position)
        grad = np.zeros((len(value), dim))
        for i in range(dim):
            pos_plus = position.copy()
            pos_plus[i] += epsilon
            pos_minus = position.copy()
            pos_minus[i] -= epsilon
            grad[:, i] = (self.evaluate(pos_plus) - self.evaluate(pos_minus)) / (2 * epsilon)
        return grad
    
    def divergence(self, position: np.ndarray) -> float:
        """计算散度 ∇·V"""
        grad = self.gradient(position)
        return float(np.trace(grad))
    
    def curl(self, position: np.ndarray) -> np.ndarray:
        """计算旋度 ∇×V"""
        epsilon = 1e-6
        result = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            pos_j_plus = position.copy()
            pos_j_plus[j] += epsilon
            pos_j_minus = position.copy()
            pos_j_minus[j] -= epsilon
            pos_k_plus = position.copy()
            pos_k_plus[k] += epsilon
            pos_k_minus = position.copy()
            pos_k_minus[k] -= epsilon
            result[i] = (self.evaluate(pos_j_plus)[k] - self.evaluate(pos_j_minus)[k] - 
                        self.evaluate(pos_k_plus)[j] + self.evaluate(pos_k_minus)[j]) / (2 * epsilon)
        return result
    
    def get_energy_density(self, position: np.ndarray) -> float:
        grad = self.gradient(position)
        return float(0.5 * np.sum(grad**2))


class ElectromagneticField(VectorField):
    """电磁场
    
    描述电磁相互作用的场，包括电场和磁场。
    """
    
    def __init__(self, 
                 electric_field_function: Callable[[np.ndarray], np.ndarray],
                 magnetic_field_function: Callable[[np.ndarray], np.ndarray]):
        self._electric_field = electric_field_function
        self._magnetic_field = magnetic_field_function
        super().__init__(lambda x: np.concatenate([self._electric_field(x), self._magnetic_field(x)]))
    
    def evaluate(self, position: np.ndarray) -> np.ndarray:
        E = np.asarray(self._electric_field(position))
        B = np.asarray(self._magnetic_field(position))
        return np.concatenate([E, B])
    
    def get_electric_field(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(self._electric_field(position))
    
    def get_magnetic_field(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(self._magnetic_field(position))
    
    def get_energy_density(self, position: np.ndarray) -> float:
        E = self.get_electric_field(position)
        B = self.get_magnetic_field(position)
        epsilon_0 = 8.854e-12
        mu_0 = 4 * np.pi * 1e-7
        return float(0.5 * (epsilon_0 * np.sum(E**2) + np.sum(B**2) / mu_0))
    
    def poynting_vector(self, position: np.ndarray) -> np.ndarray:
        """计算坡印廷矢量 S = E × B / μ₀"""
        E = self.get_electric_field(position)
        B = self.get_magnetic_field(position)
        mu_0 = 4 * np.pi * 1e-7
        return np.cross(E, B) / mu_0


class GravitationalField(ScalarField):
    """引力场
    
    描述引力相互作用的场。
    """
    
    def __init__(self, mass_distribution: list):
        self._mass_distribution = mass_distribution
    
    def evaluate(self, position: np.ndarray) -> float:
        G = 6.674e-11
        potential = 0.0
        for mass_pos, mass_value in self._mass_distribution:
            r = np.linalg.norm(position - mass_pos)
            if r > 0:
                potential -= G * mass_value / r
        return float(potential)
    
    def get_field_strength(self, position: np.ndarray) -> np.ndarray:
        return -self.gradient(position)
    
    def tidal_force(self, position: np.ndarray) -> np.ndarray:
        """计算潮汐力（二阶导数）"""
        epsilon = 1e-6
        dim = len(position)
        tidal = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                pos_pp = position.copy()
                pos_pp[i] += epsilon
                pos_pp[j] += epsilon
                pos_pm = position.copy()
                pos_pm[i] += epsilon
                pos_pm[j] -= epsilon
                pos_mp = position.copy()
                pos_mp[i] -= epsilon
                pos_mp[j] += epsilon
                pos_mm = position.copy()
                pos_mm[i] -= epsilon
                pos_mm[j] -= epsilon
                tidal[i, j] = (self.evaluate(pos_pp) - self.evaluate(pos_pm) - 
                              self.evaluate(pos_mp) + self.evaluate(pos_mm)) / (4 * epsilon ** 2)
        return tidal


class YangMillsField(VectorField):
    """杨-米尔斯场
    
    描述非阿贝尔规范相互作用的场，如胶子场。
    """
    
    def __init__(self, 
                 gauge_group: str,
                 field_function: Callable[[np.ndarray], np.ndarray],
                 coupling_constant: float = 1.0):
        super().__init__(field_function)
        self._gauge_group = gauge_group
        self._coupling = coupling_constant
    
    def get_gauge_group(self) -> str:
        return self._gauge_group
    
    def get_coupling_constant(self) -> float:
        return self._coupling
    
    def field_strength(self, position: np.ndarray) -> np.ndarray:
        """计算场强张量
        
        F_μν = ∂_μ A_ν - ∂_ν A_μ + ig[A_μ, A_ν]
        
        其中 A_μ = A_μ^a T^a 是规范场（T^a 是生成元）
        协变导数 D_μ = ∂_μ + ig A_μ
        
        Args:
            position: 空间位置
            
        Returns:
            场强张量 F[μ, ν]，反对称
        """
        dim = len(position)
        A = self.evaluate(position)
        
        gauge_dim = 1
        if self._gauge_group in ["SU(2)", "SU2"]:
            gauge_dim = 2
        elif self._gauge_group in ["SU(3)", "SU3"]:
            gauge_dim = 3
        
        if hasattr(A, '__len__') and not isinstance(A, (float, complex)):
            A_flat = np.asarray(A).flatten()
            if len(A_flat) == dim * gauge_dim:
                A_matrix = A_flat.reshape(dim, gauge_dim)
            else:
                A_matrix = A_flat
        else:
            A_matrix = np.asarray(A)
        
        epsilon = 1e-6
        F = np.zeros((dim, dim), dtype=complex)
        
        for mu in range(dim):
            for nu in range(mu + 1, dim):
                pos_mu_plus = position.copy()
                pos_mu_plus[mu] += epsilon
                A_mu_plus = np.asarray(self.evaluate(pos_mu_plus)).flatten()
                
                pos_nu_plus = position.copy()
                pos_nu_plus[nu] += epsilon
                A_nu_plus = np.asarray(self.evaluate(pos_nu_plus)).flatten()
                
                pos_mu_minus = position.copy()
                pos_mu_minus[mu] -= epsilon
                A_mu_minus = np.asarray(self.evaluate(pos_mu_minus)).flatten()
                
                pos_nu_minus = position.copy()
                pos_nu_minus[nu] -= epsilon
                A_nu_minus = np.asarray(self.evaluate(pos_nu_minus)).flatten()
                
                dA_nu_dmu = (A_mu_plus[nu::dim][:gauge_dim] if len(A_mu_plus) >= dim * gauge_dim else A_mu_plus[nu]) - \
                            (A_nu_minus[nu::dim][:gauge_dim] if len(A_nu_minus) >= dim * gauge_dim else A_nu_minus[nu])
                dA_mu_dnu = (A_nu_plus[mu::dim][:gauge_dim] if len(A_nu_plus) >= dim * gauge_dim else A_nu_plus[mu]) - \
                            (A_mu_minus[mu::dim][:gauge_dim] if len(A_mu_minus) >= dim * gauge_dim else A_mu_minus[mu])
                
                dA_nu_dmu /= epsilon
                dA_mu_dnu /= epsilon
                
                if len(A_matrix) >= dim * gauge_dim:
                    A_mu_arr = A_matrix[mu, :gauge_dim]
                    A_nu_arr = A_matrix[nu, :gauge_dim]
                else:
                    A_mu_arr = A_matrix[mu]
                    A_nu_arr = A_matrix[nu]
                
                commutator = np.dot(A_mu_arr, A_nu_arr) - np.dot(A_nu_arr, A_mu_arr) if hasattr(A_mu_arr, '__len__') and len(A_mu_arr) > 1 else 0
                
                F[mu, nu] = dA_nu_dmu - dA_mu_dnu + 1j * self._coupling * commutator
                F[nu, mu] = -F[mu, nu]
        
        return F
    
    def yang_mills_lagrangian_density(self, position: np.ndarray) -> float:
        """计算杨-米尔斯拉格朗日密度 L = -1/4 Tr(F_μν F^μν)"""
        F = self.field_strength(position)
        return float(-0.25 * np.trace(F @ F.T))


class SpinorField(Field):
    """旋量场
    
    描述自旋1/2粒子的场，如电子场、中微子场。
    
    Dirac 旋量满足 Dirac 方程：(iγ^μ ∂_μ - m)ψ = 0
    """
    
    def __init__(self, 
                 spinor_function: Callable[[np.ndarray], np.ndarray],
                 mass: float = 0.0):
        self._spinor_function = spinor_function
        self._mass = mass
        self._gamma_matrices = self._compute_gamma_matrices()
    
    def evaluate(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(self._spinor_function(position))
    
    def gradient(self, position: np.ndarray) -> np.ndarray:
        return self._compute_gradient(position)
    
    def get_energy_density(self, position: np.ndarray) -> float:
        psi = self.evaluate(position)
        psi_adjoint = self.dirac_adjoint(psi)
        return float(np.real(np.vdot(psi_adjoint, psi)))
    
    def dirac_adjoint(self, psi: np.ndarray) -> np.ndarray:
        """计算 Dirac 共轭 ψ̄ = ψ† γ⁰
        
        ψ̄ = ψ^*T γ⁰
        
        使用 Weyl/chiral 表示：γ⁰ = [[0, I], [I, 0]]
        
        Args:
            psi: Dirac 旋量（4分量）
            
        Returns:
            Dirac 共轭旋量
        """
        psi = np.asarray(psi)
        gamma_0 = self._gamma_matrices[0]
        return np.conj(psi).T @ gamma_0
    
    def dirac_equation_residual(self, position: np.ndarray, time_derivative: float = 0.0) -> np.ndarray:
        """计算 Dirac 方程残差
        
        (iγ^μ ∂_μ - m)ψ = 0
        
        对于 4D 时空，∂_μ = (∂_t, ∂_x, ∂_y, ∂_z)
        
        Args:
            position: 位置 (t, x, y, z) 或 (x, y, z)
            time_derivative: 时间导数 ∂_t ψ
            
        Returns:
            Dirac 方程的残差向量，应为零表示满足方程
        """
        psi = self.evaluate(position)
        
        spatial_gradient = self._compute_gradient(position)
        
        if len(position) == 4:
            dirac_operator = (1j * self._gamma_matrices[0] * time_derivative + 
                           1j * sum(self._gamma_matrices[i+1] * spatial_gradient[i] 
                                   for i in range(3)) 
                           - self._mass * self._gamma_matrices[0])
        else:
            dirac_operator = (1j * sum(self._gamma_matrices[i] * spatial_gradient[i-1] 
                                      for i in range(1, len(position)+1)) 
                            - self._mass * self._gamma_matrices[0])
        
        return np.dot(dirac_operator, psi)
    
    @classmethod
    def _compute_gamma_matrices(cls) -> list:
        """计算并缓存 Dirac γ 矩阵
        
        Weyl/chiral 表示：
        γ⁰ = [[0, I], [I, 0]]
        γi = [[0, σi], [-σi, 0]]
        γ⁵ = iγ⁰γ¹γ²γ³
        """
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        Z2 = np.zeros((2, 2), dtype=complex)
        
        gamma_0 = np.block([[Z2, I2], [I2, Z2]])
        gamma_1 = np.block([[Z2, sigma_x], [-sigma_x, Z2]])
        gamma_2 = np.block([[Z2, sigma_y], [-sigma_y, Z2]])
        gamma_3 = np.block([[Z2, sigma_z], [-sigma_z, Z2]])
        
        gamma_5 = 1j * gamma_0 @ gamma_1 @ gamma_2 @ gamma_3
        
        return [gamma_0, gamma_1, gamma_2, gamma_3, gamma_5]
    
    def _get_gamma_matrices(self) -> list:
        """返回 Dirac γ 矩阵"""
        return self._gamma_matrices
    
    def _compute_gradient(self, position: np.ndarray) -> np.ndarray:
        """计算旋量场的梯度
        
        返回 ∂_i ψ(x)，即旋量对空间坐标的偏导数
        维度为 (空间维度, 旋量分量数)
        
        Args:
            position: 空间位置
            
        Returns:
            旋量梯度数组，shape = (dim, 4) 对于 Dirac 旋量
        """
        epsilon = 1e-6
        dim = len(position)
        psi = self.evaluate(position)
        spinor_dim = len(psi) if psi.ndim > 0 else 1
        
        gradient = np.zeros((dim, spinor_dim), dtype=complex)
        for i in range(dim):
            pos_plus = position.copy()
            pos_plus[i] += epsilon
            pos_minus = position.copy()
            pos_minus[i] -= epsilon
            psi_plus = self.evaluate(pos_plus)
            psi_minus = self.evaluate(pos_minus)
            gradient[i] = (psi_plus - psi_minus) / (2 * epsilon)
        return gradient
