"""
对称性破缺模块

提供对称性破缺的建模：
- 显式破缺
- 自发破缺
- Higgs机制
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..symmetry_environments.base import PhysicalSymmetry


class SymmetryBreaking(ABC):
    """对称性破缺基类"""
    
    @abstractmethod
    def broken_symmetry(self) -> 'PhysicalSymmetry':
        """破缺的对称性"""
        pass
    
    @abstractmethod
    def breaking_pattern(self) -> str:
        """破缺模式 G → H"""
        pass


class ExplicitBreaking(SymmetryBreaking):
    """显式对称性破缺"""
    
    def __init__(self, 
                 symmetry: 'PhysicalSymmetry',
                 perturbation: Any):
        self.symmetry = symmetry
        self.perturbation = perturbation
        self._strength: float = 1.0
    
    @property
    def strength(self) -> float:
        """破缺强度"""
        return self._strength
    
    @strength.setter
    def strength(self, value: float):
        self._strength = value
    
    def broken_symmetry(self) -> 'PhysicalSymmetry':
        return self.symmetry
    
    def breaking_pattern(self) -> str:
        return "Explicit breaking"


class SpontaneousBreaking(SymmetryBreaking):
    """自发对称性破缺"""
    
    def __init__(self,
                 full_symmetry: 'PhysicalSymmetry',
                 residual_symmetry: 'PhysicalSymmetry',
                 order_parameter: Any):
        self.full_symmetry = full_symmetry
        self.residual_symmetry = residual_symmetry
        self.order_parameter = order_parameter
        self._goldstone_modes: Optional[List[Any]] = None
    
    def broken_symmetry(self) -> 'PhysicalSymmetry':
        return self.full_symmetry
    
    def breaking_pattern(self) -> str:
        G = self.full_symmetry.group.name if hasattr(self.full_symmetry.group, 'name') else 'G'
        H = self.residual_symmetry.group.name if hasattr(self.residual_symmetry.group, 'name') else 'H'
        return f"{G} → {H}"
    
    def goldstone_modes(self) -> List[Any]:
        """
        Goldstone模式
        
        根据Goldstone定理：破缺生成元数目 = Goldstone模式数目
        """
        if self._goldstone_modes is None:
            self._goldstone_modes = self._compute_goldstone_modes()
        return self._goldstone_modes
    
    def _compute_goldstone_modes(self) -> List['GoldstoneMode']:
        """计算Goldstone模式"""
        full_gens = self.full_symmetry.generators()
        residual_gens = self.residual_symmetry.generators() if self.residual_symmetry else []
        broken_gens = [g for g in full_gens if g not in residual_gens]
        return [self._create_goldstone_mode(g) for g in broken_gens]
    
    def _create_goldstone_mode(self, generator: Any) -> 'GoldstoneMode':
        """创建Goldstone模式"""
        return GoldstoneMode(generator)


class GoldstoneMode:
    """Goldstone模式"""
    
    def __init__(self, broken_generator: Any, 
                 mass: float = 0,
                 dispersion: str = "linear"):
        self.generator = broken_generator
        self.mass = mass
        self.dispersion = dispersion
    
    def field(self, x: np.ndarray, t: float) -> float:
        """Goldstone场 φ(x, t)"""
        return 0.0
    
    def dispersion_relation(self, k: np.ndarray) -> float:
        """
        色散关系
        
        对于无质量Goldstone模式：ω = c|k|
        """
        c = 1.0
        if self.dispersion == "linear":
            return float(c * np.linalg.norm(k))
        elif self.dispersion == "quadratic":
            return float(np.linalg.norm(k) ** 2)
        return 0.0


class HiggsMechanism(SpontaneousBreaking):
    """Higgs机制"""
    
    def __init__(self,
                 gauge_symmetry: 'PhysicalSymmetry',
                 vacuum_expectation_value: float,
                 higgs_field: Any):
        super().__init__(gauge_symmetry, gauge_symmetry, vacuum_expectation_value)
        self.vev = vacuum_expectation_value
        self.higgs_field = higgs_field
        self._massive_gauge_bosons: Optional[Dict[str, float]] = None
    
    def gauge_boson_masses(self, coupling: float = 1.0) -> Dict[str, float]:
        """
        规范玻色子质量
        
        m = g * v / 2
        
        Args:
            coupling: 耦合常数 g
        """
        if self._massive_gauge_bosons is None:
            masses = {}
            # 根据破缺生成元的数目，计算规范玻色子质量
            broken_gens = self.goldstone_modes()
            for i, mode in enumerate(broken_gens):
                mass = coupling * self.vev / 2
                masses[f'W_{i+1}'] = mass
            self._massive_gauge_bosons = masses
        return self._massive_gauge_bosons
    
    def higgs_mass(self, lambda_param: float) -> float:
        """Higgs粒子质量
        
        m_H = sqrt(2λ) * v
        
        其中 λ 是 Higgs 自耦合常数，v 是真空期望值
        
        Args:
            lambda_param: Higgs 自耦合常数 λ
            
        Returns:
            Higgs 粒子质量
        """
        return np.sqrt(2 * lambda_param) * self.vev


class DynamicalBreaking(SymmetryBreaking):
    """动力学对称性破缺"""
    
    def __init__(self, symmetry: 'PhysicalSymmetry'):
        self.symmetry = symmetry
    
    def broken_symmetry(self) -> 'PhysicalSymmetry':
        return self.symmetry
    
    def breaking_pattern(self) -> str:
        return "Dynamical breaking"


class AnomalyMatching:
    """反常匹配条件"""
    
    def __init__(self, group_before: str, group_after: str):
        self.group_before = group_before
        self.group_after = group_after
    
    def check_matching(self, anomalies_before: List[float], 
                      anomalies_after: List[float]) -> bool:
        """
        检查反常匹配条件
        
        Σ G_before = Σ G_after
        """
        return bool(np.isclose(sum(anomalies_before), sum(anomalies_after)))
