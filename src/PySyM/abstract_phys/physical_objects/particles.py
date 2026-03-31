"""
基本粒子实现

定义各种基本粒子的类，包括基本粒子基类、夸克和轻子。
"""
from typing import Dict, Any, Optional, TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from ..symmetry_operations.base import SymmetryOperation

from .abstract_physical_objects import PhysicalObject, PhysicalQuantity

class ElementaryParticle(PhysicalObject):
    """基本粒子基类
    
    所有基本粒子的基类，定义了粒子的基本属性和方法。
    与 core 模块的矩阵和群表示有良好的集成。
    """
    
    def __init__(self, mass: float, charge: float, spin: float, 
                 position: Optional[np.ndarray] = None, 
                 velocity: Optional[np.ndarray] = None,
                 momentum: Optional[np.ndarray] = None):
        """初始化基本粒子
        
        Args:
            mass: 粒子质量
            charge: 粒子电荷
            spin: 粒子自旋
            position: 粒子位置（可选）
            velocity: 粒子速度（可选）
            momentum: 粒子动量（可选）
            
        Raises:
            ValueError: 如果质量为负数
            TypeError: 如果自旋不是数字
        """
        if mass < 0:
            raise ValueError("质量不能为负数")
        if not isinstance(spin, (int, float)):
            raise TypeError("自旋必须是数字")
        
        self._mass = mass
        self._charge = charge
        self._spin = spin
        self._position = np.zeros(3) if position is None else np.array(position, dtype=float)
        self._velocity = np.zeros(3) if velocity is None else np.array(velocity, dtype=float)
        self._momentum = np.zeros(3) if momentum is None else np.array(momentum, dtype=float)
    
    @property
    def position(self) -> np.ndarray:
        """粒子位置"""
        return self._position
    
    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = np.array(value, dtype=float)
    
    @property
    def velocity(self) -> np.ndarray:
        """粒子速度"""
        return self._velocity
    
    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self._velocity = np.array(value, dtype=float)
    
    @property
    def momentum(self) -> np.ndarray:
        """粒子动量"""
        return self._momentum
    
    @momentum.setter
    def momentum(self, value: np.ndarray) -> None:
        self._momentum = np.array(value, dtype=float)
    
    def get_mass(self) -> PhysicalQuantity:
        return self._mass
    
    def get_charge(self) -> PhysicalQuantity:
        return self._charge
    
    def get_spin(self) -> PhysicalQuantity:
        return self._spin
    
    @property
    def symmetry_properties(self) -> Dict[str, Any]:
        """返回对称性性质"""
        return {
            "mass": self._mass,
            "charge": self._charge,
            "spin": self._spin,
            "position": self._position.tolist(),
            "velocity": self._velocity.tolist(),
            "momentum": self._momentum.tolist()
        }
    
    def transform(self, symmetry_operation: 'SymmetryOperation') -> 'ElementaryParticle':
        """在对称操作下变换
        
        Args:
            symmetry_operation: 对称操作对象
            
        Returns:
            变换后的粒子
        """
        import copy
        new_particle: ElementaryParticle = copy.deepcopy(self)
        new_particle._position = symmetry_operation.apply_to_vector(self._position)
        
        if hasattr(symmetry_operation, 'act_on'):
            symmetry_operation.act_on(new_particle)
        
        return new_particle
    
    def is_invariant_under(self, symmetry_operation: 'SymmetryOperation') -> bool:
        """检查是否在某对称操作下不变
        
        Args:
            symmetry_operation: 对称操作对象
            
        Returns:
            是否不变
        """
        transformed = self.transform(symmetry_operation)
        
        mass_invariant = bool(np.isclose(self._mass, transformed._mass))
        charge_invariant = bool(np.isclose(self._charge, transformed._charge))
        spin_invariant = bool(np.isclose(self._spin, transformed._spin))
        
        return mass_invariant and charge_invariant and spin_invariant
    
    def get_four_momentum(self) -> np.ndarray:
        """获取四动量
        
        Returns:
            四动量 [E, px, py, pz]
        """
        E = np.sqrt(self._mass**2 + np.sum(self._momentum**2))
        return np.concatenate([[E], self._momentum])
    
    def get_kinetic_energy(self) -> float:
        """获取动能
        
        Returns:
            动能
        """
        return 0.5 * self._mass * np.sum(self._velocity**2)
    
    def get_lorentz_boost(self, velocity: np.ndarray) -> 'ElementaryParticle':
        """执行洛伦兹变换
        
        Args:
            velocity: 参考系相对速度 (boost velocity)
            
        Returns:
            变换后的粒子
        """
        import copy
        new_particle = copy.deepcopy(self)
        
        v = np.linalg.norm(velocity)
        c = 1.0  # 自然单位制
        
        if v < 1e-10 or v >= c:
            return new_particle
        
        beta = v / c
        gamma = 1.0 / np.sqrt(1 - beta**2)
        
        v_unit = velocity / v
        v_dot_p = np.dot(v_unit, self._momentum)
        
        # 正确的洛伦兹变换公式
        # E' = γ(E - v·p)
        E = np.sqrt(self._mass**2 * c**4 + np.sum(self._momentum**2) * c**2)
        new_energy = gamma * (E - v * v_dot_p)
        
        # p'_parallel = γ(p_parallel - v*E/c²)
        # p'_perp = p_perp
        p_parallel = v_dot_p * v_unit
        p_perp = self._momentum - p_parallel
        
        new_p_parallel = gamma * (p_parallel - velocity * E / c**2)
        new_momentum = new_p_parallel + p_perp
        
        new_particle._momentum = new_momentum
        # v' = p'c²/E'
        if new_energy > 0:
            new_particle._velocity = new_momentum * c**2 / new_energy
        
        return new_particle

class Quark(ElementaryParticle):
    """夸克
    
    具有 SU(3) 颜色荷的基本粒子。
    """
    
    FLAVORS = ['up', 'down', 'charm', 'strange', 'top', 'bottom']
    COLOR_CHARGES = ['red', 'green', 'blue']
    
    def __init__(self, flavor: str, mass: float, charge: float, spin: float,
                 color: str = 'red', position: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None):
        """初始化夸克
        
        Args:
            flavor: 夸克味 ('up', 'down', 'charm', 'strange', 'top', 'bottom')
            mass: 质量
            charge: 电荷
            spin: 自旋
            color: 颜色荷 ('red', 'green', 'blue')
            position: 位置
            velocity: 速度
        """
        super().__init__(mass, charge, spin, position, velocity)
        
        if flavor not in self.FLAVORS:
            raise ValueError(f"未知的夸克味: {flavor}")
        if color not in self.COLOR_CHARGES:
            raise ValueError(f"未知的颜色荷: {color}")
        
        self._flavor = flavor
        self._color = color
    
    @property
    def flavor(self) -> str:
        """夸克味"""
        return self._flavor
    
    @property
    def color(self) -> str:
        """颜色荷"""
        return self._color
    
    @property
    def symmetry_properties(self) -> Dict[str, Any]:
        """返回对称性性质"""
        properties = super().symmetry_properties
        properties["flavor"] = self._flavor
        properties["color"] = self._color
        properties["color_group"] = "SU(3)"
        return properties
    
    def transform(self, symmetry_operation: 'SymmetryOperation') -> 'Quark':
        """在对称操作下变换夸克
        
        Args:
            symmetry_operation: 对称操作
            
        Returns:
            变换后的夸克
        """
        import copy
        new_quark: Quark = copy.deepcopy(self)
        return new_quark
    
    @staticmethod
    def get_quantum_numbers(flavor: str) -> Dict[str, Any]:
        """获取夸克的量子数
        
        Args:
            flavor: 夸克味
            
        Returns:
            量子数字典
        """
        quantum_numbers = {
            'up': {'charge': 2/3, 'isospin': 1/2, 'strangeness': 0, 'charm': 0, 'bottomness': 0, 'topness': 0},
            'down': {'charge': -1/3, 'isospin': -1/2, 'strangeness': 0, 'charm': 0, 'bottomness': 0, 'topness': 0},
            'charm': {'charge': 2/3, 'isospin': 0, 'strangeness': 0, 'charm': 1, 'bottomness': 0, 'topness': 0},
            'strange': {'charge': -1/3, 'isospin': 0, 'strangeness': -1, 'charm': 0, 'bottomness': 0, 'topness': 0},
            'top': {'charge': 2/3, 'isospin': 0, 'strangeness': 0, 'charm': 0, 'bottomness': 0, 'topness': 1},
            'bottom': {'charge': -1/3, 'isospin': 0, 'strangeness': 0, 'charm': 0, 'bottomness': -1, 'topness': 0},
        }
        return quantum_numbers.get(flavor, {})


class Lepton(ElementaryParticle):
    """轻子
    
    不参与强相互作用的基本粒子。
    """
    
    TYPES = ['electron', 'muon', 'tau', 'electron_neutrino', 'muon_neutrino', 'tau_neutrino']
    
    def __init__(self, lepton_type: str, mass: float, charge: float, spin: float,
                 position: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None):
        """初始化轻子
        
        Args:
            lepton_type: 轻子类型
            mass: 质量
            charge: 电荷
            spin: 自旋
            position: 位置
            velocity: 速度
        """
        super().__init__(mass, charge, spin, position, velocity)
        
        if lepton_type not in self.TYPES:
            raise ValueError(f"未知的轻子类型: {lepton_type}")
        
        self._lepton_type = lepton_type
    
    @property
    def type(self) -> str:
        """轻子类型"""
        return self._lepton_type
    
    @property
    def is_neutrino(self) -> bool:
        """是否为中微子"""
        return 'neutrino' in self._lepton_type
    
    @property
    def symmetry_properties(self) -> Dict[str, Any]:
        """返回对称性性质"""
        properties = super().symmetry_properties
        properties["type"] = self._lepton_type
        properties["is_neutrino"] = self.is_neutrino
        return properties
    
    @staticmethod
    def get_quantum_numbers(lepton_type: str) -> Dict[str, Any]:
        """获取轻子的量子数
        
        Args:
            lepton_type: 轻子类型
            
        Returns:
            量子数字典
        """
        quantum_numbers = {
            'electron': {'charge': -1, 'lepton_number': 1, 'family': 1},
            'muon': {'charge': -1, 'lepton_number': 1, 'family': 2},
            'tau': {'charge': -1, 'lepton_number': 1, 'family': 3},
            'electron_neutrino': {'charge': 0, 'lepton_number': 1, 'family': 1},
            'muon_neutrino': {'charge': 0, 'lepton_number': 1, 'family': 2},
            'tau_neutrino': {'charge': 0, 'lepton_number': 1, 'family': 3},
        }
        return quantum_numbers.get(lepton_type, {})