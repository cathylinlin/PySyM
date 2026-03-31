
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Type
from PySyM.core.group_theory.abstract_group import Group
from PySyM.abstract_phys.symmetry_operations.base import SymmetryOperation

class SymmetryInfo:
    """对称性信息"""
    def __init__(self, group: Group, generators: List[Any], is_continuous: bool, conserved_quantities: List[str], breaking_pattern: Optional[str] = None):
        self.group = group
        self.generators = generators
        self.is_continuous = is_continuous
        self.conserved_quantities = conserved_quantities
        self.breaking_pattern = breaking_pattern

class SymmetryRegistry:
    """
    对称性注册表
    
    管理所有已定义的对称性
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化对称性字典
            cls._symmetries: Dict[str, type] = {}
        return cls._instance
    
    @classmethod
    def register(cls, name: str, symmetry_class: type):
        """注册对称性"""
        cls._symmetries[name] = symmetry_class
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取对称性类"""
        return cls._symmetries.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有注册的对称性"""
        return list(cls._symmetries.keys())

class SymmetryType(Enum):
    """对称性类型"""
    # 连续空间对称性
    SPATIAL_TRANSLATION = auto()      # 空间平移
    SPATIAL_ROTATION = auto()         # 空间旋转
    SPATIAL_INVERSION = auto()        # 空间反演
    
    # 时间对称性
    TIME_TRANSLATION = auto()         # 时间平移
    TIME_REVERSAL = auto()            # 时间反演
    
    # 内禀对称性
    GAUGE = auto()                    # 规范对称性
    PHASE = auto()                    # 相位对称性
    PERMUTATION = auto()              # 置换对称性
    
    # 离散对称性
    PARITY = auto()                   # 宇称
    CHARGE_CONJUGATION = auto()       # 电荷共轭
    CPT = auto()                      # CPT联合对称
    
    # 其他
    LORENTZ = auto()                  # 洛伦兹对称性
    CONFORMAL = auto()                # 共形对称性
    SUPERSYMMETRY = auto()            # 超对称性


class SymmetryCategory(Enum):
    """对称性类别"""
    CONTINUOUS = "continuous"         # 连续对称性
    DISCRETE = "discrete"             # 离散对称性
    INTERNAL = "internal"             # 内禀对称性
    SPACETIME = "spacetime"           # 时空对称性
    DYNAMICAL = "dynamical"           # 动力学对称性

class SymmetryParameters:
    """对称性参数"""
    def __init__(self, continuous_params: Optional[Dict[str, float]] = None, discrete_params: Optional[Dict[str, Any]] = None):
        self.continuous_params = continuous_params if continuous_params is not None else {}
        self.discrete_params = discrete_params if discrete_params is not None else {}

class PhysicalSymmetry(ABC):
    """
    物理对称性基类
    
    定义物理系统对称性的抽象框架
    """
    
    def __init__(self, 
                symmetry_type: SymmetryType,
                group: Group,
                category: SymmetryCategory):
        self._type = symmetry_type
        self._group = group
        self._category = category
        
    @property
    def type(self) -> SymmetryType:
        """对称性类型"""
        return self._type
    
    @property
    def group(self) -> Group:
        """对称群"""
        return self._group
    
    @property
    def category(self) -> SymmetryCategory:
        """对称性类别"""
        return self._category
    
    @abstractmethod
    def create_operation(self, params: SymmetryParameters) -> SymmetryOperation:
        """创建对称操作"""
        pass
    
    @abstractmethod
    def generators(self) -> List[Any]:
        """生成元"""
        pass
    
    @abstractmethod
    def conserved_quantity(self) -> str:
        """守恒量"""
        pass
    
    def info(self) -> SymmetryInfo:
        """返回对称性信息"""
        return SymmetryInfo(
            group=self._group,
            generators=self.generators(),
            is_continuous=(self._category == SymmetryCategory.CONTINUOUS),
            conserved_quantities=[self.conserved_quantity()]
        )
