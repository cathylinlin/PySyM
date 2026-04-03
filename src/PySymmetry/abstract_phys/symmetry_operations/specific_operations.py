"""
具体对称操作实现
"""
import numpy as np
from typing import List, Any
from PySymmetry.abstract_phys.symmetry_environments.base import (
    PhysicalSymmetry, SymmetryType, SymmetryCategory, SymmetryParameters
)
from PySymmetry.abstract_phys.symmetry_operations.base import SymmetryOperation


def _rotation_from_matrix(R: np.ndarray):
    """从3x3旋转矩阵提取轴-角表示，返回 RotationOperation 或 IdentityOperation"""
    trace_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = float(np.arccos(trace_val))
    
    if abs(theta) < 1e-12:
        return IdentityOperation()
    
    if abs(theta - np.pi) < 1e-12:
        S = R + np.eye(3)
        for col in range(3):
            v = S[:, col]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                axis = v / norm
                return RotationOperation(axis, theta)
    
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    norm = np.linalg.norm(axis)
    if norm > 1e-12:
        axis = axis / norm
    else:
        axis = np.array([0.0, 0.0, 1.0])
    
    return RotationOperation(axis, theta)


# 导入 IdentityOperation 以避免循环导入
class IdentityOperation(SymmetryOperation):
    """恒等操作"""
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.abstract_group import Group
        class TrivialGroup(Group):
            def __init__(self):
                super().__init__("Trivial")
                self._identity = "e"
            
            def identity(self):
                return self._identity
            
            def multiply(self, a, b):
                return self._identity
            
            def inverse(self, element):
                return self._identity
            
            def __contains__(self, element):
                return element == self._identity
            
            def order(self):
                return 1
            
            def elements(self):
                return [self._identity]
        return TrivialGroup()
    
    @property
    def is_continuous(self):
        return False
    
    def compose(self, other):
        return other
    
    def inverse(self):
        return self
    
    def act_on(self, obj):
        return obj

class TranslationOperation(SymmetryOperation):
    """平移操作"""
    
    def __init__(self, displacement):
        self._displacement = displacement
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.continuous_groups import TranslationGroup
        return TranslationGroup(len(self._displacement))
    
    @property
    def is_continuous(self):
        return True
    
    def compose(self, other):
        if isinstance(other, TranslationOperation):
            new_displacement = self._displacement + other._displacement
            return TranslationOperation(new_displacement)
        raise ValueError("只能与平移操作组合")
    
    def inverse(self):
        return TranslationOperation(-self._displacement)
    
    def act_on(self, obj):
        if hasattr(obj, 'position'):
            obj.position += self._displacement
        return obj

class RotationOperation(SymmetryOperation):
    """旋转操作"""
    
    def __init__(self, axis, angle):
        super().__init__()
        self._axis = np.array(axis) / np.linalg.norm(axis)
        self._angle = angle
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.continuous_groups import RotationGroup
        return RotationGroup(3)
    
    @property
    def is_continuous(self):
        return True
    
    def compose(self, other):
        if isinstance(other, RotationOperation):
            # 通过矩阵乘法组合旋转
            R1 = self._rotation_matrix_3d()
            R2 = other._rotation_matrix_3d()
            R_combined = R1 @ R2
            # 从合成矩阵提取轴和角度
            return _rotation_from_matrix(R_combined)
        if isinstance(other, IdentityOperation):
            return self
        raise ValueError("只能与旋转操作或恒等操作组合")
    
    def _rotation_matrix_3d(self) -> np.ndarray:
        """Rodrigues旋转公式生成3x3旋转矩阵"""
        k = self._axis
        theta = self._angle
        K = np.array([
            [0,    -k[2],  k[1]],
            [k[2],  0,    -k[0]],
            [-k[1], k[0],  0   ]
        ])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    def inverse(self):
        return RotationOperation(self._axis, -self._angle)
    
    def act_on(self, obj):
        if hasattr(obj, 'position'):
            # 实现旋转矩阵并应用到位置
            obj.position = self._rotate_vector(obj.position)
        return obj
    
    def _rotate_vector(self, v):
        # Rodrigues旋转公式
        k = self._axis
        theta = self._angle
        return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))

class TimeTranslationOperation(SymmetryOperation):
    """时间平移操作"""
    
    def __init__(self, dt):
        super().__init__()
        self._dt = dt
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.continuous_groups import TimeTranslationGroup
        return TimeTranslationGroup()
    
    @property
    def is_continuous(self):
        return True
    
    def compose(self, other):
        if isinstance(other, TimeTranslationOperation):
            return TimeTranslationOperation(self._dt + other._dt)
        raise ValueError("只能与时间平移操作组合")
    
    def inverse(self):
        return TimeTranslationOperation(-self._dt)
    
    def act_on(self, obj):
        # 时间平移通常不直接作用于物理对象
        return obj

class ParityOperation(SymmetryOperation):
    """宇称操作"""
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.discrete_groups import ParityGroup
        return ParityGroup()
    
    @property
    def is_continuous(self):
        return False
    
    def compose(self, other):
        if isinstance(other, ParityOperation):
            return IdentityOperation()
        raise ValueError("只能与宇称操作组合")
    
    def inverse(self):
        return self
    
    def act_on(self, obj):
        if hasattr(obj, 'position'):
            obj.position = -obj.position
        return obj

class TimeReversalOperation(SymmetryOperation):
    """时间反演操作"""
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.discrete_groups import TimeReversalGroup
        return TimeReversalGroup()
    
    @property
    def is_continuous(self):
        return False
    
    def compose(self, other):
        if isinstance(other, TimeReversalOperation):
            return IdentityOperation()
        raise ValueError("只能与时间反演操作组合")
    
    def inverse(self):
        return self
    
    def act_on(self, obj):
        if hasattr(obj, 'velocity'):
            obj.velocity = -obj.velocity
        return obj

class GaugeOperation(SymmetryOperation):
    """规范操作"""
    
    def __init__(self, group, params):
        super().__init__()
        self._group = group
        self._params = params
    
    @property
    def group(self):
        return self._group
    
    @property
    def is_continuous(self):
        return True
    
    def compose(self, other):
        if isinstance(other, GaugeOperation) and self._group == other._group:
            # 通过群乘法组合规范操作的参数
            if hasattr(self._group, 'multiply') and hasattr(self._params, 'continuous_params') and hasattr(other._params, 'continuous_params'):
                try:
                    combined_params = self._group.multiply(
                        self._params.continuous_params.get('element', self._group.identity()),
                        other._params.continuous_params.get('element', self._group.identity())
                    )
                    new_params = SymmetryParameters(
                        continuous_params={'element': combined_params}
                    )
                    return GaugeOperation(self._group, new_params)
                except (AttributeError, TypeError):
                    pass
            return GaugeOperation(self._group, self._params)
        if isinstance(other, IdentityOperation):
            return self
        raise ValueError("只能与相同规范群的规范操作组合")
    
    def inverse(self):
        if hasattr(self._group, 'inverse') and hasattr(self._params, 'continuous_params'):
            try:
                elem = self._params.continuous_params.get('element', self._group.identity())
                inv_elem = self._group.inverse(elem)
                new_params = SymmetryParameters(
                    continuous_params={'element': inv_elem}
                )
                return GaugeOperation(self._group, new_params)
            except (AttributeError, TypeError):
                pass
        return GaugeOperation(self._group, self._params)
    
    def act_on(self, obj):
        # 规范变换通常作用于场
        return obj

class CompositeOperation(SymmetryOperation):
    """组合操作"""
    
    def __init__(self, operations):
        super().__init__()
        self._operations = operations
    
    @property
    def group(self):
        from PySymmetry.core.group_theory.product_group import DirectProductGroup
        groups = [op.group for op in self._operations]
        return DirectProductGroup(*groups)
    
    @property
    def is_continuous(self):
        return any(op.is_continuous for op in self._operations)
    
    def compose(self, other):
        if isinstance(other, CompositeOperation) and len(self._operations) == len(other._operations):
            new_ops = [op1.compose(op2) for op1, op2 in zip(self._operations, other._operations)]
            return CompositeOperation(new_ops)
        raise ValueError("只能与相同长度的组合操作组合")
    
    def inverse(self):
        return CompositeOperation([op.inverse() for op in reversed(self._operations)])
    
    def act_on(self, obj):
        for op in self._operations:
            obj = op.act_on(obj)
        return obj

class TranslationSymmetry(PhysicalSymmetry):
    """
    平移对称性
    
    物理系统在空间平移下的不变性
    对应守恒量：动量
    """
    
    def __init__(self, dimension: int = 3):
        # 创建平移群 R^n
        from PySymmetry.core.group_theory.continuous_groups import TranslationGroup
        group = TranslationGroup(dimension)
        
        super().__init__(
            symmetry_type=SymmetryType.SPATIAL_TRANSLATION,
            group=group,
            category=SymmetryCategory.CONTINUOUS
        )
        
        self._dimension = dimension
    
    def create_operation(self, params: SymmetryParameters) -> 'TranslationOperation':
        """创建平移操作"""
        displacement = params.continuous_params.get('displacement', 
                                                     np.zeros(self._dimension))
        return TranslationOperation(displacement)
    
    def generators(self) -> List[Any]:
        """平移生成元：动量算符"""
        from PySymmetry.abstract_phys.symmetry_operations.generators import MomentumGenerator
        return [MomentumGenerator(i) for i in range(self._dimension)]
    
    def conserved_quantity(self) -> str:
        """守恒量：动量"""
        return "momentum"
    
    @property
    def dimension(self) -> int:
        """空间维度"""
        return self._dimension

class RotationSymmetry(PhysicalSymmetry):
    """
    旋转对称性
    
    物理系统在空间旋转下的不变性
    对应守恒量：角动量
    """
    
    def __init__(self, dimension: int = 3):
        # 创建旋转群 SO(n)
        from PySymmetry.core.group_theory.continuous_groups import RotationGroup
        group = RotationGroup(dimension)
        
        super().__init__(
            symmetry_type=SymmetryType.SPATIAL_ROTATION,
            group=group,
            category=SymmetryCategory.CONTINUOUS
        )
        
        self._dimension = dimension
    
    def create_operation(self, params: SymmetryParameters) -> 'RotationOperation':
        """创建旋转操作"""
        axis = params.continuous_params.get('axis', [0, 0, 1])
        angle = params.continuous_params.get('angle', 0.0)
        return RotationOperation(axis, angle)
    
    def generators(self) -> List[Any]:
        """旋转生成元：角动量算符"""
        from PySymmetry.abstract_phys.symmetry_operations.generators import AngularMomentumGenerator
        n = self._dimension
        return [AngularMomentumGenerator(i, j) 
                for i in range(n) for j in range(i+1, n)]
    
    def conserved_quantity(self) -> str:
        """守恒量：角动量"""
        return "angular_momentum"


class TimeTranslationSymmetry(PhysicalSymmetry):
    """
    时间平移对称性
    
    物理系统在时间平移下的不变性
    对应守恒量：能量
    """
    
    def __init__(self):
        from PySymmetry.core.group_theory.continuous_groups import TimeTranslationGroup
        group = TimeTranslationGroup()
        
        super().__init__(
            symmetry_type=SymmetryType.TIME_TRANSLATION,
            group=group,
            category=SymmetryCategory.CONTINUOUS
        )
    
    def create_operation(self, params: SymmetryParameters) -> 'TimeTranslationOperation':
        """创建时间平移操作"""
        dt = params.continuous_params.get('dt', 0.0)
        return TimeTranslationOperation(dt)
    
    def generators(self) -> List[Any]:
        """时间平移生成元：哈密顿量"""
        from PySymmetry.abstract_phys.symmetry_operations.generators import HamiltonianGenerator
        return [HamiltonianGenerator()]
    
    def conserved_quantity(self) -> str:
        """守恒量：能量"""
        return "energy"


class ParitySymmetry(PhysicalSymmetry):
    """
    宇称对称性
    
    空间反演对称性 P: r -> -r
    """
    
    def __init__(self):
        from PySymmetry.core.group_theory.discrete_groups import ParityGroup
        group = ParityGroup()
        
        super().__init__(
            symmetry_type=SymmetryType.PARITY,
            group=group,
            category=SymmetryCategory.DISCRETE
        )
    
    def create_operation(self, params: SymmetryParameters = None) -> 'ParityOperation':
        """创建宇称操作"""
        return ParityOperation()
    
    def generators(self) -> List[Any]:
        """离散对称性，生成元为宇称算符"""
        from PySymmetry.abstract_phys.symmetry_operations.generators import ParityOperator
        return [ParityOperator()]
    
    def conserved_quantity(self) -> str:
        """守恒量：宇称"""
        return "parity"


class TimeReversalSymmetry(PhysicalSymmetry):
    """
    时间反演对称性
    
    时间反演 T: t -> -t
    """
    
    def __init__(self):
        from PySymmetry.core.group_theory.discrete_groups import TimeReversalGroup
        group = TimeReversalGroup()
        
        super().__init__(
            symmetry_type=SymmetryType.TIME_REVERSAL,
            group=group,
            category=SymmetryCategory.DISCRETE
        )
    
    def create_operation(self, params: SymmetryParameters = None) -> 'TimeReversalOperation':
        """创建时间反演操作"""
        return TimeReversalOperation()
    
    def generators(self) -> List[Any]:
        """时间反演算符"""
        from PySymmetry.abstract_phys.symmetry_operations.generators import TimeReversalOperator
        return [TimeReversalOperator()]
    
    def conserved_quantity(self) -> str:
        """时间反演对称性没有经典守恒量"""
        return "none"

class GaugeSymmetry(PhysicalSymmetry):
    """
    规范对称性
    
    U(1), SU(2), SU(3) 等规范对称性
    """
    
    def __init__(self, gauge_group: str = "U(1)"):
        from PySymmetry.abstract_phys.symmetry_environments.gauge_groups import GaugeGroupFactory
        group = GaugeGroupFactory.create(gauge_group)
        
        super().__init__(
            symmetry_type=SymmetryType.GAUGE,
            group=group,
            category=SymmetryCategory.INTERNAL
        )
        
        self._gauge_group_name = gauge_group
    
    def create_operation(self, params: SymmetryParameters) -> 'GaugeOperation':
        """创建规范变换"""
        return GaugeOperation(self._group, params)
    
    def generators(self) -> List[Any]:
        """规范群生成元"""
        if hasattr(self._group, 'generators'):
            return self._group.generators()
        return []
    
    def conserved_quantity(self) -> str:
        """守恒量：规范荷"""
        if self._gauge_group_name == "U(1)":
            return "charge"
        elif self._gauge_group_name == "SU(2)":
            return "isospin"
        elif self._gauge_group_name == "SU(3)":
            return "color_charge"
        return "gauge_charge"
    

class CompositeSymmetry(PhysicalSymmetry):
    """
    组合对称性
    
    多个对称性的组合
    """
    
    def __init__(self, symmetries: List[PhysicalSymmetry]):
        from PySymmetry.core.group_theory.product_group import DirectProductGroup
        
        # 构建直积群
        groups = [s.group for s in symmetries]
        combined_group = DirectProductGroup(*groups)
        
        super().__init__(
            symmetry_type=SymmetryType.CPT,  # 作为示例
            group=combined_group,
            category=SymmetryCategory.CONTINUOUS  # 需要根据实际确定
        )
        
        self._symmetries = symmetries
    
    def create_operation(self, params: List[SymmetryParameters]) -> 'CompositeOperation':
        """创建组合操作"""
        operations = [s.create_operation(p) 
                     for s, p in zip(self._symmetries, params)]
        return CompositeOperation(operations)
    
    def generators(self) -> List[Any]:
        """所有生成元的集合"""
        gens = []
        for sym in self._symmetries:
            gens.extend(sym.generators())
        return gens
    
    def conserved_quantity(self) -> str:
        """多个守恒量"""
        return ", ".join([s.conserved_quantity() for s in self._symmetries])
    
    @property
    def component_symmetries(self) -> List[PhysicalSymmetry]:
        """组成部分的对称性"""
        return self._symmetries
