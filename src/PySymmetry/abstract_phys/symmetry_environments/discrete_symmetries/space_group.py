"""
空间群模块

包含：
- 230个三维空间群
- 布拉维格子
- 倒格子与布里渊区
- 空间群操作
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .point_groups import PointGroup


# -----------------------------------------------------------------------------
# 1. 布拉维格子
# -----------------------------------------------------------------------------

class BravaisLatticeType(Enum):
    """布拉维格子类型"""
    # 三斜晶系
    TRICLINIC_P = "P"  # 简单三斜
    
    # 单斜晶系
    MONOCLINIC_P = "P"  # 简单单斜
    MONOCLINIC_C = "C"  # 底心单斜
    
    # 正交晶系
    ORTHORHOMBIC_P = "P"  # 简单正交
    ORTHORHOMBIC_C = "C"  # 底心正交
    ORTHORHOMBIC_I = "I"  # 体心正交
    ORTHORHOMBIC_F = "F"  # 面心正交
    
    # 四方晶系
    TETRAGONAL_P = "P"  # 简单四方
    TETRAGONAL_I = "I"  # 体心四方
    
    # 三方晶系
    TRIGONAL_P = "P"  # 简单三方
    TRIGONAL_R = "R"  # 菱面体
    
    # 六方晶系
    HEXAGONAL_P = "P"  # 简单六方
    
    # 立方晶系
    CUBIC_P = "P"  # 简单立方
    CUBIC_I = "I"  # 体心立方
    CUBIC_F = "F"  # 面心立方


@dataclass
class BravaisLattice:
    """布拉维格子"""
    
    lattice_type: BravaisLatticeType
    lattice_vectors: np.ndarray  # 3×3矩阵，每行是一个格矢
    
    @property
    def volume(self) -> float:
        """原胞体积"""
        return abs(np.linalg.det(self.lattice_vectors))
    
    def reciprocal_lattice(self) -> 'BravaisLattice':
        """
        倒格子
        
        b_i = 2π (a_j × a_k) / (a_1 · (a_2 × a_3))
        """
        a = self.lattice_vectors
        volume = self.volume
        
        b = np.zeros((3, 3))
        b[0] = 2 * np.pi * np.cross(a[1], a[2]) / volume
        b[1] = 2 * np.pi * np.cross(a[2], a[0]) / volume
        b[2] = 2 * np.pi * np.cross(a[0], a[1]) / volume
        
        return BravaisLattice(
            lattice_type=self.lattice_type,
            lattice_vectors=b
        )
    
    def wigner_seitz_cell(self) -> List[np.ndarray]:
        """Wigner-Seitz原胞的顶点"""
        # 找出原胞边界
        pass
    
    def brillouin_zone(self) -> List[np.ndarray]:
        """布里渊区的顶点"""
        # 倒格子的Wigner-Seitz原胞
        return self.reciprocal_lattice().wigner_seitz_cell()
    
    def high_symmetry_points(self) -> Dict[str, np.ndarray]:
        """
        高对称k点
        
        返回Γ, X, L, W, K等特殊点
        """
        points = {'Γ': np.array([0, 0, 0])}
        
        if self.lattice_type in [BravaisLatticeType.CUBIC_P]:
            points['X'] = np.array([0.5, 0, 0])
            points['M'] = np.array([0.5, 0.5, 0])
            points['R'] = np.array([0.5, 0.5, 0.5])
        
        elif self.lattice_type in [BravaisLatticeType.CUBIC_F]:
            points['X'] = np.array([0.5, 0, 0.5])
            points['L'] = np.array([0.5, 0.5, 0.5])
            points['W'] = np.array([0.5, 0.25, 0.75])
            points['K'] = np.array([0.375, 0.375, 0.75])
        
        elif self.lattice_type in [BravaisLatticeType.CUBIC_I]:
            points['H'] = np.array([0.5, -0.5, 0.5])
            points['N'] = np.array([0, 0, 0.5])
            points['P'] = np.array([0.25, 0.25, 0.25])
        
        return points
    
    def lattice_points_in_sphere(self, radius: float) -> List[Tuple[np.ndarray, int]]:
        """
        球内的格点
        
        返回: [(格矢, 模长平方)]
        """
        points = []
        
        # 估计需要搜索的范围
        n_max = int(radius / np.min(np.linalg.norm(self.lattice_vectors, axis=1))) + 1
        
        for i in range(-n_max, n_max+1):
            for j in range(-n_max, n_max+1):
                for k in range(-n_max, n_max+1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    
                    lattice_point = i * self.lattice_vectors[0] + \
                                   j * self.lattice_vectors[1] + \
                                   k * self.lattice_vectors[2]
                    
                    dist_sq = np.sum(lattice_point**2)
                    
                    if dist_sq <= radius**2:
                        points.append((lattice_point, dist_sq))
        
        # 按距离排序
        points.sort(key=lambda x: x[1])
        
        return points


# -----------------------------------------------------------------------------
# 2. 空间群操作
# -----------------------------------------------------------------------------

@dataclass
class SpaceGroupOperation:
    """
    空间群操作
    
    {R | t}: r → Rr + t
    R: 点群操作（旋转、反射等）
    t: 平移（格矢 + 分数平移）
    """
    
    rotation: np.ndarray      # 3×3旋转/反射矩阵
    translation: np.ndarray   # 平移向量
    
    @property
    def is_pure_translation(self) -> bool:
        """是否纯平移"""
        return np.allclose(self.rotation, np.eye(3))
    
    @property
    def is_screw_axis(self) -> bool:
        """是否有螺旋轴"""
        # 检查是否旋转+分数平移
        if np.allclose(self.rotation, np.eye(3)):
            return False
        # 检查平移是否沿旋转轴方向
        # 简化判断
        return not np.allclose(self.translation, 0) and not self.is_pure_translation
    
    @property
    def is_glide_plane(self) -> bool:
        """是否有滑移面"""
        # 反射+沿镜面方向的分数平移
        det = np.linalg.det(self.rotation)
        return abs(det + 1) < 1e-6 and not np.allclose(self.translation, 0)
    
    def apply(self, position: np.ndarray) -> np.ndarray:
        """应用操作于位置"""
        return self.rotation @ position + self.translation
    
    def compose(self, other: 'SpaceGroupOperation') -> 'SpaceGroupOperation':
        """
        组合两个操作
        
        {R1|t1}{R2|t2} = {R1R2|R1t2+t1}
        """
        return SpaceGroupOperation(
            rotation=self.rotation @ other.rotation,
            translation=self.rotation @ other.translation + self.translation
        )
    
    def inverse(self) -> 'SpaceGroupOperation':
        """逆操作"""
        R_inv = np.linalg.inv(self.rotation)
        return SpaceGroupOperation(
            rotation=R_inv,
            translation=-R_inv @ self.translation
        )
    
    def seitz_symbol(self) -> str:
        """Seitz符号"""
        # 简化表示
        pass


# -----------------------------------------------------------------------------
# 3. 空间群
# -----------------------------------------------------------------------------

class SpaceGroup:
    """
    空间群
    
    230个三维空间群的完整定义
    """
    
    # 空间群数据库（部分示例）
    SPACE_GROUP_DATA = {
        # P1 (No. 1)
        1: {
            'symbol': 'P1',
            'number': 1,
            'point_group': 'C1',
            'lattice_type': BravaisLatticeType.TRICLINIC_P,
            'operations': []  # 单位元
        },
        
        # P-1 (No. 2)
        2: {
            'symbol': 'P-1',
            'number': 2,
            'point_group': 'Ci',
            'lattice_type': BravaisLatticeType.TRICLINIC_P,
            'operations': [
                {'rotation': np.eye(3), 'translation': np.zeros(3)},  # E
                {'rotation': -np.eye(3), 'translation': np.zeros(3)}  # i
            ]
        },
        
        # Fm-3m (No. 225) - 面心立方
        225: {
            'symbol': 'Fm-3m',
            'number': 225,
            'point_group': 'Oh',
            'lattice_type': BravaisLatticeType.CUBIC_F,
            'operations': []  # 48个操作
        }
    }
    
    def __init__(self, space_group_number: int):
        """
        初始化空间群
        
        Parameters:
            space_group_number: 空间群编号 (1-230)
        """
        if space_group_number not in self.SPACE_GROUP_DATA:
            raise ValueError(f"空间群编号 {space_group_number} 尚未实现")
        
        self.number = space_group_number
        self.data = self.SPACE_GROUP_DATA[space_group_number]
        
        self.symbol = self.data['symbol']
        self.point_group = self.data['point_group']
        self.lattice_type = self.data['lattice_type']
        self._operations = self._parse_operations(self.data['operations'])
    
    def _parse_operations(self, ops_data: List[Dict]) -> List[SpaceGroupOperation]:
        """解析操作列表"""
        return [
            SpaceGroupOperation(
                rotation=op['rotation'],
                translation=op['translation']
            )
            for op in ops_data
        ]
    
    @property
    def operations(self) -> List[SpaceGroupOperation]:
        """所有空间群操作"""
        return self._operations
    
    @property
    def order(self) -> int:
        """群阶（包括平移操作）"""
        return len(self._operations)
    
    def wyckoff_positions(self) -> List[Dict]:
        """
        Wyckoff位置
        
        返回所有不等价格点位置
        """
        # Wyckoff字母表示多重度的不同
        # 例如 Fm-3m 中:
        # 4a: (0,0,0) - 四重
        # 4b: (0.5,0.5,0.5) - 四重
        # 8c: (0.25,0.25,0.25) - 八重
        # 等等
        pass
    
    def generate_symmetry_equivalent_positions(self, 
                                               position: np.ndarray,
                                               lattice: BravaisLattice) -> List[np.ndarray]:
        """
        生成对称等价位置
        
        考虑周期性边界条件
        """
        equivalent = []
        
        for op in self._operations:
            new_pos = op.apply(position)
            # 模格矢
            for i in range(3):
                new_pos[i] = new_pos[i] % 1.0
            
            # 检查是否已存在
            is_new = True
            for existing in equivalent:
                if np.allclose(new_pos, existing, atol=1e-6):
                    is_new = False
                    break
            
            if is_new:
                equivalent.append(new_pos)
        
        return equivalent
    
    def site_symmetry(self, position: np.ndarray) -> List[SpaceGroupOperation]:
        """
        位置点群对称性
        
        保持该位置不变的对称操作
        """
        site_ops = []
        
        for op in self._operations:
            new_pos = op.apply(position)
            # 检查是否回到原位（可能差格矢）
            if np.allclose(new_pos % 1.0, position % 1.0, atol=1e-6):
                site_ops.append(op)
        
        return site_ops
    
    def check_compatibility(self, k_point: np.ndarray) -> List[str]:
        """
        检查k点的对称性相容性
        
        返回k点的点群
        """
        pass


# -----------------------------------------------------------------------------
# 4. 布里渊区
# -----------------------------------------------------------------------------

class BrillouinZone:
    """布里渊区"""
    
    def __init__(self, lattice: BravaisLattice):
        self.lattice = lattice
        self.reciprocal = lattice.reciprocal_lattice()
        self._vertices = None
        self._faces = None
    
    @property
    def vertices(self) -> List[np.ndarray]:
        """顶点"""
        if self._vertices is None:
            self._compute_bz()
        return self._vertices
    
    def _compute_bz(self):
        """计算布里渊区几何"""
        # Wigner-Seitz原胞算法
        # 找出所有布里渊区边界平面
        pass
    
    def irreducible_bz_volume(self, space_group: SpaceGroup) -> float:
        """
        不可约布里渊区体积
        
        = 完整BZ体积 / 点群阶
        """
        total_volume = self.lattice.volume
        # 点群阶需要从空间群提取
        pass
    
    def k_path(self, 
              path: List[str] = None,
              n_points: int = 50) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        生成k点路径
        
        用于能带结构计算
        """
        high_sym_points = self.lattice.high_symmetry_points()
        
        if path is None:
            # 默认路径根据晶系确定
            path = self._default_k_path()
        
        k_points = []
        labels = []
        point_idx = 0
        
        for i in range(len(path) - 1):
            start = high_sym_points[path[i]]
            end = high_sym_points[path[i+1]]
            
            for j in range(n_points):
                t = j / (n_points - 1)
                k = start + t * (end - start)
                k_points.append(k)
            
            labels.append((path[i], point_idx))
            point_idx += n_points
        
        labels.append((path[-1], point_idx - 1))
        
        return np.array(k_points), labels
    
    def _default_k_path(self) -> List[str]:
        """默认k点路径"""
        lattice_type = self.lattice.lattice_type
        
        if lattice_type in [BravaisLatticeType.CUBIC_P]:
            return ['Γ', 'X', 'M', 'Γ', 'R', 'X']
        elif lattice_type in [BravaisLatticeType.CUBIC_F]:
            return ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L', 'K']
        elif lattice_type in [BravaisLatticeType.CUBIC_I]:
            return ['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N']
        else:
            return ['Γ']  # 简化


# -----------------------------------------------------------------------------
# 5. 常用空间群构造器
# -----------------------------------------------------------------------------

def create_fcc_lattice(a: float) -> BravaisLattice:
    """创建面心立方格子"""
    lattice_vectors = np.array([
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ])
    return BravaisLattice(BravaisLatticeType.CUBIC_F, lattice_vectors)


def create_bcc_lattice(a: float) -> BravaisLattice:
    """创建体心立方格子"""
    lattice_vectors = np.array([
        [-a/2, a/2, a/2],
        [a/2, -a/2, a/2],
        [a/2, a/2, -a/2]
    ])
    return BravaisLattice(BravaisLatticeType.CUBIC_I, lattice_vectors)


def create_simple_cubic_lattice(a: float) -> BravaisLattice:
    """创建简单立方格子"""
    lattice_vectors = a * np.eye(3)
    return BravaisLattice(BravaisLatticeType.CUBIC_P, lattice_vectors)
