"""
分子对称性模块

包含：
- 分子点群确定
- 分子轨道对称性
- 化学键对称性分析
- 振动模式分析
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from itertools import combinations
from dataclasses import dataclass

from .point_groups import PointGroup, PointGroupOperation


# -----------------------------------------------------------------------------
# 1. 分子结构
# -----------------------------------------------------------------------------

@dataclass
class Atom:
    """原子"""
    symbol: str
    position: np.ndarray
    charge: int = 0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Molecule:
    """分子"""
    
    atoms: List[Atom]
    name: str = ""
    
    def center_of_mass(self, masses: Optional[List[float]] = None) -> np.ndarray:
        """质心"""
        if masses is None:
            masses = [self._get_atomic_mass(atom.symbol) for atom in self.atoms]
        
        total_mass = sum(masses)
        com = sum(m * atom.position for m, atom in zip(masses, self.atoms)) / total_mass
        
        return com
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """获取原子质量（简化）"""
        mass_table = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453
        }
        return mass_table.get(symbol, 1.0)
    
    def inertia_tensor(self, masses: Optional[List[float]] = None) -> np.ndarray:
        """转动惯量张量"""
        if masses is None:
            masses = [self._get_atomic_mass(atom.symbol) for atom in self.atoms]
        
        # 移到质心
        com = self.center_of_mass(masses)
        
        I = np.zeros((3, 3))
        for m, atom in zip(masses, self.atoms):
            r = atom.position - com
            I[0, 0] += m * (r[1]**2 + r[2]**2)
            I[1, 1] += m * (r[0]**2 + r[2]**2)
            I[2, 2] += m * (r[0]**2 + r[1]**2)
            I[0, 1] -= m * r[0] * r[1]
            I[0, 2] -= m * r[0] * r[2]
            I[1, 2] -= m * r[1] * r[2]
        
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        
        return I
    
    def principal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """主轴"""
        I = self.inertia_tensor()
        eigenvalues, eigenvectors = np.linalg.eigh(I)
        return eigenvalues, eigenvectors


# -----------------------------------------------------------------------------
# 2. 分子点群识别
# -----------------------------------------------------------------------------

class MolecularSymmetryDetector:
    """分子对称性检测器"""
    
    def __init__(self, molecule: Molecule, tolerance: float = 1e-4):
        self.molecule = molecule
        self.tolerance = tolerance
        self._detected_symmetry = None
    
    def detect_point_group(self) -> PointGroup:
        """
        自动识别分子点群
        
        步骤：
        1. 移到质心
        2. 找旋转轴
        3. 找镜面
        4. 检查反演中心
        5. 确定点群
        """
        # 归一化坐标到质心
        com = self.molecule.center_of_mass()
        positions = [atom.position - com for atom in self.molecule.atoms]
        
        # 找主旋转轴
        rotation_axes = self._find_rotation_axes(positions)
        
        # 检查是否有旋转轴
        if not rotation_axes:
            return self._check_no_rotation_symmetry(positions)
        
        # 找最高阶旋转轴
        highest_order = max(axis['order'] for axis in rotation_axes)
        
        # 检查镜面和反演
        has_inversion = self._has_inversion_center(positions)
        mirror_planes = self._find_mirror_planes(positions)
        has_horizontal_mirror = self._has_horizontal_mirror(positions, rotation_axes)
        
        # 确定点群
        return self._determine_point_group(
            highest_order, rotation_axes, has_inversion, 
            mirror_planes, has_horizontal_mirror
        )
    
    def _find_rotation_axes(self, positions: List[np.ndarray]) -> List[Dict]:
        """找旋转轴"""
        axes = []
        
        # 可能的旋转轴方向
        # 1. 原子连线
        # 2. 坐标轴
        # 3. 通过质心的任意方向
        
        # 检查每个可能的方向
        possible_directions = self._get_possible_axis_directions(positions)
        
        for direction in possible_directions:
            # 检查不同阶数的旋转
            for order in [6, 5, 4, 3, 2]:
                if self._is_rotation_axis(positions, direction, order):
                    axes.append({
                        'direction': direction / np.linalg.norm(direction),
                        'order': order
                    })
                    break  # 只保留最高阶
        
        return axes
    
    def _get_possible_axis_directions(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """获取可能的旋转轴方向"""
        directions = []
        
        # 坐标轴
        directions.extend([
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ])
        
        # 原子连线
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:
                    direction = pos2 - pos1
                    if np.linalg.norm(direction) > self.tolerance:
                        directions.append(direction)
        
        return directions
    
    def _is_rotation_axis(self, positions: List[np.ndarray], 
                         axis: np.ndarray, order: int) -> bool:
        """检查是否是n重旋转轴"""
        axis = axis / np.linalg.norm(axis)
        theta = 2 * np.pi / order
        
        # 旋转矩阵
        R = self._rotation_matrix(axis, theta)
        
        # 对每个原子，检查旋转后是否有对应原子
        for pos in positions:
            rotated = R @ pos
            
            # 找匹配原子
            found_match = False
            for other_pos in positions:
                if np.linalg.norm(rotated - other_pos) < self.tolerance:
                    found_match = True
                    break
            
            if not found_match:
                return False
        
        return True
    
    def _rotation_matrix(self, axis: np.ndarray, theta: float) -> np.ndarray:
        """绕轴旋转矩阵"""
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(theta), np.sin(theta)
        x, y, z = axis
        
        R = np.array([
            [c + x*x*(1-c),    x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),    y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
        ])
        
        return R
    
    def _has_inversion_center(self, positions: List[np.ndarray]) -> bool:
        """检查是否有反演中心"""
        for pos in positions:
            inverted = -pos
            found_match = False
            for other_pos in positions:
                if np.linalg.norm(inverted - other_pos) < self.tolerance:
                    found_match = True
                    break
            if not found_match:
                return False
        return True
    
    def _find_mirror_planes(self, positions: List[np.ndarray]) -> List[np.ndarray]:
        """找镜面"""
        planes = []
        
        # 可能的镜面法向
        possible_normals = self._get_possible_axis_directions(positions)
        
        for normal in possible_normals:
            if self._is_mirror_plane(positions, normal):
                planes.append(normal / np.linalg.norm(normal))
        
        return planes
    
    def _is_mirror_plane(self, positions: List[np.ndarray], normal: np.ndarray) -> bool:
        """检查是否是镜面"""
        normal = normal / np.linalg.norm(normal)
        
        # 反射矩阵
        R = np.eye(3) - 2 * np.outer(normal, normal)
        
        for pos in positions:
            reflected = R @ pos
            found_match = False
            for other_pos in positions:
                if np.linalg.norm(reflected - other_pos) < self.tolerance:
                    found_match = True
                    break
            if not found_match:
                return False
        
        return True
    
    def _has_horizontal_mirror(self, positions: List[np.ndarray], 
                               rotation_axes: List[Dict]) -> bool:
        """检查是否有水平镜面"""
        if not rotation_axes:
            return False
        
        # 主轴方向
        main_axis = rotation_axes[0]['direction']
        
        # 检查垂直于主轴的镜面
        return self._is_mirror_plane(positions, main_axis)
    
    def _check_no_rotation_symmetry(self, positions: List[np.ndarray]) -> PointGroup:
        """检查无旋转轴的对称性（C1, Cs, Ci）"""
        has_inversion = self._has_inversion_center(positions)
        mirrors = self._find_mirror_planes(positions)
        
        if has_inversion:
            from .point_groups import Ci  # Ci群
            return Ci()
        elif mirrors:
            from .point_groups import Cs  # Cs群
            return Cs()
        else:
            from .point_groups import CyclicGroup
            return CyclicGroup(1)  # C1群
    
    def _determine_point_group(self, highest_order, rotation_axes, 
                               has_inversion, mirrors, has_horizontal_mirror) -> PointGroup:
        """确定点群"""
        # 根据对称性元素确定点群
        # 这需要完整的逻辑
        pass


# -----------------------------------------------------------------------------
# 3. 分子轨道对称性
# -----------------------------------------------------------------------------

class MolecularOrbitalSymmetry:
    """分子轨道对称性"""
    
    def __init__(self, point_group: PointGroup):
        self.point_group = point_group
    
    def symmetry_adapted_linear_combination(self,
                                           atomic_orbitals: List[Tuple[int, str]]) -> Dict[str, np.ndarray]:
        """
        构造对称性匹配线性组合（SALC）
        
        使用投影算符方法
        
        Parameters:
            atomic_orbitals: [(原子索引, 轨道类型)] 例如 [(0, 's'), (1, 's'), (2, 'p_x')]
        
        Returns:
            {不可约表示: SALC系数矩阵}
        """
        # 投影算符方法
        pass
    
    def orbital_symmetry_labels(self, 
                               orbital_coefficients: np.ndarray) -> List[str]:
        """
        确定分子轨道的对称性标签
        
        通过投影到不可约表示
        """
        n_orbitals = orbital_coefficients.shape[1]
        labels = []
        
        for i in range(n_orbitals):
            orbital = orbital_coefficients[:, i]
            
            # 投影到各不可约表示
            projections = self._project_to_irreps(orbital)
            
            # 找最大投影
            max_irrep = max(projections.items(), key=lambda x: x[1])
            labels.append(max_irrep[0])
        
        return labels
    
    def _project_to_irreps(self, orbital: np.ndarray) -> Dict[str, float]:
        """投影到不可约表示"""
        projections = {}
        
        for irrep in self.point_group._irrep_names:
            # 使用投影算符
            proj = self._projection_operator(orbital, irrep)
            projections[irrep] = np.linalg.norm(proj)
        
        return projections
    
    def _projection_operator(self, orbital: np.ndarray, irrep: str) -> np.ndarray:
        """投影算符"""
        # P^Γ = (d_Γ/|G|) Σ_g χ^Γ(g)* R(g)
        pass
