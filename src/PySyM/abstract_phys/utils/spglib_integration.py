"""
spglib集成模块

提供与spglib的无缝对接，包括：
- 空间群识别
- 对称操作提取
- 原子位置对称性分析
- 晶格优化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass

# 尝试导入spglib
try:
    import spglib
    SPGLIB_AVAILABLE = True
except ImportError:
    SPGLIB_AVAILABLE = False
    spglib = None

from ..symmetry_environments.discrete_symmetries.space_group import SpaceGroup, SpaceGroupOperation, BravaisLattice
from ..symmetry_environments.discrete_symmetries.point_groups import PointGroup, PointGroupOperation


# -----------------------------------------------------------------------------
# 1. 数据结构适配
# -----------------------------------------------------------------------------

@dataclass
class CrystalStructure:
    """
    晶体结构
    
    spglib需要的格式：lattice, positions, numbers
    """
    lattice: np.ndarray        # 3×3格矢矩阵
    positions: np.ndarray      # N×3分数坐标
    numbers: np.ndarray        # N个原子序数
    magmoms: Optional[np.ndarray] = None  # 磁矩（可选）
    
    def to_spglib_cell(self) -> Tuple:
        """转换为spglib格式"""
        cell = (self.lattice, self.positions, self.numbers)
        if self.magmoms is not None:
            return (self.lattice, self.positions, self.numbers, self.magmoms)
        return cell
    
    @classmethod
    def from_spglib_cell(cls, cell: Tuple) -> 'CrystalStructure':
        """从spglib格式创建"""
        if len(cell) == 3:
            lattice, positions, numbers = cell
            return cls(lattice, positions, numbers)
        elif len(cell) == 4:
            lattice, positions, numbers, magmoms = cell
            return cls(lattice, positions, numbers, magmoms)
    
    @classmethod
    def from_xyz(cls, 
                 lattice_vectors: np.ndarray,
                 atomic_positions: np.ndarray,
                 atomic_symbols: List[str]) -> 'CrystalStructure':
        """从笛卡尔坐标创建"""
        # 符号转原子序数
        from periodictable import elements
        numbers = np.array([getattr(elements, sym).number for sym in atomic_symbols])
        
        # 转换为分数坐标
        inv_lattice = np.linalg.inv(lattice_vectors)
        positions = (inv_lattice @ atomic_positions.T).T
        
        return cls(lattice_vectors, positions, numbers)


# -----------------------------------------------------------------------------
# 2. spglib适配器类
# -----------------------------------------------------------------------------

class SpglibAdapter:
    """
    spglib适配器
    
    封装spglib功能，提供统一接口
    """
    
    def __init__(self):
        if not SPGLIB_AVAILABLE:
            raise ImportError(
                "spglib未安装。请运行: pip install spglib\n"
                "或使用conda: conda install -c conda-forge spglib"
            )
        
        self.spglib = spglib
    
    # -------------------------------------------------------------------------
    # 基本对称性分析
    # -------------------------------------------------------------------------
    
    def get_spacegroup(self, 
                       structure: CrystalStructure,
                       symprec: float = 1e-5,
                       angle_tolerance: float = -1.0) -> Dict[str, Any]:
        """
        识别空间群
        
        Parameters:
            structure: 晶体结构
            symprec: 对称性判断精度（Å）
            angle_tolerance: 角度容差（度），-1表示自动
        
        Returns:
            {
                'number': 空间群编号 (1-230),
                'symbol': 国际符号（如 'Fm-3m'）,
                'hall_number': Hall编号,
                'international': 完整国际符号,
                'hall': Hall符号,
                'choice': 设置选择
            }
        """
        cell = structure.to_spglib_cell()
        
        # 获取空间群信息
        dataset = self.spglib.get_symmetry_dataset(
            cell, 
            symprec=symprec,
            angle_tolerance=angle_tolerance
        )
        
        return {
            'number': dataset['number'],
            'symbol': dataset['international_short'],
            'hall_number': dataset['hall_number'],
            'international': dataset['international'],
            'hall': dataset['hall'],
            'choice': dataset['choice'],
            'pointgroup': dataset['pointgroup'],
            'transformation_matrix': dataset['transformation_matrix'],
            'origin_shift': dataset['origin_shift']
        }
    
    def get_symmetry_operations(self,
                                structure: CrystalStructure,
                                symprec: float = 1e-5) -> List[SpaceGroupOperation]:
        """
        提取对称操作
        
        Returns:
            对称操作列表（旋转+平移）
        """
        cell = structure.to_spglib_cell()
        
        symmetry = self.spglib.get_symmetry(cell, symprec=symprec)
        
        operations = []
        for rotation, translation in zip(symmetry['rotations'], symmetry['translations']):
            op = SpaceGroupOperation(
                rotation=rotation,
                translation=translation
            )
            operations.append(op)
        
        return operations
    
    def get_symmetry_dataset(self,
                            structure: CrystalStructure,
                            symprec: float = 1e-5) -> Dict[str, Any]:
        """
        获取完整的对称性数据集
        
        包含Wyckoff位置、等效原子等信息
        """
        cell = structure.to_spglib_cell()
        dataset = self.spglib.get_symmetry_dataset(cell, symprec=symprec)
        
        return {
            # 空间群信息
            'spacegroup_number': dataset['number'],
            'international_short': dataset['international_short'],
            'international_full': dataset['international'],
            'hall_symbol': dataset['hall'],
            'pointgroup': dataset['pointgroup'],
            
            # 对称操作
            'n_operations': dataset['n_operations'],
            'rotations': dataset['rotations'],
            'translations': dataset['translations'],
            
            # Wyckoff位置
            'wyckoffs': dataset['wyckoffs'],  # Wyckoff字母
            'site_symmetry_symbols': dataset['site_symmetry_symbols'],
            'equivalent_atoms': dataset['equivalent_atoms'],
            
            # 原始晶胞信息
            'transformation_matrix': dataset['transformation_matrix'],
            'origin_shift': dataset['origin_shift'],
            
            # 标准化晶胞
            'std_lattice': dataset['std_lattice'],
            'std_positions': dataset['std_positions'],
            'std_types': dataset['std_types'],
            'std_rotation': dataset['std_rotation'],
            
            # 其他
            'mapping_to_primitive': dataset['mapping_to_primitive'],
            'mapping_to_std': dataset.get('mapping_to_std', None)
        }
    
    # -------------------------------------------------------------------------
    # 晶胞标准化和约化
    # -------------------------------------------------------------------------
    
    def find_primitive(self,
                      structure: CrystalStructure,
                      symprec: float = 1e-5) -> CrystalStructure:
        """
        找到原胞
        
        将晶胞约化为最小的周期单元
        """
        cell = structure.to_spglib_cell()
        
        primitive = self.spglib.find_primitive(cell, symprec=symprec)
        
        if primitive is None:
            return None
        
        return CrystalStructure.from_spglib_cell(primitive)
    
    def standardize_cell(self,
                        structure: CrystalStructure,
                        to_primitive: bool = False,
                        no_idealize: bool = False,
                        symprec: float = 1e-5) -> CrystalStructure:
        """
        标准化晶胞
        
        Parameters:
            to_primitive: 是否转换为原胞
            no_idealize: 是否不理想化坐标
        """
        cell = structure.to_spglib_cell()
        
        standardized = self.spglib.standardize_cell(
            cell,
            to_primitive=to_primitive,
            no_idealize=no_idealize,
            symprec=symprec
        )
        
        if standardized is None:
            return None
        
        return CrystalStructure.from_spglib_cell(standardized)
    
    def refine_cell(self,
                   structure: CrystalStructure,
                   symprec: float = 1e-5) -> CrystalStructure:
        """
        优化晶胞
        
        调整原子位置以精确满足对称性
        """
        cell = structure.to_spglib_cell()
        
        refined = self.spglib.refine_cell(cell, symprec=symprec)
        
        if refined is None:
            return None
        
        return CrystalStructure.from_spglib_cell(refined)
    
    # -------------------------------------------------------------------------
    # 对称性相关计算
    # -------------------------------------------------------------------------
    
    def get_ir_reciprocal_mesh(self,
                              structure: CrystalStructure,
                              mesh: Union[int, Tuple[int, int, int]],
                              is_shift: Tuple[int, int, int] = (0, 0, 0),
                              is_time_reversal: bool = True,
                              symprec: float = 1e-5) -> Dict[str, Any]:
        """
        生成不可约布里渊区k点网格
        
        Parameters:
            mesh: 网格大小，可以是整数或三元组
            is_shift: 是否偏移网格
            is_time_reversal: 是否考虑时间反演
        
        Returns:
            {
                'grid_points': k点网格坐标,
                'weights': 每个k点的权重,
                'grid_mapping_table': 映射表
            }
        """
        if isinstance(mesh, int):
            mesh = (mesh, mesh, mesh)
        
        cell = structure.to_spglib_cell()
        
        mapping, grid_points = self.spglib.get_ir_reciprocal_mesh(
            mesh,
            cell,
            is_shift=is_shift,
            is_time_reversal=is_time_reversal,
            symprec=symprec
        )
        
        # 统计权重
        unique, counts = np.unique(mapping, return_counts=True)
        weights = counts / len(mapping)
        
        return {
            'grid_points': grid_points,
            'grid_mapping_table': mapping,
            'unique_points': unique,
            'weights': weights
        }
    
    def get_symmetry_from_database(self, hall_number: int) -> Dict[str, Any]:
        """
        从数据库获取空间群对称性
        
        Parameters:
            hall_number: Hall编号 (1-530)
        """
        dataset = self.spglib.get_spacegroup_type(hall_number)
        
        return {
            'number': dataset['number'],
            'international_short': dataset['international_short'],
            'international_full': dataset['international'],
            'international': dataset['international'],
            'schoenflies': dataset['schoenflies'],
            'hall_number': dataset['hall_number'],
            'hall_symbol': dataset['hall_symbol'],
            'choice': dataset['choice'],
            'pointgroup_international': dataset['pointgroup_international'],
            'pointgroup_schoenflies': dataset['pointgroup_schoenflies'],
            'arithmetic_crystal_class_number': dataset['arithmetic_crystal_class_number'],
            'arithmetic_crystal_class_symbol': dataset['arithmetic_crystal_class_symbol']
        }
    
    # -------------------------------------------------------------------------
    # 原子位置对称性
    # -------------------------------------------------------------------------
    
    def get_site_symmetry(self,
                         structure: CrystalStructure,
                         atom_index: int,
                         symprec: float = 1e-5) -> List[SpaceGroupOperation]:
        """
        获取特定原子位置的点群对称性
        
        返回保持该原子位置不变的对称操作
        """
        dataset = self.get_symmetry_dataset(structure, symprec)
        
        all_operations = [
            SpaceGroupOperation(r, t)
            for r, t in zip(dataset['rotations'], dataset['translations'])
        ]
        
        position = structure.positions[atom_index]
        site_ops = []
        
        for op in all_operations:
            # 应用操作
            new_pos = op.apply(position)
            new_pos = new_pos % 1.0  # 模1
            
            # 检查是否回到原位
            if np.allclose(new_pos, position % 1.0, atol=symprec):
                site_ops.append(op)
        
        return site_ops
    
    def get_equivalent_atoms(self,
                            structure: CrystalStructure,
                            symprec: float = 1e-5) -> List[List[int]]:
        """
        找出对称等价原子组
        
        Returns:
            [[原子索引组1], [原子索引组2], ...]
        """
        dataset = self.get_symmetry_dataset(structure, symprec)
        equivalent = dataset['equivalent_atoms']
        
        # 组织成组
        groups = {}
        for i, eq in enumerate(equivalent):
            if eq not in groups:
                groups[eq] = []
            groups[eq].append(i)
        
        return list(groups.values())
    
    # -------------------------------------------------------------------------
    # 磁对称性
    # -------------------------------------------------------------------------
    
    def get_magnetic_symmetry(self,
                             structure: CrystalStructure,
                             symprec: float = 1e-5,
                             angle_tolerance: float = -1.0,
                             mag_symprec: float = None) -> Dict[str, Any]:
        """
        磁空间群分析
        
        需要structure.magmoms
        """
        if structure.magmoms is None:
            raise ValueError("需要提供磁矩信息")
        
        cell = structure.to_spglib_cell()
        
        if mag_symprec is None:
            mag_symprec = symprec
        
        dataset = self.spglib.get_magnetic_symmetry_dataset(
            cell,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec
        )
        
        return dataset


# -----------------------------------------------------------------------------
# 3. 与框架的集成
# -----------------------------------------------------------------------------

class SpglibSpaceGroup(SpaceGroup):
    """
    基于spglib的空间群类
    
    继承自SpaceGroup，使用spglib数据
    """
    
    def __init__(self, spacegroup_number: int, adapter: SpglibAdapter = None):
        if adapter is None:
            adapter = SpglibAdapter()
        
        # 从spglib获取数据
        hall_number = self._get_hall_number(spacegroup_number)
        data = adapter.get_symmetry_from_database(hall_number)
        
        self._adapter = adapter
        self._hall_number = hall_number
        self._data = data
        
        # 初始化基类
        # 注意：这里需要调用适当的基类构造函数
        # super().__init__(...)
    
    @staticmethod
    def _get_hall_number(spg_number: int) -> int:
        """空间群编号到Hall编号的映射"""
        # 这是简化的映射，实际需要完整表
        # spglib内部有这个映射
        hall_numbers = {
            1: 1,      # P1
            2: 2,      # P-1
            225: 523,  # Fm-3m
            229: 529,  # Im-3m
            # ... 完整映射
        }
        return hall_numbers.get(spg_number, spg_number)
    
    def get_operations(self) -> List[SpaceGroupOperation]:
        """获取所有对称操作"""
        # 使用spglib获取
        from spglib import get_symmetry_from_database
        symmetry = get_symmetry_from_database(self._hall_number)
        
        operations = []
        for r, t in zip(symmetry['rotations'], symmetry['translations']):
            operations.append(SpaceGroupOperation(r, t))
        
        return operations


# -----------------------------------------------------------------------------
# 4. 便捷函数
# -----------------------------------------------------------------------------

def analyze_crystal(structure: CrystalStructure, 
                   symprec: float = 1e-5) -> Dict[str, Any]:
    """
    一键分析晶体对称性
    
    返回完整的对称性分析结果
    """
    adapter = SpglibAdapter()
    
    # 空间群
    sg_info = adapter.get_spacegroup(structure, symprec)
    
    # 对称操作
    operations = adapter.get_symmetry_operations(structure, symprec)
    
    # Wyckoff位置
    dataset = adapter.get_symmetry_dataset(structure, symprec)
    
    # 等价原子
    eq_atoms = adapter.get_equivalent_atoms(structure, symprec)
    
    # 原胞
    primitive = adapter.find_primitive(structure, symprec)
    
    return {
        'spacegroup': sg_info,
        'n_symmetry_operations': len(operations),
        'symmetry_operations': operations,
        'wyckoffs': dataset['wyckoffs'],
        'site_symmetries': dataset['site_symmetry_symbols'],
        'equivalent_atom_groups': eq_atoms,
        'primitive_cell': primitive,
        'pointgroup': sg_info['pointgroup']
    }


def quick_spacegroup(lattice: np.ndarray,
                    positions: np.ndarray,
                    numbers: np.ndarray,
                    symprec: float = 1e-5) -> Tuple[int, str]:
    """
    快速识别空间群
    
    Parameters:
        lattice: 3×3格矢矩阵
        positions: N×3分数坐标
        numbers: N个原子序数
    
    Returns:
        (空间群编号, 国际符号)
    """
    structure = CrystalStructure(lattice, positions, numbers)
    adapter = SpglibAdapter()
    info = adapter.get_spacegroup(structure, symprec)
    
    return info['number'], info['symbol']


# -----------------------------------------------------------------------------
# 5. 使用示例
# -----------------------------------------------------------------------------

def example_usage():
    """使用示例"""
    
    # 示例1: 面心立方铝
    lattice_fcc = np.array([
        [0, 2.0, 2.0],
        [2.0, 0, 2.0],
        [2.0, 2.0, 0]
    ])
    positions_fcc = np.array([[0, 0, 0]])
    numbers_fcc = np.array([13])  # Al
    
    structure_fcc = CrystalStructure(lattice_fcc, positions_fcc, numbers_fcc)
    
    # 分析
    result = analyze_crystal(structure_fcc)
    print(f"空间群: {result['spacegroup']['symbol']}")
    print(f"点群: {result['pointgroup']}")
    print(f"对称操作数: {result['n_symmetry_operations']}")
    
    # 示例2: 使用spglib适配器
    adapter = SpglibAdapter()
    
    # 获取对称操作
    ops = adapter.get_symmetry_operations(structure_fcc)
    
    # 标准化晶胞
    std_structure = adapter.standardize_cell(structure_fcc)
    
    # 生成k点网格
    kpoints = adapter.get_ir_reciprocal_mesh(structure_fcc, mesh=4)
    print(f"不可约k点数: {len(kpoints['unique_points'])}")

