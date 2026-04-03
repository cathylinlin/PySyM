"""
特征标表存储与查询模块

实现方案：
1. JSON格式存储（易于维护和扩展）
2. 代码内嵌字典（快速访问）
3. 支持查询和计算
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict


# -----------------------------------------------------------------------------
# 1. 特征标表数据结构
# -----------------------------------------------------------------------------

@dataclass
class CharacterTable:
    """特征标表"""
    
    group_name: str              # 群名称（熊夫利符号）
    group_order: int             # 群阶
    irrep_names: List[str]       # 不可约表示名称
    class_names: List[str]       # 共轭类名称
    class_sizes: List[int]       # 每个类的大小
    characters: np.ndarray       # 特征标表（n_irreps × n_classes）
    
    def to_dict(self) -> Dict:
        """转换为字典（可JSON序列化）"""
        return {
            'group_name': self.group_name,
            'group_order': self.group_order,
            'irrep_names': self.irrep_names,
            'class_names': self.class_names,
            'class_sizes': self.class_sizes,
            'characters': self.characters.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CharacterTable':
        """从字典创建"""
        return cls(
            group_name=data['group_name'],
            group_order=data['group_order'],
            irrep_names=data['irrep_names'],
            class_names=data['class_names'],
            class_sizes=data['class_sizes'],
            characters=np.array(data['characters'])
        )
    
    def get_character(self, irrep: str, class_name: str) -> int:
        """获取特定不可约表示在特定类的特征标"""
        irrep_idx = self.irrep_names.index(irrep)
        class_idx = self.class_names.index(class_name)
        return self.characters[irrep_idx, class_idx]
    
    def decompose_representation(self, characters: np.ndarray) -> Dict[str, int]:
        """
        将可约表示分解为不可约表示
        
        使用公式: n_Γ = (1/|G|) Σ_g χ_Γ(g)* χ(g)
        
        Parameters:
            characters: 各共轭类的特征标
        
        Returns:
            {不可约表示名称: 重数}
        """
        decomposition = {}
        
        for i, irrep in enumerate(self.irrep_names):
            # 计算重数
            n = sum(size * self.characters[i, j] * characters[j] 
                   for j, size in enumerate(self.class_sizes))
            n = int(n / self.group_order)
            
            if n > 0:
                decomposition[irrep] = n
        
        return decomposition


# -----------------------------------------------------------------------------
# 2. 点群特征标表数据
# -----------------------------------------------------------------------------

# 完整的点群特征标表数据库
POINT_GROUP_TABLES = {
    # -------------------------------------------------------------------------
    # 非轴向群
    # -------------------------------------------------------------------------
    
    'C1': CharacterTable(
        group_name='C1',
        group_order=1,
        irrep_names=['A'],
        class_names=['E'],
        class_sizes=[1],
        characters=np.array([[1]])
    ),
    
    'Cs': CharacterTable(
        group_name='Cs',
        group_order=2,
        irrep_names=['A\'', 'A"'],
        class_names=['E', 'σh'],
        class_sizes=[1, 1],
        characters=np.array([
            [1,  1],
            [1, -1]
        ])
    ),
    
    'Ci': CharacterTable(
        group_name='Ci',
        group_order=2,
        irrep_names=['Ag', 'Au'],
        class_names=['E', 'i'],
        class_sizes=[1, 1],
        characters=np.array([
            [1,  1],
            [1, -1]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Cn群
    # -------------------------------------------------------------------------
    
    'C2': CharacterTable(
        group_name='C2',
        group_order=2,
        irrep_names=['A', 'B'],
        class_names=['E', 'C2'],
        class_sizes=[1, 1],
        characters=np.array([
            [1,  1],
            [1, -1]
        ])
    ),
    
    'C3': CharacterTable(
        group_name='C3',
        group_order=3,
        irrep_names=['A', 'E'],
        class_names=['E', 'C3', 'C3²'],
        class_sizes=[1, 1, 1],
        characters=np.array([
            [1,   1,    1],
            [2,  -1,   -1]
        ])
    ),
    
    'C4': CharacterTable(
        group_name='C4',
        group_order=4,
        irrep_names=['A', 'B', 'E'],
        class_names=['E', 'C4', 'C2', 'C4³'],
        class_sizes=[1, 1, 1, 1],
        characters=np.array([
            [1,   1,   1,   1],
            [1,  -1,   1,  -1],
            [2,   0,  -2,   0]
        ])
    ),
    
    'C6': CharacterTable(
        group_name='C6',
        group_order=6,
        irrep_names=['A', 'B', 'E1', 'E2'],
        class_names=['E', 'C6', 'C3', 'C2', 'C3²', 'C6⁵'],
        class_sizes=[1, 1, 1, 1, 1, 1],
        characters=np.array([
            [1,   1,   1,   1,   1,   1],
            [1,  -1,   1,  -1,   1,  -1],
            [2,   1,  -1,  -2,  -1,   1],
            [2,  -1,  -1,   2,  -1,  -1]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Cnv群
    # -------------------------------------------------------------------------
    
    'C2v': CharacterTable(
        group_name='C2v',
        group_order=4,
        irrep_names=['A1', 'A2', 'B1', 'B2'],
        class_names=['E', 'C2', 'σv', 'σv\''],
        class_sizes=[1, 1, 1, 1],
        characters=np.array([
            [1,  1,  1,  1],
            [1,  1, -1, -1],
            [1, -1,  1, -1],
            [1, -1, -1,  1]
        ])
    ),
    
    'C3v': CharacterTable(
        group_name='C3v',
        group_order=6,
        irrep_names=['A1', 'A2', 'E'],
        class_names=['E', '2C3', '3σv'],
        class_sizes=[1, 2, 3],
        characters=np.array([
            [1,  1,  1],
            [1,  1, -1],
            [2, -1,  0]
        ])
    ),
    
    'C4v': CharacterTable(
        group_name='C4v',
        group_order=8,
        irrep_names=['A1', 'A2', 'B1', 'B2', 'E'],
        class_names=['E', '2C4', 'C2', '2σv', '2σd'],
        class_sizes=[1, 2, 1, 2, 2],
        characters=np.array([
            [1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1],
            [1, -1,  1,  1, -1],
            [1, -1,  1, -1,  1],
            [2,  0, -2,  0,  0]
        ])
    ),
    
    'C6v': CharacterTable(
        group_name='C6v',
        group_order=12,
        irrep_names=['A1', 'A2', 'B1', 'B2', 'E1', 'E2'],
        class_names=['E', '2C6', '2C3', 'C2', '3σv', '3σd'],
        class_sizes=[1, 2, 2, 1, 3, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1, -1, -1],
            [1, -1,  1, -1,  1, -1],
            [1, -1,  1, -1, -1,  1],
            [2,  1, -1, -2,  0,  0],
            [2, -1, -1,  2,  0,  0]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Cnh群
    # -------------------------------------------------------------------------
    
    'C2h': CharacterTable(
        group_name='C2h',
        group_order=4,
        irrep_names=['Ag', 'Bg', 'Au', 'Bu'],
        class_names=['E', 'C2', 'i', 'σh'],
        class_sizes=[1, 1, 1, 1],
        characters=np.array([
            [1,  1,  1,  1],
            [1, -1,  1, -1],
            [1,  1, -1, -1],
            [1, -1, -1,  1]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Dn群
    # -------------------------------------------------------------------------
    
    'D2': CharacterTable(
        group_name='D2',
        group_order=4,
        irrep_names=['A', 'B1', 'B2', 'B3'],
        class_names=['E', 'C2(z)', 'C2(y)', 'C2(x)'],
        class_sizes=[1, 1, 1, 1],
        characters=np.array([
            [1,  1,  1,  1],
            [1,  1, -1, -1],
            [1, -1,  1, -1],
            [1, -1, -1,  1]
        ])
    ),
    
    'D3': CharacterTable(
        group_name='D3',
        group_order=6,
        irrep_names=['A1', 'A2', 'E'],
        class_names=['E', '2C3', '3C2'],
        class_sizes=[1, 2, 3],
        characters=np.array([
            [1,  1,  1],
            [1,  1, -1],
            [2, -1,  0]
        ])
    ),
    
    'D4': CharacterTable(
        group_name='D4',
        group_order=8,
        irrep_names=['A1', 'A2', 'B1', 'B2', 'E'],
        class_names=['E', '2C4', 'C2', '2C2\'', '2C2"'],
        class_sizes=[1, 2, 1, 2, 2],
        characters=np.array([
            [1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1],
            [1, -1,  1,  1, -1],
            [1, -1,  1, -1,  1],
            [2,  0, -2,  0,  0]
        ])
    ),
    
    'D6': CharacterTable(
        group_name='D6',
        group_order=12,
        irrep_names=['A1', 'A2', 'B1', 'B2', 'E1', 'E2'],
        class_names=['E', '2C6', '2C3', 'C2', '3C2\'', '3C2"'],
        class_sizes=[1, 2, 2, 1, 3, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1, -1, -1],
            [1, -1,  1, -1,  1, -1],
            [1, -1,  1, -1, -1,  1],
            [2,  1, -1, -2,  0,  0],
            [2, -1, -1,  2,  0,  0]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Dnh群
    # -------------------------------------------------------------------------
    
    'D2h': CharacterTable(
        group_name='D2h',
        group_order=8,
        irrep_names=['Ag', 'B1g', 'B2g', 'B3g', 'Au', 'B1u', 'B2u', 'B3u'],
        class_names=['E', 'C2(z)', 'C2(y)', 'C2(x)', 'i', 'σ(xy)', 'σ(xz)', 'σ(yz)'],
        class_sizes=[1, 1, 1, 1, 1, 1, 1, 1],
        characters=np.array([
            [1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1,  1, -1, -1],
            [1, -1,  1, -1,  1, -1,  1, -1],
            [1, -1, -1,  1,  1, -1, -1,  1],
            [1,  1,  1,  1, -1, -1, -1, -1],
            [1,  1, -1, -1, -1, -1,  1,  1],
            [1, -1,  1, -1, -1,  1, -1,  1],
            [1, -1, -1,  1, -1,  1,  1, -1]
        ])
    ),
    
    'D3h': CharacterTable(
        group_name='D3h',
        group_order=12,
        irrep_names=['A1\'', 'A2\'', 'E\'', 'A1"', 'A2"', 'E"'],
        class_names=['E', '2C3', '3C2', 'σh', '2S3', '3σv'],
        class_sizes=[1, 2, 3, 1, 2, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1],
            [1,  1, -1,  1,  1, -1],
            [2, -1,  0,  2, -1,  0],
            [1,  1,  1, -1, -1, -1],
            [1,  1, -1, -1, -1,  1],
            [2, -1,  0, -2,  1,  0]
        ])
    ),
    
    'D4h': CharacterTable(
        group_name='D4h',
        group_order=16,
        irrep_names=['A1g', 'A2g', 'B1g', 'B2g', 'Eg', 'A1u', 'A2u', 'B1u', 'B2u', 'Eu'],
        class_names=['E', '2C4', 'C2', '2C2\'', '2C2"', 'i', '2S4', 'σh', '2σv', '2σd'],
        class_sizes=[1, 2, 1, 2, 2, 1, 2, 1, 2, 2],
        characters=np.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1,  1,  1,  1, -1, -1],
            [1, -1,  1,  1, -1,  1, -1,  1,  1, -1],
            [1, -1,  1, -1,  1,  1, -1,  1, -1,  1],
            [2,  0, -2,  0,  0,  2,  0, -2,  0,  0],
            [1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
            [1,  1,  1, -1, -1, -1, -1, -1,  1,  1],
            [1, -1,  1,  1, -1, -1,  1, -1, -1,  1],
            [1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
            [2,  0, -2,  0,  0, -2,  0,  2,  0,  0]
        ])
    ),
    
    'D6h': CharacterTable(
        group_name='D6h',
        group_order=24,
        irrep_names=['A1g', 'A2g', 'B1g', 'B2g', 'E1g', 'E2g',
                    'A1u', 'A2u', 'B1u', 'B2u', 'E1u', 'E2u'],
        class_names=['E', '2C6', '2C3', 'C2', '3C2\'', '3C2"', 'i', '2S3', '2S6', 'σh', '3σd', '3σv'],
        class_sizes=[1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
            [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
            [1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
            [2,  1, -1, -2,  0,  0,  2,  1, -1, -2,  0,  0],
            [2, -1, -1,  2,  0,  0,  2, -1, -1,  2,  0,  0],
            [1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
            [1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1],
            [1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
            [1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
            [2,  1, -1, -2,  0,  0, -2, -1,  1,  2,  0,  0],
            [2, -1, -1,  2,  0,  0, -2,  1,  1, -2,  0,  0]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # Dnd群
    # -------------------------------------------------------------------------
    
    'D2d': CharacterTable(
        group_name='D2d',
        group_order=8,
        irrep_names=['A1', 'A2', 'B1', 'B2', 'E'],
        class_names=['E', '2S4', 'C2', '2C2\'', '2σd'],
        class_sizes=[1, 2, 1, 2, 2],
        characters=np.array([
            [1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1],
            [1, -1,  1,  1, -1],
            [1, -1,  1, -1,  1],
            [2,  0, -2,  0,  0]
        ])
    ),
    
    'D3d': CharacterTable(
        group_name='D3d',
        group_order=12,
        irrep_names=['A1g', 'A2g', 'Eg', 'A1u', 'A2u', 'Eu'],
        class_names=['E', '2C3', '3C2', 'i', '2S6', '3σd'],
        class_sizes=[1, 2, 3, 1, 2, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1],
            [1,  1, -1,  1,  1, -1],
            [2, -1,  0,  2, -1,  0],
            [1,  1,  1, -1, -1, -1],
            [1,  1, -1, -1, -1,  1],
            [2, -1,  0, -2,  1,  0]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # 立方群
    # -------------------------------------------------------------------------
    
    'T': CharacterTable(
        group_name='T',
        group_order=12,
        irrep_names=['A', 'E', 'T'],
        class_names=['E', '4C3', '4C3²', '3C2'],
        class_sizes=[1, 4, 4, 3],
        characters=np.array([
            [1,  1,  1,  1],
            [2, -1, -1,  2],
            [3,  0,  0, -1]
        ])
    ),
    
    'Th': CharacterTable(
        group_name='Th',
        group_order=24,
        irrep_names=['Ag', 'Eg', 'Tg', 'Au', 'Eu', 'Tu'],
        class_names=['E', '4C3', '4C3²', '3C2', 'i', '4S6', '4S6⁵', '3σh'],
        class_sizes=[1, 4, 4, 3, 1, 4, 4, 3],
        characters=np.array([
            [1,  1,  1,  1,  1,  1,  1,  1],
            [2, -1, -1,  2,  2, -1, -1,  2],
            [3,  0,  0, -1,  3,  0,  0, -1],
            [1,  1,  1,  1, -1, -1, -1, -1],
            [2, -1, -1,  2, -2,  1,  1, -2],
            [3,  0,  0, -1, -3,  0,  0,  1]
        ])
    ),
    
    'Td': CharacterTable(
        group_name='Td',
        group_order=24,
        irrep_names=['A1', 'A2', 'E', 'T1', 'T2'],
        class_names=['E', '8C3', '3C2', '6S4', '6σd'],
        class_sizes=[1, 8, 3, 6, 6],
        characters=np.array([
            [1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1],
            [2, -1,  2,  0,  0],
            [3,  0, -1,  1, -1],
            [3,  0, -1, -1,  1]
        ])
    ),
    
    'O': CharacterTable(
        group_name='O',
        group_order=24,
        irrep_names=['A1', 'A2', 'E', 'T1', 'T2'],
        class_names=['E', '8C3', '3C2', '6C4', '6C2\''],
        class_sizes=[1, 8, 3, 6, 6],
        characters=np.array([
            [1,  1,  1,  1,  1],
            [1,  1,  1, -1, -1],
            [2, -1,  2,  0,  0],
            [3,  0, -1,  1, -1],
            [3,  0, -1, -1,  1]
        ])
    ),
    
    'Oh': CharacterTable(
        group_name='Oh',
        group_order=48,
        irrep_names=['A1g', 'A2g', 'Eg', 'T1g', 'T2g', 
                    'A1u', 'A2u', 'Eu', 'T1u', 'T2u'],
        class_names=['E', '8C3', '6C2', '6C4', '3C2(=C4²)', 
                    'i', '6S4', '8S6', '3σh', '6σd'],
        class_sizes=[1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
        characters=np.array([
            [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [1,  1, -1, -1,  1,  1, -1,  1,  1, -1],
            [2, -1,  0,  0,  2,  2,  0, -1,  2,  0],
            [3,  0, -1,  1, -1,  3,  1,  0, -1, -1],
            [3,  0,  1, -1, -1,  3, -1,  0, -1,  1],
            [1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
            [1,  1, -1, -1,  1, -1,  1, -1, -1,  1],
            [2, -1,  0,  0,  2, -2,  0,  1, -2,  0],
            [3,  0, -1,  1, -1, -3, -1,  0,  1,  1],
            [3,  0,  1, -1, -1, -3,  1,  0,  1, -1]
        ])
    ),
    
    # -------------------------------------------------------------------------
    # 线性分子群
    # -------------------------------------------------------------------------
    
    'C∞v': CharacterTable(
        group_name='C∞v',
        group_order=-1,  # 无限群
        irrep_names=['A1=Σ⁺', 'A2=Σ⁻', 'E1=Π', 'E2=Δ', 'E3=Φ'],
        class_names=['E', '2C∞^φ', '...'],
        class_sizes=[1, 2, -1],
        characters=np.array([
            [1,  1, -1],
            [1,  1, -1],
            [2,  2*np.cos(1), -1],
            [2,  2*np.cos(2), -1],
            [2,  2*np.cos(3), -1]
        ])
    ),
    
    'D∞h': CharacterTable(
        group_name='D∞h',
        group_order=-1,  # 无限群
        irrep_names=['Σg⁺', 'Σg⁻', 'Πg', 'Δg', 'Σu⁺', 'Σu⁻', 'Πu', 'Δu'],
        class_names=['E', '2C∞^φ', '...'],
        class_sizes=[1, 2, -1],
        characters=np.array([
            [1,  1, -1],
            [1,  1, -1],
            [2,  2*np.cos(1), -1],
            [2,  2*np.cos(2), -1],
            [1,  1, -1],
            [1,  1, -1],
            [2,  2*np.cos(1), -1],
            [2,  2*np.cos(2), -1]
        ])
    )
}


# -----------------------------------------------------------------------------
# 3. 特征标表管理器
# -----------------------------------------------------------------------------

class CharacterTableDatabase:
    """特征标表数据库"""
    
    def __init__(self):
        self._tables = POINT_GROUP_TABLES.copy()
        self._file_path = None
    
    def get_table(self, group_name: str) -> CharacterTable:
        """获取特征标表"""
        # 标准化群名称
        group_name = self._normalize_name(group_name)
        
        if group_name not in self._tables:
            raise ValueError(f"群 {group_name} 的特征标表未找到")
        
        return self._tables[group_name]
    
    def _normalize_name(self, name: str) -> str:
        """标准化群名称"""
        # 移除空格
        name = name.replace(' ', '')
        
        # 处理下标
        name = name.replace('∞', '∞')
        
        return name
    
    def list_groups(self) -> List[str]:
        """列出所有可用群"""
        return list(self._tables.keys())
    
    def add_table(self, table: CharacterTable):
        """添加新的特征标表"""
        self._tables[table.group_name] = table
    
    def save_to_json(self, file_path: str):
        """保存到JSON文件"""
        data = {
            name: table.to_dict() 
            for name, table in self._tables.items()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self._file_path = file_path
    
    def load_from_json(self, file_path: str):
        """从JSON文件加载"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._tables = {
            name: CharacterTable.from_dict(table_data)
            for name, table_data in data.items()
        }
        
        self._file_path = file_path
    
    def search_by_order(self, order: int) -> List[str]:
        """按群阶搜索"""
        return [
            name for name, table in self._tables.items()
            if table.group_order == order
        ]
    
    def search_by_irrep(self, irrep_name: str) -> List[str]:
        """按不可约表示搜索"""
        return [
            name for name, table in self._tables.items()
            if irrep_name in table.irrep_names
        ]
    
    def get_product_table(self, group_name: str) -> Dict[str, Dict[str, str]]:
        """
        获取直积表
        
        Returns:
            {irrep1: {irrep2: decomposition}}
        """
        # 需要计算所有直积
        table = self.get_table(group_name)
        
        product_table = {}
        for irrep1 in table.irrep_names:
            product_table[irrep1] = {}
            for irrep2 in table.irrep_names:
                # 计算直积
                chars1 = table.characters[table.irrep_names.index(irrep1)]
                chars2 = table.characters[table.irrep_names.index(irrep2)]
                product_chars = chars1 * chars2
                
                # 分解
                decomp = table.decompose_representation(product_chars)
                product_table[irrep1][irrep2] = ' + '.join(
                    f'{count}{irrep}' if count > 1 else irrep
                    for irrep, count in decomp.items()
                )
        
        return product_table


# -----------------------------------------------------------------------------
# 4. 直积计算
# -----------------------------------------------------------------------------

class DirectProductCalculator:
    """直积计算器"""
    
    def __init__(self, database: CharacterTableDatabase = None):
        self.db = database or CharacterTableDatabase()
    
    def calculate(self, group_name: str, *irreps: str) -> Dict[str, int]:
        """
        计算不可约表示的直积
        
        Parameters:
            group_name: 群名称
            irreps: 不可约表示名称
        
        Returns:
            分解结果 {不可约表示: 重数}
        """
        table = self.db.get_table(group_name)
        
        # 初始化为第一个表示的特征标
        result_chars = table.characters[table.irrep_names.index(irreps[0])].copy()
        
        # 依次乘以其他表示
        for irrep in irreps[1:]:
            chars = table.characters[table.irrep_names.index(irrep)]
            result_chars = result_chars * chars
        
        # 分解
        return table.decompose_representation(result_chars)
    
    def contains_identity(self, group_name: str, *irreps: str) -> bool:
        """
        检查直积是否包含全对称表示
        
        用于选择定则判断
        """
        decomp = self.calculate(group_name, *irreps)
        
        # 检查是否包含A1或A1g
        identity_names = ['A1', 'A1g', 'A\'', 'Ag']
        
        return any(name in decomp for name in identity_names)


# -----------------------------------------------------------------------------
# 5. 使用示例和工具函数
# -----------------------------------------------------------------------------

def print_character_table(group_name: str):
    """打印特征标表"""
    db = CharacterTableDatabase()
    table = db.get_table(group_name)
    
    print(f"\n{group_name} 点群特征标表")
    print("=" * 60)
    
    # 打印表头
    header = "         " + "  ".join(f"{name:>8}" for name in table.class_names)
    print(header)
    
    # 打印类大小
    sizes = "类大小:  " + "  ".join(f"{size:>8}" for size in table.class_sizes)
    print(sizes)
    print("-" * 60)
    
    # 打印特征标
    for i, irrep in enumerate(table.irrep_names):
        row = f"{irrep:8}" + "  ".join(f"{int(ch):>8}" for ch in table.characters[i])
        print(row)
    
    print("=" * 60)


def find_allowed_transitions(group_name: str, 
                            initial_irrep: str, 
                            final_irrep: str,
                            operator_irreps: List[str]) -> bool:
    """
    根据群论选择定则判断跃迁是否允许
    """
    calc = DirectProductCalculator()
    
    for op_irrep in operator_irreps:
        # 检查 initial ⊗ operator ⊗ final 是否包含全对称表示
        if calc.contains_identity(group_name, initial_irrep, op_irrep, final_irrep):
            return True
    
    return False

