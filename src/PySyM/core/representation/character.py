"""类函数与特征标：``χ(g)=Tr ρ(g)``，有限群上内积 ``⟨χ₁,χ₂⟩=(1/|G|)∑_g χ₁(g)χ̄₂(g)``（与量子力学中不可约判据一致）。"""
from typing import Dict, List
import numpy as np
from .abstract_representation import GroupRepresentation
from ..group_theory.abstract_group import Group, GroupElement

class Character:
    """群表示的特征标 ``χ(g) = Tr(ρ(g))``（对矩阵表示）。
    """
    
    def __init__(self, representation: GroupRepresentation):
        self._representation = representation
        self._values = self._compute_character_values()
    
    def _compute_character_values(self) -> Dict[GroupElement, complex]:
        """计算所有群元素的特征标值"""
        return {
            g: np.trace(self._representation(g).matrix)
            for g in self._representation.group.elements()
        }
    
    def __call__(self, element: GroupElement) -> complex:
        """获取群元素的特征标值"""
        return self._values[element]
    
    def inner_product(self, other: 'Character') -> complex:
        """特征标的内积
        
        ⟨χ₁, χ₂⟩ = (1/|G|) Σ χ₁(g)χ₂(g)¯
        """
        group = self._representation.group
        if getattr(other, "_representation", None) is not None and other._representation.group != group:
            raise ValueError("只有同一群上的特征标才能计算内积")
        elements = list(group.elements())
        group_order = len(elements)
        product_sum = 0.0
        for g in elements:
            product_sum += self(g) * np.conj(other(g))
        return product_sum / group_order
    
    def is_irreducible(self) -> bool:
        """检查是否为不可约表示的特征标"""
        return np.isclose(self.inner_product(self), 1.0, atol=1e-10)
    
    def character_table(self) -> List[List[complex]]:
        """计算特征表"""
        # 特征表：每个共轭类的特征值
        group = self._representation.group
        classes = group.conjugacy_classes()
        table = []
        
        # 为每个共轭类计算特征值（使用第一个元素作为代表）
        for cls in classes:
            if cls:
                representative = next(iter(cls))
                char_value = self(representative)
                table.append([char_value])
        
        return table
    
    def is_orthogonal(self, other: 'Character') -> bool:
        """检查两个特征标是否正交"""
        return np.isclose(self.inner_product(other), 0.0, atol=1e-10)
    
    def norm(self) -> float:
        """计算特征标的范数"""
        return abs(self.inner_product(self))
    
    def conjugate(self) -> 'Character':
        """返回共轭特征标"""
        # 创建一个新的表示，其矩阵是原表示矩阵的共轭转置
        # 这里我们通过修改特征值来实现
        class ConjugateCharacter:
            def __init__(self, original_char):
                self._original_char = original_char
            
            def __call__(self, element: GroupElement) -> complex:
                return np.conj(self._original_char(element))
            
            def inner_product(self, other: 'Character') -> complex:
                return np.conj(self._original_char.inner_product(other))
            
            def is_irreducible(self) -> bool:
                return self._original_char.is_irreducible()
        
        return ConjugateCharacter(self)
    
    def tensor_product(self, other: 'Character') -> 'Character':
        """计算两个特征标的张量积"""
        # 特征标的张量积等于特征值的乘积
        class TensorProductCharacter:
            def __init__(self, char1, char2):
                self._char1 = char1
                self._char2 = char2
            
            def __call__(self, element: GroupElement) -> complex:
                return self._char1(element) * self._char2(element)
            
            def inner_product(self, other: 'Character') -> complex:
                # 内积的计算需要访问群元素
                group = self._char1._representation.group
                group_order = len(group.elements())
                product_sum = 0.0
                for g in group.elements():
                    product_sum += self(g) * np.conj(other(g))
                return product_sum / group_order
            
            def is_irreducible(self) -> bool:
                return abs(self.inner_product(self) - 1) < 1e-10
        
        return TensorProductCharacter(self, other)
    
    def restriction(self, subgroup: Group) -> 'Character':
        """将特征标限制到子群"""
        # 这里简化实现，实际应该创建一个子群的表示
        class RestrictedCharacter:
            def __init__(self, original_char, subgroup):
                self._original_char = original_char
                self._subgroup = subgroup
            
            def __call__(self, element: GroupElement) -> complex:
                return self._original_char(element)
            
            def inner_product(self, other: 'Character') -> complex:
                subgroup_order = len(self._subgroup.elements())
                product_sum = 0.0
                for g in self._subgroup.elements():
                    product_sum += self(g) * np.conj(other(g))
                return product_sum / subgroup_order
            
            def is_irreducible(self) -> bool:
                return abs(self.inner_product(self) - 1) < 1e-10
        
        return RestrictedCharacter(self, subgroup)