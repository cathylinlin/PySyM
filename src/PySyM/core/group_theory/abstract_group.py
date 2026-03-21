"""抽象群论基类模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Tuple
from dataclasses import dataclass


T = TypeVar('T')  # 群元素类型


@dataclass
class GroupProperties:
    """群属性数据类"""
    order: int                    # 群的阶
    is_abelian: bool              # 是否阿贝尔
    is_simple: bool               # 是否单群
    is_solvable: bool             # 是否可解
    center_order: int             # 中心阶数
    conjugacy_classes: int        # 共轭类数目
    generators: List[T]           # 生成元


class GroupElement(ABC):
    """群元素抽象基类"""
    
    @abstractmethod
    def __mul__(self, other: 'GroupElement') -> 'GroupElement':
        """群乘法"""
        pass
    
    @abstractmethod
    def __pow__(self, n: int) -> 'GroupElement':
        """幂运算"""
        pass
    
    @abstractmethod
    def inverse(self) -> 'GroupElement':
        """逆元"""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """必须可哈希，以便存入集合或作为字典键"""
        pass

    @abstractmethod
    def is_identity(self) -> bool:
        """是否为单位元"""
        pass
    
    @abstractmethod
    def order(self) -> int:
        """元素阶数"""
        pass
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupElement):
            return NotImplemented
        return hash(self) == hash(other)
    
    def __truediv__(self, other: 'GroupElement') -> 'GroupElement':
        """a/b = a * b^(-1)"""
        return self * other.inverse()


class Group(ABC, Generic[T]):
    """群抽象基类"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self._cache: Dict[str, any] = {}
    
    @abstractmethod
    def identity(self) -> T:
        """单位元"""
        pass
    
    @abstractmethod
    def multiply(self, a: T, b: T) -> T:
        """群乘法"""
        pass
    
    @abstractmethod
    def inverse(self, a: T) -> T:
        """逆元"""
        pass

    @abstractmethod
    def __contains__(self, element: T) -> bool:
        """判断元素是否属于该群"""
        pass

    @abstractmethod
    def order(self) -> int:
        """群的阶（无限群返回-1或抛出异常）"""
        pass
    
    def is_abelian(self) -> bool:
        """检查是否为阿贝尔群"""
        if self.order() > 1000:  # 大群采样检查
            return self._sample_abelian_check()
        elems = list(self.elements())
        for i, a in enumerate(elems):
            for b in elems[i:]:
                if self.multiply(a, b) != self.multiply(b, a):
                    return False
        return True
    
    def _sample_abelian_check(self, samples: int = 100) -> bool:
        """大群采样检查阿贝尔性"""
        import random
        elems = list(self.elements())
        for _ in range(samples):
            a = random.choice(elems)
            b = random.choice(elems)
            if self.multiply(a, b) != self.multiply(b, a):
                return False
        return True
    
    @abstractmethod
    def elements(self) -> List[T]:
        """所有群元素（有限群）"""
        pass
    
    def generate_subgroup(self, generators: List[T]) -> 'Subgroup[T]':
        """由生成元生成的子群
        
        Args:
            generators: 生成元列表
            
        Returns:
            由生成元生成的子群
        """
        if not generators:
            return Subgroup(self, [self.identity()])
        
        for g in generators:
            if g not in self:
                raise ValueError(f"元素 {g} 不属于该群")
        
        subgroup_elements: set = set()
        subgroup_elements.add(self.identity())
        
        for g in generators:
            subgroup_elements.add(g)
        
        changed = True
        while changed:
            changed = False
            current_elements = list(subgroup_elements)
            for a in current_elements:
                for b in current_elements:
                    product = self.multiply(a, b)
                    if product not in subgroup_elements:
                        subgroup_elements.add(product)
                        changed = True
                    
                    inv_product = self.multiply(a, self.inverse(b))
                    if inv_product not in subgroup_elements:
                        subgroup_elements.add(inv_product)
                        changed = True
        
        return Subgroup(self, list(subgroup_elements))
    
    def conjugacy_classes(self) -> List[List[T]]:
        """计算共轭类"""
        if self.order() > 500:
            raise ValueError("群太大，无法计算共轭类")
        
        elems = list(self.elements())
        unprocessed = set(elems)
        classes = []
        
        while unprocessed:
            rep = unprocessed.pop()
            conj_class = {rep}
            
            for g in elems:
                for h in list(conj_class):
                    # g * h * g^(-1)
                    conj = self.multiply(g, self.multiply(h, self.inverse(g)))
                    conj_class.add(conj)
            
            classes.append(list(conj_class))
            unprocessed -= conj_class
        
        return classes
    
    def center(self) -> List[T]:
        """计算群中心"""
        elems = list(self.elements())
        center = []
        for a in elems:
            if all(self.multiply(a, b) == self.multiply(b, a) for b in elems):
                center.append(a)
        return center
    
    def multiplication_table(self) -> str:
        """生成乘法表
        
        Returns:
            格式化的乘法表字符串
        """
        elems = list(self.elements())
        
        if not elems:
            return "Empty group"
        
        max_elem_len = max(len(str(e)) for e in elems)
        cell_width = max(max_elem_len + 2, 4)
        
        lines = []
        
        header = " " * cell_width + "|"
        for elem in elems:
            header += f"{str(elem):^{cell_width}}"
        lines.append(header)
        lines.append("-" * len(header))
        
        for a in elems:
            row = f"{str(a):^{cell_width}}|"
            for b in elems:
                product = self.multiply(a, b)
                row += f"{str(product):^{cell_width}}"
            lines.append(row)
        
        return "\n".join(lines)


class Subgroup(Group[T]):
    """子群类"""
    
    def __init__(self, parent_group: Group[T], elements: List[T], name: str = ""):
        super().__init__(name or f"Subgroup of {parent_group.name}")
        self._parent = parent_group
        self._elements = elements
        self._element_set = set(elements)
    
    def identity(self) -> T:
        """单位元"""
        return self._parent.identity()
    
    def multiply(self, a: T, b: T) -> T:
        """群乘法"""
        return self._parent.multiply(a, b)
    
    def inverse(self, a: T) -> T:
        """逆元"""
        return self._parent.inverse(a)
    
    def __contains__(self, element: T) -> bool:
        """判断元素是否属于该子群"""
        return element in self._element_set
    
    def order(self) -> int:
        """子群的阶"""
        return len(self._elements)
    
    def elements(self) -> List[T]:
        """所有子群元素"""
        return self._elements.copy()
    
    def is_normal(self) -> bool:
        """检查是否为正规子群"""
        parent_elems = list(self._parent.elements())
        for h in self._elements:
            for g in parent_elems:
                # g * h * g^(-1)
                conj = self._parent.multiply(
                    g, 
                    self._parent.multiply(h, self._parent.inverse(g))
                )
                if conj not in self._element_set:
                    return False
        return True
    
    def cosets_left(self) -> List[List[T]]:
        """计算左陪集"""
        parent_elems = list(self._parent.elements())
        cosets: List[List[T]] = []
        used = set()
        
        for g in parent_elems:
            if g in used:
                continue
            coset = [self._parent.multiply(g, h) for h in self._elements]
            cosets.append(coset)
            for elem in coset:
                used.add(elem)
        
        return cosets
    
    def cosets_right(self) -> List[List[T]]:
        """计算右陪集"""
        parent_elems = list(self._parent.elements())
        cosets: List[List[T]] = []
        used = set()
        
        for g in parent_elems:
            if g in used:
                continue
            coset = [self._parent.multiply(h, g) for h in self._elements]
            cosets.append(coset)
            for elem in coset:
                used.add(elem)
        
        return cosets
    
    def index(self) -> int:
        """子群的指数（陪集个数）"""
        parent_order = self._parent.order()
        subgroup_order = self.order()
        if parent_order % subgroup_order != 0:
            raise ValueError(f"子群阶数 {subgroup_order} 不整除父群阶数 {parent_order}，可能不是有效的子群")
        return parent_order // subgroup_order


class GroupFactory:
    """用于生成抽象群的工厂类"""
    
    @staticmethod
    def cyclic_group(n: int) -> Group[int]:
        """生成n阶循环群 C_n
        
        Args:
            n: 群的阶
            
        Returns:
            n阶循环群
        """
        from .specific_group import CyclicGroup
        return CyclicGroup(n)
    
    @staticmethod
    def symmetric_group(n: int) -> Group[Tuple[int, ...]]:
        """生成n次对称群 S_n
        
        Args:
            n: 对称群的次数
            
        Returns:
            n次对称群
        """
        from .specific_group import SymmetricGroup
        return SymmetricGroup(n)
    
    @staticmethod
    def dihedral_group(n: int) -> 'Group':
        """生成2n阶二面体群 D_n
        
        Args:
            n: 正多边形的边数
            
        Returns:
            2n阶二面体群
        """
        from .specific_group import DihedralGroup
        return DihedralGroup(n)
    
    @staticmethod
    def quaternion_group() -> 'Group':
        """生成四元数群 Q_8
        
        Returns:
            8阶四元数群
        """
        from .specific_group import QuaternionGroup
        return QuaternionGroup()
    
    @staticmethod
    def klein_four_group() -> Group[int]:
        """生成克莱因四元群 V_4
        
        Returns:
            克莱因四元群
        """
        from .specific_group import KleinGroup
        return KleinGroup()
    
    @staticmethod
    def alternating_group(n: int) -> Group[Tuple[int, ...]]:
        """生成n次交错群 A_n
        
        Args:
            n: 交错群的次数
            
        Returns:
            n次交错群
        """
        from .specific_group import AlternatingGroup
        return AlternatingGroup(n)
