"""抽象群论基类模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Tuple, Set, Any
from dataclasses import dataclass
from itertools import combinations

# 避免循环导入，使用延迟导入
Subgroup = None

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
    is_finite: bool               # 是否有限群


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
        """
        初始化群
        
        Args:
            name: 群的名称
        """
        self.name = name
        self._cache: Dict[str, Any] = {}
        self._properties: Optional[GroupProperties] = None
    
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
        if self._properties and self._properties.is_abelian is not None:
            return self._properties.is_abelian
        
        if self.order() > 1000:  # 大群采样检查
            return self._sample_abelian_check()
        elems = list(self.elements())
        for i, a in enumerate(elems):
            for b in elems[i:]:
                if self.multiply(a, b) != self.multiply(b, a):
                    self._update_properties(is_abelian=False)
                    return False
        self._update_properties(is_abelian=True)
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
    
    def is_simple(self) -> bool:
        """检查是否为单群（无非平凡正规子群）"""
        if self._properties and self._properties.is_simple is not None:
            return self._properties.is_simple
        
        if self.order() <= 1:
            return False
        
        # 检查所有可能的正规子群
        for subgroup in self.normal_subgroups():
            if 1 < subgroup.order() < self.order():
                self._update_properties(is_simple=False)
                return False
        
        self._update_properties(is_simple=True)
        return True
    
    def is_solvable(self) -> bool:
        """检查是否为可解群"""
        if self._properties and self._properties.is_solvable is not None:
            return self._properties.is_solvable
        
        if self.order() <= 1:
            self._update_properties(is_solvable=True)
            return True
        
        try:
            # 使用导出列（derived series）来判断可解性
            current_group = self
            max_iterations = self.order() * 2
            iteration = 0
            
            while iteration < max_iterations:
                # 计算导群 [G, G]
                derived_elements = set()
                elements = current_group.elements()
                
                for a in elements:
                    for b in elements:
                        # 交换子 [a, b] = a*b*a^{-1}*b^{-1}
                        commutator = current_group.multiply(a, current_group.multiply(b, current_group.multiply(current_group.inverse(a), current_group.inverse(b))))
                        derived_elements.add(commutator)
                
                # 生成导群
                derived_subgroup = current_group.generate_subgroup(list(derived_elements))
                
                # 如果导群是平凡群，则群可解
                if derived_subgroup.order() == 1:
                    self._update_properties(is_solvable=True)
                    return True
                
                # 如果导群与当前群相同，则群不可解
                if derived_subgroup.order() == current_group.order():
                    self._update_properties(is_solvable=False)
                    return False
                
                current_group = derived_subgroup
                iteration += 1
            
            # 达到最大迭代次数，认为不可解
            self._update_properties(is_solvable=False)
            return False
        except Exception:
            self._update_properties(is_solvable=False)
            return False
    
    def _update_properties(self, **kwargs) -> None:
        """更新群属性"""
        if self._properties is None:
            # 计算中心阶数，确保缓存正确设置
            center_order = self.center_order()
            # 计算共轭类数目
            conjugacy_classes = self.conjugacy_classes_count() if 'conjugacy_classes_count' in self._cache else None
            
            self._properties = GroupProperties(
                order=self.order(),
                is_abelian=kwargs.get('is_abelian'),
                is_simple=kwargs.get('is_simple'),
                is_solvable=kwargs.get('is_solvable'),
                center_order=center_order,
                conjugacy_classes=conjugacy_classes,
                generators=None,
                is_finite=True  # 默认假设有限
            )
        else:
            for key, value in kwargs.items():
                if hasattr(self._properties, key):
                    setattr(self._properties, key, value)
    
    @abstractmethod
    def elements(self) -> List[T]:
        """所有群元素（有限群）"""
        raise NotImplementedError("子类必须实现elements方法")
    
    def generate_subgroup(self, generators: List[T]) -> 'Subgroup[T]': # type: ignore
        """由生成元生成的子群
        
        Args:
            generators: 生成元列表
            
        Returns:
            由生成元生成的子群
        """
        global Subgroup
        if Subgroup is None:
            from .subgroup import Subgroup
        if not generators:
            return Subgroup(self, [self.identity()])
        
        for g in generators:
            if g not in self:
                raise ValueError(f"元素 {g} 不属于该群")
        
        subgroup_elements: Set[T] = set()
        subgroup_elements.add(self.identity())
        
        for g in generators:
            subgroup_elements.add(g)
        
        changed = True
        max_iterations = self.order() * self.order()
        iterations = 0
        
        while changed and iterations < max_iterations:
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
            iterations += 1
        
        return Subgroup(self, list(subgroup_elements))
    
    def conjugacy_classes(self) -> List[List[T]]:
        """计算共轭类"""
        if 'conjugacy_classes' in self._cache:
            return self._cache['conjugacy_classes']
        
        if self.order() > 500:
            raise ValueError("群太大，无法计算共轭类")
        
        elems = list(self.elements())
        elem_to_index = {elem: i for i, elem in enumerate(elems)}
        unprocessed = set(range(len(elems)))
        classes = []
        
        while unprocessed:
            i = unprocessed.pop()
            rep = elems[i]
            conj_class = {rep}
            
            for g in elems:
                for h in list(conj_class):
                    conj = self.multiply(g, self.multiply(h, self.inverse(g)))
                    conj_class.add(conj)
            
            classes.append(list(conj_class))
            unprocessed -= {elem_to_index[e] for e in conj_class if e in elem_to_index}
        
        self._cache['conjugacy_classes'] = classes
        self._cache['conjugacy_classes_count'] = len(classes)
        return classes
    
    def conjugacy_classes_count(self) -> int:
        """共轭类的数目"""
        if 'conjugacy_classes_count' in self._cache:
            return self._cache['conjugacy_classes_count']
        
        return len(self.conjugacy_classes())
    
    def generators(self) -> List[T]:
        """获取生成元"""
        if self._properties and self._properties.generators is not None:
            return self._properties.generators
        
        if self.order() > 100:
            raise ValueError("群太大，无法计算生成元")
        
        identity_elem = self.identity()
        elements = self.elements()
        
        # 特殊情况：平凡群
        if len(elements) == 1:
            self._update_properties(generators=[])
            return []
        
        generators = []
        generated_elements = {identity_elem}
        
        # 逐个尝试元素作为生成元
        for elem in elements:
            if elem == identity_elem:
                continue
            
            # 计算由当前生成元集合加上新元素生成的子群
            test_generators = generators + [elem]
            test_subgroup = self.generate_subgroup(test_generators)
            test_elements = set(test_subgroup.elements())
            
            # 如果生成的子群更大，则添加该元素到生成元列表
            if len(test_elements) > len(generated_elements):
                generators.append(elem)
                generated_elements = test_elements
                
                # 如果已经生成了整个群，停止
                if len(generated_elements) == self.order():
                    break
        
        self._update_properties(generators=generators)
        return generators
    
    def is_finite(self) -> bool:
        """检查群是否为有限群"""
        if self._properties and self._properties.is_finite is not None:
            return self._properties.is_finite
        
        # 默认假设有限，除非有特殊说明
        self._update_properties()
        return True
    
    def center(self) -> List[T]:
        """计算群中心"""
        if 'center' in self._cache:
            return self._cache['center']
        
        if self.order() > 500:
            raise ValueError("群太大，无法计算中心")
        
        elems = list(self.elements())
        center = []
        for a in elems:
            if all(self.multiply(a, b) == self.multiply(b, a) for b in elems):
                center.append(a)
        
        self._cache['center'] = center
        return center
    
    def center_order(self) -> int:
        """群中心的阶数"""
        if 'center_order' in self._cache:
            return self._cache['center_order']
        
        center = self.center()
        self._cache['center_order'] = len(center)
        return len(center)
    
    def normal_subgroups(self) -> List['Group']:
        """获取所有正规子群"""
        if 'normal_subgroups' in self._cache:
            return self._cache['normal_subgroups']
        
        if not self.is_finite():
            raise ValueError("无限群没有正规子群列表")
        
        normal_subgroups = []
        for subgroup in self.subgroups():
            if self.is_normal_subgroup(subgroup):
                normal_subgroups.append(subgroup)
        
        self._cache['normal_subgroups'] = normal_subgroups
        return normal_subgroups
    
    def is_normal_subgroup(self, subgroup: 'Group') -> bool:
        """检查子群是否为正规子群"""
        if not self.is_finite():
            raise ValueError("无限群无法检查正规子群")
        
        # 检查对所有g in G, gHg^-1 = H
        for g in self.elements():
            for h in subgroup.elements():
                conjugate = self.multiply(g, self.multiply(h, self.inverse(g)))
                if conjugate not in subgroup.elements():
                    return False
        return True
    
    def subgroups(self) -> List['Group']:
        """获取所有子群"""
        if 'subgroups' in self._cache:
            return self._cache['subgroups']
        
        if not self.is_finite():
            raise ValueError("无限群没有子群列表")
        
        elements = list(self.elements())
        n = len(elements)
        
        # 存储已经找到的子群（通过元素集合的哈希值）
        found_subgroups = set()
        subgroups = []
        
        # 从单位元开始
        identity_elem = self.identity()
        trivial_subgroup = self.generate_subgroup([identity_elem])
        subgroups.append(trivial_subgroup)
        found_subgroups.add(frozenset(trivial_subgroup.elements()))
        
        # 逐步构建子群
        # 遍历所有可能的生成元组合
        for i in range(1, n):
            # 使用组合而不是所有子集，减少重复计算
            for combo in combinations(elements, i):
                # 生成子群
                subgroup = self.generate_subgroup(list(combo))
                subgroup_elems = frozenset(subgroup.elements())
                
                # 检查是否已经找到
                if subgroup_elems not in found_subgroups:
                    found_subgroups.add(subgroup_elems)
                    subgroups.append(subgroup)
        
        # 添加群本身
        whole_group = self.generate_subgroup(elements)
        whole_group_elems = frozenset(whole_group.elements())
        if whole_group_elems not in found_subgroups:
            subgroups.append(whole_group)
        
        self._cache['subgroups'] = subgroups
        return subgroups
    


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

class FiniteGroup(Group[T]):
    """有限群"""
    pass


