"""直积/半直积群抽象基类模块"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Tuple, Callable
from .abstract_group import  GroupElement, FiniteGroup

T = TypeVar('T', bound='ProductGroupElement')

class ProductGroupElement(GroupElement, ABC):
    """直积/半直积群元素抽象基类"""
    
    def __init__(self, group: 'ProductGroup', element_id: int):
        self.group = group
        self.element_id = element_id
        self._components = None
    
    @abstractmethod
    def component(self, index: int) -> GroupElement:
        """获取直积/半直积群元素的第index个分量"""
        pass
    
    @abstractmethod
    def __mul__(self, other: 'ProductGroupElement') -> 'ProductGroupElement':
        """群乘法"""
        pass

class ProductGroup(FiniteGroup, ABC, Generic[T]):
    """直积/半直积群抽象基类"""
    
    def __init__(self, name: str, group1: FiniteGroup, group2: FiniteGroup):
        """
        初始化直积/半直积群
        
        Args:
            name: 群的名称
            group1: 第一个群
            group2: 第二个群
        """
        super().__init__(name)
        self.order_value = group1.order() * group2.order()
        self.group1 = group1
        self.group2 = group2
        self._elements = None
    
    @abstractmethod
    def _element_mul(self, a: T, b: T) -> T:
        """计算直积/半直积群元素的乘法"""
        pass
    
    @abstractmethod
    def _get_element(self, index: int) -> T:
        """获取直积/半直积群的第index个元素"""
        pass
    
    def order(self) -> int:
        """群的阶"""
        return self.order_value

    def elements(self) -> List[T]:
        """获取所有群元素"""
        if self._elements is None:
            self._elements = [self._get_element(i) for i in range(self.order())]
        return self._elements
    
    def identity(self) -> T:
        """单位元"""
        return self._get_element(0)
    
    def multiply(self, a: T, b: T) -> T:
        """群乘法"""
        return self._element_mul(a, b)
    
    def inverse(self, a: T) -> T:
        """求逆元"""
        return self._get_inverse(a)
    
    def _get_inverse(self, element: T) -> T:
        """获取元素的逆元"""
        # 默认实现：(g, h)^{-1} = (g^{-1}, h^{-1})
        g, h = element.component(0), element.component(1)
        g_inv = self.group1.inverse(g)
        h_inv = self.group2.inverse(h)
        return self._get_element_from_components(g_inv, h_inv)
    
    def _get_element_from_components(self, g: GroupElement, h: GroupElement) -> T:
        """根据两个群的元素创建直积/半直积群的元素"""
        # 子类实现
        pass
    
    def _get_components_from_index(self, index: int) -> Tuple[GroupElement, GroupElement]:
        """根据索引获取两个群的元素"""
        index1 = index % self.group1.order()
        index2 = index // self.group1.order()
        return self.group1.elements()[index1], self.group2.elements()[index2]
    
    def is_abelian(self) -> bool:
        """检查群是否为阿贝尔群"""
        return self.group1.is_abelian() and self.group2.is_abelian()
    
    def is_simple(self) -> bool:
        """检查群是否为单群"""
        # 直积/半直积群（非平凡）总是不是单群
        return False
    
    def conjugacy_classes(self) -> List[List[T]]:
        """计算共轭类"""
        # 基类提供默认实现，子类可以覆盖
        # 这里简单地返回所有元素作为单个共轭类，实际实现应该由子类提供
        return [self.elements()]




class DirectProductGroupElement(ProductGroupElement):
    """直积群元素"""
    
    def __init__(self, group: 'DirectProductGroup', element_id: int):
        super().__init__(group, element_id)
        self._components = group._get_components_from_index(element_id)
    
    def component(self, index: int) -> GroupElement:
        """获取直积群元素的第index个分量"""
        if index == 0:
            return self._components[0]
        elif index == 1:
            return self._components[1]
        else:
            raise IndexError("索引超出范围")
    
    def __mul__(self, other: 'DirectProductGroupElement') -> 'DirectProductGroupElement':
        """直积群的乘法：(g1, h1)(g2, h2) = (g1g2, h1h2)"""
        g1 = self.component(0)
        g2 = other.component(0)
        h1 = self.component(1)
        h2 = other.component(1)
        return self.group._get_element_from_components(
            self.group.group1.multiply(g1, g2),
            self.group.group2.multiply(h1, h2)
        )
    
    def __repr__(self):
        return f"({self.component(0)}, {self.component(1)})"

class DirectProductGroup(ProductGroup[DirectProductGroupElement]):
    """直积群实现"""
    
    def __init__(self, group1: FiniteGroup, group2: FiniteGroup):
        """
        初始化直积群
        :param group1: 第一个群
        :param group2: 第二个群
        """
        super().__init__(f"{group1.name} × {group2.name}", group1, group2)
    
    def _element_mul(self, a: DirectProductGroupElement, b: DirectProductGroupElement) -> DirectProductGroupElement:
        """直积群的乘法实现"""
        return a * b
    
    def _get_element(self, index: int) -> DirectProductGroupElement:
        """获取直积群的第index个元素"""
        return DirectProductGroupElement(self, index)
    
    def _get_element_from_components(self, g: GroupElement, h: GroupElement) -> DirectProductGroupElement:
        """根据两个群的元素创建直积群的元素"""
        index1 = self.group1.elements().index(g)
        index2 = self.group2.elements().index(h)
        return self._get_element(index1 + index2 * self.group1.order())
    
    def conjugacy_classes(self) -> List[List[DirectProductGroupElement]]:
        """计算直积群的共轭类"""
        # 在直积群中，共轭类是两个群共轭类的笛卡尔积
        classes1 = self.group1.conjugacy_classes()
        classes2 = self.group2.conjugacy_classes()
        
        classes = []
        for c1 in classes1:
            for c2 in classes2:
                # 构建笛卡尔积
                class_elements = [
                    self._get_element_from_components(n, h)
                    for n in c1 for h in c2
                ]
                classes.append(class_elements)
        
        return classes



# 避免循环导入，使用延迟导入
GroupHomomorphism = None

class SemidirectProductGroupElement(ProductGroupElement):
    """半直积群元素"""
    
    def __init__(self, group: 'SemidirectProductGroup', element_id: int):
        super().__init__(group, element_id)
        self._components = group._get_components_from_index(element_id)
    
    def component(self, index: int) -> GroupElement:
        """获取半直积群元素的第index个分量"""
        if index == 0:
            return self._components[0]
        elif index == 1:
            return self._components[1]
        else:
            raise IndexError("索引超出范围")
    
    def __mul__(self, other: 'SemidirectProductGroupElement') -> 'SemidirectProductGroupElement':
        """半直积群的乘法：(n1, h1)(n2, h2) = (n1φ(h1)(n2), h1h2)"""
        n1, h1 = self.component(0), self.component(1)
        n2, h2 = other.component(0), other.component(1)
        
        # 应用半直积的同态φ
        phi = self.group.phi
        n2_transformed = phi(h1)(n2)
        
        # 计算新的元素
        n_new = self.group.group1.multiply(n1, n2_transformed)
        h_new = self.group.group2.multiply(h1, h2)
        
        return self.group._get_element_from_components(n_new, h_new)
    
    def __repr__(self):
        return f"({self.component(0)}, {self.component(1)})"

class SemidirectProductGroup(ProductGroup[SemidirectProductGroupElement]):
    """半直积群实现"""
    
    def __init__(self, group1: FiniteGroup, group2: FiniteGroup, phi):
        """
        初始化半直积群
        :param group1: 正规子群N
        :param group2: 补群H
        :param phi: 同态φ: H → Aut(N)
        """
        global GroupHomomorphism
        if GroupHomomorphism is None:
            from .group_func import GroupHomomorphism
        if not isinstance(phi, GroupHomomorphism):
            raise TypeError("phi must be a GroupHomomorphism")
        super().__init__(f"{group1.name} ⋊_{phi} {group2.name}", group1, group2)
        self.phi = phi  # 同态φ
    
    def _element_mul(self, a: SemidirectProductGroupElement, b: SemidirectProductGroupElement) -> SemidirectProductGroupElement:
        """半直积群的乘法实现"""
        return a * b
    
    def _get_element(self, index: int) -> SemidirectProductGroupElement:
        """获取半直积群的第index个元素"""
        return SemidirectProductGroupElement(self, index)
    
    def _get_element_from_components(self, n: GroupElement, h: GroupElement) -> SemidirectProductGroupElement:
        """根据两个群的元素创建半直积群的元素"""
        index1 = self.group1.elements().index(n)
        index2 = self.group2.elements().index(h)
        return self._get_element(index1 + index2 * self.group1.order())
    
    def _get_inverse(self, element: SemidirectProductGroupElement) -> SemidirectProductGroupElement:
        """获取半直积群元素的逆元"""
        n, h = element.component(0), element.component(1)
        # (n, h)^{-1} = (φ(h^{-1})(n^{-1}), h^{-1})
        n_inv = self.group1.inverse(n)
        h_inv = self.group2.inverse(h)
        n_transformed = self.phi(h_inv)(n_inv)
        return self._get_element_from_components(n_transformed, h_inv)
    
    def is_abelian(self) -> bool:
        """检查半直积群是否为阿贝尔群"""
        # 半直积群是阿贝尔群当且仅当φ是平凡同态且两个群都是阿贝尔群
        if not self.group1.is_abelian() or not self.group2.is_abelian():
            return False
        
        # 检查φ是否是平凡同态（即对所有h，φ(h)是恒等自同构）
        for h in self.group2.elements():
            automorphism = self.phi(h)
            # 检查自同构是否为恒等映射
            for n in self.group1.elements():
                if automorphism(n) != n:
                    return False
        return True
    
    def conjugacy_classes(self) -> List[List[SemidirectProductGroupElement]]:
        """计算半直积群的共轭类"""
        # 半直积群的共轭类计算更复杂
        # 需要根据同态φ来计算
        classes = []
        for n in self.group1.elements():
            for h in self.group2.elements():
                # 计算(n, h)的共轭类
                class_elements = []
                for g in self.group1.elements():
                    for k in self.group2.elements():
                        # (g, k)(n, h)(g, k)^{-1}
                        g_inv = self.group1.inverse(g)
                        k_inv = self.group2.inverse(k)
                        # 计算(g, k)(n, h)(g, k)^{-1}
                        # = (g, k)(n, h)(g_inv, k_inv)
                        # = (g * φ(k)(n), k*h) * (g_inv, k_inv)
                        # = (g * φ(k)(n) * φ(k*h)(g_inv), k*h*k_inv)
                        # = (g * φ(k)(n) * φ(k)(φ(h)(g_inv)), k*h*k_inv)
                        # = (g * φ(k)(n * φ(h)(g_inv)), k*h*k_inv)
                        n_conj = self.group1.multiply(
                            g, 
                            self.phi(k)(self.group1.multiply(n, self.phi(h)(g_inv)))
                        )
                        h_conj = self.group2.multiply(
                            self.group2.multiply(k, h), 
                            k_inv
                        )
                        class_elements.append(self._get_element_from_components(n_conj, h_conj))
                classes.append(class_elements)
        return classes