from typing import TypeVar, List, Generic
from .abstract_group import Group, GroupElement, FiniteGroup
from .subgroup import Subgroup

T = TypeVar('T')

class Coset(GroupElement, Generic[T]):
    """陪集类，表示群G关于子群H的左/右陪集"""
    
    def __init__(self, group: Group[T], subgroup: 'Subgroup[T]', representative: T, is_left: bool = True):
        """
        初始化陪集
        :param group: 父群
        :param subgroup: 子群
        :param representative: 代表元
        :param is_left: 是否为左陪集(默认True)
        """
        self.group = group
        self.subgroup = subgroup
        self.representative = representative
        self.is_left = is_left
        self._elements = None  # 缓存陪集元素
    
    def elements(self) -> List[T]:
        """获取陪集中的所有元素"""
        if self._elements is None:
            if self.is_left:
                self._elements = [self.group.multiply(self.representative, h) for h in self.subgroup.elements()]
            else:
                self._elements = [self.group.multiply(h, self.representative) for h in self.subgroup.elements()]
        return self._elements
    
    def __contains__(self, element: T) -> bool:
        """判断元素是否属于陪集
        
        基于陪集的定义：
        - 左陪集 aH：元素 x 属于 aH 当且仅当 a^{-1}x 属于 H
        - 右陪集 Ha：元素 x 属于 Ha 当且仅当 xa^{-1} 属于 H
        """
        if self.is_left:
            # 左陪集：x ∈ aH ⇨ a^{-1}x ∈ H
            a_inv = self.group.inverse(self.representative)
            check_elem = self.group.multiply(a_inv, element)
        else:
            # 右陪集：x ∈ Ha ⇨ xa^{-1} ∈ H
            a_inv = self.group.inverse(self.representative)
            check_elem = self.group.multiply(element, a_inv)
        
        return check_elem in self.subgroup
    
    def __len__(self) -> int:
        """陪集的阶(等于子群的阶)"""
        return self.subgroup.order()
    
    def __repr__(self) -> str:
        direction = "左" if self.is_left else "右"
        return f"{direction}陪集({self.representative})"
    
    def __mul__(self, other: 'Coset') -> 'Coset':
        """群乘法"""
        if self.group != other.group or self.subgroup != other.subgroup:
            raise ValueError("两个陪集不属于同一个商群")
        # 陪集乘法定义: (aN)(bN) = (ab)N
        rep_product = self.group.multiply(self.representative, other.representative)
        return Coset(self.group, self.subgroup, rep_product, self.is_left)
    
    def __pow__(self, n: int) -> 'Coset':
        """幂运算"""
        if n == 0:
            return Coset(self.group, self.subgroup, self.group.identity(), self.is_left)
        elif n > 0:
            result = self
            for _ in range(n-1):
                result *= self
            return result
        else:
            return self.inverse() ** (-n)
    
    def inverse(self) -> 'Coset':
        """逆元"""
        rep_inverse = self.group.inverse(self.representative)
        return Coset(self.group, self.subgroup, rep_inverse, self.is_left)
    
    def __hash__(self) -> int:
        """哈希值
        
        基于陪集的本质属性计算哈希值，确保相等的陪集具有相同的哈希值
        """
        # 计算代表元在群中的唯一标识
        rep_hash = hash(self.representative)
        # 计算子群的唯一标识
        subgroup_hash = hash(frozenset(self.subgroup.elements()))
        return hash((rep_hash, subgroup_hash, self.is_left))
    
    def __eq__(self, other: object) -> bool:
        """相等性比较
        
        两个陪集相等当且仅当：
        - 属于同一个群
        - 属于同一个子群
        - 都是左陪集或都是右陪集
        - 代表元的差属于子群
        """
        if not isinstance(other, Coset):
            return NotImplemented
        if (self.group != other.group or 
            self.subgroup != other.subgroup or 
            self.is_left != other.is_left):
            return False
        # 检查陪集是否相等：aH = bH 当且仅当 a^{-1}b ∈ H (左陪集)
        # 或 ab^{-1} ∈ H (右陪集)
        if self.is_left:
            # 左陪集：aH = bH 当且仅当 a^{-1}b ∈ H
            a_inv = self.group.inverse(self.representative)
            ab = self.group.multiply(a_inv, other.representative)
            return ab in self.subgroup
        else:
            # 右陪集：Ha = Hb 当且仅当 ab^{-1} ∈ H
            b_inv = self.group.inverse(other.representative)
            ab = self.group.multiply(self.representative, b_inv)
            return ab in self.subgroup
    
    def is_identity(self) -> bool:
        """是否为单位元"""
        return self.representative == self.group.identity()
    
    def order(self) -> int:
        """元素阶数"""
        # 计算陪集的阶
        n = 1
        current = self
        while not current.is_identity():
            current *= self
            n += 1
        return n

class CosetSpace(Generic[T]):
    """陪集空间类，表示群G关于子群H的所有陪集集合"""
    
    def __init__(self, group: Group[T], subgroup: 'Subgroup[T]', is_left: bool = True):
        """
        初始化陪集空间
        :param group: 父群
        :param subgroup: 子群
        :param is_left: 是否为左陪集空间(默认True)
        """
        self.group = group
        self.subgroup = subgroup
        self.is_left = is_left
        self._cosets = None  # 缓存陪集列表
    
    def cosets(self) -> List[Coset[T]]:
        """获取所有陪集"""
        if self._cosets is None:
            self._cosets = []
            parent_elems = list(self.group.elements())
            used = set()
            
            for g in parent_elems:
                if g in used:
                    continue
                coset = Coset(self.group, self.subgroup, g, self.is_left)
                self._cosets.append(coset)
                for elem in coset.elements():
                    used.add(elem)
        return self._cosets
    
    def index(self) -> int:
        """陪集空间的指数(陪集个数)"""
        return len(self.cosets())
    
    def __len__(self) -> int:
        """陪集空间的指数"""
        return self.index()
    
    def __contains__(self, element: T) -> bool:
        """判断元素是否属于某个陪集"""
        return any(element in coset for coset in self.cosets())
    
    def __repr__(self) -> str:
        direction = "左" if self.is_left else "右"
        return f"{direction}陪集空间(G/{self.subgroup})"

class QuotientGroup(Group[Coset[T]]):
    """商群类，表示群G关于正规子群N的商群G/N"""
    
    def __init__(self, group: FiniteGroup[T], normal_subgroup: 'Subgroup[T]'):
        """
        初始化商群
        :param group: 父群
        :param normal_subgroup: 正规子群
        :raises ValueError: 如果子群不是正规子群
        """
        if not normal_subgroup.is_normal():
            raise ValueError("商群只能由正规子群构造")
        
        self.group = group
        self.normal_subgroup = normal_subgroup
        self._cosets = None  # 缓存陪集列表
        super().__init__(f"{group.name}/{normal_subgroup.name}")
    
    def identity(self) -> Coset[T]:
        """商群的单位元(即正规子群本身)"""
        return Coset(self.group, self.normal_subgroup, self.group.identity())
    
    def multiply(self, a: Coset[T], b: Coset[T]) -> Coset[T]:
        """商群的乘法运算"""
        # 陪集乘法定义: (aN)(bN) = (ab)N
        rep_product = self.group.multiply(a.representative, b.representative)
        return Coset(self.group, self.normal_subgroup, rep_product)
    
    def inverse(self, coset: Coset[T]) -> Coset[T]:
        """陪集的逆元"""
        rep_inverse = self.group.inverse(coset.representative)
        return Coset(self.group, self.normal_subgroup, rep_inverse)
    
    def __contains__(self, element: Coset[T]) -> bool:
        """判断陪集是否属于商群"""
        if not isinstance(element, Coset):
            return False
        return element.group == self.group and element.subgroup == self.normal_subgroup
    
    def order(self) -> int:
        """商群的阶(等于陪集空间的指数)"""
        return self.group.order() // self.normal_subgroup.order()
    
    def elements(self) -> List[Coset[T]]:
        """获取商群的所有元素(即所有陪集)"""
        if self._cosets is None:
            coset_space = CosetSpace(self.group, self.normal_subgroup, is_left=True)
            self._cosets = coset_space.cosets()  # 正规子群左右陪集相同
        return self._cosets
    
    def is_abelian(self) -> bool:
        """商群是否为阿贝尔群"""
        # 如果原群是阿贝尔群，则商群也是阿贝尔群
        if self.group.is_abelian():
            return True
        # 否则需要检查陪集乘法的交换性
        return super().is_abelian()
