from .abstract_group import Group, T

# 避免循环导入，使用延迟导入
CosetSpace = None
Coset = None


class Subgroup(Group[T]):
    """子群类"""

    def __init__(self, parent_group: Group[T], elements: list[T], name: str = ""):
        super().__init__(name or f"Subgroup of {parent_group.name}")
        self._parent = parent_group
        self._elements = elements
        # 确保元素可哈希，使用元素的哈希值作为集合元素
        self._element_set = set()
        for elem in elements:
            self._element_set.add(elem)

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

    def elements(self) -> list[T]:
        """所有子群元素"""
        return self._elements.copy()

    def is_normal(self) -> bool:
        """检查是否为正规子群"""
        return self._parent.is_normal_subgroup(self)

    def cosets_left(self) -> list["Coset[T]"]:  # type: ignore
        """计算左陪集"""
        global CosetSpace, Coset
        if CosetSpace is None or Coset is None:
            from .coset import Coset, CosetSpace
        coset_space = CosetSpace(self._parent, self, is_left=True)
        return coset_space.cosets()

    def cosets_right(self) -> list["Coset[T]"]:  # type: ignore
        """计算右陪集"""
        global CosetSpace, Coset
        if CosetSpace is None or Coset is None:
            from .coset import Coset, CosetSpace
        coset_space = CosetSpace(self._parent, self, is_left=False)
        return coset_space.cosets()

    def index(self) -> int:
        """子群的指数（陪集个数）"""
        parent_order = self._parent.order()
        subgroup_order = self.order()
        if parent_order % subgroup_order != 0:
            raise ValueError(
                f"子群阶数 {subgroup_order} 不整除父群阶数 {parent_order}，可能不是有效的子群"
            )
        return parent_order // subgroup_order
