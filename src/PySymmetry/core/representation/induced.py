import numpy as np

from ..group_theory.abstract_group import Group, GroupElement
from ..matrix_groups.general_linear import GLnElement
from .abstract_representation import GroupRepresentation

T = GroupElement


def _carrier_groups_match(g_rep: Group, g_sub: Group) -> bool:
    """子群在数学上一致时允许 `ρ.group` 与传入的 `subgroup` 非同一对象（如 Subgroup 与 AlternatingGroup）。"""
    if g_rep is g_sub:
        return True
    try:
        if g_rep.identity() != g_sub.identity():
            return False
        er, es = list(g_rep.elements()), list(g_sub.elements())
        if len(er) != len(es):
            return False
        return set(er) == set(es)
    except TypeError:
        return False


class InducedRepresentation(GroupRepresentation[T]):
    """诱导表示

    从子群的表示构造群的表示
    Ind_H^G(ρ) 表示从子群 H 的表示 ρ 诱导出的群 G 的表示
    """

    def __init__(
        self,
        group: Group[T],
        subgroup: Group[T],
        subgroup_representation: GroupRepresentation[T],
    ):
        """
        Args:
            group: 群 G
            subgroup: 子群 H
            subgroup_representation: 子群 H 的表示 ρ
        """
        if not _carrier_groups_match(subgroup_representation.group, subgroup):
            raise ValueError("子表示必须定义在与传入子群载体相同的有限群上")
        for h in subgroup.elements():
            if h not in group:
                raise ValueError("子群元素必须属于母群")

        # 计算陪集分解
        self._subgroup = subgroup
        self._subgroup_representation = subgroup_representation

        # 计算左陪集代表元
        self._coset_representatives = self._compute_coset_representatives(
            group, subgroup
        )
        self._coset_map = self._create_coset_map(
            group, subgroup, self._coset_representatives
        )

        # 诱导表示的维度 = 陪集数量 * 子表示维度
        dimension = len(self._coset_representatives) * subgroup_representation.dimension

        super().__init__(group, dimension)

    def _compute_coset_representatives(
        self, group: Group[T], subgroup: Group[T]
    ) -> list[T]:
        """计算群 G 关于子群 H 的左陪集代表元"""
        cosets = []
        used_elements = set()

        for g in group.elements():
            if g not in used_elements:
                # 计算左陪集 gH
                coset = {group.multiply(g, h) for h in subgroup.elements()}
                cosets.append(g)
                used_elements.update(coset)

        return cosets

    def _create_coset_map(
        self, group: Group[T], subgroup: Group[T], coset_representatives: list[T]
    ) -> dict[T, tuple[int, T]]:
        """创建群元素到 (陪集索引, 子群元素) 的映射"""
        coset_map = {}
        for i, g_i in enumerate(coset_representatives):
            for h in subgroup.elements():
                element = group.multiply(g_i, h)
                coset_map[element] = (i, h)
        return coset_map

    def __call__(self, element: T) -> GLnElement:
        """计算群元素在诱导表示中的矩阵"""
        n = len(self._coset_representatives)
        d = self._subgroup_representation.dimension
        subgroup_elements = set(self._subgroup.elements())

        # 初始化诱导表示矩阵
        induced_matrix = np.zeros((n * d, n * d), dtype=complex)

        # 对每个陪集代表元 g_i
        for i, g_i in enumerate(self._coset_representatives):
            # 计算 g * g_i
            g_g_i = self._group.multiply(element, g_i)

            # 找到唯一的陪集代表元 g_j 和子群元素 h 使得 g * g_i = g_j * h
            found_coset = False
            for j, g_j in enumerate(self._coset_representatives):
                # 计算 h = g_j^{-1} * g * g_i
                g_j_inv = self._group.inverse(g_j)
                h = self._group.multiply(g_j_inv, g_g_i)

                if h in subgroup_elements:
                    # 获取子表示中 h 的矩阵
                    h_matrix = self._subgroup_representation(h).matrix

                    # 填充诱导表示矩阵的对应块
                    induced_matrix[j * d : (j + 1) * d, i * d : (i + 1) * d] = h_matrix
                    found_coset = True
                    break
            if not found_coset:
                raise ValueError("无法将元素分解为陪集代表与子群元素的乘积")

        return GLnElement(induced_matrix)

    def is_homomorphism(self) -> bool:
        """验证同态性质"""
        for a in self._group.elements():
            for b in self._group.elements():
                ab = self._group.multiply(a, b)
                if not np.allclose(self(ab).matrix, self(a).matrix @ self(b).matrix):
                    return False
        return True

    def subgroup(self) -> Group[T]:
        """获取子群 H"""
        return self._subgroup

    def subgroup_representation(self) -> GroupRepresentation[T]:
        """获取子群的表示 ρ"""
        return self._subgroup_representation

    def coset_representatives(self) -> list[T]:
        """获取陪集代表元"""
        return self._coset_representatives

    def induced_character(self, element: T) -> complex:
        """计算诱导特征标

        使用 Frobenius 互反律：χ_Ind(g) = Σ_{g_i^{-1} g g_i ∈ H} χ(h)
        """
        character_sum = 0.0
        d = self._subgroup_representation.dimension

        for g_i in self._coset_representatives:
            # 计算 g_i^{-1} * g * g_i
            g_i_inv = self._group.inverse(g_i)
            conjugate = self._group.multiply(
                g_i_inv, self._group.multiply(element, g_i)
            )

            if conjugate in self._subgroup.elements():
                # 获取子表示的特征标值
                character_sum += np.trace(
                    self._subgroup_representation(conjugate).matrix
                )

        return character_sum

    def is_irreducible(self) -> bool:
        """检查诱导表示是否不可约"""
        from .character import Character

        char = Character(self)
        return char.is_irreducible()

    def decompose(self) -> list[GroupRepresentation[T]]:
        """将诱导表示分解为不可约表示的直和"""
        from .irreducible import IrreducibleRepresentationFinder

        return IrreducibleRepresentationFinder.decompose(self)

    @classmethod
    def from_trivial_subgroup(cls, group: Group[T]) -> "InducedRepresentation":
        """从平凡子群诱导表示（即正则表示）"""
        # 构造平凡子群的平凡表示
        identity = group.identity()

        class TrivialSubgroup:
            def elements(self):
                return [identity]

            def identity(self):
                return identity

            def multiply(self, a, b):
                return identity

            def inverse(self, a):
                return identity

            def __contains__(self, element):
                return element == identity

            def conjugacy_classes(self):
                return [[identity]]

        trivial_subgroup = TrivialSubgroup()
        from .matrix_representation import MatrixRepresentation

        trivial_rep = MatrixRepresentation.trivial_representation(trivial_subgroup)

        return cls(group, trivial_subgroup, trivial_rep)
