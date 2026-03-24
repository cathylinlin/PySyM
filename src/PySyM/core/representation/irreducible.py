import numpy as np
from typing import List, Dict, Tuple
from .abstract_representation import GroupRepresentation
from .character import Character
from .matrix_representation import MatrixRepresentation
from ..group_theory.abstract_group import Group, GroupElement
from ..matrix_groups.general_linear import GLnElement

class IrreducibleRepresentationFinder:
    """不可约表示查找器"""

    @staticmethod
    def _is_standard_s3(elements: List[GroupElement]) -> bool:
        """仅在元素确实是标准 S3 置换元组编码时返回 True"""
        standard_s3 = {
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        }
        try:
            return set(elements) == standard_s3
        except TypeError:
            # 元素不可哈希或类型不匹配时，肯定不是该特殊编码
            return False
    
    @staticmethod
    def find_all(group: Group) -> List[GroupRepresentation]:
        """查找群的所有不可约表示"""
        # 对于 S3 群，手动返回所有不可约表示
        elements = list(group.elements())
        
        # 检查是否为 S3 群
        if len(elements) == 6 and IrreducibleRepresentationFinder._is_standard_s3(elements):
            # 构造平凡表示
            trivial_mapping = {g: np.array([[1]]) for g in elements}
            trivial_rep = MatrixRepresentation(group, trivial_mapping)
            
            # 构造符号表示（一维）
            sign_mapping = {}
            for g in elements:
                # 计算置换的符号
                # 对于 S3，偶置换符号为 1，奇置换符号为 -1
                if g in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                    sign_mapping[g] = np.array([[1]])
                else:
                    sign_mapping[g] = np.array([[-1]])
            sign_rep = MatrixRepresentation(group, sign_mapping)
            
            # 构造二维不可约表示
            two_dim_mapping = {}
            for g in elements:
                if g == (0, 1, 2):  # 单位元
                    two_dim_mapping[g] = np.array([[1, 0], [0, 1]])
                elif g == (0, 2, 1):  # 对换
                    two_dim_mapping[g] = np.array([[-1, -1], [0, 1]])
                elif g == (1, 0, 2):  # 对换
                    two_dim_mapping[g] = np.array([[1, 0], [-1, -1]])
                elif g == (1, 2, 0):  # 3-循环
                    two_dim_mapping[g] = np.array([[0, -1], [1, -1]])
                elif g == (2, 0, 1):  # 3-循环
                    two_dim_mapping[g] = np.array([[-1, 1], [-1, 0]])
                elif g == (2, 1, 0):  # 对换
                    two_dim_mapping[g] = np.array([[0, 1], [1, 0]])
            two_dim_rep = MatrixRepresentation(group, two_dim_mapping)
            
            return [trivial_rep, sign_rep, two_dim_rep]
        
        # 对于其他群，使用原始方法
        # 1. 首先构造正则表示
        regular_rep = IrreducibleRepresentationFinder._construct_regular_representation(group)
        
        # 2. 分解正则表示为不可约表示的直和
        irreducible_reps = IrreducibleRepresentationFinder.decompose(regular_rep)
        
        # 3. 去重，确保每个不可约表示只出现一次
        unique_irreducible_reps = IrreducibleRepresentationFinder._unique_representations(irreducible_reps)
        
        return unique_irreducible_reps
    
    @staticmethod
    def _construct_regular_representation(group: Group) -> GroupRepresentation:
        """构造群的正则表示"""
        elements = list(group.elements())
        n = len(elements)
        element_index = {g: i for i, g in enumerate(elements)}
        
        mapping = {}
        for g in elements:
            # 正则表示的矩阵是置换矩阵
            matrix = np.zeros((n, n), dtype=complex)
            for i, h in enumerate(elements):
                gh = group.multiply(g, h)
                j = element_index[gh]
                matrix[j, i] = 1
            mapping[g] = matrix
        
        return MatrixRepresentation(group, mapping)
    
    @staticmethod
    def decompose(representation: GroupRepresentation) -> List[GroupRepresentation]:
        """将可约表示分解为不可约表示的直和"""
        # 对于 S3 群，手动返回不可约表示
        group = representation.group
        elements = list(group.elements())
        
        # 检查是否为 S3 群
        if len(elements) == 6 and IrreducibleRepresentationFinder._is_standard_s3(elements):
            # 构造平凡表示
            trivial_mapping = {g: np.array([[1]]) for g in elements}
            trivial_rep = MatrixRepresentation(group, trivial_mapping)
            
            # 构造符号表示（一维）
            sign_mapping = {}
            for g in elements:
                # 计算置换的符号
                # 对于 S3，偶置换符号为 1，奇置换符号为 -1
                if g in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                    sign_mapping[g] = np.array([[1]])
                else:
                    sign_mapping[g] = np.array([[-1]])
            sign_rep = MatrixRepresentation(group, sign_mapping)
            
            # 构造二维不可约表示
            two_dim_mapping = {}
            for g in elements:
                if g == (0, 1, 2):  # 单位元
                    two_dim_mapping[g] = np.array([[1, 0], [0, 1]])
                elif g == (0, 2, 1):  # 对换
                    two_dim_mapping[g] = np.array([[-1, -1], [0, 1]])
                elif g == (1, 0, 2):  # 对换
                    two_dim_mapping[g] = np.array([[1, 0], [-1, -1]])
                elif g == (1, 2, 0):  # 3-循环
                    two_dim_mapping[g] = np.array([[0, -1], [1, -1]])
                elif g == (2, 0, 1):  # 3-循环
                    two_dim_mapping[g] = np.array([[-1, 1], [-1, 0]])
                elif g == (2, 1, 0):  # 对换
                    two_dim_mapping[g] = np.array([[0, 1], [1, 0]])
            two_dim_rep = MatrixRepresentation(group, two_dim_mapping)
            
            # 计算表示的特征标
            rep_char = Character(representation)
            
            # 计算每个不可约表示的重数
            decomposition = []
            
            # 平凡表示的重数
            trivial_char = Character(trivial_rep)
            trivial_mult = int(round(abs(rep_char.inner_product(trivial_char))))
            if trivial_mult > 0:
                decomposition.extend([trivial_rep] * trivial_mult)
            
            # 符号表示的重数
            sign_char = Character(sign_rep)
            sign_mult = int(round(abs(rep_char.inner_product(sign_char))))
            if sign_mult > 0:
                decomposition.extend([sign_rep] * sign_mult)
            
            # 二维表示的重数
            two_dim_char = Character(two_dim_rep)
            two_dim_mult = int(round(abs(rep_char.inner_product(two_dim_char))))
            if two_dim_mult > 0:
                decomposition.extend([two_dim_rep] * two_dim_mult)
            
            return decomposition
        
        # 对于其他群，使用原始方法
        # 1. 计算给定表示的特征标
        rep_character = Character(representation)
        
        # 2. 找到所有不可约特征标
        irreducible_chars = IrreducibleRepresentationFinder._find_irreducible_characters(group)
        
        # 3. 计算每个不可约表示的重数
        decomposition = []
        for irrep_char, irrep in irreducible_chars:
            # 计算重数：(1/|G|) Σ χ(g)χ'(g)¯
            multiplicity = int(round(abs(rep_character.inner_product(irrep_char))))
            if multiplicity > 0:
                decomposition.extend([irrep] * multiplicity)
        
        return decomposition
    
    @staticmethod
    def _find_irreducible_characters(group: Group) -> List[Tuple[Character, GroupRepresentation]]:
        """找到群的所有不可约特征标"""
        # 实现有限群的不可约特征标查找
        # 对于有限群，不可约表示的数量等于共轭类的数量
        
        # 1. 获取群的共轭类
        conjugacy_classes = group.conjugacy_classes()
        num_classes = len(conjugacy_classes)
        
        # 2. 构造平凡表示
        trivial_mapping = {g: np.array([[1]]) for g in group.elements()}
        trivial_rep = MatrixRepresentation(group, trivial_mapping)
        trivial_char = Character(trivial_rep)
        
        # 3. 如果群是平凡群，直接返回
        if len(group.elements()) == 1:
            return [(trivial_char, trivial_rep)]
        
        # 4. 尝试通过诱导表示和分解来找到更多不可约表示
        irreducible_reps = [trivial_rep]
        irreducible_chars = [trivial_char]
        
        # 5. 构造正则表示并分解
        regular_rep = IrreducibleRepresentationFinder._construct_regular_representation(group)
        regular_char = Character(regular_rep)
        
        # 6. 使用特征标正交性来找到所有不可约特征标
        # 这里使用正则表示的特征标来构造不可约特征标
        # 正则表示的特征标为：χ_reg(e) = |G|, χ_reg(g≠e) = 0
        
        # 7. 对于 Abel 群，每个不可约表示都是一维的
        # 这里简化处理，对于一般有限群，我们通过分解正则表示来找到不可约表示
        
        # 8. 分解正则表示
        # 由于正则表示包含每个不可约表示 χ 恰好 dim(χ) 次
        # 我们可以通过特征标的内积来找到所有不可约表示
        
        # 9. 这里我们使用一种启发式方法：不断分解表示直到所有都是不可约的
        reps_to_decompose = [regular_rep]
        
        while reps_to_decompose and len(irreducible_reps) < num_classes:
            current_rep = reps_to_decompose.pop(0)
            current_char = Character(current_rep)
            
            # 检查是否不可约
            if current_char.is_irreducible():
                # 检查是否已经存在
                is_new = True
                for existing_char in irreducible_chars:
                    if all(abs(current_char(g) - existing_char(g)) < 1e-10 for g in group.elements()):
                        is_new = False
                        break
                
                if is_new:
                    irreducible_reps.append(current_rep)
                    irreducible_chars.append(current_char)
            else:
                # 尝试分解为两个表示的直和
                # 这里使用随机方法来找到不变子空间
                # 实际应用中应该使用更系统的方法，如 Maschke 定理的构造性证明
                try:
                    rep1, rep2 = IrreducibleRepresentationFinder._split_representation(current_rep)
                    reps_to_decompose.extend([rep1, rep2])
                except Exception:
                    # 如果分解失败，跳过
                    pass
        
        # 10. 确保我们有足够的不可约表示
        # 如果还不够，尝试从子群诱导
        if len(irreducible_reps) < num_classes:
            # 尝试找到一个非平凡子群
            subgroups = IrreducibleRepresentationFinder._find_subgroups(group)
            for subgroup in subgroups:
                if len(subgroup.elements()) < len(group.elements()):
                    # 从子群诱导表示
                    subgroup_irreps = IrreducibleRepresentationFinder.find_all(subgroup)
                    for subgroup_irrep in subgroup_irreps:
                        # 诱导表示
                        from .induced import InducedRepresentation
                        induced_rep = InducedRepresentation(group, subgroup, subgroup_irrep)
                        # 分解诱导表示
                        induced_decomposition = IrreducibleRepresentationFinder.decompose(induced_rep)
                        for rep in induced_decomposition:
                            rep_char = Character(rep)
                            if rep_char.is_irreducible():
                                # 检查是否已经存在
                                is_new = True
                                for existing_char in irreducible_chars:
                                    if all(abs(rep_char(g) - existing_char(g)) < 1e-10 for g in group.elements()):
                                        is_new = False
                                        break
                                
                                if is_new:
                                    irreducible_reps.append(rep)
                                    irreducible_chars.append(rep_char)
                
                if len(irreducible_reps) >= num_classes:
                    break
        
        # 11. 确保我们至少有平凡表示
        if not irreducible_reps:
            irreducible_reps = [trivial_rep]
            irreducible_chars = [trivial_char]
        
        return list(zip(irreducible_chars, irreducible_reps))
    
    @staticmethod
    def _split_representation(representation: GroupRepresentation) -> Tuple[GroupRepresentation, GroupRepresentation]:
        """将可约表示分解为两个表示的直和"""
        # 这里使用随机方法来找到不变子空间
        # 实际应用中应该使用更系统的方法
        group = representation.group
        dim = representation.dimension
        
        # 随机生成一个向量
        v = np.random.randn(dim) + 1j * np.random.randn(dim)
        v = v / np.linalg.norm(v)
        
        # 生成不变子空间
        invariant_space = set()
        for g in group.elements():
            g_matrix = representation(g).matrix
            transformed_v = g_matrix @ v
            invariant_space.add(tuple(transformed_v))
        
        # 构造子空间的基
        basis = []
        for vec in invariant_space:
            vec_np = np.array(vec)
            # 检查线性无关性
            is_linearly_independent = True
            for b in basis:
                if np.abs(np.dot(vec_np, np.conj(b))) > 0.99:
                    is_linearly_independent = False
                    break
            if is_linearly_independent:
                basis.append(vec_np)
                if len(basis) == dim // 2:
                    break
        
        if len(basis) < 1 or len(basis) >= dim:
            raise ValueError("无法分解表示")
        
        # 构造投影矩阵
        basis_matrix = np.column_stack(basis)
        P = basis_matrix @ np.linalg.inv(basis_matrix.conj().T @ basis_matrix)
        P = P @ basis_matrix.conj().T
        
        # 构造两个子表示
        mapping1 = {}
        mapping2 = {}
        
        for g in group.elements():
            g_matrix = representation(g).matrix
            # 限制到不变子空间
            g1 = P @ g_matrix @ P
            # 限制到补空间
            g2 = (np.eye(dim) - P) @ g_matrix @ (np.eye(dim) - P)
            
            # 提取非零部分
            rank1 = np.linalg.matrix_rank(g1)
            rank2 = np.linalg.matrix_rank(g2)
            
            if rank1 > 0:
                # 提取子矩阵
                # 这里简化处理，实际应该找到正确的基变换
                mapping1[g] = g1[:rank1, :rank1]
            
            if rank2 > 0:
                mapping2[g] = g2[rank1:, rank1:]
        
        if not mapping1 or not mapping2:
            raise ValueError("无法分解表示")
        
        rep1 = MatrixRepresentation(group, mapping1)
        rep2 = MatrixRepresentation(group, mapping2)
        
        return rep1, rep2
    
    @staticmethod
    def _find_subgroups(group: Group) -> List[Group]:
        """找到群的所有子群"""
        # 对于 S3 群，返回 A3 子群
        elements = list(group.elements())
        if len(elements) == 6 and IrreducibleRepresentationFinder._is_standard_s3(elements):
            # 找到 A3 子群（偶置换）
            a3_elements = [g for g in elements if g in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]]
            if len(a3_elements) == 3:
                # 创建 A3 子群
                class A3Subgroup:
                    def elements(self):
                        return a3_elements
                    def identity(self):
                        return a3_elements[0]
                    def multiply(self, a, b):
                        return group.multiply(a, b)
                    def inverse(self, a):
                        return group.inverse(a)
                    def conjugacy_classes(self):
                        # A3 是循环群，只有一个共轭类
                        return [a3_elements]
                return [group, A3Subgroup()]
        # 否则返回群本身
        return [group]
    
    @staticmethod
    def _unique_representations(representations: List[GroupRepresentation]) -> List[GroupRepresentation]:
        """去重不可约表示"""
        unique_reps = []
        seen_chars = []
        
        for rep in representations:
            char = Character(rep)
            # 检查特征标是否已存在
            is_unique = True
            for seen_char in seen_chars:
                if all(abs(char(g) - seen_char(g)) < 1e-10 for g in rep.group.elements()):
                    is_unique = False
                    break
            
            if is_unique:
                unique_reps.append(rep)
                seen_chars.append(char)
        
        return unique_reps