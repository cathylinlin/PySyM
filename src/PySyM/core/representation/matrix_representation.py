import numpy as np
from typing import Dict,TypeVar, List
from .abstract_representation import GroupRepresentation
from ..group_theory.abstract_group import Group, GroupElement
from ..matrix_groups.general_linear import GLnElement

T = TypeVar('T', bound=GroupElement)

class MatrixRepresentation(GroupRepresentation[T]):
    """矩阵表示实现
    
    将群元素映射为n×n矩阵的表示
    """
    
    def __init__(self, group: Group[T], mapping: Dict[T, np.ndarray]):
        """
        Args:
            group: 被表示的群
            mapping: 群元素到矩阵的映射字典
        """
        if not mapping:
            raise ValueError("映射不能为空")

        identity = group.identity()
        if identity not in mapping:
            raise ValueError("映射必须包含单位元")

        elements = list(group.elements())
        missing = [g for g in elements if g not in mapping]
        if missing:
            raise ValueError(f"映射缺少群元素: {missing[:3]}")

        dim = mapping[identity].shape[0]
        for g, matrix in mapping.items():
            if matrix.shape != (dim, dim):
                raise ValueError(f"元素 {g} 对应矩阵维度不一致")
            # 表示值应位于 GL(n)，即必须可逆
            if np.isclose(np.linalg.det(matrix), 0.0):
                raise ValueError(f"元素 {g} 的像矩阵不可逆，非法群表示")

        super().__init__(group, dimension=dim)
        self._mapping = mapping
    
    def __call__(self, element: T) -> GLnElement:
        """获取群元素的矩阵表示"""
        if element not in self._mapping:
            raise KeyError(f"未找到元素 {element} 的表示矩阵")
        return GLnElement(self._mapping[element])
    
    def is_homomorphism(self) -> bool:
        """验证同态性质"""
        for a in self._group.elements():
            for b in self._group.elements():
                if not np.allclose(
                    self._mapping[self._group.multiply(a, b)],
                    self._mapping[a] @ self._mapping[b]
                ):
                    return False
        return True
    
    def compose(self, other: 'MatrixRepresentation') -> 'MatrixRepresentation':
        """表示的张量积"""
        if self._group != other._group:
            raise ValueError("Tensors can only be composed for the same group")
        
        tensor_mapping = {}
        for g in self._group.elements():
            # 计算两个矩阵的张量积
            tensor_matrix = np.kron(self._mapping[g], other._mapping[g])
            tensor_mapping[g] = tensor_matrix
        
        return MatrixRepresentation(self._group, tensor_mapping)
    
    def direct_sum(self, other: 'MatrixRepresentation') -> 'MatrixRepresentation':
        """表示的直和"""
        if self._group != other._group:
            raise ValueError("Direct sum can only be composed for the same group")
        
        sum_mapping = {}
        dim1 = self.dimension
        dim2 = other.dimension
        
        for g in self._group.elements():
            # 构造直和矩阵
            matrix1 = self._mapping[g]
            matrix2 = other._mapping[g]
            
            # 创建块对角矩阵
            sum_matrix = np.zeros((dim1 + dim2, dim1 + dim2), dtype=complex)
            sum_matrix[:dim1, :dim1] = matrix1
            sum_matrix[dim1:, dim1:] = matrix2
            
            sum_mapping[g] = sum_matrix
        
        return MatrixRepresentation(self._group, sum_mapping)
    
    @classmethod
    def trivial_representation(cls, group: Group[T]) -> 'MatrixRepresentation':
        """构造平凡表示"""
        mapping = {g: np.array([[1]]) for g in group.elements()}
        return cls(group, mapping)
    
    @classmethod
    def regular_representation(cls, group: Group[T]) -> 'MatrixRepresentation':
        """构造正则表示"""
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
        
        return cls(group, mapping)
    
    def change_basis(self, basis: np.ndarray) -> 'MatrixRepresentation':
        """通过基变换得到等价表示"""
        if basis.shape != (self.dimension, self.dimension):
            raise ValueError("Basis matrix must be square with dimension equal to representation dimension")
        
        # 计算逆矩阵
        basis_inv = np.linalg.inv(basis)
        
        # 构造新的映射
        new_mapping = {}
        for g in self._group.elements():
            old_matrix = self._mapping[g]
            new_matrix = basis_inv @ old_matrix @ basis
            new_mapping[g] = new_matrix
        
        return MatrixRepresentation(self._group, new_mapping)
    
    def is_irreducible(self) -> bool:
        """检查表示是否不可约"""
        from .character import Character
        char = Character(self)
        return char.is_irreducible()
    
    def decompose(self) -> List['MatrixRepresentation']:
        """将可约表示分解为不可约表示的直和"""
        from .irreducible import IrreducibleRepresentationFinder
        return IrreducibleRepresentationFinder.decompose(self)
    
    def kernel(self) -> List[T]:
        """计算表示的核"""
        kernel_elements = []
        identity_matrix = np.eye(self.dimension)
        for element in self._group.elements():
            if np.allclose(self._mapping[element], identity_matrix):
                kernel_elements.append(element)
        return kernel_elements
    
    def image(self) -> List[np.ndarray]:
        """计算表示的像"""
        image_matrices = []
        tol = 1e-10
        
        for element in self._group.elements():
            matrix = self._mapping[element]
            if not any(np.allclose(matrix, seen, atol=tol) for seen in image_matrices):
                image_matrices.append(matrix)
        
        return image_matrices