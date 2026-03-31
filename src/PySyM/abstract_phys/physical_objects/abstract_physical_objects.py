"""
物理对象抽象基类

定义物理对象和物理空间的抽象接口，为具体实现提供统一的规范。
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING, Protocol, Union
import numpy as np
from PySyM.core.group_theory.abstract_group import Group
from PySyM.core.matrix.base import AbstractMatrix

if TYPE_CHECKING:
    from ..symmetry_operations.base import SymmetryOperation

PhysicalQuantity = Union[float, complex, np.ndarray]


class SymmetryGroup(Protocol):
    """对称群协议"""
    
    def identity(self): ...
    def multiply(self, a, b): ...
    def inverse(self, a): ...
    def __contains__(self, element) -> bool: ...

class PhysicalObject(ABC):
    """物理对象抽象基类
    
    所有物理对象（如粒子、场等）的基类，定义了基本的物理属性和对称性操作接口。
    与 core 模块的 Group、Matrix 等类有良好的集成。
    """
    
    _symmetry_group: Optional['Group'] = None
    
    @property
    @abstractmethod
    def symmetry_properties(self) -> Dict[str, Any]:
        """返回对称性性质
        
        返回一个包含对象对称性相关属性的字典，用于对称性分析。
        """
        pass
    
    @abstractmethod
    def transform(self, symmetry_operation: 'SymmetryOperation') -> 'PhysicalObject':
        """在对称操作下变换
        
        Args:
            symmetry_operation: 对称操作对象
            
        Returns:
            变换后的物理对象
        """
        pass
    
    @abstractmethod
    def is_invariant_under(self, symmetry_operation: 'SymmetryOperation') -> bool:
        """检查是否在某对称操作下不变
        
        Args:
            symmetry_operation: 对称操作对象
            
        Returns:
            布尔值，表示对象是否在该对称操作下不变
        """
        pass
    
    @abstractmethod
    def get_mass(self) -> PhysicalQuantity:
        """获取质量
        
        Returns:
            物体的质量
        """
        pass
    
    @abstractmethod
    def get_charge(self) -> PhysicalQuantity:
        """获取电荷
        
        Returns:
            物体的电荷
        """
        pass
    
    @abstractmethod
    def get_spin(self) -> PhysicalQuantity:
        """获取自旋
        
        Returns:
            物体的自旋
        """
        pass
    
    @property
    def symmetry_group(self) -> Optional['Group']:
        """获取对象的对称群
        
        Returns:
            对称群对象
        """
        return self._symmetry_group
    
    @symmetry_group.setter
    def symmetry_group(self, group: 'Group') -> None:
        """设置对象的对称群
        
        Args:
            group: 对称群对象
        """
        self._symmetry_group = group
    
    def get_representation_matrix(self, rep_type: str = 'vector') -> Optional[np.ndarray]:
        """获取对象在某表示下的矩阵形式
        
        Args:
            rep_type: 表示类型 ('vector', 'matrix', 'tensor')
            
        Returns:
            表示矩阵
        """
        return None
    
    def check_symmetry_group(self) -> bool:
        """检查对象的对称群是否合法
        
        Returns:
            是否满足对称群公理
        """
        if self._symmetry_group is None:
            return True
        
        group = self._symmetry_group
        identity = group.identity()
        
        for elem in group.elements():
            op_result = self.transform(self._create_operation_from_group_element(elem))
            if not self._verify_transform_result(op_result):
                return False
        
        return True
    
    def _create_operation_from_group_element(self, element) -> 'SymmetryOperation':
        """从群元素创建对称操作
        
        Args:
            element: 群元素
            
        Returns:
            对应的对称操作
        """
        from ..symmetry_operations.specific_operations import IdentityOperation
        return IdentityOperation()
    
    def _verify_transform_result(self, result: 'PhysicalObject') -> bool:
        """验证变换结果的合法性
        
        Args:
            result: 变换后的对象
            
        Returns:
            是否合法
        """
        return result is not None
    
    def __repr__(self):
        """返回对象的字符串表示
        
        Returns:
            包含对象基本属性的字符串
        """
        return f"{self.__class__.__name__}(mass={self.get_mass()}, charge={self.get_charge()}, spin={self.get_spin()})"



class PhysicalSpace(ABC):
    """
    物理空间基类
    
    定义物理系统所在的数学空间，提供维度、内积和范数等基本操作。
    与 core.matrix 模块有良好的集成。
    """
    
    _metric_tensor: Optional[np.ndarray] = None
    _connection: Optional[np.ndarray] = None
    
    @abstractmethod
    def dimension(self) -> int:
        """空间维度
        
        Returns:
            空间的维度数
        """
        pass
    
    @abstractmethod
    def inner_product(self, x: Any, y: Any) -> Any:
        """内积
        
        Args:
            x: 空间中的第一个向量
            y: 空间中的第二个向量
            
        Returns:
            两个向量的内积
        """
        pass
    
    @abstractmethod
    def norm(self, x: Any) -> Any:
        """范数
        
        Args:
            x: 空间中的向量
            
        Returns:
            向量的范数
        """
        pass
    
    @property
    def metric_tensor(self) -> Optional[np.ndarray]:
        """获取度规张量
        
        Returns:
            度规张量矩阵
        """
        return self._metric_tensor
    
    @metric_tensor.setter
    def metric_tensor(self, g: np.ndarray) -> None:
        """设置度规张量
        
        Args:
            g: 度规张量矩阵
        """
        if g.shape[0] != g.shape[1]:
            raise ValueError("度规张量必须是方阵")
        if g.shape[0] != self.dimension():
            raise ValueError(f"度规张量维度 {g.shape[0]} 与空间维度 {self.dimension()} 不匹配")
        self._metric_tensor = np.array(g)
    
    def get_metric_matrix(self) -> np.ndarray:
        """获取度规矩阵
        
        Returns:
            度规矩阵，若未设置则返回欧氏度规
        """
        if self._metric_tensor is not None:
            return self._metric_tensor
        return np.eye(self.dimension())
    
    def inner_product_with_metric(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用度规计算内积
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            内积结果
        """
        g = self.get_metric_matrix()
        return np.dot(x, np.dot(g, y))
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两点间的距离
        
        Args:
            x: 第一个点
            y: 第二个点
            
        Returns:
            距离
        """
        diff = np.array(x) - np.array(y)
        return float(np.sqrt(np.abs(self.inner_product_with_metric(diff, diff))))
    
    def angle(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两个向量间的夹角
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            夹角（弧度）
        """
        g = self.get_metric_matrix()
        dot_product = np.dot(x, np.dot(g, y))
        norm_x = np.sqrt(np.abs(np.dot(x, np.dot(g, x))))
        norm_y = np.sqrt(np.abs(np.dot(y, np.dot(g, y))))
        if norm_x == 0 or norm_y == 0:
            return 0.0
        return float(np.arccos(np.clip(dot_product / (norm_x * norm_y), -1, 1)))
    
    def is_orthogonal(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-10) -> bool:
        """检查两个向量是否正交
        
        Args:
            x: 第一个向量
            y: 第二个向量
            tol: 容差
            
        Returns:
            是否正交
        """
        inner = float(np.abs(self.inner_product_with_metric(x, y)))
        return inner < tol
    
    def project_onto(self, v: np.ndarray, subspace_bases: list) -> np.ndarray:
        """将向量投影到子空间
        
        Args:
            v: 被投影的向量
            subspace_bases: 子空间基向量列表
            
        Returns:
            投影后的向量
        """
        bases = np.array(subspace_bases)
        if bases.ndim == 1:
            bases = bases.reshape(1, -1)
        
        G = np.array([[self.inner_product_with_metric(bi, bj) for bj in bases] for bi in bases])
        v_proj = np.array([self.inner_product_with_metric(v, b) for b in bases])
        
        try:
            coeffs = np.linalg.solve(G, v_proj)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(G, v_proj, rcond=None)[0]
        
        return np.sum(coeffs[:, np.newaxis] * bases, axis=0)