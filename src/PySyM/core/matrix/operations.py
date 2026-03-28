from typing import Union, List
import numpy as np
from scipy.linalg import expm, logm, sqrtm

class MatrixOperations:
    """矩阵运算类"""
    
    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵加法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a + b
        """
        if a.shape != b.shape:
            raise ValueError("矩阵加法需要相同形状的矩阵")
        return a + b
    
    @staticmethod
    def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵减法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a - b
        """
        if a.shape != b.shape:
            raise ValueError("矩阵减法需要相同形状的矩阵")
        return a - b
    
    @staticmethod
    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩阵乘法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a @ b
        """
        if a.shape[1] != b.shape[0]:
            raise ValueError("矩阵乘法需要第一个矩阵的列数等于第二个矩阵的行数")
        return a @ b
    
    @staticmethod
    def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """逐元素乘法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a * b
        """
        if a.shape != b.shape:
            raise ValueError("逐元素乘法需要相同形状的矩阵")
        return a * b
    
    @staticmethod
    def elementwise_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """逐元素除法
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a / b
        """
        if a.shape != b.shape:
            raise ValueError("逐元素除法需要相同形状的矩阵")
        return a / b
    
    @staticmethod
    def transpose(matrix: np.ndarray) -> np.ndarray:
        """转置
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            matrix.T
        """
        return matrix.T
    
    @staticmethod
    def conjugate_transpose(matrix: np.ndarray) -> np.ndarray:
        """共轭转置
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            matrix.conj().T
        """
        return matrix.conj().T
    
    @staticmethod
    def kronecker_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """克罗内克积
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a ⊗ b
        """
        return np.kron(a, b)
    
    @staticmethod
    def hadamard_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """阿达玛积（逐元素乘法）
        
        Args:
            a: 第一个矩阵
            b: 第二个矩阵
            
        Returns:
            a ∘ b
        """
        if a.shape != b.shape:
            raise ValueError("阿达玛积需要相同形状的矩阵")
        return a * b
    
    @staticmethod
    def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """外积
        
        Args:
            a: 第一个向量
            b: 第二个向量
            
        Returns:
            a ⊗ b
        """
        return np.outer(a, b)
    
    @staticmethod
    def inner_product(a: np.ndarray, b: np.ndarray) -> Union[float, complex]:
        """内积
        
        Args:
            a: 第一个向量或矩阵
            b: 第二个向量或矩阵
            
        Returns:
            ⟨a, b⟩
        """
        return np.vdot(a, b)
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """点积
        
        Args:
            a: 第一个数组
            b: 第二个数组
            
        Returns:
            a · b
        """
        return np.dot(a, b)
    
    @staticmethod
    def power(matrix: np.ndarray, n: int) -> np.ndarray:
        """矩阵幂运算
        
        Args:
            matrix: 方阵
            n: 幂次
            
        Returns:
            matrix^n
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵幂运算需要方阵")
        return np.linalg.matrix_power(matrix, n)
    
    @staticmethod
    def inverse(matrix: np.ndarray) -> np.ndarray:
        """矩阵求逆
        
        Args:
            matrix: 可逆方阵
            
        Returns:
            matrix^(-1)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵求逆需要方阵")
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不可逆")
    
    @staticmethod
    def pseudo_inverse(matrix: np.ndarray) -> np.ndarray:
        """伪逆（Moore-Penrose逆）
        
        Args:
            matrix: 任意矩阵
            
        Returns:
            matrix^+
        """
        return np.linalg.pinv(matrix)
    
    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """求解线性方程组 Ax = b
        
        Args:
            A: 系数矩阵
            b: 右端向量或矩阵
            
        Returns:
            x: 解向量或矩阵
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("系数矩阵必须是方阵")
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError("线性方程组无解或有无限多解")
    
    @staticmethod
    def least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """最小二乘解
        
        Args:
            A: 系数矩阵
            b: 右端向量或矩阵
            
        Returns:
            x: 最小二乘解
        """
        return np.linalg.lstsq(A, b, rcond=None)[0]
    
    @staticmethod
    def matrix_exponential(matrix: np.ndarray) -> np.ndarray:
        """矩阵指数
        
        Args:
            matrix: 方阵
            
        Returns:
            exp(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵指数需要方阵")
        return expm(matrix)
    
    @staticmethod
    def matrix_logarithm(matrix: np.ndarray) -> np.ndarray:
        """矩阵对数
        
        Args:
            matrix: 可逆方阵
            
        Returns:
            log(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵对数需要方阵")
        try:
            return logm(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("矩阵对数计算失败，可能矩阵不可逆或不满足条件")
    
    @staticmethod
    def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        """矩阵平方根
        
        Args:
            matrix: 方阵
            
        Returns:
            sqrt(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵平方根需要方阵")
        try:
            return sqrtm(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("矩阵平方根计算失败，可能矩阵不满足条件")
    
    @staticmethod
    def trace(matrix: np.ndarray) -> float:
        """矩阵的迹
        
        Args:
            matrix: 方阵
            
        Returns:
            tr(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("矩阵的迹需要方阵")
        return np.trace(matrix)
    
    @staticmethod
    def determinant(matrix: np.ndarray) -> float:
        """行列式
        
        Args:
            matrix: 方阵
            
        Returns:
            det(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("行列式需要方阵")
        return float(np.linalg.det(matrix))
    
    @staticmethod
    def rank(matrix: np.ndarray) -> int:
        """矩阵的秩
        
        Args:
            matrix: 任意矩阵
            
        Returns:
            rank(matrix)
        """
        return np.linalg.matrix_rank(matrix)
    
    @staticmethod
    def norm(matrix: np.ndarray, ord: Union[int, float, str] = 'fro') -> float:
        """矩阵范数
        
        Args:
            matrix: 任意矩阵
            ord: 范数类型
            
        Returns:
            ||matrix||_ord
        """
        return np.linalg.norm(matrix, ord=ord)
    
    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """条件数
        
        Args:
            matrix: 方阵
            
        Returns:
            cond(matrix)
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("条件数需要方阵")
        return np.linalg.cond(matrix)
    
    @staticmethod
    def concatenate(matrices: List[np.ndarray], axis: int = 0) -> np.ndarray:
        """矩阵拼接
        
        Args:
            matrices: 矩阵列表
            axis: 拼接轴
            
        Returns:
            拼接后的矩阵
        """
        if not matrices:
            raise ValueError("矩阵列表不能为空")
        return np.concatenate(matrices, axis=axis)
    
    @staticmethod
    def stack(matrices: List[np.ndarray], axis: int = 0) -> np.ndarray:
        """矩阵堆叠
        
        Args:
            matrices: 矩阵列表
            axis: 堆叠轴
            
        Returns:
            堆叠后的矩阵
        """
        if not matrices:
            raise ValueError("矩阵列表不能为空")
        return np.stack(matrices, axis=axis)
    
    @staticmethod
    def split(matrix: np.ndarray, indices_or_sections: int, axis: int = 0) -> List[np.ndarray]:
        """矩阵分割
        
        Args:
            matrix: 输入矩阵
            indices_or_sections: 分割位置或数量
            axis: 分割轴
            
        Returns:
            分割后的矩阵列表
        """
        return np.split(matrix, indices_or_sections, axis=axis)
    
    @staticmethod
    def tile(matrix: np.ndarray, reps: tuple) -> np.ndarray:
        """矩阵重复
        
        Args:
            matrix: 输入矩阵
            reps: 重复次数
            
        Returns:
            重复后的矩阵
        """
        return np.tile(matrix, reps)
    
    @staticmethod
    def repeat(matrix: np.ndarray, repeats: int, axis: int = None) -> np.ndarray:
        """元素重复
        
        Args:
            matrix: 输入矩阵
            repeats: 重复次数
            axis: 重复轴
            
        Returns:
            重复后的矩阵
        """
        return np.repeat(matrix, repeats, axis=axis)