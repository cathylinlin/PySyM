from typing import Union
import numpy as np
from scipy.linalg import null_space, orth

class MatrixProperties:
    """矩阵性质类"""
    
    @staticmethod
    def is_square(matrix: np.ndarray) -> bool:
        """判断是否为方阵
        
        Args:
            matrix: 任意矩阵
            
        Returns:
            是否为方阵
        """
        if matrix.ndim != 2:
            return False
        return matrix.shape[0] == matrix.shape[1]
    
    @staticmethod
    def is_symmetric(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为对称矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为对称矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_hermitian(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为埃尔米特矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为埃尔米特矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        return np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_orthogonal(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为正交矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为正交矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        product = matrix @ matrix.T
        identity = np.eye(matrix.shape[0])
        return np.allclose(product, identity, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_unitary(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为酉矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为酉矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        product = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0])
        return np.allclose(product, identity, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_positive_definite(matrix: np.ndarray) -> bool:
        """判断是否为正定矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            是否为正定矩阵
        """
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def is_positive_semidefinite(matrix: np.ndarray) -> bool:
        """判断是否为半正定矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            是否为半正定矩阵
        """
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.all(eigenvalues >= 0)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def is_negative_definite(matrix: np.ndarray) -> bool:
        """判断是否为负定矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            是否为负定矩阵
        """
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.all(eigenvalues < 0)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def is_negative_semidefinite(matrix: np.ndarray) -> bool:
        """判断是否为半负定矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            是否为半负定矩阵
        """
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.all(eigenvalues <= 0)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def is_indefinite(matrix: np.ndarray) -> bool:
        """判断是否为不定矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            是否为不定矩阵
        """
        if not MatrixProperties.is_hermitian(matrix):
            return False
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.any(eigenvalues > 0) and np.any(eigenvalues < 0)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def is_singular(matrix: np.ndarray) -> bool:
        """判断是否为奇异矩阵
        
        Args:
            matrix: 方阵
            
        Returns:
            是否为奇异矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        try:
            det = np.linalg.det(matrix)
            return np.isclose(det, 0)
        except np.linalg.LinAlgError:
            return True
    
    @staticmethod
    def is_invertible(matrix: np.ndarray) -> bool:
        """判断是否可逆
        
        Args:
            matrix: 方阵
            
        Returns:
            是否可逆
        """
        return not MatrixProperties.is_singular(matrix)
    
    @staticmethod
    def is_diagonal(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为对角矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为对角矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        off_diagonal = matrix - np.diag(np.diag(matrix))
        return np.allclose(off_diagonal, 0, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_triangular(matrix: np.ndarray, lower: bool = True, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为三角矩阵
        
        Args:
            matrix: 输入矩阵
            lower: 是否为下三角矩阵（True为下三角，False为上三角）
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为三角矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        n = matrix.shape[0]
        if lower:
            upper_part = matrix[np.triu_indices(n, k=1)]
            return np.allclose(upper_part, 0, rtol=rtol, atol=atol)
        else:
            lower_part = matrix[np.tril_indices(n, k=-1)]
            return np.allclose(lower_part, 0, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_upper_triangular(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为上三角矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为上三角矩阵
        """
        return MatrixProperties.is_triangular(matrix, lower=False, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_lower_triangular(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为下三角矩阵
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为下三角矩阵
        """
        return MatrixProperties.is_triangular(matrix, lower=True, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_normal(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为正规矩阵（满足 A*A^H = A^H*A）
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为正规矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        return np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_idempotent(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为幂等矩阵（满足 A^2 = A）
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为幂等矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        return np.allclose(matrix @ matrix, matrix, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_involutory(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为对合矩阵（满足 A^2 = I）
        
        Args:
            matrix: 输入矩阵
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为对合矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix @ matrix, identity, rtol=rtol, atol=atol)
    
    @staticmethod
    def is_nilpotent(matrix: np.ndarray, power: int = None, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """判断是否为幂零矩阵（存在k使得A^k = 0）
        
        Args:
            matrix: 输入矩阵
            power: 指定幂次（如果为None，则自动判断）
            rtol: 相对容差
            atol: 绝对容差
            
        Returns:
            是否为幂零矩阵
        """
        if not MatrixProperties.is_square(matrix):
            return False
        n = matrix.shape[0]
        if power is None:
            power = n
        result = np.eye(n)
        for _ in range(power):
            result = result @ matrix
            if np.allclose(result, 0, rtol=rtol, atol=atol):
                return True
        return False
    
    @staticmethod
    def is_sparse(matrix: np.ndarray, threshold: float = 0.5) -> bool:
        """判断是否为稀疏矩阵
        
        Args:
            matrix: 输入矩阵
            threshold: 稀疏度阈值
            
        Returns:
            是否为稀疏矩阵
        """
        if matrix.ndim != 2:
            return False
        zero_ratio = np.sum(np.abs(matrix) < 1e-10) / matrix.size
        return zero_ratio > threshold
    
    @staticmethod
    def null_space(matrix: np.ndarray, tol: float = None) -> np.ndarray:
        """计算零空间
        
        Args:
            matrix: 输入矩阵
            tol: 容差
            
        Returns:
            零空间的基向量
        """
        return null_space(matrix, tol=tol)
    
    @staticmethod
    def column_space(matrix: np.ndarray) -> np.ndarray:
        """计算列空间
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            列空间的基向量
        """
        return orth(matrix.T).T
    
    @staticmethod
    def row_space(matrix: np.ndarray) -> np.ndarray:
        """计算行空间
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            行空间的基向量
        """
        return orth(matrix)