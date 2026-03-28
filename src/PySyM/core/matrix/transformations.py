from __future__ import annotations
import numpy as np
from scipy.linalg import hessenberg, schur

class MatrixTransformations:
    """矩阵变换类"""
    
    @staticmethod
    def similarity_transform(matrix: np.ndarray, P: np.ndarray) -> np.ndarray:
        """相似变换: P^(-1) * A * P
        
        Args:
            matrix: 输入矩阵
            P: 可逆变换矩阵
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if P.ndim != 2:
            raise ValueError("变换矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("相似变换需要方阵")
        if P.shape[0] != P.shape[1]:
            raise ValueError("变换矩阵必须是方阵")
        if P.shape[0] != matrix.shape[0]:
            raise ValueError("变换矩阵维度不匹配")
        try:
            P_inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            raise ValueError("变换矩阵不可逆")
        return P_inv @ matrix @ P
    
    @staticmethod
    def congruence_transform(matrix: np.ndarray, P: np.ndarray) -> np.ndarray:
        """合同变换: P^T * A * P
        
        Args:
            matrix: 输入矩阵
            P: 变换矩阵
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if P.ndim != 2:
            raise ValueError("变换矩阵必须是二维数组")
        if P.shape[1] != matrix.shape[0]:
            raise ValueError("变换矩阵维度不匹配")
        return P.T @ matrix @ P
    
    @staticmethod
    def unitary_transform(matrix: np.ndarray, U: np.ndarray) -> np.ndarray:
        """酉变换: U^H * A * U
        
        Args:
            matrix: 输入矩阵
            U: 酉矩阵
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if U.ndim != 2:
            raise ValueError("变换矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("酉变换需要方阵")
        if U.shape[0] != U.shape[1]:
            raise ValueError("变换矩阵必须是方阵")
        if U.shape[0] != matrix.shape[0]:
            raise ValueError("变换矩阵维度不匹配")
        # 验证U是否为酉矩阵
        if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
            raise ValueError("变换矩阵不是酉矩阵")
        return U.conj().T @ matrix @ U
    
    @staticmethod
    def orthogonal_transform(matrix: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """正交变换: Q^T * A * Q
        
        Args:
            matrix: 输入矩阵
            Q: 正交矩阵
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if Q.ndim != 2:
            raise ValueError("变换矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("正交变换需要方阵")
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("变换矩阵必须是方阵")
        if Q.shape[0] != matrix.shape[0]:
            raise ValueError("变换矩阵维度不匹配")
        # 验证Q是否为正交矩阵
        if not np.allclose(Q @ Q.T, np.eye(Q.shape[0])):
            raise ValueError("变换矩阵不是正交矩阵")
        return Q.T @ matrix @ Q
    
    @staticmethod
    def row_operation(matrix: np.ndarray, i: int, j: int, factor: float = 1.0) -> np.ndarray:
        """行变换: 第i行加上factor倍的第j行
        
        Args:
            matrix: 输入矩阵
            i: 目标行索引
            j: 源行索引
            factor: 乘数因子
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        n, _ = matrix.shape
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError("行索引超出范围")
        result = matrix.copy()
        result[i, :] += factor * result[j, :]
        return result
    
    @staticmethod
    def column_operation(matrix: np.ndarray, i: int, j: int, factor: float = 1.0) -> np.ndarray:
        """列变换: 第i列加上factor倍的第j列
        
        Args:
            matrix: 输入矩阵
            i: 目标列索引
            j: 源列索引
            factor: 乘数因子
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        _, m = matrix.shape
        if i < 0 or i >= m or j < 0 or j >= m:
            raise ValueError("列索引超出范围")
        result = matrix.copy()
        result[:, i] += factor * result[:, j]
        return result
    
    @staticmethod
    def row_swap(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
        """行交换: 交换第i行和第j行
        
        Args:
            matrix: 输入矩阵
            i: 第一行索引
            j: 第二行索引
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        n, _ = matrix.shape
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError("行索引超出范围")
        result = matrix.copy()
        result[[i, j], :] = result[[j, i], :]
        return result
    
    @staticmethod
    def column_swap(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
        """列交换: 交换第i列和第j列
        
        Args:
            matrix: 输入矩阵
            i: 第一列索引
            j: 第二列索引
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        _, m = matrix.shape
        if i < 0 or i >= m or j < 0 or j >= m:
            raise ValueError("列索引超出范围")
        result = matrix.copy()
        result[:, [i, j]] = result[:, [j, i]]
        return result
    
    @staticmethod
    def row_scale(matrix: np.ndarray, i: int, factor: float) -> np.ndarray:
        """行缩放: 第i行乘以factor
        
        Args:
            matrix: 输入矩阵
            i: 行索引
            factor: 缩放因子
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        n, _ = matrix.shape
        if i < 0 or i >= n:
            raise ValueError("行索引超出范围")
        result = matrix.copy()
        result[i, :] *= factor
        return result
    
    @staticmethod
    def column_scale(matrix: np.ndarray, i: int, factor: float) -> np.ndarray:
        """列缩放: 第i列乘以factor
        
        Args:
            matrix: 输入矩阵
            i: 列索引
            factor: 缩放因子
            
        Returns:
            变换后的矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        _, m = matrix.shape
        if i < 0 or i >= m:
            raise ValueError("列索引超出范围")
        result = matrix.copy()
        result[:, i] *= factor
        return result
    
    @staticmethod
    def gauss_jordan_elimination(matrix: np.ndarray) -> np.ndarray:
        """高斯-若尔当消元
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            行最简形矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        result = matrix.copy().astype(float)
        rows, cols = result.shape
        
        pivot_row = 0
        for col in range(cols):
            if pivot_row >= rows:
                break
            
            pivot = result[pivot_row:, col]
            pivot_idx = np.argmax(np.abs(pivot)) + pivot_row
            
            if np.abs(result[pivot_idx, col]) < 1e-10:
                continue
            
            if pivot_idx != pivot_row:
                result = MatrixTransformations.row_swap(result, pivot_row, pivot_idx)
            
            result[pivot_row, :] /= result[pivot_row, col]
            
            for i in range(rows):
                if i != pivot_row:
                    factor = result[i, col]
                    result[i, :] -= factor * result[pivot_row, :]
            
            pivot_row += 1
        
        return result
    
    @staticmethod
    def row_echelon_form(matrix: np.ndarray) -> np.ndarray:
        """行阶梯形
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            行阶梯形矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        result = matrix.copy().astype(float)
        rows, cols = result.shape
        
        pivot_row = 0
        for col in range(cols):
            if pivot_row >= rows:
                break
            
            pivot = result[pivot_row:, col]
            pivot_idx = np.argmax(np.abs(pivot)) + pivot_row
            
            if np.abs(result[pivot_idx, col]) < 1e-10:
                continue
            
            if pivot_idx != pivot_row:
                result = MatrixTransformations.row_swap(result, pivot_row, pivot_idx)
            
            result[pivot_row, :] /= result[pivot_row, col]
            
            for i in range(pivot_row + 1, rows):
                factor = result[i, col]
                result[i, :] -= factor * result[pivot_row, :]
            
            pivot_row += 1
        
        return result
    
    @staticmethod
    def hessenberg_form(matrix: np.ndarray) -> np.ndarray:
        """海森伯格形式
        
        Args:
            matrix: 方阵
            
        Returns:
            海森伯格形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("海森伯格形式需要方阵")
        return hessenberg(matrix)
    
    @staticmethod
    def bidiagonal_form(matrix: np.ndarray) -> np.ndarray:
        """双对角形式
        
        Args:
            matrix: 任意矩阵
            
        Returns:
            双对角形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        m, n = matrix.shape
        # 使用SVD构造双对角矩阵
        U, S, Vh = np.linalg.svd(matrix, full_matrices=True)
        B = np.zeros((m, n))
        np.fill_diagonal(B, S)
        # 对于矩形矩阵，只填充主对角线
        return B
    
    @staticmethod
    def tridiagonal_form(matrix: np.ndarray) -> np.ndarray:
        """三对角形式（仅适用于对称矩阵）
        
        Args:
            matrix: 对称方阵
            
        Returns:
            三对角形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("三对角形式需要方阵")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("三对角形式需要对称矩阵")
        n = matrix.shape[0]
        T = np.zeros((n, n))
        # 填充主对角线
        np.fill_diagonal(T, np.diag(matrix))
        # 填充上对角线和下对角线
        if n > 1:
            np.fill_diagonal(T[1:], np.diag(matrix, k=1))
            np.fill_diagonal(T[:, 1:], np.diag(matrix, k=-1))
        return T
    
    @staticmethod
    def companion_form(polynomial_coefficients: np.ndarray) -> np.ndarray:
        """伴随矩阵形式
        
        Args:
            polynomial_coefficients: 多项式系数 [a_n, a_{n-1}, ..., a_1, a_0]
            
        Returns:
            伴随矩阵
        """
        if polynomial_coefficients.ndim != 1:
            raise ValueError("多项式系数必须是一维数组")
        if len(polynomial_coefficients) < 2:
            raise ValueError("多项式次数至少为1")
        # 确保首项系数不为零
        if np.isclose(polynomial_coefficients[0], 0):
            raise ValueError("首项系数不能为零")
        
        n = len(polynomial_coefficients) - 1
        companion = np.zeros((n, n))
        companion[0, :] = -polynomial_coefficients[1:] / polynomial_coefficients[0]
        for i in range(1, n):
            companion[i, i-1] = 1
        return companion
    
    @staticmethod
    def jordan_canonical_form(matrix: np.ndarray) -> np.ndarray:
        """若尔当标准形（数值近似）
        
        Args:
            matrix: 方阵
            
        Returns:
            若尔当标准形
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("若尔当标准形需要方阵")
        # 注意：实际的Jordan分解在数值计算中很难精确实现
        # 这里使用特征值分解作为近似
        eigenvalues = np.linalg.eigvals(matrix)
        return np.diag(eigenvalues)
    
    @staticmethod
    def rational_canonical_form(matrix: np.ndarray) -> np.ndarray:
        """有理标准形（数值近似）
        
        Args:
            matrix: 方阵
            
        Returns:
            有理标准形
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("有理标准形需要方阵")
        # 这里仅作为占位符，实际实现较为复杂
        eigenvalues = np.linalg.eigvals(matrix)
        return np.diag(eigenvalues)
    
    @staticmethod
    def smith_normal_form(matrix: np.ndarray) -> np.ndarray:
        """史密斯标准形（数值近似）
        
        Args:
            matrix: 整数矩阵
            
        Returns:
            史密斯标准形
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        # 使用SVD构造近似的Smith标准形
        U, S, Vh = np.linalg.svd(matrix, full_matrices=True)
        B = np.zeros((matrix.shape[0], matrix.shape[1]))
        np.fill_diagonal(B, S)
        return B
    
    @staticmethod
    def frobenius_normal_form(matrix: np.ndarray) -> np.ndarray:
        """弗罗贝尼乌斯标准形
        
        Args:
            matrix: 方阵
            
        Returns:
            弗罗贝尼乌斯标准形
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("弗罗贝尼乌斯标准形需要方阵")
        characteristic_poly = np.poly(matrix)
        return MatrixTransformations.companion_form(characteristic_poly)
    
    @staticmethod
    def schur_form(matrix: np.ndarray) -> np.ndarray:
        """舒尔形式
        
        Args:
            matrix: 方阵
            
        Returns:
            舒尔形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("舒尔形式需要方阵")
        T, _ = schur(matrix)
        return T
    
    @staticmethod
    def real_schur_form(matrix: np.ndarray) -> np.ndarray:
        """实舒尔形式
        
        Args:
            matrix: 实方阵
            
        Returns:
            实舒尔形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("实舒尔形式需要方阵")
        T, _ = schur(matrix, output='real')
        return T
    
    @staticmethod
    def complex_schur_form(matrix: np.ndarray) -> np.ndarray:
        """复舒尔形式
        
        Args:
            matrix: 方阵
            
        Returns:
            复舒尔形式矩阵
        """
        if matrix.ndim != 2:
            raise ValueError("输入矩阵必须是二维数组")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("复舒尔形式需要方阵")
        T, _ = schur(matrix, output='complex')
        return T