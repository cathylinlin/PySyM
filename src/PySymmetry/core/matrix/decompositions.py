import numpy as np
from scipy.linalg import hessenberg, lu, polar, schur


class MatrixDecompositions:
    """矩阵分解类"""

    @staticmethod
    def eigen_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """特征值分解

        Args:
            matrix: 方阵

        Returns:
            (eigenvalues, eigenvectors): 特征值和特征向量
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("特征值分解需要方阵")
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    @staticmethod
    def svd(
        matrix: np.ndarray, full_matrices: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """奇异值分解

        Args:
            matrix: 任意矩阵
            full_matrices: 是否返回完整的U和V矩阵

        Returns:
            (U, S, Vh): 左奇异向量、奇异值、右奇异向量的共轭转置
        """
        return np.linalg.svd(matrix, full_matrices=full_matrices)

    @staticmethod
    def qr_decomposition(
        matrix: np.ndarray, mode: str = "reduced"
    ) -> tuple[np.ndarray, np.ndarray]:
        """QR分解

        Args:
            matrix: 任意矩阵
            mode: 分解模式 ('reduced', 'complete', 'r', 'raw')

        Returns:
            (Q, R): 正交矩阵和上三角矩阵
        """
        return np.linalg.qr(matrix, mode=mode)

    @staticmethod
    def cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
        """Cholesky分解

        Args:
            matrix: 正定矩阵

        Returns:
            L: 下三角矩阵，满足 A = L @ L.T
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Cholesky分解需要方阵")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Cholesky分解需要对称矩阵")
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Cholesky分解需要正定矩阵")

    @staticmethod
    def lu_decomposition(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LU分解

        Args:
            matrix: 方阵

        Returns:
            (P, L, U): 置换矩阵、下三角矩阵、上三角矩阵
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("LU分解需要方阵")
        return lu(matrix)

    @staticmethod
    def schur_decomposition(
        matrix: np.ndarray, output: str = "real"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Schur分解

        Args:
            matrix: 方阵
            output: 输出类型 ('real' 或 'complex')

        Returns:
            (T, Z): Schur形式和酉矩阵
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Schur分解需要方阵")
        return schur(matrix, output=output)

    @staticmethod
    def hessenberg_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hessenberg分解

        Args:
            matrix: 方阵

        Returns:
            (H, Q): Hessenberg形式和正交矩阵
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Hessenberg分解需要方阵")
        return hessenberg(matrix)

    @staticmethod
    def polar_decomposition(
        matrix: np.ndarray, side: str = "right"
    ) -> tuple[np.ndarray, np.ndarray]:
        """极分解

        Args:
            matrix: 任意矩阵
            side: 分解方向 ('left' 或 'right')

        Returns:
            (U, P): 酉矩阵和半正定矩阵
        """
        return polar(matrix, side=side)

    @staticmethod
    def spectral_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """谱分解（仅适用于对称/埃尔米特矩阵）

        Args:
            matrix: 对称或埃尔米特矩阵

        Returns:
            (eigenvalues, eigenvectors): 特征值和特征向量
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("谱分解需要方阵")
        if not np.allclose(matrix, matrix.T.conj()):
            raise ValueError("谱分解需要对称或埃尔米特矩阵")
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return eigenvalues, eigenvectors

    @staticmethod
    def jordan_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Jordan分解（数值近似）

        Args:
            matrix: 方阵

        Returns:
            (P, J): 变换矩阵和Jordan标准形
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Jordan分解需要方阵")
        # 注意：实际的Jordan分解在数值计算中很难精确实现
        # 这里使用特征值分解作为近似
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        # 构造近似的Jordan矩阵（仅对角元素）
        J = np.diag(eigenvalues)
        return eigenvectors, J

    @staticmethod
    def bidiagonal_decomposition(
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """双对角分解

        Args:
            matrix: 任意矩阵

        Returns:
            (U, B, Vh): 左酉矩阵、双对角矩阵、右酉矩阵的共轭转置
        """
        U, S, Vh = np.linalg.svd(matrix, full_matrices=True)
        # 构造双对角矩阵
        B = np.zeros(matrix.shape, dtype=matrix.dtype)
        np.fill_diagonal(B, S)
        return U, B, Vh
