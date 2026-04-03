import numpy as np
from scipy.linalg import circulant, hankel, hilbert, invhilbert, pascal, companion, hadamard, block_diag
from typing import Union, List, Callable

class MatrixFactory:
    """矩阵工厂类"""
    
    @staticmethod
    def zeros(rows: int, cols: int) -> np.ndarray:
        """零矩阵
        
        Args:
            rows: 行数
            cols: 列数
            
        Returns:
            零矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        return np.zeros((rows, cols))
    
    @staticmethod
    def ones(rows: int, cols: int) -> np.ndarray:
        """全1矩阵
        
        Args:
            rows: 行数
            cols: 列数
            
        Returns:
            全1矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        return np.ones((rows, cols))
    
    @staticmethod
    def identity(n: int) -> np.ndarray:
        """单位矩阵
        
        Args:
            n: 维度
            
        Returns:
            单位矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        return np.eye(n)
    
    @staticmethod
    def random(rows: int, cols: int) -> np.ndarray:
        """随机矩阵（0-1均匀分布）
        
        Args:
            rows: 行数
            cols: 列数
            
        Returns:
            随机矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        return np.random.rand(rows, cols)
    
    @staticmethod
    def random_normal(rows: int, cols: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """正态分布随机矩阵
        
        Args:
            rows: 行数
            cols: 列数
            mean: 均值
            std: 标准差
            
        Returns:
            正态分布随机矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        if std < 0:
            raise ValueError("标准差必须非负")
        return np.random.normal(mean, std, (rows, cols))
    
    @staticmethod
    def from_diagonal(diagonal: np.ndarray) -> np.ndarray:
        """从对角元素构造矩阵
        
        Args:
            diagonal: 对角元素数组
            
        Returns:
            对角矩阵
        """
        if diagonal.ndim != 1:
            raise ValueError("对角元素必须是一维数组")
        return np.diag(diagonal)
    
    @staticmethod
    def from_list(data: List[List[float]]) -> np.ndarray:
        """从列表构造矩阵
        
        Args:
            data: 二维列表
            
        Returns:
            矩阵
        """
        if not data:
            raise ValueError("列表不能为空")
        return np.array(data)
    
    @staticmethod
    def from_function(rows: int, cols: int, func: Callable[[int, int], float]) -> np.ndarray:
        """从函数构造矩阵
        
        Args:
            rows: 行数
            cols: 列数
            func: 函数 func(i, j) 返回元素值
            
        Returns:
            矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        # 使用numpy的向量化操作提高性能
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        matrix = np.vectorize(func)(i, j)
        return matrix
    
    @staticmethod
    def toeplitz(first_row: np.ndarray, first_col: np.ndarray = None) -> np.ndarray:
        """托普利茨矩阵
        
        Args:
            first_row: 第一行
            first_col: 第一列（如果为None，则使用first_row的转置）
            
        Returns:
            托普利茨矩阵
        """
        if first_row.ndim != 1:
            raise ValueError("第一行必须是一维数组")
        if first_col is not None and first_col.ndim != 1:
            raise ValueError("第一列必须是一维数组")
        return np.linalg.toeplitz(first_col if first_col is not None else first_row, first_row)
    
    @staticmethod
    def circulant(first_row: np.ndarray) -> np.ndarray:
        """循环矩阵
        
        Args:
            first_row: 第一行
            
        Returns:
            循环矩阵
        """
        if first_row.ndim != 1:
            raise ValueError("第一行必须是一维数组")
        return circulant(first_row)
    
    @staticmethod
    def hankel(first_row: np.ndarray, last_col: np.ndarray = None) -> np.ndarray:
        """汉克尔矩阵
        
        Args:
            first_row: 第一行
            last_col: 最后一列
            
        Returns:
            汉克尔矩阵
        """
        if first_row.ndim != 1:
            raise ValueError("第一行必须是一维数组")
        if last_col is not None and last_col.ndim != 1:
            raise ValueError("最后一列必须是一维数组")
        return hankel(first_row, last_col)
    
    @staticmethod
    def vandermonde(x: np.ndarray, n: int = None) -> np.ndarray:
        """范德蒙德矩阵
        
        Args:
            x: 向量
            n: 矩阵的列数（如果为None，则使用len(x)）
            
        Returns:
            范德蒙德矩阵
        """
        if x.ndim != 1:
            raise ValueError("输入必须是一维向量")
        if n is None:
            n = len(x)
        if n < 1:
            raise ValueError("列数必须大于0")
        return np.vander(x, N=n, increasing=True)
    
    @staticmethod
    def hilbert(n: int) -> np.ndarray:
        """希尔伯特矩阵
        
        Args:
            n: 维度
            
        Returns:
            希尔伯特矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        return hilbert(n)
    
    @staticmethod
    def invhilbert(n: int) -> np.ndarray:
        """希尔伯特矩阵的逆
        
        Args:
            n: 维度
            
        Returns:
            希尔伯特矩阵的逆
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        return invhilbert(n)
    
    @staticmethod
    def pascal(n: int, kind: str = 'symmetric') -> np.ndarray:
        """帕斯卡矩阵
        
        Args:
            n: 维度
            kind: 类型 ('symmetric' 或 'lower')
            
        Returns:
            帕斯卡矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        if kind not in ['symmetric', 'lower']:
            raise ValueError("类型必须是 'symmetric' 或 'lower'")
        return pascal(n, kind=kind)
    
    @staticmethod
    def companion(polynomial_coefficients: np.ndarray) -> np.ndarray:
        """伴随矩阵
        
        Args:
            polynomial_coefficients: 多项式系数 [a_n, a_{n-1}, ..., a_1, a_0]
            
        Returns:
            伴随矩阵
        """
        if polynomial_coefficients.ndim != 1:
            raise ValueError("多项式系数必须是一维数组")
        if len(polynomial_coefficients) < 2:
            raise ValueError("多项式系数长度必须至少为2")
        return companion(polynomial_coefficients)
    
    @staticmethod
    def dft(n: int) -> np.ndarray:
        """离散傅里叶变换矩阵
        
        Args:
            n: 维度
            
        Returns:
            DFT矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        # 使用numpy的向量化操作提高性能
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        omega = np.exp(-2j * np.pi / n)
        matrix = omega ** (i * j)
        return matrix / np.sqrt(n)
    
    @staticmethod
    def hadamard(n: int) -> np.ndarray:
        """哈达玛矩阵
        
        Args:
            n: 维度（必须是2的幂）
            
        Returns:
            哈达玛矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        if not (n & (n - 1) == 0):
            raise ValueError("维度必须是2的幂")
        return hadamard(n)
    
    @staticmethod
    def rotation_2d(theta: float) -> np.ndarray:
        """2D旋转矩阵
        
        Args:
            theta: 旋转角度（弧度）
            
        Returns:
            2D旋转矩阵
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    @staticmethod
    def rotation_3d(axis: str, theta: float) -> np.ndarray:
        """3D旋转矩阵
        
        Args:
            axis: 旋转轴 ('x', 'y', 或 'z')
            theta: 旋转角度（弧度）
            
        Returns:
            3D旋转矩阵
        """
        c, s = np.cos(theta), np.sin(theta)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("轴必须是 'x', 'y', 或 'z'")
    
    @staticmethod
    def reflection_2d(theta: float) -> np.ndarray:
        """2D反射矩阵
        
        Args:
            theta: 反射轴与x轴的夹角（弧度）
            
        Returns:
            2D反射矩阵
        """
        c, s = np.cos(2 * theta), np.sin(2 * theta)
        return np.array([[c, s], [s, -c]])
    
    @staticmethod
    def shear_2d(axis: str, factor: float) -> np.ndarray:
        """2D剪切矩阵
        
        Args:
            axis: 剪切轴 ('x' 或 'y')
            factor: 剪切因子
            
        Returns:
            2D剪切矩阵
        """
        if axis == 'x':
            return np.array([[1, factor], [0, 1]])
        elif axis == 'y':
            return np.array([[1, 0], [factor, 1]])
        else:
            raise ValueError("轴必须是 'x' 或 'y'")
    
    @staticmethod
    def scaling_2d(sx: float, sy: float) -> np.ndarray:
        """2D缩放矩阵
        
        Args:
            sx: x方向缩放因子
            sy: y方向缩放因子
            
        Returns:
            2D缩放矩阵
        """
        return np.diag([sx, sy])
    
    @staticmethod
    def scaling_3d(sx: float, sy: float, sz: float) -> np.ndarray:
        """3D缩放矩阵
        
        Args:
            sx: x方向缩放因子
            sy: y方向缩放因子
            sz: z方向缩放因子
            
        Returns:
            3D缩放矩阵
        """
        return np.diag([sx, sy, sz])
    
    @staticmethod
    def permutation(n: int, perm: List[int]) -> np.ndarray:
        """置换矩阵
        
        Args:
            n: 维度
            perm: 置换列表
            
        Returns:
            置换矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        if len(perm) != n:
            raise ValueError("置换列表长度必须等于维度")
        if set(perm) != set(range(n)):
            raise ValueError("置换列表必须包含0到n-1的所有整数")
        matrix = np.zeros((n, n))
        for i, j in enumerate(perm):
            matrix[i, j] = 1
        return matrix
    
    @staticmethod
    def block_diagonal(blocks: List[np.ndarray]) -> np.ndarray:
        """块对角矩阵
        
        Args:
            blocks: 矩阵块列表
            
        Returns:
            块对角矩阵
        """
        if not blocks:
            raise ValueError("矩阵块列表不能为空")
        for block in blocks:
            if block.ndim != 2:
                raise ValueError("所有矩阵块必须是二维的")
            if block.shape[0] != block.shape[1]:
                raise ValueError("所有矩阵块必须是方阵")
        return block_diag(*blocks)
    
    @staticmethod
    def tridiagonal(diagonal: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
        """三对角矩阵
        
        Args:
            diagonal: 对角线元素
            upper: 上对角线元素
            lower: 下对角线元素
            
        Returns:
            三对角矩阵
        """
        if diagonal.ndim != 1:
            raise ValueError("对角线元素必须是一维数组")
        if upper.ndim != 1:
            raise ValueError("上对角线元素必须是一维数组")
        if lower.ndim != 1:
            raise ValueError("下对角线元素必须是一维数组")
        n = len(diagonal)
        if len(upper) != n - 1:
            raise ValueError("上对角线元素长度必须为n-1")
        if len(lower) != n - 1:
            raise ValueError("下对角线元素长度必须为n-1")
        matrix = np.diag(diagonal)
        matrix += np.diag(upper, k=1)
        matrix += np.diag(lower, k=-1)
        return matrix
    
    @staticmethod
    def symmetric(rows: int, cols: int) -> np.ndarray:
        """随机对称矩阵
        
        Args:
            rows: 行数
            cols: 列数
            
        Returns:
            对称矩阵
        """
        if rows != cols:
            raise ValueError("对称矩阵必须是方阵")
        if rows < 1:
            raise ValueError("维度必须大于0")
        A = np.random.rand(rows, cols)
        return (A + A.T) / 2
    
    @staticmethod
    def positive_definite(n: int) -> np.ndarray:
        """随机正定矩阵
        
        Args:
            n: 维度
            
        Returns:
            正定矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        A = np.random.rand(n, n)
        return A @ A.T + np.eye(n)
    
    @staticmethod
    def orthogonal(n: int) -> np.ndarray:
        """随机正交矩阵
        
        Args:
            n: 维度
            
        Returns:
            正交矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        A = np.random.rand(n, n)
        Q, _ = np.linalg.qr(A)
        return Q
    
    @staticmethod
    def unitary(n: int) -> np.ndarray:
        """随机酉矩阵
        
        Args:
            n: 维度
            
        Returns:
            酉矩阵
        """
        if n < 1:
            raise ValueError("维度必须大于0")
        A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        Q, _ = np.linalg.qr(A)
        return Q
    
    @staticmethod
    def sparse(rows: int, cols: int, density: float = 0.1) -> np.ndarray:
        """稀疏矩阵
        
        Args:
            rows: 行数
            cols: 列数
            density: 非零元素密度
            
        Returns:
            稀疏矩阵
        """
        if rows < 1 or cols < 1:
            raise ValueError("行数和列数必须大于0")
        if density < 0 or density > 1:
            raise ValueError("密度必须在0到1之间")
        mask = np.random.rand(rows, cols) < density
        return np.random.rand(rows, cols) * mask
    
    @staticmethod
    def magic(n: int) -> np.ndarray:
        """魔方矩阵
        
        Args:
            n: 维度（必须是奇数或4的倍数）
            
        Returns:
            魔方矩阵
        """
        if n < 3:
            raise ValueError("维度必须至少为3")
        from scipy.linalg import magic
        return magic(n)
    
    @staticmethod
    def gallery(name: str, **kwargs) -> np.ndarray:
        """测试矩阵库
        
        Args:
            name: 矩阵名称
            **kwargs: 矩阵参数
            
        Returns:
            测试矩阵
        """
        from scipy.linalg import gallery
        return gallery(name, **kwargs)