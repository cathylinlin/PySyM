
"""矩阵群优化工具模块"""
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import qr
import logging

logger = logging.getLogger(__name__)

def optimize_special_orthogonal_group(n: int) -> np.ndarray:
    """
    优化特殊正交群矩阵生成算法
    优化方案：使用QR分解，通过分析Householder反射矩阵的行列式特性，直接调整行列式为1
    
    :param n: 矩阵维度
    :return: 生成的特殊正交矩阵
    """
    # 基础优化方案：首先生成一个标准的正交群矩阵，然后通过调整第一行乘以矩阵行列式的方式，强制使行列式变为+1
    # 生成一个标准正交矩阵
    Q = ortho_group.rvs(n)
    
    # 检查行列式
    det = np.linalg.det(Q)
    
    # 如果行列式为-1，调整第一行
    if np.isclose(det, -1):
        # 调整第一行的符号
        Q[0] = -Q[0]
    
    return Q

def optimize_special_linear_group(n: int) -> np.ndarray:
    """
    优化特殊线性群矩阵生成算法
    优化方案：使用QR分解，确保行列式为1
    
    :param n: 矩阵维度
    :return: 生成的特殊线性矩阵
    """
    # 生成随机矩阵
    A = np.random.randn(n, n)
    
    # QR分解
    Q, R = qr(A)
    
    # 调整Q的行列式为1
    det = np.linalg.det(Q)
    if np.isclose(det, -1):
        Q[:, 0] = -Q[:, 0]  # 调整第一列的符号
    
    # 生成特殊线性矩阵
    return Q

def optimize_general_linear_group(n: int) -> np.ndarray:
    """
    优化一般线性群矩阵生成算法
    优化方案：生成随机可逆矩阵
    
    :param n: 矩阵维度
    :return: 生成的一般线性矩阵
    """
    # 生成随机矩阵
    A = np.random.randn(n, n)
    
    # 确保可逆
    while np.linalg.matrix_rank(A) < n:
        A = np.random.randn(n, n)
    
    return A