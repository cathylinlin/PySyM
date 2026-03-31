#!/usr/bin/env python3
"""
矩阵工具模块
"""
import numpy as np

def is_invertible(matrix: np.ndarray) -> bool:
    """
    检查矩阵是否可逆
    :param matrix: 输入矩阵
    :return: 是否可逆
    """
    try:
        # 检查矩阵是否为方阵
        if matrix.shape[0] != matrix.shape[1]:
            return False
        # 计算矩阵的秩
        rank = np.linalg.matrix_rank(matrix)
        # 如果秩等于矩阵维度，则可逆
        return rank == matrix.shape[0]
    except Exception:
        return False
