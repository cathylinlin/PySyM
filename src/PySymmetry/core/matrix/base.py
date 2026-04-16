from abc import ABC, abstractmethod
from typing import TypeVar, Union

import numpy as np

T = TypeVar("T", bound="AbstractMatrix")


class AbstractMatrix(ABC):
    """抽象矩阵基类"""

    def __init__(self, data: np.ndarray):
        self._data = np.array(
            data, dtype=np.complex128 if np.iscomplexobj(data) else np.float64
        )
        self._validate_shape()

    @abstractmethod
    def _validate_shape(self) -> None:
        """验证矩阵形状"""
        pass

    @property
    def shape(self) -> tuple:
        """矩阵形状"""
        return self._data.shape

    @property
    def data(self) -> np.ndarray:
        """底层数据"""
        return self._data

    @property
    def ndim(self) -> int:
        """矩阵维度"""
        return self._data.ndim

    @property
    def size(self) -> int:
        """矩阵元素总数"""
        return self._data.size

    @property
    def dtype(self) -> np.dtype:
        """矩阵数据类型"""
        return self._data.dtype

    def is_square(self) -> bool:
        """判断是否为方阵"""
        return self._data.shape[0] == self._data.shape[1]

    def __add__(
        self, other: Union["AbstractMatrix", np.ndarray, float, int]
    ) -> "AbstractMatrix":
        """矩阵加法"""
        if isinstance(other, AbstractMatrix):
            other = other.data
        result = self._data + other
        return self.__class__(result)

    def __radd__(self, other: np.ndarray | float | int) -> "AbstractMatrix":
        """右加法"""
        return self.__add__(other)

    def __sub__(
        self, other: Union["AbstractMatrix", np.ndarray, float, int]
    ) -> "AbstractMatrix":
        """矩阵减法"""
        if isinstance(other, AbstractMatrix):
            other = other.data
        result = self._data - other
        return self.__class__(result)

    def __rsub__(self, other: np.ndarray | float | int) -> "AbstractMatrix":
        """右减法"""
        if isinstance(other, AbstractMatrix):
            other = other.data
        result = other - self._data
        return self.__class__(result)

    def __mul__(
        self, other: Union["AbstractMatrix", np.ndarray, float, int]
    ) -> "AbstractMatrix":
        """标量乘法或逐元素乘法"""
        if isinstance(other, AbstractMatrix):
            other = other.data
        result = self._data * other
        return self.__class__(result)

    def __rmul__(self, other: np.ndarray | float | int) -> "AbstractMatrix":
        """右乘法"""
        return self.__mul__(other)

    def __matmul__(
        self, other: Union["AbstractMatrix", np.ndarray]
    ) -> "AbstractMatrix":
        """矩阵乘法"""
        if isinstance(other, AbstractMatrix):
            other = other.data
        result = self._data @ other
        return self.__class__(result)

    def __truediv__(self, other: np.ndarray | float | int) -> "AbstractMatrix":
        """标量除法"""
        result = self._data / other
        return self.__class__(result)

    def __neg__(self) -> "AbstractMatrix":
        """取负"""
        result = -self._data
        return self.__class__(result)

    def __pos__(self) -> "AbstractMatrix":
        """正号"""
        return self

    def __abs__(self) -> "AbstractMatrix":
        """绝对值"""
        result = np.abs(self._data)
        return self.__class__(result)

    def __eq__(self, other: object) -> bool:
        """相等比较"""
        if isinstance(other, AbstractMatrix):
            return np.allclose(self._data, other.data)
        return False

    def __ne__(self, other: object) -> bool:
        """不等比较"""
        return not self.__eq__(other)

    def __getitem__(self, key) -> np.ndarray | float | complex:
        """索引访问"""
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        """索引设置"""
        self._data[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __str__(self) -> str:
        return str(self._data)

    def copy(self) -> "AbstractMatrix":
        """复制矩阵"""
        return self.__class__(self._data.copy())

    def transpose(self) -> "AbstractMatrix":
        """转置"""
        result = self._data.T
        return self.__class__(result)

    def conjugate(self) -> "AbstractMatrix":
        """共轭"""
        result = self._data.conj()
        return self.__class__(result)

    def conjugate_transpose(self) -> "AbstractMatrix":
        """共轭转置"""
        result = self._data.conj().T
        return self.__class__(result)

    def to_numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return self._data.copy()

    def astype(self, dtype: np.dtype) -> "AbstractMatrix":
        """转换数据类型"""
        result = self._data.astype(dtype)
        return self.__class__(result)
