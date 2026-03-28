"""PySyM 核心数学模块（对称性代数基础）

为上层「物理应用」模块提供可组合的数学对象与约定；本包**不**实现具体物理模型。

**与物理对接时的推荐约定**

- **有限群乘法**（``SymmetricGroup``、``AlternatingGroup``、``DihedralGroup`` 等）：
  ``multiply(a, b)`` 表示**先作用** ``b`` **再作用** ``a``（与置换合成习惯一致）。
- **矩阵表示**：同态条件为 ``ρ(ab) = ρ(a)ρ(b)``（矩阵乘法顺序与群乘法一致）；向量默认为**列向量**，
  ``ρ(g) v`` 在数值上为 ``matrix @ v``。
- **李代数** ``u(n)`` / ``su(n)``：元素为**反厄米**矩阵，与李群 ``exp`` 映射 ``exp(X)`` 一致；
  文献中常见的**厄米生成元** ``H`` 满足 ``H = -i X``，需在物理层自行换算。
- **数值性**：矩阵与李代数相等性多用 ``allclose``；与解析公式比较时请显式给定容差。

子包：``algebraic_structures``、``group_theory``、``lie_theory``、``matrix``、
``matrix_groups``、``representation``、``utils``。
"""

from . import (
    algebraic_structures,
    group_theory,
    lie_theory,
    matrix,
    matrix_groups,
    representation,
    utils, 
)


__all__ = [
    'algebraic_structures',
    'group_theory',
    'lie_theory',
    'matrix',
    'matrix_groups',
    'representation',
    'utils',
]
