"""李代数模块测试

该文件包含李代数模块的测试用例。
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from PySymmetry.core.lie_theory.lie_algebra_factory import LieAlgebraFactory
from PySymmetry.core.lie_theory.lie_algebra_representation import (
    AdjointRepresentation,
    FundamentalRepresentation,
)
from PySymmetry.core.lie_theory.lie_algebra_structure import KillingForm
from PySymmetry.core.lie_theory.lie_algebra_operations import StandardLieBracket


class TestLieAlgebra(unittest.TestCase):
    """李代数测试类"""

    def test_general_linear_lie_algebra(self):
        """测试一般线性李代数 gl(2)"""
        gl2 = LieAlgebraFactory.create_general_linear(2)
        self.assertEqual(gl2.dimension, 4)
        self.assertEqual(gl2.n, 2)

        # 测试零元素
        zero = gl2.zero()
        self.assertIsNotNone(zero)

        # 测试基
        basis = gl2.basis()
        self.assertEqual(len(basis), 4)

        # 测试李括号
        x = gl2.from_vector([1, 0, 0, 0])  # 单位矩阵的 (0,0) 元素
        y = gl2.from_vector([0, 1, 0, 0])  # 单位矩阵的 (0,1) 元素
        bracket = gl2.bracket(x, y)
        self.assertIsNotNone(bracket)

    def test_special_linear_lie_algebra(self):
        """测试特殊线性李代数 sl(2)"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        self.assertEqual(sl2.dimension, 3)
        self.assertEqual(sl2.n, 2)

        # 测试零元素
        zero = sl2.zero()
        self.assertIsNotNone(zero)

        # 测试基
        basis = sl2.basis()
        self.assertEqual(len(basis), 3)

        # 测试李括号
        x = sl2.from_vector([1, 0, 0])  # 非对角元素
        y = sl2.from_vector([0, 1, 0])  # 非对角元素
        bracket = sl2.bracket(x, y)
        self.assertIsNotNone(bracket)

    def test_orthogonal_lie_algebra(self):
        """测试正交李代数 so(3)"""
        so3 = LieAlgebraFactory.create_orthogonal(3)
        self.assertEqual(so3.dimension, 3)
        self.assertEqual(so3.n, 3)

        # 测试零元素
        zero = so3.zero()
        self.assertIsNotNone(zero)

        # 测试基
        basis = so3.basis()
        self.assertEqual(len(basis), 3)

    def test_symplectic_lie_algebra(self):
        """测试辛李代数 sp(2)（n=1 时矩阵为 2×2）"""
        sp1 = LieAlgebraFactory.create_symplectic(1)
        self.assertEqual(sp1.dimension, 3)
        self.assertEqual(sp1.n, 1)

        # 测试零元素
        zero = sp1.zero()
        self.assertIsNotNone(zero)

    def test_unitary_lie_algebra(self):
        """测试酉李代数 u(2)"""
        u2 = LieAlgebraFactory.create_unitary(2)
        self.assertEqual(u2.dimension, 4)
        self.assertEqual(u2.n, 2)

        # 测试零元素
        zero = u2.zero()
        self.assertIsNotNone(zero)

    def test_special_unitary_lie_algebra(self):
        """测试特殊酉李代数 su(2)"""
        su2 = LieAlgebraFactory.create_special_unitary(2)
        self.assertEqual(su2.dimension, 3)
        self.assertEqual(su2.n, 2)

        # 测试零元素
        zero = su2.zero()
        self.assertIsNotNone(zero)

        props = su2.properties()
        self.assertFalse(props.is_abelian)
        self.assertTrue(props.is_semisimple)

    def test_adjoint_representation(self):
        """测试伴随表示"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        adjoint = AdjointRepresentation(sl2)
        self.assertEqual(adjoint.dimension, 3)

        # 测试表示
        x = sl2.from_vector([1, 0, 0])
        matrix = adjoint(x)
        self.assertIsNotNone(matrix)

    def test_fundamental_representation(self):
        """测试基本表示"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        fundamental = FundamentalRepresentation(sl2, 2)
        self.assertEqual(fundamental.dimension, 2)

        # 测试表示
        x = sl2.from_vector([1, 0, 0])
        matrix = fundamental(x)
        self.assertIsNotNone(matrix)

    def test_killing_form(self):
        """测试基灵型"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        killing_form = KillingForm(sl2)

        # 测试基灵型矩阵
        self.assertIsNotNone(killing_form.matrix)
        self.assertEqual(killing_form.matrix.shape, (3, 3))

        # 测试非退化性
        self.assertTrue(killing_form.is_non_degenerate())

    def test_lie_bracket(self):
        """测试李括号"""
        sl2 = LieAlgebraFactory.create_special_linear(2)
        lie_bracket = StandardLieBracket(sl2)

        # 测试反对称性
        self.assertTrue(lie_bracket.is_anticommutative())

        # 测试雅可比恒等式
        x = sl2.from_vector([1, 0, 0])
        y = sl2.from_vector([0, 1, 0])
        z = sl2.from_vector([0, 0, 1])
        self.assertTrue(lie_bracket.satisfies_jacobi_identity(x, y, z))

    def test_lie_algebra_factory(self):
        """测试李代数工厂"""
        # 测试创建一般线性李代数
        gl2 = LieAlgebraFactory.create_general_linear(2)
        self.assertIsNotNone(gl2)

        # 测试创建特殊线性李代数
        sl2 = LieAlgebraFactory.create_special_linear(2)
        self.assertIsNotNone(sl2)

        # 测试根据名称创建李代数
        so3 = LieAlgebraFactory.create_lie_algebra("so", 3)
        self.assertIsNotNone(so3)

        # 测试获取李代数类
        gl_class = LieAlgebraFactory.get_lie_algebra_class("gl")
        self.assertIsNotNone(gl_class)


if __name__ == "__main__":
    unittest.main()
