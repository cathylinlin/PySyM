"""代数结构测试模块"""
import unittest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySymmetry.core.algebraic_structures import (
    IntegerRing, IntegerRingElement,
    RationalField, RationalFieldElement,
    RealField, RealFieldElement,
    FiniteField, FiniteFieldElement,
    ComplexField, ComplexFieldElement,
    PolynomialRing, PolynomialRingElement,
    MatrixRing, MatrixRingElement,
    FiniteDimensionalVectorSpace, FiniteDimensionalVectorSpaceElement,
    LinearTransformation,
    direct_sum_groups
)


class TestIntegerRing(unittest.TestCase):
    """整数环测试"""
    
    def setUp(self):
        self.ring = IntegerRing()
    
    def test_addition(self):
        """测试加法"""
        a = IntegerRingElement(1)
        b = IntegerRingElement(2)
        result = self.ring.add(a, b)
        self.assertEqual(result.value, 3)
    
    def test_multiplication(self):
        """测试乘法"""
        a = IntegerRingElement(2)
        b = IntegerRingElement(3)
        result = self.ring.multiply(a, b)
        self.assertEqual(result.value, 6)
    
    def test_inverse(self):
        """测试加法逆元"""
        a = IntegerRingElement(5)
        inverse = self.ring.inverse(a)
        self.assertEqual(inverse.value, -5)
    
    def test_is_integral_domain(self):
        """测试是否为整环"""
        self.assertTrue(self.ring.is_integral_domain())


class TestRationalField(unittest.TestCase):
    """有理数域测试"""
    
    def setUp(self):
        self.field = RationalField()
    
    def test_addition(self):
        """测试加法"""
        a = RationalFieldElement(1, 2)
        b = RationalFieldElement(1, 3)
        result = self.field.add(a, b)
        self.assertEqual(result.numerator, 5)
        self.assertEqual(result.denominator, 6)
    
    def test_multiplication(self):
        """测试乘法"""
        a = RationalFieldElement(2, 3)
        b = RationalFieldElement(3, 4)
        result = self.field.multiply(a, b)
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 2)
    
    def test_division(self):
        """测试除法"""
        a = RationalFieldElement(1, 2)
        b = RationalFieldElement(3, 4)
        result = self.field.divide(a, b)
        self.assertEqual(result.numerator, 2)
        self.assertEqual(result.denominator, 3)
    
    def test_multiplicative_inverse(self):
        """测试乘法逆元"""
        a = RationalFieldElement(2, 3)
        inverse = self.field.multiplicative_inverse(a)
        self.assertEqual(inverse.numerator, 3)
        self.assertEqual(inverse.denominator, 2)


class TestFiniteField(unittest.TestCase):
    """有限域测试"""
    
    def setUp(self):
        self.field = FiniteField(5)  # GF(5)
    
    def test_addition(self):
        """测试加法"""
        a = FiniteFieldElement(3, 5)
        b = FiniteFieldElement(4, 5)
        result = self.field.add(a, b)
        self.assertEqual(result.value, 2)  # 3 + 4 = 7 mod 5 = 2
    
    def test_multiplication(self):
        """测试乘法"""
        a = FiniteFieldElement(2, 5)
        b = FiniteFieldElement(3, 5)
        result = self.field.multiply(a, b)
        self.assertEqual(result.value, 1)  # 2 * 3 = 6 mod 5 = 1
    
    def test_division(self):
        """测试除法"""
        a = FiniteFieldElement(1, 5)
        b = FiniteFieldElement(2, 5)
        result = self.field.divide(a, b)
        self.assertEqual(result.value, 3)  # 1 / 2 = 3 mod 5
    
    def test_multiplicative_inverse(self):
        """测试乘法逆元"""
        a = FiniteFieldElement(2, 5)
        inverse = self.field.multiplicative_inverse(a)
        self.assertEqual(inverse.value, 3)  # 2 * 3 = 1 mod 5


class TestPolynomialRing(unittest.TestCase):
    """多项式环测试"""
    
    def setUp(self):
        self.ring = PolynomialRing()
    
    def test_addition(self):
        """测试加法"""
        p1 = PolynomialRingElement([1, 2])  # 1 + 2x
        p2 = PolynomialRingElement([3, 4])  # 3 + 4x
        result = self.ring.add(p1, p2)
        self.assertEqual(result.coefficients, [4, 6])  # 4 + 6x
    
    def test_multiplication(self):
        """测试乘法"""
        p1 = PolynomialRingElement([1, 1])  # 1 + x
        p2 = PolynomialRingElement([1, -1])  # 1 - x
        result = self.ring.multiply(p1, p2)
        self.assertEqual(result.coefficients, [1, 0, -1])  # 1 - x^2


class TestMatrixRing(unittest.TestCase):
    """矩阵环测试"""
    
    def setUp(self):
        self.int_ring = IntegerRing()
        self.matrix_ring = MatrixRing(self.int_ring, 2)
    
    def test_addition(self):
        """测试加法"""
        m1 = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(2)],
             [IntegerRingElement(3), IntegerRingElement(4)]],
            self.int_ring, 2, 2
        )
        m2 = MatrixRingElement(
            [[IntegerRingElement(5), IntegerRingElement(6)],
             [IntegerRingElement(7), IntegerRingElement(8)]],
            self.int_ring, 2, 2
        )
        result = self.matrix_ring.add(m1, m2)
        self.assertEqual(result.entries[0][0].value, 6)
        self.assertEqual(result.entries[0][1].value, 8)
        self.assertEqual(result.entries[1][0].value, 10)
        self.assertEqual(result.entries[1][1].value, 12)
    
    def test_multiplication(self):
        """测试乘法"""
        m1 = MatrixRingElement(
            [[IntegerRingElement(1), IntegerRingElement(2)],
             [IntegerRingElement(3), IntegerRingElement(4)]],
            self.int_ring, 2, 2
        )
        m2 = MatrixRingElement(
            [[IntegerRingElement(5), IntegerRingElement(6)],
             [IntegerRingElement(7), IntegerRingElement(8)]],
            self.int_ring, 2, 2
        )
        result = self.matrix_ring.multiply(m1, m2)
        self.assertEqual(result.entries[0][0].value, 19)  # 1*5 + 2*7 = 19
        self.assertEqual(result.entries[0][1].value, 22)  # 1*6 + 2*8 = 22
        self.assertEqual(result.entries[1][0].value, 43)  # 3*5 + 4*7 = 43
        self.assertEqual(result.entries[1][1].value, 50)  # 3*6 + 4*8 = 50


class TestVectorSpace(unittest.TestCase):
    """向量空间测试"""
    
    def setUp(self):
        self.field = RationalField()
        self.vector_space = FiniteDimensionalVectorSpace(self.field, 2)
    
    def test_addition(self):
        """测试加法"""
        v1 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 2), RationalFieldElement(1, 3)],
            self.field
        )
        v2 = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 4), RationalFieldElement(1, 5)],
            self.field
        )
        result = self.vector_space.add(v1, v2)
        self.assertEqual(result.components[0].numerator, 3)
        self.assertEqual(result.components[0].denominator, 4)
        self.assertEqual(result.components[1].numerator, 8)
        self.assertEqual(result.components[1].denominator, 15)
    
    def test_scalar_multiplication(self):
        """测试标量乘法"""
        v = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 2), RationalFieldElement(1, 3)],
            self.field
        )
        scalar = RationalFieldElement(2, 1)
        result = self.vector_space.scalar_multiply(v, scalar)
        self.assertEqual(result.components[0].numerator, 1)
        self.assertEqual(result.components[0].denominator, 1)
        self.assertEqual(result.components[1].numerator, 2)
        self.assertEqual(result.components[1].denominator, 3)


class TestLinearTransformation(unittest.TestCase):
    """线性变换测试"""
    
    def setUp(self):
        self.field = RationalField()
        self.domain = FiniteDimensionalVectorSpace(self.field, 2)
        self.codomain = FiniteDimensionalVectorSpace(self.field, 2)
        # 定义一个线性变换矩阵: [[1, 2], [3, 4]]
        self.matrix = [
            [RationalFieldElement(1, 1), RationalFieldElement(2, 1)],
            [RationalFieldElement(3, 1), RationalFieldElement(4, 1)]
        ]
        self.transformation = LinearTransformation(self.domain, self.codomain, self.matrix)
    
    def test_application(self):
        """测试线性变换的应用"""
        v = FiniteDimensionalVectorSpaceElement(
            [RationalFieldElement(1, 1), RationalFieldElement(1, 1)],
            self.field
        )
        result = self.transformation(v)
        self.assertEqual(result.components[0].numerator, 3)  # 1*1 + 2*1 = 3
        self.assertEqual(result.components[1].numerator, 7)  # 3*1 + 4*1 = 7


class TestDirectSumGroups(unittest.TestCase):
    """群的直和测试"""
    
    def setUp(self):
        self.field1 = FiniteField(2)
        self.field2 = FiniteField(3)
        self.direct_sum = direct_sum_groups([self.field1, self.field2])
    
    def test_multiplication(self):
        """测试直和群的乘法"""
        e1 = self.direct_sum.identity()
        e2 = self.direct_sum.identity()
        result = self.direct_sum.multiply(e1, e2)
        self.assertTrue(result.is_identity())


if __name__ == '__main__':
    unittest.main()
