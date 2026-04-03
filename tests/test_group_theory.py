"""群论模块测试"""
import unittest
import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PySymmetry.core.group_theory import (
    GroupFactory, Subgroup, Coset, CosetSpace, QuotientGroup
)

class TestGroupTheory(unittest.TestCase):
    """群论模块测试"""
    
    def test_cyclic_group(self):
        """测试循环群"""
        G = GroupFactory.cyclic_group(5)
        self.assertEqual(G.order(), 5)
        self.assertTrue(G.is_abelian())
        
        # 测试群运算
        a = 2
        b = 3
        self.assertEqual(G.multiply(a, b), (2 + 3) % 5)
        self.assertEqual(G.inverse(a), (-a) % 5)
    
    def test_symmetric_group(self):
        """测试对称群"""
        G = GroupFactory.symmetric_group(3)
        self.assertEqual(G.order(), 6)
        self.assertFalse(G.is_abelian())
        
        # 测试群运算
        a = (1, 0, 2)  # 交换0和1
        b = (0, 2, 1)  # 交换1和2
        expected = (1, 2, 0)  # 先应用b，再应用a
        self.assertEqual(G.multiply(a, b), expected)
    
    def test_dihedral_group(self):
        """测试二面体群"""
        G = GroupFactory.dihedral_group(3)
        self.assertEqual(G.order(), 6)
        
        # 测试群运算
        identity = (0, 0)  # 单位元
        rotation = (0, 1)  # 旋转1步
        reflection = (1, 0)  # 翻转
        
        self.assertEqual(G.multiply(rotation, rotation), (0, 2))
        self.assertEqual(G.multiply(reflection, rotation), (1, -1 % 3))
    
    def test_subgroup(self):
        """测试子群"""
        G = GroupFactory.cyclic_group(6)
        generators = [2]
        H = G.generate_subgroup(generators)
        self.assertEqual(H.order(), 3)
        self.assertTrue(2 in H)
        self.assertTrue(4 in H)
        self.assertTrue(0 in H)
    
    def test_coset(self):
        """测试陪集"""
        G = GroupFactory.cyclic_group(6)
        generators = [2]
        H = G.generate_subgroup(generators)
        
        # 测试左陪集
        coset = Coset(G, H, 1, is_left=True)
        self.assertTrue(1 in coset)
        self.assertTrue(3 in coset)
        self.assertTrue(5 in coset)
        
        # 测试右陪集
        coset_right = Coset(G, H, 1, is_left=False)
        self.assertTrue(1 in coset_right)
        self.assertTrue(3 in coset_right)
        self.assertTrue(5 in coset_right)
    
    def test_quotient_group(self):
        """测试商群"""
        G = GroupFactory.cyclic_group(6)
        generators = [3]
        H = G.generate_subgroup(generators)
        
        # 检查H是否为正规子群
        self.assertTrue(H.is_normal())
        
        # 创建商群
        Q = QuotientGroup(G, H)
        self.assertEqual(Q.order(), 3)
    
    def test_quaternion_group(self):
        """测试四元群"""
        G = GroupFactory.quaternion_group()
        self.assertEqual(G.order(), 8)
        
        # 测试群运算
        i = 'i'
        j = 'j'
        k = 'k'
        self.assertEqual(G.multiply(i, j), k)
        self.assertEqual(G.multiply(j, i), '-k')
    
    def test_klein_group(self):
        """测试Klein群"""
        G = GroupFactory.klein_four_group()
        self.assertEqual(G.order(), 4)
        self.assertTrue(G.is_abelian())
        
        # 测试群运算
        a = (1, 0)
        b = (0, 1)
        self.assertEqual(G.multiply(a, b), (1, 1))
    
    def test_alternating_group(self):
        """测试交错群"""
        G = GroupFactory.alternating_group(3)
        self.assertEqual(G.order(), 3)
        self.assertTrue(G.is_abelian())

if __name__ == '__main__':
    unittest.main()
