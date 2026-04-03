"""abstract_phys 模块测试"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
from PySymmetry.abstract_phys.physical_objects import (
    PhysicalObject,
    PhysicalSpace,
    ElementaryParticle,
    Quark,
    Lepton,
    ScalarField,
    VectorField,
)
from PySymmetry.abstract_phys.symmetry_operations import (
    IdentityOperation,
    TranslationOperation,
    RotationOperation,
    ParityOperation,
)


class ConcretePhysicalObject(PhysicalObject):
    """PhysicalObject的具体实现用于测试"""
    
    def __init__(self, mass: float = 1.0, charge: float = 0.0, spin: float = 0.5):
        self._mass = mass
        self._charge = charge
        self._spin = spin
    
    def symmetry_properties(self) -> dict:
        return {"mass": self._mass, "charge": self._charge, "spin": self._spin}
    
    def transform(self, symmetry_operation) -> 'ConcretePhysicalObject':
        return ConcretePhysicalObject(self._mass, self._charge, self._spin)
    
    def is_invariant_under(self, symmetry_operation) -> bool:
        return True
    
    def get_mass(self):
        return self._mass
    
    def get_charge(self):
        return self._charge
    
    def get_spin(self):
        return self._spin


class ConcretePhysicalSpace(PhysicalSpace):
    """PhysicalSpace的具体实现用于测试"""
    
    def __init__(self, dim: int = 3):
        self._dim = dim
    
    def dimension(self) -> int:
        return self._dim
    
    def inner_product(self, x, y):
        return np.dot(x, y)
    
    def norm(self, x):
        return np.linalg.norm(x)


class TestPhysicalObject(unittest.TestCase):
    """PhysicalObject测试"""
    
    def test_concrete_creation(self):
        """测试创建具体PhysicalObject"""
        obj = ConcretePhysicalObject(mass=2.0, charge=1.0, spin=0.5)
        self.assertEqual(obj.get_mass(), 2.0)
        self.assertEqual(obj.get_charge(), 1.0)
        self.assertEqual(obj.get_spin(), 0.5)
    
    def test_symmetry_properties(self):
        """测试对称性质"""
        obj = ConcretePhysicalObject(mass=1.5, charge=2.0, spin=1.0)
        props = obj.symmetry_properties()
        self.assertEqual(props["mass"], 1.5)
        self.assertEqual(props["charge"], 2.0)
        self.assertEqual(props["spin"], 1.0)
    
    def test_repr(self):
        """测试__repr__"""
        obj = ConcretePhysicalObject(mass=1.0, charge=2.0, spin=0.5)
        repr_str = repr(obj)
        self.assertIn("ConcretePhysicalObject", repr_str)
        self.assertIn("mass=1.0", repr_str)


class TestPhysicalSpace(unittest.TestCase):
    """PhysicalSpace测试"""
    
    def test_concrete_creation(self):
        """测试创建具体PhysicalSpace"""
        space = ConcretePhysicalSpace(dim=3)
        self.assertEqual(space.dimension(), 3)
    
    def test_inner_product(self):
        """测试内积"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        self.assertEqual(space.inner_product(x, y), 0.0)
    
    def test_norm(self):
        """测试范数"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([3.0, 4.0, 0.0])
        self.assertEqual(space.norm(x), 5.0)
    
    def test_metric_tensor(self):
        """测试度规张量设置"""
        space = ConcretePhysicalSpace(dim=3)
        g = np.eye(3)
        space.metric_tensor = g
        np.testing.assert_array_equal(space.metric_tensor, g)
    
    def test_invalid_metric_tensor(self):
        """测试无效度规张量"""
        space = ConcretePhysicalSpace(dim=3)
        with self.assertRaises(ValueError):
            space.metric_tensor = np.ones((3, 4))  # 非方阵
        
        with self.assertRaises(ValueError):
            space.metric_tensor = np.eye(4)  # 维度不匹配
    
    def test_get_metric_matrix(self):
        """测试获取度规矩阵"""
        space = ConcretePhysicalSpace(dim=3)
        metric = space.get_metric_matrix()
        np.testing.assert_array_equal(metric, np.eye(3))
    
    def test_inner_product_with_metric(self):
        """测试带度规的内积"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 0.0])
        result = space.inner_product_with_metric(x, y)
        self.assertEqual(result, 1.0)
    
    def test_distance(self):
        """测试距离"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([3.0, 4.0, 0.0])
        dist = space.distance(x, y)
        self.assertAlmostEqual(dist, 5.0)
    
    def test_angle(self):
        """测试夹角"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        angle = space.angle(x, y)
        self.assertAlmostEqual(angle, np.pi / 2)
    
    def test_is_orthogonal(self):
        """测试正交性"""
        space = ConcretePhysicalSpace(dim=3)
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        self.assertTrue(space.is_orthogonal(x, y))
    
    def test_project_onto(self):
        """测试投影"""
        space = ConcretePhysicalSpace(dim=3)
        v = np.array([1.0, 1.0, 0.0])
        subspace_bases = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        proj = space.project_onto(v, subspace_bases)
        np.testing.assert_array_almost_equal(proj, np.array([1.0, 1.0, 0.0]))


class TestElementaryParticle(unittest.TestCase):
    """基本粒子测试"""
    
    def test_creation(self):
        """测试创建粒子"""
        p = ElementaryParticle(mass=1.0, charge=1.0, spin=0.5)
        self.assertEqual(p.get_mass(), 1.0)
        self.assertEqual(p.get_charge(), 1.0)
        self.assertEqual(p.get_spin(), 0.5)
    
    def test_invalid_mass(self):
        """测试无效质量"""
        with self.assertRaises(ValueError):
            ElementaryParticle(mass=-1.0, charge=1.0, spin=0.5)
    
    def test_invalid_spin(self):
        """测试无效自旋"""
        with self.assertRaises(TypeError):
            ElementaryParticle(mass=1.0, charge=1.0, spin="invalid")
    
    def test_position_velocity(self):
        """测试位置和速度"""
        p = ElementaryParticle(
            mass=1.0, charge=1.0, spin=0.5,
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3])
        )
        np.testing.assert_array_equal(p.position, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(p.velocity, np.array([0.1, 0.2, 0.3]))
    
    def test_four_momentum(self):
        """测试四动量"""
        p = ElementaryParticle(
            mass=1.0, charge=1.0, spin=0.5,
            momentum=np.array([0.5, 0.0, 0.0])
        )
        four_momentum = p.get_four_momentum()
        self.assertEqual(len(four_momentum), 4)
        self.assertAlmostEqual(four_momentum[0], np.sqrt(1.0 + 0.25))  # E
        np.testing.assert_array_equal(four_momentum[1:], np.array([0.5, 0.0, 0.0]))
    
    def test_kinetic_energy(self):
        """测试动能"""
        p = ElementaryParticle(
            mass=2.0, charge=1.0, spin=0.5,
            velocity=np.array([1.0, 0.0, 0.0])
        )
        self.assertEqual(p.get_kinetic_energy(), 1.0)
    
    def test_symmetry_properties(self):
        """测试对称性质"""
        p = ElementaryParticle(mass=1.0, charge=1.0, spin=0.5)
        props = p.symmetry_properties
        self.assertIn("mass", props)
        self.assertIn("charge", props)
        self.assertIn("spin", props)
    
    def test_transform(self):
        """测试变换"""
        p = ElementaryParticle(mass=1.0, charge=1.0, spin=0.5)
        identity = IdentityOperation()
        transformed = p.transform(identity)
        self.assertIsInstance(transformed, ElementaryParticle)
        self.assertEqual(p.get_mass(), transformed.get_mass())
    
    def test_invariant_under(self):
        """测试不变性"""
        p = ElementaryParticle(mass=1.0, charge=1.0, spin=0.5)
        identity = IdentityOperation()
        self.assertTrue(p.is_invariant_under(identity))
    
    def test_lorentz_boost(self):
        """测试洛伦兹变换"""
        p = ElementaryParticle(
            mass=1.0, charge=1.0, spin=0.5,
            momentum=np.array([0.5, 0.0, 0.0])
        )
        # 低速情况
        boost_v = np.array([0.1, 0.0, 0.0])
        boosted = p.get_lorentz_boost(boost_v)
        self.assertIsInstance(boosted, ElementaryParticle)


class TestQuark(unittest.TestCase):
    """夸克测试"""
    
    def test_creation(self):
        """测试创建夸克"""
        q = Quark(flavor="up", mass=2.2, charge=2/3, spin=0.5, color="red")
        self.assertEqual(q.flavor, "up")
        self.assertEqual(q.color, "red")
    
    def test_invalid_flavor(self):
        """测试无效味"""
        with self.assertRaises(ValueError):
            Quark(flavor="unknown", mass=1.0, charge=0.0, spin=0.5)
    
    def test_invalid_color(self):
        """测试无效颜色"""
        with self.assertRaises(ValueError):
            Quark(flavor="up", mass=1.0, charge=0.0, spin=0.5, color="yellow")
    
    def test_quantum_numbers(self):
        """测试量子数"""
        qn = Quark.get_quantum_numbers("up")
        self.assertEqual(qn["charge"], 2/3)
        self.assertEqual(qn["isospin"], 1/2)


class TestLepton(unittest.TestCase):
    """轻子测试"""
    
    def test_creation(self):
        """测试创建轻子"""
        l = Lepton(lepton_type="electron", mass=0.511, charge=-1.0, spin=0.5)
        self.assertEqual(l.type, "electron")
    
    def test_is_neutrino(self):
        """测试中微子判断"""
        e = Lepton(lepton_type="electron", mass=0.511, charge=-1.0, spin=0.5)
        n = Lepton(lepton_type="electron_neutrino", mass=0.0, charge=0.0, spin=0.5)
        self.assertFalse(e.is_neutrino)
        self.assertTrue(n.is_neutrino)
    
    def test_invalid_type(self):
        """测试无效类型"""
        with self.assertRaises(ValueError):
            Lepton(lepton_type="unknown", mass=1.0, charge=0.0, spin=0.5)


class TestScalarField(unittest.TestCase):
    """标量场测试"""
    
    def test_creation(self):
        """测试创建标量场"""
        def simple_field(x):
            return np.sum(x**2)
        field = ScalarField(field_function=simple_field)
        result = field.evaluate(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result, 14.0)
    
    def test_evaluate(self):
        """测试场值计算"""
        def simple_field(x):
            return x[0]
        field = ScalarField(field_function=simple_field)
        result = field.evaluate(np.array([3.0, 1.0, 2.0]))
        self.assertEqual(result, 3.0)
    
    def test_gradient(self):
        """测试梯度"""
        def simple_field(x):
            return np.sum(x**2)
        field = ScalarField(field_function=simple_field)
        grad = field.gradient(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(grad, np.array([2.0, 4.0, 6.0]))


if __name__ == "__main__":
    unittest.main()
