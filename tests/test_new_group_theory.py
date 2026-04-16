"""测试新创建的群论模块"""

import pytest
import numpy as np
from PySymmetry.core.group_theory.continuous_groups import (
    TranslationGroup,
    RotationGroup,
    TimeTranslationGroup,
)
from PySymmetry.core.group_theory.discrete_groups import ParityGroup, TimeReversalGroup
from PySymmetry.abstract_phys.symmetry_operations.generators import (
    MomentumGenerator,
    AngularMomentumGenerator,
    HamiltonianGenerator,
    ParityOperator,
    TimeReversalOperator,
)
from PySymmetry.abstract_phys.symmetry_environments.gauge_groups import (
    GaugeGroupFactory,
)


class TestContinuousGroups:
    """测试连续群"""

    def test_translation_group(self):
        """测试平移群"""
        group = TranslationGroup(3)
        assert group.name == "TranslationGroup(3)"
        assert group.dimension == 3

        # 测试单位元
        identity = group.identity()
        assert np.array_equal(identity, np.zeros(3))

        # 测试逆元
        element = np.array([1, 2, 3])
        inverse = group.inverse(element)
        assert np.array_equal(inverse, np.array([-1, -2, -3]))

        # 测试组合
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        composed = group.multiply(a, b)
        assert np.array_equal(composed, np.array([1, 1, 0]))

        # 测试是否为连续群
        assert group.is_continuous() is True

    def test_rotation_group(self):
        """测试旋转群"""
        group = RotationGroup(3)
        assert group.name == "RotationGroup(3)"
        assert group.dimension == 3

        # 测试单位元
        identity = group.identity()
        assert np.array_equal(identity, np.eye(3))

        # 测试逆元
        element = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90度旋转矩阵
        inverse = group.inverse(element)
        assert np.array_equal(inverse, element.T)

        # 测试组合
        a = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90度旋转
        b = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90度旋转
        composed = group.multiply(a, b)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # 180度旋转
        assert np.allclose(composed, expected)

        # 测试是否为连续群
        assert group.is_continuous() is True

    def test_time_translation_group(self):
        """测试时间平移群"""
        group = TimeTranslationGroup()
        assert group.name == "TimeTranslationGroup"

        # 测试单位元
        identity = group.identity()
        assert identity == 0.0

        # 测试逆元
        element = 5.0
        inverse = group.inverse(element)
        assert inverse == -5.0

        # 测试组合
        a = 2.0
        b = 3.0
        composed = group.multiply(a, b)
        assert composed == 5.0

        # 测试是否为连续群
        assert group.is_continuous() is True


class TestDiscreteGroups:
    """测试离散群"""

    def test_parity_group(self):
        """测试宇称群"""
        group = ParityGroup()
        assert group.name == "ParityGroup"

        # 测试单位元
        identity = group.identity()
        assert identity == 1

        # 测试逆元
        element = -1
        inverse = group.inverse(element)
        assert inverse == -1  # 宇称操作的逆是自身

        # 测试组合
        a = -1
        b = -1
        composed = group.multiply(a, b)
        assert composed == 1  # 两个宇称操作的组合是恒等操作

        # 测试是否为连续群
        assert group.is_continuous() is False

    def test_time_reversal_group(self):
        """测试时间反演群"""
        group = TimeReversalGroup()
        assert group.name == "TimeReversalGroup"

        # 测试单位元
        identity = group.identity()
        assert identity == 1

        # 测试逆元
        element = -1
        inverse = group.inverse(element)
        assert inverse == -1  # 时间反演操作的逆是自身

        # 测试组合
        a = -1
        b = -1
        composed = group.multiply(a, b)
        assert composed == 1  # 两个时间反演操作的组合是恒等操作

        # 测试是否为连续群
        assert group.is_continuous() is False


class TestGenerators:
    """测试生成元"""

    def test_momentum_generator(self):
        """测试动量生成元"""
        gen = MomentumGenerator(0)
        assert gen.index == 0
        assert str(gen) == "MomentumGenerator(0)"

    def test_angular_momentum_generator(self):
        """测试角动量生成元"""
        gen = AngularMomentumGenerator(0, 1)
        assert gen.i == 0
        assert gen.j == 1
        assert str(gen) == "AngularMomentumGenerator(0, 1)"

    def test_hamiltonian_generator(self):
        """测试哈密顿量生成元"""
        gen = HamiltonianGenerator()
        assert str(gen) == "HamiltonianGenerator()"

    def test_parity_operator(self):
        """测试宇称算符"""
        gen = ParityOperator()
        assert str(gen) == "ParityOperator()"

    def test_time_reversal_operator(self):
        """测试时间反演算符"""
        gen = TimeReversalOperator()
        assert str(gen) == "TimeReversalOperator()"


class TestGaugeGroups:
    """测试规范群"""

    def test_gauge_group_factory(self):
        """测试规范群工厂"""
        # 测试创建U(1)规范群
        u1_group = GaugeGroupFactory.create("U(1)")
        assert u1_group.name == "U(1)"
        assert len(u1_group.generators()) == 1

        # 测试创建SU(2)规范群
        su2_group = GaugeGroupFactory.create("SU(2)")
        assert su2_group.name == "SU(2)"
        assert len(su2_group.generators()) == 3

        # 测试创建SU(3)规范群
        su3_group = GaugeGroupFactory.create("SU(3)")
        assert su3_group.name == "SU(3)"

        # 测试错误的规范群类型
        with pytest.raises(ValueError):
            GaugeGroupFactory.create("SU(4)")
