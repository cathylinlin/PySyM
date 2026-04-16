"""氢原子SO(4)对称性测试

测试氢原子的隐藏SO(4)对称性功能：
1. Runge-Lenz向量算符
2. SO(4)代数结构
3. 能级简并性验证
4. 对称性分析
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from PySymmetry.phys.quantum.hydrogen_symmetry import (
    HydrogenSO4Symmetry,
    HydrogenSymmetryAnalyzer,
    SO4Generators,
    RungeLenzVector,
    AngularMomentumOperator,
    LeviCivita,
    QuantumNumbers,
    SO4QuantumNumbers,
    analyze_hydrogen_symmetry,
    create_hydrogen_so4_analyzer,
)

from PySymmetry.phys.quantum.hamiltonian import HydrogenAtomHamiltonian
from PySymmetry.phys.quantum.analysis import (
    SO4HydrogenAnalyzer,
    detect_hydrogen_so4_symmetry,
)


class TestLeviCivita(unittest.TestCase):
    """Levi-Civita符号测试"""

    def test_epsilon_cyclic(self):
        """测试循环排列"""
        self.assertEqual(LeviCivita.epsilon(0, 1, 2), 1.0)
        self.assertEqual(LeviCivita.epsilon(1, 2, 0), 1.0)
        self.assertEqual(LeviCivita.epsilon(2, 0, 1), 1.0)

    def test_epsilon_anticyclic(self):
        """测试反循环排列"""
        self.assertEqual(LeviCivita.epsilon(0, 2, 1), -1.0)
        self.assertEqual(LeviCivita.epsilon(2, 1, 0), -1.0)
        self.assertEqual(LeviCivita.epsilon(1, 0, 2), -1.0)

    def test_epsilon_identical(self):
        """测试有相同指标"""
        self.assertEqual(LeviCivita.epsilon(0, 0, 1), 0.0)
        self.assertEqual(LeviCivita.epsilon(1, 1, 1), 0.0)


class TestAngularMomentumOperator(unittest.TestCase):
    """角动量算符测试"""

    def test_l0_operators(self):
        """测试l=0角动量算符"""
        L = AngularMomentumOperator(0)
        self.assertEqual(L.dim, 1)

        Lz = L.Lz
        self.assertEqual(Lz.shape, (1, 1))
        self.assertEqual(Lz[0, 0], 0.0)

    def test_l1_operators(self):
        """测试l=1角动量算符"""
        L = AngularMomentumOperator(1)
        self.assertEqual(L.dim, 3)

        Lz = L.Lz
        expected_diag = np.array([1, 0, -1])
        np.testing.assert_array_almost_equal(np.diag(Lz), expected_diag)

        L2 = L.L2
        np.testing.assert_array_almost_equal(L2, 2 * np.eye(3))

    def test_ladder_operators(self):
        """测试升降算符"""
        L = AngularMomentumOperator(1)
        Lp, Lm = L.ladder_operators()

        self.assertEqual(Lp.shape, (3, 3))
        self.assertEqual(Lm.shape, (3, 3))

        np.testing.assert_array_almost_equal(Lp, Lm.T.conj())


class TestRungeLenzVector(unittest.TestCase):
    """Runge-Lenz向量测试"""

    def test_creation(self):
        """测试Runge-Lenz向量创建"""
        A = RungeLenzVector(n=2, l=0)
        self.assertEqual(A.n, 2)
        self.assertEqual(A.l, 0)
        self.assertEqual(A.dim, 1)

    def test_operators(self):
        """测试算符返回"""
        A = RungeLenzVector(n=3, l=1)
        Ax, Ay, Az = A.operators()

        self.assertEqual(Ax.shape, (3, 3))
        self.assertEqual(Ay.shape, (3, 3))
        self.assertEqual(Az.shape, (3, 3))


class TestSO4Generators(unittest.TestCase):
    """SO(4)生成元测试"""

    def test_creation(self):
        """测试SO(4)生成元创建"""
        gens = SO4Generators(max_n=3)
        self.assertEqual(gens.max_n, 3)
        self.assertEqual(gens.dim, 9)

    def test_L_operators(self):
        """测试角动量算符获取"""
        gens = SO4Generators(max_n=3)

        L_ops = gens.get_L_operators(n=2, l=1)
        self.assertIn("x", L_ops)
        self.assertIn("y", L_ops)
        self.assertIn("z", L_ops)
        self.assertIn("2", L_ops)

    def test_casimir(self):
        """测试Casimir算符"""
        gens = SO4Generators(max_n=3)

        C = gens.casimir_total(n=2, l=0)
        self.assertEqual(C.shape, (1, 1))


class TestHydrogenSO4Symmetry(unittest.TestCase):
    """氢原子SO(4)对称性测试"""

    def setUp(self):
        self.analyzer = HydrogenSO4Symmetry(max_n=4, Z=1.0)

    def test_energy_levels(self):
        """测试能级计算"""
        E1 = self.analyzer.energy_level(1)
        E2 = self.analyzer.energy_level(2)
        E3 = self.analyzer.energy_level(3)

        self.assertAlmostEqual(E1, -0.5)
        self.assertAlmostEqual(E2, -0.125)
        self.assertAlmostEqual(E3, -1.0 / 18)

    def test_degeneracy(self):
        """测试简并度"""
        self.assertEqual(self.analyzer.degeneracy(1), 1)
        self.assertEqual(self.analyzer.degeneracy(2), 4)
        self.assertEqual(self.analyzer.degeneracy(3), 9)
        self.assertEqual(self.analyzer.degeneracy(4), 16)

    def test_so4_quantum_numbers(self):
        """测试SO(4)量子数"""
        qn = self.analyzer.so4_quantum_numbers(n=3, l=1)

        self.assertEqual(qn.n, 3)
        self.assertEqual(qn.j1, 1.0)
        self.assertEqual(qn.j2, 1.0)

    def test_casimir_eigenvalue(self):
        """测试Casimir算符本征值"""
        self.assertEqual(self.analyzer.casimir_eigenvalue(1), 0)
        self.assertEqual(self.analyzer.casimir_eigenvalue(2), 3)
        self.assertEqual(self.analyzer.casimir_eigenvalue(3), 8)

    def test_hamiltonian_commutes(self):
        """测试哈密顿量与生成元对易"""
        results = self.analyzer.verify_h_commutes_with_generators(tol=1e-8)

        self.assertTrue(len(results) > 0)
        for key, value in results.items():
            self.assertTrue(value, f"Failed: {key}")

    def test_so4_commutation(self):
        """测试SO(4)对易关系"""
        results = self.analyzer.verify_so4_commutation(tol=1e-8)

        self.assertTrue(len(results) > 0)

    def test_degeneracy_structure(self):
        """测试简并结构分析"""
        structure = self.analyzer.analyze_degeneracy_structure()

        self.assertEqual(len(structure), 4)

        for n in range(1, 5):
            self.assertIn(n, structure)
            self.assertEqual(structure[n]["degeneracy"], n**2)

    def test_report_generation(self):
        """测试报告生成"""
        report = self.analyzer.generate_report()

        self.assertIsInstance(report, str)
        self.assertIn("SO(4)", report)
        self.assertIn("HYDROGEN", report)


class TestHydrogenSymmetryAnalyzer(unittest.TestCase):
    """氢原子对称性分析器测试"""

    def test_detect_symmetries(self):
        """测试对称性检测"""
        analyzer = HydrogenSymmetryAnalyzer(max_n=3)
        symmetries = analyzer.detect_symmetries()

        self.assertTrue(len(symmetries) >= 3)

        names = [s["name"] for s in symmetries]
        self.assertIn("SO(4)", names)
        self.assertIn("SO(3)", names)
        self.assertIn("Parity", names)

    def test_analyze(self):
        """测试完整分析"""
        analyzer = HydrogenSymmetryAnalyzer(max_n=3)
        result = analyzer.analyze()

        self.assertIn("symmetries", result)
        self.assertIn("degeneracy_structure", result)


class TestHydrogenAtomHamiltonianSO4(unittest.TestCase):
    """氢原子哈密顿量SO(4)测试"""

    def test_so4_detection(self):
        """测试SO(4)对称性检测"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=4)

        self.assertTrue(H.has_so4_symmetry())

    def test_so4_quantum_numbers(self):
        """测试SO(4)量子数"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=4)

        qn = H.so4_quantum_numbers(n=3, l=1)

        self.assertEqual(qn["n"], 3)
        self.assertEqual(qn["j1"], 1.0)
        self.assertEqual(qn["j2"], 1.0)
        self.assertEqual(qn["casimir"], 8)

    def test_so4_report(self):
        """测试SO(4)报告生成"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=4)
        report = H.generate_so4_report()

        self.assertIsInstance(report, str)
        self.assertIn("SO(4)", report)

    def test_verify_so4(self):
        """测试SO(4)对称性验证"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=4)
        result = H.verify_so4_symmetry()

        self.assertTrue(result["detected"])


class TestSO4HydrogenAnalyzer(unittest.TestCase):
    """SO4HydrogenAnalyzer测试"""

    def test_creation_with_hamiltonian(self):
        """测试使用哈密顿量创建"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=3)
        analyzer = SO4HydrogenAnalyzer(H)

        self.assertIsNotNone(analyzer)

    def test_creation_without_hamiltonian(self):
        """测试不使用哈密顿量创建"""
        analyzer = SO4HydrogenAnalyzer(max_n=3, Z=1.0)

        self.assertIsNotNone(analyzer)

    def test_detect_so4(self):
        """测试SO(4)检测"""
        analyzer = SO4HydrogenAnalyzer(max_n=3)
        result = analyzer.detect_so4_symmetry()

        self.assertTrue(result["detected"])
        self.assertEqual(result["symmetry_type"], "SO(4)")
        self.assertTrue(result["degeneracy_explained"])

    def test_analyze_so4_structure(self):
        """测试SO(4)结构分析"""
        analyzer = SO4HydrogenAnalyzer(max_n=3)
        structure = analyzer.analyze_so4_structure()

        self.assertEqual(structure["algebra"], "so(4)")
        self.assertEqual(structure["dimension"], 6)
        self.assertEqual(len(structure["generators"]), 6)

    def test_classify_states(self):
        """测试态分类"""
        analyzer = SO4HydrogenAnalyzer(max_n=3)
        classifications = analyzer.classify_hydrogen_states()

        self.assertTrue(len(classifications) > 0)

        for cls in classifications:
            self.assertIn("n", cls)
            self.assertIn("l", cls)
            self.assertIn("m", cls)
            self.assertIn("so4_label", cls)


class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试"""

    def test_analyze_hydrogen_symmetry(self):
        """测试analyze_hydrogen_symmetry函数"""
        report = analyze_hydrogen_symmetry(max_n=4, Z=1.0)

        self.assertIsInstance(report, str)
        self.assertIn("SO(4)", report)

    def test_create_hydrogen_so4_analyzer(self):
        """测试create_hydrogen_so4_analyzer函数"""
        analyzer = create_hydrogen_so4_analyzer(max_n=3, Z=1.0)

        self.assertIsInstance(analyzer, HydrogenSymmetryAnalyzer)

    def test_detect_hydrogen_so4_symmetry(self):
        """测试detect_hydrogen_so4_symmetry函数"""
        result = detect_hydrogen_so4_symmetry(max_n=3, Z=1.0)

        self.assertTrue(result["detected"])
        self.assertEqual(result["symmetry_type"], "SO(4)")


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_analysis_workflow(self):
        """测试完整分析工作流"""
        H = HydrogenAtomHamiltonian(Z=1.0, max_n=4)

        analyzer = SO4HydrogenAnalyzer(H)

        detection = analyzer.detect_so4_symmetry()
        self.assertTrue(detection["detected"])

        structure = analyzer.analyze_so4_structure()
        self.assertEqual(structure["algebra"], "so(4)")

        report = analyzer.generate_report()
        self.assertIsInstance(report, str)

        classifications = analyzer.classify_hydrogen_states()
        self.assertEqual(len(classifications), H.dimension)


if __name__ == "__main__":
    unittest.main()
