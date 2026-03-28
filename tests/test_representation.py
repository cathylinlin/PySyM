"""群表示论回归测试：同态、特征标正交、不可约分解与诱导表示（供物理层对接）。"""
import unittest

import numpy as np

from PySyM.core.group_theory import (
    AlternatingGroup,
    CyclicGroup,
    Subgroup,
    SymmetricGroup,
)
from PySyM.core.representation import (
    Character,
    InducedRepresentation,
    IrreducibleRepresentationFinder,
    MatrixRepresentation,
)


class TestRepresentation(unittest.TestCase):
    def setUp(self):
        self.s3 = SymmetricGroup(3)

    def test_trivial_and_regular_are_homomorphisms(self):
        triv = MatrixRepresentation.trivial_representation(self.s3)
        reg = MatrixRepresentation.regular_representation(self.s3)
        self.assertTrue(triv.is_homomorphism())
        self.assertTrue(reg.is_homomorphism())
        self.assertEqual(triv.dimension, 1)
        self.assertEqual(reg.dimension, self.s3.order())

    def test_s3_irrep_dimensions_sum_of_squares(self):
        irreps = IrreducibleRepresentationFinder.find_all(self.s3)
        dims = sorted(r.dimension for r in irreps)
        self.assertEqual(dims, [1, 1, 2])
        self.assertEqual(sum(d * d for d in dims), self.s3.order())
        for r in irreps:
            self.assertTrue(
                r.is_homomorphism(),
                msg="S3 不可约表示须为群同态，供物理荷载与张量积使用",
            )

    def test_s3_character_orthogonality(self):
        irreps = IrreducibleRepresentationFinder.find_all(self.s3)
        chars = [Character(r) for r in irreps]
        for i, ci in enumerate(chars):
            self.assertTrue(ci.is_irreducible())
            for j, cj in enumerate(chars):
                ip = ci.inner_product(cj)
                if i == j:
                    self.assertAlmostEqual(float(np.real(ip)), 1.0, places=8)
                    self.assertAlmostEqual(float(np.imag(ip)), 0.0, places=8)
                else:
                    self.assertAlmostEqual(float(np.real(ip)), 0.0, places=8)
                    self.assertAlmostEqual(float(np.imag(ip)), 0.0, places=8)

    def test_tensor_and_direct_sum_homomorphism(self):
        irreps = IrreducibleRepresentationFinder.find_all(self.s3)
        tensor = irreps[0].compose(irreps[1])
        dsum = irreps[0].direct_sum(irreps[2])
        self.assertTrue(tensor.is_homomorphism())
        self.assertTrue(dsum.is_homomorphism())
        self.assertEqual(tensor.dimension, 1)
        self.assertEqual(dsum.dimension, 3)

    def test_induced_from_alternating_subgroup(self):
        s3 = self.s3
        a3 = AlternatingGroup(3)
        rho = MatrixRepresentation.trivial_representation(a3)
        ind = InducedRepresentation(s3, a3, rho)
        self.assertEqual(ind.dimension, s3.order() // a3.order())
        self.assertTrue(ind.is_homomorphism())

    def test_induced_subgroup_carrier_may_differ_from_rho_group(self):
        """ρ 定义在 AlternatingGroup 上，子群用 Subgroup 包装时载体应判定为一致。"""
        s3 = self.s3
        a3 = AlternatingGroup(3)
        h = Subgroup(s3, list(a3.elements()))
        rho = MatrixRepresentation.trivial_representation(a3)
        ind = InducedRepresentation(s3, h, rho)
        self.assertEqual(ind.dimension, 2)
        self.assertTrue(ind.is_homomorphism())

    def test_decompose_reducible_s3(self):
        irreps = IrreducibleRepresentationFinder.find_all(self.s3)
        red = irreps[0].direct_sum(irreps[1])
        parts = red.decompose()
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(Character(p).is_irreducible() for p in parts))

    def test_cyclic_trivial_irreducible(self):
        g = CyclicGroup(4)
        triv = MatrixRepresentation.trivial_representation(g)
        self.assertTrue(triv.is_homomorphism())
        self.assertTrue(Character(triv).is_irreducible())


if __name__ == "__main__":
    unittest.main()
