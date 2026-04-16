import pytest
import numpy as np
from sympy import symbols, Matrix, simplify, expand, Symbol
from src.PySymmetry.tools.lie_algebra_symbolic import (
    SymbolicLieAlgebraElement,
    SymbolicLieBracket,
    SymbolicLieAlgebra,
    SymbolicKillingForm,
    SymbolicWeylGroup,
    compute_structure_constants,
    verify_jacobi_identity,
    generate_weyl_coordinates,
)


class TestSymbolicLieAlgebraElement:
    def test_init_with_coefficients(self):
        coeffs = [1, 2, 3]
        elem = SymbolicLieAlgebraElement(coeffs)
        assert elem.coefficients == coeffs
        assert elem.dimension == 3
        assert elem.basis_labels == ["e0", "e1", "e2"]

    def test_init_with_labels(self):
        coeffs = [1, 0, 0]
        labels = ["H1", "H2", "E_alpha"]
        elem = SymbolicLieAlgebraElement(coeffs, labels)
        assert elem.basis_labels == labels

    def test_basis_element(self):
        elem = SymbolicLieAlgebraElement.basis_element(1, 3)
        assert elem.coefficients == [0, 1, 0]
        assert elem.dimension == 3

    def test_addition(self):
        a = SymbolicLieAlgebraElement([1, 2, 3])
        b = SymbolicLieAlgebraElement([4, 5, 6])
        c = a + b
        assert c.coefficients == [5, 7, 9]

    def test_subtraction(self):
        a = SymbolicLieAlgebraElement([4, 6, 8])
        b = SymbolicLieAlgebraElement([1, 2, 3])
        c = a - b
        assert c.coefficients == [3, 4, 5]

    def test_scalar_multiplication(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3])
        result = elem * 2
        assert result.coefficients == [2, 4, 6]

    def test_right_scalar_multiplication(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3])
        result = 3 * elem
        assert result.coefficients == [3, 6, 9]

    def test_negation(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3])
        neg = -elem
        assert neg.coefficients == [-1, -2, -3]

    def test_to_symbolic_vector(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3])
        vec = elem.to_symbolic_vector()
        assert isinstance(vec, Matrix)
        assert vec == Matrix([1, 2, 3])

    def test_to_expression(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3], ["x", "y", "z"])
        expr = elem.to_expression()
        assert expr is not None

    def test_simplify(self):
        x = Symbol("x")
        elem = SymbolicLieAlgebraElement([x, x + x, 3])
        simplified = elem.simplify()
        assert len(simplified.coefficients) == 3

    def test_expand(self):
        x = Symbol("x")
        y = Symbol("y")
        elem = SymbolicLieAlgebraElement([x * y, x + y])
        expanded = elem.expand()
        assert len(expanded.coefficients) == 2

    def test_str_representation(self):
        elem = SymbolicLieAlgebraElement([1, 0, 2])
        s = str(elem)
        assert "e0" in s or "1*e0" in s

    def test_repr_representation(self):
        elem = SymbolicLieAlgebraElement([1, 2, 3])
        r = repr(elem)
        assert "SymbolicLieAlgebraElement" in r


class TestSymbolicLieBracket:
    def test_init_structure_constants(self):
        struct = {(0, 1): [0, 1, 0]}
        bracket = SymbolicLieBracket(struct)
        assert bracket.structure_constants == struct

    def test_bracket_with_structure_constants(self):
        struct = {(0, 1): [0, 0, 1], (1, 0): [0, 0, -1]}
        bracket = SymbolicLieBracket(struct)

        e0 = SymbolicLieAlgebraElement([1, 0, 0])
        e1 = SymbolicLieAlgebraElement([0, 1, 0])

        result = bracket(e0, e1)
        assert result.dimension == 3

    def test_verify_anticommutative(self):
        struct = {(0, 1): [0, 0, 1], (1, 0): [0, 0, -1]}
        bracket = SymbolicLieBracket(struct)
        basis = [
            SymbolicLieAlgebraElement.basis_element(0, 3),
            SymbolicLieAlgebraElement.basis_element(1, 3),
            SymbolicLieAlgebraElement.basis_element(2, 3),
        ]
        assert bracket.verify_anticommutative(basis) == True

    def test_verify_jacobi_identity(self):
        struct = {
            (0, 1): [0, 0, 1],
            (1, 2): [0, 0, 1],
            (2, 0): [0, 0, 1],
            (1, 0): [0, 0, -1],
            (2, 1): [0, 0, -1],
            (0, 2): [0, 0, -1],
        }
        bracket = SymbolicLieBracket(struct)
        basis = [
            SymbolicLieAlgebraElement.basis_element(0, 3),
            SymbolicLieAlgebraElement.basis_element(1, 3),
            SymbolicLieAlgebraElement.basis_element(2, 3),
        ]
        assert bracket.verify_jacobi_identity(basis) == True

    def test_to_matrix(self):
        struct = {}
        bracket = SymbolicLieBracket(struct)
        matrices = bracket.to_matrix()
        assert isinstance(matrices, list)


class TestSymbolicLieAlgebra:
    def test_init_basic(self):
        algebra = SymbolicLieAlgebra("test", 3)
        assert algebra.name == "test"
        assert algebra.dimension == 3

    def test_init_with_structure_constants(self):
        struct = {(0, 1): [1]}
        algebra = SymbolicLieAlgebra("test", 2, structure_constants=struct)
        assert algebra.structure_constants == struct

    def test_init_with_basis_labels(self):
        labels = ["H", "E_plus", "E_minus"]
        algebra = SymbolicLieAlgebra("sl2", 3, basis_labels=labels)
        assert algebra.basis_labels == labels

    def test_killing_form_matrix(self):
        algebra = SymbolicLieAlgebra("test", 3)
        killing = algebra.killing_form_matrix()
        assert isinstance(killing, (Matrix, np.ndarray, list))

    def test_root_system(self):
        algebra = SymbolicLieAlgebra("test", 3)
        rs = algebra.root_system()
        assert isinstance(rs, dict)

    def test_dimension_property(self):
        algebra = SymbolicLieAlgebra("test", 5)
        assert algebra.dimension == 5


class TestSymbolicKillingForm:
    def test_init_with_lie_algebra(self):
        algebra = SymbolicLieAlgebra("test", 3)
        kf = SymbolicKillingForm(algebra)
        assert kf.lie_algebra is algebra

    def test_call_with_elements(self):
        algebra = SymbolicLieAlgebra("test", 2)
        kf = SymbolicKillingForm(algebra)
        e0 = SymbolicLieAlgebraElement.basis_element(0, 2)
        e1 = SymbolicLieAlgebraElement.basis_element(1, 2)
        result = kf(e0, e1)
        assert result is not None

    def test_is_non_degenerate(self):
        algebra = SymbolicLieAlgebra("test", 2)
        kf = SymbolicKillingForm(algebra)
        result = kf.is_non_degenerate()
        assert isinstance(result, bool)

    def test_str_representation(self):
        algebra = SymbolicLieAlgebra("test", 3)
        kf = SymbolicKillingForm(algebra)
        s = str(kf)
        assert "SymbolicKillingForm" in s


class TestSymbolicWeylGroup:
    def test_init_with_lie_algebra(self):
        algebra = SymbolicLieAlgebra("test", 3)
        wg = SymbolicWeylGroup(algebra)
        assert wg.lie_algebra is algebra

    def test_reflect(self):
        algebra = SymbolicLieAlgebra("test", 2)
        wg = SymbolicWeylGroup(algebra)
        vec = Matrix([1, 0])
        result = wg.reflect(0, vec)
        assert isinstance(result, Matrix)

    def test_order(self):
        algebra = SymbolicLieAlgebra("test", 2)
        wg = SymbolicWeylGroup(algebra)
        order = wg.order()
        assert isinstance(order, int)

    def test_str_representation(self):
        algebra = SymbolicLieAlgebra("test", 2)
        wg = SymbolicWeylGroup(algebra)
        s = str(wg)
        assert "SymbolicWeylGroup" in s


class TestUtilityFunctions:
    def test_compute_structure_constants_A(self):
        struct = compute_structure_constants("A", 3)
        assert isinstance(struct, dict)

    def test_compute_structure_constants_B(self):
        struct = compute_structure_constants("B", 3)
        assert isinstance(struct, dict)

    def test_compute_structure_constants_C(self):
        struct = compute_structure_constants("C", 3)
        assert isinstance(struct, dict)

    def test_compute_structure_constants_D(self):
        struct = compute_structure_constants("D", 4)
        assert isinstance(struct, dict)

    def test_compute_structure_constants_G2(self):
        struct = compute_structure_constants("G2", 2)
        assert isinstance(struct, dict)

    def test_verify_jacobi_A(self):
        result = verify_jacobi_identity("A", 3)
        assert isinstance(result, bool)

    def test_verify_jacobi_B(self):
        result = verify_jacobi_identity("B", 3)
        assert isinstance(result, bool)

    def test_generate_weyl_coordinates_A(self):
        coords = generate_weyl_coordinates("A", 2)
        assert isinstance(coords, list)

    def test_generate_weyl_coordinates_D(self):
        coords = generate_weyl_coordinates("D", 4)
        assert isinstance(coords, list)

    def test_generate_weyl_coordinates_G2(self):
        coords = generate_weyl_coordinates("G2", 2)
        assert isinstance(coords, list)
