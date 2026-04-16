import pytest
import numpy as np
from PySymmetry.core.group_theory.product_group import (
    DirectProductGroup,
    DirectProductGroupElement,
)
from PySymmetry.core.group_theory.specific_group import CyclicGroup


class TestDirectProductGroup:
    """Test suite for DirectProductGroup."""

    def test_creation_with_cyclic_groups(self):
        """Test DirectProductGroup creation with CyclicGroups."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        assert "C_2" in product_group.name
        assert "C_3" in product_group.name
        assert product_group.group1 == group1
        assert product_group.group2 == group2

    def test_order_calculation(self):
        """Test order calculation."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.order() == 6

    def test_elements(self):
        """Test getting all elements."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        elements = product_group.elements()
        assert len(elements) == 4
        for elem in elements:
            assert isinstance(elem, DirectProductGroupElement)

    def test_identity(self):
        """Test identity element."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        identity = product_group.identity()
        assert identity is not None
        assert isinstance(identity, DirectProductGroupElement)

    def test_multiply(self):
        """Test group multiplication."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        elements = product_group.elements()
        result = product_group.multiply(elements[0], elements[1])
        assert result is not None

    def test_inverse(self):
        """Test inverse calculation."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        elements = product_group.elements()
        for element in elements:
            inv = product_group.inverse(element)
            assert inv is not None

    def test_is_abelian_cyclic_groups(self):
        """Test abelian property for cyclic groups (always abelian)."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.is_abelian() is True

    def test_is_simple(self):
        """Test simple property - should always be False for direct product."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.is_simple() is False

    def test_conjugacy_classes(self):
        """Test conjugacy classes calculation."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        classes = product_group.conjugacy_classes()
        assert len(classes) > 0

    def test_get_components_from_index(self):
        """Test _get_components_from_index method."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        for i in range(product_group.order()):
            c1, c2 = product_group._get_components_from_index(i)
            assert c1 is not None
            assert c2 is not None


class TestDirectProductGroupElement:
    """Test suite for DirectProductGroupElement."""

    def test_component_access(self):
        """Test component access with valid indices."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        element = DirectProductGroupElement(product_group, 0)

        assert element.component(0) is not None
        assert element.component(1) is not None

    def test_component_invalid_index(self):
        """Test component access with invalid index."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        element = DirectProductGroupElement(product_group, 0)

        with pytest.raises(IndexError):
            element.component(2)

    def test_repr(self):
        """Test string representation."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        element = DirectProductGroupElement(product_group, 0)
        repr_str = repr(element)

        assert "(" in repr_str
        assert ")" in repr_str

    def test_multiplication(self):
        """Test multiplication of elements."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        elements = product_group.elements()
        result = elements[0] * elements[1]
        assert result is not None

    def test_element_id(self):
        """Test element_id attribute."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        for i in range(product_group.order()):
            element = DirectProductGroupElement(product_group, i)
            assert element.element_id == i


class TestDirectProductGroupIntegration:
    """Integration tests for DirectProductGroup."""

    def test_cyclic_groups_product(self):
        """Test product of cyclic groups."""
        group1 = CyclicGroup(3)
        group2 = CyclicGroup(4)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.order() == 12
        assert product_group.is_abelian()

        elements = product_group.elements()
        assert len(elements) == 12

    def test_identity_behavior(self):
        """Test identity element behavior."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        identity = product_group.identity()
        elements = product_group.elements()

        for elem in elements:
            result = product_group.multiply(identity, elem)
            assert result.component(0) == elem.component(0)
            assert result.component(1) == elem.component(1)

    def test_inverse_behavior(self):
        """Test inverse element behavior."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        elements = product_group.elements()
        for elem in elements:
            inv = product_group.inverse(elem)
            product = product_group.multiply(elem, inv)

            identity = product_group.identity()
            assert product.component(0) == identity.component(0)
            assert product.component(1) == identity.component(1)

    def test_get_element_from_components(self):
        """Test creating element from component groups."""
        group1 = CyclicGroup(2)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(group1, group2)

        g1 = group1.elements()[0]
        g2 = group2.elements()[0]
        element = product_group._get_element_from_components(g1, g2)

        assert isinstance(element, DirectProductGroupElement)
        assert element.component(0) == g1
        assert element.component(1) == g2


class TestDirectProductGroupEdgeCases:
    """Edge case tests for DirectProductGroup."""

    def test_trivial_group_product(self):
        """Test product with trivial group."""
        trivial = CyclicGroup(1)
        group2 = CyclicGroup(2)
        product_group = DirectProductGroup(trivial, group2)

        assert product_group.order() == 2
        elements = product_group.elements()
        assert len(elements) == 2

    def test_prime_order_groups(self):
        """Test product of prime order groups."""
        group1 = CyclicGroup(5)
        group2 = CyclicGroup(7)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.order() == 35
        assert product_group.is_abelian()

    def test_large_group(self):
        """Test with slightly larger group."""
        group1 = CyclicGroup(4)
        group2 = CyclicGroup(3)
        product_group = DirectProductGroup(group1, group2)

        assert product_group.order() == 12
        elements = product_group.elements()
        assert len(elements) == 12

        for elem in elements:
            inv = product_group.inverse(elem)
            assert inv is not None
