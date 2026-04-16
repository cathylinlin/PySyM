import pytest
import numpy as np
from PySymmetry.abstract_phys.representation.phys_representation import (
    PhysicalRepresentation,
    SU2Representation,
    SU3Representation,
    LorentzRepresentation,
    RepresentationFactory,
)


class MockHilbertSpace:
    """Mock Hilbert space for testing."""

    def __init__(self, dim=2):
        self._dim = dim

    def dimension(self):
        return self._dim


class MockGroup:
    """Mock group for testing."""

    def __init__(self, name="MockGroup"):
        self.name = name


class TestPhysicalRepresentation:
    """Test suite for PhysicalRepresentation."""

    def test_creation(self):
        """Test basic creation."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)
        assert rep.group is group
        assert rep.space is space

    def test_get_matrix_default(self):
        """Test getting default identity matrix."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)

        matrix = rep.get_matrix("e")
        assert matrix.shape == (2, 2)
        assert np.allclose(matrix, np.eye(2))

    def test_set_matrix(self):
        """Test setting a matrix."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)

        test_matrix = np.array([[1, 0], [0, -1]])
        rep.set_matrix("sigma_z", test_matrix)

        retrieved = rep.get_matrix("sigma_z")
        assert np.allclose(retrieved, test_matrix)

    def test_is_unitary_identity(self):
        """Test unitary check with identity matrix."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)
        rep.set_matrix("e", np.eye(2))

        assert rep.is_unitary() is True

    def test_is_unitary_pauli_x(self):
        """Test unitary check with Pauli X."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)
        rep.set_matrix("sigma_x", np.array([[0, 1], [1, 0]]))

        assert rep.is_unitary() is True

    def test_character_identity(self):
        """Test character of identity."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)
        rep.set_matrix("e", np.eye(2))

        char = rep.character("e")
        assert char == 2

    def test_character_pauli_z(self):
        """Test character of Pauli Z."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)
        rep.set_matrix("sigma_z", np.array([[1, 0], [0, -1]]))

        char = rep.character("sigma_z")
        assert char == 0

    def test_decompose_returns_self(self):
        """Test decompose returns self."""
        space = MockHilbertSpace(2)
        group = MockGroup()
        rep = PhysicalRepresentation(group, space)

        decomposed = rep.decompose()
        assert len(decomposed) == 1
        assert decomposed[0] is rep

    def test_tensor_product(self):
        """Test tensor product of representations."""
        space1 = MockHilbertSpace(2)
        space2 = MockHilbertSpace(2)
        group = MockGroup()

        rep1 = PhysicalRepresentation(group, space1)
        rep2 = PhysicalRepresentation(group, space2)

        result = rep1.tensor_product(rep2)
        assert isinstance(result, PhysicalRepresentation)
        assert result.space.dimension() == 4


class TestSU2Representation:
    """Test suite for SU2Representation."""

    def test_creation_spin_half(self):
        """Test creation for spin-1/2."""
        rep = SU2Representation(0.5)
        assert rep.j == 0.5

    def test_creation_spin_one(self):
        """Test creation for spin-1."""
        rep = SU2Representation(1)
        assert rep.j == 1

    def test_dimension_spin_half(self):
        """Test dimension for spin-1/2."""
        rep = SU2Representation(0.5)
        assert rep.dimension == 2

    def test_dimension_spin_one(self):
        """Test dimension for spin-1."""
        rep = SU2Representation(1)
        assert rep.dimension == 3

    def test_highest_weight(self):
        """Test highest weight."""
        rep = SU2Representation(1)
        hw = rep.highest_weight()
        assert len(hw) == 1
        assert hw[0] == 2  # 2*j = 2

    def test_is_irreducible(self):
        """Test is_irreducible returns True."""
        rep = SU2Representation(0.5)
        assert rep.is_irreducible() is True

    def test_weight_diagram_spin_half(self):
        """Test weight diagram for spin-1/2."""
        rep = SU2Representation(0.5)
        weights = rep.weight_diagram()
        assert len(weights) == 2

    def test_generators(self):
        """Test generators exist."""
        rep = SU2Representation(0.5)
        gens = rep.generators()
        assert len(gens) == 3


class TestSU3Representation:
    """Test suite for SU3Representation."""

    def test_creation(self):
        """Test basic creation."""
        rep = SU3Representation(1, 1)
        assert rep.p == 1
        assert rep.q == 1

    def test_dimension(self):
        """Test dimension."""
        rep = SU3Representation(1, 1)
        dim = rep.dimension
        assert dim > 0


class TestLorentzRepresentation:
    """Test suite for LorentzRepresentation."""

    def test_creation_scalar(self):
        """Test scalar representation."""
        rep = LorentzRepresentation(0, 0)
        assert rep.left_chirality == 0
        assert rep.right_chirality == 0

    def test_creation_spinor(self):
        """Test spinor representation."""
        rep = LorentzRepresentation(0.5, 0)
        assert rep.left_chirality == 0.5
        assert rep.right_chirality == 0

    def test_dimension_scalar(self):
        """Test scalar dimension is 1."""
        rep = LorentzRepresentation(0, 0)
        assert rep.dimension == 1

    def test_dimension_vector(self):
        """Test vector dimension is 4."""
        rep = LorentzRepresentation(0.5, 0.5)
        assert rep.dimension == 4


class TestRepresentationFactory:
    """Test suite for RepresentationFactory."""

    def test_factory_exists(self):
        """Test factory class exists."""
        assert hasattr(RepresentationFactory, "create")


class TestRepresentationProperties:
    """Test representation properties and relationships."""

    def test_spin_orbit_coupling(self):
        """Test adding spin and orbital representations."""
        space_orbital = MockHilbertSpace(3)
        space_spin = MockHilbertSpace(2)
        group = MockGroup()

        rep_orbital = SU2Representation(1)
        rep_spin = SU2Representation(0.5)

        result = rep_orbital.tensor_product(rep_spin)
        assert result.space.dimension() == 6

    def test_adding_representations(self):
        """Test direct sum of representations."""
        space1 = MockHilbertSpace(2)
        space2 = MockHilbertSpace(3)
        group = MockGroup()

        rep1 = SU2Representation(0.5)
        rep2 = SU2Representation(1)

        total_dim = rep1.dimension + rep2.dimension
        assert total_dim == 5
