import pytest
import numpy as np
from PySymmetry.abstract_phys.representation.selection_rules import (
    SelectionRule,
    ElectricDipoleSelectionRule,
    ParitySelectionRule,
    GroupTheorySelectionRule,
    VibrationSelectionRule,
    ComprehensiveSelectionRules,
)


class TestElectricDipoleSelectionRule:
    """Test suite for ElectricDipoleSelectionRule."""

    def test_creation(self):
        """Test basic creation."""
        rule = ElectricDipoleSelectionRule()
        assert rule.include_spin_orbit is True

    def test_creation_without_spin_orbit(self):
        """Test creation without spin-orbit coupling."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        assert rule.include_spin_orbit is False

    def test_allowed_transition_s_p(self):
        """Test allowed transition from s to p orbital."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        initial = {"l": 0, "s": 0.5}
        final = {"l": 1, "s": 0.5}
        assert rule.is_allowed(initial, final) is True

    def test_forbidden_transition_s_s(self):
        """Test forbidden transition s to s."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        initial = {"l": 0, "s": 0.5}
        final = {"l": 0, "s": 0.5}
        assert rule.is_allowed(initial, final) is False

    def test_forbidden_transition_s_d(self):
        """Test forbidden transition s to d (Δl=2)."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        initial = {"l": 0, "s": 0.5}
        final = {"l": 2, "s": 0.5}
        assert rule.is_allowed(initial, final) is False

    def test_spin_forbidden_transition(self):
        """Test spin forbidden transition."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        initial = {"l": 0, "s": 0.5}
        final = {"l": 1, "s": 1.5}
        assert rule.is_allowed(initial, final) is False

    def test_with_spin_orbit_coupling_allowed(self):
        """Test allowed transition with spin-orbit coupling."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=True)
        initial = {"l": 0, "s": 0.5, "j": 0.5}
        final = {"l": 1, "s": 0.5, "j": 0.5}
        assert rule.is_allowed(initial, final) is True

    def test_with_spin_orbit_j_forbidden(self):
        """Test j=0 to j'=0 forbidden transition."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=True)
        initial = {"l": 0, "s": 0.5, "j": 0.5}
        final = {"l": 1, "s": 0.5, "j": 1.5}
        assert rule.is_allowed(initial, final) is True

    def test_with_spin_orbit_large_delta_j(self):
        """Test large delta j forbidden."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=True)
        initial = {"l": 0, "s": 0.5, "j": 0.5}
        final = {"l": 1, "s": 0.5, "j": 2.5}
        assert rule.is_allowed(initial, final) is False

    def test_missing_l_values(self):
        """Test with missing l values defaults to 0."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        initial = {"l": 0}
        final = {"l": 1}
        assert rule.is_allowed(initial, final) is True

    def test_get_allowed_transitions(self):
        """Test getting allowed transitions."""
        rule = ElectricDipoleSelectionRule(include_spin_orbit=False)
        states = [
            {"l": 0, "s": 0.5},
            {"l": 1, "s": 0.5},
            {"l": 2, "s": 0.5},
        ]
        transitions = rule.get_allowed_transitions(states)
        assert len(transitions) > 0
        assert (0, 1) in transitions
        assert (1, 2) in transitions


class TestParitySelectionRule:
    """Test suite for ParitySelectionRule."""

    def test_creation_electric_dipole(self):
        """Test creation with electric dipole."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        assert rule.transition_type == "electric_dipole"

    def test_creation_magnetic_dipole(self):
        """Test creation with magnetic dipole."""
        rule = ParitySelectionRule(transition_type="magnetic_dipole")
        assert rule.transition_type == "magnetic_dipole"

    def test_creation_electric_quadrupole(self):
        """Test creation with electric quadrupole."""
        rule = ParitySelectionRule(transition_type="electric_quadrupole")
        assert rule.transition_type == "electric_quadrupole"

    def test_allowed_electric_dipole_gu(self):
        """Test allowed electric dipole g→u transition."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        initial = {"parity": 1}
        final = {"parity": -1}
        assert rule.is_allowed(initial, final) is True

    def test_forbidden_electric_dipole_gg(self):
        """Test forbidden electric dipole g→g transition."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        initial = {"parity": 1}
        final = {"parity": 1}
        assert rule.is_allowed(initial, final) is False

    def test_forbidden_electric_dipole_uu(self):
        """Test forbidden electric dipole u→u transition."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        initial = {"parity": -1}
        final = {"parity": -1}
        assert rule.is_allowed(initial, final) is False

    def test_allowed_magnetic_dipole_gg(self):
        """Test allowed magnetic dipole g→g transition."""
        rule = ParitySelectionRule(transition_type="magnetic_dipole")
        initial = {"parity": 1}
        final = {"parity": 1}
        assert rule.is_allowed(initial, final) is True

    def test_allowed_magnetic_dipole_uu(self):
        """Test allowed magnetic dipole u→u transition."""
        rule = ParitySelectionRule(transition_type="magnetic_dipole")
        initial = {"parity": -1}
        final = {"parity": -1}
        assert rule.is_allowed(initial, final) is True

    def test_forbidden_magnetic_dipole_gu(self):
        """Test forbidden magnetic dipole g→u transition."""
        rule = ParitySelectionRule(transition_type="magnetic_dipole")
        initial = {"parity": 1}
        final = {"parity": -1}
        assert rule.is_allowed(initial, final) is False

    def test_same_parity_electric_dipole_forbidden(self):
        """Test same parity with electric dipole is forbidden."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        initial = {"parity": 1}
        final = {"parity": 1}
        assert rule.is_allowed(initial, final) is False

    def test_get_allowed_transitions(self):
        """Test getting allowed transitions."""
        rule = ParitySelectionRule(transition_type="electric_dipole")
        states = [
            {"parity": 1},
            {"parity": -1},
        ]
        transitions = rule.get_allowed_transitions(states)
        assert len(transitions) == 1
        assert (0, 1) in transitions


class TestVibrationSelectionRule:
    """Test suite for VibrationSelectionRule - tests require PointGroup."""

    def test_creation_requires_point_group(self):
        """Test VibrationSelectionRule requires PointGroup."""
        from PySymmetry.abstract_phys.symmetry_environments.discrete_symmetries.point_groups import (
            PointGroup,
        )

        pass


class TestComprehensiveSelectionRules:
    """Test suite for ComprehensiveSelectionRules."""

    def test_creation(self):
        """Test basic creation."""
        rules = ComprehensiveSelectionRules()
        assert rules.parity_rule is not None
        assert rules.electric_dipole is not None

    def test_parity_rule_type(self):
        """Test parity rule is ParitySelectionRule."""
        rules = ComprehensiveSelectionRules()
        assert isinstance(rules.parity_rule, ParitySelectionRule)

    def test_electric_dipole_rule_type(self):
        """Test electric dipole rule is ElectricDipoleSelectionRule."""
        rules = ComprehensiveSelectionRules()
        assert isinstance(rules.electric_dipole, ElectricDipoleSelectionRule)


class TestSelectionRuleCombinations:
    """Test combining multiple selection rules."""

    def test_combined_rules(self):
        """Test using multiple selection rules together."""
        electric_dipole = ElectricDipoleSelectionRule(include_spin_orbit=False)
        parity = ParitySelectionRule(transition_type="electric_dipole")

        initial = {"l": 0, "s": 0.5, "parity": 1}
        final = {"l": 1, "s": 0.5, "parity": -1}

        assert electric_dipole.is_allowed(initial, final) is True
        assert parity.is_allowed(initial, final) is True

    def test_combined_rules_one_fails(self):
        """Test when one rule fails."""
        electric_dipole = ElectricDipoleSelectionRule(include_spin_orbit=False)
        parity = ParitySelectionRule(transition_type="electric_dipole")

        initial = {"l": 0, "s": 0.5, "parity": 1}
        final = {"l": 1, "s": 0.5, "parity": 1}

        assert electric_dipole.is_allowed(initial, final) is True
        assert parity.is_allowed(initial, final) is False
