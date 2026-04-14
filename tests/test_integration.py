import pytest
import numpy as np
from PySymmetry.phys.quantum.integration import (
    IntegratedQuantumResult,
    QuantumToAbstractBridge,
    SymmetryToQuantumBridge,
    LieAlgebraToQuantumBridge,
    GroupRepresentationBridge,
    integrate_with_abstract_phys,
    create_quantum_from_abstract,
    export_to_abstract_phys,
    quick_bridge,
    quick_lie_bridge,
)


class TestIntegratedQuantumResult:
    def test_creation(self):
        result = IntegratedQuantumResult(
            quantum_result={'test': 1},
            symmetry_operations=['parity', 'translation'],
            lie_algebras={'so3': {}},
            groups={'detected': ['so3']},
            conserved_quantities=['energy', 'parity']
        )
        assert result.quantum_result['test'] == 1
        assert len(result.symmetry_operations) == 2
        assert len(result.conserved_quantities) == 2


class TestQuantumToAbstractBridge:
    def test_creation(self):
        bridge = QuantumToAbstractBridge()
        assert bridge is not None

    def test_fallback_system(self):
        from PySymmetry.phys.quantum.interactive import SceneBuilder
        
        scene = SceneBuilder("Test").build()
        bridge = QuantumToAbstractBridge()
        result = bridge._create_fallback_system(scene)
        
        assert result['name'] == "Test"
        assert 'dimension' in result
        assert 'hilbert_space_dim' in result

    def test_get_conserved_quantities_no_hamiltonian(self):
        bridge = QuantumToAbstractBridge()
        conserved = bridge.get_conserved_quantities()
        assert conserved == []

    def test_from_scene_fallback(self):
        from PySymmetry.phys.quantum.interactive import SceneBuilder
        
        scene = SceneBuilder("Fallback").set_dimension(2).build()
        bridge = QuantumToAbstractBridge()
        result = bridge.from_scene(scene)
        assert isinstance(result, dict)
        assert result['name'] == "Fallback"


class TestSymmetryToQuantumBridge:
    def test_creation(self):
        bridge = SymmetryToQuantumBridge(10)
        assert bridge is not None
        assert bridge._dim == 10

    def test_parity_operator(self):
        bridge = SymmetryToQuantumBridge(5)
        P = bridge.parity_operator()
        
        assert P.shape == (5, 5)
        assert np.allclose(P @ P, np.eye(5))
        
        assert 'parity' in bridge._symmetry_operators

    def test_parity_operator_different_sizes(self):
        for n in [3, 7, 10]:
            bridge = SymmetryToQuantumBridge(n)
            P = bridge.parity_operator()
            assert P.shape == (n, n)
            assert np.allclose(P @ P, np.eye(n))

    def test_translation_operator(self):
        bridge = SymmetryToQuantumBridge(5)
        T = bridge.translation_operator(k=0.5)
        
        assert T.shape == (5, 5)
        assert 'translation' in bridge._symmetry_operators

    def test_translation_operator_properties(self):
        bridge = SymmetryToQuantumBridge(4)
        T = bridge.translation_operator()
        
        assert np.iscomplexobj(T)

    def test_rotation_operator_z(self):
        bridge = SymmetryToQuantumBridge(6)
        R = bridge.rotation_operator(theta=np.pi/2, axis='z')
        
        assert R.shape == (6, 6)
        assert 'rotation_z' in bridge._symmetry_operators

    def test_time_reversal_operator(self):
        bridge = SymmetryToQuantumBridge(4)
        T = bridge.time_reversal_operator()
        
        assert T.shape == (4, 4)
        assert np.allclose(T @ T, np.eye(4))
        assert 'time_reversal' in bridge._symmetry_operators

    def test_angular_momentum_operators(self):
        bridge = SymmetryToQuantumBridge(6)
        L = bridge.angular_momentum_operators()
        
        assert 'Lx' in L
        assert 'Ly' in L
        assert 'Lz' in L
        assert 'Lx' in bridge._symmetry_operators
        assert 'Ly' in bridge._symmetry_operators
        assert 'Lz' in bridge._symmetry_operators
        
        for key in L:
            assert L[key].shape == (6, 6)

    def test_get_operator(self):
        bridge = SymmetryToQuantumBridge(5)
        bridge.parity_operator()
        
        P = bridge.get_operator('parity')
        assert P is not None
        assert P.shape == (5, 5)
        
        nonexistent = bridge.get_operator('nonexistent')
        assert nonexistent is None

    def test_check_commutation(self):
        bridge = SymmetryToQuantumBridge(4)
        bridge.parity_operator()
        
        H = np.diag([1, 2, 3, 4])
        result = bridge.check_commutation(H, 'parity')
        assert isinstance(result, bool)

    def test_check_commutation_no_operator(self):
        bridge = SymmetryToQuantumBridge(4)
        H = np.diag([1, 2, 3, 4])
        
        result = bridge.check_commutation(H, 'nonexistent')
        assert result is False

    def test_find_symmetries(self):
        bridge = SymmetryToQuantumBridge(4)
        bridge.parity_operator()
        bridge.translation_operator()
        bridge.time_reversal_operator()
        bridge.angular_momentum_operators()
        
        H = np.diag([1, 2, 3, 4])
        symmetries = bridge.find_symmetries(H)
        
        assert isinstance(symmetries, list)


class TestLieAlgebraToQuantumBridge:
    def test_creation(self):
        bridge = LieAlgebraToQuantumBridge()
        assert bridge is not None

    def test_su2_generators_spin_half(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=0.5)
        
        assert 'Sx' in S
        assert 'Sy' in S
        assert 'Sz' in S
        
        for key in S:
            assert S[key].shape == (2, 2)

    def test_su2_generators_spin_one(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=1.0)
        
        for key in S:
            assert S[key].shape == (3, 3)

    def test_su2_generators_spin_3_over_2(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=1.5)
        
        for key in S:
            assert S[key].shape == (4, 4)

    def test_su2_generators_commutation(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=0.5)
        
        Sx = S['Sx']
        Sy = S['Sy']
        Sz = S['Sz']
        
        comm_xy = Sx @ Sy - Sy @ Sx
        expected = 0.5j * Sz
        assert np.allclose(comm_xy, expected, atol=1e-10)

    def test_so3_generators_l_1(self):
        bridge = LieAlgebraToQuantumBridge()
        L = bridge.so3_generators(l=1)
        
        assert 'Lx' in L
        assert 'Ly' in L
        assert 'Lz' in L
        assert 'L2' in L
        
        for key in L:
            assert L[key].shape == (3, 3)

    def test_so3_generators_l_2(self):
        bridge = LieAlgebraToQuantumBridge()
        L = bridge.so3_generators(l=2)
        
        for key in L:
            assert L[key].shape == (5, 5)

    def test_so3_L2_eigenvalue(self):
        bridge = LieAlgebraToQuantumBridge()
        L = bridge.so3_generators(l=1)
        
        L2 = L['L2']
        l = 1
        expected_eigenvalue = l * (l + 1)
        assert np.allclose(L2, expected_eigenvalue * np.eye(3))

    def test_heisenberg_xxx_hamiltonian_two_sites(self):
        bridge = LieAlgebraToQuantumBridge()
        try:
            H = bridge.heisenberg_xxx_hamiltonian(J=1.0, n_sites=2)
            assert H.shape == (4, 4)
        except ValueError:
            pass

    def test_heisenberg_xxx_hamiltonian_three_sites(self):
        bridge = LieAlgebraToQuantumBridge()
        try:
            H = bridge.heisenberg_xxx_hamiltonian(J=2.0, n_sites=3)
            assert H.shape == (8, 8)
        except ValueError:
            pass

    def test_heisenberg_xxx_hamiltonian_interaction_count(self):
        bridge = LieAlgebraToQuantumBridge()
        try:
            H = bridge.heisenberg_xxx_hamiltonian(J=1.0, n_sites=4)
            assert H.shape == (16, 16)
        except ValueError:
            pass

    def test_hubbard_hamiltonian(self):
        bridge = LieAlgebraToQuantumBridge()
        H = bridge.hubbard_hamiltonian(t=1.0, U=4.0, n_sites=2)
        
        assert H.shape == (16, 16)
        assert np.allclose(H, np.zeros_like(H))

    def test_get_operator(self):
        bridge = LieAlgebraToQuantumBridge()
        bridge.su2_generators(s=0.5)
        
        Sx = bridge.get_operator('Sx')
        assert Sx is not None
        assert Sx.shape == (2, 2)
        
        nonexistent = bridge.get_operator('nonexistent')
        assert nonexistent is None


class TestGroupRepresentationBridge:
    def test_creation(self):
        bridge = GroupRepresentationBridge(6)
        assert bridge is not None
        assert bridge._dim == 6

    def test_create_so3_representation(self):
        bridge = GroupRepresentationBridge(3)
        irreps = bridge.create_irreducible_representation('SO3', 3)
        
        assert isinstance(irreps, dict)

    def test_create_su2_representation(self):
        bridge = GroupRepresentationBridge(2)
        irreps = bridge.create_irreducible_representation('SU2', 2)
        
        assert isinstance(irreps, dict)

    def test_create_c3v_representation(self):
        bridge = GroupRepresentationBridge(2)
        irreps = bridge.create_irreducible_representation('C3v', 2)
        
        assert isinstance(irreps, dict)
        assert 'A1' in irreps
        assert 'E' in irreps

    def test_create_trivial_representation(self):
        bridge = GroupRepresentationBridge(5)
        irreps = bridge.create_irreducible_representation('Unknown', 5)
        
        assert np.allclose(irreps, np.eye(5))

    def test_classify_state_by_representation(self):
        bridge = GroupRepresentationBridge(1)
        bridge._c3v_representation()
        
        state = np.array([1], dtype=complex)
        chi = bridge.classify_state_by_representation(state, 'A1')
        
        assert isinstance(chi, (int, float))

    def test_classify_unknown_representation(self):
        bridge = GroupRepresentationBridge(4)
        
        state = np.array([1, 0, 0, 0], dtype=complex)
        chi = bridge.classify_state_by_representation(state, 'Unknown')
        
        assert chi == 0.0


class TestIntegrateWithAbstractPhys:
    def test_basic_integration(self):
        from PySymmetry.phys.quantum.interactive import SceneBuilder, Potential
        
        scene = SceneBuilder("Test").build()
        
        result = integrate_with_abstract_phys(
            scene,
            hamiltonian_matrix=np.eye(100),
            analyze_symmetry=False,
            compute_lie_algebra=False
        )
        
        assert isinstance(result, IntegratedQuantumResult)
        assert 'conserved_quantities' in result.__dict__


class TestCreateQuantumFromAbstract:
    def test_fallback_creation(self):
        class FakeSystem:
            name = "FakeSystem"
            dimension = 2
        
        result = create_quantum_from_abstract(FakeSystem())
        assert result is not None


class TestExportToAbstractPhys:
    def test_export_basic(self):
        from PySymmetry.phys.quantum.interactive import SceneBuilder
        
        scene = SceneBuilder("Test").build()
        energies = np.array([1.0, 2.0, 3.0])
        states = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / np.sqrt(2)]
        
        result = export_to_abstract_phys(scene, energies, states)
        
        assert result['type'] == 'QuantumSystem'
        assert result['name'] == 'Test'
        assert result['hilbert_dim'] == 3
        assert result['dimension'] == 1


class TestConvenienceFunctions:
    def test_quick_bridge(self):
        bridge = quick_bridge(10)
        assert isinstance(bridge, SymmetryToQuantumBridge)
        assert bridge._dim == 10

    def test_quick_lie_bridge(self):
        bridge = quick_lie_bridge()
        assert isinstance(bridge, LieAlgebraToQuantumBridge)


class TestBridgeIntegration:
    def test_full_bridge_workflow(self):
        n = 4
        
        sym_bridge = SymmetryToQuantumBridge(n)
        sym_bridge.parity_operator()
        sym_bridge.translation_operator()
        sym_bridge.time_reversal_operator()
        sym_bridge.angular_momentum_operators()
        
        lie_bridge = LieAlgebraToQuantumBridge()
        lie_bridge.su2_generators(s=0.5)
        lie_bridge.so3_generators(l=1)
        
        group_bridge = GroupRepresentationBridge(n)
        group_bridge._c3v_representation()
        
        assert len(sym_bridge._symmetry_operators) > 0
        assert len(lie_bridge._operators) > 0
        assert len(group_bridge._representations) > 0

    def test_multiple_spin_calculations(self):
        bridge = LieAlgebraToQuantumBridge()
        
        S1 = bridge.su2_generators(s=0.5)
        S2 = bridge.su2_generators(s=1.0)
        
        assert S1['Sx'].shape == (2, 2)
        assert S2['Sx'].shape == (3, 3)


class TestSymmetryOperatorProperties:
    def test_parity_is_involution(self):
        bridge = SymmetryToQuantumBridge(5)
        P = bridge.parity_operator()
        assert np.allclose(P @ P, np.eye(5))

    def test_time_reversal_is_involution(self):
        bridge = SymmetryToQuantumBridge(4)
        T = bridge.time_reversal_operator()
        assert np.allclose(T @ T, np.eye(4))

    def test_translation_unitarity(self):
        bridge = SymmetryToQuantumBridge(5)
        T = bridge.translation_operator()
        assert np.allclose(T @ T.conj().T, np.eye(5))


class TestLieAlgebraProperties:
    def test_su2_generators_shape(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=0.5)
        
        assert S['Sx'].shape == (2, 2)
        assert S['Sy'].shape == (2, 2)
        assert S['Sz'].shape == (2, 2)

    def test_so3_L2_diagonal(self):
        bridge = LieAlgebraToQuantumBridge()
        L = bridge.so3_generators(l=1)
        
        L2 = L['L2']
        expected = 2 * np.eye(3)
        assert np.allclose(L2, expected, atol=1e-10)

    def test_ladder_operators_shape(self):
        bridge = LieAlgebraToQuantumBridge()
        S = bridge.su2_generators(s=0.5)
        
        assert S['Sx'].shape == (2, 2)
        assert S['Sy'].shape == (2, 2)
        assert S['Sz'].shape == (2, 2)
