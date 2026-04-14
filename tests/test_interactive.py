import pytest
import numpy as np
from PySymmetry.phys.quantum.interactive import (
    Particle, Potential, QuantumScene, SceneBuilder,
    SceneHamiltonianBuilder, SceneSymmetryAnalyzer,
    simulate, SimulationResult, Visualizer,
    analyze_scene_symmetry, quick_simulate
)


class TestParticle:
    def test_particle_creation(self):
        p = Particle(
            name="electron",
            mass=1.0,
            charge=-1.0,
            spin=0.5,
            position=[0.0, 0.0, 0.0]
        )
        assert p.name == "electron"
        assert p.mass == 1.0
        assert p.charge == -1.0
        assert p.spin == 0.5
        assert isinstance(p.position, np.ndarray)
        assert len(p.position) == 3

    def test_particle_momentum(self):
        p = Particle(
            name="proton",
            mass=1836.0,
            charge=1.0,
            spin=0.5,
            position=[1.0, 0.0, 0.0],
            momentum=[0.0, 1.0, 0.0]
        )
        assert p.momentum is not None
        assert isinstance(p.momentum, np.ndarray)

    def test_particle_position_conversion(self):
        p = Particle(
            name="test",
            mass=1.0,
            charge=0.0,
            spin=0.0,
            position=np.array([1.0, 2.0, 3.0])
        )
        assert isinstance(p.position, np.ndarray)


class TestPotential:
    def test_coulomb_potential(self):
        pot = Potential.coulomb(center=np.array([0.0, 0.0, 0.0]), strength=1.0, Z=1.0)
        assert pot.name == "Coulomb"
        assert pot.potential_type == "coulomb"
        assert pot.dimension == 3
        
        V = pot.evaluate(np.array([1.0, 0.0, 0.0]))
        assert V == pytest.approx(-1.0)

    def test_coulomb_potential_at_center(self):
        pot = Potential.coulomb(center=np.array([0.0, 0.0, 0.0]), strength=1.0)
        V = pot.evaluate(np.array([0.0, 0.0, 0.0]))
        assert abs(V) > 1e5

    def test_harmonic_potential(self):
        pot = Potential.harmonic(center=np.array([0.0, 0.0, 0.0]), k=1.0)
        assert pot.name == "Harmonic"
        assert pot.potential_type == "harmonic"
        
        V = pot.evaluate(np.array([1.0, 0.0, 0.0]))
        assert V == pytest.approx(0.5)

    def test_square_well(self):
        pot = Potential.square_well(center=np.array([0.0, 0.0, 0.0]), radius=1.0, depth=2.0)
        assert pot.name == "SquareWell"
        
        V_inside = pot.evaluate(np.array([0.5, 0.0, 0.0]))
        assert V_inside == pytest.approx(-2.0)
        
        V_outside = pot.evaluate(np.array([2.0, 0.0, 0.0]))
        assert V_outside == pytest.approx(0.0)

    def test_harmonic_3d_isotropic(self):
        pot = Potential.harmonic_3d()
        assert pot.name == "Harmonic3D"
        
        V = pot.evaluate(np.array([1.0, 0.0, 0.0]))
        assert V == pytest.approx(0.5)

    def test_harmonic_3d_anisotropic(self):
        pot = Potential.harmonic_3d(kx=1.0, ky=2.0, kz=3.0)
        V = pot.evaluate(np.array([1.0, 1.0, 1.0]))
        expected = 0.5 * (1.0 + 2.0 + 3.0)
        assert V == pytest.approx(expected)

    def test_spherical_well(self):
        pot = Potential.spherical_well(radius=2.0, depth=3.0)
        assert pot.name == "SphericalWell"
        
        V = pot.evaluate(np.array([0.0, 0.0, 0.0]))
        assert V == pytest.approx(-3.0)

    def test_gaussian_well(self):
        pot = Potential.gaussian_well(center=np.array([0.0, 0.0, 0.0]), depth=1.0, width=1.0)
        assert pot.name == "GaussianWell"
        
        V_center = pot.evaluate(np.array([0.0, 0.0, 0.0]))
        assert V_center == pytest.approx(-1.0)

    def test_cylindrical_potential_z(self):
        pot = Potential.cylindrical_potential(axis='z', radius=1.0, depth=2.0)
        assert pot.name == "Cylindrical"
        
        V_inside = pot.evaluate(np.array([0.0, 0.0, 0.0]))
        assert V_inside == pytest.approx(-2.0)

    def test_step_potential(self):
        pot = Potential.step(position=0.0, height=1.0)
        assert pot.name == "Step"
        
        V_positive = pot.evaluate(np.array([1.0]))
        assert V_positive == pytest.approx(1.0)
        
        V_negative = pot.evaluate(np.array([-1.0]))
        assert V_negative == pytest.approx(0.0)

    def test_custom_potential(self):
        def my_potential(x):
            return x[0]**2
        
        pot = Potential.custom(my_potential, name="MyCustom")
        assert pot.name == "MyCustom"
        assert pot.evaluate(np.array([2.0])) == pytest.approx(4.0)


class TestQuantumScene:
    def test_scene_creation(self):
        scene = QuantumScene(name="TestScene")
        assert scene.name == "TestScene"
        assert len(scene.particles) == 0
        assert len(scene.potentials) == 0

    def test_scene_with_particles(self):
        p = Particle("e", 1.0, -1.0, 0.5, [0.0, 0.0, 0.0])
        scene = QuantumScene(name="Test", particles=[p])
        assert len(scene.particles) == 1

    def test_has_symmetry(self):
        scene = QuantumScene(name="Test")
        scene.symmetry_info = {'detected': ['parity', 'translation']}
        assert scene.has_symmetry('parity')
        assert not scene.has_symmetry('rotational')

    def test_get_conserved_quantities(self):
        scene = QuantumScene(name="Test")
        scene.symmetry_info = {'conserved': ['energy', 'parity']}
        conserved = scene.get_conserved_quantities()
        assert 'energy' in conserved
        assert 'parity' in conserved

    def test_get_grid_3d(self):
        scene = QuantumScene(name="Test", dimension=3)
        gx, gy, gz = scene.get_grid_3d()
        assert len(gx) == 50
        assert len(gy) == 50
        assert len(gz) == 50


class TestSceneBuilder:
    def test_builder_empty(self):
        builder = SceneBuilder("Empty")
        scene = builder.build()
        assert scene.name == "Empty"
        assert len(scene.particles) == 0
        assert len(scene.potentials) == 0

    def test_add_particle(self):
        builder = SceneBuilder()
        builder.add_particle("muon", mass=200.0, charge=-1.0, spin=0.5, position=[0.0])
        scene = builder.build()
        assert len(scene.particles) == 1
        assert scene.particles[0].name == "muon"

    def test_add_electron(self):
        builder = SceneBuilder()
        builder.add_electron(position=[0.0, 0.0, 0.0])
        scene = builder.build()
        assert len(scene.particles) == 1
        assert scene.particles[0].mass == 1.0
        assert scene.particles[0].charge == -1.0

    def test_add_proton(self):
        builder = SceneBuilder()
        builder.add_proton(position=[0.0, 0.0, 0.0])
        scene = builder.build()
        assert len(scene.particles) == 1
        assert scene.particles[0].mass == 1836.0
        assert scene.particles[0].charge == 1.0

    def test_add_neutron(self):
        builder = SceneBuilder()
        builder.add_neutron()
        scene = builder.build()
        assert len(scene.particles) == 1
        assert scene.particles[0].charge == 0.0

    def test_add_coulomb_potential(self):
        builder = SceneBuilder()
        builder.add_coulomb_potential(center=[0.0, 0.0, 0.0], strength=1.0, Z=1.0)
        scene = builder.build()
        assert len(scene.potentials) == 1

    def test_add_harmonic_potential(self):
        builder = SceneBuilder()
        builder.add_harmonic_potential(center=[0.0], k=1.0)
        scene = builder.build()
        assert len(scene.potentials) == 1

    def test_add_square_well(self):
        builder = SceneBuilder()
        builder.add_square_well(center=[0.0], radius=1.0, depth=2.0)
        scene = builder.build()
        assert len(scene.potentials) == 1

    def test_add_harmonic_3d(self):
        builder = SceneBuilder()
        builder.add_harmonic_3d(kx=1.0, ky=2.0, kz=3.0)
        scene = builder.build()
        assert scene.dimension == 3

    def test_add_spherical_well(self):
        builder = SceneBuilder()
        builder.add_spherical_well(radius=2.0, depth=3.0)
        scene = builder.build()
        assert scene.dimension == 3

    def test_add_gaussian_well(self):
        builder = SceneBuilder()
        builder.add_gaussian_well(center=[0.0, 0.0, 0.0], depth=1.0, width=0.5)
        scene = builder.build()
        assert scene.dimension == 3

    def test_add_custom_potential(self):
        builder = SceneBuilder()
        builder.add_custom_potential(lambda x: x[0]**2, name="Quadratic")
        scene = builder.build()
        assert len(scene.potentials) == 1

    def test_set_spatial_range(self):
        builder = SceneBuilder()
        builder.set_spatial_range(-5.0, 5.0)
        scene = builder.build()
        assert scene.spatial_range == (-5.0, 5.0)

    def test_set_spatial_range_3d(self):
        builder = SceneBuilder()
        builder.set_spatial_range_3d(x_range=(-1, 1), y_range=(-2, 2))
        scene = builder.build()
        assert scene.dimension == 3

    def test_set_dimension(self):
        builder = SceneBuilder()
        builder.set_dimension(2)
        scene = builder.build()
        assert scene.dimension == 2

    def test_set_grid_points(self):
        builder = SceneBuilder()
        builder.set_grid_points(200)
        scene = builder.build()
        assert scene.grid_points == 200

    def test_set_grid_points_3d(self):
        builder = SceneBuilder()
        builder.set_grid_points_3d(nx=20, ny=30, nz=40)
        scene = builder.build()
        assert scene.grid_points_3d == [20, 30, 40]
        assert scene.dimension == 3

    def test_set_boundary_condition(self):
        builder = SceneBuilder()
        builder.set_boundary_condition('periodic')
        scene = builder.build()
        assert scene.boundary_condition == 'periodic'

    def test_enable_spin_coupling(self):
        builder = SceneBuilder()
        builder.enable_spin_coupling(True)
        scene = builder.build()
        assert scene.spin_coupling is True

    def test_set_external_field(self):
        builder = SceneBuilder()
        builder.set_external_field('magnetic', B=1.0)
        scene = builder.build()
        assert scene.external_field['type'] == 'magnetic'
        assert scene.external_field['params']['B'] == 1.0

    def test_chain_builder(self):
        scene = (SceneBuilder("Chain")
                 .add_electron(position=[0.0])
                 .add_coulomb_potential(center=[0.0], strength=1.0)
                 .set_spatial_range(-5, 5)
                 .set_grid_points(100)
                 .set_boundary_condition('infinite')
                 .build())
        
        assert scene.name == "Chain"
        assert len(scene.particles) == 1
        assert len(scene.potentials) == 1
        assert scene.grid_points == 100

    def test_summary(self):
        builder = SceneBuilder("TestScene")
        builder.add_electron(position=[0.0])
        builder.add_harmonic_potential(center=[0.0], k=1.0)
        summary = builder.summary()
        assert "TestScene" in summary
        assert "Particles: 1" in summary
        assert "Harmonic" in summary


class TestSceneHamiltonianBuilder:
    def test_builder_creation(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=50)
        builder = SceneHamiltonianBuilder(scene)
        assert builder.dimension == 1
        assert len(builder.grid) == 50

    def test_build_kinetic_term_1d(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=10)
        builder = SceneHamiltonianBuilder(scene)
        T = builder.build_kinetic_term()
        assert T.shape == (10, 10)
        assert np.allclose(T, T.T)

    def test_build_kinetic_term_3d(self):
        scene = QuantumScene(name="Test", dimension=3, grid_points=3, grid_points_3d=[3, 3, 3])
        builder = SceneHamiltonianBuilder(scene)
        T = builder.build_kinetic_term()
        assert T.shape[0] == 27
        assert T.shape[1] == 27

    def test_build_potential_term_1d(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=10, potentials=[pot])
        builder = SceneHamiltonianBuilder(scene)
        V = builder.build_potential_term()
        assert V.shape == (10, 10)
        assert np.allclose(V, V.T)
        assert np.all(np.diag(V) >= 0)

    def test_build_potential_term_3d(self):
        pot = Potential.spherical_well(radius=2.0, depth=1.0)
        scene = QuantumScene(name="Test", dimension=3, grid_points_3d=[5, 5, 5], potentials=[pot])
        builder = SceneHamiltonianBuilder(scene)
        V = builder.build_potential_term()
        assert V.shape == (125, 125)

    def test_build_full_hamiltonian(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=20, potentials=[pot])
        builder = SceneHamiltonianBuilder(scene)
        H = builder.build()
        assert H.shape == (20, 20)
        assert np.allclose(H, H.T)

    def test_num_states(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=50)
        builder = SceneHamiltonianBuilder(scene)
        assert builder.num_states == 50


class TestSceneSymmetryAnalyzer:
    def test_analyzer_creation(self):
        scene = QuantumScene(name="Test", dimension=1)
        analyzer = SceneSymmetryAnalyzer(scene)
        assert analyzer is not None

    def test_analyze_empty_potential(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=20)
        analyzer = SceneSymmetryAnalyzer(scene)
        result = analyzer.analyze()
        assert 'detected' in result
        assert 'conserved' in result
        assert 'energy' in result['conserved']

    def test_parity_symmetry_harmonic(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=21, 
                           spatial_range=(-5.0, 5.0), potentials=[pot])
        analyzer = SceneSymmetryAnalyzer(scene)
        result = analyzer.analyze()
        assert 'parity' in result['detected']
        assert 'parity' in result['conserved']

    def test_translation_symmetry_periodic(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=20, 
                           boundary_condition='periodic')
        analyzer = SceneSymmetryAnalyzer(scene)
        result = analyzer.analyze()
        assert 'translation' in result['detected']

    def test_spherical_symmetry_coulomb(self):
        pot = Potential.coulomb(center=np.array([0.0, 0.0, 0.0]), strength=1.0)
        scene = QuantumScene(name="Test", dimension=3, grid_points_3d=[5, 5, 5], 
                           spatial_range_3d=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
                           potentials=[pot])
        analyzer = SceneSymmetryAnalyzer(scene)
        result = analyzer.analyze()
        assert 'spherical' in result['detected']
        assert 'angular_momentum' in result['conserved']

    def test_cylindrical_symmetry(self):
        pot = Potential.cylindrical_potential(axis='z', radius=1.0, depth=2.0)
        scene = QuantumScene(name="Test", dimension=3, grid_points_3d=[5, 5, 5],
                           spatial_range_3d=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
                           potentials=[pot])
        analyzer = SceneSymmetryAnalyzer(scene)
        result = analyzer.analyze()
        assert 'cylindrical' in result['detected']

    def test_generate_description(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=20)
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene.symmetry_info = {'detected': ['parity'], 'conserved': ['parity', 'energy']}
        analyzer = SceneSymmetryAnalyzer(scene)
        desc = analyzer._generate_description()
        assert len(desc) > 0


class TestAnalyzeSceneSymmetry:
    def test_convenience_function(self):
        scene = QuantumScene(name="Test", dimension=1, grid_points=20)
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        result = analyze_scene_symmetry(scene)
        assert 'detected' in result
        assert 'conserved' in result


class TestSimulate:
    def test_simulate_harmonic_oscillator(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = (SceneBuilder("HarmonicTest")
                .add_electron(position=[0.0])
                .set_spatial_range(-5, 5)
                .set_grid_points(50)
                .build())
        scene.potentials = [pot]
        
        result = simulate(scene, num_states=3, analyze_symmetry=True)
        assert isinstance(result, SimulationResult)
        assert len(result.states) == 3
        assert len(result.energies) == 3
        assert result.energies[0] <= result.energies[1] <= result.energies[2]


class TestSimulationResult:
    def test_result_summary(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=100, potentials=[pot])
        result = SimulationResult(
            scene=scene,
            hamiltonian=None,
            states=[],
            energies=np.array([0.5, 1.5, 2.5]),
            grid=np.linspace(-5, 5, 100)
        )
        summary = result.summary()
        assert "Test" in summary
        assert "Energy levels" in summary

    def test_get_conserved_quantities(self):
        scene = QuantumScene(name="Test")
        scene.symmetry_info = {'conserved': ['energy', 'parity']}
        result = SimulationResult(
            scene=scene,
            hamiltonian=None,
            states=[],
            energies=np.array([]),
            grid=np.array([])
        )
        conserved = result.get_conserved_quantities()
        assert 'energy' in conserved
        assert 'parity' in conserved


class TestVisualizer:
    def test_visualizer_creation(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=100, potentials=[pot])
        result = SimulationResult(
            scene=scene,
            hamiltonian=None,
            states=[],
            energies=np.array([]),
            grid=np.linspace(-5, 5, 100)
        )
        viz = Visualizer(result)
        assert viz is not None

    def test_plot_potential_with_matplotlib(self):
        pot = Potential.harmonic(center=np.array([0.0]), k=1.0)
        scene = QuantumScene(name="Test", dimension=1, grid_points=100, potentials=[pot])
        result = SimulationResult(
            scene=scene,
            hamiltonian=None,
            states=[],
            energies=np.array([]),
            grid=np.linspace(-5, 5, 100)
        )
        viz = Visualizer(result)
        ax = viz.plot_potential()
        assert ax is not None


class TestQuickSimulate:
    def test_quick_simulate_basic(self):
        result = quick_simulate(
            particles=[{'type': 'electron', 'position': [0.0]}],
            potentials=[{'type': 'harmonic', 'center': [0.0], 'k': 1.0}],
            x_range=(-5, 5),
            n_points=50,
            num_states=3
        )
        assert isinstance(result, SimulationResult)
        assert len(result.energies) == 3

    def test_quick_simulate_proton(self):
        result = quick_simulate(
            particles=[{'type': 'proton', 'position': [0.0]}],
            n_points=30
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_neutron(self):
        result = quick_simulate(
            particles=[{'type': 'neutron'}],
            n_points=30
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_custom_particle(self):
        result = quick_simulate(
            particles=[{'type': 'alpha', 'mass': 4.0, 'charge': 2.0, 'position': [0.0]}],
            n_points=30
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_coulomb(self):
        result = quick_simulate(
            particles=[{'type': 'electron', 'position': [0.0, 0.0, 0.0]}],
            potentials=[{'type': 'coulomb', 'center': [0.0, 0.0, 0.0], 'strength': 1.0}],
            n_points=30
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_square_well(self):
        result = quick_simulate(
            potentials=[{'type': 'square_well', 'center': [0.0], 'radius': 1.0, 'depth': 2.0}],
            n_points=50
        )
        assert isinstance(result, SimulationResult)

    def test_quick_simulate_custom_potential(self):
        result = quick_simulate(
            potentials=[{'type': 'custom', 'func': lambda x: x[0]**2, 'name': 'Quadratic'}],
            n_points=50
        )
        assert isinstance(result, SimulationResult)


class TestMultipleParticles:
    def test_two_electrons(self):
        scene = (SceneBuilder("TwoElectrons")
                .add_electron(position=[-1.0])
                .add_electron(position=[1.0])
                .set_spatial_range(-5, 5)
                .set_grid_points(50)
                .build())
        assert len(scene.particles) == 2

    def test_hydrogen_atom(self):
        scene = (SceneBuilder("Hydrogen")
                .add_electron(position=[0.0, 0.0, 0.0])
                .add_proton(position=[0.0, 0.0, 0.0])
                .add_coulomb_potential(center=[0.0, 0.0, 0.0], strength=1.0, Z=1.0)
                .set_spatial_range(-10, 10)
                .set_grid_points(100)
                .build())
        assert len(scene.particles) == 2
        assert len(scene.potentials) == 1
