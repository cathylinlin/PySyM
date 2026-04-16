"""
Universal Quantum Simulator Demo

Demonstrates how to use the interactive quantum simulator to create arbitrary scenarios.

Run with:
    python examples/interactive_demo.py
"""

import numpy as np
from PySymmetry.phys.quantum import *


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_harmonic_oscillator():
    """Harmonic Oscillator Example"""
    print_header("1. Quantum Harmonic Oscillator")

    print("""
    Harmonic potential: V(x) = 1/2 * m * omega^2 * x^2
    
    Features:
    - Equally spaced energy levels: E_n = (n + 1/2) * omega
    - Ground state is a Gaussian wave packet
    - All excited states have n nodes
    """)

    scene = (
        SceneBuilder("Harmonic Oscillator")
        .add_electron(position=[0])
        .add_harmonic_potential(center=[0], k=1.0)
        .set_spatial_range(-10, 10)
        .set_grid_points(300)
        .build()
    )

    print(f"Scene: {scene.name}")
    print(f"Particles: {len(scene.particles)}")
    print(f"Potential: Harmonic (k={1.0})")
    print()

    result = simulate(scene, num_states=5)

    print("Energy Levels:")
    print("-" * 40)
    print(f"{'n':<5} {'Energy':<15} {'Expected':<15}")
    print("-" * 40)

    for i in range(5):
        expected = i + 0.5
        actual = result.energies[i]
        error = abs(expected - actual)
        print(f"{i:<5} {actual:<15.6f} {expected:<15.6f} (error: {error:.2e})")

    return result


def example_infinite_square_well():
    """Infinite Square Well"""
    print_header("2. Infinite Square Well")

    print("""
    Potential: V = 0 for 0 < x < L, V = infinity otherwise
    
    Features:
    - Wave function zero at boundaries
    - Energy levels E_n ~ n^2
    - Wave functions sin(n*pi*x/L)
    """)

    def infinite_well(x):
        L = 5.0
        if 0 < x[0] < L:
            return 0.0
        return 1e10  # Infinite potential outside

    scene = (
        SceneBuilder("Infinite Square Well")
        .add_electron(position=[2.5])
        .add_custom_potential(infinite_well, "Infinite Well")
        .set_spatial_range(0, 5)
        .set_grid_points(200)
        .build()
    )

    print(f"Scene: {scene.name}")
    print(f"Width: L = 5.0")
    print()

    result = simulate(scene, num_states=5)

    print("Energy Levels:")
    print("-" * 40)
    L = 5.0
    print(f"{'n':<5} {'Numerical':<15} {'Analytical':<15}")
    print("-" * 40)

    for i in range(5):
        n = i + 1
        expected = n**2 * np.pi**2 / (2 * L**2)
        actual = result.energies[i]
        error = abs(expected - actual) / expected * 100
        print(f"{n:<5} {actual:<15.6f} {expected:<15.6f} ({error:.2f}%)")

    return result


def example_double_well():
    """Double Well"""
    print_header("3. Double Well (Tunneling)")

    print("""
    Potential: V(x) = -V0 * [exp(-a*(x+d)^2) + exp(-a*(x-d)^2)]
    
    Features:
    - Two symmetric wells
    - Nearly degenerate ground and first excited states (tunneling)
    - Symmetric/antisymmetric combinations
    """)

    def double_well(x):
        V0 = 5.0
        a = 1.0
        d = 3.0
        return -V0 * np.exp(-a * (x[0] + d) ** 2) - V0 * np.exp(-a * (x[0] - d) ** 2)

    scene = (
        SceneBuilder("Double Well")
        .add_electron(position=[0])
        .add_custom_potential(double_well, "Double Well")
        .set_spatial_range(-8, 8)
        .set_grid_points(400)
        .build()
    )

    print(f"Scene: {scene.name}")
    print(f"Potential: V0 = 5, separation 2d = 6")
    print()

    result = simulate(scene, num_states=6)

    print("Energy Levels (showing tunneling splitting):")
    print("-" * 40)
    for i in range(6):
        print(f"  State {i}: E = {result.energies[i]:.6f}")

    # Check for near-degeneracy
    print()
    if len(result.energies) >= 2:
        splitting = result.energies[1] - result.energies[0]
        print(f"Ground-First excited splitting: {splitting:.6f}")
        if splitting < 0.01:
            print("  -> Near-degenerate pair (tunneling!)")


def example_particle_interaction():
    """Multi-particle interaction"""
    print_header("4. Multi-particle System")

    print("""
    Simulate two particles interacting in a potential well.
    
    Hamiltonian includes:
    - Kinetic energy of each particle
    - External potential energy
    - Inter-particle interaction
    """)

    scene = (
        SceneBuilder("Two Particles")
        .add_electron(position=[-2])
        .add_electron(position=[2])
        .add_harmonic_potential(center=[0], k=0.5)
        .set_spatial_range(-8, 8)
        .set_grid_points(200)
        .build()
    )

    print(f"Scene: {scene.name}")
    print(f"Particles: 2 electrons")
    print(f"External potential: Harmonic (k=0.5)")
    print()

    result = simulate(scene, num_states=3)

    print("Energy Levels:")
    for i in range(3):
        print(f"  State {i}: E = {result.energies[i]:.6f}")


def example_custom_potential():
    """Custom potential"""
    print_header("5. Custom Potential")

    print("""
    Create arbitrary-shaped potential functions.
    
    Example: Asymmetric well with spikes
    """)

    def weird_potential(x):
        V = 0.5 * x[0] ** 2  # Quadratic
        V -= 3 * np.exp(-0.5 * x[0] ** 2)  # Gaussian dip
        V += 2 * np.sign(np.sin(x[0]))  # Oscillations
        return V

    scene = (
        SceneBuilder("Custom Potential")
        .add_electron(position=[0])
        .add_custom_potential(weird_potential, "Weird")
        .set_spatial_range(-6, 6)
        .set_grid_points(300)
        .build()
    )

    print(f"Scene: {scene.name}")
    print("Custom potential: V(x) = 0.5*x^2 - 3*exp(-0.5*x^2) + 2*sign(sin(x))")
    print()

    result = simulate(scene, num_states=5)

    print("Energy Levels:")
    for i in range(5):
        print(f"  State {i}: E = {result.energies[i]:.6f}")


def interactive_demo():
    """Interactive demo"""
    print_header("Interactive Simulator Usage Guide")

    print("""
    SceneBuilder API:
    ================
    
    # Basic usage
    scene = (SceneBuilder("My Scene")
             .add_electron(position=[0])          # Add electron
             .add_proton(position=[1])            # Add proton  
             .add_harmonic_potential(center=[0], k=1)  # Harmonic potential
             .set_spatial_range(-10, 10)          # Spatial range
             .set_grid_points(200)                # Grid points
             .build())
    
    result = simulate(scene)
    
    # Quick simulate
    result = quick_simulate(
        particles=[{'type': 'electron', 'position': [0]}],
        potentials=[{'type': 'harmonic', 'center': [0], 'k': 1}],
        x_range=(-10, 10),
        n_points=200
    )
    
    # Visualization (requires matplotlib)
    viz = Visualizer(result)
    viz.plot_all()
    
    # Get results
    psi = result.get_wavefunction(0)               # Ground state wave function
    prob = result.get_probability_density(0)       # Probability density
    exp_x = result.get_position_expectation(0)     # <x>
    """)

    print("\n" + "=" * 70)
    print("Running full demo...")
    print("=" * 70)


def run_all():
    """Run all examples"""
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  PySymmetry Universal Quantum Simulator  ".center(68) + "#")
    print("#" + "  Arbitrary Scenario Demo  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    examples = [
        ("Harmonic Oscillator", example_harmonic_oscillator),
        ("Infinite Square Well", example_infinite_square_well),
        ("Double Well", example_double_well),
        ("Multi-particle", example_particle_interaction),
        ("Custom Potential", example_custom_potential),
    ]

    results = []
    for name, func in examples:
        try:
            result = func()
            results.append((name, True, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False, None))

    print()
    print("#" * 70)
    print("#" + " Summary ".center(68) + "#")
    print("#" * 70)

    print()
    print("Scenario Test Results:")
    for name, success, _ in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name:<20} {status}")

    interactive_demo()

    print()
    print("All scenario simulations complete!")
    print()


if __name__ == "__main__":
    run_all()
