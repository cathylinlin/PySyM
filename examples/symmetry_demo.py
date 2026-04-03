"""
Quantum Symmetry Analysis Demo

Demonstrates symmetry analysis capabilities integrated with abstract_phys:
1. Uses abstract_phys SymmetryOperation for parity analysis
2. Integrates with SymmetryAnalyzer from abstract_phys
3. State classification with irreducible representations
4. Selection rules using group representation theory
5. Conservation laws via Noether's theorem

Run with:
    python examples/symmetry_demo.py
"""
import numpy as np
from PySymmetry.phys.quantum import *


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_harmonic_oscillator_symmetry():
    """Demonstrate symmetry in harmonic oscillator"""
    print_header("1. Harmonic Oscillator Symmetry")
    
    scene = (SceneBuilder("HO Symmetry Demo")
             .add_electron(position=[0])
             .add_harmonic_potential(center=[0], k=1.0)
             .set_spatial_range(-10, 10)
             .set_grid_points(300)
             .build())
    
    result = simulate(scene, num_states=8)
    
    analyzer = QuantumSymmetryAnalyzer(result.hamiltonian, result)
    
    print("\nDetected Symmetries:")
    symmetries = analyzer.detect_symmetries()
    for sym in symmetries:
        print(f"  [{sym['type'].upper()}] {sym['name']}")
        print(f"      {sym['description']}")
        if 'conserved_quantity' in sym:
            print(f"      Conserved: {sym['conserved_quantity']}")
    
    print("\nState Parity Analysis:")
    print("-" * 50)
    for i in range(min(8, len(result.states))):
        psi = result.states[i]
        if psi is not None:
            parity, desc = analyzer.analyze_parity(psi)
            print(f"  State {i}: parity = {parity:+.0f} ({desc})")
    
    print("\nConservation Laws:")
    conserved = analyzer.compute_conservation_laws()
    for name, info in conserved.items():
        print(f"  {name}: {info['eigenvalues']}")
    
    return analyzer


def demo_square_well_symmetry():
    """Demonstrate symmetry in infinite square well"""
    print_header("2. Infinite Square Well Symmetry")
    
    def infinite_well(x):
        L = 5.0
        if 0 < x[0] < L:
            return 0.0
        return 1e10
    
    scene = (SceneBuilder("Square Well Symmetry")
             .add_electron(position=[2.5])
             .add_custom_potential(infinite_well, "Infinite Well")
             .set_spatial_range(0, 5)
             .set_grid_points(200)
             .build())
    
    result = simulate(scene, num_states=6)
    
    analyzer = QuantumSymmetryAnalyzer(result.hamiltonian, result)
    symmetries = analyzer.detect_symmetries()
    
    print("\nDetected Symmetries:")
    for sym in symmetries:
        print(f"  [{sym['type'].upper()}] {sym['name']}")
        print(f"      {sym['description']}")
    
    print("\nEnergy Levels with Parity:")
    print("-" * 50)
    classifications = analyzer.classify_states()
    
    for cls in classifications[:6]:
        p = f"{cls['parity']:+.0f}" if isinstance(cls['parity'], (int, float)) else "?"
        print(f"  n={cls['index']}: E={cls['energy']:.6f}, parity={p}")
    
    print("\nDegenerate States:")
    degenerate = analyzer._find_degenerate_states()
    if degenerate:
        for group in degenerate:
            if len(group) > 1:
                print(f"  Degenerate set: states {group}")
    else:
        print("  None (all states non-degenerate)")
    
    return analyzer


def demo_double_well_tunneling():
    """Demonstrate symmetry in double well (tunneling)"""
    print_header("3. Double Well - Tunneling Symmetry")
    
    def double_well(x):
        V0 = 5.0
        a = 1.0
        d = 3.0
        return -V0 * np.exp(-a * (x[0] + d)**2) - V0 * np.exp(-a * (x[0] - d)**2)
    
    scene = (SceneBuilder("Double Well Tunneling")
             .add_electron(position=[0])
             .add_custom_potential(double_well, "Double Well")
             .set_spatial_range(-8, 8)
             .set_grid_points(400)
             .build())
    
    result = simulate(scene, num_states=6)
    
    analyzer = QuantumSymmetryAnalyzer(result.hamiltonian, result)
    
    print("\nEnergy Spectrum:")
    for i in range(min(6, len(result.energies))):
        print(f"  State {i}: E = {result.energies[i]:.6f}")
    
    print("\nParity of States:")
    for i in range(min(6, len(result.states))):
        psi = result.states[i]
        if psi is not None:
            parity, desc = analyzer.analyze_parity(psi)
            print(f"  State {i}: parity = {parity:+.0f} ({desc})")
    
    print("\nGround State Pair Analysis:")
    if len(result.energies) >= 2:
        splitting = result.energies[1] - result.energies[0]
        print(f"  Energy splitting: {splitting:.6f}")
        print(f"  -> States 0 and 1 are nearly degenerate")
        print(f"  -> This indicates tunneling between the two wells")
        
        p0, _ = analyzer.analyze_parity(result.states[0])
        p1, _ = analyzer.analyze_parity(result.states[1])
        print(f"  -> Parities: state 0 = {p0:+.0f}, state 1 = {p1:+.0f}")
        print(f"  -> Symmetric/antisymmetric pair formed by tunneling")
    
    return analyzer


def demo_selection_rules():
    """Demonstrate selection rules"""
    print_header("4. Selection Rules")
    
    scene = (SceneBuilder("Selection Rules Demo")
             .add_electron(position=[0])
             .add_harmonic_potential(center=[0], k=1.0)
             .set_spatial_range(-10, 10)
             .set_grid_points(300)
             .build())
    
    result = simulate(scene, num_states=6)
    
    analyzer = QuantumSymmetryAnalyzer(result.hamiltonian, result)
    
    print("\nElectric Dipole Selection Rules:")
    print("  Rule: parity must change (initial_parity * final_parity < 0)")
    
    rules = analyzer.compute_selection_rules(operator='dipole')
    allowed = [r for r in rules if r['allowed']]
    
    print(f"\nAllowed Transitions (based on parity):")
    for rule in allowed[:8]:
        print(f"  |{rule['initial']}> -> |{rule['final']}>: Delta E = {rule['energy_gap']:.4f}")
    
    forbidden = [r for r in rules if not r['allowed']]
    print(f"\nForbidden Transitions (same parity):")
    for rule in forbidden[:5]:
        print(f"  |{rule['initial']}> -> |{rule['final']}>: parity {rule['initial_parity']:+.0f} -> {rule['final_parity']:+.0f}")
    
    return analyzer


def demo_abstract_integration():
    """Demonstrate integration with abstract_phys"""
    print_header("5. Integration with abstract_phys")
    
    scene = (SceneBuilder("Abstract Integration Demo")
             .add_electron(position=[2.5])
             .add_custom_potential(lambda x: 0.0 if 0 < x[0] < 5 else 1e10, "Infinite Well")
             .set_spatial_range(0, 5)
             .set_grid_points(200)
             .build())
    
    result = simulate(scene, num_states=8)
    
    analyzer = QuantumSymmetryAnalyzer(result.hamiltonian, result)
    
    print("\nFull Symmetry Analysis Report:")
    print(analyzer.generate_report())
    
    print("\nStructured Result (for programmatic use):")
    structured = analyzer.to_abstract_result()
    print(f"  Detected symmetries: {len(structured.detected_symmetries)}")
    print(f"  Conserved quantities: {len(structured.conserved_quantities)}")
    print(f"  State classifications: {len(structured.state_classifications)}")
    print(f"  Selection rules: {len(structured.selection_rules)}")
    print(f"  Invariants: {structured.invariants}")
    
    return analyzer


def run_all():
    """Run all demos"""
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  Quantum Symmetry Analysis Demo  ".center(68) + "#")
    print("#" + "  (Integrated with abstract_phys)  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    demos = [
        ("Harmonic Oscillator", demo_harmonic_oscillator_symmetry),
        ("Square Well", demo_square_well_symmetry),
        ("Double Well Tunneling", demo_double_well_tunneling),
        ("Selection Rules", demo_selection_rules),
        ("Abstract Integration", demo_abstract_integration),
    ]
    
    results = []
    for name, func in demos:
        try:
            analyzer = func()
            results.append((name, True, analyzer))
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
    print("Symmetry Analysis Results:")
    for name, success, _ in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name:<25} {status}")
    
    print()
    print("Symmetry analysis complete!")
    print()


if __name__ == "__main__":
    run_all()
