"""
Quantum Analysis Demo

Demonstrates the unified quantum analysis capabilities:
1. Symmetry detection
2. State classification
3. Selection rules
4. Conservation laws
5. Invariants

Run with:
    python examples/analysis_demo.py
"""
import numpy as np
from PySymmetry.phys.quantum import (
    SceneBuilder, simulate, SimulationResult,
    QuantumAnalyzer, analyze, quick_report,
    QuantumParityOperation, QuantumTranslationOperation,
    Ket
)
import numpy as np


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_harmonic_oscillator():
    """Harmonic oscillator symmetry analysis"""
    print_header("1. Harmonic Oscillator")
    
    scene = (SceneBuilder("HO")
             .add_electron(position=[0])
             .add_harmonic_potential(center=[0], k=1.0)
             .set_spatial_range(-10, 10)
             .set_grid_points(300)
             .build())
    
    result = simulate(scene, num_states=8)
    
    analyzer = QuantumAnalyzer(result.hamiltonian, result)
    
    print("\nDetected Symmetries:")
    for sym in analyzer.detect_symmetries():
        print(f"  - {sym.name}: {sym.description}")
        if sym.conserved_quantity:
            print(f"    Conserved: {sym.conserved_quantity}")
    
    print("\nState Parities:")
    for i in range(min(8, len(result.states))):
        parity, desc = analyzer.analyze_parity(result.states[i])
        print(f"  n={i}: parity = {parity:+.0f} ({desc})")
    
    print("\nConservation Laws:")
    for name, info in analyzer.compute_conserved_quantities().items():
        print(f"  {name}: {info['eigenvalues']}")


def demo_square_well():
    """Infinite square well analysis"""
    print_header("2. Infinite Square Well")
    
    def infinite_well(x):
        if 0 < x[0] < 5:
            return 0.0
        return 1e10
    
    scene = (SceneBuilder("Square Well")
             .add_electron(position=[2.5])
             .add_custom_potential(infinite_well, "Infinite Well")
             .set_spatial_range(0, 5)
             .set_grid_points(200)
             .build())
    
    result = simulate(scene, num_states=6)
    
    analyzer = QuantumAnalyzer(result.hamiltonian, result)
    
    print("\nState Classifications:")
    print(f"{'n':<4} {'E':<12} {'Parity':<8} {'Irrep':<6}")
    print("-" * 35)
    for cls in analyzer.classify_states()[:6]:
        print(f"{cls.index:<4} {cls.energy:<12.4f} {cls.parity:+.0f}       {cls.irrep:<6}")
    
    print("\nDegenerate States:", len(analyzer._find_degenerate_states()))
    print("Invariants:", analyzer.compute_invariants())


def demo_double_well():
    """Double well - tunneling analysis"""
    print_header("3. Double Well (Tunneling)")
    
    def double_well(x):
        V0, a, d = 5.0, 1.0, 3.0
        return -V0 * np.exp(-a * (x[0] + d)**2) - V0 * np.exp(-a * (x[0] - d)**2)
    
    scene = (SceneBuilder("Double Well")
             .add_electron(position=[0])
             .add_custom_potential(double_well, "Double Well")
             .set_spatial_range(-8, 8)
             .set_grid_points(400)
             .build())
    
    result = simulate(scene, num_states=4)
    
    analyzer = QuantumAnalyzer(result.hamiltonian, result)
    
    print("\nGround State Pair:")
    print(f"  E0 = {result.energies[0]:.6f}")
    print(f"  E1 = {result.energies[1]:.6f}")
    print(f"  Splitting = {result.energies[1] - result.energies[0]:.6f}")
    
    p0, _ = analyzer.analyze_parity(result.states[0])
    p1, _ = analyzer.analyze_parity(result.states[1])
    print(f"  Parities: {p0:+.0f}, {p1:+.0f}")
    print("  -> Symmetric/antisymmetric tunneling pair")


def demo_selection_rules():
    """Selection rules demonstration"""
    print_header("4. Selection Rules")
    
    scene = (SceneBuilder("Selection Rules")
             .add_electron(position=[0])
             .add_harmonic_potential(center=[0], k=1.0)
             .set_spatial_range(-10, 10)
             .set_grid_points(300)
             .build())
    
    result = simulate(scene, num_states=6)
    analyzer = QuantumAnalyzer(result.hamiltonian, result)
    
    rules = analyzer.compute_selection_rules()
    allowed = [r for r in rules if r.allowed]
    
    print(f"\nTotal transitions: {len(rules)}")
    print(f"  Allowed (parity changes): {len(allowed)}")
    print(f"  Forbidden (same parity): {len(rules) - len(allowed)}")
    
    print("\nSample Allowed Transitions:")
    for r in allowed[:5]:
        print(f"  |{r.initial}> -> |{r.final}>: dE = {r.energy_gap:.4f}")


def demo_full_analysis():
    """Full analysis with report"""
    print_header("5. Full Analysis Report")
    
    scene = (SceneBuilder("Full Analysis")
             .add_electron(position=[2.5])
             .add_custom_potential(lambda x: 0.0 if 0 < x[0] < 5 else 1e10, "Infinite Well")
             .set_spatial_range(0, 5)
             .set_grid_points(200)
             .build())
    
    result = simulate(scene, num_states=6)
    
    print(quick_report(result.hamiltonian, result))
    
    print("\nStructured Result:")
    result_struct = analyze(result.hamiltonian, result)
    print(f"  Symmetries: {len(result_struct.symmetries)}")
    print(f"  State classifications: {len(result_struct.state_classifications)}")
    print(f"  Transition rules: {len(result_struct.transition_rules)}")
    print(f"  Conserved quantities: {len(result_struct.conserved_quantities)}")
    print(f"  Invariants: {result_struct.invariants}")


def run_all():
    """Run all demos"""
    print()
    print("#" * 70)
    print("#" + "  Quantum Analysis Demo  ".center(68) + "#")
    print("#" * 70)
    
    demos = [
        ("Harmonic Oscillator", demo_harmonic_oscillator),
        ("Square Well", demo_square_well),
        ("Double Well", demo_double_well),
        ("Selection Rules", demo_selection_rules),
        ("Full Analysis", demo_full_analysis),
    ]
    
    results = []
    for name, func in demos:
        try:
            func()
            results.append((name, True))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print()
    print("#" * 70)
    print("#  Summary  ".center(68) + "#")
    print("#" * 70)
    print()
    for name, success in results:
        print(f"  {name:<25} [{'PASS' if success else 'FAIL'}]")
    print()


if __name__ == "__main__":
    run_all()
