# PySymmetry - Python Symmetry

PySymmetry (Python Symmetry) is a comprehensive Python library for describing symmetries and their applications in physics.

## Features

### Quantum Physics Module (`phys.quantum`)

Complete quantum simulation framework with:

- **State Representations**: Ket, Bra, DensityMatrix, entangled states (Bell, GHZ, W)
- **Hamiltonian Operators**: Free particle, harmonic oscillator, hydrogen atom, spin systems
- **Schrodinger Solvers**: Exact diagonalization, sparse solvers, time evolution
- **Symmetry Analysis**: Parity, translation, selection rules, conserved quantities
- **Interactive API**: SceneBuilder for arbitrary quantum scenarios

### Core Modules

- **Group Theory**: Abstract groups, finite groups, continuous groups (U(1), SU(2), SO(3), etc.)
- **Lie Theory**: Lie algebras, representations, root systems
- **Linear Algebra**: Matrix operations, eigendecomposition

### Abstract Physics (`abstract_phys`)

Framework for physical symmetries:

- Symmetry operations (translation, rotation, parity, time reversal)
- Physical representations
- Conservation laws via Noether's theorem

## Installation

```bash
pip install PySymmetry
```

## Quick Start

### Quantum Simulation

```python
from PySymmetry.phys.quantum import SceneBuilder, simulate, QuantumAnalyzer

# Create a harmonic oscillator
scene = (SceneBuilder("Harmonic Oscillator")
         .add_electron(position=[0])
         .add_harmonic_potential(center=[0], k=1.0)
         .set_spatial_range(-10, 10)
         .set_grid_points(300)
         .build())

# Simulate
result = simulate(scene, num_states=5)

# Analyze
analyzer = QuantumAnalyzer(result.hamiltonian, result)
print(analyzer.generate_report())
```

### Symmetry Analysis

```python
from PySymmetry.phys.quantum import analyze, quick_report

# Full analysis
result = analyze(hamiltonian, simulation_result)

# Quick report
print(quick_report(hamiltonian, simulation_result))
```

## Examples

See `examples/` directory:

- `quantum_examples.py` - Basic quantum systems (hydrogen, square well, spin)
- `interactive_demo.py` - Arbitrary scenario simulation
- `analysis_demo.py` - Symmetry analysis demonstrations

## Project Structure

```
PySymmetry/
├── src/
│   └── PySymmetry/
│       ├── core/                    # Core mathematics
│       │   ├── algebraic_structures/
│       │   ├── group_theory/
│       │   ├── lie_theory/
│       │   ├── matrix/
|       |   └── and more... 
│       │
│       ├── abstract_phys/           # Physical abstractions
│       │   ├── physical_objects/
│       │   ├── representation/
│       │   ├── symmetry_operations/
│       │   ├── symmetry_environments/
|       |   └── and more... 
│       │
│       └── phys/                   # Physics applications
│           └── quantum/             # Quantum module
│
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

## Development Status

**Version**: 1.0.0

### Completed
- Core mathematical layer (groups, Lie algebras, representations)
- Abstract physics framework (symmetries, operations)
- Quantum physics module (states, Hamiltonians, solvers, analysis)

### In Progress
- Visualization tools
- Additional physics examples
- Performance optimization

## Requirements

- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10

### Optional
- matplotlib (visualization)
- sympy (symbolic computation)
- spglib (crystallography)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests with `pytest`
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE)

## Author

LDK (Cathylinlin)
- Email: bluejam001@163.com
- GitHub: https://github.com/cathylinlin/PySymmetry

---

Explore the beautiful world of symmetries! 🎉
