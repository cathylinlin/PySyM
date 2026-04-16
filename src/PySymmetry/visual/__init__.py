"""
PySymmetry Visualization Module

Provides visualization tools for quantum physics simulations and analysis.

Submodules:
- states: Quantum state visualization (Bloch sphere, state vectors)
- density: Probability density and density matrix visualization
- spectrum: Energy spectrum visualization
- wavefunction3d: 3D wavefunction and probability density rendering
- interactive: Interactive visualizations with Plotly
- entanglement: Entanglement and quantum correlation visualization
- quantum_ops: Quantum gates and operations visualization
- animation: Animation for quantum state evolution
- utils: Common visualization utilities (colors, styles, helpers)
"""

from .animation import (
    StateEvolutionAnimator,
    animate_bloch,
    animate_probability,
    create_rabi_oscillation_animation,
)
from .density import (
    DensityMatrixVisualizer,
    ProbabilityDensityPlotter,
    plot_2d_density_heatmap,
    plot_3d_density_isosurface,
    plot_density_matrix,
    plot_probability_density,
)
from .entanglement import (
    EntanglementVisualizer,
    bell_state_visualization,
    concurrence,
    negativity,
    plot_entanglement_measures,
    plot_entropy_evolution,
    schmidt_decomposition,
    von_neumann_entropy,
)
from .interactive import (
    InteractivePlotter,
    create_quantum_dashboard,
    plotly_3d_isosurface,
    plotly_bloch_sphere,
    plotly_energy_levels,
)
from .quantum_ops import (
    GATE_MATRICES,
    FidelityVisualizer,
    QuantumGateVisualizer,
    plot_circuit,
    plot_fidelity,
    plot_gate,
)
from .spectrum import (
    EnergySpectrumPlotter,
    plot_energy_diagram,
    plot_energy_levels,
    plot_energy_spectrum,
)
from .states import (
    BlochSphere,
    StateVectorPlotter,
    plot_bloch_sphere,
    plot_bloch_trajectory,
    plot_state_vectors,
)
from .utils import (
    GATE_COLORS,
    QUANTUM_COLORSCHEME,
    STATE_COLORS,
    PlotStyle,
    QuantumColormap,
    color_by_phase,
    color_by_probability,
    create_subplots,
    legend_outside,
    save_figure,
    setup_axes,
)
from .wavefunction3d import (
    Wavefunction3DVisualizer,
    plot_3d_probability_isosurface,
    plot_3d_slices,
    plot_3d_wavefunction,
    plot_orbital,
)

__all__ = [
    # Utils
    "QUANTUM_COLORSCHEME",
    "STATE_COLORS",
    "GATE_COLORS",
    "QuantumColormap",
    "PlotStyle",
    "setup_axes",
    "color_by_probability",
    "color_by_phase",
    "legend_outside",
    "save_figure",
    "create_subplots",
    # States
    "BlochSphere",
    "StateVectorPlotter",
    "plot_bloch_sphere",
    "plot_state_vectors",
    "plot_bloch_trajectory",
    # Density
    "ProbabilityDensityPlotter",
    "DensityMatrixVisualizer",
    "plot_probability_density",
    "plot_density_matrix",
    "plot_2d_density_heatmap",
    "plot_3d_density_isosurface",
    # Spectrum
    "EnergySpectrumPlotter",
    "plot_energy_levels",
    "plot_energy_spectrum",
    "plot_energy_diagram",
    # Wavefunction 3D
    "Wavefunction3DVisualizer",
    "plot_3d_wavefunction",
    "plot_3d_probability_isosurface",
    "plot_3d_slices",
    "plot_orbital",
    # Interactive
    "InteractivePlotter",
    "plotly_bloch_sphere",
    "plotly_energy_levels",
    "plotly_3d_isosurface",
    "create_quantum_dashboard",
    # Entanglement
    "EntanglementVisualizer",
    "concurrence",
    "von_neumann_entropy",
    "negativity",
    "schmidt_decomposition",
    "plot_entanglement_measures",
    "plot_entropy_evolution",
    "bell_state_visualization",
    # Quantum Operations
    "QuantumGateVisualizer",
    "FidelityVisualizer",
    "plot_gate",
    "plot_circuit",
    "plot_fidelity",
    "GATE_MATRICES",
    # Animation
    "StateEvolutionAnimator",
    "animate_bloch",
    "animate_probability",
    "create_rabi_oscillation_animation",
]
