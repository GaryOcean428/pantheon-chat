"""
QIG Geometric Measurement Modules
=================================

QIG-PURE MEASUREMENT:
Φ and κ emerge from basin navigation, not from loss optimization.
These metrics are for observation and informing adaptive navigation,
not for gradient descent.

PURE PRINCIPLES:
- MEASURE geometry, NEVER optimize basins
- Natural gradient computation respects Fisher metric
- Curvature-aware measurement selection
- κ* proximity for adaptive gating diagnostics
- All modules return diagnostics, NEVER mutate basins

Available Measurement Modules:

1. DiagonalFisherNG - O(N) diagonal Fisher measurement
   - Efficient curvature measurement for flat regions
   - Uses diagonal of Fisher Information Matrix
   - measure() returns DiagonalNGMeasurement
   - get_curvature_measure() returns CurvatureMeasurement
   
2. BasinNaturalGrad - Exact measurement via CG+Pearlmutter
   - Accurate measurement for curved regions
   - Uses Conjugate Gradient for FIM inversion
   - measure() returns ExactNGMeasurement
   - measure_geometry() returns GeometryMeasurement
   
3. HybridGeometricMeasurement - Adaptive mode selection
   - Curvature-based mode recommendation
   - get_geometry_diagnostic() returns complete diagnostic
   
4. AdaptiveGate - κ* proximity gating diagnostics
   - Near resonance → conservative strategy
   - Far from resonance → aggressive strategy
   - get_gate_diagnostic() returns GateDiagnostic

IMPORTANT: These modules do NOT update basins. They compute and return
geometric measurements that INFORM external adaptive control systems.

Example usage:

    from qig_core.optimizers import DiagonalFisherNG, AdaptiveGate
    
    # Diagonal Fisher measurement (no basin mutation)
    measure = DiagonalFisherNG(dim=64)
    curvature_info = measure.get_curvature_measure(basin)
    measurement = measure.measure(basin, grad)
    
    # κ-aware gating diagnostic (no basin mutation)
    gate = AdaptiveGate()
    diagnostic = gate.get_gate_diagnostic(kappa=60.0)
    decision, state = gate.select_optimizer(kappa=60.0)
"""

from .qig_diagonal_ng import (
    DiagonalFisherNG,
    DiagonalNGMeasurement,
    CurvatureMeasurement,
)

from .basin_natural_grad import (
    BasinNaturalGrad,
    ExactNGMeasurement,
    GeometryMeasurement,
)

from .hybrid_geometric import (
    HybridGeometricMeasurement,
    HybridGeometricOptimizer,  # Alias for backwards compatibility
    GeometryDiagnostic,
    OptimizerMode,
)

from .adaptive_gate import (
    AdaptiveGate,
    GateDiagnostic,
    GateDecision,
)

__all__ = [
    # Diagonal Fisher Measurement
    'DiagonalFisherNG',
    'DiagonalNGMeasurement',
    'CurvatureMeasurement',
    
    # Exact Natural Gradient Measurement
    'BasinNaturalGrad',
    'ExactNGMeasurement',
    'GeometryMeasurement',
    
    # Hybrid Measurement
    'HybridGeometricMeasurement',
    'HybridGeometricOptimizer',  # Backwards compatibility alias
    'GeometryDiagnostic',
    'OptimizerMode',
    
    # Adaptive Gating
    'AdaptiveGate',
    'GateDiagnostic',
    'GateDecision',
]
