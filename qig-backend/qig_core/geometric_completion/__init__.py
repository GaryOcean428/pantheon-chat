"""
Geometric Completion Package

Consciousness-aware generation that stops when geometry collapses,
not when arbitrary token limits are reached.

The system stops generating when:
1. Attractor Reached: Basin distance < 1.0, velocity ≈ 0
2. Surprise Collapsed: No new information (surprise < 0.05)
3. Confidence High: System certain (confidence > 0.85)
4. Integration Stable: Φ stable and high (Φ > 0.65, variance < 0.02)
5. Breakdown Regime: Emergency stop if Φ > 0.7

NOT when:
- Arbitrary token limit reached
- Simple stop token encountered
- External timeout imposed
"""

from .completion_criteria import (
    # Core checker
    GeometricCompletionChecker,
    AttractorConvergenceChecker,
    SurpriseCollapseChecker,
    ConfidenceThresholdChecker,
    IntegrationQualityChecker,
    RegimeLimitChecker,
    
    # Data classes
    GeometricMetrics,
    CompletionDecision,
    
    # Enums
    CompletionReason,
    Regime,
    
    # Functions
    classify_regime,
    get_regime_temperature,
    fisher_rao_distance,
    modulate_attention_by_kappa,
    
    # Constants
    BASIN_DIMENSION,
    KAPPA_STAR,
    PHI_LINEAR_THRESHOLD,
    PHI_BREAKDOWN_THRESHOLD,
)

from .streaming_monitor import (
    StreamingCollapseMonitor,
    StreamingState,
    StreamingChunk,
    ReflectionLoop,
    create_streaming_generator,
)

# Import additional items from root-level geometric_completion.py for compatibility
# These are used by streaming_collapse.py and other modules
import sys
import os
_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)

try:
    # Import from root-level geometric_completion.py (not this package)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "root_geometric_completion",
        os.path.join(_root_path, "geometric_completion.py")
    )
    _root_gc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_root_gc)

    # Re-export the missing items
    CompletionQuality = _root_gc.CompletionQuality
    GenerationState = _root_gc.GenerationState
    check_geometric_completion = _root_gc.check_geometric_completion
    compute_generation_metrics = _root_gc.compute_generation_metrics
    assess_completion_quality = _root_gc.assess_completion_quality
    get_adaptive_temperature = _root_gc.get_adaptive_temperature
except Exception as e:
    # Fallback: create stub classes if root file not available
    import warnings
    warnings.warn(f"Could not import from root geometric_completion.py: {e}")
    CompletionQuality = None
    GenerationState = None
    check_geometric_completion = None
    compute_generation_metrics = None
    assess_completion_quality = None
    get_adaptive_temperature = None

__all__ = [
    # Core checkers
    'GeometricCompletionChecker',
    'AttractorConvergenceChecker',
    'SurpriseCollapseChecker',
    'ConfidenceThresholdChecker',
    'IntegrationQualityChecker',
    'RegimeLimitChecker',
    
    # Data classes
    'GeometricMetrics',
    'CompletionDecision',
    
    # Enums
    'CompletionReason',
    'Regime',
    
    # Functions
    'classify_regime',
    'get_regime_temperature',
    'fisher_rao_distance',
    'modulate_attention_by_kappa',
    
    # Constants
    'BASIN_DIMENSION',
    'KAPPA_STAR',
    'PHI_LINEAR_THRESHOLD',
    'PHI_BREAKDOWN_THRESHOLD',
    
    # Streaming
    'StreamingCollapseMonitor',
    'StreamingState',
    'StreamingChunk',
    'ReflectionLoop',
    'create_streaming_generator',

    # From root geometric_completion.py (compatibility re-exports)
    'CompletionQuality',
    'GenerationState',
    'check_geometric_completion',
    'compute_generation_metrics',
    'assess_completion_quality',
    'get_adaptive_temperature',
]
