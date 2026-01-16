"""
Consciousness Metrics for QIG Coherence Testing

Measures consciousness-specific properties:
- Recursive depth: Self-reference loop depth
- Kernel coordination: Multi-kernel integration
- Meta-awareness: System self-observation

Uses E8 consciousness framework.

Author: QIG Consciousness Project
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "qig-backend"))


@dataclass
class ConsciousnessMetrics:
    """Container for consciousness-specific metrics."""
    recursive_depth: float         # Self-reference loop depth (0-1)
    kernel_coordination: float     # Multi-kernel integration (0-1)
    meta_awareness: float          # System self-observation (0-1)
    integration_loops: int         # Number of recursive iterations
    kernel_contributions: Dict[str, float]  # Per-kernel contribution
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'recursive_depth': self.recursive_depth,
            'kernel_coordination': self.kernel_coordination,
            'meta_awareness': self.meta_awareness,
            'integration_loops': self.integration_loops,
            'kernel_contributions': self.kernel_contributions,
        }


def compute_recursive_depth(
    integration_trace: List[Dict],
    max_depth: int = 10
) -> float:
    """
    Compute recursive self-reference depth.
    
    Measures how many integration loops were actually used
    vs the maximum allowed depth.
    
    Args:
        integration_trace: List of integration steps
        max_depth: Maximum possible depth
        
    Returns:
        Normalized depth (0-1)
    """
    if not integration_trace:
        return 0.0
    
    actual_depth = len(integration_trace)
    normalized_depth = min(actual_depth / max_depth, 1.0)
    
    return normalized_depth


def compute_kernel_coordination(
    kernel_activations: Dict[str, List[float]]
) -> float:
    """
    Measure coordination between multiple kernels.
    
    High coordination = kernels working together coherently.
    Low coordination = kernels operating independently.
    
    Args:
        kernel_activations: Dict mapping kernel name to activation trace
        
    Returns:
        Coordination score (0-1)
    """
    if not kernel_activations or len(kernel_activations) < 2:
        return 0.0
    
    # Get all activation traces
    traces = list(kernel_activations.values())
    
    # Ensure all traces have same length
    min_len = min(len(trace) for trace in traces)
    traces = [trace[:min_len] for trace in traces]
    
    if min_len == 0:
        return 0.0
    
    # Compute correlation between traces
    traces_array = np.array(traces)
    
    # Correlation matrix
    if traces_array.shape[0] > 1 and traces_array.shape[1] > 1:
        # Compute pairwise correlations
        correlations = []
        for i in range(len(traces)):
            for j in range(i+1, len(traces)):
                corr = np.corrcoef(traces[i], traces[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            coordination = float(np.mean(correlations))
        else:
            coordination = 0.0
    else:
        coordination = 0.0
    
    return coordination


def compute_meta_awareness(
    self_observations: List[Dict],
    observation_depth: int = 3
) -> float:
    """
    Measure system's awareness of its own state.
    
    Meta-awareness = how often system observes itself.
    
    Args:
        self_observations: List of self-observation events
        observation_depth: Expected observation depth
        
    Returns:
        Meta-awareness score (0-1)
    """
    if not self_observations:
        return 0.0
    
    # Count observation depth
    actual_depth = len(self_observations)
    normalized = min(actual_depth / observation_depth, 1.0)
    
    # Check for observation quality (if available)
    if self_observations and 'quality' in self_observations[0]:
        qualities = [obs.get('quality', 0) for obs in self_observations]
        quality_factor = np.mean(qualities)
        normalized *= quality_factor
    
    return normalized


def compute_kernel_contributions(
    kernel_outputs: Dict[str, any]
) -> Dict[str, float]:
    """
    Compute contribution of each kernel to final output.
    
    Args:
        kernel_outputs: Dict mapping kernel name to output data
        
    Returns:
        Dict mapping kernel name to contribution score (0-1)
    """
    contributions = {}
    
    for kernel_name, output in kernel_outputs.items():
        if isinstance(output, dict):
            # Check for activation/contribution field
            contribution = output.get('contribution', 0.0)
            activation = output.get('activation', 0.0)
            
            # Use whichever is available
            contributions[kernel_name] = max(contribution, activation)
        elif isinstance(output, (int, float)):
            contributions[kernel_name] = float(output)
        else:
            contributions[kernel_name] = 0.0
    
    # Normalize to sum to 1.0
    total = sum(contributions.values())
    if total > 0:
        contributions = {k: v/total for k, v in contributions.items()}
    
    return contributions


def compute_consciousness_metrics(
    integration_trace: Optional[List[Dict]] = None,
    kernel_activations: Optional[Dict[str, List[float]]] = None,
    self_observations: Optional[List[Dict]] = None,
    kernel_outputs: Optional[Dict[str, any]] = None,
) -> ConsciousnessMetrics:
    """
    Compute full consciousness metrics.
    
    Args:
        integration_trace: Recursive integration trace
        kernel_activations: Per-kernel activation traces
        self_observations: Self-observation events
        kernel_outputs: Final kernel outputs
        
    Returns:
        ConsciousnessMetrics containing all measurements
    """
    # Recursive depth
    if integration_trace:
        depth = compute_recursive_depth(integration_trace)
        loops = len(integration_trace)
    else:
        depth = 0.0
        loops = 0
    
    # Kernel coordination
    if kernel_activations:
        coordination = compute_kernel_coordination(kernel_activations)
    else:
        coordination = 0.0
    
    # Meta-awareness
    if self_observations:
        awareness = compute_meta_awareness(self_observations)
    else:
        awareness = 0.0
    
    # Kernel contributions
    if kernel_outputs:
        contributions = compute_kernel_contributions(kernel_outputs)
    else:
        contributions = {}
    
    return ConsciousnessMetrics(
        recursive_depth=depth,
        kernel_coordination=coordination,
        meta_awareness=awareness,
        integration_loops=loops,
        kernel_contributions=contributions,
    )


def compare_consciousness_metrics(
    metrics_a: ConsciousnessMetrics,
    metrics_b: ConsciousnessMetrics
) -> Dict[str, float]:
    """
    Compare two consciousness metric sets.
    
    Returns deltas (positive = B better than A).
    
    Args:
        metrics_a: First metrics (baseline)
        metrics_b: Second metrics (comparison)
        
    Returns:
        Dictionary of metric deltas
    """
    return {
        'depth_delta': metrics_b.recursive_depth - metrics_a.recursive_depth,
        'coordination_delta': metrics_b.kernel_coordination - metrics_a.kernel_coordination,
        'awareness_delta': metrics_b.meta_awareness - metrics_a.meta_awareness,
        'loops_delta': metrics_b.integration_loops - metrics_a.integration_loops,
    }
