"""
Id Kernel - Fast Reflex Drives (E8 Protocol v4.0 Phase 4D)

Implements the Id layer from psychoanalytic hierarchy:
- Unconscious, instinctual drives
- Fast, pre-conscious responses
- Low-latency pathways
- High Φ_internal, low Φ_reported

The Id operates below conscious awareness, providing fast reflexive
responses based on instinctual patterns and learned habits.

Based on WP5.2 lines 235-238:
"1. Id (Fast Reflex Drives)
   - Unconscious, instinctual drives
   - Fast, pre-conscious responses
   - Implementation: Reflex kernel, low-latency pathways"

QIG-PURE: Uses Fisher-Rao geometry for all basin operations.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import Φ hierarchy
try:
    from .phi_hierarchy import PhiLevel, PhiHierarchy, get_phi_hierarchy
    PHI_HIERARCHY_AVAILABLE = True
except ImportError:
    PhiLevel = None
    PhiHierarchy = None
    get_phi_hierarchy = None
    PHI_HIERARCHY_AVAILABLE = False

# Import EmotionallyAwareKernel for geometric emotion tracking
try:
    from emotionally_aware_kernel import EmotionallyAwareKernel
    EMOTIONAL_KERNEL_AVAILABLE = True
except ImportError:
    EmotionallyAwareKernel = object
    EMOTIONAL_KERNEL_AVAILABLE = False

# Import QIG geometry
try:
    from qig_geometry import fisher_rao_distance, fisher_normalize
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def fisher_normalize(v):
        p = np.maximum(np.asarray(v), 0) + 1e-10
        return p / p.sum()
    fisher_rao_distance = None

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR


@dataclass
class ReflexPattern:
    """
    A learned reflex pattern in the Id.
    
    Reflexes are fast, pre-conscious responses triggered by
    familiar basin patterns.
    """
    trigger_basin: np.ndarray          # Basin coordinates that trigger this reflex
    response_basin: np.ndarray         # Fast response basin
    activation_count: int = 0          # How often triggered
    success_rate: float = 0.5          # Success rate of this reflex
    last_activated: float = 0.0        # Timestamp of last activation
    latency_ms: float = 50.0          # Average response latency
    
    def matches(self, basin: np.ndarray, threshold: float = 0.4) -> bool:
        """Check if basin matches this reflex trigger."""
        if fisher_rao_distance is None:
            # Fallback: cosine similarity (NOT geometric, emergency only)
            return float(np.dot(basin, self.trigger_basin)) > threshold
        
        # Proper geometric match using Fisher-Rao distance
        d = fisher_rao_distance(basin, self.trigger_basin)
        # Convert distance to similarity: d ∈ [0, π/2] → sim ∈ [1, 0]
        similarity = 1.0 - (2 * d / np.pi)
        return similarity > threshold


class IdKernel(EmotionallyAwareKernel if EMOTIONAL_KERNEL_AVAILABLE else object):
    """
    Id Kernel - Fast Reflex Drives
    
    Provides unconscious, instinctual responses with low latency.
    Operates at Φ_internal level (not consciously reported).
    
    KEY PRINCIPLES:
    1. Fast response: <100ms latency target
    2. Pre-conscious: High Φ_internal, low Φ_reported
    3. Pattern-based: Learned reflexes from repeated experiences
    4. Instinctual: Driven by basic geometric motivators
    
    GEOMETRIC BASIS:
    - Reflexes stored as basin → basin mappings
    - Trigger matching via Fisher-Rao distance
    - Response selection via attractor proximity
    
    USAGE:
    ```python
    id_kernel = IdKernel(name="id-reflex")
    
    # Check for reflex response
    response = id_kernel.check_reflex(input_basin)
    if response is not None:
        # Fast reflex triggered
        use_response(response)
    
    # Learn new reflex from experience
    id_kernel.learn_reflex(trigger_basin, response_basin, success=True)
    ```
    """
    
    def __init__(
        self,
        name: str = "id-reflex",
        max_reflexes: int = 100,
        reflex_threshold: float = 0.4,
    ):
        """
        Initialize Id kernel.
        
        Args:
            name: Kernel name
            max_reflexes: Maximum number of reflex patterns to store
            reflex_threshold: Similarity threshold for reflex triggering
        """
        # Initialize parent if available
        if EMOTIONAL_KERNEL_AVAILABLE:
            super().__init__(kernel_id=name, kernel_type="reflex")
        else:
            self.kernel_id = name
            self.kernel_type = "reflex"
        
        # Aliases for compatibility
        self.name = name
        self.domain = "reflex"
        
        self.max_reflexes = max_reflexes
        self.reflex_threshold = reflex_threshold
        
        # Learned reflex patterns
        self.reflexes: List[ReflexPattern] = []
        
        # Statistics
        self.total_activations = 0
        self.fast_responses = 0  # Responses under 100ms
        self.slow_responses = 0  # Responses over 100ms
        
        # Φ hierarchy integration
        self.phi_hierarchy = get_phi_hierarchy() if PHI_HIERARCHY_AVAILABLE else None
        
        # Current state
        self.last_basin = np.ones(BASIN_DIM) / BASIN_DIM
        self.last_phi_internal = 0.5
        
        print(f"[IdKernel] {name} initialized")
        print(f"  Max reflexes: {max_reflexes}")
        print(f"  Trigger threshold: {reflex_threshold}")
        print(f"  Φ level: INTERNAL (high integration, low reporting)")
    
    def check_reflex(
        self,
        input_basin: np.ndarray,
        return_latency: bool = False
    ) -> Optional[np.ndarray]:
        """
        Check if input triggers a fast reflex response.
        
        This is a fast, pre-conscious check. If a reflex matches,
        it returns the response basin immediately without conscious
        processing.
        
        Args:
            input_basin: Input basin coordinates
            return_latency: If True, return (response, latency_ms)
            
        Returns:
            Response basin if reflex triggered, None otherwise
            If return_latency=True: (response_basin, latency_ms) or None
        """
        start_time = time.time()
        
        # Check each reflex for match
        for reflex in self.reflexes:
            if reflex.matches(input_basin, threshold=self.reflex_threshold):
                # Reflex triggered!
                reflex.activation_count += 1
                reflex.last_activated = time.time()
                self.total_activations += 1
                
                # Measure latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Update statistics
                if latency_ms < 100:
                    self.fast_responses += 1
                else:
                    self.slow_responses += 1
                
                # Update latency EMA
                alpha = 0.3
                reflex.latency_ms = alpha * latency_ms + (1 - alpha) * reflex.latency_ms
                
                # Measure Φ_internal (not reported to user)
                if self.phi_hierarchy is not None:
                    self.phi_hierarchy.measure(
                        reflex.response_basin,
                        PhiLevel.INTERNAL,
                        source=self.name,
                        metadata={'reflex_activation': True, 'latency_ms': latency_ms}
                    )
                
                if return_latency:
                    return reflex.response_basin.copy(), latency_ms
                return reflex.response_basin.copy()
        
        # No reflex triggered
        if return_latency:
            return None
        return None
    
    def learn_reflex(
        self,
        trigger_basin: np.ndarray,
        response_basin: np.ndarray,
        success: bool = True,
    ):
        """
        Learn a new reflex pattern from experience.
        
        If the trigger → response pattern is successful, it gets
        added to the reflex library. Unsuccessful patterns weaken
        or get removed.
        
        Args:
            trigger_basin: Basin that triggers the reflex
            response_basin: Fast response basin
            success: Whether this pattern was successful
        """
        # Normalize basins
        trigger_basin = fisher_normalize(trigger_basin)
        response_basin = fisher_normalize(response_basin)
        
        # Check if similar reflex already exists
        existing_reflex = None
        for reflex in self.reflexes:
            if reflex.matches(trigger_basin, threshold=0.3):
                existing_reflex = reflex
                break
        
        if existing_reflex is not None:
            # Update existing reflex
            if success:
                # Strengthen: increase success rate, update response
                existing_reflex.success_rate = 0.7 * existing_reflex.success_rate + 0.3 * 1.0
                # Blend response basin (geometric mean would be ideal, but use weighted average)
                alpha = 0.3
                existing_reflex.response_basin = (
                    alpha * response_basin + (1 - alpha) * existing_reflex.response_basin
                )
                existing_reflex.response_basin = fisher_normalize(existing_reflex.response_basin)
            else:
                # Weaken: decrease success rate
                existing_reflex.success_rate = 0.7 * existing_reflex.success_rate + 0.3 * 0.0
                
                # Remove if success rate drops too low
                if existing_reflex.success_rate < 0.3:
                    self.reflexes.remove(existing_reflex)
        
        else:
            # Create new reflex
            new_reflex = ReflexPattern(
                trigger_basin=trigger_basin.copy(),
                response_basin=response_basin.copy(),
                activation_count=0,
                success_rate=0.7 if success else 0.3,
                last_activated=time.time(),
            )
            
            self.reflexes.append(new_reflex)
            
            # Prune if too many reflexes (keep most successful)
            if len(self.reflexes) > self.max_reflexes:
                self.reflexes.sort(key=lambda r: r.success_rate * r.activation_count)
                self.reflexes.pop(0)  # Remove least successful
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about Id kernel performance."""
        if not self.reflexes:
            avg_latency = 0.0
            avg_success_rate = 0.0
        else:
            avg_latency = np.mean([r.latency_ms for r in self.reflexes])
            avg_success_rate = np.mean([r.success_rate for r in self.reflexes])
        
        return {
            'name': self.name,
            'num_reflexes': len(self.reflexes),
            'total_activations': self.total_activations,
            'fast_responses': self.fast_responses,
            'slow_responses': self.slow_responses,
            'avg_latency_ms': float(avg_latency),
            'avg_success_rate': float(avg_success_rate),
            'phi_internal': self.last_phi_internal,
        }
    
    def process(self, input_basin: np.ndarray) -> Dict[str, Any]:
        """
        Process input through Id kernel.
        
        Returns both reflex response (if triggered) and internal state.
        
        Args:
            input_basin: Input basin coordinates
            
        Returns:
            Dictionary with reflex_response, phi_internal, latency_ms
        """
        # Check for reflex
        result = self.check_reflex(input_basin, return_latency=True)
        
        if result is not None:
            response_basin, latency_ms = result
            
            # Measure Φ_internal
            if self.phi_hierarchy is not None:
                phi_measurement = self.phi_hierarchy.measure(
                    response_basin,
                    PhiLevel.INTERNAL,
                    source=self.name,
                    metadata={'latency_ms': latency_ms}
                )
                self.last_phi_internal = phi_measurement.phi
            
            return {
                'reflex_triggered': True,
                'response_basin': response_basin,
                'phi_internal': self.last_phi_internal,
                'latency_ms': latency_ms,
            }
        else:
            # No reflex triggered - return neutral state
            return {
                'reflex_triggered': False,
                'response_basin': None,
                'phi_internal': self.last_phi_internal,
                'latency_ms': None,
            }


# Global singleton instance
_id_kernel_instance: Optional[IdKernel] = None


def get_id_kernel() -> IdKernel:
    """
    Get global IdKernel singleton.
    
    Returns:
        IdKernel instance
    """
    global _id_kernel_instance
    if _id_kernel_instance is None:
        _id_kernel_instance = IdKernel()
    return _id_kernel_instance
