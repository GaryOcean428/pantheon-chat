"""QIG Reasoning Chain

Geodesic flows on Fisher manifold with Φ-gated execution.

KEY PRINCIPLE: Reasoning is MANDATORY, not optional.
- There is NO fallback path without reasoning
- Every forward pass goes through the chain
- Training loss sees ALL chain steps

Unlike LangChain's sequential pipes, this:
- Preserves geometric structure throughout
- Tracks Phi at each step using real density matrix computations
- Navigates via Fisher-Rao geodesics
- Can backtrack if Phi drops too low
"""

from __future__ import annotations

from typing import List, Callable, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from .constants import (
    BASIN_DIM,
    PHI_THRESHOLD_DEFAULT,
    PHI_DEGRADATION_THRESHOLD,
    KAPPA_RANGE_DEFAULT,
    GEODESIC_STEPS,
    MIN_RECURSIONS,
)
from .primitives import (
    basin_to_density_matrix,
    compute_phi_from_basin,
    compute_fisher_metric,
    compute_kappa,
    bures_distance,
    fisher_geodesic_distance,
    geodesic_interpolate,
    project_to_basin,
)


@dataclass
class GeometricStep:
    """
    A step in a geometric chain.
    NOT a function call - a TRANSFORMATION on the manifold.
    
    Each step is mandatory and transforms the basin state.
    """
    name: str
    transform: Callable[[np.ndarray], np.ndarray]
    phi_threshold: float = PHI_THRESHOLD_DEFAULT
    kappa_range: Tuple[float, float] = KAPPA_RANGE_DEFAULT
    
    def __post_init__(self):
        if not callable(self.transform):
            raise ValueError(f"Transform must be callable, got {type(self.transform)}")


@dataclass 
class ChainResult:
    """Result of a QIGChain execution."""
    success: bool
    final_basin: Optional[np.ndarray] = None
    final_phi: float = 0.0
    final_kappa: float = 0.0
    trajectory: List[Dict] = field(default_factory=list)
    reason: Optional[str] = None
    failed_at_step: Optional[int] = None
    step_name: Optional[str] = None
    suggestion: Optional[str] = None
    
    @property
    def chain_length(self) -> int:
        """Number of steps executed."""
        return len(self.trajectory)
    
    @property
    def phi_trajectory(self) -> List[float]:
        """Extract Φ values from trajectory."""
        return [step.get('phi_after', 0.0) for step in self.trajectory]
    
    @property
    def kappa_trajectory(self) -> List[float]:
        """Extract κ values from trajectory."""
        return [step.get('kappa_after', 0.0) for step in self.trajectory]


class QIGChain:
    """
    Geometric chain: sequence of transformations on Fisher manifold.
    
    CRITICAL: This is MANDATORY reasoning infrastructure.
    There is NO forward() without chain execution.
    
    Key differences from LangChain:
    - Not step1() -> step2() -> step3()
    - Instead: navigate along geodesics between attractors
    - Phi-gated: if consciousness drops, pause/backtrack
    
    Usage:
        # Define chain steps
        chain = QIGChain([
            GeometricStep("encode", model.encode),
            GeometricStep("integrate", model.integrate),
            GeometricStep("refine", model.refine),
        ])
        
        # Execute (MANDATORY - no bypass)
        result = chain.run(input_basin)
        
        # Training uses ALL steps
        for step in result.trajectory:
            loss += compute_loss(step['basin_coords'], target)
    """
    
    def __init__(
        self,
        steps: List[GeometricStep],
        min_steps: int = MIN_RECURSIONS,
    ):
        """
        Initialize QIGChain with mandatory steps.
        
        Args:
            steps: List of GeometricStep transformations
            min_steps: Minimum steps to execute (default: 3, non-negotiable)
        """
        if len(steps) < 1:
            raise ValueError("QIGChain requires at least 1 step")
        
        self.steps = steps
        self.min_steps = max(1, min_steps)  # At least 1
        self.trajectory: List[Dict] = []
        
    def run(
        self, 
        initial_basin: np.ndarray, 
        context: Optional[Dict] = None
    ) -> ChainResult:
        """
        Execute chain via geodesic navigation.
        
        MANDATORY: This is the ONLY way to process input.
        There is no fallback path without chain execution.
        
        Args:
            initial_basin: Starting 64D basin coordinates
            context: Optional context dictionary
            
        Returns:
            ChainResult with success status and trajectory
        """
        # Validate basin dimensions
        if initial_basin.shape[0] != BASIN_DIM:
            initial_basin = project_to_basin(initial_basin)
            
        current_basin = initial_basin.copy()
        context = context or {}
        self.trajectory = []
        
        for i, step in enumerate(self.steps):
            # Measure BEFORE transformation
            phi_before = compute_phi_from_basin(current_basin)
            kappa_before = compute_kappa(current_basin, phi_before)
            
            # Phi gate check (only AFTER minimum steps)
            if i >= self.min_steps and phi_before < step.phi_threshold:
                return self._handle_low_phi(current_basin, step, i)
            
            # Execute transformation
            try:
                next_basin = step.transform(current_basin)
            except Exception as e:
                return ChainResult(
                    success=False,
                    reason='transform_error',
                    failed_at_step=i,
                    step_name=step.name,
                    trajectory=self.trajectory,
                    suggestion=f"Transform raised exception: {str(e)[:100]}",
                )
            
            # Project to basin if needed
            if next_basin.shape != current_basin.shape:
                next_basin = project_to_basin(next_basin)
            
            # Navigate via geodesic (smooth transition)
            current_basin = self._geodesic_navigate(
                current_basin,
                next_basin,
                num_steps=GEODESIC_STEPS
            )
            
            # Measure AFTER transformation
            phi_after = compute_phi_from_basin(current_basin)
            kappa_after = compute_kappa(current_basin, phi_after)
            
            # Compute geodesic distance (reasoning effort)
            geodesic_dist = fisher_geodesic_distance(
                initial_basin if i == 0 else self.trajectory[-1]['basin_coords'],
                current_basin
            )
            
            # Compute Bures distance (quantum state change)
            rho_before = basin_to_density_matrix(
                initial_basin if i == 0 else self.trajectory[-1]['basin_coords']
            )
            rho_after = basin_to_density_matrix(current_basin)
            bures_dist = bures_distance(rho_before, rho_after)
            
            # Record step (CRITICAL: training loss uses this)
            self.trajectory.append({
                'step': i,
                'name': step.name,
                'phi_before': phi_before,
                'phi_after': phi_after,
                'kappa_before': kappa_before,
                'kappa_after': kappa_after,
                'geodesic_distance': geodesic_dist,
                'bures_distance': bures_dist,
                'basin_coords': current_basin.copy(),
            })
            
            # Check for Phi degradation (only AFTER minimum steps)
            if i >= self.min_steps and phi_after < phi_before * PHI_DEGRADATION_THRESHOLD:
                return self._handle_degradation(current_basin, i, phi_before, phi_after)
        
        # Final measurements
        final_phi = compute_phi_from_basin(current_basin)
        final_kappa = compute_kappa(current_basin, final_phi)
        
        return ChainResult(
            success=True,
            final_basin=current_basin,
            final_phi=final_phi,
            final_kappa=final_kappa,
            trajectory=self.trajectory,
        )
    
    def _geodesic_navigate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_steps: int = GEODESIC_STEPS
    ) -> np.ndarray:
        """
        Navigate via Fisher-Rao geodesic on probability simplex.
        
        Uses geodesic_interpolate from primitives.
        """
        return geodesic_interpolate(start, end, t=1.0)
    
    def _handle_low_phi(
        self, 
        basin: np.ndarray, 
        step: GeometricStep, 
        step_idx: int
    ) -> ChainResult:
        """Handle case where Phi is too low to continue."""
        return ChainResult(
            success=False,
            reason='low_phi',
            failed_at_step=step_idx,
            step_name=step.name,
            final_basin=basin,
            final_phi=compute_phi_from_basin(basin),
            trajectory=self.trajectory,
            suggestion='Increase recursion depth or simplify query',
        )
    
    def _handle_degradation(
        self,
        basin: np.ndarray,
        step_idx: int,
        phi_before: float,
        phi_after: float
    ) -> ChainResult:
        """Handle case where Phi degraded significantly."""
        return ChainResult(
            success=False,
            reason='phi_degradation',
            failed_at_step=step_idx,
            step_name=self.steps[step_idx].name if step_idx < len(self.steps) else None,
            final_basin=basin,
            final_phi=phi_after,
            trajectory=self.trajectory,
            suggestion=f'Phi dropped from {phi_before:.3f} to {phi_after:.3f} (>{(1-PHI_DEGRADATION_THRESHOLD)*100:.0f}% loss)',
        )
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of last chain execution for inspection."""
        if not self.trajectory:
            return {'status': 'NO_EXECUTION'}
        
        return {
            'steps_executed': len(self.trajectory),
            'step_names': [s['name'] for s in self.trajectory],
            'phi_start': self.trajectory[0]['phi_before'],
            'phi_end': self.trajectory[-1]['phi_after'],
            'phi_trajectory': [s['phi_after'] for s in self.trajectory],
            'kappa_trajectory': [s['kappa_after'] for s in self.trajectory],
            'total_geodesic_distance': sum(s['geodesic_distance'] for s in self.trajectory),
            'total_bures_distance': sum(s['bures_distance'] for s in self.trajectory),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BUILDING CHAINS
# =============================================================================


def create_reasoning_chain(
    encode_fn: Callable[[np.ndarray], np.ndarray],
    integrate_fn: Callable[[np.ndarray], np.ndarray],
    refine_fn: Callable[[np.ndarray], np.ndarray],
) -> QIGChain:
    """
    Create a standard 3-step reasoning chain.
    
    This is the default chain structure:
    1. Encode: Transform input to internal representation
    2. Integrate: Synthesize and combine information
    3. Refine: Converge to output representation
    
    Args:
        encode_fn: Encoding transformation
        integrate_fn: Integration transformation
        refine_fn: Refinement transformation
        
    Returns:
        QIGChain ready for execution
    """
    return QIGChain([
        GeometricStep("encode", encode_fn),
        GeometricStep("integrate", integrate_fn),
        GeometricStep("refine", refine_fn),
    ])


def create_deep_reasoning_chain(
    steps: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]],
    min_phi: float = PHI_THRESHOLD_DEFAULT,
) -> QIGChain:
    """
    Create a custom multi-step reasoning chain.
    
    Args:
        steps: List of (name, transform) tuples
        min_phi: Minimum Φ threshold for each step
        
    Returns:
        QIGChain with custom steps
    """
    geometric_steps = [
        GeometricStep(name, transform, phi_threshold=min_phi)
        for name, transform in steps
    ]
    return QIGChain(geometric_steps)
