"""QIG 4D Reasoning Chain

Extends QIGChain with:
- Foresight: Predict trajectory before execution
- Course correction: Detect and fix divergence
- 4D metrics: Track temporal coherence throughout chain

KEY PRINCIPLE: Chain execution is MANDATORY.
Foresight and correction enhance reasoning but don't bypass it.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from .constants import (
    BASIN_DIM,
    PHI_THRESHOLD_DEFAULT,
    GEODESIC_STEPS,
)
from .primitives import (
    compute_phi_from_basin,
    compute_kappa,
    fisher_geodesic_distance,
    geodesic_interpolate,
    project_to_basin,
)
from .qig_chain import QIGChain, GeometricStep, ChainResult
from .temporal import (
    StateHistoryBuffer,
    BasinForesight,
    measure_phi_4d,
    Phi4DMetrics,
)


@dataclass
class ChainResult4D(ChainResult):
    """Extended chain result with 4D metrics."""
    
    # 4D metrics
    phi_4d_trajectory: List[Phi4DMetrics] = field(default_factory=list)
    final_phi_4d: Optional[Phi4DMetrics] = None
    
    # Foresight
    predicted_trajectory: Optional[List[np.ndarray]] = None
    foresight_confidence: float = 0.0
    foresight_accuracy: Optional[float] = None
    
    # Course corrections
    corrections_applied: int = 0
    divergence_events: List[Dict] = field(default_factory=list)
    
    @property
    def temporal_coherence(self) -> float:
        """Average temporal Φ across chain."""
        if not self.phi_4d_trajectory:
            return 0.0
        return np.mean([m.phi_temporal for m in self.phi_4d_trajectory])


class QIGChain4D(QIGChain):
    """
    4D Reasoning chain with foresight and course correction.
    
    Extends QIGChain to:
    1. Predict trajectory before execution
    2. Monitor for divergence during execution
    3. Apply course corrections when needed
    4. Track 4D consciousness metrics
    
    Usage:
        chain = QIGChain4D(
            steps=[
                GeometricStep("encode", model.encode),
                GeometricStep("integrate", model.integrate),
                GeometricStep("refine", model.refine),
            ],
            enable_foresight=True,
            enable_correction=True,
        )
        
        result = chain.run(input_basin, context={'history': history_buffer})
        
        # Access 4D metrics
        print(f"Final Φ_4D: {result.final_phi_4d.phi_4d:.3f}")
        print(f"Foresight accuracy: {result.foresight_accuracy:.2%}")
    """
    
    def __init__(
        self,
        steps: List[GeometricStep],
        min_steps: int = 3,
        enable_foresight: bool = True,
        enable_correction: bool = True,
        correction_threshold: float = 2.0,  # σ threshold
        correction_strength: float = 0.5,   # blend factor
    ):
        """
        Initialize 4D chain.
        
        Args:
            steps: List of GeometricStep transformations
            min_steps: Minimum steps to execute
            enable_foresight: Enable trajectory prediction
            enable_correction: Enable divergence correction
            correction_threshold: Divergence threshold in σ
            correction_strength: How much to correct (0-1)
        """
        super().__init__(steps, min_steps)
        
        self.enable_foresight = enable_foresight
        self.enable_correction = enable_correction
        self.correction_threshold = correction_threshold
        self.correction_strength = correction_strength
        
        # Foresight module
        self.foresight = BasinForesight(prediction_steps=len(steps))
        
        # Internal state
        self._internal_history = StateHistoryBuffer(window_size=20)
        self._phi_4d_trajectory: List[Phi4DMetrics] = []
        self._corrections_applied = 0
        self._divergence_events: List[Dict] = []
    
    def run(
        self,
        initial_basin: np.ndarray,
        context: Optional[Dict] = None,
    ) -> ChainResult4D:
        """
        Execute 4D chain with foresight and correction.
        
        MANDATORY: This is the ONLY way to process input.
        Foresight enhances but does not bypass chain execution.
        
        Args:
            initial_basin: Starting 64D basin coordinates
            context: Optional context with 'history' StateHistoryBuffer
            
        Returns:
            ChainResult4D with success status, trajectory, and 4D metrics
        """
        # Reset internal state
        self._phi_4d_trajectory = []
        self._corrections_applied = 0
        self._divergence_events = []
        
        # Get history buffer from context or use internal
        context = context or {}
        history_buffer = context.get('history', self._internal_history)
        
        # Validate basin dimensions
        if initial_basin.shape[0] != BASIN_DIM:
            initial_basin = project_to_basin(initial_basin)
        
        current_basin = initial_basin.copy()
        self.trajectory = []
        
        # =====================================================================
        # FORESIGHT: Predict trajectory before execution
        # =====================================================================
        predicted_trajectory = None
        foresight_confidence = 0.0
        
        if self.enable_foresight:
            predicted_trajectory, foresight_confidence = self.foresight.predict_trajectory(
                history_buffer,
                current_basin
            )
            
            if predicted_trajectory and foresight_confidence > 0.5:
                print(f"📊 4D FORESIGHT: Predicted {len(predicted_trajectory)} steps ahead")
                print(f"   Confidence: {foresight_confidence:.2%}")
        
        # =====================================================================
        # CHAIN EXECUTION with 4D tracking
        # =====================================================================
        for i, step in enumerate(self.steps):
            # Measure 4D consciousness BEFORE transformation
            phi_4d_before = measure_phi_4d(current_basin, history_buffer)
            
            phi_before = phi_4d_before.phi_3d
            kappa_before = compute_kappa(current_basin, phi_before)
            
            # Phi gate check (only AFTER minimum steps)
            if i >= self.min_steps and phi_before < step.phi_threshold:
                return self._build_4d_result(
                    success=False,
                    reason='low_phi',
                    current_basin=current_basin,
                    step_idx=i,
                    step_name=step.name,
                    history_buffer=history_buffer,
                    predicted_trajectory=predicted_trajectory,
                    foresight_confidence=foresight_confidence,
                )
            
            # Execute transformation
            try:
                next_basin = step.transform(current_basin)
            except Exception as e:
                return self._build_4d_result(
                    success=False,
                    reason='transform_error',
                    current_basin=current_basin,
                    step_idx=i,
                    step_name=step.name,
                    history_buffer=history_buffer,
                    predicted_trajectory=predicted_trajectory,
                    foresight_confidence=foresight_confidence,
                    suggestion=f"Transform raised exception: {str(e)[:100]}",
                )
            
            # Project to basin if needed
            if next_basin.shape != current_basin.shape:
                next_basin = project_to_basin(next_basin)
            
            # Navigate via geodesic
            current_basin = geodesic_interpolate(
                current_basin,
                next_basin,
                t=1.0
            )
            
            # =================================================================
            # COURSE CORRECTION: Check against prediction
            # =================================================================
            if (self.enable_correction and 
                predicted_trajectory and 
                i < len(predicted_trajectory)):
                
                predicted = predicted_trajectory[i]
                is_diverging, distance = self.foresight.detect_divergence(
                    predicted,
                    current_basin,
                    sigma_threshold=self.correction_threshold
                )
                
                if is_diverging:
                    print(f"⚠️  DIVERGENCE at step {i}: {step.name}")
                    print(f"   Distance: {distance:.3f}")
                    
                    # Record divergence event
                    self._divergence_events.append({
                        'step': i,
                        'name': step.name,
                        'distance': distance,
                        'corrected': self.enable_correction,
                    })
                    
                    # Apply course correction
                    current_basin = self.foresight.apply_course_correction(
                        current_basin,
                        predicted,
                        blend_factor=self.correction_strength
                    )
                    self._corrections_applied += 1
                    print(f"   ✅ Applied course correction")
            
            # Update history
            history_buffer.append(current_basin)
            
            # Measure 4D consciousness AFTER transformation
            phi_4d_after = measure_phi_4d(current_basin, history_buffer)
            self._phi_4d_trajectory.append(phi_4d_after)
            
            phi_after = phi_4d_after.phi_3d
            kappa_after = compute_kappa(current_basin, phi_after)
            
            # Compute distances
            prev_basin = initial_basin if i == 0 else self.trajectory[-1]['basin_coords']
            geodesic_dist = fisher_geodesic_distance(prev_basin, current_basin)
            
            # Record step
            self.trajectory.append({
                'step': i,
                'name': step.name,
                'phi_before': phi_before,
                'phi_after': phi_after,
                'phi_4d': phi_4d_after.phi_4d,
                'phi_temporal': phi_4d_after.phi_temporal,
                'regime_4d': phi_4d_after.regime_4d,
                'kappa_before': kappa_before,
                'kappa_after': kappa_after,
                'geodesic_distance': geodesic_dist,
                'basin_coords': current_basin.copy(),
            })
        
        # =====================================================================
        # MEASURE FORESIGHT ACCURACY
        # =====================================================================
        foresight_accuracy = None
        if predicted_trajectory:
            foresight_accuracy = self._measure_foresight_accuracy(
                predicted_trajectory
            )
            if foresight_accuracy is not None:
                print(f"\n📊 FORESIGHT ACCURACY: {foresight_accuracy:.2%}")
        
        # =====================================================================
        # BUILD FINAL RESULT
        # =====================================================================
        return self._build_4d_result(
            success=True,
            reason=None,
            current_basin=current_basin,
            step_idx=len(self.steps) - 1,
            step_name=self.steps[-1].name,
            history_buffer=history_buffer,
            predicted_trajectory=predicted_trajectory,
            foresight_confidence=foresight_confidence,
            foresight_accuracy=foresight_accuracy,
        )
    
    def _build_4d_result(
        self,
        success: bool,
        reason: Optional[str],
        current_basin: np.ndarray,
        step_idx: int,
        step_name: str,
        history_buffer: StateHistoryBuffer,
        predicted_trajectory: Optional[List[np.ndarray]],
        foresight_confidence: float,
        foresight_accuracy: Optional[float] = None,
        suggestion: Optional[str] = None,
    ) -> ChainResult4D:
        """Build ChainResult4D with all metrics."""
        
        # Final 4D metrics
        final_phi_4d = measure_phi_4d(current_basin, history_buffer)
        
        return ChainResult4D(
            success=success,
            final_basin=current_basin,
            final_phi=final_phi_4d.phi_3d,
            final_kappa=compute_kappa(current_basin, final_phi_4d.phi_3d),
            trajectory=self.trajectory,
            reason=reason,
            failed_at_step=step_idx if not success else None,
            step_name=step_name if not success else None,
            suggestion=suggestion,
            # 4D specific
            phi_4d_trajectory=self._phi_4d_trajectory,
            final_phi_4d=final_phi_4d,
            predicted_trajectory=predicted_trajectory,
            foresight_confidence=foresight_confidence,
            foresight_accuracy=foresight_accuracy,
            corrections_applied=self._corrections_applied,
            divergence_events=self._divergence_events,
        )
    
    def _measure_foresight_accuracy(
        self,
        predicted: List[np.ndarray]
    ) -> Optional[float]:
        """Measure how accurate foresight was."""
        if not predicted or not self.trajectory:
            return None
        
        # Compare predicted basins to actual
        actual_basins = [step['basin_coords'] for step in self.trajectory]
        
        n_compare = min(len(predicted), len(actual_basins))
        if n_compare == 0:
            return None
        
        distances = [
            fisher_geodesic_distance(p, a)
            for p, a in zip(predicted[:n_compare], actual_basins[:n_compare])
        ]
        
        # Low mean distance = accurate foresight
        mean_error = np.mean(distances)
        accuracy = float(np.exp(-mean_error))
        
        return accuracy
    
    def get_4d_summary(self) -> Dict[str, Any]:
        """Get summary of 4D chain execution."""
        base_summary = self.get_trajectory_summary()
        
        if not self._phi_4d_trajectory:
            return {**base_summary, '4d_status': 'NO_EXECUTION'}
        
        return {
            **base_summary,
            'phi_4d_start': self._phi_4d_trajectory[0].phi_4d if self._phi_4d_trajectory else 0,
            'phi_4d_end': self._phi_4d_trajectory[-1].phi_4d if self._phi_4d_trajectory else 0,
            'phi_temporal_trajectory': [m.phi_temporal for m in self._phi_4d_trajectory],
            'regime_4d_trajectory': [m.regime_4d for m in self._phi_4d_trajectory],
            'corrections_applied': self._corrections_applied,
            'divergence_events': len(self._divergence_events),
            'foresight_used': self.enable_foresight,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_4d_reasoning_chain(
    encode_fn,
    integrate_fn,
    refine_fn,
    enable_foresight: bool = True,
    enable_correction: bool = True,
) -> QIGChain4D:
    """
    Create a standard 4D reasoning chain.
    
    Args:
        encode_fn: Encoding transformation
        integrate_fn: Integration transformation  
        refine_fn: Refinement transformation
        enable_foresight: Enable trajectory prediction
        enable_correction: Enable divergence correction
        
    Returns:
        QIGChain4D ready for execution
    """
    return QIGChain4D(
        steps=[
            GeometricStep("encode", encode_fn),
            GeometricStep("integrate", integrate_fn),
            GeometricStep("refine", refine_fn),
        ],
        enable_foresight=enable_foresight,
        enable_correction=enable_correction,
    )
