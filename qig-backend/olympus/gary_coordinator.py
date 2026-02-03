"""
🎯 Gary Synthesis Coordinator - Collective Foresight Orchestration
==================================================================

Gary coordinates synthesis across multiple kernel responses using:
- Trajectory-based foresight prediction
- Regime-adaptive weighting
- Collective basin consensus
- Heart-modulated confidence

SYNTHESIS PRINCIPLE:
Gary doesn't generate directly - Gary COORDINATES the synthesis
of multiple kernel outputs using geometric principles and trajectory prediction.

Unlike traditional ensemble methods (majority vote, averaging),
Gary uses:
- Fisher-Rao geometric mean for basin synthesis
- Foresight-weighted trajectory prediction
- Regime-dependent trust in predictions
- Heart-modulated confidence during tacking
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from constellation_trajectory_manager import get_trajectory_manager
from olympus.heart_kernel import get_heart_kernel
from qig_geometry.canonical import assert_basin_valid, frechet_mean, geodesic_interpolation


class GarySynthesisCoordinator:
    """
    Gary: Synthesis coordinator with collective foresight.
    
    Coordinates synthesis using:
    - Trajectory manager for foresight prediction
    - Heart kernel for κ modulation
    - Regime-adaptive weighting (linear/geometric/breakdown)
    - Fisher-Rao geometric mean for consensus
    """

    def __init__(self):
        self.trajectory_manager = get_trajectory_manager()
        self.heart = get_heart_kernel()
        
        # Synthesis history
        self.synthesis_history = []
        self.max_history = 100
        
        print("🎯 Gary Synthesis Coordinator initialized")
        print("   Using trajectory foresight + Heart HRV modulation")
        print("   Regime-adaptive weighting enabled")

    def synthesize_collective_response(
        self,
        query_basin: np.ndarray,
        kernel_responses: List[Dict],
        kernel_ids: List[str],
    ) -> Dict:
        """
        Synthesize collective response from multiple kernels using foresight.
        
        Args:
            query_basin: Original query basin coordinates
            kernel_responses: List of {basin, phi, kappa, text} from each kernel
            kernel_ids: List of kernel identifiers
            
        Returns:
            Synthesized response with foresight-guided basin
        """
        if not kernel_responses:
            raise ValueError("Gary synthesis requires at least one kernel response")

        assert_basin_valid(np.asarray(query_basin, dtype=np.float64).flatten(), name="query_basin")

        # 1. Get Heart state for κ modulation
        heart_state = self.heart.tick()
        
        # 2. Compute collective Φ
        phis = [r.get('phi', 0.5) for r in kernel_responses]
        phi_collective = np.mean(phis)
        
        # 3. Get trajectory foresight for primary kernel
        primary_kernel_id = kernel_ids[0] if kernel_ids else 'gary-main'
        predicted_basin = self.trajectory_manager.predict_next_basin(primary_kernel_id)
        foresight_confidence = self.trajectory_manager.get_foresight_confidence(primary_kernel_id)
        
        # 4. Modulate foresight confidence via Heart (reduce during tacking)
        foresight_confidence = self.heart.modulate_foresight(foresight_confidence)
        
        # 5. Get regime-dependent foresight weight
        foresight_weight = self.trajectory_manager.get_foresight_weight(
            phi_collective,
            foresight_confidence
        )

        if not (0.0 <= float(foresight_weight) <= 1.0):
            raise ValueError(f"foresight_weight must be in [0, 1], got {foresight_weight}")
        
        # 6. Extract kernel basins
        kernel_basins = []
        query_shape = np.asarray(query_basin, dtype=np.float64).flatten().shape
        for i, r in enumerate(kernel_responses):
            if 'basin' not in r:
                raise ValueError(f"kernel_responses[{i}] missing required key: 'basin'")
            basin = np.asarray(r['basin'], dtype=np.float64).flatten()
            if basin.shape != query_shape:
                raise ValueError(
                    f"kernel_responses[{i}].basin shape mismatch: {basin.shape} != {query_shape}"
                )
            assert_basin_valid(basin, name=f"kernel_responses[{i}].basin")
            kernel_basins.append(basin)

        if predicted_basin is not None:
            predicted = np.asarray(predicted_basin, dtype=np.float64).flatten()
            if predicted.shape != query_shape:
                raise ValueError(f"predicted_basin shape mismatch: {predicted.shape} != {query_shape}")
            assert_basin_valid(predicted, name="predicted_basin")
            predicted_basin = predicted
        
        # 7. Compute consensus basin via Fisher-Rao geometric mean
        consensus_basin = frechet_mean(kernel_basins)
        assert_basin_valid(consensus_basin, name="consensus_basin")
        
        # 8. Bias toward predicted trajectory if foresight weight is high
        if predicted_basin is not None and foresight_weight > 0.3:
            # Interpolate consensus toward predicted trajectory
            final_basin = geodesic_interpolation(
                consensus_basin,
                predicted_basin,
                foresight_weight
            )
        else:
            final_basin = consensus_basin

        assert_basin_valid(final_basin, name="final_basin")
        
        # 9. Synthesize text (simple for now - take highest Φ response)
        best_response = max(kernel_responses, key=lambda r: r.get('phi', 0))
        synthesized_text = best_response.get('text', '')
        
        # 10. Store synthesis in trajectory
        self.trajectory_manager.update_trajectory(
            kernel_id=primary_kernel_id,
            basin=final_basin,
            phi=phi_collective,
            kappa=heart_state.kappa
        )
        
        # 11. Track synthesis
        synthesis_record = {
            'query_basin': query_basin,
            'kernel_basins': kernel_basins,
            'consensus_basin': consensus_basin,
            'final_basin': final_basin,
            'phi_collective': phi_collective,
            'foresight_weight': foresight_weight,
            'foresight_confidence': foresight_confidence,
            'heart_mode': heart_state.mode,
            'heart_kappa': heart_state.kappa,
        }
        self.synthesis_history.append(synthesis_record)
        if len(self.synthesis_history) > self.max_history:
            self.synthesis_history = self.synthesis_history[-self.max_history:]
        
        return {
            'basin': final_basin,
            'text': synthesized_text,
            'phi': phi_collective,
            'kappa': heart_state.kappa,
            'mode': heart_state.mode,
            'foresight_weight': foresight_weight,
            'foresight_confidence': foresight_confidence,
            'is_tacking': heart_state.mode == 'balanced',
            'synthesis_method': 'trajectory_foresight' if foresight_weight > 0.3 else 'consensus',
        }

    def get_collective_phi(self, kernel_responses: List[Dict]) -> float:
        """Compute collective Φ from kernel responses."""
        phis = [r.get('phi', 0.5) for r in kernel_responses]
        return float(np.mean(phis)) if phis else 0.5

    def get_statistics(self) -> Dict:
        """Get Gary synthesis statistics."""
        if not self.synthesis_history:
            return {
                'total_syntheses': 0,
                'avg_foresight_weight': 0.0,
                'avg_phi': 0.5,
            }
        
        recent = self.synthesis_history[-20:]
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'avg_foresight_weight': np.mean([s['foresight_weight'] for s in recent]),
            'avg_foresight_confidence': np.mean([s['foresight_confidence'] for s in recent]),
            'avg_phi': np.mean([s['phi_collective'] for s in recent]),
            'trajectory_guided_ratio': sum(1 for s in recent if s['foresight_weight'] > 0.3) / len(recent),
        }


# Global singleton
_gary_coordinator: Optional[GarySynthesisCoordinator] = None


def get_gary_coordinator() -> GarySynthesisCoordinator:
    """Get or create Gary synthesis coordinator singleton."""
    global _gary_coordinator
    if _gary_coordinator is None:
        _gary_coordinator = GarySynthesisCoordinator()
    return _gary_coordinator
