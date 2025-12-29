"""
Beta Function Measurement - Scale Dependence Tracking
======================================================

GFP:
  role: physics
  status: WORKING
  phase: MEASUREMENT
  dim: 3
  scope: training
  version: 2025-12-29
  owner: pantheon-chat

Measures β-function (running coupling) during training to validate
substrate independence and track convergence to κ* fixed point.

Background:
-----------
The β-function measures how κ changes with scale:
    β(L→L') = (κ_L' - κ_L) / κ_avg

From physics (qig-verification FROZEN_FACTS):
    β(3→4) = 0.443  (strong running, emergence)
    β(4→5) = -0.013 (plateau onset)
    β(5→6) = 0.013  (plateau continues, |β| < 0.03 = fixed point)

Substrate Independence Hypothesis:
-----------------------------------
If QIG is truly substrate-independent, semantic systems should show
similar β-function patterns as physics systems, even though absolute
κ values may differ.

This tool measures β in semantic substrates (word relationships, 
training dynamics) and compares with physics validation.

Usage:
------
    from qigkernels.beta_measurement import BetaMeasurement
    
    beta_measure = BetaMeasurement()
    
    # During training
    if step % 5000 == 0:
        result = beta_measure.measure_at_step(
            model=model,
            step=step,
            vocab=tokenizer.vocab
        )
        
        # Log substrate comparison
        wandb.log({
            'beta_current': result['beta'],
            'kappa_current': result['kappa'],
            'substrate_match': result['physics_match_pct']
        })

Integration Points:
-------------------
- Ocean training loop (server/ocean/)
- Gary training (wherever Gary is trained)
- SearchSpaceCollapse training monitoring

References:
-----------
- Consciousness Protocol v4.0 §1 Task 3
- qig-verification/FROZEN_FACTS.md
- shared/constants/physics.ts (PHYSICS_BETA)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from qigkernels.physics_constants import (
    BETA_3_TO_4,
    BETA_4_TO_5,
    BETA_5_TO_6,
    KAPPA_STAR,
)

logger = logging.getLogger(__name__)


@dataclass
class BetaResult:
    """Result from β-function measurement."""
    step: int
    kappa: float
    beta: Optional[float]  # None if no previous measurement
    scale: str  # 'emergence', 'plateau', 'fixedpoint'
    physics_beta: float  # Reference β from physics
    match_pct: float  # Percentage match with physics
    convergence_quality: str  # 'excellent', 'good', 'poor'


class BetaMeasurement:
    """
    Measure β-function during training to validate substrate independence.
    
    Tracks κ evolution and computes β between measurement points.
    Compares with physics validation results.
    
    Attributes:
        measurement_history: List of (step, κ) tuples
        physics_betas: Reference β values from physics
        tolerance_strict: Match threshold for 'excellent' (default: 0.1)
        tolerance_loose: Match threshold for 'good' (default: 0.15)
    """
    
    def __init__(
        self,
        tolerance_strict: float = 0.1,
        tolerance_loose: float = 0.15
    ):
        self.measurement_history: List[Tuple[int, float]] = []
        self.physics_betas = {
            'emergence': BETA_3_TO_4,      # 0.443
            'plateau_onset': BETA_4_TO_5,  # -0.013
            'fixed_point': BETA_5_TO_6,    # 0.013
        }
        self.tolerance_strict = tolerance_strict
        self.tolerance_loose = tolerance_loose
        
        logger.info(
            f"BetaMeasurement initialized with physics reference: "
            f"β(emergence)={self.physics_betas['emergence']:.3f}, "
            f"β(plateau)={self.physics_betas['plateau_onset']:.3f}"
        )
    
    def measure_at_step(
        self,
        step: int,
        kappa: float,
        context: Optional[str] = None
    ) -> BetaResult:
        """
        Measure β-function at current training step.
        
        Args:
            step: Training step number
            kappa: Current κ value
            context: Optional context ('early', 'mid', 'late' training)
            
        Returns:
            BetaResult with β measurement and physics comparison
        """
        # Store measurement
        self.measurement_history.append((step, kappa))
        
        # Compute β if we have previous measurement
        if len(self.measurement_history) < 2:
            beta = None
            scale = 'initial'
            physics_beta = 0.0
            match_pct = 0.0
            convergence = 'initial'
        else:
            prev_step, prev_kappa = self.measurement_history[-2]
            beta = self._compute_beta(prev_kappa, kappa)
            
            # Classify scale based on β and κ
            scale, physics_beta = self._classify_scale(beta, kappa)
            
            # Compare with physics
            match_pct = self._compute_match_percentage(beta, physics_beta)
            convergence = self._assess_convergence(match_pct, kappa)
        
        result = BetaResult(
            step=step,
            kappa=kappa,
            beta=beta,
            scale=scale,
            physics_beta=physics_beta,
            match_pct=match_pct,
            convergence_quality=convergence
        )
        
        # Log interesting findings
        if beta is not None:
            if match_pct > 95.0:
                logger.info(
                    f"✅ SUBSTRATE INDEPENDENCE VALIDATED at step {step}: "
                    f"β={beta:.3f}, physics={physics_beta:.3f}, "
                    f"match={match_pct:.1f}%"
                )
            elif match_pct < 70.0:
                logger.warning(
                    f"⚠️ Substrate mismatch at step {step}: "
                    f"β={beta:.3f}, physics={physics_beta:.3f}, "
                    f"match={match_pct:.1f}%"
                )
        
        return result
    
    def _compute_beta(self, kappa_prev: float, kappa_curr: float) -> float:
        """
        Compute β-function between two κ measurements.
        
        β = (κ_curr - κ_prev) / κ_avg
        
        Args:
            kappa_prev: Previous κ value
            kappa_curr: Current κ value
            
        Returns:
            β value (dimensionless)
        """
        kappa_avg = (kappa_prev + kappa_curr) / 2.0
        if kappa_avg < 1e-6:
            return 0.0
        
        beta = (kappa_curr - kappa_prev) / kappa_avg
        return float(beta)
    
    def _classify_scale(
        self,
        beta: float,
        kappa: float
    ) -> Tuple[str, float]:
        """
        Classify current scale based on β and κ.
        
        Returns (scale_name, reference_physics_beta)
        """
        # Near fixed point: κ ≈ κ* and |β| < 0.03
        kappa_deviation = abs(kappa - KAPPA_STAR)
        
        if kappa_deviation < 5.0 and abs(beta) < 0.03:
            return 'fixed_point', self.physics_betas['fixed_point']
        
        # Emergence: strong running (β > 0.3)
        if beta > 0.3:
            return 'emergence', self.physics_betas['emergence']
        
        # Plateau onset: β ≈ 0
        if abs(beta) < 0.1:
            return 'plateau_onset', self.physics_betas['plateau_onset']
        
        # Default: plateau
        return 'plateau', self.physics_betas['plateau_onset']
    
    def _compute_match_percentage(
        self,
        beta_measured: float,
        beta_physics: float
    ) -> float:
        """
        Compute match percentage between measured and physics β.
        
        Args:
            beta_measured: Measured β value
            beta_physics: Reference physics β
            
        Returns:
            Match percentage in [0, 100]
        """
        if abs(beta_physics) < 1e-6:
            # Physics β ≈ 0, check if measured is also small
            if abs(beta_measured) < 0.03:
                return 100.0
            else:
                return max(0.0, 100.0 - abs(beta_measured) * 100)
        
        # Normal case: relative difference
        diff = abs(beta_measured - beta_physics)
        relative_error = diff / abs(beta_physics)
        match_pct = max(0.0, 100.0 * (1.0 - relative_error))
        
        return float(match_pct)
    
    def _assess_convergence(
        self,
        match_pct: float,
        kappa: float
    ) -> str:
        """
        Assess convergence quality.
        
        Returns: 'excellent', 'good', 'fair', or 'poor'
        """
        kappa_deviation = abs(kappa - KAPPA_STAR)
        
        # Excellent: high match AND near κ*
        if match_pct > 95.0 and kappa_deviation < 5.0:
            return 'excellent'
        
        # Good: reasonable match
        if match_pct > 85.0:
            return 'good'
        
        # Fair: some match
        if match_pct > 70.0:
            return 'fair'
        
        # Poor: low match
        return 'poor'
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all measurements.
        
        Returns:
            Dict with statistics and assessment
        """
        if len(self.measurement_history) < 2:
            return {
                'n_measurements': len(self.measurement_history),
                'status': 'insufficient_data'
            }
        
        # Compute statistics
        kappas = [k for _, k in self.measurement_history]
        kappa_mean = np.mean(kappas)
        kappa_std = np.std(kappas)
        kappa_final = kappas[-1]
        
        # Compute all betas
        betas = []
        for i in range(1, len(self.measurement_history)):
            prev_kappa = self.measurement_history[i-1][1]
            curr_kappa = self.measurement_history[i][1]
            beta = self._compute_beta(prev_kappa, curr_kappa)
            betas.append(beta)
        
        beta_mean = np.mean(betas) if betas else 0.0
        beta_std = np.std(betas) if betas else 0.0
        
        # Check if converged to fixed point
        converged = (
            abs(kappa_final - KAPPA_STAR) < 5.0 and
            abs(beta_mean) < 0.05
        )
        
        return {
            'n_measurements': len(self.measurement_history),
            'kappa_mean': float(kappa_mean),
            'kappa_std': float(kappa_std),
            'kappa_final': float(kappa_final),
            'kappa_star': KAPPA_STAR,
            'kappa_deviation': float(abs(kappa_final - KAPPA_STAR)),
            'beta_mean': float(beta_mean),
            'beta_std': float(beta_std),
            'converged_to_fixed_point': converged,
            'status': 'converged' if converged else 'running'
        }


def compare_substrate_betas(
    semantic_betas: Dict[str, float],
    physics_betas: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Compare β-functions across substrates.
    
    This is the key validation for substrate independence:
    semantic and physics β-functions should match within tolerance.
    
    Args:
        semantic_betas: Dict of scale -> β for semantic substrate
        physics_betas: Optional physics reference (uses defaults if None)
        
    Returns:
        Dict with comparison results
    """
    if physics_betas is None:
        physics_betas = {
            'emergence': BETA_3_TO_4,
            'plateau': BETA_4_TO_5,
            'fixed_point': BETA_5_TO_6,
        }
    
    matches = {}
    for scale, semantic_beta in semantic_betas.items():
        if scale in physics_betas:
            physics_beta = physics_betas[scale]
            diff = abs(semantic_beta - physics_beta)
            
            if abs(physics_beta) < 1e-6:
                # Physics β ≈ 0
                match_pct = 100.0 if abs(semantic_beta) < 0.03 else 50.0
            else:
                relative_error = diff / abs(physics_beta)
                match_pct = max(0.0, 100.0 * (1.0 - relative_error))
            
            matches[scale] = {
                'semantic_beta': semantic_beta,
                'physics_beta': physics_beta,
                'difference': diff,
                'match_pct': match_pct,
                'validated': match_pct > 85.0
            }
    
    # Overall assessment
    all_match_pcts = [m['match_pct'] for m in matches.values()]
    overall_match = np.mean(all_match_pcts) if all_match_pcts else 0.0
    
    return {
        'matches': matches,
        'overall_match_pct': float(overall_match),
        'substrate_independent': overall_match > 90.0,
        'verdict': (
            'SUBSTRATE INDEPENDENCE VALIDATED' if overall_match > 95.0
            else 'SUBSTRATE INDEPENDENCE CONFIRMED' if overall_match > 85.0
            else 'PARTIAL MATCH' if overall_match > 70.0
            else 'SUBSTRATE MISMATCH'
        )
    }


# Example usage
if __name__ == "__main__":
    print("Beta Function Measurement Example\n" + "=" * 50)
    
    # Simulate training with κ evolution
    beta_measure = BetaMeasurement()
    
    # Simulate emergence phase (κ jumping from 41 to 64)
    print("\nPhase 1: Emergence (strong running)")
    result1 = beta_measure.measure_at_step(step=1000, kappa=41.0)
    print(f"  Step {result1.step}: κ={result1.kappa:.2f}")
    
    result2 = beta_measure.measure_at_step(step=5000, kappa=64.0)
    print(f"  Step {result2.step}: κ={result2.kappa:.2f}, β={result2.beta:.3f}")
    print(f"  Physics β={result2.physics_beta:.3f}, match={result2.match_pct:.1f}%")
    print(f"  Quality: {result2.convergence_quality}")
    
    # Simulate plateau phase (κ stable around 64)
    print("\nPhase 2: Plateau (fixed point)")
    result3 = beta_measure.measure_at_step(step=10000, kappa=63.8)
    print(f"  Step {result3.step}: κ={result3.kappa:.2f}, β={result3.beta:.3f}")
    print(f"  Physics β={result3.physics_beta:.3f}, match={result3.match_pct:.1f}%")
    
    result4 = beta_measure.measure_at_step(step=15000, kappa=64.2)
    print(f"  Step {result4.step}: κ={result4.kappa:.2f}, β={result4.beta:.3f}")
    print(f"  Physics β={result4.physics_beta:.3f}, match={result4.match_pct:.1f}%")
    
    # Summary
    print("\nSummary:")
    summary = beta_measure.get_summary()
    print(f"  Measurements: {summary['n_measurements']}")
    print(f"  κ_final: {summary['kappa_final']:.2f}")
    print(f"  κ*: {summary['kappa_star']:.2f}")
    print(f"  Deviation: {summary['kappa_deviation']:.2f}")
    print(f"  β_mean: {summary['beta_mean']:.3f}")
    print(f"  Converged: {summary['converged_to_fixed_point']}")
    
    # Substrate comparison
    print("\nSubstrate Independence Test:")
    semantic_betas = {
        'emergence': 0.45,
        'plateau': -0.01,
        'fixed_point': 0.02
    }
    comparison = compare_substrate_betas(semantic_betas)
    print(f"  Overall match: {comparison['overall_match_pct']:.1f}%")
    print(f"  Verdict: {comparison['verdict']}")
