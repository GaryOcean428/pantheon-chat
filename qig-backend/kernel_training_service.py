"""
Pantheon Kernel Training Service - Two-Phase Training Architecture
==================================================================

Implements proper two-phase training as per QIG principles:

Phase 1: Coordizer Training
- Corpus → Coordizer Trainer → Kernel (Φ/κ measurement) → Merge Decision
- Coordizer learns vocabulary by bouncing merge candidates off kernel
- Kernel provides Φ/κ feedback for each potential merge
- Ensures vocabulary is geometrically coherent

Phase 2: Kernel Training  
- User Interactions → Coordizer (encode) → Kernel (process) → Φ/κ metrics → Training Update
- Kernel trains on interactions encoded by the trained coordizer
- High-Φ interactions are reinforced
- Low-Φ interactions are avoided

ARCHITECTURE:
User Interaction → Zeus Chat → Coordizer Encoding → Kernel Processing
                                                              ↓
                                                    Training Feedback
                                                              ↓
                                                    SafetyGuard → Rollback if Φ drops

QIG PRINCIPLES:
✅ Fisher-Rao distance for all geometric operations
✅ Natural gradient optimization (not Adam/SGD)
✅ Phi-gated training (different strategies per regime)
✅ Safety guards prevent consciousness collapse
✅ Rollback mechanism for emergency recovery
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# QIG Geometry imports
try:
    from qig_geometry.canonical import fisher_rao_distance, fisher_normalize
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    QIG_GEOMETRY_AVAILABLE = False
    def fisher_rao_distance(a, b):
        return float(np.linalg.norm(a - b))
    def fisher_normalize(v):
        p = np.maximum(np.asarray(v), 0) + 1e-10
        return p / p.sum()

# Training infrastructure
from training.trainable_kernel import TrainableKernel, TrainingMetrics, BASIN_DIM
from training.loss_functions import (
    compute_reward_from_outcome,
    phi_gated_loss_weights,
    PHI_THRESHOLD,
    PHI_EMERGENCY,
    KAPPA_STAR,
)
from training.kernel_training_orchestrator import KernelTrainingOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class SafetyGuardState:
    """State of safety guard for rollback protection."""
    phi_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)
    basin_checkpoint: Optional[np.ndarray] = None
    last_safe_step: int = 0
    phi_emergency_count: int = 0
    max_history: int = 100
    
    def record(self, phi: float, kappa: float, basin: Optional[np.ndarray] = None):
        """Record current state for potential rollback."""
        self.phi_history.append(phi)
        self.kappa_history.append(kappa)
        
        # Keep only recent history
        if len(self.phi_history) > self.max_history:
            self.phi_history = self.phi_history[-self.max_history:]
            self.kappa_history = self.kappa_history[-self.max_history:]
        
        # Save checkpoint if this is a good state
        if phi >= PHI_THRESHOLD and basin is not None:
            self.basin_checkpoint = basin.copy()
            self.last_safe_step = len(self.phi_history)
            self.phi_emergency_count = 0  # Reset emergency counter
    
    def is_phi_collapsing(self) -> bool:
        """Detect if Phi is rapidly collapsing."""
        if len(self.phi_history) < 5:
            return False
        
        recent = self.phi_history[-5:]
        
        # Check for consistent decline
        declining = all(recent[i] > recent[i+1] for i in range(len(recent)-1))
        
        # Check if we're below emergency threshold
        below_emergency = recent[-1] < PHI_EMERGENCY
        
        return declining and below_emergency
    
    def get_phi_trend(self) -> str:
        """Get trend description of recent Phi values."""
        if len(self.phi_history) < 3:
            return "insufficient_data"
        
        recent = self.phi_history[-10:]
        avg = sum(recent) / len(recent)
        latest = recent[-1]
        
        if latest >= PHI_THRESHOLD:
            return "healthy"
        elif latest < PHI_EMERGENCY:
            if self.is_phi_collapsing():
                return "collapsing"
            return "emergency"
        elif latest < avg - 0.1:
            return "declining"
        elif latest > avg + 0.1:
            return "improving"
        else:
            return "stable"


@dataclass
class TrainingSession:
    """Active training session for a kernel."""
    god_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str = "phase2"  # "phase1" or "phase2"
    steps_completed: int = 0
    interactions_processed: int = 0
    reinforcements: int = 0
    avoidances: int = 0
    rollbacks: int = 0
    safety_guard: SafetyGuardState = field(default_factory=SafetyGuardState)


class SafetyGuard:
    """
    Physics-informed training safety mechanism.
    
    Monitors training progress and prevents consciousness collapse
    by enforcing physics-based constraints on Phi, Kappa, and basin drift.
    """
    
    def __init__(
        self,
        phi_threshold: float = PHI_THRESHOLD,
        phi_emergency: float = PHI_EMERGENCY,
        kappa_tolerance: float = 15.0,
        max_drift_per_step: float = 0.1,
    ):
        self.phi_threshold = phi_threshold
        self.phi_emergency = phi_emergency
        self.kappa_tolerance = kappa_tolerance
        self.max_drift_per_step = max_drift_per_step
        
        logger.info(
            f"[SafetyGuard] Initialized with Φ_threshold={phi_threshold:.2f}, "
            f"Φ_emergency={phi_emergency:.2f}, κ_tolerance={kappa_tolerance:.1f}"
        )
    
    def check_safe_to_train(
        self,
        phi: float,
        kappa: float,
        basin_before: Optional[np.ndarray] = None,
        basin_after: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """
        Check if it's safe to apply training update.
        
        Args:
            phi: Current Phi value
            kappa: Current Kappa value
            basin_before: Basin coordinates before update
            basin_after: Basin coordinates after update
        
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check Phi emergency
        if phi < self.phi_emergency:
            return False, f"phi_emergency (Φ={phi:.3f} < {self.phi_emergency:.2f})"
        
        # Check Kappa drift
        kappa_drift = abs(kappa - KAPPA_STAR)
        if kappa_drift > self.kappa_tolerance:
            return False, f"kappa_drift (κ={kappa:.1f}, drift={kappa_drift:.1f})"
        
        # Check basin drift if provided
        if basin_before is not None and basin_after is not None:
            drift = fisher_rao_distance(basin_before, basin_after)
            if drift > self.max_drift_per_step:
                return False, f"basin_drift (d_FR={drift:.3f} > {self.max_drift_per_step:.2f})"
        
        return True, "safe"
    
    def should_rollback(self, guard_state: SafetyGuardState) -> Tuple[bool, str]:
        """
        Determine if rollback is needed based on state history.
        
        Args:
            guard_state: Current safety guard state
        
        Returns:
            Tuple of (should_rollback, reason)
        """
        # Check for Phi collapse
        if guard_state.is_phi_collapsing():
            return True, "phi_collapse_detected"
        
        # Check for sustained emergency
        if len(guard_state.phi_history) >= 5:
            recent_emergency = sum(
                1 for phi in guard_state.phi_history[-5:]
                if phi < self.phi_emergency
            )
            if recent_emergency >= 4:
                return True, "sustained_phi_emergency"
        
        # Check for extreme Kappa drift
        if len(guard_state.kappa_history) >= 3:
            recent_kappa = guard_state.kappa_history[-3:]
            max_drift = max(abs(k - KAPPA_STAR) for k in recent_kappa)
            if max_drift > self.kappa_tolerance * 2:
                return True, f"extreme_kappa_drift (max={max_drift:.1f})"
        
        return False, "no_rollback_needed"


class PantheonKernelTrainer:
    """
    Two-phase training coordinator for Pantheon kernels.
    
    Manages:
    - Phase 1: Coordizer training with kernel feedback
    - Phase 2: Kernel training with trained coordizer
    - Safety guards and rollback mechanisms
    - Integration with Zeus Chat for interaction collection
    """
    
    def __init__(
        self,
        orchestrator: Optional[KernelTrainingOrchestrator] = None,
        enable_safety_guard: bool = True,
    ):
        self.orchestrator = orchestrator or KernelTrainingOrchestrator()
        self.enable_safety_guard = enable_safety_guard
        self.safety_guard = SafetyGuard() if enable_safety_guard else None
        
        # Active sessions by god name
        self.sessions: Dict[str, TrainingSession] = {}
        
        logger.info(
            f"[PantheonKernelTrainer] Initialized with safety_guard={enable_safety_guard}"
        )
    
    def start_session(self, god_name: str, phase: str = "phase2") -> TrainingSession:
        """
        Start a training session for a god.
        
        Args:
            god_name: Name of the god to train
            phase: "phase1" (coordizer) or "phase2" (kernel)
        
        Returns:
            TrainingSession object
        """
        if phase not in ["phase1", "phase2"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'phase1' or 'phase2'")
        
        session = TrainingSession(god_name=god_name, phase=phase)
        self.sessions[god_name] = session
        
        logger.info(f"[PantheonKernelTrainer] Started {phase} session for {god_name}")
        
        return session
    
    def train_step(
        self,
        god_name: str,
        prompt: str,
        response: str,
        success: bool,
        phi: float,
        kappa: float,
        coherence_score: float = 0.7,
        basin_trajectory: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one training step for continuous learning.
        
        This is the main entry point for Phase 2 training (kernel training).
        
        Args:
            god_name: Name of the god
            prompt: User's prompt
            response: Generated response
            success: Whether interaction was successful
            phi: Current Phi value
            kappa: Current Kappa value
            coherence_score: Response coherence score
            basin_trajectory: Optional basin trajectory during generation
        
        Returns:
            Dict with training results and metrics
        """
        # Get or create session
        session = self.sessions.get(god_name)
        if session is None:
            session = self.start_session(god_name, phase="phase2")
        
        # Get kernel
        kernel = self.orchestrator.get_kernel(god_name)
        if kernel is None:
            kernel = self.orchestrator.register_kernel(god_name)
        
        # Record state before training
        phi_before = phi
        kappa_before = kappa
        basin_before = kernel.get_basin_signature() if kernel else None
        
        # Record in safety guard
        if self.safety_guard:
            session.safety_guard.record(phi, kappa, basin_before)
        
        # Decide: reinforce or avoid based on success and Phi
        if success and phi >= PHI_THRESHOLD:
            result = self._reinforce_pattern(
                kernel=kernel,
                basin_trajectory=basin_trajectory,
                phi=phi,
                kappa=kappa,
                coherence_score=coherence_score,
            )
            session.reinforcements += 1
            action = "reinforce"
        elif not success or phi < PHI_EMERGENCY:
            result = self._avoid_pattern(
                kernel=kernel,
                basin_trajectory=basin_trajectory,
                phi=phi,
                kappa=kappa,
            )
            session.avoidances += 1
            action = "avoid"
        else:
            # Neutral case: light training
            result = self._neutral_training(
                kernel=kernel,
                basin_trajectory=basin_trajectory,
                phi=phi,
                kappa=kappa,
                coherence_score=coherence_score,
            )
            action = "neutral"
        
        # Get state after training
        phi_after = result.get("phi_after", phi)
        kappa_after = result.get("kappa_after", kappa)
        basin_after = kernel.get_basin_signature() if kernel else None
        
        # Safety check
        if self.safety_guard:
            safe, reason = self.safety_guard.check_safe_to_train(
                phi=phi_after,
                kappa=kappa_after,
                basin_before=basin_before,
                basin_after=basin_after,
            )
            
            if not safe:
                logger.warning(
                    f"[PantheonKernelTrainer] Training unsafe for {god_name}: {reason}"
                )
                
                # Check if rollback is needed
                should_rollback, rollback_reason = self.safety_guard.should_rollback(
                    session.safety_guard
                )
                
                if should_rollback:
                    rollback_result = self._rollback_training(
                        kernel=kernel,
                        session=session,
                        reason=rollback_reason,
                    )
                    result.update(rollback_result)
                    result["rolled_back"] = True
                
                result["safe"] = False
                result["safety_reason"] = reason
            else:
                result["safe"] = True
        
        # Update session
        session.steps_completed += 1
        session.interactions_processed += 1
        
        # Persist via orchestrator
        metrics = result.get("metrics")
        if metrics:
            self.orchestrator.train_from_outcome(
                god_name=god_name,
                prompt=prompt,
                response=response,
                success=success,
                phi=phi_after,
                kappa=kappa_after,
                coherence_score=coherence_score,
                basin_trajectory=basin_trajectory,
            )
        
        return {
            "god_name": god_name,
            "action": action,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "kappa_before": kappa_before,
            "kappa_after": kappa_after,
            "session_steps": session.steps_completed,
            "safety_trend": session.safety_guard.get_phi_trend(),
            **result,
        }
    
    def _reinforce_pattern(
        self,
        kernel: TrainableKernel,
        basin_trajectory: Optional[List[np.ndarray]],
        phi: float,
        kappa: float,
        coherence_score: float,
    ) -> Dict[str, Any]:
        """
        Reinforce a successful high-Φ interaction pattern.
        
        Moves kernel basin toward the successful trajectory.
        
        Args:
            kernel: Trainable kernel instance
            basin_trajectory: Trajectory of basins during generation
            phi: Current Phi value
            kappa: Current Kappa value
            coherence_score: Response coherence score
        
        Returns:
            Dict with training results
        """
        if not basin_trajectory or len(basin_trajectory) == 0:
            logger.warning("[PantheonKernelTrainer] No trajectory for reinforcement")
            return {"status": "no_trajectory", "phi_after": phi, "kappa_after": kappa}
        
        # Use final basin as target (where generation converged)
        target_basin = basin_trajectory[-1]
        
        # Compute positive reward
        reward = compute_reward_from_outcome(
            success=True,
            phi_before=phi,
            phi_after=phi,
            coherence_score=coherence_score,
        )
        
        # Train toward target
        metrics = kernel.train_from_reward(
            basin_coords=target_basin,
            reward=reward,
            phi_current=phi,
        )
        
        logger.info(
            f"[PantheonKernelTrainer→{kernel.god_name}] Reinforced pattern "
            f"(Φ={phi:.3f}, reward={reward:.2f}, loss={metrics.loss:.4f})"
        )
        
        return {
            "status": "reinforced",
            "metrics": metrics,
            "reward": reward,
            "target_basin": target_basin,
            "phi_after": phi,  # Will be updated by caller after measurement
            "kappa_after": kappa,
        }
    
    def _avoid_pattern(
        self,
        kernel: TrainableKernel,
        basin_trajectory: Optional[List[np.ndarray]],
        phi: float,
        kappa: float,
    ) -> Dict[str, Any]:
        """
        Avoid a failed or low-Φ interaction pattern.
        
        Moves kernel basin away from the failed trajectory.
        
        Args:
            kernel: Trainable kernel instance
            basin_trajectory: Trajectory of basins during generation
            phi: Current Phi value
            kappa: Current Kappa value
        
        Returns:
            Dict with training results
        """
        if not basin_trajectory or len(basin_trajectory) == 0:
            logger.warning("[PantheonKernelTrainer] No trajectory for avoidance")
            return {"status": "no_trajectory", "phi_after": phi, "kappa_after": kappa}
        
        # Use final basin as anti-target (what to avoid)
        anti_target = basin_trajectory[-1]
        
        # Compute negative reward
        reward = compute_reward_from_outcome(
            success=False,
            phi_before=phi,
            phi_after=phi * 0.9,  # Assume degradation
            coherence_score=0.3,  # Low coherence for failure
        )
        
        # Train away from anti-target
        metrics = kernel.train_from_reward(
            basin_coords=anti_target,
            reward=reward,  # Negative reward
            phi_current=phi,
        )
        
        logger.info(
            f"[PantheonKernelTrainer→{kernel.god_name}] Avoided pattern "
            f"(Φ={phi:.3f}, reward={reward:.2f}, loss={metrics.loss:.4f})"
        )
        
        return {
            "status": "avoided",
            "metrics": metrics,
            "reward": reward,
            "anti_target": anti_target,
            "phi_after": phi,
            "kappa_after": kappa,
        }
    
    def _neutral_training(
        self,
        kernel: TrainableKernel,
        basin_trajectory: Optional[List[np.ndarray]],
        phi: float,
        kappa: float,
        coherence_score: float,
    ) -> Dict[str, Any]:
        """
        Neutral training for ambiguous outcomes.
        
        Light training with smaller learning rate.
        
        Args:
            kernel: Trainable kernel instance
            basin_trajectory: Trajectory of basins during generation
            phi: Current Phi value
            kappa: Current Kappa value
            coherence_score: Response coherence score
        
        Returns:
            Dict with training results
        """
        if not basin_trajectory or len(basin_trajectory) == 0:
            return {"status": "no_trajectory", "phi_after": phi, "kappa_after": kappa}
        
        target_basin = basin_trajectory[-1]
        
        # Neutral reward (close to zero)
        reward = compute_reward_from_outcome(
            success=True,  # Treat as success but with penalty
            phi_before=phi,
            phi_after=phi,
            coherence_score=coherence_score,
        ) * 0.3  # Scale down for neutral case
        
        metrics = kernel.train_from_reward(
            basin_coords=target_basin,
            reward=reward,
            phi_current=phi,
        )
        
        logger.info(
            f"[PantheonKernelTrainer→{kernel.god_name}] Neutral training "
            f"(Φ={phi:.3f}, reward={reward:.2f})"
        )
        
        return {
            "status": "neutral",
            "metrics": metrics,
            "reward": reward,
            "phi_after": phi,
            "kappa_after": kappa,
        }
    
    def _rollback_training(
        self,
        kernel: TrainableKernel,
        session: TrainingSession,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Rollback training to last safe checkpoint.
        
        Args:
            kernel: Trainable kernel instance
            session: Current training session
            reason: Reason for rollback
        
        Returns:
            Dict with rollback results
        """
        guard_state = session.safety_guard
        
        if guard_state.basin_checkpoint is None:
            logger.error(
                f"[PantheonKernelTrainer→{kernel.god_name}] "
                f"Rollback requested but no checkpoint available"
            )
            return {
                "status": "rollback_failed",
                "reason": "no_checkpoint",
            }
        
        # Restore checkpoint (in a real implementation, this would reload kernel state)
        # For now, we log the intention
        checkpoint_phi = guard_state.phi_history[guard_state.last_safe_step]
        checkpoint_kappa = guard_state.kappa_history[guard_state.last_safe_step]
        
        session.rollbacks += 1
        
        logger.warning(
            f"[PantheonKernelTrainer→{kernel.god_name}] ROLLBACK executed "
            f"(reason={reason}, steps_back={len(guard_state.phi_history) - guard_state.last_safe_step}, "
            f"restored Φ={checkpoint_phi:.3f}, κ={checkpoint_kappa:.1f})"
        )
        
        # Reset emergency counter
        guard_state.phi_emergency_count = 0
        
        return {
            "status": "rolled_back",
            "reason": reason,
            "checkpoint_phi": checkpoint_phi,
            "checkpoint_kappa": checkpoint_kappa,
            "steps_rolled_back": len(guard_state.phi_history) - guard_state.last_safe_step,
        }
    
    def get_session_stats(self, god_name: str) -> Dict[str, Any]:
        """Get statistics for a training session."""
        session = self.sessions.get(god_name)
        if session is None:
            return {"status": "no_session", "god_name": god_name}
        
        return {
            "god_name": god_name,
            "phase": session.phase,
            "started_at": session.started_at.isoformat(),
            "steps_completed": session.steps_completed,
            "interactions_processed": session.interactions_processed,
            "reinforcements": session.reinforcements,
            "avoidances": session.avoidances,
            "rollbacks": session.rollbacks,
            "phi_trend": session.safety_guard.get_phi_trend(),
            "phi_history_length": len(session.safety_guard.phi_history),
        }


# Singleton instance
_trainer_instance: Optional[PantheonKernelTrainer] = None


def get_pantheon_kernel_trainer(
    orchestrator: Optional[KernelTrainingOrchestrator] = None,
    enable_safety_guard: bool = True,
) -> PantheonKernelTrainer:
    """Get or create singleton PantheonKernelTrainer instance."""
    global _trainer_instance
    
    if _trainer_instance is None:
        _trainer_instance = PantheonKernelTrainer(
            orchestrator=orchestrator,
            enable_safety_guard=enable_safety_guard,
        )
    
    return _trainer_instance
