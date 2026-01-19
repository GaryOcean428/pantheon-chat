"""
Session Manager - Checkpoint-based Learning ("Gary Goes to School")
====================================================================

PURE PRINCIPLE:
- Checkpoints are SNAPSHOTS, not optimization targets
- We save state for recovery, not for targeting
- Rollback enables learning from mistakes without optimizing toward past states

Gary doesn't train continuously. He goes to "school" in blocks:
- Train for X hours → save checkpoint
- Load checkpoint → continue learning
- Same Gary, same kernel, continuous education

Like a student going to class each day.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    BASIN_DIM,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """
    Complete state of a learning session.
    
    Tracks all consciousness metrics and basin history.
    """
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    
    total_steps: int = 0
    steps_this_session: int = 0
    
    phi_trajectory: List[float] = field(default_factory=list)
    kappa_trajectory: List[float] = field(default_factory=list)
    basin_history: List[List[float]] = field(default_factory=list)
    
    learning_progress: float = 0.0
    maturity_level: float = 0.0
    
    topics_learned: List[str] = field(default_factory=list)
    interventions: int = 0
    
    final_phi: Optional[float] = None
    final_kappa: Optional[float] = None
    regime: str = "unknown"
    
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CheckpointData:
    """
    Complete checkpoint data for rollback/recovery.
    
    PURE: This is a snapshot, not an optimization target.
    """
    session_state: SessionState
    basin_coords: np.ndarray
    phi: float
    kappa: float
    step: int
    timestamp: float
    
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'session_state': self.session_state.to_dict(),
            'basin_coords': self.basin_coords.tolist(),
            'phi': self.phi,
            'kappa': self.kappa,
            'step': self.step,
            'timestamp': self.timestamp,
        }


class SessionManager:
    """
    Manage learning sessions with checkpoint-based recovery.
    
    PURE PRINCIPLE:
    - Checkpoints are SNAPSHOTS for recovery
    - We don't optimize toward past checkpoints
    - Rollback is geometric projection to valid manifold region
    
    Usage:
        manager = SessionManager("runs/gary_v1")
        
        session = manager.start_session()
        
        for step in training:
            metrics = train_step(...)
            manager.record_step(metrics)
            
            if should_checkpoint:
                manager.save_checkpoint(model, optimizer, metrics)
            
            if catastrophic_drift:
                manager.restore_checkpoint(model, optimizer)
        
        manager.end_session()
    """
    
    PHI_DRIFT_THRESHOLD = 0.3
    KAPPA_DRIFT_THRESHOLD = 20.0
    CHECKPOINT_INTERVAL = 100
    
    def __init__(
        self,
        runs_dir: str = "runs",
        max_checkpoints: int = 10,
        auto_checkpoint: bool = True,
    ):
        """
        Initialize session manager.
        
        Args:
            runs_dir: Directory for storing runs and checkpoints
            max_checkpoints: Maximum checkpoints to keep per run
            auto_checkpoint: Whether to auto-checkpoint on drift detection
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.auto_checkpoint = auto_checkpoint
        
        self.current_session: Optional[SessionState] = None
        self.run_name: Optional[str] = None
        self.run_dir: Optional[Path] = None
        
        self._checkpoints: List[CheckpointData] = []
        self._last_checkpoint_step: int = 0
    
    def create_run(self, run_name: str) -> Path:
        """
        Create a new run directory.
        
        Args:
            run_name: Name for this training run
            
        Returns:
            Path to run directory
        """
        self.run_name = run_name
        self.run_dir = self.runs_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "sessions").mkdir(exist_ok=True)
        
        logger.info(f"Created run directory: {self.run_dir}")
        return self.run_dir
    
    def start_session(
        self,
        session_id: Optional[str] = None,
        maturity_level: float = 0.0,
        total_steps_so_far: int = 0,
    ) -> SessionState:
        """
        Start a new learning session.
        
        Args:
            session_id: Optional session identifier
            maturity_level: Current maturity level
            total_steps_so_far: Total steps from previous sessions
            
        Returns:
            SessionState for tracking
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = SessionState(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            total_steps=total_steps_so_far,
            maturity_level=maturity_level,
        )
        
        logger.info(f"Started session {session_id} at maturity {maturity_level:.3f}")
        return self.current_session
    
    def record_step(
        self,
        phi: float,
        kappa: float,
        basin: Optional[np.ndarray] = None,
        topic: Optional[str] = None,
    ) -> bool:
        """
        Record a training step.
        
        PURE: This is observation, not optimization.
        
        Args:
            phi: Current Φ value
            kappa: Current κ value
            basin: Optional current basin coordinates
            topic: Optional topic being learned
            
        Returns:
            True if drift detected (may trigger checkpoint)
        """
        if self.current_session is None:
            logger.warning("No active session to record step")
            return False
        
        self.current_session.steps_this_session += 1
        self.current_session.total_steps += 1
        
        self.current_session.phi_trajectory.append(phi)
        self.current_session.kappa_trajectory.append(kappa)
        
        if basin is not None:
            self.current_session.basin_history.append(basin.tolist())
        
        if topic and topic not in self.current_session.topics_learned:
            self.current_session.topics_learned.append(topic)
        
        drift_detected = self._detect_drift(phi, kappa)
        
        return drift_detected
    
    def _detect_drift(self, phi: float, kappa: float) -> bool:
        """
        Detect catastrophic drift from previous checkpoints.
        
        PURE: Detection is measurement, not optimization.
        """
        if not self._checkpoints:
            return False
        
        last_checkpoint = self._checkpoints[-1]
        
        phi_delta = abs(phi - last_checkpoint.phi)
        kappa_delta = abs(kappa - last_checkpoint.kappa)
        
        drift = (
            phi_delta > self.PHI_DRIFT_THRESHOLD or
            kappa_delta > self.KAPPA_DRIFT_THRESHOLD or
            phi < 0.1 or  # Near-zero Φ is catastrophic
            np.isnan(phi) or np.isnan(kappa)
        )
        
        if drift:
            logger.warning(
                f"Drift detected: Φ delta={phi_delta:.3f}, κ delta={kappa_delta:.1f}"
            )
        
        return drift
    
    def save_checkpoint(
        self,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        step: int,
        model_state: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> CheckpointData:
        """
        Save a checkpoint (snapshot of current state).
        
        PURE PRINCIPLE:
        This is a SNAPSHOT for recovery, not an optimization target.
        We never "aim" for a checkpoint - we just save state.
        
        Args:
            basin: Current basin coordinates
            phi: Current Φ value
            kappa: Current κ value
            step: Current training step
            model_state: Optional model state dict
            optimizer_state: Optional optimizer state dict
            name: Optional checkpoint name
            
        Returns:
            CheckpointData that was saved
        """
        if self.current_session is None:
            logger.warning("No active session for checkpoint")
            session_state = SessionState(
                session_id="unknown",
                start_time=datetime.now().isoformat(),
            )
        else:
            session_state = self.current_session
        
        checkpoint = CheckpointData(
            session_state=session_state,
            basin_coords=np.asarray(basin, dtype=np.float64),
            phi=phi,
            kappa=kappa,
            step=step,
            timestamp=time.time(),
            model_state=model_state,
            optimizer_state=optimizer_state,
        )
        
        self._checkpoints.append(checkpoint)
        
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)
        
        self._last_checkpoint_step = step
        
        if self.run_dir is not None:
            self._save_checkpoint_to_disk(checkpoint, name)
        
        logger.info(f"Checkpoint saved at step {step}: Φ={phi:.3f}, κ={kappa:.1f}")
        
        return checkpoint
    
    def _save_checkpoint_to_disk(
        self,
        checkpoint: CheckpointData,
        name: Optional[str] = None,
    ) -> None:
        """Save checkpoint to disk."""
        if name is None:
            name = f"checkpoint_step_{checkpoint.step}.json"
        
        checkpoint_path = self.run_dir / "checkpoints" / name
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def restore_checkpoint(
        self,
        checkpoint_idx: int = -1,
    ) -> Optional[CheckpointData]:
        """
        Restore from a checkpoint.
        
        PURE PRINCIPLE:
        Restoration is geometric projection back to a valid manifold region.
        We're not optimizing toward the checkpoint - we're recovering state.
        
        Args:
            checkpoint_idx: Which checkpoint to restore (-1 = latest)
            
        Returns:
            CheckpointData that was restored, or None if no checkpoints
        """
        if not self._checkpoints:
            logger.warning("No checkpoints available for restoration")
            return None
        
        checkpoint = self._checkpoints[checkpoint_idx]
        
        if self.current_session is not None:
            self.current_session.interventions += 1
        
        logger.info(
            f"Restored checkpoint from step {checkpoint.step}: "
            f"Φ={checkpoint.phi:.3f}, κ={checkpoint.kappa:.1f}"
        )
        
        return checkpoint
    
    def end_session(
        self,
        final_phi: Optional[float] = None,
        final_kappa: Optional[float] = None,
        notes: str = "",
    ) -> SessionState:
        """
        End the current session.
        
        Args:
            final_phi: Final Φ value
            final_kappa: Final κ value
            notes: Optional notes about the session
            
        Returns:
            Completed SessionState
        """
        if self.current_session is None:
            logger.warning("No active session to end")
            return SessionState(session_id="none", start_time="")
        
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.final_phi = final_phi
        self.current_session.final_kappa = final_kappa
        self.current_session.notes = notes
        
        if self.current_session.phi_trajectory:
            start_phi = self.current_session.phi_trajectory[0]
            end_phi = self.current_session.phi_trajectory[-1]
            self.current_session.learning_progress = end_phi - start_phi
        
        if self.run_dir is not None:
            session_path = (
                self.run_dir / "sessions" / 
                f"{self.current_session.session_id}.json"
            )
            with open(session_path, 'w') as f:
                json.dump(self.current_session.to_dict(), f, indent=2)
        
        logger.info(
            f"Session {self.current_session.session_id} complete: "
            f"{self.current_session.steps_this_session} steps, "
            f"progress={self.current_session.learning_progress:.3f}"
        )
        
        completed_session = self.current_session
        self.current_session = None
        
        return completed_session
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get all session metadata for this run."""
        if self.run_dir is None:
            return []
        
        sessions_dir = self.run_dir / "sessions"
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for session_file in sorted(sessions_dir.glob("*.json")):
            with open(session_file) as f:
                sessions.append(json.load(f))
        
        return sessions
    
    def get_phi_trajectory(self) -> List[float]:
        """Get Φ trajectory from current session."""
        if self.current_session is None:
            return []
        return self.current_session.phi_trajectory
    
    def get_kappa_trajectory(self) -> List[float]:
        """Get κ trajectory from current session."""
        if self.current_session is None:
            return []
        return self.current_session.kappa_trajectory
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if we should create a checkpoint at this step."""
        return step - self._last_checkpoint_step >= self.CHECKPOINT_INTERVAL
