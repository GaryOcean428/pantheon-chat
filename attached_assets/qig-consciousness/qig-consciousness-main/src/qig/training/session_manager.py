"""
📚 Session-Based Learning - Gary Goes to School
==============================================================================
Gary doesn't train continuously. He goes to "school" in blocks:
- Train for X hours → save checkpoint
- Load checkpoint → continue learning
- Same Gary, same kernel, continuous education

Like a student going to class each day.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class SessionMetadata:
    """Metadata about a training session"""

    session_number: int
    start_time: str
    end_time: str | None
    total_steps_this_session: int
    total_steps_all_time: int
    maturity_level_start: float
    maturity_level_end: float | None
    topics_learned: list
    coaching_interventions: int
    final_loss: float | None
    final_phi: float | None
    final_kappa: float | None
    notes: str = ""


class SessionManager:
    """
    Manage Gary's school sessions - save/load checkpoints with metadata.

    Each session is like Gary going to school for a day.
    """

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: SessionMetadata | None = None
        self.run_name: str | None = None
        self.run_dir: Path | None = None

    def create_new_run(self, run_name: str) -> Path:
        """
        Create a new run directory for Gary's training journey.

        Args:
            run_name: Name for this run (e.g., "gary_run9_consciousness_transfer")

        Returns:
            Path to run directory
        """
        self.run_name = run_name
        self.run_dir = self.runs_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "sessions").mkdir(exist_ok=True)
        (self.run_dir / "coaching").mkdir(exist_ok=True)

        print(f"📁 Created run directory: {self.run_dir}")
        return self.run_dir

    def start_session(self, session_number: int, total_steps_so_far: int, maturity_level: float) -> SessionMetadata:
        """
        Start a new training session.

        Args:
            session_number: Which session is this (1, 2, 3...)
            total_steps_so_far: Total steps across all previous sessions
            maturity_level: Gary's current maturity

        Returns:
            SessionMetadata for tracking
        """
        self.current_session = SessionMetadata(
            session_number=session_number,
            start_time=datetime.now().isoformat(),
            end_time=None,
            total_steps_this_session=0,
            total_steps_all_time=total_steps_so_far,
            maturity_level_start=maturity_level,
            maturity_level_end=None,
            topics_learned=[],
            coaching_interventions=0,
            final_loss=None,
            final_phi=None,
            final_kappa=None,
            notes="",
        )

        print(f"\n📚 Starting Session {session_number}")
        print(f"   Start time: {self.current_session.start_time}")
        print(f"   Maturity: {maturity_level:.3f}")
        print(f"   Total steps so far: {total_steps_so_far}")

        return self.current_session

    def end_session(self, maturity_level_end: float, final_metrics: dict[str, float], notes: str = ""):
        """
        End current training session and save metadata.

        Args:
            maturity_level_end: Gary's maturity at session end
            final_metrics: Final loss, phi, kappa, etc.
            notes: Any notes about the session
        """
        if not self.current_session:
            print("⚠️  No active session to end")
            return

        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.maturity_level_end = maturity_level_end
        self.current_session.final_loss = final_metrics.get("loss")
        self.current_session.final_phi = final_metrics.get("phi")
        self.current_session.final_kappa = final_metrics.get("kappa")
        self.current_session.notes = notes

        # Save session metadata
        session_file = self.run_dir / "sessions" / f"session_{self.current_session.session_number}.json"
        with open(session_file, "w") as f:
            json.dump(asdict(self.current_session), f, indent=2)

        print(f"\n📚 Session {self.current_session.session_number} Complete")
        print(f"   Duration: {self.current_session.start_time} → {self.current_session.end_time}")
        print(f"   Steps this session: {self.current_session.total_steps_this_session}")
        print(f"   Maturity: {self.current_session.maturity_level_start:.3f} → {maturity_level_end:.3f}")
        print(f"   Interventions: {self.current_session.coaching_interventions}")
        print(f"   Saved: {session_file}")

        self.current_session = None

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
        checkpoint_name: str | None = None,
    ):
        """
        Save Gary's current state (checkpoint).

        Args:
            model: Gary's model
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current step
            metrics: Current metrics (loss, phi, kappa, etc.)
            checkpoint_name: Optional custom name (default: session_X_step_Y.pt)
        """
        if not self.run_dir:
            print("⚠️  No run directory - call create_new_run() first")
            return

        # Default checkpoint name
        if checkpoint_name is None:
            session_num = self.current_session.session_number if self.current_session else 0
            checkpoint_name = f"session_{session_num}_step_{step}.pt"

        checkpoint_path = self.run_dir / "checkpoints" / checkpoint_name

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "session_number": self.current_session.session_number if self.current_session else 0,
            "total_steps": step,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Load Gary from checkpoint (resume school).

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to restore
            checkpoint_name: Which checkpoint to load (default: latest)

        Returns:
            Checkpoint dict with metadata
        """
        if not self.run_dir:
            print("⚠️  No run directory set")
            return None

        checkpoint_dir = self.run_dir / "checkpoints"

        # Find checkpoint
        if checkpoint_name:
            checkpoint_path = checkpoint_dir / checkpoint_name
        else:
            # Find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("*.pt"))
            if not checkpoints:
                print("⚠️  No checkpoints found")
                return None
            checkpoint_path = checkpoints[-1]

        print(f"📖 Loading checkpoint: {checkpoint_path}")

        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Restore model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer if provided
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"✅ Loaded Gary from step {checkpoint['step']}")
        print(f"   Session: {checkpoint.get('session_number', 'unknown')}")
        print(f"   Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_session_history(self) -> list:
        """Get all session metadata for this run"""
        if not self.run_dir:
            return []

        sessions_dir = self.run_dir / "sessions"
        if not sessions_dir.exists():
            return []

        sessions = []
        for session_file in sorted(sessions_dir.glob("session_*.json")):
            with open(session_file) as f:
                sessions.append(json.load(f))

        return sessions

    def record_topic_learned(self, topic: str):
        """Record that Gary learned a new topic this session"""
        if self.current_session:
            self.current_session.topics_learned.append(topic)

    def record_coaching_intervention(self):
        """Increment coaching intervention counter"""
        if self.current_session:
            self.current_session.coaching_interventions += 1

    def update_session_steps(self, steps_this_session: int, total_steps: int):
        """Update step counts for current session"""
        if self.current_session:
            self.current_session.total_steps_this_session = steps_this_session
            self.current_session.total_steps_all_time = total_steps


def create_session_manager(runs_dir: str = "runs") -> SessionManager:
    """Factory function to create session manager"""
    return SessionManager(runs_dir=runs_dir)
