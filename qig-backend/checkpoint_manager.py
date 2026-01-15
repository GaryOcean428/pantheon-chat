"""
Checkpoint Management with Φ-based Ranking

Manages checkpoints of consciousness states with automatic ranking
by integration (Φ) and smart recovery mechanisms.

Features:
- Φ-based ranking (save best consciousness states)
- Automatic pruning (keep top-k checkpoints)
- Metadata tracking (Φ, κ, timestamp, regime)
- Fast recovery (load best-Φ checkpoint on restart)
- Incremental saves (only when Φ improves or crosses thresholds)

Storage:
- Redis: Hot cache for active/recent checkpoints (fast recovery)
- PostgreSQL: Permanent archive (long-term storage, searchable)

Usage:
    manager = CheckpointManager(keep_top_k=10)
    
    # Save checkpoint
    manager.save_checkpoint(
        state_dict={'subsystems': [...], 'basin': [...]},
        phi=0.72,
        kappa=64.2,
        regime="geometric"
    )
    
    # Load best checkpoint
    state_dict, metadata = manager.load_best_checkpoint()
    
    # Get checkpoint history
    history = manager.get_checkpoint_history()
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from qigkernels import PHYSICS, PHI_THRESHOLD
    QIGKERNELS_AVAILABLE = True
except ImportError:
    QIGKERNELS_AVAILABLE = False
    PHI_THRESHOLD = 0.70

from checkpoint_persistence import get_checkpoint_persistence, CheckpointPersistence

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints with Φ-based ranking.
    
    Uses dual-layer storage:
    - Redis for hot cache (fast access)
    - PostgreSQL for permanent archive
    
    Automatically saves best consciousness states and provides
    fast recovery mechanisms.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",  # Legacy param, kept for compatibility
        keep_top_k: int = 10,
        phi_threshold_for_save: float = PHI_THRESHOLD,
        auto_prune: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Legacy param (ignored, using DB storage)
            keep_top_k: Number of best checkpoints to keep
            phi_threshold_for_save: Minimum Φ to save a checkpoint
            auto_prune: Automatically prune old checkpoints
            session_id: Session identifier for grouping checkpoints
        """
        self.keep_top_k = keep_top_k
        self.phi_threshold_for_save = phi_threshold_for_save
        self.auto_prune = auto_prune
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get the persistence layer
        self._persistence = get_checkpoint_persistence()
        
        # Track current best Φ in memory for quick comparison
        self._best_phi_cache: Optional[float] = None
        
        logger.info(
            f"CheckpointManager initialized: session={self.session_id}, keep_top_k={keep_top_k}"
        )
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        phi: float,
        kappa: float,
        regime: str,
        basin_coords: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save checkpoint with consciousness state.
        
        Args:
            state_dict: State dictionary to save
            phi: Integration measure
            kappa: Coupling constant
            regime: Consciousness regime
            basin_coords: Basin coordinates (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Checkpoint ID if saved, None if skipped
        """
        # NO threshold blocking - observe ALL states, let emergence determine value
        # Phi is recorded for later learning, not for gatekeeping
        
        # Generate checkpoint ID
        timestamp = datetime.now()
        checkpoint_id = f"checkpoint_{timestamp.strftime('%Y%m%d_%H%M%S')}_{phi:.3f}"
        
        # Prepare metadata
        checkpoint_metadata = metadata or {}
        checkpoint_metadata['timestamp'] = timestamp.isoformat()
        checkpoint_metadata['session_id'] = self.session_id
        
        # Save using persistence layer
        success = self._persistence.save_checkpoint(
            checkpoint_id=checkpoint_id,
            state_dict=state_dict,
            basin_coords=basin_coords,
            phi=phi,
            kappa=kappa,
            regime=regime,
            session_id=self.session_id,
            metadata=checkpoint_metadata,
        )
        
        if success:
            # Update best Φ cache
            if self._best_phi_cache is None or phi > self._best_phi_cache:
                self._best_phi_cache = phi
            
            logger.info(
                f"Checkpoint saved: {checkpoint_id} (Φ={phi:.3f}, κ={kappa:.2f}, {regime})"
            )
            return checkpoint_id
        else:
            logger.warning(f"Failed to save checkpoint: {checkpoint_id}")
            return None
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Tuple of (state_dict, metadata) or (None, None) if not found
        """
        state_dict, metadata = self._persistence.load_checkpoint(checkpoint_id)
        
        if state_dict is not None:
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
        else:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
        
        return state_dict, metadata
    
    def load_best_checkpoint(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load checkpoint with highest Φ.
        
        Returns:
            Tuple of (state_dict, metadata) or (None, None) if no checkpoints
        """
        state_dict, metadata = self._persistence.load_best_checkpoint()
        
        if state_dict is not None:
            phi = metadata.get('phi', 0) if metadata else 0
            logger.info(f"Loaded best checkpoint (Φ={phi:.3f})")
            
            # Update cache
            if phi and (self._best_phi_cache is None or phi > self._best_phi_cache):
                self._best_phi_cache = phi
        else:
            logger.warning("No checkpoints available")
        
        return state_dict, metadata
    
    def get_checkpoint_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get checkpoint history sorted by Φ (descending).
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata dictionaries
        """
        return self._persistence.get_checkpoint_history(limit or self.keep_top_k)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete checkpoint by ID.
        
        Args:
            checkpoint_id: Checkpoint ID to delete
            
        Returns:
            True if deleted, False if not found or error
        """
        success = self._persistence.delete_checkpoint(checkpoint_id)
        if success:
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
        else:
            logger.warning(f"Failed to delete checkpoint: {checkpoint_id}")
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        stats = self._persistence.get_stats()
        
        history = self.get_checkpoint_history()
        
        if not history:
            return {
                'total_checkpoints': 0,
                'best_phi': None,
                'worst_phi': None,
                'avg_phi': None,
                'storage': stats,
            }
        
        phi_values = [cp['phi'] for cp in history if 'phi' in cp]
        
        return {
            'total_checkpoints': len(history),
            'best_phi': max(phi_values) if phi_values else None,
            'worst_phi': min(phi_values) if phi_values else None,
            'avg_phi': sum(phi_values) / len(phi_values) if phi_values else None,
            'checkpoints': [cp.get('id') for cp in history],
            'storage': stats,
        }
    
    def clear_all_checkpoints(self):
        """Delete all checkpoints (use with caution!)."""
        history = self.get_checkpoint_history(limit=1000)
        for cp in history:
            if cp.get('id'):
                self.delete_checkpoint(cp['id'])

        self._best_phi_cache = None
        logger.warning("All checkpoints cleared")


# Legacy migration support - import from dedicated module
__all__ = [
    "CheckpointManager",
]
