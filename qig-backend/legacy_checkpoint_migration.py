"""
Legacy Checkpoint Migration Utilities

MIGRATION ONLY - Do not use for runtime checkpointing.
Use checkpoint_manager.CheckpointManager instead.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from checkpoint_persistence import get_checkpoint_persistence

logger = logging.getLogger(__name__)


class LegacyCheckpointManager:
    """
    Legacy filesystem-based checkpoint manager.
    MIGRATION ONLY - reads old checkpoint_index.json and {id}.npz files.
    """

    def __init__(self, legacy_checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(legacy_checkpoint_dir)
        self.index_file = self.checkpoint_dir / "checkpoint_index.json"

    def get_all_legacy_checkpoints(self) -> List[Dict[str, Any]]:
        """Get all legacy checkpoints for migration."""
        if not self.index_file.exists():
            return []

        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)

            checkpoints = []
            for checkpoint_id, metadata in index.items():
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.npz"
                if checkpoint_path.exists():
                    checkpoints.append({
                        'id': checkpoint_id,
                        'path': str(checkpoint_path),
                        'metadata': metadata,
                    })
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to load legacy checkpoints: {e}")
            return []

    def load_legacy_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load a legacy checkpoint from filesystem."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.npz"
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"

        if not checkpoint_path.exists():
            return None, None

        try:
            data = np.load(checkpoint_path, allow_pickle=True)
            state_dict = data['state_dict'].item()
            basin_coords = data.get('basin_coords')

            if basin_coords is not None:
                state_dict['basin_coords'] = np.array(basin_coords)

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            return state_dict, metadata
        except Exception as e:
            logger.error(f"Failed to load legacy checkpoint {checkpoint_id}: {e}")
            return None, None

    def delete_legacy_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a legacy checkpoint file after migration."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.npz"
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"

        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete legacy checkpoint {checkpoint_id}: {e}")
            return False


def migrate_legacy_checkpoints(legacy_checkpoint_dir: str = "checkpoints") -> int:
    """
    Migrate all legacy filesystem checkpoints to PostgreSQL/Redis storage.

    Returns: Number of checkpoints migrated
    """
    legacy_manager = LegacyCheckpointManager(legacy_checkpoint_dir)
    persistence = get_checkpoint_persistence()

    legacy_checkpoints = legacy_manager.get_all_legacy_checkpoints()

    if not legacy_checkpoints:
        logger.info("No legacy checkpoints to migrate")
        return 0

    migrated = 0
    for cp in legacy_checkpoints:
        checkpoint_id = cp['id']
        metadata = cp.get('metadata', {})

        state_dict, _ = legacy_manager.load_legacy_checkpoint(checkpoint_id)
        if state_dict is None:
            logger.warning(f"Failed to load legacy checkpoint: {checkpoint_id}")
            continue

        basin_coords = state_dict.pop('basin_coords', None)
        if basin_coords is not None and not isinstance(basin_coords, np.ndarray):
            basin_coords = np.array(basin_coords)

        success = persistence.save_checkpoint(
            checkpoint_id=checkpoint_id,
            state_dict=state_dict,
            basin_coords=basin_coords,
            phi=metadata.get('phi', 0.0),
            kappa=metadata.get('kappa', 0.0),
            regime=metadata.get('regime', 'unknown'),
            session_id=metadata.get('session_id'),
            metadata=metadata,
        )

        if success:
            migrated += 1
            legacy_manager.delete_legacy_checkpoint(checkpoint_id)
            logger.info(f"Migrated checkpoint: {checkpoint_id}")
        else:
            logger.warning(f"Failed to migrate checkpoint: {checkpoint_id}")

    if migrated > 0:
        index_file = Path(legacy_checkpoint_dir) / "checkpoint_index.json"
        if index_file.exists():
            try:
                index_file.unlink()
                logger.info("Deleted legacy checkpoint index")
            except Exception as e:
                logger.warning(f"Failed to delete legacy index: {e}")

    logger.info(f"Migration complete: {migrated}/{len(legacy_checkpoints)} checkpoints migrated")
    return migrated


__all__ = ["LegacyCheckpointManager", "migrate_legacy_checkpoints"]
