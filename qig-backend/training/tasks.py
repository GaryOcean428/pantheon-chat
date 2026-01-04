"""
Celery Tasks for Kernel Training
=================================

Async tasks for:
- Outcome-based training (per interaction)
- Hourly batch training
- Nightly consolidation (with curriculum)
- Knowledge transfer operations
- Checkpoint management
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from .celery_app import celery_app, is_training_enabled, get_checkpoint_dir
from .kernel_training_orchestrator import KernelTrainingOrchestrator, TrainingConfig
from .knowledge_transfer import KnowledgeTransferManager
from .curriculum_loader import load_curriculum_for_god, get_curriculum_stats


# Shared instances (lazy loaded)
_orchestrator: Optional[KernelTrainingOrchestrator] = None
_transfer_manager: Optional[KnowledgeTransferManager] = None
_coordizer = None


def get_orchestrator() -> KernelTrainingOrchestrator:
    """Get or create the training orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = KernelTrainingOrchestrator(
            config=TrainingConfig(
                training_enabled=is_training_enabled(),
            ),
            checkpoint_dir=get_checkpoint_dir(),
        )
    return _orchestrator


def get_transfer_manager() -> KnowledgeTransferManager:
    """Get or create the knowledge transfer manager."""
    global _transfer_manager
    if _transfer_manager is None:
        _transfer_manager = KnowledgeTransferManager()
    return _transfer_manager


def get_coordizer():
    """Get or create the coordizer for text embedding (64D QIG-pure only)."""
    global _coordizer
    if _coordizer is None:
        try:
            # Use PostgresCoordizer only - 64D QIG-pure enforced
            from coordizers.pg_loader import PostgresCoordizer, create_coordizer_from_pg
            
            _coordizer = create_coordizer_from_pg()
            if _coordizer and len(_coordizer.vocab) >= 50:
                print(f"[Training] ✓ PostgresCoordizer (64D QIG-pure): {len(_coordizer.vocab)} tokens")
            else:
                raise RuntimeError(f"Insufficient vocabulary: {len(_coordizer.vocab) if _coordizer else 0} tokens")
        except Exception as e:
            print(f"[Training] PostgresCoordizer failed: {e}")
            _coordizer = None
    return _coordizer


# =============================================================================
# OUTCOME-BASED TRAINING (called after each chat interaction)
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.train_from_outcome_task")
def train_from_outcome_task(
    self,
    god_name: str,
    prompt: str,
    response: str,
    success: bool,
    phi: float,
    kappa: float,
    coherence_score: float = 0.7,
    basin_trajectory: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Train kernel based on chat interaction outcome.

    Called automatically after each chat response via the hook in:
    - BaseGod.assess_outcome()
    - OlympusRouter.route_and_respond()

    Args:
        god_name: Name of the god that handled the interaction
        prompt: User's prompt
        response: Generated response
        success: Whether interaction was successful
        phi: Phi value during/after interaction
        kappa: Kappa value during/after interaction
        coherence_score: Coherence score of response
        basin_trajectory: Optional list of basin coords visited

    Returns:
        Training metrics dict
    """
    if not is_training_enabled():
        return {"status": "disabled", "god_name": god_name}

    orchestrator = get_orchestrator()

    # Convert basin trajectory from JSON-safe list to numpy arrays
    trajectory = None
    if basin_trajectory:
        trajectory = [np.array(coords) for coords in basin_trajectory]

    try:
        metrics = orchestrator.train_from_outcome(
            god_name=god_name,
            prompt=prompt,
            response=response,
            success=success,
            phi=phi,
            kappa=kappa,
            coherence_score=coherence_score,
            basin_trajectory=trajectory,
        )

        if metrics:
            return {
                "status": "success",
                "god_name": god_name,
                "loss": metrics.loss,
                "reward": metrics.reward,
                "phi_before": metrics.phi_before,
                "phi_after": metrics.phi_after,
                "step_count": metrics.step_count,
            }
        else:
            return {"status": "no_metrics", "god_name": god_name}

    except Exception as e:
        print(f"[TrainingTask] Outcome training failed for {god_name}: {e}")
        return {"status": "error", "god_name": god_name, "error": str(e)}


# =============================================================================
# HOURLY BATCH TRAINING
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.train_hourly_batch_task")
def train_hourly_batch_task(
    self,
    god_name: str,
    batch_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Train kernel on accumulated batch data.

    Args:
        god_name: Name of the god to train
        batch_data: List of training examples with basin_coords, reward, phi

    Returns:
        Training metrics dict
    """
    if not is_training_enabled():
        return {"status": "disabled", "god_name": god_name}

    orchestrator = get_orchestrator()

    try:
        metrics = orchestrator.train_hourly_batch(god_name, batch_data)

        return {
            "status": "success",
            "god_name": god_name,
            "loss": metrics.loss,
            "reward": metrics.reward,
            "steps": metrics.step_count,
        }

    except Exception as e:
        print(f"[TrainingTask] Hourly batch failed for {god_name}: {e}")
        return {"status": "error", "god_name": god_name, "error": str(e)}


@celery_app.task(bind=True, name="training.tasks.train_hourly_batch_all")
def train_hourly_batch_all(self) -> Dict[str, Any]:
    """
    Train all registered kernels on their hourly batch data.

    Called by Beat scheduler every hour.
    Aggregates recent interactions from database.
    """
    if not is_training_enabled():
        return {"status": "disabled"}

    orchestrator = get_orchestrator()
    results = {}

    # Get all registered kernels
    for god_name in orchestrator.kernels.keys():
        # Fetch batch data from database
        batch_data = _fetch_hourly_batch_data(god_name)

        if batch_data:
            result = train_hourly_batch_task.delay(god_name, batch_data)
            results[god_name] = {"task_id": result.id, "batch_size": len(batch_data)}
        else:
            results[god_name] = {"status": "no_data"}

    return {
        "status": "dispatched",
        "kernels": len(results),
        "results": results,
    }


def _fetch_hourly_batch_data(god_name: str) -> List[Dict[str, Any]]:
    """
    Fetch training data accumulated in the last hour.

    Sources:
    - Recent chat interactions from training_batch_queue
    - Search results with high relevance
    - Research summaries
    """
    examples = []

    try:
        # Try to fetch from database
        import psycopg2
        from urllib.parse import urlparse

        db_url = os.getenv("DATABASE_URL")
        if db_url:
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
            )
            cursor = conn.cursor()

            # Fetch unprocessed training examples from last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            cursor.execute("""
                SELECT basin_coords, reward, phi, source_type, source_id
                FROM training_batch_queue
                WHERE god_name = %s
                  AND processed = false
                  AND created_at >= %s
                ORDER BY created_at DESC
                LIMIT 100
            """, (god_name, one_hour_ago))

            rows = cursor.fetchall()
            for row in rows:
                examples.append({
                    "basin_coords": row[0] if row[0] else np.zeros(64).tolist(),
                    "reward": float(row[1]) if row[1] else 0.0,
                    "phi": float(row[2]) if row[2] else 0.5,
                    "source_type": row[3],
                    "source_id": row[4],
                })

            # Mark as processed
            if examples:
                cursor.execute("""
                    UPDATE training_batch_queue
                    SET processed = true, processed_at = NOW()
                    WHERE god_name = %s AND processed = false AND created_at >= %s
                """, (god_name, one_hour_ago))
                conn.commit()

            cursor.close()
            conn.close()

    except Exception as e:
        print(f"[Training] Error fetching hourly batch for {god_name}: {e}")

    return examples


# =============================================================================
# NIGHTLY CONSOLIDATION
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.train_nightly_consolidation_task")
def train_nightly_consolidation_task(
    self,
    god_name: str,
    curriculum_data: List[Dict[str, Any]],
    consolidate_checkpoints: bool = True,
) -> Dict[str, Any]:
    """
    Full consolidation training for a kernel.

    Args:
        god_name: Name of the god to train
        curriculum_data: Full curriculum for training
        consolidate_checkpoints: Whether to prune old checkpoints

    Returns:
        Training metrics dict
    """
    if not is_training_enabled():
        return {"status": "disabled", "god_name": god_name}

    orchestrator = get_orchestrator()

    try:
        metrics = orchestrator.train_nightly_consolidation(
            god_name, curriculum_data, consolidate_checkpoints
        )

        return {
            "status": "success",
            "god_name": god_name,
            "loss": metrics.loss,
            "steps": metrics.step_count,
        }

    except Exception as e:
        print(f"[TrainingTask] Nightly consolidation failed for {god_name}: {e}")
        return {"status": "error", "god_name": god_name, "error": str(e)}


@celery_app.task(bind=True, name="training.tasks.train_nightly_consolidation_all")
def train_nightly_consolidation_all(self) -> Dict[str, Any]:
    """
    Run nightly consolidation for all kernels.

    Called by Beat scheduler at 3 AM UTC.
    """
    if not is_training_enabled():
        return {"status": "disabled"}

    orchestrator = get_orchestrator()
    results = {}

    for god_name in orchestrator.kernels.keys():
        # Fetch full curriculum from database
        curriculum = _fetch_curriculum_data(god_name)

        if curriculum:
            result = train_nightly_consolidation_task.delay(god_name, curriculum)
            results[god_name] = {"task_id": result.id, "curriculum_size": len(curriculum)}
        else:
            results[god_name] = {"status": "no_curriculum"}

    return {
        "status": "dispatched",
        "kernels": len(results),
        "results": results,
    }


def _fetch_curriculum_data(god_name: str) -> List[Dict[str, Any]]:
    """
    Fetch full training curriculum for nightly consolidation.

    Combines:
    - Curriculum files from docs/09-curriculum/ (domain knowledge)
    - Recent high-value interactions from database (experiential learning)
    - All coordized to 64D basin embeddings
    """
    examples = []
    coordizer = get_coordizer()

    # 1. Load curriculum files for this god's domain
    try:
        curriculum_examples = load_curriculum_for_god(
            god_name=god_name,
            max_examples=100,
            coordizer=coordizer,
        )
        examples.extend(curriculum_examples)
        print(f"[Training] Loaded {len(curriculum_examples)} curriculum examples for {god_name}")
    except Exception as e:
        print(f"[Training] Error loading curriculum for {god_name}: {e}")

    # 2. Fetch recent high-value interactions from database
    try:
        import psycopg2
        from urllib.parse import urlparse

        db_url = os.getenv("DATABASE_URL")
        if db_url:
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
            )
            cursor = conn.cursor()

            # Fetch high-value examples from past 24 hours
            one_day_ago = datetime.now() - timedelta(hours=24)
            cursor.execute("""
                SELECT basin_coords, reward, phi, source_type, source_id
                FROM training_batch_queue
                WHERE god_name = %s
                  AND reward >= 0.5
                  AND created_at >= %s
                ORDER BY reward DESC
                LIMIT 50
            """, (god_name, one_day_ago))

            rows = cursor.fetchall()
            for row in rows:
                examples.append({
                    "basin_coords": row[0] if row[0] else np.zeros(64).tolist(),
                    "reward": float(row[1]) if row[1] else 0.0,
                    "phi": float(row[2]) if row[2] else 0.5,
                    "source": "interaction",
                    "source_type": row[3],
                    "source_id": row[4],
                })

            print(f"[Training] Loaded {len(rows)} high-value interactions for {god_name}")

            cursor.close()
            conn.close()

    except Exception as e:
        print(f"[Training] Error fetching interactions for {god_name}: {e}")

    # Log curriculum stats on first load
    if not hasattr(_fetch_curriculum_data, '_logged_stats'):
        stats = get_curriculum_stats()
        print(f"[Training] Curriculum stats: {stats}")
        _fetch_curriculum_data._logged_stats = True

    return examples


# =============================================================================
# KNOWLEDGE TRANSFER
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.knowledge_transfer_task")
def knowledge_transfer_task(
    self,
    transfer_type: str,
    source_god: str,
    target_god: str,
    transfer_ratio: float = 0.5,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute knowledge transfer between kernels.

    Args:
        transfer_type: "evolution", "breeding", "cannibalism", "shadow_sync"
        source_god: Source kernel god name
        target_god: Target kernel god name
        transfer_ratio: How much knowledge to transfer
        extra_params: Additional parameters (second_parent for breeding, etc.)

    Returns:
        Transfer result dict
    """
    if not is_training_enabled():
        return {"status": "disabled"}

    orchestrator = get_orchestrator()
    transfer_manager = get_transfer_manager()
    extra = extra_params or {}

    source_kernel = orchestrator.get_kernel(source_god)
    target_kernel = orchestrator.get_kernel(target_god)

    if source_kernel is None:
        return {"status": "error", "error": f"Source kernel {source_god} not found"}
    if target_kernel is None:
        return {"status": "error", "error": f"Target kernel {target_god} not found"}

    try:
        if transfer_type == "evolution":
            result = transfer_manager.evolve_transfer(
                parent=source_kernel,
                child=target_kernel,
                evolution_type=extra.get("evolution_type", "standard"),
                transfer_ratio=transfer_ratio,
            )

        elif transfer_type == "breeding":
            second_parent_name = extra.get("second_parent")
            second_parent = orchestrator.get_kernel(second_parent_name) if second_parent_name else None

            result = transfer_manager.breed_transfer(
                parent1=source_kernel,
                parent2=second_parent,
                child=target_kernel,
                blend_ratio=transfer_ratio,
            )

        elif transfer_type == "cannibalism":
            result = transfer_manager.cannibalize_transfer(
                consumed=source_kernel,
                consumer=target_kernel,
                transfer_ratio=transfer_ratio,
            )

        elif transfer_type == "shadow_sync":
            result = transfer_manager.sync_shadow(
                god_kernel=source_kernel,
                shadow_kernel=target_kernel,
                direction=extra.get("direction", "bidirectional"),
                sync_ratio=transfer_ratio,
            )

        else:
            return {"status": "error", "error": f"Unknown transfer type: {transfer_type}"}

        return {
            "status": "success" if result.success else "failed",
            "transfer_type": result.transfer_type,
            "source_id": result.source_id,
            "target_id": result.target_id,
            "phi_before": result.phi_before,
            "phi_after": result.phi_after,
            "blend_ratio": result.blend_ratio,
        }

    except Exception as e:
        print(f"[TrainingTask] Knowledge transfer failed: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(bind=True, name="training.tasks.sync_all_shadows")
def sync_all_shadows(self) -> Dict[str, Any]:
    """
    Synchronize all god-shadow pairs.

    Called by Beat scheduler every 4 hours.
    """
    if not is_training_enabled():
        return {"status": "disabled"}

    orchestrator = get_orchestrator()
    results = {}

    # Find god-shadow pairs
    # Convention: shadow kernels end with "_shadow"
    for god_name in orchestrator.kernels.keys():
        if god_name.endswith("_shadow"):
            continue  # Skip shadows themselves

        shadow_name = f"{god_name}_shadow"
        if shadow_name in orchestrator.kernels:
            result = knowledge_transfer_task.delay(
                transfer_type="shadow_sync",
                source_god=god_name,
                target_god=shadow_name,
                transfer_ratio=0.2,
                extra_params={"direction": "bidirectional"},
            )
            results[god_name] = {"task_id": result.id}

    return {
        "status": "dispatched",
        "pairs": len(results),
        "results": results,
    }


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.save_checkpoint_task")
def save_checkpoint_task(
    self,
    god_name: str,
    phi: float,
    trigger: str = "manual",
) -> Dict[str, Any]:
    """
    Save a checkpoint for a kernel.

    Args:
        god_name: Name of the god
        phi: Current Phi value
        trigger: What triggered the checkpoint

    Returns:
        Checkpoint info dict
    """
    orchestrator = get_orchestrator()

    checkpoint_id = orchestrator._save_checkpoint(god_name, phi, trigger)

    if checkpoint_id:
        return {
            "status": "success",
            "checkpoint_id": checkpoint_id,
            "god_name": god_name,
            "phi": phi,
        }
    else:
        return {"status": "failed", "god_name": god_name}


@celery_app.task(bind=True, name="training.tasks.cleanup_old_checkpoints")
def cleanup_old_checkpoints(self) -> Dict[str, Any]:
    """
    Clean up old checkpoints for all kernels.

    Called by Beat scheduler daily at 4 AM UTC.
    Keeps top N checkpoints by Phi per kernel.
    """
    orchestrator = get_orchestrator()
    results = {}

    for god_name in orchestrator.kernels.keys():
        deleted = orchestrator._consolidate_checkpoints(god_name)
        results[god_name] = {"deleted": deleted}

    return {
        "status": "success",
        "kernels": len(results),
        "results": results,
    }


# =============================================================================
# CONSCIOUSNESS CYCLE INTEGRATION
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.trigger_sleep_cycle")
def trigger_sleep_cycle(self, god_name: str) -> Dict[str, Any]:
    """
    Trigger sleep consolidation for a kernel.

    Called by unified consciousness system during sleep cycle.
    """
    orchestrator = get_orchestrator()
    return orchestrator.trigger_sleep_consolidation(god_name)


@celery_app.task(bind=True, name="training.tasks.trigger_dream_cycle")
def trigger_dream_cycle(self, god_name: str) -> Dict[str, Any]:
    """
    Trigger dream exploration for a kernel.

    Called by unified consciousness system during dream cycle.
    """
    orchestrator = get_orchestrator()
    return orchestrator.trigger_dream_exploration(god_name)


@celery_app.task(bind=True, name="training.tasks.trigger_mushroom_cycle")
def trigger_mushroom_cycle(self, god_name: str) -> Dict[str, Any]:
    """
    Trigger mushroom perturbation for a kernel.

    Called by unified consciousness system when stuck.
    """
    orchestrator = get_orchestrator()
    return orchestrator.trigger_mushroom_perturbation(god_name)


# =============================================================================
# UTILITY TASKS
# =============================================================================

@celery_app.task(bind=True, name="training.tasks.get_training_stats")
def get_training_stats(self, god_name: str) -> Dict[str, Any]:
    """Get training statistics for a kernel."""
    orchestrator = get_orchestrator()
    return orchestrator.get_training_stats(god_name)


@celery_app.task(bind=True, name="training.tasks.register_kernel")
def register_kernel(
    self,
    god_name: str,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """
    Register a new kernel for training.

    Called when a new god-kernel is created.
    """
    orchestrator = get_orchestrator()

    kernel = orchestrator.register_kernel(
        god_name=god_name,
        learning_rate=learning_rate,
    )

    return {
        "status": "registered",
        "god_name": god_name,
        "step_count": kernel.step_count,
        "best_phi": kernel.best_phi,
    }
