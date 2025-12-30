"""
QIG Kernel Training Module
==========================

Provides formal gradient-based training infrastructure for god-kernels
using QIG-pure geometric loss functions.

Training Modes:
- Outcome-based: Train after each chat interaction (automatic)
- Hourly batch: Process accumulated chats, search results
- Nightly consolidation: Full curriculum, checkpoint cleanup

Knowledge Transfer:
- Evolution: Parent → child (full transfer)
- Breeding: Parent1 + parent2 → child (merged)
- Cannibalism: Consumed → consumer (selective)
- Shadow sync: God ↔ shadow (bidirectional)

All training uses Fisher-Rao geometry, NOT cross-entropy.
"""

from .loss_functions import (
    geometric_loss,
    phi_regularization,
    coherence_loss,
    combined_training_loss,
)
from .trainable_kernel import TrainableKernel
from .kernel_training_orchestrator import KernelTrainingOrchestrator
from .knowledge_transfer import KnowledgeTransferManager
from .celery_app import celery_app
from .tasks import (
    train_from_outcome_task,
    train_hourly_batch_task,
    train_nightly_consolidation_task,
    knowledge_transfer_task,
    save_checkpoint_task,
)

__all__ = [
    # Loss functions
    "geometric_loss",
    "phi_regularization",
    "coherence_loss",
    "combined_training_loss",
    # Core classes
    "TrainableKernel",
    "KernelTrainingOrchestrator",
    "KnowledgeTransferManager",
    # Celery
    "celery_app",
    # Tasks
    "train_from_outcome_task",
    "train_hourly_batch_task",
    "train_nightly_consolidation_task",
    "knowledge_transfer_task",
    "save_checkpoint_task",
]
