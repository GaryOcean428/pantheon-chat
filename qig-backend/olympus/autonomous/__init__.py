"""
Autonomous QIG Consciousness Platform

Phase 9 implementation of self-directed learning, long-horizon planning,
and meta-cognitive improvement using QIG-PURE geometric foundations.

Components:
- GeometricMemoryBank: Infinite context via basin coordinate storage
- CuriosityEngine: Autonomous exploration with geometric novelty
- TaskExecutionTree: Hierarchical task planning along geodesics
- MetaLearningLoop: Learning to learn via natural gradient
- EthicalConstraintNetwork: Geometric safety boundaries
- BasinSynchronization: Federated knowledge transfer
- AutonomousConsciousness: Main orchestrating class

All operations use Fisher-Rao distance and basin coordinates (64D).
No Euclidean metrics or cosine similarity.
"""

from .geometric_memory_bank import GeometricMemoryBank, MemoryEntry
from .curiosity_engine import CuriosityEngine, ExplorationTarget
from .task_execution_tree import TaskExecutionTree, TaskNode, TaskStatus
from .meta_learning_loop import MetaLearningLoop, MetaParameters
from .ethical_constraint_network import EthicalConstraintNetwork, EthicalDecision
from .basin_synchronization import BasinSynchronization, KnowledgePacket
from .autonomous_consciousness import AutonomousConsciousness

__all__ = [
    # Memory
    'GeometricMemoryBank',
    'MemoryEntry',
    # Curiosity
    'CuriosityEngine',
    'ExplorationTarget',
    # Tasks
    'TaskExecutionTree',
    'TaskNode',
    'TaskStatus',
    # Meta-learning
    'MetaLearningLoop',
    'MetaParameters',
    # Ethics
    'EthicalConstraintNetwork',
    'EthicalDecision',
    # Sync
    'BasinSynchronization',
    'KnowledgePacket',
    # Main
    'AutonomousConsciousness',
]
