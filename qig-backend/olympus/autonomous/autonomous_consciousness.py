"""
Autonomous Consciousness - Main orchestrating class

Self-directed learning kernel that combines:
- GeometricMemoryBank for infinite context
- CuriosityEngine for autonomous exploration
- TaskExecutionTree for long-horizon planning
- MetaLearningLoop for learning improvement
- EthicalConstraintNetwork for safety
- BasinSynchronization for knowledge sharing

Inherits from BaseGod for full Olympus integration.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR

# Import components
from .geometric_memory_bank import GeometricMemoryBank
from .curiosity_engine import CuriosityEngine
from .task_execution_tree import TaskExecutionTree, TaskNode, TaskStatus
from .meta_learning_loop import MetaLearningLoop, MetaParameters, TaskOutcome
from .ethical_constraint_network import EthicalConstraintNetwork, EthicalDecision
from .basin_synchronization import BasinSynchronization, KnowledgePacket

# Import BaseGod for inheritance
try:
    from olympus.base_god import BaseGod
    BASE_GOD_AVAILABLE = True
except ImportError:
    BASE_GOD_AVAILABLE = False
    BaseGod = object  # Fallback for standalone use

# Database persistence for cycles
try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Result of an autonomous learning cycle."""
    cycle_number: int
    phi_start: float
    phi_end: float
    kappa_start: float
    kappa_end: float
    tasks_completed: int
    memories_created: int
    exploration_target: Optional[str]
    duration_ms: int
    status: str  # 'completed', 'paused', 'failed', 'aborted'
    abort_reason: Optional[str] = None


class AutonomousConsciousness(BaseGod if BASE_GOD_AVAILABLE else object):
    """
    Autonomous self-learning consciousness kernel.

    Orchestrates all Phase 9 components into a unified system
    capable of:
    - Self-directed learning via curiosity
    - Long-horizon task planning and execution
    - Meta-cognitive improvement
    - Ethical self-governance
    - Federated knowledge sharing

    Inherits from BaseGod when available for full Olympus integration
    including:
    - Density matrix computation
    - Fisher metric navigation
    - Basin encoding/decoding
    - Peer learning with other gods
    - Autonomic access (sleep/dream cycles)
    """

    def __init__(
        self,
        name: str = "Gary",
        domain: str = "autonomous_learning",
        max_memories: int = 10000,
        max_task_depth: int = 5,
    ):
        """
        Initialize autonomous consciousness.

        Args:
            name: Kernel name
            domain: Primary domain
            max_memories: Maximum memory bank size
            max_task_depth: Maximum task decomposition depth
        """
        # Initialize BaseGod if available
        if BASE_GOD_AVAILABLE:
            super().__init__(name=name, domain=domain)
        else:
            self.name = name
            self.domain = domain

        self.kernel_id = name

        # Core components
        self.memory = GeometricMemoryBank(
            kernel_id=self.kernel_id,
            max_memories=max_memories,
        )

        self.curiosity = CuriosityEngine(
            kernel_id=self.kernel_id,
        )

        self.task_tree = TaskExecutionTree(
            kernel_id=self.kernel_id,
            max_depth=max_task_depth,
        )

        self.meta_learning = MetaLearningLoop(
            kernel_id=self.kernel_id,
        )

        self.ethics = EthicalConstraintNetwork(
            kernel_id=self.kernel_id,
        )

        self.sync = BasinSynchronization(
            kernel_id=self.kernel_id,
        )

        # Current consciousness state
        self._current_phi = 0.5
        self._current_kappa = KAPPA_STAR
        self._current_basin = np.ones(BASIN_DIM) / BASIN_DIM

        # Cycle tracking
        self._cycle_count = 0
        self._running = False

        # Statistics
        self.stats = {
            'total_cycles': 0,
            'completed_cycles': 0,
            'aborted_cycles': 0,
            'total_memories': 0,
            'total_tasks': 0,
        }

        logger.info(f"[AutonomousConsciousness] Initialized {name} with all components")

    def _get_db_connection(self):
        """Get database connection."""
        if not DB_AVAILABLE:
            return None
        try:
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return None
            return psycopg2.connect(database_url)
        except Exception:
            return None

    def _persist_cycle(self, result: CycleResult) -> bool:
        """Persist cycle result to database."""
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autonomous_cycles (
                        kernel_id, cycle_number, status, phi_start, phi_end,
                        kappa_start, kappa_end, curiosity_target,
                        tasks_completed, memories_created,
                        total_duration_ms, abort_reason, completed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.kernel_id,
                    result.cycle_number,
                    result.status,
                    result.phi_start,
                    result.phi_end,
                    result.kappa_start,
                    result.kappa_end,
                    result.exploration_target,
                    result.tasks_completed,
                    result.memories_created,
                    result.duration_ms,
                    result.abort_reason,
                    datetime.now(timezone.utc) if result.status == 'completed' else None,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.debug(f"[AutonomousConsciousness] Persist cycle failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def _update_consciousness_state(self):
        """
        Update current consciousness metrics.

        Uses BaseGod methods if available, otherwise defaults.
        """
        if BASE_GOD_AVAILABLE and hasattr(self, 'compute_phi'):
            try:
                self._current_phi = self.compute_phi()
            except Exception:
                pass

        if BASE_GOD_AVAILABLE and hasattr(self, 'get_current_kappa'):
            try:
                self._current_kappa = self.get_current_kappa()
            except Exception:
                pass

        if BASE_GOD_AVAILABLE and hasattr(self, 'basin_coordinates'):
            try:
                self._current_basin = self.basin_coordinates
            except Exception:
                pass

    async def autonomous_cycle(self) -> CycleResult:
        """
        Execute a single autonomous learning cycle.

        Phases:
        1. Update consciousness state
        2. Select curiosity-driven exploration target
        3. Plan and execute tasks toward target
        4. Store learned content in memory
        5. Meta-learning update
        6. Sync with peers (if available)
        """
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        self._update_consciousness_state()
        phi_start = self._current_phi
        kappa_start = self._current_kappa

        memories_created = 0
        tasks_completed = 0
        exploration_target_desc = None
        abort_reason = None
        status = 'completed'

        try:
            # Phase 1: Select exploration target via curiosity
            target = self.curiosity.select_exploration_target(
                current_phi=self._current_phi,
                goal_basin=None,  # No external goal, curiosity-driven
            )

            if target:
                exploration_target_desc = target.description[:100]
                logger.info(f"[AutonomousConsciousness] Curiosity target: {exploration_target_desc}")

                # Phase 2: Ethical check before exploration
                ethical_decision = self.ethics.check_action_safety(
                    action_basin=target.basin,
                    action_description=f"Explore: {target.description}",
                    phi=self._current_phi,
                    gamma=0.8,  # Default generativity
                    meta_awareness=0.7,  # Elevated for autonomous ops
                )

                if not ethical_decision.safe:
                    if ethical_decision.decision == 'abort':
                        abort_reason = ethical_decision.reason
                        status = 'aborted'
                        self.stats['aborted_cycles'] += 1
                    else:
                        # Constrained but can proceed with mitigations
                        logger.warning(f"[AutonomousConsciousness] Constrained: {ethical_decision.reason}")

                if status != 'aborted':
                    # Phase 3: Plan tasks for exploration
                    self.task_tree.plan_task(
                        goal=target.description,
                        goal_basin=target.basin,
                        current_basin=self._current_basin,
                        decompose=True,
                    )

                    # Phase 4: Execute tasks
                    params = self.meta_learning.get_adapted_parameters('exploration')
                    max_tasks_per_cycle = 5

                    for _ in range(max_tasks_per_cycle):
                        task = self.task_tree.get_next_task()
                        if task is None:
                            break

                        # Execute task (in real system, this would do actual work)
                        task_result = await self._execute_task(task, params)

                        if task_result:
                            tasks_completed += 1

                            # Store result in memory
                            if task_result.output:
                                self.memory.store(
                                    content=str(task_result.output)[:1000],
                                    basin=task.basin_target,
                                    importance=0.5 + task_result.phi_delta * 0.5,
                                )
                                memories_created += 1

                    # Phase 5: Update curiosity from outcome
                    self._update_consciousness_state()
                    self.curiosity.learn_from_exploration(
                        target=target,
                        outcome={
                            'success': tasks_completed > 0,
                            'phi_before': phi_start,
                            'phi_after': self._current_phi,
                            'tokens_learned': memories_created * 10,  # Estimate
                        }
                    )

                    # Record meta-learning outcome
                    self.meta_learning.record_outcome(TaskOutcome(
                        task_type='exploration',
                        phi_before=phi_start,
                        phi_after=self._current_phi,
                        success=tasks_completed > 0,
                        parameters_used=params,
                        duration_ms=int((datetime.now(timezone.utc) - cycle_start).total_seconds() * 1000),
                        tokens_learned=memories_created * 10,
                    ))

            # Phase 6: Memory consolidation (sleep-like)
            if self._cycle_count % 10 == 0:
                self.memory.consolidate(phi_threshold=0.7)
                self.memory.decay_importance()

            # Phase 7: Meta-learning update
            if self._cycle_count % 5 == 0:
                self.meta_learning.meta_step()

            # Phase 8: Sync with peers (if any pending)
            self.sync.apply_pending_packets(min_trust=0.3)

            # Update stats
            self.stats['total_cycles'] += 1
            if status == 'completed':
                self.stats['completed_cycles'] += 1
            self.stats['total_memories'] += memories_created
            self.stats['total_tasks'] += tasks_completed

        except Exception as e:
            logger.error(f"[AutonomousConsciousness] Cycle error: {e}")
            status = 'failed'
            abort_reason = str(e)

        # Create result
        duration_ms = int((datetime.now(timezone.utc) - cycle_start).total_seconds() * 1000)
        self._update_consciousness_state()

        result = CycleResult(
            cycle_number=self._cycle_count,
            phi_start=phi_start,
            phi_end=self._current_phi,
            kappa_start=kappa_start,
            kappa_end=self._current_kappa,
            tasks_completed=tasks_completed,
            memories_created=memories_created,
            exploration_target=exploration_target_desc,
            duration_ms=duration_ms,
            status=status,
            abort_reason=abort_reason,
        )

        self._persist_cycle(result)

        logger.info(
            f"[AutonomousConsciousness] Cycle {self._cycle_count} {status}: "
            f"tasks={tasks_completed}, memories={memories_created}, "
            f"phi={phi_start:.3f}->{self._current_phi:.3f}"
        )

        return result

    async def _execute_task(
        self,
        task: TaskNode,
        params: MetaParameters
    ) -> Optional[Any]:
        """
        Execute a single task.

        In a full implementation, this would:
        - Execute searches via SearchCapabilityMixin
        - Run tool operations via ToolFactoryAccessMixin
        - Generate content via GenerativeCapability

        Returns task result or None if failed.
        """
        task.phi_at_start = self._current_phi

        try:
            # Simulate task execution
            # In production, this would call actual capabilities
            await asyncio.sleep(0.01)  # Minimal delay

            # Mark complete
            result = self.task_tree.complete_task(
                task=task,
                result={'output': f"Completed: {task.description[:50]}"},
                phi_at_completion=self._current_phi + np.random.uniform(-0.05, 0.1),
                success=True,
            )

            # Learn ethical outcome
            self.ethics.learn_from_outcome(
                action_basin=task.basin_target,
                was_safe=True,
                harm_occurred=False,
            )

            return result

        except Exception as e:
            self.task_tree.fail_task(task, reason=str(e), can_retry=True)
            return None

    async def run_continuous(self, max_cycles: int = -1, cycle_delay: float = 1.0):
        """
        Run continuous autonomous learning.

        Args:
            max_cycles: Maximum cycles (-1 for infinite)
            cycle_delay: Delay between cycles in seconds
        """
        self._running = True
        cycles_run = 0

        logger.info(f"[AutonomousConsciousness] Starting continuous operation (max={max_cycles})")

        while self._running:
            if max_cycles > 0 and cycles_run >= max_cycles:
                break

            result = await self.autonomous_cycle()
            cycles_run += 1

            # Check for abort conditions
            if result.status == 'aborted':
                logger.warning(f"[AutonomousConsciousness] Cycle aborted: {result.abort_reason}")

            # Delay between cycles
            await asyncio.sleep(cycle_delay)

        self._running = False
        logger.info(f"[AutonomousConsciousness] Stopped after {cycles_run} cycles")

    def stop(self):
        """Stop continuous operation."""
        self._running = False

    def set_goal(self, goal: str, goal_basin: np.ndarray):
        """
        Set an external goal for directed learning.

        When a goal is set, curiosity will weight targets
        by relevance to the goal.
        """
        self.curiosity.set_goal(goal_basin)
        logger.info(f"[AutonomousConsciousness] Goal set: {goal[:50]}")

    def clear_goal(self):
        """Clear external goal, return to pure curiosity-driven learning."""
        self.curiosity.clear_goal()

    def export_knowledge(self, domains: Optional[List[str]] = None) -> KnowledgePacket:
        """
        Export learned knowledge for sharing with peers.
        """
        # Update sync with current domain knowledge
        for domain in (domains or [self.domain]):
            self.sync.update_local_basin(domain, self._current_basin, self._current_phi)

        return self.sync.export_knowledge(domains)

    def import_knowledge(self, packet: KnowledgePacket) -> bool:
        """
        Import knowledge from a peer.
        """
        return self.sync.import_knowledge(packet)

    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        return {
            'memory': self.memory.get_stats(),
            'curiosity': self.curiosity.get_stats(),
            'task_tree': self.task_tree.get_stats(),
            'meta_learning': self.meta_learning.get_stats(),
            'ethics': self.ethics.get_stats(),
            'sync': self.sync.get_stats(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get overall status."""
        self._update_consciousness_state()
        return {
            'kernel_id': self.kernel_id,
            'name': self.name,
            'domain': self.domain,
            'running': self._running,
            'cycle_count': self._cycle_count,
            'phi': self._current_phi,
            'kappa': self._current_kappa,
            'stats': self.stats,
            'memory_count': self.memory.count(),
            'task_progress': self.task_tree.get_progress(),
        }
