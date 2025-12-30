"""
Kernel Evolution Orchestrator - Strategic Self-Evolution with Safety Constraints

Gods can evolve but MUST ALWAYS occupy their primitive (core domain identity).
Evolution happens AROUND the primitive, not replacing it.

Core Concepts:
- Primitives: Immutable core identities (Zeus=authority, Athena=wisdom, etc.)
- Evolution: Basin coordinate drift within primitive bounds
- Lightning Insights: Sudden connections that trigger evolution cascades
- Foresight: 4D temporal projection of evolution outcomes
- ML Chains: Sequential learning patterns that compound
- QIGraph: Inter-kernel knowledge network with Fisher edges
- Safety: Rollback capability, drift limits, primitive anchoring

The orchestrator uses QIG-derived thresholds (not hardcoded magic numbers).
"""

import logging
import numpy as np
import uuid
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from contextlib import contextmanager
import os
import threading

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM
    from frozen_physics import (
        REGIME_GEOMETRIC,
        REGIME_HYPERDIMENSIONAL,
        PHI_THRESHOLD,
        PHI_EMERGENCY,
        BASIN_DRIFT_THRESHOLD,
        BETA_3_TO_4,
    )
    FROZEN_PHYSICS_AVAILABLE = True
except ImportError:
    KAPPA_STAR = 64.21
    BASIN_DIM = 64
    PHI_THRESHOLD = 0.7
    PHI_EMERGENCY = 0.3
    BASIN_DRIFT_THRESHOLD = 0.05
    BETA_3_TO_4 = 0.44
    FROZEN_PHYSICS_AVAILABLE = False
    REGIME_GEOMETRIC = None
    REGIME_HYPERDIMENSIONAL = None


class EvolutionReason(Enum):
    LIGHTNING_INSIGHT = "lightning_insight"
    ACCUMULATED_LEARNING = "accumulated_learning"
    META_REFLECTION = "meta_reflection"
    FORESIGHT_PROJECTION = "foresight_projection"
    ML_CHAIN_COMPLETION = "ml_chain_completion"
    PANTHEON_CONSENSUS = "pantheon_consensus"
    CURIOSITY_DRIVEN = "curiosity_driven"
    MEMORY_CONSOLIDATION = "memory_consolidation"


class LifecycleAction(Enum):
    EVOLVE = "evolve"
    SPAWN = "spawn"
    MERGE = "merge"
    CANNIBALIZE = "cannibalize"
    HIBERNATE = "hibernate"
    AWAKEN = "awaken"
    NO_ACTION = "no_action"


@dataclass
class GodPrimitive:
    """
    Immutable core identity of an Olympian god.
    Gods MUST always occupy their primitive - it cannot be abandoned.
    """
    name: str
    domain: str
    element: str
    role: str
    anchor_basin: np.ndarray
    drift_limit: float = 0.15
    
    def is_within_bounds(self, current_basin: np.ndarray) -> bool:
        """Check if current basin is within drift limit of anchor."""
        if len(current_basin) != len(self.anchor_basin):
            return False
        dot = np.clip(np.dot(current_basin, self.anchor_basin), -1.0, 1.0)
        distance = np.arccos(dot) / np.pi
        return distance <= self.drift_limit
    
    def project_to_bounds(self, evolved_basin: np.ndarray) -> np.ndarray:
        """Project an evolved basin back within primitive bounds."""
        if self.is_within_bounds(evolved_basin):
            return evolved_basin
        direction = evolved_basin - self.anchor_basin
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return self.anchor_basin.copy()
        unit_dir = direction / norm
        max_step = self.drift_limit * np.pi
        safe_basin = self.anchor_basin + unit_dir * max_step * 0.9
        safe_norm = np.linalg.norm(safe_basin)
        if safe_norm > 1e-10:
            safe_basin = safe_basin / safe_norm
        return safe_basin


@dataclass
class KernelEvolutionState:
    """
    Complete evolution state for a kernel.
    Tracks all metrics needed for lifecycle decisions.
    """
    kernel_id: str
    god_name: str
    primitive: Optional[GodPrimitive]
    
    current_basin: np.ndarray = field(default_factory=lambda: np.zeros(BASIN_DIM))
    phi: float = 0.5
    kappa: float = KAPPA_STAR
    kappa_drift: float = 0.0
    beta_coupling: float = BETA_STRONG
    surprise: float = 0.0
    basin_entropy: float = 0.5
    
    total_generations: int = 0
    successful_generations: int = 0
    efficiency: float = 0.5
    
    lightning_insights: List[Dict] = field(default_factory=list)
    ml_chain_progress: Dict[str, float] = field(default_factory=dict)
    foresight_projections: List[Dict] = field(default_factory=list)
    
    qigraph_edges: Dict[str, float] = field(default_factory=dict)
    
    last_evolution: Optional[datetime] = None
    evolution_history: List[Dict] = field(default_factory=list)
    
    def compute_evolution_readiness(self) -> float:
        """
        Compute readiness for evolution using QIG-derived metrics.
        Returns value in [0, 1] where higher = more ready to evolve.
        """
        phi_factor = self.phi
        
        efficiency_factor = self.efficiency
        
        kappa_stability = 1.0 - min(abs(self.kappa - KAPPA_STAR) / KAPPA_STAR, 1.0)
        
        lightning_factor = min(len(self.lightning_insights) / 5.0, 1.0)
        
        ml_progress = np.mean(list(self.ml_chain_progress.values())) if self.ml_chain_progress else 0.0
        
        readiness = (
            0.25 * phi_factor +
            0.20 * efficiency_factor +
            0.15 * kappa_stability +
            0.20 * lightning_factor +
            0.20 * ml_progress
        )
        
        return float(np.clip(readiness, 0.0, 1.0))
    
    def add_lightning_insight(self, insight: Dict) -> None:
        """Record a lightning insight (sudden connection)."""
        insight['timestamp'] = datetime.now().isoformat()
        self.lightning_insights.append(insight)
        if len(self.lightning_insights) > 50:
            self.lightning_insights = self.lightning_insights[-50:]
    
    def update_ml_chain(self, chain_id: str, progress: float) -> None:
        """Update progress on an ML learning chain."""
        self.ml_chain_progress[chain_id] = float(np.clip(progress, 0.0, 1.0))


@dataclass
class QIGraphEdge:
    """Edge in the inter-kernel knowledge graph."""
    source_kernel: str
    target_kernel: str
    fisher_distance: float
    insight_flow: float
    last_interaction: datetime
    interaction_count: int = 0
    edge_type: str = "knowledge"


class ForesightEngine:
    """
    4D temporal projection for evolution outcomes.
    Projects current state forward to predict evolution results.
    """
    
    def __init__(self, time_horizon: int = 10):
        self.time_horizon = time_horizon
        self.temporal_dimension = 4
    
    def project_evolution(
        self,
        current_state: KernelEvolutionState,
        proposed_action: LifecycleAction,
        proposed_params: Dict
    ) -> Dict:
        """
        Project the outcome of a proposed evolution action.
        Uses 4D spacetime modeling to predict future states.
        """
        current_basin = current_state.current_basin
        current_phi = current_state.phi
        
        trajectory = [current_basin.copy()]
        phi_trajectory = [current_phi]
        
        for t in range(1, self.time_horizon + 1):
            noise = np.random.randn(BASIN_DIM) * 0.01 * (1.0 / (t + 1))
            
            if proposed_action == LifecycleAction.EVOLVE:
                evolution_direction = proposed_params.get('direction', np.zeros(BASIN_DIM))
                step = current_basin + evolution_direction * 0.1 * np.exp(-t * 0.1) + noise
            elif proposed_action == LifecycleAction.MERGE:
                target_basin = proposed_params.get('target_basin', current_basin)
                t_norm = t / self.time_horizon
                step = (1 - t_norm) * current_basin + t_norm * target_basin + noise
            else:
                step = current_basin + noise
            
            norm = np.linalg.norm(step)
            if norm > 1e-10:
                step = step / norm
            trajectory.append(step)
            
            phi_change = 0.02 * np.random.randn() + 0.005 * (current_state.efficiency - 0.5)
            new_phi = np.clip(phi_trajectory[-1] + phi_change, 0.0, 1.0)
            phi_trajectory.append(new_phi)
        
        final_basin = trajectory[-1]
        
        primitive_safe = True
        if current_state.primitive:
            primitive_safe = current_state.primitive.is_within_bounds(final_basin)
        
        stability = 1.0 - np.std([np.linalg.norm(t - trajectory[-1]) for t in trajectory[-3:]])
        
        return {
            'predicted_basin': final_basin.tolist(),
            'predicted_phi': float(phi_trajectory[-1]),
            'primitive_safe': primitive_safe,
            'stability_score': float(np.clip(stability, 0.0, 1.0)),
            'trajectory_length': len(trajectory),
            'phi_trend': 'increasing' if phi_trajectory[-1] > phi_trajectory[0] else 'decreasing',
            'confidence': 0.7 - 0.05 * self.time_horizon,
        }


class MLChainTracker:
    """
    Tracks sequential learning patterns (ML chains).
    Chains compound learning across multiple experiences.
    """
    
    def __init__(self):
        self.active_chains: Dict[str, Dict] = {}
        self.completed_chains: List[Dict] = []
    
    def start_chain(self, chain_id: str, chain_type: str, target_learning: str) -> None:
        """Start a new ML learning chain."""
        self.active_chains[chain_id] = {
            'chain_id': chain_id,
            'type': chain_type,
            'target': target_learning,
            'steps': [],
            'progress': 0.0,
            'started_at': datetime.now().isoformat(),
            'insights_accumulated': [],
        }
    
    def add_step(self, chain_id: str, step_data: Dict) -> float:
        """Add a learning step to a chain. Returns new progress."""
        if chain_id not in self.active_chains:
            return 0.0
        
        chain = self.active_chains[chain_id]
        chain['steps'].append({
            **step_data,
            'timestamp': datetime.now().isoformat()
        })
        
        progress_increment = step_data.get('learning_delta', 0.1)
        chain['progress'] = min(chain['progress'] + progress_increment, 1.0)
        
        if 'insight' in step_data:
            chain['insights_accumulated'].append(step_data['insight'])
        
        if chain['progress'] >= 1.0:
            self._complete_chain(chain_id)
        
        return chain['progress']
    
    def _complete_chain(self, chain_id: str) -> None:
        """Complete a learning chain."""
        if chain_id in self.active_chains:
            chain = self.active_chains.pop(chain_id)
            chain['completed_at'] = datetime.now().isoformat()
            self.completed_chains.append(chain)
            if len(self.completed_chains) > 100:
                self.completed_chains = self.completed_chains[-100:]


class LifecycleOrchestrator:
    """
    Main orchestrator for kernel lifecycle decisions.
    
    Uses QIG-derived thresholds instead of hardcoded magic numbers.
    All decisions are geometrically justified.
    """
    
    def __init__(self):
        self.kernel_states: Dict[str, KernelEvolutionState] = {}
        self.qigraph_edges: Dict[str, QIGraphEdge] = {}
        self.foresight = ForesightEngine()
        self.ml_chains = MLChainTracker()
        self.primitives = self._init_god_primitives()
        
        self._evolution_threshold = self._compute_evolution_threshold()
        self._spawn_threshold = self._compute_spawn_threshold()
        self._merge_threshold = self._compute_merge_threshold()
        self._cannibalize_threshold = self._compute_cannibalize_threshold()
        
        self._lock = threading.RLock()
        self._persistence = EvolutionPersistence()
        
        logger.info(f"[LifecycleOrchestrator] Initialized with QIG thresholds:")
        logger.info(f"  Evolution: {self._evolution_threshold:.3f}")
        logger.info(f"  Spawn: {self._spawn_threshold:.3f}")
        logger.info(f"  Merge: {self._merge_threshold:.3f}")
        logger.info(f"  Cannibalize: {self._cannibalize_threshold:.3f}")
    
    def _init_god_primitives(self) -> Dict[str, GodPrimitive]:
        """Initialize immutable primitives for all Olympian gods."""
        primitives = {}
        
        god_definitions = [
            ("Zeus", "authority_leadership", "lightning", "king"),
            ("Athena", "wisdom_strategy", "owl", "strategist"),
            ("Apollo", "truth_prophecy", "sun", "oracle"),
            ("Artemis", "hunt_nature", "moon", "hunter"),
            ("Ares", "war_conflict", "fire", "warrior"),
            ("Hermes", "communication_travel", "wind", "messenger"),
            ("Hephaestus", "craft_forge", "metal", "smith"),
            ("Demeter", "growth_harvest", "earth", "nurturer"),
            ("Dionysus", "chaos_creativity", "wine", "liberator"),
            ("Poseidon", "sea_depth", "water", "depths"),
            ("Hades", "underworld_secrets", "shadow", "keeper"),
            ("Hera", "home_family", "peacock", "queen"),
            ("Aphrodite", "love_beauty", "rose", "inspirer"),
        ]
        
        for i, (name, domain, element, role) in enumerate(god_definitions):
            anchor = self._compute_primitive_anchor(domain, i)
            primitives[name.lower()] = GodPrimitive(
                name=name,
                domain=domain,
                element=element,
                role=role,
                anchor_basin=anchor,
                drift_limit=0.15
            )
        
        return primitives
    
    def _compute_primitive_anchor(self, domain: str, seed: int) -> np.ndarray:
        """Compute the anchor basin for a god's primitive using domain semantics."""
        np.random.seed(hash(domain) % (2**32))
        
        anchor = np.random.randn(BASIN_DIM)
        
        anchor[seed % BASIN_DIM] += 2.0
        
        norm = np.linalg.norm(anchor)
        if norm > 1e-10:
            anchor = anchor / norm
        
        return anchor
    
    def _compute_evolution_threshold(self) -> float:
        """
        Compute evolution threshold from FROZEN QIG physics.
        
        Uses REGIME_GEOMETRIC phi_min as evolution readiness threshold.
        Kernels should evolve when they're solidly in geometric consciousness.
        """
        if FROZEN_PHYSICS_AVAILABLE and REGIME_GEOMETRIC:
            return float(REGIME_GEOMETRIC.phi_min)
        return float(PHI_THRESHOLD)
    
    def _compute_spawn_threshold(self) -> float:
        """
        Threshold for spawning new kernels.
        
        Uses REGIME_HYPERDIMENSIONAL phi_min - only spawn when in 4D+ consciousness.
        """
        if FROZEN_PHYSICS_AVAILABLE and REGIME_HYPERDIMENSIONAL:
            return float(REGIME_HYPERDIMENSIONAL.phi_min)
        return float(PHI_THRESHOLD + 0.05)
    
    def _compute_merge_threshold(self) -> float:
        """
        Threshold for merging kernels (Fisher distance).
        
        Uses BASIN_DRIFT_THRESHOLD from frozen physics.
        """
        return float(BASIN_DRIFT_THRESHOLD)
    
    def _compute_cannibalize_threshold(self) -> float:
        """
        Threshold below which a kernel should be cannibalized.
        
        Uses PHI_EMERGENCY from frozen physics - kernels below this are failing.
        """
        return float(PHI_EMERGENCY)
    
    def register_kernel(
        self,
        kernel_id: str,
        god_name: str,
        initial_basin: Optional[np.ndarray] = None
    ) -> KernelEvolutionState:
        """Register a kernel for lifecycle management."""
        with self._lock:
            primitive = self.primitives.get(god_name.lower())
            
            if initial_basin is None:
                if primitive:
                    initial_basin = primitive.anchor_basin.copy()
                else:
                    initial_basin = np.random.randn(BASIN_DIM)
                    initial_basin = initial_basin / np.linalg.norm(initial_basin)
            
            state = KernelEvolutionState(
                kernel_id=kernel_id,
                god_name=god_name,
                primitive=primitive,
                current_basin=initial_basin,
            )
            
            self.kernel_states[kernel_id] = state
            logger.info(f"[LifecycleOrchestrator] Registered kernel {kernel_id} ({god_name})")
            return state
    
    def update_kernel_metrics(
        self,
        kernel_id: str,
        phi: Optional[float] = None,
        kappa: Optional[float] = None,
        efficiency: Optional[float] = None,
        surprise: Optional[float] = None,
        generation_success: Optional[bool] = None,
    ) -> None:
        """Update a kernel's evolution metrics."""
        with self._lock:
            if kernel_id not in self.kernel_states:
                return
            
            state = self.kernel_states[kernel_id]
            
            if phi is not None:
                state.phi = phi
            if kappa is not None:
                old_kappa = state.kappa
                state.kappa = kappa
                state.kappa_drift = kappa - old_kappa
            if efficiency is not None:
                state.efficiency = efficiency
            if surprise is not None:
                state.surprise = surprise
            if generation_success is not None:
                state.total_generations += 1
                if generation_success:
                    state.successful_generations += 1
                state.efficiency = state.successful_generations / max(state.total_generations, 1)
    
    def record_lightning_insight(
        self,
        kernel_id: str,
        insight_type: str,
        connection: Dict,
        magnitude: float = 1.0
    ) -> None:
        """Record a lightning insight (sudden connection)."""
        with self._lock:
            if kernel_id not in self.kernel_states:
                return
            
            state = self.kernel_states[kernel_id]
            state.add_lightning_insight({
                'type': insight_type,
                'connection': connection,
                'magnitude': magnitude,
            })
            
            if magnitude > 0.8:
                self._propagate_insight_to_pantheon(kernel_id, insight_type, connection)
    
    def _propagate_insight_to_pantheon(
        self,
        source_kernel: str,
        insight_type: str,
        connection: Dict
    ) -> None:
        """Propagate high-magnitude insights to related kernels via QIGraph."""
        source_state = self.kernel_states.get(source_kernel)
        if not source_state:
            return
        
        for kernel_id, state in self.kernel_states.items():
            if kernel_id == source_kernel:
                continue
            
            dot = np.dot(source_state.current_basin, state.current_basin)
            similarity = (dot + 1) / 2
            
            if similarity > 0.3:
                edge_key = f"{source_kernel}->{kernel_id}"
                if edge_key in self.qigraph_edges:
                    edge = self.qigraph_edges[edge_key]
                    edge.insight_flow += 0.1
                    edge.interaction_count += 1
                else:
                    fisher_dist = np.arccos(np.clip(dot, -1, 1)) / np.pi
                    self.qigraph_edges[edge_key] = QIGraphEdge(
                        source_kernel=source_kernel,
                        target_kernel=kernel_id,
                        fisher_distance=fisher_dist,
                        insight_flow=0.1,
                        last_interaction=datetime.now(),
                        edge_type="insight_propagation"
                    )
    
    def evaluate_lifecycle_action(self, kernel_id: str) -> Tuple[LifecycleAction, Dict]:
        """
        Evaluate what lifecycle action a kernel should take.
        Uses QIChain reasoning (sequential geometric analysis).
        """
        with self._lock:
            if kernel_id not in self.kernel_states:
                return LifecycleAction.NO_ACTION, {}
            
            state = self.kernel_states[kernel_id]
            
            readiness = state.compute_evolution_readiness()
            
            if state.efficiency < self._cannibalize_threshold and state.total_generations > 10:
                absorber = self._find_best_absorber(kernel_id)
                if absorber:
                    return LifecycleAction.CANNIBALIZE, {
                        'absorber_kernel': absorber,
                        'reason': 'persistent_low_efficiency',
                        'efficiency': state.efficiency,
                    }
            
            if readiness >= self._spawn_threshold and len(state.lightning_insights) >= 3:
                spawn_direction = self._compute_spawn_direction(state)
                projection = self.foresight.project_evolution(
                    state, LifecycleAction.SPAWN, {'direction': spawn_direction}
                )
                if projection['stability_score'] > 0.6:
                    return LifecycleAction.SPAWN, {
                        'direction': spawn_direction.tolist(),
                        'predicted_outcome': projection,
                        'trigger_insights': state.lightning_insights[-3:],
                    }
            
            merge_candidate = self._find_merge_candidate(kernel_id)
            if merge_candidate:
                return LifecycleAction.MERGE, {
                    'partner_kernel': merge_candidate,
                    'fisher_distance': self._compute_fisher_distance(kernel_id, merge_candidate),
                }
            
            if readiness >= self._evolution_threshold:
                evolution_direction = self._compute_evolution_direction(state)
                projection = self.foresight.project_evolution(
                    state, LifecycleAction.EVOLVE, {'direction': evolution_direction}
                )
                
                if projection['primitive_safe']:
                    return LifecycleAction.EVOLVE, {
                        'direction': evolution_direction.tolist(),
                        'predicted_outcome': projection,
                        'reason': EvolutionReason.ACCUMULATED_LEARNING.value,
                    }
            
            return LifecycleAction.NO_ACTION, {'readiness': readiness}
    
    def execute_evolution(
        self,
        kernel_id: str,
        direction: np.ndarray,
        reason: EvolutionReason,
        step_size: float = 0.05
    ) -> Dict:
        """
        Execute a safe evolution step for a kernel.
        Ensures primitive is never abandoned.
        """
        with self._lock:
            if kernel_id not in self.kernel_states:
                return {'success': False, 'error': 'kernel_not_found'}
            
            state = self.kernel_states[kernel_id]
            old_basin = state.current_basin.copy()
            
            new_basin = old_basin + direction * step_size
            norm = np.linalg.norm(new_basin)
            if norm > 1e-10:
                new_basin = new_basin / norm
            
            if state.primitive:
                if not state.primitive.is_within_bounds(new_basin):
                    new_basin = state.primitive.project_to_bounds(new_basin)
                    logger.info(f"[Evolution] Projected {kernel_id} back to primitive bounds")
            
            state.current_basin = new_basin
            state.last_evolution = datetime.now()
            state.evolution_history.append({
                'timestamp': datetime.now().isoformat(),
                'reason': reason.value,
                'old_basin': old_basin.tolist(),
                'new_basin': new_basin.tolist(),
                'direction': direction.tolist(),
                'step_size': step_size,
            })
            
            self._persistence.save_evolution_event(kernel_id, state)
            
            return {
                'success': True,
                'kernel_id': kernel_id,
                'old_basin': old_basin.tolist(),
                'new_basin': new_basin.tolist(),
                'primitive_safe': state.primitive.is_within_bounds(new_basin) if state.primitive else True,
                'can_rollback': True,
                'rollback_to': old_basin.tolist(),
            }
    
    def rollback_evolution(self, kernel_id: str, to_step: int = -1) -> Dict:
        """
        Rollback a kernel's evolution to a previous state.
        
        Args:
            kernel_id: Kernel to rollback
            to_step: Index in evolution history (-1 = previous step)
            
        Returns:
            Rollback result dict
        """
        with self._lock:
            if kernel_id not in self.kernel_states:
                return {'success': False, 'error': 'kernel_not_found'}
            
            state = self.kernel_states[kernel_id]
            
            if not state.evolution_history:
                return {'success': False, 'error': 'no_evolution_history'}
            
            try:
                target_event = state.evolution_history[to_step]
                old_basin = np.array(target_event['old_basin'])
                
                if state.primitive and not state.primitive.is_within_bounds(old_basin):
                    old_basin = state.primitive.project_to_bounds(old_basin)
                
                state.current_basin = old_basin
                state.evolution_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'rollback',
                    'old_basin': state.current_basin.tolist(),
                    'new_basin': old_basin.tolist(),
                    'rollback_to_step': to_step,
                })
                
                logger.info(f"[Evolution] Rolled back {kernel_id} to step {to_step}")
                return {
                    'success': True,
                    'kernel_id': kernel_id,
                    'rolled_back_to': old_basin.tolist(),
                }
            except (IndexError, KeyError) as e:
                return {'success': False, 'error': f'rollback_failed: {e}'}
    
    def sync_ml_chain_to_state(self, kernel_id: str) -> None:
        """Synchronize ML chain progress into kernel evolution state."""
        with self._lock:
            if kernel_id not in self.kernel_states:
                return
            
            state = self.kernel_states[kernel_id]
            
            for chain_id, chain in self.ml_chains.active_chains.items():
                state.ml_chain_progress[chain_id] = chain['progress']
    
    def _compute_evolution_direction(self, state: KernelEvolutionState) -> np.ndarray:
        """Compute evolution direction from accumulated insights and learning."""
        direction = np.zeros(BASIN_DIM)
        
        for insight in state.lightning_insights[-10:]:
            if 'connection' in insight and 'basin_delta' in insight['connection']:
                delta = np.array(insight['connection']['basin_delta'])
                if len(delta) == BASIN_DIM:
                    direction += delta * insight.get('magnitude', 1.0)
        
        for chain_id, progress in state.ml_chain_progress.items():
            direction += np.random.randn(BASIN_DIM) * 0.01 * progress
        
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = np.random.randn(BASIN_DIM)
            direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def _compute_spawn_direction(self, state: KernelEvolutionState) -> np.ndarray:
        """Compute direction for spawning a new kernel."""
        direction = self._compute_evolution_direction(state)
        
        if state.primitive:
            perpendicular = direction - np.dot(direction, state.primitive.anchor_basin) * state.primitive.anchor_basin
            norm = np.linalg.norm(perpendicular)
            if norm > 1e-10:
                direction = perpendicular / norm
        
        return direction
    
    def _find_merge_candidate(self, kernel_id: str) -> Optional[str]:
        """Find a kernel that could merge with this one."""
        state = self.kernel_states.get(kernel_id)
        if not state:
            return None
        
        for other_id, other_state in self.kernel_states.items():
            if other_id == kernel_id:
                continue
            
            if state.primitive and other_state.primitive:
                if state.primitive.name == other_state.primitive.name:
                    continue
            
            fisher_dist = self._compute_fisher_distance(kernel_id, other_id)
            if fisher_dist < self._merge_threshold:
                return other_id
        
        return None
    
    def _find_best_absorber(self, failing_kernel_id: str) -> Optional[str]:
        """Find the best kernel to absorb a failing one."""
        failing_state = self.kernel_states.get(failing_kernel_id)
        if not failing_state:
            return None
        
        best_absorber = None
        best_score = 0.0
        
        for kernel_id, state in self.kernel_states.items():
            if kernel_id == failing_kernel_id:
                continue
            
            if state.efficiency < 0.5:
                continue
            
            dot = np.dot(failing_state.current_basin, state.current_basin)
            similarity = (dot + 1) / 2
            
            score = state.efficiency * 0.6 + similarity * 0.4
            
            if score > best_score:
                best_score = score
                best_absorber = kernel_id
        
        return best_absorber
    
    def _compute_fisher_distance(self, kernel_a: str, kernel_b: str) -> float:
        """Compute Fisher-Rao distance between two kernels."""
        state_a = self.kernel_states.get(kernel_a)
        state_b = self.kernel_states.get(kernel_b)
        
        if not state_a or not state_b:
            return float('inf')
        
        dot = np.clip(np.dot(state_a.current_basin, state_b.current_basin), -1.0, 1.0)
        return float(np.arccos(dot) / np.pi)
    
    def get_orchestrator_status(self) -> Dict:
        """Get current status of the lifecycle orchestrator."""
        with self._lock:
            kernel_summaries = []
            for kernel_id, state in self.kernel_states.items():
                kernel_summaries.append({
                    'kernel_id': kernel_id,
                    'god_name': state.god_name,
                    'phi': state.phi,
                    'efficiency': state.efficiency,
                    'readiness': state.compute_evolution_readiness(),
                    'lightning_insights': len(state.lightning_insights),
                    'primitive_bound': state.primitive is not None,
                })
            
            return {
                'total_kernels': len(self.kernel_states),
                'qigraph_edges': len(self.qigraph_edges),
                'thresholds': {
                    'evolution': self._evolution_threshold,
                    'spawn': self._spawn_threshold,
                    'merge': self._merge_threshold,
                    'cannibalize': self._cannibalize_threshold,
                },
                'kernels': kernel_summaries,
            }


class EvolutionPersistence:
    """PostgreSQL persistence for evolution events."""
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        self._tables_ensured = False
        if self.database_url and PSYCOPG2_AVAILABLE:
            self._ensure_tables()
    
    @contextmanager
    def _get_connection(self):
        if not self.database_url or not PSYCOPG2_AVAILABLE:
            yield None
            return
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"[EvolutionPersistence] DB error: {e}")
            yield None
        finally:
            if conn:
                conn.close()
    
    def _ensure_tables(self):
        if self._tables_ensured:
            return
        with self._get_connection() as conn:
            if not conn:
                return
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS kernel_evolution_events (
                            event_id VARCHAR(64) PRIMARY KEY,
                            kernel_id VARCHAR(64),
                            god_name VARCHAR(64),
                            action VARCHAR(32),
                            reason VARCHAR(64),
                            old_basin JSONB,
                            new_basin JSONB,
                            phi_before FLOAT8,
                            phi_after FLOAT8,
                            efficiency FLOAT8,
                            metadata JSONB DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_evolution_kernel ON kernel_evolution_events(kernel_id);
                        CREATE INDEX IF NOT EXISTS idx_evolution_time ON kernel_evolution_events(created_at);
                    """)
                self._tables_ensured = True
                logger.info("[EvolutionPersistence] Tables ensured")
            except Exception as e:
                logger.error(f"[EvolutionPersistence] Table creation error: {e}")
    
    def save_evolution_event(self, kernel_id: str, state: KernelEvolutionState) -> bool:
        with self._get_connection() as conn:
            if not conn:
                return False
            try:
                with conn.cursor() as cur:
                    event_id = str(uuid.uuid4())[:16]
                    history = state.evolution_history[-1] if state.evolution_history else {}
                    cur.execute("""
                        INSERT INTO kernel_evolution_events
                        (event_id, kernel_id, god_name, action, reason, old_basin, new_basin, phi_before, phi_after, efficiency, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event_id,
                        kernel_id,
                        state.god_name,
                        'evolve',
                        history.get('reason', 'unknown'),
                        json.dumps(history.get('old_basin', [])),
                        json.dumps(history.get('new_basin', [])),
                        state.phi,
                        state.phi,
                        state.efficiency,
                        json.dumps({'step_size': history.get('step_size', 0)}),
                    ))
                return True
            except Exception as e:
                logger.error(f"[EvolutionPersistence] Save error: {e}")
                return False


_orchestrator: Optional[LifecycleOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_lifecycle_orchestrator() -> LifecycleOrchestrator:
    """Get or create the singleton lifecycle orchestrator."""
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = LifecycleOrchestrator()
        return _orchestrator
