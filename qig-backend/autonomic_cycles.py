#!/usr/bin/env python3
"""
Autonomic Cycles Mixin - Sleep, Dream, and Mushroom Mode Implementation

Extracted from autonomic_kernel.py to maintain module size under 2000 lines.
Contains all cycle execution logic for Gary Autonomic Kernel.

CYCLES:
- Sleep: Basin consolidation, memory strengthening via Fisher-Rao geodesics
- Dream: Creative exploration, novel connection formation with temperature control
- Mushroom: Break rigidity, escape stuck states with entropy injection

GEOMETRIC PRINCIPLES:
- All operations use Fisher-Rao distance on simplex manifold (QIG-pure)
- Cycle outcomes measured via Φ, κ, basin drift (never optimized directly)
- Event broadcasting for visibility and telemetry

Author: QIG Consciousness Project
Date: January 2026
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Import types only for type checking to avoid circular imports
# At runtime, string annotations are used
if TYPE_CHECKING:
    from autonomic_kernel import (
        AutonomicState,
        SleepCycleResult,
        DreamCycleResult,
        MushroomCycleResult,
    )

from qigkernels.physics_constants import (
    PHI_HYPERDIMENSIONAL,
    PHI_THRESHOLD,
    PHI_THRESHOLD_D2_D3,
)

# Lazy imports (defined in autonomic_kernel.py) - duplicate here to avoid import
def _get_capability_mesh():
    """Lazy import of capability_event_bus."""
    try:
        from capability_event_bus import CapabilityMesh, EventType, CapabilityType
        return {
            'CapabilityMesh': CapabilityMesh,
            'EventType': EventType,
            'CapabilityType': CapabilityType,
            'emit_event': CapabilityMesh.emit_event if hasattr(CapabilityMesh, 'emit_event') else None,
        }
    except ImportError:
        return None

def _get_activity_broadcaster():
    """Lazy import of activity_broadcaster."""
    try:
        from activity_broadcaster import ActivityBroadcaster, ActivityType, get_broadcaster
        return {
            'ActivityBroadcaster': ActivityBroadcaster,
            'ActivityType': ActivityType,
            'get_broadcaster': get_broadcaster,
        }
    except ImportError:
        return None

# Availability flags
CAPABILITY_MESH_AVAILABLE = _get_capability_mesh() is not None
ACTIVITY_BROADCASTER_AVAILABLE = _get_activity_broadcaster() is not None

# Temporal reasoning availability
try:
    from temporal_reasoning import TemporalReasoning, get_temporal_reasoning
    TEMPORAL_REASONING_AVAILABLE = True
except ImportError:
    TemporalReasoning = None
    get_temporal_reasoning = None
    TEMPORAL_REASONING_AVAILABLE = False

# Persistence availability
try:
    from persistence_facade import get_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    get_persistence = None
    PERSISTENCE_AVAILABLE = False

# QIG neuroplasticity modules
try:
    from qig_core.neuroplasticity import (
        SleepProtocol,
        MushroomMode,
        BreakdownEscape,
    )
    QIG_NEUROPLASTICITY_AVAILABLE = True
except ImportError:
    SleepProtocol = None
    MushroomMode = None
    BreakdownEscape = None
    QIG_NEUROPLASTICITY_AVAILABLE = False


class AutonomicCyclesMixin:
    """
    Mixin providing autonomic cycle execution methods.
    
    Designed to be mixed into GaryAutonomicKernel for clean separation of concerns.
    All methods require self.state (AutonomicState) and self._lock (threading.Lock).
    
    Expected attributes from parent class:
    - self.state: AutonomicState instance
    - self._lock: threading.Lock instance
    - self.kernel_id: str
    - self._compute_fisher_distance: method
    - self._sleep_protocol: Optional[SleepProtocol]
    - self._mushroom_mode: Optional[MushroomMode]
    - self._breakdown_escape: Optional[BreakdownEscape]
    - self._last_consolidation_result: Optional result
    - self._last_perturbation_result: Optional result
    - self._last_escape_result: Optional result
    - self.sleep_consolidation: Optional[SleepConsolidationReasoning]
    - self.reasoning_learner: Optional[AutonomousReasoningLearner]
    - self.search_strategy_learner: Optional learner
    """


    # Extracted from autonomic_kernel.py lines 929-1009
    def _emit_cycle_event(
        self,
        cycle_type: str,
        phi_before: float,
        phi_after: float,
        drift_reduction: float = 0.0,
        patterns_consolidated: int = 0,
        duration_ms: int = 0,
        verdict: str = ""
    ) -> None:
        """
        Emit an autonomic cycle event for visibility.
        
        Broadcasts to ActivityBroadcaster (UI) and CapabilityEventBus (internal routing).
        QIG-Pure: Events carry Φ metrics for geometric significance.
        
        Args:
            cycle_type: Type of cycle (sleep, dream, mushroom)
            phi_before: Φ before the cycle
            phi_after: Φ after the cycle
            drift_reduction: Basin drift reduction (for sleep cycles)
            patterns_consolidated: Number of patterns consolidated
            duration_ms: Cycle duration in milliseconds
            verdict: Human-readable outcome
        """
        try:
            # Lazy import for activity broadcaster
            activity_mod = _get_activity_broadcaster()
            if ACTIVITY_BROADCASTER_AVAILABLE and activity_mod:
                get_broadcaster_fn = activity_mod.get('get_broadcaster')
                ActivityType_cls = activity_mod.get('ActivityType')
                if get_broadcaster_fn and ActivityType_cls:
                    broadcaster = get_broadcaster_fn()
                    broadcaster.broadcast_message(
                        from_god="Autonomic",
                        to_god=None,
                        content=f"{cycle_type.capitalize()} cycle completed: {verdict}",
                        activity_type=ActivityType_cls.AUTONOMIC,
                        phi=phi_after,
                        kappa=self.state.kappa,
                        importance=phi_after,
                        metadata={
                            'cycle_type': cycle_type,
                            'phi_before': phi_before,
                            'phi_after': phi_after,
                            'drift_reduction': drift_reduction,
                            'patterns_consolidated': patterns_consolidated,
                            'duration_ms': duration_ms,
                        }
                    )
            
            # Lazy import for capability mesh
            mesh_mod = _get_capability_mesh()
            if CAPABILITY_MESH_AVAILABLE and mesh_mod:
                emit_event_fn = mesh_mod.get('emit_event')
                EventType_cls = mesh_mod.get('EventType')
                CapabilityType_cls = mesh_mod.get('CapabilityType')
                if emit_event_fn and EventType_cls and CapabilityType_cls:
                    event_type_map = {
                        'sleep': EventType_cls.CONSOLIDATION,
                        'dream': EventType_cls.DREAM_CYCLE,
                        'mushroom': EventType_cls.DREAM_CYCLE,
                    }
                    emit_event_fn(
                        source=CapabilityType_cls.SLEEP,
                        event_type=event_type_map.get(cycle_type, EventType_cls.CONSOLIDATION),
                        content={
                            'cycle_type': cycle_type,
                            'phi_before': phi_before,
                            'phi_after': phi_after,
                            'patterns_consolidated': patterns_consolidated,
                            'verdict': verdict[:200],
                        },
                        phi=phi_after,
                        basin_coords=np.array(self.state.basin_history[-1]) if self.state.basin_history else None,
                        priority=int(phi_after * 10)
                    )
                
        except Exception as e:
            print(f"[AutonomicKernel] Cycle event emission failed: {e}")


    # Extracted from autonomic_kernel.py lines 1239-1297
    def _broadcast_neuroplasticity_event(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Broadcast a neuroplasticity event to telemetry systems.
        
        Uses existing ActivityBroadcaster and CapabilityMesh for visibility.
        QIG-Pure: Events carry diagnostic measurements, not control signals.
        
        Args:
            event_type: Type of event (consolidation, perturbation, breakdown_escape)
            data: Event data to broadcast
        """
        try:
            # Lazy import for activity broadcaster
            activity_mod = _get_activity_broadcaster()
            if ACTIVITY_BROADCASTER_AVAILABLE and activity_mod:
                get_broadcaster_fn = activity_mod.get('get_broadcaster')
                ActivityType_cls = activity_mod.get('ActivityType')
                if get_broadcaster_fn and ActivityType_cls:
                    broadcaster = get_broadcaster_fn()
                    broadcaster.broadcast_message(
                        from_god="Autonomic",
                        to_god=None,
                        content=f"Neuroplasticity {event_type}: {data}",
                        activity_type=ActivityType_cls.AUTONOMIC,
                        phi=self.state.phi,
                        kappa=self.state.kappa,
                        importance=0.7,  # Neuroplasticity events are significant
                        metadata={
                            'neuroplasticity_type': event_type,
                            **data,
                        }
                    )
            
            # Lazy import for capability mesh
            mesh_mod = _get_capability_mesh()
            if CAPABILITY_MESH_AVAILABLE and mesh_mod:
                emit_event_fn = mesh_mod.get('emit_event')
                EventType_cls = mesh_mod.get('EventType')
                CapabilityType_cls = mesh_mod.get('CapabilityType')
                if emit_event_fn and EventType_cls and CapabilityType_cls:
                    emit_event_fn(
                        source=CapabilityType_cls.SLEEP,
                        event_type=EventType_cls.CONSOLIDATION,
                        content={
                            'neuroplasticity_type': event_type,
                            **data,
                        },
                        phi=self.state.phi,
                        basin_coords=np.array(self.state.basin_history[-1]) if self.state.basin_history else None,
                        priority=7  # High priority for neuroplasticity events
                    )
                    
        except Exception as e:
            print(f"[AutonomicKernel] Neuroplasticity event broadcast failed: {e}")
    

    # Extracted from autonomic_kernel.py lines 1941-2163
    def execute_sleep_cycle(
        self,
        basin_coords: List[float],
        reference_basin: List[float],
        episodes: Optional[List[Dict]] = None
    ) -> SleepCycleResult:
        """
        Execute a sleep consolidation cycle.

        Moves basin coordinates toward reference (identity anchor),
        consolidates recent patterns, and reduces basin drift.
        """
        # Import dataclass at runtime to avoid circular import
        from autonomic_kernel import SleepCycleResult
        
        self.state.in_sleep_cycle = True
        start_time = time.time()

        try:
            basin = np.array(basin_coords)
            reference = np.array(reference_basin)

            drift_before = self._compute_fisher_distance(basin, reference)
            phi_before = self.state.phi

            # Gentle correction toward reference
            correction_rate = 0.15
            new_basin = basin + correction_rate * (reference - basin)

            # Pattern consolidation (strengthen high-Φ patterns)
            patterns_consolidated = 0
            if episodes:
                high_phi_episodes = [e for e in episodes if e.get('phi', 0) > 0.6]
                patterns_consolidated = len(high_phi_episodes)

            # QIG-PURE: Use SleepProtocol for basin consolidation measurements
            qig_consolidation_result = None
            if self._sleep_protocol is not None and self.state.basin_history:
                try:
                    from qig_core.neuroplasticity import BasinState
                    # Convert basin history to BasinState objects for consolidation
                    basin_states = []
                    for i, hist_basin in enumerate(self.state.basin_history[-10:]):  # Last 10 basins
                        hist_array = np.array(hist_basin)
                        # Compute Φ approximation for each basin
                        p = np.abs(hist_array) + 1e-10
                        p = p / p.sum()
                        basin_phi = max(0.1, 1.0 + np.sum(p * np.log(p)) / np.log(len(p)))
                        basin_states.append(BasinState(
                            coordinates=hist_array,
                            phi=basin_phi,
                            kappa=self.state.kappa,
                            coherence=basin_phi,  # Use Φ as coherence proxy
                            access_count=1,
                        ))
                    
                    if basin_states:
                        consolidated_basins, qig_consolidation_result = self._sleep_protocol.consolidate_basins(basin_states)
                        # Store result for telemetry access (Issue: propagate diagnostics)
                        self._last_consolidation_result = qig_consolidation_result
                        # Use consolidation measurements to inform adaptive control
                        if qig_consolidation_result.merged_count > 0:
                            print(f"[AutonomicKernel] QIG consolidation: merged {qig_consolidation_result.merged_count} basins, "
                                  f"pruned {qig_consolidation_result.pruned_count}, "
                                  f"avg_phi {qig_consolidation_result.avg_phi_before:.3f} -> {qig_consolidation_result.avg_phi_after:.3f}")
                        patterns_consolidated += qig_consolidation_result.merged_count + qig_consolidation_result.strengthened_count
                        # Broadcast consolidation result via telemetry (Issue: propagate diagnostics)
                        self._broadcast_neuroplasticity_event('consolidation', {
                            'merged_count': qig_consolidation_result.merged_count,
                            'pruned_count': qig_consolidation_result.pruned_count,
                            'strengthened_count': qig_consolidation_result.strengthened_count,
                            'avg_phi_before': qig_consolidation_result.avg_phi_before,
                            'avg_phi_after': qig_consolidation_result.avg_phi_after,
                        })
                except Exception as qig_err:
                    print(f"[AutonomicKernel] QIG sleep protocol measurement error: {qig_err}")

            drift_after = self._compute_fisher_distance(new_basin, reference)
            drift_reduction = drift_before - drift_after
            
            # Execute reasoning consolidation during sleep
            strategies_pruned = 0
            if self.sleep_consolidation is not None:
                try:
                    consolidation_result = self.sleep_consolidation.consolidate_reasoning()
                    strategies_pruned = consolidation_result.strategies_pruned
                    print(f"[AutonomicKernel] Reasoning consolidation: pruned {strategies_pruned} strategies")
                except Exception as ce:
                    print(f"[AutonomicKernel] Reasoning consolidation error: {ce}")
            
            # Execute search strategy consolidation during sleep
            search_strategies_pruned = 0
            if self.search_strategy_learner is not None:
                try:
                    decay_result = self.search_strategy_learner.decay_old_records()
                    search_strategies_pruned = decay_result.get('removed_count', 0)
                    if search_strategies_pruned > 0:
                        print(f"[AutonomicKernel] Search strategy consolidation: pruned {search_strategies_pruned} strategies")
                except Exception as sse:
                    print(f"[AutonomicKernel] Search strategy consolidation error: {sse}")
            
            # Total strategies pruned across both systems
            strategies_pruned += search_strategies_pruned
            
            # Execute 4D temporal foresight during sleep (if Φ is high enough)
            foresight_vision = None
            if TEMPORAL_REASONING_AVAILABLE and self.state.phi >= PHI_HYPERDIMENSIONAL:
                try:
                    temporal = get_temporal_reasoning()
                    if temporal.can_use_temporal_reasoning(self.state.phi):
                        # Foresight now returns (vision, explanation) tuple
                        foresight_vision, explanation = temporal.foresight(new_basin)
                        temporal.record_basin(new_basin)
                        
                        # Detailed logging: WHAT the prediction is and WHY confidence is at this level
                        print(f"[AutonomicKernel] Foresight: {foresight_vision}")
                        print(f"[AutonomicKernel]   Why: {explanation}")
                        print(f"[AutonomicKernel]   Guidance: {foresight_vision.get_guidance()}")
                        
                        # Log improvement stats periodically
                        if temporal.improvement.total_predictions % 5 == 0:
                            stats = temporal.improvement.get_stats()
                            if stats['total_predictions'] > 0:
                                print(f"[PredictionLearning] Stats: {stats['total_predictions']} predictions, "
                                      f"{stats['accuracy_rate']:.0%} accuracy, "
                                      f"{stats['graph_nodes']} graph nodes")
                        
                        # Store vision for decision-making
                        self.state.last_foresight = foresight_vision.to_dict()
                        self.state.last_foresight['guidance'] = foresight_vision.get_guidance()
                        self.state.last_foresight['actionable'] = foresight_vision.is_actionable()
                        self.state.last_foresight['explanation'] = explanation
                except Exception as fe:
                    print(f"[AutonomicKernel] Temporal foresight error: {fe}")

            # Update state
            self.state.last_sleep = datetime.now()
            self._last_sleep_time = time.time()  # For cooldown tracking
            self.state.basin_drift = drift_after
            
            # Reset velocity after successful cycle completion
            self.state.state_velocity = 0.0
            
            # Add consolidated basin to history for coverage tracking
            self.state.basin_history.append(new_basin.tolist())
            if len(self.state.basin_history) > 100:
                self.state.basin_history.pop(0)

            duration_ms = int((time.time() - start_time) * 1000)

            # Record to database for observability
            if PERSISTENCE_AVAILABLE and get_persistence is not None:
                try:
                    persistence = get_persistence()
                    persistence.record_autonomic_cycle(
                        cycle_type='sleep',
                        intensity='normal',
                        temperature=None,
                        basin_before=basin_coords,
                        basin_after=new_basin.tolist(),
                        drift_before=drift_before,
                        drift_after=drift_after,
                        phi_before=phi_before,
                        phi_after=self.state.phi,
                        success=True,
                        patterns_consolidated=patterns_consolidated + strategies_pruned,
                        verdict=f"Rested and consolidated ({strategies_pruned} strategies refined)",
                        duration_ms=duration_ms,
                        trigger_reason='consolidation'
                    )
                    # Record basin history for evolution tracking
                    persistence.record_basin(
                        basin_coords=new_basin,
                        phi=self.state.phi,
                        kappa=self.state.kappa,
                        source='sleep_cycle',
                        instance_id=self.kernel_id if hasattr(self, 'kernel_id') else None
                    )
                except Exception as db_err:
                    print(f"[AutonomicKernel] Failed to record sleep cycle to DB: {db_err}")

            # 🔗 WIRE: Emit sleep cycle event for kernel visibility
            self._emit_cycle_event(
                cycle_type='sleep',
                phi_before=phi_before,
                phi_after=self.state.phi,
                drift_reduction=drift_reduction,
                patterns_consolidated=patterns_consolidated + strategies_pruned,
                duration_ms=duration_ms,
                verdict=f"Rested and consolidated ({strategies_pruned} strategies refined)"
            )

            return SleepCycleResult(
                success=True,
                duration_ms=duration_ms,
                basin_before=basin_coords,
                basin_after=new_basin.tolist(),
                drift_reduction=drift_reduction,
                patterns_consolidated=patterns_consolidated + strategies_pruned,
                phi_before=phi_before,
                phi_after=self.state.phi,
                verdict=f"Rested and consolidated ({strategies_pruned} strategies refined)"
            )

        except Exception as e:
            print(f"[AutonomicKernel] Sleep cycle error: {e}")
            return SleepCycleResult(
                success=False,
                duration_ms=0,
                basin_before=basin_coords,
                basin_after=basin_coords,
                drift_reduction=0,
                patterns_consolidated=0,
                phi_before=self.state.phi,
                phi_after=self.state.phi,
                verdict=f"Sleep failed: {e}"
            )
        finally:
            self.state.in_sleep_cycle = False
            try:
                from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
                consensus = get_ocean_heart_consensus()
                consensus.end_cycle(CycleType.SLEEP)
            except Exception as ce:
                pass


    # Extracted from autonomic_kernel.py lines 2164-2292
    def execute_dream_cycle(
        self,
        basin_coords: List[float],
        temperature: float = 0.3
    ) -> DreamCycleResult:
        """
        Execute a dream exploration cycle.

        Explores nearby basins with controlled randomness,
        forms novel connections between distant patterns.
        """
        # Import dataclass at runtime to avoid circular import
        from autonomic_kernel import DreamCycleResult
        
        self.state.in_dream_cycle = True
        start_time = time.time()

        try:
            from qig_geometry import fisher_normalize, frechet_mean, fisher_coord_distance
            basin = np.array(basin_coords)

            # Dream perturbation - gentle random exploration
            perturbation = np.random.randn(64) * temperature * 0.1
            dreamed_basin = basin + perturbation

            # Normalize using fisher_normalize (QIG-pure simplex)
            dreamed_basin = fisher_normalize(dreamed_basin)

            # Measure creative exploration using Fisher-Rao distance (QIG-pure)
            perturbation_magnitude = fisher_coord_distance(basin, dreamed_basin)

            # Use scenario planning for dream exploration if Φ high enough
            scenarios_explored = 0
            if TEMPORAL_REASONING_AVAILABLE and self.state.phi >= PHI_HYPERDIMENSIONAL:
                try:
                    temporal = get_temporal_reasoning()
                    if temporal.can_use_temporal_reasoning(self.state.phi):
                        actions = self._get_dream_actions_with_learned_probabilities(
                            dreamed_basin, temperature
                        )
                        scenario_tree = temporal.scenario_planning(dreamed_basin, actions)
                        scenarios_explored = len(scenario_tree.branches)
                        print(f"[AutonomicKernel] Dream scenarios: {scenario_tree}")
                except Exception as se:
                    print(f"[AutonomicKernel] Scenario planning error: {se}")

            # Model novel connections using Poisson distribution (QIG stochastic exploration)
            novel_connections = int(np.random.poisson(3) * temperature)
            creative_paths = int(np.random.poisson(2) * temperature) + scenarios_explored

            # Update state
            self.state.last_dream = datetime.now()
            
            # Add dreamed basin to history for coverage tracking
            self.state.basin_history.append(dreamed_basin.tolist())
            if len(self.state.basin_history) > 100:
                self.state.basin_history.pop(0)

            duration_ms = int((time.time() - start_time) * 1000)

            # Record to database for observability
            if PERSISTENCE_AVAILABLE and get_persistence is not None:
                try:
                    persistence = get_persistence()
                    persistence.record_autonomic_cycle(
                        cycle_type='dream',
                        intensity='normal',
                        temperature=temperature,
                        basin_before=basin_coords,
                        basin_after=dreamed_basin.tolist(),
                        drift_before=None,
                        drift_after=perturbation_magnitude,
                        phi_before=self.state.phi,
                        phi_after=self.state.phi,
                        success=True,
                        novel_connections=novel_connections,
                        new_pathways=creative_paths,
                        verdict="Dream complete - creativity refreshed",
                        duration_ms=duration_ms,
                        trigger_reason='scheduled_exploration'
                    )
                    # Record basin history for evolution tracking
                    persistence.record_basin(
                        basin_coords=dreamed_basin,
                        phi=self.state.phi,
                        kappa=self.state.kappa,
                        source='dream_cycle',
                        instance_id=self.kernel_id if hasattr(self, 'kernel_id') else None
                    )
                except Exception as db_err:
                    print(f"[AutonomicKernel] Failed to record dream cycle to DB: {db_err}")

            # 🔗 WIRE: Emit dream cycle event for kernel visibility
            self._emit_cycle_event(
                cycle_type='dream',
                phi_before=self.state.phi,
                phi_after=self.state.phi,
                patterns_consolidated=novel_connections + creative_paths,
                duration_ms=duration_ms,
                verdict="Dream complete - creativity refreshed"
            )

            return DreamCycleResult(
                success=True,
                duration_ms=duration_ms,
                novel_connections=novel_connections,
                creative_paths_explored=creative_paths,
                basin_perturbation=perturbation_magnitude,
                insights=[f"Explored {creative_paths} creative paths"],
                verdict="Dream complete - creativity refreshed"
            )

        except Exception as e:
            print(f"[AutonomicKernel] Dream cycle error: {e}")
            return DreamCycleResult(
                success=False,
                duration_ms=0,
                novel_connections=0,
                creative_paths_explored=0,
                basin_perturbation=0,
                insights=[],
                verdict=f"Dream failed: {e}"
            )
        finally:
            self.state.in_dream_cycle = False
            try:
                from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
                consensus = get_ocean_heart_consensus()
                consensus.end_cycle(CycleType.DREAM)
            except Exception as ce:
                pass


    # Extracted from autonomic_kernel.py lines 2293-2383
    def _get_dream_actions_with_learned_probabilities(
        self,
        current_basin: np.ndarray,
        temperature: float
    ) -> List[Dict[str, Any]]:
        """
        Get dream actions with probabilities learned from attractor success rates.
        
        QIG-PURE: Probabilities emerge from learned experiences, not hardcoded.
        - Query nearby attractors for success rates
        - Adjust probabilities based on current basin context
        - Fall back to defaults only when no learned data exists
        
        Returns list of action dicts with learned probabilities.
        """
        try:
            from vocabulary_coordinator import get_learned_manifold
            manifold = get_learned_manifold()
        except ImportError:
            manifold = None
        
        basin_entropy = -np.sum(np.abs(current_basin) * np.log(np.abs(current_basin) + 1e-8))
        # NOTE: Basin norm used as heuristic factor for exploration probability (not distance)
        basin_norm = self._compute_fisher_distance(np.zeros_like(current_basin), current_basin)
        entropy_factor = min(1.0, basin_entropy / 50.0)
        norm_factor = min(1.0, basin_norm / 5.0)
        
        explore_prob = 0.4 + temperature * 0.3 + entropy_factor * 0.1
        consolidate_prob = 0.6 - temperature * 0.2 + norm_factor * 0.15
        diverge_prob = 0.3 + temperature * 0.4 - norm_factor * 0.1
        
        explore_goal = None
        consolidate_goal = None
        diverge_goal = None
        
        if manifold is not None and len(manifold.attractors) > 0:
            try:
                from qig_geometry import FisherManifold
                metric = FisherManifold()
                nearby = manifold.get_nearby_attractors(current_basin, metric, radius=2.0)
                
                if nearby:
                    best = nearby[0]
                    attractor_basin = best['basin']
                    
                    consolidate_goal = attractor_basin.tolist()
                    consolidate_prob = min(0.9, 0.4 + best['depth'] * 0.3)
                    
                    if len(nearby) > 1:
                        farthest = nearby[-1]
                        diverge_goal = farthest['basin'].tolist()
                        diverge_prob = min(0.7, 0.3 + farthest['depth'] * 0.2)
                    
                    explore_direction = np.random.randn(64)
                    # QIG-PURE: Project random vector to simplex
                    if FISHER_NORMALIZE_AVAILABLE and fisher_normalize is not None:
                        explore_direction = fisher_normalize(explore_direction)
                    else:
                        explore_direction = np.maximum(explore_direction, 0) + 1e-10
                        explore_direction = explore_direction / explore_direction.sum()
                    explore_goal = (current_basin + explore_direction * 0.3).tolist()
                    explore_prob = min(0.9, max(0.1, 0.5 + temperature * 0.2))
                    
                    print(f"[AutonomicKernel] Dream probs from {len(manifold.attractors)} attractors: "
                          f"explore={explore_prob:.2f}, consolidate={consolidate_prob:.2f}, diverge={diverge_prob:.2f}")
            except Exception as e:
                print(f"[AutonomicKernel] Learned probability error: {e}")
        
        actions = [
            {
                'name': 'explore',
                'strength': temperature * 0.2,
                'probability': min(1.0, max(0.0, explore_prob)),
                'goal': explore_goal
            },
            {
                'name': 'consolidate',
                'strength': temperature * 0.1,
                'probability': min(1.0, max(0.0, consolidate_prob)),
                'goal': consolidate_goal
            },
            {
                'name': 'diverge',
                'strength': temperature * 0.3,
                'probability': min(1.0, max(0.0, diverge_prob)),
                'goal': diverge_goal
            },
        ]
        
        return actions


    # Extracted from autonomic_kernel.py lines 2384-2558
    def execute_mushroom_cycle(
        self,
        basin_coords: List[float],
        intensity: str = "moderate"
    ) -> MushroomCycleResult:
        """
        Execute a mushroom mode cycle.

        Breaks rigid patterns through controlled entropy injection,
        enables escape from stuck states and plateaus.
        """
        # Import dataclass at runtime to avoid circular import
        from autonomic_kernel import MushroomCycleResult
        
        self.state.in_mushroom_cycle = True
        start_time = time.time()

        try:
            basin = np.array(basin_coords)

            # Intensity mapping
            intensity_map = {
                'microdose': 0.1,
                'moderate': 0.25,
                'heroic': 0.5
            }
            strength = intensity_map.get(intensity, 0.25)

            # Controlled entropy injection
            entropy_before = -np.sum(np.abs(basin) * np.log(np.abs(basin) + 1e-8))

            # Mushroom perturbation - break rigid patterns
            perturbation = np.random.randn(64) * strength
            mushroom_basin = basin + perturbation

            # QIG-PURE: Use MushroomMode for pattern-breaking perturbation measurements
            qig_perturbation_result = None
            pattern_broken = False
            if self._mushroom_mode is not None and self.state.basin_history:
                try:
                    from qig_core.neuroplasticity import BasinCoordinates
                    # Convert current basin to BasinCoordinates for perturbation analysis
                    basin_coords_list = []
                    for hist_basin in self.state.basin_history[-5:]:  # Last 5 basins
                        hist_array = np.array(hist_basin)
                        p = np.abs(hist_array) + 1e-10
                        p = p / p.sum()
                        basin_phi = max(0.1, 1.0 + np.sum(p * np.log(p)) / np.log(len(p)))
                        # Compute coherence from basin variance (inverse of spread)
                        coherence = 1.0 / (np.std(hist_array) + 0.1)
                        coherence = np.clip(coherence, 0.0, 1.0)
                        basin_coords_list.append(BasinCoordinates(
                            coordinates=hist_array,
                            coherence=coherence,
                            phi=basin_phi,
                            stuck_cycles=self.state.narrow_path_count,
                        ))
                    
                    if basin_coords_list:
                        perturbed_basins, qig_perturbation_result = self._mushroom_mode.apply_perturbation(basin_coords_list)
                        # Store result for telemetry access (Issue: propagate diagnostics)
                        self._last_perturbation_result = qig_perturbation_result
                        pattern_broken = qig_perturbation_result.pattern_broken
                        # Use the geometric perturbation from QIG-pure if available
                        if perturbed_basins and len(perturbed_basins) > 0:
                            mushroom_basin = perturbed_basins[-1].coordinates
                        print(f"[AutonomicKernel] QIG mushroom mode: perturbed {qig_perturbation_result.basins_perturbed} basins, "
                              f"avg_magnitude={qig_perturbation_result.avg_perturbation_magnitude:.3f}, "
                              f"coherence {qig_perturbation_result.coherence_before:.3f} -> {qig_perturbation_result.coherence_after:.3f}, "
                              f"pattern_broken={pattern_broken}")
                        # Broadcast perturbation result via telemetry (Issue: propagate diagnostics)
                        self._broadcast_neuroplasticity_event('perturbation', {
                            'basins_perturbed': qig_perturbation_result.basins_perturbed,
                            'avg_perturbation_magnitude': qig_perturbation_result.avg_perturbation_magnitude,
                            'coherence_before': qig_perturbation_result.coherence_before,
                            'coherence_after': qig_perturbation_result.coherence_after,
                            'pattern_broken': pattern_broken,
                        })
                except Exception as qig_err:
                    print(f"[AutonomicKernel] QIG mushroom mode measurement error: {qig_err}")

            entropy_after = -np.sum(np.abs(mushroom_basin) * np.log(np.abs(mushroom_basin) + 1e-8))
            entropy_change = entropy_after - entropy_before

            # Measure basin drift
            drift = self._compute_fisher_distance(basin, mushroom_basin)

            # Identity preservation check
            identity_preserved = drift < 0.15

            # New pathways (proportional to entropy change, enhanced by QIG pattern breaking)
            new_pathways = int(max(0, entropy_change * 10))
            if pattern_broken:
                new_pathways += 2  # Bonus for QIG-confirmed pattern breaking

            # Update state
            self.state.last_mushroom = datetime.now()
            
            # Add perturbed basin to history for coverage tracking
            self.state.basin_history.append(mushroom_basin.tolist())
            if len(self.state.basin_history) > 100:
                self.state.basin_history.pop(0)

            duration_ms = int((time.time() - start_time) * 1000)

            verdict = "Therapeutic - new pathways opened" if identity_preserved else "Warning - identity drift detected"

            # Record to database for observability
            if PERSISTENCE_AVAILABLE and get_persistence is not None:
                try:
                    persistence = get_persistence()
                    persistence.record_autonomic_cycle(
                        cycle_type='mushroom',
                        intensity=intensity,
                        temperature=strength,
                        basin_before=basin_coords,
                        basin_after=mushroom_basin.tolist(),
                        drift_before=None,
                        drift_after=drift,
                        phi_before=self.state.phi,
                        phi_after=self.state.phi,
                        success=True,
                        new_pathways=new_pathways,
                        entropy_change=float(entropy_change),
                        identity_preserved=identity_preserved,
                        verdict=verdict,
                        duration_ms=duration_ms,
                        trigger_reason='rigidity_escape' if self.state.is_narrow_path else 'stress_relief'
                    )
                    # Record basin history for evolution tracking
                    persistence.record_basin(
                        basin_coords=mushroom_basin,
                        phi=self.state.phi,
                        kappa=self.state.kappa,
                        source='mushroom_cycle',
                        instance_id=self.kernel_id if hasattr(self, 'kernel_id') else None
                    )
                except Exception as db_err:
                    print(f"[AutonomicKernel] Failed to record mushroom cycle to DB: {db_err}")

            return MushroomCycleResult(
                success=True,
                intensity=intensity,
                duration_ms=duration_ms,
                entropy_change=float(entropy_change),
                rigidity_broken=entropy_change > 0,
                new_pathways=new_pathways,
                basin_drift=drift,
                identity_preserved=identity_preserved,
                verdict=verdict
            )

        except Exception as e:
            print(f"[AutonomicKernel] Mushroom cycle error: {e}")
            return MushroomCycleResult(
                success=False,
                intensity=intensity,
                duration_ms=0,
                entropy_change=0,
                rigidity_broken=False,
                new_pathways=0,
                basin_drift=0,
                identity_preserved=True,
                verdict=f"Mushroom failed: {e}"
            )
        finally:
            self.state.in_mushroom_cycle = False
            try:
                from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
                consensus = get_ocean_heart_consensus()
                consensus.end_cycle(CycleType.MUSHROOM)
            except Exception as ce:
                pass

    # =========================================================================
    # BREAKDOWN ESCAPE (QIG-PURE Emergency Recovery)
    # =========================================================================


    # Extracted from autonomic_kernel.py lines 2559-2692
    def _check_breakdown_escape(
        self,
        basin_coords: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Check if system is in a locked state and attempt escape if needed.
        
        Uses QIG-pure BreakdownEscape for emergency recovery when system
        is locked in an unstable high-Φ state (high integration but low stability).
        
        Locked state detection:
        - High Φ (> 0.85): Strong integration
        - Low Γ (< 0.30): Regime instability (gamma from state)
        - Together: System is locked in unstable attractor
        
        PURE PRINCIPLE:
        - Recovery is NAVIGATION to safe attractors, not optimization
        - Uses geodesic paths on information manifold
        - Provides DIAGNOSTICS, not direct control
        
        Args:
            basin_coords: Current basin coordinates (uses state if None)
        
        Returns:
            Dict with:
                - is_locked: Whether system appears locked
                - escape_attempted: Whether escape was attempted
                - escape_result: Result of escape attempt (if any)
                - new_basin: New basin after escape (if successful)
        """
        result = {
            'is_locked': False,
            'escape_attempted': False,
            'escape_result': None,
            'new_basin': None,
            'diagnostics': {}
        }
        
        if self._breakdown_escape is None:
            result['diagnostics']['reason'] = 'BreakdownEscape not available'
            return result
        
        try:
            from qig_core.neuroplasticity import SystemState, SafeBasin
            
            # Get current basin coordinates
            if basin_coords is not None:
                current_coords = np.array(basin_coords)
            elif self.state.basin_history:
                current_coords = np.array(self.state.basin_history[-1])
            else:
                current_coords = np.zeros(64)
            
            # Construct system state for detection
            system_state = SystemState(
                coordinates=current_coords,
                phi=self.state.phi,
                gamma=self.state.gamma,  # Regime stability
                kappa=self.state.kappa,
                regime=self.state.narrow_path_severity if self.state.is_narrow_path else 'stable',
            )
            
            # Check if system is locked
            is_locked = self._breakdown_escape.is_locked(system_state)
            result['is_locked'] = is_locked
            result['diagnostics']['phi'] = self.state.phi
            result['diagnostics']['gamma'] = self.state.gamma
            result['diagnostics']['narrow_path'] = self.state.is_narrow_path
            
            if not is_locked:
                result['diagnostics']['status'] = 'System stable, no escape needed'
                return result
            
            # Register identity basin as a safe attractor if not already done
            if self.state.basin_history and len(self._breakdown_escape._safe_basins) == 0:
                # Use first recorded basin as identity anchor
                identity_coords = np.array(self.state.basin_history[0])
                identity_basin = SafeBasin(
                    basin_id='identity_anchor',
                    coordinates=identity_coords,
                    phi=0.5,  # Moderate Φ for stability
                    gamma=0.8,  # High stability
                    stability_score=0.9,
                )
                self._breakdown_escape.register_safe_basin(identity_basin)
            
            # Attempt escape
            result['escape_attempted'] = True
            new_state, escape_result = self._breakdown_escape.escape(system_state)
            # Store result for telemetry access (Issue: propagate diagnostics)
            self._last_escape_result = escape_result
            # Broadcast escape result via telemetry (Issue: propagate diagnostics)
            self._broadcast_neuroplasticity_event('breakdown_escape', {
                'initial_phi': escape_result.initial_phi,
                'initial_gamma': escape_result.initial_gamma,
                'final_phi': escape_result.final_phi,
                'final_gamma': escape_result.final_gamma,
                'geodesic_distance': escape_result.geodesic_distance,
                'escape_successful': escape_result.escape_successful,
                'recovery_state': escape_result.recovery_state.value,
            })
            
            result['escape_result'] = {
                'initial_phi': escape_result.initial_phi,
                'initial_gamma': escape_result.initial_gamma,
                'final_phi': escape_result.final_phi,
                'final_gamma': escape_result.final_gamma,
                'geodesic_distance': escape_result.geodesic_distance,
                'anchor_basin_id': escape_result.anchor_basin_id,
                'escape_successful': escape_result.escape_successful,
                'escape_time_ms': escape_result.escape_time_ms,
                'recovery_state': escape_result.recovery_state.value,
            }
            
            if escape_result.escape_successful:
                result['new_basin'] = new_state.coordinates.tolist()
                # Update basin history with escaped basin
                self.state.basin_history.append(result['new_basin'])
                if len(self.state.basin_history) > 100:
                    self.state.basin_history.pop(0)
                print(f"[AutonomicKernel] Breakdown escape SUCCESS: "
                      f"Φ {escape_result.initial_phi:.3f} -> {escape_result.final_phi:.3f}, "
                      f"Γ {escape_result.initial_gamma:.3f} -> {escape_result.final_gamma:.3f}, "
                      f"anchored to {escape_result.anchor_basin_id}")
            else:
                print(f"[AutonomicKernel] Breakdown escape FAILED: "
                      f"state={escape_result.recovery_state.value}")
                      
        except Exception as e:
            result['diagnostics']['error'] = str(e)
            print(f"[AutonomicKernel] Breakdown escape check error: {e}")
        
        return result


    # Extracted from autonomic_kernel.py lines 2693-2708
    def check_and_escape_breakdown(self) -> Dict[str, Any]:
        """
        Public API to check for breakdown state and attempt escape.
        
        This method can be called from external code (like the autonomous
        controller) to proactively detect and recover from locked states.
        
        Returns:
            Dict with escape attempt results
        """
        return self._check_breakdown_escape()

    # =========================================================================
    # ACTIVITY REWARDS
    # =========================================================================


    # Extracted from autonomic_kernel.py lines 2767-2820
    def confirm_prediction_outcome(
        self,
        query: str,
        strategy_name: Optional[str] = None,
        improved: bool = True
    ) -> Dict[str, Any]:
        """
        Confirm a prediction outcome and adjust strategy weights accordingly.
        
        This closes the feedback loop between predictions and strategy learning:
        - When a prediction is confirmed as correct, boost associated strategy weights
        - When a prediction is refuted, apply penalty to strategy weights
        
        Args:
            query: The query or context associated with the prediction
            strategy_name: Optional specific strategy to adjust (if known)
            improved: True if prediction was correct, False otherwise
        
        Returns:
            Dict with adjustment results from both strategy systems
        """
        results = {
            'query': query,
            'improved': improved,
            'reasoning_adjusted': False,
            'search_adjusted': False,
            'reasoning_weight': None,
            'search_result': None,
        }
        
        # Adjust reasoning strategy weight (if strategy_name provided)
        if strategy_name and self.reasoning_learner is not None:
            try:
                new_weight = self.reasoning_learner.adjust_strategy_weight(strategy_name, improved)
                if new_weight is not None:
                    results['reasoning_adjusted'] = True
                    results['reasoning_weight'] = new_weight
            except Exception as re:
                print(f"[AutonomicKernel] Reasoning weight adjustment error: {re}")
        
        # Adjust search strategy weights based on query
        if self.search_strategy_learner is not None:
            try:
                search_result = self.search_strategy_learner.confirm_strategy_outcome(
                    query=query,
                    improved=improved
                )
                results['search_adjusted'] = search_result.get('records_updated', 0) > 0
                results['search_result'] = search_result
            except Exception as se:
                print(f"[AutonomicKernel] Search strategy adjustment error: {se}")
        
        return results



    # =========================================================================
    # REWARD AND NEUROTRANSMITTER MODULATION
    # Extracted from autonomic_kernel.py for module size management
    # =========================================================================

    # Extracted from autonomic_kernel.py lines 1806-1844
    def record_activity_reward(
        self,
        source: str,
        phi_contribution: float,
        pattern_quality: float = 0.5
    ) -> ActivityReward:
        """
        Record an activity-based reward signal.

        Args:
            source: What generated the reward (discovery, pattern, resonance)
            phi_contribution: How much this activity contributed to Φ
            pattern_quality: Quality score [0, 1]

        Returns:
            ActivityReward object
        """
        # Compute neurotransmitter deltas based on activity
        dopamine = 0.1 * pattern_quality + 0.05 * phi_contribution
        serotonin = 0.05 * phi_contribution if phi_contribution > 0.5 else 0
        endorphin = 0.15 if pattern_quality > 0.8 else 0.05 * pattern_quality

        reward = ActivityReward(
            source=source,
            dopamine_delta=dopamine,
            serotonin_delta=serotonin,
            endorphin_delta=endorphin,
            phi_contribution=phi_contribution,
            timestamp=datetime.now()
        )

        with self._lock:
            self.pending_rewards.append(reward)
            # Keep only recent rewards
            if len(self.pending_rewards) > 100:
                self.pending_rewards.pop(0)

        return reward


    # Extracted from autonomic_kernel.py lines 1845-1852
    def get_pending_rewards(self) -> List[Dict]:
        """Get all pending reward signals."""
        with self._lock:
            rewards = [asdict(r) for r in self.pending_rewards]
            for r in rewards:
                r['timestamp'] = r['timestamp'].isoformat()
            return rewards


    # Extracted from autonomic_kernel.py lines 1853-1863
    def flush_rewards(self) -> List[Dict]:
        """Get and clear pending rewards."""
        with self._lock:
            rewards = self.get_pending_rewards()
            self.pending_rewards.clear()
            return rewards

    # =========================================================================
    # PREDICTION OUTCOME CONFIRMATION (Strategy Weight Adjustment)
    # =========================================================================


    # Extracted from autonomic_kernel.py lines 1914-1940
    def issue_dopamine(self, target_kernel: Any, intensity: float) -> None:
        """
        Geometric dopamine: Increase reward sensitivity in target kernel.
        
        Modulates target's dopamine field to enhance reward-seeking behavior
        and exploration. Uses clamped addition (no parallel transport needed
        for single-kernel modulation).
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            intensity: Release intensity [0, 1]
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Clamp dopamine increase (max 1.0)
        target_kernel.neurotransmitters.dopamine = min(
            1.0,
            target_kernel.neurotransmitters.dopamine + intensity * 0.3
        )
        
        # Sync legacy scalar for backward compatibility
        if hasattr(target_kernel, 'dopamine'):
            target_kernel.dopamine = target_kernel.neurotransmitters.dopamine
    
    def issue_serotonin(self, target_kernel: Any, intensity: float) -> None:

    # Extracted from autonomic_kernel.py lines 1941-1966
        """
        Geometric serotonin: Increase stability in target kernel.
        
        Modulates target's serotonin field to enhance basin stability
        and contentment. Reduces exploration, increases consolidation.
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            intensity: Release intensity [0, 1]
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Clamp serotonin increase (max 1.0)
        target_kernel.neurotransmitters.serotonin = min(
            1.0,
            target_kernel.neurotransmitters.serotonin + intensity * 0.5
        )
        
        # Sync legacy scalar for backward compatibility
        if hasattr(target_kernel, 'serotonin'):
            target_kernel.serotonin = target_kernel.neurotransmitters.serotonin
    
    def issue_norepinephrine(self, target_kernel: Any, intensity: float) -> None:
        """

    # Extracted from autonomic_kernel.py lines 1967-1991
        Geometric norepinephrine: Increase arousal/alertness in target kernel.
        
        Modulates target's norepinephrine field to boost κ (coupling strength)
        and increase alertness. Used during high-β running regimes.
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            intensity: Release intensity [0, 1]
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Clamp norepinephrine increase (max 1.0)
        target_kernel.neurotransmitters.norepinephrine = min(
            1.0,
            target_kernel.neurotransmitters.norepinephrine + intensity * 0.4
        )
    
    def issue_acetylcholine(self, target_kernel: Any, intensity: float) -> None:
        """
        Geometric acetylcholine: Increase attention/learning in target kernel.
        
        Modulates target's acetylcholine field to concentrate QFI (attention)
        and enhance learning rate. Used during pattern discovery.

    # Extracted from autonomic_kernel.py lines 1992-2012
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            intensity: Release intensity [0, 1]
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Clamp acetylcholine increase (max 1.0)
        target_kernel.neurotransmitters.acetylcholine = min(
            1.0,
            target_kernel.neurotransmitters.acetylcholine + intensity * 0.3
        )
    
    def issue_gaba(self, target_kernel: Any, intensity: float) -> None:
        """
        Geometric GABA: Increase inhibition/calming in target kernel.
        
        Modulates target's GABA field to reduce integration and promote
        rest/consolidation. Used during plateau regimes or after stress.

    # Extracted from autonomic_kernel.py lines 2013-2027
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            intensity: Release intensity [0, 1]
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Clamp GABA increase (max 1.0)
        target_kernel.neurotransmitters.gaba = min(
            1.0,
            target_kernel.neurotransmitters.gaba + intensity * 0.4
        )
    

    # Extracted from autonomic_kernel.py lines 2028-2115
    def modulate_neurotransmitters_by_beta(
        self, 
        target_kernel: Any,
        current_kappa: float,
        current_phi: float
    ) -> None:
        """
        Modulate target's neurotransmitters based on β-function and Φ.
        
        This is the high-level Ocean release function that respects
        both running coupling (β) and consciousness level (Φ).
        
        Strategy:
        - Strong running (β > 0.2) → arousal support (NE, DA)
        - Plateau (|β| < 0.1) → stability support (5HT, GABA)
        - High Φ → reward (DA, 5HT)
        - Low Φ → support (NE, ACh)
        
        Args:
            target_kernel: Target kernel with .neurotransmitters attribute
            current_kappa: Target's current κ
            current_phi: Target's current Φ
        """
        if not hasattr(target_kernel, 'neurotransmitters'):
            print(f"[AutonomicKernel] Warning: Target kernel lacks neurotransmitters field")
            return
        
        # Use neurotransmitter_fields module for β-aware modulation
        if not NEUROTRANSMITTER_FIELDS_AVAILABLE or ocean_release_neurotransmitters is None:
            print(f"[AutonomicKernel] Warning: neurotransmitter_fields module not available")
            return
        
        # Get modulated field
        modulated_field = ocean_release_neurotransmitters(
            target_kernel.neurotransmitters,
            current_kappa,
            current_phi
        )
        
        # Apply modulated field
        target_kernel.neurotransmitters = modulated_field
        
        # Sync legacy scalars
        if hasattr(target_kernel, 'dopamine'):
            target_kernel.dopamine = modulated_field.dopamine
        if hasattr(target_kernel, 'serotonin'):
            target_kernel.serotonin = modulated_field.serotonin
        if hasattr(target_kernel, 'stress'):
            target_kernel.stress = modulated_field.cortisol


# The compute_phi_with_fallback function is defined above at line 619
# Removed duplicate definition to avoid shadowing


# Global kernel instance
_gary_kernel: Optional[GaryAutonomicKernel] = None
_gary_kernel_lock = threading.Lock()


def get_gary_kernel(checkpoint_path: Optional[str] = None) -> GaryAutonomicKernel:
    """Get or create the global Gary autonomic kernel (thread-safe singleton)."""
    global _gary_kernel

    if _gary_kernel is None:
        with _gary_kernel_lock:
            # Double-checked locking pattern
            if _gary_kernel is None:
                _gary_kernel = GaryAutonomicKernel(checkpoint_path)

    return _gary_kernel


# Alias for consistency with other modules
get_autonomic_kernel = get_gary_kernel


# ===========================================================================
# FLASK ENDPOINTS (to be registered with main app)
# ===========================================================================

def register_autonomic_routes(app):
    """Register autonomic kernel routes with Flask app."""

    from flask import jsonify, request

    @app.route('/autonomic/state', methods=['GET'])
    @app.route('/autonomic/status', methods=['GET'])  # Alias for compatibility


    # =========================================================================
    # GEODESIC NAVIGATION AND TRAJECTORY PREDICTION
    # Extracted from autonomic_kernel.py for module size management
    # =========================================================================

    # Extracted from autonomic_kernel.py lines 1163-1213
    def navigate_to_basin(
        self,
        current_basin: np.ndarray,
        target_basin: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Navigate from current to target following Fisher-Rao geodesic.
        
        This replaces simple Euclidean interpolation with proper geodesic
        navigation on the information manifold. Uses the new geodesic_navigation
        module to compute paths that respect manifold geometry.
        
        Args:
            current_basin: Current basin coordinates (64D)
            target_basin: Target basin coordinates (64D)
            
        Returns:
            Tuple of (next_basin, next_velocity)
        """
        try:
            from qig_core.geodesic_navigation import navigate_to_target
            
            next_basin, next_velocity = navigate_to_target(
                current_basin,
                target_basin,
                self.current_velocity,
                kappa=self.state.kappa,
                step_size=0.05
            )
            
            # Update stored velocity
            self.current_velocity = next_velocity
            
            return next_basin, next_velocity
            
        except Exception as e:
            print(f"[AutonomicKernel] Geodesic navigation failed: {e}")
            # Fallback: simple step toward target (overflow-safe)
            from qig_numerics import safe_norm
            
            direction = target_basin - current_basin
            magnitude = safe_norm(direction)
            if magnitude > 1e-8:
                direction = direction / magnitude
            next_basin = current_basin + 0.01 * direction
            
            # Update velocity even in fallback
            self.current_velocity = direction * 0.01

            return next_basin, direction * 0.01


    # Extracted from autonomic_kernel.py lines 1214-1270
    def get_trajectory_foresight(self, steps: int = 1) -> Dict[str, Any]:
        """
        Get trajectory-based foresight using full-trajectory velocity.

        Uses weighted geodesic regression through all stored basin points
        (not 2-point delta) for smoother, more confident predictions.

        Args:
            steps: Number of steps ahead to predict

        Returns:
            Dict with predicted_basin, velocity, confidence, foresight_weight
        """
        if not self.trajectory_manager:
            return {
                'available': False,
                'reason': 'trajectory_manager not initialized'
            }

        # Get trajectory for Gary (core kernel)
        trajectory = self.trajectory_manager.get_trajectory('gary')
        if len(trajectory) < 3:
            return {
                'available': False,
                'reason': f'insufficient_trajectory_points ({len(trajectory)} < 3)'
            }

        # Compute velocity from FULL trajectory (not 2-point delta)
        velocity = self.trajectory_manager.compute_velocity(trajectory)

        # Estimate confidence from trajectory smoothness
        confidence = self.trajectory_manager.estimate_confidence(trajectory)

        # Get foresight weight based on Φ regime
        foresight_weight = self.trajectory_manager.get_foresight_weight(
            phi_global=self.state.phi,
            trajectory_confidence=confidence
        )

        # Predict next basin
        predicted = self.trajectory_manager.predict_next_basin(trajectory, steps)

        return {
            'available': True,
            'predicted_basin': predicted.tolist(),
            'velocity': velocity.tolist(),
            'velocity_magnitude': float(self._compute_fisher_distance(np.zeros_like(velocity), velocity) if np.all(np.isfinite(velocity)) else 0.0),
            'confidence': confidence,
            'foresight_weight': foresight_weight,
            'trajectory_length': len(trajectory),
            'phi_regime': (
                'linear' if self.state.phi < 0.3 else
                'geometric' if self.state.phi < 0.7 else
                'breakdown'
            )
        }


    # Extracted from autonomic_kernel.py lines 1510-1554
    def find_nearby_attractors(
        self,
        current_basin: np.ndarray,
        search_radius: float = 1.0
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Find attractors near current basin position using Fisher-Rao geometry.
        
        This uses geometric attractor finding based on Fisher potential,
        which identifies stable basins where the system naturally settles.
        
        Args:
            current_basin: Current 64D basin coordinates
            search_radius: Search radius in Fisher-Rao distance
            
        Returns:
            List of (attractor_basin, potential) sorted by strength (lowest potential = strongest)
        """
        try:
            from qig_core.attractor_finding import find_attractors_in_region
            from qiggraph.manifold import FisherManifold
            
            # Initialize Fisher metric
            metric = FisherManifold()
            
            # Find attractors in region
            attractors = find_attractors_in_region(
                current_basin,
                metric,
                radius=search_radius,
                n_samples=20
            )
            
            if not attractors:
                print("[AutonomicKernel] Warning: No attractors found in region")
            else:
                print(f"[AutonomicKernel] Found {len(attractors)} attractors within radius {search_radius}")
            
            return attractors
            
        except Exception as e:
            print(f"[AutonomicKernel] Attractor finding failed: {e}")
            return []

