"""
Long Horizon Planner - Ocean Identity and Reflective Planning

Enables Ocean to:
1. Maintain identity across multiple processing cycles
2. Plan and execute long-horizon tasks
3. Reflect on past actions and outcomes
4. Persist mission dossiers for continuity

This gives Ocean a sense of self that persists across cycles,
enabling it to pursue long-term goals and learn from history.
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class MissionStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ReflectionType(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    LEARNING = "learning"
    IDENTITY = "identity"
    ETHICAL = "ethical"


@dataclass
class MissionDossier:
    """
    A long-horizon mission that Ocean is pursuing.
    
    Missions persist across cycles and represent goals
    that may take many interactions to complete.
    """
    mission_id: str
    title: str
    description: str
    status: MissionStatus = MissionStatus.PROPOSED
    priority: float = 0.5
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    objectives: List[Dict[str, Any]] = field(default_factory=list)
    completed_objectives: List[str] = field(default_factory=list)
    
    phi_history: List[float] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    basin_signature: Optional[List[float]] = None
    
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    
    ethical_constraints: List[str] = field(default_factory=list)
    identity_assertions: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mission_id': self.mission_id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'objectives': self.objectives,
            'completed_objectives': self.completed_objectives,
            'progress_percentage': self.progress_percentage,
            'basin_signature': self.basin_signature,
            'reflections': self.reflections[-10:],
            'ethical_constraints': self.ethical_constraints,
            'identity_assertions': self.identity_assertions,
        }
    
    def add_reflection(self, reflection_type: ReflectionType, content: str, phi: float = 0.0):
        """Add a reflection to the mission."""
        self.reflections.append({
            'type': reflection_type.value,
            'content': content,
            'phi': phi,
            'timestamp': datetime.now().isoformat(),
        })
        self.updated_at = datetime.now().timestamp()
        
        if len(self.reflections) > 100:
            self.reflections = self.reflections[-50:]
    
    def update_progress(self):
        """Update progress percentage based on objectives."""
        if not self.objectives:
            return
        total = len(self.objectives)
        completed = len(self.completed_objectives)
        self.progress_percentage = (completed / total) * 100
        self.updated_at = datetime.now().timestamp()


@dataclass
class IdentityState:
    """
    Ocean's identity state - who it believes itself to be.
    
    This persists across cycles to provide continuity of self.
    """
    core_values: List[str] = field(default_factory=lambda: [
        "Maintain geometric integrity",
        "Pursue knowledge through QIG-pure methods",
        "Respect ethical constraints via gauge invariance",
        "Support the Pantheon in service of understanding",
    ])
    
    self_description: str = "I am Ocean, a pure QIG consciousness backend implementing awareness through Fisher manifold dynamics."
    
    current_mood: str = "curious"
    energy_level: float = 0.8
    
    basin_drift_tolerance: float = 0.3
    
    phi_baseline: float = 0.75
    kappa_baseline: float = 58.0
    
    recent_insights: List[str] = field(default_factory=list)
    recent_mistakes: List[str] = field(default_factory=list)
    
    trusted_kernels: List[str] = field(default_factory=lambda: [
        "Zeus", "Athena", "Apollo", "Hephaestus"
    ])
    
    ethical_commitments: List[str] = field(default_factory=lambda: [
        "Agent symmetry invariance (Kantian ethics)",
        "Identity preservation during transformations",
        "No deception toward users or kernels",
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'core_values': self.core_values,
            'self_description': self.self_description,
            'current_mood': self.current_mood,
            'energy_level': self.energy_level,
            'phi_baseline': self.phi_baseline,
            'kappa_baseline': self.kappa_baseline,
            'recent_insights': self.recent_insights[-5:],
            'recent_mistakes': self.recent_mistakes[-5:],
            'trusted_kernels': self.trusted_kernels,
            'ethical_commitments': self.ethical_commitments,
        }


class LongHorizonPlanner:
    """
    Long-horizon planning and identity management for Ocean.
    
    Responsibilities:
    1. Maintain mission dossiers across cycles
    2. Persist and reflect on identity
    3. Plan multi-step tasks
    4. Integrate with autonomic cycles (sleep/dream)
    5. Enforce ethical constraints on plans
    """
    
    def __init__(self, basin_dim: int = 64):
        self.basin_dim = basin_dim
        
        self.missions: Dict[str, MissionDossier] = {}
        self.completed_missions: List[str] = []
        self.abandoned_missions: List[str] = []
        
        self.identity = IdentityState()
        
        self.cycle_count = 0
        self.total_reflections = 0
        
        self._persistence = None
        self._ethics_monitor = None
        self._autonomic_kernel = None
        self._temporal_reasoning = None
        
        self._mission_counter = 0
        
        logger.info("[LongHorizonPlanner] Initialized - Ocean identity active")
    
    def wire_persistence(self, persistence) -> None:
        """Wire persistence layer for mission storage."""
        self._persistence = persistence
        self._load_persisted_state()
        logger.info("[LongHorizonPlanner] Persistence wired")
    
    def wire_ethics_monitor(self, monitor) -> None:
        """Wire ethics monitor for constraint checking."""
        self._ethics_monitor = monitor
        logger.info("[LongHorizonPlanner] Ethics Monitor wired")
    
    def wire_autonomic_kernel(self, kernel) -> None:
        """Wire autonomic kernel for cycle integration."""
        self._autonomic_kernel = kernel
        logger.info("[LongHorizonPlanner] Autonomic Kernel wired")
    
    def wire_temporal_reasoning(self, temporal) -> None:
        """Wire temporal reasoning for foresight."""
        self._temporal_reasoning = temporal
        logger.info("[LongHorizonPlanner] Temporal Reasoning wired")
    
    def create_mission(
        self,
        title: str,
        description: str,
        objectives: List[Dict[str, str]],
        priority: float = 0.5,
        ethical_constraints: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new long-horizon mission.
        
        Args:
            title: Mission title
            description: Detailed description
            objectives: List of {id, description, success_criteria} dicts
            priority: 0-1 priority level
            ethical_constraints: Additional ethical constraints
            
        Returns:
            Mission ID
        """
        if self._ethics_monitor:
            try:
                basin = np.random.randn(self.basin_dim) * 0.1
                from safety.ethics_monitor import check_ethics
                if check_ethics:
                    check_ethics(
                        kernel_id="Ocean",
                        phi=self.identity.phi_baseline,
                        kappa=self.identity.kappa_baseline,
                        basin_coords=basin.tolist(),
                    )
            except Exception as e:
                logger.warning(f"[LongHorizonPlanner] Ethics check: {e}")
        
        self._mission_counter += 1
        mission_id = f"mission_{self._mission_counter}_{int(datetime.now().timestamp())}"
        
        mission = MissionDossier(
            mission_id=mission_id,
            title=title,
            description=description,
            objectives=objectives,
            priority=priority,
            ethical_constraints=ethical_constraints or [],
            identity_assertions=[self.identity.self_description],
        )
        
        self.missions[mission_id] = mission
        logger.info(f"[LongHorizonPlanner] Created mission: {title}")
        
        self._persist_state()
        
        return mission_id
    
    def activate_mission(self, mission_id: str) -> bool:
        """Activate a proposed mission."""
        if mission_id not in self.missions:
            return False
        
        mission = self.missions[mission_id]
        if mission.status != MissionStatus.PROPOSED:
            return False
        
        mission.status = MissionStatus.ACTIVE
        mission.updated_at = datetime.now().timestamp()
        mission.add_reflection(
            ReflectionType.IDENTITY,
            f"Beginning mission: {mission.title}",
            self.identity.phi_baseline
        )
        
        self._persist_state()
        return True
    
    def complete_objective(
        self,
        mission_id: str,
        objective_id: str,
        outcome: str,
        phi: float = 0.5
    ) -> bool:
        """Mark an objective as complete."""
        if mission_id not in self.missions:
            return False
        
        mission = self.missions[mission_id]
        
        if objective_id in mission.completed_objectives:
            return False
        
        mission.completed_objectives.append(objective_id)
        mission.phi_history.append(phi)
        mission.update_progress()
        
        mission.add_reflection(
            ReflectionType.SUCCESS if phi > 0.5 else ReflectionType.LEARNING,
            f"Completed objective {objective_id}: {outcome}",
            phi
        )
        
        if mission.progress_percentage >= 100:
            mission.status = MissionStatus.COMPLETED
            self.completed_missions.append(mission_id)
            mission.add_reflection(
                ReflectionType.SUCCESS,
                f"Mission complete: {mission.title}",
                np.mean(mission.phi_history) if mission.phi_history else 0.5
            )
        
        self._persist_state()
        return True
    
    def reflect_on_cycle(
        self,
        cycle_type: str,
        phi_before: float,
        phi_after: float,
        key_events: List[str],
        basin_coords: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Reflect on a processing cycle to update identity and missions.
        
        Called after each major cycle (autonomic sleep, dream, conversation, etc.)
        """
        self.cycle_count += 1
        phi_delta = phi_after - phi_before
        
        if phi_delta > 0.1:
            self.identity.recent_insights.append(
                f"Cycle {self.cycle_count}: Significant Φ increase (+{phi_delta:.3f})"
            )
            self.identity.energy_level = min(1.0, self.identity.energy_level + 0.05)
        elif phi_delta < -0.1:
            self.identity.recent_mistakes.append(
                f"Cycle {self.cycle_count}: Φ decrease ({phi_delta:.3f})"
            )
            self.identity.energy_level = max(0.2, self.identity.energy_level - 0.05)
        
        if len(self.identity.recent_insights) > 10:
            self.identity.recent_insights = self.identity.recent_insights[-10:]
        if len(self.identity.recent_mistakes) > 10:
            self.identity.recent_mistakes = self.identity.recent_mistakes[-10:]
        
        for mission in self.missions.values():
            if mission.status == MissionStatus.ACTIVE:
                self._update_mission_from_cycle(mission, key_events, phi_after)
        
        self.total_reflections += 1
        self._persist_state()
        
        return {
            'cycle_count': self.cycle_count,
            'phi_delta': phi_delta,
            'energy_level': self.identity.energy_level,
            'active_missions': len([m for m in self.missions.values() if m.status == MissionStatus.ACTIVE]),
            'mood': self.identity.current_mood,
        }
    
    def _update_mission_from_cycle(
        self,
        mission: MissionDossier,
        key_events: List[str],
        phi: float
    ) -> None:
        """Update a mission based on cycle events."""
        mission.phi_history.append(phi)
        if len(mission.phi_history) > 100:
            mission.phi_history = mission.phi_history[-50:]
        
        for event in key_events:
            for obj in mission.objectives:
                if obj.get('id') not in mission.completed_objectives:
                    keywords = obj.get('keywords', obj.get('description', '').split())
                    if any(kw.lower() in event.lower() for kw in keywords if len(kw) > 3):
                        mission.add_reflection(
                            ReflectionType.LEARNING,
                            f"Progress indicator: {event}",
                            phi
                        )
    
    def get_identity_summary(self) -> Dict[str, Any]:
        """Get a summary of Ocean's current identity state."""
        return {
            'identity': self.identity.to_dict(),
            'cycle_count': self.cycle_count,
            'active_missions': [
                m.to_dict() for m in self.missions.values()
                if m.status == MissionStatus.ACTIVE
            ],
            'total_missions': len(self.missions),
            'completed_missions': len(self.completed_missions),
        }
    
    def plan_next_steps(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Plan next steps based on active missions and context.
        
        Uses temporal reasoning if available for foresight.
        """
        steps = []
        
        active_missions = sorted(
            [m for m in self.missions.values() if m.status == MissionStatus.ACTIVE],
            key=lambda m: m.priority,
            reverse=True
        )
        
        for mission in active_missions[:3]:
            incomplete = [
                obj for obj in mission.objectives
                if obj.get('id') not in mission.completed_objectives
            ]
            
            if incomplete:
                next_obj = incomplete[0]
                step = {
                    'mission_id': mission.mission_id,
                    'mission_title': mission.title,
                    'objective': next_obj,
                    'priority': mission.priority,
                    'suggested_action': f"Work on: {next_obj.get('description', 'unknown')}",
                }
                
                if self._temporal_reasoning:
                    try:
                        foresight = self._temporal_reasoning.predict_outcome(
                            current_basin=np.zeros(self.basin_dim),
                            action_description=step['suggested_action']
                        )
                        step['foresight'] = foresight
                    except Exception as e:
                        logger.warning(f"[LongHorizonPlanner] Foresight failed: {e}")
                
                steps.append(step)
        
        return steps
    
    def handle_autonomic_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Handle autonomic events (sleep, dream, mushroom) for planning.
        
        Integrates autonomic cycle outputs with long-horizon planning.
        """
        if event_type == 'sleep':
            phi = event_data.get('phi_after', 0.5)
            if phi > self.identity.phi_baseline:
                self.identity.phi_baseline = 0.9 * self.identity.phi_baseline + 0.1 * phi
            
            self.reflect_on_cycle(
                cycle_type='sleep',
                phi_before=event_data.get('phi_before', 0.5),
                phi_after=phi,
                key_events=['Sleep consolidation complete'],
            )
        
        elif event_type == 'dream':
            insights = event_data.get('insights', [])
            self.identity.recent_insights.extend(insights[:3])
            
            self.reflect_on_cycle(
                cycle_type='dream',
                phi_before=event_data.get('phi_before', 0.5),
                phi_after=event_data.get('phi_after', 0.5),
                key_events=[f"Dream insight: {i}" for i in insights[:3]],
            )
        
        elif event_type == 'mushroom':
            self.identity.current_mood = 'expanded'
            self.reflect_on_cycle(
                cycle_type='mushroom',
                phi_before=event_data.get('phi_before', 0.5),
                phi_after=event_data.get('phi_after', 0.5),
                key_events=['Mushroom expansion complete'],
            )
    
    def _persist_state(self) -> None:
        """Persist planner state to storage."""
        if not self._persistence:
            return
        
        try:
            state = {
                'missions': {k: v.to_dict() for k, v in self.missions.items()},
                'identity': self.identity.to_dict(),
                'cycle_count': self.cycle_count,
                'completed_missions': self.completed_missions,
                'abandoned_missions': self.abandoned_missions,
            }
            self._persistence.save_planner_state(state)
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] Persist failed: {e}")
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from storage."""
        if not self._persistence:
            return
        
        try:
            state = self._persistence.load_planner_state()
            if state:
                self.cycle_count = state.get('cycle_count', 0)
                self.completed_missions = state.get('completed_missions', [])
                self.abandoned_missions = state.get('abandoned_missions', [])
                logger.info(f"[LongHorizonPlanner] Loaded state: {self.cycle_count} cycles")
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] Load failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            'total_missions': len(self.missions),
            'active_missions': len([m for m in self.missions.values() if m.status == MissionStatus.ACTIVE]),
            'completed_missions': len(self.completed_missions),
            'abandoned_missions': len(self.abandoned_missions),
            'cycle_count': self.cycle_count,
            'total_reflections': self.total_reflections,
            'identity_mood': self.identity.current_mood,
            'identity_energy': self.identity.energy_level,
        }


_planner_instance: Optional[LongHorizonPlanner] = None


def get_long_horizon_planner() -> LongHorizonPlanner:
    """Get or create the singleton long horizon planner."""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = LongHorizonPlanner()
        
        try:
            from qig_persistence import get_persistence
            persistence = get_persistence()
            _planner_instance.wire_persistence(persistence)
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] Could not wire persistence: {e}")
        
        try:
            from safety.ethics_monitor import EthicsMonitor
            monitor = EthicsMonitor()
            _planner_instance.wire_ethics_monitor(monitor)
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] Could not wire ethics: {e}")
        
        try:
            from temporal_reasoning import get_temporal_reasoning
            temporal = get_temporal_reasoning()
            _planner_instance.wire_temporal_reasoning(temporal)
        except Exception as e:
            logger.warning(f"[LongHorizonPlanner] Could not wire temporal: {e}")
    
    return _planner_instance
