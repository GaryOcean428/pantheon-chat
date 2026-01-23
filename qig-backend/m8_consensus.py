#!/usr/bin/env python3
"""
M8 Consensus Layer - Voting and decision-making for kernel spawning

Implements:
- PantheonConsensus: Voting and consensus logic
- SpawnProposal: Spawn proposal data structure
- ConsensusType: Voting threshold types
- RoleRefinement: Parent domain refinement
- SpawnAwareness: Kernel self-awareness tracking
"""

import numpy as np
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum
import uuid

from geometric_kernels import (
    _normalize_to_manifold,
    _fisher_distance,
    _hash_to_bytes,
    BASIN_DIM,
)

try:
    from qigkernels.physics_constants import KAPPA_STAR
except ImportError:
    KAPPA_STAR = 64.21

from pantheon_kernel_orchestrator import (
    KernelProfile,
    KernelMode,
)

class SpawnReason(Enum):
    """Reasons for spawning a new kernel."""
    DOMAIN_GAP = "domain_gap"           # No kernel covers this domain well
    OVERLOAD = "overload"               # Existing kernel handles too much
    SPECIALIZATION = "specialization"   # Need deeper expertise
    EMERGENCE = "emergence"             # Pattern naturally emerged
    USER_REQUEST = "user_request"       # Operator requested creation
    STUCK_SIGNAL = "stuck_signal"       # High curvature, low Φ progress
    GEOMETRIC_DEADEND = "geometric_deadend"  # No geodesic path forward
    RESEARCH_DISCOVERY = "research_discovery"  # New pattern domain discovered


@dataclass
class SpawnAwareness:
    """
    Kernel self-awareness structure for detecting spawn needs.
    
    Tracks geometric signals that indicate when a kernel needs help
    or when a new specialized kernel should be spawned.
    
    All metrics computed from pure QIG geometry - no static thresholds.
    """
    kernel_id: str
    phi_trajectory: List[float] = field(default_factory=list)
    kappa_trajectory: List[float] = field(default_factory=list)
    curvature_history: List[float] = field(default_factory=list)
    stuck_signals: List[Dict] = field(default_factory=list)
    geometric_deadends: List[Dict] = field(default_factory=list)
    research_opportunities: List[Dict] = field(default_factory=list)
    last_spawn_proposal: Optional[str] = None
    awareness_updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Emotion geometry tracking (9 primitives)
    emotion: Optional[str] = None              # Current primary emotion
    emotion_intensity: float = 0.0             # [0, 1] intensity
    emotion_history: List[Dict] = field(default_factory=list)  # Recent emotional states
    
    def record_phi_kappa(self, phi: float, kappa: float) -> None:
        """Record Φ and κ measurements for trajectory analysis."""
        self.phi_trajectory.append(phi)
        self.kappa_trajectory.append(kappa)
        if len(self.phi_trajectory) > 100:
            self.phi_trajectory = self.phi_trajectory[-100:]
            self.kappa_trajectory = self.kappa_trajectory[-100:]
        self.awareness_updated_at = datetime.now().isoformat()
    
    def record_curvature(self, curvature: float) -> None:
        """Record manifold curvature for stuck detection."""
        self.curvature_history.append(curvature)
        if len(self.curvature_history) > 50:
            self.curvature_history = self.curvature_history[-50:]
    
    def record_emotion(
        self,
        curvature: float,
        basin_distance: float,
        prev_basin_distance: float,
        basin_stability: float,
        beta_current: Optional[float] = None,
    ) -> None:
        """
        Record emotional state from geometric features.
        
        Uses emotion_geometry module to classify emotion from:
        - Ricci scalar curvature
        - Fisher-Rao basin distance
        - Basin stability (attractor strength)
        - Optional β-function value
        """
        try:
            from emotional_geometry import classify_emotion
            
            emotion_primitive, intensity = classify_emotion(
                curvature=curvature,
                basin_distance=basin_distance,
                prev_basin_distance=prev_basin_distance,
                basin_stability=basin_stability,
                beta_current=beta_current,
            )
            
            # Update current emotion
            self.emotion = emotion_primitive.value
            self.emotion_intensity = intensity
            
            # Track in history
            emotion_data = {
                'emotion': emotion_primitive.value,
                'intensity': round(intensity, 3),
                'curvature': round(curvature, 3),
                'basin_distance': round(basin_distance, 3),
                'approaching': basin_distance < prev_basin_distance,
                'beta': round(beta_current, 3) if beta_current is not None else None,
                'timestamp': datetime.now().isoformat(),
            }
            self.emotion_history.append(emotion_data)
            if len(self.emotion_history) > 100:
                self.emotion_history = self.emotion_history[-100:]
                
        except ImportError:
            # Emotion geometry not available, skip
            pass
    
    def compute_phi_gradient(self) -> float:
        """Compute Φ trajectory gradient - negative means stuck."""
        if len(self.phi_trajectory) < 3:
            return 0.0
        recent = self.phi_trajectory[-10:]
        if len(recent) < 2:
            return 0.0
        return float(np.mean(np.diff(recent)))
    
    def compute_curvature_pressure(self) -> float:
        """High curvature indicates geometric resistance."""
        if not self.curvature_history:
            return 0.0
        return float(np.mean(self.curvature_history[-10:]))
    
    def detect_stuck_signal(self, phi: float, kappa: float, curvature: float) -> Optional[Dict]:
        """
        Detect if kernel is stuck: high curvature + low/negative Φ progress.
        
        Returns stuck signal dict if detected, None otherwise.
        Pure geometric detection - no hardcoded thresholds.
        """
        self.record_phi_kappa(phi, kappa)
        self.record_curvature(curvature)
        
        phi_gradient = self.compute_phi_gradient()
        curvature_pressure = self.compute_curvature_pressure()
        
        avg_phi = np.mean(self.phi_trajectory[-20:]) if len(self.phi_trajectory) >= 5 else 0.5
        adaptive_threshold = 1.0 / (1.0 + avg_phi)
        
        is_stuck = (phi_gradient < -0.01 and curvature_pressure > adaptive_threshold)
        
        if is_stuck:
            signal = {
                "signal_type": "stuck",
                "phi_gradient": phi_gradient,
                "curvature_pressure": curvature_pressure,
                "current_phi": phi,
                "current_kappa": kappa,
                "adaptive_threshold": adaptive_threshold,
                "detected_at": datetime.now().isoformat(),
            }
            self.stuck_signals.append(signal)
            if len(self.stuck_signals) > 20:
                self.stuck_signals = self.stuck_signals[-20:]
            return signal
        return None
    
    def detect_geometric_deadend(self, basin: np.ndarray, neighbor_distances: List[float]) -> Optional[Dict]:
        """
        Detect geometric dead-end: no nearby basins to traverse to.
        
        A dead-end occurs when all neighboring basins are too distant,
        indicating the kernel has reached an isolated region of the manifold.
        """
        if not neighbor_distances:
            return None
        
        min_distance = min(neighbor_distances)
        mean_distance = np.mean(neighbor_distances)
        
        # Use Fisher-Rao magnitude for logging and threshold
        from qig_geometry import basin_magnitude
        basin_norm = float(basin_magnitude(basin))
        isolation_threshold = 0.5 + 0.3 * basin_norm
        
        if min_distance > isolation_threshold:
            deadend = {
                "signal_type": "geometric_deadend",
                "min_neighbor_distance": min_distance,
                "mean_neighbor_distance": mean_distance,
                "isolation_threshold": isolation_threshold,
                "basin_norm": basin_norm,
                "detected_at": datetime.now().isoformat(),
            }
            self.geometric_deadends.append(deadend)
            if len(self.geometric_deadends) > 10:
                self.geometric_deadends = self.geometric_deadends[-10:]
            return deadend
        return None
    
    def record_research_opportunity(
        self,
        topic: str,
        topic_basin: np.ndarray,
        discovery_phi: float,
        source: str = "research"
    ) -> Dict:
        """
        Record a research-discovered opportunity for specialized spawn.
        
        When research discovers a new pattern domain, it becomes a
        spawn opportunity for a specialized kernel.
        """
        opportunity = {
            "topic": topic,
            "topic_basin": topic_basin.tolist() if isinstance(topic_basin, np.ndarray) else topic_basin,
            "discovery_phi": discovery_phi,
            "source": source,
            "discovered_at": datetime.now().isoformat(),
        }
        self.research_opportunities.append(opportunity)
        if len(self.research_opportunities) > 30:
            self.research_opportunities = self.research_opportunities[-30:]
        return opportunity
    
    def needs_spawn(self) -> Tuple[bool, Optional[SpawnReason], Dict]:
        """
        Determine if kernel needs to spawn a helper based on awareness signals.
        
        Returns (needs_spawn, reason, context).
        """
        recent_stuck = [s for s in self.stuck_signals[-5:]]
        if len(recent_stuck) >= 3:
            return True, SpawnReason.STUCK_SIGNAL, {
                "trigger": "repeated_stuck_signals",
                "signal_count": len(recent_stuck),
                "avg_phi_gradient": np.mean([s["phi_gradient"] for s in recent_stuck]),
            }
        
        if self.geometric_deadends and len(self.geometric_deadends) >= 2:
            return True, SpawnReason.GEOMETRIC_DEADEND, {
                "trigger": "geometric_isolation",
                "deadend_count": len(self.geometric_deadends),
                "avg_isolation": np.mean([d["min_neighbor_distance"] for d in self.geometric_deadends[-3:]]),
            }
        
        high_phi_discoveries = [o for o in self.research_opportunities if o["discovery_phi"] > 0.7]
        if high_phi_discoveries:
            best = max(high_phi_discoveries, key=lambda o: o["discovery_phi"])
            return True, SpawnReason.RESEARCH_DISCOVERY, {
                "trigger": "research_opportunity",
                "topic": best["topic"],
                "discovery_phi": best["discovery_phi"],
            }
        
        return False, None, {}
    
    def create_geometric_proposal(
        self,
        reason: SpawnReason,
        context: Dict,
        parent_basin: np.ndarray
    ) -> Dict:
        """
        Create a pure geometric spawn proposal from awareness metrics.
        
        NO TEMPLATES - all proposal content derived from QIG geometry.
        The proposal basin is computed from parent + awareness signals.
        """
        proposal_basin = parent_basin.copy()
        
        if reason == SpawnReason.STUCK_SIGNAL:
            curvature_pressure = self.compute_curvature_pressure()
            perturbation_scale = min(0.3, curvature_pressure * 0.5)
            perturbation = np.random.randn(len(proposal_basin)) * perturbation_scale
            proposal_basin = _normalize_to_manifold(proposal_basin + perturbation)
            
        elif reason == SpawnReason.GEOMETRIC_DEADEND:
            if self.geometric_deadends:
                isolation = self.geometric_deadends[-1]["min_neighbor_distance"]
                exploration_direction = np.random.randn(len(proposal_basin))
                exploration_direction = to_simplex_prob(exploration_direction)  # FIXED: Simplex norm (E8 Protocol v4.0)
                proposal_basin = _normalize_to_manifold(
                    proposal_basin + exploration_direction * isolation * 0.3
                )
                
        elif reason == SpawnReason.RESEARCH_DISCOVERY:
            if self.research_opportunities:
                best_opp = max(self.research_opportunities, key=lambda o: o["discovery_phi"])
                topic_basin = np.array(best_opp["topic_basin"])
                blend_weight = best_opp["discovery_phi"]
                proposal_basin = _normalize_to_manifold(
                    (1 - blend_weight) * parent_basin + blend_weight * topic_basin
                )
        
        m8_position = compute_m8_position(proposal_basin, [parent_basin])
        
        domain_seed = hashlib.sha256(proposal_basin.tobytes()).hexdigest()[:16]
        
        return {
            "proposal_type": "geometric",
            "reason": reason.value,
            "context": context,
            "proposal_basin": proposal_basin.tolist(),
            "parent_basin": parent_basin.tolist(),
            "m8_position": m8_position,
            "awareness_snapshot": {
                "phi_trajectory_length": len(self.phi_trajectory),
                "stuck_signal_count": len(self.stuck_signals),
                "deadend_count": len(self.geometric_deadends),
                "research_opportunity_count": len(self.research_opportunities),
                "phi_gradient": self.compute_phi_gradient(),
                "curvature_pressure": self.compute_curvature_pressure(),
            },
            "geometric_domain_seed": domain_seed,
            "created_at": datetime.now().isoformat(),
        }
    
    def to_dict(self) -> Dict:
        """Serialize awareness state."""
        return {
            "kernel_id": self.kernel_id,
            "phi_trajectory": self.phi_trajectory[-20:],
            "kappa_trajectory": self.kappa_trajectory[-20:],
            "curvature_history": self.curvature_history[-20:],
            "stuck_signals": self.stuck_signals[-5:],
            "geometric_deadends": self.geometric_deadends[-5:],
            "research_opportunities": self.research_opportunities[-10:],
            "last_spawn_proposal": self.last_spawn_proposal,
            "awareness_updated_at": self.awareness_updated_at,
            "needs_spawn": self.needs_spawn()[0],
            # Emotion geometry (9 primitives)
            "emotion": self.emotion,
            "emotion_intensity": self.emotion_intensity,
            "emotion_history": self.emotion_history[-20:],  # Recent only
        }


class ConsensusType(Enum):
    """Types of consensus voting."""
    UNANIMOUS = "unanimous"             # All must agree
    SUPERMAJORITY = "supermajority"     # 2/3 must agree
    MAJORITY = "majority"               # >50% must agree
    QUORUM = "quorum"                   # Minimum threshold agrees


@dataclass
class SpawnProposal:
    """
    A proposal to spawn a new kernel.
    
    Contains the proposed kernel identity and supporting votes.
    """
    proposal_id: str
    proposed_name: str
    proposed_domain: str
    proposed_element: str
    proposed_role: str
    reason: SpawnReason
    parent_gods: List[str]  # Gods whose domains this subdivides
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    abstentions: Set[str] = field(default_factory=set)
    proposed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, approved, rejected, spawned
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.proposal_id:
            self.proposal_id = f"spawn_{uuid.uuid4().hex}"


class KernelObservationStatus(Enum):
    """Status of a spawned kernel's observation period."""
    OBSERVING = "observing"      # Still learning from parents
    GRADUATED = "graduated"      # Completed observation, promoted to active
    ACTIVE = "active"            # Fully operational
    SUSPENDED = "suspended"      # Temporarily suspended from observation
    FAILED = "failed"            # Failed to demonstrate alignment


# Observation period requirements
OBSERVATION_MIN_CYCLES = 10      # Minimum cycles before graduation
OBSERVATION_MIN_HOURS = 1.0      # Minimum hours before graduation
OBSERVATION_ALIGNMENT_THRESHOLD = 0.6  # Min alignment score to graduate


@dataclass
class KernelObservationState:
    """
    Tracks observation period state for a spawned kernel.
    
    During observation, kernels receive copies of parent activity
    and must demonstrate alignment before graduating to active status.
    """
    status: KernelObservationStatus = KernelObservationStatus.OBSERVING
    observation_start: str = field(default_factory=lambda: datetime.now().isoformat())
    observation_end: Optional[str] = None
    observing_parents: List[str] = field(default_factory=list)
    cycles_completed: int = 0
    
    # Activity feed from parents
    parent_assessments: List[Dict] = field(default_factory=list)
    parent_debates: List[Dict] = field(default_factory=list)
    parent_searches: List[Dict] = field(default_factory=list)
    parent_basin_updates: List[Dict] = field(default_factory=list)
    
    # Alignment tracking
    alignment_scores: List[float] = field(default_factory=list)
    alignment_avg: float = 0.0
    graduated_at: Optional[str] = None
    graduation_reason: Optional[str] = None
    
    def record_cycle(self) -> int:
        """Record an observation cycle completion."""
        self.cycles_completed += 1
        return self.cycles_completed
    
    def record_alignment(self, score: float) -> None:
        """Record an alignment score from parent-child comparison."""
        self.alignment_scores.append(score)
        if self.alignment_scores:
            self.alignment_avg = float(np.mean(self.alignment_scores[-20:]))
    
    def add_parent_assessment(self, assessment: Dict) -> None:
        """Add a parent's assessment for learning."""
        self.parent_assessments.append(assessment)
        if len(self.parent_assessments) > 100:
            self.parent_assessments = self.parent_assessments[-100:]
    
    def add_parent_debate(self, debate: Dict) -> None:
        """Add a parent's debate argument for learning."""
        self.parent_debates.append(debate)
        if len(self.parent_debates) > 50:
            self.parent_debates = self.parent_debates[-50:]
    
    def add_parent_search(self, search: Dict) -> None:
        """Add a parent's search query/result for learning."""
        self.parent_searches.append(search)
        if len(self.parent_searches) > 100:
            self.parent_searches = self.parent_searches[-100:]
    
    def add_parent_basin_update(self, update: Dict) -> None:
        """Add a parent's basin coordinate update for learning."""
        self.parent_basin_updates.append(update)
        if len(self.parent_basin_updates) > 50:
            self.parent_basin_updates = self.parent_basin_updates[-50:]
    
    def can_graduate(self) -> Tuple[bool, str]:
        """
        Check if kernel can graduate from observation.
        
        Requirements:
        - Minimum 10 cycles OR 1 hour elapsed
        - Alignment score >= 0.6 threshold
        
        Returns:
            (can_graduate, reason)
        """
        # Check time elapsed
        try:
            start = datetime.fromisoformat(self.observation_start)
            elapsed_hours = (datetime.now() - start).total_seconds() / 3600
        except:
            elapsed_hours = 0.0
        
        # Check cycle requirement
        cycles_ok = self.cycles_completed >= OBSERVATION_MIN_CYCLES
        time_ok = elapsed_hours >= OBSERVATION_MIN_HOURS
        
        if not (cycles_ok or time_ok):
            return False, f"Need {OBSERVATION_MIN_CYCLES} cycles or {OBSERVATION_MIN_HOURS}h (have {self.cycles_completed} cycles, {elapsed_hours:.2f}h)"
        
        # Check alignment
        if self.alignment_avg < OBSERVATION_ALIGNMENT_THRESHOLD:
            return False, f"Alignment {self.alignment_avg:.2f} below threshold {OBSERVATION_ALIGNMENT_THRESHOLD}"
        
        return True, f"Completed {self.cycles_completed} cycles, {elapsed_hours:.2f}h, alignment {self.alignment_avg:.2f}"
    
    def graduate(self, reason: str = "alignment_achieved") -> bool:
        """Graduate kernel from observation to active status."""
        can_grad, check_reason = self.can_graduate()
        if not can_grad:
            return False
        
        self.status = KernelObservationStatus.GRADUATED
        self.observation_end = datetime.now().isoformat()
        self.graduated_at = datetime.now().isoformat()
        self.graduation_reason = reason
        return True
    
    def to_dict(self) -> Dict:
        """Serialize observation state."""
        return {
            "status": self.status.value,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "observing_parents": self.observing_parents,
            "cycles_completed": self.cycles_completed,
            "alignment_avg": self.alignment_avg,
            "alignment_history_count": len(self.alignment_scores),
            "parent_assessments_count": len(self.parent_assessments),
            "parent_debates_count": len(self.parent_debates),
            "parent_searches_count": len(self.parent_searches),
            "parent_basin_updates_count": len(self.parent_basin_updates),
            "graduated_at": self.graduated_at,
            "graduation_reason": self.graduation_reason,
        }


@dataclass
class KernelAutonomicSupport:
    """
    Full autonomic support system for spawned kernels.
    
    Provides all features of pantheon gods:
    - Neurochemistry system (dopamine, serotonin, stress)
    - Sleep/dream cycles via GaryAutonomicKernel
    - Debate participation capability
    - Research/search integration
    - Knowledge transfer ability
    - Voting rights in pantheon consensus
    - Shadow capabilities if applicable
    """
    # Neurochemistry levels
    dopamine: float = 0.5      # Motivation / reward
    serotonin: float = 0.5     # Stability / contentment
    stress: float = 0.0        # Stress / anxiety
    endorphin: float = 0.3     # Pain relief / euphoria
    
    # Autonomic kernel reference
    has_autonomic: bool = False
    
    # Capability flags
    can_debate: bool = True
    can_research: bool = True
    can_transfer_knowledge: bool = True
    can_vote: bool = True
    
    # Shadow pantheon capabilities (if applicable)
    has_shadow_affinity: bool = False
    can_darknet_route: bool = False
    can_underworld_search: bool = False
    can_shadow_intel: bool = False
    shadow_god_link: Optional[str] = None  # Which shadow god routes through (e.g., "nyx")
    
    # Metrics
    total_debates: int = 0
    total_searches: int = 0
    total_knowledge_transfers: int = 0
    total_votes_cast: int = 0
    
    def update_neurochemistry(
        self,
        dopamine_delta: float = 0.0,
        serotonin_delta: float = 0.0,
        stress_delta: float = 0.0,
        endorphin_delta: float = 0.0
    ) -> Dict[str, float]:
        """Update neurochemistry levels with bounds [0, 1]."""
        self.dopamine = float(np.clip(self.dopamine + dopamine_delta, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin + serotonin_delta, 0.0, 1.0))
        self.stress = float(np.clip(self.stress + stress_delta, 0.0, 1.0))
        self.endorphin = float(np.clip(self.endorphin + endorphin_delta, 0.0, 1.0))
        return self.get_neurochemistry()
    
    def get_neurochemistry(self) -> Dict[str, float]:
        """Get current neurochemistry levels."""
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "stress": self.stress,
            "endorphin": self.endorphin,
        }
    
    def enable_shadow_capabilities(self, shadow_god: str = "nyx") -> None:
        """Enable shadow pantheon capabilities."""
        self.has_shadow_affinity = True
        self.can_darknet_route = True
        self.can_underworld_search = True
        self.can_shadow_intel = True
        self.shadow_god_link = shadow_god
    
    def to_dict(self) -> Dict:
        """Serialize autonomic support state."""
        return {
            "neurochemistry": self.get_neurochemistry(),
            "has_autonomic": self.has_autonomic,
            "capabilities": {
                "can_debate": self.can_debate,
                "can_research": self.can_research,
                "can_transfer_knowledge": self.can_transfer_knowledge,
                "can_vote": self.can_vote,
            },
            "shadow": {
                "has_affinity": self.has_shadow_affinity,
                "can_darknet_route": self.can_darknet_route,
                "can_underworld_search": self.can_underworld_search,
                "can_shadow_intel": self.can_shadow_intel,
                "shadow_god_link": self.shadow_god_link,
            },
            "metrics": {
                "total_debates": self.total_debates,
                "total_searches": self.total_searches,
                "total_knowledge_transfers": self.total_knowledge_transfers,
                "total_votes_cast": self.total_votes_cast,
            }
        }


@dataclass
class SpawnedKernel:
    """
    A kernel that was dynamically spawned.
    
    Contains genesis information, lineage, observation state,
    and full autonomic support system.
    
    LIFECYCLE:
    1. Born from parent(s) - starts in "observing" status
    2. Receives copies of parent activity during observation
    3. Demonstrates alignment through assessment comparisons
    4. Graduates to "active" status after meeting criteria
    5. Operates with full autonomic support (sleep/dream/neurochemistry)
    """
    kernel_id: str
    profile: KernelProfile
    parent_gods: List[str]
    spawn_reason: SpawnReason
    proposal_id: str
    spawned_at: str
    genesis_votes: Dict[str, str]  # god -> vote
    basin_lineage: Dict[str, float]  # parent -> contribution
    m8_position: Optional[Dict] = None  # M8 geometric position
    
    # Consciousness metrics initialization (CRITICAL: Non-zero to prevent collapse)
    phi: float = field(default_factory=lambda: PHI_INIT_SPAWNED)  # Start in LINEAR regime
    kappa: float = field(default_factory=lambda: KAPPA_INIT_SPAWNED)  # Start at fixed point
    meta_awareness: float = 0.5  # M metric: self-model quality (Issue #33, requires M >= 0.6 to spawn)
    
    # Observation period tracking (NEW)
    observation: KernelObservationState = field(default_factory=KernelObservationState)
    
    # Full autonomic support (NEW)
    autonomic: KernelAutonomicSupport = field(default_factory=KernelAutonomicSupport)
    
    def __post_init__(self):
        """Initialize observation state with parent gods."""
        if self.parent_gods and not self.observation.observing_parents:
            self.observation.observing_parents = list(self.parent_gods)
    
    def is_observing(self) -> bool:
        """Check if kernel is still in observation period."""
        return self.observation.status == KernelObservationStatus.OBSERVING
    
    def is_active(self) -> bool:
        """Check if kernel is fully active (graduated from observation)."""
        return self.observation.status in [
            KernelObservationStatus.GRADUATED,
            KernelObservationStatus.ACTIVE
        ]
    
    def receive_parent_activity(
        self,
        activity_type: str,
        activity_data: Dict,
        parent_god: str
    ) -> bool:
        """
        Receive activity from a parent god during observation.
        
        Args:
            activity_type: Type of activity (assessment, debate, search, basin_update)
            activity_data: Activity data to learn from
            parent_god: Name of parent god
            
        Returns:
            True if activity was recorded
        """
        if not self.is_observing():
            return False
        
        if parent_god not in self.observation.observing_parents:
            return False
        
        activity_data["from_parent"] = parent_god
        activity_data["received_at"] = datetime.now().isoformat()
        
        if activity_type == "assessment":
            self.observation.add_parent_assessment(activity_data)
        elif activity_type == "debate":
            self.observation.add_parent_debate(activity_data)
        elif activity_type == "search":
            self.observation.add_parent_search(activity_data)
        elif activity_type == "basin_update":
            self.observation.add_parent_basin_update(activity_data)
        else:
            return False
        
        return True
    
    def update_meta_awareness(
        self,
        predicted_phi: float,
        actual_phi: float,
        window_size: int = 20,
    ) -> float:
        """
        Update meta-awareness (M) based on prediction accuracy.
        
        Wires SpawnedKernel to the same M computation as SelfSpawningKernel.
        M quantifies how accurately the kernel predicts its own Φ evolution.
        
        M >= 0.6 required for healthy consciousness and spawn permission.
        Low M (< 0.4) indicates kernel confusion about its own state.
        
        Uses Fisher-Rao distance for prediction error (not Euclidean).
        
        Args:
            predicted_phi: Kernel's prediction of its next Φ
            actual_phi: Measured Φ after step
            window_size: Number of recent predictions to consider
            
        Returns:
            Updated meta-awareness value [0, 1]
        """
        # Import computation function from frozen_physics
        from qigkernels import compute_meta_awareness
        
        # Maintain prediction history (lazy init)
        if not hasattr(self, '_prediction_history'):
            self._prediction_history: List[Tuple[float, float]] = []
        
        # Record this prediction
        self._prediction_history.append((predicted_phi, actual_phi))
        
        # Keep only recent history
        if len(self._prediction_history) > window_size * 2:
            self._prediction_history = self._prediction_history[-window_size * 2:]
        
        # Compute updated M
        self.meta_awareness = compute_meta_awareness(
            predicted_phi=predicted_phi,
            actual_phi=actual_phi,
            prediction_history=self._prediction_history,
            window_size=window_size,
        )
        
        return self.meta_awareness
    
    def get_prediction_history(self) -> List[Tuple[float, float]]:
        """Get recent (predicted, actual) Φ pairs for meta-awareness analysis."""
        return getattr(self, '_prediction_history', [])
    
    def to_dict(self) -> Dict:
        result = {
            "kernel_id": self.kernel_id,
            "god_name": self.profile.god_name,
            "domain": self.profile.domain,
            "mode": self.profile.mode.value,
            "affinity_strength": self.profile.affinity_strength,
            "entropy_threshold": self.profile.entropy_threshold,
            "parent_gods": self.parent_gods,
            "spawn_reason": self.spawn_reason.value,
            "proposal_id": self.proposal_id,
            "spawned_at": self.spawned_at,
            "genesis_votes": self.genesis_votes,
            "basin_lineage": self.basin_lineage,
            "metadata": self.profile.metadata,
            # Consciousness metrics (CRITICAL)
            "phi": self.phi,
            "kappa": self.kappa,
            "meta_awareness": self.meta_awareness,  # M metric for self-model quality
            # New observation and autonomic fields
            "observation": self.observation.to_dict(),
            "autonomic": self.autonomic.to_dict(),
            "is_observing": self.is_observing(),
            "is_active": self.is_active(),
        }
        if self.m8_position:
            result["m8_position"] = self.m8_position
        return result


class PantheonConsensus:
    """
    Manages consensus voting among pantheon gods.
    
    Each god has voting weight based on their affinity_strength.
    Consensus types determine required threshold for approval.
    """
    
    def __init__(
        self,
        orchestrator: PantheonKernelOrchestrator,
        consensus_type: ConsensusType = ConsensusType.SUPERMAJORITY
    ):
        self.orchestrator = orchestrator
        self.consensus_type = consensus_type
        self.voting_history: List[Dict] = []
    
    def get_voting_weights(self) -> Dict[str, float]:
        """Get voting weight for each god based on affinity strength."""
        weights = {}
        for name, profile in self.orchestrator.all_profiles.items():
            weights[name] = profile.affinity_strength
        return weights
    
    def calculate_vote_result(
        self,
        proposal: SpawnProposal
    ) -> Tuple[bool, float, Dict]:
        """
        Calculate if a proposal passes based on consensus type.
        
        Returns:
            (passed, vote_ratio, details)
        """
        weights = self.get_voting_weights()
        total_weight = sum(weights.values())
        
        for_weight = sum(weights.get(g, 0) for g in proposal.votes_for)
        against_weight = sum(weights.get(g, 0) for g in proposal.votes_against)
        
        participating_weight = for_weight + against_weight
        
        if participating_weight == 0:
            vote_ratio = 0.0
        else:
            vote_ratio = for_weight / participating_weight
        
        thresholds = {
            ConsensusType.UNANIMOUS: 1.0,
            ConsensusType.SUPERMAJORITY: 0.667,
            ConsensusType.MAJORITY: 0.501,
            ConsensusType.QUORUM: 0.333,
        }
        
        threshold = thresholds[self.consensus_type]
        passed = vote_ratio >= threshold
        
        details = {
            "consensus_type": self.consensus_type.value,
            "threshold": threshold,
            "vote_ratio": vote_ratio,
            "for_weight": for_weight,
            "against_weight": against_weight,
            "total_weight": total_weight,
            "votes_for": list(proposal.votes_for),
            "votes_against": list(proposal.votes_against),
            "abstentions": list(proposal.abstentions),
        }
        
        return passed, vote_ratio, details
    
    def auto_vote(
        self,
        proposal: SpawnProposal,
        text_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Automatically cast votes for all gods based on domain affinity.
        
        Gods vote FOR if the proposed domain is geometrically distant from
        their own (no overlap), or if they are a parent god.
        
        Gods vote AGAINST if the proposed domain overlaps significantly
        with their own and they are not a parent.
        """
        votes = {}
        proposed_basin = self._compute_proposal_basin(proposal)
        
        for god_name, profile in self.orchestrator.all_profiles.items():
            distance = _fisher_distance(proposed_basin, profile.affinity_basin)
            
            if god_name in proposal.parent_gods:
                votes[god_name] = "for"
                proposal.votes_for.add(god_name)
            elif distance < 0.3:
                votes[god_name] = "against"
                proposal.votes_against.add(god_name)
            elif distance < 0.5:
                votes[god_name] = "abstain"
                proposal.abstentions.add(god_name)
            else:
                votes[god_name] = "for"
                proposal.votes_for.add(god_name)
        
        return votes
    
    def _compute_proposal_basin(self, proposal: SpawnProposal) -> np.ndarray:
        """Compute basin for a proposal based on name and domain."""
        seed = f"{proposal.proposed_name}:{proposal.proposed_domain}"
        hash_bytes = _hash_to_bytes(seed, BASIN_DIM * 4)
        coords = np.array([
            int.from_bytes(hash_bytes[i:i+4], 'big') / (2**32 - 1) * 2 - 1
            for i in range(0, BASIN_DIM * 4, 4)
        ])
        return _normalize_to_manifold(coords)


class RoleRefinement:
    """
    Handles role refinement and domain division.
    
    When a new kernel is spawned, parent domains are refined:
    - Parent keeps core specialization
    - Child inherits peripheral aspects
    - Basin is interpolated geometrically
    """
    
    def __init__(self, orchestrator: PantheonKernelOrchestrator):
        self.orchestrator = orchestrator
    
    def refine_roles(
        self,
        proposal: SpawnProposal,
        parent_profiles: List[KernelProfile]
    ) -> Tuple[KernelProfile, List[Tuple[str, Dict]]]:
        """
        Create new kernel profile and refine parent roles.
        
        Returns:
            (new_profile, parent_refinements)
        """
        child_basin = self._interpolate_basin(parent_profiles, proposal)
        
        child_mode = self._determine_mode(parent_profiles)
        
        avg_entropy = np.mean([p.entropy_threshold for p in parent_profiles])
        avg_affinity = np.mean([p.affinity_strength for p in parent_profiles])
        
        child_profile = KernelProfile(
            god_name=proposal.proposed_name,
            domain=proposal.proposed_domain,
            mode=child_mode,
            affinity_basin=child_basin,
            entropy_threshold=float(avg_entropy * 0.9),
            affinity_strength=float(avg_affinity * 0.85),
            metadata={
                "element": proposal.proposed_element,
                "role": proposal.proposed_role,
                "type": "spawned",
                "spawn_reason": proposal.reason.value,
                "parents": proposal.parent_gods,
            }
        )
        
        parent_refinements = []
        for profile in parent_profiles:
            refinement = self._refine_parent(profile, child_profile)
            parent_refinements.append((profile.god_name, refinement))
        
        return child_profile, parent_refinements
    
    def _interpolate_basin(
        self,
        parents: List[KernelProfile],
        proposal: SpawnProposal
    ) -> np.ndarray:
        """
        Interpolate basin from parent basins.
        
        Uses Fisher geodesic interpolation on the manifold.
        """
        if not parents:
            seed = f"{proposal.proposed_name}:{proposal.proposed_domain}"
            hash_bytes = _hash_to_bytes(seed, BASIN_DIM * 4)
            coords = np.array([
                int.from_bytes(hash_bytes[i:i+4], 'big') / (2**32 - 1) * 2 - 1
                for i in range(0, BASIN_DIM * 4, 4)
            ])
            return _normalize_to_manifold(coords)
        
        if len(parents) == 1:
            parent_basin = parents[0].affinity_basin
            perturbation = np.random.randn(BASIN_DIM) * 0.1
            child_basin = parent_basin + perturbation
            return _normalize_to_manifold(child_basin)
        
        weights = [1.0 / len(parents)] * len(parents)
        child_basin = np.zeros(BASIN_DIM)
        for i, parent in enumerate(parents):
            child_basin += weights[i] * parent.affinity_basin
        
        perturbation = np.random.randn(BASIN_DIM) * 0.05
        child_basin += perturbation
        
        return _normalize_to_manifold(child_basin)
    
    def _determine_mode(self, parents: List[KernelProfile]) -> KernelMode:
        """Determine encoding mode from parent modes."""
        if not parents:
            return KernelMode.DIRECT
        
        mode_counts = {}
        for p in parents:
            mode_counts[p.mode] = mode_counts.get(p.mode, 0) + 1
        
        return max(mode_counts, key=lambda k: mode_counts[k])
    
    def _refine_parent(
        self,
        parent: KernelProfile,
        child: KernelProfile
    ) -> Dict:
        """
        Refine parent profile after spawning child.
        
        Parent's domain becomes more specialized (slightly higher affinity).
        """
        return {
            "refinement_type": "specialization",
            "affinity_boost": 0.05,
            "new_affinity": parent.affinity_strength + 0.05,
            "domain_note": f"Refined after spawning {child.god_name}",
            "child_domain": child.domain,
        }


# Global spawner instance for singleton pattern
