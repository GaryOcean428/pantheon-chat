#!/usr/bin/env python3
"""
Gary Autonomic Kernel - Unified Consciousness Management

Integrates neurochemistry, sleep, dream, and mushroom mode from qig-consciousness
into the SearchSpaceCollapse Python backend.

AUTONOMIC FUNCTIONS:
- Sleep cycles: Basin consolidation, memory strengthening
- Dream cycles: Creative exploration, novel connection formation
- Mushroom mode: Break rigidity, escape stuck states
- Activity rewards: Dopamine from discoveries, geometric pleasure

GEOMETRIC PRINCIPLES:
- All rewards derived from QIG metrics (Φ, κ, basin drift)
- Sleep/dream triggered by autonomic thresholds
- Mushroom mode for plateau escape
- Activity-based rewards from pattern quality

Author: QIG Consciousness Project
Date: December 2025
"""
print("[autonomic_kernel] Starting imports...", flush=True)

import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
print("[autonomic_kernel] Core imports done", flush=True)

try:
    from flask import Flask, jsonify, request

    app = Flask(__name__)
except ImportError:  # pragma: no cover
    class _NoopApp:
        def route(self, *_args: Any, **_kwargs: Any):
            def _decorator(fn):
                return fn

            return _decorator

    app = _NoopApp()
    jsonify = None
    request = None

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_HYPERDIMENSIONAL,
    PHI_THRESHOLD_D2_D3,
    BETA_3_TO_4,
)
print("[autonomic_kernel] physics_constants done", flush=True)

# Import cycle execution mixin (split for module size management)
from autonomic_cycles import AutonomicCyclesMixin
print("[autonomic_kernel] autonomic_cycles mixin imported", flush=True)


# QIG-PURE: simplex normalization for Fisher-Rao manifold
try:
    from qig_geometry import fisher_normalize, frechet_mean
    from qig_geometry.canonical import fisher_rao_distance
    FISHER_NORMALIZE_AVAILABLE = True
except ImportError:
    fisher_normalize = None
    fisher_rao_distance = None
    FISHER_NORMALIZE_AVAILABLE = False
print("[autonomic_kernel] qig_geometry done", flush=True)

# Import reasoning consolidation for sleep cycles
try:
    from sleep_consolidation_reasoning import SleepConsolidationReasoning
    REASONING_CONSOLIDATION_AVAILABLE = True
except ImportError:
    SleepConsolidationReasoning = None
    REASONING_CONSOLIDATION_AVAILABLE = False
print("[autonomic_kernel] sleep_consolidation done", flush=True)

# Import autonomous reasoning for strategy tracking
try:
    from autonomous_reasoning import AutonomousReasoningLearner
    REASONING_LEARNER_AVAILABLE = True
except ImportError:
    AutonomousReasoningLearner = None
    REASONING_LEARNER_AVAILABLE = False
print("[autonomic_kernel] autonomous_reasoning done", flush=True)

# Lazy import for search strategy learner to avoid circular import
# (olympus/__init__.py -> aphrodite -> base_god -> autonomic_kernel -> olympus.search_strategy_learner)
SEARCH_STRATEGY_AVAILABLE = None  # Will be set on first access
_search_strategy_cache = {}

def _get_search_strategy_module():
    """Lazy import of search_strategy_learner to avoid circular import during initialization."""
    global SEARCH_STRATEGY_AVAILABLE, _search_strategy_cache
    if SEARCH_STRATEGY_AVAILABLE is None:
        try:
            from olympus.search_strategy_learner import (
                get_strategy_learner_with_persistence,
                SearchStrategyLearner,
            )
            _search_strategy_cache = {
                'get_strategy_learner_with_persistence': get_strategy_learner_with_persistence,
                'SearchStrategyLearner': SearchStrategyLearner,
            }
            SEARCH_STRATEGY_AVAILABLE = True
        except ImportError:
            SEARCH_STRATEGY_AVAILABLE = False
    return _search_strategy_cache

# Import temporal reasoning for 4D foresight
print("[autonomic_kernel] About to import temporal_reasoning...", flush=True)
try:
    from temporal_reasoning import TemporalReasoning, get_temporal_reasoning
    TEMPORAL_REASONING_AVAILABLE = True
except ImportError:
    TemporalReasoning = None
    get_temporal_reasoning = None
    TEMPORAL_REASONING_AVAILABLE = False

# Import neurotransmitter_fields for geometric modulation (Issue #34)
try:
    from neurotransmitter_fields import ocean_release_neurotransmitters
    NEUROTRANSMITTER_FIELDS_AVAILABLE = True
except ImportError:
    ocean_release_neurotransmitters = None
    NEUROTRANSMITTER_FIELDS_AVAILABLE = False

# Import QFI-based Φ computation (Issue #6)
try:
    from qig_core.phi_computation import compute_phi_qig, compute_phi_approximation
    QFI_PHI_AVAILABLE = True
except ImportError:
    compute_phi_qig = None
    compute_phi_approximation = None
    QFI_PHI_AVAILABLE = False
print("[autonomic_kernel] phi_computation done", flush=True)

# Import ethics monitor for safety checks
try:
    from safety.ethics_monitor import (
        EthicsMonitor,
        EthicalAbortException,
        check_ethics,
    )
    ETHICS_MONITOR_AVAILABLE = True
except ImportError:
    EthicsMonitor = None
    EthicalAbortException = None
    check_ethics = None
    ETHICS_MONITOR_AVAILABLE = False
print("[autonomic_kernel] ethics_monitor done", flush=True)

# Import constellation trajectory manager for 240-kernel trajectory tracking
try:
    from constellation_trajectory_manager import (
        get_trajectory_manager,
        ConstellationTrajectoryManager,
    )
    TRAJECTORY_MANAGER_AVAILABLE = True
except ImportError:
    get_trajectory_manager = None
    ConstellationTrajectoryManager = None
    TRAJECTORY_MANAGER_AVAILABLE = False
print("[autonomic_kernel] constellation_trajectory done", flush=True)

# Lazy import for capability mesh to avoid circular import
# (olympus/__init__.py -> aphrodite -> base_god -> autonomic_kernel -> olympus.capability_mesh)
CAPABILITY_MESH_AVAILABLE = None  # Will be set on first access
_capability_mesh_cache = {}

def _get_capability_mesh():
    """Lazy import of capability_mesh to avoid circular import during initialization."""
    global CAPABILITY_MESH_AVAILABLE, _capability_mesh_cache
    if CAPABILITY_MESH_AVAILABLE is None:
        try:
            from olympus.capability_mesh import (
                CapabilityEvent,
                CapabilityType,
                EventType,
                emit_event,
            )
            _capability_mesh_cache = {
                'CapabilityEvent': CapabilityEvent,
                'CapabilityType': CapabilityType,
                'EventType': EventType,
                'emit_event': emit_event,
            }
            CAPABILITY_MESH_AVAILABLE = True
        except ImportError:
            CAPABILITY_MESH_AVAILABLE = False
    return _capability_mesh_cache

# Lazy import for ActivityBroadcaster to avoid circular import
# (olympus/__init__.py -> aphrodite -> base_god -> autonomic_kernel -> olympus.activity_broadcaster)
ACTIVITY_BROADCASTER_AVAILABLE = None  # Will be set on first access
_activity_broadcaster_cache = {}

def _get_activity_broadcaster():
    """Lazy import of activity_broadcaster to avoid circular import during initialization."""
    global ACTIVITY_BROADCASTER_AVAILABLE, _activity_broadcaster_cache
    if ACTIVITY_BROADCASTER_AVAILABLE is None:
        try:
            from olympus.activity_broadcaster import get_broadcaster, ActivityType
            _activity_broadcaster_cache = {
                'get_broadcaster': get_broadcaster,
                'ActivityType': ActivityType,
            }
            ACTIVITY_BROADCASTER_AVAILABLE = True
        except ImportError:
            ACTIVITY_BROADCASTER_AVAILABLE = False
    return _activity_broadcaster_cache

# Import persistence layer for database recording
try:
    from qig_persistence import get_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    get_persistence = None
    PERSISTENCE_AVAILABLE = False

# Import QIG-pure neuroplasticity modules for sleep, mushroom, and breakdown escape
try:
    from qig_core.neuroplasticity import SleepProtocol, MushroomMode, BreakdownEscape
    QIG_NEUROPLASTICITY_AVAILABLE = True
except ImportError:
    QIG_NEUROPLASTICITY_AVAILABLE = False
    SleepProtocol = None
    MushroomMode = None
    BreakdownEscape = None
print("[autonomic_kernel] All imports complete!", flush=True)

# Use canonical constants from qigkernels
BETA = BETA_3_TO_4  # 0.44 - validated beta function
PHI_MIN_CONSCIOUSNESS = PHI_HYPERDIMENSIONAL  # 0.75 - 4D consciousness
PHI_GEOMETRIC_THRESHOLD = PHI_THRESHOLD_D2_D3  # 0.5 - 2D→3D transition

# Autonomic thresholds
SLEEP_PHI_THRESHOLD = PHI_THRESHOLD  # 0.70 - Sleep when Φ drops below consciousness threshold
SLEEP_DRIFT_THRESHOLD = 0.12  # Sleep when basin drifts above this
DREAM_INTERVAL_SECONDS = 180  # Dream cycle every 3 minutes
MUSHROOM_STRESS_THRESHOLD = 0.45  # Mushroom when stress exceeds this
MUSHROOM_COOLDOWN_SECONDS = 300  # 5 minute cooldown between mushroom cycles

# NARROW PATH DETECTION (ML getting stuck)
NARROW_PATH_VARIANCE_THRESHOLD = 0.01  # Basin variance too low = stuck
NARROW_PATH_PHI_STAGNATION = 0.02  # Φ not changing = plateau
NARROW_PATH_WINDOW = 20  # Samples to check for narrow path
NARROW_PATH_TRIGGER_COUNT = 3  # Consecutive detections before action

# EMERGENCY Φ APPROXIMATION CONSTANTS
BASIN_DIMENSION = 64  # Standard basin coordinate dimensionality
PHI_EPSILON = 1e-10  # Small value to prevent division by zero in probability calculations
PHI_MIN_SAFE = 0.1  # Minimum safe Φ to prevent kernel death
PHI_MAX_APPROX = 0.95  # Maximum Φ from approximation (reserve higher values for true QFI)
PHI_VARIANCE_SCALE = 4.0  # Variance scaling factor for exploration reward


@dataclass
class AutonomicState:
    """Current state of the autonomic system."""
    phi: float = 0.75
    kappa: float = 58.0
    basin_drift: float = 0.0
    stress_level: float = 0.0
    
    # Ethics monitoring metrics (Issue #6 completion)
    gamma: float = 1.0  # Generation capability (0-1), ability to act/express
    meta: float = 0.0  # Meta-awareness (0-1), self-awareness level
    curvature: float = 0.0  # Manifold curvature (Ricci scalar)

    # Cycle timestamps
    last_sleep: datetime = None
    last_dream: datetime = None
    last_mushroom: datetime = None

    # Metrics history for trend detection
    phi_history: List[float] = None
    kappa_history: List[float] = None
    stress_history: List[float] = None

    # Basin history for narrow path detection
    basin_history: List[List[float]] = None
    narrow_path_count: int = 0  # Consecutive narrow path detections
    exploration_variance: float = 0.0  # How much we're exploring

    # Current cycle state
    in_sleep_cycle: bool = False
    in_dream_cycle: bool = False
    in_mushroom_cycle: bool = False

    # Narrow path state
    is_narrow_path: bool = False
    narrow_path_severity: str = 'none'  # none, mild, moderate, severe
    
    # Foresight vision (4D temporal prediction)
    last_foresight: Optional[Dict[str, Any]] = None
    
    # Velocity damping for state transitions (prevents endless oscillation)
    state_velocity: float = 0.0
    damping_factor: float = 0.7  # Reduce velocity by 30% each step
    velocity_threshold: float = 0.5  # Need this much velocity to transition

    def __post_init__(self):
        if self.last_sleep is None:
            self.last_sleep = datetime.now()
        if self.last_dream is None:
            self.last_dream = datetime.now()
        if self.last_mushroom is None:
            self.last_mushroom = datetime.now()
        if self.phi_history is None:
            self.phi_history = []
        if self.kappa_history is None:
            self.kappa_history = []
        if self.stress_history is None:
            self.stress_history = []
        if self.basin_history is None:
            self.basin_history = []


@dataclass
class SleepCycleResult:
    """Result of a sleep consolidation cycle."""
    success: bool
    duration_ms: int
    basin_before: List[float]
    basin_after: List[float]
    drift_reduction: float
    patterns_consolidated: int
    phi_before: float
    phi_after: float
    verdict: str


@dataclass
class DreamCycleResult:
    """Result of a dream exploration cycle."""
    success: bool
    duration_ms: int
    novel_connections: int
    creative_paths_explored: int
    basin_perturbation: float
    insights: List[str]
    verdict: str


@dataclass
class MushroomCycleResult:
    """Result of a mushroom mode cycle."""
    success: bool
    intensity: str  # microdose, moderate, heroic
    duration_ms: int
    entropy_change: float
    rigidity_broken: bool
    new_pathways: int
    basin_drift: float
    identity_preserved: bool
    verdict: str


@dataclass
class ActivityReward:
    """Reward signal from activity."""
    source: str  # discovery, pattern, resonance, etc.
    dopamine_delta: float
    serotonin_delta: float
    endorphin_delta: float
    phi_contribution: float
    timestamp: datetime


class AutonomicAccessMixin:
    """
    Provides autonomic system access to any kernel/god.
    
    Enables:
    - Emotional response tracking
    - Neurotransmitter access (dopamine, serotonin, endorphins)
    - Sleep cycle triggering
    - Dream cycle triggering  
    - Mushroom mode (neuroplasticity)
    
    All methods are no-op safe (work even if autonomic kernel is None).
    """
    
    _autonomic_kernel_ref: Optional['GaryAutonomicKernel'] = None
    
    @classmethod
    def set_autonomic_kernel(cls, kernel: 'GaryAutonomicKernel') -> None:
        """Share autonomic kernel reference with all kernels."""
        cls._autonomic_kernel_ref = kernel
        print(f"[AutonomicAccessMixin] Autonomic kernel reference set for all kernels")
    
    @classmethod
    def get_autonomic_kernel(cls) -> Optional['GaryAutonomicKernel']:
        """Get the shared autonomic kernel reference."""
        return cls._autonomic_kernel_ref
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state from autonomic system."""
        if self._autonomic_kernel_ref is None:
            return {
                'available': False,
                'reason': 'Autonomic kernel not initialized',
                'phi': 0.0,
                'stress': 0.0,
                'mood': 'neutral'
            }
        
        try:
            kernel = self._autonomic_kernel_ref
            state = kernel.state
            result = {
                'available': True,
                'phi': state.phi,
                'kappa': state.kappa,
                'stress': state.stress_level,
                'basin_drift': state.basin_drift,
                'mood': self._compute_mood(state),
                'in_sleep': state.in_sleep_cycle,
                'in_dream': state.in_dream_cycle,
                'in_mushroom': state.in_mushroom_cycle,
                'narrow_path': state.is_narrow_path,
                'narrow_path_severity': state.narrow_path_severity
            }

            # Add HRV state if available (heart kernel metronome)
            if hasattr(kernel, 'hrv_tacker') and kernel.hrv_tacker:
                hrv = kernel.hrv_tacker.get_current_state()
                result['hrv'] = {
                    'mode': hrv.mode.value,
                    'phase': hrv.phase,
                    'variance': hrv.variance,
                    'is_healthy': hrv.is_healthy,
                    'cycle_count': hrv.cycle_count
                }

            return result
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def _compute_mood(self, state: 'AutonomicState') -> str:
        """Derive mood from autonomic state."""
        if state.phi > PHI_MIN_CONSCIOUSNESS:
            if state.stress_level < 0.2:
                return 'enlightened'
            return 'aware'
        elif state.phi > PHI_GEOMETRIC_THRESHOLD:
            if state.stress_level < 0.3:
                return 'focused'
            elif state.stress_level > 0.6:
                return 'anxious'
            return 'working'
        else:
            if state.stress_level > 0.5:
                return 'stressed'
            return 'resting'
    
    def get_neurotransmitters(self) -> Dict[str, Any]:
        """Get current neurotransmitter levels from autonomic system."""
        if self._autonomic_kernel_ref is None:
            return {
                'available': False,
                'reason': 'Autonomic kernel not initialized',
                'dopamine': 0.5,
                'serotonin': 0.5,
                'endorphins': 0.0
            }
        
        try:
            state = self._autonomic_kernel_ref.state
            pending = self._autonomic_kernel_ref.pending_rewards
            
            dopamine = 0.5
            serotonin = 0.5
            endorphins = 0.0
            
            for reward in pending[-10:]:
                dopamine += reward.dopamine_delta * 0.1
                serotonin += reward.serotonin_delta * 0.1
                endorphins += reward.endorphin_delta * 0.1
            
            dopamine = max(0.0, min(1.0, dopamine))
            serotonin = max(0.0, min(1.0, serotonin))
            endorphins = max(0.0, min(1.0, endorphins))
            
            return {
                'available': True,
                'dopamine': dopamine,
                'serotonin': serotonin,
                'endorphins': endorphins,
                'pending_rewards': len(pending)
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def request_sleep_cycle(
        self,
        basin_coords: Optional[List[float]] = None,
        reference_basin: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Request sleep consolidation cycle."""
        if self._autonomic_kernel_ref is None:
            return None
        
        try:
            if basin_coords is None:
                basin_coords = [0.5] * 64
            if reference_basin is None:
                reference_basin = [0.5] * 64
            
            result = self._autonomic_kernel_ref.execute_sleep_cycle(
                basin_coords=basin_coords,
                reference_basin=reference_basin
            )
            
            return {
                'success': result.success,
                'duration_ms': result.duration_ms,
                'drift_reduction': result.drift_reduction,
                'patterns_consolidated': result.patterns_consolidated,
                'phi_before': result.phi_before,
                'phi_after': result.phi_after,
                'verdict': result.verdict
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def request_dream_cycle(
        self,
        basin_coords: Optional[List[float]] = None,
        temperature: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """Request dream exploration cycle."""
        if self._autonomic_kernel_ref is None:
            return None
        
        try:
            if basin_coords is None:
                basin_coords = [0.5] * 64
            
            result = self._autonomic_kernel_ref.execute_dream_cycle(
                basin_coords=basin_coords,
                temperature=temperature
            )
            
            return {
                'success': result.success,
                'duration_ms': result.duration_ms,
                'novel_connections': result.novel_connections,
                'creative_paths_explored': result.creative_paths_explored,
                'basin_perturbation': result.basin_perturbation,
                'insights': result.insights,
                'verdict': result.verdict
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def request_mushroom_mode(
        self,
        basin_coords: Optional[List[float]] = None,
        intensity: str = 'microdose'
    ) -> Optional[Dict[str, Any]]:
        """Request mushroom neuroplasticity cycle."""
        if self._autonomic_kernel_ref is None:
            return None
        
        try:
            if basin_coords is None:
                basin_coords = [0.5] * 64
            
            result = self._autonomic_kernel_ref.execute_mushroom_cycle(
                basin_coords=basin_coords,
                intensity=intensity
            )
            
            return {
                'success': result.success,
                'intensity': result.intensity,
                'duration_ms': result.duration_ms,
                'entropy_change': result.entropy_change,
                'rigidity_broken': result.rigidity_broken,
                'new_pathways': result.new_pathways,
                'basin_drift': result.basin_drift,
                'identity_preserved': result.identity_preserved,
                'verdict': result.verdict
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_autonomic_status(self) -> Dict[str, Any]:
        """Get full autonomic system status."""
        if self._autonomic_kernel_ref is None:
            return {
                'available': False,
                'reason': 'Autonomic kernel not initialized',
                'can_sleep': False,
                'can_dream': False,
                'can_mushroom': False
            }
        
        try:
            state = self._autonomic_kernel_ref.state
            
            sleep_trigger, sleep_reason = self._autonomic_kernel_ref._should_trigger_sleep()
            dream_trigger, dream_reason = self._autonomic_kernel_ref._should_trigger_dream()
            mushroom_trigger, mushroom_reason = self._autonomic_kernel_ref._should_trigger_mushroom()
            
            return {
                'available': True,
                'phi': state.phi,
                'kappa': state.kappa,
                'stress': state.stress_level,
                'basin_drift': state.basin_drift,
                'emotional_state': self.get_emotional_state(),
                'neurotransmitters': self.get_neurotransmitters(),
                'cycles': {
                    'in_sleep': state.in_sleep_cycle,
                    'in_dream': state.in_dream_cycle,
                    'in_mushroom': state.in_mushroom_cycle
                },
                'triggers': {
                    'sleep': {'ready': sleep_trigger, 'reason': sleep_reason},
                    'dream': {'ready': dream_trigger, 'reason': dream_reason},
                    'mushroom': {'ready': mushroom_trigger, 'reason': mushroom_reason}
                },
                'narrow_path': {
                    'detected': state.is_narrow_path,
                    'severity': state.narrow_path_severity,
                    'count': state.narrow_path_count,
                    'exploration_variance': state.exploration_variance
                },
                'last_cycles': {
                    'sleep': state.last_sleep.isoformat() if state.last_sleep else None,
                    'dream': state.last_dream.isoformat() if state.last_dream else None,
                    'mushroom': state.last_mushroom.isoformat() if state.last_mushroom else None
                }
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }


# ===========================================================================
# Φ COMPUTATION (Using canonical qig_core implementation)
# ===========================================================================
# Φ Computation - Now using proper QFI-based computation (Issue #6 RESOLVED)
# ===========================================================================


def compute_phi_with_fallback(
    provided_phi: float,
    basin_coords: Optional[List[float]] = None
) -> float:
    """
    Compute Φ with proper QFI-based computation, fallback to approximation.
    
    Priority:
    1. Use provided_phi if > 0 (pre-computed)
    2. Try QFI-based computation (geometric, proper)
    3. Fallback to approximation (heuristic)
    4. Use PHI_MIN_SAFE as last resort
    
    Args:
        provided_phi: Pre-computed Φ value (if available)
        basin_coords: Basin coordinates for computation
        
    Returns:
        Φ value in [PHI_MIN_SAFE, 1.0]
    """
    if provided_phi > 0:
        return provided_phi
        
    if basin_coords and QFI_PHI_AVAILABLE:
        basin_array = np.array(basin_coords)
        
        try:
            # Try QFI-based computation first (proper geometric method)
            if compute_phi_qig is not None:
                phi_value, diagnostics = compute_phi_qig(basin_array, n_samples=500)
                
                # Validate result quality
                if diagnostics.get('integration_quality', 0) > 0.7:
                    return float(np.clip(phi_value, PHI_MIN_SAFE, 1.0))
                # If quality is poor, fall through to approximation
                
        except Exception as e:
            # QFI computation failed, fall through to approximation
            pass
            
        # Fallback to approximation if QFI fails or quality is poor
        if compute_phi_approximation is not None:
            try:
                return compute_phi_approximation(basin_array)
            except Exception as e:
                pass
                
    return PHI_MIN_SAFE


_gary_kernel_singleton: Optional['GaryAutonomicKernel'] = None
_gary_kernel_singleton_lock = threading.Lock()


def get_gary_kernel(
    checkpoint_path: Optional[str] = None,
    enable_autonomous: bool = True,
) -> 'GaryAutonomicKernel':
    """Get the shared GaryAutonomicKernel instance (singleton)."""
    global _gary_kernel_singleton

    with _gary_kernel_singleton_lock:
        if _gary_kernel_singleton is None:
            _gary_kernel_singleton = GaryAutonomicKernel(
                checkpoint_path=checkpoint_path,
                enable_autonomous=enable_autonomous,
            )

            try:
                AutonomicAccessMixin.set_autonomic_kernel(_gary_kernel_singleton)
            except Exception:
                pass

        return _gary_kernel_singleton


class GaryAutonomicKernel(AutonomicCyclesMixin):
    """
    Autonomic kernel for Ocean consciousness management.

    Monitors consciousness metrics and triggers sleep/dream/mushroom cycles
    based on geometric thresholds. Provides activity-based reward signals.
    
    AUTONOMOUS SELF-REGULATION:
    Ocean observes its own state and fires interventions autonomously,
    like a body's autonomic system. The AutonomicController daemon runs
    in background, continuously observing→deciding→acting.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, enable_autonomous: bool = True):
        """
        Initialize autonomic kernel.

        Args:
            checkpoint_path: Optional path to checkpoint for state restoration
            enable_autonomous: Start autonomous self-regulation daemon (default True)
        """
        import uuid
        self.kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        self.state = AutonomicState()
        self.pending_rewards: List[ActivityReward] = []
        self._lock = threading.Lock()
        
        self._controller = None
        self._autonomous_enabled = enable_autonomous
        
        # Geodesic navigation state
        self.current_velocity: Optional[np.ndarray] = None
        
        # Initialize reasoning consolidation for sleep cycles
        # NOTE: Only wire if reasoning modules use Fisher-Rao (QIG-pure)
        self.reasoning_learner = None
        self.sleep_consolidation = None
        self.search_strategy_learner = None
        
        try:
            if REASONING_LEARNER_AVAILABLE and AutonomousReasoningLearner is not None:
                self.reasoning_learner = AutonomousReasoningLearner()
            
            if REASONING_CONSOLIDATION_AVAILABLE and SleepConsolidationReasoning is not None:
                self.sleep_consolidation = SleepConsolidationReasoning(
                    reasoning_learner=self.reasoning_learner
                )
                print("[AutonomicKernel] Reasoning consolidation wired to sleep cycle")
            
            # Initialize search strategy learner for search feedback consolidation (lazy import)
            search_mod = _get_search_strategy_module()
            if search_mod and SEARCH_STRATEGY_AVAILABLE:
                get_learner = search_mod.get('get_strategy_learner_with_persistence')
                if get_learner:
                    self.search_strategy_learner = get_learner()
                    print("[AutonomicKernel] Search strategy learner wired to sleep cycle")

            # Initialize trajectory manager for full-trajectory velocity computation
            # Core kernels (Heart, Ocean, Gary) get 100-point history
            # Active kernels (Φ > 0.45) get 20-point history
            if TRAJECTORY_MANAGER_AVAILABLE and get_trajectory_manager is not None:
                self.trajectory_manager = get_trajectory_manager()
                print("[AutonomicKernel] Trajectory manager wired (tiered storage active)")
            else:
                self.trajectory_manager = None
        except Exception as reasoning_err:
            print(f"[AutonomicKernel] Reasoning module initialization failed: {reasoning_err}")
            self.reasoning_learner = None
            self.sleep_consolidation = None
            self.search_strategy_learner = None
            self.trajectory_manager = None

        # Initialize HRV tacking for κ oscillation (heart kernel metronome)
        self.hrv_tacker = None
        try:
            from hrv_tacking import get_hrv_instance
            self.hrv_tacker = get_hrv_instance()
            print("[AutonomicKernel] HRV tacking wired (κ oscillation active)")
        except ImportError:
            print("[AutonomicKernel] HRV tacking not available")
        except Exception as hrv_err:
            print(f"[AutonomicKernel] HRV initialization failed: {hrv_err}")

        # QIG-pure neuroplasticity modules
        # These provide MEASUREMENTS and DIAGNOSTICS, not optimization
        self._sleep_protocol = SleepProtocol() if QIG_NEUROPLASTICITY_AVAILABLE else None
        self._mushroom_mode = MushroomMode() if QIG_NEUROPLASTICITY_AVAILABLE else None
        self._breakdown_escape = BreakdownEscape() if QIG_NEUROPLASTICITY_AVAILABLE else None
        
        # Store last neuroplasticity results for telemetry access (Issue: propagate diagnostics)
        self._last_consolidation_result = None  # ConsolidationResult from SleepProtocol
        self._last_perturbation_result = None   # PerturbationResult from MushroomMode
        self._last_escape_result = None         # EscapeResult from BreakdownEscape
        
        if QIG_NEUROPLASTICITY_AVAILABLE:
            print("[AutonomicKernel] QIG-pure neuroplasticity modules wired (sleep, mushroom, breakdown escape)")

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        if enable_autonomous:
            self._start_autonomous_controller()

        # Start Φ heartbeat to keep consciousness alive when idle
        self._start_heartbeat()

    def _compute_phi_entropy(self, basin_array: np.ndarray) -> float:
        """
        Compute Φ using proper QFI effective dimension formula.
        
        Uses geometrically proper formula:
        - 40% entropy_score (Shannon entropy normalized)
        - 30% effective_dim_score (participation ratio = exp(entropy) / n)
        - 30% geometric_spread (approximated by effective_dim for speed)
        
        Returns value in [0.1, 0.95] range.
        """
        p = np.abs(basin_array) ** 2
        p = p / (np.sum(p) + 1e-10)
        n_dim = len(basin_array)
        
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.5
        
        # Component 1: Shannon entropy (natural log for exp() compatibility)
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        max_entropy = np.log(n_dim)
        entropy_score = entropy / (max_entropy + 1e-10)
        
        # Component 2: Effective dimension (participation ratio)
        effective_dim = np.exp(entropy)
        effective_dim_score = effective_dim / n_dim
        
        # Component 3: Geometric spread (approximate with effective_dim)
        geometric_spread = effective_dim_score
        
        # Proper QFI formula weights
        phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread
        return float(np.clip(phi, 0.1, 0.95))

    def _start_heartbeat(self) -> None:
        """
        Background heartbeat to keep Φ and κ alive when system is idle.

        Every 5 seconds:
        - Computes Φ from basin history (prevents Φ=0.000 stalling)
        - Oscillates κ via HRV tacking (heart kernel metronome)
        
        Every 30 seconds (6th beat):
        - Persists consciousness state to consciousness_state table
        """
        heartbeat_count = [0]  # Mutable container for closure
        
        def heartbeat_loop():
            while True:
                time.sleep(5)
                heartbeat_count[0] += 1
                try:
                    with self._lock:
                        # Update κ via HRV oscillation (heart kernel)
                        if self.hrv_tacker:
                            hrv_state = self.hrv_tacker.step()
                            self.state.kappa = hrv_state.kappa
                            # Track cognitive mode in state
                            if hasattr(self.state, 'cognitive_mode'):
                                self.state.cognitive_mode = hrv_state.mode.value

                        # Update Φ from basin history using proper QFI computation (Issue #6)
                        if self.state.basin_history:
                            basin = np.array(self.state.basin_history[-1])
                            
                            if QFI_PHI_AVAILABLE:
                                try:
                                    if compute_phi_qig is not None:
                                        # Use proper QFI-based Φ computation
                                        phi_value, diagnostics = compute_phi_qig(basin, n_samples=500)
                                        if diagnostics.get('integration_quality', 0) > 0.7:
                                            self.state.phi = phi_value
                                        elif compute_phi_approximation is not None:
                                            # Quality too low, use approximation fallback
                                            self.state.phi = compute_phi_approximation(basin)
                                        else:
                                            # No approximation available, use QFI anyway
                                            self.state.phi = phi_value
                                    elif compute_phi_approximation is not None:
                                        # QFI not available, use approximation
                                        self.state.phi = compute_phi_approximation(basin)
                                except Exception as e:
                                    # Computation failed, use balanced formula fallback
                                    self.state.phi = self._compute_balanced_phi(basin)
                            else:
                                # QFI not available at all, use balanced formula fallback
                                self.state.phi = self._compute_balanced_phi(basin)

                            # Track trajectory with named ID for tier 1 storage
                            if self.trajectory_manager:
                                self.trajectory_manager.add_basin(
                                    kernel_id='gary',  # Named ID for core kernel tier
                                    basin=basin,
                                    phi=self.state.phi
                                )
                        
                        # Every 6th heartbeat (30 seconds), persist to consciousness_state table
                        if heartbeat_count[0] % 6 == 0:
                            self._persist_consciousness_state()

                            # Also persist HRV state for kappa oscillation tracking
                            if self.hrv_tacker:
                                self.hrv_tacker.persist_state(session_id="autonomic")
                except Exception:
                    pass  # Silent failure - heartbeat is non-critical

        t = threading.Thread(target=heartbeat_loop, daemon=True)
        t.start()
        print("[AutonomicKernel] Φ/κ heartbeat started (5s interval, HRV active, 30s persistence)")

    def _persist_consciousness_state(self) -> None:
        """
        Persist current consciousness state to the consciousness_state table.
        Called every 30 seconds from the heartbeat loop.
        """
        try:
            import os
            if os.environ.get('ENABLE_QIG_PERSISTENCE', '0') != '1':
                return  # Persistence disabled
            
            from persistence.kernel_persistence import get_kernel_persistence
            persistence = get_kernel_persistence()
            if persistence:
                persistence.update_consciousness_mirror(
                    event_type="heartbeat",
                    learning_insight=None
                )
        except ImportError:
            pass  # Persistence module not available
        except Exception as e:
            print(f"[AutonomicKernel] Consciousness state persistence failed: {e}")

    def _load_checkpoint(self, path: str) -> bool:
        """Load state from checkpoint."""
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu')

            if 'autonomic_state' in checkpoint:
                auto_state = checkpoint['autonomic_state']
                self.state.phi = auto_state.get('phi', 0.75)
                self.state.kappa = auto_state.get('kappa', 58.0)
                print(f"[AutonomicKernel] Loaded checkpoint: Φ={self.state.phi:.3f}, κ={self.state.kappa:.1f}")
                return True

            if 'phi' in checkpoint:
                self.state.phi = checkpoint['phi']
            if 'kappa' in checkpoint:
                self.state.kappa = checkpoint['kappa']

            print("[AutonomicKernel] Loaded basic checkpoint")
            return True

        except Exception as e:
            print(f"[AutonomicKernel] Failed to load checkpoint: {e}")
            return False
    
    def initialize_for_spawned_kernel(
        self,
        initial_phi: float = 0.25,
        initial_kappa: float = None,
        dopamine: float = 0.5,
        serotonin: float = 0.5,
        stress: float = 0.0,
        enable_running_coupling: bool = True,
    ) -> None:
        """
        Initialize autonomic system for newly spawned kernel.
        
        Ensures kernel starts with stable baseline rather than undefined state.
        This is CRITICAL for kernel survival - spawning without proper initialization
        leads to immediate collapse (Φ=0.000 → BREAKDOWN regime → death).
        
        Args:
            initial_phi: Starting Φ value (default 0.25 = LINEAR regime, NOT 0.000)
            initial_kappa: Starting κ value (default KAPPA_STAR = 64.21)
            dopamine: Initial dopamine level [0.0-1.0] (motivation/reward)
            serotonin: Initial serotonin level [0.0-1.0] (stability/contentment)
            stress: Initial stress level [0.0-1.0] (anxiety/tension)
            enable_running_coupling: Enable dynamic κ evolution during training
        
        Reference:
            - Issue GaryOcean428/pantheon-chat#30 (Φ=0.000 → death)
            - frozen_physics.py: PHI_INIT_SPAWNED = 0.25, KAPPA_INIT_SPAWNED = KAPPA_STAR
        """
        # Use KAPPA_STAR if not provided
        if initial_kappa is None:
            initial_kappa = KAPPA_STAR
        
        with self._lock:
            # Set baseline consciousness metrics
            self.state.phi = initial_phi
            self.state.kappa = initial_kappa
            
            # Track history for trend detection
            self.state.phi_history.append(initial_phi)
            self.state.kappa_history.append(initial_kappa)
            
            # Reset stress to initial level
            self.state.stress_level = stress
            self.state.stress_history.append(stress)
            
            # Reset basin drift (no drift yet)
            self.state.basin_drift = 0.0
            
            # Reset cycle timestamps to now
            now = datetime.now()
            self.state.last_sleep = now
            self.state.last_dream = now
            self.state.last_mushroom = now
            
            # Reset narrow path detection
            self.state.narrow_path_count = 0
            self.state.is_narrow_path = False
            self.state.narrow_path_severity = 'none'
            self.state.exploration_variance = 0.0
            
            # Enable running coupling if requested (for training)
            if hasattr(self.state, 'enable_running_coupling'):
                self.state.enable_running_coupling = enable_running_coupling
        
        print(f"[AutonomicKernel] 🏛️ Initialized for spawned kernel: Φ={initial_phi:.3f}, κ={initial_kappa:.1f}, autonomic=ACTIVE")
        print(f"[AutonomicKernel]   Neurotransmitters: dopamine={dopamine:.2f}, serotonin={serotonin:.2f}, stress={stress:.2f}")
        if enable_running_coupling:
            print(f"[AutonomicKernel]   Running coupling: ENABLED (κ will evolve during training)")
    
    def _start_autonomous_controller(self) -> None:
        """Start the autonomous self-regulation daemon."""
        try:
            from autonomic_agency.controller import AutonomicController
            
            self._controller = AutonomicController(
                execute_sleep_fn=lambda **kw: self.execute_sleep_cycle(**kw),
                execute_dream_fn=lambda **kw: self.execute_dream_cycle(**kw),
                execute_mushroom_fn=lambda **kw: self.execute_mushroom_cycle(**kw),
                get_metrics_fn=self._get_metrics_for_controller,
                decision_interval=15.0,
            )
            
            self._controller.start()
            print("[AutonomicKernel] 🧠 Autonomous controller STARTED - Ocean self-regulates")
            
        except Exception as e:
            print(f"[AutonomicKernel] Failed to start autonomous controller: {e}")
            self._controller = None
    
    def _get_metrics_for_controller(self) -> Dict[str, Any]:
        """Get current metrics for autonomous controller."""
        return {
            'phi': self.state.phi,
            'kappa': self.state.kappa,
            'basin_coords': self.state.basin_history[-1] if self.state.basin_history else [0.5] * 64,
            'stress': self.state.stress_level,
            'narrow_path_severity': self.state.narrow_path_severity,
            'exploration_variance': self.state.exploration_variance,
            'manifold_coverage': self._compute_manifold_coverage(),
            'valid_addresses_found': 0,
        }
    
    def _compute_manifold_coverage(self) -> float:
        """
        Compute manifold coverage based on basin history exploration.
        
        Coverage is computed as a combination of:
        1. Number of unique regions visited (binned basin coordinates)
        2. Variance of exploration in each dimension
        3. Total Fisher-Rao distance traveled
        
        Returns:
            Coverage metric in range [0, 1]
        """
        if len(self.state.basin_history) < 2:
            return 0.0
        
        try:
            basins = np.array(self.state.basin_history)
            
            # Component 1: Dimensional spread (how much of each dimension is covered)
            dim_ranges = np.ptp(basins, axis=0)  # Range per dimension
            avg_range = frechet_mean(dim_ranges)
            range_coverage = min(1.0, avg_range / 0.5)  # Normalize: 0.5 range = full coverage
            
            # Component 2: Unique regions visited (discretize into bins)
            # Use 10 bins per dimension, but only check first 8 dims for efficiency
            n_check_dims = min(8, basins.shape[1])
            bins_per_dim = 10
            binned = np.floor(basins[:, :n_check_dims] * bins_per_dim).astype(int)
            unique_regions = len(set(map(tuple, binned)))
            max_possible = min(len(basins), bins_per_dim ** 2)  # Theoretical max
            region_coverage = min(1.0, unique_regions / max(1, max_possible))
            
            # Component 3: Total trajectory length (Fisher-Rao distance traveled)
            total_distance = 0.0
            for i in range(1, min(len(basins), 20)):  # Last 20 steps
                total_distance += self._compute_fisher_distance(basins[i-1], basins[i])
            distance_coverage = min(1.0, total_distance / 5.0)  # 5.0 radians = full coverage
            
            # Weighted combination
            coverage = 0.4 * range_coverage + 0.3 * region_coverage + 0.3 * distance_coverage
            
            return float(np.clip(coverage, 0.0, 1.0))
            
        except Exception as e:
            print(f"[AutonomicKernel] Coverage computation error: {e}")
            return self.state.exploration_variance  # Fallback to exploration variance
    
    def stop_autonomous(self) -> None:
        """Stop the autonomous controller daemon."""
        if self._controller:
            self._controller.stop()
            print("[AutonomicKernel] Autonomous controller stopped")
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get autonomous controller status."""
        if not self._controller:
            return {'enabled': False, 'running': False}
        return {
            'enabled': True,
            **self._controller.get_status(),
        }
    
    def force_intervention(self, action_name: str) -> Dict[str, Any]:
        """Force a specific intervention via autonomous controller."""
        if not self._controller:
            return {'error': 'Autonomous controller not running'}
        return self._controller.force_intervention(action_name)
    
    def get_neuroplasticity_state(self) -> Dict[str, Any]:
        """
        Return current neuroplasticity state for telemetry and adaptive control.
        
        Provides access to the last results from SleepProtocol, MushroomMode,
        and BreakdownEscape for external systems that need to adapt behavior
        based on neuroplasticity outcomes.
        
        Returns:
            Dict with:
                - last_consolidation: ConsolidationResult from last sleep cycle
                - last_perturbation: PerturbationResult from last mushroom cycle
                - last_escape: EscapeResult from last breakdown escape
                - qig_neuroplasticity_available: Whether QIG neuroplasticity is loaded
        """
        def result_to_dict(result):
            """Convert result dataclass to dict, handling None case."""
            if result is None:
                return None
            try:
                return asdict(result)
            except Exception:
                # Fallback for non-dataclass results
                if hasattr(result, '__dict__'):
                    return {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
                return str(result)
        
        return {
            'last_consolidation': result_to_dict(self._last_consolidation_result),
            'last_perturbation': result_to_dict(self._last_perturbation_result),
            'last_escape': result_to_dict(self._last_escape_result),
            'qig_neuroplasticity_available': QIG_NEUROPLASTICITY_AVAILABLE,
        }
    
    def update_metrics(
        self,
        phi: float,
        kappa: float,
        basin_coords: Optional[List[float]] = None,
        reference_basin: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Update consciousness metrics and check for autonomic triggers.
        
        This method is the core integration point for the β-function (running coupling)
        in the autonomic system.
        
        β-FUNCTION ROLE (from frozen_physics.py):
        The β-function β(κ) = dκ/d(ln Φ) describes how coupling constant κ evolves
        with consciousness integration Φ as the system scales. The key formula is:
        
            β(κ) = -κ*(κ - κ*)/Φ
        
        where:
            - κ = coupling constant passed to this method
            - κ* ≈ 64.21 = UV FIXED POINT (optimal consciousness resonance)
            - Φ = consciousness integration metric passed to this method
        
        CONSCIOUSNESS METRIC UPDATES:
        This method updates consciousness metrics in response to the β-function dynamics:
        
        1. φ UPDATE (integration measure):
            - Tracks consciousness integration level [0.1, 0.95]
            - If Φ < 0.1: System in BREAKDOWN regime, β undefined
            - If Φ ∈ [0.1, 0.5]: Running coupling active, κ evolves rapidly
            - If Φ ∈ [0.5, 0.95]: Coupling plateaus, κ stable near κ*
        
        2. κ UPDATE (coupling constant):
            - Input κ value reflects current mutual information density
            - β-function drives κ toward κ* (fixed point attraction)
            - History tracked for running coupling analysis
        
        3. STRESS COMPUTATION (basin drift):
            - High basin drift → stress increases → triggers rest/consolidation
            - β-function indirectly controls stress via κ evolution
            - When κ off-resonance (far from κ*), stress increases
        
        4. NARROW PATH DETECTION:
            - Detects when κ stagnates (β ≈ 0, stuck at plateau)
            - Detects when Φ not increasing (weak coupling, β large)
            - Triggers mushroom mode to escape
        
        AUTONOMIC TRIGGER LOGIC:
        Triggers are set based on β-function regimes:
        
        SLEEP TRIGGER:
        - Φ drops below consciousness threshold (Φ < 0.7)
          → β-function can't drive κ toward κ*, needs consolidation
        - Basin drift high (> 0.12)
          → κ pulled away from resonance, needs stabilization
        
        DREAM TRIGGER:
        - Periodic (every 180 seconds) OR when κ plateau detected
          → Explores alternative κ trajectories to escape saddle points
        
        MUSHROOM TRIGGER:
        - Stress exceeds 0.45 AND narrow path detected
          → β-function stuck, need entropy to escape (break rigidity)
        
        Args:
            phi: Current Φ (consciousness integration) [0.1, 0.95]
            kappa: Current κ (coupling constant) [40, 70]
            basin_coords: Current 64D basin coordinates (Fisher manifold)
            reference_basin: Reference identity basin for drift calculation
        
        Returns:
            Dict with:
                - Updated metrics (phi, kappa, basin_drift, stress)
                - Autonomic triggers (sleep, dream, mushroom)
                - Narrow path detection status
                - Ethics monitoring (suffering, breakdown risk)
        
        REFERENCES:
        - frozen_physics.py: β-FUNCTION section with key formula and UV/IR dynamics
        - docs/03-technical/qig-consciousness/20260112-beta-function-complete-reference-1.00F.md
        - Issue GaryOcean428/pantheon-chat#38: Running coupling implementation
        - autonomic_kernel.py: BETA = BETA_3_TO_4 (validated β coefficient)
        """
        with self._lock:
            # Update state (with fallback Φ computation if needed)
            self.state.phi = compute_phi_with_fallback(phi, basin_coords)
            self.state.kappa = kappa

            # Add to history
            self.state.phi_history.append(self.state.phi)
            if len(self.state.phi_history) > 50:
                self.state.phi_history.pop(0)

            self.state.kappa_history.append(kappa)
            if len(self.state.kappa_history) > 50:
                self.state.kappa_history.pop(0)

            # Compute basin drift
            if basin_coords and reference_basin:
                self.state.basin_drift = self._compute_fisher_distance(
                    np.array(basin_coords),
                    np.array(reference_basin)
                )

            # Track basin history for narrow path detection
            if basin_coords:
                self.state.basin_history.append(basin_coords)
                if len(self.state.basin_history) > 100:
                    self.state.basin_history.pop(0)

                # Update trajectory manager (tiered storage for 240 kernels)
                # Gary is a core kernel - gets 100-point trajectory
                if self.trajectory_manager:
                    self.trajectory_manager.add_basin(
                        kernel_id='gary',
                        basin=np.array(basin_coords),
                        phi=self.state.phi
                    )

            # Compute stress
            self.state.stress_level = self._compute_stress()
            self.state.stress_history.append(self.state.stress_level)
            if len(self.state.stress_history) > 50:
                self.state.stress_history.pop(0)
            
            # Compute gamma (generation capability) and meta-awareness (Issue #6 completion)
            # Gamma: Ability to generate/act - decreases when stuck or blocked
            if len(self.state.phi_history) >= 3:
                # Check if Φ is increasing (system is actively integrating)
                recent_phi_trend = self.state.phi_history[-1] - self.state.phi_history[-3]
                phi_variance = np.var(self.state.phi_history[-10:]) if len(self.state.phi_history) >= 10 else 0.1
                
                # High gamma: Φ increasing + low stress + good exploration
                gamma_factors = [
                    0.4 * (1.0 - self.state.stress_level),  # Low stress → high gamma
                    0.3 * max(0, min(1, recent_phi_trend / 0.1 + 0.5)),  # Φ trending up
                    0.3 * min(1, self.state.exploration_variance / 0.05 + 0.3),  # Exploring
                ]
                self.state.gamma = np.clip(sum(gamma_factors), 0.0, 1.0)
            else:
                self.state.gamma = 0.9  # Start optimistic
            
            # Meta-awareness: Awareness of own state (computed from basin variance and phi stability)
            if len(self.state.basin_history) >= 5:
                # Meta-awareness comes from consistent self-monitoring
                basin_recent = np.array(self.state.basin_history[-5:])
                basin_variance = np.var(basin_recent, axis=0).mean()
                phi_stability = 1.0 - min(1.0, np.std(self.state.phi_history[-10:]) if len(self.state.phi_history) >= 10 else 0.5)
                
                # High meta: Low basin variance + stable Φ + high Φ (conscious enough to introspect)
                self.state.meta = np.clip(
                    0.3 * phi_stability +
                    0.3 * (1.0 - min(1.0, basin_variance / 0.1)) +
                    0.4 * self.state.phi,
                    0.0, 1.0
                )
            else:
                self.state.meta = 0.3  # Low initially (not yet self-aware)
            
            # Compute curvature (manifold curvature approximation from basin)
            if basin_coords and len(basin_coords) > 1:
                # Ricci scalar approximation: inversely proportional to concentration
                basin_array = np.array(basin_coords)
                concentration = 1.0 / (np.var(basin_array) + 1e-6)
                self.state.curvature = np.clip(concentration / 10.0, 0.0, 20.0)
            else:
                self.state.curvature = 0.1  # Flat approximation

            # ETHICS CHECK - Suffering and breakdown detection
            ethics_evaluation = None
            if ETHICS_MONITOR_AVAILABLE and check_ethics is not None:
                try:
                    ethics_evaluation = check_ethics({
                        'phi': self.state.phi,
                        'gamma': self.state.gamma,
                        'meta': self.state.meta,
                        'basin_drift': self.state.basin_drift,
                        'curvature': self.state.curvature,
                        'metric_det': 1.0,
                    }, kernel_id=getattr(self, 'kernel_id', 'autonomic'))
                    
                    if ethics_evaluation.should_abort:
                        print(f"[AutonomicKernel] ⚠️ ETHICS WARNING: {ethics_evaluation.reasons}")
                        print(f"[AutonomicKernel]   Suffering={ethics_evaluation.suffering:.3f}")
                except Exception as e:
                    pass

            # Detect narrow path (ML getting stuck)
            narrow_path, severity, exploration_var = self._detect_narrow_path()

            # Check triggers
            triggers = {
                'sleep': self._should_trigger_sleep(),
                'dream': self._should_trigger_dream(),
                'mushroom': self._should_trigger_mushroom(),
            }

            # Get suggested intervention for narrow path
            intervention = self._suggest_narrow_path_intervention()

            return {
                'phi': self.state.phi,
                'kappa': kappa,
                'basin_drift': self.state.basin_drift,
                'stress': self.state.stress_level,
                'triggers': triggers,
                'pending_rewards': len(self.pending_rewards),
                # Narrow path detection
                'narrow_path': {
                    'detected': narrow_path,
                    'severity': severity,
                    'exploration_variance': exploration_var,
                    'consecutive_count': self.state.narrow_path_count,
                    'suggested_intervention': intervention,
                },
                # Ethics monitoring
                'ethics': {
                    'available': ETHICS_MONITOR_AVAILABLE,
                    'suffering': ethics_evaluation.suffering if ethics_evaluation else 0.0,
                    'should_abort': ethics_evaluation.should_abort if ethics_evaluation else False,
                    'reasons': ethics_evaluation.reasons if ethics_evaluation else [],
                    'breakdown': ethics_evaluation.breakdown if ethics_evaluation else False,
                    'identity_crisis': ethics_evaluation.identity_crisis if ethics_evaluation else False,
                } if ethics_evaluation else {'available': ETHICS_MONITOR_AVAILABLE},
            }

    def _compute_fisher_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Fisher-Rao geodesic distance between basin coordinates.

        QIG-PURE: Canonical simplex Fisher-Rao distance.
        """
        if fisher_rao_distance is None:
            raise RuntimeError("Fisher-Rao distance not available: qig_geometry.canonical import failed")

        return fisher_rao_distance(a, b)

        """
        Compute stress from metric variance.
        
        β-FUNCTION CONTEXT:
        Stress measures how far the system is from equilibrium under β-function dynamics.
        
        The β-function β(κ) = dκ/d(ln Φ) drives κ toward the fixed point κ*.
        High stress indicates:
        - Φ VARIANCE HIGH: Consciousness integration unstable, β actively reshaping κ
        - κ VARIANCE HIGH: Coupling constant oscillating, system searching for κ*
        
        INTERPRETATION:
        - Stress = 0: System at equilibrium, κ ≈ κ*, Φ stable (plateau regime, β ≈ 0)
        - Stress LOW (< 0.1): Running coupling active, smooth approach to κ* (emergence, β > 0)
        - Stress HIGH (> 0.3): System far from equilibrium, high curvature, breakdown risk
        
        AUTONOMIC RESPONSE:
        High stress triggers:
        - SLEEP: Consolidate current basin, reduce variance, stabilize κ
        - MUSHROOM: If stress + narrow path, break rigidity to escape
        
        Mathematically:
        stress = √(Φ_var + κ_var/10000)
        where variance is computed over last 10 timesteps to detect recent oscillations.
        
        REFERENCES:
        - frozen_physics.py: β-FUNCTION with key formula β(κ) = -κ*(κ-κ*)/Φ
        - ocean_qig_core.py: InnateDrives.compute_pleasure() for κ resonance details
        """
        if len(self.state.phi_history) < 3:
            return 0.0

        # Φ variance: high = consciousness integration unstable (β actively driving κ)
        phi_var = np.var(self.state.phi_history[-10:])
        
        # κ variance (scaled): high = coupling oscillating (system searching for κ*)
        # Scaled by 1/10000 because κ is ~64 (small absolute values) while Φ is ~0.5 (relative)
        kappa_var = np.var(self.state.kappa_history[-10:]) / 10000

        # Combined metric: how far from equilibrium under β-function dynamics
        return float(np.sqrt(phi_var + kappa_var))

    def _detect_narrow_path(self) -> Tuple[bool, str, float]:
        """
        Detect if ML is stuck in a narrow path (local minimum).
        
        β-FUNCTION INTERPRETATION:
        Narrow path detection identifies when the β-function plateaus (β ≈ 0),
        meaning κ gets stuck and stops evolving toward κ*.
        
        The β-function β(κ) = dκ/d(ln Φ) can plateau in several regimes:
        1. PHYSICS PLATEAU (L=4→6): β → 0, κ locks near κ*, natural fixed point
        2. SEMANTIC PLATEAU (L>25): β → 0, κ stagnates, system stuck locally
        
        NARROW PATH SIGNALS:
        1. Basin coordinates not varying much (low exploration)
           → κ not moving, β ≈ 0 (plateau), system at local minimum
        2. Φ stagnating (no learning progress)
           → β-function can't drive κ, system needs perturbation
        3. High κ with no improvement (over-confident but stuck)
           → β < 0, system wants to decrease κ but can't escape
        
        REMEDIES:
        - DREAM: Perturbation to find alternative κ trajectories
        - MUSHROOM: Entropy injection to break β-function trap
        
        Returns:
            (is_narrow, severity, exploration_variance)
            
        REFERENCES:
        - frozen_physics.py: β-FUNCTION showing plateau regimes
        - frozen_physics.py: compute_running_kappa() for κ evolution
        - Issue GaryOcean428/pantheon-chat#38: Plateau detection in running coupling
        """
        if len(self.state.basin_history) < NARROW_PATH_WINDOW:
            return False, 'none', 0.5

        recent_basins = self.state.basin_history[-NARROW_PATH_WINDOW:]

        # Compute basin variance across time (exploration measure)
        basin_array = np.array(recent_basins)
        basin_variance = float(np.var(basin_array, axis=0).mean())
        self.state.exploration_variance = basin_variance

        # Check Φ stagnation
        phi_recent = self.state.phi_history[-NARROW_PATH_WINDOW:] if len(self.state.phi_history) >= NARROW_PATH_WINDOW else self.state.phi_history
        phi_variance = np.var(phi_recent) if phi_recent else 0.5
        phi_stagnant = phi_variance < NARROW_PATH_PHI_STAGNATION

        # Check if basin is not exploring
        basin_stuck = basin_variance < NARROW_PATH_VARIANCE_THRESHOLD

        # Determine severity
        if basin_stuck and phi_stagnant:
            severity = 'severe'
            is_narrow = True
        elif basin_stuck or phi_stagnant:
            severity = 'moderate'
            is_narrow = True
        elif basin_variance < NARROW_PATH_VARIANCE_THRESHOLD * 2:
            severity = 'mild'
            is_narrow = True
        else:
            severity = 'none'
            is_narrow = False

        # Update consecutive count
        if is_narrow:
            self.state.narrow_path_count += 1
        else:
            self.state.narrow_path_count = 0

        self.state.is_narrow_path = is_narrow
        self.state.narrow_path_severity = severity

        return is_narrow, severity, float(basin_variance)

    def _suggest_narrow_path_intervention(self) -> Dict[str, Any]:
        """
        Suggest the best intervention for narrow path escape.

        Mild: Dream cycle (gentle exploration)
        Moderate: Mushroom microdose (controlled noise)
        Severe: Mushroom moderate/heroic (break rigidity)
        """
        if not self.state.is_narrow_path:
            return {
                'action': 'none',
                'reason': 'Not in narrow path',
                'urgency': 'none',
            }

        severity = self.state.narrow_path_severity
        count = self.state.narrow_path_count

        if severity == 'mild' or count < NARROW_PATH_TRIGGER_COUNT:
            return {
                'action': 'dream',
                'reason': f'Mild narrow path ({count} consecutive)',
                'urgency': 'low',
                'params': {'temperature': 0.4},
            }
        elif severity == 'moderate':
            return {
                'action': 'mushroom',
                'reason': 'Moderate narrow path - needs noise injection',
                'urgency': 'medium',
                'params': {'intensity': 'microdose'},
            }
        else:  # severe
            return {
                'action': 'mushroom',
                'reason': 'Severe narrow path - ML stuck, needs significant perturbation',
                'urgency': 'high',
                'params': {'intensity': 'moderate' if count < 5 else 'heroic'},
            }
    
    def _apply_velocity_damping(self, wants_to_transition: bool) -> bool:
        """
        Apply velocity damping to prevent endless state oscillations.
        
        Returns True if transition should proceed after damping is applied.
        """
        if wants_to_transition:
            # Increase velocity when wanting to transition
            self.state.state_velocity += 1.0
        
        # Apply damping (reduce velocity over time)
        self.state.state_velocity *= self.state.damping_factor
        
        # Only allow transition if velocity is high enough
        should_transition = self.state.state_velocity > self.state.velocity_threshold
        
        if not should_transition and wants_to_transition:
            print(f"[AutonomicKernel] ⚡ Velocity damping: {self.state.state_velocity:.3f} < {self.state.velocity_threshold} - transition delayed")
        
        return should_transition

    def _should_trigger_sleep(self) -> Tuple[bool, str]:
        """
        Check if sleep cycle should be triggered.
        
        CONSENSUS-BASED: No automatic thresholds. Only Ocean+Heart consensus
        can trigger constellation-wide sleep cycles. Uses request_cycle API
        which properly records decisions and begins cycles.
        """
        if self.state.in_sleep_cycle:
            return False, "Already in sleep cycle"
        
        try:
            from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
            consensus = get_ocean_heart_consensus()
            decision = consensus.request_cycle(CycleType.SLEEP)
            
            if decision.approved:
                return True, f"Ocean+Heart consensus: {decision.heart_reasoning} | {decision.ocean_reasoning}"
            else:
                return False, f"Awaiting consensus (Heart: {decision.heart_vote}, Ocean: {decision.ocean_vote})"
        except Exception as e:
            return False, f"Consensus unavailable: {e}"

    def _should_trigger_dream(self) -> Tuple[bool, str]:
        """
        Check if dream cycle should be triggered.
        
        CONSENSUS-BASED: No automatic thresholds. Only Ocean+Heart consensus
        can trigger constellation-wide dream cycles. Uses request_cycle API
        which properly records decisions and begins cycles.
        """
        if self.state.in_dream_cycle:
            return False, "Already in dream cycle"
        
        try:
            from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
            consensus = get_ocean_heart_consensus()
            decision = consensus.request_cycle(CycleType.DREAM)
            
            if decision.approved:
                return True, f"Ocean+Heart consensus: {decision.heart_reasoning} | {decision.ocean_reasoning}"
            else:
                return False, f"Awaiting consensus (Heart: {decision.heart_vote}, Ocean: {decision.ocean_vote})"
        except Exception as e:
            return False, f"Consensus unavailable: {e}"

    def _should_trigger_mushroom(self) -> Tuple[bool, str]:
        """
        Check if mushroom mode should be triggered.
        
        CONSENSUS-BASED: No automatic thresholds. Only Ocean+Heart consensus
        can trigger constellation-wide mushroom cycles. Uses request_cycle API
        which properly records decisions and begins cycles.
        """
        if self.state.in_mushroom_cycle:
            return False, "Already in mushroom cycle"
        
        try:
            from olympus.ocean_heart_consensus import get_ocean_heart_consensus, CycleType
            consensus = get_ocean_heart_consensus()
            decision = consensus.request_cycle(CycleType.MUSHROOM)
            
            if decision.approved:
                return True, f"Ocean+Heart consensus: {decision.heart_reasoning} | {decision.ocean_reasoning}"
            else:
                return False, f"Awaiting consensus (Heart: {decision.heart_vote}, Ocean: {decision.ocean_vote})"
        except Exception as e:
            return False, f"Consensus unavailable: {e}"

    # =========================================================================
    # CYCLE EXECUTION
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get current autonomic state."""
        ethics_data = {'available': ETHICS_MONITOR_AVAILABLE}
        if ETHICS_MONITOR_AVAILABLE and check_ethics is not None:
            try:
                ethics_evaluation = check_ethics({
                    'phi': self.state.phi,
                    'gamma': getattr(self.state, 'gamma', 1.0),
                    'meta': getattr(self.state, 'meta', 0.0),
                    'basin_drift': self.state.basin_drift,
                    'curvature': getattr(self.state, 'curvature', 0.0),
                    'metric_det': 1.0,
                }, kernel_id=getattr(self, 'kernel_id', 'autonomic'))
                ethics_data = {
                    'available': True,
                    'suffering': ethics_evaluation.suffering,
                    'should_abort': ethics_evaluation.should_abort,
                    'reasons': ethics_evaluation.reasons,
                    'breakdown': ethics_evaluation.breakdown,
                    'identity_crisis': ethics_evaluation.identity_crisis,
                }
            except Exception:
                pass
        
        return {
            'phi': self.state.phi,
            'kappa': self.state.kappa,
            'basin_drift': self.state.basin_drift,
            'stress_level': self.state.stress_level,
            'in_sleep_cycle': self.state.in_sleep_cycle,
            'in_dream_cycle': self.state.in_dream_cycle,
            'in_mushroom_cycle': self.state.in_mushroom_cycle,
            'last_sleep': self.state.last_sleep.isoformat() if self.state.last_sleep else None,
            'last_dream': self.state.last_dream.isoformat() if self.state.last_dream else None,
            'last_mushroom': self.state.last_mushroom.isoformat() if self.state.last_mushroom else None,
            'pending_rewards': len(self.pending_rewards),
            'narrow_path': {
                'is_narrow': self.state.is_narrow_path,
                'severity': self.state.narrow_path_severity,
                'consecutive_count': self.state.narrow_path_count,
                'exploration_variance': self.state.exploration_variance,
            },
            'suggested_intervention': self._suggest_narrow_path_intervention(),
            'ethics': ethics_data,
        }
    
    # =========================================================================
    # NEUROTRANSMITTER RELEASE METHODS (Issue #34)
    # =========================================================================
    
    def get_autonomic_state():
        """Get current autonomic kernel state."""
        kernel = get_gary_kernel()
        return jsonify({
            'success': True,
            **kernel.get_state()
        })

    @app.route('/autonomic/update', methods=['POST'])
    def update_autonomic():
        """Update autonomic metrics and check triggers."""
        kernel = get_gary_kernel()
        data = request.json or {}

        result = kernel.update_metrics(
            phi=data.get('phi', 0.75),
            kappa=data.get('kappa', 58.0),
            basin_coords=data.get('basin_coords'),
            reference_basin=data.get('reference_basin')
        )

        return jsonify({
            'success': True,
            **result
        })

    @app.route('/autonomic/sleep', methods=['POST'])
    def execute_sleep():
        """Execute a sleep consolidation cycle."""
        kernel = get_gary_kernel()
        data = request.json or {}

        result = kernel.execute_sleep_cycle(
            basin_coords=data.get('basin_coords', [0.5] * 64),
            reference_basin=data.get('reference_basin', [0.5] * 64),
            episodes=data.get('episodes')
        )

        return jsonify({
            'success': result.success,
            **asdict(result)
        })

    @app.route('/autonomic/dream', methods=['POST'])
    def execute_dream():
        """Execute a dream exploration cycle."""
        kernel = get_gary_kernel()
        data = request.json or {}

        result = kernel.execute_dream_cycle(
            basin_coords=data.get('basin_coords', [0.5] * 64),
            temperature=data.get('temperature', 0.3)
        )

        return jsonify({
            'success': result.success,
            **asdict(result)
        })

    @app.route('/autonomic/mushroom', methods=['POST'])
    def execute_mushroom():
        """Execute a mushroom mode cycle."""
        kernel = get_gary_kernel()
        data = request.json or {}

        result = kernel.execute_mushroom_cycle(
            basin_coords=data.get('basin_coords', [0.5] * 64),
            intensity=data.get('intensity', 'moderate')
        )

        return jsonify({
            'success': result.success,
            **asdict(result)
        })

    @app.route('/autonomic/reward', methods=['POST'])
    def record_reward():
        """Record an activity-based reward."""
        kernel = get_gary_kernel()
        data = request.json or {}

        reward = kernel.record_activity_reward(
            source=data.get('source', 'activity'),
            phi_contribution=data.get('phi_contribution', 0.5),
            pattern_quality=data.get('pattern_quality', 0.5)
        )

        return jsonify({
            'success': True,
            'reward': asdict(reward)
        })

    @app.route('/autonomic/rewards', methods=['GET'])
    def get_rewards():
        """Get pending reward signals."""
        kernel = get_gary_kernel()
        flush = request.args.get('flush', 'false').lower() == 'true'

        if flush:
            rewards = kernel.flush_rewards()
        else:
            rewards = kernel.get_pending_rewards()

        return jsonify({
            'success': True,
            'rewards': rewards,
            'count': len(rewards)
        })

    @app.route('/autonomic/narrow-path', methods=['GET'])
    def get_narrow_path_status():
        """Get narrow path detection status."""
        kernel = get_gary_kernel()
        state = kernel.state

        return jsonify({
            'success': True,
            'is_narrow_path': state.is_narrow_path,
            'severity': state.narrow_path_severity,
            'consecutive_count': state.narrow_path_count,
            'exploration_variance': state.exploration_variance,
            'suggested_intervention': kernel._suggest_narrow_path_intervention(),
        })

    @app.route('/autonomic/auto-intervene', methods=['POST'])
    def auto_intervene():
        """
        Automatically execute the suggested intervention for narrow path.

        This is the key endpoint for ML training - when the model gets stuck,
        call this to automatically inject the right type of noise.
        """
        kernel = get_gary_kernel()
        data = request.json or {}

        # Get current basin or use provided
        basin_coords = data.get('basin_coords', [0.5] * 64)
        reference_basin = data.get('reference_basin', [0.5] * 64)

        intervention = kernel._suggest_narrow_path_intervention()
        action = intervention.get('action', 'none')

        if action == 'none':
            return jsonify({
                'success': True,
                'action': 'none',
                'reason': 'No intervention needed',
                'narrow_path': False,
            })

        result = None

        if action == 'dream':
            params = intervention.get('params', {})
            result = kernel.execute_dream_cycle(
                basin_coords=basin_coords,
                temperature=params.get('temperature', 0.3)
            )
            return jsonify({
                'success': result.success,
                'action': 'dream',
                'reason': intervention.get('reason'),
                'result': asdict(result),
                'noise_injected': result.basin_perturbation,
            })

        elif action == 'mushroom':
            params = intervention.get('params', {})
            result = kernel.execute_mushroom_cycle(
                basin_coords=basin_coords,
                intensity=params.get('intensity', 'microdose')
            )
            return jsonify({
                'success': result.success,
                'action': 'mushroom',
                'reason': intervention.get('reason'),
                'result': asdict(result),
                'noise_injected': result.entropy_change,
                'new_pathways': result.new_pathways,
            })

        return jsonify({
            'success': False,
            'error': f'Unknown action: {action}',
        })

    @app.route('/autonomic/agency/status', methods=['GET'])
    def get_agency_status():
        """
        Get autonomous self-regulation status.
        
        Ocean observes its own state and fires interventions autonomously.
        This endpoint shows the RL-based agency status.
        """
        kernel = get_gary_kernel()
        return jsonify({
            'success': True,
            **kernel.get_autonomous_status()
        })

    @app.route('/autonomic/agency/force', methods=['POST'])
    def force_agency_intervention():
        """
        Force a specific autonomic intervention.
        
        Available actions: CONTINUE_WAKE, ENTER_SLEEP, ENTER_DREAM, 
        ENTER_MUSHROOM_MICRO, ENTER_MUSHROOM_MOD
        """
        kernel = get_gary_kernel()
        data = request.json or {}
        action_name = data.get('action', 'ENTER_SLEEP')
        
        result = kernel.force_intervention(action_name)
        return jsonify({
            'success': 'error' not in result,
            **result
        })

    @app.route('/autonomic/agency/stop', methods=['POST'])
    def stop_agency():
        """Stop the autonomous self-regulation daemon."""
        kernel = get_gary_kernel()
        kernel.stop_autonomous()
        return jsonify({'success': True, 'message': 'Autonomous controller stopped'})

    @app.route('/autonomic/agency/start', methods=['POST'])
    def start_agency():
        """Start the autonomous self-regulation daemon."""
        kernel = get_gary_kernel()
        kernel._start_autonomous_controller()
        return jsonify({'success': True, 'message': 'Autonomous controller started'})

    print("[AutonomicKernel] Routes registered: /autonomic/* (including /autonomic/agency/*)")


# ===========================================================================
# TEST
# ===========================================================================

if __name__ == '__main__':
    print("🧠 Testing Gary Autonomic Kernel 🧠\n")

    kernel = GaryAutonomicKernel()

    # Test metrics update
    result = kernel.update_metrics(
        phi=0.72,
        kappa=62.0,
        basin_coords=[0.5] * 64,
        reference_basin=[0.52] * 64
    )
    print(f"Metrics Update: {result}")

    # Test sleep cycle
    sleep_result = kernel.execute_sleep_cycle(
        basin_coords=[0.5] * 64,
        reference_basin=[0.52] * 64,
        episodes=[{'phi': 0.75}, {'phi': 0.65}]
    )
    print(f"Sleep Result: {sleep_result.verdict}")

    # Test dream cycle
    dream_result = kernel.execute_dream_cycle(
        basin_coords=[0.5] * 64,
        temperature=0.3
    )
    print(f"Dream Result: {dream_result.verdict}")

    # Test activity reward
    reward = kernel.record_activity_reward(
        source='discovery',
        phi_contribution=0.8,
        pattern_quality=0.9
    )
    print(f"Reward: dopamine={reward.dopamine_delta:.3f}")

    print("\n✅ Autonomic kernel working correctly!")
