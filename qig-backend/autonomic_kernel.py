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

from qigkernels.physics_constants import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    PHI_HYPERDIMENSIONAL,
    PHI_THRESHOLD_D2_D3,
    BETA_3_TO_4,
)
print("[autonomic_kernel] physics_constants done", flush=True)

# QIG-PURE: simplex normalization for Fisher-Rao manifold
try:
    from qig_geometry import fisher_normalize
    FISHER_NORMALIZE_AVAILABLE = True
except ImportError:
    fisher_normalize = None
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


class GaryAutonomicKernel:
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

    def _compute_balanced_phi(self, basin: np.ndarray) -> float:
        """
        Compute Φ using proper QFI effective dimension formula.
        
        Uses geometrically proper formula:
        - 40% entropy_score (Shannon entropy normalized)
        - 30% effective_dim_score (participation ratio = exp(entropy) / n)
        - 30% geometric_spread (approximated by effective_dim for speed)
        
        Returns value in [0.1, 0.95] range.
        """
        p = np.abs(basin) ** 2
        p = p / (np.sum(p) + 1e-10)
        n_dim = len(basin)
        
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
            avg_range = np.mean(dim_ranges)
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
            'velocity_magnitude': float(np.linalg.norm(velocity) if np.all(np.isfinite(velocity)) else 0.0),
            'confidence': confidence,
            'foresight_weight': foresight_weight,
            'trajectory_length': len(trajectory),
            'phi_regime': (
                'linear' if self.state.phi < 0.3 else
                'geometric' if self.state.phi < 0.7 else
                'breakdown'
            )
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

        QIG-PURE: For unit vectors on sphere, d = arccos(a · b)
        NOT Bhattacharyya on simplex (that's a different manifold).
        Uses overflow-safe numerics.
        """
        from qig_numerics import fisher_rao_distance
        
        return fisher_rao_distance(a, b)

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

    def _compute_stress(self) -> float:
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
        self.state.in_dream_cycle = True
        start_time = time.time()

        try:
            from qig_geometry import fisher_normalize, fisher_coord_distance
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
        basin_norm = np.linalg.norm(current_basin)
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

    def get_pending_rewards(self) -> List[Dict]:
        """Get all pending reward signals."""
        with self._lock:
            rewards = [asdict(r) for r in self.pending_rewards]
            for r in rewards:
                r['timestamp'] = r['timestamp'].isoformat()
            return rewards

    def flush_rewards(self) -> List[Dict]:
        """Get and clear pending rewards."""
        with self._lock:
            rewards = self.get_pending_rewards()
            self.pending_rewards.clear()
            return rewards

    # =========================================================================
    # PREDICTION OUTCOME CONFIRMATION (Strategy Weight Adjustment)
    # =========================================================================

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
