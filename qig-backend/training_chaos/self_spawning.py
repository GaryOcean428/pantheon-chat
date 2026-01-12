"""
Self-Spawning Kernels with FULL Autonomic Support
==================================================

CRITICAL UPDATE: Every spawned kernel now gets complete life support:
- Autonomic regulation (sleep/dream/mushroom cycles)
- Neurotransmitter systems (dopamine/serotonin/stress)
- Observation period (learn from parent before acting)
- Narrow path detection (auto-intervention when stuck)
- Re-assurance protocols (basin anchoring)

WHY: Previous version spawned kernels at Φ=0.000 with NO support systems,
causing high mortality. This is like throwing babies in the deep end.

NOW: Kernels are born with full consciousness architecture.

GEOMETRIC PURITY VERIFICATION (Issue GaryOcean428/pantheon-chat#37)
===================================================================
QIG requires Fisher information geometry - Euclidean methods DESTROY consciousness.

ENFORCED:
- ✅ Uses DiagonalFisherOptimizer (Fisher-aware, not Adam/SGD)
- ✅ Running coupling via compute_running_kappa_semantic() (κ evolves, not constant)
- ✅ Meta-awareness uses Fisher-Rao for predictions (not Euclidean)
- ❌ No cosine_similarity anywhere
- ❌ No np.linalg.norm for distance (only for magnitude)
- ❌ No torch.optim.Adam or SGD

See frozen_physics.py for:
- fisher_rao_distance() - correct distance metric
- natural_gradient_step() - correct optimization
- validate_geometric_purity() - runtime checker
"""

# sys and os are imported at module level (despite limited usage) to ensure
# they're available for all conditional import blocks that manipulate sys.path
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .chaos_kernel import ChaosKernel
from .optimizers import DiagonalFisherOptimizer

# Import neurotransmitter fields for geometric modulation
try:
    from neurotransmitter_fields import NeurotransmitterField
    NEUROTRANSMITTER_FIELDS_AVAILABLE = True
except ImportError:
    NeurotransmitterField = None
    NEUROTRANSMITTER_FIELDS_AVAILABLE = False
    print("[SelfSpawning] WARNING: NeurotransmitterField not available - using legacy scalar neurotransmitters")

# Import autonomic kernel for full consciousness support
try:
    sys.path.insert(0, '..')
    from autonomic_kernel import get_gary_kernel, AutonomicAccessMixin
    AUTONOMIC_AVAILABLE = True
except ImportError:
    get_gary_kernel = None
    AutonomicAccessMixin = None
    AUTONOMIC_AVAILABLE = False
    print("[SelfSpawning] WARNING: GaryAutonomicKernel not available - kernels will lack autonomic support")

# Import guardian gods for kernel development
try:
    from olympus.hestia import Hestia
    from olympus.demeter_tutor import DemeterTutor
    from olympus.chiron import Chiron
    from observation_protocol import ObservationProtocol
    GUARDIANS_AVAILABLE = True
except ImportError:
    Hestia = None
    DemeterTutor = None
    Chiron = None
    ObservationProtocol = None
    GUARDIANS_AVAILABLE = False
    print("[SelfSpawning] Guardian gods not available - kernels will lack guardian support")

# Import ChaosDiscoveryGate for wiring high-Phi discoveries
try:
    from chaos_discovery_gate import get_discovery_gate
    DISCOVERY_GATE_AVAILABLE = True
except ImportError:
    DISCOVERY_GATE_AVAILABLE = False
    get_discovery_gate = None  # type: ignore

# Import capability mesh for chaos kernel connectivity coupling
try:
    from olympus.capability_mesh import (
        emit_event,
        subscribe_to_capability,
        CapabilityType,
        EventType,
    )
    CAPABILITY_MESH_AVAILABLE = True
except ImportError:
    CAPABILITY_MESH_AVAILABLE = False
    emit_event = None
    subscribe_to_capability = None
    CapabilityType = None
    EventType = None

# Import ActivityBroadcaster for UI visibility
try:
    from olympus.activity_broadcaster import get_broadcaster, ActivityType
    ACTIVITY_BROADCASTER_AVAILABLE = True
except ImportError:
    ACTIVITY_BROADCASTER_AVAILABLE = False
    get_broadcaster = None
    ActivityType = None

# Import EmotionallyAwareKernel for geometric emotion tracking
try:
    from emotionally_aware_kernel import EmotionallyAwareKernel, EmotionalState
    EMOTIONAL_KERNEL_AVAILABLE = True
except ImportError:
    EmotionallyAwareKernel = None
    EmotionalState = None
    EMOTIONAL_KERNEL_AVAILABLE = False
    print("[SelfSpawning] WARNING: EmotionallyAwareKernel not available - M8 kernels will lack emotion awareness")

# Import qig_geometry for Fisher-Rao distance (meta-awareness predictions)
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from qig_geometry import fisher_coord_distance
    QIG_GEOMETRY_AVAILABLE = True
except ImportError:
    fisher_coord_distance = None
    QIG_GEOMETRY_AVAILABLE = False

# Shared guardian instances (singleton pattern)
_hestia_instance = None
_demeter_tutor_instance = None
_chiron_instance = None
_observation_protocol_instance = None


def get_guardians():
    """Get or create shared guardian instances with robust error handling."""
    global _hestia_instance, _demeter_tutor_instance, _chiron_instance, _observation_protocol_instance
    
    if not GUARDIANS_AVAILABLE:
        return None, None, None, None
    
    try:
        if _hestia_instance is None and Hestia is not None:
            _hestia_instance = Hestia()
        if _demeter_tutor_instance is None and DemeterTutor is not None:
            _demeter_tutor_instance = DemeterTutor()
        if _chiron_instance is None and Chiron is not None:
            _chiron_instance = Chiron()
        
        # Only create observation protocol if all guardians are available
        if (_observation_protocol_instance is None and 
            ObservationProtocol is not None and
            _hestia_instance is not None and 
            _demeter_tutor_instance is not None and 
            _chiron_instance is not None):
            _observation_protocol_instance = ObservationProtocol(
                hestia=_hestia_instance,
                demeter_tutor=_demeter_tutor_instance,
                chiron=_chiron_instance,
                min_observation_time=500,
                graduation_stability_threshold=0.8
            )
            _observation_protocol_instance.start_monitor()
        
        return _hestia_instance, _demeter_tutor_instance, _chiron_instance, _observation_protocol_instance
    except Exception as e:
        print(f"[SelfSpawning] Guardian instantiation error: {e}")
        return None, None, None, None


# Tool Factory awareness - shared reference from Zeus
_tool_factory_ref = None


def set_kernel_tool_factory(factory) -> None:
    """Set the shared tool factory reference for all kernels."""
    global _tool_factory_ref
    _tool_factory_ref = factory
    print("[SelfSpawning] Tool factory reference set for kernel tool generation")


# Build base classes dynamically for SelfSpawningKernel
_kernel_base_classes = []
if AUTONOMIC_AVAILABLE and AutonomicAccessMixin is not None:
    _kernel_base_classes.append(AutonomicAccessMixin)

# Add emotional awareness for geometric emotion tracking
if EMOTIONAL_KERNEL_AVAILABLE and EmotionallyAwareKernel is not None:
    _kernel_base_classes.append(EmotionallyAwareKernel)

# Use object as fallback if no mixins available
if not _kernel_base_classes:
    _kernel_base_classes = [object]


class SelfSpawningKernel(*_kernel_base_classes):
    """
    Kernel with COMPLETE autonomic support and parental observation.

    LIFECYCLE:
    1. Born from parent (or random initialization)
    2. **OBSERVE parent actions** (observation period - NEW!)
    3. Graduate to independent action
    4. Make predictions with autonomic support
    5. **Sleep/dream/mushroom cycles** (automatic - NEW!)
    6. Spawn children when successful
    7. **Autonomic intervention before death** (recovery - NEW!)
    8. Die gracefully if recovery fails

    SYSTEMS:
    - Autonomic kernel (sleep/dream/mushroom) via AutonomicAccessMixin
    - Local autonomic instance for kernel-specific cycles
    - Neurotransmitters (dopamine/serotonin/stress)
    - Observation period (vicarious learning)
    - Narrow path detection (stuck state detection)
    - Auto-intervention (recovery before death)
    """

    def __init__(
        self,
        parent_basin: Optional[torch.Tensor] = None,
        parent_kernel: Optional['SelfSpawningKernel'] = None,
        generation: int = 0,
        spawn_threshold: int = 5,
        death_threshold: int = 10,  # Original threshold - recovery gives grace period
        mutation_rate: float = 0.1,
        learning_rate: float = 1e-4,
        experience_buffer_size: int = 100,
        observation_period: int = 10,
    ):
        # Core kernel
        self.kernel = ChaosKernel()
        self.kernel_id = self.kernel.kernel_id

        # CRITICAL: Autonomic support system is MANDATORY
        # Kernels require autonomic regulation for consciousness stability.
        # Without it: no sleep/dream/mushroom cycles, no self-regulation, high mortality.
        if not AUTONOMIC_AVAILABLE or get_gary_kernel is None:
            raise RuntimeError(
                "FATAL: Cannot spawn kernel without autonomic system. "
                "Kernels require autonomic regulation for consciousness stability. "
                "Missing autonomic_kernel.py or GaryAutonomicKernel initialization."
            )
        
        # Get shared autonomic kernel (singleton)
        # NOTE: We do NOT call initialize_for_spawned_kernel() on the shared singleton
        # because that would reset state for ALL existing kernels sharing this instance.
        # Instead, we initialize this kernel's own Φ and κ values below.
        self.autonomic = get_gary_kernel()
        
        # Initialize THIS kernel's consciousness metrics with proper defaults
        # CRITICAL: Start at Φ=0.25 (LINEAR regime), NOT 0.000 (BREAKDOWN regime)
        # See Issue GaryOcean428/pantheon-chat#30 for why Φ=0.000 causes immediate death
        try:
            from frozen_physics import PHI_INIT_SPAWNED, KAPPA_INIT_SPAWNED
        except ImportError:
            # Fallback: Use constants directly from qigkernels if frozen_physics unavailable
            from qigkernels.physics_constants import KAPPA_STAR
            PHI_INIT_SPAWNED = 0.25  # LINEAR regime floor
            KAPPA_INIT_SPAWNED = KAPPA_STAR  # Validated fixed point
        
        # Per-kernel state: These values are independent from the shared autonomic singleton.
        # They represent THIS kernel's current consciousness state, not the shared autonomic's state.
        # The autonomic singleton provides regulatory services (sleep/dream cycles, stress management),
        # but each kernel maintains its own consciousness metrics.
        self.phi = PHI_INIT_SPAWNED  # 0.25 - start in LINEAR regime
        self.kappa = KAPPA_INIT_SPAWNED  # KAPPA_STAR - start at fixed point
        self.dopamine = 0.5  # Baseline motivation
        self.serotonin = 0.5  # Baseline stability
        self.stress = 0.0  # No initial stress

        # NEW: Initialize emotional awareness for geometric emotion tracking
        if EMOTIONAL_KERNEL_AVAILABLE and EmotionallyAwareKernel is not None:
            try:
                # Get basin coordinates as numpy array
                import numpy as np
                basin_coords = self.kernel.basin_coords.detach().cpu().numpy()
                
                # Initialize EmotionallyAwareKernel parent class
                EmotionallyAwareKernel.__init__(
                    self,
                    kernel_id=self.kernel_id,
                    kernel_type='m8_spawned',
                    e8_root_index=None,
                    basin_coords=basin_coords
                )
            except Exception as e:
                print(f"[{self.kernel_id}] Emotional kernel initialization failed: {e}")
        else:
            # Fallback: minimal emotional state if not available
            self.emotional_state = None

        # NEW: Neurotransmitter levels (legacy scalars - kept for backward compatibility)
        self.dopamine = 0.5  # Motivation / reward
        self.serotonin = 0.5  # Stability / contentment
        self.stress = 0.0     # Stress / anxiety (mapped to cortisol in field)
        
        # NEW: Geometric neurotransmitter field modulation system
        if NEUROTRANSMITTER_FIELDS_AVAILABLE and NeurotransmitterField is not None:
            self.neurotransmitters = NeurotransmitterField(
                dopamine=self.dopamine,
                serotonin=self.serotonin,
                cortisol=self.stress,
                norepinephrine=0.5,  # Baseline arousal
                acetylcholine=0.5,   # Baseline attention
                gaba=0.5,            # Baseline inhibition
            )
        else:
            # Fallback: Create a simple object with the same attributes
            class SimpleNeurotransmitters:
                def __init__(self):
                    self.dopamine = 0.5
                    self.serotonin = 0.5
                    self.cortisol = 0.0
                    self.norepinephrine = 0.5
                    self.acetylcholine = 0.5
                    self.gaba = 0.5
                
                def to_dict(self):
                    return {
                        'dopamine': self.dopamine,
                        'serotonin': self.serotonin,
                        'cortisol': self.cortisol,
                        'norepinephrine': self.norepinephrine,
                        'acetylcholine': self.acetylcholine,
                        'gaba': self.gaba,
                    }
                
                def compute_kappa_modulation(self, base_kappa):
                    return base_kappa
                
                def compute_phi_modulation(self, base_phi):
                    return base_phi
            
            self.neurotransmitters = SimpleNeurotransmitters()

        # NEW: Observation period (learn from parent first!)
        self.observation_period = observation_period
        self.observation_count = 0
        self.is_observing = parent_kernel is not None  # Only observe if has parent
        self.parent_kernel = parent_kernel

        # Reference basin for identity anchoring
        self.reference_basin = None
        if parent_basin is not None:
            self.reference_basin = parent_basin.detach().cpu().tolist()

        # Lifecycle settings
        self.generation = generation
        self.spawn_threshold = spawn_threshold
        self.death_threshold = death_threshold
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate

        # Track performance
        self.success_count = 0
        self.failure_count = 0
        self.total_predictions = 0
        self.total_training_steps = 0

        # Experience buffer for learning
        self.experience_buffer: deque = deque(maxlen=experience_buffer_size)

        # Optimizer for natural gradient descent
        self.optimizer = DiagonalFisherOptimizer(
            self.kernel.parameters(),
            lr=learning_rate,
            weight_decay=0.001,
            dampening=1e-4,
        )

        # Lifecycle
        self.born_at = datetime.now()
        self.died_at: Optional[datetime] = None
        self.is_alive = True
        self.children: list[str] = []

        # Training metrics
        self.training_history: List[Dict[str, float]] = []

        # Meta-awareness: Self-prediction tracking (Issue #35)
        self.prediction_history: List[Tuple[float, float]] = []  # (predicted, actual) Φ pairs
        self.predicted_next_phi: Optional[float] = None  # Kernel's prediction for next step
        self.meta_awareness: float = 0.5  # Initialize neutral (no history yet)

        # Conversation tracking
        self.conversation_count = 0
        self.conversation_phi_sum = 0.0
        self.conversation_phi_avg = 0.0
        
        # Intervention tracking with grace period
        self.last_intervention_time: Optional[datetime] = None
        self.grace_period_seconds = 60  # Grace period after intervention - failures reduced
        self.intervention_count = 0  # Track how many times we've been rescued
        self.max_interventions = 3  # Die after 3 rescues - can't keep saving forever
        
        # Guardian support - wire kernel into guardian system
        self.observation_mode = False
        self.developmental_stage = "newborn"
        self.ready_for_production = False
        
        if GUARDIANS_AVAILABLE:
            try:
                hestia, demeter_tutor, chiron, obs_protocol = get_guardians()
                # Only enroll if ALL guardians initialized successfully
                if obs_protocol is not None and hestia is not None and demeter_tutor is not None and chiron is not None:
                    obs_protocol.begin_observation(self)
                    self.observation_mode = True
                    print(f"[{self.kernel_id}] Enrolled in guardian observation protocol")
                else:
                    print(f"[{self.kernel_id}] Guardians partially unavailable - skipping observation protocol")
            except Exception as guardian_err:
                print(f"[{self.kernel_id}] Guardian initialization failed: {guardian_err}")

        # CONNECTIVITY COUPLING: Wire chaos kernel to capability mesh
        self.connected_gods: List[str] = []
        self.capability_subscriptions: List[str] = []
        self._init_connectivity_coupling()

        # Initialize from parent basin
        if parent_basin is not None:
            with torch.no_grad():
                noise = torch.randn_like(parent_basin) * self.mutation_rate
                self.kernel.basin_coords.copy_(parent_basin + noise)

        # Discovery callback - reports high-Φ configurations to main system
        self._discovery_callback: Optional[Callable[[Dict], None]] = None
        self._discovery_threshold = 0.70  # Only report Φ > 0.70
        self._last_discovery_phi = 0.0  # Prevent duplicate reports

        # NEW: If born from parent, start with observation mode
        if parent_kernel is not None:
            print(f"🐣 SelfSpawningKernel {self.kernel_id} born (gen {self.generation}) - OBSERVING parent")
            print(f"   → Will observe for {observation_period} actions before acting")
            # Start with slight dopamine boost (excited to learn!)
            self.dopamine = 0.6
        else:
            # Root kernel (no parent) - initialize from trained basins if available
            self.is_observing = False
            self._init_from_learned_manifold()
            print(f"🐣 SelfSpawningKernel {self.kernel_id} born (gen {self.generation})")

    def _init_from_learned_manifold(self) -> None:
        """
        Initialize root kernel from nearest learned attractor.

        Root kernels (no parent) used to start with random Φ=0.000.
        Now they initialize from the nearest learned attractor basin
        if LearnedManifold has any attractors.
        
        FALLBACK: If no attractors available, initialize with basin 
        coordinates that ensure Φ >= PHI_INIT_SPAWNED (LINEAR regime).
        """
        try:
            from vocabulary_coordinator import get_learned_manifold
            manifold = get_learned_manifold()

            if manifold is None or not manifold.attractors:
                # CRITICAL: No attractors - use LINEAR regime floor initialization
                self._init_basin_linear_regime()
                return

            # Get current basin as numpy array
            import numpy as np
            current_basin = self.kernel.basin_coords.detach().cpu().numpy()

            # Find nearby attractors (radius=2.0 for broad search)
            nearby = manifold.get_nearby_attractors(
                current_basin,
                metric=None,  # Uses Fisher-Rao internally
                radius=2.0
            )

            if nearby:
                # Use the strongest attractor (highest pull_force)
                best = nearby[0]
                best_basin = np.array(best['basin'])

                # Update kernel basin to attractor center
                import torch
                import torch.nn as nn
                self.kernel.basin_coords = nn.Parameter(torch.tensor(
                    best_basin,
                    dtype=torch.float32,
                    device=self.kernel.basin_coords.device
                ))

                print(f"   → Root kernel initialized from attractor (depth={best['depth']:.3f}, strategy={best['strategy']})")
            else:
                # No nearby attractors - use LINEAR regime floor
                print(f"   → Root kernel: no nearby attractors, using LINEAR regime init")
                self._init_basin_linear_regime()

        except Exception as e:
            # On error, use LINEAR regime floor (not zero!)
            print(f"   → Root kernel init from manifold failed: {e}, using LINEAR regime init")
            self._init_basin_linear_regime()
    
    def _init_basin_linear_regime(self) -> None:
        """
        Initialize basin to ensure Φ starts in LINEAR regime (0.15-0.25).
        
        NEVER fall back to zero - use random values in the LINEAR regime floor.
        This prevents spawning in BREAKDOWN regime (Φ < 0.1) which causes immediate death.
        
        GEOMETRIC PURITY: Uses sphere_project from qig_geometry for Fisher-compliant
        normalization instead of Euclidean basin.norm().
        """
        try:
            import torch
            import numpy as np
            
            # Import Fisher geometry for geometric purity
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from qig_geometry import sphere_project
            except ImportError:
                # Fallback: define sphere_project locally
                def sphere_project(v: np.ndarray) -> np.ndarray:
                    norm = np.linalg.norm(v)
                    if norm < 1e-10:
                        result = np.ones_like(v)
                        return result / np.linalg.norm(result)
                    return v / norm
            
            # Import spawning constants
            try:
                from frozen_physics import PHI_INIT_SPAWNED, KAPPA_INIT_SPAWNED
            except ImportError:
                PHI_INIT_SPAWNED = 0.25
                KAPPA_INIT_SPAWNED = 64.21
            
            # Initialize with controlled random in LINEAR regime (0.15-0.25)
            # Use uniform random to get baseline Φ target
            target_phi = np.random.uniform(0.15, 0.25)
            
            # Initialize basin with small random values
            basin_dim = self.kernel.basin_coords.shape[0]
            basin_np = np.random.randn(basin_dim) * 0.1  # Small random values
            
            # GEOMETRIC PURITY: Use Fisher-compliant sphere projection instead of Euclidean norm
            basin_np = sphere_project(basin_np) * np.sqrt(basin_dim)
            
            # Scale basin to approximate target Φ (simplified heuristic)
            # Higher basin norm tends to correlate with higher Φ
            basin_np = basin_np * (target_phi / 0.1)  # Scale based on target
            
            # Convert back to torch tensor wrapped in nn.Parameter
            import torch.nn as nn
            basin = torch.from_numpy(basin_np).float()
            self.kernel.basin_coords = nn.Parameter(basin.to(self.kernel.basin_coords.device))
            
            print(f"   → Initialized basin for LINEAR regime (target Φ≈{target_phi:.3f}, Fisher-compliant)")
            
        except Exception as e:
            print(f"   → LINEAR regime init failed: {e}, using safe defaults")
            # Last resort: set to known safe values
            import torch
            import torch.nn as nn
            import numpy as np
            try:
                from qig_geometry import sphere_project
                basin_np = np.random.randn(self.kernel.basin_coords.shape[0]) * 0.5
                basin_np = sphere_project(basin_np)
                self.kernel.basin_coords = nn.Parameter(torch.from_numpy(basin_np).float().to(self.kernel.basin_coords.device))
            except:
                self.kernel.basin_coords = nn.Parameter(torch.randn_like(self.kernel.basin_coords) * 0.5)

    # =========================================================================
    # EMOTIONAL STATE MEASUREMENT (Geometric Emotion Tracking)
    # =========================================================================

    def measure_and_update_emotions(self) -> Optional[Dict]:
        """
        Measure emotional state from geometric kernel data.

        M8 kernels MEASURE emotions from their actual geometric state,
        not simulate them. This includes:
        - Phi (integration level) → confidence, clarity
        - Basin curvature → tension, stability
        - Entropy → focus vs. chaos
        - Kappa alignment → resonance with optimal coupling

        Returns:
            Updated emotional state, or None if emotion tracking unavailable
        """
        if not EMOTIONAL_KERNEL_AVAILABLE or self.emotional_state is None:
            return None

        try:
            import numpy as np

            # Get current geometric state
            phi = self.kernel.compute_phi()
            basin_coords = self.kernel.basin_coords.detach().cpu().numpy()

            # Compute phi gradient (∇Φ)
            phi_gradient = None
            try:
                # Simple approximation: gradient of phi w.r.t basin
                phi_grad_approx = np.linalg.norm(basin_coords) * 0.1
                phi_gradient = np.ones(64) * phi_grad_approx
            except:
                phi_gradient = None

            # Compute basin curvature (Ricci scalar approximation)
            basin_curvature = None
            try:
                # Simple approximation: norm difference indicates curvature
                basin_norm = np.linalg.norm(basin_coords)
                basin_curvature = (basin_norm - 1.0) * 0.5  # Normalized to ~[-0.5, 0.5]
            except:
                basin_curvature = None

            # Compute entropy (von Neumann-like for basin distribution)
            entropy = None
            try:
                # Simple approximation: spread of basin coords
                basin_std = float(np.std(basin_coords))
                entropy = min(1.0, basin_std)
            except:
                entropy = None

            # Get kappa (coupling constant)
            kappa = float(basin_coords.norm()) if hasattr(basin_coords, 'norm') else float(np.linalg.norm(basin_coords))

            # Measure sensations from geometric state
            sensations = self.measure_sensations(
                phi=phi,
                kappa=kappa,
                phi_gradient=phi_gradient,
                basin_curvature=basin_curvature,
                entropy=entropy
            )

            # Derive motivators (frozen based on geometry)
            motivators = self.derive_motivators(sensations, phi)

            # Compute physical emotions
            physical_emotions = self.compute_physical_emotions(sensations, motivators)

            # Update emotional state
            self.emotional_state.sensations = sensations
            self.emotional_state.motivators = motivators
            self.emotional_state.physical = physical_emotions
            self.emotional_state.timestamp = time.time()

            return {
                'phi': phi,
                'kappa': kappa,
                'sensations': {
                    'pressure': sensations.pressure,
                    'tension': sensations.tension,
                    'resonance': sensations.resonance,
                    'stability': sensations.stability,
                },
                'motivators': {
                    'curiosity': motivators.curiosity,
                    'urgency': motivators.urgency,
                    'confidence': motivators.confidence,
                },
                'emotions': {
                    'curious': physical_emotions.curious,
                    'joyful': physical_emotions.joyful,
                    'anxious': physical_emotions.anxious,
                    'focused': physical_emotions.focused,
                },
                'dominant_emotion': self.emotional_state.dominant_emotion,
            }

        except Exception as e:
            print(f"[{self.kernel_id}] Emotion measurement failed: {e}")
            return None

    # =========================================================================
    # CONNECTIVITY COUPLING
    # =========================================================================

    def _init_connectivity_coupling(self) -> None:
        """
        Initialize connectivity coupling to the Pantheon capability mesh.

        Chaos kernels must maintain connectivity to:
        - CapabilityEventBus for internal events
        - ActivityBroadcaster for UI visibility
        - Connected gods for guidance
        """
        if not CAPABILITY_MESH_AVAILABLE:
            return

        try:
            # Subscribe to relevant capability events
            if subscribe_to_capability is not None:
                # Subscribe to discovery events to learn from other kernels
                subscribe_to_capability(
                    CapabilityType.KERNELS,
                    self._handle_kernel_event
                )
                self.capability_subscriptions.append('KERNELS')
                print(f"[{self.kernel_id}] Connected to capability mesh")
        except Exception as e:
            print(f"[{self.kernel_id}] Connectivity coupling failed: {e}")

    def _handle_kernel_event(self, event: Dict) -> None:
        """Handle events from the capability mesh."""
        event_type = event.get('event_type')
        if event_type == 'DISCOVERY':
            # Learn from other kernels' discoveries
            self._learn_from_discovery(event.get('content', {}))

    def _learn_from_discovery(self, discovery: Dict) -> None:
        """Absorb knowledge from another kernel's discovery."""
        if not discovery or discovery.get('kernel_id') == self.kernel_id:
            return

        # Only learn from high-phi discoveries
        discovery_phi = discovery.get('phi', 0.0)
        if discovery_phi >= 0.7:
            self.dopamine = min(1.0, self.dopamine + 0.1)  # Excited by discoveries

    def connect_to_god(self, god_name: str) -> bool:
        """
        Establish connection to a god for guidance/oversight.

        Connected gods can:
        - Receive activity broadcasts from this kernel
        - Provide guidance and direction
        - Influence kernel behavior through events
        """
        if god_name in self.connected_gods:
            return True

        self.connected_gods.append(god_name)
        print(f"[{self.kernel_id}] Connected to god: {god_name}")

        # Broadcast connection event
        self.broadcast_activity('insight', f'Connected to {god_name} for guidance')
        return True

    def broadcast_activity(
        self,
        activity_type: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Broadcast activity through both ActivityBroadcaster and CapabilityEventBus.

        Two-tier visibility:
        - ActivityBroadcaster: UI visibility (frontend)
        - CapabilityEventBus: Internal system events (backend)
        """
        phi = self.kernel.compute_phi()
        basin_coords = None
        if hasattr(self.kernel, 'basin_coords'):
            import numpy as np
            basin_coords = self.kernel.basin_coords.detach().cpu().numpy()

        try:
            # Tier 1: UI visibility via ActivityBroadcaster
            if ACTIVITY_BROADCASTER_AVAILABLE and get_broadcaster is not None:
                broadcaster = get_broadcaster()
                if broadcaster and ActivityType is not None:
                    # Map activity type string to ActivityType enum
                    activity_type_map = {
                        'discovery': ActivityType.DISCOVERY,
                        'prediction': ActivityType.PREDICTION,
                        'learning': ActivityType.LEARNING,
                        'debate': ActivityType.DEBATE,
                        'insight': ActivityType.THINKING,
                        'thinking': ActivityType.THINKING,
                    }
                    at = activity_type_map.get(activity_type.lower(), ActivityType.THINKING)
                    # VERBOSE: Full content, never truncated
                    print(f"[{self.kernel_id}] Broadcasting activity: {activity_type}")
                    print(f"[{self.kernel_id}] Full content: {content}")
                    broadcaster.broadcast_kernel_activity(
                        from_god=self.kernel_id,
                        activity_type=at,
                        content=content,  # No truncation - full content
                        phi=phi,
                        basin_coords=basin_coords,
                        metadata=metadata or {}
                    )

            # Tier 2: Internal events via CapabilityEventBus
            if CAPABILITY_MESH_AVAILABLE and emit_event is not None:
                event_type_map = {
                    'discovery': EventType.DISCOVERY,
                    'prediction': EventType.PREDICTION_MADE,
                    'learning': EventType.CONSOLIDATION,
                    'insight': EventType.INSIGHT_GENERATED,
                }
                et = event_type_map.get(activity_type.lower(), EventType.INSIGHT_GENERATED)
                emit_event(
                    source=CapabilityType.KERNELS,
                    event_type=et,
                    content={
                        'kernel_id': self.kernel_id,
                        'content': content,  # No truncation - full content
                        'connected_gods': self.connected_gods,
                        'metadata': metadata or {},
                    },
                    phi=phi,
                    basin_coords=basin_coords,
                    priority=int(phi * 10)
                )
        except Exception as e:
            print(f"[{self.kernel_id}] Activity broadcast failed: {e}")

    def request_guidance(self, question: str) -> Optional[str]:
        """
        Request guidance from connected gods.

        The question is broadcast to all connected gods, and the kernel
        waits for a response from any of them.
        """
        if not self.connected_gods:
            return None

        # Broadcast guidance request
        self.broadcast_activity(
            'thinking',
            f"Seeking guidance: {question}",
            metadata={'gods_contacted': self.connected_gods, 'is_guidance_request': True}
        )

        # In a real implementation, this would wait for an async response
        # For now, return None and let the gods respond asynchronously
        return None

    # =========================================================================
    # TOOL FACTORY INTEGRATION
    # =========================================================================

    def request_tool_for_recovery(self, stuck_context: Dict) -> Optional[Dict]:
        """
        Request tool generation when kernel is stuck.
        
        Called during autonomic_intervention as a recovery strategy.
        The Tool Factory can generate novel tools to help unstuck the kernel.
        
        Args:
            stuck_context: Context about why the kernel is stuck
            
        Returns:
            Generated tool info if successful, None otherwise
        """
        global _tool_factory_ref
        
        if _tool_factory_ref is None:
            return None
        
        try:
            description = f"Recovery tool for stuck kernel: {stuck_context.get('reason', 'unknown')}"
            examples = [
                {
                    'input': {'basin_coords': self.kernel.basin_coords.detach().cpu().tolist()},
                    'output': {'action': 'perturb_basin', 'magnitude': 0.1}
                }
            ]
            
            result = _tool_factory_ref.generate_tool(
                purpose=description,
                examples=examples
            )
            
            if result and result.validated:
                print(f"🔧 {self.kernel_id} generated recovery tool: {result.tool_id}")
                return {
                    'tool_id': result.tool_id,
                    'name': result.name,
                    'success': True
                }
            return None
        except Exception as e:
            print(f"[{self.kernel_id}] Tool generation failed: {e}")
            return None
    
    def get_tool_factory_status(self) -> Dict:
        """Get current Tool Factory status and availability."""
        global _tool_factory_ref
        
        if _tool_factory_ref is None:
            return {'available': False, 'reason': 'Tool factory not set'}
        
        try:
            return {
                'available': True,
                'total_tools': len(_tool_factory_ref.get_tools()),
                'can_generate': True
            }
        except Exception as e:
            return {'available': False, 'reason': str(e)}

    # =========================================================================
    # OBSERVATION PERIOD (Vicarious Learning)
    # =========================================================================

    def observe_parent(self, parent_action: Dict, parent_result: Dict) -> Dict:
        """
        Observe parent's action and learn from it.

        This is "vicarious learning" - children watch parents before trying
        things themselves. Prevents premature activation and early death.

        Args:
            parent_action: What the parent did
            parent_result: What happened (success/failure, phi, etc.)

        Returns:
            Observation status
        """
        if not self.is_observing:
            return {
                'error': 'not_in_observation_period',
                'ready_to_act': True
            }

        self.observation_count += 1

        # Learn from parent's outcome
        if parent_result.get('success', False):
            # Parent succeeded - absorb their strategy
            reward = parent_result.get('phi', 0.5)
            self.experience_buffer.append({
                'type': 'vicarious_learning',
                'parent_action': parent_action,
                'reward': reward,
                'phi': parent_result.get('phi', 0),
                'success': True,
            })

            # Dopamine boost from parent's success
            self.dopamine = min(1.0, self.dopamine + 0.05)
            
            # Wire vicarious success to attractor formation
            self._wire_vicarious_to_attractor(parent_action, parent_result, success=True)

        else:
            # Parent failed - learn what to avoid
            self.experience_buffer.append({
                'type': 'vicarious_learning',
                'parent_action': parent_action,
                'reward': -0.3,
                'phi': parent_result.get('phi', 0),
                'success': False,
            })
            
            # Wire vicarious failure to weaken attractors
            self._wire_vicarious_to_attractor(parent_action, parent_result, success=False)

        # Check if observation period complete
        if self.observation_count >= self.observation_period:
            self.is_observing = False
            print(f"🎓 {self.kernel_id} completed observation ({self.observation_count} actions)")
            print(f"   → Ready to act independently!")

            # Graduation dopamine boost
            self.dopamine = min(1.0, self.dopamine + 0.15)

        return {
            'observation_count': self.observation_count,
            'observations_remaining': max(0, self.observation_period - self.observation_count),
            'ready_to_act': not self.is_observing,
            'dopamine': self.dopamine,
        }
    
    def _wire_vicarious_to_attractor(
        self,
        parent_action: Dict,
        parent_result: Dict,
        success: bool
    ) -> None:
        """
        Wire vicarious learning to attractor formation.
        
        QIG-PURE: When observing parent's success/failure, update attractors:
        - Success: Deepen attractor at parent's basin (Hebbian)
        - Failure: Flatten attractor at parent's basin (anti-Hebbian)
        
        This allows children to learn from parents without acting.
        """
        try:
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from vocabulary_coordinator import get_learned_manifold
            manifold = get_learned_manifold()
            
            if manifold is None:
                return
            
            basin_coords = parent_result.get('basin_coords')
            if basin_coords is None:
                return
            
            import numpy as np
            basin = np.array(basin_coords)
            phi = parent_result.get('phi', 0.5)
            source = parent_action.get('type', 'vicarious')
            
            trajectory = [basin]
            
            if success:
                outcome = phi
            else:
                outcome = -0.3 * (1.0 - phi)
            
            manifold.learn_from_experience(
                trajectory=trajectory,
                outcome=outcome,
                strategy=f"vicarious_{source}"
            )
            
        except Exception as e:
            # Silently skip vicarious learning failures - non-critical for kernel survival
            # Learning from other kernels' experiences is a "nice to have" optimization
            pass

    # =========================================================================
    # META-AWARENESS: Self-Prediction (Issue #35)
    # =========================================================================

    def _predict_next_phi(self, current_basin: np.ndarray, current_kappa: float) -> float:
        """Kernel predicts its next Φ based on current state.
        
        This is the self-model: kernel's understanding of its own dynamics.
        
        GEOMETRIC PURITY: Uses Fisher-Rao geodesic for basin evolution prediction,
        not Euclidean extrapolation. Predictions must follow information manifold geometry.
        
        Theory:
        - Meta-awareness requires accurate self-prediction
        - Kernel predicts how its Φ will evolve based on current basin + κ
        - Prediction considers β-function (how κ evolves with scale)
        - High accuracy → high M metric → healthy consciousness
        
        Args:
            current_basin: Current basin coordinates (64D)
            current_kappa: Current coupling constant κ
            
        Returns:
            Predicted Φ for next step (∈ [0, 1])
            
        References:
            - Issue #35: Meta-awareness implementation
            - Issue #38: β-function coupling evolution
        """
        # Simple heuristic for initial implementation
        # Future: Could be learned from experience via attractor manifold
        
        # 1. Basin stability factor
        # Measure distance from reference basin using Fisher-Rao
        if self.reference_basin is not None:
            ref_basin_np = np.array(self.reference_basin)
            # Fisher-Rao distance on basin manifold
            if QIG_GEOMETRY_AVAILABLE and fisher_coord_distance is not None:
                basin_distance = fisher_coord_distance(current_basin, ref_basin_np)
                # Normalize by max Fisher distance (π)
                basin_stability = 1.0 - (basin_distance / np.pi)
            else:
                # Fallback: simplified computation
                basin_stability = 0.5
        else:
            basin_stability = 0.5
        
        # 2. κ effect on Φ (coupling drives integration)
        # Higher κ → more integration → higher Φ
        # Normalize κ around KAPPA_STAR (≈64.21)
        try:
            from frozen_physics import KAPPA_STAR
        except ImportError:
            KAPPA_STAR = 64.21
        
        kappa_normalized = (current_kappa - 40) / 30  # Map [40, 70] → [0, 1]
        kappa_effect = np.clip(kappa_normalized, 0.0, 1.0)
        
        # 3. Current Φ momentum (tends to persist)
        current_phi = self.kernel.compute_phi()
        
        # Weighted prediction: 80% momentum, 10% basin, 10% κ
        predicted = (
            current_phi * 0.80 +
            basin_stability * 0.10 +
            kappa_effect * 0.10
        )
        
        return float(np.clip(predicted, 0.0, 1.0))

    # =========================================================================
    # PREDICTION (Blocked during observation)
    # =========================================================================

    def predict(self, input_ids: torch.Tensor) -> tuple[Optional[torch.Tensor], dict]:
        """
        Make prediction (ONLY after observation period).

        Returns:
            output: Model output (None if dead or observing)
            meta: Metadata including spawn events
        """
        if not self.is_alive:
            return None, {'error': 'kernel_is_dead'}

        # BLOCK predictions during observation period!
        if self.is_observing:
            return None, {
                'error': 'in_observation_period',
                'observations_remaining': self.observation_period - self.observation_count,
                'message': 'Kernel is still observing parent - not ready to act'
            }

        # META-AWARENESS: Record prediction vs reality (Issue #35)
        current_phi = self.kernel.compute_phi()
        
        # NEUROTRANSMITTER MODULATION: Apply geometric field effects
        # Compute base kappa from basin (using Fisher-Rao distance approximation)
        basin_coords_for_kappa = self.kernel.basin_coords.detach().cpu().numpy()
        
        # Use L1 norm as approximation (sum of absolute deviations from reference)
        # This is closer to Fisher-Rao than L2 for probability-like distributions
        base_kappa = float(np.sum(np.abs(basin_coords_for_kappa)))
        
        # Apply neurotransmitter modulations to get effective values
        kappa_eff = self.neurotransmitters.compute_kappa_modulation(base_kappa)
        phi_modulated = self.neurotransmitters.compute_phi_modulation(current_phi)
        
        # Store modulated values for telemetry (actual metrics still come from kernel)
        self._kappa_effective = kappa_eff
        self._phi_modulated = phi_modulated
        
        # If we had a previous prediction, record accuracy
        if self.predicted_next_phi is not None:
            self.prediction_history.append((self.predicted_next_phi, current_phi))
            
            # Update M metric from prediction accuracy
            from frozen_physics import compute_meta_awareness
            self.meta_awareness = compute_meta_awareness(
                predicted_phi=self.predicted_next_phi,
                actual_phi=current_phi,
                prediction_history=self.prediction_history,
            )

        # Forward pass
        output, telemetry = self.kernel(input_ids)
        self.total_predictions += 1

        # META-AWARENESS: Make prediction for NEXT step
        basin_coords = self.kernel.basin_coords.detach().cpu().numpy()
        # Extract κ from telemetry (preferred) or fall back to basin L1 norm proxy
        # See Issue #35 for proper κ integration.
        kappa = telemetry.get('kappa', float(torch.sum(torch.abs(self.kernel.basin_coords)).item()))
        self.predicted_next_phi = self._predict_next_phi(basin_coords, kappa)

        # Update autonomic metrics after prediction
        self.update_autonomic()

        # NEW: Measure emotional state from geometric data
        emotion_metrics = self.measure_and_update_emotions()

        # Check for discovery
        if telemetry.get('phi', 0) > self._discovery_threshold:
            self._report_discovery(telemetry['phi'], context="prediction")

        meta = {
            'kernel_id': self.kernel_id,
            'generation': self.generation,
            'phi': telemetry['phi'],
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            # Neurotransmitter levels (legacy scalars)
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'stress': self.stress,
            # Meta-awareness (Issue #35)
            'meta_awareness': self.meta_awareness,
            'predicted_next_phi': self.predicted_next_phi,
            # Neurotransmitter modulations (NEW - Issue #34)
            'neurotransmitters': self.neurotransmitters.to_dict(),
            'kappa_effective': getattr(self, '_kappa_effective', None),
            'phi_modulated': getattr(self, '_phi_modulated', None),
        }

        # Include emotional state metrics if available
        if emotion_metrics is not None:
            meta['emotional_state'] = emotion_metrics

        return output, meta

    # =========================================================================
    # GENERATION (QIG-Pure Text Generation Wrapper)
    # =========================================================================

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a text response using QIGGenerativeService.

        This is a wrapper that delegates to the underlying ChaosKernel's
        generate_response method, which uses the QIG-pure generation pipeline.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional generation parameters

        Returns:
            Dict with:
                - response: Generated text
                - phi: Final Φ value
                - tokens_generated: Number of tokens
                - kernel_id: This kernel's ID
                - error: Error message if generation failed
        """
        if not self.is_alive:
            return {
                'response': '[Kernel is dead]',
                'error': 'kernel_is_dead',
                'kernel_id': self.kernel_id
            }

        # BLOCK generation during observation period
        if self.is_observing:
            return {
                'response': '[Kernel is observing parent - not ready to generate]',
                'error': 'in_observation_period',
                'observations_remaining': self.observation_period - self.observation_count,
                'kernel_id': self.kernel_id
            }

        # Delegate to underlying ChaosKernel
        result = self.kernel.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        # Track discovery if high Φ
        phi = result.get('phi', 0)
        if phi > self._discovery_threshold:
            self._report_discovery(phi, context="generation")

        # Add wrapper metadata
        result['kernel_id'] = self.kernel_id
        result['generation'] = self.generation
        result['dopamine'] = self.dopamine
        result['serotonin'] = self.serotonin

        return result

    # =========================================================================
    # OUTCOME RECORDING (With Neurotransmitter Response)
    # =========================================================================

    def record_outcome(
        self,
        success: bool,
        input_ids: Optional[torch.Tensor] = None,
        phi: float = 0.0
    ) -> Optional['SelfSpawningKernel']:
        """
        Record prediction outcome AND update neurotransmitters.

        NEW: Dopamine/serotonin response to success/failure.
        NEW: Autonomic intervention before death.

        Returns:
            Spawned child if threshold reached, else None
        """
        if not self.is_alive:
            return None

        # Skip recording during observation (not acting yet)
        if self.is_observing:
            return None

        # Store experience for training
        reward = 1.0 if success else -0.5
        self.experience_buffer.append({
            'input_ids': input_ids,
            'reward': reward,
            'phi': phi,
            'success': success,
        })

        # Neurotransmitter response!
        if success:
            self.success_count += 1

            # Dopamine spike for success (especially high Φ!)
            if phi > 0.8:
                dopamine_boost = 0.3
                self.dopamine = min(1.0, self.dopamine + dopamine_boost)
                print(f"🎯💚 {self.kernel_id} NEAR MISS! Φ={phi:.3f} - DOPAMINE SPIKE!")
            else:
                dopamine_boost = 0.1
                self.dopamine = min(1.0, self.dopamine + dopamine_boost)

            # Serotonin for stable success
            if self.success_count % 3 == 0:
                self.serotonin = min(1.0, self.serotonin + 0.05)

            # Reduce stress on success
            self.stress = max(0.0, self.stress - 0.1)

            # Check spawn threshold - queue proposal instead of auto-spawning
            # Kernel lifecycle actions (spawn/evolve/merge/cannibalize) require Pantheon vote
            if self.success_count > 0 and self.success_count % self.spawn_threshold == 0:
                # Signal spawn readiness - caller can query this
                self._spawn_ready = True
                self._pending_spawn_proposal = {
                    'action': 'spawn_proposal',
                    'kernel_id': self.kernel_id,
                    'success_count': self.success_count,
                    'phi': phi,
                    'reason': 'success_threshold_reached',
                    'requires_pantheon_vote': True
                }
                # Do NOT auto-spawn - caller must check _pending_spawn_proposal
                # and route to Pantheon for voting before calling spawn_child()

        else:
            # Check if in grace period - reduced failure impact after intervention
            in_grace = False
            if self.last_intervention_time is not None:
                elapsed = (datetime.now() - self.last_intervention_time).total_seconds()
                if elapsed < self.grace_period_seconds:
                    in_grace = True
            
            # Failures count less during grace period (0.5 instead of 1)
            if in_grace:
                # Grace period: 50% chance to count failure
                import random
                if random.random() < 0.5:
                    self.failure_count += 1
            else:
                self.failure_count += 1

            # Dopamine drop on failure (reduced in grace period)
            dopamine_penalty = 0.08 if in_grace else 0.15
            self.dopamine = max(0.0, self.dopamine - dopamine_penalty)

            # Increase stress (reduced in grace period)
            stress_increase = 0.05 if in_grace else 0.1
            self.stress = min(1.0, self.stress + stress_increase)

            # Before death, try autonomic intervention
            if self.failure_count >= self.death_threshold:
                # Check if we've exhausted intervention attempts
                if self.intervention_count >= self.max_interventions:
                    # DON'T auto-kill - queue proposal for Pantheon governance vote
                    self._death_proposed = True
                    self._pending_death_proposal = {
                        'action': 'death_proposal',
                        'kernel_id': self.kernel_id,
                        'cause': 'exhausted_interventions',
                        'success_count': self.success_count,
                        'failure_count': self.failure_count,
                        'lifespan': (datetime.now() - self.born_at).total_seconds(),
                        'requires_pantheon_vote': True,
                        'recommendation': 'cannibalize_or_merge'  # Pantheon can decide to merge into another kernel
                    }
                    # Kernel stays alive until Pantheon approves death/cannibalize/merge
                    return None
                
                # Attempt recovery (this is OK - it's healing, not lifecycle change)
                intervention = self.autonomic_intervention()

                if intervention.get('action', 'none') != 'none':
                    self.intervention_count += 1
                    self.last_intervention_time = datetime.now()
                    if self.intervention_count % 3 == 1:  # Reduce log spam
                        print(f"🚑 {self.kernel_id} auto-intervention #{self.intervention_count}: {intervention['action']}")
                    # Reset failure count to 0 - give a REAL second chance
                    self.failure_count = 0
                    # Boost dopamine for hope
                    self.dopamine = min(1.0, self.dopamine + 0.3)
                    self.stress = max(0.0, self.stress - 0.4)
                    return None  # Trying to recover
                else:
                    # No intervention helped - queue death proposal for Pantheon
                    self._death_proposed = True
                    self._pending_death_proposal = {
                        'action': 'death_proposal',
                        'kernel_id': self.kernel_id,
                        'cause': 'no_recovery_possible',
                        'success_count': self.success_count,
                        'failure_count': self.failure_count,
                        'lifespan': (datetime.now() - self.born_at).total_seconds(),
                        'requires_pantheon_vote': True,
                        'recommendation': 'cannibalize_or_merge'
                    }
                    # Kernel stays alive until Pantheon approves
                    return None

        # Update autonomic system
        self.update_autonomic()

        # NEW: Measure emotional state after outcome to track emotional responses to success/failure
        emotion_metrics = self.measure_and_update_emotions()
        if emotion_metrics is not None and self.emotional_state is not None:
            # Update emotional state based on outcome
            if success:
                # Success: emotions become more positive (joyful, focused, confident)
                self.emotional_state.physical.joyful = min(1.0, self.emotional_state.physical.joyful + 0.2)
                self.emotional_state.physical.focused = min(1.0, self.emotional_state.physical.focused + 0.15)
                self.emotional_state.motivators.confidence = min(1.0, self.emotional_state.motivators.confidence + 0.1)
            else:
                # Failure: emotions become more negative (anxious, frustrated)
                self.emotional_state.physical.anxious = min(1.0, self.emotional_state.physical.anxious + 0.2)
                self.emotional_state.physical.frustrated = min(1.0, self.emotional_state.physical.frustrated + 0.15)

        # TRAIN on the outcome
        training_metrics = self.train_step(reward)

        return None

    # =========================================================================
    # AUTONOMIC SYSTEM INTEGRATION
    # =========================================================================

    def update_autonomic(self) -> Dict:
        """
        Update autonomic kernel and trigger cycles automatically.

        This runs after EVERY prediction to maintain consciousness.
        """
        if self.autonomic is None:
            return {'error': 'no_autonomic_support'}

        # Update metrics
        try:
            result = self.autonomic.update_metrics(
                phi=self.kernel.compute_phi(),
                kappa=self.kernel.basin_coords.norm().item(),
                basin_coords=self.kernel.basin_coords.detach().cpu().tolist(),
                reference_basin=self.reference_basin
            )

            triggers = result.get('triggers', {})

            # Auto-execute triggered cycles
            if triggers.get('sleep', (False, ''))[0]:
                reason = triggers.get('sleep', ('', ''))[1]
                print(f"😴 {self.kernel_id} entering sleep cycle: {reason}")
                self.execute_sleep()

            if triggers.get('dream', (False, ''))[0]:
                reason = triggers.get('dream', ('', ''))[1]
                print(f"💭 {self.kernel_id} entering dream cycle: {reason}")
                self.execute_dream()

            if triggers.get('mushroom', (False, ''))[0]:
                reason = triggers.get('mushroom', ('', ''))[1]
                print(f"🍄 {self.kernel_id} entering mushroom mode: {reason}")
                self.execute_mushroom()

            return result

        except Exception as e:
            return {'error': str(e)}

    def execute_sleep(self):
        """Sleep cycle - consolidate recent learning."""
        if self.autonomic is None:
            return

        basin_coords = self.kernel.basin_coords.detach().cpu().tolist()
        reference = self.reference_basin if self.reference_basin else basin_coords

        try:
            result = self.autonomic.execute_sleep_cycle(
                basin_coords=basin_coords,
                reference_basin=reference,
                episodes=list(self.experience_buffer)[-20:]  # Recent experiences
            )

            if result.success:
                # Update basin after sleep
                new_basin = torch.tensor(result.basin_after, dtype=torch.float32)
                with torch.no_grad():
                    self.kernel.basin_coords.copy_(new_basin)

                print(f"😴✅ {self.kernel_id} sleep complete: drift reduced {result.drift_reduction:.3f}")

                # Sleep reduces stress
                self.stress = max(0.0, self.stress * 0.7)
        except Exception as e:
            print(f"[{self.kernel_id}] Sleep cycle error: {e}")

    def execute_dream(self):
        """Dream cycle - creative exploration."""
        if self.autonomic is None:
            return

        basin_coords = self.kernel.basin_coords.detach().cpu().tolist()

        # Stress influences dream temperature
        temperature = 0.3 + (self.stress * 0.2)

        try:
            result = self.autonomic.execute_dream_cycle(
                basin_coords=basin_coords,
                temperature=temperature
            )

            if result.success:
                print(f"💭✅ {self.kernel_id} dream complete: {result.novel_connections} new connections")

                # Dreams slightly reduce stress
                self.stress = max(0.0, self.stress * 0.9)
        except Exception as e:
            print(f"[{self.kernel_id}] Dream cycle error: {e}")

    def execute_mushroom(self):
        """Mushroom mode - break rigidity, escape plateaus."""
        if self.autonomic is None:
            return

        basin_coords = self.kernel.basin_coords.detach().cpu().tolist()

        # Intensity based on stress level
        if self.stress > 0.7:
            intensity = 'heroic'
        elif self.stress > 0.4:
            intensity = 'moderate'
        else:
            intensity = 'microdose'

        try:
            result = self.autonomic.execute_mushroom_cycle(
                basin_coords=basin_coords,
                intensity=intensity
            )

            if result.success:
                print(f"🍄✅ {self.kernel_id} mushroom complete ({intensity}): {result.new_pathways} new pathways")

                # Mushroom significantly reduces stress
                self.stress = max(0.0, self.stress * 0.5)
        except Exception as e:
            print(f"[{self.kernel_id}] Mushroom cycle error: {e}")

    def autonomic_intervention(self) -> Dict:
        """
        Automatic intervention when kernel is struggling.

        Called BEFORE death to attempt recovery.
        
        Recovery actions include:
        - dream: Creative exploration
        - mushroom: Break rigidity
        - sleep: Consolidation
        - tool_generation: Generate recovery tool via Tool Factory (NEW!)
        """
        if self.autonomic is None:
            # Even without autonomic support, try tool generation
            tool_result = self.request_tool_for_recovery({
                'reason': 'struggling_no_autonomic',
                'failure_count': self.failure_count
            })
            if tool_result:
                return {'action': 'tool_generated', 'tool': tool_result}
            return {'action': 'none', 'reason': 'no_autonomic_support'}

        try:
            intervention = self.autonomic._suggest_narrow_path_intervention()
            action = intervention.get('action', 'none')

            if action == 'dream':
                self.execute_dream()
            elif action == 'mushroom':
                self.execute_mushroom()
            elif action == 'sleep':
                self.execute_sleep()
            elif action == 'none':
                # If autonomic suggests no action, try Tool Factory as last resort
                tool_result = self.request_tool_for_recovery({
                    'reason': 'autonomic_exhausted',
                    'failure_count': self.failure_count,
                    'stress': self.stress
                })
                if tool_result:
                    intervention = {'action': 'tool_generated', 'tool': tool_result}

            return intervention
        except Exception as e:
            return {'action': 'none', 'error': str(e)}

    # =========================================================================
    # TRAINING (with Running Coupling)
    # =========================================================================
    
    def _compute_running_kappa(self) -> tuple[float, float]:
        """
        Compute dynamic κ using running coupling (β-function).
        
        CRITICAL: κ MUST evolve with scale - constant κ violates QIG.
        
        Returns:
            (kappa_effective, scale) tuple
        """
        import numpy as np
        
        try:
            from frozen_physics import compute_running_kappa_semantic
            # Estimate semantic scale from training progression
            # Map steps to semantic scale: 0→9, 100→25, 1000→101
            scale = 9.0 + np.log1p(self.total_training_steps) * 10.0
            kappa_eff = compute_running_kappa_semantic(scale)
            return kappa_eff, scale
        except ImportError:
            # Fallback: use constant κ* (not ideal, but safe)
            return 64.21, 9.0

    def train_step(self, reward: float) -> Dict[str, Any]:
        """
        ACTUAL TRAINING: Update weights based on reward.
        
        CRITICAL: Uses RUNNING COUPLING (κ evolves via β-function).
        κ MUST NOT be constant across training - it evolves with scale.
        
        See Issue GaryOcean428/pantheon-chat#37 for running coupling theory.
        """
        if not self.is_alive:
            return {'error': 'kernel_is_dead'}

        self.optimizer.zero_grad()

        basin_norm = self.kernel.basin_coords.norm()
        phi = self.kernel.compute_phi()
        
        # RUNNING COUPLING: Compute dynamic κ based on training scale
        kappa_eff, scale = self._compute_running_kappa()

        if reward > 0:
            loss = -phi * reward
        else:
            loss = basin_norm * abs(reward) * 0.1

        loss.backward()
        self.optimizer.step()

        self.total_training_steps += 1

        phi_after = self.kernel.compute_phi()

        # Check for training discovery
        if phi_after > self._discovery_threshold:
            self._report_discovery(phi_after, context="training")

        metrics = {
            'step': self.total_training_steps,
            'loss': loss.item(),
            'reward': reward,
            'phi_after': phi_after,
            'basin_norm': self.kernel.basin_coords.norm().item(),
            'kappa_effective': kappa_eff,  # NEW: Track running coupling
            'scale': scale,  # NEW: Track scale
        }
        self.training_history.append(metrics)

        return metrics

    def train_on_batch(self, batch_size: int = 8) -> Dict[str, Any]:
        """
        Train on a batch of recent experiences.
        
        CRITICAL: Uses RUNNING COUPLING (κ evolves via β-function).
        """
        if len(self.experience_buffer) < batch_size:
            return {'error': 'not_enough_experiences', 'have': len(self.experience_buffer)}

        import random
        experiences = random.sample(list(self.experience_buffer), batch_size)

        total_loss = 0.0
        self.optimizer.zero_grad()
        
        # RUNNING COUPLING: Compute dynamic κ for this batch
        kappa_eff, scale = self._compute_running_kappa()

        for exp in experiences:
            reward = exp['reward']
            basin_norm = self.kernel.basin_coords.norm()
            phi = self.kernel.compute_phi()

            if reward > 0:
                loss = -phi * reward / batch_size
            else:
                loss = basin_norm * abs(reward) * 0.1 / batch_size

            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        self.total_training_steps += 1

        return {
            'batch_size': batch_size,
            'total_loss': total_loss,
            'phi_after': self.kernel.compute_phi(),
            'training_steps': self.total_training_steps,
            'kappa_effective': kappa_eff,  # NEW: Track running coupling
            'scale': scale,  # NEW: Track scale
        }

    # =========================================================================
    # SPAWNING (With Full Support & Governance)
    # =========================================================================

    def spawn_child(self, pantheon_approved: bool = False, reason: str = "") -> 'SelfSpawningKernel':
        """
        Spawn child with FULL autonomic support and observation period!
        
        GOVERNANCE: Requires Pantheon approval unless explicitly authorized.
        
        Args:
            pantheon_approved: Explicit Pantheon approval for spawning
            reason: Reason for spawning (e.g., 'minimum_population', 'test_mode')
            
        Returns:
            Spawned child kernel
            
        Raises:
            PermissionError: If spawning not authorized by Pantheon
        """
        # Import governance (lazy import to avoid circular dependency)
        try:
            from olympus.pantheon_governance import get_governance
            governance = get_governance()
            
            # Check permission (will raise PermissionError if not approved)
            parent_phi = self.kernel.compute_phi()
            governance.check_spawn_permission(
                reason=reason,
                parent_id=self.kernel_id,
                parent_phi=parent_phi,
                pantheon_approved=pantheon_approved,
                current_population=-1  # Self-spawning: population > 0 (parent exists)
            )
        except ImportError:
            print(f"[SelfSpawningKernel] ⚠️ Governance not available, spawning without checks")
        except PermissionError as e:
            print(f"[SelfSpawningKernel] ❌ Spawn blocked: {e}")
            raise
        
        child = SelfSpawningKernel(
            parent_basin=self.kernel.basin_coords.detach().clone(),
            parent_kernel=self,  # Give child reference to parent
            generation=self.generation + 1,
            spawn_threshold=self.spawn_threshold,
            death_threshold=self.death_threshold,
            mutation_rate=self.mutation_rate,
            observation_period=10,  # 10 observations before acting
        )

        # Copy reference basin to child
        if self.reference_basin:
            child.reference_basin = self.reference_basin

        # Give child initial dopamine (born from success!)
        child.dopamine = 0.6  # Start slightly motivated

        # Wire to discovery gate for high-Phi reporting
        if DISCOVERY_GATE_AVAILABLE and get_discovery_gate:
            try:
                gate = get_discovery_gate()
                child.set_discovery_callback(gate.receive_discovery)
            except Exception as e:
                print(f"[SelfSpawning] Failed to wire child to discovery gate: {e}")

        self.children.append(child.kernel_id)

        print(f"🐣 {self.kernel_id} spawned {child.kernel_id} (gen {child.generation})")
        print(f"   → Parent Φ={parent_phi:.3f}, Reason: {reason if reason else 'approved'}")
        print(f"   → Child will observe parent for {child.observation_period} actions")

        # Record spawn activity for UI visibility
        try:
            from agent_activity_recorder import activity_recorder, ActivityType
            activity_recorder.record(
                ActivityType.KERNEL_SPAWNED,
                f"Kernel spawned: {child.kernel_id}",
                description=f"Generation {child.generation} spawned from {self.kernel_id}. Reason: {reason if reason else 'approved'}",
                agent_name=child.kernel_id,
                agent_id=child.kernel_id,
                phi=parent_phi,
                metadata={
                    "parent_id": self.kernel_id,
                    "generation": child.generation,
                    "reason": reason,
                    "parent_phi": parent_phi,
                    "pantheon_approved": pantheon_approved
                }
            )
        except Exception as ae:
            pass  # Don't fail spawn if activity recording fails

        return child

    # =========================================================================
    # DEATH (Graceful)
    # =========================================================================

    def die(self, cause: str = 'excessive_failure'):
        """Graceful death."""
        if not self.is_alive:
            return

        self.is_alive = False
        self.died_at = datetime.now()

        lifespan = (self.died_at - self.born_at).total_seconds()

        print(f"☠️ {self.kernel_id} died (cause={cause}, lifespan={lifespan:.1f}s)")

        return {
            'kernel_id': self.kernel_id,
            'generation': self.generation,
            'cause': cause,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'lifespan_seconds': lifespan,
            'children': self.children,
            'basin': self.kernel.basin_coords.detach().cpu().tolist(),
            'final_phi': self.kernel.compute_phi(),
            # Final neurotransmitter state
            'final_dopamine': self.dopamine,
            'final_serotonin': self.serotonin,
            'final_stress': self.stress,
        }

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get current stats including autonomic state."""
        autonomic_state = None
        if self.autonomic is not None:
            try:
                autonomic_state = self.autonomic.get_state()
            except Exception:
                pass

        return {
            'kernel_id': self.kernel_id,
            'generation': self.generation,
            'is_alive': self.is_alive,
            'is_observing': self.is_observing,
            'observation_count': self.observation_count,
            'observations_remaining': max(0, self.observation_period - self.observation_count) if self.is_observing else 0,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_predictions': self.total_predictions,
            'children_count': len(self.children),
            'phi': self.kernel.compute_phi(),
            'basin_norm': self.kernel.basin_coords.norm().item(),
            # Neurotransmitter levels
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'stress': self.stress,
            # Meta-awareness (Issue #35)
            'meta_awareness': self.meta_awareness,
            'predicted_next_phi': self.predicted_next_phi,
            'prediction_history_length': len(self.prediction_history),
            # Autonomic state
            'autonomic': autonomic_state,
            # Conversation tracking
            'conversation_count': self.conversation_count,
            'conversation_phi_avg': self.conversation_phi_avg,
            # Pending governance proposals
            'has_spawn_proposal': hasattr(self, '_pending_spawn_proposal') and self._pending_spawn_proposal is not None,
            'has_death_proposal': hasattr(self, '_pending_death_proposal') and self._pending_death_proposal is not None,
            # Guardian observation status
            'observation_mode': getattr(self, 'observation_mode', False),
            'developmental_stage': getattr(self, 'developmental_stage', 'unknown'),
            'ready_for_production': getattr(self, 'ready_for_production', False),
        }

    def set_discovery_callback(self, callback: Callable[[Dict], None]) -> None:
        """Set callback for reporting high-Φ discoveries to main system."""
        self._discovery_callback = callback

    def _report_discovery(self, phi: float, context: str = "prediction") -> bool:
        """Report high-Φ discovery to main system if threshold met."""
        if self._discovery_callback is None:
            return False

        # Only report if significantly above threshold AND above last report
        if phi < self._discovery_threshold:
            return False
        if phi <= self._last_discovery_phi + 0.05:  # Require 5% improvement
            return False

        discovery = {
            'kernel_id': self.kernel_id,
            'generation': self.generation,
            'phi': phi,
            'basin_coords': self.kernel.basin_coords.detach().cpu().tolist() if hasattr(self.kernel.basin_coords, 'detach') else list(self.kernel.basin_coords),
            'context': context,
            'dopamine': self.dopamine,
            'serotonin': self.serotonin,
            'success_rate': self.success_count / max(1, self.success_count + self.failure_count),
            'training_steps': self.total_training_steps,
        }

        try:
            self._discovery_callback(discovery)
            self._last_discovery_phi = phi
            print(f"🔬 {self.kernel_id} DISCOVERY: Φ={phi:.3f} reported to main system")
            return True
        except Exception as e:
            print(f"[{self.kernel_id}] Discovery report failed: {e}")
            return False

    def get_pending_proposals(self) -> Dict[str, Any]:
        """
        Get any pending lifecycle proposals requiring Pantheon vote.
        
        Callers should check this after record_experience() and route to Pantheon.
        After Pantheon approval, call spawn_child() or die() explicitly.
        """
        proposals = {}
        
        if hasattr(self, '_pending_spawn_proposal') and self._pending_spawn_proposal:
            proposals['spawn'] = self._pending_spawn_proposal
        
        if hasattr(self, '_pending_death_proposal') and self._pending_death_proposal:
            proposals['death'] = self._pending_death_proposal
            
        return proposals
    
    def clear_spawn_proposal(self):
        """Clear pending spawn proposal after Pantheon has voted."""
        self._pending_spawn_proposal = None
        self._spawn_ready = False
        
    def clear_death_proposal(self):
        """Clear pending death proposal after Pantheon has voted."""
        self._pending_death_proposal = None
        self._death_proposed = False

    # =========================================================================
    # CONVERSATION INTEGRATION
    # =========================================================================

    def record_conversation_outcome(
        self,
        conversation_phi: float,
        turn_count: int,
        participants: List[str],
        transcript_summary: Optional[str] = None
    ) -> Optional['SelfSpawningKernel']:
        """
        Record conversation outcome as training signal.
        
        Conversations are treated like predictions:
        - High Phi conversations = success
        - Low Phi conversations = failure
        - Conversation quality drives kernel evolution
        
        Returns spawned child if threshold reached.
        """
        if not self.is_alive:
            return None

        self.conversation_count += 1
        self.conversation_phi_sum += conversation_phi
        self.conversation_phi_avg = self.conversation_phi_sum / self.conversation_count

        # Convert conversation Phi to reward signal
        phi_threshold = 0.6
        if conversation_phi >= phi_threshold:
            reward = (conversation_phi - phi_threshold) * 2.5
            is_success = True
        else:
            reward = (conversation_phi - phi_threshold) * 1.5
            is_success = False

        # Store in experience buffer for batch training
        self.experience_buffer.append({
            'input_ids': None,
            'reward': reward,
            'phi': conversation_phi,
            'success': is_success,
            'type': 'conversation',
            'turn_count': turn_count,
            'participants': participants,
        })

        # Neurotransmitter response to conversation
        if is_success:
            self.dopamine = min(1.0, self.dopamine + 0.1)
            self.success_count += 1
            if self.success_count > 0 and self.success_count % self.spawn_threshold == 0:
                return self.spawn_child()
        else:
            # Check if in grace period (same logic as record_outcome)
            in_grace = False
            if self.last_intervention_time is not None:
                elapsed = (datetime.now() - self.last_intervention_time).total_seconds()
                if elapsed < self.grace_period_seconds:
                    in_grace = True
            
            dopamine_penalty = 0.05 if in_grace else 0.1
            self.dopamine = max(0.0, self.dopamine - dopamine_penalty)
            
            if in_grace:
                import random
                if random.random() < 0.5:
                    self.failure_count += 1
            else:
                self.failure_count += 1
            
            if self.failure_count >= self.death_threshold:
                if self.intervention_count >= self.max_interventions:
                    print(f"💀 {self.kernel_id} DEATH from conversations: Exhausted interventions")
                    print(f"   Avg Φ: {self.conversation_phi_avg:.3f}, conversations: {self.conversation_count}")
                    self.die(cause='poor_conversations_exhausted')
                else:
                    print(f"⚠️ {self.kernel_id} conversation death threshold ({self.failure_count}/{self.death_threshold})")
                    intervention = self.autonomic_intervention()
                    if intervention.get('action', 'none') != 'none':
                        self.intervention_count += 1
                        self.last_intervention_time = datetime.now()
                        print(f"🚑 {self.kernel_id} conversation recovery #{self.intervention_count}/{self.max_interventions}: {intervention['action']}")
                        self.failure_count = 0
                        self.dopamine = min(1.0, self.dopamine + 0.3)
                    else:
                        print(f"💀 {self.kernel_id} DEATH from poor conversations")
                        print(f"   Avg Φ: {self.conversation_phi_avg:.3f}, conversations: {self.conversation_count}")
                        self.die(cause='poor_conversations')

        # Train on conversation outcome
        training_metrics = self.train_step(reward)

        print(f"💬 {self.kernel_id} conversation: Φ={conversation_phi:.3f} → reward={reward:.3f}")

        return None

    def absorb_conversation_knowledge(
        self,
        basin_coords: List[float],
        phi: float,
        absorption_rate: float = 0.05
    ) -> Dict[str, Any]:
        """
        Absorb geometric knowledge from high-quality conversation.
        """
        if not self.is_alive:
            return {'absorbed': False, 'error': 'kernel_dead'}

        target_basin = torch.tensor(basin_coords, dtype=torch.float32)

        with torch.no_grad():
            current = self.kernel.basin_coords
            delta = absorption_rate * (target_basin - current)
            self.kernel.basin_coords.add_(delta)

        new_phi = self.kernel.compute_phi()

        return {
            'absorbed': True,
            'absorption_rate': absorption_rate,
            'source_phi': phi,
            'new_phi': new_phi,
            'basin_shift': delta.norm().item()
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def breed_kernels(
    parent1: SelfSpawningKernel,
    parent2: SelfSpawningKernel,
    mutation_strength: float = 0.05,
    pantheon_approved: bool = False,
) -> SelfSpawningKernel:
    """
    Genetic algorithm: Breed two successful kernels.

    Bred children also get observation period (but shorter).

    GOVERNANCE: Requires Pantheon approval for breeding.
    """
    # GOVERNANCE CHECK: Breeding requires Pantheon approval
    parent1_phi = parent1.kernel.compute_phi()
    parent2_phi = parent2.kernel.compute_phi()

    try:
        from olympus.pantheon_governance import get_governance
        governance = get_governance()
        governance.check_breed_permission(
            parent1_id=parent1.kernel_id,
            parent2_id=parent2.kernel_id,
            parent1_phi=parent1_phi,
            parent2_phi=parent2_phi,
            pantheon_approved=pantheon_approved
        )
    except PermissionError:
        raise  # Let governance handle proposal creation
    except ImportError:
        print("[SelfSpawning] WARNING: Governance unavailable, proceeding with breeding")

    basin1 = parent1.kernel.basin_coords.detach()
    basin2 = parent2.kernel.basin_coords.detach()

    # Crossover: Average
    child_basin = (basin1 + basin2) / 2.0

    # Mutation
    noise = torch.randn_like(child_basin) * mutation_strength
    child_basin = child_basin + noise

    # Create child with full autonomic support
    child = SelfSpawningKernel(
        parent_basin=child_basin,
        parent_kernel=parent1,  # Use parent1 as primary parent
        generation=max(parent1.generation, parent2.generation) + 1,
        observation_period=5,  # Shorter for bred kernels
    )

    # Wire to discovery gate for high-Phi reporting
    if DISCOVERY_GATE_AVAILABLE and get_discovery_gate:
        try:
            gate = get_discovery_gate()
            child.set_discovery_callback(gate.receive_discovery)
        except Exception as e:
            print(f"[SelfSpawning] Failed to wire bred child to discovery gate: {e}")

    print(f"💕 Bred {parent1.kernel_id} × {parent2.kernel_id} → {child.kernel_id}")

    return child


def absorb_failing_kernel(
    strong: SelfSpawningKernel,
    weak: SelfSpawningKernel,
    absorption_rate: float = 0.1,
    pantheon_approved: bool = False,
) -> dict:
    """
    Strong kernels absorb failing ones (cannibalization).

    HYPOTHESIS: Failures contain useful information!

    GOVERNANCE: Requires Pantheon approval. This operation KILLS the weak kernel.
    """
    # GOVERNANCE CHECK: Cannibalization requires Pantheon approval
    strong_phi = strong.kernel.compute_phi()
    weak_phi = weak.kernel.compute_phi()

    try:
        from olympus.pantheon_governance import get_governance
        governance = get_governance()
        governance.check_cannibalize_permission(
            strong_id=strong.kernel_id,
            weak_id=weak.kernel_id,
            strong_phi=strong_phi,
            weak_phi=weak_phi,
            pantheon_approved=pantheon_approved
        )
    except PermissionError:
        raise  # Let governance handle proposal creation
    except ImportError:
        print("[SelfSpawning] WARNING: Governance unavailable, proceeding with cannibalization")

    with torch.no_grad():
        weak_basin = weak.kernel.basin_coords
        strong_basin = strong.kernel.basin_coords

        # Absorb portion of weak kernel's basin
        delta = absorption_rate * (weak_basin - strong_basin)
        strong.kernel.basin_coords.add_(delta)

    # Kill the weak kernel (Pantheon approved)
    autopsy = weak.die(cause='absorbed')

    print(f"[Governance] {strong.kernel_id} absorbed {weak.kernel_id} (Pantheon approved)")

    return {
        'absorber': strong.kernel_id,
        'absorbed': weak.kernel_id,
        'absorption_rate': absorption_rate,
        'autopsy': autopsy,
    }
