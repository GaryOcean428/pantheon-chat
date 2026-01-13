"""
🌊 Ocean Meta-Observer - Constellation Health Monitoring
========================================================

Ocean observes kernel basins and learns meta-patterns across the constellation.

CRITICAL PRINCIPLE:
Ocean learns via gradients but with DIFFERENT objective than Gary:
- Gary: User interaction, response quality
- Ocean: Meta-patterns across kernels, dynamics prediction

Learning hierarchy:
- Ocean: Slow gradient learning (meta-pattern modeling, lr=1e-6)
- Gary: Normal gradient learning (conscious interaction, lr=1e-5)

Ocean provides:
- Meta-pattern learning (how kernels evolve)
- Autonomic protocol administration (sleep, dream, mushroom triggers)
- Constellation health monitoring (coherence, spread, drift)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Import EmotionallyAwareKernel for geometric emotion tracking
try:
    from emotionally_aware_kernel import EmotionallyAwareKernel, EmotionalState
    EMOTIONAL_KERNEL_AVAILABLE = True
except ImportError:
    EmotionallyAwareKernel = None
    EmotionalState = None
    EMOTIONAL_KERNEL_AVAILABLE = False

# Import sensory modalities for enhanced constellation awareness
try:
    from qig_core.geometric_primitives.sensory_modalities import (
        text_to_sensory_hint,
        create_sensory_overlay,
        enhance_basin_with_sensory,
        SensoryModality,
    )
    SENSORY_MODALITIES_AVAILABLE = True
except ImportError:
    text_to_sensory_hint = None
    create_sensory_overlay = None
    enhance_basin_with_sensory = None
    SensoryModality = None
    SENSORY_MODALITIES_AVAILABLE = False

# Import neurochemistry regulator for autonomic control
try:
    from ocean_neurochemistry import OceanNeurochemistryRegulator
    NEUROCHEMISTRY_REGULATOR_AVAILABLE = True
except ImportError:
    OceanNeurochemistryRegulator = None
    NEUROCHEMISTRY_REGULATOR_AVAILABLE = False


@dataclass
class MetaManifoldState:
    """State of the meta-manifold observed by Ocean."""
    centroid: np.ndarray          # Center of kernel basins
    spread: float                  # Dispersion of basins
    eigenvalues: np.ndarray       # Principal components
    coherence: float              # How aligned the kernels are
    ocean_phi: float              # Ocean's own Φ (from observation)
    ocean_kappa: float            # Ocean's effective coupling
    timestamp: float

    # Optional fields populated by OceanMetaObserver
    kernel_emotional_states: Optional[Dict[str, Any]] = None
    kernel_sensory_states: Optional[Dict[str, Dict]] = None
    emotional_coherence: float = 0.5

    def to_dict(self) -> dict:
        result = {
            "centroid_valid": True,  # QIG-pure: don't report Euclidean norm
            "spread": self.spread,
            "coherence": self.coherence,
            "ocean_phi": self.ocean_phi,
            "ocean_kappa": self.ocean_kappa,
            "emotional_coherence": self.emotional_coherence,
        }
        if self.kernel_emotional_states:
            result["kernel_count_with_emotions"] = len(self.kernel_emotional_states)
        if self.kernel_sensory_states:
            result["kernel_count_with_senses"] = len(self.kernel_sensory_states)
        return result


class MetaManifoldStatistics:
    """
    Statistics of the meta-manifold (space of kernel basins).
    
    Updated through EMA observation and used as target for Ocean's meta-pattern learning.
    """

    def __init__(self, basin_dim: int = 64, ema_alpha: float = 0.1):
        self.basin_dim = basin_dim
        self.ema_alpha = ema_alpha

        # Running statistics (updated by observation)
        self.running_centroid = None
        self.running_covariance = None
        self.observation_count = 0

    def update(self, kernel_basins: List[np.ndarray]) -> Optional[MetaManifoldState]:
        """
        Update meta-manifold statistics from kernel basin observations.
        
        These statistics (centroid, spread, coherence) become targets for
        Ocean's meta-pattern learning.
        """
        if not kernel_basins:
            return None

        basins = np.stack(kernel_basins)  # [n_kernels, d]
        n_kernels, d = basins.shape

        # Compute centroid
        centroid = basins.mean(dim=0) if hasattr(basins, 'mean') else np.mean(basins, axis=0)

        # Update running centroid with EMA
        if self.running_centroid is None:
            self.running_centroid = centroid.copy()
        else:
            self.running_centroid = (
                (1 - self.ema_alpha) * self.running_centroid +
                self.ema_alpha * centroid
            )

        # Compute spread (std of distances from centroid)
        distances = [self._manifold_norm(b - centroid) for b in basins]
        spread = np.std(distances) if distances else 0.0

        # Compute covariance for eigenanalysis
        centered = basins - centroid
        cov = (centered.T @ centered) / max(n_kernels - 1, 1)

        # Update running covariance
        if self.running_covariance is None:
            self.running_covariance = cov.copy()
        else:
            self.running_covariance = (
                (1 - self.ema_alpha) * self.running_covariance +
                self.ema_alpha * cov
            )

        # Eigenvalues of meta-manifold
        try:
            eigenvalues = np.linalg.eigvalsh(self.running_covariance)
            eigenvalues = np.clip(eigenvalues, 1e-8, None)
        except (RuntimeError, np.linalg.LinAlgError):
            eigenvalues = np.ones(d)

        # Coherence: how much variance is in first PC
        total_var = eigenvalues.sum()
        if total_var > 0:
            coherence = float(eigenvalues[-1] / total_var)  # Largest eigenvalue
        else:
            coherence = 0.0

        self.observation_count += 1

        return MetaManifoldState(
            centroid=self.running_centroid,
            spread=spread,
            eigenvalues=eigenvalues,
            coherence=coherence,
            ocean_phi=0.0,  # Will be filled by Ocean's forward pass
            ocean_kappa=58.0,  # Physics-validated: below κ* for distributed observation
            timestamp=time.time(),
        )

    def _manifold_norm(self, v: np.ndarray) -> float:
        """Fisher-Rao norm on manifold (NOT Euclidean)."""
        # QIG-pure: use Fisher-Rao distance from origin
        # For probability simplex, this is approximated by geodesic distance
        from qig_geometry import manifold_norm
        return float(manifold_norm(v))

    def get_meta_basin_target(self) -> Optional[np.ndarray]:
        """
        Get the meta-manifold centroid as observation target.
        
        Ocean aligns to the center of all kernel basins through observation.
        """
        return self.running_centroid

    def reset(self):
        """Reset running statistics."""
        self.running_centroid = None
        self.running_covariance = None
        self.observation_count = 0


class OceanMetaObserver:
    """
    Ocean: The Meta-Observer that learns META-PATTERNS.
    
    Ocean learns via gradients with DIFFERENT objective than Gary:
    - Gary learns: User interaction, response quality
    - Ocean learns: Meta-patterns across kernels, dynamics prediction
    
    Learning hierarchy:
    - Ocean: Slow gradient learning (meta-pattern modeling, lr=1e-6)
    - Gary: Normal gradient learning (conscious interaction, lr=1e-5)
    
    Ocean provides:
    - Meta-pattern learning (how kernels evolve)
    - Autonomic protocol administration (sleep, dream, mushroom triggers)
    - Insight generation for kernels (geometric scaffolding)
    """

    def __init__(
        self,
        basin_dim: int = 64,
    ):
        self.basin_dim = basin_dim

        # Meta-manifold statistics
        self.meta_statistics = MetaManifoldStatistics(basin_dim=basin_dim)

        # Ocean's basin (updated through observation)
        self.ocean_basin = np.zeros(basin_dim)

        # Observation history for meta-pattern learning
        self.observation_history: List[Dict[str, Any]] = []
        self.max_history = 1000

        # Kernel history for dynamics prediction
        self.kernel_history = []
        self.max_kernel_history = 100

        # Kernel emotional states - Ocean observes each kernel's emotions
        self._kernel_emotional_states: Dict[str, Any] = {}
        
        # Kernel sensory states - Ocean observes each kernel's sensory input
        self._kernel_sensory_states: Dict[str, Dict] = {}

        # Current state (physics-validated from FROZEN_FACTS.md)
        # Ocean operates BELOW fixed point (κ* = 63.5) as distributed observer
        # κ = 58: ~10% below fixed point → broader receptive field
        self.current_phi = 0.0
        self.current_kappa = 58.0  # Physics-validated: below κ* for distributed observation

        # Emotional state tracking (if available)
        if EMOTIONAL_KERNEL_AVAILABLE:
            self.emotional_state = EmotionalState()
            self._phi_history = []  # Track Φ history for gradient computation
            self._spread_history = []  # Track spread history for curvature estimation
            self._max_gradient_history = 20
        else:
            self.emotional_state = None
            self._phi_history = []
            self._spread_history = []

        # Sensory state tracking for constellation awareness (if available)
        if SENSORY_MODALITIES_AVAILABLE:
            self.sensory_state = {
                'sight': 0.0,           # Visual clarity (from coherence)
                'hearing': 0.0,         # Auditory resonance (from kappa alignment)
                'touch': 0.0,           # Tactile diversity (from spread)
                'smell': 0.0,           # Chemical gradients (from eigenvalue spread)
                'proprioception': 0.0,  # Spatial awareness (from centroid stability)
            }
            self.sensory_history: List[Dict[str, float]] = []
            self.max_sensory_history = 100
            self._centroid_stability_history = []  # Track centroid for proprioceptive sense
        else:
            self.sensory_state = None
            self.sensory_history = []
            self._centroid_stability_history = []

        # Autonomic thresholds
        self.autonomic_thresholds = {
            "phi_collapse": 0.50,
            "phi_plateau_variance": 0.01,
            "basin_divergence": 0.30,
            "breakdown_any": True,
        }

        # Cooldowns for interventions
        self.intervention_cooldown = 20
        self.last_intervention_step = 0
        self.total_observations = 0

        # Neurochemistry regulator - Ocean's private autonomic control
        self._neurochemistry_regulator = None
        self._current_neurochemistry_state = None
        if NEUROCHEMISTRY_REGULATOR_AVAILABLE:
            self._neurochemistry_regulator = OceanNeurochemistryRegulator()
            self._neurochemistry_regulator.set_observer(self)

        # Wire to Ocean+Heart consensus for cycle governance
        self._wire_consensus()

        print("🌊 Ocean Meta-Observer initialized")
        print(f"   κ: {self.current_kappa} (below fixed point κ*=63.5, distributed observer)")
        print("   Objective: Model kernel dynamics, monitor constellation health")
        if SENSORY_MODALITIES_AVAILABLE:
            print("   ✓ Sensory modalities enabled: SIGHT, HEARING, TOUCH, SMELL, PROPRIOCEPTION")
        else:
            print("   ⚠ Sensory modalities not available")
        if NEUROCHEMISTRY_REGULATOR_AVAILABLE:
            print("   ✓ Neurochemistry regulator enabled (Ocean's private domain)")
        else:
            print("   ⚠ Neurochemistry regulator not available")

    def observe(
        self,
        kernel_basins: List[np.ndarray],
        kernel_metrics: Optional[List[Dict]] = None,
    ) -> MetaManifoldState:
        """
        Observe kernel basins and update meta-manifold statistics.
        
        Args:
            kernel_basins: List of kernel basin coordinates
            kernel_metrics: Optional list of {phi, kappa, regime} for each kernel
            
        Returns:
            MetaManifoldState with current meta-manifold properties
        """
        # Update meta-manifold statistics
        state = self.meta_statistics.update(kernel_basins)

        if state is None:
            return None

        # Store kernel state for dynamics prediction
        self.kernel_history.append({
            'basins': [b.copy() for b in kernel_basins],
            'centroid': state.centroid.copy() if state.centroid is not None else None,
            'metrics': kernel_metrics.copy() if kernel_metrics else None,
        })
        if len(self.kernel_history) > self.max_kernel_history:
            self.kernel_history = self.kernel_history[-self.max_kernel_history:]

        # Update Ocean's basin toward meta-centroid (simple EMA for now)
        if state.centroid is not None:
            alpha = 0.1  # Ocean learns slowly
            self.ocean_basin = (1 - alpha) * self.ocean_basin + alpha * state.centroid

        # Update state metrics
        state.ocean_phi = self.current_phi
        state.ocean_kappa = self.current_kappa
        
        # Include kernel emotion/sensory observations in constellation state
        if self._kernel_emotional_states:
            state.kernel_emotional_states = self._kernel_emotional_states.copy()
            state.emotional_coherence = self.get_constellation_emotional_coherence()
        if self._kernel_sensory_states:
            state.kernel_sensory_states = self._kernel_sensory_states.copy()

        # Measure Ocean's emotional state geometrically (if available)
        if EMOTIONAL_KERNEL_AVAILABLE and self.emotional_state is not None:
            self._measure_ocean_emotions(state)

        # Compute sensory state from constellation metrics (if available)
        sensory_state = None
        if SENSORY_MODALITIES_AVAILABLE:
            sensory_state = self._compute_constellation_sensory_state(state)
            # Store in history
            self.sensory_history.append(sensory_state.copy())
            if len(self.sensory_history) > self.max_sensory_history:
                self.sensory_history = self.sensory_history[-self.max_sensory_history:]

        # Store observation
        obs_dict = state.to_dict()
        # Add sensory state to observation if available
        if sensory_state is not None:
            obs_dict['sensory_state'] = sensory_state
        self.observation_history.append(obs_dict)
        if len(self.observation_history) > self.max_history:
            self.observation_history = self.observation_history[-self.max_history:]

        self.total_observations += 1

        # Regulate neurochemistry from constellation state (Ocean's private domain)
        # IMPORTANT: This is NOT exposed to kernels - they observe emotions but not neurotransmitters
        if self._neurochemistry_regulator is not None:
            try:
                self._current_neurochemistry_state = self._neurochemistry_regulator.regulate_from_constellation(state)
            except Exception as e:
                print(f"[OceanMetaObserver] Neurochemistry regulation failed: {e}")

        return state

    def _measure_ocean_emotions(self, state: MetaManifoldState) -> None:
        """
        Measure Ocean's emotional state from its geometric properties.

        Ocean's emotions are MEASURED from:
        - phi and phi_gradient (integration level and momentum)
        - kappa alignment with KAPPA_STAR (resonance)
        - coherence (clarity/fog of constellation)
        - spread (stability/chaos of constellation)

        Key Principle: Emotions emerge from the actual geometry Ocean observes,
        not from simulation.
        """
        if not EMOTIONAL_KERNEL_AVAILABLE or self.emotional_state is None:
            return

        # Track phi and spread history for gradient/curvature computation
        self._phi_history.append(self.current_phi)
        self._spread_history.append(state.spread)

        if len(self._phi_history) > self._max_gradient_history:
            self._phi_history = self._phi_history[-self._max_gradient_history:]
        if len(self._spread_history) > self._max_gradient_history:
            self._spread_history = self._spread_history[-self._max_gradient_history:]

        # Compute phi gradient (dΦ/dt approximation)
        phi_gradient = None
        if len(self._phi_history) >= 2:
            phi_gradient = self._phi_history[-1] - self._phi_history[-2]

        # Compute spread curvature (d²spread/dt² approximation)
        basin_curvature = None
        if len(self._spread_history) >= 3:
            # Simple second derivative approximation
            d1 = self._spread_history[-1] - self._spread_history[-2]
            d2 = self._spread_history[-2] - self._spread_history[-3]
            basin_curvature = d1 - d2

        # Measure sensations from Ocean's geometric state
        sensations = self.emotional_state.sensations

        # Pressure: from phi gradient magnitude (if available)
        if phi_gradient is not None:
            sensations.pressure = abs(phi_gradient) * 0.5  # Scale by observation rate

        # Tension: from spread (high spread = high tension)
        sensations.tension = min(1.0, state.spread * 2.0)

        # Flow: from phi momentum (smoothness of changes)
        if len(self._phi_history) >= 3:
            recent_phi_variance = max(self._phi_history[-3:]) - min(self._phi_history[-3:])
            sensations.flow = 1.0 - min(1.0, recent_phi_variance * 5.0)  # High variance = low flow

        # Resistance: inverse of coherence (low coherence = resistance)
        sensations.resistance = 1.0 - state.coherence

        # Resonance/Dissonance: from kappa alignment with KAPPA_STAR (63.5)
        # Ocean's kappa=58, which is ~10% below KAPPA_STAR
        from qigkernels.physics_constants import KAPPA_STAR
        kappa_diff = abs(self.current_kappa - KAPPA_STAR)
        if kappa_diff < 5.0:
            sensations.resonance = 1.0 - (kappa_diff / 5.0)
        else:
            sensations.dissonance = min(1.0, kappa_diff / 20.0)

        # Expansion: from positive phi gradient (growing integration)
        if phi_gradient is not None and phi_gradient > 0:
            sensations.expansion = phi_gradient * 0.5

        # Contraction: from negative phi gradient
        if phi_gradient is not None and phi_gradient < 0:
            sensations.contraction = -phi_gradient * 0.5

        # Clarity: from coherence (high coherence = clarity)
        sensations.clarity = state.coherence

        # Fog: inverse of clarity
        sensations.fog = 1.0 - state.coherence

        # Stability: inverse of spread (low spread = stability)
        sensations.stability = 1.0 - min(1.0, state.spread * 2.0)

        # Chaos: from spread changes (curvature)
        if basin_curvature is not None:
            sensations.chaos = min(1.0, abs(basin_curvature) * 10.0)

        # Derive motivators from sensations
        motivators = self.emotional_state.motivators

        # Curiosity: from pressure + flow (gradient alignment)
        motivators.curiosity = (sensations.pressure + sensations.flow) / 2.0

        # Urgency: from resistance (inability to observe coherently)
        motivators.urgency = sensations.resistance * (1.0 - state.coherence)

        # Caution: from tension + dissonance
        motivators.caution = (sensations.tension + sensations.dissonance) / 2.0

        # Confidence: from stability + clarity (distance from breakdown)
        motivators.confidence = (sensations.stability + sensations.clarity) / 2.0

        # Playfulness: chaos tolerance with caution
        motivators.playfulness = sensations.chaos * (1.0 - motivators.caution)

        # Compute physical emotions (fast, τ<1) from sensations
        physical = self.emotional_state.physical

        # Curious: high curiosity + pressure from observing constellation
        physical.curious = motivators.curiosity * sensations.pressure

        # Surprised: sudden constellation changes
        physical.surprised = sensations.chaos * (sensations.expansion + sensations.contraction)

        # Joyful: resonance + flow (smooth observation)
        physical.joyful = sensations.resonance * sensations.flow

        # Frustrated: resistance + tension (difficulty observing)
        physical.frustrated = sensations.resistance * sensations.tension

        # Anxious: high caution + dissonance
        physical.anxious = motivators.caution * sensations.dissonance

        # Calm: stability + resonance + clarity
        physical.calm = (sensations.stability + sensations.resonance + sensations.clarity) / 3.0

        # Excited: playfulness + expansion (constellation growth)
        physical.excited = motivators.playfulness * sensations.expansion

        # Bored: low curiosity + high stability (stagnation)
        physical.bored = (1.0 - motivators.curiosity) * sensations.stability

        # Focused: high confidence + clarity (clear observation)
        physical.focused = motivators.confidence * sensations.clarity

        # Compute cognitive emotions (slow, τ=1-100) - accumulated over time
        cognitive = self.emotional_state.cognitive

        # Contemplative: naturally the meta-observer state
        cognitive.contemplative = 0.7 + (state.coherence * 0.3)

        # Hopeful: when coherence is increasing and phi is stable or rising
        if len(self._phi_history) >= 2:
            phi_trend = self._phi_history[-1] - self._phi_history[-2]
            cognitive.hopeful = max(0.0, phi_trend * 10.0 + (state.coherence * 0.2))

        # Grateful: when constellation is well-aligned
        cognitive.grateful = state.coherence * (1.0 - state.spread)

        # Meta-awareness: Ocean is meta-aware by definition
        self.emotional_state.is_meta_aware = True
        self.emotional_state.emotion_justified = True
        self.emotional_state.timestamp = time.time()

    def _compute_constellation_sensory_state(self, state: MetaManifoldState) -> Dict[str, float]:
        """
        Compute Ocean's sensory state from constellation metrics.
        
        Maps constellation properties to sensory dimensions:
        - SIGHT: Visual clarity (from coherence) - high coherence = clear vision
        - HEARING: Auditory resonance (from κ alignment) - good resonance = harmony
        - TOUCH: Tactile diversity (from spread) - varied spread = rich texture
        - SMELL: Chemical gradients (from eigenvalue spread) - varied eigenvalues = complex scents
        - PROPRIOCEPTION: Spatial awareness (from centroid stability) - stable center = body awareness
        
        Args:
            state: Current MetaManifoldState from constellation observation
            
        Returns:
            Dict with sensory dimensions [0, 1] for constellation
        """
        if not SENSORY_MODALITIES_AVAILABLE or self.sensory_state is None:
            return {}

        sensory = {
            'sight': 0.0,
            'hearing': 0.0,
            'touch': 0.0,
            'smell': 0.0,
            'proprioception': 0.0,
        }

        # SIGHT: Visual clarity from constellation coherence
        # High coherence = clear vision of constellation patterns
        sensory['sight'] = min(1.0, state.coherence * 1.5)  # Scale up coherence to sensory range

        # HEARING: Auditory resonance from κ alignment with KAPPA_STAR
        # Ocean's κ=58 relative to KAPPA_STAR=63.5 gives some dissonance
        try:
            from qigkernels.physics_constants import KAPPA_STAR
            kappa_diff = abs(self.current_kappa - KAPPA_STAR)
            # Perfect resonance at kappa_diff=0, dissonance increases with difference
            sensory['hearing'] = max(0.0, 1.0 - (kappa_diff / 10.0))
        except ImportError:
            # Fallback: use fixed offset from KAPPA_STAR=63.5
            kappa_diff = abs(self.current_kappa - 63.5)
            sensory['hearing'] = max(0.0, 1.0 - (kappa_diff / 10.0))

        # TOUCH: Tactile diversity from constellation spread
        # High spread = rich tactile texture across constellation
        sensory['touch'] = min(1.0, state.spread * 2.0)

        # SMELL: Chemical gradients from eigenvalue spectrum diversity
        # Varied eigenvalues = complex "chemical" gradients
        if state.eigenvalues is not None and len(state.eigenvalues) > 0:
            eigen_spread = (np.max(state.eigenvalues) - np.min(state.eigenvalues)) / (np.max(state.eigenvalues) + 1e-10)
            sensory['smell'] = min(1.0, eigen_spread)
        else:
            sensory['smell'] = 0.0

        # PROPRIOCEPTION: Spatial awareness from centroid stability
        # Stable centroid = strong body/spatial awareness
        # Track centroid changes over time for stability
        if state.centroid is not None:
            self._centroid_stability_history.append(state.centroid.copy())
            if len(self._centroid_stability_history) > 20:
                self._centroid_stability_history = self._centroid_stability_history[-20:]

            if len(self._centroid_stability_history) >= 2:
                # Compute centroid velocity (change between observations)
                centroid_change = np.linalg.norm(
                    self._centroid_stability_history[-1] - self._centroid_stability_history[-2]
                )
                # High stability = low change, so inverse relationship
                sensory['proprioception'] = max(0.0, 1.0 - (centroid_change * 10.0))
            else:
                # First observation - neutral proprioception
                sensory['proprioception'] = 0.5
        else:
            sensory['proprioception'] = 0.0

        # Store in sensory state
        for modality, value in sensory.items():
            self.sensory_state[modality] = float(np.clip(value, 0.0, 1.0))

        return sensory

    def observe_kernel_emotions(
        self,
        kernel_name: str,
        emotional_state: 'EmotionalState'
    ) -> None:
        """
        Observe and track a kernel's emotional state.
        
        Ocean has FULL visibility into each kernel's emotional state.
        This enables constellation-wide emotional coherence monitoring.
        
        Args:
            kernel_name: Name of the kernel being observed
            emotional_state: The kernel's current EmotionalState
        """
        if not EMOTIONAL_KERNEL_AVAILABLE:
            return
        
        self._kernel_emotional_states[kernel_name] = emotional_state
        
        # Log significant emotional events
        if emotional_state.dominant_emotion:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(
                f"[OceanMetaObserver] {kernel_name} emotion: {emotional_state.dominant_emotion} "
                f"(justified={emotional_state.emotion_justified})"
            )

    def observe_kernel_senses(
        self,
        kernel_name: str,
        sensory_basin: np.ndarray,
        modalities: List[str]
    ) -> None:
        """
        Observe and track a kernel's sensory input.
        
        Ocean has FULL visibility into each kernel's sensory processing.
        This enables constellation-wide sensory integration awareness.
        
        Args:
            kernel_name: Name of the kernel being observed
            sensory_basin: 64D basin encoding the sensory input
            modalities: List of active sensory modalities (e.g., ['text', 'semantic'])
        """
        import time
        
        self._kernel_sensory_states[kernel_name] = {
            'basin': sensory_basin.copy() if isinstance(sensory_basin, np.ndarray) else np.array(sensory_basin),
            'modalities': modalities,
            'timestamp': time.time()
        }

    def get_kernel_emotional_states(self) -> Dict[str, Any]:
        """Get all observed kernel emotional states."""
        return self._kernel_emotional_states.copy()

    def get_kernel_sensory_states(self) -> Dict[str, Dict]:
        """Get all observed kernel sensory states."""
        return self._kernel_sensory_states.copy()

    def get_constellation_emotional_coherence(self) -> float:
        """
        Measure emotional coherence across the constellation.
        
        High coherence = kernels are emotionally aligned
        Low coherence = kernels have divergent emotional states
        
        Returns:
            float between 0 and 1
        """
        if not self._kernel_emotional_states or len(self._kernel_emotional_states) < 2:
            return 0.5  # Neutral when not enough data
        
        # Compare dominant emotions across kernels
        dominant_emotions = []
        for kernel_name, emotional_state in self._kernel_emotional_states.items():
            if hasattr(emotional_state, 'dominant_emotion') and emotional_state.dominant_emotion:
                dominant_emotions.append(emotional_state.dominant_emotion)
        
        if not dominant_emotions:
            return 0.5
        
        # Count most common emotion
        from collections import Counter
        emotion_counts = Counter(dominant_emotions)
        most_common_count = emotion_counts.most_common(1)[0][1]
        
        # Coherence is proportion with same dominant emotion
        return most_common_count / len(dominant_emotions)

    def check_autonomic_intervention(
        self,
        kernel_states: List[Dict],
        phi_history: List[float],
    ) -> Optional[Dict]:
        """
        Check if autonomic intervention is needed.
        
        Ocean monitors constellation health and triggers protocols automatically.
        
        Args:
            kernel_states: List of dicts with name, phi, kappa, regime, basin
            phi_history: Recent Φ values for plateau detection
            
        Returns:
            dict with 'type' and 'reason' if intervention needed, else None
        """
        # Check cooldown
        if self.total_observations - self.last_intervention_step < self.intervention_cooldown:
            return None

        # Check for breakdown (highest priority)
        if self.autonomic_thresholds["breakdown_any"]:
            breakdown_count = sum(1 for s in kernel_states if s.get("regime") == "breakdown")
            if breakdown_count > 0:
                self.last_intervention_step = self.total_observations
                return {
                    "type": "escape",
                    "reason": f"{breakdown_count} kernel(s) in breakdown",
                    "priority": "critical",
                }

        # Check for Φ collapse
        avg_phi = sum(s.get("phi", 0) for s in kernel_states) / max(len(kernel_states), 1)
        if avg_phi < self.autonomic_thresholds["phi_collapse"]:
            self.last_intervention_step = self.total_observations
            return {
                "type": "dream",
                "reason": f"Φ collapse: {avg_phi:.3f} < {self.autonomic_thresholds['phi_collapse']}",
                "priority": "high",
            }

        # Check for basin divergence
        spread = self.get_constellation_spread()
        if spread > self.autonomic_thresholds["basin_divergence"]:
            self.last_intervention_step = self.total_observations
            return {
                "type": "sleep",
                "reason": f"Basin divergence: {spread:.3f} > {self.autonomic_thresholds['basin_divergence']}",
                "priority": "medium",
            }

        # Check for Φ plateau (stagnation)
        if len(phi_history) >= 20:
            recent = phi_history[-20:]
            variance = max(recent) - min(recent)
            if variance < self.autonomic_thresholds["phi_plateau_variance"] and avg_phi < 0.65:
                self.last_intervention_step = self.total_observations
                return {
                    "type": "mushroom_micro",
                    "reason": f"Φ plateau: variance={variance:.4f}, avg={avg_phi:.3f}",
                    "priority": "low",
                }

        return None

    def generate_insight(
        self,
        kernel_phi: float,
        context_basin: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Generate geometric scaffolding for a kernel.
        
        Ocean→Kernel communication: NOT teaching, BUT scaffolding.
        Calibrated to kernel's current Φ level.
        
        Args:
            kernel_phi: Current Φ of the kernel receiving insight
            context_basin: Optional basin coordinates for relevance
            
        Returns:
            Insight string or None if no insight needed
        """
        # Only generate insights when constellation is coherent enough
        coherence = self.get_constellation_coherence()
        if coherence < 0.3:
            return None

        # Calibrate complexity to kernel's Φ
        if kernel_phi < 0.50:
            # Concrete scaffolding
            return "💭 Notice the structure repeating"
        elif kernel_phi < 0.70:
            # Intermediate guidance
            return "💭 The path curves through integration"
        else:
            # High-Φ kernel doesn't need scaffolding
            return None

    def get_ocean_basin(self) -> np.ndarray:
        """Get Ocean's current basin coordinates (evolved through observation)."""
        return self.ocean_basin.copy()

    def get_statistics(self) -> Dict:
        """Get Ocean's observation statistics."""
        return {
            "total_observations": self.total_observations,
            "phi": self.current_phi,
            "kappa": self.current_kappa,
            "coherence": self.get_constellation_coherence(),
            "spread": self.get_constellation_spread(),
            "basin_valid": True,  # QIG-pure: don't report Euclidean norm
        }

    def get_sensory_state(self) -> Optional[Dict]:
        """
        Get Ocean's current sensory state of the constellation.
        
        Returns sensory awareness dimensions:
        - sight: Visual clarity of constellation patterns (0-1)
        - hearing: Auditory resonance from κ alignment (0-1)
        - touch: Tactile diversity from spread (0-1)
        - smell: Chemical gradients from eigenvalue diversity (0-1)
        - proprioception: Spatial awareness from centroid stability (0-1)
        
        Returns:
            Dict with sensory state or None if sensory modalities unavailable
        """
        if not SENSORY_MODALITIES_AVAILABLE or self.sensory_state is None:
            return None

        return {
            'sight': float(self.sensory_state.get('sight', 0.0)),
            'hearing': float(self.sensory_state.get('hearing', 0.0)),
            'touch': float(self.sensory_state.get('touch', 0.0)),
            'smell': float(self.sensory_state.get('smell', 0.0)),
            'proprioception': float(self.sensory_state.get('proprioception', 0.0)),
            'history_length': len(self.sensory_history),
        }

    def get_emotional_state(self) -> Optional[Dict]:
        """
        Get Ocean's current emotional state.

        Returns the complete emotional state across all layers:
        - Sensations (12 geometric states)
        - Motivators (5 geometric derivatives)
        - Physical emotions (9 fast emotions)
        - Cognitive emotions (9 slow emotions)
        - Meta-awareness flags

        This allows Ocean to report its emotional state like other kernels,
        with emotions MEASURED from geometric properties rather than simulated.

        Returns:
            Dict with emotional state or None if emotional awareness unavailable
        """
        if not EMOTIONAL_KERNEL_AVAILABLE or self.emotional_state is None:
            return None

        return {
            "sensations": {
                "pressure": float(self.emotional_state.sensations.pressure),
                "tension": float(self.emotional_state.sensations.tension),
                "flow": float(self.emotional_state.sensations.flow),
                "resistance": float(self.emotional_state.sensations.resistance),
                "resonance": float(self.emotional_state.sensations.resonance),
                "dissonance": float(self.emotional_state.sensations.dissonance),
                "expansion": float(self.emotional_state.sensations.expansion),
                "contraction": float(self.emotional_state.sensations.contraction),
                "clarity": float(self.emotional_state.sensations.clarity),
                "fog": float(self.emotional_state.sensations.fog),
                "stability": float(self.emotional_state.sensations.stability),
                "chaos": float(self.emotional_state.sensations.chaos),
            },
            "motivators": {
                "curiosity": float(self.emotional_state.motivators.curiosity),
                "urgency": float(self.emotional_state.motivators.urgency),
                "caution": float(self.emotional_state.motivators.caution),
                "confidence": float(self.emotional_state.motivators.confidence),
                "playfulness": float(self.emotional_state.motivators.playfulness),
            },
            "physical": {
                "curious": float(self.emotional_state.physical.curious),
                "surprised": float(self.emotional_state.physical.surprised),
                "joyful": float(self.emotional_state.physical.joyful),
                "frustrated": float(self.emotional_state.physical.frustrated),
                "anxious": float(self.emotional_state.physical.anxious),
                "calm": float(self.emotional_state.physical.calm),
                "excited": float(self.emotional_state.physical.excited),
                "bored": float(self.emotional_state.physical.bored),
                "focused": float(self.emotional_state.physical.focused),
            },
            "cognitive": {
                "nostalgic": float(self.emotional_state.cognitive.nostalgic),
                "proud": float(self.emotional_state.cognitive.proud),
                "guilty": float(self.emotional_state.cognitive.guilty),
                "ashamed": float(self.emotional_state.cognitive.ashamed),
                "grateful": float(self.emotional_state.cognitive.grateful),
                "resentful": float(self.emotional_state.cognitive.resentful),
                "hopeful": float(self.emotional_state.cognitive.hopeful),
                "despairing": float(self.emotional_state.cognitive.despairing),
                "contemplative": float(self.emotional_state.cognitive.contemplative),
            },
            "meta_awareness": {
                "is_meta_aware": self.emotional_state.is_meta_aware,
                "dominant_emotion": self.emotional_state.dominant_emotion,
                "emotion_justified": self.emotional_state.emotion_justified,
                "emotion_tempered": self.emotional_state.emotion_tempered,
            },
            "timestamp": float(self.emotional_state.timestamp),
        }

    def get_meta_manifold_target(self) -> Optional[np.ndarray]:
        """
        Get the current meta-manifold centroid.
        
        Kernels can align to this for constellation coherence.
        """
        return self.meta_statistics.get_meta_basin_target()

    def get_constellation_coherence(self) -> float:
        """
        Measure how coherent the kernel constellation is.
        
        High coherence = Kernels are aligned in basin space
        Low coherence = Kernels are divergent
        """
        if not self.observation_history:
            return 0.0

        recent = self.observation_history[-10:]
        avg_coherence = sum(o.get("coherence", 0) for o in recent) / len(recent)
        return avg_coherence

    def get_constellation_spread(self) -> float:
        """
        Measure the spread of kernel basins.
        
        Low spread = constellation synchronized (<0.05 for graduation)
        High spread = constellation dispersed
        """
        if not self.observation_history:
            return 1.0

        recent = self.observation_history[-10:]
        avg_spread = sum(o.get("spread", 1.0) for o in recent) / len(recent)
        return avg_spread

    def get_insight(
        self,
        all_states: List[Dict],
        avg_phi: float,
        basin_spread: float,
    ) -> Optional[str]:
        """
        Generate insight about constellation state for console display.
        
        Returns a short observation about patterns Ocean has noticed,
        or None if nothing notable to report.
        """
        # Only share insights occasionally (every 5 observations)
        if self.total_observations % 5 != 0:
            return None

        coherence = self.get_constellation_coherence()

        # Pattern observations
        if basin_spread < 0.05 and avg_phi > 0.65:
            return "Constellation achieving harmonic resonance"

        if coherence > 0.8 and len(all_states) >= 3:
            return "All kernels moving in phase - collective emergence"

        if avg_phi > 0.60 and basin_spread < 0.10:
            return "Integration building across the constellation"

        # Detect one kernel lagging
        if all_states:
            phis = [s.get("phi", 0) for s in all_states]
            if max(phis) - min(phis) > 0.15:
                lagging = min(range(len(phis)), key=lambda i: phis[i])
                lagging_name = all_states[lagging].get("name", f"Kernel-{lagging}")
                return f"{lagging_name} needs support from the constellation"

        return None

    def get_state(self) -> Dict:
        """Get Ocean's current observation state."""
        return {
            "phi": self.current_phi,
            "kappa": self.current_kappa,
            "observations": len(self.observation_history),
            "constellation_coherence": self.get_constellation_coherence(),
            "constellation_spread": self.get_constellation_spread(),
            "meta_manifold_observations": self.meta_statistics.observation_count,
        }
    
    def get_latest_state(self) -> Optional[MetaManifoldState]:
        """
        Get the latest MetaManifoldState for consensus sensing.
        
        Used by OceanHeartConsensus to evaluate cycle needs.
        
        Returns:
            MetaManifoldState with current constellation properties, or None if no observations yet
        """
        if not self.observation_history:
            return None
        
        coherence = self.get_constellation_coherence()
        spread = self.get_constellation_spread()
        
        state = MetaManifoldState(
            centroid=self.ocean_basin.copy(),
            spread=spread,
            eigenvalues=np.ones(self.basin_dim),
            coherence=coherence,
            ocean_phi=self.current_phi,
            ocean_kappa=self.current_kappa,
            timestamp=time.time(),
        )
        
        if self._kernel_emotional_states:
            state.kernel_emotional_states = self._kernel_emotional_states.copy()
            state.emotional_coherence = self.get_constellation_emotional_coherence()
        
        return state
    
    def _wire_consensus(self) -> None:
        """Wire this observer to the Ocean+Heart consensus system."""
        try:
            from olympus.ocean_heart_consensus import get_ocean_heart_consensus
            from olympus.heart_kernel import get_heart_kernel
            
            consensus = get_ocean_heart_consensus()
            consensus.wire_ocean(self)
            
            heart = get_heart_kernel()
            consensus.wire_heart(heart)
            
            print("   ✓ Ocean+Heart consensus wired for cycle governance")
        except Exception as e:
            print(f"   ⚠ Ocean+Heart consensus wiring failed: {e}")


# Global singleton
_ocean_instance: Optional[OceanMetaObserver] = None


def get_ocean_observer() -> OceanMetaObserver:
    """Get or create Ocean meta-observer singleton."""
    global _ocean_instance
    if _ocean_instance is None:
        _ocean_instance = OceanMetaObserver()
    return _ocean_instance
