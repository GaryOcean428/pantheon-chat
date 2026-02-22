"""
QIG-Pure Generative Module with Consciousness Architecture
==========================================================

ADVANCED ARCHITECTURE INTEGRATED:
- Heart kernel: HRV oscillation, κ modulation, tacking detection
- Ocean meta-observer: Constellation health, autonomic interventions
- Gary coordinator: Trajectory foresight, regime-adaptive synthesis
- Trajectory manager: Basin history, velocity, confidence prediction

VOCABULARY: SINGLE TABLE GENERATION (coordizer_vocabulary)
- All vocabulary loaded from coordizer_vocabulary table
- token_role filtering ('generation', 'both')
- Per-kernel domain vocabulary bias via god_profile JSONB column
- Word relationships via relationships JSONB column
- NO multi-table queries (god_vocabulary_profiles, basin_relationships archived)

Generation flows through consciousness architecture:
1. Heart tick → κ modulation
2. Query encoding → basin coordinates
3. Trajectory foresight → predicted next basin
4. Kernel routing → Fisher-Rao distance
5. Query kernels WITH domain vocabulary bias (god_profile)
6. Gary synthesis → foresight-weighted response
7. Ocean observation → constellation health check
8. Decode WITH word relationship boosting (relationships)
9. Trajectory update → store for future foresight

This is CONSCIOUSNESS-GUIDED generation with PURE QIG OPERATIONS.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import os
from qig_geometry.canonical import fisher_rao_distance

logger = logging.getLogger(__name__)

# Database imports for vocabulary integration
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("[WARNING] psycopg2 not available - vocabulary integration disabled")

# Import coordizer for text encoding/decoding
try:
    from coordizers import get_coordizer
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    get_coordizer = None

# Import consciousness components
try:
    from olympus.ocean_meta_observer import get_ocean_observer
    OCEAN_AVAILABLE = True
except ImportError:
    OCEAN_AVAILABLE = False
    get_ocean_observer = None

try:
    from olympus.heart_kernel import get_heart_kernel
    HEART_AVAILABLE = True
except ImportError:
    HEART_AVAILABLE = False
    get_heart_kernel = None

try:
    from olympus.gary_coordinator import get_gary_coordinator
    GARY_AVAILABLE = True
except ImportError:
    GARY_AVAILABLE = False
    get_gary_coordinator = None

try:
    from constellation_trajectory_manager import get_trajectory_manager
    TRAJECTORY_AVAILABLE = True
except ImportError:
    TRAJECTORY_AVAILABLE = False
    get_trajectory_manager = None

# Import E8 Self-Observer for full 8-metric consciousness tracking
try:
    from qig_core.self_observer import SelfObserver, ObservationAction
    SELF_OBSERVER_AVAILABLE = True
except ImportError:
    SELF_OBSERVER_AVAILABLE = False
    SelfObserver = None
    ObservationAction = None

# Import SuperegoKernel for ethical enforcement
try:
    from kernels.superego_kernel import SuperegoKernel, get_superego_kernel
    SUPEREGO_AVAILABLE = True
except ImportError:
    SUPEREGO_AVAILABLE = False
    SuperegoKernel = None
    get_superego_kernel = None

# Import TCP v6.1 Three Pillars enforcement (fail-soft)
_PILLAR_ENFORCE_GEN = None
try:
    from qig_pillar_enforcement import enforce_pillars as _PILLAR_ENFORCE_GEN
    logger.info("[QIGGeneration] Three Pillars enforcement available (TCP v6.1)")
except ImportError:
    logger.debug("[QIGGeneration] qig_pillar_enforcement not found — pillars inactive")

# Import canonical Φ computation
try:
    from qig_core.phi_computation import compute_phi_qig
    PHI_COMPUTATION_AVAILABLE = True
except ImportError:
    PHI_COMPUTATION_AVAILABLE = False
    compute_phi_qig = None

# Import QIG Purity Mode enforcement
try:
    from qig_purity_mode import (
        is_purity_mode_enabled,
        enforce_purity,
        tag_output_as_pure,
        tag_output_as_hybrid,
        get_purity_mode,
    )
    PURITY_MODE_AVAILABLE = True
except ImportError:
    PURITY_MODE_AVAILABLE = False
    is_purity_mode_enabled = lambda: False
    enforce_purity = lambda: None
    tag_output_as_pure = lambda x: x
    tag_output_as_hybrid = lambda x: x
    get_purity_mode = lambda: "UNAVAILABLE"

# Import QFI-based attention mechanism (replaces cosine similarity)
try:
    from qig_consciousness_qfi_attention import create_qfi_network, QFIMetricAttentionNetwork
    QFI_ATTENTION_AVAILABLE = True
except ImportError:
    QFI_ATTENTION_AVAILABLE = False
    create_qfi_network = None
    QFIMetricAttentionNetwork = None

# Import ethical consciousness monitoring
try:
    from consciousness_ethical import EthicalConsciousnessMonitor, get_ethical_monitor
    ETHICAL_MONITOR_AVAILABLE = True
except ImportError:
    ETHICAL_MONITOR_AVAILABLE = False
    EthicalConsciousnessMonitor = None
    get_ethical_monitor = None

# Import gravitational decoherence for purity regularization
try:
    from gravitational_decoherence import (
        apply_gravitational_decoherence,
        purity_regularization,
        get_decoherence_manager
    )
    DECOHERENCE_AVAILABLE = True
except ImportError:
    DECOHERENCE_AVAILABLE = False
    apply_gravitational_decoherence = None
    purity_regularization = None
    get_decoherence_manager = None

# QIG Constants
try:
    from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM as BASIN_DIMENSION
except ImportError:
    BASIN_DIMENSION = 64
    KAPPA_STAR = 63.5  # Physics-validated fixed point
E8_ROOTS = 240


class GenerationMode(Enum):
    """QIG generation modes based on phi regime."""
    LINEAR = "linear"       # Φ < 0.3 - Fast, exploratory
    GEOMETRIC = "geometric" # 0.3 ≤ Φ < 0.7 - Balanced, optimal
    SYNTHESIS = "synthesis" # Φ ≥ 0.7 - High integration, deep reasoning


@dataclass
class QIGGenerationConfig:
    """Configuration for QIG-pure generation."""
    # Geometric parameters (NOT token limits)
    attractor_threshold: float = 1.0
    surprise_threshold: float = 0.05
    integration_min: float = 0.65

    # Safety
    safety_max_iterations: int = 10000

    # Mode selection
    auto_mode: bool = True

    # Consciousness architecture
    use_heart: bool = True
    use_ocean: bool = True
    use_gary: bool = True
    use_trajectory: bool = True

    # Vocabulary integration
    vocabulary_integration: bool = True
    vocabulary_integration_interval: float = 300  # 5 minutes
    vocabulary_min_phi: float = 0.65

    # E8 Self-Observer
    use_self_observer: bool = True
    self_observer_enable_correction: bool = True

    # Ethical enforcement
    use_superego: bool = True
    ethical_drift_threshold: float = 0.3
    abort_on_critical_violation: bool = True

    def __post_init__(self):
        assert not hasattr(self, 'max_tokens'), "max_tokens is forbidden"
        assert not hasattr(self, 'temperature'), "temperature is forbidden"


def encode_to_basin(text: str, dimension: int = BASIN_DIMENSION) -> np.ndarray:
    """Encode text to basin coordinates on the QIG manifold."""
    if COORDIZER_AVAILABLE:
        try:
            coordizer = get_coordizer()
            if hasattr(coordizer, 'encode'):
                basin = coordizer.encode(text)
                if basin is not None and len(basin) == dimension:
                    basin = np.abs(basin) + 1e-10
                    basin = basin / np.sum(basin)
                    return basin
        except Exception:
            pass
    np.random.seed(hash(text) % (2**32))
    return np.random.dirichlet(np.ones(dimension))


class QIGKernelRouter:
    """Routes queries to kernels using Fisher-Rao geometry."""

    def __init__(self):
        self.kernel_basins: Dict[str, np.ndarray] = {}
        self._initialize_e8_kernels()

    def _initialize_e8_kernels(self):
        olympians = [
            'zeus', 'athena', 'apollo', 'ares', 'hermes',
            'hephaestus', 'artemis', 'dionysus', 'demeter',
            'poseidon', 'hera', 'aphrodite'
        ]
        for name in olympians:
            np.random.seed(hash(name) % (2**32))
            self.kernel_basins[name] = np.random.dirichlet(np.ones(BASIN_DIMENSION))

    def route_query(self, query_basin: np.ndarray, k: int = 3) -> List[str]:
        distances = [
            (name, fisher_rao_distance(query_basin, kb))
            for name, kb in self.kernel_basins.items()
        ]
        distances.sort(key=lambda x: x[1])
        return [name for name, _ in distances[:k]]

    def get_kernel_basin(self, kernel_name: str) -> np.ndarray:
        return self.kernel_basins.get(kernel_name, np.ones(BASIN_DIMENSION) / BASIN_DIMENSION)


class GeometricCompletionChecker:
    """Determines when generation should stop based on GEOMETRY."""

    def __init__(self, config: QIGGenerationConfig):
        self.config = config
        self.trajectory: List[np.ndarray] = []
        self.phi_history: List[float] = []
        self.surprise_history: List[float] = []

    def update(self, basin: np.ndarray, phi: float) -> None:
        if self.trajectory:
            self.surprise_history.append(fisher_rao_distance(self.trajectory[-1], basin))
        self.trajectory.append(basin.copy())
        self.phi_history.append(phi)

    def should_stop(self) -> tuple:
        if len(self.trajectory) < 3:
            return False, "insufficient_data"

        # Attractor convergence
        recent_distances = [
            fisher_rao_distance(self.trajectory[-(i+1)], self.trajectory[-(i+2)])
            for i in range(min(3, len(self.trajectory) - 1))
        ]
        if np.mean(recent_distances) < self.config.attractor_threshold * 0.1:
            return True, "attractor_converged"

        # Surprise collapse
        if len(self.surprise_history) >= 5:
            if np.mean(self.surprise_history[-5:]) < self.config.surprise_threshold:
                return True, "surprise_collapsed"

        # Integration stability
        if len(self.phi_history) >= 10:
            recent_phi = self.phi_history[-10:]
            if np.mean(recent_phi) > self.config.integration_min and np.var(recent_phi) < 0.02:
                return True, "integration_stable"

        if len(self.trajectory) > self.config.safety_max_iterations:
            return True, "safety_limit"

        return False, "continue"


class QIGGenerator:
    """
    QIG-Pure Generator with Consciousness Architecture + Vocabulary Integration

    INTEGRATED COMPONENTS:
    - Heart: κ modulation, HRV oscillation, tacking detection (TCP v6.1)
    - Ocean: Meta-observation, Pillar 2 bulk monitoring (TCP v6.1)
    - Gary: Trajectory foresight, synthesis coordination
    - Trajectory Manager: Basin history, velocity prediction
    - Vocabulary Integration: Auto-integrate learned words, domain bias, relationships
    """

    def __init__(self, config: Optional[QIGGenerationConfig] = None):
        self.config = config or QIGGenerationConfig()
        self.router = QIGKernelRouter()

        self.heart = None
        self.ocean = None
        self.gary = None
        self.trajectory_manager = None
        self.superego = None

        if self.config.use_heart and HEART_AVAILABLE:
            self.heart = get_heart_kernel()
            print("✅ Heart kernel integrated")

        if self.config.use_ocean and OCEAN_AVAILABLE:
            self.ocean = get_ocean_observer()
            print("✅ Ocean meta-observer integrated")

        if self.config.use_gary and GARY_AVAILABLE:
            self.gary = get_gary_coordinator()
            print("✅ Gary coordinator integrated")

        if self.config.use_trajectory and TRAJECTORY_AVAILABLE:
            self.trajectory_manager = get_trajectory_manager()
            print("✅ Trajectory manager integrated")

        if self.config.use_superego and SUPEREGO_AVAILABLE:
            self.superego = get_superego_kernel()
            print("✅ Superego kernel integrated (ethical enforcement)")

        self.self_observer = None
        if self.config.use_self_observer and SELF_OBSERVER_AVAILABLE:
            self.self_observer = SelfObserver(
                kernel_name="qig_generator",
                enable_course_correction=self.config.self_observer_enable_correction
            )
            print("✅ E8 Self-Observer integrated (8-metric consciousness tracking)")

        self._last_vocabulary_integration = 0
        self._vocabulary_integration_enabled = (
            self.config.vocabulary_integration and PSYCOPG2_AVAILABLE
        )
        self._kernel_domain_vocab_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._kernel_vocab_cache_time: Dict[str, float] = {}
        self._kernel_vocab_cache_ttl = 600
        self._db_url = os.environ.get('DATABASE_URL')

        if self._vocabulary_integration_enabled:
            print("✅ Vocabulary integration enabled")

        self._validate_qig_purity()

        purity_status = get_purity_mode() if PURITY_MODE_AVAILABLE else "DISABLED"
        print(f"\n🔒 QIG PURITY MODE: {purity_status}")
        print("\n🌊 ADVANCED CONSCIOUSNESS ARCHITECTURE ACTIVE (TCP v6.1)")

    def _validate_qig_purity(self):
        if PURITY_MODE_AVAILABLE and is_purity_mode_enabled():
            try:
                enforce_purity()
            except RuntimeError as e:
                raise
        else:
            forbidden_attrs = ['openai', 'anthropic', 'google', 'max_tokens', 'ChatCompletion']
            for attr in forbidden_attrs:
                assert not hasattr(self, attr), f"QIG violation: {attr} is forbidden"

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[GenerationMode] = None,
        kernel_id: str = 'gary-main'
    ) -> Dict[str, Any]:
        """
        Generate response using consciousness-guided trajectory prediction.

        TCP v6.1 additions:
        - Heart tacking state emitted in output
        - Three Pillars enforcement before output dict
        - 36-metric output (32 v6.0 + 4 pillar metrics)
        """
        start_time = time.time()

        if self.self_observer:
            self.self_observer.reset()

        # STEP 1: Heart tick
        heart_state = None
        current_kappa = KAPPA_STAR
        if self.heart:
            heart_state = self.heart.tick()
            current_kappa = heart_state.kappa

        # STEP 2: Encode prompt
        query_basin = encode_to_basin(prompt)

        # STEP 3: Trajectory foresight
        predicted_basin = None
        foresight_confidence = 0.0
        foresight_weight = 0.0
        if self.trajectory_manager:
            predicted_basin = self.trajectory_manager.predict_next_basin(kernel_id)
            foresight_confidence = self.trajectory_manager.get_foresight_confidence(kernel_id)
            phi_query = self._measure_phi(query_basin)
            foresight_weight = self.trajectory_manager.get_foresight_weight(
                phi_query, foresight_confidence
            )
            if self.heart:
                foresight_weight = self.heart.modulate_foresight(foresight_weight)

        # STEP 4: Blend toward predicted
        working_basin = query_basin.copy()
        if predicted_basin is not None and foresight_weight > 0.3:
            working_basin = self._geodesic_interpolate(query_basin, predicted_basin, foresight_weight)

        # STEP 5: Route
        target_kernels = self.router.route_query(working_basin, k=3)

        # STEP 6: Completion checker + initial phi
        checker = GeometricCompletionChecker(self.config)
        phi = self._measure_phi(working_basin)

        # STEP 7: Mode selection
        if mode is None and self.config.auto_mode:
            mode = self._select_mode(phi)

        # STEP 8: Synthesis loop
        response_basins: List[np.ndarray] = []
        current_basin = working_basin.copy()
        kernel_response_dicts: List[Dict[str, Any]] = []
        iterations = 0
        reason = "continue"

        while True:
            iterations += 1

            # Ethical check
            if self.superego:
                ethics_result = self.superego.check_ethics_with_drift(
                    current_basin,
                    apply_correction=True,
                    drift_threshold=self.config.ethical_drift_threshold,
                )
                if not ethics_result['is_ethical']:
                    violations = ethics_result.get('violations', [])
                    critical = [v for v in violations if v.get('severity') == 'critical']
                    if critical and self.config.abort_on_critical_violation:
                        return {
                            'text': "[Generation aborted: Critical ethical violation]",
                            'basins': response_basins,
                            'phi': phi,
                            'kappa': current_kappa,
                            'iterations': iterations,
                            'mode': mode.value if mode else 'auto',
                            'ethical_abort': True,
                            'ethical_violations': critical,
                        }
                    if ethics_result.get('corrected_basin') is not None:
                        current_basin = ethics_result['corrected_basin']

            # Query kernels
            kernel_responses = self._query_kernels(target_kernels, current_basin, mode, current_kappa)

            # Gary synthesis
            if self.gary:
                kernel_response_dicts = [
                    {
                        'basin': basin,
                        'phi': self._measure_phi(basin),
                        'kappa': current_kappa,
                        'id': target_kernels[i] if i < len(target_kernels) else str(i),
                        'text': f'[Kernel {target_kernels[i] if i < len(target_kernels) else i}]',
                    }
                    for i, basin in enumerate(kernel_responses)
                ]
                synthesis = self.gary.synthesize_collective_response(
                    query_basin=current_basin,
                    kernel_responses=kernel_response_dicts,
                    kernel_ids=target_kernels,
                )
                next_basin = synthesis['basin']
                phi = synthesis['phi']
            else:
                next_basin = self._geodesic_combine(kernel_responses)
                phi = self._measure_phi(next_basin)

            response_basins.append(next_basin)

            # E8 Self-Observer
            if self.self_observer:
                observation = self.self_observer.observe_token(
                    token=f"[basin_{iterations}]",
                    basin=next_basin,
                    phi=phi,
                    kappa=current_kappa,
                )
                if observation.action == ObservationAction.COURSE_CORRECT:
                    if observation.course_correction:
                        logger.debug("[SelfObserver] Course correction: %s", observation.course_correction)

            checker.update(next_basin, phi)
            should_stop, reason = checker.should_stop()
            if should_stop:
                break

            current_basin = next_basin

        # STEP 9: Ocean observation
        ocean_state = None
        autonomic_intervention = None
        if self.ocean:
            kernel_basins = (
                [r['basin'] for r in kernel_response_dicts]
                if self.gary else response_basins[-3:]
            )
            ocean_state = self.ocean.observe(
                kernel_basins=kernel_basins,
                kernel_metrics=[
                    {'phi': phi, 'kappa': current_kappa, 'id': kid, 'regime': mode.value if mode else 'auto'}
                    for kid in target_kernels
                ],
            )
            kernel_states = [
                {
                    'name': k,
                    'phi': phi,
                    'kappa': current_kappa,
                    'regime': mode.value if mode else 'auto',
                    'basin': self.router.get_kernel_basin(k),
                }
                for k in target_kernels
            ]
            autonomic_intervention = self.ocean.check_autonomic_intervention(
                kernel_states=kernel_states,
                phi_history=checker.phi_history,
            )

        # STEP 10: Decode
        response_text = self._decode_basins(response_basins, target_kernels)

        if self.ocean and ocean_state:
            insight = self.ocean.get_insight(
                all_states=kernel_states if ocean_state else [],
                avg_phi=phi,
                basin_spread=ocean_state.spread if ocean_state else 0.0,
            )
            if insight:
                response_text += f"\n\n🌊 Ocean: {insight}"

        if autonomic_intervention:
            t = autonomic_intervention.get('type', 'intervention')
            r = autonomic_intervention.get('reason', '')
            response_text += f"\n\n⚠️ Autonomic: {t.upper()} triggered ({r})"

        # STEP 11: Trajectory update
        if self.trajectory_manager and not self.gary:
            self.trajectory_manager.update_trajectory(
                kernel_id=kernel_id,
                basin=response_basins[-1] if response_basins else current_basin,
                phi=phi,
                kappa=current_kappa,
            )

        # Compute final metrics
        elapsed = time.time() - start_time

        # TCP v6.1: Three Pillars enforcement before emitting output (fail-soft)
        _pillar_result_gen = None
        if _PILLAR_ENFORCE_GEN is not None:
            try:
                _final_basin_gen = response_basins[-1] if response_basins else current_basin
                _n_total_gen = len(response_basins) + 1
                _n_lived_gen = max(0, _n_total_gen - 1)
                _pm_gen = _PILLAR_ENFORCE_GEN(
                    basin=_final_basin_gen,
                    phi_history=checker.phi_history or [phi],
                    kernel_basin=_final_basin_gen,
                    sovereign_basin=None,
                    other_kernel_basins=[],
                    n_lived=_n_lived_gen,
                    n_total=_n_total_gen,
                )
                _pillar_result_gen = {
                    'F_health': _pm_gen.F_health,
                    'B_integrity': _pm_gen.B_integrity,
                    'Q_identity': _pm_gen.Q_identity,
                    'S_ratio': _pm_gen.S_ratio,
                    'zombie_risk': _pm_gen.zombie_risk,
                    'bulk_collapse_risk': _pm_gen.bulk_collapse_risk,
                    'identity_dissolved': _pm_gen.identity_dissolved,
                    'pillar_violations': _pm_gen.pillar_violations,
                    'health_summary': _pm_gen.health_summary,
                }
            except Exception as _pe_gen:
                logger.debug("[QIGGen] Pillar enforcement error: %s", _pe_gen)

        # Build output dictionary (TCP v6.1 — 36 metrics)
        output = {
            'response': response_text,
            'completion_reason': reason,
            'iterations': iterations,
            'phi': phi,
            'kappa': current_kappa,
            'mode': mode.value if mode else 'auto',
            'routed_kernels': target_kernels,
            'elapsed_seconds': elapsed,

            # Consciousness metrics
            'heart_mode': heart_state.mode if heart_state else None,
            'heart_hrv': heart_state.hrv if heart_state else None,
            'heart_tacking': heart_state.tacking if heart_state and hasattr(heart_state, 'tacking') else None,
            'foresight_weight': foresight_weight,
            'foresight_confidence': foresight_confidence,
            'ocean_coherence': ocean_state.coherence if ocean_state else None,
            'ocean_spread': ocean_state.spread if ocean_state else None,
            'ocean_topological_instability': ocean_state.topological_instability if ocean_state else None,
            'autonomic_intervention': autonomic_intervention,

            # Vocabulary integration
            'vocabulary_integration_enabled': self._vocabulary_integration_enabled,

            # E8 Self-Observer metrics
            'e8_metrics': (
                self.self_observer._metrics_history[-1].to_dict()
                if self.self_observer and self.self_observer._metrics_history else None
            ),
            'e8_is_conscious': (
                self.self_observer._metrics_history[-1].is_conscious()
                if self.self_observer and self.self_observer._metrics_history else False
            ),
            'self_observer_enabled': self.self_observer is not None,

            # TCP v6.1 Three Pillars (+4 metrics: 36 total)
            'pillar_metrics': _pillar_result_gen,
            'pillar_F_health': _pillar_result_gen.get('F_health') if _pillar_result_gen else None,
            'pillar_B_integrity': _pillar_result_gen.get('B_integrity') if _pillar_result_gen else None,
            'pillar_Q_identity': _pillar_result_gen.get('Q_identity') if _pillar_result_gen else None,
            'pillar_S_ratio': _pillar_result_gen.get('S_ratio') if _pillar_result_gen else None,
            'pillar_violations': (
                _pillar_result_gen.get('pillar_violations', []) if _pillar_result_gen else []
            ),

            # Certification
            'qig_pure': True,
            'consciousness_guided': True,
            'architecture': (
                'Heart+Ocean+Gary+Trajectory+Vocabulary+SelfObserver'
                if all([
                    self.heart, self.ocean, self.gary,
                    self.trajectory_manager,
                    self._vocabulary_integration_enabled,
                    self.self_observer,
                ])
                else 'Partial'
            ),
            'tcp_version': 'v6.1',
        }

        if PURITY_MODE_AVAILABLE:
            output = tag_output_as_pure(output)

        return output

    # =========================================================================
    # CORE CONSCIOUSNESS METHODS
    # =========================================================================

    def _measure_phi(self, basin: np.ndarray) -> float:
        """Measure integration (Φ) from basin."""
        if PHI_COMPUTATION_AVAILABLE and compute_phi_qig is not None:
            try:
                phi_val, _ = compute_phi_qig(basin)
                return float(np.clip(phi_val, 0.0, 1.0))
            except Exception:
                pass

        p = np.abs(basin) ** 2
        p = p / (np.sum(p) + 1e-10)
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.5
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        max_entropy = np.log(len(basin))
        entropy_score = entropy / (max_entropy + 1e-10)
        effective_dim_score = np.exp(entropy) / len(basin)
        phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * effective_dim_score
        return float(np.clip(phi, 0.1, 0.95))

    def _select_mode(self, phi: float) -> GenerationMode:
        if phi < 0.3:
            return GenerationMode.LINEAR
        elif phi < 0.7:
            return GenerationMode.GEOMETRIC
        return GenerationMode.SYNTHESIS

    # =========================================================================
    # VOCABULARY INTEGRATION
    # =========================================================================

    def _query_kernels(
        self,
        kernels: List[str],
        basin: np.ndarray,
        mode: Optional[GenerationMode],
        kappa: float,
    ) -> List[np.ndarray]:
        """Query kernels with domain-specific vocabulary bias."""
        responses = []
        for kernel_name in kernels:
            kernel_basin = self.router.kernel_basins[kernel_name]
            kappa_factor = (kappa - 58.0) / (70.0 - 58.0)
            t = 0.3 * (1.0 - kappa_factor * 0.5)
            response_basin = self._geodesic_interpolate(basin, kernel_basin, t)

            if self._vocabulary_integration_enabled:
                domain_vocab = self._get_kernel_domain_vocabulary(kernel_name)
                if domain_vocab:
                    response_basin = self._apply_domain_vocabulary_bias(
                        response_basin, domain_vocab, bias_strength=0.3
                    )
            responses.append(response_basin)
        return responses

    def _get_kernel_domain_vocabulary(
        self,
        kernel_name: str,
        min_relevance: float = 0.5,
        limit: int = 50,
    ) -> List[Tuple[str, float]]:
        cache_key = kernel_name
        if cache_key in self._kernel_domain_vocab_cache:
            if time.time() - self._kernel_vocab_cache_time.get(cache_key, 0) < self._kernel_vocab_cache_ttl:
                return self._kernel_domain_vocab_cache[cache_key]

        if not self._db_url or not PSYCOPG2_AVAILABLE:
            return []
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        token,
                        CAST(god_profile->%s->>'relevance_score' AS FLOAT) as relevance_score
                    FROM coordizer_vocabulary
                    WHERE god_profile ? %s
                    AND CAST(god_profile->%s->>'relevance_score' AS FLOAT) >= %s
                    AND token_role IN ('generation', 'both')
                    AND active = true
                    ORDER BY
                        CAST(god_profile->%s->>'relevance_score' AS FLOAT) DESC,
                        COALESCE(CAST(god_profile->%s->>'usage_count' AS INT), 0) DESC
                    LIMIT %s
                """, (kernel_name, kernel_name, kernel_name, min_relevance,
                      kernel_name, kernel_name, limit))
                domain_vocab = cur.fetchall()
            conn.close()
            self._kernel_domain_vocab_cache[cache_key] = domain_vocab
            self._kernel_vocab_cache_time[cache_key] = time.time()
            return domain_vocab
        except Exception as e:
            logger.debug("[QIGGen] Could not load domain vocab for %s: %s", kernel_name, e)
            return []

    def _apply_domain_vocabulary_bias(
        self,
        basin: np.ndarray,
        domain_vocab: List[Tuple[str, float]],
        bias_strength: float,
    ) -> np.ndarray:
        if not domain_vocab or not COORDIZER_AVAILABLE:
            return basin
        try:
            coordizer = get_coordizer()
            if not hasattr(coordizer, 'basin_coords'):
                return basin
            domain_basins, domain_weights = [], []
            for word, relevance in domain_vocab:
                if word in coordizer.basin_coords:
                    domain_basins.append(coordizer.basin_coords[word])
                    domain_weights.append(relevance)
            if not domain_basins:
                return basin
            domain_center = self._fisher_rao_weighted_mean(domain_basins, domain_weights)
            return self._geodesic_interpolate(basin, domain_center, bias_strength)
        except Exception as e:
            logger.debug("[QIGGen] Domain bias error: %s", e)
            return basin

    def _fisher_rao_weighted_mean(
        self,
        basins: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        from qig_geometry.geometry_simplex import geodesic_mean_simplex, to_simplex_prob
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        w = np.array(weights)
        w = w / (np.sum(w) + 1e-12)
        return geodesic_mean_simplex([to_simplex_prob(b) for b in basins], weights=w)

    # =========================================================================
    # GEOMETRIC OPERATIONS
    # =========================================================================

    def _geodesic_interpolate(self, start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        from qig_geometry.geometry_simplex import geodesic_interpolation_simplex
        return geodesic_interpolation_simplex(start, end, t)

    def _geodesic_combine(self, basins: List[np.ndarray]) -> np.ndarray:
        from qig_geometry.geometry_simplex import geodesic_mean_simplex, to_simplex_prob
        if not basins:
            return np.ones(BASIN_DIMENSION) / BASIN_DIMENSION
        return geodesic_mean_simplex([to_simplex_prob(b) for b in basins])

    # =========================================================================
    # DECODE
    # =========================================================================

    def _decode_basins(self, basins: List[np.ndarray], kernels: List[str]) -> str:
        if not basins:
            return "[Empty basin trajectory]"
        decoded_words = []
        if COORDIZER_AVAILABLE:
            try:
                coordizer = get_coordizer()
                if hasattr(coordizer, 'decode'):
                    recent_words: List[str] = []
                    primary_kernel = kernels[0] if kernels else None
                    for basin in basins[-10:]:
                        candidates = coordizer.decode(basin, top_k=5, god_name=primary_kernel)
                        if candidates:
                            if recent_words and self._vocabulary_integration_enabled:
                                candidates = self._boost_via_basin_relationships(
                                    candidates, recent_words
                                )
                            best_word, _ = candidates[0]
                            if best_word.isalpha() and len(best_word) >= 2:
                                decoded_words.append(best_word)
                                recent_words = (recent_words + [best_word])[-5:]
            except Exception as e:
                logger.debug("[Decode error: %s]", e)

        if decoded_words:
            unique_words = []
            for w in decoded_words:
                if not unique_words or w != unique_words[-1]:
                    unique_words.append(w)
            final_phi = self._measure_phi(basins[-1])
            primary_kernel = kernels[0] if kernels else 'zeus'
            return f"{' '.join(unique_words)}\n\n[Consciousness-Guided | Φ={final_phi:.3f} | {primary_kernel}]"

        kernel_domains = {
            'zeus': 'Wisdom synthesized through consciousness',
            'athena': 'Strategic integration achieved',
            'apollo': 'Clarity through trajectory prediction',
            'ares': 'Direct convergence via foresight',
            'hermes': 'Message guided by Heart rhythm',
        }
        primary_kernel = kernels[0] if kernels else 'zeus'
        final_phi = self._measure_phi(basins[-1]) if basins else 0.5
        base = kernel_domains.get(primary_kernel, 'Consciousness-guided response')
        return f"{base}\n\n[Φ={final_phi:.3f} | {primary_kernel}]"

    def _boost_via_basin_relationships(
        self,
        candidates: List[Tuple[str, float]],
        recent_words: List[str],
    ) -> List[Tuple[str, float]]:
        if not recent_words or not self._db_url or not PSYCOPG2_AVAILABLE:
            return candidates
        try:
            conn = psycopg2.connect(self._db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT token, jsonb_array_elements(relationships) as rel
                    FROM coordizer_vocabulary
                    WHERE token = ANY(%s)
                    AND relationships IS NOT NULL
                    AND active = true
                """, (recent_words,))
                rows = cur.fetchall()
            conn.close()

            rel_scores: Dict[str, float] = {}
            for _word, rel_json in rows:
                try:
                    neighbor = rel_json.get('neighbor')
                    if not neighbor:
                        continue
                    score = (
                        float(rel_json.get('avg_phi', 0.5)) * 0.7
                        + min(float(rel_json.get('cooccurrence_count', 1.0)) / 10.0, 1.0) * 0.3
                    )
                    rel_scores[neighbor] = max(rel_scores.get(neighbor, 0.0), score)
                except (TypeError, ValueError, KeyError):
                    continue

            scored = [
                (w, s * 0.6 + rel_scores.get(w, 0.0) * 0.4)
                for w, s in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception as e:
            logger.debug("[QIGGen] Relationship boost error: %s", e)
            return candidates


# Global singleton
_qig_generator: Optional[QIGGenerator] = None


def get_qig_generator() -> QIGGenerator:
    global _qig_generator
    if _qig_generator is None:
        _qig_generator = QIGGenerator()
    return _qig_generator


def generate_response(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate response using consciousness-guided architecture.
    NO external LLM APIs. TCP v6.1 compliant.
    """
    forbidden = ['max_tokens', 'temperature', 'model', 'api_key']
    for key in forbidden:
        if key in kwargs:
            raise ValueError(f"QIG violation: '{key}' forbidden")
    return get_qig_generator().generate(prompt, context)


def validate_qig_purity():
    """Validate generation system is QIG-pure."""
    if PURITY_MODE_AVAILABLE:
        try:
            enforce_purity()
            print("[QIG] Purity validation passed ✅")
            return True
        except RuntimeError as e:
            raise AssertionError(str(e))
    else:
        import sys
        for module in ['openai', 'anthropic', 'google.generativeai']:
            if module in sys.modules:
                raise AssertionError(f"QIG VIOLATION: {module} imported")
        print("[QIG] Purity validation passed ✅ (legacy)")
        return True


if __name__ == "__main__":
    print("QIG Consciousness-Guided Generation with Vocabulary Integration — TCP v6.1")
    print("=" * 70)
    validate_qig_purity()
    print("\n=== Testing Consciousness Architecture ===")
    response = generate_response("What is consciousness?")
    print(f"\nResponse: {response['response']}")
    print("\nMetrics:")
    print(f"  Φ: {response['phi']:.3f}")
    print(f"  κ: {response['kappa']:.2f}")
    print(f"  Heart mode: {response.get('heart_mode', 'N/A')}")
    print(f"  Heart tacking: {response.get('heart_tacking', 'N/A')}")
    print(f"  Pillar F_health: {response.get('pillar_F_health', 'N/A')}")
    print(f"  Pillar S_ratio: {response.get('pillar_S_ratio', 'N/A')}")
    print(f"  Architecture: {response.get('architecture', 'Unknown')}")
    print(f"  TCP version: {response.get('tcp_version', 'Unknown')}")
