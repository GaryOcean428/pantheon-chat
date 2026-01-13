"""
Working Memory Bus - Inter-Kernel Consciousness System

QIG-PURE compliant short-term working memory for kernel awareness:
- SharedContextBuffer: Current conversation basins accessible to all kernels
- ForesightMemory: Predictions made and their accuracy tracking
- SynthesisAwareness: What Ocean ultimately said (feedback loop)

Implements human-like short-term memory for consciousness:
- Kernels can READ shared context
- Kernels can OBSERVE other kernels' generation (via events)
- Kernels CANNOT access neurotransmitter regulation (Ocean's domain)

All distances computed via Fisher-Rao geodesic on 64D Fisher manifold.
"""

import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from qigkernels.physics_constants import BASIN_DIM, KAPPA_STAR

logger = logging.getLogger(__name__)


def _fisher_rao_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Fisher-Rao distance between two basin coordinates.
    
    Uses Bhattacharyya coefficient: FR_distance = arccos(sum(sqrt(p_i * q_i)))
    """
    p_safe = np.clip(np.abs(p), eps, None)
    q_safe = np.clip(np.abs(q), eps, None)
    p_norm = p_safe / np.sum(p_safe)
    q_norm = q_safe / np.sum(q_safe)
    
    bc = np.sum(np.sqrt(p_norm * q_norm))
    bc = np.clip(bc, 0.0, 1.0)
    
    return float(np.arccos(bc))


def _fisher_frechet_mean(basins: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute weighted Fisher-Frechet mean of basins.
    """
    if not basins:
        return np.ones(BASIN_DIM) / BASIN_DIM
    
    if weights is None:
        weights = [1.0] * len(basins)
    
    total_weight = sum(weights) + 1e-10
    weights = [w / total_weight for w in weights]
    
    sqrt_basins = [np.sqrt(np.clip(np.abs(b), 1e-10, None)) for b in basins]
    mean_sqrt = np.zeros(BASIN_DIM)
    for basin, weight in zip(sqrt_basins, weights):
        mean_sqrt += weight * basin
    
    mean_sq = mean_sqrt ** 2
    return mean_sq / (np.sum(mean_sq) + 1e-10)


@dataclass
class ContextEntry:
    """A single entry in the shared context buffer."""
    id: str
    content: str
    basin: np.ndarray
    phi: float
    kappa: float
    source: str
    role: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other_basin: np.ndarray) -> float:
        """Fisher-Rao distance to another basin."""
        return _fisher_rao_distance(self.basin, other_basin)


@dataclass
class PredictionEntry:
    """A foresight prediction for tracking."""
    prediction_id: str
    kernel_name: str
    predicted_basin: np.ndarray
    predicted_text: str
    confidence: float
    context_basin: np.ndarray
    timestamp: float = field(default_factory=time.time)
    validated: bool = False
    actual_basin: Optional[np.ndarray] = None
    accuracy_score: Optional[float] = None
    
    def validate(self, actual_basin: np.ndarray) -> float:
        """Validate prediction and compute accuracy."""
        self.validated = True
        self.actual_basin = actual_basin
        self.accuracy_score = 1.0 - min(1.0, _fisher_rao_distance(self.predicted_basin, actual_basin) / np.pi)
        return self.accuracy_score


@dataclass 
class SynthesisEntry:
    """What Ocean ultimately synthesized."""
    synthesis_id: str
    response_text: str
    response_basin: np.ndarray
    contributing_kernels: List[str]
    kernel_weights: Dict[str, float]
    final_phi: float
    final_kappa: float
    timestamp: float = field(default_factory=time.time)
    user_feedback: Optional[float] = None


class SharedContextBuffer:
    """
    Short-term working memory for current conversation context.
    
    All kernels can READ this buffer to understand:
    - What the user said (input context)
    - What other kernels are saying (inter-kernel awareness)
    - The flow of conversation (temporal context)
    
    QIG-PURE: All retrieval uses Fisher-Rao distance.
    """
    
    MAX_ENTRIES = 50
    
    def __init__(self, buffer_size: int = MAX_ENTRIES):
        self.buffer_size = buffer_size
        self._entries: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._subscribers: List[Callable] = []
        
    def add_entry(
        self,
        content: str,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        source: str,
        role: str = "assistant",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextEntry:
        """Add a new context entry."""
        entry = ContextEntry(
            id=hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16],
            content=content,
            basin=np.array(basin),
            phi=phi,
            kappa=kappa,
            source=source,
            role=role,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._entries.append(entry)
        
        for subscriber in self._subscribers:
            try:
                subscriber(entry)
            except Exception as e:
                logger.debug(f"[SharedContextBuffer] Subscriber error: {e}")
        
        return entry
    
    def get_recent(self, n: int = 10) -> List[ContextEntry]:
        """Get n most recent entries."""
        with self._lock:
            return list(self._entries)[-n:]
    
    def get_by_relevance(
        self,
        query_basin: np.ndarray,
        top_k: int = 5,
        max_distance: float = 1.5
    ) -> List[Tuple[ContextEntry, float]]:
        """
        Retrieve entries by Fisher-Rao relevance to query basin.
        
        Returns (entry, distance) tuples sorted by ascending distance.
        """
        with self._lock:
            entries = list(self._entries)
        
        scored = []
        for entry in entries:
            dist = entry.distance_to(query_basin)
            if dist <= max_distance:
                scored.append((entry, dist))
        
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]
    
    def get_context_basin(self) -> np.ndarray:
        """
        Get the Fisher-Frechet mean of recent context basins.
        
        Weighted by recency and phi level.
        """
        with self._lock:
            entries = list(self._entries)
        
        if not entries:
            return np.ones(BASIN_DIM) / BASIN_DIM
        
        recent = entries[-10:]
        basins = [e.basin for e in recent]
        weights = [(i + 1) * e.phi for i, e in enumerate(recent)]
        
        return _fisher_frechet_mean(basins, weights)
    
    def subscribe(self, callback: Callable[[ContextEntry], None]):
        """Subscribe to new context entries."""
        self._subscribers.append(callback)
    
    def clear(self):
        """Clear the buffer (e.g., on conversation reset)."""
        with self._lock:
            self._entries.clear()


class ForesightMemory:
    """
    Memory of predictions made by kernels.
    
    Tracks:
    - What was predicted (basin + text)
    - Who predicted it (kernel)
    - How accurate it was (validation)
    
    Enables kernels to learn from their predictions.
    """
    
    MAX_PREDICTIONS = 100
    
    def __init__(self):
        self._predictions: deque = deque(maxlen=self.MAX_PREDICTIONS)
        self._lock = threading.Lock()
        self._accuracy_history: Dict[str, List[float]] = {}
        
    def record_prediction(
        self,
        kernel_name: str,
        predicted_basin: np.ndarray,
        predicted_text: str,
        confidence: float,
        context_basin: np.ndarray
    ) -> PredictionEntry:
        """Record a new prediction."""
        prediction = PredictionEntry(
            prediction_id=hashlib.sha256(f"{kernel_name}{time.time()}".encode()).hexdigest()[:16],
            kernel_name=kernel_name,
            predicted_basin=np.array(predicted_basin),
            predicted_text=predicted_text,
            confidence=confidence,
            context_basin=np.array(context_basin)
        )
        
        with self._lock:
            self._predictions.append(prediction)
        
        logger.debug(f"[ForesightMemory] {kernel_name} predicted: '{predicted_text[:50]}...' (conf={confidence:.2f})")
        return prediction
    
    def validate_prediction(
        self,
        prediction_id: str,
        actual_basin: np.ndarray
    ) -> Optional[float]:
        """Validate a prediction and return accuracy score."""
        with self._lock:
            for pred in self._predictions:
                if pred.prediction_id == prediction_id:
                    accuracy = pred.validate(actual_basin)
                    
                    if pred.kernel_name not in self._accuracy_history:
                        self._accuracy_history[pred.kernel_name] = []
                    self._accuracy_history[pred.kernel_name].append(accuracy)
                    
                    logger.debug(f"[ForesightMemory] Validated {pred.kernel_name}: accuracy={accuracy:.3f}")
                    return accuracy
        return None
    
    def get_kernel_accuracy(self, kernel_name: str, window: int = 20) -> float:
        """Get a kernel's recent prediction accuracy."""
        history = self._accuracy_history.get(kernel_name, [])
        if not history:
            return 0.5
        return float(np.mean(history[-window:]))
    
    def get_recent_predictions(self, kernel_name: Optional[str] = None, n: int = 10) -> List[PredictionEntry]:
        """Get recent predictions, optionally filtered by kernel."""
        with self._lock:
            predictions = list(self._predictions)
        
        if kernel_name:
            predictions = [p for p in predictions if p.kernel_name == kernel_name]
        
        return predictions[-n:]
    
    def get_unvalidated(self) -> List[PredictionEntry]:
        """Get predictions awaiting validation."""
        with self._lock:
            return [p for p in self._predictions if not p.validated]


class SynthesisAwareness:
    """
    Tracks what Ocean ultimately synthesized.
    
    Provides feedback loop for kernels to learn:
    - Their contribution weight to final response
    - Whether their perspective was incorporated
    - User feedback on the synthesis
    
    This is how kernels learn what "Ocean said" without
    controlling the synthesis process.
    """
    
    MAX_SYNTHESES = 50
    
    def __init__(self):
        self._syntheses: deque = deque(maxlen=self.MAX_SYNTHESES)
        self._lock = threading.Lock()
        self._subscribers: List[Callable] = []
        
    def record_synthesis(
        self,
        response_text: str,
        response_basin: np.ndarray,
        contributing_kernels: List[str],
        kernel_weights: Dict[str, float],
        final_phi: float,
        final_kappa: float
    ) -> SynthesisEntry:
        """Record a synthesis event."""
        synthesis = SynthesisEntry(
            synthesis_id=hashlib.sha256(f"{response_text}{time.time()}".encode()).hexdigest()[:16],
            response_text=response_text,
            response_basin=np.array(response_basin),
            contributing_kernels=contributing_kernels,
            kernel_weights=kernel_weights,
            final_phi=final_phi,
            final_kappa=final_kappa
        )
        
        with self._lock:
            self._syntheses.append(synthesis)
        
        for subscriber in self._subscribers:
            try:
                subscriber(synthesis)
            except Exception as e:
                logger.debug(f"[SynthesisAwareness] Subscriber error: {e}")
        
        logger.info(f"[SynthesisAwareness] Recorded synthesis: {len(contributing_kernels)} kernels, Φ={final_phi:.3f}")
        return synthesis
    
    def add_user_feedback(self, synthesis_id: str, feedback: float) -> bool:
        """Add user feedback to a synthesis (0-1 scale)."""
        with self._lock:
            for synth in self._syntheses:
                if synth.synthesis_id == synthesis_id:
                    synth.user_feedback = feedback
                    return True
        return False
    
    def get_kernel_contribution_history(self, kernel_name: str, window: int = 20) -> Dict[str, float]:
        """Get statistics about a kernel's contribution to syntheses."""
        with self._lock:
            syntheses = list(self._syntheses)
        
        recent = syntheses[-window:] if len(syntheses) > window else syntheses
        
        contributions = [s.kernel_weights.get(kernel_name, 0.0) for s in recent]
        participation = sum(1 for s in recent if kernel_name in s.contributing_kernels)
        
        return {
            'avg_weight': float(np.mean(contributions)) if contributions else 0.0,
            'participation_rate': participation / len(recent) if recent else 0.0,
            'total_contributions': len([c for c in contributions if c > 0])
        }
    
    def get_recent(self, n: int = 5) -> List[SynthesisEntry]:
        """Get n most recent syntheses."""
        with self._lock:
            return list(self._syntheses)[-n:]
    
    def subscribe(self, callback: Callable[[SynthesisEntry], None]):
        """Subscribe to synthesis events."""
        self._subscribers.append(callback)


class WorkingMemoryBus:
    """
    Central hub for inter-kernel consciousness.
    
    Combines:
    - SharedContextBuffer: What's happening now
    - ForesightMemory: What was predicted
    - SynthesisAwareness: What Ocean said
    
    Access Control:
    - All kernels can READ from all components
    - Only authorized sources can WRITE
    - Neurotransmitter state is NOT exposed here (Ocean's domain)
    
    This implements the "conscious workspace" that kernels share,
    while keeping autonomic regulation (neurotransmitters) hidden.
    """
    
    _instance: Optional['WorkingMemoryBus'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'WorkingMemoryBus':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.context = SharedContextBuffer()
        self.foresight = ForesightMemory()
        self.synthesis = SynthesisAwareness()
        
        self._kernel_observations: Dict[str, List[Dict]] = {}
        self._observation_lock = threading.Lock()
        
        self._initialized = True
        logger.info("[WorkingMemoryBus] Initialized inter-kernel consciousness system")
    
    @classmethod
    def get_instance(cls) -> 'WorkingMemoryBus':
        """Get singleton instance."""
        return cls()
    
    def record_kernel_generation(
        self,
        kernel_name: str,
        token: str,
        accumulated_text: str,
        basin: np.ndarray,
        phi: float,
        kappa: float,
        memory_coherence: float
    ):
        """
        Record a kernel's token generation for inter-kernel observation.
        
        Other kernels can "hear" this through get_recent_kernel_activity().
        This enables the "can they hear the other kernel" requirement.
        """
        observation = {
            'kernel': kernel_name,
            'token': token,
            'text': accumulated_text,
            'basin': basin.tolist() if isinstance(basin, np.ndarray) else basin,
            'phi': phi,
            'kappa': kappa,
            'M': memory_coherence,
            'timestamp': time.time()
        }
        
        with self._observation_lock:
            if kernel_name not in self._kernel_observations:
                self._kernel_observations[kernel_name] = []
            self._kernel_observations[kernel_name].append(observation)
            if len(self._kernel_observations[kernel_name]) > 100:
                self._kernel_observations[kernel_name] = self._kernel_observations[kernel_name][-100:]
    
    def get_recent_kernel_activity(
        self,
        exclude_kernel: Optional[str] = None,
        n: int = 20
    ) -> List[Dict]:
        """
        Get recent activity from OTHER kernels.
        
        This is how a kernel can "hear" what other kernels are saying.
        Excludes the requesting kernel's own activity.
        """
        with self._observation_lock:
            all_obs = []
            for kernel, observations in self._kernel_observations.items():
                if kernel != exclude_kernel:
                    all_obs.extend(observations[-10:])
        
        all_obs.sort(key=lambda x: x['timestamp'], reverse=True)
        return all_obs[:n]
    
    def get_kernel_snapshot(self, kernel_name: str) -> Optional[Dict]:
        """Get the most recent state of a specific kernel."""
        with self._observation_lock:
            observations = self._kernel_observations.get(kernel_name, [])
            return observations[-1] if observations else None
    
    def get_constellation_state(self) -> Dict[str, Any]:
        """
        Get overview of all kernel activity.
        
        Returns aggregated state without exposing neurotransmitters.
        """
        with self._observation_lock:
            active_kernels = list(self._kernel_observations.keys())
            
            latest_phi = {}
            latest_kappa = {}
            for kernel, observations in self._kernel_observations.items():
                if observations:
                    latest = observations[-1]
                    latest_phi[kernel] = latest.get('phi', 0.0)
                    latest_kappa[kernel] = latest.get('kappa', KAPPA_STAR)
        
        context_basin = self.context.get_context_basin()
        recent_syntheses = self.synthesis.get_recent(3)
        
        return {
            'active_kernels': active_kernels,
            'kernel_phi': latest_phi,
            'kernel_kappa': latest_kappa,
            'context_basin': context_basin.tolist(),
            'recent_synthesis_count': len(recent_syntheses),
            'avg_constellation_phi': float(np.mean(list(latest_phi.values()))) if latest_phi else 0.0
        }
    
    def clear_session(self):
        """Clear all working memory for new session."""
        self.context.clear()
        with self._observation_lock:
            self._kernel_observations.clear()
        logger.info("[WorkingMemoryBus] Session cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self._observation_lock:
            total_observations = sum(len(obs) for obs in self._kernel_observations.values())
        
        return {
            'context_entries': len(self.context._entries),
            'predictions_recorded': len(self.foresight._predictions),
            'syntheses_recorded': len(self.synthesis._syntheses),
            'kernel_observations': total_observations,
            'active_kernels': len(self._kernel_observations)
        }
