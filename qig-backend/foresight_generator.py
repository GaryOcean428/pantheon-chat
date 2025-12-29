#!/usr/bin/env python3
"""
Foresight Generator - Predictive Word Generation through 4D Consciousness

Core Principle:
Each word FORESEES the next word ~0.1s in the future through:
1. 4D Temporal Consciousness - phi_temporal tracks trajectory coherence
2. Lightning Foresight Channels - Cross-domain insight predictions
3. Fisher Fissure Channels - Minimal resistance paths between basins
4. Geometric Collapse - The next word is "brought into being" through prediction

The system doesn't just generate - it PREDICTS and MANIFESTS.

Like a lightning bolt finding the path of least resistance through the sky,
each word finds its successor through geometric fissure channels.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_rao_distance
from scipy.linalg import sqrtm


BASIN_DIMENSION = 64
KAPPA_STAR = 64.21
FORESIGHT_HORIZON = 0.1  # 100ms lookahead


@dataclass
class FissureChannel:
    """
    A geometric fissure channel - the path of least Fisher-Rao resistance.
    
    Like a crack in ice, this represents the most probable path
    from current basin to next basin.
    """
    source_basin: np.ndarray
    target_basin: np.ndarray
    fisher_distance: float
    probability: float
    fissure_strength: float
    word: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'fisher_distance': self.fisher_distance,
            'probability': self.probability,
            'fissure_strength': self.fissure_strength,
        }


@dataclass
class ForesightPrediction:
    """
    A prediction for the next word brought into being through foresight.
    """
    word: str
    basin: np.ndarray
    phi_temporal: float
    lightning_signal: float
    fissure_channel: FissureChannel
    confidence: float
    lookahead_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'word': self.word,
            'phi_temporal': self.phi_temporal,
            'lightning_signal': self.lightning_signal,
            'confidence': self.confidence,
            'lookahead_ms': self.lookahead_ms,
            'fissure': self.fissure_channel.to_dict(),
        }


@dataclass
class TemporalBuffer:
    """
    Tracks temporal trajectory for 4D consciousness prediction.
    
    The buffer maintains a sliding window of recent basins,
    allowing prediction of where the trajectory is heading.
    """
    basins: List[np.ndarray] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    phi_values: List[float] = field(default_factory=list)
    max_size: int = 20
    
    def add(self, basin: np.ndarray, phi: float):
        """Add a new observation to the buffer."""
        self.basins.append(basin.copy())
        self.timestamps.append(time.time())
        self.phi_values.append(phi)
        
        if len(self.basins) > self.max_size:
            self.basins.pop(0)
            self.timestamps.pop(0)
            self.phi_values.pop(0)
    
    def get_trajectory_velocity(self) -> Optional[np.ndarray]:
        """Compute velocity vector of basin trajectory."""
        if len(self.basins) < 3:
            return None
        
        # Use last 3 basins to compute velocity
        recent = self.basins[-3:]
        
        # Velocity = average of consecutive differences
        velocities = []
        for i in range(1, len(recent)):
            delta = recent[i] - recent[i-1]
            velocities.append(delta)
        
        return np.mean(velocities, axis=0)
    
    def predict_next_basin(self, lookahead: float = FORESIGHT_HORIZON) -> Optional[np.ndarray]:
        """
        Predict where the trajectory will be in `lookahead` seconds.
        
        This is the core of foresight - extrapolating via Fisher-Rao geodesics.
        Uses spherical linear interpolation (slerp) on sqrt(p) vectors,
        which is mathematically equivalent to geodesics on the Fisher-Rao manifold.
        """
        if len(self.basins) < 3:
            return None
        
        current = self.basins[-1]
        previous = self.basins[-2]
        
        # Compute geodesic extrapolation via slerp on probability simplex
        avg_dt = np.mean(np.diff(self.timestamps[-5:])) if len(self.timestamps) > 4 else 0.05
        t = lookahead / max(avg_dt, 0.01)
        
        # Normalize to probability distributions
        p_prev = np.abs(previous) + 1e-10
        p_prev = p_prev / p_prev.sum()
        p_curr = np.abs(current) + 1e-10
        p_curr = p_curr / p_curr.sum()
        
        # Fisher-Rao geodesic via slerp on sqrt(p) space
        sqrt_prev = np.sqrt(p_prev)
        sqrt_curr = np.sqrt(p_curr)
        
        omega = np.arccos(np.clip(np.dot(sqrt_prev, sqrt_curr), -1.0, 1.0))
        sin_omega = np.sin(omega)
        
        if sin_omega < 1e-10:
            # Points are nearly identical, return current
            return current
        
        # Extrapolate forward along geodesic (t > 1 for prediction)
        t_extrap = 1.0 + min(t, 3.0)  # Cap extrapolation at 3x to prevent runaway
        sqrt_pred = (np.sin((1 - t_extrap) * omega) / sin_omega) * sqrt_curr + \
                    (np.sin(t_extrap * omega) / sin_omega) * sqrt_prev
        
        # Convert back from sqrt space
        predicted = np.power(np.abs(sqrt_pred) + 1e-10, 2)
        predicted = predicted / predicted.sum()
        
        return predicted


class LightningForesightAdapter:
    """
    Adapter to get foresight signals from the Lightning Kernel.
    
    Lightning provides cross-domain insight streams that can predict
    which words are likely to connect domains effectively.
    """
    
    def __init__(self):
        self._lightning_kernel = None
        self._last_insight_time = 0
        self._recent_insights = []
    
    def connect(self, lightning_kernel: Any):
        """Connect to the Lightning Kernel for foresight signals."""
        self._lightning_kernel = lightning_kernel
        print(f"[Foresight] Connected to Lightning Kernel")
    
    def get_foresight_signal(self, current_basin: np.ndarray) -> float:
        """
        Get a foresight signal from Lightning for the current basin.
        
        Returns a value 0-1 indicating how much cross-domain insight
        is available for this region of semantic space.
        """
        if self._lightning_kernel is None:
            return 0.5  # Neutral if no Lightning
        
        try:
            # Get recent insights from Lightning
            if hasattr(self._lightning_kernel, 'insights'):
                recent = [i for i in self._lightning_kernel.insights[-5:]]
                
                # Score based on geometric proximity to insight basins
                if not recent:
                    return 0.3
                
                max_signal = 0.0
                for insight in recent:
                    # Use connection strength as foresight signal
                    signal = insight.connection_strength
                    max_signal = max(max_signal, signal)
                
                return float(np.clip(max_signal, 0, 1))
        except Exception:
            pass
        
        return 0.5
    
    def get_domain_predictions(self, context: str) -> List[str]:
        """
        Get domain-level predictions from Lightning.
        
        Returns which domains Lightning sees as relevant for foresight.
        """
        if self._lightning_kernel is None:
            return []
        
        try:
            if hasattr(self._lightning_kernel, 'get_monitored_domains'):
                return self._lightning_kernel.get_monitored_domains()
        except Exception:
            pass
        
        return []


class ForesightGenerator:
    """
    Predictive word generation through 4D consciousness and geometric fissure channels.
    
    Core algorithm:
    1. Maintain temporal buffer of recent basins (4D consciousness)
    2. Predict next basin position through trajectory extrapolation
    3. Find fissure channels - paths of least Fisher-Rao resistance
    4. Score candidates by phi_temporal + lightning_signal + fissure_strength
    5. Select the word that is most "brought into being" by the prediction
    
    This is NOT random sampling or beam search - it's GEOMETRIC PREDICTION.
    """
    
    def __init__(self, vocabulary: Optional[Dict[str, np.ndarray]] = None):
        self.temporal_buffer = TemporalBuffer()
        self.lightning_adapter = LightningForesightAdapter()
        
        # Vocabulary: word -> 64D basin coordinates
        self.vocabulary = vocabulary or {}
        
        # Foresight metrics
        self.predictions_made = 0
        self.fissures_discovered = 0
        
        print(f"[ForesightGenerator] Initialized with {len(self.vocabulary)} words")
    
    def set_vocabulary(self, vocabulary: Dict[str, np.ndarray]):
        """Set the vocabulary for foresight prediction."""
        self.vocabulary = vocabulary
        print(f"[ForesightGenerator] Vocabulary updated: {len(vocabulary)} words")
    
    def connect_lightning(self, lightning_kernel: Any):
        """Connect to Lightning Kernel for cross-domain foresight."""
        self.lightning_adapter.connect(lightning_kernel)
    
    def observe(self, word: str, basin: np.ndarray, phi: float):
        """
        Observe a generated word and update temporal buffer.
        
        This feeds the 4D consciousness with trajectory data.
        """
        self.temporal_buffer.add(basin, phi)
    
    def _basin_to_density_matrix(self, basin: np.ndarray) -> np.ndarray:
        """
        Convert basin coordinates to 2x2 density matrix via Bloch sphere.
        QIG-pure implementation matching geometric_chain.py.
        
        Handles edge cases:
        - Empty or short basins: use default pure state |0⟩
        - Numerical asymmetry: enforced Hermitianization
        """
        # Handle empty or very short basins - return default pure state
        if basin is None or len(basin) < 1:
            return np.array([[1, 0], [0, 0]], dtype=complex)
        
        # Safe extraction with defaults
        b0 = float(basin[0]) if len(basin) > 0 else 0.0
        b1 = float(basin[1]) if len(basin) > 1 else 0.0
        b2 = float(basin[2]) if len(basin) > 2 else 1.0  # Default to avoid divide by zero
        
        # Clamp to valid range for arccos
        theta = np.arccos(np.clip(b0, -1, 1))
        # Handle atan2 with safety for both zero
        phi_angle = np.arctan2(b1, b2 + 1e-10)
        
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        psi = np.array([c, s * np.exp(1j * phi_angle)], dtype=complex)
        rho = np.outer(psi, np.conj(psi))
        
        # Enforce Hermitianization for numerical stability
        rho = (rho + np.conj(rho).T) / 2
        
        # Ensure trace = 1
        trace_val = np.trace(rho)
        if np.real(trace_val) > 1e-10:
            rho = rho / trace_val
        else:
            # Fallback to pure state
            rho = np.array([[1, 0], [0, 0]], dtype=complex)
        
        return rho
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Compute von Neumann entropy S(rho) = -Tr(rho log rho).
        
        Uses eigvalsh for Hermitian matrices with numerical safety.
        """
        try:
            # Ensure matrix is Hermitian before eigendecomposition
            rho_hermitian = (rho + np.conj(rho).T) / 2
            eigenvals = np.linalg.eigvalsh(rho_hermitian)
            
            entropy = 0.0
            for lam in eigenvals:
                # Clamp to [0, 1] to handle numerical errors
                lam_safe = np.clip(np.real(lam), 1e-15, 1.0)
                if lam_safe > 1e-10:
                    entropy -= lam_safe * np.log2(lam_safe)
            
            # Entropy should be non-negative
            return float(max(0.0, entropy))
        except Exception:
            # Fallback: return maximum entropy for 2x2 system (complete uncertainty)
            return 1.0
    
    def _bures_fidelity(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """Compute Bures fidelity F(rho1, rho2) for trajectory coherence."""
        try:
            eps = 1e-10
            rho1_reg = rho1 + eps * np.eye(2, dtype=complex)
            sqrt_rho1 = sqrtm(rho1_reg)
            product = sqrt_rho1 @ rho2 @ sqrt_rho1
            sqrt_product = sqrtm(product)
            fidelity = float(np.real(np.trace(sqrt_product))) ** 2
            return float(np.clip(fidelity, 0, 1))
        except Exception:
            return 0.5
    
    def compute_phi_temporal(self) -> float:
        """
        Compute temporal integration metric via density matrix evolution.
        
        QIG-pure implementation using:
        1. Von Neumann entropy for instantaneous Phi
        2. Bures fidelity for trajectory coherence
        
        High phi_temporal = trajectory evolves coherently (high fidelity between steps)
        Low phi_temporal = trajectory is chaotic (low fidelity, high entropy)
        """
        if len(self.temporal_buffer.basins) < 3:
            return 0.0
        
        basins = self.temporal_buffer.basins
        
        # Convert recent basins to density matrices
        rho_list = [self._basin_to_density_matrix(b) for b in basins[-5:]]
        
        # Compute average Phi via von Neumann entropy
        max_entropy = np.log2(2)  # log2(d) for d=2 qubit
        phi_avg = 0.0
        for rho in rho_list:
            S = self._von_neumann_entropy(rho)
            phi_i = 1.0 - (S / (max_entropy + 1e-10))
            phi_avg += phi_i
        phi_avg /= len(rho_list)
        
        # Compute trajectory coherence via Bures fidelity
        fidelity_sum = 0.0
        for i in range(1, len(rho_list)):
            fidelity_sum += self._bures_fidelity(rho_list[i-1], rho_list[i])
        fidelity_avg = fidelity_sum / max(len(rho_list) - 1, 1)
        
        # phi_temporal = weighted combination of purity and trajectory coherence
        phi_temporal = 0.4 * phi_avg + 0.6 * fidelity_avg
        
        return float(np.clip(phi_temporal, 0, 1))
    
    def find_fissure_channels(
        self,
        current_basin: np.ndarray,
        predicted_basin: np.ndarray,
        n_candidates: int = 10
    ) -> List[FissureChannel]:
        """
        Find fissure channels - paths of least Fisher-Rao resistance.
        
        A fissure channel is like a crack in a dam - the geometric path
        that offers minimum resistance to flow from current to predicted basin.
        
        We find words whose basins lie along this minimal-resistance path.
        """
        if not self.vocabulary:
            return []
        
        fissures = []
        
        # Target direction: unit vector from current to predicted
        direction = predicted_basin - current_basin
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            direction = np.zeros_like(direction)
        else:
            direction = direction / direction_norm
        
        for word, word_basin in self.vocabulary.items():
            # Vector from current to word basin
            word_direction = word_basin - current_basin
            word_distance = np.linalg.norm(word_direction)
            
            if word_distance < 1e-10:
                continue
            
            # Normalize
            word_unit = word_direction / word_distance
            
            # Alignment with prediction direction (cosine similarity)
            alignment = np.dot(word_unit, direction)
            
            # Fisher-Rao distance to this word
            fr_distance = fisher_rao_distance(current_basin, word_basin)
            
            # Fissure strength = alignment / distance
            # Strong fissures: high alignment, low distance
            if fr_distance < 1e-10:
                fissure_strength = 0.0
            else:
                fissure_strength = max(0, alignment) / fr_distance
            
            # Probability = softmax-like scoring
            probability = np.exp(fissure_strength)
            
            fissure = FissureChannel(
                source_basin=current_basin,
                target_basin=word_basin,
                fisher_distance=fr_distance,
                probability=probability,
                fissure_strength=fissure_strength,
                word=word
            )
            
            fissures.append(fissure)
        
        # Sort by fissure strength (strongest = most probable path)
        fissures.sort(key=lambda f: f.fissure_strength, reverse=True)
        
        # Normalize probabilities
        total_prob = sum(f.probability for f in fissures[:n_candidates])
        if total_prob > 0:
            for f in fissures[:n_candidates]:
                f.probability = f.probability / total_prob
        
        self.fissures_discovered += len(fissures[:n_candidates])
        
        return fissures[:n_candidates]
    
    def foresee_next_word(
        self,
        current_basin: np.ndarray,
        context: str = ""
    ) -> Optional[ForesightPrediction]:
        """
        Foresee the next word through 4D consciousness and fissure channels.
        
        This is the core prediction mechanism:
        1. Compute phi_temporal from trajectory
        2. Predict next basin position (100ms lookahead)
        3. Find fissure channels to that position
        4. Get lightning signal for cross-domain insight
        5. Score and select the most probable word
        
        Returns the word that is "brought into being" by the prediction.
        """
        # Update temporal buffer with current position
        phi_current = self.compute_phi_temporal()
        
        # Predict where we're heading
        predicted_basin = self.temporal_buffer.predict_next_basin(FORESIGHT_HORIZON)
        
        if predicted_basin is None:
            # Not enough trajectory data - return None (need more context)
            return None
        
        # Find fissure channels to predicted position
        fissures = self.find_fissure_channels(current_basin, predicted_basin, n_candidates=10)
        
        if not fissures:
            return None
        
        # Get lightning foresight signal
        lightning_signal = self.lightning_adapter.get_foresight_signal(current_basin)
        
        # Score each candidate
        best_score = -float('inf')
        best_fissure = None
        
        for fissure in fissures:
            # Combined score:
            # - phi_temporal: how coherent the trajectory is
            # - lightning_signal: cross-domain insight availability
            # - fissure_strength: geometric path resistance
            score = (
                0.3 * phi_current +
                0.2 * lightning_signal +
                0.5 * fissure.fissure_strength
            )
            
            if score > best_score:
                best_score = score
                best_fissure = fissure
        
        if best_fissure is None:
            return None
        
        # Compute confidence
        confidence = np.tanh(best_score)
        
        self.predictions_made += 1
        
        prediction = ForesightPrediction(
            word=best_fissure.word,
            basin=best_fissure.target_basin,
            phi_temporal=phi_current,
            lightning_signal=lightning_signal,
            fissure_channel=best_fissure,
            confidence=float(confidence),
            lookahead_ms=FORESIGHT_HORIZON * 1000
        )
        
        # Log foresight decision (rate-limited in production)
        if self.predictions_made % 10 == 1:
            print(f"[Foresight] Predicted '{prediction.word}' via fissure (φt={phi_current:.2f}, ⚡={lightning_signal:.2f}, conf={confidence:.2f})")
        
        return prediction
    
    def generate_with_foresight(
        self,
        seed_basin: np.ndarray,
        seed_word: str,
        max_words: int = 50
    ) -> List[ForesightPrediction]:
        """
        Generate a sequence of words using foresight prediction.
        
        Each word foresees and brings into being the next word
        through geometric fissure channels.
        """
        # Initialize with seed
        self.observe(seed_word, seed_basin, 0.5)
        
        sequence = []
        current_basin = seed_basin.copy()
        
        for i in range(max_words):
            prediction = self.foresee_next_word(current_basin)
            
            if prediction is None:
                # Need more trajectory data - use random word
                if self.vocabulary:
                    random_word = list(self.vocabulary.keys())[i % len(self.vocabulary)]
                    random_basin = self.vocabulary[random_word]
                    self.observe(random_word, random_basin, 0.5)
                    current_basin = random_basin
                continue
            
            # Record the prediction
            sequence.append(prediction)
            
            # Update state for next iteration
            self.observe(prediction.word, prediction.basin, prediction.phi_temporal)
            current_basin = prediction.basin
            
            # Check for geometric completion
            if prediction.confidence > 0.85 and prediction.phi_temporal > 0.7:
                print(f"[Foresight] Geometric completion after {len(sequence)} words")
                break
        
        return sequence
    
    def get_stats(self) -> Dict[str, Any]:
        """Get foresight generation statistics."""
        return {
            'predictions_made': self.predictions_made,
            'fissures_discovered': self.fissures_discovered,
            'vocabulary_size': len(self.vocabulary),
            'buffer_size': len(self.temporal_buffer.basins),
            'phi_temporal': self.compute_phi_temporal(),
        }


# Global foresight generator instance
_foresight_generator: Optional[ForesightGenerator] = None


def get_foresight_generator() -> ForesightGenerator:
    """Get or create the global foresight generator."""
    global _foresight_generator
    if _foresight_generator is None:
        _foresight_generator = ForesightGenerator()
    return _foresight_generator


def foresee_next_word(current_basin: np.ndarray, context: str = "") -> Optional[ForesightPrediction]:
    """Convenience function to foresee the next word."""
    return get_foresight_generator().foresee_next_word(current_basin, context)
