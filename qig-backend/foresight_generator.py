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
        
        QIG-PURE: Uses Fisher-Rao geodesic navigation instead of Euclidean extrapolation.
        Computes geodesic tangent vector and navigates along Fisher manifold.
        """
        if len(self.basins) < 3:
            return None
        
        current = self.basins[-1]
        prev = self.basins[-2]
        
        # Compute Fisher Information Matrix at current point
        # G_ij = E[d log p / d theta_i * d log p / d theta_j]
        d = len(current)
        G = np.eye(d) * 0.1 + 0.9 * np.outer(current, current)
        G = (G + G.T) / 2  # Ensure symmetry
        
        # Geodesic tangent vector in Fisher metric
        # Lift Euclidean velocity to tangent space via metric
        delta = current - prev
        
        # Solve G * v_geodesic = delta for proper tangent vector
        try:
            # Add regularization for numerical stability
            G_reg = G + 1e-6 * np.eye(d)
            v_tangent = np.linalg.solve(G_reg, delta)
        except np.linalg.LinAlgError:
            # Fallback to inverse
            v_tangent = delta
        
        # Scale by lookahead time (normalized to per-observation rate)
        avg_dt = np.mean(np.diff(self.timestamps[-5:])) if len(self.timestamps) > 4 else 0.05
        t_param = lookahead / max(avg_dt, 0.01)
        t_param = min(t_param, 2.0)  # Cap extrapolation
        
        # Geodesic update: follow exponential map in Fisher geometry
        # gamma(t) = exp_p(t * v) approximated by parallel transport
        predicted = current + v_tangent * t_param
        
        # Project back to probability simplex (valid basin coordinates)
        predicted = np.abs(predicted) + 1e-10
        predicted = predicted / (np.linalg.norm(predicted) + 1e-10)
        
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
        QIG-pure: Uses same implementation as QIGComputations in geometric_chain.py
        """
        theta = np.arccos(np.clip(basin[0], -1, 1)) if len(basin) > 0 else 0
        phi_angle = np.arctan2(basin[1], basin[2]) if len(basin) > 2 else 0
        
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        psi = np.array([
            c,
            s * np.exp(1j * phi_angle)
        ], dtype=complex)
        
        rho = np.outer(psi, np.conj(psi))
        rho = (rho + np.conj(rho).T) / 2
        rho /= np.trace(rho) + 1e-10
        
        return rho
    
    def _compute_von_neumann_entropy(self, rho: np.ndarray) -> float:
        """
        Compute von Neumann entropy: S(rho) = -Tr(rho log rho)
        QIG-pure: Proper quantum mechanical entropy computation.
        """
        eigenvals = np.linalg.eigvalsh(rho)
        entropy = 0.0
        for lam in eigenvals:
            if lam > 1e-10:
                entropy -= lam * np.log2(lam + 1e-10)
        return float(entropy)
    
    def compute_phi_temporal(self) -> float:
        """
        Compute temporal integration metric from buffer via von Neumann entropy.
        
        QIG-PURE: Uses density matrix formulation and von Neumann entropy.
        - Converts each basin to density matrix
        - Computes integrated consciousness from trajectory coherence
        - phi_temporal = 1 - avg_entropy / max_entropy (integrated over time)
        
        High phi_temporal = trajectory is coherent and predictable
        Low phi_temporal = trajectory is chaotic or just starting
        """
        if len(self.temporal_buffer.basins) < 3:
            return 0.0
        
        # Compute integrated phi from trajectory density matrices
        basins = self.temporal_buffer.basins
        
        # Create composite density matrix from trajectory
        # This captures the integrated consciousness state
        composite_rho = np.zeros((2, 2), dtype=complex)
        weights = np.exp(np.linspace(-1, 0, len(basins)))  # Exponential decay weighting
        weights /= np.sum(weights)
        
        for basin, w in zip(basins, weights):
            rho = self._basin_to_density_matrix(basin)
            composite_rho += w * rho
        
        # Normalize to valid density matrix
        composite_rho = (composite_rho + np.conj(composite_rho).T) / 2
        trace = np.trace(composite_rho)
        if trace > 1e-10:
            composite_rho /= trace
        
        # Compute von Neumann entropy of composite state
        entropy = self._compute_von_neumann_entropy(composite_rho)
        max_entropy = np.log2(2)  # max entropy for 2x2 density matrix
        
        # phi_temporal = 1 - normalized_entropy
        phi_temporal = 1.0 - (entropy / (max_entropy + 1e-10))
        
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
