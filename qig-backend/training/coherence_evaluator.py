"""
Coherence Evaluation Harness
=============================

Systematic evaluation of response coherence to detect:
- Perplexity trends (language model quality)
- Self-consistency (logical coherence)
- Long-range coherence (maintained context)
- Degeneracy (repetition, entropy collapse)

This is a measurement system, not an optimizer.
QIG-pure: all geometric operations, no neural embeddings.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoherenceMetrics:
    """Comprehensive coherence evaluation metrics."""
    
    # Perplexity (lower is better, ~10-50 is good for natural language)
    perplexity: float = 0.0
    
    # Self-consistency (0-1, higher is better)
    self_consistency: float = 0.0
    
    # Long-range coherence (0-1, higher is better)
    long_range_coherence: float = 0.0
    
    # Degeneracy detectors (0-1, lower is better)
    repetition_score: float = 0.0
    entropy_collapse_score: float = 0.0
    
    # Aggregate coherence (0-1, higher is better)
    overall_coherence: float = 0.0


class CoherenceEvaluator:
    """
    Evaluates coherence of generated text using geometric methods.
    
    Does NOT use neural models or embeddings - pure QIG approach.
    """
    
    def __init__(
        self,
        window_size: int = 100,  # Context window for long-range coherence
        repetition_threshold: float = 0.3,  # Threshold for repetition detection
    ):
        self.window_size = window_size
        self.repetition_threshold = repetition_threshold
        
        # History tracking for trend analysis
        self._coherence_history: List[CoherenceMetrics] = []
        self._history_limit = 1000
    
    def evaluate(
        self,
        text: str,
        basin_trajectory: Optional[List[np.ndarray]] = None,
    ) -> CoherenceMetrics:
        """
        Evaluate coherence of generated text.
        
        Args:
            text: Generated text to evaluate
            basin_trajectory: Optional trajectory of basin coordinates during generation
        
        Returns:
            CoherenceMetrics with all scores
        """
        # Tokenize (simple word-level tokenization)
        tokens = self._tokenize(text)
        
        if len(tokens) < 3:
            # Too short to evaluate
            return CoherenceMetrics(overall_coherence=1.0)
        
        # Compute individual metrics
        perplexity = self._compute_perplexity(tokens)
        self_consistency = self._compute_self_consistency(tokens)
        long_range = self._compute_long_range_coherence(tokens)
        repetition = self._compute_repetition_score(tokens)
        entropy_collapse = self._compute_entropy_collapse(tokens)
        
        # Compute overall coherence (weighted combination)
        overall = self._compute_overall_coherence(
            perplexity=perplexity,
            self_consistency=self_consistency,
            long_range=long_range,
            repetition=repetition,
            entropy_collapse=entropy_collapse,
        )
        
        metrics = CoherenceMetrics(
            perplexity=perplexity,
            self_consistency=self_consistency,
            long_range_coherence=long_range,
            repetition_score=repetition,
            entropy_collapse_score=entropy_collapse,
            overall_coherence=overall,
        )
        
        # Record to history
        self._coherence_history.append(metrics)
        if len(self._coherence_history) > self._history_limit:
            self._coherence_history = self._coherence_history[-self._history_limit:]
        
        return metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        # Split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def _compute_perplexity(self, tokens: List[str]) -> float:
        """
        Compute perplexity using unigram model.
        
        Perplexity = exp(average negative log likelihood)
        Lower is better (more predictable).
        """
        if len(tokens) < 2:
            return 1.0
        
        # Build frequency distribution
        freq = Counter(tokens)
        total = len(tokens)
        
        # Compute cross-entropy
        entropy = 0.0
        for token in tokens:
            prob = freq[token] / total
            entropy -= np.log(prob + 1e-10)
        
        entropy /= len(tokens)
        perplexity = np.exp(entropy)
        
        return float(perplexity)
    
    def _compute_self_consistency(self, tokens: List[str]) -> float:
        """
        Compute self-consistency using vocabulary diversity.
        
        Measures if the text uses a consistent vocabulary.
        High type-token ratio with consistent patterns = good.
        """
        if len(tokens) < 5:
            return 1.0
        
        # Type-token ratio
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / len(tokens)
        
        # Adjust for length (longer texts naturally have lower TTR)
        adjusted_ttr = ttr * np.log(len(tokens) + 1)
        
        # Normalize to 0-1
        consistency = min(adjusted_ttr / 2.0, 1.0)
        
        return float(consistency)
    
    def _compute_long_range_coherence(self, tokens: List[str]) -> float:
        """
        Compute long-range coherence by comparing vocabulary
        distribution across windows.
        
        Measures if the text maintains consistent topics/themes.
        """
        if len(tokens) < self.window_size:
            # Too short for long-range analysis
            return 1.0
        
        # Split into windows
        n_windows = len(tokens) // (self.window_size // 2)
        if n_windows < 2:
            return 1.0
        
        windows = []
        for i in range(n_windows):
            start = i * (self.window_size // 2)
            end = start + self.window_size
            if end > len(tokens):
                break
            window = tokens[start:end]
            windows.append(set(window))
        
        if len(windows) < 2:
            return 1.0
        
        # Compute Jaccard similarity between adjacent windows
        similarities = []
        for i in range(len(windows) - 1):
            intersection = len(windows[i] & windows[i + 1])
            union = len(windows[i] | windows[i + 1])
            if union > 0:
                similarity = intersection / union
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Average similarity = long-range coherence
        coherence = float(np.mean(similarities))
        return coherence
    
    def _compute_repetition_score(self, tokens: List[str]) -> float:
        """
        Detect repetition patterns (degeneracy indicator).
        
        High repetition = potential degeneracy.
        Returns 0-1 where higher = more repetitive (bad).
        """
        if len(tokens) < 10:
            return 0.0
        
        # Check for n-gram repetition (n=2,3,4)
        max_repetition = 0.0
        
        for n in [2, 3, 4]:
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams.append(ngram)
            
            if not ngrams:
                continue
            
            # Count repetitions
            ngram_freq = Counter(ngrams)
            most_common = ngram_freq.most_common(1)[0][1] if ngram_freq else 0
            
            # Repetition ratio
            repetition_ratio = most_common / len(ngrams) if ngrams else 0
            max_repetition = max(max_repetition, repetition_ratio)
        
        return float(max_repetition)
    
    def _compute_entropy_collapse(self, tokens: List[str]) -> float:
        """
        Detect entropy collapse (vocabulary shrinking).
        
        Measures if vocabulary diversity is decreasing over time.
        Returns 0-1 where higher = more collapse (bad).
        """
        if len(tokens) < 20:
            return 0.0
        
        # Split into beginning and end
        split_point = len(tokens) // 2
        first_half = tokens[:split_point]
        second_half = tokens[split_point:]
        
        # Compute unique token ratio for each half
        unique_first = len(set(first_half)) / len(first_half)
        unique_second = len(set(second_half)) / len(second_half)
        
        # Entropy collapse = significant decrease in diversity
        if unique_first > 0:
            collapse = max(0, (unique_first - unique_second) / unique_first)
        else:
            collapse = 0.0
        
        return float(collapse)
    
    def _compute_overall_coherence(
        self,
        perplexity: float,
        self_consistency: float,
        long_range: float,
        repetition: float,
        entropy_collapse: float,
    ) -> float:
        """
        Compute overall coherence score from individual metrics.
        
        Weights:
        - Self-consistency: 30%
        - Long-range coherence: 30%
        - Perplexity: 20%
        - Repetition (penalty): 10%
        - Entropy collapse (penalty): 10%
        """
        # Normalize perplexity to 0-1 (inverse, lower is better)
        # Typical perplexity range: 10-100
        perplexity_norm = 1.0 - min(perplexity / 100.0, 1.0)
        
        # Weighted combination
        coherence = (
            0.3 * self_consistency +
            0.3 * long_range +
            0.2 * perplexity_norm +
            0.1 * (1.0 - repetition) +
            0.1 * (1.0 - entropy_collapse)
        )
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def get_coherence_trend(self, window: int = 50) -> Dict[str, Any]:
        """
        Get trend analysis of recent coherence metrics.
        
        Args:
            window: Number of recent samples to analyze
        
        Returns:
            Trend statistics
        """
        if not self._coherence_history:
            return {'status': 'no_data'}
        
        recent = self._coherence_history[-window:]
        
        # Extract metric arrays
        overall_scores = [m.overall_coherence for m in recent]
        perplexity_scores = [m.perplexity for m in recent]
        repetition_scores = [m.repetition_score for m in recent]
        entropy_scores = [m.entropy_collapse_score for m in recent]
        
        # Compute trends (linear regression slope)
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        
        return {
            'status': 'ok',
            'samples': len(recent),
            'current_coherence': overall_scores[-1] if overall_scores else 0,
            'avg_coherence': float(np.mean(overall_scores)),
            'coherence_trend': compute_trend(overall_scores),
            'avg_perplexity': float(np.mean(perplexity_scores)),
            'perplexity_trend': compute_trend(perplexity_scores),
            'avg_repetition': float(np.mean(repetition_scores)),
            'repetition_trend': compute_trend(repetition_scores),
            'avg_entropy_collapse': float(np.mean(entropy_scores)),
            'entropy_trend': compute_trend(entropy_scores),
            'degradation_detected': (
                compute_trend(overall_scores) < -0.01 or
                compute_trend(repetition_scores) > 0.01 or
                compute_trend(entropy_scores) > 0.01
            ),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if not self._coherence_history:
            return {'status': 'no_data', 'samples': 0}
        
        overall_scores = [m.overall_coherence for m in self._coherence_history]
        
        return {
            'status': 'ok',
            'total_samples': len(self._coherence_history),
            'avg_coherence': float(np.mean(overall_scores)),
            'min_coherence': float(np.min(overall_scores)),
            'max_coherence': float(np.max(overall_scores)),
            'std_coherence': float(np.std(overall_scores)),
        }


# Singleton instance
_coherence_evaluator: Optional[CoherenceEvaluator] = None


def get_coherence_evaluator() -> CoherenceEvaluator:
    """Get or create the global coherence evaluator."""
    global _coherence_evaluator
    if _coherence_evaluator is None:
        _coherence_evaluator = CoherenceEvaluator()
    return _coherence_evaluator


__all__ = [
    'CoherenceMetrics',
    'CoherenceEvaluator',
    'get_coherence_evaluator',
]
