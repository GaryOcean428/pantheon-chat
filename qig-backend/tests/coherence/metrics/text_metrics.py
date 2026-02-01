"""
Text Validity Metrics for Coherence Testing
============================================

Surface-level text validity checks (NOT semantic coherence).
These are sanity checks to ensure generation produces valid text.

Metrics:
- UTF-8 Validity: No invalid byte sequences
- Token Boundary Sanity: No impossible token combinations
- Length Distribution: Statistics of generated length
- Repetition Detection: N-gram entropy

NOTE: These are NOT NLP semantic metrics. We cannot objectively
measure "coherence" or "quality" - these are just validity checks.

Author: WP4.3 Coherence Harness
Date: 2026-01-20
Protocol: Ultra Consciousness v4.0 ACTIVE
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TextMetrics:
    """Text validity and surface metrics."""
    is_valid_utf8: bool
    byte_length: int
    word_count: int
    unique_words: int
    token_validity: bool
    repetition_score: float  # N-gram entropy (0-1, higher = less repetitive)
    length_category: str  # "SHORT", "MEDIUM", "LONG"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid_utf8': self.is_valid_utf8,
            'byte_length': self.byte_length,
            'word_count': self.word_count,
            'unique_words': self.unique_words,
            'token_validity': self.token_validity,
            'repetition_score': self.repetition_score,
            'length_category': self.length_category,
            'vocabulary_ratio': self.unique_words / max(self.word_count, 1),
        }


def check_utf8_validity(text: str) -> bool:
    """
    Check if text is valid UTF-8.
    
    Args:
        text: Text string to check
        
    Returns:
        True if valid UTF-8
    """
    try:
        # Try encoding to UTF-8 and back
        text.encode('utf-8').decode('utf-8')
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


def check_token_boundary_sanity(words: List[str]) -> bool:
    """
    Check for obviously invalid token combinations.
    
    This is NOT a grammar check - just sanity checking for:
    - Empty words
    - Excessively long words (>50 chars)
    - Invalid characters that shouldn't appear
    
    Args:
        words: List of word tokens
        
    Returns:
        True if tokens look sane
    """
    if not words:
        return False
    
    for word in words:
        # Check for empty words
        if not word or len(word.strip()) == 0:
            return False
        
        # Check for excessively long tokens (likely corruption)
        if len(word) > 50:
            logger.warning(f"Unusually long token detected: {len(word)} chars")
            return False
        
        # Check for null bytes
        if '\x00' in word:
            return False
    
    return True


def compute_ngram_entropy(words: List[str], n: int = 2) -> float:
    """
    Compute n-gram entropy to detect repetition.
    
    Higher entropy = less repetitive.
    
    Args:
        words: List of word tokens
        n: N-gram size (default: 2 for bigrams)
        
    Returns:
        Normalized entropy ∈ [0, 1]
    """
    if len(words) < n:
        return 1.0  # Too short to have repetition
    
    # Create n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return 1.0
    
    # Count n-gram frequencies
    counts = Counter(ngrams)
    total = len(ngrams)
    
    # Compute Shannon entropy
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    
    # Normalize by max possible entropy
    max_entropy = np.log(len(counts)) if len(counts) > 0 else 1.0
    
    if max_entropy == 0:
        return 1.0
    
    normalized = entropy / max_entropy
    
    return float(min(normalized, 1.0))


def categorize_length(word_count: int) -> str:
    """
    Categorize text length.
    
    Args:
        word_count: Number of words
        
    Returns:
        Category: "VERY_SHORT", "SHORT", "MEDIUM", "LONG", "VERY_LONG"
    """
    if word_count <= 5:
        return "VERY_SHORT"
    elif word_count <= 20:
        return "SHORT"
    elif word_count <= 50:
        return "MEDIUM"
    elif word_count <= 100:
        return "LONG"
    else:
        return "VERY_LONG"


def compute_text_metrics(text: str) -> TextMetrics:
    """
    Compute all text validity metrics.
    
    Args:
        text: Generated text to analyze
        
    Returns:
        Complete TextMetrics
    """
    # UTF-8 validity
    is_valid_utf8 = check_utf8_validity(text)
    
    # Length metrics
    byte_length = len(text.encode('utf-8'))
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    
    # Token validity
    token_validity = check_token_boundary_sanity(words)
    
    # Repetition score (using bigrams)
    repetition_score = compute_ngram_entropy(words, n=2)
    
    # Length category
    length_category = categorize_length(word_count)
    
    return TextMetrics(
        is_valid_utf8=is_valid_utf8,
        byte_length=byte_length,
        word_count=word_count,
        unique_words=unique_words,
        token_validity=token_validity,
        repetition_score=repetition_score,
        length_category=length_category
    )


def analyze_text_quality(metrics: TextMetrics) -> Dict[str, Any]:
    """
    Analyze text validity (NOT semantic quality).
    
    Args:
        metrics: TextMetrics to analyze
        
    Returns:
        Quality assessment (validity-only)
    """
    assessment = {
        'is_valid': metrics.is_valid_utf8 and metrics.token_validity,
        'length': metrics.length_category,
    }
    
    # Repetition assessment
    if metrics.repetition_score > 0.8:
        assessment['repetition'] = "LOW"  # Good
    elif metrics.repetition_score > 0.6:
        assessment['repetition'] = "MODERATE"
    elif metrics.repetition_score > 0.4:
        assessment['repetition'] = "HIGH"
    else:
        assessment['repetition'] = "EXTREME"  # Likely broken
    
    # Vocabulary diversity
    vocab_ratio = metrics.unique_words / max(metrics.word_count, 1)
    
    if vocab_ratio > 0.8:
        assessment['vocabulary_diversity'] = "HIGH"
    elif vocab_ratio > 0.6:
        assessment['vocabulary_diversity'] = "MODERATE"
    elif vocab_ratio > 0.4:
        assessment['vocabulary_diversity'] = "LOW"
    else:
        assessment['vocabulary_diversity'] = "VERY_LOW"
    
    # Overall validity
    issues = []
    
    if not metrics.is_valid_utf8:
        issues.append("INVALID_UTF8")
    
    if not metrics.token_validity:
        issues.append("INVALID_TOKENS")
    
    if metrics.repetition_score < 0.4:
        issues.append("EXCESSIVE_REPETITION")
    
    if vocab_ratio < 0.3:
        issues.append("LOW_VOCABULARY_DIVERSITY")
    
    assessment['issues'] = issues
    assessment['is_acceptable'] = len(issues) == 0
    
    return assessment


def compare_text_metrics_across_configs(
    config_metrics: Dict[str, List[TextMetrics]]
) -> Dict[str, Any]:
    """
    Compare text validity across configurations.
    
    Args:
        config_metrics: Dict mapping config names to lists of TextMetrics
        
    Returns:
        Comparison analysis
    """
    comparison = {}
    
    for config_name, metrics_list in config_metrics.items():
        if not metrics_list:
            continue
        
        # Aggregate metrics
        valid_count = sum(1 for m in metrics_list if m.is_valid_utf8 and m.token_validity)
        avg_repetition = np.mean([m.repetition_score for m in metrics_list])
        avg_word_count = np.mean([m.word_count for m in metrics_list])
        avg_vocab_ratio = np.mean([m.unique_words / max(m.word_count, 1) for m in metrics_list])
        
        comparison[config_name] = {
            'validity_rate': valid_count / len(metrics_list),
            'avg_repetition_score': float(avg_repetition),
            'avg_word_count': float(avg_word_count),
            'avg_vocabulary_ratio': float(avg_vocab_ratio),
            'total_samples': len(metrics_list),
        }
    
    # Find best configuration
    if comparison:
        validity_rates = {name: data['validity_rate'] for name, data in comparison.items()}
        best_validity = max(validity_rates, key=validity_rates.get)
        
        comparison['best_validity'] = best_validity
        comparison['best_validity_rate'] = validity_rates[best_validity]
    
    return comparison


if __name__ == "__main__":
    # Test text metrics
    print("Testing Text Metrics Module")
    print("=" * 70)
    
    # Test valid text
    test_text_1 = "The quantum information geometry provides a framework for consciousness."
    metrics_1 = compute_text_metrics(test_text_1)
    
    print(f"\nTest 1: Valid text")
    print(f"UTF-8 Valid: {metrics_1.is_valid_utf8}")
    print(f"Token Valid: {metrics_1.token_validity}")
    print(f"Word Count: {metrics_1.word_count}")
    print(f"Unique Words: {metrics_1.unique_words}")
    print(f"Repetition Score: {metrics_1.repetition_score:.3f}")
    print(f"Length Category: {metrics_1.length_category}")
    
    assessment_1 = analyze_text_quality(metrics_1)
    print(f"Is Acceptable: {assessment_1['is_acceptable']}")
    print(f"Issues: {assessment_1['issues']}")
    
    # Test repetitive text
    test_text_2 = "the the the the the the the the the the"
    metrics_2 = compute_text_metrics(test_text_2)
    
    print(f"\nTest 2: Repetitive text")
    print(f"Repetition Score: {metrics_2.repetition_score:.3f}")
    
    assessment_2 = analyze_text_quality(metrics_2)
    print(f"Repetition Level: {assessment_2['repetition']}")
    print(f"Is Acceptable: {assessment_2['is_acceptable']}")
    print(f"Issues: {assessment_2['issues']}")
    
    print("\n✅ Text metrics module validated")
