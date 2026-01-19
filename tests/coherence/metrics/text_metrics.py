"""
Text Validity Metrics for QIG Coherence Testing

Surface-level text quality checks (NOT semantic analysis):
- UTF-8 validity: No invalid byte sequences
- Length distribution: Reasonable output length
- Repetition detection: N-gram entropy
- Token boundary sanity: No impossible sequences

These are basic validity checks, not NLP semantics.

Author: QIG Consciousness Project
Date: January 2026
"""

import re
import unicodedata
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class TextMetrics:
    """Container for text validity metrics."""
    is_valid_utf8: bool            # UTF-8 validity
    length: int                    # Character count
    word_count: int                # Word count
    repetition_score: float        # 0-1, higher = more repetitive
    entropy: float                 # N-gram entropy
    unique_words: int              # Vocabulary size
    avg_word_length: float         # Average word length
    has_invalid_sequences: bool    # Impossible token sequences
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'is_valid_utf8': self.is_valid_utf8,
            'length': self.length,
            'word_count': self.word_count,
            'repetition_score': self.repetition_score,
            'entropy': self.entropy,
            'unique_words': self.unique_words,
            'avg_word_length': self.avg_word_length,
            'has_invalid_sequences': self.has_invalid_sequences,
        }


def check_utf8_validity(text: str) -> bool:
    """
    Check if text is valid UTF-8.
    
    Args:
        text: Input text
        
    Returns:
        True if valid UTF-8
    """
    try:
        # Try encoding/decoding
        text.encode('utf-8').decode('utf-8')
        
        # Check for replacement characters (indicates encoding issues)
        if '\ufffd' in text:
            return False
        
        # Check for control characters (except whitespace)
        for char in text:
            if unicodedata.category(char)[0] == 'C' and char not in '\n\r\t':
                return False
        
        return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False


def compute_repetition_score(text: str, n: int = 3) -> float:
    """
    Compute repetition score using n-gram analysis.
    
    Higher score = more repetitive.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Repetition score (0-1)
    """
    if len(text) < n:
        return 0.0
    
    # Extract n-grams
    ngrams = []
    words = text.lower().split()
    
    if len(words) < n:
        return 0.0
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    # Count duplicates
    counts = Counter(ngrams)
    total = len(ngrams)
    unique = len(counts)
    
    # Repetition = 1 - (unique / total)
    repetition = 1.0 - (unique / total)
    
    return repetition


def compute_ngram_entropy(text: str, n: int = 2) -> float:
    """
    Compute n-gram entropy (Shannon entropy).
    
    Higher entropy = more diverse/unpredictable.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Entropy value
    """
    if len(text) < n:
        return 0.0
    
    # Character-level n-grams
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    # Compute probabilities
    counts = Counter(ngrams)
    total = len(ngrams)
    
    # Shannon entropy
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy


def detect_invalid_sequences(text: str) -> bool:
    """
    Detect obviously invalid token sequences.
    
    Checks for:
    - Multiple consecutive punctuation (except ellipsis)
    - Spaces before punctuation
    - Missing spaces after punctuation
    - Repeated words (same word 3+ times in a row)
    
    Args:
        text: Input text
        
    Returns:
        True if invalid sequences detected
    """
    # Multiple consecutive punctuation (except ...)
    if re.search(r'[.!?,;:]{3,}', text):
        if not re.search(r'\.\.\.+', text):  # Allow ellipsis
            return True
    
    # Space before punctuation
    if re.search(r'\s+[.!?,;:]', text):
        return True
    
    # Missing space after punctuation (except in numbers)
    if re.search(r'[.!?,;:][a-zA-Z]', text):
        return True
    
    # Repeated words (3+ times)
    words = text.split()
    for i in range(len(words) - 2):
        if len(words[i]) > 2 and words[i] == words[i+1] == words[i+2]:
            return True
    
    return False


def compute_text_metrics(text: str) -> TextMetrics:
    """
    Compute all text validity metrics.
    
    Args:
        text: Generated text
        
    Returns:
        TextMetrics containing all measurements
    """
    # UTF-8 validity
    is_valid = check_utf8_validity(text)
    
    # Length metrics
    length = len(text)
    words = text.split()
    word_count = len(words)
    
    # Unique words
    unique_words = len(set(word.lower() for word in words))
    
    # Average word length
    if words:
        avg_word_length = sum(len(word) for word in words) / len(words)
    else:
        avg_word_length = 0.0
    
    # Repetition
    repetition = compute_repetition_score(text, n=3)
    
    # Entropy
    entropy = compute_ngram_entropy(text, n=2)
    
    # Invalid sequences
    has_invalid = detect_invalid_sequences(text)
    
    return TextMetrics(
        is_valid_utf8=is_valid,
        length=length,
        word_count=word_count,
        repetition_score=repetition,
        entropy=entropy,
        unique_words=unique_words,
        avg_word_length=avg_word_length,
        has_invalid_sequences=has_invalid,
    )


def compare_text_metrics(
    metrics_a: TextMetrics,
    metrics_b: TextMetrics
) -> Dict[str, any]:
    """
    Compare two text metric sets.
    
    Args:
        metrics_a: First metrics
        metrics_b: Second metrics
        
    Returns:
        Dictionary of comparisons
    """
    return {
        'both_valid_utf8': metrics_a.is_valid_utf8 and metrics_b.is_valid_utf8,
        'length_delta': metrics_b.length - metrics_a.length,
        'repetition_delta': metrics_a.repetition_score - metrics_b.repetition_score,  # Lower better
        'entropy_delta': metrics_b.entropy - metrics_a.entropy,  # Higher better
        'unique_words_delta': metrics_b.unique_words - metrics_a.unique_words,
        'both_no_invalid': not metrics_a.has_invalid_sequences and not metrics_b.has_invalid_sequences,
    }


def is_text_valid(metrics: TextMetrics, min_length: int = 10, max_repetition: float = 0.7) -> bool:
    """
    Check if text passes basic validity criteria.
    
    Args:
        metrics: Text metrics
        min_length: Minimum acceptable length
        max_repetition: Maximum acceptable repetition
        
    Returns:
        True if text is valid
    """
    return (
        metrics.is_valid_utf8 and
        metrics.length >= min_length and
        metrics.repetition_score <= max_repetition and
        not metrics.has_invalid_sequences and
        metrics.word_count > 0
    )
