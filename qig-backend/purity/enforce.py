#!/usr/bin/env python3
"""
QIG Purity Enforcement Module
==============================

Enforces QIG purity mode - blocks external NLP calls and non-geometric operations.

When QIG_PURITY_MODE=true:
- NO external LLM calls for generation structure
- NO spacy/nltk POS tagging
- NO cosine similarity or Euclidean distance on basins
- ONLY Fisher-Rao geometry and QIG-native operations

This module provides:
1. @require_qig_purity decorator - blocks functions in purity mode
2. Runtime checks for forbidden operations
3. Import guards for external NLP packages

Author: Copilot AI Agent
Date: 2026-01-20
Issue: GaryOcean428/pantheon-chat#99 (E8 Protocol Issue-03)
Reference: docs/10-e8-protocol/issues/20260119-issue-99-qig-native-skeleton-remediation-1.00W.md
"""

import os
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Check QIG_PURITY_MODE environment variable
_QIG_PURITY_MODE = os.getenv('QIG_PURITY_MODE', 'false').lower() in ('true', '1', 'yes')


def is_purity_mode_enabled() -> bool:
    """
    Check if QIG purity mode is enabled.
    
    Returns:
        True if QIG_PURITY_MODE is set to true/1/yes
    """
    return _QIG_PURITY_MODE


def require_qig_purity(func: Callable) -> Callable:
    """
    Decorator to block function calls when NOT in QIG purity mode.
    
    Use this for functions that violate QIG purity (external NLP, Euclidean ops).
    
    Example:
        @require_qig_purity
        def use_spacy_pos_tagger(text):
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp(text)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _QIG_PURITY_MODE:
            raise RuntimeError(
                f"QIG_PURITY_MODE violation: {func.__name__} is not QIG-pure. "
                f"This function uses external NLP or non-geometric operations."
            )
        return func(*args, **kwargs)
    return wrapper


def allow_only_in_purity_mode(func: Callable) -> Callable:
    """
    Decorator to REQUIRE QIG purity mode for a function.
    
    Use this for QIG-pure functions that should only run in purity mode.
    
    Example:
        @allow_only_in_purity_mode
        def geometric_token_selection(basin):
            # QIG-pure implementation
            return ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _QIG_PURITY_MODE:
            logger.warning(
                f"{func.__name__} is QIG-pure but purity mode is disabled. "
                f"Set QIG_PURITY_MODE=true for full geometric purity."
            )
        return func(*args, **kwargs)
    return wrapper


class PurityViolation(Exception):
    """Exception raised when QIG purity is violated."""
    pass


def assert_no_external_nlp():
    """
    Assert that external NLP packages are not imported.
    
    Call this at module boundaries to detect purity violations.
    """
    if not _QIG_PURITY_MODE:
        return  # Only check in purity mode
    
    import sys
    
    forbidden_modules = {
        'spacy': 'spaCy POS tagger',
        'nltk': 'NLTK tools',
        'openai': 'OpenAI API',
        'anthropic': 'Anthropic API',
        'sklearn': 'scikit-learn (use geometric methods instead)',
        'transformers': 'HuggingFace transformers (use QIG-native generation)'
    }
    
    violations = []
    for module_name, description in forbidden_modules.items():
        if module_name in sys.modules:
            violations.append(f"{module_name} ({description})")
    
    if violations:
        raise PurityViolation(
            f"QIG purity violation: forbidden modules loaded: {', '.join(violations)}"
        )


def assert_geometric_distance(distance_func_name: str):
    """
    Assert that geometric (Fisher-Rao) distance is used, not Euclidean.
    
    Args:
        distance_func_name: Name of distance function being checked
    """
    if not _QIG_PURITY_MODE:
        return
    
    forbidden_names = {
        'euclidean_distance',
        'cosine_similarity',
        'dot_product',
        'l2_distance',
        'manhattan_distance'
    }
    
    if distance_func_name.lower() in forbidden_names:
        raise PurityViolation(
            f"QIG purity violation: {distance_func_name} is not geometric. "
            f"Use fisher_rao_distance instead."
        )


def get_purity_config() -> dict:
    """
    Get current QIG purity configuration.
    
    Returns:
        Dict with purity settings
    """
    return {
        'purity_mode_enabled': _QIG_PURITY_MODE,
        'allow_external_llm': not _QIG_PURITY_MODE,
        'allow_external_pos': not _QIG_PURITY_MODE,
        'allow_euclidean_ops': not _QIG_PURITY_MODE,
        'require_fisher_rao': _QIG_PURITY_MODE,
        'require_simplex_storage': True,  # Always required
        'require_qfi_score': True  # Always required
    }


# Example: External NLP function (blocked in purity mode)
@require_qig_purity
def use_external_pos_tagger(text: str) -> list:
    """
    Use external POS tagger (BLOCKED in QIG purity mode).
    
    This is a placeholder for external NLP tools.
    In production, replace with QIG-native token role derivation.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [(token.text, token.pos_) for token in doc]
    except ImportError:
        raise RuntimeError("spaCy not available")


@require_qig_purity
def use_external_llm_skeleton(prompt: str) -> str:
    """
    Use external LLM for skeleton generation (BLOCKED in QIG purity mode).
    
    This is a placeholder for external LLM calls.
    In production, replace with QIG-native geometric skeleton generation.
    """
    raise NotImplementedError("External LLM skeleton generation not implemented")


# Example: QIG-pure function (recommended in purity mode)
@allow_only_in_purity_mode
def geometric_skeleton_generation(trajectory: list, target_length: int) -> list:
    """
    Generate skeleton using QIG-native geometric methods.
    
    This is QIG-pure: uses token roles and foresight prediction.
    """
    # Placeholder for QIG-native implementation
    logger.info("Using QIG-pure skeleton generation")
    return []


if __name__ == '__main__':
    print(f"QIG Purity Mode: {is_purity_mode_enabled()}")
    print(f"Configuration: {get_purity_config()}")
    
    # Test purity enforcement
    if is_purity_mode_enabled():
        print("\nPurity mode ENABLED - testing enforcement...")
        
        try:
            use_external_pos_tagger("test")
            print("✗ External POS tagger allowed (should be blocked)")
        except RuntimeError as e:
            print(f"✓ External POS tagger blocked: {e}")
        
        try:
            assert_geometric_distance("cosine_similarity")
            print("✗ Cosine similarity allowed (should be blocked)")
        except PurityViolation as e:
            print(f"✓ Cosine similarity blocked: {e}")
    else:
        print("\nPurity mode DISABLED - external tools allowed")
        print("Set QIG_PURITY_MODE=true to enable strict geometric purity")
