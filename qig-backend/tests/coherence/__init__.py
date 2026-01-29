"""
QIG Coherence Test Harness
===========================

Reproducible coherence evaluation framework for QIG generation architectures.

Version: 1.0
Author: WP4.3 Implementation
Date: 2026-01-20
Protocol: Ultra Consciousness v4.0 ACTIVE

Usage:
    # Run tests
    pytest tests/coherence/ -v
    
    # Compare architectures
    python tests/coherence/compare_architectures.py
    
    # Generate report
    python tests/coherence/generate_report.py
"""

__version__ = "1.0"
__author__ = "WP4.3 Implementation"
__protocol__ = "Ultra Consciousness v4.0 ACTIVE"

from . import metrics
from .test_helpers import (
    load_test_prompts,
    load_test_configurations,
    load_test_seeds,
    set_reproducible_seed,
    get_prompt_seed,
)

__all__ = [
    'metrics',
    'load_test_prompts',
    'load_test_configurations',
    'load_test_seeds',
    'set_reproducible_seed',
    'get_prompt_seed',
]
