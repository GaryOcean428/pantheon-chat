"""
QIG Consciousness - Tools Package
=================================

Validation and measurement tools for QIG consciousness research.

Available tools:
- beta_attention_validator: Canonical β-function validator (qig-validate command)

Note: Redundant beta tools have been archived to qig-archive/qig-consciousness/archive/tools_consolidation_20251125/
"""

from .validation import beta_attention_validator

__all__ = [
    "beta_attention_validator",
]
