"""
Curriculum-Only Mode Guard for Python

Provides curriculum-only mode checking for Python backend components.
When QIG_CURRICULUM_ONLY=true, all external web searches are blocked.

Usage:
    from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock
    
    if is_curriculum_only_enabled():
        raise CurriculumOnlyBlock("External search blocked in curriculum-only mode")
"""

import os
import logging

logger = logging.getLogger(__name__)


class CurriculumOnlyBlock(Exception):
    """
    Exception raised when an operation is blocked by curriculum-only mode.
    
    This indicates that external web searches are disabled because the system
    is operating in curriculum-only training mode.
    """
    pass


def is_curriculum_only_enabled() -> bool:
    """
    Check if curriculum-only mode is enabled.
    
    Returns:
        True if QIG_CURRICULUM_ONLY environment variable is set to 'true'
        False otherwise
        
    When curriculum-only mode is enabled:
    - All external web searches are blocked
    - Only curriculum-based training data is used
    - System operates on pre-validated tokens only
    """
    return os.environ.get('QIG_CURRICULUM_ONLY', '').lower() == 'true'


def check_curriculum_guard(operation_name: str = "operation") -> None:
    """
    Check curriculum-only mode and raise exception if blocked.
    
    Args:
        operation_name: Name of the operation being blocked (for logging)
        
    Raises:
        CurriculumOnlyBlock: If curriculum-only mode is enabled
        
    Usage:
        check_curriculum_guard("external web search")
    """
    if is_curriculum_only_enabled():
        msg = f"{operation_name} blocked by curriculum-only mode"
        logger.warning(f"[CurriculumGuard] {msg}")
        raise CurriculumOnlyBlock(msg)
