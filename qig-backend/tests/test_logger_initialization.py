"""
Test that logger is properly initialized before use in ocean_qig_core.py

This test validates the fix for the critical NameError where 'logger' was not defined
at line 57, causing Python backend crashes.

Author: GitHub Copilot
Date: 2026-02-01
"""

import ast
import pytest
from pathlib import Path


def test_logger_defined_before_use():
    """
    Verify that logger is defined before any logger.* calls in ocean_qig_core.py
    
    This is a static analysis test that parses the Python AST to ensure
    logger is assigned before being used, preventing NameError crashes.
    """
    ocean_qig_file = Path(__file__).parent.parent / "ocean_qig_core.py"
    
    with open(ocean_qig_file, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    logger_defined_line = None
    first_logger_use_line = None
    
    # Walk the AST to find logger assignment and usage
    for node in ast.walk(tree):
        # Check for logger assignment: logger = logging.getLogger(...)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'logger':
                    if logger_defined_line is None or node.lineno < logger_defined_line:
                        logger_defined_line = node.lineno
        
        # Check for logger attribute access: logger.info(), logger.warning(), etc.
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'logger':
                if first_logger_use_line is None or node.lineno < first_logger_use_line:
                    first_logger_use_line = node.lineno
    
    # Assert that logger is defined before it's used
    assert logger_defined_line is not None, "logger variable should be defined in ocean_qig_core.py"
    assert first_logger_use_line is not None, "logger should be used in ocean_qig_core.py"
    assert logger_defined_line < first_logger_use_line, (
        f"logger must be defined (line {logger_defined_line}) before first use "
        f"(line {first_logger_use_line}). This prevents NameError crashes."
    )


def test_logger_in_import_block():
    """
    Verify that the import block that uses logger comes after logger is defined.
    
    Specifically checks that the first gravitational_decoherence import (which logs on
    success/failure) comes after logger initialization.
    """
    ocean_qig_file = Path(__file__).parent.parent / "ocean_qig_core.py"
    
    with open(ocean_qig_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger_init_line = None
    first_decoherence_import_line = None
    
    for i, line in enumerate(lines, start=1):
        # Find logger initialization (first occurrence)
        if 'logger = logging.getLogger' in line and logger_init_line is None:
            logger_init_line = i
        
        # Find first gravitational_decoherence import
        if 'from gravitational_decoherence import' in line and first_decoherence_import_line is None:
            first_decoherence_import_line = i
    
    # If decoherence import exists, it must come after logger init
    if first_decoherence_import_line is not None:
        assert logger_init_line is not None, "logger must be initialized if decoherence import exists"
        assert logger_init_line < first_decoherence_import_line, (
            f"logger initialization (line {logger_init_line}) must come before "
            f"first gravitational_decoherence import (line {first_decoherence_import_line}) "
            f"because the import block uses logger.info() and logger.warning()"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
