"""Pytest configuration for qig-backend tests.

Configures pytest to use importlib mode to avoid __init__.py relative import issues.
"""
import pytest

# This file's presence enables proper test discovery.
# The import-mode is set via pytest.ini or pyproject.toml
