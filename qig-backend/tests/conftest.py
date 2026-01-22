"""Pytest configuration for qig-backend tests.

Configures pytest to use importlib mode to avoid __init__.py relative import issues.
"""
import sys
from pathlib import Path
import pytest

# Add qig-backend directory to Python path for imports
# This must happen before any test modules try to import from qig_geometry, etc.
qig_backend_dir = Path(__file__).parent.parent
if str(qig_backend_dir) not in sys.path:
    sys.path.insert(0, str(qig_backend_dir))

# This file's presence enables proper test discovery.
# The import-mode is set via pytest.ini or pyproject.toml
