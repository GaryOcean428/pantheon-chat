"""
Pytest Configuration for QIG Backend Tests

Sets up proper import paths to avoid relative import issues with __init__.py
"""

import sys
import os

# Add qig-backend directory to path BEFORE any imports
# This allows direct imports like 'from buffer_of_thoughts import ...'
qig_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if qig_backend_dir not in sys.path:
    sys.path.insert(0, qig_backend_dir)

# Prevent pytest from importing qig-backend as a package
# by removing it from the namespace if accidentally imported
if 'qig_backend' in sys.modules:
    del sys.modules['qig_backend']

# Common test fixtures can go here
import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def random_basin():
    """Generate a random 64D basin coordinate."""
    np.random.seed(42)
    return list(np.random.dirichlet(np.ones(64)))


@pytest.fixture
def uniform_basin():
    """Generate a uniform 64D basin coordinate."""
    return [1.0 / 64] * 64
