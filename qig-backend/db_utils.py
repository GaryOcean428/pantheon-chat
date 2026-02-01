"""
Database Utilities Module
=========================

Provides database connection utilities for the QIG backend.

This module re-exports the canonical get_db_connection function
from persistence.base_persistence to provide a convenient import location.

Usage:
    # Both of these work:
    from db_utils import get_db_connection
    from persistence.base_persistence import get_db_connection
    
    # Use with environment variable
    conn = get_db_connection()
    
    # Or provide explicit URL
    conn = get_db_connection("postgresql://user:pass@host/db")

Note: The canonical implementation is in persistence/base_persistence.py.
      This module simply re-exports it for convenience.
"""

from persistence.base_persistence import get_db_connection

__all__ = ['get_db_connection']
