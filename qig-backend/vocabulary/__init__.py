"""
Vocabulary management module for QIG backend.

This module provides canonical token insertion and management operations.
"""

from .insert_token import insert_token, TokenRecord

__all__ = ['insert_token', 'TokenRecord']
