"""
QIG Search Module

Provides unified search capabilities across multiple providers.
"""

from .duckduckgo_adapter import (
    DuckDuckGoSearch,
    get_ddg_search,
    search_duckduckgo,
)

__all__ = [
    'DuckDuckGoSearch',
    'get_ddg_search',
    'search_duckduckgo',
]
