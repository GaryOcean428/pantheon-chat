"""
QIG Search Module

Provides unified search capabilities across multiple providers.
Uses geometric reasoning to select the best provider for each query.
"""

from .duckduckgo_adapter import (
    DuckDuckGoSearch,
    get_ddg_search,
    search_duckduckgo,
)

from .provider_selector import (
    GeometricProviderSelector,
    ProviderStats,
    get_provider_selector,
)

__all__ = [
    'DuckDuckGoSearch',
    'get_ddg_search',
    'search_duckduckgo',
    'GeometricProviderSelector',
    'ProviderStats',
    'get_provider_selector',
]
