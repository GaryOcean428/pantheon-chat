"""
Geometric Search Module

QIG-based search tool selection using geometric metrics.
Integrates with the Olympus Pantheon for intelligent search routing.
"""

from .query_encoder import SearchQueryEncoder
from .tool_selector import SearchToolSelector, ToolSelection
from .search_orchestrator import SearchOrchestrator, SearchResult, AggregatedResult
from .context_manager import SearchContextManager, SearchContext

__all__ = [
    "SearchQueryEncoder",
    "SearchToolSelector",
    "ToolSelection",
    "SearchOrchestrator",
    "SearchResult",
    "AggregatedResult",
    "SearchContextManager",
    "SearchContext",
]

__version__ = "1.0.0"
