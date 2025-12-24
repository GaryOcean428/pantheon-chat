"""
Geometric Search Module

QIG-based search tool selection using geometric metrics.
Integrates with the Olympus Pantheon for intelligent search routing.
QIG-PURE: Result synthesis uses internal generative service, no external LLMs.
"""

from .query_encoder import SearchQueryEncoder
from .tool_selector import SearchToolSelector, ToolSelection
from .search_orchestrator import (
    SearchOrchestrator, 
    SearchResult, 
    AggregatedResult,
    GENERATIVE_SERVICE_AVAILABLE
)
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
    "GENERATIVE_SERVICE_AVAILABLE",
]

__version__ = "1.0.0"
