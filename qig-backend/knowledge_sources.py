"""
Pluggable Knowledge Source System

Provides a unified interface for any knowledge source to be used for autonomous learning.
Kernels can dynamically register new sources at runtime.

ARCHITECTURE:
1. KnowledgeSource (ABC): Base interface for all sources
2. KnowledgeSourceRegistry: Manages source registration/discovery
3. KnowledgeOrchestrator: Selects and queries sources based on context
4. Adapters: Wrap existing systems (DuckDuckGo, Wikipedia, GitHub, arXiv, Shadow)

FULL AUTONOMY: Any source can be added. Kernels self-register sources.
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SourceCapability(Enum):
    """What a source can provide."""
    WEB_SEARCH = "web_search"
    ENCYCLOPEDIC = "encyclopedic"
    CODE_REPOS = "code_repos"
    ACADEMIC = "academic"
    DARKNET = "darknet"
    NEWS = "news"
    REALTIME = "realtime"
    DOMAIN_SPECIFIC = "domain_specific"
    RAG = "rag"
    TOOL_DISCOVERY = "tool_discovery"


@dataclass
class SourceMetadata:
    """Metadata about a knowledge source."""
    name: str
    capabilities: List[SourceCapability]
    cost_per_query: float = 0.0
    priority: int = 1
    trust_level: float = 0.8
    avg_latency_ms: int = 1000
    requires_api_key: bool = False
    api_key_env: Optional[str] = None
    domains: List[str] = field(default_factory=list)
    enabled: bool = True
    registered_by: Optional[str] = None
    registered_at: Optional[datetime] = None


@dataclass
class SourceQuery:
    """A query to a knowledge source."""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 5
    capabilities_needed: List[SourceCapability] = field(default_factory=list)
    requester_kernel: Optional[str] = None


@dataclass
class SourceResult:
    """Result from a knowledge source."""
    source_name: str
    content: str
    title: str = ""
    url: str = ""
    relevance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeSource(ABC):
    """
    Abstract base class for all knowledge sources.
    
    Implement this to add ANY new knowledge source to the system.
    Kernels can create and register their own sources.
    """
    
    @property
    @abstractmethod
    def metadata(self) -> SourceMetadata:
        """Return source metadata."""
        pass
    
    @abstractmethod
    def query(self, query: SourceQuery) -> List[SourceResult]:
        """Execute a query against this source."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if source is available."""
        pass
    
    def can_handle(self, capabilities: List[SourceCapability]) -> bool:
        """Check if this source can handle the requested capabilities."""
        if not capabilities:
            return True
        return any(cap in self.metadata.capabilities for cap in capabilities)


class KnowledgeSourceRegistry:
    """
    Central registry for all knowledge sources.
    
    Supports:
    - Static sources (builtin adapters)
    - Dynamic sources (kernel-registered at runtime)
    - Hot-loading new sources without restart
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sources: Dict[str, KnowledgeSource] = {}
        self._source_stats: Dict[str, Dict] = {}
        self._listeners: List[Callable] = []
        self._initialized = True
        
        logger.info("[KnowledgeSourceRegistry] Initialized")
    
    def register(
        self, 
        source: KnowledgeSource,
        registered_by: Optional[str] = None
    ) -> bool:
        """
        Register a knowledge source.
        
        Kernels can call this to add new sources dynamically.
        """
        name = source.metadata.name
        
        if name in self._sources:
            logger.warning(f"[KnowledgeSourceRegistry] Source '{name}' already exists, updating")
        
        if not source.health_check():
            logger.warning(f"[KnowledgeSourceRegistry] Source '{name}' failed health check but registering anyway")
        
        source.metadata.registered_by = registered_by
        source.metadata.registered_at = datetime.now()
        
        self._sources[name] = source
        self._source_stats[name] = {
            'query_count': 0,
            'success_count': 0,
            'total_latency_ms': 0,
            'last_used': None
        }
        
        logger.info(f"[KnowledgeSourceRegistry] Registered source '{name}' "
                   f"(capabilities: {[c.value for c in source.metadata.capabilities]}, "
                   f"registered_by: {registered_by})")
        
        for listener in self._listeners:
            try:
                listener('register', source)
            except Exception as e:
                logger.error(f"[KnowledgeSourceRegistry] Listener error: {e}")
        
        return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a source."""
        if name not in self._sources:
            return False
        
        source = self._sources.pop(name)
        self._source_stats.pop(name, None)
        
        logger.info(f"[KnowledgeSourceRegistry] Unregistered source '{name}'")
        
        for listener in self._listeners:
            try:
                listener('unregister', source)
            except Exception:
                pass
        
        return True
    
    def get(self, name: str) -> Optional[KnowledgeSource]:
        """Get a source by name."""
        return self._sources.get(name)
    
    def list_all(self) -> List[SourceMetadata]:
        """List all registered sources."""
        return [s.metadata for s in self._sources.values()]
    
    def find_by_capability(
        self, 
        capabilities: List[SourceCapability],
        enabled_only: bool = True
    ) -> List[KnowledgeSource]:
        """Find sources that can handle given capabilities."""
        matches = []
        for source in self._sources.values():
            if enabled_only and not source.metadata.enabled:
                continue
            if source.can_handle(capabilities):
                matches.append(source)
        
        matches.sort(key=lambda s: (s.metadata.priority, -s.metadata.cost_per_query), reverse=True)
        return matches
    
    def record_query_stats(
        self, 
        source_name: str, 
        success: bool, 
        latency_ms: int
    ):
        """Record query statistics for a source."""
        if source_name in self._source_stats:
            stats = self._source_stats[source_name]
            stats['query_count'] += 1
            if success:
                stats['success_count'] += 1
            stats['total_latency_ms'] += latency_ms
            stats['last_used'] = datetime.now()
    
    def get_stats(self, source_name: str) -> Optional[Dict]:
        """Get statistics for a source."""
        return self._source_stats.get(source_name)
    
    def add_listener(self, callback: Callable):
        """Add listener for source registration events."""
        self._listeners.append(callback)


class KnowledgeOrchestrator:
    """
    Orchestrates queries across multiple knowledge sources.
    
    Selects best sources based on:
    - Required capabilities
    - Cost/budget
    - Trust level
    - Historical performance
    - Kernel preferences
    """
    
    def __init__(self, registry: Optional[KnowledgeSourceRegistry] = None):
        self.registry = registry or get_source_registry()
        self._budget_remaining = float('inf')
        self._max_concurrent = 3
    
    def query(
        self,
        query: SourceQuery,
        use_all_matching: bool = False,
        max_sources: int = 3
    ) -> List[SourceResult]:
        """
        Query knowledge sources.
        
        Args:
            query: The query to execute
            use_all_matching: Query all matching sources (not just best)
            max_sources: Maximum number of sources to query
        
        Returns:
            Combined results from all queried sources
        """
        sources = self.registry.find_by_capability(query.capabilities_needed)
        
        if not sources:
            logger.warning(f"[KnowledgeOrchestrator] No sources found for capabilities: {query.capabilities_needed}")
            return []
        
        sources_to_use = sources[:max_sources] if not use_all_matching else sources
        
        all_results = []
        for source in sources_to_use:
            if source.metadata.cost_per_query > self._budget_remaining:
                continue
            
            start_time = time.time()
            try:
                results = source.query(query)
                latency_ms = int((time.time() - start_time) * 1000)
                
                self.registry.record_query_stats(source.metadata.name, True, latency_ms)
                self._budget_remaining -= source.metadata.cost_per_query
                
                all_results.extend(results)
                
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                self.registry.record_query_stats(source.metadata.name, False, latency_ms)
                logger.error(f"[KnowledgeOrchestrator] Source '{source.metadata.name}' failed: {e}")
        
        all_results.sort(key=lambda r: r.relevance, reverse=True)
        return all_results
    
    def query_all_sources(
        self,
        query_text: str,
        context: Optional[Dict] = None,
        requester: Optional[str] = None
    ) -> List[SourceResult]:
        """
        Query ALL available sources for comprehensive research.
        
        This is the main entry point for autonomous learning.
        """
        query = SourceQuery(
            query=query_text,
            context=context or {},
            requester_kernel=requester
        )
        
        return self.query(query, use_all_matching=True, max_sources=10)
    
    def set_budget(self, budget: float):
        """Set remaining budget for queries."""
        self._budget_remaining = budget


def _create_duckduckgo_adapter():
    """Create adapter for DuckDuckGo search."""
    try:
        from search.duckduckgo_adapter import DuckDuckGoAdapter as DDGBase
        
        class DuckDuckGoSource(KnowledgeSource):
            def __init__(self):
                self._adapter = DDGBase()
                self._meta = SourceMetadata(
                    name="duckduckgo",
                    capabilities=[SourceCapability.WEB_SEARCH, SourceCapability.NEWS],
                    cost_per_query=0.0,
                    priority=5,
                    trust_level=0.7
                )
            
            @property
            def metadata(self) -> SourceMetadata:
                return self._meta
            
            def query(self, query: SourceQuery) -> List[SourceResult]:
                results = self._adapter.search(query.query, max_results=query.max_results)
                return [
                    SourceResult(
                        source_name="duckduckgo",
                        content=r.get('body', ''),
                        title=r.get('title', ''),
                        url=r.get('href', ''),
                        relevance=0.7
                    )
                    for r in results.get('results', [])
                ]
            
            def health_check(self) -> bool:
                return True
        
        return DuckDuckGoSource()
    except ImportError:
        return None


def _create_wikipedia_adapter():
    """Create adapter for Wikipedia."""
    try:
        from external_knowledge import get_external_kb
        
        class WikipediaSource(KnowledgeSource):
            def __init__(self):
                self._kb = None
                self._meta = SourceMetadata(
                    name="wikipedia",
                    capabilities=[SourceCapability.ENCYCLOPEDIC],
                    cost_per_query=0.0,
                    priority=4,
                    trust_level=0.85
                )
            
            @property
            def metadata(self) -> SourceMetadata:
                return self._meta
            
            def _get_kb(self):
                if self._kb is None:
                    self._kb = get_external_kb()
                return self._kb
            
            def query(self, query: SourceQuery) -> List[SourceResult]:
                kb = self._get_kb()
                results = kb.query_wikipedia(query.query, max_results=query.max_results)
                return [
                    SourceResult(
                        source_name="wikipedia",
                        content=r.get('content', '') or r.get('extract', ''),
                        title=r.get('title', ''),
                        url=r.get('url', ''),
                        relevance=0.8
                    )
                    for r in results
                ]
            
            def health_check(self) -> bool:
                return True
        
        return WikipediaSource()
    except ImportError:
        return None


def _create_github_adapter():
    """Create adapter for GitHub."""
    try:
        from research.web_scraper import get_scraper
        
        class GitHubSource(KnowledgeSource):
            def __init__(self):
                self._meta = SourceMetadata(
                    name="github",
                    capabilities=[SourceCapability.CODE_REPOS, SourceCapability.TOOL_DISCOVERY],
                    cost_per_query=0.0,
                    priority=4,
                    trust_level=0.9
                )
            
            @property
            def metadata(self) -> SourceMetadata:
                return self._meta
            
            def query(self, query: SourceQuery) -> List[SourceResult]:
                scraper = get_scraper()
                data = scraper._scrape_github(query.query)
                if not data or 'repositories' not in data:
                    return []
                
                return [
                    SourceResult(
                        source_name="github",
                        content=repo.get('description', ''),
                        title=repo.get('name', ''),
                        url=repo.get('url', ''),
                        relevance=0.75,
                        metadata={'stars': repo.get('stars', 0)}
                    )
                    for repo in data['repositories'][:query.max_results]
                ]
            
            def health_check(self) -> bool:
                return True
        
        return GitHubSource()
    except ImportError:
        return None


def _create_arxiv_adapter():
    """Create adapter for arXiv."""
    try:
        from research.web_scraper import get_scraper
        
        class ArxivSource(KnowledgeSource):
            def __init__(self):
                self._meta = SourceMetadata(
                    name="arxiv",
                    capabilities=[SourceCapability.ACADEMIC],
                    cost_per_query=0.0,
                    priority=5,
                    trust_level=0.95
                )
            
            @property
            def metadata(self) -> SourceMetadata:
                return self._meta
            
            def query(self, query: SourceQuery) -> List[SourceResult]:
                scraper = get_scraper()
                data = scraper._scrape_arxiv(query.query)
                if not data or 'papers' not in data:
                    return []
                
                return [
                    SourceResult(
                        source_name="arxiv",
                        content=paper.get('abstract', ''),
                        title=paper.get('title', ''),
                        url=paper.get('url', ''),
                        relevance=0.9,
                        metadata={'authors': paper.get('authors', [])}
                    )
                    for paper in data['papers'][:query.max_results]
                ]
            
            def health_check(self) -> bool:
                return True
        
        return ArxivSource()
    except ImportError:
        return None


def _create_shadow_adapter():
    """Create adapter for Shadow Pantheon darknet capabilities."""
    try:
        from olympus.shadow_scrapy import get_shadow_scrapy
        
        class ShadowSource(KnowledgeSource):
            def __init__(self):
                self._meta = SourceMetadata(
                    name="shadow",
                    capabilities=[SourceCapability.DARKNET, SourceCapability.WEB_SEARCH],
                    cost_per_query=0.0,
                    priority=3,
                    trust_level=0.6
                )
            
            @property
            def metadata(self) -> SourceMetadata:
                return self._meta
            
            def query(self, query: SourceQuery) -> List[SourceResult]:
                shadow = get_shadow_scrapy()
                results = shadow.stealth_search(query.query, max_results=query.max_results)
                return [
                    SourceResult(
                        source_name="shadow",
                        content=r.get('content', ''),
                        title=r.get('title', ''),
                        url=r.get('url', ''),
                        relevance=0.6
                    )
                    for r in results
                ]
            
            def health_check(self) -> bool:
                try:
                    shadow = get_shadow_scrapy()
                    return shadow.is_tor_available()
                except:
                    return False
        
        return ShadowSource()
    except ImportError:
        return None


_registry_instance: Optional[KnowledgeSourceRegistry] = None


def get_source_registry() -> KnowledgeSourceRegistry:
    """Get or create the global source registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = KnowledgeSourceRegistry()
        _initialize_builtin_sources(_registry_instance)
    return _registry_instance


def _initialize_builtin_sources(registry: KnowledgeSourceRegistry):
    """Initialize all builtin knowledge sources."""
    adapters = [
        ('duckduckgo', _create_duckduckgo_adapter),
        ('wikipedia', _create_wikipedia_adapter),
        ('github', _create_github_adapter),
        ('arxiv', _create_arxiv_adapter),
        ('shadow', _create_shadow_adapter),
    ]
    
    for name, creator in adapters:
        try:
            source = creator()
            if source:
                registry.register(source, registered_by='system')
        except Exception as e:
            logger.warning(f"[KnowledgeSourceRegistry] Failed to create {name} adapter: {e}")


def get_orchestrator() -> KnowledgeOrchestrator:
    """Get the knowledge orchestrator."""
    return KnowledgeOrchestrator(get_source_registry())


class DynamicSource(KnowledgeSource):
    """
    A dynamically-created source that kernels can instantiate.
    
    Allows kernels to create new sources by providing:
    - A name
    - Capabilities
    - A query function
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[SourceCapability],
        query_fn: Callable[[SourceQuery], List[SourceResult]],
        health_fn: Optional[Callable[[], bool]] = None,
        **kwargs
    ):
        self._meta = SourceMetadata(
            name=name,
            capabilities=capabilities,
            **kwargs
        )
        self._query_fn = query_fn
        self._health_fn = health_fn or (lambda: True)
    
    @property
    def metadata(self) -> SourceMetadata:
        return self._meta
    
    def query(self, query: SourceQuery) -> List[SourceResult]:
        return self._query_fn(query)
    
    def health_check(self) -> bool:
        return self._health_fn()


def register_kernel_source(
    kernel_name: str,
    source_name: str,
    capabilities: List[str],
    query_fn: Callable,
    **kwargs
) -> bool:
    """
    API for kernels to register their own knowledge sources.
    
    Example:
        def my_query_fn(query):
            return [SourceResult(...)]
        
        register_kernel_source(
            kernel_name="athena",
            source_name="strategy_papers",
            capabilities=["academic"],
            query_fn=my_query_fn
        )
    """
    caps = [SourceCapability(c) for c in capabilities if c in [e.value for e in SourceCapability]]
    
    source = DynamicSource(
        name=source_name,
        capabilities=caps,
        query_fn=query_fn,
        **kwargs
    )
    
    registry = get_source_registry()
    return registry.register(source, registered_by=kernel_name)
