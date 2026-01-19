"""
Tavily Search Client - Full SDK Capabilities

Provides complete access to Tavily's search, extract, crawl, and map APIs
for high-quality QIG RAG learning.

Capabilities:
- search: Web search with advanced filtering and AI answers
- extract: Content extraction from URLs
- crawl: Intelligent website crawling with instructions
- map: Website structure mapping

All results are returned in QIG-compatible formats for geometric learning.

BUDGET ENFORCEMENT: All paid API calls check the budget orchestrator
before execution. If the daily cost cap is exceeded, calls will be blocked.

CURRICULUM-ONLY MODE: All external searches are blocked when QIG_CURRICULUM_ONLY=true
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Import curriculum guard - centralized check
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Budget orchestrator for cost control
_budget_orchestrator = None

def _get_budget_orchestrator():
    """Lazy import budget orchestrator to avoid circular imports."""
    global _budget_orchestrator
    if _budget_orchestrator is None:
        try:
            from search.search_budget_orchestrator import get_budget_orchestrator
            _budget_orchestrator = get_budget_orchestrator()
        except ImportError:
            logger.warning("[TavilyClient] Budget orchestrator not available - no cost limits")
    return _budget_orchestrator

def _check_budget(provider: str = 'tavily') -> bool:
    """Check if budget allows this API call. Returns False if blocked."""
    orchestrator = _get_budget_orchestrator()
    if not orchestrator:
        return True  # Allow if orchestrator not available

    if not orchestrator.consume_quota(provider):
        logger.warning(f"[TavilyClient] BLOCKED by budget orchestrator (provider={provider})")
        return False
    return True


@dataclass
class TavilySearchResult:
    """Structured search result from Tavily."""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None
    raw_content: Optional[str] = None


@dataclass
class TavilySearchResponse:
    """Complete search response from Tavily."""
    query: str
    answer: Optional[str]
    results: List[TavilySearchResult]
    response_time: float
    follow_up_questions: Optional[List[str]] = None
    images: Optional[List[Dict[str, Any]]] = None


@dataclass
class TavilyExtractResult:
    """Extracted content from a URL."""
    url: str
    raw_content: str
    success: bool
    error: Optional[str] = None


@dataclass
class TavilyCrawlResult:
    """Crawled content from a website."""
    base_url: str
    pages: List[Dict[str, Any]]
    total_pages: int
    success: bool
    error: Optional[str] = None


@dataclass
class TavilyMapResult:
    """Website structure map."""
    base_url: str
    urls: List[str]
    total_urls: int
    success: bool
    error: Optional[str] = None


class TavilySearchClient:
    """
    Full-featured Tavily client for QIG learning.
    
    Provides all Tavily API capabilities:
    - search(): Advanced web search with AI answers
    - extract(): Content extraction from URLs
    - crawl(): Intelligent website crawling
    - map(): Website structure discovery
    
    Usage:
        client = TavilySearchClient()
        
        # Search
        results = client.search("quantum information geometry", include_answer=True)
        
        # Extract content
        content = client.extract(["https://arxiv.org/abs/..."])
        
        # Crawl with instructions
        pages = client.crawl("https://docs.example.com", 
                            instructions="Find all API documentation")
        
        # Map website structure
        sitemap = client.map("https://example.com", max_depth=2)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily client.
        
        Args:
            api_key: Tavily API key. If not provided, uses TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        self.client: Any = None
        self.available = False
        
        if not self.api_key:
            logger.warning("[TavilyClient] TAVILY_API_KEY not set - Tavily search disabled")
            return
        
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.api_key)
            self.available = True
            logger.info("[TavilyClient] Initialized with full SDK capabilities")
        except ImportError:
            logger.error("[TavilyClient] tavily-python not installed. Run: pip install tavily-python")
        except Exception as e:
            logger.error(f"[TavilyClient] Failed to initialize: {e}")
    
    def search(
        self,
        query: str,
        search_depth: str = "advanced",
        topic: str = "general",
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        days: Optional[int] = None
    ) -> Optional[TavilySearchResponse]:
        """
        Execute advanced web search.
        
        Args:
            query: Search query
            search_depth: "basic" or "advanced" (more thorough)
            topic: "general" or "news"
            max_results: Maximum results (1-20)
            include_answer: Include AI-generated answer
            include_raw_content: Include full page content
            include_images: Include relevant images
            include_domains: Whitelist domains (e.g., ["arxiv.org", "github.com"])
            exclude_domains: Blacklist domains
            days: Limit to recent days (for news topic)
            
        Returns:
            TavilySearchResponse or None if unavailable or budget exceeded
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            logger.warning("[TavilyClient] Search blocked by curriculum-only mode")
            return None
        
        if not self.available:
            logger.warning("[TavilyClient] Search unavailable - client not initialized")
            return None

        # BUDGET CHECK: Consume quota before making API call
        if not _check_budget('tavily'):
            logger.warning("[TavilyClient] Search blocked by budget cap")
            return None

        try:
            logger.info(f"[TavilyClient] Searching: '{query[:50]}...' (depth={search_depth})")
            
            kwargs = {
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
            }
            
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            if days and topic == "news":
                kwargs["days"] = days
            
            response = self.client.search(**kwargs)
            
            results = [
                TavilySearchResult(
                    title=r.get('title', ''),
                    url=r.get('url', ''),
                    content=r.get('content', ''),
                    score=r.get('score', 0.0),
                    published_date=r.get('published_date'),
                    raw_content=r.get('raw_content')
                )
                for r in response.get('results', [])
            ]
            
            search_response = TavilySearchResponse(
                query=query,
                answer=response.get('answer'),
                results=results,
                response_time=response.get('response_time', 0.0),
                follow_up_questions=response.get('follow_up_questions'),
                images=response.get('images')
            )
            
            logger.info(f"[TavilyClient] Search returned {len(results)} results")
            return search_response
            
        except Exception as e:
            logger.error(f"[TavilyClient] Search error: {e}")
            return None
    
    def extract(
        self,
        urls: Union[str, List[str]],
        extract_depth: str = "basic"
    ) -> List[TavilyExtractResult]:
        """
        Extract content from URLs.
        
        Args:
            urls: Single URL or list of URLs to extract
            extract_depth: "basic" or "advanced" (more thorough extraction)
            
        Returns:
            List of TavilyExtractResult (empty if budget exceeded)
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            logger.warning("[TavilyClient] Extract blocked by curriculum-only mode")
            return []
        
        if not self.available:
            logger.warning("[TavilyClient] Extract unavailable - client not initialized")
            return []

        # BUDGET CHECK: Consume quota before making API call
        if not _check_budget('tavily'):
            logger.warning("[TavilyClient] Extract blocked by budget cap")
            return []

        if isinstance(urls, str):
            urls = [urls]

        try:
            logger.info(f"[TavilyClient] Extracting from {len(urls)} URLs")
            
            response = self.client.extract(urls=urls)
            
            results = []
            for r in response.get('results', []):
                results.append(TavilyExtractResult(
                    url=r.get('url', ''),
                    raw_content=r.get('raw_content', ''),
                    success=True
                ))
            
            for r in response.get('failed_results', []):
                results.append(TavilyExtractResult(
                    url=r.get('url', ''),
                    raw_content='',
                    success=False,
                    error=r.get('error', 'Unknown error')
                ))
            
            logger.info(f"[TavilyClient] Extracted {len([r for r in results if r.success])} URLs successfully")
            return results
            
        except Exception as e:
            logger.error(f"[TavilyClient] Extract error: {e}")
            return []
    
    def crawl(
        self,
        url: str,
        instructions: Optional[str] = None,
        max_pages: int = 10,
        max_depth: int = 2,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None
    ) -> Optional[TavilyCrawlResult]:
        """
        Crawl website with optional instructions.
        
        Args:
            url: Starting URL
            instructions: Natural language crawling instructions
                          (e.g., "Find all API documentation and examples")
            max_pages: Maximum pages to crawl (1-50)
            max_depth: Maximum crawl depth
            include_paths: Path patterns to include
            exclude_paths: Path patterns to exclude
            
        Returns:
            TavilyCrawlResult or None if unavailable or budget exceeded
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            logger.warning("[TavilyClient] Crawl blocked by curriculum-only mode")
            return None
        
        if not self.available:
            logger.warning("[TavilyClient] Crawl unavailable - client not initialized")
            return None

        # BUDGET CHECK: Consume quota before making API call
        if not _check_budget('tavily'):
            logger.warning("[TavilyClient] Crawl blocked by budget cap")
            return None

        try:
            logger.info(f"[TavilyClient] Crawling: {url} (max_pages={max_pages})")
            
            kwargs = {
                "url": url,
                "max_pages": max_pages,
                "max_depth": max_depth,
            }
            
            if instructions:
                kwargs["instructions"] = instructions
            if include_paths:
                kwargs["include_paths"] = include_paths
            if exclude_paths:
                kwargs["exclude_paths"] = exclude_paths
            
            response = self.client.crawl(**kwargs)
            
            pages = response.get('results', [])
            
            result = TavilyCrawlResult(
                base_url=url,
                pages=pages,
                total_pages=len(pages),
                success=True
            )
            
            logger.info(f"[TavilyClient] Crawled {len(pages)} pages")
            return result
            
        except Exception as e:
            logger.error(f"[TavilyClient] Crawl error: {e}")
            return TavilyCrawlResult(
                base_url=url,
                pages=[],
                total_pages=0,
                success=False,
                error=str(e)
            )
    
    def map(
        self,
        url: str,
        max_depth: int = 2,
        limit: int = 100
    ) -> Optional[TavilyMapResult]:
        """
        Map website structure to discover URLs.
        
        Args:
            url: Starting URL
            max_depth: Maximum crawl depth
            limit: Maximum URLs to return
            
        Returns:
            TavilyMapResult or None if unavailable or budget exceeded
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            logger.warning("[TavilyClient] Map blocked by curriculum-only mode")
            return None
        
        if not self.available:
            logger.warning("[TavilyClient] Map unavailable - client not initialized")
            return None

        # BUDGET CHECK: Consume quota before making API call
        if not _check_budget('tavily'):
            logger.warning("[TavilyClient] Map blocked by budget cap")
            return None

        try:
            logger.info(f"[TavilyClient] Mapping: {url} (depth={max_depth}, limit={limit})")
            
            response = self.client.map(
                url=url,
                max_depth=max_depth,
                limit=limit
            )
            
            urls = response.get('urls', [])
            
            result = TavilyMapResult(
                base_url=url,
                urls=urls,
                total_urls=len(urls),
                success=True
            )
            
            logger.info(f"[TavilyClient] Mapped {len(urls)} URLs")
            return result
            
        except Exception as e:
            logger.error(f"[TavilyClient] Map error: {e}")
            return TavilyMapResult(
                base_url=url,
                urls=[],
                total_urls=0,
                success=False,
                error=str(e)
            )
    
    def research(
        self,
        query: str,
        topic: str = "general",
        include_domains: Optional[List[str]] = None,
        max_iterations: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Conduct deep research on a topic.
        
        This combines search + extract + crawl for comprehensive research:
        1. Search for relevant sources
        2. Extract content from top results
        3. Optionally crawl linked pages for more context
        
        Args:
            query: Research query
            topic: "general" or "news"
            include_domains: Priority domains for research
            max_iterations: Maximum search/extract iterations
            
        Returns:
            Dict with research findings or None
        """
        if not self.available:
            logger.warning("[TavilyClient] Research unavailable - client not initialized")
            return None
        
        try:
            logger.info(f"[TavilyClient] Starting research: '{query[:50]}...'")
            
            all_content = []
            all_sources = []
            
            search_result = self.search(
                query=query,
                search_depth="advanced",
                topic=topic,
                max_results=10,
                include_answer=True,
                include_raw_content=True,
                include_domains=include_domains
            )
            
            if not search_result:
                return None
            
            if search_result.answer:
                all_content.append({
                    "type": "ai_answer",
                    "content": search_result.answer,
                    "source": "tavily_synthesis"
                })
            
            for result in search_result.results:
                all_sources.append({
                    "url": result.url,
                    "title": result.title,
                    "score": result.score
                })
                if result.raw_content:
                    all_content.append({
                        "type": "page_content",
                        "url": result.url,
                        "title": result.title,
                        "content": result.raw_content[:5000]
                    })
            
            top_urls = [r.url for r in search_result.results[:5] if not r.raw_content]
            if top_urls:
                extracts = self.extract(top_urls)
                for ext in extracts:
                    if ext.success:
                        all_content.append({
                            "type": "extracted_content",
                            "url": ext.url,
                            "content": ext.raw_content[:5000]
                        })
            
            research_result = {
                "query": query,
                "ai_synthesis": search_result.answer,
                "sources": all_sources,
                "content": all_content,
                "follow_up_questions": search_result.follow_up_questions,
                "total_sources": len(all_sources),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"[TavilyClient] Research complete: {len(all_sources)} sources, {len(all_content)} content pieces")
            return research_result
            
        except Exception as e:
            logger.error(f"[TavilyClient] Research error: {e}")
            return None


_tavily_client: Optional[TavilySearchClient] = None


def get_tavily_client() -> TavilySearchClient:
    """Get or create singleton Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilySearchClient()
    return _tavily_client
