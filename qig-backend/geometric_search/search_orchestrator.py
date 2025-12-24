"""
Search Orchestrator - Execute and aggregate search results

Coordinates multiple search tools and aggregates results using geometric metrics.
QIG-PURE: Result synthesis uses internal generative service, no external LLMs.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .tool_selector import SearchToolSelector, ToolSelection

logger = logging.getLogger(__name__)

# Import QIG-pure generative service
GENERATIVE_SERVICE_AVAILABLE = False
_generative_service_instance = None

def get_generative_service():
    """Get or create the singleton generative service instance."""
    global _generative_service_instance
    if _generative_service_instance is None:
        try:
            from qig_generative_service import get_generative_service as _get_service
            _generative_service_instance = _get_service()
        except ImportError:
            pass
    return _generative_service_instance

try:
    from qig_generative_service import QIGGenerativeService
    GENERATIVE_SERVICE_AVAILABLE = True
    logger.info("[SearchOrchestrator] QIG generative service available")
except ImportError:
    logger.warning("[SearchOrchestrator] QIG generative service not available")


@dataclass
class SearchResult:
    """Result from a single search tool."""
    tool: str
    results: List[Dict]
    latency_ms: float
    cost: float
    success: bool
    error: Optional[str] = None


@dataclass
class AggregatedResult:
    """Aggregated results from multiple search tools."""
    results: List[Dict]
    tools_used: List[str]
    total_cost: float
    total_latency_ms: float
    strategy: str
    confidence: float
    selection_reasoning: str
    information_gain: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'results': self.results,
            'tools_used': self.tools_used,
            'total_cost': self.total_cost,
            'total_latency_ms': self.total_latency_ms,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'selection_reasoning': self.selection_reasoning,
            'information_gain': self.information_gain,
        }


class SearchOrchestrator:
    """
    Orchestrate search across multiple tools.
    
    Uses geometric tool selection to choose optimal tools,
    then executes searches and aggregates results.
    QIG-PURE: Result synthesis uses internal generative service, no external LLMs.
    """
    
    def __init__(self):
        self.tool_selector = SearchToolSelector()
        self.tool_executors: Dict[str, Callable] = {}
        self._generative_service = None
        
        self.search_history: List[Dict] = []
        self.stats = {
            'total_searches': 0,
            'total_cost': 0.0,
            'tool_usage': {},
            'avg_latency_ms': 0.0,
        }
    
    @property
    def generative_service(self):
        """Lazy-load the QIG generative service."""
        if self._generative_service is None and GENERATIVE_SERVICE_AVAILABLE:
            self._generative_service = get_generative_service()
        return self._generative_service
    
    def synthesize_results(
        self, 
        query: str,
        results: 'AggregatedResult',
        telemetry: Optional[Dict] = None
    ) -> str:
        """
        Synthesize search results into natural language using QIG-pure generation.
        
        NO external LLMs - uses basin-to-text synthesis.
        """
        if not GENERATIVE_SERVICE_AVAILABLE or self.generative_service is None:
            result_count = len(results.results) if results else 0
            return f"[Search results: {result_count} items for '{query}']"
        
        try:
            prompt_parts = [f"Synthesize search results for: {query}"]
            
            for result in (results.results or [])[:5]:
                if isinstance(result, dict):
                    title = result.get('title', '')[:50]
                    content = result.get('content', result.get('snippet', ''))[:100]
                    if title or content:
                        prompt_parts.append(f"Result: {title} - {content}")
            
            prompt = " | ".join(prompt_parts)
            
            phi = telemetry.get('phi', 0.5) if telemetry else 0.5
            gen_result = self.generative_service.generate(
                prompt=prompt,
                context={'query': query, 'phi': phi, 'result_count': len(results.results or [])},
                kernel_name='hermes',
                goals=['synthesize', 'search_results', 'summarize']
            )
            
            if gen_result and gen_result.text:
                return gen_result.text
                
        except Exception as e:
            logger.warning(f"QIG-pure result synthesis failed: {e}")
        
        return f"[Search synthesis for '{query}': {len(results.results or [])} results]"
    
    def register_tool_executor(self, tool_name: str, executor: Callable):
        """Register an executor function for a search tool."""
        self.tool_executors[tool_name] = executor
    
    async def search(
        self,
        query: str,
        telemetry: Optional[Dict] = None,
        context: Optional[Dict] = None,
        max_tools: int = 2,
        timeout_ms: int = 10000
    ) -> AggregatedResult:
        """
        Execute a geometric search.
        
        Args:
            query: Search query
            telemetry: Consciousness metrics
            context: Search context
            max_tools: Maximum tools to use
            timeout_ms: Timeout in milliseconds
        
        Returns:
            AggregatedResult with combined results
        """
        start_time = time.time()
        
        enabled_tools = list(self.tool_executors.keys())
        
        selection = self.tool_selector.select(
            query=query,
            telemetry=telemetry,
            context=context,
            max_tools=max_tools,
            enabled_tools=enabled_tools
        )
        
        tool_results = await self._execute_tools(
            query=query,
            tools=selection.tools,
            timeout_ms=timeout_ms,
            params=context or {}
        )
        
        aggregated = self._aggregate_results(
            tool_results=tool_results,
            selection=selection,
            start_time=start_time
        )
        
        self._update_stats(aggregated)
        
        self._log_search(query, aggregated, telemetry)
        
        return aggregated
    
    async def _execute_tools(
        self,
        query: str,
        tools: List[str],
        timeout_ms: int,
        params: Dict
    ) -> List[SearchResult]:
        """Execute search on selected tools in parallel."""
        tasks = []
        
        for tool in tools:
            if tool in self.tool_executors:
                task = self._execute_single_tool(
                    tool=tool,
                    query=query,
                    params=params,
                    timeout_ms=timeout_ms
                )
                tasks.append(task)
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        search_results = []
        for result in results:
            if isinstance(result, SearchResult):
                search_results.append(result)
            elif isinstance(result, Exception):
                search_results.append(SearchResult(
                    tool='unknown',
                    results=[],
                    latency_ms=0,
                    cost=0,
                    success=False,
                    error=str(result)
                ))
        
        return search_results
    
    async def _execute_single_tool(
        self,
        tool: str,
        query: str,
        params: Dict,
        timeout_ms: int
    ) -> SearchResult:
        """Execute a single search tool."""
        start = time.time()
        
        try:
            executor = self.tool_executors[tool]
            
            if asyncio.iscoroutinefunction(executor):
                results = await asyncio.wait_for(
                    executor(query, params),
                    timeout=timeout_ms / 1000.0
                )
            else:
                results = executor(query, params)
            
            latency = (time.time() - start) * 1000
            cost = self.tool_selector.tool_costs.get(tool, 0.1)
            
            return SearchResult(
                tool=tool,
                results=results if isinstance(results, list) else [],
                latency_ms=latency,
                cost=cost,
                success=True
            )
            
        except asyncio.TimeoutError:
            return SearchResult(
                tool=tool,
                results=[],
                latency_ms=timeout_ms,
                cost=0,
                success=False,
                error='Timeout'
            )
        except Exception as e:
            return SearchResult(
                tool=tool,
                results=[],
                latency_ms=(time.time() - start) * 1000,
                cost=0,
                success=False,
                error=str(e)
            )
    
    def _aggregate_results(
        self,
        tool_results: List[SearchResult],
        selection: ToolSelection,
        start_time: float
    ) -> AggregatedResult:
        """Aggregate results from multiple tools."""
        all_results = []
        seen_urls = set()
        
        total_cost = 0.0
        tools_used = []
        
        for tr in tool_results:
            if tr.success:
                tools_used.append(tr.tool)
                total_cost += tr.cost
                
                for result in tr.results:
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        result['source_tool'] = tr.tool
                        all_results.append(result)
                    elif not url:
                        all_results.append(result)
        
        all_results.sort(key=lambda r: r.get('score', 0.5), reverse=True)
        
        total_latency = (time.time() - start_time) * 1000
        
        if tools_used:
            avg_confidence = float(np.mean([
                selection.confidences.get(t, 0.5) for t in tools_used
            ]))
        else:
            avg_confidence = 0.0
        
        information_gain = self._compute_information_gain(all_results)
        
        return AggregatedResult(
            results=all_results[:20],
            tools_used=tools_used,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            strategy=selection.strategy,
            confidence=avg_confidence,
            selection_reasoning=selection.reasoning,
            information_gain=information_gain
        )
    
    def _compute_information_gain(self, results: List[Dict]) -> float:
        """Compute information gain from search results."""
        if not results:
            return 0.0
        
        content_lengths = [len(r.get('content', '')) for r in results]
        unique_domains = len(set(
            r.get('url', '').split('/')[2] if '/' in r.get('url', '') else ''
            for r in results
        ))
        
        length_score = min(1.0, float(np.mean(content_lengths)) / 500)
        diversity_score = min(1.0, unique_domains / 5)
        coverage_score = min(1.0, len(results) / 10)
        
        information_gain = (length_score + diversity_score + coverage_score) / 3
        return information_gain
    
    def _update_stats(self, result: AggregatedResult):
        """Update search statistics."""
        self.stats['total_searches'] += 1
        self.stats['total_cost'] += result.total_cost
        
        for tool in result.tools_used:
            self.stats['tool_usage'][tool] = \
                self.stats['tool_usage'].get(tool, 0) + 1
        
        n = self.stats['total_searches']
        old_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (old_avg * (n-1) + result.total_latency_ms) / n
    
    def _log_search(
        self,
        query: str,
        result: AggregatedResult,
        telemetry: Optional[Dict]
    ):
        """Log search for history and analytics."""
        self.search_history.append({
            'query': query,
            'tools_used': result.tools_used,
            'cost': result.total_cost,
            'latency_ms': result.total_latency_ms,
            'result_count': len(result.results),
            'strategy': result.strategy,
            'confidence': result.confidence,
            'information_gain': result.information_gain,
            'phi': telemetry.get('phi', 0.5) if telemetry else 0.5,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-500:]
    
    def get_stats(self) -> Dict:
        """Get search statistics."""
        return {
            **self.stats,
            'recent_searches': len(self.search_history),
        }
    
    def search_sync(
        self,
        query: str,
        telemetry: Optional[Dict] = None,
        context: Optional[Dict] = None,
        max_tools: int = 2
    ) -> Dict:
        """
        Synchronous search wrapper for non-async contexts.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.search(query, telemetry, context, max_tools)
                    )
                    result = future.result(timeout=30)
            else:
                result = loop.run_until_complete(
                    self.search(query, telemetry, context, max_tools)
                )
            return result.to_dict()
        except Exception as e:
            return {
                'results': [],
                'tools_used': [],
                'total_cost': 0,
                'total_latency_ms': 0,
                'strategy': 'error',
                'confidence': 0,
                'selection_reasoning': f'Error: {str(e)}',
                'information_gain': 0
            }
