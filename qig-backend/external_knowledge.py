"""
External Knowledge Base Connector

Integrates external knowledge sources with QIG-RAG for enhanced retrieval.
All external data is converted to basin coordinates via geometric encoding.

ARCHITECTURE:
- Budget-aware SearchProviderManager (Tavily, Perplexity, Google, DuckDuckGo)
- Wikipedia API fallback for encyclopedic knowledge
- DuckDuckGo Instant Answers fallback for quick facts
- Results encoded to 64D basin coordinates
- Fisher-Rao distance used for relevance ranking
- External knowledge weighted lower than local geometric memory
- Results feed back to VocabularyCoordinator for learning
"""

import os
import time
import requests
from typing import List, Dict, Optional, Any
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

BASIN_DIMENSION = 64
EXTERNAL_WEIGHT = 0.7  # Weight external results lower than local
BUDGET_SEARCH_WEIGHT = 0.9  # Budget search results are higher quality
CACHE_TTL = 3600  # 1 hour cache

_external_kb: Optional['ExternalKnowledgeBase'] = None


class ExternalKnowledgeBase:
    """
    Connector for external knowledge sources.
    All results are encoded to basin coordinates for geometric integration.

    Priority order:
    1. Budget-aware search (Tavily, Perplexity, Google) - if budget available
    2. Wikipedia API - free, always available
    3. DuckDuckGo Instant Answers - free, always available

    All results feed into VocabularyCoordinator for learning.
    """

    def __init__(self, encoder=None):
        self.encoder = encoder
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Wikipedia and DuckDuckGo as fallback
        self.wikipedia_endpoint = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wikipedia_search_endpoint = "https://en.wikipedia.org/w/api.php"
        self.ddg_endpoint = "https://api.duckduckgo.com/"

        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'PantheonChat/1.0 (QIG Knowledge System)'
        })

        # Try to wire budget-aware search
        self._search_manager = None
        self._budget_orchestrator = None
        self._vocab_coordinator = None
        self._wire_budget_search()

        sources = ["Wikipedia", "DuckDuckGo"]
        if self._search_manager:
            sources.extend(["Tavily", "Perplexity", "Google"])
        print(f"[ExternalKnowledge] Initialized with: {', '.join(sources)}")

    def _wire_budget_search(self):
        """Wire budget-aware search providers for richer results."""
        try:
            from search.search_providers import get_search_manager
            from search.search_budget_orchestrator import get_budget_orchestrator
            self._search_manager = get_search_manager()
            self._budget_orchestrator = get_budget_orchestrator()
            print("[ExternalKnowledge] Budget-aware search wired (Tavily/Perplexity/Google)")
        except ImportError as e:
            print(f"[ExternalKnowledge] Budget search not available: {e}")
        except Exception as e:
            print(f"[ExternalKnowledge] Budget search wiring failed: {e}")

        # Wire vocabulary coordinator for learning from search results
        try:
            from vocabulary_coordinator import get_vocabulary_coordinator
            self._vocab_coordinator = get_vocabulary_coordinator()
            print("[ExternalKnowledge] VocabularyCoordinator wired for search learning")
        except ImportError:
            pass
        except Exception as e:
            print(f"[ExternalKnowledge] VocabularyCoordinator wiring failed: {e}")
    
    def set_encoder(self, encoder):
        """Set the conversation encoder for basin coordinate generation."""
        self.encoder = encoder

    def query_budget_search(self, query: str, importance: int = 2, max_results: int = 5) -> List[Dict]:
        """
        Query budget-aware search providers (Tavily, Perplexity, Google).

        Uses SearchBudgetOrchestrator to select best provider within budget.
        Results are encoded to basin coordinates and feed into vocabulary learning.

        Args:
            query: Search query
            importance: 1=LOW, 2=MODERATE, 3=HIGH, 4=CRITICAL
            max_results: Maximum results to return

        Returns:
            List of results with basin coordinates
        """
        if not self._search_manager or not self._budget_orchestrator:
            return []

        cache_key = f"budget:{query}:{importance}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached.get('results', [])

        results = []

        try:
            # Use search manager with importance-based provider selection
            search_result = self._search_manager.search(
                query=query,
                importance=importance,
                max_results=max_results
            )

            if search_result.get('success'):
                for item in search_result.get('results', []):
                    content = item.get('snippet', '') or item.get('content', '')
                    title = item.get('title', '')

                    # Encode to basin coordinates
                    basin_coords = None
                    if self.encoder and content:
                        try:
                            basin_coords = self.encoder.encode(content[:500])
                        except Exception:
                            pass

                    results.append({
                        'source': f"budget_{search_result.get('provider', 'unknown')}",
                        'title': title,
                        'content': content[:1000] if content else '',
                        'url': item.get('url', ''),
                        'basin_coords': basin_coords,
                        'weight': BUDGET_SEARCH_WEIGHT,
                        'provider': search_result.get('provider'),
                        'timestamp': time.time()
                    })

                # Feed results into vocabulary learning
                self._learn_from_results(results, query)

            self._set_cache(cache_key, {'results': results})

        except Exception as e:
            print(f"[ExternalKnowledge] Budget search error: {e}")

        return results

    def _learn_from_results(self, results: List[Dict], query: str):
        """Feed search results into VocabularyCoordinator for learning."""
        if not self._vocab_coordinator or not results:
            return

        try:
            # Combine result content for vocabulary training
            combined_text = ' '.join([
                r.get('content', '') for r in results if r.get('content')
            ])

            if combined_text and len(combined_text) > 50:
                metrics = self._vocab_coordinator.train_from_text(
                    text=combined_text[:5000],
                    source=f"search:{query[:500]}",
                    context_phi=0.6
                )
                if metrics.get('words_learned', 0) > 0:
                    print(f"[ExternalKnowledge] Learned {metrics['words_learned']} words from search")
        except Exception as e:
            print(f"[ExternalKnowledge] Vocabulary learning error: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached result is still valid."""
        if key not in self._cache:
            return False
        timestamp = self._cache_timestamps.get(key, 0)
        return (time.time() - timestamp) < CACHE_TTL
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if valid."""
        if self._is_cache_valid(key):
            return self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Dict):
        """Cache a result."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
    
    def query_wikipedia(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Query Wikipedia for relevant articles.
        Returns summaries encoded to basin coordinates.
        """
        print(f"[ExternalKnowledge] Wikipedia query: {query}")
        cache_key = f"wiki:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"[ExternalKnowledge] Wikipedia cache hit")
            return cached.get('results', [])
        
        results = []
        
        try:
            search_params = {
                'action': 'opensearch',
                'search': query,
                'limit': max_results,
                'namespace': 0,
                'format': 'json'
            }
            
            search_response = self._session.get(
                self.wikipedia_search_endpoint,
                params=search_params,
                timeout=10
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            print(f"[ExternalKnowledge] Wikipedia search found {len(search_data[1]) if len(search_data) > 1 else 0} titles")
            
            if len(search_data) >= 2:
                titles = search_data[1][:max_results]
                print(f"[ExternalKnowledge] Fetching summaries for: {titles}")
                
                for title in titles:
                    try:
                        summary_url = f"{self.wikipedia_endpoint}{title.replace(' ', '_')}"
                        summary_response = self._session.get(summary_url, timeout=10)
                        print(f"[ExternalKnowledge] Summary status for '{title}': {summary_response.status_code}")
                        
                        if summary_response.status_code == 200:
                            data = summary_response.json()
                            
                            content = data.get('extract', '')
                            if content:
                                basin_coords = None
                                if self.encoder:
                                    try:
                                        basin_coords = self.encoder.encode(content[:500])
                                    except Exception as enc_e:
                                        print(f"[ExternalKnowledge] Encoding error: {enc_e}")
                                
                                results.append({
                                    'source': 'wikipedia',
                                    'title': data.get('title', title),
                                    'content': content[:1000],
                                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                    'basin_coords': basin_coords,
                                    'weight': EXTERNAL_WEIGHT,
                                    'timestamp': time.time()
                                })
                                print(f"[ExternalKnowledge] Added Wikipedia result: {data.get('title', title)}")
                            else:
                                print(f"[ExternalKnowledge] No extract for '{title}'")
                        else:
                            print(f"[ExternalKnowledge] Failed to get summary for '{title}'")
                    except Exception as e:
                        print(f"[ExternalKnowledge] Wikipedia summary error for {title}: {e}")
                        continue
            
            self._set_cache(cache_key, {'results': results})
            
        except Exception as e:
            print(f"[ExternalKnowledge] Wikipedia search error: {e}")
        
        return results
    
    def query_duckduckgo_instant(self, query: str) -> List[Dict]:
        """
        Query DuckDuckGo Instant Answers API.
        Returns quick facts encoded to basin coordinates.
        """
        print(f"[ExternalKnowledge] DuckDuckGo query: {query}")
        cache_key = f"ddg:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            print(f"[ExternalKnowledge] DDG cache hit")
            return cached.get('results', [])
        
        results = []
        
        try:
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = self._session.get(
                self.ddg_endpoint,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            abstract = data.get('Abstract', '')
            if abstract:
                basin_coords = None
                if self.encoder:
                    basin_coords = self.encoder.encode(abstract[:500])
                
                results.append({
                    'source': 'duckduckgo',
                    'title': data.get('Heading', query),
                    'content': abstract,
                    'url': data.get('AbstractURL', ''),
                    'basin_coords': basin_coords,
                    'weight': EXTERNAL_WEIGHT,
                    'answer_type': data.get('Type', 'A'),
                    'timestamp': time.time()
                })
            
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and 'Text' in topic:
                    text = topic['Text']
                    basin_coords = None
                    if self.encoder and text:
                        basin_coords = self.encoder.encode(text[:300])
                    
                    results.append({
                        'source': 'duckduckgo_related',
                        'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' '),
                        'content': text,
                        'url': topic.get('FirstURL', ''),
                        'basin_coords': basin_coords,
                        'weight': EXTERNAL_WEIGHT * 0.8,
                        'timestamp': time.time()
                    })
            
            self._set_cache(cache_key, {'results': results})
            
        except Exception as e:
            print(f"[ExternalKnowledge] DuckDuckGo error: {e}")
        
        return results
    
    def query_all_sources(
        self,
        query: str,
        max_wiki: int = 3,
        include_ddg: bool = True,
        importance: int = 2
    ) -> List[Dict]:
        """
        Query all external sources with budget-aware priority.

        Priority:
        1. Budget search (Tavily/Perplexity/Google) if budget available
        2. Wikipedia (always free)
        3. DuckDuckGo Instant (always free)

        Args:
            query: Search query
            max_wiki: Max Wikipedia results
            include_ddg: Include DuckDuckGo results
            importance: Search importance (1-4) for budget allocation

        Returns:
            Combined results sorted by weight
        """
        all_results = []

        # Try budget search first (better quality results)
        if self._search_manager:
            budget_results = self.query_budget_search(query, importance=importance, max_results=5)
            all_results.extend(budget_results)

        # Then query free sources in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.query_wikipedia, query, max_wiki): 'wikipedia'
            }

            if include_ddg:
                futures[executor.submit(self.query_duckduckgo_instant, query)] = 'duckduckgo'

            for future in as_completed(futures, timeout=10):
                source = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"[ExternalKnowledge] {source} query failed: {e}")

        # Sort by weight (budget results first)
        all_results.sort(key=lambda x: x.get('weight', 0), reverse=True)

        return all_results
    
    def search_with_geometric_ranking(
        self,
        query: str,
        query_basin: Optional[np.ndarray] = None,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search external sources and rank by Fisher-Rao distance.
        
        Args:
            query: Text query
            query_basin: Pre-computed basin coordinates for query
            max_results: Max results to return
            
        Returns:
            Results sorted by geometric distance (closest first)
        """
        if query_basin is None and self.encoder:
            query_basin = self.encoder.encode(query)
        
        external_results = self.query_all_sources(query)
        
        if query_basin is None:
            return external_results[:max_results]
        
        for result in external_results:
            result_basin = result.get('basin_coords')
            if result_basin is not None:
                result_basin = np.asarray(result_basin, dtype=np.float64)
                distance = self._fisher_rao_distance(query_basin, result_basin)
                similarity = 1.0 - distance / np.pi
                result['distance'] = float(distance)
                result['similarity'] = float(np.clip(similarity, 0, 1))
            else:
                result['distance'] = float('inf')
                result['similarity'] = 0.0
        
        external_results.sort(key=lambda x: x['distance'])
        
        return external_results[:max_results]
    
    def _fisher_rao_distance(self, basin1: np.ndarray, basin2: np.ndarray) -> float:
        """
        Compute Fisher-Rao distance between two basin coordinates.
        Uses geodesic distance on probability simplex.
        """
        basin1 = np.asarray(basin1, dtype=np.float64)
        basin2 = np.asarray(basin2, dtype=np.float64)
        
        # Use L2 magnitude for validation checks only
        norm1 = np.sqrt(np.sum(basin1 ** 2))
        norm2 = np.sqrt(np.sum(basin2 ** 2))
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return np.pi
        
        basin1_norm = basin1 / norm1
        basin2_norm = basin2 / norm2
        
        dot_product = np.clip(np.dot(basin1_norm, basin2_norm), -1.0, 1.0)
        
        return float(np.arccos(dot_product))


def get_external_knowledge_base(encoder=None) -> ExternalKnowledgeBase:
    """Get singleton instance of external knowledge base."""
    global _external_kb
    
    if _external_kb is None:
        _external_kb = ExternalKnowledgeBase(encoder=encoder)
    elif encoder is not None and _external_kb.encoder is None:
        _external_kb.set_encoder(encoder)
    
    return _external_kb


def reset_external_knowledge_base():
    """Reset the singleton (for testing)."""
    global _external_kb
    _external_kb = None
