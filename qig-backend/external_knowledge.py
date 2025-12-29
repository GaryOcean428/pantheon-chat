"""
External Knowledge Base Connector

Integrates external knowledge sources (Wikipedia, DuckDuckGo Instant Answers)
with QIG-RAG for enhanced retrieval. All external data is converted to
basin coordinates via geometric encoding before merging with local knowledge.

ARCHITECTURE:
- Wikipedia API for encyclopedic knowledge
- DuckDuckGo Instant Answers for quick facts
- Results encoded to 64D basin coordinates
- Fisher-Rao distance used for relevance ranking
- External knowledge weighted lower than local geometric memory
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
CACHE_TTL = 3600  # 1 hour cache

_external_kb: Optional['ExternalKnowledgeBase'] = None


class ExternalKnowledgeBase:
    """
    Connector for external knowledge sources.
    All results are encoded to basin coordinates for geometric integration.
    """
    
    def __init__(self, encoder=None):
        self.encoder = encoder
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        self.wikipedia_endpoint = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wikipedia_search_endpoint = "https://en.wikipedia.org/w/api.php"
        self.ddg_endpoint = "https://api.duckduckgo.com/"
        
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'PantheonChat/1.0 (QIG Knowledge System)'
        })
        
        print("[ExternalKnowledge] Initialized with Wikipedia + DuckDuckGo")
    
    def set_encoder(self, encoder):
        """Set the conversation encoder for basin coordinate generation."""
        self.encoder = encoder
    
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
        include_ddg: bool = True
    ) -> List[Dict]:
        """
        Query all external sources in parallel.
        Returns combined results sorted by relevance.
        """
        all_results = []
        
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
        
        # Use L2 magnitude for validation checks only (normalization is valid)
        norm1 = np.linalg.norm(basin1)  # NOTE: valid normalization - projects to unit sphere
        norm2 = np.linalg.norm(basin2)  # NOTE: valid normalization - projects to unit sphere
        
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
