"""
Search Source Indexer

Indexes URLs from premium search providers (Tavily, Perplexity, Google)
to the discovered_sources table for future reference and quality tracking.
"""

import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SearchSourceIndexer:
    """
    Indexes search result URLs to discovered_sources table.
    
    Lazy-loads persistence to avoid circular imports.
    """
    
    def __init__(self):
        self._persistence = None
        self._db_url = None
    
    def _get_db_connection(self):
        """Get database connection lazily."""
        import os
        import psycopg2
        
        if self._db_url is None:
            self._db_url = os.environ.get('DATABASE_URL')
        
        if not self._db_url:
            return None
        
        try:
            return psycopg2.connect(self._db_url)
        except Exception as e:
            logger.error(f"[SearchSourceIndexer] DB connection failed: {e}")
            return None
    
    def _normalize_url(self, url: str) -> Optional[str]:
        """Validate and normalize URL."""
        if not url:
            return None
        
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            return None
        
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return None
            return url
        except Exception:
            return None
    
    def _map_provider_to_origin(self, provider: str) -> str:
        """Map search provider to origin value."""
        mapping = {
            'tavily': 'search:tavily',
            'perplexity': 'search:perplexity',
            'google': 'search:google',
            'duckduckgo': 'search:duckduckgo',
        }
        return mapping.get(provider, f'search:{provider}')
    
    def _map_provider_to_category(self, provider: str, title: str = '') -> str:
        """Determine category based on provider and content."""
        if 'api' in title.lower() or 'documentation' in title.lower():
            return 'documentation'
        if 'research' in title.lower() or 'paper' in title.lower():
            return 'research'
        if 'news' in title.lower():
            return 'news'
        return 'general'
    
    def record_results(
        self,
        provider: str,
        query: str,
        results: List[Dict],
        kernel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record search result URLs to discovered_sources table.
        
        Args:
            provider: Search provider name (tavily, perplexity, google)
            query: The search query that produced these results
            results: List of result dicts with 'url', 'title', 'content' keys
            kernel_id: Optional kernel ID that triggered the search
        
        Returns:
            Summary of indexed sources
        """
        if not results:
            return {'indexed': 0, 'skipped': 0, 'errors': []}
        
        conn = self._get_db_connection()
        if not conn:
            logger.warning("[SearchSourceIndexer] No database connection, skipping indexing")
            return {'indexed': 0, 'skipped': 0, 'errors': ['no_database']}
        
        origin = self._map_provider_to_origin(provider)
        indexed = 0
        skipped = 0
        errors = []
        
        try:
            with conn.cursor() as cur:
                for result in results:
                    url = self._normalize_url(result.get('url', ''))
                    if not url:
                        skipped += 1
                        continue
                    
                    title = result.get('title', '')
                    category = self._map_provider_to_category(provider, title)
                    
                    try:
                        import json as json_module
                        from psycopg2.extras import Json
                        
                        metadata = Json({
                            'query': query[:100],
                            'title': title[:200],
                            'provider': provider
                        })
                        
                        cur.execute("""
                            INSERT INTO discovered_sources 
                                (url, category, origin, hit_count, phi_avg, phi_max,
                                 success_count, failure_count, is_active, metadata)
                            VALUES (%s, %s, %s, 1, 0.5, 0.5, 0, 0, true, %s)
                            ON CONFLICT (url) DO UPDATE SET
                                hit_count = discovered_sources.hit_count + 1,
                                updated_at = NOW(),
                                metadata = COALESCE(discovered_sources.metadata, '{}'::jsonb) || 
                                           COALESCE(EXCLUDED.metadata, '{}'::jsonb)
                        """, (
                            url,
                            category,
                            origin,
                            metadata
                        ))
                        indexed += 1
                    except Exception as e:
                        errors.append(f"{url}: {str(e)}")
                        logger.error(f"[SearchSourceIndexer] Failed to index {url}: {e}")
                
                conn.commit()
                
            if indexed > 0:
                logger.info(
                    f"[SearchSourceIndexer] Indexed {indexed} sources from {provider} "
                    f"(query: '{query[:50]}...', skipped: {skipped})"
                )
        except Exception as e:
            logger.error(f"[SearchSourceIndexer] Batch indexing failed: {e}")
            errors.append(str(e))
        finally:
            conn.close()
        
        return {
            'indexed': indexed,
            'skipped': skipped,
            'errors': errors if errors else None,
            'origin': origin
        }


_indexer: Optional[SearchSourceIndexer] = None


def get_source_indexer() -> SearchSourceIndexer:
    """Get singleton SearchSourceIndexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = SearchSourceIndexer()
    return _indexer


def index_search_results(
    provider: str,
    query: str,
    results: List[Dict],
    kernel_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to index search results."""
    return get_source_indexer().record_results(provider, query, results, kernel_id)
