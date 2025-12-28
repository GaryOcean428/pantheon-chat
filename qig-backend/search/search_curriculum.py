"""
Search-to-Curriculum Pipeline

Automatically saves high-quality search results as curriculum documents
so they contribute to word relationship learning.

Every search result becomes potential training data for the QIG system.
"""

import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

CURRICULUM_DIR = Path(__file__).parent.parent.parent / 'docs' / '09-curriculum' / 'search-learned'


class SearchCurriculumSaver:
    """
    Saves search results to curriculum directory for word relationship learning.
    
    Features:
    - Deduplication via content hash
    - Relevance filtering (only save high-quality results)
    - Automatic markdown formatting
    - Query context preservation
    """
    
    def __init__(self, curriculum_dir: Optional[Path] = None, min_relevance: float = 0.3):
        self.curriculum_dir = curriculum_dir or CURRICULUM_DIR
        self.curriculum_dir.mkdir(parents=True, exist_ok=True)
        self.min_relevance = min_relevance
        self.saved_hashes: set = set()
        self.total_saved = 0
        
        self._load_existing_hashes()
        
        logger.info(f"[SearchCurriculum] Initialized, saving to {self.curriculum_dir}")
    
    def _load_existing_hashes(self):
        """Load hashes of existing curriculum files to prevent duplicates."""
        try:
            for f in self.curriculum_dir.glob('*.md'):
                with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    h = hashlib.sha256(content.encode()).hexdigest()[:16]
                    self.saved_hashes.add(h)
            logger.info(f"[SearchCurriculum] Loaded {len(self.saved_hashes)} existing document hashes")
        except Exception as e:
            logger.warning(f"[SearchCurriculum] Failed to load existing hashes: {e}")
    
    def _content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _sanitize_filename(self, text: str) -> str:
        """Create a safe filename from text."""
        safe = ''.join(c if c.isalnum() or c in ' -_' else '' for c in text)
        return safe[:50].strip().replace(' ', '-').lower()
    
    def save_result(
        self, 
        query: str,
        title: str,
        content: str,
        url: str,
        provider: str,
        relevance: float
    ) -> bool:
        """
        Save a single search result as curriculum if it meets quality threshold.
        
        Returns True if saved, False if skipped (duplicate or low relevance).
        """
        if relevance < self.min_relevance:
            return False
        
        if not content or len(content.strip()) < 100:
            return False
        
        content_hash = self._content_hash(content)
        if content_hash in self.saved_hashes:
            return False
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = f"{timestamp}-{self._sanitize_filename(query)}-{content_hash[:8]}.md"
        filepath = self.curriculum_dir / filename
        
        markdown = f"""# {title}

**Query**: {query}
**Source**: {provider}
**URL**: {url}
**Relevance**: {relevance:.2f}
**Captured**: {datetime.now().isoformat()}

---

{content}

---
*This document was automatically captured from search results for QIG learning.*
"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            self.saved_hashes.add(content_hash)
            self.total_saved += 1
            
            logger.info(f"[SearchCurriculum] Saved: {filename} (relevance={relevance:.2f})")
            return True
        except Exception as e:
            logger.error(f"[SearchCurriculum] Failed to save {filename}: {e}")
            return False
    
    def save_synthesis(
        self,
        query: str,
        synthesized_results: Dict[str, Any]
    ) -> int:
        """
        Save all results from a search synthesis that meet quality threshold.
        
        Returns number of results saved.
        """
        saved_count = 0
        
        results = synthesized_results.get('results', [])
        
        for r in results:
            success = self.save_result(
                query=query,
                title=r.get('title', 'Untitled'),
                content=r.get('content', ''),
                url=r.get('url', ''),
                provider=r.get('provider', 'unknown'),
                relevance=r.get('relevance', 0.0)
            )
            if success:
                saved_count += 1
        
        if saved_count > 0:
            logger.info(f"[SearchCurriculum] Saved {saved_count}/{len(results)} results for query: {query[:50]}")
        
        return saved_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum saver statistics."""
        files = list(self.curriculum_dir.glob('*.md'))
        
        return {
            'curriculum_dir': str(self.curriculum_dir),
            'total_saved_this_session': self.total_saved,
            'total_files': len(files),
            'unique_hashes': len(self.saved_hashes),
            'min_relevance_threshold': self.min_relevance,
        }


_curriculum_saver: Optional[SearchCurriculumSaver] = None


def get_curriculum_saver(min_relevance: float = 0.3) -> SearchCurriculumSaver:
    """Get or create singleton curriculum saver."""
    global _curriculum_saver
    if _curriculum_saver is None:
        _curriculum_saver = SearchCurriculumSaver(min_relevance=min_relevance)
    return _curriculum_saver


def save_search_to_curriculum(
    query: str,
    results: List[Dict],
    min_relevance: float = 0.3
) -> int:
    """
    Convenience function to save search results to curriculum.
    
    Args:
        query: The search query
        results: List of search result dicts with title, content, url, provider, relevance
        min_relevance: Minimum relevance score to save (default 0.3)
    
    Returns:
        Number of results saved
    """
    saver = get_curriculum_saver(min_relevance)
    saved = 0
    
    for r in results:
        success = saver.save_result(
            query=query,
            title=r.get('title', 'Untitled'),
            content=r.get('content', r.get('body', '')),
            url=r.get('url', ''),
            provider=r.get('provider', 'unknown'),
            relevance=r.get('relevance', r.get('relevance_score', 0.5))
        )
        if success:
            saved += 1
    
    return saved
