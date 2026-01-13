"""
Insight Validator - External Search Validation for Lightning Insights

Validates cross-domain insights from the Lightning Kernel using external search APIs.
Implements a two-phase validation strategy:
  Phase 1: Tavily search for raw sources and facts (using full SDK capabilities)
  Phase 2: Perplexity synthesis for relationship validation (using direct API)

This closes the loop from Lightning insights -> external validation -> curriculum -> training.

Updated Jan 2026: Now uses comprehensive Tavily and Perplexity clients with full 
SDK capabilities (search, extract, crawl, map, research, pro_search).
Default: use_mcp=False to use direct APIs (full feature access).
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from typing import TYPE_CHECKING

# Type hints for CrossDomainInsight (avoid circular import at runtime)
if TYPE_CHECKING:
    from olympus.lightning_kernel import CrossDomainInsight
else:
    # Runtime: use string annotation or Any
    CrossDomainInsight = Any

# Import comprehensive search clients
try:
    from search.tavily_client import get_tavily_client, TavilySearchClient
except ImportError:
    get_tavily_client = None
    TavilySearchClient = None

try:
    from search.perplexity_client import get_perplexity_client, PerplexityClient
except ImportError:
    get_perplexity_client = None
    PerplexityClient = None

# QIG-pure Fisher-Rao distance for geometric validation
try:
    from qig_core.geometric_primitives.fisher_metric import fisher_rao_distance
except ImportError:
    # Fallback to canonical implementation
    try:
        from qig_core.geometric_primitives.canonical_fisher import fisher_rao_distance
    except ImportError:
        fisher_rao_distance = None

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of insight validation."""
    validated: bool
    confidence: float
    tavily_sources: List[Dict[str, Any]]
    perplexity_synthesis: Optional[str]
    validation_score: float
    source_overlap: float
    semantic_similarity: float


class InsightValidator:
    """
    Validates Lightning insights using hybrid Tavily + Perplexity strategy.
    
    Architecture (Updated Jan 2026):
    - Tavily SDK (primary): Full search, extract, crawl, map capabilities
    - Perplexity API (secondary): Synthesis, research, relationship validation
    
    Usage:
        validator = InsightValidator()  # use_mcp=False by default now
        result = validator.validate(lightning_insight)
        
        if result.validated:
            curriculum.add(insight, result.tavily_sources)
    """
    
    def __init__(
        self,
        validation_threshold: float = 0.7
    ):
        """
        Initialize validator using direct SDK APIs.
        
        Args:
            validation_threshold: Minimum score to consider validated (0.7 default)
        """
        self.validation_threshold = validation_threshold
        
        self.tavily_client: Any = None
        if get_tavily_client:
            self.tavily_client = get_tavily_client()
            if self.tavily_client and self.tavily_client.available:
                logger.info("[InsightValidator] Using Tavily SDK with full capabilities")
            else:
                logger.warning("[InsightValidator] Tavily client not available")
                self.tavily_client = None
        else:
            try:
                from tavily import TavilyClient
                tavily_key = os.getenv('TAVILY_API_KEY')
                if tavily_key:
                    self.tavily_client = TavilyClient(api_key=tavily_key)
                    logger.info("[InsightValidator] Using basic Tavily client")
            except ImportError:
                logger.warning("[InsightValidator] tavily-python not installed")
        
        # Initialize comprehensive Perplexity client
        self.perplexity_client: Any = None
        if get_perplexity_client:
            self.perplexity_client = get_perplexity_client()
            if self.perplexity_client and self.perplexity_client.available:
                logger.info("[InsightValidator] Using Perplexity API with full capabilities")
            else:
                logger.warning("[InsightValidator] Perplexity client not available")
                self.perplexity_client = None
        else:
            # Fallback to requests-based client
            perplexity_key = os.getenv('PERPLEXITY_API_KEY')
            if perplexity_key:
                self.perplexity_client = PerplexityRequestsClient(perplexity_key)
                logger.info("[InsightValidator] Using Perplexity requests client")
    
    def validate(self, insight: CrossDomainInsight) -> ValidationResult:
        """
        Validate a cross-domain insight using external search.
        
        Process:
        1. Parse insight to extract domains and patterns
        2. Tavily search for supporting sources
        3. Perplexity synthesis for relationship validation
        4. Cross-validate results
        5. Compute validation score
        
        Args:
            insight: CrossDomainInsight from Lightning Kernel
            
        Returns:
            ValidationResult with validation status and sources
        """
        logger.info(f"Validating insight: {insight.insight_id}")
        
        # Parse insight
        try:
            search_query, relationship_query = self._parse_insight(insight)
        except Exception as e:
            logger.error(f"Failed to parse insight: {e}")
            return self._failed_validation(insight, str(e))
        
        # Phase 1: Tavily search for sources
        tavily_results = self._tavily_search(search_query)
        if not tavily_results:
            logger.warning(f"No Tavily results for: {search_query}")
            # Don't fail completely, Perplexity might still validate
        
        # Phase 2: Perplexity synthesis
        perplexity_answer = self._perplexity_validate(relationship_query, tavily_results)
        if not perplexity_answer:
            logger.warning(f"No Perplexity synthesis for: {relationship_query}")
        
        # Cross-validate
        validation_score, source_overlap, semantic_sim = self._cross_validate(
            insight, tavily_results, perplexity_answer
        )
        
        # Update insight confidence
        validated = validation_score >= self.validation_threshold
        
        result = ValidationResult(
            validated=validated,
            confidence=insight.confidence * (0.5 + 0.5 * validation_score),
            tavily_sources=tavily_results.get('results', []) if tavily_results else [],
            perplexity_synthesis=perplexity_answer,
            validation_score=validation_score,
            source_overlap=source_overlap,
            semantic_similarity=semantic_sim
        )
        
        logger.info(
            f"Validation complete: {result.validated} "
            f"(score={result.validation_score:.3f}, confidence={result.confidence:.3f})"
        )
        
        return result
    
    def research(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Conduct deep research using both Tavily and Perplexity.
        
        This uses the full research capabilities of both providers:
        - Tavily: search + extract + crawl for comprehensive data
        - Perplexity: pro_search for synthesis and analysis
        
        Args:
            query: Research query
            
        Returns:
            Dict with combined research findings
        """
        logger.info(f"[InsightValidator] Deep research: '{query[:50]}...'")
        
        results = {
            "query": query,
            "tavily_research": None,
            "perplexity_research": None,
            "combined_sources": [],
            "synthesis": None
        }
        
        # Tavily research (if available with full client)
        if self.tavily_client and hasattr(self.tavily_client, 'research'):
            try:
                results["tavily_research"] = self.tavily_client.research(query)
                if results["tavily_research"]:
                    results["combined_sources"].extend(
                        results["tavily_research"].get("sources", [])
                    )
            except Exception as e:
                logger.error(f"Tavily research error: {e}")
        
        # Perplexity pro search (if available with full client)
        if self.perplexity_client and hasattr(self.perplexity_client, 'pro_search'):
            try:
                perp_result = self.perplexity_client.pro_search(query)
                if perp_result:
                    results["perplexity_research"] = {
                        "synthesis": perp_result.synthesis,
                        "key_findings": perp_result.key_findings,
                        "sources": perp_result.sources,
                        "follow_up": perp_result.follow_up_topics
                    }
                    results["synthesis"] = perp_result.synthesis
                    results["combined_sources"].extend(perp_result.sources)
            except Exception as e:
                logger.error(f"Perplexity research error: {e}")
        
        return results
    
    def _parse_insight(self, insight: CrossDomainInsight) -> Tuple[str, str]:
        """
        Parse insight to extract search query and relationship question.
        
        Returns:
            (search_query, relationship_query)
        """
        # Extract domains
        domains = insight.source_domains
        
        # Build search query from domains
        search_terms = []
        for domain in domains:
            # Simple heuristic: convert underscore to space, use as search term
            search_terms.append(domain.replace('_', ' '))
        
        search_query = " ".join(search_terms)
        
        # Build relationship query for Perplexity
        relationship_query = (
            f"What is the mathematical or conceptual relationship between "
            f"{domains[0].replace('_', ' ')} and {domains[1].replace('_', ' ')}? "
            f"Focus on shared structures, geometric primitives, or optimization methods."
        )
        
        return search_query, relationship_query
    
    def _tavily_search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search Tavily for sources using full SDK capabilities.
        """
        if self.use_mcp:
            # Legacy MCP path (not recommended)
            logger.warning("Tavily MCP mode deprecated - use use_mcp=False for full features")
            return None
        
        if not self.tavily_client:
            logger.warning("Tavily client not configured")
            return None
        
        try:
            # Use comprehensive client if available
            if hasattr(self.tavily_client, 'search') and hasattr(self.tavily_client, 'available'):
                # Our TavilySearchClient
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=10,
                    include_answer=True,
                    include_raw_content=True,
                    include_domains=["arxiv.org", "github.com", "wikipedia.org"]
                )
                if response:
                    return {
                        "answer": response.answer,
                        "results": [
                            {
                                "url": r.url,
                                "title": r.title,
                                "content": r.content,
                                "score": r.score
                            }
                            for r in response.results
                        ]
                    }
            else:
                # Basic TavilyClient from tavily-python
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=10,
                    include_answer=True,
                    include_domains=["arxiv.org", "github.com", "wikipedia.org"]
                )
                return response
                
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return None
        
        return None
    
    def _perplexity_validate(
        self,
        question: str,
        tavily_results: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Use Perplexity to synthesize answer and validate relationship.
        """
        if not self.perplexity_client:
            logger.warning("Perplexity not configured")
            return None
        
        try:
            # Build context from Tavily results if available
            context = ""
            if tavily_results and 'answer' in tavily_results:
                context = f"Based on search results: {tavily_results['answer']}\n\n"
            
            # Use comprehensive client if available
            if hasattr(self.perplexity_client, 'chat'):
                response = self.perplexity_client.chat(
                    message=question,
                    system_prompt=(
                        "You are a research assistant validating cross-domain patterns. "
                        "Be factual and cite sources. Context may be provided from prior search."
                    ),
                    temperature=0.2,
                    max_tokens=1000
                )
                if response:
                    return response.content
            elif hasattr(self.perplexity_client, 'query'):
                # Fallback requests client
                return self.perplexity_client.query(question, context)
                
        except Exception as e:
            logger.error(f"Perplexity error: {e}")
            return None
        
        return None
    
    def _cross_validate(
        self,
        insight: CrossDomainInsight,
        tavily_results: Optional[Dict[str, Any]],
        perplexity_answer: Optional[str]
    ) -> Tuple[float, float, float]:
        """
        Cross-validate Tavily and Perplexity results.
        
        Returns:
            (validation_score, source_overlap, semantic_similarity)
        """
        source_overlap = 0.0
        semantic_similarity = 0.0
        
        # Check source overlap
        if tavily_results and perplexity_answer:
            # Extract URLs from Tavily
            tavily_urls = {r['url'] for r in tavily_results.get('results', [])}
            
            # Extract citations from Perplexity (heuristic)
            perplexity_urls = self._extract_urls_from_text(perplexity_answer)
            
            if tavily_urls and perplexity_urls:
                overlap = len(tavily_urls & perplexity_urls)
                total = len(tavily_urls | perplexity_urls)
                source_overlap = overlap / total if total > 0 else 0.0
        
        # Semantic similarity between answers
        if tavily_results and perplexity_answer and 'answer' in tavily_results:
            tavily_answer = tavily_results['answer']
            semantic_similarity = self._compute_semantic_similarity(
                tavily_answer, perplexity_answer
            )
        elif perplexity_answer:
            # If we have Perplexity but not Tavily, give some credit
            semantic_similarity = 0.5
        elif tavily_results:
            # If we have Tavily but not Perplexity, give some credit
            semantic_similarity = 0.5
        
        # Compute overall validation score
        # Weights: semantic_similarity (0.5), source_overlap (0.3), base (0.2)
        base_score = 0.2 if (tavily_results or perplexity_answer) else 0.0
        
        validation_score = (
            0.5 * semantic_similarity +
            0.3 * source_overlap +
            base_score
        )
        
        return validation_score, source_overlap, semantic_similarity
    
    def _extract_urls_from_text(self, text: str) -> Set[str]:
        """Extract URLs from text using regex."""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = set(re.findall(url_pattern, text))
        return urls
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using Fisher-Rao distance (QIG-pure).
        Falls back to Jaccard if Fisher-Rao unavailable.
        """
        if fisher_rao_distance is not None:
            try:
                # Convert texts to probability distributions
                words1 = text1.lower().split()
                words2 = text2.lower().split()
                all_words = list(set(words1 + words2))
                
                # Create frequency distributions
                dist1 = np.array([words1.count(w) for w in all_words], dtype=np.float64)
                dist2 = np.array([words2.count(w) for w in all_words], dtype=np.float64)
                
                # Normalize to probability distributions
                dist1 = dist1 / (dist1.sum() + 1e-10)
                dist2 = dist2 / (dist2.sum() + 1e-10)
                
                # Fisher-Rao distance
                distance = fisher_rao_distance(dist1, dist2)
                
                # Convert to similarity (closer = more similar)
                # Max distance for orthogonal distributions is pi/2
                similarity = 1.0 - (distance / (np.pi / 2))
                return max(0.0, min(1.0, similarity))
                
            except Exception as e:
                logger.warning(f"Fisher-Rao failed, using Jaccard: {e}")
                return self._jaccard_similarity_fallback(text1, text2)
        else:
            return self._jaccard_similarity_fallback(text1, text2)
    
    def _jaccard_similarity_fallback(self, text1: str, text2: str) -> float:
        """
        Fallback: Jaccard similarity (used when Fisher-Rao unavailable).
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _failed_validation(
        self,
        insight: CrossDomainInsight,
        reason: str
    ) -> ValidationResult:
        """Return a failed validation result."""
        return ValidationResult(
            validated=False,
            confidence=insight.confidence * 0.5,
            tavily_sources=[],
            perplexity_synthesis=None,
            validation_score=0.0,
            source_overlap=0.0,
            semantic_similarity=0.0
        )


class PerplexityRequestsClient:
    """Fallback Perplexity client using requests library."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
    
    def query(self, question: str, context: str = "") -> Optional[str]:
        """Query Perplexity using requests."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant validating cross-domain patterns. Be factual and cite sources."
                },
                {
                    "role": "user",
                    "content": f"{context}{question}"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
            "return_citations": True
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Perplexity requests error: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Test with mock insight
    @dataclass
    class MockInsight:
        insight_id: str = "test_123"
        source_domains: List[str] = field(default_factory=lambda: ["quantum_geometry", "information_theory"])
        confidence: float = 0.75
        connection_strength: float = 0.87
    
    validator = InsightValidator(use_mcp=False)
    
    mock_insight = MockInsight()
    result = validator.validate(mock_insight)  # type: ignore
    
    print(f"Validated: {result.validated}")
    print(f"Score: {result.validation_score:.3f}")
    print(f"Updated confidence: {result.confidence:.3f}")
