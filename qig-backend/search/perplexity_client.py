"""
Perplexity API Client - Full Research Capabilities

Provides complete access to Perplexity's AI-powered research capabilities
for high-quality QIG RAG learning.

Capabilities:
- chat: Standard chat completions with citations
- search: Grounded search with real-time information
- pro_search: Deep research with comprehensive synthesis

Uses the OpenAI-compatible API with Perplexity's specialized models:
- sonar: Fast, grounded responses
- sonar-pro: Advanced research with more comprehensive answers
- sonar-deep-research: Deep multi-step research (when available)

All results include citations for verification and learning.

BUDGET ENFORCEMENT: All paid API calls check the budget orchestrator
before execution. If the daily cost cap is exceeded, calls will be blocked.

CURRICULUM-ONLY MODE: All external searches are blocked when QIG_CURRICULUM_ONLY=true
"""

import os
import logging
import requests
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
            logger.warning("[PerplexityClient] Budget orchestrator not available - no cost limits")
    return _budget_orchestrator

def _check_budget(provider: str = 'perplexity') -> bool:
    """Check if budget allows this API call. Returns False if blocked."""
    orchestrator = _get_budget_orchestrator()
    if not orchestrator:
        return True  # Allow if orchestrator not available

    if not orchestrator.consume_quota(provider):
        logger.warning(f"[PerplexityClient] BLOCKED by budget orchestrator (provider={provider})")
        return False
    return True


@dataclass
class PerplexityCitation:
    """A citation from Perplexity response."""
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass
class PerplexityResponse:
    """Complete response from Perplexity."""
    content: str
    model: str
    citations: List[PerplexityCitation]
    usage: Dict[str, int]
    response_time: float
    related_questions: Optional[List[str]] = None


@dataclass 
class PerplexityResearchResult:
    """Deep research result from Perplexity."""
    query: str
    synthesis: str
    key_findings: List[str]
    sources: List[Dict[str, Any]]
    citations: List[PerplexityCitation]
    follow_up_topics: List[str]
    confidence: float
    timestamp: str


class PerplexityClient:
    """
    Full-featured Perplexity client for QIG learning.
    
    Provides all Perplexity API capabilities:
    - chat(): Standard chat with citations
    - search(): Grounded search queries
    - pro_search(): Deep research with synthesis
    - validate_insight(): Validate cross-domain insights
    
    Usage:
        client = PerplexityClient()
        
        # Simple search
        result = client.search("What is Fisher-Rao distance?")
        
        # Deep research
        research = client.pro_search("quantum information geometry applications")
        
        # Validate an insight
        validation = client.validate_insight(
            "Fisher-Rao distance in information geometry relates to Bures metric in quantum mechanics"
        )
    """
    
    API_BASE = "https://api.perplexity.ai"
    
    MODELS = {
        "sonar": "sonar",
        "sonar-pro": "sonar-pro", 
        "sonar-reasoning": "sonar-reasoning",
        "sonar-reasoning-pro": "sonar-reasoning-pro",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key. If not provided, uses PERPLEXITY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.available = False
        
        if not self.api_key:
            logger.warning("[PerplexityClient] PERPLEXITY_API_KEY not set - Perplexity disabled")
            return
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.available = True
        logger.info("[PerplexityClient] Initialized with full API capabilities")
    
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "sonar-pro",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        return_citations: bool = True,
        return_related_questions: bool = False,
        search_recency_filter: Optional[str] = None
    ) -> Optional[PerplexityResponse]:
        """
        Make API request to Perplexity.
        
        Args:
            messages: Chat messages
            model: Model to use
            temperature: Response temperature (0-1)
            max_tokens: Maximum response tokens
            return_citations: Include source citations
            return_related_questions: Include follow-up questions
            search_recency_filter: Time filter ("hour", "day", "week", "month", "year")
            
        Returns:
            PerplexityResponse or None on error or budget exceeded
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            logger.warning("[PerplexityClient] Request blocked by curriculum-only mode")
            return None
        
        if not self.available:
            return None

        # BUDGET CHECK: Consume quota before making API call
        if not _check_budget('perplexity'):
            logger.warning("[PerplexityClient] Request blocked by budget cap")
            return None

        try:
            import time
            start_time = time.time()
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "return_citations": return_citations,
                "return_related_questions": return_related_questions,
            }
            
            if search_recency_filter:
                payload["search_recency_filter"] = search_recency_filter
            
            response = requests.post(
                f"{self.API_BASE}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            elapsed = time.time() - start_time
            
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            citations = []
            for url in data.get('citations', []):
                if isinstance(url, str):
                    citations.append(PerplexityCitation(url=url))
                elif isinstance(url, dict):
                    citations.append(PerplexityCitation(
                        url=url.get('url', ''),
                        title=url.get('title'),
                        snippet=url.get('snippet')
                    ))
            
            return PerplexityResponse(
                content=content,
                model=data.get('model', model),
                citations=citations,
                usage=data.get('usage', {}),
                response_time=elapsed,
                related_questions=data.get('related_questions')
            )
            
        except requests.exceptions.Timeout:
            logger.error("[PerplexityClient] Request timeout")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"[PerplexityClient] HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"[PerplexityClient] Request error: {e}")
            return None
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: str = "sonar-pro",
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Optional[PerplexityResponse]:
        """
        Standard chat completion with citations.
        
        Args:
            message: User message/question
            system_prompt: Optional system prompt
            model: Model to use (sonar, sonar-pro)
            temperature: Response temperature
            max_tokens: Maximum response tokens
            
        Returns:
            PerplexityResponse with answer and citations
        """
        if not self.available:
            logger.warning("[PerplexityClient] Chat unavailable - client not initialized")
            return None
        
        logger.info(f"[PerplexityClient] Chat: '{message[:50]}...'")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        return self._make_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def search(
        self,
        query: str,
        recency: Optional[str] = None,
        model: str = "sonar-pro"
    ) -> Optional[PerplexityResponse]:
        """
        Grounded search with real-time information.
        
        Args:
            query: Search query
            recency: Time filter ("hour", "day", "week", "month", "year")
            model: Model to use
            
        Returns:
            PerplexityResponse with grounded answer and citations
        """
        if not self.available:
            logger.warning("[PerplexityClient] Search unavailable - client not initialized")
            return None
        
        logger.info(f"[PerplexityClient] Search: '{query[:50]}...'")
        
        system_prompt = (
            "You are a research assistant. Provide accurate, factual answers "
            "based on current information. Always cite your sources."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        return self._make_request(
            messages=messages,
            model=model,
            return_citations=True,
            return_related_questions=True,
            search_recency_filter=recency
        )
    
    def pro_search(
        self,
        query: str,
        context: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Optional[PerplexityResearchResult]:
        """
        Deep research with comprehensive synthesis.
        
        Performs multi-step research:
        1. Initial broad search
        2. Follow-up on key areas
        3. Synthesis of findings
        
        Args:
            query: Research query
            context: Additional context to consider
            focus_areas: Specific areas to focus on
            
        Returns:
            PerplexityResearchResult with comprehensive findings
        """
        if not self.available:
            logger.warning("[PerplexityClient] Pro search unavailable - client not initialized")
            return None
        
        logger.info(f"[PerplexityClient] Pro search: '{query[:50]}...'")
        
        system_prompt = """You are an expert research assistant conducting deep research.
        
Your task:
1. Thoroughly research the topic
2. Identify key findings and insights
3. Synthesize information from multiple sources
4. Provide a comprehensive answer with citations

Format your response as:
## Key Findings
- Finding 1
- Finding 2
...

## Synthesis
[Your comprehensive synthesis]

## Follow-up Topics
- Topic 1
- Topic 2
..."""
        
        user_message = query
        if context:
            user_message = f"Context: {context}\n\nResearch question: {query}"
        if focus_areas:
            user_message += f"\n\nFocus areas: {', '.join(focus_areas)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self._make_request(
            messages=messages,
            model="sonar-pro",
            temperature=0.3,
            max_tokens=4000,
            return_citations=True,
            return_related_questions=True
        )
        
        if not response:
            return None
        
        content = response.content
        key_findings = []
        synthesis = content
        follow_up_topics = []
        
        if "## Key Findings" in content:
            parts = content.split("## Key Findings")
            if len(parts) > 1:
                findings_section = parts[1].split("##")[0]
                for line in findings_section.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        key_findings.append(line[2:])
        
        if "## Synthesis" in content:
            parts = content.split("## Synthesis")
            if len(parts) > 1:
                synthesis = parts[1].split("##")[0].strip()
        
        if "## Follow-up Topics" in content:
            parts = content.split("## Follow-up Topics")
            if len(parts) > 1:
                topics_section = parts[1]
                for line in topics_section.strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        follow_up_topics.append(line[2:])
        
        if response.related_questions:
            follow_up_topics.extend(response.related_questions)
        
        return PerplexityResearchResult(
            query=query,
            synthesis=synthesis,
            key_findings=key_findings,
            sources=[{"url": c.url, "title": c.title} for c in response.citations],
            citations=response.citations,
            follow_up_topics=follow_up_topics[:5],
            confidence=min(0.9, 0.5 + 0.1 * len(response.citations)),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def validate_insight(
        self,
        insight: str,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate a cross-domain insight using Perplexity.
        
        Args:
            insight: The insight statement to validate
            domains: Source domains of the insight
            
        Returns:
            Dict with validation results
        """
        if not self.available:
            logger.warning("[PerplexityClient] Validation unavailable - client not initialized")
            return {"validated": False, "reason": "Client not available"}
        
        logger.info(f"[PerplexityClient] Validating insight: '{insight[:50]}...'")
        
        system_prompt = """You are a fact-checking research assistant.

Evaluate the following insight for accuracy:
1. Is it factually correct?
2. Are the claimed relationships supported by evidence?
3. What sources support or contradict this?

Provide a validation score (0-1) and explanation."""
        
        user_message = f"Validate this insight:\n\n{insight}"
        if domains:
            user_message += f"\n\nDomains: {', '.join(domains)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self._make_request(
            messages=messages,
            model="sonar-pro",
            temperature=0.1,
            max_tokens=1500,
            return_citations=True
        )
        
        if not response:
            return {"validated": False, "reason": "API error"}
        
        content = response.content.lower()
        
        score = 0.5
        if any(word in content for word in ["correct", "accurate", "valid", "supported", "confirmed"]):
            score += 0.2
        if any(word in content for word in ["incorrect", "inaccurate", "invalid", "unsupported", "contradicted"]):
            score -= 0.2
        if len(response.citations) >= 3:
            score += 0.1
        if len(response.citations) >= 5:
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        
        return {
            "validated": score >= 0.6,
            "score": score,
            "explanation": response.content,
            "citations": [c.url for c in response.citations],
            "citation_count": len(response.citations),
            "model": response.model
        }


_perplexity_client: Optional[PerplexityClient] = None


def get_perplexity_client() -> PerplexityClient:
    """Get or create singleton Perplexity client."""
    global _perplexity_client
    if _perplexity_client is None:
        _perplexity_client = PerplexityClient()
    return _perplexity_client
