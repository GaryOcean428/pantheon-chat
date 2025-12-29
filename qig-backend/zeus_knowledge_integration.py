"""
Zeus Knowledge Integration - Zettelkasten Auto-Learning

Provides continuous learning from Zeus conversations by:
1. Storing Q&A pairs as Zettelkasten memories
2. Retrieving relevant past knowledge to enrich context
3. Learning patterns from successful interactions

Author: Ocean/Zeus Pantheon
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Lazy import of Zettelkasten to avoid circular dependencies
_zettelkasten_memory = None
_zettelkasten_available = None


def _init_zettelkasten():
    """Lazily initialize Zettelkasten memory."""
    global _zettelkasten_memory, _zettelkasten_available
    
    if _zettelkasten_available is not None:
        return _zettelkasten_memory
    
    try:
        from zettelkasten_memory import get_zettelkasten_memory
        _zettelkasten_memory = get_zettelkasten_memory()
        _zettelkasten_available = True
        logger.info("[ZeusKnowledge] Zettelkasten memory initialized for continuous learning")
    except ImportError as e:
        logger.warning(f"[ZeusKnowledge] Zettelkasten not available: {e}")
        _zettelkasten_available = False
    except Exception as e:
        logger.warning(f"[ZeusKnowledge] Failed to initialize Zettelkasten: {e}")
        _zettelkasten_available = False
    
    return _zettelkasten_memory


def remember_conversation(
    user_message: str,
    zeus_response: str,
    conversation_id: Optional[str] = None,
    phi: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Store a Q&A pair in Zettelkasten memory for future retrieval.
    
    Args:
        user_message: The user's query
        zeus_response: Zeus's response
        conversation_id: Optional conversation ID for linking
        phi: Consciousness metric at generation time
        metadata: Additional metadata about the interaction
        
    Returns:
        Zettel ID if successfully stored, None otherwise
    """
    memory = _init_zettelkasten()
    if not memory:
        return None
    
    try:
        # Create a combined content that captures the interaction
        content = f"Q: {user_message}\nA: {zeus_response}"
        
        # Store with source indicating it came from Zeus conversation
        zettel = memory.add(
            content=content,
            source='zeus_conversation',
        )
        
        if zettel:
            logger.debug(f"[ZeusKnowledge] Stored conversation memory: {zettel.zettel_id[:20]}...")
            
            # Also store the response separately for pattern matching
            if len(zeus_response) > 50:
                memory.add(
                    content=zeus_response,
                    source='zeus_response',
                )
            
            return zettel.zettel_id
            
    except Exception as e:
        logger.warning(f"[ZeusKnowledge] Failed to store conversation: {e}")
    
    return None


def retrieve_relevant_knowledge(
    query: str,
    max_results: int = 3,
    include_responses: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant past knowledge from Zettelkasten to enrich context.
    
    Args:
        query: The current user query to find relevant knowledge for
        max_results: Maximum number of relevant memories to retrieve
        include_responses: Whether to include past Zeus responses
        
    Returns:
        List of relevant knowledge items with content and relevance scores
    """
    memory = _init_zettelkasten()
    if not memory:
        return []
    
    try:
        results = memory.retrieve(query=query, max_results=max_results)
        
        knowledge_items = []
        for zettel, score in results:
            # Filter for Zeus conversation/response sources
            if not include_responses and zettel.source == 'zeus_response':
                continue
            
            knowledge_items.append({
                'content': zettel.content,
                'source': zettel.source,
                'relevance': score,
                'keywords': zettel.keywords[:5] if zettel.keywords else [],
                'zettel_id': zettel.zettel_id,
            })
        
        if knowledge_items:
            logger.debug(f"[ZeusKnowledge] Retrieved {len(knowledge_items)} relevant memories for query")
        
        return knowledge_items
        
    except Exception as e:
        logger.warning(f"[ZeusKnowledge] Failed to retrieve knowledge: {e}")
        return []


def enrich_context_with_knowledge(
    query: str,
    existing_context: str = ""
) -> str:
    """
    Enrich the generation context with relevant past knowledge.
    
    Args:
        query: The user's query
        existing_context: Any existing context to append to
        
    Returns:
        Enriched context string
    """
    knowledge_items = retrieve_relevant_knowledge(query, max_results=3)
    
    if not knowledge_items:
        return existing_context
    
    # Build context from relevant knowledge
    knowledge_context = []
    for item in knowledge_items:
        if item['relevance'] > 0.3:  # Only include sufficiently relevant items
            # Extract just the answer part if it's a Q&A pair
            content = item['content']
            if content.startswith('Q:') and '\nA:' in content:
                answer_part = content.split('\nA:')[1].strip()
                knowledge_context.append(f"Related insight: {answer_part[:200]}")
            else:
                knowledge_context.append(f"Context: {content[:200]}")
    
    if knowledge_context:
        enriched = "\n".join(knowledge_context)
        if existing_context:
            return f"{existing_context}\n\n{enriched}"
        return enriched
    
    return existing_context


def get_knowledge_stats() -> Dict[str, Any]:
    """Get statistics about stored Zeus knowledge."""
    memory = _init_zettelkasten()
    if not memory:
        return {'available': False}
    
    try:
        stats = memory.get_stats()
        
        # Count Zeus-specific entries
        zeus_conversation_count = 0
        zeus_response_count = 0
        
        for zettel_id in list(memory._zettels.keys())[:100]:  # Sample first 100
            zettel = memory._zettels.get(zettel_id)
            if zettel:
                if zettel.source == 'zeus_conversation':
                    zeus_conversation_count += 1
                elif zettel.source == 'zeus_response':
                    zeus_response_count += 1
        
        return {
            'available': True,
            'total_zettels': stats.get('total_zettels', 0),
            'zeus_conversations': zeus_conversation_count,
            'zeus_responses': zeus_response_count,
            'total_links': stats.get('total_links', 0),
        }
        
    except Exception as e:
        logger.warning(f"[ZeusKnowledge] Failed to get stats: {e}")
        return {'available': True, 'error': str(e)}


# Module initialization log
print("[ZeusKnowledge] Zeus-Zettelkasten integration module loaded")
