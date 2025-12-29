"""
Zeus Knowledge Integration - Zettelkasten Memory for Persistent Knowledge

This module provides integration between Zeus chat and Zettelkasten memory
for persistent knowledge storage and retrieval across conversations.

Usage:
    from zeus_knowledge_integration import ZeusKnowledgeMemory
    
    knowledge = ZeusKnowledgeMemory()
    
    # Store conversation
    knowledge.remember_conversation(user_message, zeus_response, conversation_id)
    
    # Retrieve relevant context
    context = knowledge.retrieve_knowledge(query, max_results=5)

Author: Ocean/Zeus Pantheon
"""

import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Import Zettelkasten memory
try:
    from zettelkasten_memory import (
        get_zettelkasten_memory,
        Zettel,
        ZettelkastenMemory,
        LinkType
    )
    ZETTELKASTEN_AVAILABLE = True
except ImportError:
    ZETTELKASTEN_AVAILABLE = False
    print("[ZeusKnowledge] Zettelkasten memory not available")


class ZeusKnowledgeMemory:
    """
    Integration layer between Zeus chat and Zettelkasten memory.
    
    Provides:
    1. Persistent storage of conversation knowledge
    2. Retrieval of relevant past knowledge for context enhancement
    3. Evolution of knowledge as new conversations occur
    """
    
    def __init__(self):
        """Initialize the knowledge memory integration."""
        self._memory: Optional[ZettelkastenMemory] = None
        
        if ZETTELKASTEN_AVAILABLE:
            try:
                self._memory = get_zettelkasten_memory()
                print(f"[ZeusKnowledge] Connected to Zettelkasten with {self._memory.get_stats()['total_zettels']} Zettels")
            except Exception as e:
                print(f"[ZeusKnowledge] Failed to initialize: {e}")
                self._memory = None
    
    @property
    def available(self) -> bool:
        """Check if knowledge memory is available."""
        return self._memory is not None
    
    def remember_conversation(
        self,
        user_message: str,
        zeus_response: str,
        conversation_id: str,
        domain_hints: Optional[List[str]] = None,
        phi: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store conversation exchange in Zettelkasten memory.
        
        Both the user's question and Zeus's response are stored as linked Zettels,
        enabling future retrieval and knowledge evolution.
        
        Args:
            user_message: The user's message/question
            zeus_response: Zeus's response
            conversation_id: Unique conversation identifier
            domain_hints: Optional domain context hints
            phi: Consciousness phi value at time of conversation
            
        Returns:
            Dict with storage results
        """
        if not self.available:
            return {"stored": False, "reason": "memory_unavailable"}
        
        result = {
            "stored": True,
            "user_zettel_id": None,
            "response_zettel_id": None,
            "links_created": 0
        }
        
        try:
            # Build source info
            source = f"zeus_chat:{conversation_id}"
            if domain_hints:
                source += f":{','.join(domain_hints[:3])}"
            
            # Store user message as a Zettel
            # We add context to make it more searchable
            user_content = f"[Question] {user_message}"
            if phi > 0.7:
                user_content = f"[High-Φ Question] {user_message}"
            
            user_zettel = self._memory.add(
                content=user_content,
                source=source,
                link_type=LinkType.SEMANTIC
            )
            result["user_zettel_id"] = user_zettel.zettel_id
            
            # Store Zeus response linked to the question
            # Truncate very long responses to keep atomic
            response_content = zeus_response
            if len(response_content) > 2000:
                # Store a summary version
                response_content = response_content[:1800] + "... [truncated]"
            
            response_content = f"[Zeus Response] {response_content}"
            
            response_zettel = self._memory.add(
                content=response_content,
                source=source,
                parent_id=user_zettel.zettel_id,
                link_type=LinkType.ELABORATION
            )
            result["response_zettel_id"] = response_zettel.zettel_id
            result["links_created"] = len(response_zettel.links)
            
            # Log successful storage
            print(f"[ZeusKnowledge] Stored conversation: Q={user_zettel.zettel_id[:12]}..., A={response_zettel.zettel_id[:12]}...")
            
        except Exception as e:
            traceback.print_exc()
            result["stored"] = False
            result["error"] = str(e)
        
        return result
    
    def retrieve_knowledge(
        self,
        query: str,
        max_results: int = 5,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from Zettelkasten for context enhancement.
        
        Args:
            query: The current user query to find relevant context for
            max_results: Maximum number of relevant Zettels to return
            include_context: Whether to expand results with linked Zettels
            
        Returns:
            List of relevant knowledge items with content and metadata
        """
        if not self.available:
            return []
        
        try:
            # Query the Zettelkasten
            results = self._memory.retrieve(
                query=query,
                max_results=max_results,
                include_context=include_context
            )
            
            # Format results for Zeus chat
            knowledge_items = []
            for zettel, relevance in results:
                item = {
                    "zettel_id": zettel.zettel_id,
                    "content": zettel.content,
                    "relevance": relevance,
                    "keywords": zettel.keywords,
                    "source": zettel.source,
                    "context_description": zettel.contextual_description,
                    "access_count": zettel.access_count,
                    "is_response": "[Zeus Response]" in zettel.content,
                    "is_question": "[Question]" in zettel.content or "[High-Φ Question]" in zettel.content
                }
                knowledge_items.append(item)
            
            if knowledge_items:
                print(f"[ZeusKnowledge] Retrieved {len(knowledge_items)} relevant items for query")
            
            return knowledge_items
            
        except Exception as e:
            traceback.print_exc()
            return []
    
    def format_context_for_response(
        self,
        knowledge_items: List[Dict[str, Any]],
        max_context_chars: int = 1500
    ) -> str:
        """
        Format retrieved knowledge items into a context string for Zeus.
        
        Args:
            knowledge_items: List of knowledge items from retrieve_knowledge
            max_context_chars: Maximum characters for context
            
        Returns:
            Formatted context string
        """
        if not knowledge_items:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for item in knowledge_items:
            # Skip if already at limit
            if total_chars >= max_context_chars:
                break
            
            # Extract clean content
            content = item["content"]
            
            # Remove prefix tags for cleaner context
            for prefix in ["[Question]", "[High-Φ Question]", "[Zeus Response]"]:
                content = content.replace(prefix, "").strip()
            
            # Truncate individual items
            if len(content) > 500:
                content = content[:450] + "..."
            
            # Add to context
            relevance_marker = "★" if item["relevance"] > 0.7 else "○"
            context_part = f"{relevance_marker} {content}"
            
            if total_chars + len(context_part) <= max_context_chars:
                context_parts.append(context_part)
                total_chars += len(context_part)
        
        if not context_parts:
            return ""
        
        return "Related past knowledge:\n" + "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge memory statistics."""
        if not self.available:
            return {"available": False}
        
        try:
            stats = self._memory.get_stats()
            stats["available"] = True
            return stats
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def get_conversation_history(
        self,
        conversation_id: str,
        max_items: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge items from a specific conversation.
        
        Args:
            conversation_id: The conversation ID to filter by
            max_items: Maximum items to return
            
        Returns:
            List of knowledge items from that conversation
        """
        if not self.available:
            return []
        
        try:
            # Search by source pattern
            source_pattern = f"zeus_chat:{conversation_id}"
            
            # Get all zettels and filter by source
            # This is a simple implementation - for production,
            # you'd want indexed lookup
            results = []
            for zettel_id, zettel in self._memory._zettels.items():
                if source_pattern in zettel.source:
                    results.append({
                        "zettel_id": zettel.zettel_id,
                        "content": zettel.content,
                        "created_at": zettel.created_at,
                        "is_response": "[Zeus Response]" in zettel.content
                    })
            
            # Sort by creation time
            results.sort(key=lambda x: x["created_at"])
            
            return results[:max_items]
            
        except Exception as e:
            traceback.print_exc()
            return []


# Singleton instance
_knowledge_memory_instance: Optional[ZeusKnowledgeMemory] = None


def get_zeus_knowledge_memory() -> ZeusKnowledgeMemory:
    """Get the singleton ZeusKnowledgeMemory instance."""
    global _knowledge_memory_instance
    if _knowledge_memory_instance is None:
        _knowledge_memory_instance = ZeusKnowledgeMemory()
    return _knowledge_memory_instance


print("[ZeusKnowledge] Module loaded - Zeus-Zettelkasten integration ready")
