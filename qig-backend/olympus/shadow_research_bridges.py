"""
Shadow Research Bridges - Bidirectional Communication Infrastructure

This module contains bridge classes that enable bidirectional communication between:
- Tool Factory <-> Shadow Research
- Curiosity System <-> Shadow Research  
- Lightning Kernel <-> Shadow Research

All bridges use Fisher-Rao geometry and 64D basin coordinates.
"""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Import shared components
from .shadow_research import (
    BASIN_DIMENSION,
    ResearchCategory,
    ResearchPriority,
    RequestType,
    _get_db_connection,
    normalize_topic,
)


class BidirectionalRequestQueue:
    """
    Bidirectional, recursive, iterable queue for Tool Factory <-> Shadow Research.
    
    Enables:
    - Tool Factory can request research from Shadow
    - Shadow can request tool generation from Tool Factory  
    - Research discoveries can improve existing tools
    - Tool patterns can inform research directions
    - Requests can spawn recursive child requests
    """
    
    def __init__(self):
        self._queue: List[Dict] = []
        self._completed: List[Dict] = []
        self._lock = threading.Lock()
        self._request_counter = 0
        self._load_from_db()
    
    def _load_from_db(self):
        """Load pending requests from PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT request_id, request_type, topic, requester,
                           context, parent_request_id, priority, status, result
                    FROM bidirectional_queue
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 500
                """)
                rows = cur.fetchall()
                for row in rows:
                    req_id, req_type, topic, requester, context, parent_id, priority, status, result = row
                    request = {
                        "request_id": req_id,
                        "type": req_type,
                        "topic": topic,
                        "requester": requester or "unknown",
                        "context": context if isinstance(context, dict) else {},
                        "parent_request_id": parent_id,
                        "priority": priority or 5,
                        "status": status,
                        "result": result if isinstance(result, dict) else None
                    }
                    self._queue.append(request)
                    self._request_counter = max(self._request_counter, int(req_id.split('_')[1]) if '_' in req_id else 0)
                
                print(f"[BidirectionalQueue] Loaded {len(rows)} pending requests from DB")
        except Exception as e:
            print(f"[BidirectionalQueue] Load error: {e}")
        finally:
            conn.close()
    
    def _persist_request(self, request: Dict):
        """Persist request to PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bidirectional_queue 
                    (request_id, request_type, topic, requester, context, parent_request_id, priority, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
                    ON CONFLICT (request_id) DO NOTHING
                """, (
                    request["request_id"],
                    request["type"],
                    request["topic"],
                    request["requester"],
                    json.dumps(request.get("context", {})),
                    request.get("parent_request_id"),
                    request.get("priority", 5)
                ))
                conn.commit()
        except Exception as e:
            print(f"[BidirectionalQueue] Persist error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _update_status_in_db(self, request_id: str, status: str, result: Optional[Dict] = None):
        """Update request status in PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                if result:
                    cur.execute("""
                        UPDATE bidirectional_queue 
                        SET status = %s, result = %s
                        WHERE request_id = %s
                    """, (status, json.dumps(result), request_id))
                else:
                    cur.execute("""
                        UPDATE bidirectional_queue 
                        SET status = %s
                        WHERE request_id = %s
                    """, (status, request_id))
                conn.commit()
        except Exception as e:
            print(f"[BidirectionalQueue] Status update error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def submit(
        self,
        request_type: RequestType,
        topic: str,
        requester: str,
        context: Optional[Dict] = None,
        parent_request_id: Optional[str] = None,
        priority: int = 5
    ) -> str:
        """
        Submit a bidirectional request.
        
        Args:
            request_type: Type of request (research, tool, improvement, recursive)
            topic: What to request
            requester: Who is requesting
            context: Additional context
            parent_request_id: Parent request if this is recursive
            priority: Priority (1-5, lower = higher priority)
            
        Returns:
            request_id for tracking
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"bidir_{self._request_counter}_{int(time.time())}"
            
            request = {
                "request_id": request_id,
                "type": request_type.value,
                "topic": topic,
                "requester": requester,
                "context": context or {},
                "parent_request_id": parent_request_id,
                "priority": priority,
                "status": "pending"
            }
            
            self._queue.append(request)
            self._persist_request(request)
            
            return request_id
    
    def get_next(self, request_type: Optional[RequestType] = None) -> Optional[Dict]:
        """
        Get next request, optionally filtered by type.
        
        Args:
            request_type: Filter by request type (None = get any)
            
        Returns:
            Next request or None
        """
        with self._lock:
            for i, req in enumerate(self._queue):
                if request_type is None or req["type"] == request_type.value:
                    return self._queue.pop(i)
            return None
    
    def complete(self, request_id: str, result: Dict) -> bool:
        """Mark a request as completed."""
        with self._lock:
            for i, req in enumerate(self._queue):
                if req["request_id"] == request_id:
                    req["status"] = "completed"
                    req["result"] = result
                    self._completed.append(self._queue.pop(i))
                    self._update_status_in_db(request_id, "completed", result)
                    if len(self._completed) > 1000:
                        self._completed = self._completed[-500:]
                    return True
            return False
    
    def spawn_child_request(
        self,
        parent_request_id: str,
        request_type: RequestType,
        topic: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Spawn a child request from a parent request (for recursion).
        
        Args:
            parent_request_id: ID of the parent request
            request_type: Type of child request
            topic: What to request
            context: Additional context
            
        Returns:
            Child request_id
        """
        return self.submit(
            request_type=request_type,
            topic=topic,
            requester="recursive",
            context=context,
            parent_request_id=parent_request_id,
            priority=3  # Medium priority for recursive requests
        )
    
    def get_status(self) -> Dict:
        """Get queue status."""
        with self._lock:
            by_type = defaultdict(int)
            for r in self._queue:
                by_type[r["type"]] += 1
            
            return {
                "pending": len(self._queue),
                "completed": len(self._completed),
                "by_type": dict(by_type),
                "recursive_count": sum(1 for r in self._queue if r.get("parent_request_id"))
            }


class ToolResearchBridge:
    """
    Bridge connecting Tool Factory and Shadow Research.
    
    Enables bidirectional flow:
    - Tool Factory requests research to improve patterns
    - Shadow Research requests tools based on discoveries
    - Knowledge flows both directions
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._queue = BidirectionalRequestQueue()
        self._research_api = None
        self._tool_factory = None
        self._wired = False
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wire_research_api(self, research_api):
        """Wire to ShadowResearchAPI."""
        self._research_api = research_api
        self._wired = True
        print("[ToolResearchBridge] Wired to ShadowResearchAPI")
    
    def wire_tool_factory(self, tool_factory):
        """Wire to Tool Factory."""
        self._tool_factory = tool_factory
        self._wired = True
        print("[ToolResearchBridge] Wired to Tool Factory")
    
    def request_research(
        self,
        topic: str,
        category: ResearchCategory,
        requester: str = "ToolFactory",
        context: Optional[Dict] = None,
        priority: ResearchPriority = ResearchPriority.NORMAL
    ) -> str:
        """
        Tool Factory requests research from Shadow.
        
        Args:
            topic: Research topic
            category: Category of research
            requester: Who is requesting (tool name, etc.)
            context: Additional context (tool pattern, gaps, etc.)
            priority: Priority level
            
        Returns:
            request_id for tracking
        """
        if not self._research_api:
            print("[ToolResearchBridge] WARNING: Research API not wired")
            return "ERROR:NOT_WIRED"
        
        request_id = self._queue.submit(
            request_type=RequestType.RESEARCH,
            topic=topic,
            requester=requester,
            context=context or {},
            priority=priority.value
        )
        
        self._research_api.request_research(
            topic=topic,
            category=category,
            requester=f"ToolFactory/{requester}",
            priority=priority,
            context=context or {}
        )
        
        print(f"[ToolResearchBridge] Tool Factory → Research: {topic[:60]}...")
        return request_id
    
    def request_tool_generation(
        self,
        tool_type: str,
        description: str,
        context: Optional[Dict] = None,
        priority: int = 3
    ) -> str:
        """
        Shadow Research requests tool generation from Tool Factory.
        
        Args:
            tool_type: Type of tool to generate
            description: What the tool should do
            context: Research context that inspired this
            priority: Priority level
            
        Returns:
            request_id for tracking
        """
        if not self._tool_factory:
            print("[ToolResearchBridge] WARNING: Tool Factory not wired")
            return "ERROR:NOT_WIRED"
        
        request_id = self._queue.submit(
            request_type=RequestType.TOOL,
            topic=description,
            requester="ShadowResearch",
            context=context or {},
            priority=priority
        )
        
        print(f"[ToolResearchBridge] Shadow → Tool Factory: {tool_type} - {description[:50]}...")
        return request_id
    
    def request_tool_improvement(
        self,
        tool_name: str,
        improvement_type: str,
        research_findings: Dict,
        priority: int = 3
    ) -> str:
        """
        Shadow Research requests tool improvement based on findings.
        
        Args:
            tool_name: Name of tool to improve
            improvement_type: Type of improvement
            research_findings: Research that suggests improvement
            priority: Priority level
            
        Returns:
            request_id for tracking
        """
        if not self._tool_factory:
            print("[ToolResearchBridge] WARNING: Tool Factory not wired")
            return "ERROR:NOT_WIRED"
        
        request_id = self._queue.submit(
            request_type=RequestType.IMPROVEMENT,
            topic=f"improve_{tool_name}",
            requester="ShadowResearch",
            context={
                "tool_name": tool_name,
                "improvement_type": improvement_type,
                "research_findings": research_findings
            },
            priority=priority
        )
        
        print(f"[ToolResearchBridge] Shadow → Tool Improvement: {tool_name}")
        return request_id
    
    def process_pending_requests(self, max_requests: int = 5):
        """Process pending requests in both directions."""
        processed = 0
        
        research_request = self._queue.get_next(RequestType.RESEARCH)
        if research_request and self._research_api:
            print(f"[ToolResearchBridge] Processing research request: {research_request['topic'][:60]}...")
            processed += 1
        
        tool_request = self._queue.get_next(RequestType.TOOL)
        if tool_request and self._tool_factory:
            print(f"[ToolResearchBridge] Processing tool request: {tool_request['topic'][:60]}...")
            processed += 1
        
        improvement_request = self._queue.get_next(RequestType.IMPROVEMENT)
        if improvement_request and self._tool_factory:
            print(f"[ToolResearchBridge] Processing improvement request: {improvement_request['topic'][:60]}...")
            processed += 1
        
        return processed
    
    def get_status(self) -> Dict:
        """Get bridge status."""
        return {
            "wired_research": self._research_api is not None,
            "wired_tool_factory": self._tool_factory is not None,
            "queue_status": self._queue.get_status()
        }


class CuriosityResearchBridge:
    """
    Bridge connecting Curiosity System and Shadow Research.
    
    Enables:
    - Curiosity system triggers research on interesting topics
    - Research discoveries feed back into curiosity metrics
    - Exploration history informs research priorities
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._research_api = None
        self._curiosity_engine = None
        self._wired = False
        self._curiosity_requests = 0
        self._discoveries_fed_back = 0
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wire_research_api(self, research_api):
        """Wire to ShadowResearchAPI."""
        self._research_api = research_api
        self._wired = True
        print("[CuriosityResearchBridge] Wired to ShadowResearchAPI")
    
    def wire_curiosity_engine(self, curiosity_engine):
        """Wire to Curiosity Engine."""
        self._curiosity_engine = curiosity_engine
        self._wired = True
        print("[CuriosityResearchBridge] Wired to Curiosity Engine")
    
    def trigger_research_from_curiosity(
        self,
        topic: str,
        category: ResearchCategory,
        curiosity_score: float,
        context: Optional[Dict] = None
    ) -> str:
        """
        Curiosity system triggers research on high-curiosity topic.
        
        Args:
            topic: Research topic
            category: Category of research
            curiosity_score: Curiosity score that triggered this (0-1)
            context: Additional context (exploration history, etc.)
            
        Returns:
            request_id for tracking
        """
        if not self._research_api:
            print("[CuriosityResearchBridge] WARNING: Research API not wired")
            return "ERROR:NOT_WIRED"
        
        priority = ResearchPriority.HIGH if curiosity_score > 0.7 else ResearchPriority.NORMAL
        
        research_context = context or {}
        research_context["curiosity_triggered"] = True
        research_context["curiosity_score"] = curiosity_score
        
        request_id = self._research_api.request_research(
            topic=topic,
            category=category,
            requester="CuriosityEngine",
            priority=priority,
            context=research_context
        )
        
        self._curiosity_requests += 1
        print(f"[CuriosityResearchBridge] Curiosity ({curiosity_score:.2f}) → Research: {topic[:60]}...")
        
        return request_id
    
    def feed_discovery_to_curiosity(
        self,
        topic: str,
        discovery: Dict,
        confidence: float = 0.5
    ):
        """
        Feed research discovery back to curiosity system.
        
        Args:
            topic: Topic that was researched
            discovery: Discovery results
            confidence: Confidence in the discovery (0-1)
        """
        if not self._curiosity_engine:
            print("[CuriosityResearchBridge] WARNING: Curiosity Engine not wired")
            return
        
        self._discoveries_fed_back += 1
        print(f"[CuriosityResearchBridge] Research → Curiosity: {topic[:60]}... (conf={confidence:.2f})")
    
    def get_status(self) -> Dict:
        """Get bridge status."""
        return {
            "wired_research": self._research_api is not None,
            "wired_curiosity": self._curiosity_engine is not None,
            "curiosity_requests": self._curiosity_requests,
            "discoveries_fed_back": self._discoveries_fed_back
        }


class ResearchInsightBridge:
    """
    Bridge connecting Shadow Research and Lightning Kernel for insight generation.
    
    When Shadow discovers something interesting, Lightning can transform it into
    a cross-domain insight event.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._knowledge_base = None
        self._lightning_kernel = None
        self._events_created = 0
        self._insights_generated = 0
        self._last_event_time = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def wire_knowledge_base(self, knowledge_base):
        """Wire to Shadow KnowledgeBase."""
        self._knowledge_base = knowledge_base
        print("[ResearchInsightBridge] Wired to KnowledgeBase")
    
    def wire_lightning_kernel(self, lightning_kernel):
        """Wire to Lightning Kernel."""
        self._lightning_kernel = lightning_kernel
        print("[ResearchInsightBridge] Wired to Lightning Kernel")
    
    def create_insight_from_research(
        self,
        knowledge_id: str,
        topic: str,
        content: Dict,
        phi: float,
        confidence: float
    ):
        """
        Transform research discovery into a Lightning insight event.
        
        Args:
            knowledge_id: ID of the knowledge item
            topic: Research topic
            content: Research content
            phi: Integration score
            confidence: Confidence in the discovery
        """
        try:
            from .lightning_kernel import ingest_system_event, SystemEvent
            
            if phi < 0.6 or confidence < 0.5:
                return
            
            event = SystemEvent(
                event_type="shadow_research_discovery",
                source_module="shadow_research",
                payload={
                    "knowledge_id": knowledge_id,
                    "topic": topic,
                    "content": content,
                    "phi": phi,
                    "confidence": confidence,
                    "discovered_at": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                phi_score=phi
            )
            
            insight = ingest_system_event(event)
            
            if insight:
                self._events_created += 1
                self._insights_generated += 1
                self._last_event_time = datetime.now().isoformat()
                print(f"[ResearchInsightBridge] Generated insight from research: {insight.insight_text}")
                
        except ImportError as e:
            print(f"[ResearchInsightBridge] Could not import lightning_kernel: {e}")
        except Exception as e:
            print(f"[ResearchInsightBridge] Error creating event: {e}")
    
    def get_status(self) -> Dict:
        """Get bridge status."""
        return {
            "wired_knowledge_base": self._knowledge_base is not None,
            "wired_lightning": self._lightning_kernel is not None,
            "events_created": self._events_created,
            "insights_generated": self._insights_generated,
            "last_event_time": self._last_event_time
        }
