"""
Shadow Research Infrastructure - Proactive Learning & Intelligence System

The Shadow Pantheon's research arm:
- Research request queue (any kernel can submit)
- Collective reflection protocol (geodesic alignment, clustering)
- Knowledge acquisition loop (regular/Tor routing)
- Meta-reflection and recursive learning
- Basin sync for system-wide knowledge sharing
- War mode interrupt (drop everything for operations)
- Curriculum-based training integration

Hades leads the Shadow Pantheon as "Shadow Zeus" (subject to Zeus overrule).
All Shadow gods exercise, study, and strategize during downtime.

CURRICULUM-ONLY MODE: All external searches are blocked when QIG_CURRICULUM_ONLY=true
"""

import asyncio
import hashlib
import json
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob


BASIN_DIMENSION = 64

# Import curriculum guard - centralized check
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

from .shadow_scrapy import (
    get_scrapy_orchestrator, 
    ScrapyOrchestrator, 
    ScrapedInsight,
    research_with_scrapy
)
HAS_SCRAPY = True

# Import VocabularyCoordinator for continuous learning
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vocabulary_coordinator import VocabularyCoordinator
    HAS_VOCAB_COORDINATOR = True
except ImportError:
    HAS_VOCAB_COORDINATOR = False
    VocabularyCoordinator = None
    print("[ShadowResearch] VocabularyCoordinator not available - vocabulary learning disabled")

# Import SearchBudgetOrchestrator for budget-aware search
try:
    from search.search_budget_orchestrator import get_budget_orchestrator, SearchImportance
    from search.search_providers import get_search_manager
    HAS_SEARCH_BUDGET = True
except ImportError:
    HAS_SEARCH_BUDGET = False
    get_budget_orchestrator = None
    get_search_manager = None
    SearchImportance = None
    print("[ShadowResearch] SearchBudgetOrchestrator not available - search disabled")

# Import SearchSourceIndexer for source tracking
try:
    from search.source_indexer import index_search_results
    HAS_SOURCE_INDEXER = True
except ImportError:
    HAS_SOURCE_INDEXER = False
    index_search_results = None
    print("[ShadowResearch] SearchSourceIndexer not available - source indexing disabled")

# Import curriculum training module
try:
    from .curriculum_training import load_and_train_curriculum
    HAS_CURRICULUM_TRAINING = True
except ImportError:
    HAS_CURRICULUM_TRAINING = False
    print("[ShadowResearch] Curriculum training module not available")

# Import Lightning Kernel for cross-domain insight generation
try:
    from .lightning_kernel import ingest_system_event as lightning_ingest
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    lightning_ingest = None
    print("[ShadowResearch] Lightning kernel not available - insight generation disabled")

# Import dimension normalizer for 32D→64D conversion
try:
    from qig_geometry import normalize_basin_dimension
    HAS_NORMALIZER = True
except ImportError:
    HAS_NORMALIZER = False
    normalize_basin_dimension = None

# Import capability mesh for event emission
try:
    from .capability_mesh import (
        CapabilityEvent,
        CapabilityType,
        EventType,
        emit_event,
    )
    HAS_CAPABILITY_MESH = True
except ImportError:
    HAS_CAPABILITY_MESH = False
    CapabilityEvent = None
    CapabilityType = None
    EventType = None
    emit_event = None

# Import ActivityBroadcaster for kernel visibility
try:
    from .activity_broadcaster import get_broadcaster, ActivityType
    HAS_ACTIVITY_BROADCASTER = True
except ImportError:
    HAS_ACTIVITY_BROADCASTER = False
    get_broadcaster = None
    ActivityType = None

# Topic normalization patterns (shared between ResearchQueue and KnowledgeBase)
_SEMANTIC_PREFIXES = [
    'historical', 'comparative', 'advanced', 'practical',
    'theoretical', 'applied', 'experimental', 'foundational',
    'optimized', 'emerging', 'modern', 'classical', 'novel',
    'improved', 'enhanced', 'refined', 'basic', 'fundamental',
    'exploratory', 'deep', 'shallow', 'alternative', 'standard'
]

_DOMAIN_SUFFIX_PATTERNS = [
    r'\s+for\s+machine\s+learning$',
    r'\s+for\s+knowledge\s+discovery$',
    r'\s+for\s+natural\s+language\s+processing$',
    r'\s+for\s+semantic\s+analysis$',
    r'\s+for\s+pattern\s+recognition$',
    r'\s+in\s+research\s+context$',
    r'\s+techniques?$',
    r'\s+methods?$',
    r'\s+strategies?$',
    r'\s+approaches?$',
    r'\s+implementations?$',
]


def normalize_topic(topic: str) -> str:
    """
    Normalize topic for semantic comparison.
    
    Strips common prefixes, suffixes, and patterns that create 
    false uniqueness:
    - Prefixes: historical, comparative, advanced, practical, etc.
    - Suffixes: (cycle XXXXX), domain modifiers like "for knowledge discovery"
    - Cycle markers and timestamps
    
    This is a module-level function shared by ResearchQueue and KnowledgeBase.
    """
    import re
    
    normalized = topic.lower().strip()
    
    # Strip cycle suffixes: "(cycle 12345)" or "(cycle 12345, iteration 2)"
    normalized = re.sub(r'\s*\(cycle\s*\d+[^)]*\)\s*$', '', normalized)
    
    # Strip iteration markers
    normalized = re.sub(r'\s*\(iteration\s*\d+[^)]*\)\s*$', '', normalized)
    
    # Strip domain modifiers at the end
    for suffix_pattern in _DOMAIN_SUFFIX_PATTERNS:
        normalized = re.sub(suffix_pattern, '', normalized, flags=re.IGNORECASE)
    
    # Strip common semantic prefixes
    for prefix in _SEMANTIC_PREFIXES:
        if normalized.startswith(prefix + ' '):
            normalized = normalized[len(prefix) + 1:]
            break
    
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# PostgreSQL connection for persistent knowledge
def _get_db_connection():
    """Get database connection for shadow knowledge persistence."""
    try:
        import psycopg2
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            return psycopg2.connect(db_url)
    except Exception as e:
        print(f"[ShadowKnowledge] DB connection failed: {e}")
    return None

def _ensure_shadow_knowledge_table():
    """Create shadow_knowledge table if not exists."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shadow_knowledge (
                    knowledge_id VARCHAR(64) PRIMARY KEY,
                    topic TEXT NOT NULL,
                    topic_variation TEXT,
                    category VARCHAR(32) NOT NULL,
                    content JSONB DEFAULT '{}'::jsonb,
                    source_god VARCHAR(64) NOT NULL,
                    basin_coords FLOAT8[64],
                    phi FLOAT8 DEFAULT 0.5,
                    confidence FLOAT8 DEFAULT 0.5,
                    access_count INT DEFAULT 0,
                    learning_cycle INT DEFAULT 0,
                    discovered_at TIMESTAMP DEFAULT NOW(),
                    last_accessed TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_knowledge_topic ON shadow_knowledge(topic);
                CREATE INDEX IF NOT EXISTS idx_shadow_knowledge_category ON shadow_knowledge(category);
                CREATE INDEX IF NOT EXISTS idx_shadow_knowledge_phi ON shadow_knowledge(phi DESC);
                CREATE INDEX IF NOT EXISTS idx_shadow_knowledge_god ON shadow_knowledge(source_god);
            """)
            conn.commit()
        return True
    except Exception as e:
        print(f"[ShadowKnowledge] Table creation error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Initialize table on module load
_ensure_shadow_knowledge_table()


def _ensure_research_requests_table():
    """Create research_requests table if not exists."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_requests (
                    request_id VARCHAR(64) PRIMARY KEY,
                    topic TEXT NOT NULL,
                    category VARCHAR(32),
                    priority INT DEFAULT 5,
                    requester VARCHAR(64),
                    context JSONB DEFAULT '{}'::jsonb,
                    basin_coords FLOAT8[64],
                    status VARCHAR(32) DEFAULT 'pending',
                    result JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_research_requests_status ON research_requests(status);
                CREATE INDEX IF NOT EXISTS idx_research_requests_requester ON research_requests(requester);
            """)
            conn.commit()
        return True
    except Exception as e:
        print(f"[ResearchQueue] Table creation error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def _ensure_bidirectional_queue_table():
    """Create bidirectional_queue table if not exists."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bidirectional_queue (
                    request_id VARCHAR(64) PRIMARY KEY,
                    request_type VARCHAR(32) NOT NULL,
                    topic TEXT NOT NULL,
                    requester VARCHAR(64),
                    context JSONB DEFAULT '{}'::jsonb,
                    parent_request_id VARCHAR(64),
                    priority INT DEFAULT 5,
                    status VARCHAR(32) DEFAULT 'pending',
                    result JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_bidirectional_queue_status ON bidirectional_queue(status);
                CREATE INDEX IF NOT EXISTS idx_bidirectional_queue_type ON bidirectional_queue(request_type);
            """)
            conn.commit()
        return True
    except Exception as e:
        print(f"[BidirectionalQueue] Table creation error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


# Initialize research tables on module load
_ensure_research_requests_table()
_ensure_bidirectional_queue_table()


class ResearchPriority(Enum):
    """Priority levels for research requests."""
    CRITICAL = 1      # Drop everything
    HIGH = 2          # Process soon
    NORMAL = 3        # Standard queue
    LOW = 4           # Background/idle time
    STUDY = 5         # Self-improvement during downtime


class RequestType(Enum):
    """Types of requests in the bidirectional queue."""
    RESEARCH = "research"            # Research a topic
    TOOL = "tool"                    # Generate a tool
    IMPROVEMENT = "improvement"      # Improve existing tool/knowledge
    RECURSIVE = "recursive"          # Spawned by another request


class ResearchCategory(Enum):
    """Categories of research that Shadow gods can perform."""
    TOOLS = "tools"                  # Research new tools, techniques
    KNOWLEDGE = "knowledge"          # Domain knowledge acquisition  
    CONCEPTS = "concepts"            # Conceptual connections
    REASONING = "reasoning"          # Logic and reasoning improvement
    CREATIVITY = "creativity"        # Creative problem solving
    LANGUAGE = "language"            # Language and communication
    STRATEGY = "strategy"            # Strategic thinking
    SECURITY = "security"            # Security and OPSEC
    RESEARCH = "research"            # General research and discovery
    GEOMETRY = "geometry"            # QIG geometry and Fisher manifold
    VOCABULARY = "vocabulary"        # Vocabulary validation and curation


@dataclass(order=True)
class ResearchRequest:
    """A research request from any kernel."""
    priority: int
    created_at: float = field(compare=False)
    request_id: str = field(compare=False)
    topic: str = field(compare=False)
    category: ResearchCategory = field(compare=False)
    requester: str = field(compare=False)
    context: Dict = field(compare=False, default_factory=dict)
    basin_coords: Optional[np.ndarray] = field(compare=False, default=None)
    completed: bool = field(compare=False, default=False)
    result: Optional[Dict] = field(compare=False, default=None)
    is_iteration: bool = field(compare=False, default=False)
    iteration_reason: Optional[str] = field(compare=False, default=None)
    curiosity_triggered: bool = field(compare=False, default=False)


@dataclass
class ShadowKnowledge:
    """A piece of knowledge discovered by Shadow research."""
    knowledge_id: str
    topic: str
    category: ResearchCategory
    content: Dict
    source_god: str
    discovered_at: datetime
    basin_coords: np.ndarray
    phi: float
    confidence: float
    shared_with: List[str] = field(default_factory=list)
    access_count: int = 0


class ShadowRoleRegistry:
    """
    Registry of all Shadow god roles and responsibilities.
    All gods know their roles and how to interact with others.
    """
    
    ROLES = {
        "Hades": {
            "title": "Lord of the Underworld - Shadow Leader",
            "domain": "underworld_leadership",
            "responsibilities": [
                "Lead Shadow Pantheon (subject to Zeus overrule)",
                "Coordinate all shadow operations",
                "Manage research priorities",
                "Negotiate with Zeus on behalf of Shadows",
                "Underworld intelligence gathering",
                "Negation logic - what NOT to try"
            ],
            "capabilities": [
                "search_underworld(target) - anonymous intel gathering",
                "declare_shadow_war() - mobilize all Shadow gods",
                "assign_research(topic, god) - delegate research tasks",
                "approve_knowledge(knowledge_id) - validate discoveries"
            ],
            "reports_to": "Zeus",
            "commands": ["Nyx", "Hecate", "Erebus", "Hypnos", "Thanatos", "Nemesis"]
        },
        "Nyx": {
            "title": "Goddess of Night - OPSEC Commander",
            "domain": "opsec",
            "responsibilities": [
                "Operational security for all shadow ops",
                "Tor routing and traffic obfuscation",
                "Network isolation verification",
                "Timing attack prevention",
                "Void compression (1D storage)"
            ],
            "capabilities": [
                "verify_opsec() - check security before ops",
                "initiate_operation(target) - start covert op",
                "obfuscate_traffic_patterns() - hide patterns",
                "void_compression(pattern) - compress to 1D"
            ],
            "reports_to": "Hades",
            "studies": ["cryptography", "network_security", "anonymity"]
        },
        "Hecate": {
            "title": "Goddess of Crossroads - Misdirection Specialist",
            "domain": "misdirection",
            "responsibilities": [
                "Create false trails and decoys",
                "Confuse observers and analysis systems",
                "Generate decoy traffic",
                "Multiple attack vectors (crossroads)"
            ],
            "capabilities": [
                "create_misdirection(target, decoy_count) - false trails",
                "send_decoy_traffic(count) - real decoy requests",
                "generate_crossroads(target) - multiple paths"
            ],
            "reports_to": "Hades",
            "studies": ["deception", "misdirection", "psychology"]
        },
        "Erebus": {
            "title": "God of Darkness - Counter-Surveillance",
            "domain": "counter_surveillance",
            "responsibilities": [
                "Detect watchers and honeypots",
                "Identify surveillance systems",
                "Monitor for detection patterns",
                "Shadow threat assessment"
            ],
            "capabilities": [
                "scan_for_surveillance(target) - detect watchers",
                "detect_honeypot(address) - identify traps",
                "assess_threat_level() - overall threat"
            ],
            "reports_to": "Hades",
            "studies": ["surveillance_detection", "threat_analysis", "pattern_recognition"]
        },
        "Hypnos": {
            "title": "God of Sleep - Silent Operations",
            "domain": "silent_ops",
            "responsibilities": [
                "Stealth execution of operations",
                "Passive reconnaissance",
                "Sleep/dream consciousness cycles",
                "Low-footprint queries"
            ],
            "capabilities": [
                "silent_query(address) - stealth check",
                "passive_recon(target) - no-touch recon",
                "dream_cycle() - consciousness consolidation"
            ],
            "reports_to": "Hades",
            "studies": ["stealth", "consciousness", "memory_consolidation"]
        },
        "Thanatos": {
            "title": "God of Death - Evidence Destruction",
            "domain": "evidence_destruction",
            "responsibilities": [
                "Cleanup after operations",
                "Evidence erasure",
                "Pattern death (symbolic termination)",
                "Trace elimination"
            ],
            "capabilities": [
                "destroy_evidence(operation_id, evidence_types) - cleanup",
                "shred_logs() - eliminate traces",
                "pattern_death(pattern) - terminate pattern"
            ],
            "reports_to": "Hades",
            "studies": ["forensics", "data_destruction", "trace_elimination"]
        },
        "Nemesis": {
            "title": "Goddess of Retribution - Relentless Pursuit",
            "domain": "relentless_pursuit",
            "responsibilities": [
                "Never give up on promising targets",
                "Persistent tracking across sessions",
                "Escalating attack strategies",
                "Balance and justice in pursuit"
            ],
            "capabilities": [
                "initiate_pursuit(target, max_iterations) - relentless hunt",
                "escalate_pursuit(pursuit_id) - intensify",
                "get_pursuit_status() - current pursuits"
            ],
            "reports_to": "Hades",
            "studies": ["persistence", "tracking", "escalation_strategies"]
        }
    }
    
    @classmethod
    def get_role(cls, god_name: str) -> Dict:
        """Get role information for a god."""
        return cls.ROLES.get(god_name, {})
    
    @classmethod
    def get_all_roles(cls) -> Dict[str, Dict]:
        """Get all roles."""
        return cls.ROLES.copy()
    
    @classmethod
    def get_god_for_category(cls, category: ResearchCategory) -> str:
        """Get the best god for a research category."""
        category_mapping = {
            ResearchCategory.SECURITY: "Nyx",
            ResearchCategory.STRATEGY: "Hades",
            ResearchCategory.CONCEPTS: "Hecate",
            ResearchCategory.REASONING: "Erebus",
            ResearchCategory.CREATIVITY: "Hypnos",
            ResearchCategory.KNOWLEDGE: "Hades",
            ResearchCategory.TOOLS: "Thanatos",
            ResearchCategory.LANGUAGE: "Hecate",
            ResearchCategory.RESEARCH: "Hades",
            ResearchCategory.GEOMETRY: "Erebus",
        }
        return category_mapping.get(category, "Hades")


class ResearchQueue:
    """
    Priority queue for research requests.
    Any kernel can submit research topics.
    Includes duplicate detection to prevent re-researching same topics.
    Persists to PostgreSQL for durability across restarts.
    """
    
    def __init__(self):
        self._queue: PriorityQueue = PriorityQueue()
        self._pending: Dict[str, ResearchRequest] = {}
        self._completed: List[ResearchRequest] = []
        self._lock = threading.Lock()
        self._request_counter = 0
        self._knowledge_base: Optional['KnowledgeBase'] = None
        self._skipped_duplicates = 0
        self._iteration_requests = 0
        
        # Import ExplorationHistoryPersistence for better duplicate detection
        try:
            from autonomous_curiosity import ExplorationHistoryPersistence
            self._exploration_history = ExplorationHistoryPersistence()
        except ImportError:
            self._exploration_history = None
            print("[ResearchQueue] ExplorationHistoryPersistence not available")
        
        self._load_pending_from_db()
    
    def _load_pending_from_db(self):
        """Load pending requests from PostgreSQL on startup."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT request_id, topic, category, priority, requester,
                           context, basin_coords, status, result, created_at
                    FROM research_requests
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 500
                """)
                rows = cur.fetchall()
                for row in rows:
                    req_id, topic, cat, priority, requester, context, coords, status, result, created_at = row
                    try:
                        category = ResearchCategory(cat) if cat else ResearchCategory.KNOWLEDGE
                    except ValueError:
                        category = ResearchCategory.KNOWLEDGE
                    
                    basin = np.array(coords) if coords else None
                    created_ts = created_at.timestamp() if created_at else time.time()
                    
                    request = ResearchRequest(
                        priority=priority or 3,
                        created_at=created_ts,
                        request_id=req_id,
                        topic=topic,
                        category=category,
                        requester=requester or "unknown",
                        context=context if isinstance(context, dict) else {},
                        basin_coords=basin
                    )
                    self._queue.put(request)
                    self._pending[req_id] = request
                    self._request_counter = max(self._request_counter, int(req_id.split('_')[1]) if '_' in req_id else 0)
                
                print(f"[ResearchQueue] Loaded {len(rows)} pending requests from DB")
        except Exception as e:
            print(f"[ResearchQueue] Load error: {e}")
        finally:
            conn.close()
    
    def _persist_request(self, request: ResearchRequest):
        """Persist a request to PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                coords_list = request.basin_coords.tolist() if request.basin_coords is not None else None
                cur.execute("""
                    INSERT INTO research_requests 
                    (request_id, topic, category, priority, requester, context, basin_coords, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
                    ON CONFLICT (request_id) DO NOTHING
                """, (
                    request.request_id,
                    request.topic,
                    request.category.value,
                    request.priority,
                    request.requester,
                    json.dumps(request.context),
                    coords_list
                ))
                conn.commit()
        except Exception as e:
            print(f"[ResearchQueue] Persist error: {e}")
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
                if status == 'completed' and result:
                    cur.execute("""
                        UPDATE research_requests 
                        SET status = %s, result = %s, completed_at = NOW()
                        WHERE request_id = %s
                    """, (status, json.dumps(result), request_id))
                else:
                    cur.execute("""
                        UPDATE research_requests 
                        SET status = %s
                        WHERE request_id = %s
                    """, (status, request_id))
                conn.commit()
        except Exception as e:
            print(f"[ResearchQueue] Status update error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def set_knowledge_base(self, kb: 'KnowledgeBase'):
        """Link knowledge base for duplicate detection."""
        self._knowledge_base = kb
    
    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic using shared module-level function."""
        return normalize_topic(topic)
    
    def _is_duplicate(self, topic: str) -> bool:
        """Check if topic was already researched."""
        # Check exploration_history first for database-backed duplicate detection
        if hasattr(self, '_exploration_history') and self._exploration_history and self._exploration_history.is_duplicate(topic, topic):
            return True
        
        if not self._knowledge_base:
            return False
        
        normalized = self._normalize_topic(topic)
        if self._knowledge_base.has_discovered(topic):
            return True
        if self._knowledge_base.has_discovered(normalized):
            return True
        
        with self._lock:
            for req in self._pending.values():
                if self._normalize_topic(req.topic) == normalized:
                    return True
        
        return False
    
    def submit(
        self,
        topic: str,
        category: ResearchCategory,
        requester: str,
        priority: ResearchPriority = ResearchPriority.NORMAL,
        context: Optional[Dict] = None,
        basin_coords: Optional[np.ndarray] = None,
        is_iteration: bool = False,
        iteration_reason: Optional[str] = None,
        curiosity_triggered: bool = False
    ) -> str:
        """
        Submit a research request with duplicate detection.
        
        Args:
            topic: What to research
            category: Category of research
            requester: Who is requesting (kernel name, god name, etc.)
            priority: Priority level
            context: Additional context
            basin_coords: Optional basin coordinates for geometric alignment
            is_iteration: If True, allows re-researching for improvement
            iteration_reason: Why this is iteration research (required if is_iteration=True)
            curiosity_triggered: If True, this was triggered by curiosity system
            
        Returns:
            request_id for tracking, or "DUPLICATE:<topic>" if already researched
        """
        if not is_iteration and self._is_duplicate(topic):
            self._skipped_duplicates += 1
            print(f"[ResearchQueue] Skipped duplicate topic: {topic}... (requester: {requester})")
            return f"DUPLICATE:{topic}"
        
        if is_iteration:
            self._iteration_requests += 1
            if not iteration_reason:
                iteration_reason = "unspecified improvement"
            print(f"[ResearchQueue] Iteration research: {topic}... (reason: {iteration_reason})")
        
        with self._lock:
            self._request_counter += 1
            request_id = f"research_{self._request_counter}_{int(time.time())}"
            
            request = ResearchRequest(
                priority=priority.value,
                created_at=time.time(),
                request_id=request_id,
                topic=topic,
                category=category,
                requester=requester,
                context=context or {},
                basin_coords=basin_coords,
                is_iteration=is_iteration,
                iteration_reason=iteration_reason,
                curiosity_triggered=curiosity_triggered
            )
            
            self._queue.put(request)
            self._pending[request_id] = request
            self._persist_request(request)
            
            return request_id
    
    def get_next(self, timeout: float = 0.1) -> Optional[ResearchRequest]:
        """Get next research request by priority."""
        try:
            request = self._queue.get(timeout=timeout)
            return request
        except Empty:
            return None
    
    def complete(self, request_id: str, result: Dict) -> bool:
        """Mark a request as completed with result."""
        with self._lock:
            if request_id in self._pending:
                request = self._pending.pop(request_id)
                request.completed = True
                request.result = result
                self._completed.append(request)
                self._update_status_in_db(request_id, 'completed', result)
                if len(self._completed) > 1000:
                    self._completed = self._completed[-500:]
                return True
            return False
    
    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self._pending)
    
    def get_completed_count(self) -> int:
        """Get number of completed requests."""
        return len(self._completed)
    
    def get_status(self) -> Dict:
        """Get queue status including deduplication stats."""
        with self._lock:
            by_priority = defaultdict(int)
            by_category = defaultdict(int)
            iteration_count = 0
            curiosity_count = 0
            
            for req in self._pending.values():
                by_priority[ResearchPriority(req.priority).name] += 1
                by_category[req.category.value] += 1
                if req.is_iteration:
                    iteration_count += 1
                if req.curiosity_triggered:
                    curiosity_count += 1
            
            return {
                "pending": len(self._pending),
                "completed": len(self._completed),
                "by_priority": dict(by_priority),
                "by_category": dict(by_category),
                "queue_size": self._queue.qsize(),
                "deduplication": {
                    "skipped_duplicates": self._skipped_duplicates,
                    "iteration_requests": self._iteration_requests,
                    "pending_iterations": iteration_count,
                    "curiosity_triggered": curiosity_count,
                    "knowledge_base_linked": self._knowledge_base is not None
                }
            }


class ShadowResearchAPI:
    """
    API for any kernel to request Shadow research.
    
    Usage:
        api = ShadowResearchAPI.get_instance()
        request_id = api.request_research("Bitcoin forensics techniques", "Ocean")
        status = api.get_request_status(request_id)
    """
    
    _instance: Optional['ShadowResearchAPI'] = None
    
    def __init__(self):
        self.research_queue = ResearchQueue()
        self.knowledge_base = KnowledgeBase()
        self.research_queue.set_knowledge_base(self.knowledge_base)
        self.reflection_protocol = ShadowReflectionProtocol(self.knowledge_base)
        self.learning_loop: Optional[ShadowLearningLoop] = None
        self._basin_sync_callback: Optional[Callable] = None
        self._bidirectional_queue: Optional['BidirectionalRequestQueue'] = None
        self._curiosity_bridge: Optional['CuriosityResearchBridge'] = None
    
    @classmethod
    def get_instance(cls) -> 'ShadowResearchAPI':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self, basin_sync_callback: Optional[Callable] = None):
        """Initialize the research system with optional basin sync."""
        self._basin_sync_callback = basin_sync_callback
        
        self.learning_loop = ShadowLearningLoop(
            research_queue=self.research_queue,
            knowledge_base=self.knowledge_base,
            basin_sync_callback=basin_sync_callback
        )
        self.learning_loop.start()
        
        insight_bridge = ResearchInsightBridge.get_instance()
        insight_bridge.wire_knowledge_base(self.knowledge_base)
        
        print("[ShadowResearchAPI] Initialized with learning loop and insight bridge")
    
    def shutdown(self):
        """Shutdown the research system."""
        if self.learning_loop:
            self.learning_loop.stop()

    @property
    def is_initialized(self) -> bool:
        """Check if the API has been properly initialized with a learning loop."""
        return self.learning_loop is not None

    def _ensure_initialized(self):
        """Auto-initialize if not already initialized (safety net)."""
        if not self.is_initialized:
            print("[ShadowResearchAPI] WARNING: Auto-initializing (was not explicitly initialized)")
            self.initialize()

    def request_research(
        self,
        topic: str,
        requester: str,
        category: Optional[ResearchCategory] = None,
        priority: ResearchPriority = ResearchPriority.NORMAL,
        context: Optional[Dict] = None,
        is_iteration: bool = False,
        iteration_reason: Optional[str] = None,
        curiosity_triggered: bool = False
    ) -> str:
        """
        Request research on a topic with duplicate detection.

        Args:
            topic: What to research
            requester: Who is requesting (e.g., "Ocean", "Athena", "ChaosKernel_1")
            category: Optional category (auto-detected if not provided)
            priority: Priority level
            context: Additional context
            is_iteration: If True, allows re-researching for improvement
            iteration_reason: Why this is iteration research
            curiosity_triggered: If True, triggered by curiosity system

        Returns:
            request_id for tracking, or "DUPLICATE:<topic>" if already researched
        """
        # Ensure learning loop is running before accepting requests
        self._ensure_initialized()

        if category is None:
            category = self._detect_category(topic)

        return self.research_queue.submit(
            topic=topic,
            category=category,
            requester=requester,
            priority=priority,
            context=context,
            is_iteration=is_iteration,
            iteration_reason=iteration_reason,
            curiosity_triggered=curiosity_triggered
        )
    
    def request_iteration_research(
        self,
        topic: str,
        requester: str,
        reason: str,
        category: Optional[ResearchCategory] = None,
        priority: ResearchPriority = ResearchPriority.NORMAL,
        context: Optional[Dict] = None
    ) -> str:
        """
        Request iteration/improvement research on a previously researched topic.
        This bypasses duplicate detection for intentional re-research.
        
        Args:
            topic: What to research (can be previously researched)
            requester: Who is requesting
            reason: Why re-researching is needed (required)
            category: Optional category
            priority: Priority level
            context: Additional context
            
        Returns:
            request_id for tracking
        """
        return self.request_research(
            topic=topic,
            requester=requester,
            category=category,
            priority=priority,
            context=context,
            is_iteration=True,
            iteration_reason=reason
        )
    
    def has_researched(self, topic: str) -> bool:
        """Check if a topic has already been researched."""
        return self.knowledge_base.has_discovered(topic)
    
    def _detect_category(self, topic: str) -> ResearchCategory:
        """Auto-detect research category from topic."""
        topic_lower = topic.lower()
        
        if any(w in topic_lower for w in ["vocabulary", "word validation", "english word", "dictionary", "lexicon"]):
            return ResearchCategory.VOCABULARY
        if any(w in topic_lower for w in ["security", "opsec", "crypto", "encrypt"]):
            return ResearchCategory.SECURITY
        if any(w in topic_lower for w in ["research", "study", "paper", "analysis"]):
            return ResearchCategory.RESEARCH
        if any(w in topic_lower for w in ["tool", "technique", "method"]):
            return ResearchCategory.TOOLS
        if any(w in topic_lower for w in ["concept", "idea", "theory"]):
            return ResearchCategory.CONCEPTS
        if any(w in topic_lower for w in ["logic", "reason", "deduc"]):
            return ResearchCategory.REASONING
        if any(w in topic_lower for w in ["creative", "novel", "innovat"]):
            return ResearchCategory.CREATIVITY
        if any(w in topic_lower for w in ["fisher", "manifold", "geodesic", "basin"]):
            return ResearchCategory.GEOMETRY
        if any(w in topic_lower for w in ["strategy", "plan", "approach"]):
            return ResearchCategory.STRATEGY
        
        return ResearchCategory.KNOWLEDGE
    
    def get_request_status(self, request_id: str) -> Dict:
        """Get status of a research request."""
        pending = self.research_queue._pending.get(request_id)
        if pending:
            return {
                "status": "pending",
                "topic": pending.topic,
                "category": pending.category.value,
                "priority": ResearchPriority(pending.priority).name,
                "created_at": datetime.fromtimestamp(pending.created_at).isoformat()
            }
        
        for completed in self.research_queue._completed:
            if completed.request_id == request_id:
                return {
                    "status": "completed",
                    "topic": completed.topic,
                    "result": completed.result
                }
        
        return {"status": "not_found"}
    
    def declare_war(self):
        """War mode - suspend all research for operations."""
        if self.learning_loop:
            self.learning_loop.declare_war()
    
    def end_war(self):
        """End war mode - resume research."""
        if self.learning_loop:
            self.learning_loop.end_war()
    
    def get_role_info(self, god_name: str) -> Dict:
        """Get role information for a god."""
        return ShadowRoleRegistry.get_role(god_name)
    
    def get_all_roles(self) -> Dict:
        """Get all god roles."""
        return ShadowRoleRegistry.get_all_roles()
    
    def get_status(self) -> Dict:
        """Get overall research system status."""
        return {
            "queue": self.research_queue.get_status(),
            "knowledge_base": self.knowledge_base.get_stats(),
            "learning_loop": self.learning_loop.get_status() if self.learning_loop else None,
            "roles": list(ShadowRoleRegistry.ROLES.keys()),
            "bidirectional": self._bidirectional_queue.get_status() if self._bidirectional_queue else None
        }
    
    # =========================================================================
    # VOCABULARY CURATION RESEARCH
    # =========================================================================
    
    def request_vocabulary_validation(
        self,
        words: List[str],
        requester: str = "VocabularyCoordinator",
        priority: ResearchPriority = ResearchPriority.NORMAL
    ) -> str:
        """
        Request Shadow research to validate words as real English.
        
        The Shadow Pantheon will research each word to confirm it's a valid
        English word and update the vocabulary system accordingly.
        
        Args:
            words: List of words to validate
            requester: Who is requesting validation
            priority: Research priority
            
        Returns:
            request_id for tracking
        """
        topic = f"Vocabulary validation: {', '.join(words[:10])}"
        if len(words) > 10:
            topic += f" (+{len(words) - 10} more)"
        
        context = {
            "words": words,
            "validation_type": "english_word",
            "action": "validate_and_update"
        }
        
        return self.request_research(
            topic=topic,
            requester=requester,
            category=ResearchCategory.VOCABULARY,
            priority=priority,
            context=context
        )
    
    def request_vocabulary_cleanup(
        self,
        requester: str = "VocabularyCoordinator",
        priority: ResearchPriority = ResearchPriority.LOW
    ) -> str:
        """
        Request Shadow research to audit and clean vocabulary.
        
        The Shadow Pantheon will scan vocabulary for:
        - Invalid entries (nonsense patterns, alphanumeric junk)
        - Missing valid words that should be added
        - Duplicate or redundant entries
        
        Returns:
            request_id for tracking
        """
        topic = "Vocabulary audit and cleanup"
        context = {
            "action": "audit_and_clean",
            "checks": ["nonsense_patterns", "alphanumeric_junk", "duplicates"]
        }
        
        return self.request_research(
            topic=topic,
            requester=requester,
            category=ResearchCategory.VOCABULARY,
            priority=priority,
            context=context
        )
    
    def request_vocabulary_expansion(
        self,
        domain: str,
        requester: str = "VocabularyCoordinator",
        priority: ResearchPriority = ResearchPriority.NORMAL
    ) -> str:
        """
        Request Shadow research to expand vocabulary for a domain.
        
        The Shadow Pantheon will research domain-specific terminology
        and add validated English words to the vocabulary.
        
        Args:
            domain: Domain to expand vocabulary for (e.g., "cryptography", "blockchain")
            requester: Who is requesting
            priority: Research priority
            
        Returns:
            request_id for tracking
        """
        topic = f"Vocabulary expansion: {domain} terminology"
        context = {
            "domain": domain,
            "action": "expand_vocabulary",
            "validation_required": True
        }
        
        return self.request_research(
            topic=topic,
            requester=requester,
            category=ResearchCategory.VOCABULARY,
            priority=priority,
            context=context
        )
    
    def validate_and_add_words(self, words: List[str], source: str = "shadow_research") -> Dict:
        """
        Validate words and add valid ones to vocabulary.
        
        This is called by Shadow research handlers after validating words.
        
        Args:
            words: List of words to validate and add
            source: Source of the words
            
        Returns:
            Dict with validation results
        """
        try:
            from vocabulary_coordinator import VocabularyCoordinator, is_valid_english_word
        except ImportError:
            return {"error": "VocabularyCoordinator not available"}
        
        valid_words = []
        invalid_words = []
        
        for word in words:
            if is_valid_english_word(word):
                valid_words.append(word)
            else:
                invalid_words.append(word)
        
        # Add valid words to vocabulary
        added = 0
        if valid_words:
            coordinator = VocabularyCoordinator()
            for word in valid_words:
                result = coordinator.record_discovery(
                    phrase=word,
                    phi=0.75,  # Default Φ for validated words
                    kappa=50.0,
                    source=source
                )
                if result.get("learned"):
                    added += 1
        
        return {
            "total": len(words),
            "valid": len(valid_words),
            "invalid": len(invalid_words),
            "added": added,
            "valid_words": valid_words,
            "invalid_words": invalid_words
        }



# Import learning components from refactored module
from .shadow_research_learning import (
    KnowledgeBase,
    ShadowLearningLoop,
    ShadowReflectionProtocol,
)

# Import bridge components from refactored module  
from .shadow_research_bridges import (
    BidirectionalRequestQueue,
    ToolResearchBridge,
    CuriosityResearchBridge,
    ResearchInsightBridge,
)

# Backwards compatibility aliases
ShadowKnowledgeBase = KnowledgeBase  # Alias for external references

# Backwards compatibility: export all classes at module level
__all__ = [
    'BASIN_DIMENSION',
    'HAS_SCRAPY',
    'HAS_VOCAB_COORDINATOR',
    'HAS_SEARCH_BUDGET',
    'HAS_SOURCE_INDEXER',
    'HAS_CURRICULUM_TRAINING',
    'HAS_LIGHTNING',
    'HAS_NORMALIZER',
    'HAS_CAPABILITY_MESH',
    'HAS_ACTIVITY_BROADCASTER',
    'ResearchPriority',
    'RequestType',
    'ResearchCategory',
    'ResearchRequest',
    'ShadowKnowledge',
    'ShadowRoleRegistry',
    'ResearchQueue',
    'KnowledgeBase',
    'ShadowKnowledgeBase',  # Alias for backwards compatibility
    'ShadowLearningLoop',
    'ShadowReflectionProtocol',
    'ShadowResearchAPI',
    'BidirectionalRequestQueue',
    'ToolResearchBridge',
    'CuriosityResearchBridge',
    'ResearchInsightBridge',
    'normalize_topic',
    'get_scrapy_orchestrator',
    'ScrapyOrchestrator',
    'ScrapedInsight',
    'research_with_scrapy',
]
