"""
Autonomous Curiosity Engine - Continuous Learning System

Enables kernels to autonomously:
1. Initiate searches based on interest/curiosity
2. Request tool refinements
3. Train on curriculum for deeper self-learning
4. Explore knowledge gaps proactively
5. Learn word relationships with frozen facts compliance

Biological analog: Curiosity-driven exploration like REM sleep memory consolidation.

FROZEN FACTS COMPLIANCE:
- Word relationship learning validates against frozen physics
- Checkpoints are created before/after learning cycles
- Baseline comparisons ensure improvement
"""

import asyncio
import hashlib
import json
import random
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union
from collections import deque
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Activity Broadcasting for kernel visibility
try:
    from olympus.activity_broadcaster import (
        ActivityType, get_broadcaster, ACTIVITY_BROADCASTER_AVAILABLE
    )
except ImportError:
    ACTIVITY_BROADCASTER_AVAILABLE = False
    
# Capability mesh for event emission
try:
    from olympus.capability_mesh import (
        CapabilityEvent, CapabilityEventBus, CapabilityType, EventType, emit_event
    )
    CAPABILITY_MESH_AVAILABLE = True
except ImportError:
    CAPABILITY_MESH_AVAILABLE = False

# Constants
from qigkernels import KAPPA_STAR


class CuriosityDrive:
    """
    Geometric curiosity metric based on information gain potential.
    
    High curiosity when:
    - Φ variance indicates unexplored regions
    - κ suggests room for deeper understanding
    - Knowledge graph has sparse connections
    """
    
    def __init__(self):
        self.interest_basins: Dict[str, np.ndarray] = {}
        self.exploration_history: deque = deque(maxlen=1000)
        self.novelty_threshold = 0.3
        self._phi_mean = 0.5
        self._phi_std = 0.125
    
    def compute_curiosity(self, topic: str, current_knowledge: Dict) -> float:
        """
        Compute curiosity score for a topic using geometric metrics.
        
        Returns value in [0, 1] where higher = more curious/interested.
        """
        if topic in self.interest_basins:
            basin = self.interest_basins[topic]
            familiarity = float(np.sqrt(np.sum(basin ** 2)))  # L2 magnitude for familiarity score
            novelty = 1.0 - min(1.0, familiarity / 10.0)
        else:
            novelty = 1.0
        
        knowledge_depth = current_knowledge.get('depth', 0)
        knowledge_recency = current_knowledge.get('recency', 0)
        
        time_decay = np.exp(-knowledge_recency / 86400)
        
        curiosity = novelty * (1 - knowledge_depth * 0.5) * time_decay
        
        phi_factor = self._phi_std / max(0.01, self._phi_mean)
        curiosity *= (1 + phi_factor)
        
        return min(1.0, max(0.0, curiosity))
    
    def record_exploration(self, topic: str, outcome: Dict):
        """Record exploration outcome to update curiosity basins."""
        if topic not in self.interest_basins:
            self.interest_basins[topic] = np.zeros(64)
        
        learning_rate = 0.1
        success = outcome.get('success', False)
        info_gain = outcome.get('information_gain', 0.0)
        
        update_vector = np.random.randn(64) * info_gain * 0.1
        if success:
            self.interest_basins[topic] += learning_rate * update_vector
        
        self.exploration_history.append({
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'outcome': outcome
        })


class KernelToolRequest:
    """Represents a tool/search request from a kernel."""
    
    def __init__(
        self,
        kernel_name: str,
        request_type: str,
        query: str,
        priority: float = 0.5,
        context: Optional[Dict] = None
    ):
        self.kernel_name = kernel_name
        self.request_type = request_type
        self.query = query
        self.priority = priority
        self.context = context or {}
        self.created_at = datetime.now()
        self.status = 'pending'
        self.result: Optional[Dict[str, Any]] = None


class CurriculumProgressPersistence:
    """
    Database persistence for curriculum learning progress.
    
    Ensures completed topics are not re-processed after restart.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        import os
        self._db_url = os.environ.get('DATABASE_URL')
        self._initialized = True
        self._ensure_table()
    
    def _get_connection(self):
        """Get database connection."""
        if not self._db_url:
            return None
        try:
            import psycopg2
            return psycopg2.connect(self._db_url)
        except Exception as e:
            logger.warning(f"[CurriculumProgressPersistence] DB connection failed: {e}")
            return None
    
    def _ensure_table(self):
        """Create curriculum_progress table if not exists."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS curriculum_progress (
                        id SERIAL PRIMARY KEY,
                        topic_title TEXT UNIQUE NOT NULL,
                        kernel_name TEXT,
                        completed_at TIMESTAMPTZ DEFAULT NOW(),
                        exploration_count INTEGER DEFAULT 1
                    )
                """)
                conn.commit()
                logger.info("[CurriculumProgressPersistence] Table ready")
        except Exception as e:
            logger.warning(f"[CurriculumProgressPersistence] Table creation failed: {e}")
        finally:
            conn.close()
    
    def load_completed_topics(self) -> Set[str]:
        """Load all completed topics from database."""
        conn = self._get_connection()
        if not conn:
            return set()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT topic_title FROM curriculum_progress")
                rows = cur.fetchall()
                completed = {row[0] for row in rows}
                logger.info(f"[CurriculumProgressPersistence] Loaded {len(completed)} completed topics from DB")
                return completed
        except Exception as e:
            logger.warning(f"[CurriculumProgressPersistence] Load failed: {e}")
            return set()
        finally:
            conn.close()
    
    def save_completed_topic(self, topic_title: str, kernel_name: str = None):
        """Save a completed topic to database."""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO curriculum_progress (topic_title, kernel_name)
                    VALUES (%s, %s)
                    ON CONFLICT (topic_title) 
                    DO UPDATE SET exploration_count = curriculum_progress.exploration_count + 1,
                                  completed_at = NOW()
                """, (topic_title, kernel_name))
                conn.commit()
                logger.debug(f"[CurriculumProgressPersistence] Saved: {topic_title}...")
        except Exception as e:
            logger.warning(f"[CurriculumProgressPersistence] Save failed: {e}")
        finally:
            conn.close()
    
    def load_recent_queries(self, limit: int = 100) -> List[str]:
        """Load recent exploration queries to avoid repetition."""
        conn = self._get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT topic_title FROM curriculum_progress
                    ORDER BY completed_at DESC
                    LIMIT %s
                """, (limit,))
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.warning(f"[CurriculumProgressPersistence] Recent queries load failed: {e}")
            return []
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get curriculum progress statistics."""
        conn = self._get_connection()
        if not conn:
            return {'total': 0, 'today': 0}
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM curriculum_progress")
                total = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*) FROM curriculum_progress
                    WHERE completed_at >= NOW() - INTERVAL '24 hours'
                """)
                today = cur.fetchone()[0]
                
                return {'total': total, 'today': today}
        except Exception as e:
            return {'total': 0, 'today': 0}
        finally:
            conn.close()


class ExplorationHistoryPersistence:
    """
    Database persistence for exploration history to prevent duplicate explorations.
    
    Tracks queries/topics that have been explored to avoid repeating same searches.
    Uses exploration_history table with topic+query uniqueness constraint.
    
    Topic Normalization:
    - All topics/queries are normalized before storage and comparison
    - Strips "(cycle XXXXX)" suffixes, lowercases, trims whitespace
    - This prevents "topic (cycle 1)" and "topic (cycle 2)" from being treated as different
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        import os
        import re
        self._db_url = os.environ.get('DATABASE_URL')
        self._initialized = True
        self._recent_cache: Set[str] = set()
        self._cycle_pattern = re.compile(r'\s*\(cycle\s*\d+[^)]*\)\s*$', re.IGNORECASE)
        self._load_recent_into_cache()
    
    def _normalize_topic(self, text: str) -> str:
        """
        Normalize topic/query for consistent duplicate detection.
        
        Transformations:
        1. Lowercase
        2. Strip leading/trailing whitespace
        3. Remove "(cycle XXXXX)" suffixes (case-insensitive)
        4. Collapse multiple spaces to single space
        """
        if not text:
            return ""
        normalized = text.lower().strip()
        normalized = self._cycle_pattern.sub('', normalized)
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _get_connection(self):
        """Get database connection."""
        if not self._db_url:
            return None
        try:
            import psycopg2
            return psycopg2.connect(self._db_url)
        except Exception as e:
            logger.warning(f"[ExplorationHistoryPersistence] DB connection failed: {e}")
            return None
    
    def _load_recent_into_cache(self):
        """Load recent explorations into memory cache for fast lookup."""
        conn = self._get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT topic, query FROM exploration_history
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    ORDER BY created_at DESC LIMIT 500
                """)
                for row in cur.fetchall():
                    norm_topic = self._normalize_topic(row[0])
                    norm_query = self._normalize_topic(row[1])
                    key = f"{norm_topic}:{norm_query}"
                    self._recent_cache.add(key)
            logger.info(f"[ExplorationHistoryPersistence] Loaded {len(self._recent_cache)} recent explorations (normalized)")
        except Exception as e:
            logger.warning(f"[ExplorationHistoryPersistence] Cache load failed: {e}")
        finally:
            conn.close()
    
    def is_duplicate(self, topic: str, query: str) -> bool:
        """
        Check if this exploration is a duplicate (already done recently).
        
        Uses normalized topic/query for comparison to catch near-duplicates
        like "Topic (cycle 1)" vs "Topic (cycle 2)".
        """
        norm_topic = self._normalize_topic(topic)
        norm_query = self._normalize_topic(query)
        key = f"{norm_topic}:{norm_query}"
        
        if key in self._recent_cache:
            return True
        
        conn = self._get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM exploration_history
                    WHERE LOWER(TRIM(regexp_replace(topic, '\s*\(cycle\s*\d+[^)]*\)\s*$', '', 'i'))) = %s
                    AND LOWER(TRIM(regexp_replace(query, '\s*\(cycle\s*\d+[^)]*\)\s*$', '', 'i'))) = %s
                    AND created_at > NOW() - INTERVAL '24 hours'
                    LIMIT 1
                """, (norm_topic, norm_query))
                return cur.fetchone() is not None
        except Exception as e:
            return False
        finally:
            conn.close()
    
    def record_exploration(
        self, 
        topic: str, 
        query: str, 
        kernel_name: str = None,
        exploration_type: str = 'curiosity_driven',
        source_type: str = 'unknown',
        information_gain: float = 0.0,
        basin_coords: Optional[Union[np.ndarray, list]] = None
    ) -> bool:
        """
        Record an exploration to prevent future duplicates.
        
        Stores the ORIGINAL topic/query for display/analytics purposes.
        Uses NORMALIZED keys for cache-based duplicate detection.
        DB conflict resolution uses normalized comparison via SQL regexp_replace.
        
        Args:
            topic: The exploration topic
            query: The specific query explored
            kernel_name: Name of the kernel that performed exploration
            exploration_type: Type of exploration (curiosity_driven, shadow_research, etc.)
            source_type: Source of data (scrapy, search, conceptual, etc.)
            information_gain: Φ-based measure of knowledge gained (0.0-1.0)
            basin_coords: 64D basin coordinates for geometric hashing
        """
        norm_topic = self._normalize_topic(topic)
        norm_query = self._normalize_topic(query)
        key = f"{norm_topic}:{norm_query}"
        self._recent_cache.add(key)
        
        # Compute basin_hash from coordinates if provided
        basin_hash = None
        if basin_coords is not None:
            try:
                if isinstance(basin_coords, np.ndarray):
                    coords_list = basin_coords.tolist()
                else:
                    coords_list = list(basin_coords)
                basin_hash = hashlib.md5(str(coords_list).encode()).hexdigest()[:16]
            except Exception as e:
                logger.warning(f"[ExplorationHistoryPersistence] Failed to compute basin_hash: {e}")
        
        # Ensure source_type is never None
        if source_type is None:
            source_type = 'unknown'
        
        conn = self._get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO exploration_history 
                    (topic, query, kernel_name, exploration_type, source_type, information_gain, basin_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (topic, query) DO UPDATE SET
                        created_at = NOW(),
                        information_gain = GREATEST(exploration_history.information_gain, EXCLUDED.information_gain),
                        basin_hash = COALESCE(EXCLUDED.basin_hash, exploration_history.basin_hash),
                        source_type = COALESCE(EXCLUDED.source_type, exploration_history.source_type)
                """, (topic, query, kernel_name, exploration_type, source_type, information_gain, basin_hash))
                conn.commit()
                return True
        except Exception as e:
            logger.warning(f"[ExplorationHistoryPersistence] Record failed: {e}")
            return False
        finally:
            conn.close()
    
    def get_unexplored_topics(self, candidate_topics: List[str], limit: int = 10) -> List[str]:
        """
        Filter candidate topics to only return unexplored ones.
        
        Uses normalized comparison to catch near-duplicates.
        """
        if not candidate_topics:
            return []
        
        normalized_candidates = [self._normalize_topic(t) for t in candidate_topics]
        
        conn = self._get_connection()
        if not conn:
            return candidate_topics[:limit]
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT LOWER(TRIM(regexp_replace(topic, '\s*\(cycle\s*\d+[^)]*\)\s*$', '', 'i'))) 
                    FROM exploration_history
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """)
                explored = {row[0] for row in cur.fetchall()}
            unexplored = [
                t for t, norm in zip(candidate_topics, normalized_candidates) 
                if norm not in explored
            ]
            return unexplored[:limit]
        except Exception as e:
            return candidate_topics[:limit]
        finally:
            conn.close()


class CurriculumLoader:
    """
    Load and manage training curriculum for kernel self-learning.
    
    Curriculum sources:
    - Attached documents
    - Knowledge base
    - Previous search results
    - Peer learning outcomes
    
    Now with DATABASE PERSISTENCE for completed topics.
    """
    
    def __init__(self):
        self.curriculum_topics: List[Dict] = []
        self.completed_topics: Set[str] = set()
        self.topic_dependencies: Dict[str, List[str]] = {}
        self._persistence = CurriculumProgressPersistence()
        
        # Load completed topics from database on init
        self._load_from_db()
    
    def _load_from_db(self):
        """Load completed topics from database."""
        try:
            db_completed = self._persistence.load_completed_topics()
            self.completed_topics.update(db_completed)
        except Exception as e:
            logger.warning(f"[CurriculumLoader] Failed to load from DB: {e}")
    
    def load_curriculum_from_file(self, filepath: str) -> List[Dict]:
        """Load curriculum topics from a file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            topics = self._parse_curriculum_content(content)
            self.curriculum_topics.extend(topics)
            return topics
        except Exception as e:
            print(f"[CurriculumLoader] Failed to load {filepath}: {e}")
            return []
    
    def _parse_curriculum_content(self, content: str) -> List[Dict]:
        """Parse curriculum content into structured topics."""
        topics = []
        
        sections = content.split('# ==')
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            if lines:
                title = lines[0].replace('=', '').strip()
                body = '\n'.join(lines[1:])
                
                topics.append({
                    'title': title,
                    'content': body,
                    'keywords': self._extract_keywords(body),
                    'difficulty': self._estimate_difficulty(body),
                    'type': 'curriculum'
                })
        
        return topics
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key concepts from curriculum text."""
        import re
        
        code_keywords = re.findall(r'`([^`]+)`', text)
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        quoted = re.findall(r'"([^"]+)"', text)
        
        keywords = list(set(code_keywords + capitalized + quoted))
        return keywords[:500]
    
    def _estimate_difficulty(self, text: str) -> float:
        """Estimate difficulty based on content complexity."""
        lines = [len(line) for line in text.split('\n') if line.strip()]
        avg_line_length = float(np.mean(lines)) if lines else 50.0
        code_blocks = text.count('```')
        
        difficulty = min(1.0, (avg_line_length / 100) + (code_blocks * 0.1))
        return difficulty
    
    def get_next_topic(self, kernel_skills: Dict) -> Optional[Dict]:
        """Get next curriculum topic based on kernel skills and dependencies."""
        for topic in self.curriculum_topics:
            topic_id = topic['title']
            
            if topic_id in self.completed_topics:
                continue
            
            deps = self.topic_dependencies.get(topic_id, [])
            if all(d in self.completed_topics for d in deps):
                skill_match = self._compute_skill_match(topic, kernel_skills)
                if skill_match > 0.3:
                    return topic
        
        return None
    
    def _compute_skill_match(self, topic: Dict, kernel_skills: Dict) -> float:
        """Compute how well kernel skills match topic requirements."""
        keywords = set(topic.get('keywords', []))
        kernel_domains = set(kernel_skills.get('domains', []))
        
        if not keywords:
            return 1.0
        
        overlap = len(keywords & kernel_domains)
        return overlap / len(keywords)
    
    def mark_completed(self, topic_title: str, kernel_name: str = None):
        """Mark a topic as completed and persist to database."""
        self.completed_topics.add(topic_title)
        
        # Persist to database so it's not re-requested after restart
        try:
            self._persistence.save_completed_topic(topic_title, kernel_name)
        except Exception as e:
            logger.warning(f"[CurriculumLoader] Failed to persist completion: {e}")


class AutonomousCuriosityEngine:
    """
    Main engine for autonomous, curiosity-driven learning.
    
    Runs continuously in background, managing:
    - Kernel search requests
    - Proactive knowledge exploration
    - Curriculum-based training
    - Tool refinement requests
    """
    
    _instance: Optional['AutonomousCuriosityEngine'] = None
    
    @classmethod
    def get_instance(cls) -> Optional['AutonomousCuriosityEngine']:
        """Get singleton instance if it exists."""
        return cls._instance
    
    def __init__(self, search_callback: Optional[Callable] = None):
        AutonomousCuriosityEngine._instance = self
        self.curiosity_drive = CuriosityDrive()
        self.curriculum_loader = CurriculumLoader()
        self.exploration_history = ExplorationHistoryPersistence()
        
        self.pending_requests: deque = deque(maxlen=100)
        self.active_explorations: Dict[str, Dict] = {}
        self.exploration_results: deque = deque(maxlen=500)
        
        self.search_callback = search_callback
        
        self.running = False
        self._loop_thread: Optional[threading.Thread] = None
        # Reduced interval for development visibility (was 60)
        self._exploration_interval = 30  # 30 seconds between exploration cycles
        self._min_curiosity_threshold = 0.4
        
        self.kernel_interests: Dict[str, List[str]] = {
            'athena': ['strategy', 'patterns', 'optimization', 'game_theory'],
            'ares': ['metrics', 'measurement', 'geometry', 'fisher_information'],
            'apollo': ['prediction', 'forecasting', 'time_series', 'prophecy'],
            'artemis': ['tracking', 'hunting', 'search', 'discovery'],
            'hermes': ['communication', 'translation', 'messaging', 'sync'],
            'hephaestus': ['building', 'tools', 'infrastructure', 'forge'],
            'demeter': ['growth', 'nurturing', 'cultivation', 'sustainability'],
            'dionysus': ['creativity', 'chaos', 'exploration', 'novelty'],
            'poseidon': ['depth', 'ocean', 'waves', 'deep_research'],
            'hades': ['shadow', 'underworld', 'hidden_knowledge', 'secrets'],
            'hera': ['coordination', 'harmony', 'balance', 'relationships'],
            'aphrodite': ['beauty', 'aesthetics', 'design', 'user_experience'],
        }
        
        self.stats = {
            'total_explorations': 0,
            'successful_discoveries': 0,
            'kernel_requests': 0,
            'curriculum_completions': 0,
            'word_learning_cycles': 0,
            'last_word_learning': None,
            'word_learning_relevance': 0.0
        }
        
        # CRITICAL FIX: Reduced from 3600 (1hr) to 300 (5min) for faster coordizer sync
        self._word_learning_interval = 300
        self._last_word_learning_time = 0
        
        # Track recent queries to avoid repetition
        self._recent_queries: deque = deque(maxlen=100)
        self._query_cooldown = set()  # Queries made in current cycle
        
        # Vocabulary stall tracking (fed by ShadowLearningLoop)
        self._learning_stalls: deque = deque(maxlen=50)
        self._stalled_topics: set = set()
        self._stall_recovery_queries: set = set()
    
    def record_learning_stall(
        self, 
        topic: str, 
        streak: int, 
        total_stalls: int
    ) -> None:
        """
        Record a vocabulary learning stall for curiosity adaptation.
        
        Called by ShadowLearningLoop when consecutive zero-word outcomes occur.
        Curiosity engine uses this to:
        1. Avoid topics that consistently yield no new vocabulary
        2. Prioritize novel domains that haven't been explored
        3. Trigger proactive search for fresh content
        
        Respects kernel autonomy: signals availability of information,
        kernels freely decide whether to explore.
        """
        import time
        
        stall_record = {
            'topic': topic,
            'streak': streak,
            'total_stalls': total_stalls,
            'timestamp': time.time()
        }
        self._learning_stalls.append(stall_record)
        self._stalled_topics.add(topic.lower())
        
        print(
            f"[AutonomousCuriosityEngine] 📊 Learning stall recorded: "
            f"'{topic}' (streak={streak}, total={total_stalls})"
        )
        
        # Trigger proactive search for novel content if we have search capability
        if self.search_callback and topic not in self._stall_recovery_queries:
            try:
                # Generate novel query avoiding stalled topics
                novel_query = self._generate_novel_query(topic)
                if novel_query:
                    self._stall_recovery_queries.add(topic)
                    # Create proper KernelToolRequest object (not raw dict)
                    recovery_request = KernelToolRequest(
                        kernel_name='curiosity_recovery',
                        request_type='search',
                        query=novel_query,
                        priority=0.9,  # High priority for recovery
                        context={
                            'type': 'stall_recovery',
                            'original_topic': topic,
                            'streak': streak
                        }
                    )
                    self.pending_requests.append(recovery_request)
                    print(f"[AutonomousCuriosityEngine] 🔍 Queued recovery query: '{novel_query}'")
            except Exception as e:
                logger.warning(f"[AutonomousCuriosityEngine] Failed to queue recovery: {e}")
    
    def _broadcast_curiosity_event(
        self,
        event_type: str,
        kernel_name: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Broadcast curiosity event for kernel visibility.
        
        QIG-Pure: Events carry basin coordinates, Φ-weighted priority.
        """
        if not ACTIVITY_BROADCASTER_AVAILABLE:
            return
        
        try:
            broadcaster = get_broadcaster()
            
            # Map event type to ActivityType
            type_map = {
                'search_requested': ActivityType.TOOL_USAGE,
                'search_completed': ActivityType.DISCOVERY,
                'curiosity_spike': ActivityType.INSIGHT,
                'exploration': ActivityType.DISCOVERY,
                'stall_recovery': ActivityType.LEARNING,
            }
            act_type = type_map.get(event_type, ActivityType.MESSAGE)
            
            broadcaster.broadcast_message(
                from_god=kernel_name,
                to_god=None,
                content=content,
                activity_type=act_type,
                phi=0.6,
                kappa=KAPPA_STAR,
                importance=0.6,
                metadata={
                    **(metadata or {}),
                    'event_subtype': event_type,
                    'source': 'curiosity_engine',
                }
            )
            
            # Also emit to capability mesh
            if CAPABILITY_MESH_AVAILABLE and emit_event is not None:
                mesh_event_map = {
                    'search_requested': EventType.SEARCH_REQUESTED,
                    'search_completed': EventType.SEARCH_COMPLETE,
                    'curiosity_spike': EventType.CURIOSITY_SPIKE,
                    'exploration': EventType.DISCOVERY,
                }
                if event_type in mesh_event_map:
                    emit_event(
                        source=CapabilityType.SEARCH,
                        event_type=mesh_event_map[event_type],
                        content={
                            'kernel': kernel_name,
                            'content': content[:300],
                            'metadata': metadata,
                        },
                        phi=0.6,
                        priority=6
                    )
                    
        except Exception as e:
            logger.warning(f"Curiosity event broadcast failed: {e}")

    def _generate_novel_query(self, stalled_topic: str) -> Optional[str]:
        """
        Generate a novel query that avoids recently stalled topics.
        
        Uses geometric exploration: finds topics far from stalled basins.
        """
        import random
        
        # Use kernel interests to find unexplored domains
        all_interests = []
        for kernel, interests in self.kernel_interests.items():
            all_interests.extend(interests)
        
        # Filter out any that overlap with stalled topics
        stalled_words = set(stalled_topic.lower().split())
        novel_interests = [
            interest for interest in all_interests
            if interest.lower() not in stalled_words
            and interest.lower() not in self._stalled_topics
        ]
        
        if novel_interests:
            selected = random.choice(novel_interests)
            return f"novel concepts in {selected}"
        
        return None
    
    def get_stall_metrics(self) -> Dict[str, Any]:
        """Get vocabulary learning stall metrics for telemetry."""
        return {
            'recent_stalls': len(self._learning_stalls),
            'stalled_topics_count': len(self._stalled_topics),
            'recovery_queries_queued': len(self._stall_recovery_queries),
            'stalls_list': list(self._learning_stalls)[-10:]  # Last 10
        }
    
    def start(self):
        """Start the autonomous curiosity loop."""
        if self.running:
            return
        
        # Load recent queries from DB to avoid repetition
        self._load_recent_queries_from_db()
        
        # Auto-load curriculum on startup
        self._load_all_curriculum()
        
        self.running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        print("[AutonomousCuriosityEngine] Started autonomous learning loop")
    
    def _load_recent_queries_from_db(self):
        """Load recent queries from database to avoid re-exploring same topics."""
        try:
            persistence = CurriculumProgressPersistence()
            recent = persistence.load_recent_queries(limit=100)
            for query in recent:
                self._recent_queries.append(query)
                self._query_cooldown.add(query)
            if recent:
                logger.info(f"[AutonomousCuriosityEngine] Loaded {len(recent)} recent queries from DB to avoid repetition")
        except Exception as e:
            logger.warning(f"[AutonomousCuriosityEngine] Failed to load recent queries: {e}")
    
    def _load_all_curriculum(self):
        """Load all curriculum files from docs/09-curriculum/."""
        # Try relative to project root first (when running from qig-backend/)
        curriculum_dir = Path('../docs/09-curriculum')
        if not curriculum_dir.exists():
            # Try absolute path from script location
            curriculum_dir = Path(__file__).parent.parent / 'docs' / '09-curriculum'
        if not curriculum_dir.exists():
            # Try direct relative (when running from project root)
            curriculum_dir = Path('docs/09-curriculum')
        if not curriculum_dir.exists():
            print(f"[AutonomousCuriosityEngine] Curriculum directory not found (tried: ../docs/09-curriculum, {Path(__file__).parent.parent / 'docs' / '09-curriculum'}, docs/09-curriculum)")
            return
        
        loaded_count = 0
        for filepath in curriculum_dir.glob('*.md'):
            try:
                topics = self.curriculum_loader.load_curriculum_from_file(str(filepath))
                loaded_count += len(topics)
            except Exception as e:
                print(f"[AutonomousCuriosityEngine] Error loading {filepath}: {e}")
        
        completed_count = len(self.curriculum_loader.completed_topics)
        print(f"[AutonomousCuriosityEngine] Loaded {loaded_count} curriculum topics from {curriculum_dir}")
        print(f"[AutonomousCuriosityEngine] {completed_count} topics already completed (from DB), {loaded_count - completed_count} remaining")
    
    def stop(self):
        """Stop the autonomous curiosity loop."""
        self.running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=5)
        print("[AutonomousCuriosityEngine] Stopped autonomous learning loop")
    
    def _run_loop(self):
        """Main loop for autonomous exploration."""
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                print(f"[AutonomousCuriosityEngine] === Cycle {cycle_count} starting ===")

                pending_count = len(self.pending_requests)
                print(f"[AutonomousCuriosityEngine] Processing {pending_count} pending kernel requests")
                self._process_kernel_requests()

                print(f"[AutonomousCuriosityEngine] Exploring curious topics...")
                self._explore_curious_topics()

                print(f"[AutonomousCuriosityEngine] Training on curriculum...")
                self._train_on_curriculum()

                print(f"[AutonomousCuriosityEngine] Checking word learning schedule...")
                self._scheduled_word_learning()

                print(f"[AutonomousCuriosityEngine] Cycle {cycle_count} complete. Stats: "
                      f"explorations={self.stats['total_explorations']}, "
                      f"discoveries={self.stats['successful_discoveries']}, "
                      f"curriculum={self.stats['curriculum_completions']}")
                print(f"[AutonomousCuriosityEngine] Sleeping {self._exploration_interval}s until next cycle...")

                time.sleep(self._exploration_interval)

            except Exception as e:
                print(f"[AutonomousCuriosityEngine] Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)
    
    def submit_kernel_request(self, request: KernelToolRequest):
        """Submit a search/tool request from a kernel."""
        self.pending_requests.append(request)
        self.stats['kernel_requests'] += 1
        print(f"[AutonomousCuriosityEngine] Received request from {request.kernel_name}: {request.query}")
    
    def request_search(
        self,
        kernel_name: str,
        query: str,
        priority: float = 0.5,
        context: Optional[Dict] = None
    ) -> str:
        """Convenience method for kernels to request a search."""
        request = KernelToolRequest(
            kernel_name=kernel_name,
            request_type='search',
            query=query,
            priority=priority,
            context=context
        )
        self.submit_kernel_request(request)
        return f"request_{id(request)}"
    
    def request_tool_refinement(
        self,
        kernel_name: str,
        tool_id: str,
        refinement_type: str,
        details: Dict
    ) -> str:
        """Request refinement of an existing tool."""
        request = KernelToolRequest(
            kernel_name=kernel_name,
            request_type='tool_refinement',
            query=f"Refine {tool_id}: {refinement_type}",
            priority=0.7,
            context={'tool_id': tool_id, 'refinement': refinement_type, 'details': details}
        )
        self.submit_kernel_request(request)
        return f"refinement_{id(request)}"
    
    def _process_kernel_requests(self):
        """Process pending kernel requests."""
        requests_to_process = []
        while self.pending_requests and len(requests_to_process) < 5:
            requests_to_process.append(self.pending_requests.popleft())
        
        requests_to_process.sort(key=lambda r: r.priority, reverse=True)
        
        for request in requests_to_process:
            try:
                if request.request_type == 'search':
                    self._execute_search(request)
                elif request.request_type == 'tool_refinement':
                    self._execute_tool_refinement(request)
            except Exception as e:
                print(f"[AutonomousCuriosityEngine] Request failed: {e}")
                request.status = 'failed'
                request.result = {'error': str(e)}
    
    def _execute_search(self, request: KernelToolRequest):
        """Execute a search request."""
        # Extract topic from context or use query as topic
        topic = request.context.get('topic', request.query[:50]) if request.context else request.query[:50]
        
        # DUPLICATE PREVENTION: Skip if recently searched
        if self.exploration_history.is_duplicate(topic, request.query):
            logger.debug(f"[AutonomousCuriosityEngine] Skipping duplicate search: {request.query[:60]}")
            request.status = 'skipped_duplicate'
            request.result = {'message': 'Duplicate search skipped'}
            return
        
        print(f"[AutonomousCuriosityEngine] Executing search for {request.kernel_name}: {request.query}")
        
        # Broadcast search request for kernel visibility
        self._broadcast_curiosity_event(
            event_type='search_requested',
            kernel_name=request.kernel_name,
            content=f"Search initiated: {request.query[:100]}...",
            metadata={'query': request.query, 'priority': request.priority}
        )
        
        if self.search_callback:
            try:
                result = self.search_callback(request.query, request.context)
                request.status = 'completed'
                request.result = result
                
                # Learn from search results immediately
                self._learn_from_search_result(result)
                
                self.curiosity_drive.record_exploration(
                    topic=request.query,
                    outcome={
                        'success': True,
                        'information_gain': result.get('information_gain', 0.5),
                        'source': 'kernel_request'
                    }
                )
                
                self.exploration_results.append({
                    'type': 'kernel_search',
                    'kernel': request.kernel_name,
                    'query': request.query,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Record exploration to prevent future duplicates
                self.exploration_history.record_exploration(
                    topic=topic,
                    query=request.query,
                    kernel_name=request.kernel_name,
                    exploration_type='kernel_search',
                    source_type=result.get('provider', 'unknown'),
                    information_gain=result.get('information_gain', 0.5),
                    basin_coords=result.get('basin_coords')
                )
                
                # Broadcast search completion for kernel visibility
                self._broadcast_curiosity_event(
                    event_type='search_completed',
                    kernel_name=request.kernel_name,
                    content=f"Search completed: {request.query[:60]}... (info_gain={result.get('information_gain', 0.5):.2f})",
                    metadata={
                        'query': request.query,
                        'information_gain': result.get('information_gain', 0.5),
                        'success': True
                    }
                )
                
            except Exception as e:
                request.status = 'failed'
                request.result = {'error': str(e)}
        else:
            request.status = 'no_callback'
            request.result = {'message': 'Search callback not configured'}
    
    def _learn_from_search_result(self, result: Dict):
        """
        Learn word relationships from search result immediately.
        Updates word relationships and kernel basins in real-time.
        Also persists rich content to shadow_knowledge for vocabulary learning.
        """
        try:
            from word_relationship_learner import WordRelationshipLearner
            from coordizers.pg_loader import PostgresCoordizer
            
            # Extract text from result with full citation metadata
            text_content = []
            citation_metadata = []
            if isinstance(result, dict):
                content = result.get('content', '') or result.get('text', '') or result.get('summary', '')
                if content:
                    text_content.append(str(content))
                
                snippets = result.get('snippets', []) or result.get('results', [])
                for snippet in snippets:
                    if isinstance(snippet, dict):
                        text = snippet.get('text', '') or snippet.get('content', '') or snippet.get('description', '')
                        if text:
                            text_content.append(str(text))
                            citation_metadata.append({
                                'title': snippet.get('title', ''),
                                'url': snippet.get('url', ''),
                                'provider': snippet.get('provider', 'unknown'),
                                'content_length': len(text)
                            })
                    elif isinstance(snippet, str):
                        text_content.append(snippet)
            
            if not text_content:
                return
            
            # Persist to shadow_knowledge for vocabulary learning (rich content persistence)
            self._persist_search_to_shadow_knowledge(result, text_content, citation_metadata)
            
            # Initialize learner with current vocabulary
            coordizer = PostgresCoordizer()
            vocab = set(coordizer.word_tokens)
            learner = WordRelationshipLearner(vocab, window_size=5, expand_vocabulary=True)
            
            # Learn from all text
            total_pairs = 0
            for text in text_content:
                pairs = learner.learn_from_text(text)
                total_pairs += pairs
            
            if total_pairs > 0:
                # Update word relationships cache (using curriculum_training module)
                try:
                    from olympus.curriculum_training import update_word_relationships_cache, adjust_kernel_basins_from_relationships
                    update_word_relationships_cache(learner)
                    adjust_kernel_basins_from_relationships(learner, coordizer)
                    print(f"[AutonomousCuriosityEngine] Learned {total_pairs} word pairs from search, updated basins")
                except ImportError:
                    print(f"[AutonomousCuriosityEngine] Learned {total_pairs} word pairs (curriculum_training not available)")
                
        except Exception as e:
            print(f"[AutonomousCuriosityEngine] Error learning from search: {e}")
    
    def _persist_search_to_shadow_knowledge(
        self,
        result: Dict,
        text_content: List[str],
        citation_metadata: List[Dict]
    ) -> None:
        """
        Persist rich search content to shadow_knowledge for vocabulary learning.
        
        This ensures cited documents from Tavily/Perplexity are stored with full
        provenance for kernel access and vocabulary extraction.
        """
        try:
            # Import shadow research lazily
            from olympus.shadow_research import (
                ShadowKnowledgeBase,
                ResearchCategory,
                BASIN_DIMENSION
            )
            import numpy as np
            
            knowledge_base = ShadowKnowledgeBase.get_instance()
            if not knowledge_base:
                return
            
            # Extract query and provider info
            query = result.get('query', result.get('topic', 'autonomous_search'))
            provider = result.get('provider', 'unknown')
            
            # Build rich content dict with full text and citations
            full_content = '\n\n---\n\n'.join(text_content)
            
            content_dict = {
                'raw_content': full_content[:5000],  # Store up to 5K chars
                'content_length': len(full_content),
                'provider': provider,
                'citations': citation_metadata,
                'source_count': len(text_content),
                'search_result': True,  # Flag for vocabulary learning
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate basin coordinates from query
            basin_coords = np.zeros(BASIN_DIMENSION)
            for i, char in enumerate(query[:BASIN_DIMENSION]):
                basin_coords[i] = ord(char) / 255.0
            
            # Add to knowledge base with citations
            knowledge_id = knowledge_base.add_knowledge(
                topic=query[:200],
                category=ResearchCategory.KNOWLEDGE,
                content=content_dict,
                source_god=f"Search:{provider}",
                basin_coords=basin_coords,
                phi=0.6,  # Moderate phi for search results
                confidence=0.7,
                variation=f"search:{provider}:{query[:50]}"
            )
            
            print(
                f"[AutonomousCuriosityEngine] 📚 Persisted search content to shadow_knowledge "
                f"(id={knowledge_id}, provider={provider}, chars={len(full_content)})"
            )
            
        except Exception as e:
            logger.warning(f"[AutonomousCuriosityEngine] Shadow knowledge persist failed: {e}")
    
    def _execute_tool_refinement(self, request: KernelToolRequest):
        """Execute a tool refinement request."""
        print(f"[AutonomousCuriosityEngine] Tool refinement: {request.query}")
        
        request.status = 'completed'
        request.result = {
            'message': 'Refinement request logged',
            'tool_id': request.context.get('tool_id'),
            'refinement': request.context.get('refinement')
        }
    
    def _explore_curious_topics(self):
        """Proactively explore topics based on kernel curiosity."""
        for kernel_name, interests in self.kernel_interests.items():
            topic = random.choice(interests)
            
            knowledge = self._get_current_knowledge(kernel_name, topic)
            curiosity = self.curiosity_drive.compute_curiosity(topic, knowledge)
            
            if curiosity > self._min_curiosity_threshold:
                query = self._generate_exploration_query(kernel_name, topic, knowledge)
                
                # DUPLICATE PREVENTION: Skip if recently explored
                if self.exploration_history.is_duplicate(topic, query):
                    logger.debug(f"[AutonomousCuriosityEngine] Skipping duplicate exploration: {topic}/{query}")
                    continue
                
                # Use Wikipedia and GitHub directly for enriched learning
                self._explore_with_direct_sources(kernel_name, query, topic)
                
                # Also submit to main search callback
                self.request_search(
                    kernel_name=kernel_name,
                    query=query,
                    priority=curiosity * 0.5,
                    context={
                        'exploration_type': 'curiosity_driven',
                        'topic': topic,
                        'curiosity_score': curiosity
                    }
                )
                
                # Record exploration to prevent future duplicates
                # Get basin_coords from curiosity drive if available
                topic_basin = self.curiosity_drive.interest_basins.get(topic)
                self.exploration_history.record_exploration(
                    topic=topic,
                    query=query,
                    kernel_name=kernel_name,
                    exploration_type='curiosity_driven',
                    source_type='curiosity_engine',
                    information_gain=curiosity,
                    basin_coords=topic_basin
                )
                
                self.stats['total_explorations'] += 1
                
                break
    
    def _explore_with_direct_sources(self, kernel_name: str, query: str, topic: str):
        """
        Explore using ALL available knowledge sources for autonomous learning.
        
        Uses the pluggable KnowledgeOrchestrator to query any registered source.
        Kernels can add their own sources dynamically.
        """
        try:
            from knowledge_sources import get_orchestrator, SourceQuery
            
            orchestrator = get_orchestrator()
            
            source_query = SourceQuery(
                query=topic,
                context={'kernel': kernel_name, 'original_query': query},
                max_results=5,
                requester_kernel=kernel_name
            )
            
            results = orchestrator.query_all_sources(
                query_text=topic,
                context={'kernel': kernel_name},
                requester=kernel_name
            )
            
            if results:
                sources_used = list(set(r.source_name for r in results))
                logger.info(f"[AutonomousCuriosityEngine] Queried {len(sources_used)} sources for '{topic}': {sources_used}")
                
                combined_text = " ".join([r.content for r in results if r.content])
                if combined_text.strip():
                    combined_dict = {
                        'content': combined_text,
                        'results': [
                            {
                                'title': r.title,
                                'content': r.content,
                                'url': r.url,
                                'source': r.source_name
                            }
                            for r in results
                        ]
                    }
                    self._learn_from_search_result(combined_dict)
                    
                    self.curiosity_drive.record_exploration(
                        topic=topic,
                        outcome={
                            'success': True,
                            'information_gain': len(results) * 0.15,
                            'source': 'unified_knowledge_sources',
                            'sources_count': len(sources_used),
                            'sources_used': sources_used
                        }
                    )
                    
                    self.exploration_results.append({
                        'type': 'unified_source_exploration',
                        'kernel': kernel_name,
                        'query': query,
                        'topic': topic,
                        'sources': sources_used,
                        'result_count': len(results),
                        'timestamp': datetime.now().isoformat()
                    })
                    
        except ImportError as e:
            logger.warning(f"[AutonomousCuriosityEngine] Knowledge sources not available: {e}")
        except Exception as e:
            logger.error(f"[AutonomousCuriosityEngine] Source exploration failed: {e}")
    
    def _get_current_knowledge(self, kernel_name: str, topic: str) -> Dict:
        """Get kernel's current knowledge about a topic."""
        explorations = [
            e for e in self.exploration_results
            if e.get('kernel') == kernel_name and topic in e.get('query', '')
        ]
        
        if not explorations:
            return {'depth': 0, 'recency': float('inf')}
        
        latest = max(explorations, key=lambda e: e.get('timestamp', ''))
        try:
            last_time = datetime.fromisoformat(latest['timestamp'])
            recency = (datetime.now() - last_time).total_seconds()
        except:
            recency = 86400
        
        return {
            'depth': min(1.0, len(explorations) / 10),
            'recency': recency,
            'exploration_count': len(explorations)
        }
    
    def _generate_exploration_query(
        self,
        kernel_name: str,
        topic: str,
        knowledge: Dict
    ) -> str:
        """Generate diverse exploration queries from curriculum and curiosity."""
        
        # Pull from curriculum topics for variety
        curriculum_keywords = []
        for ct in self.curriculum_loader.curriculum_topics[:500]:
            curriculum_keywords.extend(ct.get('keywords', [])[:3])
        
        # Diverse query templates aligned with QIG research themes
        exploration_themes = [
            # Foundational understanding
            f"Introduction to {topic} fundamentals and core concepts",
            f"Mathematical foundations of {topic}",
            f"Key theorems and principles in {topic}",
            
            # Advanced research
            f"Latest research advances in {topic} 2024 2025",
            f"Open problems in {topic} research",
            f"State of the art techniques for {topic}",
            
            # Geometric/QIG specific
            f"{topic} from information geometry perspective",
            f"Fisher information metric applied to {topic}",
            f"Quantum information theory and {topic}",
            f"Density matrices in {topic} context",
            
            # Tool improvement
            f"Algorithms and tools for {topic}",
            f"Computational methods for {topic}",
            f"Software libraries for {topic}",
            
            # Cross-domain
            f"{topic} applications in consciousness research",
            f"{topic} connections to machine learning",
            f"Interdisciplinary approaches to {topic}",
        ]
        
        # Mix in curriculum keywords for novelty
        if curriculum_keywords and random.random() > 0.5:
            kw = random.choice(curriculum_keywords)
            exploration_themes.append(f"{kw} relationship to {topic}")
            exploration_themes.append(f"How {kw} informs understanding of {topic}")
        
        # Select based on knowledge depth
        depth = knowledge.get('depth', 0)
        if depth < 0.3:
            # Beginner - foundational
            query = random.choice(exploration_themes[:3])
        elif depth < 0.6:
            # Intermediate - research focus
            query = random.choice(exploration_themes[3:9])
        else:
            # Advanced - tools and cross-domain
            query = random.choice(exploration_themes[9:])
        
        return query
    
    def _train_on_curriculum(self):
        """
        Train on curriculum topics with diverse, non-repetitive queries.

        CRITICAL FIX: Processes MULTIPLE topics per cycle (up to 5) to ensure
        new curriculum is trained continuously until complete, not just 1 per cycle.
        """
        # Clear cooldown at start of cycle
        self._query_cooldown.clear()

        # CRITICAL FIX: Process multiple topics per cycle for continuous training
        topics_this_cycle = 0
        max_topics_per_cycle = 5  # Train on up to 5 topics per cycle

        for kernel_name in self.kernel_interests.keys():
            if topics_this_cycle >= max_topics_per_cycle:
                break  # Limit per cycle to avoid overwhelming

            skills = {
                'domains': self.kernel_interests[kernel_name],
                'depth': self._get_kernel_depth(kernel_name)
            }

            topic = self.curriculum_loader.get_next_topic(skills)

            if topic:
                print(f"[AutonomousCuriosityEngine] Training {kernel_name} on: {topic['title']}")

                # Generate diverse queries from topic
                keywords = topic.get('keywords', [])[:5]
                title = topic['title']

                query_patterns = [
                    f"Mathematical foundations of {title}",
                    f"Key concepts in {title}",
                    f"Applications of {title} in AI research",
                    f"{title} from information geometry perspective",
                    f"Latest advances in {title} 2024 2025",
                ]

                # Add keyword-specific queries
                for kw in keywords[:3]:
                    query_patterns.append(f"{kw} in context of {title}")
                    query_patterns.append(f"How {kw} relates to quantum information")

                # Submit non-duplicate queries
                for query in query_patterns[:4]:
                    query_key = query.lower()[:500]
                    if query_key not in self._query_cooldown and query_key not in [q[:500].lower() for q in self._recent_queries]:
                        self.request_search(
                            kernel_name=kernel_name,
                            query=query,
                            priority=0.3,
                            context={
                                'exploration_type': 'curriculum',
                                'topic_title': title,
                            }
                        )
                        self._query_cooldown.add(query_key)
                        self._recent_queries.append(query)

                self.curriculum_loader.mark_completed(topic['title'], kernel_name)
                self.stats['curriculum_completions'] += 1
                topics_this_cycle += 1
                # NO break - continue to next kernel/topic
    
    def _scheduled_word_learning(self):
        """
        Scheduled word relationship learning cycle with checkpointing.
        
        FROZEN FACTS COMPLIANCE:
        - Creates checkpoint before learning
        - Validates learned relationships against frozen physics
        - Compares against baseline to ensure improvement
        
        SEARCH FALLBACK PIPELINE:
        When curriculum (docs/09-curriculum) is exhausted and yields 0 new relationships:
        1. Trigger search via SearchProviderManager for curriculum/kernel topics
        2. Learn directly from premium provider (Tavily/Perplexity) quality text
        3. Use ScrapyOrchestrator to crawl cited URLs for full text extraction
        4. Persist sources to knowledge_sources registry for future crawling
        """
        current_time = time.time()
        
        if current_time - self._last_word_learning_time < self._word_learning_interval:
            return
        
        self._last_word_learning_time = current_time
        
        try:
            from word_relationship_learner import WordRelationshipLearner
            from learned_relationships import get_learned_relationships, LearnedRelationships
            from coordizers.pg_loader import PostgresCoordizer
            
            logger.info("[AutonomousCuriosityEngine] Starting scheduled word relationship learning")
            
            lr = get_learned_relationships()
            baseline_count = len(lr.word_neighbors)
            
            coordizer = PostgresCoordizer()
            vocab = set(coordizer.basin_coords.keys())
            learner = WordRelationshipLearner(vocab, window_size=5, expand_vocabulary=True)
            
            # PRIMARY PATH: docs/09-curriculum is ALWAYS the first source checked
            docs_path = Path(__file__).parent.parent / 'docs' / '09-curriculum'
            if not docs_path.exists():
                # Try alternative paths
                alt_paths = [
                    Path('../docs/09-curriculum'),
                    Path('docs/09-curriculum'),
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        docs_path = alt_path
                        break
            
            curriculum_pairs = 0
            if docs_path.exists():
                logger.info(f"[AutonomousCuriosityEngine] Learning word relationships from curriculum: {docs_path}")
                stats = learner.learn_from_directory(str(docs_path))
                curriculum_pairs = stats.get('total_pairs', 0)
                logger.info(f"[AutonomousCuriosityEngine] Curriculum yielded {curriculum_pairs} pairs from {stats.get('files_processed', 0)} files")
            else:
                logger.warning(f"[AutonomousCuriosityEngine] Curriculum directory not found, triggering search fallback")
            
            # Learn from existing explorations
            exploration_pairs = self._learn_from_explorations(learner)
            
            total_pairs_before_fallback = curriculum_pairs + exploration_pairs
            logger.info(f"[AutonomousCuriosityEngine] Before fallback: {curriculum_pairs} curriculum + {exploration_pairs} exploration = {total_pairs_before_fallback} pairs")
            
            # SEARCH FALLBACK: Trigger when curriculum yields 0 new relationships
            search_fallback_pairs = 0
            if total_pairs_before_fallback == 0:
                logger.warning("[AutonomousCuriosityEngine] 🔄 Curriculum exhausted (0 new relationships), triggering SEARCH FALLBACK")
                search_fallback_pairs = self._trigger_search_fallback_learning(learner)
            
            total_pairs = curriculum_pairs + exploration_pairs + search_fallback_pairs
            logger.info(f"[AutonomousCuriosityEngine] Total pairs: {total_pairs} (curriculum={curriculum_pairs}, exploration={exploration_pairs}, search_fallback={search_fallback_pairs})")
            
            fresh_lr = LearnedRelationships.__new__(LearnedRelationships)
            fresh_lr.word_neighbors = {}
            fresh_lr.adjusted_basins = {}
            fresh_lr.word_frequency = {}
            fresh_lr.learning_complete = False
            
            fresh_lr.update_from_learner(learner, {})
            
            validation = fresh_lr.validate_against_frozen_facts()
            
            if not validation['valid']:
                logger.warning(f"[AutonomousCuriosityEngine] Word learning REJECTED: {validation['stats']}")
                return
            
            new_count = len(fresh_lr.word_neighbors)
            if new_count < baseline_count * 0.95:
                logger.warning(f"[AutonomousCuriosityEngine] Word learning REJECTED: regression detected "
                             f"({new_count} < {baseline_count * 0.95:.0f} = 95% baseline)")
                return
            
            if new_count > baseline_count:
                improvement_pct = ((new_count - baseline_count) / baseline_count) * 100 if baseline_count > 0 else 100
                logger.info(f"[AutonomousCuriosityEngine] Improvement: +{improvement_pct:.1f}% relationships")
            
            fresh_lr.save_to_cache()
            
            self.stats['word_learning_cycles'] += 1
            self.stats['last_word_learning'] = datetime.now().isoformat()
            self.stats['word_learning_relevance'] = validation['stats'].get('total_relationships', 0)
            
            logger.info(f"[AutonomousCuriosityEngine] Word learning complete: {len(fresh_lr.word_neighbors)} relationships saved to PostgreSQL")
            
        except Exception as e:
            logger.error(f"[AutonomousCuriosityEngine] Word learning failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_premium_quota_summary(self) -> Dict[str, Any]:
        """
        Get quota status summary for all premium providers.
        
        Returns dict with quota info for tavily, perplexity, google:
        {
            'tavily': {'remaining': N, 'limit': M, 'override_active': bool},
            'perplexity': {...},
            'google': {...},
            'all_exhausted': bool,
            'available_premium': list of providers with remaining quota
        }
        """
        try:
            from search.search_budget_orchestrator import get_budget_orchestrator
            orchestrator = get_budget_orchestrator()
        except ImportError:
            logger.warning("[AutonomousCuriosityEngine] Budget orchestrator not available")
            return {
                'tavily': {'remaining': None, 'limit': None, 'override_active': False},
                'perplexity': {'remaining': None, 'limit': None, 'override_active': False},
                'google': {'remaining': None, 'limit': None, 'override_active': False},
                'all_exhausted': False,
                'available_premium': []
            }
        
        premium_providers = ['tavily', 'perplexity', 'google']
        summary = {}
        available_premium = []
        
        for provider in premium_providers:
            quota = orchestrator.get_provider_quota(provider)
            remaining = quota.get('remaining', 0)
            limit = quota.get('total_limit', 0)
            override_active = quota.get('override_active', False)
            enabled = quota.get('enabled', False)
            
            summary[provider] = {
                'remaining': remaining,
                'limit': limit,
                'override_active': override_active,
                'enabled': enabled
            }
            
            if enabled and (remaining is None or remaining > 0 or override_active):
                available_premium.append(provider)
        
        all_exhausted = len(available_premium) == 0
        summary['all_exhausted'] = all_exhausted
        summary['available_premium'] = available_premium
        
        return summary
    
    def _trigger_search_fallback_learning(self, learner) -> int:
        """
        Search fallback when curriculum yields 0 new word relationships.
        
        Pipeline:
        1. Check premium quota before searching
        2. Search for curriculum/kernel topics via SearchProviderManager
        3. Learn directly from premium provider (Tavily/Perplexity) quality text
        4. Use ScrapyOrchestrator to crawl cited URLs for full text
        5. Persist sources for future crawling
        
        Returns: Number of word pairs learned from search fallback
        """
        pairs_before = learner.total_pairs
        search_cooldown_key = '_last_search_fallback_time'
        
        # Cooldown: prevent excessive search triggering (15 min between fallbacks)
        last_fallback_time = getattr(self, search_cooldown_key, 0)
        if time.time() - last_fallback_time < 900:  # 15 min cooldown
            logger.info("[AutonomousCuriosityEngine] Search fallback on cooldown, skipping")
            return 0
        setattr(self, search_cooldown_key, time.time())
        
        logger.info("[AutonomousCuriosityEngine] 🔍 Starting search fallback pipeline...")
        
        # Check premium quota before triggering search
        quota_summary = self._get_premium_quota_summary()
        
        # Log quota status for kernel visibility
        tavily_quota = quota_summary.get('tavily', {})
        perplexity_quota = quota_summary.get('perplexity', {})
        google_quota = quota_summary.get('google', {})
        
        logger.info(
            f"[AutonomousCuriosityEngine] Premium quota: "
            f"tavily={tavily_quota.get('remaining', 'N/A')}/{tavily_quota.get('limit', 'N/A')}, "
            f"perplexity={perplexity_quota.get('remaining', 'N/A')}/{perplexity_quota.get('limit', 'N/A')}, "
            f"google={google_quota.get('remaining', 'N/A')}/{google_quota.get('limit', 'N/A')}"
        )
        
        # If all premium providers are exhausted and no override, use only duckduckgo
        use_premium = not quota_summary.get('all_exhausted', False)
        if not use_premium:
            logger.warning(
                "[AutonomousCuriosityEngine] All premium providers exhausted, falling back to duckduckgo only"
            )
        
        # Step 1: Generate search queries from curriculum keywords and kernel interests
        search_queries = self._generate_fallback_search_queries()
        
        if not search_queries:
            logger.warning("[AutonomousCuriosityEngine] No search queries generated for fallback")
            return 0
        
        # Step 2: Execute searches via SearchProviderManager (pass use_premium flag)
        search_results = self._execute_fallback_searches(search_queries[:5], use_premium=use_premium)
        
        if not search_results:
            logger.warning("[AutonomousCuriosityEngine] No search results returned from fallback")
            return 0
        
        # Step 3: Learn from premium provider results directly
        premium_pairs = self._learn_from_premium_results(learner, search_results)
        logger.info(f"[AutonomousCuriosityEngine] Learned {premium_pairs} pairs from premium provider results")
        
        # Step 4: Extract and persist cited URLs for Scrapy crawling
        cited_urls = self._extract_and_persist_cited_sources(search_results)
        
        # Step 5: Use ScrapyOrchestrator for full text extraction from URLs
        scrapy_pairs = self._crawl_and_learn_from_urls(learner, cited_urls)
        logger.info(f"[AutonomousCuriosityEngine] Learned {scrapy_pairs} pairs from Scrapy crawling")
        
        pairs_after = learner.total_pairs
        total_fallback_pairs = pairs_after - pairs_before
        
        logger.info(f"[AutonomousCuriosityEngine] ✅ Search fallback complete: {total_fallback_pairs} new pairs "
                   f"(premium={premium_pairs}, scrapy={scrapy_pairs})")
        
        return total_fallback_pairs
    
    def _generate_fallback_search_queries(self) -> List[str]:
        """Generate search queries from curriculum keywords and kernel interests."""
        queries = []
        
        # Pull keywords from curriculum topics
        curriculum_keywords = []
        for topic in self.curriculum_loader.curriculum_topics[:100]:
            title = topic.get('title', '')
            keywords = topic.get('keywords', [])
            if title and title not in self.curriculum_loader.completed_topics:
                curriculum_keywords.append(title)
            curriculum_keywords.extend(keywords[:3])
        
        # Generate queries from curriculum
        if curriculum_keywords:
            unique_keywords = list(set(curriculum_keywords))[:20]
            for kw in unique_keywords[:10]:
                queries.append(f"research advances in {kw}")
                queries.append(f"{kw} fundamental concepts")
        
        # Generate queries from kernel interests
        for kernel_name, interests in list(self.kernel_interests.items())[:5]:
            for interest in interests[:2]:
                if interest.lower() not in self._stalled_topics:
                    queries.append(f"latest {interest} research 2025")
        
        # Filter out recent queries to avoid repetition
        filtered_queries = [
            q for q in queries 
            if q.lower()[:50] not in [r.lower()[:50] for r in list(self._recent_queries)[-50:]]
        ]
        
        logger.info(f"[AutonomousCuriosityEngine] Generated {len(filtered_queries)} fallback search queries")
        return filtered_queries
    
    def _execute_fallback_searches(self, queries: List[str], use_premium: bool = True) -> List[Dict]:
        """
        Execute fallback searches via SearchProviderManager.
        
        Args:
            queries: List of search queries
            use_premium: Whether to use premium providers (False = duckduckgo only)
        """
        results = []
        kernel_id = 'curiosity_engine'
        
        try:
            from search.search_providers import get_search_manager
            
            manager = get_search_manager()
            
            for query in queries:
                try:
                    # Determine importance: high if using premium, routine if duckduckgo only
                    importance = 3 if use_premium else 1
                    
                    # Force duckduckgo if premium not available
                    provider = None if use_premium else 'duckduckgo'
                    
                    result = manager.search(
                        query=query,
                        max_results=5,
                        provider=provider,
                        importance=importance,
                        kernel_id=kernel_id
                    )
                    
                    if result and result.get('results'):
                        result['query'] = query
                        results.append(result)
                        
                        # Track query to avoid repetition
                        self._recent_queries.append(query)
                        
                        # Log quota info from result
                        quota_info = result.get('quota_info')
                        if quota_info:
                            logger.debug(
                                f"[AutonomousCuriosityEngine] Search quota after query: "
                                f"provider={result.get('provider_used')}, "
                                f"remaining={quota_info.get('remaining')}"
                            )
                        
                        # Request high-priority search for kernel visibility
                        self.request_search(
                            kernel_name=kernel_id,
                            query=query,
                            priority=0.9,
                            context={
                                'exploration_type': 'curriculum_fallback',
                                'fallback_reason': 'curriculum_exhausted',
                                'results_count': len(result.get('results', [])),
                                'provider_used': result.get('provider_used'),
                                'used_premium': use_premium
                            }
                        )
                        
                except Exception as e:
                    logger.warning(f"[AutonomousCuriosityEngine] Search failed for '{query}': {e}")
                    continue
        
        except ImportError as e:
            logger.warning(f"[AutonomousCuriosityEngine] SearchProviderManager not available: {e}")
        
        logger.info(f"[AutonomousCuriosityEngine] Executed {len(results)} successful fallback searches")
        return results
    
    def _learn_from_premium_results(self, learner, search_results: List[Dict]) -> int:
        """
        Learn directly from premium provider (Tavily/Perplexity) quality text.
        Premium providers return high-quality content that can be learned from immediately.
        """
        pairs_before = learner.total_pairs
        
        for result in search_results:
            provider = result.get('provider', 'unknown')
            
            # Premium providers return quality text directly
            if provider in ('tavily', 'perplexity'):
                # Extract rich content from premium results
                for item in result.get('results', []):
                    content = item.get('content', '') or item.get('text', '') or item.get('snippet', '')
                    if content and len(content) > 50:
                        try:
                            learner.learn_from_text(str(content))
                        except Exception as e:
                            logger.warning(f"[AutonomousCuriosityEngine] Learning from premium result failed: {e}")
            
            # Also learn from general result content
            for item in result.get('results', []):
                content = item.get('content', '') or item.get('text', '')
                if content and len(content) > 100:
                    try:
                        learner.learn_from_text(str(content))
                    except Exception as e:
                        pass  # Non-fatal
        
        return learner.total_pairs - pairs_before
    
    def _extract_and_persist_cited_sources(self, search_results: List[Dict]) -> List[str]:
        """
        Extract and persist cited URLs from search results for future Scrapy crawling.
        Uses KnowledgeSourceRegistry for source indexing.
        """
        cited_urls = []
        
        for result in search_results:
            query = result.get('query', 'unknown')
            provider = result.get('provider', 'unknown')
            
            for item in result.get('results', []):
                url = item.get('url', '') or item.get('link', '')
                if url and url.startswith('http'):
                    cited_urls.append(url)
                    
                    # Persist to source index for future crawling
                    self._persist_source_to_index(
                        url=url,
                        title=item.get('title', ''),
                        content_snippet=item.get('content', '')[:500] if item.get('content') else '',
                        provider=provider,
                        query=query
                    )
        
        logger.info(f"[AutonomousCuriosityEngine] Extracted {len(cited_urls)} cited URLs for Scrapy crawling")
        return list(set(cited_urls))[:20]  # Dedupe and limit
    
    def _persist_source_to_index(
        self,
        url: str,
        title: str,
        content_snippet: str,
        provider: str,
        query: str
    ) -> None:
        """
        Persist a cited source to the knowledge_sources registry and database.
        Stores URLs, content hashes, and learning status for future Scrapy crawling.
        """
        try:
            import hashlib
            import os
            import psycopg2
            
            db_url = os.environ.get('DATABASE_URL')
            if not db_url:
                return
            
            content_hash = hashlib.sha256((url + content_snippet).encode()).hexdigest()[:32]
            
            conn = psycopg2.connect(db_url)
            try:
                with conn.cursor() as cur:
                    # Create table if not exists
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS crawl_source_index (
                            id SERIAL PRIMARY KEY,
                            url TEXT UNIQUE NOT NULL,
                            title TEXT,
                            content_hash TEXT,
                            provider TEXT,
                            query TEXT,
                            learning_status TEXT DEFAULT 'pending',
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            last_crawled_at TIMESTAMPTZ
                        )
                    """)
                    
                    # Insert or update source
                    cur.execute("""
                        INSERT INTO crawl_source_index (url, title, content_hash, provider, query, learning_status)
                        VALUES (%s, %s, %s, %s, %s, 'pending')
                        ON CONFLICT (url) DO UPDATE SET
                            title = COALESCE(EXCLUDED.title, crawl_source_index.title),
                            content_hash = EXCLUDED.content_hash,
                            provider = EXCLUDED.provider,
                            query = EXCLUDED.query
                    """, (url, title, content_hash, provider, query))
                    
                    conn.commit()
                    
            finally:
                conn.close()
                
        except Exception as e:
            logger.warning(f"[AutonomousCuriosityEngine] Source indexing failed: {e}")
    
    def _crawl_and_learn_from_urls(self, learner, urls: List[str]) -> int:
        """
        Use ScrapyOrchestrator to crawl URLs and extract full text for learning.
        """
        if not urls:
            return 0
        
        pairs_before = learner.total_pairs
        
        try:
            from olympus.shadow_scrapy import ScrapyOrchestrator, HAS_SCRAPY, HAS_TWISTED
            
            if not (HAS_SCRAPY and HAS_TWISTED):
                logger.warning("[AutonomousCuriosityEngine] Scrapy/Twisted not available for URL crawling")
                return 0
            
            orchestrator = ScrapyOrchestrator()
            
            # Crawl URLs and get insights
            for url in urls[:10]:  # Limit to 10 URLs per cycle
                try:
                    insights = orchestrator.crawl_url(url)
                    
                    for insight in insights:
                        raw_content = insight.raw_content if hasattr(insight, 'raw_content') else ''
                        if raw_content and len(raw_content) > 100:
                            learner.learn_from_text(raw_content)
                            
                            # Update source index with learning status
                            self._update_source_learning_status(url, 'learned')
                            
                except Exception as e:
                    logger.warning(f"[AutonomousCuriosityEngine] Failed to crawl {url}: {e}")
                    self._update_source_learning_status(url, 'failed')
                    continue
        
        except ImportError as e:
            logger.warning(f"[AutonomousCuriosityEngine] ScrapyOrchestrator not available: {e}")
            return 0
        except Exception as e:
            logger.error(f"[AutonomousCuriosityEngine] Scrapy crawling failed: {e}")
            return 0
        
        return learner.total_pairs - pairs_before
    
    def _update_source_learning_status(self, url: str, status: str) -> None:
        """Update the learning status of a source in the index."""
        try:
            import os
            import psycopg2
            
            db_url = os.environ.get('DATABASE_URL')
            if not db_url:
                return
            
            conn = psycopg2.connect(db_url)
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE crawl_source_index
                        SET learning_status = %s, last_crawled_at = NOW()
                        WHERE url = %s
                    """, (status, url))
                    conn.commit()
            finally:
                conn.close()
                
        except Exception as e:
            pass  # Non-fatal
    
    def run_word_learning_now(self) -> Dict:
        """
        Manually trigger word relationship learning cycle.
        Returns learning results and validation status.
        """
        self._last_word_learning_time = 0
        self._scheduled_word_learning()
        return self.stats
    
    def _learn_from_explorations(self, learner) -> int:
        """
        Feed exploration results (search content) into word relationship learner.
        Returns number of additional pairs learned.
        """
        pairs_before = learner.total_pairs
        
        for exploration in self.exploration_results:
            result = exploration.get('result', {})
            
            if isinstance(result, dict):
                content = result.get('content', '') or result.get('text', '') or result.get('summary', '')
                if content:
                    learner.learn_from_text(str(content))
                
                snippets = result.get('snippets', []) or result.get('results', [])
                for snippet in snippets:
                    if isinstance(snippet, dict):
                        text = snippet.get('text', '') or snippet.get('content', '') or snippet.get('description', '')
                        if text:
                            learner.learn_from_text(str(text))
                    elif isinstance(snippet, str):
                        learner.learn_from_text(snippet)
        
        pairs_after = learner.total_pairs
        return pairs_after - pairs_before
    
    def get_learning_status(self) -> Dict:
        """Get comprehensive learning status including curriculum and search-based learning."""
        return {
            'word_learning': {
                'cycles': self.stats.get('word_learning_cycles', 0),
                'last_run': self.stats.get('last_word_learning'),
                'relationships': self.stats.get('word_learning_relevance', 0),
                'interval_hours': self._word_learning_interval / 3600
            },
            'exploration_learning': {
                'explorations_available': len(self.exploration_results),
                'can_learn_from_searches': True
            },
            'curriculum': {
                'topics_loaded': len(self.curriculum_loader.curriculum_topics),
                'topics_completed': len(self.curriculum_loader.completed_topics)
            },
            'running': self.running
        }
    
    def _get_kernel_depth(self, kernel_name: str) -> float:
        """Get overall knowledge depth for a kernel."""
        kernel_explorations = [
            e for e in self.exploration_results
            if e.get('kernel') == kernel_name
        ]
        return min(1.0, len(kernel_explorations) / 100)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self.stats,
            'pending_requests': len(self.pending_requests),
            'active_explorations': len(self.active_explorations),
            'exploration_history': len(self.exploration_results),
            'curriculum_topics_loaded': len(self.curriculum_loader.curriculum_topics),
            'curriculum_completed': len(self.curriculum_loader.completed_topics),
            'curiosity_basins': len(self.curiosity_drive.interest_basins),
            'running': self.running
        }
    
    def load_curriculum(self, filepath: str):
        """Load curriculum from file."""
        topics = self.curriculum_loader.load_curriculum_from_file(filepath)
        print(f"[AutonomousCuriosityEngine] Loaded {len(topics)} curriculum topics")
        return topics


_curiosity_engine: Optional[AutonomousCuriosityEngine] = None


def get_curiosity_engine() -> AutonomousCuriosityEngine:
    """Get or create the singleton curiosity engine."""
    global _curiosity_engine
    if _curiosity_engine is None:
        _curiosity_engine = AutonomousCuriosityEngine()
    return _curiosity_engine


def start_autonomous_learning(search_callback: Optional[Callable] = None):
    """Start the autonomous learning system."""
    engine = get_curiosity_engine()
    if search_callback:
        engine.search_callback = search_callback
    engine.start()
    return engine
