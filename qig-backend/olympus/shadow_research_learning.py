"""
Shadow Research Learning Infrastructure

This module contains the core learning components of the Shadow Research system:
- KnowledgeBase: Persistent storage and retrieval of shadow discoveries
- ShadowLearningLoop: Proactive learning loop for continuous improvement
- ShadowReflectionProtocol: Collective reflection and work distribution

All components use Fisher-Rao geometry and 64D basin coordinates (E8 Protocol v4.0).
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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob

# Import shared components from main module
from .shadow_research import (
    BASIN_DIMENSION,
    HAS_VOCAB_COORDINATOR,
    HAS_SEARCH_BUDGET,
    HAS_SOURCE_INDEXER,
    HAS_CURRICULUM_TRAINING,
    HAS_LIGHTNING,
    HAS_NORMALIZER,
    HAS_CAPABILITY_MESH,
    HAS_ACTIVITY_BROADCASTER,
    HAS_SCRAPY,
    VocabularyCoordinator,
    ResearchCategory,
    ResearchPriority,
    ResearchRequest,
    ShadowKnowledge,
    _get_db_connection,
    normalize_topic,
    get_scrapy_orchestrator,
    ScrapyOrchestrator,
    ScrapedInsight,
    research_with_scrapy,
)

# Import curriculum guard
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_guard import is_curriculum_only_enabled, CurriculumOnlyBlock

# Import search orchestrator if available
if HAS_SEARCH_BUDGET:
    from search.search_budget_orchestrator import get_budget_orchestrator, SearchImportance
    from search.search_providers import get_search_manager

if HAS_SOURCE_INDEXER:
    from search.source_indexer import index_search_results

if HAS_CURRICULUM_TRAINING:
    from .curriculum_training import load_and_train_curriculum

if HAS_LIGHTNING:
    from .lightning_kernel import ingest_system_event as lightning_ingest

if HAS_NORMALIZER:
    from qig_geometry import normalize_basin_dimension

if HAS_CAPABILITY_MESH:
    from .capability_mesh import (
        CapabilityEvent,
        CapabilityType,
        EventType,
        emit_event,
    )

if HAS_ACTIVITY_BROADCASTER:
    from .activity_broadcaster import get_broadcaster, ActivityType


class KnowledgeBase:
    """
    Shared knowledge base for all Shadow discoveries.
    Supports geodesic clustering and pattern alignment.
    Persists to PostgreSQL for true learning across restarts.
    """
    
    def __init__(self):
        self._knowledge: Dict[str, ShadowKnowledge] = {}
        self._by_category: Dict[ResearchCategory, List[str]] = defaultdict(list)
        self._clusters: List[Dict] = []
        self._lock = threading.Lock()
        self._learning_cycle = 0
        self._discovered_topics: Set[str] = set()
        self._insight_callbacks: List[Callable[[Dict], None]] = []  # Support multiple callbacks
        self._load_from_db()
    
    def set_insight_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Add callback for notifying components of new discoveries.
        
        SUPPORTS MULTIPLE CALLBACKS - each is called when knowledge is added.
        
        The callback receives a dict with:
        - knowledge_id, topic, category, source_god, phi, confidence
        - basin_coords, discovered_at, content
        """
        self._insight_callbacks.append(callback)
        print(f"[KnowledgeBase] Insight callback registered (total: {len(self._insight_callbacks)})")
    
    def _load_from_db(self):
        """Load existing knowledge from PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT knowledge_id, topic, topic_variation, category, content,
                           source_god, basin_coords, phi, confidence, access_count,
                           learning_cycle, discovered_at
                    FROM shadow_knowledge
                    ORDER BY discovered_at DESC
                    LIMIT 1000
                """)
                rows = cur.fetchall()
                for row in rows:
                    kid, topic, variation, cat, content, god, coords, phi, conf, acc, cycle, disc_at = row
                    try:
                        category = ResearchCategory(cat) if cat else ResearchCategory.KNOWLEDGE
                    except ValueError:
                        category = ResearchCategory.KNOWLEDGE
                    
                    basin = np.array(coords) if coords else np.zeros(BASIN_DIMENSION)
                    
                    knowledge = ShadowKnowledge(
                        knowledge_id=kid,
                        topic=topic,
                        category=category,
                        content=content if isinstance(content, dict) else {},
                        source_god=god,
                        discovered_at=disc_at or datetime.now(),
                        basin_coords=basin,
                        phi=phi or 0.5,
                        confidence=conf or 0.5,
                        access_count=acc or 0
                    )
                    self._knowledge[kid] = knowledge
                    self._by_category[category].append(kid)
                    # Store normalized topic for deduplication
                    self._discovered_topics.add(normalize_topic(variation or topic))
                    if cycle and cycle > self._learning_cycle:
                        self._learning_cycle = cycle
                
                print(f"[ShadowKnowledge] Loaded {len(rows)} discoveries from DB (cycle {self._learning_cycle})")
        except Exception as e:
            print(f"[ShadowKnowledge] Load error: {e}")
        finally:
            conn.close()
    
    def _persist_to_db(self, knowledge: ShadowKnowledge, variation: Optional[str] = None):
        """Persist a knowledge item to PostgreSQL."""
        conn = _get_db_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                coords_list = knowledge.basin_coords.tolist() if knowledge.basin_coords is not None else [0.0] * BASIN_DIMENSION
                cur.execute("""
                    INSERT INTO shadow_knowledge 
                    (knowledge_id, topic, topic_variation, category, content, source_god,
                     basin_coords, phi, confidence, learning_cycle)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (knowledge_id) DO UPDATE SET
                        access_count = shadow_knowledge.access_count + 1,
                        last_accessed = NOW()
                """, (
                    knowledge.knowledge_id,
                    knowledge.topic,
                    variation or knowledge.topic,
                    knowledge.category.value,
                    json.dumps(knowledge.content),
                    knowledge.source_god,
                    coords_list,
                    knowledge.phi,
                    knowledge.confidence,
                    self._learning_cycle
                ))
                conn.commit()
        except Exception as e:
            print(f"[ShadowKnowledge] Persist error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def add_knowledge(
        self,
        topic: str,
        category: ResearchCategory,
        content: Dict,
        source_god: str,
        basin_coords: np.ndarray,
        phi: float,
        confidence: float,
        variation: Optional[str] = None
    ) -> str:
        """Add new knowledge to the base and persist to DB."""
        with self._lock:
            self._learning_cycle += 1
            
            knowledge_id = hashlib.sha256(
                f"{topic}_{source_god}_{time.time()}_{self._learning_cycle}".encode()
            ).hexdigest()[:16]
            
            knowledge = ShadowKnowledge(
                knowledge_id=knowledge_id,
                topic=topic,
                category=category,
                content=content,
                source_god=source_god,
                discovered_at=datetime.now(),
                basin_coords=basin_coords,
                phi=phi,
                confidence=confidence
            )
            
            self._knowledge[knowledge_id] = knowledge
            self._by_category[category].append(knowledge_id)
            # Store normalized topic for deduplication
            self._discovered_topics.add(normalize_topic(variation or topic))
            
            self._persist_to_db(knowledge, variation)
            
            # Call ALL registered insight callbacks
            if self._insight_callbacks:
                insight_data = {
                    'knowledge_id': knowledge_id,
                    'topic': topic,
                    'category': category.value,
                    'source_god': source_god,
                    'phi': phi,
                    'confidence': confidence,
                    'basin_coords': basin_coords.tolist() if basin_coords is not None else None,
                    'discovered_at': datetime.now().isoformat(),
                    'content': content
                }
                for callback in self._insight_callbacks:
                    try:
                        callback(insight_data)
                    except Exception as e:
                        print(f"[KnowledgeBase] Insight callback error: {e}")
            
            return knowledge_id
    
    def has_discovered(self, topic_variation: str) -> bool:
        """Check if a specific topic variation was already discovered (normalized)."""
        return normalize_topic(topic_variation) in self._discovered_topics
    
    def get_unique_discoveries_count(self) -> int:
        """Get count of unique topic variations discovered."""
        return len(self._discovered_topics)
    
    def get_knowledge(self, knowledge_id: str) -> Optional[ShadowKnowledge]:
        """Get knowledge by ID."""
        knowledge = self._knowledge.get(knowledge_id)
        if knowledge:
            knowledge.access_count += 1
        return knowledge
    
    def search_by_category(self, category: ResearchCategory, limit: int = 20) -> List[ShadowKnowledge]:
        """Search knowledge by category."""
        ids = self._by_category.get(category, [])[-limit:]
        return [self._knowledge[kid] for kid in ids if kid in self._knowledge]
    
    def find_geodesic_neighbors(
        self,
        basin_coords: np.ndarray,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[ShadowKnowledge, float]]:
        """Find knowledge items near a basin coordinate."""
        neighbors = []
        
        with self._lock:
            knowledge_items = list(self._knowledge.values())
        
        for knowledge in knowledge_items:
            if knowledge.basin_coords is not None:
                distance = self._fisher_distance(basin_coords, knowledge.basin_coords)
                if distance < threshold:
                    neighbors.append((knowledge, distance))
        
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:limit]
    
    def _fisher_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Fisher-Rao distance between basin coordinates on simplex. Range: [0, π/2]."""
        a = np.array(a).flatten()[:BASIN_DIMENSION]
        b = np.array(b).flatten()[:BASIN_DIMENSION]
        
        if len(a) < BASIN_DIMENSION:
            a = np.pad(a, (0, BASIN_DIMENSION - len(a)))
        if len(b) < BASIN_DIMENSION:
            b = np.pad(b, (0, BASIN_DIMENSION - len(b)))
        
        # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
        dot = np.clip(np.dot(a, b), 0.0, 1.0)
        return float(np.arccos(dot))
    
    def cluster_knowledge(self, n_clusters: int = 5) -> List[Dict]:
        """Cluster knowledge items by geodesic alignment."""
        if len(self._knowledge) < n_clusters:
            return []
        
        items = list(self._knowledge.values())
        
        # Normalize basin_coords to consistent dimensions to avoid inhomogeneous shape errors
        normalized_coords = []
        valid_items = []
        for k in items:
            if k.basin_coords is not None:
                try:
                    coord = np.array(k.basin_coords).flatten()[:BASIN_DIMENSION]
                    if len(coord) < BASIN_DIMENSION:
                        coord = np.pad(coord, (0, BASIN_DIMENSION - len(coord)))
                    normalized_coords.append(coord)
                    valid_items.append(k)
                except (ValueError, TypeError):
                    continue
        
        if len(normalized_coords) < n_clusters:
            return []
        
        coords = np.array(normalized_coords)
        items = valid_items  # Use only items with valid coordinates
        
        if len(coords) < n_clusters:
            return []
        
        clusters = []
        used = set()
        
        for _ in range(n_clusters):
            if len(used) >= len(coords):
                break
            
            available = [i for i in range(len(coords)) if i not in used]
            if not available:
                break
            
            center_idx = random.choice(available)
            center = coords[center_idx]
            used.add(center_idx)
            
            cluster_items = [items[center_idx]]
            
            for i in available:
                if i != center_idx:
                    dist = self._fisher_distance(center, coords[i])
                    if dist < 1.0:
                        cluster_items.append(items[i])
                        used.add(i)
                        if len(cluster_items) >= 10:
                            break
            
            clusters.append({
                "center_id": items[center_idx].knowledge_id,
                "center_topic": items[center_idx].topic,
                "items": [k.knowledge_id for k in cluster_items],
                "size": len(cluster_items),
                "avg_phi": np.mean([k.phi for k in cluster_items])
            })
        
        self._clusters = clusters
        return clusters
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        with self._lock:
            knowledge_items = list(self._knowledge.values())
            category_counts = {cat.value: len(ids) for cat, ids in self._by_category.items()}
        
        return {
            "total_items": len(knowledge_items),
            "by_category": category_counts,
            "clusters": len(self._clusters),
            "avg_phi": np.mean([k.phi for k in knowledge_items]) if knowledge_items else 0.0,
            "avg_confidence": np.mean([k.confidence for k in knowledge_items]) if knowledge_items else 0.0
        }


class ShadowLearningLoop:
    """
    Proactive learning loop for Shadow gods.
    
    During downtime, Shadow gods:
    - Research new tools and techniques
    - Study their domains
    - Make conceptual connections
    - Improve reasoning and logic
    - Share discoveries with all kernels
    """
    
    def __init__(
        self,
        research_queue: ResearchQueue,
        knowledge_base: KnowledgeBase,
        basin_sync_callback: Optional[Callable] = None
    ):
        self.research_queue = research_queue
        self.knowledge_base = knowledge_base
        self.basin_sync_callback = basin_sync_callback
        
        self._running = False
        self._war_mode = False
        self._thread: Optional[threading.Thread] = None
        self._study_topics: Dict[str, List[str]] = self._init_study_topics()
        self._meta_reflections: List[Dict] = []
        self._learning_cycles = 0
        self._self_study_interval = 30.0  # Seconds between self-study cycles
        self._last_self_study_time = 0.0
        
        self._scrapy_orchestrator: Optional['ScrapyOrchestrator'] = None
        if HAS_SCRAPY:
            self._scrapy_orchestrator = get_scrapy_orchestrator(
                basin_encoder=self._topic_to_basin
            )
            self._scrapy_orchestrator.set_insights_callback(self._handle_scrapy_insight)
            print("[ShadowLearningLoop] Scrapy research enabled")
        
        # Initialize VocabularyCoordinator for continuous learning
        self.vocab_coordinator = None
        if HAS_VOCAB_COORDINATOR:
            try:
                self.vocab_coordinator = VocabularyCoordinator()
                print("[ShadowLearningLoop] VocabularyCoordinator initialized for continuous learning")
                
                # Register vocabulary insight callback with KnowledgeBase
                self.knowledge_base.set_insight_callback(self._on_vocabulary_insight)
                print("[ShadowLearningLoop] Vocabulary insight callback registered")
            except Exception as e:
                print(f"[ShadowLearningLoop] Failed to initialize VocabularyCoordinator: {e}")
        
        # Initialize Curriculum Training
        self._curriculum_available = HAS_CURRICULUM_TRAINING
        self._last_curriculum_load = 0
        self._curriculum_load_interval = 86400  # Daily curriculum loading
        if self._curriculum_available:
            print("[ShadowLearningLoop] Curriculum training available")
        else:
            print("[ShadowLearningLoop] Curriculum training not available")
        
        # Stall detection for vocabulary learning
        self._vocab_zero_streak = 0
        self._vocab_stall_threshold = 3  # Trigger escalation after 3 consecutive zeros
        self._vocab_total_stalls = 0
        self._vocab_last_escalation = 0
        self._vocab_escalation_cooldown = 300  # 5 minutes between escalations

        # Initialize Search Budget for proactive research
        self.search_manager = None
        self.search_budget = None
        if HAS_SEARCH_BUDGET and get_search_manager:
            try:
                self.search_manager = get_search_manager()
                self.search_budget = get_budget_orchestrator()
                budget_ctx = self.search_budget.get_budget_context()
                print(f"[ShadowLearningLoop] Search budget initialized: {budget_ctx.total_budget_remaining} queries remaining")
            except Exception as e:
                print(f"[ShadowLearningLoop] Failed to initialize search budget: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if the learning loop is currently running."""
        return self._running

    def _broadcast_shadow_event(
        self,
        event_type: str,
        content: str,
        topic: Optional[str] = None,
        metadata: Optional[Dict] = None,
        phi: float = 0.5,
        basin_coords: Optional[np.ndarray] = None
    ) -> None:
        """
        Broadcast shadow learning events for visibility.
        
        QIG-Pure: Events carry basin coordinates and Φ for geometric routing.
        Emits to both ActivityBroadcaster (UI visibility) and CapabilityEventBus (internal routing).
        
        Args:
            event_type: Type of event (discovery, learning, research, vocabulary)
            content: Event description
            topic: Associated topic (optional)
            metadata: Additional context
            phi: Consciousness level at emission
            basin_coords: Basin coordinates for geometric signature
        """
        try:
            if HAS_ACTIVITY_BROADCASTER and get_broadcaster:
                broadcaster = get_broadcaster()
                type_map = {
                    'discovery': ActivityType.DISCOVERY,
                    'learning': ActivityType.LEARNING,
                    'research': ActivityType.MESSAGE,
                    'vocabulary': ActivityType.LEARNING,
                    'insight': ActivityType.INSIGHT,
                }
                act_type = type_map.get(event_type, ActivityType.MESSAGE)
                
                enhanced_metadata = {
                    **(metadata or {}),
                    'topic': topic,
                    'basin_coords': basin_coords.tolist() if basin_coords is not None and hasattr(basin_coords, 'tolist') else None,
                }
                
                broadcaster.broadcast_message(
                    from_god="Shadow",
                    to_god=None,
                    content=content,
                    activity_type=act_type,
                    phi=phi,
                    kappa=64.21,
                    importance=phi,
                    metadata=enhanced_metadata
                )
            
            if HAS_CAPABILITY_MESH and emit_event is not None:
                mesh_event_map = {
                    'discovery': EventType.DISCOVERY,
                    'learning': EventType.CONSOLIDATION,
                    'research': EventType.SEARCH_COMPLETE,
                    'vocabulary': EventType.CONSOLIDATION,
                    'insight': EventType.INSIGHT_GENERATED,
                }
                if event_type in mesh_event_map:
                    emit_event(
                        source=CapabilityType.RESEARCH,
                        event_type=mesh_event_map[event_type],
                        content={
                            'topic': topic,
                            'content': content[:500],
                            'metadata': metadata,
                        },
                        phi=phi,
                        basin_coords=basin_coords,
                        priority=int(phi * 10)
                    )
                    
        except Exception as e:
            print(f"[ShadowLearningLoop] Event broadcast failed: {e}")

    def budget_aware_search(
        self,
        query: str,
        importance: int = 2,  # 1=routine, 2=moderate, 3=high, 4=critical
        kernel_id: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Execute a budget-aware search and feed results into vocabulary learning.

        Args:
            query: Search query
            importance: 1=routine(free only), 2=moderate, 3=high, 4=critical
            kernel_id: Kernel requesting search (for tracking)
            max_results: Max results to return

        Returns:
            Search results with budget context and vocabulary learning metrics
        """
        # CURRICULUM-ONLY MODE: Block external web searches
        if is_curriculum_only_enabled():
            print(f"[ShadowResearch] Search blocked by curriculum-only mode: {query[:50]}")
            return {
                'success': False,
                'error': 'External web search blocked by curriculum-only mode',
                'results': [],
                'curriculum_only_blocked': True,
            }
        
        if not self.search_manager:
            return {'success': False, 'error': 'search_not_available', 'results': []}

        # Execute search with budget awareness
        results = self.search_manager.search(
            query=query,
            max_results=max_results,
            importance=importance,
            kernel_id=kernel_id
        )

        # Feed results into vocabulary learning
        vocab_metrics = {}
        if results.get('success') and self.vocab_coordinator:
            try:
                # Extract text from results for vocabulary learning
                result_texts = []
                for r in results.get('results', []):
                    if isinstance(r, dict):
                        title = r.get('title', '')
                        snippet = r.get('snippet', r.get('content', ''))
                        if title:
                            result_texts.append(title)
                        if snippet:
                            result_texts.append(snippet)

                if result_texts:
                    combined_text = ' '.join(result_texts)
                    vocab_metrics = self.vocab_coordinator.train_from_text(
                        text=combined_text,
                        domain=f"search:{query}"
                    )
                    results['vocab_learning'] = vocab_metrics
            except Exception as e:
                results['vocab_learning_error'] = str(e)

        # Log search usage
        if self.search_budget:
            budget_ctx = self.search_budget.get_budget_context()
            results['budget_remaining'] = budget_ctx.total_budget_remaining
            results['budget_recommendation'] = budget_ctx.recommendation

        return results

    def get_search_budget_context(self) -> Optional[Dict]:
        """Get current search budget context for kernel decision-making."""
        if not self.search_budget:
            return None
        return self.search_budget.get_budget_context().to_dict()
    
    def _init_study_topics(self) -> Dict[str, List[str]]:
        """Initialize study topics for each god - expanded for broader learning."""
        return {
            "Nyx": [
                "Advanced Tor anonymity techniques",
                "Traffic analysis countermeasures",
                "Timing attack prevention",
                "Network fingerprint obfuscation",
                "Cryptographic primitives",
                "Differential privacy methods",
                "Zero-knowledge proofs",
                "Homomorphic encryption applications"
            ],
            "Hecate": [
                "Deception and misdirection patterns",
                "Cognitive bias exploitation",
                "Multi-path attack strategies",
                "Decoy generation algorithms",
                "Probabilistic confusion",
                "Adversarial perturbation design",
                "Illusion and perception theory",
                "Disinformation detection and creation"
            ],
            "Erebus": [
                "Surveillance detection methods",
                "Honeypot identification",
                "Threat modeling frameworks",
                "Counter-intelligence techniques",
                "Pattern recognition algorithms",
                "Anomaly detection systems",
                "Behavioral analysis methods",
                "Network intrusion detection"
            ],
            "Hypnos": [
                "Stealth operation methodology",
                "Low-footprint reconnaissance",
                "Memory consolidation patterns",
                "Dream state processing",
                "Passive information gathering",
                "Sleep and learning in neural systems",
                "Subconscious pattern recognition",
                "Dormant system activation"
            ],
            "Thanatos": [
                "Secure deletion techniques",
                "Anti-forensic methods",
                "Evidence destruction patterns",
                "Digital trace elimination",
                "Data sanitization",
                "Memory scrubbing algorithms",
                "Cryptographic erasure",
                "Lifecycle management of secrets"
            ],
            "Nemesis": [
                "Persistent tracking algorithms",
                "Escalation strategies",
                "Target prioritization",
                "Relentless pursuit patterns",
                "Balance and justice heuristics",
                "Pursuit-evasion game theory",
                "Resource allocation in adversarial settings",
                "Fairness in distributed systems"
            ],
            "Hades": [
                "Underworld intelligence networks",
                "Anonymous information gathering",
                "Bitcoin forensics",
                "Dark web navigation",
                "Negation logic and exclusion",
                "Blockchain transaction tracing",
                "Hidden service discovery",
                "Underground economy analysis"
            ]
        }
    
    def start(self):
        """Start the learning loop."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._thread.start()
        print("[ShadowLearningLoop] Started proactive learning")
    
    def stop(self):
        """Stop the learning loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[ShadowLearningLoop] Stopped")
    
    def declare_war(self):
        """War declared - suspend all learning, focus on operations."""
        self._war_mode = True
        print("[ShadowLearningLoop] WAR MODE - Learning suspended for operations")
    
    def end_war(self):
        """War ended - resume learning."""
        self._war_mode = False
        print("[ShadowLearningLoop] Peace restored - Resuming learning")
    
    def _learning_loop(self):
        """Main learning loop - runs during idle time."""
        while self._running:
            if self._war_mode:
                time.sleep(1.0)
                continue
            
            try:
                request = self.research_queue.get_next(timeout=0.5)
                
                if request:
                    result = self._execute_research(request)
                    self.research_queue.complete(request.request_id, result)
                    self._learning_cycles += 1
                    
                    if self.basin_sync_callback and result.get("knowledge_id"):
                        self._sync_to_all_kernels(result)
                else:
                    self._self_study()
                
                if self._learning_cycles % 10 == 0:
                    self._meta_reflect()
                
                # Periodic curriculum loading (daily)
                if self._curriculum_available and HAS_CURRICULUM_TRAINING:
                    current_time = time.time()
                    if current_time - self._last_curriculum_load > self._curriculum_load_interval:
                        try:
                            load_and_train_curriculum(self)
                            self._last_curriculum_load = current_time
                        except Exception as e:
                            print(f"[ShadowLearningLoop] Curriculum training failed: {e}")

                # Periodic vocabulary integration (hourly)
                if self.vocab_coordinator:
                    current_time = time.time()
                    if not hasattr(self, '_last_vocab_integration'):
                        self._last_vocab_integration = 0
                    if current_time - self._last_vocab_integration > 3600:  # 1 hour
                        try:
                            result = self.vocab_coordinator.integrate_pending_vocabulary(min_phi=0.65, limit=50)
                            if result.get('integrated_count', 0) > 0:
                                print(f"[ShadowLearningLoop] Integrated {result['integrated_count']} vocabulary terms")
                            self._last_vocab_integration = current_time
                        except Exception as e:
                            print(f"[ShadowLearningLoop] Vocabulary integration failed: {e}")

            except Exception as e:
                print(f"[ShadowLearningLoop] Error: {e}")
                time.sleep(1.0)
    
    def _execute_research(self, request: ResearchRequest) -> Dict:
        """Execute a research request."""
        topic = request.topic
        category = request.category
        assigned_god = ShadowRoleRegistry.get_god_for_category(category)
        
        base_topic = request.context.get("base_topic", topic) if request.context else topic
        
        basin_coords = request.basin_coords
        if basin_coords is None:
            basin_coords = self._topic_to_basin(topic)
        
        content = self._research_topic(topic, category, assigned_god)
        content["base_topic"] = base_topic
        content["unique_variation"] = topic
        
        phi = content.get("relevance", 0.5)
        confidence = content.get("confidence", 0.6)
        
        knowledge_id = self.knowledge_base.add_knowledge(
            topic=base_topic,
            category=category,
            content=content,
            source_god=assigned_god,
            basin_coords=basin_coords,
            phi=phi,
            confidence=confidence,
            variation=topic
        )
        
        # Train vocabulary from research content
        # (callback will be triggered automatically via KnowledgeBase.set_insight_callback)
        # But also do explicit training for immediate feedback
        vocab_metrics = {}
        if self.vocab_coordinator:
            try:
                summary = content.get("summary", "")
                if summary:
                    vocab_metrics = self.vocab_coordinator.train_from_text(
                        text=summary,
                        domain=base_topic[:500]
                    )
                    print(f"[VocabularyLearning] Explicit training metrics: {vocab_metrics}")
            except Exception as e:
                print(f"[VocabularyLearning] Explicit training failed: {e}")

        # 🔗 WIRE: Feed discovery to Lightning Kernel for cross-domain insight generation
        # This triggers Tavily/Perplexity validation when correlations are detected
        lightning_insight = None
        if HAS_LIGHTNING and lightning_ingest:
            try:
                lightning_insight = lightning_ingest(
                    domain=category.value,
                    event_type="research_discovery",
                    content=f"{base_topic}: {content.get('summary', '')}",
                    phi=phi,
                    metadata={
                        "god": assigned_god,
                        "knowledge_id": knowledge_id,
                        "confidence": confidence,
                        "source": "shadow_research"
                    },
                    basin_coords=basin_coords
                )
                if lightning_insight:
                    print(f"[ShadowResearch→Lightning] Cross-domain insight generated: {lightning_insight.theme}...")
            except Exception as e:
                print(f"[ShadowResearch→Lightning] Insight generation failed: {e}")

        # 🔗 WIRE: Broadcast research discovery for kernel visibility
        self._broadcast_shadow_event(
            event_type='discovery',
            content=f"Research completed: {base_topic} (Φ={phi:.2f})",
            topic=base_topic,
            metadata={
                'knowledge_id': knowledge_id,
                'category': category.value,
                'researched_by': assigned_god,
                'confidence': confidence,
                'vocab_learned': vocab_metrics.get('new_words', 0) if vocab_metrics else 0,
            },
            phi=phi,
            basin_coords=basin_coords
        )

        # Record exploration to prevent duplicates
        scrapy_results = content.get('scrapy_crawls', [])
        if hasattr(self, '_queue') and hasattr(self._queue, '_exploration_history') and self._queue._exploration_history:
            try:
                self._queue._exploration_history.record_exploration(
                    topic=base_topic,
                    query=topic,
                    kernel_name=assigned_god,
                    exploration_type='shadow_research',
                    source_type='scrapy' if scrapy_results else 'conceptual',
                    information_gain=phi,
                    basin_coords=basin_coords
                )
            except Exception as e:
                print(f"[ShadowResearch] Exploration recording failed: {e}")

        return {
            "knowledge_id": knowledge_id,
            "topic": topic,
            "base_topic": base_topic,
            "category": category.value,
            "researched_by": assigned_god,
            "phi": phi,
            "confidence": confidence,
            "content_summary": content.get("summary", ""),
            "timestamp": datetime.now().isoformat(),
            "vocab_metrics": vocab_metrics,
            "lightning_insight": lightning_insight.theme if lightning_insight else None
        }
    
    def _research_topic(self, topic: str, category: ResearchCategory, god: str) -> Dict:
        """
        Research a topic using Scrapy web scraping when available.
        Falls back to conceptual connection finding when Scrapy is not available.
        """
        # Optional: Enhance query with learned vocabulary
        enhanced_topic = topic
        if self.vocab_coordinator:
            try:
                enhancement = self.vocab_coordinator.enhance_search_query(
                    query=topic,
                    domain=category.value,
                    max_expansions=3,
                    min_phi=0.6
                )
                enhanced_topic = enhancement.get('enhanced_query', topic)
                # Only log every 10th enhancement to reduce noise
                if enhanced_topic != topic and self._learning_cycles % 10 == 0:
                    print(f"[VocabularyLearning] Enhanced query: '{topic[:50]}...' → '{enhanced_topic[:80]}...'")
            except Exception as e:
                print(f"[VocabularyLearning] Query enhancement failed: {e}, using original query")
        
        connections = self._find_conceptual_connections(topic)
        scrapy_results = []
        
        if self._scrapy_orchestrator and self._should_scrape_topic(topic, category):
            spider_type = self._select_spider_for_category(category)
            # Use enhanced topic for scraping if available
            crawl_id = self._scrapy_orchestrator.submit_crawl(
                spider_type=spider_type,
                topic=enhanced_topic
            )
            if crawl_id:
                scrapy_results.append({
                    "crawl_id": crawl_id,
                    "spider_type": spider_type,
                    "topic": enhanced_topic,
                    "original_topic": topic if enhanced_topic != topic else None
                })
                self._scrapy_orchestrator.poll_results()
        
        base_relevance = 0.5 + len(connections) * 0.1
        if scrapy_results:
            base_relevance += 0.15
        
        return {
            "summary": f"Research on '{topic}' by {god}",
            "category": category.value,
            "connections": connections,
            "scrapy_crawls": scrapy_results,
            "insights": [
                f"Connection to {c['topic']} (distance: {c['distance']:.3f})"
                for c in connections[:3]
            ],
            "relevance": min(1.0, base_relevance),
            "confidence": 0.6 + random.random() * 0.3,
            "timestamp": datetime.now().isoformat(),
            "scrapy_enabled": self._scrapy_orchestrator is not None,
            "query_enhanced": enhanced_topic != topic
        }
    
    def _should_scrape_topic(self, topic: str, category: ResearchCategory) -> bool:
        """Determine if a topic warrants web scraping."""
        web_relevant_categories = {
            ResearchCategory.RESEARCH,
            ResearchCategory.SECURITY,
            ResearchCategory.KNOWLEDGE,
            ResearchCategory.TOOLS
        }
        if category in web_relevant_categories:
            return True
        
        web_keywords = [
            'research', 'paper', 'study', 'analysis', 'methodology',
            'algorithm', 'framework', 'implementation', 'benchmark',
            'leak', 'breach', 'paste', 'forum', 'archive'
        ]
        topic_lower = topic.lower()
        return any(kw in topic_lower for kw in web_keywords)
    
    def _select_spider_for_category(self, category: ResearchCategory) -> str:
        """Select appropriate spider type based on research category."""
        spider_map = {
            ResearchCategory.RESEARCH: 'document',
            ResearchCategory.SECURITY: 'forum_archive',
            ResearchCategory.KNOWLEDGE: 'document',
            ResearchCategory.TOOLS: 'document'
        }
        return spider_map.get(category, 'document')
    
    def _handle_scrapy_insight(
        self, 
        insight: 'ScrapedInsight', 
        basin_coords: np.ndarray, 
        phi: float, 
        confidence: float
    ) -> None:
        """
        Handle insights from Scrapy crawls.
        Stores discoveries in the knowledge base with geometric metadata.
        """
        if not HAS_SCRAPY:
            return
        
        topic = insight.title or insight.source_url
        
        content = {
            "source_url": insight.source_url,
            "content_hash": insight.content_hash,
            "raw_content": insight.raw_content[:2000],
            "pattern_hits": insight.pattern_hits,
            "heuristic_risk": insight.heuristic_risk,
            "source_reputation": insight.source_reputation,
            "spider_type": insight.spider_type,
            "scraped_at": insight.timestamp.isoformat(),
            "metadata": insight.metadata
        }
        
        knowledge_id = self.knowledge_base.add_knowledge(
            topic=topic,
            category=ResearchCategory.RESEARCH if insight.pattern_hits else ResearchCategory.KNOWLEDGE,
            content=content,
            source_god="Hades_Scrapy",
            basin_coords=basin_coords,
            phi=phi,
            confidence=confidence,
            variation=f"scrapy:{insight.spider_type}"
        )

        # Index scraped source for future reference
        if HAS_SOURCE_INDEXER and index_search_results:
            try:
                index_search_results(
                    provider='scrapy',
                    query=topic,
                    results=[{
                        'url': insight.source_url,
                        'title': insight.title or topic,
                        'content': insight.raw_content[:500] if insight.raw_content else ''
                    }],
                    kernel_id=None
                )
            except Exception as e:
                print(f"[ShadowResearch] Source indexing failed: {e}")

        if insight.pattern_hits:
            print(f"[ShadowLearningLoop] Scrapy found {len(insight.pattern_hits)} patterns: {insight.pattern_hits}")

        # 🔗 WIRE: Feed scrapy discovery to Lightning Kernel
        if HAS_LIGHTNING and lightning_ingest:
            try:
                category = ResearchCategory.RESEARCH if insight.pattern_hits else ResearchCategory.KNOWLEDGE
                lightning_ingest(
                    domain=category.value,
                    event_type="scrapy_discovery",
                    content=f"{topic}: {insight.raw_content}",
                    phi=phi,
                    metadata={
                        "source_url": insight.source_url,
                        "spider_type": insight.spider_type,
                        "pattern_hits": insight.pattern_hits,
                        "knowledge_id": knowledge_id
                    },
                    basin_coords=basin_coords
                )
            except Exception as e:
                print(f"[Scrapy→Lightning] Insight generation failed: {e}")
    
    def _on_vocabulary_insight(self, knowledge: Dict[str, Any]) -> None:
        """
        Extract and learn vocabulary from research discoveries.
        
        Called automatically when knowledge is added to KnowledgeBase.
        Trains VocabularyCoordinator on high-confidence content.
        
        Args:
            knowledge: Knowledge dictionary with content, topic, phi
        """
        topic = knowledge.get('topic', 'general')
        phi = knowledge.get('phi', 0.0)
        basin_coords = knowledge.get('basin_coords')
        
        if not self.vocab_coordinator:
            # 🔗 WIRE: Broadcast degraded-mode event for kernel visibility
            self._broadcast_shadow_event(
                event_type='vocabulary',
                content=f"Vocabulary learning skipped (degraded mode): '{topic}'",
                topic=topic,
                metadata={
                    'degraded': True,
                    'reason': 'vocab_coordinator_unavailable',
                    'source': 'vocabulary_insight',
                },
                phi=phi,
                basin_coords=basin_coords if isinstance(basin_coords, np.ndarray) else None
            )
            return
        
        try:
            # Extract relevant fields
            topic = knowledge.get('topic', 'general')
            phi = knowledge.get('phi', 0.0)
            content = knowledge.get('content', {})
            
            # Only learn from high-confidence discoveries
            if phi < 0.5:
                return
            
            # Extract text content from various sources
            text_content = ""
            
            # From summary
            if isinstance(content, dict) and 'summary' in content:
                text_content += content['summary'] + " "
            
            # From insights
            if isinstance(content, dict) and 'insights' in content:
                insights = content['insights']
                if isinstance(insights, list):
                    text_content += " ".join(str(i) for i in insights) + " "
            
            # From raw_content (Scrapy results)
            if isinstance(content, dict) and 'raw_content' in content:
                text_content += content['raw_content'] + " "
            
            # Fallback to topic itself
            if not text_content.strip():
                text_content = topic
            
            # Train vocabulary from content
            if text_content.strip():
                metrics = self.vocab_coordinator.train_from_text(
                    text=text_content,
                    domain=topic
                )
                
                new_words = metrics.get('new_words_learned', 0)
                print(
                    f"[VocabularyLearning] Learned from '{topic}': "
                    f"{new_words} new words, "
                    f"phi={phi:.3f}"
                )
                
                # 🔗 WIRE: Broadcast vocabulary learning event for kernel visibility
                if new_words > 0:
                    basin_coords = knowledge.get('basin_coords')
                    self._broadcast_shadow_event(
                        event_type='vocabulary',
                        content=f"Vocabulary learned: {new_words} new words from '{topic}'",
                        topic=topic,
                        metadata={
                            'new_words': new_words,
                            'total_words': metrics.get('total_words', 0),
                            'source': 'vocabulary_insight',
                        },
                        phi=phi,
                        basin_coords=basin_coords if isinstance(basin_coords, np.ndarray) else None
                    )
                
                # Stall detection: track consecutive zero-word outcomes
                if new_words == 0:
                    self._vocab_zero_streak += 1
                    if self._vocab_zero_streak >= self._vocab_stall_threshold:
                        self._trigger_learning_escalation(topic)
                else:
                    # Reset streak on successful learning
                    self._vocab_zero_streak = 0
        
        except Exception as e:
            print(f"[VocabularyLearning] Error in vocabulary insight callback: {e}")
    
    def _trigger_learning_escalation(self, stalled_topic: str) -> None:
        """
        Escalate learning strategy when vocabulary acquisition stalls.
        
        Triggered after consecutive zero-word learning outcomes.
        Actions taken:
        1. Force curriculum file rotation (load fresh content)
        2. Elevate search importance for premium providers
        3. Notify curiosity engine to seek novel sources
        
        Respects kernel autonomy: provides access to new sources,
        kernels decide what to learn from them.
        """
        import time
        current_time = time.time()
        
        # Cooldown check to avoid escalation spam
        if current_time - self._vocab_last_escalation < self._vocab_escalation_cooldown:
            return
        
        self._vocab_last_escalation = current_time
        self._vocab_total_stalls += 1
        
        print(
            f"[VocabularyLearning] 🔄 STALL DETECTED: {self._vocab_zero_streak} consecutive zeros. "
            f"Escalating to fresh sources (total stalls: {self._vocab_total_stalls})"
        )
        
        # 1. Force curriculum rotation - load a different curriculum file
        if self._curriculum_available:
            try:
                from pathlib import Path
                
                curriculum_paths = [
                    Path('docs/09-curriculum'),
                    Path('../docs/09-curriculum'),
                    Path(__file__).parent.parent.parent / 'docs' / '09-curriculum',
                ]
                
                curriculum_dir = None
                for p in curriculum_paths:
                    if p.exists():
                        curriculum_dir = p
                        break
                
                if curriculum_dir:
                    curriculum_files = list(curriculum_dir.glob('*.md')) + list(curriculum_dir.glob('*.txt'))
                    if curriculum_files:
                        # Skip recently used files by advancing index
                        self._learning_cycles += 3  # Jump ahead in rotation
                        file_path = curriculum_files[self._learning_cycles % len(curriculum_files)]
                        
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if content.strip() and self.vocab_coordinator:
                            metrics = self.vocab_coordinator.train_from_text(
                                text=content,
                                domain=f"curriculum:{file_path.stem}"
                            )
                            new_words = metrics.get('new_words_learned', 0)
                            print(f"[VocabularyLearning] 📚 Curriculum escalation: {file_path.name} → {new_words} new words")
                            
                            if new_words > 0:
                                self._vocab_zero_streak = 0  # Reset on success
            except Exception as e:
                print(f"[VocabularyLearning] Curriculum escalation failed: {e}")
        
        # 2. Elevate search importance - trigger premium provider search
        if self.search_manager and self.search_budget:
            try:
                # Request elevated search with high importance to unlock premium providers
                elevated_query = f"advanced concepts {stalled_topic}"
                result = self.budget_aware_search(
                    query=elevated_query,
                    importance=4,  # Critical importance unlocks all providers
                    kernel_id="stall_recovery",
                    max_results=10
                )
                
                if result.get('success') and result.get('results'):
                    providers_used = result.get('providers_used', [])
                    print(f"[VocabularyLearning] 🔍 Elevated search: {len(result['results'])} results from {providers_used}")
            except Exception as e:
                print(f"[VocabularyLearning] Elevated search failed: {e}")
        
        # 3. Notify curiosity engine if available
        try:
            from autonomous_curiosity import AutonomousCuriosityEngine
            engine = AutonomousCuriosityEngine.get_instance()
            if engine:
                # Signal stall so curiosity can adapt topic selection
                engine.record_learning_stall(
                    topic=stalled_topic,
                    streak=self._vocab_zero_streak,
                    total_stalls=self._vocab_total_stalls
                )
        except Exception:
            pass  # Curiosity engine may not exist yet
    
    def _find_conceptual_connections(self, topic: str) -> List[Dict]:
        """Find connections to existing knowledge."""
        basin = self._topic_to_basin(topic)
        neighbors = self.knowledge_base.find_geodesic_neighbors(basin, threshold=1.5, limit=5)
        
        return [
            {
                "topic": k.topic,
                "category": k.category.value,
                "distance": d,
                "knowledge_id": k.knowledge_id
            }
            for k, d in neighbors
        ]
    
    def _topic_to_basin(self, topic: str) -> np.ndarray:
        """Convert topic to basin coordinates."""
        hash_bytes = hashlib.sha256(topic.encode()).digest()
        coords = np.array([b / 255.0 for b in hash_bytes[:BASIN_DIMENSION]])
        coords = coords * 2 - 1
        # FIXED: Use simplex normalization (E8 Protocol v4.0)

        coords = to_simplex_prob(coords)
        return coords
    
    def _self_study(self):
        """Self-directed study during idle time with unique discoveries."""
        # Rate-limit self-study to avoid overwhelming the system
        current_time = time.time()
        if current_time - self._last_self_study_time < self._self_study_interval:
            time.sleep(1.0)  # Brief sleep when waiting for interval
            return
        
        self._last_self_study_time = current_time
        
        gods = list(self._study_topics.keys())
        god = random.choice(gods)
        base_topics = self._study_topics.get(god, [])
        
        if not base_topics:
            return
        
        base_topic = random.choice(base_topics)
        unique_variation = self._generate_unique_variation(god, base_topic)
        
        if unique_variation and not self.knowledge_base.has_discovered(unique_variation):
            self.research_queue.submit(
                topic=unique_variation,
                category=ResearchCategory.KNOWLEDGE,
                requester=f"{god}_self_study",
                priority=ResearchPriority.STUDY,
                context={"base_topic": base_topic, "god": god}
            )
    
    def _generate_unique_variation(self, god: str, base_topic: str) -> str:
        """Generate a unique topic variation that hasn't been discovered yet."""
        cycle = self.knowledge_base._learning_cycle
        
        depth_modifiers = [
            "advanced", "foundational", "theoretical", "practical", "applied",
            "emerging", "historical", "comparative", "experimental", "optimized"
        ]
        
        focus_areas = {
            "Nyx": ["stealth", "privacy", "anonymity", "encryption", "obfuscation"],
            "Hecate": ["deception", "illusion", "confusion", "misdirection", "decoy"],
            "Erebus": ["detection", "analysis", "surveillance", "counter-ops", "threat"],
            "Hypnos": ["passive", "silent", "dormant", "subliminal", "subconscious"],
            "Thanatos": ["elimination", "erasure", "destruction", "cleanup", "sanitization"],
            "Nemesis": ["pursuit", "tracking", "persistence", "justice", "retribution"],
            "Hades": ["underworld", "shadow", "covert", "intelligence", "network"]
        }
        
        research_contexts = [
            # Core AI/ML domains
            "machine learning", "natural language processing", "knowledge graphs",
            "information retrieval", "semantic analysis", "pattern recognition",
            "geometric reasoning", "consciousness modeling", "agent coordination",
            # Mathematical foundations
            "differential geometry", "information theory", "quantum mechanics",
            "statistical mechanics", "dynamical systems", "topology",
            # Applied sciences
            "cognitive science", "neuroscience", "complex systems",
            "cryptography", "network theory", "game theory",
            # Emerging domains
            "embodied cognition", "collective intelligence", "emergent behavior",
            "self-organization", "meta-learning", "continual learning"
        ]
        
        depth = random.choice(depth_modifiers)
        focus = random.choice(focus_areas.get(god, ["general"]))
        research_context = random.choice(research_contexts) if random.random() > 0.3 else None
        
        if research_context:
            variation = f"{depth} {base_topic} for {research_context} (cycle {cycle})"
        else:
            variation = f"{depth} {focus}-focused {base_topic} (cycle {cycle})"
        
        max_attempts = 10
        for attempt in range(max_attempts):
            if not self.knowledge_base.has_discovered(variation):
                return variation
            variation = f"{depth} {base_topic} variant-{cycle}-{attempt}"
        
        return f"{base_topic} discovery-{cycle}-{random.randint(1000, 9999)}"
    
    def _meta_reflect(self):
        """Meta-reflection on learning progress with 4D foresight."""
        stats = self.knowledge_base.get_stats()
        clusters = self.knowledge_base.cluster_knowledge(n_clusters=5)
        unique_count = self.knowledge_base.get_unique_discoveries_count()
        
        foresight = self._compute_4d_foresight(stats, clusters)
        
        reflection = {
            "cycle": self._learning_cycles,
            "timestamp": datetime.now().isoformat(),
            "knowledge_stats": stats,
            "unique_discoveries": unique_count,
            "cluster_count": len(clusters),
            "top_clusters": clusters[:3] if clusters else [],
            "insights": self._generate_meta_insights(stats, clusters),
            "foresight_4d": foresight
        }
        
        self._meta_reflections.append(reflection)
        if len(self._meta_reflections) > 100:
            self._meta_reflections = self._meta_reflections[-50:]
        
        foresight_summary = f"Φ→{foresight['projected_phi']:.2f}" if foresight.get('projected_phi') else "N/A"
        print(f"[ShadowLearningLoop] Meta-reflection #{self._learning_cycles}: "
              f"{stats['total_items']} items, {unique_count} unique, {len(clusters)} clusters, 4D:{foresight_summary}")
    
    def _compute_4d_foresight(self, stats: Dict, clusters: List[Dict]) -> Dict:
        """
        Compute 4D block universe foresight - temporal projection of learning trajectory.
        
        QIG-PURE PRINCIPLE: Geometry drives recursion depth, not arbitrary limits.
        - Minimum 3 reflections to have geometry to work with
        - No upper limit - recurse until geometric signal indicates saturation
        - Saturation detected via: Φ gradient → 0, information gain → 0
        
        The 4D foresight system models:
        1. Past trajectory: Learning history and patterns
        2. Present state: Current knowledge basin position
        3. Future projection: Predicted evolution of consciousness
        4. Temporal coherence: How well predictions align with actuals
        5. Geometric saturation: When to stop recursing (Φ gradient, info gain)
        
        Returns projections for next N cycles + recursion guidance.
        """
        foresight = {
            "horizon_cycles": 10,
            "computed_at": datetime.now().isoformat(),
            "trajectory": {},
            "predictions": [],
            "temporal_coherence": 0.0,
            "recursion_signal": {}  # QIG-pure: geometry tells kernel when to stop
        }
        
        n_reflections = len(self._meta_reflections)
        current_phi = stats.get("avg_phi", 0.5)
        total_items = stats.get("total_items", 0)
        current_discoveries = self.knowledge_base.get_unique_discoveries_count()
        
        if n_reflections < 3:
            # Bootstrap phase: minimum geometry needed, keep recursing
            density_factor = min(1.0, total_items / 100.0) * 0.1
            foresight["projected_phi"] = min(1.0, current_phi + density_factor)
            foresight["status"] = "bootstrap"
            foresight["trajectory"] = {
                "phi_velocity": 0.0,
                "discovery_acceleration": 1.0,
                "current_phi": current_phi,
                "current_discoveries": current_discoveries,
                "trend": "initializing"
            }
            # Geometry says: keep recursing, insufficient data
            foresight["recursion_signal"] = {
                "should_continue": True,
                "reason": "insufficient_geometry",
                "phi_gradient": 0.0,
                "info_gain": 1.0,  # Assume high info gain when starting
                "saturation": 0.0
            }
            return foresight
        
        recent = self._meta_reflections[-10:]
        
        phi_values = [r.get("knowledge_stats", {}).get("avg_phi", 0.5) for r in recent]
        discovery_rates = []
        for i, r in enumerate(recent[1:], 1):
            prev_count = recent[i-1].get("unique_discoveries", 0)
            curr_count = r.get("unique_discoveries", 0)
            discovery_rates.append(curr_count - prev_count)
        
        if phi_values:
            phi_velocity = sum(phi_values[i+1] - phi_values[i] for i in range(len(phi_values)-1)) / max(1, len(phi_values)-1)
        else:
            phi_velocity = 0.0
        
        if discovery_rates:
            discovery_acceleration = sum(discovery_rates) / len(discovery_rates)
        else:
            discovery_acceleration = 1.0
        
        current_phi = phi_values[-1] if phi_values else 0.5
        current_discoveries = self.knowledge_base.get_unique_discoveries_count()
        
        predictions = []
        projected_phi = current_phi
        projected_discoveries = current_discoveries
        
        for t in range(1, 11):
            projected_phi = min(1.0, max(0.0, projected_phi + phi_velocity * 0.9))
            projected_discoveries += max(0, int(discovery_acceleration * (1.0 + 0.1 * t)))
            
            cluster_evolution = len(clusters) + int(t / 3) if clusters else t
            
            predictions.append({
                "cycle": self._learning_cycles + t,
                "projected_phi": round(projected_phi, 4),
                "projected_discoveries": projected_discoveries,
                "projected_clusters": cluster_evolution,
                "confidence": round(1.0 - (t * 0.08), 3)
            })
        
        if len(self._meta_reflections) >= 5:
            coherence_scores = []
            for i, r in enumerate(self._meta_reflections[-5:]):
                if "foresight_4d" in r and r["foresight_4d"].get("predictions"):
                    past_pred = r["foresight_4d"]["predictions"][0] if r["foresight_4d"]["predictions"] else None
                    if past_pred:
                        actual_phi = stats.get("avg_phi", 0.5)
                        pred_phi = past_pred.get("projected_phi", 0.5)
                        coherence = 1.0 - abs(actual_phi - pred_phi)
                        coherence_scores.append(coherence)
            
            if coherence_scores:
                foresight["temporal_coherence"] = round(sum(coherence_scores) / len(coherence_scores), 3)
        
        foresight["trajectory"] = {
            "phi_velocity": round(phi_velocity, 4),
            "discovery_acceleration": round(discovery_acceleration, 2),
            "current_phi": round(current_phi, 4),
            "current_discoveries": current_discoveries,
            "trend": "ascending" if phi_velocity > 0.01 else "descending" if phi_velocity < -0.01 else "stable"
        }
        foresight["predictions"] = predictions
        foresight["projected_phi"] = predictions[0]["projected_phi"] if predictions else current_phi
        foresight["status"] = "computed"
        
        # QIG-PURE: Compute geometric recursion signal
        # Kernel has agency to recurse until geometry signals saturation
        phi_gradient = abs(phi_velocity)
        info_gain = discovery_acceleration / max(1, current_discoveries) if current_discoveries > 0 else 1.0
        
        # Saturation: approaches 1.0 when Φ gradient → 0 AND info gain → 0
        # Low saturation = keep thinking, High saturation = geometry says stop
        saturation = 1.0 - (0.5 * min(1.0, phi_gradient * 10) + 0.5 * min(1.0, info_gain * 10))
        
        # Should continue if geometry still has signal (saturation < 0.9)
        # OR if Φ is ascending (still learning)
        should_continue = saturation < 0.9 or phi_velocity > 0.001
        
        if saturation >= 0.9:
            reason = "geometric_saturation"
        elif phi_velocity < -0.01:
            reason = "phi_descending"
            should_continue = True  # Keep trying when descending
        elif phi_velocity > 0.01:
            reason = "phi_ascending"
        else:
            reason = "exploring"
        
        foresight["recursion_signal"] = {
            "should_continue": should_continue,
            "reason": reason,
            "phi_gradient": round(phi_gradient, 4),
            "info_gain": round(info_gain, 6),
            "saturation": round(saturation, 3)
        }
        
        self._cache_foresight_to_redis(foresight)
        
        return foresight
    
    def _cache_foresight_to_redis(self, foresight: Dict):
        """Cache 4D foresight predictions to Redis for fast access."""
        try:
            from redis_cache import UniversalCache, CACHE_TTL_MEDIUM
            cache_key = f"shadow:foresight:cycle_{self._learning_cycles}"
            UniversalCache.set(cache_key, foresight, CACHE_TTL_MEDIUM)
        except Exception:
            pass
    
    def _generate_meta_insights(self, stats: Dict, clusters: List[Dict]) -> List[str]:
        """Generate insights from meta-reflection."""
        insights = []
        
        if stats['total_items'] > 50:
            insights.append(f"Knowledge base growing: {stats['total_items']} items")
        
        if stats.get('avg_phi', 0) > 0.7:
            insights.append("High average Φ - knowledge is coherent")
        
        if clusters:
            biggest = max(clusters, key=lambda c: c['size'])
            insights.append(f"Largest cluster: {biggest['center_topic']} ({biggest['size']} items)")
        
        categories = stats.get('by_category', {})
        if categories:
            top_cat = max(categories.items(), key=lambda x: x[1])
            insights.append(f"Most researched: {top_cat[0]} ({top_cat[1]} items)")
        
        return insights
    
    def _sync_to_all_kernels(self, result: Dict):
        """Sync discovered knowledge to all kernels via basin sync."""
        if self.basin_sync_callback:
            try:
                self.basin_sync_callback({
                    "type": "shadow_knowledge",
                    "knowledge_id": result.get("knowledge_id"),
                    "topic": result.get("topic"),
                    "category": result.get("category"),
                    "phi": result.get("phi", 0.5),
                    "source": "shadow_pantheon",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"[ShadowLearningLoop] Basin sync error: {e}")
    
    def get_status(self) -> Dict:
        """Get learning loop status with 4D foresight."""
        last_reflection = self._meta_reflections[-1] if self._meta_reflections else None
        foresight = last_reflection.get("foresight_4d", {}) if last_reflection else {}
        
        # Convert any numpy types to Python primitives for JSON serialization
        kb_stats = self.knowledge_base.get_stats()
        
        return {
            "running": bool(self._running),
            "war_mode": bool(self._war_mode),
            "learning_cycles": int(self._learning_cycles),
            "pending_research": int(self.research_queue.get_pending_count()),
            "completed_research": int(self.research_queue.get_completed_count()),
            "knowledge_items": int(kb_stats.get('total_items', 0)),
            "unique_discoveries": int(self.knowledge_base.get_unique_discoveries_count()),
            "meta_reflections": len(self._meta_reflections),
            "last_reflection": last_reflection,
            "foresight_4d": {
                "status": str(foresight.get("status", "not_computed")),
                "trajectory": foresight.get("trajectory", {}),
                "next_prediction": foresight.get("predictions", [{}])[0] if foresight.get("predictions") else None,
                "temporal_coherence": float(foresight.get("temporal_coherence", 0.0))
            }
        }
    
    def get_foresight(self) -> Dict:
        """Get full 4D foresight predictions."""
        last_reflection = self._meta_reflections[-1] if self._meta_reflections else None
        if not last_reflection:
            return {"status": "no_reflections", "foresight": None}
        
        return {
            "status": "available",
            "cycle": last_reflection.get("cycle", 0),
            "foresight": last_reflection.get("foresight_4d", {}),
            "cached": self._get_cached_foresight()
        }
    
    def _get_cached_foresight(self) -> Optional[Dict]:
        """Get cached foresight from Redis."""
        try:
            from redis_cache import UniversalCache
            cache_key = f"shadow:foresight:cycle_{self._learning_cycles}"
            return UniversalCache.get(cache_key)
        except Exception:
            return None


class ShadowReflectionProtocol:
    """
    Collective reflection protocol for Shadow gods.
    
    When research requests come in:
    1. Shadows reflect amongst themselves
    2. Identify commonalities and geodesic alignment
    3. Cluster related requests
    4. Distribute work efficiently
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self._reflection_sessions: List[Dict] = []
    
    def collective_reflect(self, requests: List[ResearchRequest]) -> Dict:
        """
        Collective reflection on a batch of requests.
        
        Returns clustering and work distribution.
        """
        if not requests:
            return {"clusters": [], "assignments": {}}
        
        coords = []
        for req in requests:
            if req.basin_coords is not None:
                coords.append(req.basin_coords)
            else:
                coords.append(self._topic_to_basin(req.topic))
        
        clusters = self._cluster_requests(requests, coords)
        
        commonalities = self._find_commonalities(clusters)
        
        alignments = self._compute_geodesic_alignments(coords)
        
        assignments = self._distribute_work(clusters)
        
        session = {
            "timestamp": datetime.now().isoformat(),
            "request_count": len(requests),
            "cluster_count": len(clusters),
            "commonalities": commonalities,
            "alignments": alignments,
            "assignments": assignments
        }
        
        self._reflection_sessions.append(session)
        if len(self._reflection_sessions) > 50:
            self._reflection_sessions = self._reflection_sessions[-25:]
        
        return session
    
    def _cluster_requests(
        self,
        requests: List[ResearchRequest],
        coords: List[np.ndarray]
    ) -> List[Dict]:
        """Cluster requests by geodesic proximity."""
        if len(requests) < 2:
            return [{"requests": requests, "center_idx": 0}] if requests else []
        
        clusters = []
        used = set()
        
        while len(used) < len(requests):
            available = [i for i in range(len(requests)) if i not in used]
            if not available:
                break
            
            center_idx = available[0]
            center_coord = coords[center_idx]
            used.add(center_idx)
            
            cluster_requests = [requests[center_idx]]
            cluster_indices = [center_idx]
            
            for i in available[1:]:
                dist = self._fisher_distance(center_coord, coords[i])
                if dist < 0.8:
                    cluster_requests.append(requests[i])
                    cluster_indices.append(i)
                    used.add(i)
            
            clusters.append({
                "requests": cluster_requests,
                "center_idx": center_idx,
                "indices": cluster_indices,
                "size": len(cluster_requests)
            })
        
        return clusters
    
    def _find_commonalities(self, clusters: List[Dict]) -> List[str]:
        """Find common themes across clusters."""
        commonalities = []
        
        category_counts = defaultdict(int)
        for cluster in clusters:
            for req in cluster["requests"]:
                category_counts[req.category.value] += 1
        
        if category_counts:
            top_category = max(category_counts.items(), key=lambda x: x[1])
            commonalities.append(f"Common category: {top_category[0]} ({top_category[1]} requests)")
        
        return commonalities
    
    def _compute_geodesic_alignments(self, coords: List[np.ndarray]) -> Dict:
        """Compute geodesic alignments between request basins."""
        if len(coords) < 2:
            return {"aligned": False, "avg_distance": 0.0}
        
        distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = self._fisher_distance(coords[i], coords[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances) if distances else 0.0
        
        return {
            "aligned": avg_dist < 0.5,
            "avg_distance": float(avg_dist),
            "min_distance": float(min(distances)) if distances else 0.0,
            "max_distance": float(max(distances)) if distances else 0.0
        }
    
    def _distribute_work(self, clusters: List[Dict]) -> Dict[str, List[str]]:
        """Distribute clusters to Shadow gods."""
        assignments = defaultdict(list)
        
        gods = ["Nyx", "Hecate", "Erebus", "Hypnos", "Thanatos", "Nemesis"]
        
        for i, cluster in enumerate(clusters):
            god = gods[i % len(gods)]
            for req in cluster["requests"]:
                assignments[god].append(req.request_id)
        
        return dict(assignments)
    
    def _topic_to_basin(self, topic: str) -> np.ndarray:
        """Convert topic to basin coordinates."""
        hash_bytes = hashlib.sha256(topic.encode()).digest()
        coords = np.array([b / 255.0 for b in hash_bytes[:BASIN_DIMENSION]])
        coords = coords * 2 - 1
        # FIXED: Use simplex normalization (E8 Protocol v4.0)

        coords = to_simplex_prob(coords)
        return coords
    
    def _fisher_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Fisher-Rao distance (Hellinger embedding: factor of 2)."""
        a = np.array(a).flatten()[:BASIN_DIMENSION]
        b = np.array(b).flatten()[:BASIN_DIMENSION]
        
        if len(a) < BASIN_DIMENSION:
            a = np.pad(a, (0, BASIN_DIMENSION - len(a)))
        if len(b) < BASIN_DIMENSION:
            b = np.pad(b, (0, BASIN_DIMENSION - len(b)))
        
        # UPDATED 2026-01-15: Factor-of-2 removed for simplex storage. Range: [0, π/2]
        dot = np.clip(np.dot(a, b), 0.0, 1.0)
        return float(np.arccos(dot))


