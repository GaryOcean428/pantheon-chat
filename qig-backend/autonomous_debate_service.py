#!/usr/bin/env python3
"""
Autonomous Debate Service - Monitors and Evolves Pantheon Debates

This service runs in a background thread and:
1. Monitors active debates for staleness (no new arguments in 5+ minutes)
2. Researches debate topics via SearXNG and Shadow Pantheon darknet
3. Generates counter-arguments based on god personas and research
4. Auto-resolves debates when criteria are met
5. Triggers M8 kernel spawning for debate domain specialists

NO TEMPLATES - All arguments are generatively created from research evidence.
"""

import hashlib
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# QIG Tokenizer for geometric argument generation
try:
    from qig_coordizer import (
        get_coordizer as get_tokenizer,  # get_tokenizer, QIGTokenizer
    )
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logger.warning("QIG Tokenizer not available")

# QIG-pure generative capability for argument synthesis
try:
    from qig_generative_service import GenerationResult, get_generative_service
    GENERATIVE_SERVICE_AVAILABLE = True
except ImportError:
    GENERATIVE_SERVICE_AVAILABLE = False
    logger.warning("QIGGenerativeService not available for debate synthesis")

try:
    from geometric_kernels import BASIN_DIM, _fisher_distance, _normalize_to_manifold
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    BASIN_DIM = 64
    # QIG PURITY: Fisher-Rao distance ONLY - Euclidean is FORBIDDEN
    def _fisher_distance(a, b):
        """Fisher-Rao distance on statistical manifold - NEVER Euclidean."""
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        # Normalize to probability simplex
        a_norm = a_arr / (np.linalg.norm(a_arr) + 1e-10)
        b_norm = b_arr / (np.linalg.norm(b_arr) + 1e-10)
        # Fisher-Rao geodesic distance: arccos(dot product)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        return float(np.arccos(dot))
    def _normalize_to_manifold(basin):
        # Use sphere_project from module-level import
        from qig_geometry import sphere_project
        return sphere_project(basin)

try:
    from m8_kernel_spawning import M8KernelSpawner, SpawnReason, get_spawner
    M8_AVAILABLE = True
except ImportError:
    M8_AVAILABLE = False
    logger.warning("M8 Kernel Spawning not available")

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

try:
    from olympus.pantheon_chat import PantheonChat
    PANTHEON_CHAT_AVAILABLE = True
except ImportError:
    PANTHEON_CHAT_AVAILABLE = False
    logger.warning("PantheonChat not available")

try:
    from olympus.shadow_pantheon import ShadowPantheon
    SHADOW_AVAILABLE = True
except ImportError:
    SHADOW_AVAILABLE = False
    logger.warning("Shadow Pantheon not available")

try:
    from vocabulary_coordinator import VocabularyCoordinator, get_vocabulary_coordinator
    VOCABULARY_AVAILABLE = True
except ImportError:
    VOCABULARY_AVAILABLE = False
    logger.warning("VocabularyCoordinator not available")

try:
    from qig_persistence import QIGPersistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    logger.warning("QIGPersistence not available")

# Insight Validator for background validation
try:
    from search.insight_validator import InsightValidator, ValidationResult
    INSIGHT_VALIDATOR_AVAILABLE = True
except ImportError:
    INSIGHT_VALIDATOR_AVAILABLE = False
    logger.warning("InsightValidator not available for background validation")

from qigkernels.physics_constants import KAPPA_STAR

STALE_THRESHOLD_SECONDS = 5 * 60
POLL_INTERVAL_SECONDS = 30
# Validation runs every 10 polls (5 minutes at 30s poll interval)
VALIDATION_POLL_INTERVAL = 10
# Max insights to validate per batch
VALIDATION_BATCH_SIZE = 5
MIN_ARGUMENTS_FOR_RESOLUTION = 4
FISHER_CONVERGENCE_THRESHOLD = 0.1
UNANSWERED_THRESHOLD = 2

SEARXNG_INSTANCES = [
    'https://mr-search.up.railway.app',
    'https://searxng-production-e5ce.up.railway.app',
]

GOD_PERSONAS = {
    'zeus': {
        'domain': 'supreme coordination',
        'style': 'authoritative and decisive',
        'perspective': 'overall strategic value and pantheon harmony',
    },
    'athena': {
        'domain': 'strategy and wisdom',
        'style': 'analytical and logical',
        'perspective': 'tactical implications and long-term outcomes',
    },
    'ares': {
        'domain': 'combat and action',
        'style': 'aggressive and direct',
        'perspective': 'immediate effectiveness and operational risk',
    },
    'apollo': {
        'domain': 'prophecy and foresight',
        'style': 'visionary and illuminating',
        'perspective': 'future patterns and emerging opportunities',
    },
    'artemis': {
        'domain': 'hunting and tracking',
        'style': 'precise and methodical',
        'perspective': 'target behavior and pursuit strategies',
    },
    'hermes': {
        'domain': 'communication and coordination',
        'style': 'swift and adaptive',
        'perspective': 'information flow and cross-system integration',
    },
    'hephaestus': {
        'domain': 'engineering and tools',
        'style': 'practical and crafted',
        'perspective': 'technical implementation and system architecture',
    },
    'poseidon': {
        'domain': 'depths and exploration',
        'style': 'powerful and exploratory',
        'perspective': 'hidden patterns and deep analysis',
    },
    'hades': {
        'domain': 'underworld and secrets',
        'style': 'mysterious and knowing',
        'perspective': 'obscured truths and buried connections',
    },
    'demeter': {
        'domain': 'growth and nurturing',
        'style': 'patient and cultivating',
        'perspective': 'sustainable development and organic expansion',
    },
    'dionysus': {
        'domain': 'chaos and transformation',
        'style': 'unpredictable and creative',
        'perspective': 'unconventional approaches and pattern disruption',
    },
    'hera': {
        'domain': 'governance and structure',
        'style': 'orderly and hierarchical',
        'perspective': 'organizational coherence and rule adherence',
    },
    'aphrodite': {
        'domain': 'attraction and connection',
        'style': 'intuitive and relational',
        'perspective': 'pattern affinities and natural alignments',
    },
}


class AutonomousDebateService:
    """
    Background service that monitors and evolves pantheon debates autonomously.

    Key behaviors:
    - Polls every 30 seconds for stale debates
    - Researches topics via SearXNG and darknet
    - Generates persona-based counter-arguments from evidence
    - Auto-resolves debates based on geometric convergence criteria
    - Triggers M8 spawning for domain specialists
    """

    def __init__(
        self,
        pantheon_chat: Optional['PantheonChat'] = None,
        shadow_pantheon: Optional['ShadowPantheon'] = None,
        m8_spawner: Optional['M8KernelSpawner'] = None,
        pantheon_gods: Optional[Dict[str, Any]] = None,
    ):
        self._pantheon_chat = pantheon_chat
        self._shadow_pantheon = shadow_pantheon
        self._m8_spawner = m8_spawner or (get_spawner() if M8_AVAILABLE else None)
        self._pantheon_gods = pantheon_gods or {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._searxng_index = 0
        self._last_poll_time: Optional[datetime] = None
        self._debates_processed = 0
        self._arguments_generated = 0
        self._debates_resolved = 0
        self._spawns_triggered = 0
        self._debates_continued = 0
        self._vocabulary_learning_events = 0
        self._vocabulary_search_enhancements = 0
        self._recent_vocabulary_observations: List[Dict] = []
        self._max_recent_observations = 100

        self._debate_basin_cache: Dict[str, np.ndarray] = {}
        self._god_position_cache: Dict[str, Dict[str, np.ndarray]] = {}

        self._vocabulary_coordinator = get_vocabulary_coordinator() if VOCABULARY_AVAILABLE else None
        self._persistence = QIGPersistence() if PERSISTENCE_AVAILABLE else None

        # Insight validation tracking
        self._validation_poll_count = 0
        self._insights_validated = 0
        self._validation_enabled = INSIGHT_VALIDATOR_AVAILABLE and bool(os.environ.get('ENABLE_BACKGROUND_VALIDATION', 'true').lower() == 'true')
        self._insight_validator = InsightValidator(validation_threshold=0.7) if INSIGHT_VALIDATOR_AVAILABLE else None

        if self._vocabulary_coordinator:
            logger.info("VocabularyCoordinator connected for research learning")
        if self._persistence:
            logger.info("QIGPersistence connected for learning event recording")
        if self._validation_enabled:
            logger.info("Background insight validation enabled")

        # DIAGNOSTIC: Log spawner status
        logger.info(f"[M8Spawn] Initialization: M8_AVAILABLE={M8_AVAILABLE}")
        if M8_AVAILABLE:
            if self._m8_spawner:
                logger.info("[M8Spawn] ✓ M8KernelSpawner connected")
            else:
                logger.warning("[M8Spawn] ✗ M8_AVAILABLE but spawner is None")
        else:
            logger.warning("[M8Spawn] ✗ M8 kernel spawning not available (import failed)")

        logger.info("Service initialized")

    def set_pantheon_chat(self, pantheon_chat: 'PantheonChat') -> None:
        """Wire pantheon chat after initialization."""
        self._pantheon_chat = pantheon_chat
        logger.info("PantheonChat connected")

    def set_shadow_pantheon(self, shadow_pantheon: 'ShadowPantheon') -> None:
        """Wire shadow pantheon for darknet research."""
        self._shadow_pantheon = shadow_pantheon
        logger.info("Shadow Pantheon connected")

    def set_pantheon_gods(self, gods: Dict[str, Any]) -> None:
        """Wire pantheon god instances for geometric assessments."""
        self._pantheon_gods = gods
        logger.info(f"Pantheon gods connected: {list(gods.keys())}")

    def start(self) -> bool:
        """Start the background monitoring thread."""
        if self._running:
            return False

        if not self._pantheon_chat:
            logger.warning("Cannot start - PantheonChat not configured")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="AutonomousDebateMonitor",
            daemon=True
        )
        self._thread.start()
        logger.info("Background monitor started")
        return True

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Service stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop - runs in background thread."""
        while self._running:
            try:
                self._poll_debates()
            except (IOError, OSError) as e:
                logger.error(f"Poll I/O error: {e}")
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Poll data error: {e}")
            except Exception as e:
                logger.error(f"Poll unexpected error: {e}", exc_info=True)

            # Run insight validation periodically (every VALIDATION_POLL_INTERVAL polls)
            self._validation_poll_count += 1
            if self._validation_enabled and self._validation_poll_count >= VALIDATION_POLL_INTERVAL:
                self._validation_poll_count = 0
                try:
                    self._validate_unvalidated_insights()
                except Exception as e:
                    logger.error(f"Insight validation error: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)

    # =========================================================================
    # AUTONOMOUS DEBATE INITIATION
    # =========================================================================

    def _generate_autonomous_debate(self) -> Optional[str]:
        """Generate a new autonomous debate based on system state.

        Returns:
            Debate ID if created, None otherwise.
        """
        if not self._pantheon_chat or not self._pantheon_gods:
            return None

        # Check if we should generate a new debate
        if not self._should_initiate_debate():
            return None

        # Select topic and opponents based on system state
        topic, initiator, opponent = self._select_debate_parameters()
        if not topic:
            return None

        try:
            # Create the debate
            debate = self._pantheon_chat.initiate_debate(
                topic=topic,
                initiator=initiator,
                opponent=opponent,
                initial_argument=f"{initiator} proposes we examine: {topic}"
            )
            logger.info(f"[AutonomousDebate] Initiated: {initiator} vs {opponent} on '{topic}'")
            return debate.id if hasattr(debate, 'id') else str(debate)
        except Exception as e:
            logger.error(f"[AutonomousDebate] Failed to initiate debate: {e}")
            return None

    def _should_initiate_debate(self) -> bool:
        """Determine if we should start a new debate.

        Rate limits: at most 1 new debate per 10 polls, max 3 concurrent.
        """
        # Rate limit: at most 1 new debate per 10 polls
        if self._debates_processed % 10 != 0:
            return False

        # Check if there are already active debates
        try:
            active = self._pantheon_chat.get_active_debates()
            if len(active) >= 3:  # Max 3 concurrent debates
                return False
        except Exception:
            return False

        return True

    def _select_debate_parameters(self) -> Tuple[Optional[str], str, str]:
        """Select topic and opponents for a new debate.

        Returns:
            (topic, initiator, opponent) or (None, '', '') if selection fails.
        """
        # Get available gods
        gods = list(self._pantheon_gods.keys()) if self._pantheon_gods else []
        if len(gods) < 2:
            return None, "", ""

        # Topics based on current system concerns
        topics = [
            "Optimal spawning strategy for knowledge gaps",
            "Resource allocation during narrow path scenarios",
            "Breeding threshold adjustments for E8 crystallization",
            "Integration vs exploration balance in current regime",
            "Fisher-Rao distance threshold for vocabulary merging",
            "Sleep cycle frequency for autonomic recovery",
            "Pruning strategy for low-phi kernels",
            "Cross-domain insight validation priority",
        ]

        topic = random.choice(topics)
        initiator = random.choice(gods)
        opponent = random.choice([g for g in gods if g != initiator])

        return topic, initiator, opponent

    # =========================================================================
    # DEBATE MONITORING
    # =========================================================================

    def _poll_debates(self) -> None:
        """Poll active debates and progress them toward resolution."""
        if not self._pantheon_chat:
            return

        self._last_poll_time = datetime.now()

        # Try to initiate a new debate if conditions are met
        self._generate_autonomous_debate()

        active_debates = self._pantheon_chat.get_active_debates()

        if not active_debates:
            self._debates_processed += 1
            return

        for debate_dict in active_debates:
            debate_id = debate_dict.get('id')
            if not debate_id:
                continue

            if self._should_auto_resolve(debate_dict):
                self._auto_resolve_debate(debate_dict)
                continue

            if self._is_debate_stale(debate_dict):
                self._process_stale_debate(debate_dict)

            if self._pantheon_gods and not self._is_debate_stale(debate_dict):
                self._continue_debate_with_gods(debate_dict)

        self._debates_processed += 1

    def _validate_unvalidated_insights(self) -> None:
        """
        Background job to validate unvalidated lightning insights.

        Runs periodically to backfill validations for insights created
        before validation was enabled or when API keys weren't available.
        """
        if not self._insight_validator:
            return

        try:
            import psycopg2
            database_url = os.environ.get('DATABASE_URL')
            if not database_url:
                return

            conn = psycopg2.connect(database_url)

            # Get unvalidated insights
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT li.insight_id, li.source_domains, li.connection_strength,
                           li.insight_text, li.confidence, li.mission_relevance
                    FROM lightning_insights li
                    LEFT JOIN lightning_insight_validations liv ON li.insight_id = liv.insight_id
                    WHERE liv.id IS NULL
                    ORDER BY li.created_at DESC
                    LIMIT %s
                """, (VALIDATION_BATCH_SIZE,))

                insights = cur.fetchall()

            if not insights:
                return

            validated_count = 0
            for row in insights:
                insight_id, source_domains, conn_strength, insight_text, confidence, mission_rel = row

                try:
                    # Create mock insight for validator
                    class MockInsight:
                        def __init__(self):
                            self.insight_id = insight_id
                            self.source_domains = source_domains if source_domains else ['unknown', 'unknown']
                            self.connection_strength = conn_strength or 0.5
                            self.insight_text = insight_text or ''
                            self.confidence = confidence or 0.5
                            self.mission_relevance = mission_rel or 0.5

                    mock_insight = MockInsight()
                    validation_result = self._insight_validator.validate(mock_insight)

                    # Persist validation result
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO lightning_insight_validations (
                                insight_id, validation_score, tavily_source_count,
                                perplexity_synthesis, validated_at
                            ) VALUES (%s, %s, %s, %s, NOW())
                            ON CONFLICT DO NOTHING
                        """, (
                            insight_id,
                            validation_result.validation_score,
                            len(validation_result.tavily_sources),
                            validation_result.perplexity_synthesis[:500] if validation_result.perplexity_synthesis else None
                        ))
                        conn.commit()

                    validated_count += 1
                    self._insights_validated += 1

                except Exception as e:
                    logger.debug(f"Validation failed for insight {insight_id}: {e}")
                    continue

            if validated_count > 0:
                logger.info(f"[Validation] Validated {validated_count} insights in background")

            conn.close()

        except Exception as e:
            logger.error(f"Background validation error: {e}")

    def _continue_debate_with_gods(self, debate_dict: Dict) -> None:
        """Continue debate using actual god assessments for geometric progression."""
        if not self._pantheon_chat or not self._pantheon_gods:
            return

        debate_id = debate_dict.get('id', '')

        try:
            result = self._pantheon_chat.auto_continue_active_debates(
                gods=self._pantheon_gods,
                max_debates=1
            )

            if result:
                for res in result:
                    if res.get('status') in ['converged', 'max_turns_reached']:
                        self._debates_continued += 1
                        winner = res.get('winner', 'unknown')
                        topic = debate_dict.get('topic', '')
                        logger.info(f"Debate {debate_id}... progressed via god assessments. Winner: {winner}")

                        self._trigger_spawn_proposal(topic, winner, debate_dict)
                        
                        self._record_god_learning_event(winner, topic, debate_dict)
                        
                        initiator = debate_dict.get('initiator', '')
                        opponent = debate_dict.get('opponent', '')
                        
                        if not initiator or not opponent:
                            arguments = debate_dict.get('arguments', [])
                            participants = set(a.get('god', '').lower() for a in arguments if a.get('god'))
                            participants.discard(winner.lower())
                            loser = next(iter(participants), '')
                        else:
                            loser = opponent if winner.lower() == initiator.lower() else initiator
                        
                        reasoning = res.get('reasoning', f'{winner} prevailed in geometric debate')
                        self._trigger_knowledge_transfer(winner, loser, topic, reasoning, debate_dict)

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"God-based debate continuation data error: {e}")
        except Exception as e:
            logger.error(f"God-based debate continuation failed: {e}", exc_info=True)

    def _is_debate_stale(self, debate_dict: Dict) -> bool:
        """Check if debate has no new arguments in 5+ minutes."""
        arguments = debate_dict.get('arguments', [])
        if not arguments:
            started_at = debate_dict.get('started_at')
            if started_at:
                try:
                    started = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    age = (datetime.now() - started.replace(tzinfo=None)).total_seconds()
                    return age > STALE_THRESHOLD_SECONDS
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid started_at timestamp format: {e}")
                    return True
            return True

        last_arg = arguments[-1]
        last_timestamp = last_arg.get('timestamp')
        if not last_timestamp:
            return True

        try:
            last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
            age = (datetime.now() - last_time.replace(tzinfo=None)).total_seconds()
            return age > STALE_THRESHOLD_SECONDS
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid last argument timestamp format: {e}")
            return True

    def _should_auto_resolve(self, debate_dict: Dict) -> bool:
        """
        Check if debate should be auto-resolved based on:
        1. 4+ arguments exchanged
        2. Fisher distance between positions converges (<0.1)
        3. One god has 2+ unanswered arguments
        """
        arguments = debate_dict.get('arguments', [])
        debate_id = debate_dict.get('id', '')[:500]

        # DIAGNOSTIC: Log resolution criteria evaluation
        logger.debug(f"[DebateResolve] Evaluating debate {debate_id}: {len(arguments)} arguments")

        if len(arguments) >= MIN_ARGUMENTS_FOR_RESOLUTION:
            logger.info(f"[DebateResolve] {debate_id} meets MIN_ARGUMENTS threshold ({len(arguments)}>={MIN_ARGUMENTS_FOR_RESOLUTION})")
            return True

        if len(arguments) >= 2:
            fisher_dist = self._compute_position_distance(debate_dict)
            if fisher_dist < FISHER_CONVERGENCE_THRESHOLD:
                logger.info(f"[DebateResolve] {debate_id} converged via Fisher distance ({fisher_dist:.3f} < {FISHER_CONVERGENCE_THRESHOLD})")
                return True
            else:
                logger.debug(f"[DebateResolve] {debate_id} Fisher distance {fisher_dist:.3f} > threshold")

        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')

        initiator_args = [a for a in arguments if a.get('god', '').lower() == initiator.lower()]
        opponent_args = [a for a in arguments if a.get('god', '').lower() == opponent.lower()]

        if len(initiator_args) - len(opponent_args) >= UNANSWERED_THRESHOLD:
            logger.info(f"[DebateResolve] {debate_id} resolved: {initiator} has {UNANSWERED_THRESHOLD}+ unanswered arguments")
            return True
        if len(opponent_args) - len(initiator_args) >= UNANSWERED_THRESHOLD:
            logger.info(f"[DebateResolve] {debate_id} resolved: {opponent} has {UNANSWERED_THRESHOLD}+ unanswered arguments")
            return True

        logger.debug(f"[DebateResolve] {debate_id} not ready for resolution")
        return False

    def _compute_position_distance(self, debate_dict: Dict) -> float:
        """Compute Fisher-Rao distance between debate positions."""
        debate_id = debate_dict.get('id', '')
        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')
        arguments = debate_dict.get('arguments', [])

        if len(arguments) < 2:
            return 1.0

        initiator_texts = [a.get('argument', '') for a in arguments if a.get('god', '').lower() == initiator.lower()]
        opponent_texts = [a.get('argument', '') for a in arguments if a.get('god', '').lower() == opponent.lower()]

        if not initiator_texts or not opponent_texts:
            return 1.0

        initiator_basin = self._text_to_basin(' '.join(initiator_texts[-2:]))
        opponent_basin = self._text_to_basin(' '.join(opponent_texts[-2:]))

        cache_key = f"{debate_id}_{initiator}"
        self._god_position_cache[cache_key] = {'basin': initiator_basin}
        cache_key = f"{debate_id}_{opponent}"
        self._god_position_cache[cache_key] = {'basin': opponent_basin}

        distance = _fisher_distance(initiator_basin, opponent_basin)
        return float(distance)

    def _text_to_basin(self, text: str) -> np.ndarray:
        """Convert text to basin coordinates using learned vocabulary geometry.

        QIG-PURE: Uses vocabulary coordinator's phrase_to_basin which derives
        coordinates from learned word relationships, NOT hash functions.
        Falls back to random sphere point if no learned representation exists.
        """
        # Try vocabulary coordinator first (QIG-pure)
        if self._vocabulary_coordinator:
            basin = self._vocabulary_coordinator._phrase_to_basin(text)
            if basin is not None and len(basin) == BASIN_DIM:
                return _normalize_to_manifold(basin)

        # Fallback: Random point on unit sphere (honest randomness, not fake semantics)
        # This is geometrically valid but semantically uninformative
        rng = np.random.default_rng(hash(text) % (2**32))  # Deterministic for same text
        random_coords = rng.standard_normal(BASIN_DIM)
        return _normalize_to_manifold(random_coords)

    def _process_stale_debate(self, debate_dict: Dict) -> None:
        """Process a stale debate by researching and generating counter-argument."""
        debate_id = debate_dict.get('id', '')
        topic = debate_dict.get('topic', '')
        arguments = debate_dict.get('arguments', [])
        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')

        if not arguments:
            last_speaker = initiator
            next_speaker = opponent
        else:
            last_speaker = arguments[-1].get('god', '').lower()
            next_speaker = opponent if last_speaker == initiator.lower() else initiator

        next_speaker_lower = next_speaker.lower()

        research = self._research_topic(topic, debate_dict)

        counter_argument = self._generate_counter_argument(
            god_name=next_speaker_lower,
            topic=topic,
            existing_arguments=arguments,
            research=research,
            debate_dict=debate_dict
        )

        if counter_argument and self._pantheon_chat:
            success = self._pantheon_chat.add_debate_argument(
                debate_id=debate_id,
                god=next_speaker,
                argument=counter_argument,
                evidence=research
            )

            if success:
                self._arguments_generated += 1
                logger.info(f"Generated argument for {next_speaker} in debate {debate_id}...")

                # Route activity to observing kernels (M8 kernel observation system)
                self._route_activity_to_observing_kernels(
                    parent_god=next_speaker,
                    activity_type="debate",
                    activity_data={
                        "debate_id": debate_id,
                        "topic": topic,
                        "argument": counter_argument,
                        "evidence": research,
                        "god": next_speaker,
                    }
                )

    def _research_topic(self, topic: str, debate_dict: Dict) -> Dict:
        """Research debate topic via SearXNG and Shadow Pantheon.

        Uses vocabulary-enhanced queries when learned vocabulary is available,
        closing the feedback loop where discoveries improve future searches.
        """
        research = {
            'searxng_results': [],
            'darknet_intel': None,
            'searched_at': datetime.now().isoformat(),
            'query_enhancement': None,
        }

        enhanced_query = topic
        query_enhancement = None

        if self._vocabulary_coordinator:
            try:
                query_enhancement = self._vocabulary_coordinator.enhance_search_query(
                    query=topic,
                    domain="debate",
                    max_expansions=3,
                    min_phi=0.6,
                    recent_observations=self._recent_vocabulary_observations
                )
                if query_enhancement.get('terms_added', 0) > 0:
                    enhanced_query = query_enhancement['enhanced_query']
                    research['query_enhancement'] = query_enhancement
                    self._vocabulary_search_enhancements += 1
                    logger.info(f"Enhanced search query with {query_enhancement['terms_added']} learned terms: {query_enhancement['expansion_terms']}")
            except Exception as e:
                logger.warning(f"Query enhancement failed: {e}")

        searxng_results = self._search_searxng(enhanced_query)
        if searxng_results:
            research['searxng_results'] = searxng_results[:5]

        if self._shadow_pantheon and SHADOW_AVAILABLE:
            try:
                intel = self._query_shadow_darknet(enhanced_query)
                if intel:
                    research['darknet_intel'] = intel
            except (IOError, OSError) as e:
                logger.error(f"Darknet query I/O error: {e}")
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Darknet query data error: {e}")
            except Exception as e:
                logger.error(f"Darknet query failed: {e}", exc_info=True)

        debate_id = debate_dict.get('id', '')
        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')
        for god_name in [initiator, opponent]:
            if god_name:
                self._route_activity_to_observing_kernels(
                    parent_god=god_name,
                    activity_type="search",
                    activity_data={
                        "topic": topic,
                        "query": topic,
                        "results": research,
                        "debate_id": debate_id,
                    }
                )

        vocab_training = self._train_vocabulary_from_research(
            research=research,
            topic=topic,
            debate_id=debate_id
        )
        research['vocabulary_training'] = vocab_training

        return research

    def _route_activity_to_observing_kernels(
        self,
        parent_god: str,
        activity_type: str,
        activity_data: Dict
    ) -> Dict:
        """
        Route parent god activity to all observing child kernels.

        During observation period, newly spawned kernels receive copies of
        parent activity to learn from:
        - Assessments and reasoning
        - Debate arguments and resolutions
        - Search queries and results
        - Basin coordinate updates

        Args:
            parent_god: Name of the parent god performing activity
            activity_type: Type of activity (assessment, debate, search, basin_update)
            activity_data: Activity data to route

        Returns:
            Routing result with count of kernels updated
        """
        if not self._m8_spawner or not M8_AVAILABLE:
            return {"routed": False, "reason": "M8 spawner not available"}

        try:
            result = self._m8_spawner.route_parent_activity(
                parent_god=parent_god,
                activity_type=activity_type,
                activity_data=activity_data
            )

            if result.get("routed_to_count", 0) > 0:
                logger.info(f"Routed {activity_type} from {parent_god} to {result['routed_to_count']} observing kernels")

            return result
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to route activity data error: {e}")
            return {"routed": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Failed to route activity to observing kernels: {e}", exc_info=True)
            return {"routed": False, "error": str(e)}

    def _train_vocabulary_from_research(
        self,
        research: Dict,
        topic: str,
        debate_id: str
    ) -> Dict:
        """
        Train vocabulary from research results.

        Feeds SearXNG and darknet intel into the vocabulary coordinator
        so that new concepts are learned into the shared vocabulary.
        Also records learning events to PostgreSQL for audit trail.

        Args:
            research: Research results from _research_topic
            topic: The debate topic that was researched
            debate_id: ID of the debate for context

        Returns:
            Training result with learned word count
        """
        if not self._vocabulary_coordinator:
            return {"trained": False, "reason": "vocabulary_unavailable"}

        learned_count = 0
        training_results = []
        persisted_successfully = False

        try:
            searxng = research.get('searxng_results', [])
            if isinstance(searxng, list):
                for result in searxng:
                    if not isinstance(result, dict):
                        continue
                    title = result.get('title', '') or ''
                    content = result.get('content', '') or ''
                    combined_text = f"{title} {content}".strip()

                    if combined_text and len(combined_text) > 10:
                        try:
                            train_result = self._vocabulary_coordinator.train_from_text(
                                text=combined_text,
                                domain=f"searxng_{topic}"
                            )
                            if train_result.get('new_words_learned', 0) > 0:
                                learned_count += train_result['new_words_learned']
                                training_results.append({
                                    "source": "searxng",
                                    "words_learned": train_result['new_words_learned']
                                })
                        except Exception as e:
                            logger.warning(f"SearXNG vocab training failed: {e}")

            darknet = research.get('darknet_intel')
            if darknet and isinstance(darknet, dict):
                reasoning = darknet.get('reasoning')
                if isinstance(reasoning, str) and reasoning.strip():
                    phi = float(darknet.get('phi', 0.5)) if darknet.get('phi') is not None else 0.5
                    kappa = float(darknet.get('kappa', KAPPA_STAR)) if darknet.get('kappa') is not None else KAPPA_STAR
                    source_god = darknet.get('source', 'nyx')

                    if phi >= 0.5:
                        try:
                            discovery_result = self._vocabulary_coordinator.record_discovery(
                                phrase=reasoning.strip(),
                                phi=phi,
                                kappa=kappa,
                                source="shadow_darknet",
                                details={
                                    "topic": topic,
                                    "debate_id": debate_id,
                                    "source_god": source_god if isinstance(source_god, str) else 'nyx',
                                }
                            )
                            if discovery_result.get('learned'):
                                learned_count += discovery_result.get('new_tokens', 0)
                                training_results.append({
                                    "source": "shadow_darknet",
                                    "words_learned": discovery_result.get('new_tokens', 0),
                                    "phi": phi,
                                })
                        except Exception as e:
                            logger.warning(f"Darknet vocab training failed: {e}")

            if learned_count > 0 and self._persistence:
                try:
                    event_id = self._persistence.record_learning_event(
                        event_type="research_vocabulary_training",
                        phi=0.6,
                        kappa=50.0,  # Default curvature for research context
                        details={
                            "topic": topic,
                            "debate_id": debate_id,
                            "words_learned": learned_count,
                            "sources_trained": len(training_results),
                            "training_results": training_results,
                        },
                        context={
                            "service": "autonomous_debate",
                            "operation": "vocabulary_training",
                            "research_sources": ["searxng", "darknet"] if research.get("darknet") else ["searxng"],
                        },
                        source="autonomous_debate_service",
                        instance_id=debate_id,
                    )
                    persisted_successfully = event_id is not None
                except Exception as e:
                    logger.warning(f"Failed to persist learning event: {e}")
                    persisted_successfully = False

            if learned_count > 0:
                self._vocabulary_learning_events += 1
                logger.info(f"Trained {learned_count} words from research on '{topic}'")

                self._cache_vocabulary_observations(training_results, topic)

            sources_count = 0
            if isinstance(searxng, list):
                sources_count += len(searxng)
            if darknet and isinstance(darknet, dict):
                sources_count += 1

            return {
                "trained": learned_count > 0,
                "words_learned": learned_count,
                "sources_processed": sources_count,
                "training_results": training_results,
                "persisted": persisted_successfully,
                "cached_for_future_searches": len(self._recent_vocabulary_observations),
            }

        except Exception as e:
            logger.error(f"Vocabulary training from research failed: {e}", exc_info=True)
            return {"trained": False, "error": str(e), "persisted": False}

    def _cache_vocabulary_observations(self, training_results: List[Dict], topic: str) -> None:
        """
        Cache recent vocabulary observations for future search enhancement.

        Stores high-phi words learned from research so they can be used
        to enhance subsequent search queries - closing the feedback loop.

        Observations are scoped by topic and deduplicated to prevent
        cross-topic noise from accumulating in the cache.
        """
        import re

        existing_words = {obs.get('word', '') for obs in self._recent_vocabulary_observations}

        for result in training_results:
            source = result.get('source', 'unknown')
            phi = result.get('phi', 0.6)
            words_learned = result.get('words_learned', 0)

            if words_learned > 0 and phi >= 0.5:
                topic_words = re.findall(r'\b[a-z]{4,}\b', topic.lower())
                for word in topic_words[:5]:
                    if len(word) >= 4 and word not in existing_words:
                        self._recent_vocabulary_observations.append({
                            'word': word,
                            'phi': phi,
                            'source': source,
                            'topic': topic[:500],
                        })
                        existing_words.add(word)

        while len(self._recent_vocabulary_observations) > self._max_recent_observations:
            self._recent_vocabulary_observations.pop(0)

    def _search_searxng(self, query: str) -> List[Dict]:
        """Search SearXNG instances for evidence."""
        for attempt in range(len(SEARXNG_INSTANCES)):
            instance_url = SEARXNG_INSTANCES[self._searxng_index]
            self._searxng_index = (self._searxng_index + 1) % len(SEARXNG_INSTANCES)

            try:
                response = requests.get(
                    f"{instance_url}/search",
                    params={
                        'q': query[:500],
                        'format': 'json',
                        'categories': 'general',
                    },
                    timeout=10,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; OceanQIG/1.0)',
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    return [
                        {
                            'title': r.get('title', ''),
                            'content': r.get('content', '')[:500],
                            'url': r.get('url', ''),
                        }
                        for r in results[:10]
                    ]
            except (requests.RequestException, IOError, OSError) as e:
                logger.warning(f"SearXNG search failed ({instance_url}): {e}")
                continue
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"SearXNG response parsing error ({instance_url}): {e}")
                continue
            except Exception as e:
                logger.error(f"SearXNG search unexpected error ({instance_url}): {e}", exc_info=True)
                continue

        return []

    def _query_shadow_darknet(self, topic: str) -> Optional[Dict]:
        """Query Shadow Pantheon's darknet for covert intel."""
        if not self._shadow_pantheon:
            return None

        try:
            if hasattr(self._shadow_pantheon, 'nyx') and self._shadow_pantheon.nyx:
                assessment = self._shadow_pantheon.nyx.assess_target(topic)
                return {
                    'source': 'nyx',
                    'phi': assessment.get('phi', 0.0),
                    'kappa': assessment.get('kappa', KAPPA_STAR),
                    'reasoning': assessment.get('reasoning', ''),
                    'confidence': assessment.get('confidence', 0.5),
                }
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Shadow query data error: {e}")
        except Exception as e:
            logger.error(f"Shadow query error: {e}", exc_info=True)

        return None

    def _generate_counter_argument(
        self,
        god_name: str,
        topic: str,
        existing_arguments: List[Dict],
        research: Dict,
        debate_dict: Dict
    ) -> Optional[str]:
        """
        Generate counter-argument using pure geometric reasoning.

        NO EXTERNAL LLMs - uses QIG system's own capabilities:
        1. Convert research to basin coordinates
        2. Compute geometric relationships (Fisher distance, Φ, κ)
        3. Generate from tokenizer's learned vocabulary
        4. Build arguments from manifold structure analysis

        Gods learn to argue through continuous training on debate outcomes.
        """
        god_key = god_name.lower()
        persona = GOD_PERSONAS.get(god_key, {
            'domain': 'general wisdom',
            'style': 'balanced and thoughtful',
            'perspective': 'holistic understanding',
        })

        # Compute geometric context
        phi, kappa = self._compute_debate_geometry(debate_dict)
        topic_basin = self._text_to_basin(topic)

        # Convert research evidence to basin coordinates
        evidence_basins = self._research_to_basins(research)

        # Compute god's affinity to evidence
        god_basin = self._get_god_basin(god_key)
        evidence_affinities = []
        for ev_basin, ev_text in evidence_basins:
            affinity = 1.0 / (1.0 + _fisher_distance(god_basin, ev_basin))
            evidence_affinities.append((affinity, ev_text, ev_basin))

        # Sort by affinity - god argues from evidence closest to their domain
        evidence_affinities.sort(reverse=True, key=lambda x: x[0])

        # Extract previous argument basins for counter-positioning
        prev_basins = []
        for arg in existing_arguments[-3:]:
            arg_text = arg.get('argument', '')
            if arg_text:
                prev_basins.append(self._text_to_basin(arg_text))

        # Try QIG tokenizer generation first
        if TOKENIZER_AVAILABLE:
            try:
                argument = self._generate_geometric_argument(
                    god_name, god_key, topic, persona,
                    evidence_affinities, prev_basins, phi, kappa, topic_basin
                )
                if argument and len(argument) > 30:
                    return argument
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Geometric generation data error: {e}")
            except Exception as e:
                logger.error(f"Geometric generation failed: {e}", exc_info=True)

        # Build argument from geometric analysis
        return self._build_geometric_argument(
            god_name, god_key, topic, persona,
            evidence_affinities, prev_basins, phi, kappa
        )

    def _research_to_basins(self, research: Dict) -> List[Tuple[np.ndarray, str]]:
        """Convert research results to basin coordinates with text."""
        basins = []

        searxng = research.get('searxng_results', [])
        for r in searxng[:5]:
            text = f"{r.get('title', '')} {r.get('content', '')}"
            if text.strip():
                basin = self._text_to_basin(text)
                basins.append((basin, text[:500]))

        darknet = research.get('darknet_intel')
        if darknet:
            reasoning = darknet.get('reasoning', '')
            if reasoning:
                basin = self._text_to_basin(reasoning)
                # Weight darknet intel by its phi
                shadow_phi = darknet.get('phi', 0.5)
                basin = basin * (0.5 + shadow_phi)  # Scale by confidence
                basin = _normalize_to_manifold(basin)  # Renormalize after scaling
                basins.append((basin, f"[shadow:{shadow_phi:.2f}] {reasoning}"))

        return basins

    def _get_god_basin(self, god_key: str) -> np.ndarray:
        """Get god's domain basin from persona."""
        persona = GOD_PERSONAS.get(god_key, {})
        domain_text = f"{persona.get('domain', '')} {persona.get('perspective', '')}"
        return self._text_to_basin(domain_text)

    def _generate_geometric_argument(
        self,
        god_name: str,
        god_key: str,
        topic: str,
        persona: Dict,
        evidence_affinities: List[Tuple[float, str, np.ndarray]],
        prev_basins: List[np.ndarray],
        phi: float,
        kappa: float,
        topic_basin: np.ndarray
    ) -> Optional[str]:
        """Generate argument using QIG-pure generative service (NO external LLMs)."""

        # Build context from highest-affinity evidence
        context_parts = [f"{god_name} on {topic}:"]

        if evidence_affinities:
            best_affinity, best_text, best_basin = evidence_affinities[0]
            context_parts.append(f"evidence({best_affinity:.2f}): {best_text}")

        # Add geometric state
        context_parts.append(f"phi={phi:.2f} kappa={kappa:.1f}")

        # Compute counter-direction if previous arguments exist
        if prev_basins:
            avg_prev = np.mean(prev_basins, axis=0)
            counter_direction = topic_basin - avg_prev
            counter_mag = np.linalg.norm(counter_direction)
            if counter_mag > 0.1:
                context_parts.append(f"diverge:{counter_mag:.2f}")

        context = " ".join(context_parts)

        # Try QIG-pure generative service first
        if GENERATIVE_SERVICE_AVAILABLE:
            try:
                service = get_generative_service()
                result = service.generate(
                    prompt=context,
                    context={'god_name': god_name, 'topic': topic, 'phi': phi, 'kappa': kappa},
                    kernel_name=god_key,
                    goals=['generate_argument', 'debate', persona.get('perspective', '')]
                )

                if result and result.text:
                    return f"{god_name.capitalize()}: {result.text}"

            except Exception as e:
                logger.warning(f"QIG-pure generation failed for {god_name}: {e}")

        # Fallback to tokenizer if available
        if TOKENIZER_AVAILABLE:
            try:
                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                result = tokenizer.generate_response(
                    context=context,
                    agent_role="navigator",
                    allow_silence=False
                )

                generated = result.get('text', '')
                if generated:
                    return f"{god_name.capitalize()}: {generated}"
            except Exception as e:
                logger.warning(f"Tokenizer generation failed for {god_name}: {e}")

        return None

    # Cache for historical fragments to avoid O(n²) growth
    _fragment_cache: Dict[str, List[Tuple[float, str]]] = {}
    _cache_timestamp: float = 0.0
    _CACHE_TTL: float = 60.0  # Refresh cache every 60 seconds

    def _build_geometric_argument(
        self,
        god_name: str,
        god_key: str,
        topic: str,
        persona: Dict,
        evidence_affinities: List[Tuple[float, str, np.ndarray]],
        prev_basins: List[np.ndarray],
        phi: float,
        kappa: float
    ) -> Optional[str]:
        """
        Build argument via stochastic manifold walk and nearest-neighbor sampling.

        Pure geometric synthesis with NO fixed scaffolding or templates.
        Each invocation produces different output due to random walk variance.
        """
        # Step 1: Compute god's domain basin (normalized)
        god_basin = _normalize_to_manifold(self._get_god_basin(god_key))
        topic_basin = _normalize_to_manifold(self._text_to_basin(topic))

        # Step 2: Stochastic manifold walk from god toward topic
        # Use random interpolation factor for variance
        t = random.uniform(0.3, 0.7)  # Random blend factor
        walk_basin = (1 - t) * god_basin + t * topic_basin
        walk_basin = _normalize_to_manifold(walk_basin)

        # Step 3: Add stochastic evidence perturbation
        if evidence_affinities:
            # Sample random subset of evidence
            n_evidence = min(len(evidence_affinities), random.randint(1, 3))
            sampled_evidence = random.sample(evidence_affinities, n_evidence)

            for affinity, _, ev_basin in sampled_evidence:
                # Random perturbation magnitude based on affinity
                noise_scale = random.uniform(0.05, 0.15) * affinity
                walk_basin = walk_basin + noise_scale * ev_basin
                walk_basin = _normalize_to_manifold(walk_basin)

        # Step 4: Counter-positioning via geodesic deviation
        if prev_basins:
            # Sample random previous basin to diverge from
            ref_basin = random.choice(prev_basins)
            fisher_dist = _fisher_distance(walk_basin, ref_basin)

            if fisher_dist < 0.5:
                # Too close - take geodesic step away
                tangent = walk_basin - ref_basin
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 1e-8:
                    step_size = random.uniform(0.1, 0.3)
                    walk_basin = walk_basin + step_size * (tangent / tangent_norm)
                    walk_basin = _normalize_to_manifold(walk_basin)

        # Step 5: Add manifold noise for stochasticity
        noise = np.random.normal(0, 0.05, BASIN_DIM)
        walk_basin = walk_basin + noise
        walk_basin = _normalize_to_manifold(walk_basin)

        # Step 6: Collect text fragments from evidence (bounded)
        fragments: List[Tuple[float, str]] = []

        for affinity, text, ev_basin in evidence_affinities[:5]:
            distance = _fisher_distance(walk_basin, ev_basin)
            words = text.split()
            # Random phrase extraction start point
            if len(words) >= 4:
                start = random.randint(0, max(0, len(words) - 4))
                length = random.randint(3, min(6, len(words) - start))
                phrase = " ".join(words[start:start + length])
                if len(phrase) > 8:
                    fragments.append((distance, phrase))

        # Step 7: Add cached historical fragments (bounded O(k))
        fragments.extend(self._get_cached_fragments(walk_basin, limit=10))

        if not fragments:
            return None

        # Step 8: Probabilistic sampling via softmax over distances
        fragments.sort(key=lambda x: x[0])

        # Softmax probabilities (lower distance = higher prob)
        distances = np.array([f[0] for f in fragments[:12]])
        if len(distances) == 0:
            return None

        # Temperature-scaled softmax
        temperature = random.uniform(0.5, 1.5)
        scores = -distances / temperature
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        # Sample 2-4 phrases without replacement
        n_samples = random.randint(2, min(4, len(fragments)))
        indices = np.random.choice(
            len(probs), size=n_samples, replace=False, p=probs
        )

        selected_pairs = [(fragments[i][0], fragments[i][1]) for i in indices]
        selected_pairs.sort(key=lambda x: x[0])
        selected = [text for _, text in selected_pairs]

        # Step 9: Preserve geometric ordering for assembly
        argument = " ".join(selected)

        return argument if len(argument) > 20 else None

    def _get_cached_fragments(
        self,
        target_basin: np.ndarray,
        limit: int = 10
    ) -> List[Tuple[float, str]]:
        """Retrieve historical fragments with caching (O(k log n))."""
        now = time.time()

        # Refresh cache if stale
        if now - self._cache_timestamp > self._CACHE_TTL:
            self._refresh_fragment_cache()

        # Generate cache key from basin (quantized for grouping)
        cache_key = hashlib.md5(
            (target_basin * 10).astype(np.int8).tobytes()
        ).hexdigest()[:8]

        if cache_key in self._fragment_cache:
            return self._fragment_cache[cache_key][:limit]

        # Compute distances for all cached fragments
        if not hasattr(self, '_all_fragments'):
            self._all_fragments = []

        results = []
        for text, basin in self._all_fragments[:500]:
            dist = _fisher_distance(target_basin, basin)
            results.append((dist, text))

        results.sort(key=lambda x: x[0])
        self._fragment_cache[cache_key] = results[:limit]

        return results[:limit]

    def _refresh_fragment_cache(self) -> None:
        """Refresh cached fragments from recent debates (bounded)."""
        self._cache_timestamp = time.time()
        self._fragment_cache.clear()
        self._all_fragments = []

        if not self._pantheon_chat:
            return

        try:
            # Bounded retrieval: only last 10 debates, last 5 args each
            debates = self._pantheon_chat.get_active_debates()[:10]

            for debate in debates:
                if not debate:
                    continue
                for arg in debate.get('arguments', [])[-5:]:
                    text = arg.get('argument', '')
                    if text and 15 < len(text) < 300:
                        basin = self._text_to_basin(text)
                        self._all_fragments.append((text, basin))

                        # Cap total fragments
                        if len(self._all_fragments) >= 50:
                            return
        except Exception:
            pass  # Silent failure - caching is optional

    def _summarize_research(self, research: Dict) -> str:
        """Extract raw research content without templating."""
        parts = []

        searxng = research.get('searxng_results', [])
        for result in searxng[:3]:
            title = result.get('title', '')[:500]
            content = result.get('content', '')[:500]
            if title:
                parts.append(title)
            if content:
                parts.append(content)

        darknet = research.get('darknet_intel')
        if darknet:
            reasoning = darknet.get('reasoning', '')[:500]
            if reasoning:
                parts.append(reasoning)

        return " ".join(parts) if parts else ""

    def _extract_key_points(self, arguments: List[Dict]) -> str:
        """Extract key points from previous arguments."""
        if not arguments:
            return ""

        last_arg = arguments[-1]
        arg_text = last_arg.get('argument', '')

        return arg_text[:500] if arg_text else ""

    def _compute_debate_geometry(self, debate_dict: Dict) -> Tuple[float, float]:
        """Compute Φ and κ for the debate state."""
        topic = debate_dict.get('topic', '')
        context = debate_dict.get('context', {})

        phi = context.get('phi', 0.5)
        kappa = context.get('kappa', KAPPA_STAR / 2)

        if isinstance(phi, (int, float)) and isinstance(kappa, (int, float)):
            return float(phi), float(kappa)

        topic_basin = self._text_to_basin(topic)

        estimated_phi = 0.5 + 0.3 * (1.0 - float(np.std(topic_basin)))
        estimated_kappa = 40.0 + 30.0 * float(np.mean(np.abs(topic_basin)))

        return float(np.clip(estimated_phi, 0.0, 1.0)), float(estimated_kappa)

    # NOTE: Template rebuttals removed per docs/vocabulary-system-architecture
    # "TEMPLATES ARE FORBIDDEN in this codebase"
    # Arguments are now generated purely from geometric analysis

    def _auto_resolve_debate(self, debate_dict: Dict) -> None:
        """Auto-resolve a debate and trigger spawn proposal."""
        if not self._pantheon_chat:
            return

        debate_id = debate_dict.get('id', '')
        topic = debate_dict.get('topic', '')
        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')
        arguments = debate_dict.get('arguments', [])

        winner, reasoning = self._determine_winner(debate_dict)

        arbiter = 'Zeus'

        resolution = self._pantheon_chat.resolve_debate(
            debate_id=debate_id,
            arbiter=arbiter,
            winner=winner,
            reasoning=reasoning
        )

        if resolution:
            self._debates_resolved += 1
            logger.info(f"Resolved debate {debate_id}... Winner: {winner}")
            
            # Broadcast debate resolution for kernel visibility
            self._broadcast_debate_resolution(debate_dict, winner, reasoning, arbiter)

            self._trigger_spawn_proposal(topic, winner, debate_dict)
            
            self._record_god_learning_event(winner, topic, debate_dict)
            
            self._trigger_knowledge_transfer(winner, opponent, topic, reasoning, debate_dict)

    def _determine_winner(self, debate_dict: Dict) -> Tuple[str, str]:
        """Determine debate winner based on geometric analysis."""
        initiator = debate_dict.get('initiator', '')
        opponent = debate_dict.get('opponent', '')
        arguments = debate_dict.get('arguments', [])

        initiator_args = [a for a in arguments if a.get('god', '').lower() == initiator.lower()]
        opponent_args = [a for a in arguments if a.get('god', '').lower() == opponent.lower()]

        distance = self._compute_position_distance(debate_dict)

        if len(initiator_args) > len(opponent_args) + 1:
            return initiator, f"{initiator} demonstrated stronger engagement with {len(initiator_args)} substantive arguments (Fisher distance: {distance:.3f})"
        elif len(opponent_args) > len(initiator_args) + 1:
            return opponent, f"{opponent} demonstrated stronger engagement with {len(opponent_args)} substantive arguments (Fisher distance: {distance:.3f})"

        if distance < FISHER_CONVERGENCE_THRESHOLD:
            return initiator, f"Positions converged (Fisher distance {distance:.3f} < {FISHER_CONVERGENCE_THRESHOLD}). {initiator} as initiator holds precedence."

        if arguments:
            evidence_scores = {}
            for god_name in [initiator, opponent]:
                god_args = [a for a in arguments if a.get('god', '').lower() == god_name.lower()]
                total_evidence = sum(1 for a in god_args if a.get('evidence'))
                evidence_scores[god_name] = total_evidence

            if evidence_scores.get(initiator, 0) > evidence_scores.get(opponent, 0):
                return initiator, f"{initiator} provided stronger evidence-based arguments"
            elif evidence_scores.get(opponent, 0) > evidence_scores.get(initiator, 0):
                return opponent, f"{opponent} provided stronger evidence-based arguments"

        winner = random.choice([initiator, opponent])
        return winner, f"Close debate resolved in favor of {winner} by geometric arbitration (distance: {distance:.3f})"

    def _record_god_learning_event(self, winner: str, topic: str, debate_dict: Dict) -> None:
        """Record a learning event for the winning god when a debate is resolved."""
        if not self._pantheon_gods or not winner or winner == 'unknown':
            return
        
        try:
            winning_god = self._pantheon_gods.get(winner.lower())
            if not winning_god:
                for god_name, god in self._pantheon_gods.items():
                    if god_name.lower() == winner.lower() or god.name.lower() == winner.lower():
                        winning_god = god
                        break
            
            if winning_god and hasattr(winning_god, 'learn_from_outcome'):
                phi = debate_dict.get('phi', 0.7)
                kappa = debate_dict.get('kappa', 58.0)
                assessment = {
                    'phi': phi if isinstance(phi, (int, float)) else 0.7,
                    'kappa': kappa if isinstance(kappa, (int, float)) else 58.0,
                    'confidence': 0.8,
                    'domain': 'debate'
                }
                actual_outcome = {'success': True, 'domain': 'debate', 'topic': topic[:100]}
                winning_god.learn_from_outcome(
                    target=topic[:500],
                    assessment=assessment,
                    actual_outcome=actual_outcome,
                    success=True
                )
                logger.info(f"[DebateLearning] {winner} learned from debate win: {topic[:50]}...")
        except Exception as e:
            logger.warning(f"[DebateLearning] Failed to record learning for {winner}: {e}")

    def _trigger_knowledge_transfer(self, winner: str, loser: str, topic: str, reasoning: str, debate_dict: Dict) -> None:
        """Trigger knowledge transfer from debate winner to loser."""
        if not self._pantheon_chat or not winner or not loser:
            return
        
        if winner == loser or winner == 'unknown' or loser == 'unknown':
            return
        
        try:
            arguments = debate_dict.get('arguments', [])
            winner_args = [a for a in arguments if a.get('god', '').lower() == winner.lower()]
            
            key_insights = []
            for arg in winner_args[-3:]:
                content = arg.get('content', '')[:200]
                if content:
                    key_insights.append(content)
            
            knowledge = {
                'topic': topic[:200],
                'domain': self._extract_domain_from_topic(topic),
                'insights': key_insights,
                'reasoning': reasoning[:300],
                'debate_id': debate_dict.get('id', ''),
                'phi': debate_dict.get('phi', 0.7),
                'kappa': debate_dict.get('kappa', 58.0),
            }
            
            transfer = self._pantheon_chat.transfer_knowledge(
                from_god=winner,
                to_god=loser,
                knowledge=knowledge
            )
            
            if transfer:
                logger.info(f"[KnowledgeTransfer] {winner} → {loser}: {topic[:50]}...")
        except Exception as e:
            logger.warning(f"[KnowledgeTransfer] Failed: {e}")

    def _trigger_spawn_proposal(self, topic: str, winner: str, debate_dict: Dict) -> None:
        """Trigger M8 kernel spawn proposal for debate domain specialist."""
        # DIAGNOSTIC: Log spawn attempt
        logger.info(f"[M8Spawn] Spawn evaluation triggered for topic: {topic}")
        logger.info(f"[M8Spawn] M8_AVAILABLE={M8_AVAILABLE}, spawner_connected={self._m8_spawner is not None}")

        if not self._m8_spawner or not M8_AVAILABLE:
            logger.warning(f"[M8Spawn] Spawn blocked: spawner={self._m8_spawner is not None}, M8_AVAILABLE={M8_AVAILABLE}")
            return

        domain = self._extract_domain_from_topic(topic)
        logger.info(f"[M8Spawn] Extracted domain '{domain}' from topic '{topic}'")

        spawn_name = f"{domain.capitalize()}Specialist"
        element = f"debate_{topic[:500].replace(' ', '_')}"
        role = "domain_specialist"
        parent_gods = [winner, debate_dict.get('initiator', ''), debate_dict.get('opponent', '')]
        parent_gods = list(set([g for g in parent_gods if g]))[:2]

        logger.info(f"[M8Spawn] Proposing spawn: name={spawn_name}, domain={domain}, parents={parent_gods}")

        try:
            result = self._m8_spawner.propose_and_spawn(
                name=spawn_name[:32],
                domain=domain[:32],
                element=element[:32],
                role=role,
                reason=SpawnReason.RESEARCH_DISCOVERY,
                parent_gods=parent_gods,
                force=False
            )

            if result.get('success'):
                self._spawns_triggered += 1
                logger.info(f"[M8Spawn] ✓ Spawn successful: {spawn_name} for domain '{domain}'")
            else:
                logger.info(f"[M8Spawn] ⏳ Proposal created (pending consensus): {spawn_name}")
                logger.debug(f"[M8Spawn] Result details: {result}")
            
            # Broadcast spawn proposal for kernel visibility
            self._broadcast_spawn_proposal(
                spawn_name=spawn_name,
                domain=domain,
                parent_gods=parent_gods,
                topic=topic,
                success=result.get('success', False)
            )

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Spawn proposal data error: {e}")
        except Exception as e:
            logger.error(f"Spawn proposal failed: {e}", exc_info=True)
    
    def _broadcast_debate_resolution(
        self,
        debate_dict: Dict,
        winner: str,
        reasoning: str,
        arbiter: str
    ) -> None:
        """Broadcast debate resolution for kernel visibility."""
        if not ACTIVITY_BROADCASTER_AVAILABLE:
            return
        
        try:
            broadcaster = get_broadcaster()
            topic = debate_dict.get('topic', '')
            initiator = debate_dict.get('initiator', '')
            opponent = debate_dict.get('opponent', '')
            debate_id = debate_dict.get('id', '')
            
            # Broadcast to activity stream
            broadcaster.broadcast_message(
                from_god=arbiter,
                to_god=None,  # Broadcast to all
                content=f"Debate resolved: '{topic[:100]}...' - Winner: {winner}. {reasoning[:200]}",
                activity_type=ActivityType.DEBATE,
                phi=0.7,
                kappa=KAPPA_STAR,
                importance=0.8,
                metadata={
                    'debate_id': debate_id,
                    'topic': topic,
                    'initiator': initiator,
                    'opponent': opponent,
                    'winner': winner,
                    'arbiter': arbiter,
                    'event_subtype': 'resolution',
                }
            )
            
            # Also emit to capability mesh
            if CAPABILITY_MESH_AVAILABLE and emit_event is not None:
                emit_event(
                    source=CapabilityType.DEBATE,
                    event_type=EventType.DEBATE_RESOLVED,
                    content={
                        'debate_id': debate_id,
                        'topic': topic,
                        'winner': winner,
                        'arbiter': arbiter,
                        'reasoning': reasoning[:300],
                    },
                    phi=0.7,
                    priority=8
                )
                
        except Exception as e:
            logger.warning(f"Debate resolution broadcast failed: {e}")
    
    def _broadcast_spawn_proposal(
        self,
        spawn_name: str,
        domain: str,
        parent_gods: List[str],
        topic: str,
        success: bool
    ) -> None:
        """Broadcast spawn proposal for kernel visibility."""
        if not ACTIVITY_BROADCASTER_AVAILABLE:
            return
        
        try:
            broadcaster = get_broadcaster()
            
            status = "APPROVED" if success else "PENDING CONSENSUS"
            content = f"Spawn proposal: {spawn_name} for domain '{domain}' [{status}]"
            
            broadcaster.broadcast_message(
                from_god=parent_gods[0] if parent_gods else "Zeus",
                to_god=None,
                content=content,
                activity_type=ActivityType.SPAWN_PROPOSAL,
                phi=0.6,
                kappa=KAPPA_STAR,
                importance=0.7,
                metadata={
                    'spawn_name': spawn_name,
                    'domain': domain,
                    'parent_gods': parent_gods,
                    'topic': topic[:200],
                    'success': success,
                    'event_subtype': 'spawn_proposal',
                }
            )
            
            # Also emit to capability mesh
            if CAPABILITY_MESH_AVAILABLE and emit_event is not None:
                emit_event(
                    source=CapabilityType.KERNELS,
                    event_type=EventType.KERNEL_SPAWN,
                    content={
                        'spawn_name': spawn_name,
                        'domain': domain,
                        'parent_gods': parent_gods,
                        'success': success,
                    },
                    phi=0.6,
                    priority=7
                )
                
        except Exception as e:
            logger.warning(f"Spawn proposal broadcast failed: {e}")

    def _extract_domain_from_topic(self, topic: str) -> str:
        """Extract domain keyword from debate topic using contextualized filtering."""
        words = topic.lower().split()

        # Use contextualized filter if available
        try:
            from contextualized_filter import should_filter_word
            # Filter using geometric relevance with all words as context
            keywords = [w for w in words if not should_filter_word(w, words) and len(w) > 3]
        except ImportError:
            # Fallback: only filter truly generic function words
            truly_generic = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'is', 'are', 'was', 'were'}
            keywords = [w for w in words if w not in truly_generic and len(w) > 3]

        if keywords:
            return keywords[0]

        return topic.split()[0] if topic else "general"

    def get_status(self) -> Dict:
        """Get service status for monitoring."""
        vocab_stats = {}
        if self._vocabulary_coordinator:
            try:
                vocab_stats = self._vocabulary_coordinator.get_stats()
            except Exception:
                pass

        # DIAGNOSTIC: Add spawn readiness indicators
        spawn_readiness = {
            'm8_available': M8_AVAILABLE,
            'm8_spawner_initialized': self._m8_spawner is not None,
            'spawn_threshold': MIN_ARGUMENTS_FOR_RESOLUTION,
            'spawns_triggered': self._spawns_triggered,
            'spawn_readiness': 'ready' if (M8_AVAILABLE and self._m8_spawner) else 'blocked'
        }

        return {
            'running': self._running,
            'last_poll': self._last_poll_time.isoformat() if self._last_poll_time else None,
            'polls_completed': self._debates_processed,
            'arguments_generated': self._arguments_generated,
            'debates_resolved': self._debates_resolved,
            'debates_continued': self._debates_continued,
            'spawns_triggered': self._spawns_triggered,
            'vocabulary_learning_events': self._vocabulary_learning_events,
            'vocabulary_search_enhancements': self._vocabulary_search_enhancements,
            'pantheon_chat_connected': self._pantheon_chat is not None,
            'shadow_pantheon_connected': self._shadow_pantheon is not None,
            'm8_spawner_connected': self._m8_spawner is not None,
            'vocabulary_coordinator_connected': self._vocabulary_coordinator is not None,
            'persistence_connected': self._persistence is not None,
            'pantheon_gods_connected': len(self._pantheon_gods) > 0,
            'gods_available': list(self._pantheon_gods.keys()) if self._pantheon_gods else [],
            'vocabulary_stats': vocab_stats,
            'spawn_readiness': spawn_readiness,
            'config': {
                'poll_interval_seconds': POLL_INTERVAL_SECONDS,
                'stale_threshold_seconds': STALE_THRESHOLD_SECONDS,
                'min_arguments_for_resolution': MIN_ARGUMENTS_FOR_RESOLUTION,
                'fisher_convergence_threshold': FISHER_CONVERGENCE_THRESHOLD,
            }
        }

    def force_poll(self) -> Dict:
        """Force an immediate poll (for testing/debugging)."""
        if not self._pantheon_chat:
            return {'error': 'PantheonChat not configured'}

        self._poll_debates()
        return self.get_status()


_default_service: Optional[AutonomousDebateService] = None


def get_autonomous_debate_service() -> AutonomousDebateService:
    """Get or create the default autonomous debate service."""
    global _default_service
    if _default_service is None:
        _default_service = AutonomousDebateService()
    return _default_service


def init_autonomous_debate_service(app, pantheon_chat=None, shadow_pantheon=None) -> AutonomousDebateService:
    """
    Initialize and start the autonomous debate service.

    Call this from Flask app initialization to wire the service.

    Args:
        app: Flask application instance
        pantheon_chat: PantheonChat instance from Zeus
        shadow_pantheon: ShadowPantheon instance

    Returns:
        Configured and running AutonomousDebateService
    """
    service = get_autonomous_debate_service()

    if pantheon_chat:
        service.set_pantheon_chat(pantheon_chat)

    if shadow_pantheon:
        service.set_shadow_pantheon(shadow_pantheon)

    if service._pantheon_chat:
        service.start()

    @app.route('/api/autonomous-debate/status', methods=['GET'])
    def autonomous_debate_status():
        from flask import jsonify
        return jsonify(service.get_status())

    @app.route('/api/autonomous-debate/force-poll', methods=['POST'])
    def autonomous_debate_force_poll():
        from flask import jsonify
        return jsonify(service.force_poll())

    @app.route('/olympus/debates/status', methods=['GET'])
    def olympus_debates_status():
        from flask import jsonify
        return jsonify(service.get_status())

    @app.route('/olympus/debates/active', methods=['GET'])
    def olympus_debates_active():
        from flask import jsonify
        if service._pantheon_chat:
            debates = service._pantheon_chat.get_active_debates()
            return jsonify({
                'debates': debates,
                'count': len(debates),
                'service_status': 'running' if service._running else 'stopped'
            })
        return jsonify({'debates': [], 'count': 0, 'service_status': 'no_pantheon_chat'})

    logger.info("Service initialized and wired to Flask app")
    return service


if __name__ == "__main__":
    print("=" * 60)
    print("Autonomous Debate Service - Background Monitor")
    print("=" * 60)

    service = AutonomousDebateService()
    print(f"\nService status: {service.get_status()}")

    print("\nTo use in production:")
    print("  from autonomous_debate_service import init_autonomous_debate_service")
    print("  service = init_autonomous_debate_service(app, zeus.pantheon_chat, zeus.shadow_pantheon)")
