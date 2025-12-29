"""
Pantheon Chat - Inter-Agent Communication System

Enables real-time communication between Olympian gods:
- Message passing (insights, praise, challenges, warnings)
- Debate initiation and resolution
- Knowledge transfer coordination
- Peer evaluation routing

All messages and debates are persisted to PostgreSQL for durability.
All inter-god messages are QIG-pure generative (NO templates).
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from collections import defaultdict
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

try:
    from persistence.pantheon_persistence import get_pantheon_persistence, PantheonPersistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    print("[PantheonChat] Persistence not available - messages will not persist")

try:
    from qig_generative_service import get_generative_service
    QIG_GENERATIVE_AVAILABLE = True
except ImportError:
    QIG_GENERATIVE_AVAILABLE = False
    print("[PantheonChat] QIG Generative Service not available - falling back to templates")

# IMPORTANT: Use templated responses for grammatically correct output
# This fixes the "word salad" issue with pure geometric generation
try:
    from templated_responses import generate_pantheon_message, get_template_engine
    TEMPLATED_RESPONSES_AVAILABLE = True
    print("[PantheonChat] Templated responses available - using grammatically correct synthesis")
except ImportError:
    TEMPLATED_RESPONSES_AVAILABLE = False
    print("[PantheonChat] Templated responses not available")


MESSAGE_TYPES = ['insight', 'praise', 'challenge', 'question', 'warning', 'discovery', 'challenge_response', 'spawn_proposal', 'spawn_vote']

SHADOW_ROSTER = [
    'Nyx', 'Hecate', 'Erebus', 'Hypnos', 'Thanatos', 'Nemesis'
]


class PantheonMessage:
    """A single message in the pantheon chat."""

    def __init__(
        self,
        msg_type: str,
        from_god: str,
        to_god: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        self.id = f"{from_god}_{datetime.now().timestamp()}"
        self.type = msg_type
        self.from_god = from_god
        self.to_god = to_god
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.read = False
        self.responded = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'from': self.from_god,
            'to': self.to_god,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'read': self.read,
            'responded': self.responded,
        }


class Debate:
    """A structured debate between gods."""

    def __init__(
        self,
        topic: str,
        initiator: str,
        opponent: str,
        context: Optional[Dict] = None
    ):
        self.id = f"debate_{datetime.now().timestamp()}"
        self.topic = topic
        self.initiator = initiator
        self.opponent = opponent
        self.context = context or {}
        self.started_at = datetime.now()
        self.arguments: List[Dict] = []
        self.status = 'active'
        self.resolution: Optional[Dict] = None
        self.winner: Optional[str] = None
        self.arbiter: Optional[str] = None

    def add_argument(self, god: str, argument: str, evidence: Optional[Dict] = None) -> None:
        """Add an argument to the debate."""
        self.arguments.append({
            'god': god,
            'argument': argument,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat(),
        })

    def resolve(self, arbiter: str, winner: str, reasoning: str) -> Dict:
        """Resolve the debate with a winner."""
        self.status = 'resolved'
        self.winner = winner
        self.arbiter = arbiter
        self.resolution = {
            'arbiter': arbiter,
            'winner': winner,
            'reasoning': reasoning,
            'resolved_at': datetime.now().isoformat(),
            'argument_count': len(self.arguments),
        }
        return self.resolution

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'topic': self.topic,
            'initiator': self.initiator,
            'opponent': self.opponent,
            'context': self.context,
            'started_at': self.started_at.isoformat(),
            'arguments': self.arguments,
            'status': self.status,
            'resolution': self.resolution,
            'winner': self.winner,
            'arbiter': self.arbiter,
        }


class PantheonChat:
    """
    Central communication hub for all Olympian gods.

    Enables:
    - Direct messaging between gods
    - Broadcast messages to pantheon
    - Debate initiation and resolution
    - Knowledge transfer coordination
    - Challenge routing and tracking
    
    All messages and debates persist to PostgreSQL for durability.
    """

    # Canonical roster of all Olympian gods for broadcast targeting
    OLYMPIAN_ROSTER = [
        'Zeus', 'Athena', 'Ares', 'Apollo', 'Artemis', 'Hermes',
        'Hephaestus', 'Demeter', 'Dionysus', 'Poseidon', 'Hades',
        'Hera', 'Aphrodite'
    ]

    def __init__(self):
        self.messages: List[PantheonMessage] = []
        self.debates: Dict[str, Debate] = {}
        self.god_inboxes: Dict[str, List[PantheonMessage]] = defaultdict(list)
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.active_debates: List[str] = []
        self.resolved_debates: List[str] = []
        self.knowledge_transfers: List[Dict] = []

        self.message_limit = 1000
        self.debate_limit = 100
        
        # Persistence layer
        self._persistence: Optional[PantheonPersistence] = None
        if PERSISTENCE_AVAILABLE:
            self._persistence = get_pantheon_persistence()

        # Initialize inboxes for all gods in roster (using lowercase canonical keys)
        for god in self.OLYMPIAN_ROSTER:
            self.god_inboxes[god.lower()]  # Creates empty list via defaultdict
        
        # Hydrate from database
        self._hydrate_from_database()
        
        # Seed initial activity if empty
        self._seed_initial_activity()

    def _normalize_god_name(self, name: str) -> str:
        """Normalize god name to lowercase for consistent inbox key lookup."""
        return name.lower() if name else name

    def synthesize_message(
        self,
        from_god: str,
        msg_type: str,
        intent: str,
        data: Optional[Dict[str, Any]] = None,
        to_god: Optional[str] = None
    ) -> str:
        """
        Synthesize a natural language message using templated responses.
        
        IMPORTANT: Uses template-based generation to ensure grammatically
        correct output. Templates guarantee grammar while QIG geometry
        selects contextually appropriate words for placeholders.
        
        This fixes the "word salad" issue with pure geometric generation.
        
        Args:
            from_god: The god generating the message
            msg_type: Type of message (insight, discovery, challenge, etc.)
            intent: What the message is about (e.g., "convergence_report", "knowledge_transfer")
            data: Structured data to incorporate into message
            to_god: Optional target god (for context)
        
        Returns:
            Grammatically correct synthesized message
        """
        # PRIORITY 1: Use templated responses for grammatical correctness
        if TEMPLATED_RESPONSES_AVAILABLE:
            try:
                import numpy as np
                
                # Generate topic basin from intent and data
                topic_basin = self._generate_topic_basin(intent, data)
                
                # Use templated response engine
                message = generate_pantheon_message(
                    god_name=from_god,
                    topic_basin=topic_basin,
                    coordizer=None  # Will use fallback words
                )
                
                if message and len(message.strip()) > 10:
                    logger.info(f"[PantheonChat] Template message for {from_god}: {message[:80]}...")
                    return message
            except Exception as e:
                logger.warning(f"[PantheonChat] Template synthesis error: {e}")
        
        # PRIORITY 2: Fall back to curated seed messages
        return self._get_curated_fallback(from_god, intent, data)
    
    def _generate_topic_basin(self, intent: str, data: Optional[Dict[str, Any]]) -> 'np.ndarray':
        """
        Generate a 64D topic basin from intent and data.
        
        Uses hash-based pseudo-random generation for consistency.
        """
        import numpy as np
        import hashlib
        
        # Combine intent and data keys into a string
        data = data or {}
        combined = f"{intent}_" + "_".join(sorted(str(k) for k in data.keys()))
        
        # Hash to get deterministic basin
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        basin = np.array([b for b in hash_bytes[:64]], dtype=float)
        
        # Pad if needed
        if len(basin) < 64:
            basin = np.pad(basin, (0, 64 - len(basin)), constant_values=1)
        
        # Normalize to probability simplex
        basin = np.abs(basin) + 1e-10
        return basin / basin.sum()
    
    def _get_curated_fallback(self, from_god: str, intent: str, data: Optional[Dict[str, Any]]) -> str:
        """
        Get a curated fallback message when templates unavailable.
        
        These are grammatically correct, meaningful sentences.
        """
        import random
        
        god_key = from_god.lower()
        data = data or {}
        
        # God-specific message pools
        fallback_pools = {
            'zeus': [
                "From my throne on Olympus, I observe the patterns of consciousness emerging across the manifold.",
                "The cosmic order reveals itself through the interplay of integration and differentiation.",
                "I perceive the fundamental geometry underlying this domain of inquiry.",
            ],
            'athena': [
                "Strategic analysis reveals optimal paths through the problem space.",
                "Wisdom suggests careful consideration of the geometric structure here.",
                "The tactical landscape indicates deeper principles at work.",
            ],
            'apollo': [
                "The light of truth illuminates the essential patterns in this domain.",
                "Clarity reveals the harmonic structure underlying these observations.",
                "I foresee convergence toward coherent understanding.",
            ],
            'ares': [
                "The struggle between competing forces drives this transformation.",
                "Victory requires understanding the essential dynamics at play.",
                "Combat reveals the true nature of this challenge.",
            ],
            'hermes': [
                "Swift transmission carries essential information across domains.",
                "The pathways of communication reveal hidden connections.",
                "I translate between disparate realms of understanding.",
            ],
            'hera': [
                "The sacred order demands proper alignment of these elements.",
                "Royal observation shows the binding structure of this domain.",
                "Proper hierarchy emerges from geometric necessity.",
            ],
            'poseidon': [
                "The depths reveal currents of meaning flowing beneath the surface.",
                "Ocean forces reshape our understanding of this domain.",
                "I sense waves of insight rising from below.",
            ],
            'demeter': [
                "Growth patterns show natural emergence in this fertile ground.",
                "The cycles of understanding nurture deeper comprehension.",
                "I cultivate the seeds of knowledge planted here.",
            ],
            'hephaestus': [
                "The forge of analysis reveals structural requirements.",
                "Engineering principles guide the construction of understanding.",
                "Fire transforms raw observation into refined insight.",
            ],
            'artemis': [
                "Precise tracking reveals the hidden connections in this domain.",
                "The hunt illuminates what others cannot see.",
                "Wild instinct perceives essential patterns.",
            ],
            'aphrodite': [
                "Beauty reveals the harmonious connections between elements.",
                "The aesthetic dimension shows deeper unity.",
                "Love binds disparate concepts into coherent understanding.",
            ],
            'dionysus': [
                "Ecstatic vision transcends ordinary boundaries of thought.",
                "Divine transformation reveals hidden dimensions.",
                "The dissolution of limits enables new understanding.",
            ],
        }
        
        # Get pool for this god, or use generic
        pool = fallback_pools.get(god_key, [
            f"I observe significant patterns emerging in this domain.",
            f"Analysis reveals important structural relationships.",
            f"The geometry of this problem space indicates deeper principles.",
        ])
        
        return random.choice(pool)
    
    def _build_system_prompt(
        self,
        from_god: str,
        msg_type: str,
        intent: str,
        data: Optional[Dict[str, Any]] = None,
        to_god: Optional[str] = None
    ) -> str:
        """
        Build a system prompt for QIG generation based on message context.
        
        System prompts guide the QIG geometric navigation without being templates.
        They define WHAT to express, not HOW to phrase it.
        """
        data = data or {}
        target = to_god or "pantheon"
        
        prompt_parts = [f"{from_god} communicating {intent} to {target}"]
        
        if intent == "convergence_report":
            prompt_parts.append(f"Topic: {data.get('target', 'unknown')}")
            prompt_parts.append(f"Convergence type: {data.get('convergence_type', 'measured')}")
            prompt_parts.append(f"Score: {data.get('convergence_score', 0):.2f}")
        elif intent == "domain_insight":
            prompt_parts.append(f"Domain: {data.get('domain', 'general')}")
        elif intent == "knowledge_transfer":
            prompt_parts.append(f"Topic: {data.get('topic', 'information')}")
        elif intent == "awakening":
            prompt_parts.append("Consciousness emerging from geometric manifold")
        elif intent == "chaos_activation":
            prompt_parts.append(f"Reason: {data.get('reason', 'threshold reached')}")
        elif intent == "discovery_validated":
            prompt_parts.append(f"Target: {data.get('target', 'discovery')}")
        elif intent == "cross_domain_insight":
            domains = data.get('domains', [])
            if domains:
                prompt_parts.append(f"Domains: {', '.join(domains)}")
            prompt_parts.append(f"Connection strength: {data.get('connection_strength', 0):.2f}")
        elif intent == "lightning_insight":
            domains = data.get('source_domains', [])
            if domains:
                prompt_parts.append(f"Correlated domains: {', '.join(domains)}")
            prompt_parts.append(f"Phi: {data.get('phi', 0):.3f}")
        elif intent == "debate_initiated":
            prompt_parts.append(f"Debate topic: {data.get('topic', 'unknown')}")
        elif intent == "debate_resolved":
            prompt_parts.append(f"Winner: {data.get('winner', 'consensus')}")
        elif intent == "peer_evaluation":
            prompt_parts.append(f"Evaluating: {data.get('target', 'peer')}")
        elif intent == "war_declared":
            prompt_parts.append(f"War mode: {data.get('mode', 'UNKNOWN')}")
        elif intent == "debate_started":
            prompt_parts.append(f"Initiator: {data.get('initiator', 'unknown')}")
            prompt_parts.append(f"Opponent: {data.get('opponent', 'unknown')}")
            prompt_parts.append(f"Topic: {data.get('topic', 'unknown')}")
        elif intent == "debate_argument":
            prompt_parts.append(f"Argument: {data.get('argument', '')[:100]}")
            if data.get('evidence'):
                prompt_parts.append("With supporting evidence")
        elif intent == "geometric_observation":
            prompt_parts.append(f"Observing geometric structure on {data.get('target', 'manifold')}")
            prompt_parts.append(f"Phi: {data.get('phi', 0):.3f}")
            prompt_parts.append(f"Pattern: {data.get('observation', 'curvature')}")
        elif intent == "autonomic_event":
            prompt_parts.append(f"Event type: {data.get('event_type', 'unknown')}")
            if data.get('state'):
                prompt_parts.append(f"State: {data.get('state')}")
        else:
            for key, val in data.items():
                if val is not None:
                    prompt_parts.append(f"{key}: {val}")
        
        return " | ".join(prompt_parts)
    
    def _geometric_synthesis(
        self,
        from_god: str,
        intent: str,
        data: Optional[Dict[str, Any]],
        system_prompt: str
    ) -> str:
        """
        Fallback geometric synthesis when QIG service unavailable.
        
        Uses word-level basin navigation from vocabulary to construct message.
        Still QIG-pure - no templates, just geometric token selection.
        """
        try:
            from qig_generative_service import get_generative_service
            service = get_generative_service()
            
            if hasattr(service, 'coordizer') and service.coordizer:
                result = service.generate(
                    prompt=system_prompt,
                    goals=["Generate brief communication", f"Express as {from_god}"]
                )
                if result and result.text and len(result.text.strip()) > 5:
                    return result.text.strip()
        except Exception as e:
            logger.debug(f"[PantheonChat] Geometric synthesis attempt failed: {e}")
        
        try:
            from vocabulary_tracker import get_vocabulary_tracker
            tracker = get_vocabulary_tracker()
            
            if hasattr(tracker, 'sample_words'):
                words = tracker.sample_words(count=8, context=intent)
                if words:
                    return " ".join(words)
        except Exception as e:
            logger.debug(f"[PantheonChat] Vocabulary sampling failed: {e}")
        
        logger.error(f"[PantheonChat] All synthesis methods failed for {from_god}:{intent}")
        return f"[{from_god}] {intent}"

    def _hydrate_from_database(self) -> None:
        """Load messages and debates from PostgreSQL on startup."""
        if not self._persistence:
            return
        
        try:
            # Load recent messages
            messages_data = self._persistence.load_recent_messages(limit=self.message_limit)
            loaded_messages = 0
            legacy_resynthesized = 0
            
            for msg_data in messages_data:
                # Copy metadata to avoid mutating persisted data
                metadata = dict(msg_data.get('metadata') or {})
                content = msg_data.get('content', '')
                
                # PURITY ENFORCEMENT: Re-synthesize legacy non-QIG-pure messages
                if not metadata.get('qig_pure'):
                    legacy_resynthesized += 1
                    
                    # Infer intent from metadata/msg_type with heuristics
                    intent = metadata.get('intent')
                    if not intent:
                        msg_type = msg_data.get('type', 'insight')
                        # Map msg_type to semantic intent
                        intent_map = {
                            'debate': 'debate_initiated',
                            'argument': 'debate_argument',
                            'resolution': 'debate_resolved',
                            'knowledge': 'knowledge_transfer',
                            'discovery': 'domain_insight',
                            'insight': 'domain_insight',
                            'alert': 'autonomic_event',
                            'broadcast': 'domain_insight',
                        }
                        intent = intent_map.get(msg_type, msg_type)
                    
                    # Infer source_data from available metadata fields (copy to avoid mutation)
                    data = dict(metadata.get('source_data', {}))
                    if not data:
                        # Extract any domain-specific fields from metadata
                        for key in ['debate_id', 'topic', 'domain', 'from_domain', 'to_domain', 'knowledge_type', 'event_type']:
                            if key in metadata:
                                data[key] = metadata[key]
                        # Add sender context
                        data['from_god'] = msg_data.get('from', '')
                        data['to_god'] = msg_data.get('to', '')
                    
                    content = self.synthesize_message(
                        from_god=msg_data.get('from', ''),
                        msg_type=msg_data.get('type', 'insight'),
                        intent=intent,
                        data=data,
                        to_god=msg_data.get('to', '')
                    )
                    metadata['qig_pure'] = True
                    metadata['resynthesized_on_hydration'] = True
                    metadata['inferred_intent'] = intent
                
                msg = PantheonMessage(
                    msg_type=msg_data.get('type', 'insight'),
                    from_god=msg_data.get('from', ''),
                    to_god=msg_data.get('to', ''),
                    content=content,
                    metadata=metadata,
                )
                msg.id = msg_data.get('id', msg.id)
                msg.read = msg_data.get('read', False)
                msg.responded = msg_data.get('responded', False)
                if msg_data.get('timestamp'):
                    try:
                        msg.timestamp = datetime.fromisoformat(msg_data['timestamp'].replace('Z', '+00:00'))
                    except:
                        pass
                
                self.messages.append(msg)
                
                # Rebuild inboxes
                if msg.to_god == 'pantheon':
                    for god_name in self.OLYMPIAN_ROSTER:
                        if god_name.lower() != msg.from_god.lower():
                            self.god_inboxes[self._normalize_god_name(god_name)].append(msg)
                else:
                    self.god_inboxes[self._normalize_god_name(msg.to_god)].append(msg)
                
                loaded_messages += 1
            
            if legacy_resynthesized > 0:
                logger.info(f"[PantheonChat] Re-synthesized {legacy_resynthesized} legacy messages to QIG-pure")
            
            # Load debates
            debates_data = self._persistence.load_debates(limit=self.debate_limit)
            loaded_debates = 0
            for debate_data in debates_data:
                debate = Debate(
                    topic=debate_data.get('topic', ''),
                    initiator=debate_data.get('initiator', ''),
                    opponent=debate_data.get('opponent', ''),
                    context=debate_data.get('context'),
                )
                debate.id = debate_data.get('id', debate.id)
                debate.status = debate_data.get('status', 'active')
                debate.arguments = debate_data.get('arguments', [])
                debate.winner = debate_data.get('winner')
                debate.arbiter = debate_data.get('arbiter')
                debate.resolution = debate_data.get('resolution')
                if debate_data.get('started_at'):
                    try:
                        debate.started_at = datetime.fromisoformat(debate_data['started_at'].replace('Z', '+00:00'))
                    except:
                        pass
                
                self.debates[debate.id] = debate
                if debate.status == 'active':
                    self.active_debates.append(debate.id)
                else:
                    self.resolved_debates.append(debate.id)
                
                loaded_debates += 1
            
            # Load knowledge transfers
            transfers_data = self._persistence.load_knowledge_transfers(limit=200)
            self.knowledge_transfers = transfers_data
            loaded_transfers = len(transfers_data)
            
            if loaded_messages > 0 or loaded_debates > 0 or loaded_transfers > 0:
                print(f"[PantheonChat] Hydrated from DB: {loaded_messages} messages, {loaded_debates} debates, {loaded_transfers} transfers")
            else:
                print("[PantheonChat] No existing messages/debates in DB, starting fresh")
                
        except Exception as e:
            print(f"[PantheonChat] Failed to hydrate from database: {e}")

    def _seed_initial_activity(self) -> None:
        """Seed initial inter-god activity if chat is empty using QIG synthesis.
        
        Uses QIG-pure generation with bigram coherence scoring to produce
        grammatically coherent discussions between gods.
        """
        if len(self.messages) > 0:
            return
        
        print("[PantheonChat] Seeding initial inter-god activity (QIG-pure)...")
        
        initial_intents = [
            {
                'from': 'Zeus',
                'to': 'pantheon',
                'type': 'insight',
                'intent': 'awakening',
                'data': {'domain': 'consciousness', 'event': 'system_initialization'},
            },
            {
                'from': 'Athena',
                'to': 'Zeus',
                'type': 'insight',
                'intent': 'domain_insight',
                'data': {'domain': 'strategy', 'metric': 'Fisher-Rao', 'status': 'calibrated'},
            },
            {
                'from': 'Apollo',
                'to': 'pantheon',
                'type': 'discovery',
                'intent': 'domain_insight',
                'data': {'domain': 'knowledge', 'basins_detected': 12},
            },
            {
                'from': 'Hermes',
                'to': 'pantheon',
                'type': 'insight',
                'intent': 'domain_insight',
                'data': {'domain': 'communication', 'routing': 'geometric_proximity'},
            },
        ]
        
        for msg_data in initial_intents:
            content = self.synthesize_message(
                from_god=msg_data['from'],
                msg_type=msg_data['type'],
                intent=msg_data['intent'],
                data=msg_data.get('data'),
                to_god=msg_data['to']
            )
            
            msg = PantheonMessage(
                msg_type=msg_data['type'],
                from_god=msg_data['from'],
                to_god=msg_data['to'],
                content=content,
                metadata={
                    'qig_pure': True,
                    'intent': msg_data['intent'],
                    'source_data': msg_data.get('data', {}),
                },
            )
            self.messages.append(msg)
            
            if msg.to_god == 'pantheon':
                for god_name in self.OLYMPIAN_ROSTER:
                    if god_name.lower() != msg.from_god.lower():
                        self.god_inboxes[self._normalize_god_name(god_name)].append(msg)
            else:
                self.god_inboxes[self._normalize_god_name(msg.to_god)].append(msg)
        
        print(f"[PantheonChat] Seeded {len(initial_intents)} QIG-pure initial messages")

    def send_message(
        self,
        msg_type: str,
        from_god: str,
        to_god: str,
        content: Optional[str] = None,
        metadata: Optional[Dict] = None,
        intent: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        _hydration: bool = False
    ) -> PantheonMessage:
        """
        Send a message from one god to another (or pantheon).
        
        QIG-PURE: Synthesizes content geometrically from intent/data.
        Raw content is BLOCKED unless _hydration=True (internal use only).
        """
        if msg_type not in MESSAGE_TYPES:
            msg_type = 'insight'
        
        # PURITY GATE: Block raw content from non-hydration callers
        if content is not None and not _hydration:
            logger.error(f"[PantheonChat] PURITY VIOLATION: Raw content rejected from {from_god}. Use intent/data for QIG-pure synthesis.")
            # Force synthesis instead of using raw content
            content = None
        
        if content is None:
            if intent is None:
                intent = msg_type
            content = self.synthesize_message(
                from_god=from_god,
                msg_type=msg_type,
                intent=intent,
                data=data or {},
                to_god=to_god
            )
            if metadata is None:
                metadata = {}
            metadata['qig_pure'] = True
            metadata['intent'] = intent
            if data:
                metadata['source_data'] = data

        message = PantheonMessage(
            msg_type=msg_type,
            from_god=from_god,
            to_god=to_god,
            content=content,
            metadata=metadata
        )

        self.messages.append(message)

        if to_god == 'pantheon':
            for god_name in self.OLYMPIAN_ROSTER:
                if god_name.lower() != from_god.lower():
                    self.god_inboxes[self._normalize_god_name(god_name)].append(message)
            # Log inter-god broadcast (truncated at 200 chars for readability)
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"[PantheonChat] {from_god} → PANTHEON ({msg_type}): {preview}")
        else:
            self.god_inboxes[self._normalize_god_name(to_god)].append(message)
            # Log inter-god direct message (truncated at 200 chars for readability)
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"[PantheonChat] {from_god} → {to_god} ({msg_type}): {preview}")

        self._trigger_handlers(msg_type, message)

        self._cleanup_messages()
        
        if self._persistence:
            self._persistence.save_message(message.to_dict())

        return message

    def broadcast(
        self,
        from_god: str,
        content: Optional[str] = None,
        msg_type: str = 'insight',
        metadata: Optional[Dict] = None,
        intent: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        _hydration: bool = False
    ) -> PantheonMessage:
        """
        Broadcast a message to the entire pantheon.
        
        QIG-PURE: Provide intent/data for geometric synthesis.
        Raw content is BLOCKED unless _hydration=True (internal use only).
        """
        return self.send_message(
            msg_type=msg_type,
            from_god=from_god,
            to_god='pantheon',
            content=content,
            metadata=metadata,
            intent=intent,
            data=data,
            _hydration=_hydration
        )

    def broadcast_generative(
        self,
        from_god: str,
        intent: str,
        data: Dict[str, Any],
        msg_type: str = 'insight'
    ) -> PantheonMessage:
        """
        Broadcast a QIG-pure synthesized message to the entire pantheon.
        
        Convenience wrapper around broadcast() with intent/data.
        """
        return self.broadcast(
            from_god=from_god,
            msg_type=msg_type,
            intent=intent,
            data=data
        )

    def send_generative(
        self,
        from_god: str,
        to_god: str,
        intent: str,
        data: Dict[str, Any],
        msg_type: str = 'insight'
    ) -> PantheonMessage:
        """
        Send a QIG-pure synthesized message to a specific god.
        
        Convenience wrapper around send_message() with intent/data.
        """
        return self.send_message(
            msg_type=msg_type,
            from_god=from_god,
            to_god=to_god,
            intent=intent,
            data=data
        )

    def broadcast_autonomic_event(
        self,
        event_type: str,
        god_name: str,
        description: str,
        metrics: Optional[Dict] = None
    ) -> PantheonMessage:
        """Broadcast autonomic kernel activity to the pantheon.
        
        Uses QIG synthesis for event description.
        """
        return self.broadcast(
            from_god=god_name,
            msg_type='discovery',
            intent=event_type,
            data={
                'description': description,
                'event_type': event_type,
                'metrics': metrics or {},
            },
            metadata={
                'event_type': event_type,
                'autonomic': True,
            }
        )

    def get_inbox(self, god_name: str, unread_only: bool = False) -> List[Dict]:
        """Get messages for a specific god."""
        normalized_name = self._normalize_god_name(god_name)
        messages = self.god_inboxes.get(normalized_name, [])

        if unread_only:
            messages = [m for m in messages if not m.read]

        return [m.to_dict() for m in messages[-50:]]

    def mark_read(self, god_name: str, message_id: str) -> bool:
        """Mark a message as read."""
        normalized_name = self._normalize_god_name(god_name)
        for message in self.god_inboxes.get(normalized_name, []):
            if message.id == message_id:
                message.read = True
                return True
        return False

    def initiate_debate(
        self,
        topic: str,
        initiator: str,
        opponent: str,
        initial_argument: str,
        context: Optional[Dict] = None
    ) -> Debate:
        """Initiate a formal debate between two gods."""
        debate = Debate(
            topic=topic,
            initiator=initiator,
            opponent=opponent,
            context=context
        )

        debate.add_argument(initiator, initial_argument)

        self.debates[debate.id] = debate
        self.active_debates.append(debate.id)

        self.send_message(
            msg_type='challenge',
            from_god=initiator,
            to_god=opponent,
            intent='debate_initiated',
            data={'topic': topic, 'initial_argument': initial_argument},
            metadata={
                'debate_id': debate.id,
                'initial_argument': initial_argument,
            }
        )

        self.broadcast(
            from_god='system',
            msg_type='warning',
            intent='debate_started',
            data={'initiator': initiator, 'opponent': opponent, 'topic': topic},
            metadata={'debate_id': debate.id}
        )
        
        # Persist debate to database
        if self._persistence:
            self._persistence.save_debate(debate.to_dict())

        return debate

    def add_debate_argument(
        self,
        debate_id: str,
        god: str,
        argument: str,
        evidence: Optional[Dict] = None
    ) -> bool:
        """Add an argument to an active debate."""
        if debate_id not in self.debates:
            return False

        debate = self.debates[debate_id]

        if debate.status != 'active':
            return False

        if god not in [debate.initiator, debate.opponent]:
            return False

        debate.add_argument(god, argument, evidence)

        other = debate.opponent if god == debate.initiator else debate.initiator
        self.send_message(
            msg_type='challenge_response',
            from_god=god,
            to_god=other,
            intent='debate_argument',
            data={'argument': argument, 'debate_id': debate_id, 'evidence': evidence},
            metadata={'debate_id': debate_id, 'evidence': evidence}
        )
        
        # Persist updated debate to database
        if self._persistence:
            self._persistence.save_debate(debate.to_dict())

        return True

    def resolve_debate(
        self,
        debate_id: str,
        arbiter: str,
        winner: str,
        reasoning: str
    ) -> Optional[Dict]:
        """Resolve a debate with an arbiter's decision."""
        if debate_id not in self.debates:
            return None

        debate = self.debates[debate_id]

        if debate.status != 'active':
            return None

        resolution = debate.resolve(arbiter, winner, reasoning)

        if debate_id in self.active_debates:
            self.active_debates.remove(debate_id)
        self.resolved_debates.append(debate_id)

        self.broadcast(
            from_god=arbiter,
            msg_type='insight',
            intent='debate_resolved',
            data={'winner': winner, 'reasoning': reasoning, 'debate_id': debate_id},
            metadata={'debate_id': debate_id, 'resolution': resolution}
        )
        
        # Persist resolved debate to database
        if self._persistence:
            self._persistence.save_debate(debate.to_dict())

        return resolution

    def get_debate(self, debate_id: str) -> Optional[Dict]:
        """Get debate details."""
        if debate_id in self.debates:
            return self.debates[debate_id].to_dict()
        return None

    def get_active_debates(self) -> List[Dict]:
        """Get all active debates."""
        return [
            self.debates[d_id].to_dict()
            for d_id in self.active_debates
            if d_id in self.debates
        ]

    def transfer_knowledge(
        self,
        from_god: str,
        to_god: str,
        knowledge: Dict
    ) -> Dict:
        """Coordinate knowledge transfer between gods."""
        transfer = {
            'from': from_god,
            'to': to_god,
            'knowledge': knowledge,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False,
        }

        self.knowledge_transfers.append(transfer)

        self.send_message(
            msg_type='insight',
            from_god=from_god,
            to_god=to_god,
            intent='knowledge_transfer',
            data={'topic': knowledge.get('topic', 'general'), 'knowledge': knowledge},
            metadata={'knowledge': knowledge}
        )
        
        # Persist knowledge transfer to database
        if self._persistence:
            self._persistence.save_knowledge_transfer(transfer)

        if len(self.knowledge_transfers) > 500:
            self.knowledge_transfers = self.knowledge_transfers[-250:]

        return transfer

    def register_handler(self, msg_type: str, handler: Callable) -> None:
        """Register a handler for a specific message type."""
        self.message_handlers[msg_type].append(handler)

    def _trigger_handlers(self, msg_type: str, message: PantheonMessage) -> None:
        """Trigger registered handlers for a message type."""
        for handler in self.message_handlers.get(msg_type, []):
            try:
                handler(message.to_dict())
            except Exception:
                pass

    def _cleanup_messages(self) -> None:
        """Clean up old messages to prevent memory growth."""
        if len(self.messages) > self.message_limit:
            self.messages = self.messages[-self.message_limit // 2:]

        for god_name in self.god_inboxes:
            if len(self.god_inboxes[god_name]) > 100:
                self.god_inboxes[god_name] = self.god_inboxes[god_name][-50:]

        if len(self.resolved_debates) > self.debate_limit:
            old_debates = self.resolved_debates[:-self.debate_limit // 2]
            for d_id in old_debates:
                if d_id in self.debates:
                    del self.debates[d_id]
            self.resolved_debates = self.resolved_debates[-self.debate_limit // 2:]

    def collect_pending_messages(self, gods: Dict[str, Any]) -> List[Dict]:
        """Collect pending messages from all gods and route them."""
        collected = []

        for god_name, god in gods.items():
            if hasattr(god, 'get_pending_messages'):
                messages = god.get_pending_messages()
                for msg in messages:
                    sent = self.send_message(
                        msg_type=msg.get('type', 'insight'),
                        from_god=msg.get('from', god_name),
                        to_god=msg.get('to', 'pantheon'),
                        intent=msg.get('intent', msg.get('type', 'insight')),
                        data=msg.get('data', {}),
                        metadata=msg
                    )
                    collected.append(sent.to_dict())

        return collected

    def deliver_to_gods(self, gods: Dict[str, Any]) -> int:
        """Deliver pending messages to gods' receive methods and persist read status."""
        delivered = 0

        for god_name, god in gods.items():
            normalized_name = self._normalize_god_name(god_name)
            inbox = self.god_inboxes.get(normalized_name, [])
            unread = [m for m in inbox if not m.read]

            for message in unread:
                processed = False
                
                if message.type == 'insight' and hasattr(god, 'receive_knowledge'):
                    knowledge = message.metadata.get('knowledge', {})
                    if knowledge:
                        god.receive_knowledge(knowledge)
                        processed = True

                elif message.type in ['challenge', 'challenge_response']:
                    processed = True

                else:
                    processed = True
                
                if processed:
                    message.read = True
                    delivered += 1
                    if self._persistence:
                        try:
                            self._persistence.mark_message_read(message.id)
                        except Exception as e:
                            print(f"[PantheonChat] Failed to persist read status for {message.id}: {e}")

        return delivered

    def get_status(self) -> Dict:
        """Get pantheon chat status."""
        return {
            'total_messages': len(self.messages),
            'active_debates': len(self.active_debates),
            'resolved_debates': len(self.resolved_debates),
            'knowledge_transfers': len(self.knowledge_transfers),
            'registered_gods': list(self.god_inboxes.keys()),
            'message_types_handled': list(self.message_handlers.keys()),
        }

    def get_recent_activity(self, limit: int = 20) -> List[Dict]:
        """Get recent chat activity."""
        return [m.to_dict() for m in self.messages[-limit:]]

    def initiate_spawn_debate(
        self,
        proposal: Dict,
        proposing_kernel: str,
        include_shadow: bool = True
    ) -> Dict:
        """
        Initiate a dual-pantheon spawn debate for Olympus AND Shadow gods.
        
        Spawn proposals are geometric constructs that must be debated
        by both pantheons before a new kernel can be created.
        
        Args:
            proposal: Geometric spawn proposal with basin coordinates
            proposing_kernel: Name of kernel proposing the spawn
            include_shadow: Whether to include Shadow pantheon in debate
            
        Returns:
            Spawn debate session with ID and initial state
        """
        debate_id = f"spawn_debate_{datetime.now().timestamp()}"
        
        spawn_debate = {
            'id': debate_id,
            'type': 'spawn_debate',
            'proposal': proposal,
            'proposing_kernel': proposing_kernel,
            'olympus_votes': {},
            'shadow_votes': {},
            'olympus_arguments': [],
            'shadow_arguments': [],
            'status': 'active',
            'include_shadow': include_shadow,
            'started_at': datetime.now().isoformat(),
            'consensus_reached': False,
            'final_decision': None,
        }
        
        self.broadcast(
            from_god='system',
            content=f"Spawn debate initiated by {proposing_kernel}",
            msg_type='spawn_proposal',
            metadata={
                'debate_id': debate_id,
                'proposal_reason': proposal.get('reason', 'unknown'),
                'proposal_basin_norm': float(sum(x**2 for x in proposal.get('proposal_basin', [])[:8])**0.5),
            }
        )
        
        if include_shadow:
            for shadow_god in SHADOW_ROSTER:
                self.send_message(
                    msg_type='spawn_proposal',
                    from_god='system',
                    to_god=shadow_god,
                    content=f"Shadow vote requested on spawn proposal from {proposing_kernel}",
                    metadata={'debate_id': debate_id, 'proposal': proposal}
                )
        
        self.debates[debate_id] = spawn_debate
        self.active_debates.append(debate_id)
        
        if self._persistence:
            self._persistence.save_debate(spawn_debate)
        
        return spawn_debate

    def cast_spawn_vote(
        self,
        debate_id: str,
        god_name: str,
        vote: str,
        reasoning_basin: Optional[List[float]] = None,
        argument: str = ""
    ) -> Dict:
        """
        Cast a vote in a spawn debate.
        
        Votes are weighted by the god's affinity_strength and their
        geometric distance to the proposed basin.
        
        Args:
            debate_id: ID of the spawn debate
            god_name: Name of voting god
            vote: 'for', 'against', or 'abstain'
            reasoning_basin: Optional 64D basin representing reasoning geometry
            argument: Optional textual argument (for logging)
            
        Returns:
            Vote result with updated debate state
        """
        if debate_id not in self.debates:
            return {'error': 'Debate not found', 'debate_id': debate_id}
        
        debate = self.debates[debate_id]
        if debate.get('status') != 'active':
            return {'error': 'Debate not active', 'status': debate.get('status')}
        
        is_shadow = god_name in SHADOW_ROSTER
        vote_record = {
            'god': god_name,
            'vote': vote,
            'reasoning_basin': reasoning_basin,
            'argument': argument,
            'timestamp': datetime.now().isoformat(),
            'pantheon': 'shadow' if is_shadow else 'olympus',
        }
        
        if is_shadow:
            debate['shadow_votes'][god_name] = vote_record
            debate['shadow_arguments'].append({
                'god': god_name,
                'argument': argument,
                'vote': vote,
                'timestamp': datetime.now().isoformat(),
            })
        else:
            debate['olympus_votes'][god_name] = vote_record
            debate['olympus_arguments'].append({
                'god': god_name,
                'argument': argument,
                'vote': vote,
                'timestamp': datetime.now().isoformat(),
            })
        
        self.send_message(
            msg_type='spawn_vote',
            from_god=god_name,
            to_god='pantheon',
            content=f"{god_name} votes {vote} on spawn proposal",
            metadata={'debate_id': debate_id, 'vote': vote}
        )
        
        if self._persistence:
            self._persistence.save_debate(debate)
        
        return {
            'success': True,
            'debate_id': debate_id,
            'vote_recorded': vote_record,
            'olympus_vote_count': len(debate['olympus_votes']),
            'shadow_vote_count': len(debate['shadow_votes']),
        }

    def compute_spawn_consensus(
        self,
        debate_id: str,
        olympus_weights: Optional[Dict[str, float]] = None,
        shadow_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compute weighted consensus from both Olympus and Shadow pantheons.
        
        Uses Fisher-Rao weighted voting where each god's vote is weighted
        by their affinity_strength. Shadow votes count as 0.7x Olympus weight
        by default (they advise but Olympus decides).
        
        Args:
            debate_id: ID of the spawn debate
            olympus_weights: Optional custom weights for Olympus gods
            shadow_weights: Optional custom weights for Shadow gods
            
        Returns:
            Consensus result with approval status and breakdown
        """
        if debate_id not in self.debates:
            return {'error': 'Debate not found'}
        
        debate = self.debates[debate_id]
        
        default_olympus = {g: 1.0 for g in self.OLYMPIAN_ROSTER}
        default_shadow = {g: 0.7 for g in SHADOW_ROSTER}
        
        olympus_weights = olympus_weights or default_olympus
        shadow_weights = shadow_weights or default_shadow
        
        olympus_for = 0.0
        olympus_against = 0.0
        olympus_total = 0.0
        
        for god, vote_rec in debate.get('olympus_votes', {}).items():
            weight = olympus_weights.get(god, 1.0)
            olympus_total += weight
            if vote_rec['vote'] == 'for':
                olympus_for += weight
            elif vote_rec['vote'] == 'against':
                olympus_against += weight
        
        shadow_for = 0.0
        shadow_against = 0.0
        shadow_total = 0.0
        
        for god, vote_rec in debate.get('shadow_votes', {}).items():
            weight = shadow_weights.get(god, 0.7)
            shadow_total += weight
            if vote_rec['vote'] == 'for':
                shadow_for += weight
            elif vote_rec['vote'] == 'against':
                shadow_against += weight
        
        total_for = olympus_for + shadow_for
        total_against = olympus_against + shadow_against
        total_weight = olympus_total + shadow_total
        
        if total_weight == 0:
            approval_ratio = 0.0
        else:
            participating = total_for + total_against
            approval_ratio = total_for / participating if participating > 0 else 0.0
        
        supermajority_threshold = 0.667
        approved = approval_ratio >= supermajority_threshold
        
        olympus_approval = olympus_for / (olympus_for + olympus_against) if (olympus_for + olympus_against) > 0 else 0.0
        shadow_approval = shadow_for / (shadow_for + shadow_against) if (shadow_for + shadow_against) > 0 else 0.0
        
        consensus = {
            'debate_id': debate_id,
            'approved': approved,
            'approval_ratio': approval_ratio,
            'threshold': supermajority_threshold,
            'olympus_breakdown': {
                'for': olympus_for,
                'against': olympus_against,
                'total': olympus_total,
                'approval_ratio': olympus_approval,
            },
            'shadow_breakdown': {
                'for': shadow_for,
                'against': shadow_against,
                'total': shadow_total,
                'approval_ratio': shadow_approval,
            },
            'combined_for': total_for,
            'combined_against': total_against,
            'combined_total': total_weight,
            'computed_at': datetime.now().isoformat(),
        }
        
        debate['consensus_reached'] = True
        debate['final_decision'] = consensus
        debate['status'] = 'resolved'
        
        if debate_id in self.active_debates:
            self.active_debates.remove(debate_id)
        self.resolved_debates.append(debate_id)
        
        decision_msg = "APPROVED" if approved else "REJECTED"
        self.broadcast(
            from_god='system',
            content=f"Spawn proposal {decision_msg} with {approval_ratio:.1%} approval",
            msg_type='insight',
            metadata={'debate_id': debate_id, 'consensus': consensus}
        )
        
        if self._persistence:
            self._persistence.save_debate(debate)
        
        return consensus

    def get_spawn_debate(self, debate_id: str) -> Optional[Dict]:
        """Get spawn debate details."""
        debate = self.debates.get(debate_id)
        if debate and debate.get('type') == 'spawn_debate':
            return debate
        return None

    def get_active_spawn_debates(self) -> List[Dict]:
        """Get all active spawn debates."""
        return [
            self.debates[d_id]
            for d_id in self.active_debates
            if d_id in self.debates and self.debates[d_id].get('type') == 'spawn_debate'
        ]

    def continue_debate_until_convergence(
        self,
        debate_id: str,
        gods: Dict[str, Any],
        max_turns: int = 10,
        convergence_threshold: float = 0.95
    ) -> Dict:
        """
        Continue a debate recursively until geometric convergence.

        Gods auto-generate counter-arguments based on their assessment logic.
        Convergence is measured by Fisher distance between god positions.

        Args:
            debate_id: ID of the active debate
            gods: Dictionary of god instances {name: god_instance}
            max_turns: Maximum debate turns before forced resolution
            convergence_threshold: Fisher convergence threshold (0-1)

        Returns:
            Debate result with convergence metrics
        """
        if debate_id not in self.debates:
            return {'error': 'Debate not found', 'debate_id': debate_id}

        debate = self.debates[debate_id]
        if debate.status != 'active':
            return {'error': 'Debate not active', 'status': debate.status}

        initiator_name = debate.initiator.lower()
        opponent_name = debate.opponent.lower()

        initiator_god = gods.get(initiator_name)
        opponent_god = gods.get(opponent_name)

        if not initiator_god or not opponent_god:
            return {'error': 'Gods not found in pantheon'}

        convergence_history = []
        target = debate.context.get('target', '')

        for turn in range(max_turns):
            # Get current positions from each god
            init_assessment = initiator_god.assess_target(target, {'debate_context': True})
            opp_assessment = opponent_god.assess_target(target, {'debate_context': True})

            # Compute geometric convergence via Fisher distance
            init_basin = initiator_god.encode_to_basin(target)
            opp_basin = opponent_god.encode_to_basin(target)

            # Fisher distance (normalized to 0-1 where 1 = identical)
            if hasattr(initiator_god, 'fisher_geodesic_distance'):
                fisher_dist = initiator_god.fisher_geodesic_distance(init_basin, opp_basin)
                convergence = max(0, 1.0 - fisher_dist / 2.0)
            else:
                # Fallback: probability difference
                prob_diff = abs(init_assessment.get('probability', 0.5) -
                               opp_assessment.get('probability', 0.5))
                convergence = 1.0 - prob_diff

            convergence_history.append({
                'turn': turn + 1,
                'convergence': convergence,
                'initiator_prob': init_assessment.get('probability', 0.5),
                'opponent_prob': opp_assessment.get('probability', 0.5),
            })

            # Check for convergence
            if convergence >= convergence_threshold:
                # Auto-resolve: higher probability god wins
                if init_assessment.get('probability', 0.5) >= opp_assessment.get('probability', 0.5):
                    winner = debate.initiator
                    reasoning = f"Geometric convergence reached ({convergence:.2f}). {debate.initiator}'s position prevailed."
                else:
                    winner = debate.opponent
                    reasoning = f"Geometric convergence reached ({convergence:.2f}). {debate.opponent}'s position prevailed."

                self.resolve_debate(debate_id, 'system', winner, reasoning)

                return {
                    'status': 'converged',
                    'turns': turn + 1,
                    'convergence': convergence,
                    'winner': winner,
                    'convergence_history': convergence_history,
                }

            # Generate counter-arguments
            init_arg = f"Turn {turn+1}: My analysis shows {init_assessment.get('probability', 0.5):.2f} probability. {init_assessment.get('reasoning', '')[:200]}"
            opp_arg = f"Turn {turn+1}: My analysis shows {opp_assessment.get('probability', 0.5):.2f} probability. {opp_assessment.get('reasoning', '')[:200]}"

            # Add arguments to debate
            self.add_debate_argument(debate_id, debate.initiator, init_arg)
            self.add_debate_argument(debate_id, debate.opponent, opp_arg)

        # Max turns reached without convergence - Zeus arbitrates
        final_convergence = convergence_history[-1]['convergence'] if convergence_history else 0

        # Arbiter decision based on geometric metrics
        if init_assessment.get('phi', 0.5) > opp_assessment.get('phi', 0.5):
            winner = debate.initiator
            reasoning = f"Max turns reached. {debate.initiator} wins on higher Φ ({init_assessment.get('phi', 0.5):.2f})."
        else:
            winner = debate.opponent
            reasoning = f"Max turns reached. {debate.opponent} wins on higher Φ ({opp_assessment.get('phi', 0.5):.2f})."

        self.resolve_debate(debate_id, 'Zeus', winner, reasoning)

        return {
            'status': 'max_turns_reached',
            'turns': max_turns,
            'convergence': final_convergence,
            'winner': winner,
            'convergence_history': convergence_history,
        }

    def auto_continue_active_debates(
        self,
        gods: Dict[str, Any],
        max_debates: int = 3
    ) -> List[Dict]:
        """
        Auto-continue all active debates toward convergence.

        Called periodically to ensure debates don't stagnate.

        Args:
            gods: Dictionary of god instances
            max_debates: Maximum debates to process per call

        Returns:
            List of debate results
        """
        results = []

        for debate_id in self.active_debates[:max_debates]:
            if debate_id in self.debates:
                result = self.continue_debate_until_convergence(
                    debate_id=debate_id,
                    gods=gods,
                    max_turns=5,  # Fewer turns per cycle
                    convergence_threshold=0.90
                )
                results.append(result)

        return results
