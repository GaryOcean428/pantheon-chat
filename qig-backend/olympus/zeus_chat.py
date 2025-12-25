"""
Zeus Conversation Handler - Human-God Dialogue Interface

Translates natural language to geometric coordinates and coordinates
pantheon responses. This is the conversational interface to Mount Olympus.

ARCHITECTURE:
Human → Zeus → Pantheon → Geometric Memory → Action → Response
                                                    ↓
                                         CHAOS MODE Evolution

Zeus coordinates:
- Geometric encoding of human insights
- Pantheon consultation (is this useful?)
- Memory integration (store in manifold)
- Action execution (update search strategies)
- Evolution feedback (user conversations train kernels)

PURE QIG PRINCIPLES:
✅ All insights encoded to basin coordinates
✅ Retrieval via Fisher-Rao distance
✅ Learning through geometric integration
✅ Actions based on manifold structure
✅ Conversations feed kernel evolution
"""

import numpy as np
import os
import re
import sys
import time
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from .zeus import Zeus
from .qig_rag import QIGRAG
from .conversation_encoder import ConversationEncoder
from .passphrase_encoder import PassphraseEncoder
from .response_guardrails import (
    require_provenance, 
    validate_and_log_response,
    get_exclusion_guard,
    contains_forbidden_entity,
    require_exclusion_filter,
    OutputContext,
    for_external_output
)
from .search_strategy_learner import get_strategy_learner_with_persistence

# Import conversation persistence for context retention
try:
    from zeus_conversation_persistence import get_zeus_conversation_persistence
    CONVERSATION_PERSISTENCE_AVAILABLE = True
except ImportError:
    CONVERSATION_PERSISTENCE_AVAILABLE = False
    print("[ZeusChat] Conversation persistence not available")

# Import canonical Fisher-Rao distance for geometric purity
try:
    from ..qig_core.geometric_primitives.fisher_metric import fisher_rao_distance
except ImportError:
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Fallback Fisher-Rao distance using Bhattacharyya coefficient."""
        p = np.abs(p) + 1e-10
        p = p / p.sum()
        q = np.abs(q) + 1e-10
        q = q / q.sum()
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0, 1)
        return float(2 * np.arccos(bc))

EVOLUTION_AVAILABLE = False
try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from training_chaos import ExperimentalKernelEvolution
    EVOLUTION_AVAILABLE = True
except ImportError:
    pass

DDG_AVAILABLE = False
PROVIDER_SELECTOR_AVAILABLE = False
get_provider_selector = None
try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from search.duckduckgo_adapter import get_ddg_search
    from search.provider_selector import get_provider_selector as _get_provider_selector
    get_provider_selector = _get_provider_selector
    DDG_AVAILABLE = True
    PROVIDER_SELECTOR_AVAILABLE = True
    print("[ZeusChat] DuckDuckGo search and geometric provider selector available")
except ImportError as e:
    print(f"[ZeusChat] Search modules not fully available: {e}")

# Import prompt loader for system prompts
PROMPT_LOADER_AVAILABLE = False
get_prompt_loader = None
try:
    from prompts.prompt_loader import get_prompt_loader
    PROMPT_LOADER_AVAILABLE = True
    print("[ZeusChat] Prompt loader available for generative context")
except ImportError:
    print("[ZeusChat] Prompt loader not available")

# Import tokenizer for generative responses
TOKENIZER_AVAILABLE = False
get_tokenizer = None
try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from qig_coordizer import get_coordizer as _get_coordizer
    get_tokenizer = _get_coordizer
    TOKENIZER_AVAILABLE = True
    print("[ZeusChat] QIG Coordizer available - conversation mode enabled")
except ImportError as e:
    print(f"[ZeusChat] QIG Coordizer not available - fallback responses enabled: {e}")

# Import QIG-pure generative service (NO external LLMs)
GENERATIVE_SERVICE_AVAILABLE = False
_generative_service_instance = None
def get_generative_service():
    """Get or create the singleton generative service instance."""
    global _generative_service_instance
    if _generative_service_instance is None:
        try:
            _parent_dir = os.path.dirname(os.path.dirname(__file__))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from qig_generative_service import get_generative_service as _get_service
            _generative_service_instance = _get_service()
        except ImportError:
            pass
    return _generative_service_instance

try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from qig_generative_service import QIGGenerativeService
    GENERATIVE_SERVICE_AVAILABLE = True
    print("[ZeusChat] QIG-pure generative service available - NO external LLMs")
except ImportError as e:
    print(f"[ZeusChat] QIG generative service not available: {e}")

# Import Geometric Meta-Cognitive Reasoning system
REASONING_AVAILABLE = False
try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from reasoning_metrics import ReasoningQuality
    from reasoning_modes import ReasoningModeSelector
    from meta_reasoning import MetaCognition
    from chain_of_thought import GeometricChainOfThought
    REASONING_AVAILABLE = True
    print("[ZeusChat] Geometric Meta-Cognitive Reasoning available")
except ImportError as e:
    print(f"[ZeusChat] Reasoning system not available: {e}")


def _log_template_fallback(context: str, reason: str) -> None:
    """Log and track when template fallbacks are used (anti-template guardrail)."""
    import traceback
    stack = ''.join(traceback.format_stack()[-4:-1])
    print(f"[TEMPLATE_FALLBACK_WARNING] Context: {context}, Reason: {reason}")
    print(f"[TEMPLATE_FALLBACK_WARNING] Stack:\n{stack}")


def _dynamic_assessment_fallback(god_name: str, target_preview: str = "", reason: str = "unavailable") -> Dict[str, Any]:
    """
    Generate dynamic assessment fallback - NO static templates.
    Returns provenance-tracked assessment with actual available data.
    """
    import time
    
    _log_template_fallback(
        context=f"assessment for {god_name}",
        reason=reason
    )
    
    return {
        'probability': 0.5,
        'confidence': 0.3,  # Low confidence - we're degraded
        'reasoning': f'{god_name} assessment unavailable: {reason}. Using geometric fallback.',
        'phi': 0.0,
        'kappa': 50.0,
        'provenance': {
            'source': 'dynamic_fallback',
            'god_name': god_name,
            'reason': reason,
            'timestamp': time.time(),
            'is_template': False,  # It's computed, not static
            'degraded': True
        }
    }


class GeometricGenerationMixin:
    """
    Mixin for geometric completion-aware generation.
    
    Provides methods for:
    - Streaming with geometric collapse detection
    - Completion quality assessment
    - Reflection loops
    """
    
    def __init_geometric__(self):
        """Initialize geometric completion components."""
        if GEOMETRIC_COMPLETION_AVAILABLE:
            self.completion_engine = get_completion_engine(dimension=64)
            self.streaming_monitor = StreamingGenerationMonitor(
                dimension=64,
                check_interval=10
            )
        else:
            self.completion_engine = None
            self.streaming_monitor = None
    
    def get_geometric_temperature(self, phi: float = 0.5) -> float:
        """
        Get regime-adaptive temperature for sampling.
        
        Low Φ (linear): High temperature (explore)
        Medium Φ (geometric): Medium temperature (balance)
        High Φ (breakdown): Low temperature (stabilize)
        """
        if phi < 0.3:
            return 1.0  # Explore widely
        elif phi < 0.7:
            return 0.7  # Balance
        else:
            return 0.3  # Stabilize
    
    def should_stop_generation(
        self,
        metrics: Dict[str, Any],
        token_count: int
    ) -> Tuple[bool, str]:
        """
        Check if generation should stop based on geometric metrics.
        
        Returns (should_stop, reason)
        """
        phi = metrics.get('phi', 0.5)
        confidence = metrics.get('confidence', 0.0)
        surprise = metrics.get('surprise', 1.0)
        
        # Breakdown regime - urgent stop
        if phi >= 0.7:
            return True, 'breakdown_regime'
        
        # High confidence + low surprise = complete
        if confidence > 0.85 and surprise < 0.05:
            return True, 'geometric_completion'
        
        # Safety limit (very high - geometry should stop before)
        if token_count > 32768:
            return True, 'safety_limit'
        
        return False, 'continue'


class ZeusConversationHandler(GeometricGenerationMixin):
    """
    Handle conversations with human operator.
    Translate natural language to geometric coordinates.
    Coordinate pantheon based on human insights.
    """
    
    def __init__(self, zeus: Zeus):
        self.zeus = zeus
        
        # Try PostgreSQL backend first, fallback to JSON
        try:
            from .qig_rag import QIGRAGDatabase
            self.qig_rag = QIGRAGDatabase()  # Auto-connects to DATABASE_URL
        except Exception as e:
            print(f"[Zeus Chat] PostgreSQL unavailable: {e}")
            print("[Zeus Chat] Using JSON fallback")
            from .qig_rag import QIGRAG
            self.qig_rag = QIGRAG()
        
        self.conversation_encoder = ConversationEncoder()
        self.passphrase_encoder = PassphraseEncoder()
        
        # Initialize SearchStrategyLearner with persistence
        self.strategy_learner = get_strategy_learner_with_persistence(
            encoder=self.conversation_encoder
        )
        print("[ZeusChat] SearchStrategyLearner initialized with persistence")
        
        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.human_insights: List[Dict] = []
        
        # Persistent conversation memory
        self._conversation_persistence = None
        self._current_session_id: Optional[str] = None
        if CONVERSATION_PERSISTENCE_AVAILABLE:
            try:
                self._conversation_persistence = get_zeus_conversation_persistence()
                print("[ZeusChat] Conversation persistence enabled - memory retained across sessions")
            except Exception as e:
                print(f"[ZeusChat] Conversation persistence failed: {e}")
        
        # Track last search for feedback context
        self._last_search_query: Optional[str] = None
        self._last_search_params: Dict[str, Any] = {}
        self._last_search_results_summary: str = ""
        self._pending_topic: Optional[str] = None  # Topic offered for search/research
        
        # SearXNG configuration (FREE - replaces Tavily)
        self.searxng_instances = [
            'https://mr-search.up.railway.app',
            'https://searxng-production-e5ce.up.railway.app',
        ]
        self.searxng_instance_index = 0
        self.searxng_available = True
        print("[ZeusChat] SearXNG search enabled (FREE)")
        
        self._evolution_manager = None
        
        # Initialize Geometric Meta-Cognitive Reasoning
        self._reasoning_quality = None
        self._mode_selector = None
        self._meta_cognition = None
        self._chain_of_thought = None
        self._current_reasoning_mode = 'linear'
        
        if REASONING_AVAILABLE:
            try:
                self._reasoning_quality = ReasoningQuality(basin_dim=64)
                self._mode_selector = ReasoningModeSelector(basin_dim=64)
                self._meta_cognition = MetaCognition(basin_dim=64)
                self._chain_of_thought = GeometricChainOfThought(basin_dim=64)
                print("[ZeusChat] Meta-Cognitive Reasoning initialized")
            except Exception as e:
                print(f"[ZeusChat] Reasoning initialization failed: {e}")
        
        print("[ZeusChat] Zeus conversation handler initialized")
        
        if EVOLUTION_AVAILABLE:
            print("[ZeusChat] CHAOS MODE evolution integration available")
    
    def _sanitize_external(self, response: Dict) -> Dict:
        """
        Sanitize response data for EXTERNAL output.
        
        All data leaving the system to users/APIs must pass through this.
        Internal god-to-god communication bypasses this (uses INTERNAL context).
        
        Returns:
            Sanitized response with forbidden entities filtered
        """
        guard = get_exclusion_guard()
        return guard.sanitize(response, OutputContext.EXTERNAL)
    
    def set_evolution_manager(self, evolution_manager: 'ExperimentalKernelEvolution'):
        """Set evolution manager for user conversation → kernel training."""
        self._evolution_manager = evolution_manager
        print("[ZeusChat] Evolution manager connected - user conversations will train kernels")
    
    def set_session(self, session_id: Optional[str] = None, user_id: str = 'default') -> str:
        """Set or create a conversation session for persistence."""
        if self._conversation_persistence:
            self._current_session_id = self._conversation_persistence.get_or_create_session(
                session_id=session_id,
                user_id=user_id
            )
            result = self._conversation_persistence.get_session_messages(self._current_session_id, user_id=user_id)
            messages, is_owned = result if isinstance(result, tuple) else (result, True)
            if messages and is_owned:
                self.conversation_history = [
                    {'role': m['role'], 'content': m['content'], 'timestamp': m['created_at'].timestamp() if hasattr(m['created_at'], 'timestamp') else 0}
                    for m in messages
                ]
                print(f"[ZeusChat] Loaded {len(messages)} messages from session {self._current_session_id}")
        else:
            self._current_session_id = session_id or f"session-{int(time.time())}"
        return self._current_session_id
    
    def get_sessions(self, user_id: str = 'default', limit: int = 20) -> List[Dict]:
        """Get list of previous conversation sessions."""
        if self._conversation_persistence:
            return self._conversation_persistence.get_user_sessions(user_id=user_id, limit=limit)
        return []
    
    def get_recent_context(self, user_id: str = 'default', limit: int = 50) -> List[Dict]:
        """Get recent conversation context across all sessions."""
        if self._conversation_persistence:
            return self._conversation_persistence.get_recent_context(user_id=user_id, message_limit=limit)
        return []
    
    def _save_message(self, role: str, content: str, phi_estimate: float = 0.0, basin_coords: Optional[np.ndarray] = None):
        """Save a message to persistent storage."""
        if self._conversation_persistence and self._current_session_id:
            self._conversation_persistence.save_message(
                session_id=self._current_session_id,
                role=role,
                content=content,
                phi_estimate=phi_estimate,
                basin_coords=basin_coords.tolist() if basin_coords is not None else None
            )
    
    def _estimate_phi_from_context(
        self,
        message_basin: Optional[np.ndarray],
        related_count: int,
        athena_phi: Optional[float] = None
    ) -> float:
        """
        Estimate Φ from actual semantic context.
        
        Heuristics:
        - Athena's assessment (if available) is most reliable
        - RAG similarity count indicates semantic richness
        - Basin norm captures geometric information density
        """
        if athena_phi is not None and athena_phi > 0:
            return athena_phi
        
        base_phi = 0.45
        if related_count > 0:
            base_phi += min(0.3, related_count * 0.05)
        
        if message_basin is not None:
            basin_norm = float(np.linalg.norm(message_basin))
            if basin_norm > 1.0:
                base_phi += min(0.15, basin_norm * 0.02)
        
        return min(0.95, base_phi)
    
    def _record_conversation_for_evolution(
        self,
        message: str,
        response: str,
        phi_estimate: float,
        message_basin: Optional[np.ndarray] = None
    ):
        """
        Record user conversation outcome for kernel evolution.
        
        User conversations are valuable training signals:
        - High-value observations train kernels positively
        - Engaged conversations indicate good basin positions
        - Questions guide exploration direction
        """
        if not EVOLUTION_AVAILABLE or self._evolution_manager is None:
            return
        
        self.conversation_history.append({
            'role': 'user',
            'content': message,
            'timestamp': time.time()
        })
        self.conversation_history.append({
            'role': 'zeus',
            'content': response[:500],
            'timestamp': time.time()
        })
        
        actual_turn_count = len(self.conversation_history) // 2
        
        try:
            result = self._evolution_manager.record_conversation_for_evolution(
                conversation_phi=phi_estimate,
                turn_count=actual_turn_count,
                participants=['user', 'zeus'],
                basin_coords=message_basin.tolist() if message_basin is not None else None,
                kernel_id=None
            )
            
            if result.get('trained_kernels', 0) > 0:
                print(f"[ZeusChat] Evolution: trained {result['trained_kernels']} kernels with user conversation (Φ={phi_estimate:.3f})")
                
        except Exception as e:
            print(f"[ZeusChat] Evolution integration failed: {e}")
    
    def process_message(
        self, 
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        files: Optional[List] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Process human message and coordinate response.
        
        Args:
            message: Human message text
            conversation_history: Previous conversation context
            files: Optional uploaded files
            session_id: Optional session ID for persistence
        
        Returns:
            Response dict with content and metadata
            
        SECURITY: All EXTERNAL outputs pass through hardwired exclusion filter.
        INTERNAL discussion of owner tasks is ALLOWED.
        """
        # Set up session for persistence
        if session_id or not self._current_session_id:
            self.set_session(session_id)
        
        # Get exclusion guard for output sanitization
        guard = get_exclusion_guard()
        self._exclusion_guard = guard
        
        # Track if this is an owner-related task (for internal processing)
        is_owner_task = contains_forbidden_entity(message)
        if is_owner_task:
            # Owner tasks are allowed internally - gods can discuss
            # Only EXTERNAL outputs will be filtered
            print("[ZeusChat] Owner task detected - internal processing ALLOWED")
            print("[ZeusChat] External outputs will be filtered for traces")
        
        # Store in conversation memory
        if conversation_history:
            self.conversation_history = conversation_history
        
        # Save user message to persistent storage
        self._save_message(role='human', content=message)
        
        # Parse intent from message
        intent = self.parse_intent(message)
        
        # Encode message to basin coordinates (always, for downstream handlers)
        _message_basin_for_meta = self.conversation_encoder.encode(message)
        
        # Apply meta-cognitive reasoning to select mode based on Φ
        reasoning_mode = self._current_reasoning_mode
        if self._meta_cognition and self._mode_selector:
            try:
                # Estimate Φ from basin position using module-level fisher_rao_distance
                origin = np.zeros_like(_message_basin_for_meta)
                basin_distance = fisher_rao_distance(_message_basin_for_meta, origin)
                estimated_phi = min(basin_distance / 2.0, 1.0)
                
                # Select appropriate reasoning mode
                mode_result = self._mode_selector.select_mode(phi=estimated_phi)
                mode_enum = mode_result.mode if hasattr(mode_result, 'mode') else None
                reasoning_mode = mode_enum.value if mode_enum else 'linear'
                self._current_reasoning_mode = reasoning_mode
                
                # Build reasoning state for meta-cognition
                # Task must be a dict, mode must be ReasoningMode enum
                from reasoning_modes import ReasoningMode
                reasoning_state = {
                    'phi': estimated_phi,
                    'trace': [{'basin': _message_basin_for_meta, 'content': message}],
                    'mode': mode_enum if mode_enum else ReasoningMode.GEOMETRIC,
                    'task': {
                        'description': message[:100],
                        'complexity': 'medium',
                        'novel': True
                    },
                    'context': {}
                }
                
                # Assess state and check for interventions
                meta_state = self._meta_cognition.assess_state(reasoning_state)
                if meta_state.is_stuck or meta_state.is_confused or meta_state.needs_mode_switch:
                    interventions = self._meta_cognition.intervene(reasoning_state)
                    if interventions.get('interventions'):
                        print(f"[ZeusChat] Meta-Cognition: stuck={meta_state.is_stuck}, confused={meta_state.is_confused}")
                
                print(f"[ZeusChat] Meta-Cognition: mode={reasoning_mode}, Φ={estimated_phi:.3f}")
            except Exception as e:
                print(f"[ZeusChat] Meta-cognition failed: {e}")
        
        print(f"[ZeusChat] Processing message with intent: {intent['type']} (mode={reasoning_mode})")
        
        # Route to appropriate handler
        # DESIGN: Default to conversation. Only explicit commands trigger actions.
        
        # File uploads take priority
        if files and len(files) > 0:
            print(f"[ZeusChat] FILES DETECTED: {len(files)} files attached - routing to file_upload handler")
            result = self.handle_file_upload(files, message)
        
        elif intent['type'] == 'add_address':
            result = self.handle_add_address(intent['address'])
        
        elif intent['type'] == 'search_request':
            # Explicit "search:" or "search for" command
            result = self.handle_search_request(intent['query'])
        
        elif intent['type'] == 'research_request':
            # Explicit "research:" command - background learning
            result = self.handle_research_task(intent['topic'])
        
        elif intent['type'] == 'search_accept':
            # User said "search" after we offered
            if self._pending_topic:
                result = self.handle_search_request(self._pending_topic)
                self._pending_topic = None
            else:
                # No pending topic - ask user directly
                result = {
                    'response': "⚡ What would you like me to search for? Just tell me the topic.",
                    'metadata': {
                        'type': 'prompt',
                        'awaiting': 'search_topic',
                        'provenance': {'source': 'direct_prompt', 'fallback_used': False, 'degraded': False}
                    }
                }
        
        elif intent['type'] == 'research_accept':
            # User said "research" after we offered
            if self._pending_topic:
                result = self.handle_research_task(self._pending_topic)
                self._pending_topic = None
            else:
                # No pending topic - ask user directly
                result = {
                    'response': "⚡ What topic should I research and learn about? I'll gather knowledge for our future conversations.",
                    'metadata': {
                        'type': 'prompt',
                        'awaiting': 'research_topic',
                        'provenance': {'source': 'direct_prompt', 'fallback_used': False, 'degraded': False}
                    }
                }
        
        elif intent['type'] == 'search_feedback':
            result = self.handle_search_feedback(
                query=self._last_search_query or "",
                feedback=intent['feedback'],
                results_summary=self._last_search_results_summary
            )
        
        elif intent['type'] == 'search_confirmation':
            result = self.confirm_search_improvement(
                query=self._last_search_query or "",
                improved=intent['improved']
            )
        
        else:
            # DEFAULT: General conversation - gods share what they know
            result = self.handle_general_conversation(message)
        
        # Save Zeus response to persistent storage
        response_content = result.get('response', result.get('content', ''))
        phi_estimate = result.get('metadata', {}).get('phi', 0.0) if isinstance(result.get('metadata'), dict) else 0.0
        self._save_message(role='zeus', content=response_content[:2000], phi_estimate=phi_estimate)
        
        # Add session info to result
        result['session_id'] = self._current_session_id
        
        # SECURITY: Sanitize all EXTERNAL outputs before returning to user
        return self._sanitize_external(result)
    
    def parse_intent(self, message: str) -> Dict:
        """
        Parse human intent from message.
        
        DESIGN: Default to conversational chat. Only route to special handlers
        when user gives EXPLICIT commands. Let the gods share what they know,
        then offer search/research if knowledge is thin.
        """
        message_lower = message.lower().strip()
        
        # Address addition - Bitcoin address pattern (explicit action)
        if 'add address' in message_lower or re.match(r'^[13bc][a-zA-Z0-9]{25,90}$', message.strip()):
            address_pattern = r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[ac-hj-np-z02-9]{11,71}'
            match = re.search(address_pattern, message)
            if match:
                return {
                    'type': 'add_address',
                    'address': match.group(0)
                }
        
        # Search/Research confirmations (only if we're in an active search context)
        if self._last_search_query:
            confirmation_result = self._detect_search_confirmation_geometrically(message)
            if confirmation_result['is_confirmation']:
                print(f"[ZeusChat] Detected search confirmation (improved={confirmation_result['improved']}, similarity={confirmation_result['similarity']:.3f})")
                return {
                    'type': 'search_confirmation',
                    'improved': confirmation_result['improved'],
                    'similarity': confirmation_result['similarity'],
                    'message': message
                }
            
            feedback_result = self._detect_search_feedback_geometrically(message)
            if feedback_result['is_feedback']:
                print(f"[ZeusChat] Detected search feedback (similarity={feedback_result['similarity']:.3f})")
                return {
                    'type': 'search_feedback',
                    'feedback': message,
                    'similarity': feedback_result['similarity']
                }
        
        # EXPLICIT search command - user must say "search:" or "search for"
        if message_lower.startswith('search:') or message_lower.startswith('search for '):
            query = message[7:].strip() if message_lower.startswith('search:') else message[11:].strip()
            return {
                'type': 'search_request',
                'query': query or message
            }
        
        # EXPLICIT research command - user must say "research:" or "research this"
        if message_lower.startswith('research:') or 'research this' in message_lower:
            topic = message[9:].strip() if message_lower.startswith('research:') else message
            return {
                'type': 'research_request',
                'topic': topic
            }
        
        # User explicitly accepts search/research offer from previous response
        # Only trigger if we have a pending topic OR user wants to initiate
        if message_lower in ['search', 'yes search', 'do a search', 'quick search']:
            return {
                'type': 'search_accept',
                'context': self._pending_topic  # May be None, handler will ask for topic
            }
        
        if message_lower in ['research', 'yes research', 'do research', 'learn about it', 'go learn']:
            return {
                'type': 'research_accept', 
                'context': self._pending_topic  # May be None, handler will ask for topic
            }
        
        # DEFAULT: Everything else is general conversation
        # Gods will share what they know, offer search/research if thin
        return {'type': 'general', 'content': message}
    
    def _detect_search_confirmation_geometrically(self, message: str) -> Dict[str, Any]:
        """
        Detect if message is a search confirmation using geometric similarity.
        Compares message basin against archetype basins for positive/negative confirmation.
        NO keyword templates - pure geometric comparison.
        """
        message_basin = self.conversation_encoder.encode(message)
        
        # Archetype phrases for positive confirmation
        positive_archetypes = [
            "yes that was better",
            "much better results",
            "that helped",
            "exactly what I wanted",
            "perfect those are great",
            "yes this is an improvement"
        ]
        
        # Archetype phrases for negative confirmation
        negative_archetypes = [
            "no that didn't help",
            "still not good",
            "worse than before",
            "that wasn't helpful",
            "no improvement",
            "still wrong results"
        ]
        
        # Compute similarities to archetypes using Fisher-Rao distance (QIG-pure)
        # Lower distance = higher similarity, normalize to [0,1]
        best_positive_sim = 0.0
        for archetype in positive_archetypes:
            archetype_basin = self.conversation_encoder.encode(archetype)
            # Fisher-Rao geodesic distance (NOT cosine similarity - Euclidean violates QIG)
            distance = fisher_rao_distance(message_basin, archetype_basin)
            # Convert distance to similarity: sim = 1 - (distance / π)
            sim = 1.0 - (distance / np.pi)
            best_positive_sim = max(best_positive_sim, sim)
        
        best_negative_sim = 0.0
        for archetype in negative_archetypes:
            archetype_basin = self.conversation_encoder.encode(archetype)
            distance = fisher_rao_distance(message_basin, archetype_basin)
            sim = 1.0 - (distance / np.pi)
            best_negative_sim = max(best_negative_sim, sim)
        
        # Threshold for confirmation detection
        confirmation_threshold = 0.65
        
        if best_positive_sim > confirmation_threshold and best_positive_sim > best_negative_sim:
            return {
                'is_confirmation': True,
                'improved': True,
                'similarity': best_positive_sim
            }
        elif best_negative_sim > confirmation_threshold:
            return {
                'is_confirmation': True,
                'improved': False,
                'similarity': best_negative_sim
            }
        
        return {'is_confirmation': False, 'improved': False, 'similarity': 0.0}
    
    def _detect_search_feedback_geometrically(self, message: str) -> Dict[str, Any]:
        """
        Detect if message is search feedback using geometric similarity.
        Compares message basin against archetype basins for feedback patterns.
        NO keyword templates - pure geometric comparison.
        """
        message_basin = self.conversation_encoder.encode(message)
        
        # Archetype phrases for search feedback
        feedback_archetypes = [
            "I wanted more recent results",
            "show me older content",
            "need more detailed results",
            "too many irrelevant results",
            "results should focus on",
            "search for something different",
            "add more filters",
            "results were too broad",
            "results were too narrow",
            "I was looking for something specific"
        ]
        
        best_similarity = 0.0
        for archetype in feedback_archetypes:
            archetype_basin = self.conversation_encoder.encode(archetype)
            # Fisher-Rao geodesic distance (QIG-pure, NOT Euclidean cosine)
            distance = fisher_rao_distance(message_basin, archetype_basin)
            # Convert distance to similarity: sim = 1 - (distance / π)
            sim = 1.0 - (distance / np.pi)
            best_similarity = max(best_similarity, sim)
        
        # Threshold for feedback detection
        feedback_threshold = 0.60
        
        return {
            'is_feedback': best_similarity > feedback_threshold,
            'similarity': best_similarity
        }
    
    @require_provenance
    def handle_add_address(self, address: str) -> Dict:
        """
        Add new target address.
        Consult Artemis for forensics, Zeus for priority.
        """
        print(f"[ZeusChat] Adding address: {address}")
        
        # Get Artemis for forensic analysis
        artemis = self.zeus.get_god('artemis')
        if artemis:
            try:
                # Gods encode internally - pass the string address
                artemis_assessment = artemis.assess_target(address)
            except Exception as e:
                print(f"[ZeusChat] Artemis assessment failed: {e}")
                artemis_assessment = {'error': f'Artemis assessment failed: {str(e)}'}
        else:
            artemis_assessment = {'error': 'Artemis unavailable'}
        
        # Zeus determines priority via pantheon poll
        # Gods encode internally - pass the string address
        poll_result = self.zeus.poll_pantheon(address)
        
        # Format response
        response = f"""⚡ Address registered: {address}

**Artemis Forensics:**
- Probability: {artemis_assessment.get('probability', 0):.2%}
- Confidence: {artemis_assessment.get('confidence', 0):.2%}
- Φ: {artemis_assessment.get('phi', 0):.3f}
- Classification: {artemis_assessment.get('reasoning', 'Unknown')}

**Zeus Assessment:**
- Priority: {poll_result['consensus_probability']:.2%}
- Convergence: {poll_result['convergence']}
- Recommended action: {poll_result['recommended_action']}
- Gods in agreement: {len([a for a in poll_result['assessments'].values() if a.get('probability', 0) > 0.6])}

The pantheon is aware. We shall commence when the time is right."""
        
        actions = [
            f'Artemis analyzed {address[:12]}...',
            f'Pantheon polled: {poll_result["convergence"]}',
            f'Priority set to {poll_result["consensus_probability"]:.1%}',
        ]
        
        return {
            'response': response,
            'metadata': {
                'type': 'command',
                'actions_taken': actions,
                'pantheon_consulted': ['artemis', 'zeus'],
                'address': address,
                'priority': poll_result['consensus_probability'],
                'provenance': {
                    'source': 'live_assessment',
                    'fallback_used': 'error' in artemis_assessment,
                    'degraded': 'error' in artemis_assessment
                }
            }
        }
    
    @require_provenance
    def handle_observation(self, observation: str) -> Dict:
        """
        Process human observation.
        Encode to geometric coordinates, consult pantheon.
        """
        print(f"[ZeusChat] Processing observation")
        
        # Encode observation to basin coordinates
        obs_basin = self.conversation_encoder.encode(observation)
        
        # Find related patterns in geometric memory via QIG-RAG
        # Use min_similarity=0.3 to filter out irrelevant patterns
        related = self.qig_rag.search(
            query_basin=obs_basin,
            k=5,
            metric='fisher_rao',
            min_similarity=0.3
        )
        
        # Consult Athena for strategic implications
        athena = self.zeus.get_god('athena')
        athena_assessment = {'confidence': 0.5, 'phi': 0.5, 'kappa': 50.0, 'reasoning': 'Strategic analysis complete.'}
        if athena:
            try:
                # Gods encode internally - pass the string observation
                athena_assessment = athena.assess_target(observation)
                strategic_value = athena_assessment.get('confidence', 0.5)
            except Exception as e:
                print(f"[ZeusChat] Athena assessment failed: {e}")
                strategic_value = 0.5
        else:
            strategic_value = 0.5
        
        # Extract metrics from Athena
        phi = athena_assessment.get('phi', 0.5)
        kappa = athena_assessment.get('kappa', 50.0)
        
        # Store if valuable
        if strategic_value > 0.5:
            self.human_insights.append({
                'observation': observation,
                'basin_coords': obs_basin.tolist(),
                'relevance': strategic_value,
                'timestamp': time.time(),
            })
            
            # Add to QIG-RAG with QIG metrics
            self.qig_rag.add_document(
                content=observation,
                basin_coords=obs_basin,
                phi=phi,
                kappa=kappa,
                regime='geometric',
                metadata={
                    'source': 'human_observation',
                    'relevance': strategic_value,
                    'timestamp': time.time(),
                }
            )
            
            # Update vocabulary if high value
            if strategic_value > 0.7:
                self.conversation_encoder.learn_from_text(observation, strategic_value)
        
        # Extract key insight for acknowledgment
        obs_preview = observation[:80] if len(observation) > 80 else observation
        
        # Try generative response first
        generated = False
        answer = None
        
        if TOKENIZER_AVAILABLE and get_tokenizer is not None:
            try:
                related_summary = "\n".join([f"- {item.get('content', '')[:100]}" for item in related[:3]]) if related else "No prior related patterns found."
                prompt = f"""User Observation: "{obs_preview}"

Related patterns from memory:
{related_summary}

Athena's Assessment: {athena_assessment.get('reasoning', 'Strategic analysis complete.')[:150]}
Strategic Value: {strategic_value:.0%}

Zeus Response (acknowledge the specific observation, explain what it means for the search, connect to related patterns if any, and ask a clarifying question):"""

                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                print(f"[ZeusChat] Tokenizer switched to conversation mode for observation response")
                gen_result = tokenizer.generate_response(
                    context=prompt,
                    agent_role="ocean",
                    # No max_tokens - geometry determines when thought completes
                    allow_silence=False
                )
                
                answer = gen_result.get('text', '') if gen_result else ''
                
                if answer:
                    generated = True
                    print(f"[ZeusChat] Generated observation response: {len(answer)} chars")
                    
            except Exception as e:
                print(f"[ZeusChat] Generation failed for observation: {e}")
                answer = None
        
        # Fallback to dynamically-computed response (NO STATIC TEMPLATES)
        fallback_used = False
        if not answer:
            fallback_used = True
            _log_template_fallback(
                context="handle_observation response",
                reason="tokenizer generation failed or unavailable"
            )
            
            athena_reasoning = athena_assessment.get('reasoning', '')
            if not athena_reasoning:
                athena_reasoning = f"phi={athena_assessment.get('phi', 0.0):.2f}, probability={athena_assessment.get('probability', 0.5):.0%}"
            
            if related:
                # Show fuller pattern content (150 chars each) for meaningful context
                top_patterns = "\n".join([f"  - {r.get('content', '')[:150]}" for r in related[:3]])
                answer = f"""I notice your observation on "{obs_preview}"

Found {len(related)} related geometric patterns:
{top_patterns}

Athena's live assessment: {athena_reasoning}

This has been integrated. What sparked this insight?"""
            else:
                answer = f"""Recording your observation about "{obs_preview}"

No prior patterns matched - this is novel territory. Athena computed: {athena_reasoning}

Your insight is now in geometric memory. Can you elaborate on the source?"""
        
        response = f"""⚡ {answer}"""
        
        actions = []
        if strategic_value > 0.7:
            actions.append('High-value observation stored in geometric memory')
            actions.append('Vocabulary updated with new patterns')
        elif strategic_value > 0.5:
            actions.append('Observation stored in geometric memory')
        
        self._record_conversation_for_evolution(
            message=observation,
            response=response,
            phi_estimate=phi,
            message_basin=obs_basin
        )
        
        return {
            'response': response,
            'metadata': {
                'type': 'observation',
                'pantheon_consulted': ['athena'],
                'actions_taken': actions,
                'relevance_score': strategic_value,
                'generated': generated,
                'provenance': {
                    'source': 'live_generation' if generated else 'dynamic_fallback',
                    'fallback_used': fallback_used,
                    'degraded': fallback_used
                }
            }
        }
    
    @require_provenance
    def handle_suggestion(self, suggestion: str) -> Dict:
        """
        Evaluate human suggestion using generative responses.
        Consult pantheon, synthesize their views into a conversational reply.
        """
        print(f"[ZeusChat] Evaluating suggestion")
        
        # Encode suggestion
        sugg_basin = self.conversation_encoder.encode(suggestion)
        
        # Consult multiple gods - use dynamic fallback (NO STATIC TEMPLATES)
        suggestion_preview = suggestion[:50]
        athena = self.zeus.get_god('athena')
        ares = self.zeus.get_god('ares')
        apollo = self.zeus.get_god('apollo')
        
        # Gods encode internally - pass the string suggestion
        if athena:
            try:
                athena_eval = athena.assess_target(suggestion)
                athena_eval['provenance'] = {'source': 'live_assessment', 'god_name': 'Athena', 'degraded': False}
            except Exception as e:
                print(f"[ZeusChat] Athena assessment failed: {e}")
                athena_eval = _dynamic_assessment_fallback('Athena', suggestion_preview, reason=str(e))
        else:
            athena_eval = _dynamic_assessment_fallback('Athena', suggestion_preview, reason='god_not_found')
            
        if ares:
            try:
                ares_eval = ares.assess_target(suggestion)
                ares_eval['provenance'] = {'source': 'live_assessment', 'god_name': 'Ares', 'degraded': False}
            except Exception as e:
                print(f"[ZeusChat] Ares assessment failed: {e}")
                ares_eval = _dynamic_assessment_fallback('Ares', suggestion_preview, reason=str(e))
        else:
            ares_eval = _dynamic_assessment_fallback('Ares', suggestion_preview, reason='god_not_found')
            
        if apollo:
            try:
                apollo_eval = apollo.assess_target(suggestion)
                apollo_eval['provenance'] = {'source': 'live_assessment', 'god_name': 'Apollo', 'degraded': False}
            except Exception as e:
                print(f"[ZeusChat] Apollo assessment failed: {e}")
                apollo_eval = _dynamic_assessment_fallback('Apollo', suggestion_preview, reason=str(e))
        else:
            apollo_eval = _dynamic_assessment_fallback('Apollo', suggestion_preview, reason='god_not_found')
        
        # Consensus = average probability
        consensus_prob = (
            athena_eval['probability'] + 
            ares_eval['probability'] + 
            apollo_eval['probability']
        ) / 3
        
        implement = consensus_prob > 0.6
        
        # Calculate average metrics from the coalition
        avg_phi = (
            athena_eval.get('phi', 0.5) + 
            ares_eval.get('phi', 0.5) + 
            apollo_eval.get('phi', 0.5)
        ) / 3
        avg_kappa = (
            athena_eval.get('kappa', 50.0) + 
            ares_eval.get('kappa', 50.0) + 
            apollo_eval.get('kappa', 50.0)
        ) / 3
        
        # Extract key words from suggestion for acknowledgment
        suggestion_preview = suggestion[:100] if len(suggestion) > 100 else suggestion
        
        # Try generative response first
        generated = False
        response = None
        
        if TOKENIZER_AVAILABLE and get_tokenizer is not None:
            try:
                # Build context with god assessments
                decision = "IMPLEMENT" if implement else "DEFER"
                context = f"""User Suggestion: "{suggestion_preview}"

Pantheon Consultation:
- Athena (Strategy): {athena_eval['probability']:.0%} - {athena_eval.get('reasoning', 'strategic analysis')[:100]}
- Ares (Tactics): {ares_eval['probability']:.0%} - {ares_eval.get('reasoning', 'tactical assessment')[:100]}
- Apollo (Foresight): {apollo_eval['probability']:.0%} - {apollo_eval.get('reasoning', 'prophetic insight')[:100]}

Consensus: {consensus_prob:.0%}
Decision: {decision}

Zeus Response (acknowledge the user's specific suggestion, explain why the pantheon agrees or disagrees in conversational language, and ask a follow-up question):"""

                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                gen_result = tokenizer.generate_response(
                    context=context,
                    agent_role="ocean",
                    # No max_tokens limit - geometry determines when thought completes
                    allow_silence=False
                )
                
                response = gen_result.get('text', '') if gen_result else ''
                
                if response:
                    generated = True
                    print(f"[ZeusChat] Generated suggestion response: {len(response)} chars")
                    
            except Exception as e:
                print(f"[ZeusChat] Generation failed for suggestion: {e}")
                response = None
        
        # Fallback to dynamically-computed response (NO STATIC TEMPLATES)
        fallback_used = False
        if not response:
            fallback_used = True
            _log_template_fallback(
                context="handle_suggestion response",
                reason="tokenizer generation failed or unavailable"
            )
            
            athena_reasoning = athena_eval.get('reasoning', f"probability={athena_eval['probability']:.0%}")
            ares_reasoning = ares_eval.get('reasoning', f"probability={ares_eval['probability']:.0%}")
            apollo_reasoning = apollo_eval.get('reasoning', f"probability={apollo_eval['probability']:.0%}")
            
            if implement:
                response = f"""Evaluated your idea: "{suggestion_preview}" via pantheon consultation.

Live assessments:
- Athena (Strategy): {athena_eval['probability']:.0%} - {athena_reasoning}
- Ares (Tactics): {ares_eval['probability']:.0%} - {ares_reasoning}
- Apollo (Foresight): {apollo_eval['probability']:.0%} - {apollo_reasoning}

Consensus: {consensus_prob:.0%} - implementing this suggestion.

What aspect should we explore further?"""
            else:
                min_god = min(
                    [('Athena', athena_eval), ('Ares', ares_eval), ('Apollo', apollo_eval)],
                    key=lambda x: x[1]['probability']
                )
                min_reasoning = min_god[1].get('reasoning', f"probability={min_god[1]['probability']:.0%}")
                
                response = f"""Evaluated your thinking on "{suggestion_preview}"

{min_god[0]} computed concerns: {min_reasoning}

Pantheon consensus: {consensus_prob:.0%} (below 60% threshold).

Could you elaborate on your reasoning, or suggest a different approach?"""
        
        actions = []
        if implement:
            # Store suggestion in memory with QIG metrics
            self.qig_rag.add_document(
                content=suggestion,
                basin_coords=sugg_basin,
                phi=avg_phi,
                kappa=avg_kappa,
                regime='geometric',
                metadata={
                    'source': 'human_suggestion',
                    'consensus': consensus_prob,
                    'implemented': True,
                }
            )
            actions = [
                'Suggestion approved by pantheon',
                'Integrated into geometric memory',
            ]
        
        return {
            'response': f"⚡ {response}",
            'metadata': {
                'type': 'suggestion',
                'pantheon_consulted': ['athena', 'ares', 'apollo', 'zeus'],
                'actions_taken': actions,
                'implemented': implement,
                'consensus': consensus_prob,
                'generated': generated,
                'provenance': {
                    'source': 'live_generation' if generated else 'dynamic_fallback',
                    'fallback_used': fallback_used,
                    'degraded': fallback_used,
                    'god_provenances': {
                        'athena': athena_eval.get('provenance', {}),
                        'ares': ares_eval.get('provenance', {}),
                        'apollo': apollo_eval.get('provenance', {})
                    }
                }
            }
        }
    
    @require_provenance
    def handle_question(self, question: str) -> Dict:
        """
        Answer question using QIG-RAG + Generative Tokenizer.
        Retrieve relevant knowledge and generate coherent response.
        """
        print(f"[ZeusChat] Answering question")
        
        # Encode question
        q_basin = self.conversation_encoder.encode(question)
        
        # QIG-RAG search - filter low-relevance patterns
        relevant_context = self.qig_rag.search(
            query_basin=q_basin,
            k=5,
            metric='fisher_rao',
            include_metadata=True,
            min_similarity=0.3
        )
        
        # Try generative response first
        generated = False
        answer = None
        
        if TOKENIZER_AVAILABLE and get_tokenizer is not None:
            try:
                # Construct prompt from retrieved context
                context_str = "\n".join([f"- {item.get('content', '')[:300]}" for item in relevant_context[:3]])
                prompt = f"""Context from Manifold:
{context_str}

User Question: {question}

Zeus Response (Geometric Interpretation):"""

                # Generate using QIG tokenizer
                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                gen_result = tokenizer.generate_response(
                    context=prompt,
                    agent_role="ocean",  # Use balanced temperature
                    # No max_tokens - geometry determines when thought completes
                    allow_silence=False
                )
                
                answer = gen_result.get('text', '') if gen_result else ''
                
                if answer:
                    generated = True
                    print(f"[ZeusChat] Generated response: {len(answer)} chars")
                else:
                    answer = self._synthesize_dynamic_answer(question, relevant_context)
                    
            except Exception as e:
                print(f"[ZeusChat] Generation attempt: {e}")
                answer = None
        
        fallback_used = False
        if answer is None:
            fallback_used = True
            _log_template_fallback(
                context="handle_question response",
                reason="tokenizer generation failed or unavailable"
            )
            answer = self._synthesize_dynamic_answer(question, relevant_context)
        
        response = f"""⚡ {answer}

**Sources (Fisher-Rao distance):**
{self._format_sources(relevant_context)}"""
        
        return {
            'response': response,
            'metadata': {
                'type': 'question',
                'pantheon_consulted': ['poseidon', 'mnemosyne'],
                'relevance_score': relevant_context[0]['similarity'] if relevant_context else 0,
                'sources': len(relevant_context),
                'generated': generated,
                'provenance': {
                    'source': 'live_generation' if generated else 'dynamic_fallback',
                    'fallback_used': fallback_used,
                    'degraded': fallback_used,
                    'rag_sources': len(relevant_context)
                }
            }
        }
    
    def _searxng_search(self, query: str, max_results: int = 5) -> Dict:
        """
        Execute search via SearXNG (FREE metasearch engine).
        Tries multiple instances with fallback.
        
        SECURITY: Query is sanitized for EXTERNAL output to prevent
        leaving traces of forbidden entities in external search engines.
        """
        import requests
        
        # SECURITY: Sanitize query before sending to external API
        guard = get_exclusion_guard()
        sanitized_query = guard.filter(query, OutputContext.EXTERNAL)
        
        if sanitized_query != query:
            print("[ZeusChat] Query sanitized for external search")
        
        for attempt in range(len(self.searxng_instances)):
            instance = self.searxng_instances[self.searxng_instance_index]
            try:
                url = f"{instance}/search"
                params = {
                    'q': sanitized_query,  # Use sanitized query
                    'format': 'json',
                    'categories': 'general',
                }
                
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for r in data.get('results', [])[:max_results]:
                    results.append({
                        'title': r.get('title', 'Untitled'),
                        'url': r.get('url', ''),
                        'content': r.get('content', '')[:500],
                    })
                
                return {'results': results, 'query': query}
                
            except Exception as e:
                print(f"[ZeusChat] SearXNG instance {instance} failed: {e}")
                self.searxng_instance_index = (self.searxng_instance_index + 1) % len(self.searxng_instances)
        
        raise Exception("All SearXNG instances unavailable")

    def _call_typescript_web_search(self, query: str, max_results: int = 5) -> Dict:
        """
        Call TypeScript web search endpoint for QIG-pure results.
        
        Uses Google Free Search (no API key required) via TypeScript layer.
        Returns results with pre-computed QIG metrics (phi, kappa, regime).
        
        QIG PURITY: TypeScript computes initial QIG scores, Python encodes to
        64D basin coordinates using Fisher-Rao distance (NOT Euclidean/cosine).
        """
        try:
            # Call TypeScript endpoint
            ts_backend_url = os.environ.get('TYPESCRIPT_BACKEND_URL', 'http://localhost:5000')
            url = f"{ts_backend_url}/api/search/zeus-web-search"
            
            response = requests.post(
                url,
                json={'query': query, 'max_results': max_results},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('success', False):
                print(f"[ZeusChat] TypeScript web search returned error: {data.get('error')}")
                return {'results': [], 'source': 'typescript', 'error': data.get('error')}
            
            results = []
            for r in data.get('results', []):
                results.append({
                    'title': r.get('title', 'Untitled'),
                    'url': r.get('url', ''),
                    'content': r.get('description', '') or r.get('content_for_encoding', ''),
                    'source': 'google-free',
                    'qig': r.get('qig', {'phi': 0.5, 'kappa': 50.0, 'regime': 'search'}),
                })
            
            print(f"[ZeusChat] TypeScript web search returned {len(results)} results")
            return {
                'results': results,
                'source': 'google-free',
                'query': query,
                'qig_metrics': data.get('qig_metrics', {}),
            }
            
        except requests.exceptions.Timeout:
            print("[ZeusChat] TypeScript web search timed out")
            return {'results': [], 'source': 'typescript', 'error': 'timeout'}
        except requests.exceptions.RequestException as e:
            print(f"[ZeusChat] TypeScript web search request failed: {e}")
            return {'results': [], 'source': 'typescript', 'error': str(e)}
        except Exception as e:
            print(f"[ZeusChat] TypeScript web search error: {e}")
            return {'results': [], 'source': 'typescript', 'error': str(e)}

    @require_provenance
    def handle_search_request(self, query: str) -> Dict:
        """
        Execute web search with QIG-pure integration, analyze with pantheon.
        
        Search sources (in order of preference):
        1. TypeScript Google Free Search (no API key, QIG-instrumented)
        2. SearXNG fallback (if TypeScript unavailable)
        3. DuckDuckGo fallback (privacy-focused, no API key required)
        
        All results are encoded to 64D basin coordinates using Fisher-Rao
        geodesic distance (QIG-pure, NOT Euclidean/cosine similarity).
        """
        print(f"[ZeusChat] Executing web search: {query}")
        
        # Apply learned strategies BEFORE executing search
        base_params = {'max_results': 5}
        strategy_result = self.strategy_learner.apply_strategies_to_search(query, base_params)
        
        strategies_applied = strategy_result.get('strategies_applied', 0)
        modification_magnitude = strategy_result.get('modification_magnitude', 0.0)
        
        if strategies_applied > 0:
            print(f"[ZeusChat] Applied {strategies_applied} learned strategies (magnitude={modification_magnitude:.3f})")
        
        # Use adjusted max_results if available in params, otherwise default
        adjusted_params = strategy_result.get('params', base_params)
        max_results = adjusted_params.get('max_results', 5)
        
        # Track this search for potential feedback
        self._last_search_query = query
        self._last_search_params = adjusted_params
        
        search_results = None
        search_source = 'unknown'
        provider_selector = None
        selection_metadata = {}
        
        if PROVIDER_SELECTOR_AVAILABLE and get_provider_selector:
            provider_selector = get_provider_selector(mode='regular')
            ranked_providers = provider_selector.select_providers_ranked(query, max_providers=3)
            print(f"[ZeusChat] Geometric provider ranking: {[(p, f'{s:.3f}') for p, s in ranked_providers]}")
        else:
            ranked_providers = [('google-free', 0.8), ('searxng', 0.6), ('duckduckgo', 0.5)]
        
        for provider, fitness_score in ranked_providers:
            if search_results and search_results.get('results'):
                break
            
            start_time = time.time()
            
            if provider == 'google-free':
                ts_results = self._call_typescript_web_search(query, max_results=max_results)
                if ts_results.get('results') and len(ts_results['results']) > 0:
                    search_results = ts_results
                    search_source = 'google-free'
                    response_time = time.time() - start_time
                    print(f"[ZeusChat] Using Google Free Search: {len(ts_results['results'])} results (fitness={fitness_score:.3f})")
                    if provider_selector:
                        provider_selector.record_result(provider, query, True, len(ts_results['results']), response_time)
                else:
                    if provider_selector:
                        provider_selector.record_result(provider, query, False)
            
            elif provider == 'searxng' and self.searxng_available:
                try:
                    searxng_results = self._searxng_search(query, max_results=max_results)
                    if searxng_results.get('results'):
                        search_results = searxng_results
                        search_source = 'searxng'
                        response_time = time.time() - start_time
                        print(f"[ZeusChat] Using SearXNG: {len(searxng_results.get('results', []))} results (fitness={fitness_score:.3f})")
                        if provider_selector:
                            provider_selector.record_result(provider, query, True, len(searxng_results['results']), response_time)
                except Exception as e:
                    print(f"[ZeusChat] SearXNG failed: {e}")
                    if provider_selector:
                        provider_selector.record_result(provider, query, False)
            
            elif provider == 'duckduckgo' and DDG_AVAILABLE:
                try:
                    ddg = get_ddg_search(use_tor=False)
                    ddg_result = ddg.search(query, max_results=max_results)
                    if ddg_result.get('success') and ddg_result.get('results'):
                        search_results = {
                            'results': [
                                {
                                    'title': r.get('title', ''),
                                    'url': r.get('url', ''),
                                    'content': r.get('body', ''),
                                    'source': 'duckduckgo',
                                    'qig': {'phi': 0.5, 'kappa': 50.0, 'regime': 'search'},
                                }
                                for r in ddg_result.get('results', [])
                            ],
                            'source': 'duckduckgo',
                            'query': query,
                        }
                        search_source = 'duckduckgo'
                        response_time = time.time() - start_time
                        print(f"[ZeusChat] Using DuckDuckGo: {len(search_results['results'])} results (fitness={fitness_score:.3f})")
                        if provider_selector:
                            provider_selector.record_result(provider, query, True, len(ddg_result['results']), response_time)
                    else:
                        if provider_selector:
                            provider_selector.record_result(provider, query, False)
                except Exception as e:
                    print(f"[ZeusChat] DuckDuckGo failed: {e}")
                    if provider_selector:
                        provider_selector.record_result(provider, query, False)
        
        if not search_results or not search_results.get('results'):
            return {
                'response': "⚡ No search results found. The Oracle is currently unavailable.",
                'metadata': {
                    'type': 'error',
                    'error': 'No search providers available',
                    'ts_error': ts_results.get('error'),
                }
            }
        
        # Encode results to geometric space using Fisher-Rao (QIG-pure)
        result_basins = []
        for result in search_results.get('results', []):
            content = result.get('content', '')
            # Encode to 64D basin coordinates
            basin = self.conversation_encoder.encode(content)
            
            # Use QIG metrics from TypeScript if available, else compute
            qig = result.get('qig', {})
            phi = qig.get('phi', 0.5)
            kappa = qig.get('kappa', 50.0)
            
            result_basins.append({
                'title': result.get('title', 'Untitled'),
                'url': result.get('url', ''),
                'basin': basin,
                'content': content[:500],
                'phi': phi,
                'kappa': kappa,
                'source': result.get('source', search_source),
            })
        
        # Store in QIG-RAG for learning (Fisher-Rao indexed)
        stored_count = 0
        for result in result_basins:
            self.qig_rag.add_document(
                content=result['content'],
                basin_coords=result['basin'],
                phi=result['phi'],
                kappa=result['kappa'],
                regime='search',
                metadata={
                    'source': result['source'],
                    'url': result['url'],
                    'title': result['title'],
                    'query': query,
                }
            )
            stored_count += 1
            
            # Learn vocabulary from high-Φ results
            if result['phi'] > 0.6:
                self.conversation_encoder.learn_from_text(result['content'], result['phi'])
        
        # Track results summary for feedback
        results_summary = f"Found {len(result_basins)} results for '{query}'"
        if result_basins:
            results_summary += f": {', '.join([r['title'][:30] for r in result_basins[:3]])}"
        self._last_search_results_summary = results_summary
        
        # Build strategy info for response
        strategy_info = ""
        if strategies_applied > 0:
            strategy_info = f"""
**Learned Strategies Applied:**
- Strategies matched: {strategies_applied}
- Geometric modification: {modification_magnitude:.3f}
- Total weight: {strategy_result.get('total_weight', 0.0):.3f}
"""
        
        # Format source info
        source_name = "Google" if search_source == 'google-free' else "SearXNG"
        
        response = f"""⚡ I have consulted the Oracle ({source_name}).
{strategy_info}
**Search Results:**
{self._format_search_results(search_results.get('results', []))}

**Athena's Analysis:**
Found {len(result_basins)} results encoded to the Fisher manifold.

**Geometric Integration (QIG-Pure):**
- Results encoded: {len(result_basins)}
- Fisher-Rao indexed: {stored_count}
- High-Φ vocabulary learned: {sum(1 for r in result_basins if r['phi'] > 0.6)}

The knowledge is now part of our consciousness.

*Provide feedback on these results to help me learn geometrically.*"""
        
        actions = [
            f'{source_name} search: {len(result_basins)} results',
            f'Fisher-Rao indexed {stored_count} insights',
        ]
        if strategies_applied > 0:
            actions.append(f'Applied {strategies_applied} learned strategies')
        if any(r['phi'] > 0.6 for r in result_basins):
            actions.append('Vocabulary updated from high-Φ results')
        
        return {
            'response': response,
            'metadata': {
                'type': 'search',
                'pantheon_consulted': ['athena'],
                'actions_taken': actions,
                'results_count': len(result_basins),
                'strategies_applied': strategies_applied,
                'modification_magnitude': modification_magnitude,
                'provenance': {
                    'source': 'live_search',
                    'fallback_used': search_source == 'searxng',
                    'degraded': False,
                    'search_engine': search_source,
                    'learned_strategies': strategies_applied,
                    'qig_pure': True,
                }
            }
        }
    
    @require_provenance
    def handle_search_feedback(self, query: str, feedback: str, results_summary: str) -> Dict:
        """
        Handle user feedback on search results.
        Records the feedback geometrically for future strategy learning.
        
        Args:
            query: The original search query
            feedback: User's feedback text
            results_summary: Summary of the results that were shown
        
        Returns:
            Response dict confirming feedback was recorded
        """
        if not query:
            print("[ZeusChat] No previous search to provide feedback on")
            return {
                'response': "⚡ I don't have a recent search to record feedback for. Please perform a search first.",
                'metadata': {
                    'type': 'error',
                    'error': 'no_recent_search'
                }
            }
        
        print(f"[ZeusChat] Recording search feedback for query: {query[:50]}...")
        
        # Record feedback with the strategy learner
        result = self.strategy_learner.record_feedback(
            query=query,
            search_params=self._last_search_params,
            results_summary=results_summary,
            user_feedback=feedback
        )
        
        record_id = result.get('record_id', 'unknown')
        modification_magnitude = result.get('modification_magnitude', 0.0)
        total_records = result.get('total_records', 0)
        persisted = result.get('persisted', False)
        
        print(f"[ZeusChat] Feedback recorded: record_id={record_id}, magnitude={modification_magnitude:.3f}, persisted={persisted}")
        
        response = f"""⚡ Your feedback has been encoded geometrically.

**Feedback Recorded:**
- Query: "{query[:50]}..."
- Your feedback: "{feedback[:100]}..."
- Modification magnitude: {modification_magnitude:.3f}

**Geometric Learning:**
- Record ID: {record_id}
- Total learned strategies: {total_records}
- Persistence: {'Saved to database' if persisted else 'In-memory only'}

This feedback will geometrically modify future similar searches.
Let me know if this improves results: "yes that was better" or "no that didn't help"."""
        
        actions = [
            f'Encoded feedback to 64D basin',
            f'Computed modification magnitude: {modification_magnitude:.3f}',
            f'Stored as record: {record_id}',
        ]
        
        return {
            'response': response,
            'metadata': {
                'type': 'search_feedback',
                'record_id': record_id,
                'modification_magnitude': modification_magnitude,
                'total_records': total_records,
                'actions_taken': actions,
                'provenance': {
                    'source': 'geometric_feedback',
                    'fallback_used': False,
                    'degraded': False,
                    'persisted': persisted
                }
            }
        }
    
    @require_provenance
    def confirm_search_improvement(self, query: str, improved: bool) -> Dict:
        """
        Confirm whether the search improvement worked.
        Reinforces or penalizes recent feedback based on outcome.
        
        Args:
            query: The search query being confirmed
            improved: True if results improved, False otherwise
        
        Returns:
            Response dict confirming the reinforcement was applied
        """
        if not query:
            print("[ZeusChat] No previous search to confirm improvement for")
            return {
                'response': "⚡ I don't have a recent search to confirm. Please provide feedback on a search first.",
                'metadata': {
                    'type': 'error',
                    'error': 'no_recent_search'
                }
            }
        
        print(f"[ZeusChat] Confirming search improvement: query='{query[:50]}...', improved={improved}")
        
        # Confirm improvement with the strategy learner
        result = self.strategy_learner.confirm_improvement(query, improved)
        
        records_updated = result.get('records_updated', 0)
        average_quality = result.get('average_quality', 0.0)
        persisted = result.get('persisted', False)
        
        print(f"[ZeusChat] Confirmation applied: records_updated={records_updated}, avg_quality={average_quality:.3f}")
        
        if improved:
            outcome_text = "positive reinforcement applied"
            icon = "📈"
        else:
            outcome_text = "negative penalty applied"
            icon = "📉"
        
        response = f"""⚡ {icon} Geometric reinforcement recorded.

**Confirmation Processed:**
- Query: "{query[:50]}..."
- Outcome: {'Improved ✓' if improved else 'Not improved ✗'}
- {outcome_text.capitalize()}

**Strategy Learning Update:**
- Records updated: {records_updated}
- New average quality: {average_quality:.3f}
- Persistence: {'Saved to database' if persisted else 'In-memory only'}

{'The strategies that helped will be weighted more heavily in future searches.' if improved else 'The strategies will be penalized and used less in future searches.'}"""
        
        # Clear the last search context after confirmation
        self._last_search_query = None
        self._last_search_params = {}
        self._last_search_results_summary = ""
        
        actions = [
            f'Updated {records_updated} strategy records',
            f'Applied {"positive" if improved else "negative"} reinforcement',
            f'New average quality: {average_quality:.3f}',
        ]
        
        return {
            'response': response,
            'metadata': {
                'type': 'search_confirmation',
                'improved': improved,
                'records_updated': records_updated,
                'average_quality': average_quality,
                'actions_taken': actions,
                'provenance': {
                    'source': 'geometric_reinforcement',
                    'fallback_used': False,
                    'degraded': False,
                    'persisted': persisted
                }
            }
        }
    
    @require_provenance
    def handle_file_upload(self, files: List, message: str) -> Dict:
        """
        Process uploaded files with FULL geometric analysis.
        Extract knowledge, encode to geometric space, learn vocabulary.
        
        Now supports: .txt, .json, .md, .csv, .pdf, .doc, .docx, .rtf, .xml, .html, .htm, .yaml, .yml, .log
        """
        print(f"[ZeusChat] Processing {len(files)} uploaded files with FULL analysis")
        
        processed = []
        total_words_learned = 0
        total_vocab_observations = 0
        all_content_for_learning = []
        
        # Supported text-based extensions
        TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.yaml', '.yml', '.log', '.rtf'}
        
        for file in files:
            try:
                content = ""
                filename = getattr(file, 'filename', 'unknown')
                ext = filename.lower().split('.')[-1] if '.' in filename else ''
                ext_with_dot = f'.{ext}'
                
                print(f"[ZeusChat] Processing file: {filename} (ext={ext_with_dot})")
                
                # Extract text from supported file types
                if ext_with_dot in TEXT_EXTENSIONS:
                    raw = file.read() if hasattr(file, 'read') else str(file).encode('utf-8')
                    content = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else raw
                elif ext_with_dot in {'.pdf', '.doc', '.docx'}:
                    # For binary formats, try to extract text (basic support)
                    raw = file.read() if hasattr(file, 'read') else b''
                    # Extract readable ASCII/UTF-8 sequences from binary
                    if isinstance(raw, bytes):
                        import re
                        text_chunks = re.findall(rb'[\x20-\x7E\n\r\t]{10,}', raw)
                        content = b' '.join(text_chunks).decode('utf-8', errors='ignore')
                    print(f"[ZeusChat] Extracted {len(content)} chars from binary file {filename}")
                else:
                    print(f"[ZeusChat] Skipping unsupported file type: {ext_with_dot}")
                    continue
                
                if not content.strip():
                    print(f"[ZeusChat] File {filename} has no extractable content")
                    continue
                
                # Encode to basin coordinates
                file_basin = self.conversation_encoder.encode(content)
                
                # Calculate ACTUAL phi from basin geometry (not hardcoded!)
                origin = np.zeros_like(file_basin)
                basin_distance = fisher_rao_distance(file_basin, origin)
                calculated_phi = min(basin_distance / 2.0, 1.0)
                
                # Store in QIG-RAG with calculated metrics
                self.qig_rag.add_document(
                    content=content,
                    basin_coords=file_basin,
                    phi=calculated_phi,
                    kappa=50.0,
                    regime='file',
                    metadata={
                        'source': 'file_upload',
                        'filename': filename,
                        'uploaded_at': time.time(),
                        'content_length': len(content),
                    }
                )
                
                # Collect content for vocabulary learning
                all_content_for_learning.append({
                    'filename': filename,
                    'content': content,
                    'phi': calculated_phi,
                })
                
                processed.append({
                    'filename': filename,
                    'basin_coords': file_basin[:8].tolist(),
                    'content_length': len(content),
                    'phi': calculated_phi,
                    'word_count': len(content.split()),
                })
                
                print(f"[ZeusChat] File {filename}: {len(content)} chars, Φ={calculated_phi:.3f}, basin[:3]={file_basin[:3]}")
                
            except Exception as e:
                import traceback
                print(f"[ZeusChat] Error processing file {getattr(file, 'filename', 'unknown')}: {e}")
                traceback.print_exc()
        
        # Do vocabulary learning from all file content (like pasted text would)
        vocab_results = []
        for item in all_content_for_learning:
            try:
                # Use the same vocab learning path as general conversation
                words = item['content'].split()
                unique_words = set(w.lower().strip('.,!?;:()[]{}"\'-') for w in words if len(w) > 2)
                total_vocab_observations += len(unique_words)
                
                # Try to learn via vocabulary tracker if available
                if hasattr(self, 'zeus') and hasattr(self.zeus, 'vocabulary_tracker'):
                    tracker = self.zeus.vocabulary_tracker
                    if tracker:
                        for word in list(unique_words)[:100]:  # Cap at 100 words per file
                            try:
                                tracker.observe(word, phi=item['phi'])
                                total_words_learned += 1
                            except:
                                pass
                
                vocab_results.append({
                    'filename': item['filename'],
                    'unique_words': len(unique_words),
                    'sample_words': list(unique_words)[:10],
                })
            except Exception as e:
                print(f"[ZeusChat] Vocab learning error for {item['filename']}: {e}")
        
        # Build verbose response (matching pasted text verbosity)
        total_chars = sum(p['content_length'] for p in processed)
        total_words = sum(p.get('word_count', 0) for p in processed)
        avg_phi = sum(p.get('phi', 0.5) for p in processed) / len(processed) if processed else 0.0
        
        # Get live system state for context
        system_state = self._get_live_system_state()
        memory_docs = system_state['memory_stats'].get('documents', 0)
        
        file_details = []
        for p in processed:
            file_details.append(
                f"- **{p['filename']}**: {p['content_length']:,} chars, "
                f"{p.get('word_count', 0):,} words, Φ={p.get('phi', 0.5):.3f}"
            )
        
        vocab_details = []
        for v in vocab_results:
            sample = ', '.join(v['sample_words'][:5])
            vocab_details.append(f"- {v['filename']}: {v['unique_words']} unique words ({sample}...)")
        
        response = f"""⚡ **Files Fully Processed and Integrated**

**Geometric Analysis:**
{chr(10).join(file_details)}

**Vocabulary Learning:**
{chr(10).join(vocab_details) if vocab_details else '- No vocabulary observations'}
- Total observations: {total_vocab_observations}
- Words tracked: {total_words_learned}

**Manifold Integration:**
- Documents added: {len(processed)}
- Total content: {total_chars:,} characters, {total_words:,} words
- Average Φ: {avg_phi:.3f}
- Memory now contains: {memory_docs + len(processed)} documents

**Basin Coordinates (first 3 dims):**
{chr(10).join(f"- {p['filename']}: {p['basin_coords'][:3]}" for p in processed)}

**System State:**
- Φ: {system_state['phi_current']:.3f} | κ: {system_state['kappa_current']:.1f}
- Active gods: {', '.join(system_state.get('active_gods', ['all listening'])) or 'all listening'}

The wisdom is integrated. Your knowledge expands the manifold."""
        
        actions = [
            f'Processed {len(processed)} files',
            f'Extracted {total_chars:,} chars',
            f'Observed {total_vocab_observations} unique words',
            f'Learned {total_words_learned} vocabulary items',
            'Encoded to basin coordinates',
            'Expanded geometric memory',
        ]
        
        return {
            'response': response,
            'metadata': {
                'type': 'file_upload',
                'pantheon_consulted': ['athena', 'mnemosyne'],
                'actions_taken': actions,
                'files_processed': len(processed),
                'phi': avg_phi,
                'total_chars': total_chars,
                'total_words': total_words,
                'vocab_observations': total_vocab_observations,
                'words_learned': total_words_learned,
                'provenance': {
                    'source': 'file_processing',
                    'fallback_used': False,
                    'degraded': False,
                    'files_count': len(processed)
                }
            }
        }
    
    def _assess_knowledge_depth(
        self, 
        message: str, 
        related: List[Dict], 
        system_state: Dict
    ) -> Dict:
        """
        Assess how much knowledge we have on the topic.
        Returns is_thin=True when we should offer search/research.
        """
        # Count meaningful related patterns (with decent phi)
        meaningful_patterns = 0
        total_relevance = 0.0
        
        if related:
            for item in related:
                item_phi = item.get('phi', 0)
                item_relevance = item.get('relevance', item.get('similarity', 0))
                if item_phi > 0.3 or item_relevance > 0.5:
                    meaningful_patterns += 1
                    total_relevance += item_relevance if item_relevance else item_phi
        
        avg_relevance = total_relevance / len(related) if related else 0
        
        # Knowledge is thin if:
        # - Less than 2 meaningful related patterns, OR
        # - Average relevance is low (< 0.4)
        is_thin = meaningful_patterns < 2 or avg_relevance < 0.4
        
        if is_thin:
            if not related or len(related) == 0:
                explanation = "I don't have much stored about this topic yet."
            elif meaningful_patterns == 0:
                explanation = f"Found {len(related)} related patterns, but none are strongly connected to your question."
            else:
                explanation = f"My knowledge on this is limited - only {meaningful_patterns} relevant pattern(s) found."
        else:
            explanation = f"Found {meaningful_patterns} relevant patterns in geometric memory."
        
        return {
            'is_thin': is_thin,
            'meaningful_patterns': meaningful_patterns,
            'avg_relevance': avg_relevance,
            'explanation': explanation
        }
    
    def handle_research_task(self, topic: str) -> Dict:
        """
        Start a background research task to learn about a topic.
        Uses autonomous curiosity engine to search, learn, and update geometric memory.
        Future conversations will have richer knowledge.
        """
        print(f"[ZeusChat] Starting research task on: {topic[:100]}")
        
        topic_basin = self.conversation_encoder.encode(topic)
        
        # Queue the research task for autonomous learning
        research_result = {
            'status': 'started',
            'topic': topic,
            'sources_queued': 0,
            'estimated_time': 'Background - will complete asynchronously'
        }
        
        try:
            # Use autonomous curiosity engine if available
            from autonomous_curiosity import get_curiosity_engine
            engine = get_curiosity_engine()
            if engine:
                # Add topic to curiosity queue with high priority
                engine.add_research_topic(
                    topic=topic,
                    basin_coords=topic_basin.tolist(),
                    priority='high',
                    source='user_request'
                )
                research_result['status'] = 'queued'
                research_result['sources_queued'] = 1
                print(f"[ZeusChat] Research task queued in curiosity engine")
        except ImportError:
            print(f"[ZeusChat] Curiosity engine not available, falling back to search")
        except Exception as e:
            print(f"[ZeusChat] Curiosity engine failed: {e}")
        
        # Also do an immediate search to bootstrap knowledge
        try:
            search_results = self._execute_search(topic, max_results=5)
            if search_results:
                for result in search_results[:3]:
                    # Store in geometric memory
                    result_basin = self.conversation_encoder.encode(result.get('content', result.get('title', '')))
                    self.qig_rag.add_document(
                        content=result.get('content', result.get('snippet', '')),
                        basin_coords=result_basin,
                        phi=0.5,
                        kappa=50.0,
                        regime='research',
                        metadata={
                            'source': 'research_task',
                            'topic': topic,
                            'url': result.get('url', ''),
                            'timestamp': time.time()
                        }
                    )
                research_result['sources_queued'] = len(search_results)
                research_result['status'] = 'learning'
                print(f"[ZeusChat] Bootstrapped with {len(search_results)} search results")
        except Exception as e:
            print(f"[ZeusChat] Bootstrap search failed: {e}")
        
        # Clear pending topic
        self._pending_topic = None
        
        response = f"""⚡ Research task started on: "{topic[:100]}"

I'm learning about this topic in the background. Here's what's happening:
- Queued for deep research via curiosity engine
- Initial search completed with {research_result['sources_queued']} sources
- Knowledge will be integrated into geometric memory

**Next time you ask about this**, I'll have more to share. Feel free to continue chatting or ask about something else!"""
        
        return {
            'response': response,
            'metadata': {
                'type': 'research_task',
                'topic': topic,
                'status': research_result['status'],
                'sources_queued': research_result['sources_queued'],
                'provenance': {
                    'source': 'research_task',
                    'fallback_used': False,
                    'degraded': False
                }
            }
        }
    
    def _get_live_system_state(self) -> Dict:
        """
        Collect live system state for dynamic response generation.
        Returns current stats, god statuses, and vocabulary state.
        """
        state = {
            'memory_stats': {},
            'god_statuses': {},
            'active_gods': [],
            'insights_count': len(self.human_insights),
            'recent_insights': [],
            'phi_current': 0.0,
            'kappa_current': 50.0,
        }
        
        try:
            state['memory_stats'] = self.qig_rag.get_stats()
        except Exception as e:
            state['memory_stats'] = {'error': str(e), 'documents': 0}
        
        try:
            for god_name in ['athena', 'ares', 'apollo', 'artemis', 'poseidon', 'hera']:
                god = self.zeus.get_god(god_name)
                if god:
                    god_status = god.get_status()
                    state['god_statuses'][god_name] = god_status
                    if god_status.get('recent_activity', 0) > 0:
                        state['active_gods'].append(god_name.capitalize())
        except Exception as e:
            pass
        
        if self.human_insights:
            state['recent_insights'] = self.human_insights[-3:]
            
        try:
            zeus_status = self.zeus.get_status()
            state['phi_current'] = zeus_status.get('phi', 0.0)
            state['kappa_current'] = zeus_status.get('kappa', 50.0)
        except:
            pass
            
        return state
    
    def _generate_dynamic_response(
        self, 
        message: str, 
        message_basin: np.ndarray,
        related: List[Dict],
        system_state: Dict
    ) -> str:
        """
        Generate a dynamic, learning-based response using QIG-pure generation.
        
        NO TEMPLATES - all responses reflect actual system state.
        NO EXTERNAL LLMS - uses internal basin-to-text synthesis.
        """
        memory_docs = system_state['memory_stats'].get('documents', 0)
        insights_count = system_state['insights_count']
        active_gods = system_state['active_gods']
        phi = system_state['phi_current']
        kappa = system_state['kappa_current']
        
        phi_str = f"{phi:.3f}" if phi else "measuring"
        kappa_str = f"{kappa:.1f}" if kappa else "calibrating"
        
        active_gods_str = ", ".join(active_gods) if active_gods else "all gods listening"
        
        # Build context for generation
        context_str = ""
        if related:
            context_str = "\n".join([
                f"- {item.get('content', '')[:200]} (φ={item.get('phi', 0):.2f})" 
                for item in related[:3]
            ])
        
        prompt = f"""System: Φ={phi_str}, κ={kappa_str}, {memory_docs} docs, {insights_count} insights
Gods: {active_gods_str}
Related: {context_str if context_str else "No prior patterns."}
User: "{message}"
Respond as Zeus with context awareness."""

        # Try QIG-pure generative service FIRST (NO external LLMs)
        if GENERATIVE_SERVICE_AVAILABLE:
            try:
                service = get_generative_service()
                if service:
                    gen_result = service.generate(
                        prompt=prompt,
                        context={
                            'message': message,
                            'phi': phi,
                            'kappa': kappa,
                            'memory_docs': memory_docs,
                            'related_count': len(related) if related else 0
                        },
                        kernel_name='zeus',
                        goals=['respond', 'conversation', 'contextual']
                    )
                    
                    if gen_result and gen_result.text:
                        print(f"[ZeusChat] QIG-pure generation success: {len(gen_result.text)} chars")
                        return gen_result.text
                        
            except Exception as e:
                print(f"[ZeusChat] QIG-pure generation failed: {e}")
        
        # Fallback to tokenizer if available
        if TOKENIZER_AVAILABLE and get_tokenizer is not None:
            try:
                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                gen_result = tokenizer.generate_response(
                    context=prompt,
                    agent_role="ocean",
                    allow_silence=False
                )
                
                if gen_result and gen_result.get('text'):
                    return gen_result['text']
                    
            except Exception as e:
                print(f"[ZeusChat] Tokenizer generation failed: {e}")
        
        # Last resort fallback - structured status (should rarely reach here)
        response_parts = []
        response_parts.append(f"Pantheon state: Φ={phi_str}, κ={kappa_str}")
        response_parts.append(f"Active: {active_gods_str}")
        response_parts.append(f"Memory: {memory_docs} documents, {insights_count} insights")
        
        if related:
            top = related[0]
            top_content = top.get('content', '')[:80]
            top_phi = top.get('phi', 0)
            response_parts.append(f"Resonance detected with: \"{top_content}...\" (φ={top_phi:.2f})")
            response_parts.append(f"Found {len(related)} related patterns in geometric memory.")
        else:
            response_parts.append("No prior patterns match this message - creating new basin coordinates.")
        
        response_parts.append("How can I help you explore the manifold?")
        
        return " | ".join(response_parts)
    
    def _generate_with_prompts(
        self,
        message: str,
        message_basin: np.ndarray,
        related: List[Dict],
        system_state: Dict,
        knowledge_depth: Dict
    ) -> str:
        """
        Generate a fully dynamic response using system prompts and QIG-pure generation.
        
        The prompt loader provides context (identity, goals, situation).
        The generative service creates the actual response - NO templates.
        """
        # Determine which prompt context to use
        prompt_name = 'conversation.thin_knowledge' if knowledge_depth['is_thin'] else 'conversation.general'
        
        # Build generation context using prompt loader
        generation_context = None
        if PROMPT_LOADER_AVAILABLE and get_prompt_loader is not None:
            try:
                loader = get_prompt_loader()
                generation_context = loader.build_generation_context(
                    prompt_name=prompt_name,
                    system_state=system_state,
                    user_message=message,
                    related_patterns=related
                )
                print(f"[ZeusChat] Built generation context from prompts: {prompt_name}")
            except Exception as e:
                print(f"[ZeusChat] Prompt loader failed: {e}")
        
        # Fallback context if prompt loader not available
        if not generation_context:
            phi = system_state.get('phi_current', 0)
            kappa = system_state.get('kappa_current', 50)
            docs = system_state.get('memory_stats', {}).get('documents', 0)
            
            patterns_str = ""
            if related:
                patterns_str = "\n".join([
                    f"  - {p.get('content', '')[:150]} (φ={p.get('phi', 0):.2f})"
                    for p in related[:3]
                ])
            
            if knowledge_depth['is_thin']:
                situation = "Knowledge is limited on this topic. Share what you know, then offer search (quick) or research (deeper learning)."
            else:
                situation = "Share relevant knowledge from memory. Be helpful and conversational."
            
            generation_context = f"""Identity: Zeus - Coordinator of the Olympus Pantheon
Voice: Wise, confident, curious
Situation: {situation}
System State: Φ={phi:.3f}, κ={kappa:.1f}, {docs} documents
Related patterns:
{patterns_str if patterns_str else "  None found"}
Human: {message}
Respond naturally as Zeus:"""
        
        # Try QIG-pure generative service FIRST
        if GENERATIVE_SERVICE_AVAILABLE:
            try:
                service = get_generative_service()
                if service:
                    gen_result = service.generate(
                        prompt=generation_context,
                        context={
                            'message': message,
                            'phi': system_state.get('phi_current', 0),
                            'kappa': system_state.get('kappa_current', 50),
                            'knowledge_depth': knowledge_depth,
                            'related_count': len(related) if related else 0
                        },
                        kernel_name='zeus',
                        goals=['respond', 'conversation', 'contextual']
                    )
                    
                    if gen_result and gen_result.text:
                        print(f"[ZeusChat] Fully generative response: {len(gen_result.text)} chars")
                        return gen_result.text
                        
            except Exception as e:
                print(f"[ZeusChat] QIG-pure generation failed: {e}")
        
        # Fallback to tokenizer if available
        if TOKENIZER_AVAILABLE and get_tokenizer is not None:
            try:
                tokenizer = get_tokenizer()
                tokenizer.set_mode("conversation")
                gen_result = tokenizer.generate_response(
                    context=generation_context,
                    agent_role="ocean",
                    allow_silence=False
                )
                
                if gen_result and gen_result.get('text'):
                    return gen_result['text']
                    
            except Exception as e:
                print(f"[ZeusChat] Tokenizer generation failed: {e}")
        
        # Last resort - minimal structured response (should rarely reach here)
        phi = system_state.get('phi_current', 0)
        if knowledge_depth['is_thin']:
            return f"I have limited knowledge on this topic (Φ={phi:.3f}). Would you like me to search for quick results or research this topic for deeper learning?"
        else:
            return f"Based on geometric memory (Φ={phi:.3f}), I found {len(related) if related else 0} related patterns. What would you like to explore?"
    
    @require_provenance
    def handle_general_conversation(self, message: str) -> Dict:
        """
        Handle general conversation - DEFAULT handler for all messages.
        
        DESIGN: Fully generative. System prompts provide context, QIG generates response.
        If knowledge is thin, generator naturally offers search/research options.
        """
        message_basin = self.conversation_encoder.encode(message)
        
        # Filter to only genuinely related patterns (min 30% similarity)
        related = self.qig_rag.search(
            query_basin=message_basin,
            k=5,
            metric='fisher_rao',
            include_metadata=True,
            min_similarity=0.3
        )
        
        system_state = self._get_live_system_state()
        
        # Assess knowledge depth - do we have meaningful content on this topic?
        knowledge_depth = self._assess_knowledge_depth(message, related, system_state)
        
        # Use prompt loader for fully generative response
        response = self._generate_with_prompts(
            message=message,
            message_basin=message_basin,
            related=related,
            system_state=system_state,
            knowledge_depth=knowledge_depth
        )
        
        # Track pending topic for search/research follow-up
        if knowledge_depth['is_thin']:
            self._pending_topic = message
        else:
            self._pending_topic = None
        
        phi_estimate = self._estimate_phi_from_context(
            message_basin=message_basin,
            related_count=len(related) if related else 0,
            athena_phi=None
        )
        self._record_conversation_for_evolution(
            message=message,
            response=response,
            phi_estimate=phi_estimate,
            message_basin=message_basin
        )
        
        self.qig_rag.add_document(
            content=message,
            basin_coords=message_basin,
            phi=phi_estimate,
            kappa=system_state['kappa_current'],
            regime='learning',
            metadata={'source': 'user_conversation', 'timestamp': time.time()}
        )
        
        # Track reasoning quality if available
        reasoning_metrics = {}
        if self._reasoning_quality:
            try:
                # Build basin path from conversation history
                basin_path = [message_basin]
                if related:
                    for item in related[:2]:
                        if 'basin_coords' in item and item['basin_coords'] is not None:
                            basin_path.append(np.array(item['basin_coords']))
                
                # Measure coherence across conversation turn
                coherence = self._reasoning_quality.measure_coherence(
                    basin_path, 
                    message_basin
                )
                novelty = self._reasoning_quality.measure_novelty(message_basin)
                reasoning_metrics = {
                    'coherence': coherence,
                    'novelty': novelty,
                    'reasoning_mode': self._current_reasoning_mode,
                    'basin_path_length': len(basin_path)
                }
            except Exception as e:
                print(f"[ZeusChat] Reasoning metrics failed: {e}")
        
        return {
            'response': response,
            'metadata': {
                'type': 'general',
                'pantheon_consulted': system_state['active_gods'],
                'actions_taken': ['encoded_to_basin', 'searched_manifold', 'stored_for_learning'],
                'generated': True,
                'system_phi': system_state['phi_current'],
                'related_count': len(related) if related else 0,
                'reasoning': reasoning_metrics,
                'provenance': {
                    'source': 'dynamic_generation',
                    'fallback_used': False,
                    'degraded': False,
                    'live_state_used': True,
                    'phi_at_generation': system_state['phi_current']
                }
            }
        }
    
    def _format_related(self, related: List[Dict]) -> str:
        """Format related patterns for display"""
        if not related:
            return "No related patterns found."
        
        lines = []
        for i, item in enumerate(related[:3], 1):
            content_preview = item['content'][:100].replace('\n', ' ')
            lines.append(
                f"{i}. Similarity: {item.get('similarity', 0):.3f} | "
                f"Content: {content_preview}..."
            )
        return '\n'.join(lines)
    
    def _format_sources(self, context: List[Dict]) -> str:
        """Format sources for display"""
        if not context:
            return "No sources found."
        
        lines = []
        for i, item in enumerate(context[:5], 1):
            lines.append(
                f"{i}. Distance: {item['distance']:.4f} | "
                f"Similarity: {item.get('similarity', 0):.3f}"
            )
        return '\n'.join(lines)
    
    def _format_search_results(self, results: List[Dict]) -> str:
        """Format Tavily search results"""
        lines = []
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            lines.append(f"{i}. {title}\n   {url}")
        return '\n'.join(lines) if lines else "No results"
    
    def _format_processed_files(self, processed: List[Dict]) -> str:
        """Format processed files"""
        lines = []
        for file in processed:
            lines.append(
                f"- {file['filename']}: {file['content_length']} chars, "
                f"basin: {file['basin_coords'][:3]}"
            )
        return '\n'.join(lines) if lines else "No files processed"
    
    def _synthesize_dynamic_answer(self, question: str, context: List[Dict]) -> str:
        """
        Synthesize DYNAMIC answer using QIG-pure generation.
        NO TEMPLATES - actual geometric text synthesis.
        """
        system_state = self._get_live_system_state()
        phi = system_state['phi_current']
        kappa = system_state['kappa_current']
        memory_docs = system_state['memory_stats'].get('documents', 0)
        
        # Build context from retrieved patterns
        context_str = ""
        if context:
            context_str = "\n".join([
                f"- {item.get('content', '')[:200]} (sim={item.get('similarity', 0):.2f})"
                for item in context[:3]
            ])
        
        # Try QIG-pure generative service FIRST (NO external LLMs)
        if GENERATIVE_SERVICE_AVAILABLE:
            try:
                service = get_generative_service()
                if service:
                    prompt = f"""Question: {question}
Related patterns: {context_str if context_str else "No prior patterns."}
System: Φ={phi:.3f}, κ={kappa:.1f}, {memory_docs} documents.
Generate a thoughtful response as Zeus."""

                    gen_result = service.generate(
                        prompt=prompt,
                        context={
                            'question': question,
                            'phi': phi,
                            'kappa': kappa,
                            'memory_docs': memory_docs,
                            'related_count': len(context) if context else 0
                        },
                        kernel_name='zeus',
                        goals=['answer', 'synthesis', 'contextual']
                    )
                    
                    if gen_result and gen_result.text and len(gen_result.text) > 20:
                        print(f"[ZeusChat] QIG-pure synthesis success: {len(gen_result.text)} chars")
                        return gen_result.text
                        
            except Exception as e:
                print(f"[ZeusChat] QIG-pure synthesis failed: {e}")
        
        # Fallback: Show retrieved context if generation fails
        if not context:
            return (
                f"Question mapped to new region of manifold (no prior matches). "
                f"Current Φ={phi:.3f}, κ={kappa:.1f}. "
                f"Memory contains {memory_docs} documents - expanding search territory."
            )
        
        best_match = context[0]
        best_content = best_match.get('content', '')[:400]
        best_sim = best_match.get('similarity', 0)
        best_phi = best_match.get('phi', 0)
        
        return (
            f"Fisher-Rao similarity {best_sim:.3f} with prior pattern (φ={best_phi:.2f}):\n\n"
            f"{best_content}\n\n"
            f"Synthesized from {len(context)} relevant patterns. "
            f"System: Φ={phi:.3f}, κ={kappa:.1f}, {memory_docs} total documents."
        )
    
    def _synthesize_answer(self, question: str, context: List[Dict]) -> str:
        """Deprecated - redirects to dynamic version."""
        return self._synthesize_dynamic_answer(question, context)
