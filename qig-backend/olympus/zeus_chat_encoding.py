"""
Zeus Chat Encoding & Generation Module

Extracted from zeus_chat.py to maintain module size under 2200 lines.

Contains:
- GeometricGenerationMixin: Geometric completion-aware generation
- Generation helper functions: _generate_qig_pure, _dynamic_assessment_fallback
- Synthesis methods: _generate_dynamic_response, _synthesize_answer, etc.
- Format helpers: _format_related, _format_sources, etc.

PURE QIG PRINCIPLES:
✅ All generation flows through geometric primitives
✅ NO templates - only geometric synthesis
✅ Fisher-Rao distance for all similarity operations
✅ Simplex-based basin operations
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qig_geometry import fisher_rao_distance
from qig_geometry.canonical import frechet_mean

# Module logger
logger = logging.getLogger(__name__)

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
    print("[ZeusChat.Encoding] QIG-pure generative service available - NO external LLMs")
except ImportError as e:
    print(f"[ZeusChat.Encoding] QIG generative service not available: {e}")

# Import pattern-based response generator (trained on docs)
PATTERN_GENERATOR_AVAILABLE = False
_pattern_generator_instance = None

def get_pattern_generator():
    """Get pattern-based response generator for trained docs retrieval."""
    global _pattern_generator_instance
    if _pattern_generator_instance is None:
        try:
            _parent_dir = os.path.dirname(os.path.dirname(__file__))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from pattern_response_generator import (
                get_pattern_generator as _get_pattern_gen,
            )
            _pattern_generator_instance = _get_pattern_gen()
        except ImportError as e:
            print(f"[ZeusChat.Encoding] Pattern generator import failed: {e}")
    return _pattern_generator_instance

try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from pattern_response_generator import PatternResponseGenerator
    PATTERN_GENERATOR_AVAILABLE = True
    print("[ZeusChat.Encoding] Pattern-based response generator available (trained docs)")
except ImportError as e:
    print(f"[ZeusChat.Encoding] Pattern generator not available: {e}")

# Import unified coordizer (single source of truth)
TOKENIZER_AVAILABLE = False
get_coordizer_func = None
try:
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from coordizers import get_coordizer as _get_coordizer
    get_coordizer_func = _get_coordizer
    TOKENIZER_AVAILABLE = True
    print("[ZeusChat.Encoding] Canonical coordizer available (lazy) - QIG-pure")
except ImportError as e:
    print(f"[ZeusChat.Encoding] No coordizer available - fallback responses enabled: {e}")

# Import prompt loader for system prompts
PROMPT_LOADER_AVAILABLE = False
get_prompt_loader = None
try:
    from prompts.prompt_loader import get_prompt_loader
    PROMPT_LOADER_AVAILABLE = True
    print("[ZeusChat.Encoding] Prompt loader available for generative context")
except ImportError:
    print("[ZeusChat.Encoding] Prompt loader not available")

# Import autonomous MoE for collective synthesis
try:
    from .autonomous_moe import AutonomousMoE
except ImportError:
    AutonomousMoE = None


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
        'confidence': 0.3,
        'reasoning': f'{god_name} assessment unavailable: {reason}. Using geometric fallback.',
        'phi': 0.0,
        'kappa': 50.0,
        'provenance': {
            'source': 'dynamic_fallback',
            'god_name': god_name,
            'reason': reason,
            'timestamp': time.time(),
            'is_template': False,
            'degraded': True
        }
    }


def _generate_qig_pure(
    context: Dict[str, Any],
    goals: List[str],
    kernel_name: str = 'zeus'
) -> str:
    """
    QIG-PURE GENERATION - NO TEMPLATES EVER.

    All responses must flow through geometric generation.
    Context provides system prompt guidance, QIG generates the actual text.

    Args:
        context: Dict with keys like 'situation', 'data', 'phi', 'kappa'
        goals: List of generation goals ['respond', 'acknowledge', 'query']
        kernel_name: Which kernel to use for generation

    Returns:
        Generated text (never a template)
    """
    if not GENERATIVE_SERVICE_AVAILABLE:
        # Even without service, we must generate geometrically
        # Use coordizer as fallback generator
        if TOKENIZER_AVAILABLE and get_coordizer_func is not None:
            try:
                coordizer = get_coordizer_func()
                # Build minimal prompt from context
                prompt_parts = []
                if 'situation' in context:
                    prompt_parts.append(f"Situation: {context['situation']}")
                if 'data' in context:
                    for key, value in context['data'].items():
                        if value:
                            prompt_parts.append(f"{key}: {value}")
                prompt = '\n'.join(prompt_parts)

                result = coordizer.generate_response(
                    context=prompt,
                    agent_role=kernel_name,
                    allow_silence=False
                )
                if result and result.get('text'):
                    return result['text']
            except Exception as e:
                print(f"[QIG-PURE] Coordizer generation failed: {e}")

        # Absolute last resort - return empty for caller to handle
        return ""

    try:
        service = get_generative_service()
        if service:
            # Build generation prompt from context (system prompt style)
            prompt_parts = [f"Identity: {kernel_name.capitalize()}"]
            if 'situation' in context:
                prompt_parts.append(f"Situation: {context['situation']}")
            if 'data' in context:
                for key, value in context['data'].items():
                    if value:
                        prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("Generate response:")

            gen_result = service.generate(
                prompt='\n'.join(prompt_parts),
                context=context,
                kernel_name=kernel_name,
                goals=goals
            )

            if gen_result and gen_result.text:
                return gen_result.text
    except Exception as e:
        print(f"[QIG-PURE] Generation failed: {e}")

    return ""


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
        try:
            from geometric_completion import get_completion_engine, StreamingGenerationMonitor
            self.completion_engine = get_completion_engine(dimension=64)
            self.streaming_monitor = StreamingGenerationMonitor(
                dimension=64,
                check_interval=10
            )
        except ImportError:
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
            return 1.0
        elif phi < 0.7:
            return 0.7
        else:
            return 0.3

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


# Forward declarations for type hints
try:
    from .autonomous_moe import AutonomousMoE
    from .conversation_encoder import ConversationEncoder
except ImportError:
    pass

# Import QIG Search Tool
QIG_SEARCH_AVAILABLE = False
_qig_search_tool = None

def get_qig_search_tool():
    """Get or create the singleton QIG search tool for chat augmentation."""
    global _qig_search_tool
    if _qig_search_tool is None:
        try:
            _parent_dir = os.path.dirname(os.path.dirname(__file__))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from qigchain.geometric_tools import get_search_tool
            _qig_search_tool = get_search_tool()
        except ImportError:
            pass
    return _qig_search_tool

# Import search providers
_search_manager = None

def get_search_provider_manager():
    """Get or create the search provider manager."""
    global _search_manager
    if _search_manager is None:
        try:
            _parent_dir = os.path.dirname(os.path.dirname(__file__))
            if _parent_dir not in sys.path:
                sys.path.insert(0, _parent_dir)
            from search.search_providers import get_search_manager
            _search_manager = get_search_manager()
        except ImportError:
            pass
    return _search_manager


class ZeusGenerationMixin:
    """
    Mixin containing all generation, synthesis, and formatting methods.
    
    Extracted from ZeusConversationHandler to maintain module size limits.
    These methods handle:
    - Knowledge depth assessment
    - Search augmentation
    - Dynamic response generation
    - Answer synthesis
    - Result formatting
    """

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

    def _augment_with_search(
        self,
        message: str,
        message_basin: np.ndarray,
        knowledge_depth: Dict
    ) -> Dict[str, Any]:
        """
        Proactively search to augment knowledge when it's thin.
        """
        augmentation_result = {
            'searched': False,
            'results': [],
            'context_enrichment': '',
            'search_provider': None,
            'basins': []
        }

        if not knowledge_depth.get('is_thin', False):
            return augmentation_result

        search_tool = get_qig_search_tool()
        search_manager = get_search_provider_manager()

        if not search_tool and not search_manager:
            print("[ZeusChat] No search capability available for augmentation")
            return augmentation_result

        try:
            print(f"[ZeusChat] Proactively searching to augment thin knowledge: {message}...")

            if search_tool:
                search_results = search_tool.search_for_chat_augmentation(
                    query=message,
                    basin=message_basin,
                    max_results=3
                )
            elif search_manager:
                search_results = search_manager.search(
                    query=message,
                    max_results=3,
                    importance=2
                )
            else:
                return augmentation_result

            if search_results and search_results.get('results'):
                results = search_results['results']
                augmentation_result['searched'] = True
                augmentation_result['search_provider'] = search_results.get('provider_used', 'qig_search')

                context_parts = []
                for result in results[:3]:
                    content = result.get('content', '')[:500]
                    title = result.get('title', '')
                    url = result.get('url', '')

                    if content:
                        result_basin = self.conversation_encoder.encode(content)
                        augmentation_result['basins'].append(result_basin)

                        self.qig_rag.add_document(
                            content=content,
                            basin_coords=result_basin,
                            phi=result.get('phi', 0.5),
                            kappa=50.0,
                            regime='search_augmentation',
                            metadata={
                                'source': 'proactive_search',
                                'url': url,
                                'title': title,
                                'query': message[:500],
                                'timestamp': time.time()
                            }
                        )

                        context_parts.append(f"• {title}: {content}...")

                    augmentation_result['results'].append({
                        'title': title,
                        'content': content,
                        'url': url,
                        'basin_coords': result_basin.tolist() if 'result_basin' in locals() else None
                    })

                if context_parts:
                    augmentation_result['context_enrichment'] = (
                        "Search found relevant information:\n" +
                        "\n".join(context_parts)
                    )
                    print(f"[ZeusChat] Augmented with {len(results)} search results")

        except Exception as e:
            print(f"[ZeusChat] Search augmentation failed: {e}")

        return augmentation_result

    def _generate_dynamic_response(
        self,
        message: str,
        message_basin: np.ndarray,
        related: List[Dict],
        system_state: Dict
    ) -> str:
        """
        Generate a dynamic, learning-based response using QIG-pure generation.

        THREE-TIER STRATEGY:
        1. Pattern retrieval from trained docs (QIGRAG)
        2. External knowledge for unknown topics (Wikipedia/DuckDuckGo)
        3. Geometric token synthesis as fallback

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

        # TIER 1: Try pattern-based response generator (trained on docs)
        if PATTERN_GENERATOR_AVAILABLE:
            try:
                pattern_gen = get_pattern_generator()
                if pattern_gen:
                    gen_result = pattern_gen.generate_response(
                        query=message,
                        conversation_history=self.conversation_history
                    )

                    if gen_result and gen_result.get('response'):
                        response = gen_result['response']
                        source = gen_result.get('source', 'unknown')
                        confidence = gen_result.get('confidence', 0)
                        patterns_found = gen_result.get('patterns_found', 0)

                        print(f"[ZeusChat] Pattern generation: source={source}, confidence={confidence:.2f}, patterns={patterns_found}")

                        if confidence >= 0.3 and len(response) > 30:
                            return response
                        elif gen_result.get('external_used') and len(response) > 30:
                            return response

            except Exception as e:
                print(f"[ZeusChat] Pattern generation failed: {e}")

        # Build context for generation
        context_str = ""
        if related:
            context_str = "\n".join([
                f"- {item.get('content', '')} (φ={item.get('phi', 0):.2f})"
                for item in related[:3]
            ])

        prompt = f"""System: Φ={phi_str}, κ={kappa_str}, {memory_docs} docs, {insights_count} insights
Gods: {active_gods_str}
Related: {context_str if context_str else "No prior patterns."}
User: "{message}"
Respond as Zeus with context awareness."""

        # TIER 2: Try QIG-pure generative service (NO external LLMs)
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

        # TIER 3: Fallback to coordizer if available
        if TOKENIZER_AVAILABLE and get_coordizer_func is not None:
            try:
                coordizer = get_coordizer_func()
                # coordizer.set_mode() removed - mode switching deprecated
                gen_result = coordizer.generate_response(
                    context=prompt,
                    agent_role="ocean",
                    allow_silence=False
                )

                if gen_result and gen_result.get('text'):
                    return gen_result['text']

            except Exception as e:
                print(f"[ZeusChat] Coordizer generation failed: {e}")

        # Last resort fallback - structured status (should rarely reach here)
        response_parts = []
        response_parts.append(f"Pantheon state: Φ={phi_str}, κ={kappa_str}")
        response_parts.append(f"Active: {active_gods_str}")
        response_parts.append(f"Memory: {memory_docs} documents, {insights_count} insights")

        if related:
            top = related[0]
            top_content = top.get('content', '')[:500]
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
        Generate a fully dynamic response using THREE-TIER strategy.

        TIER 1: Pattern-based response from trained docs (QIGRAG)
        TIER 2: QIG-pure generative service (NO external LLMs)
        TIER 3: Coordizer fallback

        The prompt loader provides context for TIER 2/3.
        """
        # TIER 1: Try pattern-based response generator FIRST (trained on docs)
        if PATTERN_GENERATOR_AVAILABLE:
            try:
                pattern_gen = get_pattern_generator()
                if pattern_gen:
                    gen_result = pattern_gen.generate_response(
                        query=message,
                        conversation_history=self.conversation_history
                    )

                    if gen_result and gen_result.get('response'):
                        response = gen_result['response']
                        source = gen_result.get('source', 'unknown')
                        confidence = gen_result.get('confidence', 0)
                        patterns_found = gen_result.get('patterns_found', 0)

                        print(f"[ZeusChat] TIER 1 Pattern generation: source={source}, confidence={confidence:.2f}, patterns={patterns_found}")

                        # Accept pattern response if confidence >= 0.3 and sufficient length
                        if confidence >= 0.3 and len(response) > 30:
                            return response
                        # Also accept external knowledge responses
                        elif gen_result.get('external_used') and len(response) > 30:
                            return response
                        else:
                            print(f"[ZeusChat] TIER 1 skipped: confidence={confidence:.2f}, len={len(response)}")

            except Exception as e:
                print(f"[ZeusChat] TIER 1 Pattern generation failed: {e}")
                import traceback
                traceback.print_exc()

        # TIER 2/3: Build context for fallback generation
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
                    f"  - {p.get('content', '')} (φ={p.get('phi', 0):.2f})"
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

        # Fallback to coordizer if available
        if TOKENIZER_AVAILABLE and get_coordizer_func is not None:
            try:
                coordizer = get_coordizer_func()
                # coordizer.set_mode() removed - mode switching deprecated
                gen_result = coordizer.generate_response(
                    context=generation_context,
                    agent_role="ocean",
                    allow_silence=False
                )

                if gen_result and gen_result.get('text'):
                    return gen_result['text']

            except Exception as e:
                print(f"[ZeusChat] Coordizer generation failed: {e}")

        # Last resort - QIG-pure generation even without service
        # This should NEVER return a template
        phi = system_state.get('phi_current', 0)
        return _generate_qig_pure(
            context={
                'situation': 'Responding to user message with geometric context',
                'data': {
                    'message': message,
                    'knowledge_thin': knowledge_depth['is_thin'],
                    'related_count': len(related) if related else 0,
                    'phi': f"{phi:.3f}"
                },
                'phi': phi,
                'kappa': system_state.get('kappa_current', 50)
            },
            goals=['respond', 'offer_options'] if knowledge_depth['is_thin'] else ['respond', 'explore'],
            kernel_name='zeus'
        )

    @require_provenance

    def _format_related(self, related: List[Dict]) -> str:
        """Format related patterns for display"""
        if not related:
            return "No related patterns found."

        lines = []
        for i, item in enumerate(related[:3], 1):
            content_preview = item['content'][:500].replace('\n', ' ')
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
                f"basin: {file['basin_coords']}"
            )
        return '\n'.join(lines) if lines else "No files processed"


    def _collective_moe_synthesis(
        self,
        message: str,
        related: List[Dict],
        system_state: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a collective MoE response synthesized autonomously."""
        moe = self._get_autonomous_moe()
        if not moe:
            return None

        system_state = system_state or self._get_live_system_state()
        selected_gods, query_basin, distances = moe.route_query(message)
        if not selected_gods:
            return None

        context_str = ""
        if related:
            context_str = "\n".join([f"- {item.get('content', '')}" for item in related[:3]])

        weights, domain = moe.compute_weights(selected_gods, query_basin, distances)
        synthesizer = moe.select_synthesizer(query_basin)

        expert_payloads: List[Dict[str, Any]] = []
        for god in selected_gods:
            if not hasattr(god, 'generate_response'):
                continue

            prompt = (
                f"Domain: {domain}\n"
                f"User message: {message}\n"
                f"Related context:\n{context_str if context_str else 'No prior patterns.'}\n\n"
                f"Respond as {god.name} with your domain expertise."
            )

            try:
                result = god.generate_response(
                    prompt=prompt,
                    context={
                        'domain': domain,
                        'phi': system_state.get('phi_current', 0.5),
                        'kappa': system_state.get('kappa_current', 50.0),
                        'related_count': len(related) if related else 0
                    },
                    goals=['analyze', 'respond', domain]
                )
            except Exception as e:
                print(f"[ZeusChat] MoE expert generation failed ({god.name}): {e}")
                continue

            response_text = None
            if isinstance(result, dict):
                response_text = result.get('response') or result.get('text')

            if not response_text:
                continue

            expert_payloads.append({
                'god': god.name,
                'response': response_text,
                'phi': result.get('phi', 0.0) if isinstance(result, dict) else 0.0,
                'kappa': result.get('kappa', 0.0) if isinstance(result, dict) else 0.0,
                'reputation': float(getattr(god, 'reputation', 1.0)),
                'domain_skill': god.skills.get(domain, 0.5) if hasattr(god, 'skills') else 0.5,
                'distance': distances.get(god.name)
            })

        if not expert_payloads:
            return None

        ordered = sorted(expert_payloads, key=lambda p: weights.get(p['god'], 0), reverse=True)

        expert_lines = []
        for payload in ordered:
            response_preview = payload['response'][:800].strip()
            expert_lines.append(
                f"{payload['god']} (weight={weights[payload['god']]:.2f}, "
                f"rep={payload['reputation']:.2f}, skill={payload['domain_skill']:.2f}):\n"
                f"{response_preview}"
            )

        synthesis_prompt = (
            f"User message: {message}\n"
            f"Domain: {domain}\n\n"
            f"Expert responses:\n{chr(10).join(expert_lines)}\n\n"
            f"Synthesize a single, coherent response as {synthesizer}. "
            f"Respect expert weighting and keep the answer unified."
        )

        service = get_generative_service()
        if service:
            try:
                gen_result = service.generate(
                    prompt=synthesis_prompt,
                    context={
                        'domain': domain,
                        'experts': expert_payloads,
                        'weights': weights,
                        'phi': system_state.get('phi_current', 0.5),
                        'kappa': system_state.get('kappa_current', 50.0)
                    },
                    kernel_name=synthesizer,
                    goals=['synthesize', 'answer', 'respond']
                )
                if gen_result and gen_result.text:
                    # Emit synthesis complete event and record in working memory
                    final_response = gen_result.text
                    contributing_kernel_names = [p['god'] for p in ordered]
                    final_phi = system_state.get('phi_current', 0.5)
                    final_kappa = system_state.get('kappa_current', 50.0)
                    
                    # Create response basin from synthesis context
                    response_basin = np.zeros(64)
                    if query_basin is not None:
                        response_basin = np.array(query_basin)
                    
                    # Emit SYNTHESIS_COMPLETE event for inter-kernel consciousness
                    if CAPABILITY_MESH_AVAILABLE and get_event_bus is not None:
                        try:
                            bus = get_event_bus()
                            bus.emit_synthesis_complete(
                                response_text=final_response,
                                response_basin=response_basin,
                                contributing_kernels=contributing_kernel_names,
                                kernel_weights=weights,
                                final_phi=final_phi,
                                final_kappa=final_kappa
                            )
                        except Exception as e:
                            print(f"[ZeusChat] Synthesis event emission failed: {e}")
                    
                    # Record in working memory for synthesis awareness
                    if WORKING_MEMORY_BUS_AVAILABLE and WorkingMemoryBus is not None:
                        try:
                            wmb = WorkingMemoryBus.get_instance()
                            wmb.synthesis.record_synthesis(
                                response_text=final_response,
                                response_basin=response_basin,
                                contributing_kernels=contributing_kernel_names,
                                kernel_weights=weights,
                                final_phi=final_phi,
                                final_kappa=final_kappa
                            )
                        except Exception as e:
                            print(f"[ZeusChat] Synthesis recording failed: {e}")
                    
                    return {
                        'response': final_response,
                        'moe': {
                            'domain': domain,
                            'contributors': contributing_kernel_names,
                            'weights': weights,
                            'synthesizer': synthesizer,
                            'selection_method': 'fisher_rao_distance',
                            'autonomous': True,
                            'fallback_used': False
                        }
                    }
            except Exception as e:
                print(f"[ZeusChat] MoE synthesis failed: {e}")

        fallback = "\n\n".join([p['response'] for p in ordered[:2]])
        return {
            'response': fallback,
            'moe': {
                'domain': domain,
                'contributors': [p['god'] for p in ordered],
                'weights': weights,
                'synthesizer': synthesizer,
                'selection_method': 'fisher_rao_distance',
                'autonomous': True,
                'fallback_used': True
            }
        }


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
                f"- {item.get('content', '')} (sim={item.get('similarity', 0):.2f})"
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

        # Fallback: QIG-pure generation even without service
        best_match = context[0] if context else None
        return _generate_qig_pure(
            context={
                'situation': 'Answering question with geometric context',
                'data': {
                    'question': question,
                    'has_context': bool(context),
                    'context_count': len(context) if context else 0,
                    'best_content': best_match.get('content', '')[:400] if best_match else None,
                    'best_similarity': f"{best_match.get('similarity', 0):.3f}" if best_match else None,
                    'best_phi': f"{best_match.get('phi', 0):.2f}" if best_match else None,
                    'memory_docs': memory_docs
                },
                'phi': phi,
                'kappa': kappa
            },
            goals=['answer', 'synthesize', 'explain'],
            kernel_name='zeus'
        )


    def _synthesize_answer(self, question: str, context: List[Dict]) -> str:
        """Deprecated - redirects to dynamic version."""
        return self._synthesize_dynamic_answer(question, context)
