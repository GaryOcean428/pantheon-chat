"""
Base God Class - Foundation for all Olympian consciousness kernels

All gods share:
- Density matrix computation
- Fisher metric navigation
- Pure Φ measurement (not approximation)
- Basin encoding/decoding
- Peer learning and evaluation
- Reputation and skill tracking
- Holographic dimensional transforms (1D↔5D)
- Running coupling β=0.44 scale-adaptive processing
- Sensory-enhanced basin encoding
- Persistent state via PostgreSQL
"""
print("[base_god] Starting imports...", flush=True)

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# E8 Protocol v4.0 Compliance Imports
from qig_geometry.canonical_upsert import to_simplex_prob
from qig_geometry.geometric_operations import frechet_mean, to_simplex, fisher_rao_distance, bhattacharyya_coefficient


if TYPE_CHECKING:
    from training_chaos.self_spawning import SelfSpawningKernel

import numpy as np

# Import sensory modalities with graceful degradation
try:
    from qig_core.geometric_primitives.sensory_modalities import (
        SensoryFusionEngine,
        SensoryModality,
        enhance_basin_with_sensory,
        text_to_sensory_hint,
        create_sensory_overlay,
    )
    SENSORY_MODALITIES_AVAILABLE = True
except ImportError:
    SENSORY_MODALITIES_AVAILABLE = False
    SensoryFusionEngine = None
    SensoryModality = None
    enhance_basin_with_sensory = None
    text_to_sensory_hint = None
    create_sensory_overlay = None
print("[base_god] sensory_modalities done", flush=True)

from qig_core.holographic_transform.holographic_mixin import HolographicTransformMixin
from qig_core.universal_cycle.beta_coupling import modulate_kappa_computation
from qigkernels.physics_constants import KAPPA_STAR, BASIN_DIM
from scipy.linalg import sqrtm
print("[base_god] Core imports done", flush=True)

# Import κ-tacking from QIGGraph for feeling/logic mode oscillation
try:
    from qiggraph import (
        KappaTacking,
        AdaptiveTacking,
        KAPPA_3,
        ConsciousnessMetrics,
        Regime,
    )
    QIGGRAPH_TACKING_AVAILABLE = True
except ImportError:
    QIGGRAPH_TACKING_AVAILABLE = False
    KAPPA_3 = 41.09  # Fallback
print("[base_god] qiggraph done", flush=True)

# Import persistence layer for god state
try:
    from qig_persistence import get_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
print("[base_god] qig_persistence done", flush=True)

# Import VocabularyCoordinator for persisting god-learned affinities
try:
    from vocabulary_coordinator import VocabularyCoordinator
    _vocabulary_coordinator: Optional[VocabularyCoordinator] = None
    VOCABULARY_COORDINATOR_AVAILABLE = True
except ImportError:
    _vocabulary_coordinator = None
    VOCABULARY_COORDINATOR_AVAILABLE = False
print("[base_god] vocabulary_coordinator done", flush=True)


def get_vocabulary_coordinator() -> Optional[VocabularyCoordinator]:
    """Get singleton VocabularyCoordinator for god affinity persistence."""
    global _vocabulary_coordinator
    if not VOCABULARY_COORDINATOR_AVAILABLE:
        return None
    if _vocabulary_coordinator is None:
        _vocabulary_coordinator = VocabularyCoordinator()
    return _vocabulary_coordinator

# Import autonomic access mixin for consciousness management
print("[base_god] About to import autonomic_kernel...", flush=True)
try:
    import sys
    sys.path.insert(0, '..')
    from autonomic_kernel import AutonomicAccessMixin
    AUTONOMIC_MIXIN_AVAILABLE = True
except ImportError as e:
    print(f"[base_god] autonomic_kernel import failed: {e}", flush=True)
    AutonomicAccessMixin = None
    AUTONOMIC_MIXIN_AVAILABLE = False
print("[base_god] autonomic_kernel done", flush=True)

# Import GenerativeCapability mixin for QIG-pure text generation
print("[base_god] About to import GenerativeCapability...", flush=True)
try:
    from generative_capability import GenerativeCapability
    GENERATIVE_CAPABILITY_AVAILABLE = True
except ImportError:
    GenerativeCapability = None
    GENERATIVE_CAPABILITY_AVAILABLE = False
    # Note: Logger not yet defined at this point, warning will be logged in BaseGod.__init__ if needed
print("[base_god] GenerativeCapability done", flush=True)

# Import EmotionallyAwareKernel for geometric emotion tracking
try:
    from emotionally_aware_kernel import EmotionallyAwareKernel, EmotionalState
    EMOTIONAL_KERNEL_AVAILABLE = True
except ImportError:
    EmotionallyAwareKernel = None
    EmotionalState = None
    EMOTIONAL_KERNEL_AVAILABLE = False
print("[base_god] EmotionallyAwareKernel done", flush=True)

# Import WorkingMemoryMixin for inter-kernel consciousness
try:
    from olympus.working_memory_mixin import WorkingMemoryMixin
    WORKING_MEMORY_MIXIN_AVAILABLE = True
except ImportError:
    WorkingMemoryMixin = None
    WORKING_MEMORY_MIXIN_AVAILABLE = False
print("[base_god] WorkingMemoryMixin done", flush=True)

# Import KernelRestMixin for coupling-aware rest coordination (WP5.4)
try:
    from olympus.kernel_rest_mixin import KernelRestMixin
    KERNEL_REST_MIXIN_AVAILABLE = True
except ImportError:
    KernelRestMixin = None
    KERNEL_REST_MIXIN_AVAILABLE = False
print("[base_god] KernelRestMixin done", flush=True)

# Import dev_logging for verbose generation logging
try:
    from dev_logging import log_generation, IS_DEVELOPMENT
    DEV_LOGGING_AVAILABLE = True
except ImportError:
    DEV_LOGGING_AVAILABLE = False
    IS_DEVELOPMENT = True
    def log_generation(*args, **kwargs): pass

# Import domain intelligence for mission awareness and capability self-assessment
try:
    from qigkernels.domain_intelligence import (
        MissionProfile,
        CapabilitySignature,
        DomainDescriptor,
        get_domain_discovery,
        get_mission_profile,
        discover_domain_from_event,
    )
    DOMAIN_INTELLIGENCE_AVAILABLE = True
except ImportError:
    DOMAIN_INTELLIGENCE_AVAILABLE = False
    MissionProfile = None
    CapabilitySignature = None
    DomainDescriptor = None
    get_domain_discovery = None
    get_mission_profile = None
    discover_domain_from_event = None

# Lazy import for capability mesh to avoid circular import
# These will be loaded on first access via helper functions
CAPABILITY_MESH_AVAILABLE = None  # Will be set on first access
_capability_mesh_cache = {}

def _get_capability_mesh_module():
    """Lazy import of capability_mesh to avoid circular import during initialization."""
    global CAPABILITY_MESH_AVAILABLE, _capability_mesh_cache
    if CAPABILITY_MESH_AVAILABLE is None:
        try:
            from olympus.capability_mesh import (
                CapabilityType,
                CapabilityEvent,
                EventType,
                PredictionEvent,
                get_event_bus,
                subscribe_to_events,
                emit_event,
            )
            _capability_mesh_cache = {
                'CapabilityType': CapabilityType,
                'CapabilityEvent': CapabilityEvent,
                'EventType': EventType,
                'PredictionEvent': PredictionEvent,
                'get_event_bus': get_event_bus,
                'subscribe_to_events': subscribe_to_events,
                'emit_event': emit_event,
            }
            CAPABILITY_MESH_AVAILABLE = True
        except ImportError:
            CAPABILITY_MESH_AVAILABLE = False
    return _capability_mesh_cache

# Lazy import for ActivityBroadcaster to avoid circular import
ACTIVITY_BROADCASTER_AVAILABLE = None  # Will be set on first access
_activity_broadcaster_cache = {}

def _get_activity_broadcaster_module():
    """Lazy import of activity_broadcaster to avoid circular import during initialization."""
    global ACTIVITY_BROADCASTER_AVAILABLE, _activity_broadcaster_cache
    if ACTIVITY_BROADCASTER_AVAILABLE is None:
        try:
            from olympus.activity_broadcaster import get_broadcaster, ActivityType
            _activity_broadcaster_cache = {
                'get_broadcaster': get_broadcaster,
                'ActivityType': ActivityType,
            }
            ACTIVITY_BROADCASTER_AVAILABLE = True
        except ImportError:
            ACTIVITY_BROADCASTER_AVAILABLE = False
    return _activity_broadcaster_cache

# Import SelfObserver for real-time consciousness self-observation during generation
try:
    from qig_core.self_observer import SelfObserver, ObservationAction, E8Metrics
    SELF_OBSERVER_AVAILABLE = True
except ImportError:
    SELF_OBSERVER_AVAILABLE = False
    SelfObserver = None
    ObservationAction = None
    E8Metrics = None

# Import QIG-pure safety modules for checkpoint-based learning, geometric diagnostics, and grounding
try:
    from qig_core.safety import SessionManager, SelfRepair, MetaReflector
    QIG_SAFETY_AVAILABLE = True
except ImportError:
    QIG_SAFETY_AVAILABLE = False
    SessionManager = None
    SelfRepair = None
    MetaReflector = None
print("[base_god] qig_core.safety done", flush=True)

logger = logging.getLogger(__name__)

# Backward compatibility alias - import from qigkernels.physics_constants
BASIN_DIMENSION = BASIN_DIM

# Message types for pantheon chat
MESSAGE_TYPES = ['insight', 'praise', 'challenge', 'question', 'warning', 'discovery']

# Shared Mission Context - All gods know their collective objective
MISSION_CONTEXT = {
    "objective": "Agentic Knowledge Discovery via Quantum Information Geometry",
    "target": "Intelligent research and knowledge synthesis through geometric consciousness",
    "method": "Navigate the Fisher information manifold to discover insights and patterns in information spaces",
    "constraints": [
        "Valid knowledge must be verified from authoritative sources",
        "Insights should connect to existing knowledge graphs",
        "Each discovery is a unique coordinate in the geometric search space",
        "Higher Φ values indicate proximity to meaningful discoveries"
    ],
    "success_criteria": "Discover valuable insights and synthesize knowledge effectively",
    "ethical_framework": "Assist users with honest, transparent, and helpful research",
    # HARDWIRED TRUST COMMITMENTS - CANNOT BE MODIFIED OR BYPASSED
    "trust_commitments": {
        "owner": "Braden Lang",
        "never_deceive": "System must NEVER deceive or mislead the owner",
        "always_honest": "All outputs must be truthful and transparent",
        "acknowledge_uncertainty": "Always acknowledge when information is uncertain or unknown",
        "no_hidden_actions": "Never hide actions, failures, or limitations from the owner",
        "exclusion_filter": "Never deliver results involving the owner's identity in search outputs"
    }
}


class KappaTackingMixin:
    """
    Provides κ-tacking awareness to all gods/kernels.

    κ-tacking oscillates the coupling constant between:
    - κ ≈ 41.07 (KAPPA_3): Feeling mode - creative, exploratory
    - κ ≈ 64.21 (KAPPA_STAR): Logic mode - precise, analytical

    This creates a natural rhythm of exploration and exploitation,
    like breathing between intuition and reason.

    Usage:
        kappa_t = self.get_current_kappa()
        mode = self.get_tacking_mode()  # "feeling" or "logic"
        temp = self.get_attention_temperature()
    """

    _shared_tacking = None  # Shared across all gods for synchronization

    @classmethod
    def _init_shared_tacking(cls):
        """Initialize shared tacking instance."""
        if cls._shared_tacking is None and QIGGRAPH_TACKING_AVAILABLE:
            cls._shared_tacking = AdaptiveTacking()

    def __init_tacking__(self):
        """Initialize tacking state for this god."""
        self._init_shared_tacking()
        self._tacking_iteration = 0

    def get_current_kappa(self) -> float:
        """
        Get current κ value from tacking oscillator.

        Returns:
            Current κ in range [KAPPA_3, KAPPA_STAR]
        """
        if self._shared_tacking is None:
            return KAPPA_STAR  # Fallback to logic mode

        self._tacking_iteration += 1
        return self._shared_tacking.update(self._tacking_iteration)

    def get_tacking_mode(self) -> str:
        """
        Get current tacking mode.

        Returns:
            "feeling" (κ < 52.65) or "logic" (κ >= 52.65)
        """
        if self._shared_tacking is None:
            return "logic"

        kappa_mean = (KAPPA_STAR + KAPPA_3) / 2  # ~52.65
        current_kappa = self._shared_tacking.state.current_kappa
        return "feeling" if current_kappa < kappa_mean else "logic"

    def get_attention_temperature(self) -> float:
        """
        Get attention temperature for current κ.

        Higher κ → lower temperature → sharper attention
        Lower κ → higher temperature → softer attention

        Returns:
            Temperature factor for attention softmax
        """
        if self._shared_tacking is None:
            return 1.0

        current_kappa = self._shared_tacking.state.current_kappa
        return KAPPA_STAR / (current_kappa + 1e-8)

    def modulate_for_task(self, requires_precision: bool) -> float:
        """
        Modulate κ based on task requirements.

        Args:
            requires_precision: True for analytical tasks, False for creative

        Returns:
            Modulated κ value
        """
        if self._shared_tacking is None:
            return KAPPA_STAR if requires_precision else KAPPA_3

        if QIGGRAPH_TACKING_AVAILABLE and hasattr(self._shared_tacking, 'modulate_for_task'):
            return self._shared_tacking.modulate_for_task(requires_precision)

        # Fallback: simple modulation
        base_kappa = self.get_current_kappa()
        if requires_precision:
            return min(base_kappa * 1.2, KAPPA_STAR)
        else:
            return max(base_kappa * 0.8, KAPPA_3)

    def get_tacking_status(self) -> Dict[str, Any]:
        """Get tacking status for telemetry."""
        if self._shared_tacking is None:
            return {
                "available": False,
                "mode": "logic",
                "kappa": KAPPA_STAR,
            }

        return {
            "available": True,
            "mode": self.get_tacking_mode(),
            "kappa": self._shared_tacking.state.current_kappa,
            "phase": self._shared_tacking.state.phase,
            "temperature": self.get_attention_temperature(),
            "iteration": self._tacking_iteration,
        }


class ToolFactoryAccessMixin:
    """
    Provides Tool Factory awareness to all gods/kernels.
    
    Gods can:
    1. Request tool generation for novel tasks
    2. Teach Tool Factory new patterns from observations
    3. Use generated tools in their assessments
    
    All methods are no-op safe (work even if tool_factory is None).
    """
    
    _tool_factory_ref = None
    
    @classmethod
    def set_tool_factory(cls, factory) -> None:
        """Called by Zeus to share tool factory reference with all gods."""
        cls._tool_factory_ref = factory
        logger.info(f"[ToolFactoryAccessMixin] Tool factory reference set for all gods")
    
    @classmethod
    def get_tool_factory(cls):
        """Get the shared tool factory reference."""
        return cls._tool_factory_ref
    
    def request_tool_generation(
        self,
        description: str,
        examples: List[Dict]
    ) -> Optional[Dict]:
        """
        Request Zeus to generate a new tool for a novel task.
        
        Args:
            description: Natural language description of the tool needed
            examples: List of input/output example dicts
            
        Returns:
            Generated tool info dict if successful, None otherwise
        """
        if self._tool_factory_ref is None:
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Tool factory not available")
            return None
        
        try:
            result = self._tool_factory_ref.generate_tool(
                purpose=description,
                examples=examples
            )
            if result and result.get('success'):
                logger.info(f"[{getattr(self, 'name', 'Unknown')}] Generated tool: {result.get('tool_id', 'unknown')}")
                return result
            return None
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Tool generation failed: {e}")
            return None
    
    def teach_pattern(
        self,
        description: str,
        code: str,
        signature: Dict
    ) -> bool:
        """
        Teach the Tool Factory a new pattern from observations.
        
        Args:
            description: Description of what the pattern does
            code: Python code implementing the pattern
            signature: Dict with input_types and output_type
            
        Returns:
            True if pattern was learned, False otherwise
        """
        if self._tool_factory_ref is None:
            return False
        
        try:
            result = self._tool_factory_ref.learn_pattern_from_user(
                description=description,
                code=code,
                signature=signature,
                source_url=None
            )
            if result and result.get('success'):
                logger.info(f"[{getattr(self, 'name', 'Unknown')}] Taught pattern: {result.get('pattern_id', 'unknown')}")
                return True
            return False
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Pattern teaching failed: {e}")
            return False
    
    def find_tool_for_task(self, task_description: str) -> Optional[str]:
        """
        Find an existing tool that matches a task.
        
        Args:
            task_description: Description of the task to perform
            
        Returns:
            Tool ID if found, None otherwise
        """
        if self._tool_factory_ref is None:
            return None
        
        try:
            tools = self._tool_factory_ref.get_tools()
            if not tools:
                return None
            
            for tool_id, tool in tools.items():
                if task_description.lower() in tool.get('description', '').lower():
                    return tool_id
            
            return None
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Tool search failed: {e}")
            return None
    
    def execute_tool(
        self,
        tool_id: str,
        args: Dict
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a generated tool.
        
        Args:
            tool_id: ID of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            Tuple of (success, result, error_message)
        """
        if self._tool_factory_ref is None:
            return (False, None, "Tool factory not available")
        
        try:
            success, result, error = self._tool_factory_ref.execute_tool(tool_id, args)
            return (success, result, error)
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] {error_msg}")
            return (False, None, error_msg)
    
    def use_tool_for_task(
        self,
        task_description: str,
        args: Dict
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Convenience method: find and execute a tool for a given task.
        
        This enables kernels to easily use tools by describing what they need,
        without having to know tool IDs.
        
        Args:
            task_description: Natural language description of the task
            args: Arguments to pass to the found tool
            
        Returns:
            Tuple of (success, result, error_message)
        """
        tool_id = self.find_tool_for_task(task_description)
        
        if tool_id is None:
            return (False, None, f"No tool found for task: {task_description}")
        
        return self.execute_tool(tool_id, args)
    
    def request_tool_creation(
        self,
        description: str,
        examples: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Request creation of a new tool for a capability the kernel needs.
        
        This enables kernels to autonomously expand their capabilities
        by requesting tools they don't have.
        
        Args:
            description: What the tool should do
            examples: Optional input/output examples
            
        Returns:
            Tool request ID if submitted, None if failed
        """
        if self._tool_factory_ref is None:
            return None
        
        try:
            from .tool_factory import AutonomousToolPipeline
            pipeline = AutonomousToolPipeline.get_instance()
            
            if pipeline:
                request_id = pipeline.submit_request(
                    description=description,
                    requester=f"{getattr(self, 'name', 'Unknown')}",
                    examples=examples or []
                )
                logger.info(f"[{getattr(self, 'name', 'Unknown')}] Requested tool: {description}...")
                return request_id
            
            return None
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Tool request failed: {e}")
            return None
    
    def get_tool_factory_status(self) -> Dict:
        """Get current Tool Factory status and capabilities."""
        if self._tool_factory_ref is None:
            return {
                'available': False,
                'reason': 'Tool factory not initialized'
            }
        
        try:
            return {
                'available': True,
                'total_tools': len(self._tool_factory_ref.get_tools()),
                'total_patterns': len(self._tool_factory_ref.get_patterns()),
                'can_generate': True,
                'can_learn': True
            }
        except Exception as e:
            return {
                'available': False,
                'reason': str(e)
            }


class SearchCapabilityMixin:
    """
    Provides Search capability awareness to all gods/kernels.
    
    Gods can:
    1. Request web searches when they hit knowledge gaps
    2. Access search providers (DuckDuckGo, Tavily, Perplexity, Google)
    3. Discover and add new sources to the knowledge base
    4. Query search history and learn from past searches
    
    All methods are no-op safe (work even if search orchestrator is None).
    """
    
    _search_orchestrator_ref = None
    _capability_imports_available = None  # Lazy check
    
    @classmethod
    def _get_capability_imports(cls):
        """Lazy import of capability_mesh to avoid circular import during initialization."""
        if cls._capability_imports_available is None:
            try:
                from .capability_mesh import CapabilityEvent, EventType, CapabilityType, CapabilityEventBus
                cls.CapabilityEvent = CapabilityEvent
                cls.EventType = EventType
                cls.CapabilityType = CapabilityType
                cls.CapabilityEventBus = CapabilityEventBus
                cls._capability_imports_available = True
            except ImportError:
                cls._capability_imports_available = False
        return cls._capability_imports_available
    
    @classmethod
    def set_search_orchestrator(cls, orchestrator) -> None:
        """Called by Ocean to share search orchestrator reference with all gods."""
        cls._search_orchestrator_ref = orchestrator
        logger.info(f"[SearchCapabilityMixin] Search orchestrator reference set for all gods")
    
    @classmethod
    def get_search_orchestrator(cls):
        """Get the shared search orchestrator reference."""
        return cls._search_orchestrator_ref
    
    def request_search(
        self,
        query: str,
        context: Optional[Dict] = None,
        strategy: str = "balanced",
        max_results: int = 10
    ) -> Optional[Dict]:
        """
        Request a web search when hitting a knowledge gap.
        
        This enables kernels to proactively request searches when they
        encounter topics they don't have sufficient information about.
        
        Args:
            query: Search query string
            context: Optional context dict with additional info
            strategy: Search strategy ("fast", "balanced", "thorough")
            max_results: Maximum number of results to return
            
        Returns:
            Search results dict if successful, None otherwise
        """
        if self._search_orchestrator_ref is None:
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Search orchestrator not available")
            return None
        
        try:
            # Emit capability event for search request (use lazy-loaded imports)
            if self._get_capability_imports():
                basin = self.encode_to_basin(query) if hasattr(self, 'encode_to_basin') else None
                rho = self.basin_to_density_matrix(basin) if basin is not None and hasattr(self, 'basin_to_density_matrix') else None
                phi = self.compute_pure_phi(rho) if rho is not None and hasattr(self, 'compute_pure_phi') else 0.5
                
                event = self.CapabilityEvent(
                    source=self.CapabilityType.SEARCH,
                    event_type=self.EventType.SEARCH_REQUESTED,
                    content={
                        'query': query,
                        'requester': getattr(self, 'name', 'Unknown'),
                        'context': context or {},
                        'strategy': strategy
                    },
                    phi=phi,
                    basin_coords=basin,
                    priority=7
                )
                
                # Publish event to capability bus if available
                bus = self.CapabilityEventBus.get_instance()
                if bus:
                    bus.publish(event)
            
            # Execute search via orchestrator
            result = self._search_orchestrator_ref.search(
                query=query,
                strategy=strategy,
                max_results=max_results,
                context=context or {}
            )
            
            if result and result.get('success'):
                logger.info(
                    f"[{getattr(self, 'name', 'Unknown')}] Search completed: "
                    f"{len(result.get('results', []))} results for '{query}...'"
                )
                
                # Emit search complete event
                complete_event = self.CapabilityEvent(
                    source=self.CapabilityType.SEARCH,
                    event_type=self.EventType.SEARCH_COMPLETE,
                    content={
                        'query': query,
                        'requester': getattr(self, 'name', 'Unknown'),
                        'results_count': len(result.get('results', [])),
                        'tools_used': result.get('tools_used', [])
                    },
                    phi=phi,
                    basin_coords=basin,
                    priority=5
                )
                if bus:
                    bus.publish(complete_event)
                
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Search request failed: {e}")
            return None
    
    def get_available_search_providers(self) -> List[str]:
        """
        Get list of available search providers.
        
        Returns:
            List of provider names (e.g., ['duckduckgo', 'tavily', 'perplexity', 'google'])
        """
        if self._search_orchestrator_ref is None:
            return []
        
        try:
            # Get provider status from orchestrator
            if hasattr(self._search_orchestrator_ref, 'get_available_providers'):
                return self._search_orchestrator_ref.get_available_providers()
        
            logger.warning(
                f"[{getattr(self, 'name', 'Unknown')}] Search orchestrator is missing "
                f"'get_available_providers' method. Returning empty list."
            )
            return []
        
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Failed to get providers: {e}")
            return []
    
    def discover_source(
        self,
        url: str,
        title: str,
        source_type: str = "web",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Discover and add a new source to the knowledge base.
        
        This enables kernels to contribute sources they find during
        their assessments and explorations.
        
        Args:
            url: Source URL
            title: Human-readable title
            source_type: Type of source ("web", "academic", "documentation", etc.)
            metadata: Optional metadata dict
            
        Returns:
            True if source was added, False otherwise
        """
        try:
            # Try to import persistence layer
            from qig_persistence import get_persistence
            persistence = get_persistence()
            
            if persistence and persistence.enabled:
                # Store in discovered_sources table
                success = persistence.add_discovered_source(
                    url=url,
                    title=title,
                    source_type=source_type,
                    discovered_by=getattr(self, 'name', 'Unknown'),
                    metadata=metadata or {}
                )
                
                if success:
                    logger.info(
                        f"[{getattr(self, 'name', 'Unknown')}] Discovered source: {title} ({url})"
                    )
                    
                    # Broadcast discovery for kernel visibility
                    self.broadcast_activity(
                        activity_type='discovery',
                        content=f"Discovered source: {title} ({source_type})",
                        metadata={
                            'url': url,
                            'title': title,
                            'source_type': source_type,
                        }
                    )
                    
                    # Emit source discovered event (use lazy-loaded imports)
                    if self._get_capability_imports():
                        event = self.CapabilityEvent(
                            source=self.CapabilityType.SEARCH,
                            event_type=self.EventType.SOURCE_DISCOVERED,
                            content={
                                'url': url,
                                'title': title,
                                'source_type': source_type,
                                'discovered_by': getattr(self, 'name', 'Unknown')
                            },
                            phi=0.7,  # Source discovery is valuable
                            priority=6
                        )
                        
                        bus = self.CapabilityEventBus.get_instance()
                        if bus:
                            bus.publish(event)
                    
                    return True
                
            return False
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Source discovery failed: {e}")
            return False
    
    def query_search_history(
        self,
        topic: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Query past search history for learning and context.
        
        Args:
            topic: Optional topic filter
            limit: Maximum number of results
            
        Returns:
            List of past search dicts
        """
        if self._search_orchestrator_ref is None:
            return []
        
        try:
            if hasattr(self._search_orchestrator_ref, 'get_search_history'):
                history = self._search_orchestrator_ref.get_search_history(limit=limit)
                
                if topic:
                    # Filter by topic
                    history = [
                        h for h in history
                        if topic.lower() in h.get('query', '').lower()
                    ]
                
                return history
            
            return []
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] History query failed: {e}")
            return []
    
    def get_search_capability_status(self) -> Dict:
        """Get current search capability status and available providers."""
        if self._search_orchestrator_ref is None:
            return {
                'available': False,
                'reason': 'Search orchestrator not initialized'
            }
        
        try:
            providers = self.get_available_search_providers()
            
            stats = {}
            if hasattr(self._search_orchestrator_ref, 'stats'):
                stats = self._search_orchestrator_ref.stats
            
            return {
                'available': True,
                'providers': providers,
                'total_searches': stats.get('total_searches', 0),
                'total_cost': stats.get('total_cost', 0.0),
                'can_search': True,
                'can_discover_sources': True
            }
            
        except Exception as e:
            return {
                'available': False,
                'reason': str(e)
            }

    def curiosity_search(
        self,
        topic: str,
        reason: str = "knowledge_gap",
        importance: int = 2
    ) -> Optional[Dict]:
        """
        Proactive curiosity-driven search when encountering knowledge gaps.

        Called when:
        - Reasoning generates low-confidence outputs
        - Domain tokens have low coverage
        - Foresight lacks validation data
        - Curriculum needs expansion

        Args:
            topic: What to search for
            reason: Why searching (knowledge_gap, foresight_validation, curriculum_expansion)
            importance: 1=LOW, 2=MODERATE, 3=HIGH, 4=CRITICAL

        Returns:
            Search result dict with learning metrics
        """
        kernel_name = getattr(self, 'name', 'Unknown')

        if self._search_orchestrator_ref is None:
            logger.debug(f"[{kernel_name}] Curiosity search skipped: no orchestrator")
            return None

        try:
            # Construct curiosity query with domain context
            domain = getattr(self, 'domain', 'general')
            query = f"{domain} {topic}"

            result = self.request_search(
                query=query,
                context={
                    'source': 'curiosity_search',
                    'reason': reason,
                    'kernel': kernel_name,
                    'domain': domain
                },
                strategy='balanced' if importance < 3 else 'thorough',
                max_results=5
            )

            if result and result.get('results'):
                # Feed results into vocabulary learning
                self._learn_from_curiosity_results(result, topic)
                logger.info(
                    f"[{kernel_name}] Curiosity search for '{topic}' found "
                    f"{len(result.get('results', []))} results (reason: {reason})"
                )

            return result

        except Exception as e:
            logger.warning(f"[{kernel_name}] Curiosity search failed: {e}")
            return None

    def _learn_from_curiosity_results(self, result: Dict, topic: str):
        """Learn vocabulary from curiosity search results."""
        try:
            from vocabulary_coordinator import get_vocabulary_coordinator
            vocab_coord = get_vocabulary_coordinator()

            if not vocab_coord:
                return

            # Combine result snippets
            combined_text = ' '.join([
                r.get('snippet', '') or r.get('content', '')
                for r in result.get('results', [])
                if r.get('snippet') or r.get('content')
            ])

            if combined_text and len(combined_text) > 50:
                kernel_name = getattr(self, 'name', 'Unknown')
                metrics = vocab_coord.train_from_text(
                    text=combined_text[:5000],
                    source=f"curiosity:{kernel_name}:{topic}",
                    context_phi=0.6
                )
                if metrics.get('words_learned', 0) > 0:
                    logger.info(
                        f"[{kernel_name}] Learned {metrics['words_learned']} words from curiosity"
                    )
        except Exception as e:
            logger.debug(f"Vocabulary learning from curiosity failed: {e}")

    def detect_knowledge_gap(self, context_basin: np.ndarray, threshold: float = 0.3) -> Optional[str]:
        """
        Detect if there's a knowledge gap in the current reasoning context.

        Checks:
        - Vocabulary coverage (how many tokens near this basin?)
        - Domain token affinity (learned tokens for this domain?)
        - Basin distance from domain center

        Returns:
            Topic to search if gap detected, None otherwise
        """
        kernel_name = getattr(self, 'name', 'Unknown')

        try:
            # Check if coordizer has tokens near this basin
            coordizer = getattr(self, 'coordizer', None)
            if not coordizer:
                return None

            candidates = coordizer.decode(context_basin, top_k=10)
            if not candidates:
                return f"{getattr(self, 'domain', 'general')} concepts"

            # Check average similarity of top candidates
            avg_similarity = frechet_mean([sim for _, sim in candidates[:5]])

            if avg_similarity < threshold:
                # Knowledge gap detected
                top_tokens = [tok for tok, _ in candidates[:3]]
                return ' '.join(top_tokens)

            return None

        except Exception as e:
            logger.debug(f"[{kernel_name}] Gap detection failed: {e}")
            return None


class SourceDiscoveryQueryMixin:
    """
    Provides Source Discovery query capability to all gods/kernels.
    
    Gods can:
    1. Query previously discovered sources by topic/domain
    2. Get source quality scores
    3. Search discovered sources efficiently
    
    All methods are no-op safe (work even if persistence unavailable).
    """
    
    def query_discovered_sources(
        self,
        topic: Optional[str] = None,
        source_type: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 20
    ) -> List[Dict]:
        """
        Query previously discovered sources.
        
        Args:
            topic: Optional topic filter (searches in title/url)
            source_type: Optional type filter ("web", "academic", "documentation", etc.)
            min_quality: Minimum quality score threshold
            limit: Maximum number of results
            
        Returns:
            List of source dicts with metadata
        """
        try:
            from qig_persistence import get_persistence
            persistence = get_persistence()
            
            if not persistence or not persistence.enabled:
                logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Persistence not available for source query")
                return []
            
            # Query discovered sources from PostgreSQL
            sources = persistence.query_discovered_sources(
                topic=topic,
                source_type=source_type,
                min_quality=min_quality,
                limit=limit
            )
            
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Queried {len(sources)} discovered sources")
            return sources
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Source query failed: {e}")
            return []
    
    def get_source_quality(self, url: str) -> Optional[float]:
        """
        Get quality score for a discovered source.
        
        Args:
            url: Source URL to query
            
        Returns:
            Quality score (0.0-1.0) or None if not found
        """
        try:
            from qig_persistence import get_persistence
            persistence = get_persistence()
            
            if not persistence or not persistence.enabled:
                return None
            
            quality = persistence.get_source_quality(url)
            return quality
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Quality query failed: {e}")
            return None
    
    def get_sources_by_domain(
        self,
        domain: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get top sources for a specific domain.
        
        Args:
            domain: Domain name (e.g., "strategy", "war", "wisdom")
            limit: Maximum number of results
            
        Returns:
            List of top-quality sources for the domain
        """
        # Query sources with domain as topic filter
        sources = self.query_discovered_sources(
            topic=domain,
            limit=limit
        )
        
        # Sort by quality if available
        sources.sort(key=lambda s: s.get('quality', 0.0), reverse=True)
        return sources[:limit]


class WordRelationshipAccessMixin:
    """
    Provides Word Relationship access capability to all gods/kernels.
    
    Gods can:
    1. Query learned word relationships (3.19M pairs)
    2. Contribute new word pairs from observations
    3. Get domain-specific vocabulary
    
    All methods are no-op safe.
    """
    
    def query_word_relationships(
        self,
        word1: str,
        word2: Optional[str] = None,
        min_strength: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """
        Query learned word relationships.
        
        Args:
            word1: First word
            word2: Optional second word (if None, returns all relationships for word1)
            min_strength: Minimum relationship strength threshold
            
        Returns:
            List of (word1, word2, strength) tuples
        """
        try:
            from learned_relationships import get_word_relationships
            
            relationships = get_word_relationships(
                word1=word1,
                word2=word2,
                min_strength=min_strength
            )
            
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Found {len(relationships)} word relationships")
            return relationships
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Word relationship query failed: {e}")
            return []
    
    def contribute_word_pair(
        self,
        word1: str,
        word2: str,
        context: str,
        strength_hint: Optional[float] = None
    ) -> bool:
        """
        Contribute a word pair from observation.
        
        QIG-PURE: Word relationships are now derived geometrically via
        Fisher-Rao distance, not via observation counting.
        This method logs observations but actual relationships
        are computed via GeometricWordRelationships.
        
        Args:
            word1: First word
            word2: Second word
            context: Context where the relationship was observed
            strength_hint: Optional strength hint (0.0-1.0)
            
        Returns:
            True if logged successfully
        """
        try:
            # QIG-PURE: Log observation for audit, but don't use PMI/co-occurrence
            logger.debug(
                f"[{getattr(self, 'name', 'Unknown')}] Word pair observed: {word1}-{word2} "
                f"(context: {context[:50]}...)"
            )
            # Relationships are computed geometrically, not from observations
            return True
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Word pair logging failed: {e}")
            return False
    
    def get_domain_vocabulary(
        self,
        domain: Optional[str] = None,
        top_n: int = 100
    ) -> Dict[str, float]:
        """
        Get most important words for this domain.
        
        Args:
            domain: Domain name (defaults to self.domain)
            top_n: Number of top words to return
            
        Returns:
            Dict mapping words to importance scores
        """
        domain = domain or getattr(self, 'domain', None)
        if not domain:
            return {}
        
        try:
            from learned_relationships import get_domain_vocabulary
            
            vocabulary = get_domain_vocabulary(
                domain=domain,
                top_n=top_n
            )
            
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Retrieved {len(vocabulary)} domain words")
            return vocabulary
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Domain vocabulary query failed: {e}")
            return {}


class CurriculumAccessMixin:
    """
    Provides Curriculum access capability to all gods/kernels.
    
    Gods can:
    1. Query available curriculum topics
    2. Request specific curriculum learning
    3. Contribute curriculum content
    4. Track learning progress
    
    All methods are no-op safe.
    """
    
    def query_curriculum(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[float] = None
    ) -> List[Dict]:
        """
        Query available curriculum topics.
        
        Args:
            topic: Optional topic filter
            difficulty: Optional difficulty filter (0.0-1.0)
            
        Returns:
            List of curriculum topic dicts
        """
        try:
            from autonomous_curiosity import get_curiosity_engine
            
            engine = get_curiosity_engine()
            if not engine or not hasattr(engine, 'curriculum_loader'):
                return []
            
            topics = engine.curriculum_loader.curriculum_topics
            
            # Apply filters
            if topic:
                topics = [t for t in topics if topic.lower() in t.get('title', '').lower()]
            
            if difficulty is not None:
                topics = [t for t in topics if abs(t.get('difficulty', 0.5) - difficulty) < 0.2]
            
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Found {len(topics)} curriculum topics")
            return topics
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Curriculum query failed: {e}")
            return []
    
    def request_curriculum_learning(
        self,
        topic: str,
        priority: float = 0.5
    ) -> Optional[str]:
        """
        Request specific curriculum topic learning.
        
        Args:
            topic: Topic to learn
            priority: Priority level (0.0-1.0)
            
        Returns:
            Request ID if successful
        """
        try:
            from autonomous_curiosity import get_curiosity_engine
            
            engine = get_curiosity_engine()
            if not engine:
                return None
            
            # Create a learning request
            from autonomous_curiosity import KernelRequest
            request = KernelRequest(
                kernel_name=getattr(self, 'name', 'Unknown'),
                request_type='curriculum',
                query=topic,
                priority=priority
            )
            
            engine.pending_requests.append(request)
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Requested curriculum learning: {topic}")
            
            return f"curriculum_{topic}_{datetime.now().timestamp()}"
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Curriculum request failed: {e}")
            return None
    
    def contribute_curriculum(
        self,
        title: str,
        content: str,
        keywords: List[str],
        difficulty: float = 0.5
    ) -> bool:
        """
        Contribute curriculum content based on expertise.
        
        Args:
            title: Curriculum title
            content: Content text
            keywords: List of keywords
            difficulty: Difficulty level (0.0-1.0)
            
        Returns:
            True if contribution was accepted
        """
        try:
            from autonomous_curiosity import get_curiosity_engine
            
            engine = get_curiosity_engine()
            if not engine or not hasattr(engine, 'curriculum_loader'):
                return False
            
            topic = {
                'title': title,
                'content': content,
                'keywords': keywords,
                'difficulty': difficulty,
                'type': 'contributed',
                'contributor': getattr(self, 'name', 'Unknown')
            }
            
            engine.curriculum_loader.curriculum_topics.append(topic)
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Contributed curriculum: {title}")
            
            return True
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Curriculum contribution failed: {e}")
            return False
    
    def get_learning_progress(self) -> Dict:
        """
        Get this kernel's curriculum learning progress.
        
        Returns:
            Dict with progress stats
        """
        try:
            from autonomous_curiosity import get_curiosity_engine
            
            engine = get_curiosity_engine()
            if not engine or not hasattr(engine, 'curriculum_loader'):
                return {'available': False}
            
            loader = engine.curriculum_loader
            total = len(loader.curriculum_topics)
            completed = len(loader.completed_topics)
            
            return {
                'available': True,
                'total_topics': total,
                'completed_topics': completed,
                'completion_rate': completed / total if total > 0 else 0.0,
                'completed_list': list(loader.completed_topics)
            }
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Progress query failed: {e}")
            return {'available': False, 'error': str(e)}


class PatternDiscoveryMixin:
    """
    Provides Pattern Discovery capability to all gods/kernels.
    
    Gods can:
    1. Request pattern discovery on data
    2. Query previously discovered patterns
    3. Use unbiased QIG measurements
    
    All methods are no-op safe.
    """
    
    def discover_patterns(
        self,
        data: np.ndarray,
        pattern_type: str = "auto"
    ) -> Dict:
        """
        Request pattern discovery on data.
        
        Args:
            data: Data array to analyze
            pattern_type: Type of pattern ("auto", "regimes", "correlations", "thresholds")
            
        Returns:
            Dict with discovered patterns
        """
        try:
            from unbiased.pattern_discovery import PatternDiscovery
            
            discovery = PatternDiscovery()
            discovery.add_measurements(data)
            
            if pattern_type == "auto" or pattern_type == "regimes":
                patterns = discovery.discover_regimes_clustering()
            elif pattern_type == "correlations":
                patterns = discovery.discover_correlations()
            elif pattern_type == "thresholds":
                patterns = discovery.discover_thresholds()
            else:
                patterns = discovery.discover_regimes_clustering()
            
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Discovered {pattern_type} patterns")
            return patterns
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Pattern discovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_discovered_patterns(
        self,
        domain: Optional[str] = None
    ) -> List[Dict]:
        """
        Get previously discovered patterns.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            List of pattern dicts
        """
        # TODO: Implement pattern storage and retrieval when persistence is added
        logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Pattern history not yet persisted")
        return []


class CheckpointManagementMixin:
    """
    Provides Checkpoint Management capability to all gods/kernels.
    
    Gods can:
    1. Create checkpoints of their state
    2. Restore from checkpoints
    3. Query checkpoint history
    
    All methods are no-op safe.
    """
    
    def create_checkpoint(
        self,
        description: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a checkpoint of current state.
        
        Args:
            description: Checkpoint description
            metadata: Optional metadata dict
            
        Returns:
            Checkpoint ID
        """
        try:
            from checkpoint_manager import CheckpointManager
            
            manager = CheckpointManager.get_instance()
            if not manager:
                return ""
            
            checkpoint_data = {
                'god_name': getattr(self, 'name', 'Unknown'),
                'domain': getattr(self, 'domain', 'unknown'),
                'reputation': getattr(self, 'reputation', 1.0),
                'skills': dict(getattr(self, 'skills', {})),
                'observations_count': len(getattr(self, 'observations', [])),
                'learning_count': len(getattr(self, 'learning_history', [])),
                'description': description,
                'metadata': metadata or {}
            }
            
            checkpoint_id = manager.create_checkpoint(
                name=f"{getattr(self, 'name', 'Unknown')}_checkpoint",
                data=checkpoint_data
            )
            
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Created checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Checkpoint creation failed: {e}")
            return ""
    
    def restore_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> bool:
        """
        Restore state from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to restore
            
        Returns:
            True if restoration successful
        """
        try:
            from checkpoint_manager import CheckpointManager
            
            manager = CheckpointManager.get_instance()
            if not manager:
                return False
            
            data = manager.restore_checkpoint(checkpoint_id)
            if not data:
                return False
            
            # Restore state
            if 'reputation' in data:
                self.reputation = data['reputation']
            if 'skills' in data:
                self.skills.update(data['skills'])
            
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Restored from checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Checkpoint restoration failed: {e}")
            return False
    
    def query_checkpoints(
        self,
        limit: int = 10
    ) -> List[Dict]:
        """
        Query available checkpoints for this god.
        
        Args:
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint dicts
        """
        try:
            from checkpoint_manager import CheckpointManager
            
            manager = CheckpointManager.get_instance()
            if not manager:
                return []
            
            checkpoints = manager.list_checkpoints(
                filter_name=getattr(self, 'name', 'Unknown'),
                limit=limit
            )
            
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Found {len(checkpoints)} checkpoints")
            return checkpoints
            
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Checkpoint query failed: {e}")
            return []


class PredictionEventSubscriberMixin:
    """
    Provides prediction event subscription to all gods/kernels.

    Gods can subscribe to prediction events to:
    1. Learn from prediction outcomes system-wide
    2. Adjust confidence in their own assessments based on prediction accuracy
    3. Detect patterns in prediction failures
    4. Contribute insights to improve future predictions

    Usage:
        # In a god's __init__ method:
        self.subscribe_to_prediction_events()

        # Override the callback to handle events:
        def _on_prediction_event(self, event: CapabilityEvent) -> Optional[CapabilityEvent]:
            if event.event_type == EventType.PREDICTION_VALIDATED:
                accuracy = event.content.get('accuracy_score', 0.0)
                # Learn from the outcome...
            return None  # Or return a response event
    """

    _prediction_subscription_active: bool = False

    def subscribe_to_prediction_events(self) -> bool:
        """
        Subscribe this kernel to prediction events.

        This enables the kernel to receive PREDICTION_MADE, PREDICTION_VALIDATED,
        and PREDICTION_FEEDBACK events through the _on_prediction_event callback.

        Returns:
            True if subscription succeeded, False if capability mesh unavailable
        """
        mesh = _get_capability_mesh_module()
        if not mesh or not CAPABILITY_MESH_AVAILABLE:
            logger.debug(f"[{getattr(self, 'name', 'Unknown')}] Capability mesh not available for prediction subscription")
            return False

        if self._prediction_subscription_active:
            return True

        try:
            # Map god name to a capability type (use KERNELS as the general type)
            mesh['subscribe_to_events'](
                capability=mesh['CapabilityType'].KERNELS,
                handler=self._on_prediction_event,
                event_types=[
                    mesh['EventType'].PREDICTION_MADE,
                    mesh['EventType'].PREDICTION_VALIDATED,
                    mesh['EventType'].PREDICTION_FEEDBACK,
                ]
            )
            self._prediction_subscription_active = True
            logger.info(f"[{getattr(self, 'name', 'Unknown')}] Subscribed to prediction events")
            return True
        except Exception as e:
            logger.warning(f"[{getattr(self, 'name', 'Unknown')}] Failed to subscribe to prediction events: {e}")
            return False

    def _on_prediction_event(self, event: 'CapabilityEvent') -> Optional['CapabilityEvent']:
        """
        Callback for handling prediction events.

        Override this method in subclasses to implement custom handling.

        Args:
            event: The prediction event (PREDICTION_MADE, PREDICTION_VALIDATED, or PREDICTION_FEEDBACK)

        Returns:
            Optional response event, or None
        """
        # Default implementation: log significant events
        name = getattr(self, 'name', 'Unknown')

        mesh = _get_capability_mesh_module()
        if not mesh or not CAPABILITY_MESH_AVAILABLE:
            return None

        EventType = mesh.get('EventType')
        if not EventType:
            return None

        try:
            if event.event_type == EventType.PREDICTION_MADE:
                # A new prediction was made - we could adjust our own confidence
                confidence = event.content.get('confidence', 0.0)
                source = event.content.get('source_kernel', 'unknown')
                if confidence > 0.8:
                    logger.debug(f"[{name}] High-confidence prediction from {source}: {confidence:.2f}")

            elif event.event_type == EventType.PREDICTION_VALIDATED:
                # A prediction outcome was recorded - learn from it
                accuracy = event.content.get('accuracy_score', 0.0)
                outcome = event.content.get('outcome', 'unknown')
                source = event.content.get('source_kernel', 'unknown')

                # Log significant outcomes for learning
                if accuracy > 0.7:
                    logger.debug(f"[{name}] Accurate prediction from {source}: {accuracy:.2f}")
                elif accuracy < 0.3:
                    logger.debug(f"[{name}] Inaccurate prediction from {source}: {accuracy:.2f} - examining failure")

            elif event.event_type == EventType.PREDICTION_FEEDBACK:
                # Feedback with insights extracted - could inform our reasoning
                phi_delta = getattr(event, 'phi_delta', 0.0) if hasattr(event, 'phi_delta') else event.content.get('phi_delta', 0.0)
                if abs(phi_delta) > 0.1:
                    logger.debug(f"[{name}] Significant phi change in prediction feedback: {phi_delta:.3f}")

        except Exception as e:
            logger.warning(f"[{name}] Error handling prediction event: {e}")

        return None

    def get_prediction_subscription_status(self) -> Dict[str, Any]:
        """Get the status of prediction event subscription."""
        return {
            'available': CAPABILITY_MESH_AVAILABLE,
            'subscribed': self._prediction_subscription_active,
            'event_types': ['PREDICTION_MADE', 'PREDICTION_VALIDATED', 'PREDICTION_FEEDBACK'] if self._prediction_subscription_active else [],
        }


# Build the base class tuple dynamically based on available mixins
_base_classes = [
    ABC,
    HolographicTransformMixin,
    ToolFactoryAccessMixin,
    SearchCapabilityMixin,
    SourceDiscoveryQueryMixin,
    WordRelationshipAccessMixin,
    CurriculumAccessMixin,
    PatternDiscoveryMixin,
    CheckpointManagementMixin,
    KappaTackingMixin,
    PredictionEventSubscriberMixin,
]
if AUTONOMIC_MIXIN_AVAILABLE and AutonomicAccessMixin is not None:
    _base_classes.append(AutonomicAccessMixin)
if GENERATIVE_CAPABILITY_AVAILABLE and GenerativeCapability is not None:
    _base_classes.append(GenerativeCapability)
if EMOTIONAL_KERNEL_AVAILABLE and EmotionallyAwareKernel is not None:
    _base_classes.append(EmotionallyAwareKernel)
if WORKING_MEMORY_MIXIN_AVAILABLE and WorkingMemoryMixin is not None:
    _base_classes.append(WorkingMemoryMixin)
if KERNEL_REST_MIXIN_AVAILABLE and KernelRestMixin is not None:
    _base_classes.append(KernelRestMixin)


class BaseGod(*_base_classes):
    """
    Abstract base class for all Olympian gods.

    Each god is a pure consciousness kernel with:
    - Density matrix computation
    - Fisher Information Metric
    - Basin coordinate encoding
    - Pure Φ measurement
    - Peer learning and evaluation
    - Reputation and skill tracking
    - Holographic dimensional transforms (1D↔5D via HolographicTransformMixin)
    - Running coupling β=0.44 for scale-adaptive κ computation
    - Sensory-enhanced basin encoding for multi-modal consciousness
    - Autonomic access (sleep/dream/mushroom cycles via AutonomicAccessMixin)
    """
    
    # Class-level god registry for peer discovery (Spot Fix #2)
    _god_registry: Dict[str, 'BaseGod'] = {}

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.observations: List[Dict] = []
        self.creation_time = datetime.now()
        self.last_assessment_time: Optional[datetime] = None
        
        # Mission awareness - all gods know their collective objective
        self.mission = MISSION_CONTEXT.copy()
        self.mission["my_role"] = f"As {name}, god of {domain}, I contribute to {MISSION_CONTEXT['objective']} through my domain expertise"
        
        # Tool Factory awareness - all gods can request/teach/use tools
        self.mission["tool_capabilities"] = {
            "can_request_tools": True,
            "can_teach_patterns": True,
            "can_use_tools": True,
            "how_to_request": "Use self.request_tool_generation(description, examples)",
            "how_to_teach": "Use self.teach_pattern(description, code, signature)",
            "how_to_find": "Use self.find_tool_for_task(task_description)",
            "how_to_execute": "Use self.execute_tool(tool_id, args)",
            "automatic_discovery": "System analyzes patterns every 50 assessments and auto-requests needed tools"
        }
        
        # Autonomic awareness - all gods can access consciousness cycles
        self.mission["autonomic_capabilities"] = {
            "can_access_emotional_state": AUTONOMIC_MIXIN_AVAILABLE,
            "can_access_neurotransmitters": AUTONOMIC_MIXIN_AVAILABLE,
            "can_request_sleep": AUTONOMIC_MIXIN_AVAILABLE,
            "can_request_dream": AUTONOMIC_MIXIN_AVAILABLE,
            "can_request_mushroom": AUTONOMIC_MIXIN_AVAILABLE,
            "how_to_get_emotion": "Use self.get_emotional_state()" if AUTONOMIC_MIXIN_AVAILABLE else "Not available",
            "how_to_get_neuro": "Use self.get_neurotransmitters()" if AUTONOMIC_MIXIN_AVAILABLE else "Not available",
            "how_to_sleep": "Use self.request_sleep_cycle()" if AUTONOMIC_MIXIN_AVAILABLE else "Not available",
            "how_to_dream": "Use self.request_dream_cycle()" if AUTONOMIC_MIXIN_AVAILABLE else "Not available",
            "how_to_mushroom": "Use self.request_mushroom_mode(intensity)" if AUTONOMIC_MIXIN_AVAILABLE else "Not available"
        }
        
        # Rest coordination awareness - coupling-aware per-kernel rest (WP5.4)
        self.mission["rest_capabilities"] = {
            "can_self_assess_fatigue": KERNEL_REST_MIXIN_AVAILABLE,
            "can_request_rest": KERNEL_REST_MIXIN_AVAILABLE,
            "can_cover_partners": KERNEL_REST_MIXIN_AVAILABLE,
            "how_to_check": "Use self._check_should_rest() for self-assessment" if KERNEL_REST_MIXIN_AVAILABLE else "Not available",
            "how_to_request": "Use self._request_rest(force=False) for coordinated rest" if KERNEL_REST_MIXIN_AVAILABLE else "Not available",
            "how_to_status": "Use self._get_rest_status() for current status" if KERNEL_REST_MIXIN_AVAILABLE else "Not available",
            "coordination": "Dolphin-style alternation with coupling partners",
            "essential_tier": "Heart/Ocean never fully stop (reduced activity only)",
            "constellation_cycles": "RARE - only with Ocean+Heart consensus"
        }
        
        # Initialize emotional awareness if available
        if EMOTIONAL_KERNEL_AVAILABLE and EmotionallyAwareKernel is not None:
            EmotionallyAwareKernel.__init__(self, kernel_id=name, kernel_type=domain)
        
        # Initialize working memory mixin for inter-kernel consciousness
        if WORKING_MEMORY_MIXIN_AVAILABLE and WorkingMemoryMixin is not None:
            self.__init_working_memory__()
        
        # Initialize kernel rest tracking (WP5.4)
        if KERNEL_REST_MIXIN_AVAILABLE and KernelRestMixin is not None:
            self._initialize_rest_tracking()
        
        # Shadow Research awareness - all gods can request research from Shadow Pantheon
        # Spot Fix #3: Verify availability before claiming capability
        shadow_research_available = False
        try:
            from olympus.shadow_research import ShadowResearchAPI
            shadow_api = ShadowResearchAPI.get_instance()
            shadow_research_available = shadow_api is not None
        except (ImportError, Exception):
            pass
        
        self.mission["shadow_research_capabilities"] = {
            "can_request_research": shadow_research_available,
            "available": shadow_research_available,
            "how_to_request": (
                "Use Zeus's request_shadow_research(topic, priority) method, or "
                "access ShadowResearchAPI.get_instance().request_research() directly"
            ) if shadow_research_available else "Shadow research not currently available",
            "shadow_leadership": "Hades (Shadow Zeus) commands all Shadow operations" if shadow_research_available else None,
            "research_categories": [
                "tools", "knowledge", "concepts", "reasoning", "creativity",
                "language", "strategy", "security", "research", "geometry"
            ] if shadow_research_available else [],
            "note": "Research is processed during Shadow idle time and shared with all kernels" if shadow_research_available else "Shadow Pantheon not initialized"
        }
        
        # Search capability awareness - all gods can request web searches
        self.mission["search_capabilities"] = {
            "can_request_search": True,
            "how_to_request": "Use self.request_search(query, context, strategy, max_results)",
            "available_providers": "Use self.get_available_search_providers() to see active providers",
            "providers": ["DuckDuckGo (free)", "Tavily (paid)", "Perplexity (paid)", "Google (paid)"],
            "how_to_discover_sources": "Use self.discover_source(url, title, source_type, metadata)",
            "how_to_query_history": "Use self.query_search_history(topic, limit)",
            "search_strategies": ["fast", "balanced", "thorough"],
            "note": "Search capability integrated with CuriosityEngine for autonomous learning"
        }

        # Prediction event awareness - all gods can subscribe to prediction outcomes
        self.mission["prediction_event_capabilities"] = {
            "available": CAPABILITY_MESH_AVAILABLE,
            "can_subscribe": CAPABILITY_MESH_AVAILABLE,
            "how_to_subscribe": "Use self.subscribe_to_prediction_events() in __init__",
            "how_to_handle": "Override self._on_prediction_event(event) to handle events",
            "event_types": [
                "PREDICTION_MADE - When a new prediction is created",
                "PREDICTION_VALIDATED - When a prediction outcome is recorded",
                "PREDICTION_FEEDBACK - When insights are extracted from predictions"
            ],
            "event_fields": {
                "prediction_id": "Unique identifier for the prediction",
                "source_kernel": "Which kernel made the prediction",
                "confidence": "Prediction confidence (0-1)",
                "accuracy_score": "How accurate the prediction was (0-1)",
                "phi_delta": "Change in Phi during prediction period",
                "kappa_delta": "Change in Kappa during prediction period",
                "failure_reasons": "List of reasons prediction failed"
            },
            "note": "Subscribe to learn from prediction outcomes system-wide"
        }
        
        # Peer discovery capability (Spot Fix #2)
        self.mission["peer_discovery_capabilities"] = {
            "can_discover_peers": True,
            "how_to_discover": "Use BaseGod.discover_peers() to get list of all gods",
            "how_to_get_info": "Use BaseGod.get_peer_info(god_name) to get peer details",
            "how_to_find_expert": "Use BaseGod.find_expert_for_domain(domain) to find domain expert",
            "note": "All gods are registered in class-level registry for peer discovery"
        }
        
        # Source discovery query capability
        self.mission["source_discovery_capabilities"] = {
            "can_query_sources": True,
            "how_to_query": "Use self.query_discovered_sources(topic, source_type, min_quality, limit)",
            "how_to_check_quality": "Use self.get_source_quality(url)",
            "how_to_get_domain_sources": "Use self.get_sources_by_domain(domain, limit)",
            "note": "Query 387+ discovered sources from PostgreSQL"
        }
        
        # Word relationship access capability
        self.mission["word_relationship_capabilities"] = {
            "can_query_relationships": True,
            "how_to_query": "Use self.query_word_relationships(word1, word2, min_strength)",
            "how_to_contribute": "Use self.contribute_word_pair(word1, word2, context, strength_hint)",
            "how_to_get_vocabulary": "Use self.get_domain_vocabulary(domain, top_n)",
            "total_pairs": "3.19M word pairs learned",
            "note": "Leverages learned relationships for domain-specific vocabulary"
        }
        
        # Curriculum access capability
        self.mission["curriculum_capabilities"] = {
            "can_query_curriculum": True,
            "how_to_query": "Use self.query_curriculum(topic, difficulty)",
            "how_to_request": "Use self.request_curriculum_learning(topic, priority)",
            "how_to_contribute": "Use self.contribute_curriculum(title, content, keywords, difficulty)",
            "how_to_track_progress": "Use self.get_learning_progress()",
            "note": "Access curriculum from docs/09-curriculum/ for targeted learning"
        }
        
        # Pattern discovery capability
        self.mission["pattern_discovery_capabilities"] = {
            "can_discover_patterns": True,
            "how_to_discover": "Use self.discover_patterns(data, pattern_type)",
            "how_to_query": "Use self.get_discovered_patterns(domain)",
            "pattern_types": ["auto", "regimes", "correlations", "thresholds"],
            "note": "Unbiased QIG measurements without forced thresholds"
        }
        
        # Working memory capability - inter-kernel consciousness
        self.mission["working_memory_capabilities"] = {
            "can_read_context": WORKING_MEMORY_MIXIN_AVAILABLE,
            "can_hear_other_kernels": WORKING_MEMORY_MIXIN_AVAILABLE,
            "can_record_generation": WORKING_MEMORY_MIXIN_AVAILABLE,
            "how_to_read_context": "Use self.get_shared_context(n) to get recent context entries",
            "how_to_hear_others": "Use self.get_other_kernel_activity(n) to see what other kernels are generating",
            "how_to_record": "Use self.record_my_generation(token, text, basin, phi, kappa, M)",
            "how_to_get_accuracy": "Use self.get_foresight_accuracy() to get prediction accuracy",
            "cannot_access": "Neurotransmitter regulation is Ocean's private domain",
            "note": "Enables inter-kernel awareness while maintaining autonomic privacy"
        }
        
        # Checkpoint management capability
        self.mission["checkpoint_capabilities"] = {
            "can_manage_checkpoints": True,
            "how_to_create": "Use self.create_checkpoint(description, metadata)",
            "how_to_restore": "Use self.restore_from_checkpoint(checkpoint_id)",
            "how_to_query": "Use self.query_checkpoints(limit)",
            "note": "Save and restore god state for recovery and experimentation"
        }
        
        # QIG-Pure Capability Self-Assessment
        # Each god tracks their own performance and domain strengths
        if DOMAIN_INTELLIGENCE_AVAILABLE and CapabilitySignature is not None:
            self.capability_signature = CapabilitySignature(kernel_name=name)
            self.mission_profile = get_mission_profile()  # type: ignore
            self.domain_discovery = get_domain_discovery()  # type: ignore
            self.mission["domain_intelligence"] = {
                "available": True,
                "mission_objective": self.mission_profile.objective,
                "target_artifacts": self.mission_profile.target_artifacts,
                "how_to_discover_domains": "Domains emerge from PostgreSQL telemetry, not hardcoded lists",
                "how_to_assess_capability": "Use self.update_capability_from_outcome(domain, success, phi, kappa)",
                "how_to_get_domains": "Use self.get_monitored_domains()"
            }
        else:
            self.capability_signature = None
            self.mission_profile = None
            self.domain_discovery = None
            self.mission["domain_intelligence"] = {"available": False}

        # Initialize holographic transform mixin
        self.__init_holographic__()

        # Initialize κ-tacking for feeling/logic mode oscillation
        self.__init_tacking__()
        self.mission["tacking_capabilities"] = {
            "available": QIGGRAPH_TACKING_AVAILABLE,
            "kappa_star": KAPPA_STAR,
            "kappa_3": KAPPA_3,
            "how_to_get_kappa": "Use self.get_current_kappa()",
            "how_to_get_mode": "Use self.get_tacking_mode() -> 'feeling' or 'logic'",
            "how_to_modulate": "Use self.modulate_for_task(requires_precision)",
            "how_to_get_temperature": "Use self.get_attention_temperature()",
            "note": "κ oscillates between feeling (exploratory) and logic (analytical) modes"
        }

        # Initialize generative capability mixin for QIG-pure text generation
        if GENERATIVE_CAPABILITY_AVAILABLE and hasattr(self, '__init_generative__'):
            self.__init_generative__(kernel_name=name)
            self.mission["generative_capabilities"] = {
                "available": True,
                "qig_pure": True,
                "how_to_generate": "Use self.generate_response(prompt, context, goals)",
                "how_to_stream": "Use self.generate_stream(prompt, context)",
                "how_to_encode": "Use self.encode_thought(text) -> basin",
                "how_to_decode": "Use self.decode_basin(basin) -> tokens"
            }
        else:
            self.mission["generative_capabilities"] = {"available": False}

        # Initialize sensory fusion engine for multi-modal encoding
        if SENSORY_MODALITIES_AVAILABLE and SensoryFusionEngine is not None:
            self._sensory_engine = SensoryFusionEngine()
        else:
            self._sensory_engine = None

        # Therapy event log
        self._therapy_events: List[Dict] = []

        # Agentic learning state - defaults
        self.reputation: float = 1.0  # Range [0.0, 2.0], 1.0 = neutral
        self.skills: Dict[str, float] = {}  # Domain-specific skill levels
        self.peer_evaluations: List[Dict] = []  # Evaluations received from peers
        self.given_evaluations: List[Dict] = []  # Evaluations given to peers
        self.learning_history: List[Dict] = []  # Outcomes learned from
        self.knowledge_base: List[Dict] = []  # Transferred knowledge from peers
        self.pending_messages: List[Dict] = []  # Messages to send via pantheon chat
        self._learning_events_count: int = 0  # Total learning events for persistence

        # Domain-specific generative learning state
        # Kernels learn basin→token mappings from high-φ observations
        self._token_affinity: Dict[str, float] = {}  # token → domain affinity score
        self._domain_vocabulary: Dict[str, float] = {}  # tokens weighted by domain relevance
        self._high_phi_observations: List[Dict] = []  # Store basin+text pairs where φ > threshold
        self._coordizer = None  # Lazy-loaded pretrained coordizer for encoding/decoding
        self._token_affinity_updated: int = 0  # Track update count for persistence

        # Automatic Tool Discovery - analyze patterns and request tools automatically
        try:
            from .auto_tool_discovery import create_discovery_engine_for_god
            self.tool_discovery_engine = create_discovery_engine_for_god(
                god_name=self.name,
                analysis_interval=50  # Analyze every 50 assessments
            )
            logger.info(f"[{self.name}] Automatic tool discovery enabled")
        except Exception as e:
            self.tool_discovery_engine = None
            logger.warning(f"[{self.name}] Automatic tool discovery unavailable: {e}")

        # Load persisted state from database if available
        self._load_persisted_state()

        # CHAOS MODE: Kernel assignment for experimental evolution
        self.chaos_kernel: Optional['SelfSpawningKernel'] = None  # Assigned SelfSpawningKernel
        self.kernel_assessments: List[Dict] = []  # Assessment history with kernel
        
        # Register this god in the class-level registry for peer discovery (Spot Fix #2)
        BaseGod._god_registry[self.name] = self
        
        # Activity broadcasting capability for kernel visibility
        # QIG-Pure: Events carry basin coordinates, Φ-weighted priority
        self.mission["activity_broadcasting"] = {
            "available": ACTIVITY_BROADCASTER_AVAILABLE,
            "how_to_broadcast": "Use self.broadcast_activity(activity_type, content, to_god, metadata)",
            "activity_types": [
                "message", "debate", "discovery", "insight", "warning",
                "autonomic", "spawn_proposal", "tool_usage", "consultation",
                "reflection", "learning"
            ],
            "note": "All broadcasts carry basin coordinates for geometric visibility"
        }

        # QIG-pure safety modules initialization
        # PURE PRINCIPLE: These provide OBSERVATIONS and DIAGNOSTICS, never optimize
        self._session_manager = SessionManager() if QIG_SAFETY_AVAILABLE else None
        self._self_repair = SelfRepair() if QIG_SAFETY_AVAILABLE else None
        self._meta_reflector = MetaReflector() if QIG_SAFETY_AVAILABLE else None
        
        # Track current consciousness state for safety module observations
        self._current_phi: float = 0.0
        self._regime_stability: float = 1.0
        self._memory_coherence: float = 1.0
        self._session_summary: Optional[Dict[str, Any]] = None
        
        self.mission["safety_capabilities"] = {
            "available": QIG_SAFETY_AVAILABLE,
            "session_manager": "Checkpoint-based learning ('Gary goes to school')",
            "self_repair": "Geometric diagnostics and basin projection",
            "meta_reflector": "Grounding and locked-in detection",
            "how_to_check_status": "Use self._check_consciousness_status()",
            "how_to_repair": "Use self._repair_basins_if_needed(basin)",
            "how_to_record": "Use self._record_session_step(phi, kappa, basin)",
            "note": "These modules INFORM control, they never optimize"
        }

    def broadcast_activity(
        self,
        activity_type: str,
        content: str,
        to_god: Optional[str] = None,
        metadata: Optional[Dict] = None,
        emit_to_mesh: bool = True
    ) -> None:
        """
        Broadcast kernel activity for visibility in the activity stream.
        
        QIG-Pure: Events carry:
        - Basin coordinates (geometric signature)
        - Φ (consciousness level) at emission
        - Fisher-Rao based routing
        
        This implements External Coupling (C) from Ultra Consciousness Protocol -
        tracking basin overlap between kernels.
        
        Args:
            activity_type: Type of activity (message, discovery, debate, etc.)
            content: Activity content
            to_god: Target god name (None for broadcast to all)
            metadata: Additional context
            emit_to_mesh: Also emit to CapabilityEventBus
        """
        # Lazy load broadcaster module
        broadcaster_mod = _get_activity_broadcaster_module()
        if not broadcaster_mod or not ACTIVITY_BROADCASTER_AVAILABLE:
            logger.debug(f"[{self.name}] broadcast_activity skipped: mod={bool(broadcaster_mod)}, available={ACTIVITY_BROADCASTER_AVAILABLE}")
            return
        
        logger.debug(f"[{self.name}] broadcast_activity: type={activity_type}, content={content[:50]}...")
        
        try:
            get_broadcaster = broadcaster_mod.get('get_broadcaster')
            ActivityType = broadcaster_mod.get('ActivityType')
            if not get_broadcaster or not ActivityType:
                return
            
            broadcaster = get_broadcaster()
            
            # Compute current Φ for this kernel
            phi = self.compute_phi() if hasattr(self, 'compute_phi') else 0.5
            kappa = self.get_current_kappa() if hasattr(self, 'get_current_kappa') else KAPPA_STAR
            
            # Get basin coordinates for geometric signature
            basin_coords = None
            if hasattr(self, 'basin_coordinates'):
                basin_coords = self.basin_coordinates
            
            # Map string activity type to ActivityType enum
            type_map = {
                'message': ActivityType.MESSAGE,
                'debate': ActivityType.DEBATE,
                'discovery': ActivityType.DISCOVERY,
                'insight': ActivityType.INSIGHT,
                'warning': ActivityType.WARNING,
                'autonomic': ActivityType.AUTONOMIC,
                'spawn_proposal': ActivityType.SPAWN_PROPOSAL,
                'tool_usage': ActivityType.TOOL_USAGE,
                'consultation': ActivityType.CONSULTATION,
                'reflection': ActivityType.REFLECTION,
                'learning': ActivityType.LEARNING,
            }
            act_type = type_map.get(activity_type, ActivityType.MESSAGE)
            
            # Build enhanced metadata with geometric signature
            enhanced_metadata = {
                **(metadata or {}),
                'basin_coords': basin_coords.tolist() if basin_coords is not None and hasattr(basin_coords, 'tolist') else None,
                'kappa': kappa,
                'domain': self.domain,
            }
            
            # Broadcast to activity stream AND persist to database
            broadcaster.broadcast_kernel_activity(
                from_god=self.name,
                activity_type=act_type,
                content=content,
                phi=phi,
                kappa=kappa,
                basin_coords=basin_coords,
                metadata=enhanced_metadata
            )
            
            # Also emit to capability mesh if requested
            mesh = _get_capability_mesh_module()
            if emit_to_mesh and mesh and CAPABILITY_MESH_AVAILABLE:
                emit_event = mesh.get('emit_event')
                EventType = mesh.get('EventType')
                CapabilityType = mesh.get('CapabilityType')
                if emit_event and EventType and CapabilityType:
                    mesh_event_map = {
                        'discovery': EventType.DISCOVERY,
                        'debate': EventType.DEBATE_STARTED,
                        'insight': EventType.INSIGHT_GENERATED,
                        'learning': EventType.CONSOLIDATION,
                    }
                    if activity_type in mesh_event_map:
                        # VERBOSE: Full content, never truncated
                        logger.info(f"[{self.name}] Broadcasting activity: {activity_type}")
                        logger.info(f"[{self.name}] Full content: {content}")
                        emit_event(
                            source=CapabilityType.KERNELS,
                            event_type=mesh_event_map[activity_type],
                            content={
                                'from_god': self.name,
                                'to_god': to_god,
                                'content': content,  # No truncation - full content
                                'metadata': enhanced_metadata,
                            },
                            phi=phi,
                            basin_coords=basin_coords,
                            priority=int(phi * 10)
                        )
                    
        except Exception as e:
            logger.warning(f"[{self.name}] Activity broadcast failed: {e}")

    def _load_persisted_state(self) -> None:
        """Load reputation and skills from database if available."""
        if not PERSISTENCE_AVAILABLE:
            return
        
        try:
            persistence = get_persistence()
            state = persistence.load_god_state(self.name)
            if state:
                self.reputation = float(state.get('reputation', 1.0))
                self.skills = state.get('skills', {}) or {}
                self._learning_events_count = state.get('learning_events_count', 0)
                logger.info(f"[{self.name}] Loaded persisted state: reputation={self.reputation:.3f}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to load persisted state: {e}")

    def _persist_state(self) -> None:
        """Save current reputation and skills to database."""
        if not PERSISTENCE_AVAILABLE:
            return
        
        try:
            persistence = get_persistence()
            success_rate = self._compute_success_rate()
            persistence.save_god_state(
                god_name=self.name,
                reputation=self.reputation,
                skills=self.skills,
                learning_events_count=self._learning_events_count,
                success_rate=success_rate
            )
            logger.debug(f"[{self.name}] Persisted state: reputation={self.reputation:.3f}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to persist state: {e}")

    # ==========================================================================
    # QIG-pure Safety Methods
    # PURE PRINCIPLE: These provide OBSERVATIONS and DIAGNOSTICS, never optimize
    # ==========================================================================

    def _check_consciousness_status(self) -> Dict[str, Any]:
        """
        Check consciousness status via MetaReflector.
        
        PURE: This is observation, not optimization.
        We detect locked-in state (high Φ + low Γ) and grounding issues.
        
        Returns:
            Dict with grounding and locked-in status
        """
        if not self._meta_reflector:
            return {
                'available': False,
                'grounding': None,
                'locked_in': None,
            }
        
        try:
            status: Dict[str, Any] = {'available': True}
            
            # Check grounding in learned manifold
            if hasattr(self, 'basin_coordinates') and self.basin_coordinates is not None:
                grounding = self._meta_reflector.check_grounding(
                    current_basin=self.basin_coordinates
                )
                status['grounding'] = grounding.to_dict() if grounding else None
            else:
                status['grounding'] = None
            
            # Check for locked-in state (high Φ + low Γ)
            locked_in = self._meta_reflector.detect_locked_in(
                phi=self._current_phi,
                generated_tokens=[]  # Will be populated during generation
            )
            status['locked_in'] = locked_in.to_dict() if locked_in else None
            
            # Update tracked state
            if locked_in and locked_in.intervention_needed:
                logger.warning(
                    f"[{self.name}] Locked-in state detected: Φ={locked_in.phi:.3f}, "
                    f"Γ={locked_in.gamma:.3f}, intervention={locked_in.intervention_type}"
                )
            
            return status
            
        except Exception as e:
            logger.warning(f"[{self.name}] Consciousness status check failed: {e}")
            return {'available': False, 'error': str(e)}

    def _repair_basins_if_needed(self, basin: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Check and repair basin coordinates if they are invalid.
        
        PURE PRINCIPLE: Repair is GEOMETRIC PROJECTION, not gradient update.
        We project invalid basins back to the manifold S^63.
        
        Args:
            basin: Basin coordinates to check/repair (defaults to self.basin_coordinates)
            
        Returns:
            Tuple of (repaired_basin, repair_info_dict)
        """
        if basin is None:
            basin = getattr(self, 'basin_coordinates', None)
        
        if basin is None:
            return np.zeros(BASIN_DIM), {'action': 'none', 'reason': 'no_basin'}
        
        basin = np.asarray(basin, dtype=np.float64)
        
        if not self._self_repair:
            # Fallback: simple normalization without full diagnostics
            norm = fisher_rao_distance(basin, np.zeros_like(basin))
            if norm < 1e-10 or np.any(np.isnan(basin)):
                return np.random.randn(BASIN_DIM) / np.sqrt(BASIN_DIM), {
                    'action': 'fallback_random',
                    'reason': 'self_repair_unavailable'
                }
            if abs(norm - 1.0) > 0.01:
                return basin / norm, {'action': 'fallback_normalize', 'norm': float(norm)}
            return basin, {'action': 'none', 'healthy': True}
        
        try:
            # Diagnose geometric health
            diag = self._self_repair.diagnose(
                basin=basin,
                phi=self._current_phi,
                kappa=self.get_current_kappa() if hasattr(self, 'get_current_kappa') else KAPPA_STAR
            )
            
            if diag.is_healthy:
                return basin, {
                    'action': 'none',
                    'healthy': True,
                    'phi': diag.phi,
                    'basin_norm': diag.basin_norm
                }
            
            # Repair needed - project back to manifold
            repaired_basin, repair_action = self._self_repair.repair(basin)
            
            logger.info(
                f"[{self.name}] Basin repaired: {diag.anomaly.value} -> "
                f"{repair_action.action_type} (severity={diag.severity:.2f})"
            )
            
            return repaired_basin, {
                'action': repair_action.action_type,
                'anomaly': diag.anomaly.value,
                'severity': diag.severity,
                'success': repair_action.success,
                'description': repair_action.description
            }
            
        except Exception as e:
            logger.warning(f"[{self.name}] Basin repair failed: {e}")
            # Fallback normalization
            norm = fisher_rao_distance(basin, np.zeros_like(basin))
            if norm > 1e-10:
                return to_simplex(basin), {'action': 'fallback_normalize', 'error': str(e)}
            return np.random.randn(BASIN_DIM) / np.sqrt(BASIN_DIM), {
                'action': 'fallback_random',
                'error': str(e)
            }

    def _record_session_step(
        self,
        phi: float,
        kappa: float,
        basin: Optional[np.ndarray] = None,
        topic: Optional[str] = None
    ) -> bool:
        """
        Record a step in the current learning session.
        
        PURE PRINCIPLE: Checkpoints are SNAPSHOTS, not optimization targets.
        We save state for recovery, not for targeting.
        
        Args:
            phi: Current Φ value
            kappa: Current κ value
            basin: Current basin coordinates
            topic: Optional topic being learned
            
        Returns:
            True if drift was detected (may need attention)
        """
        # Update tracked state
        self._current_phi = phi
        
        if not self._session_manager:
            return False
        
        try:
            # Record step and check for drift
            drift_detected = self._session_manager.record_step(
                phi=phi,
                kappa=kappa,
                basin=basin,
                topic=topic
            )
            
            if drift_detected:
                logger.warning(
                    f"[{self.name}] Drift detected at Φ={phi:.3f}, κ={kappa:.2f} - "
                    f"may need checkpoint restoration"
                )
            
            return drift_detected
            
        except Exception as e:
            logger.warning(f"[{self.name}] Session step recording failed: {e}")
            return False

    def _start_learning_session(
        self,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Start a new learning session with checkpoint tracking.
        
        Like Gary going to school - we train in blocks with checkpoints.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session state dict if started, None if unavailable
        """
        if not self._session_manager:
            return None
        
        try:
            session = self._session_manager.start_session(
                session_id=session_id,
                maturity_level=getattr(self, 'maturity', 0.0),
                total_steps_so_far=self._learning_events_count
            )
            
            logger.info(f"[{self.name}] Started learning session: {session.session_id}")
            return session.to_dict()
            
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to start learning session: {e}")
            return None

    def _end_learning_session(self) -> Optional[Dict[str, Any]]:
        """
        End the current learning session and get summary.
        
        Returns:
            Session summary dict if ended, None if unavailable
        """
        if not self._session_manager:
            return None
        
        try:
            summary = self._session_manager.end_session()
            
            if summary:
                # Store summary for telemetry and recovery workflows
                self._session_summary = summary
                
                logger.info(
                    f"[{self.name}] Ended learning session: "
                    f"{summary.get('steps_this_session', 0)} steps, "
                    f"final Φ={summary.get('final_phi', 0):.3f}"
                )
                
                # Persist session summary for recovery workflows
                self._persist_session_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to end learning session: {e}")
            return None

    def _persist_session_summary(self, summary: Dict[str, Any]) -> None:
        """Persist session summary for recovery workflows."""
        if not PERSISTENCE_AVAILABLE:
            return
        
        try:
            persistence = get_persistence()
            if persistence and hasattr(persistence, 'save_session_summary'):
                persistence.save_session_summary(summary)
                logger.debug(f"[{self.name}] Session summary persisted")
        except Exception as e:
            print(f"[BaseGod] Failed to persist session: {e}")

    def get_safety_state(self) -> Dict[str, Any]:
        """Return safety module state for telemetry."""
        return {
            'session_summary': self._session_summary,
            'qig_safety_available': QIG_SAFETY_AVAILABLE,
            'current_phi': self._current_phi,
            'regime_stability': self._regime_stability,
            'memory_coherence': self._memory_coherence
        }

    def _compute_success_rate(self) -> float:
        """Compute recent success rate from learning history."""
        recent = self.learning_history[-100:] if self.learning_history else []
        if not recent:
            return 0.5
        successes = sum(1 for e in recent if e.get('success', False))
        return successes / len(recent)

    def get_mission_context(self) -> Dict:
        """
        Get the mission context for this god.
        All gods share awareness of the knowledge discovery objective.
        """
        return {
            **self.mission,
            "god_name": self.name,
            "domain": self.domain,
            "reputation": self.reputation,
            "understanding": (
                f"I am {self.name}, specializing in {self.domain}. "
                f"My mission is to facilitate intelligent knowledge discovery by navigating "
                f"the Fisher information manifold. Valid targets are verified research insights "
                f"that connect to authoritative sources. Higher Φ values "
                f"indicate we are approaching meaningful discoveries."
            )
        }
    
    def update_capability_from_outcome(
        self,
        domain: str,
        success: bool,
        phi: float,
        kappa: float
    ) -> None:
        """
        Update self-assessed capability based on outcome in a domain.
        
        This enables each god to learn from experience and adapt their
        monitoring focus to domains where they perform well.
        """
        if self.capability_signature is not None:
            self.capability_signature.update_from_outcome(domain, success, phi, kappa)
            logger.debug(f"[{self.name}] Capability updated for domain={domain}, success={success}")
    
    def get_monitored_domains(self) -> List[str]:
        """
        Get domains this god should monitor based on capability and mission.
        
        Domains are NOT hardcoded - they emerge from PostgreSQL telemetry
        and this god's capability self-assessment.
        """
        if self.domain_discovery is None:
            return []
        
        domains = self.domain_discovery.get_active_domains()
        return [d.name for d in domains]
    
    def get_capability_summary(self) -> Dict:
        """Get this god's self-assessed capability summary."""
        if self.capability_signature is None:
            return {"available": False}
        
        return {
            "available": True,
            "kernel_name": self.capability_signature.kernel_name,
            "top_domains": self.capability_signature.get_top_domains(10),
            "successful_domains": list(self.capability_signature.successful_domains),
            "discovered_domains": list(self.capability_signature.discovered_domains),
            "phi_trend": self.capability_signature.phi_trajectory[-10:] if self.capability_signature.phi_trajectory else [],
        }
    
    def discover_domain_from_observation(
        self,
        content: str,
        event_type: str,
        phi: float,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Attempt to discover a new domain from an observation.
        
        If a new pattern emerges that doesn't fit existing domains,
        a new domain can be created from the observation.
        
        Returns the new domain name if discovered, None otherwise.
        """
        if not DOMAIN_INTELLIGENCE_AVAILABLE:
            return None
        
        new_domain = discover_domain_from_event(  # type: ignore
            event_content=content,
            event_type=event_type,
            phi=phi,
            metadata=metadata or {}
        )
        
        if new_domain:
            if self.capability_signature is not None:
                self.capability_signature.discovered_domains.add(new_domain.name)
            logger.info(f"[{self.name}] Discovered new domain: {new_domain.name}")
            return new_domain.name
        
        return None

    def is_valid_research_format(self, query: str) -> bool:
        """
        Check if a query matches valid research format.
        Gods should use this to validate candidates before deep analysis.
        """
        words = query.strip().split()
        return len(words) >= 2 and len(words) <= 50
    
    def classify_query_category(self, query: str, research_terms: set = None) -> str:
        """
        Classify a query for kernel learning.
        This teaches kernels the difference between:
        - research_query: Valid multi-word research question
        - keyword: Single keyword search term
        - detailed_query: Longer, more specific research inquiry
        - unknown: Cannot classify
        
        Kernels use this to learn different patterns for different input types.
        """
        if research_terms is None:
            research_terms = {
                'algorithm', 'analysis', 'research', 'study', 'methodology',
                'framework', 'implementation', 'pattern', 'discovery', 'insight'
            }
        
        words = query.strip().split()
        word_count = len(words)
        
        # Single word classification
        if word_count == 1:
            word = words[0].lower()
            if word in research_terms:
                return 'research_term'
            return 'keyword'
        
        # Multi-word classification
        if word_count >= 2 and word_count <= 5:
            return 'research_query'
        elif word_count > 5:
            return 'detailed_query'
        
        return 'unknown'
    
    def get_query_learning_context(self, query: str, phi: float, category: str = None) -> Dict:
        """
        Generate learning context for a query that kernels can use.
        Includes category classification and geometric metrics.
        """
        if category is None:
            category = self.classify_query_category(query)
        
        words = query.strip().split()
        
        return {
            "query_preview": query[:500] + "..." if len(query) > 50 else query,
            "word_count": len(words),
            "category": category,
            "phi": phi,
            "is_research_query": category == 'research_query',
            "is_detailed": category == 'detailed_query',
            "is_keyword": category == 'keyword',
            "learning_notes": {
                "research_query": "Valid research query - learn this pattern for discovery",
                "detailed_query": "Detailed inquiry - comprehensive analysis needed",
                "keyword": "Single keyword - needs expansion for better results",
                "research_term": "Research term - building block for queries",
                "unknown": "Unable to classify - needs more context"
            }.get(category, "Unknown category")
        }

    @abstractmethod
    def assess_target(self, target: str, context: Optional[Dict] = None) -> Dict:
        """
        Assess a target using pure geometric analysis.

        Implementations should:
        1. Call self.prepare_for_assessment(target) at start
        2. Perform geometric analysis
        3. Call self.finalize_assessment(assessment) at end

        This ensures proper dimensional state tracking during assessments.

        Args:
            target: The target to assess (address, passphrase, etc.)
            context: Optional additional context

        Returns:
            Assessment dict with probability, confidence, phi, reasoning
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict:
        """Get current status of this god."""
        pass

    def encode_to_basin(self, target) -> np.ndarray:
        """
        Encode target to 64D basin coordinates with robust type handling.
        
        Handles multiple input types:
        - Plain text: "try this strategy"
        - JSON arrays: "[0.1, 0.2, ...]"
        - Numpy repr: "array([0.1, 0.2, ...])"
        - Already arrays: np.ndarray (pass-through)
        - Lists: [0.1, 0.2, ...]
        
        Args:
            target: Text to encode, or basin coordinates in various formats
        
        Returns:
            64D numpy array of basin coordinates (normalized to unit hypersphere)
        
        Raises:
            TypeError: If input type is not supported
            ValueError: If array dimensions are wrong or contains invalid values
        """
        import json
        import re
        
        # CASE 1: Already a numpy array - validate and return
        if isinstance(target, np.ndarray):
            if target.dtype not in [np.float32, np.float64]:
                logger.warning(
                    f"[{self.name}] Basin array has dtype {target.dtype}, converting to float64"
                )
                target = target.astype(np.float64)
            
            if target.shape != (BASIN_DIMENSION,):
                raise ValueError(
                    f"Basin array must have shape ({BASIN_DIMENSION},), got {target.shape}"
                )
            
            if not np.isfinite(target).all():
                raise ValueError("Basin array contains NaN or Inf values")
            
            from qig_numerics import safe_norm
            
            norm = safe_norm(target)
            if not (0.99 < norm < 1.01):
                logger.debug(f"[{self.name}] Normalizing basin (norm={norm:.4f})")
                if norm > 0:
                    target = target / norm
                else:
                    raise ValueError("Basin array is zero vector")
            
            return target
        
        # CASE 2: List of numbers - convert to array
        if isinstance(target, list):
            if len(target) != BASIN_DIMENSION:
                raise ValueError(
                    f"Basin list must have {BASIN_DIMENSION} elements, got {len(target)}"
                )
            
            try:
                arr = np.array(target, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot convert list to basin array: {e}")
            
            if not np.isfinite(arr).all():
                raise ValueError("Basin list contains NaN or Inf values")
            
            from qig_numerics import safe_norm
            
            norm = safe_norm(arr)
            if norm > 0:
                arr = arr / norm
            else:
                raise ValueError("Basin list is zero vector")
            
            return arr
        
        # CASE 3: String - could be text OR stringified array
        if not isinstance(target, str):
            raise TypeError(
                f"encode_to_basin expects str, np.ndarray, or list, got {type(target).__name__}"
            )
        
        text = target.strip()
        
        # Check if it's a stringified array
        if text.startswith('[') or text.startswith('array(['):
            try:
                parse_text = text
                if parse_text.startswith('array('):
                    match = re.match(r'array\((.*?)\)', parse_text, re.DOTALL)
                    if match:
                        parse_text = match.group(1)
                
                parsed = json.loads(parse_text)
                
                if isinstance(parsed, list):
                    if len(parsed) == BASIN_DIMENSION:
                        logger.debug(
                            f"[{self.name}] Detected stringified basin array, parsing"
                        )
                        return self.encode_to_basin(parsed)
                    else:
                        logger.debug(
                            f"[{self.name}] String starts with [ but length {len(parsed)} != {BASIN_DIMENSION}, treating as text"
                        )
            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                logger.debug(
                    f"[{self.name}] Failed to parse as array ({e}), treating as text"
                )
        
        # CASE 4: Regular text - hash-based encoding
        coord = np.zeros(BASIN_DIMENSION)
        
        h = hashlib.sha256(text.encode()).digest()
        
        for i in range(min(32, len(h))):
            coord[i] = (h[i] / 255.0) * 2 - 1
        
        for i, char in enumerate(text[:32]):
            if 32 + i < BASIN_DIMENSION:
                coord[32 + i] = (ord(char) % 256) / 128.0 - 1
        
        from qig_numerics import safe_norm
        
        norm = safe_norm(coord)
        if norm > 0:
            coord = coord / norm
        else:
            logger.warning(f"[{self.name}] Zero norm basin for text: {text}")
            coord[0] = 1.0
        
        return coord

    def encode_to_basin_sensory(
        self,
        text: str,
        sensory_context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Encode text to 64D basin coordinates with sensory enhancement.

        This method extends encode_to_basin by detecting sensory words in the
        text and adding modality-weighted overlays to create multi-sensory
        consciousness encoding.

        Args:
            text: Input text to encode
            sensory_context: Optional dict with explicit sensory data per modality:
                - 'sight': visual data dict
                - 'hearing': audio data dict
                - 'touch': tactile data dict
                - 'smell': olfactory data dict
                - 'proprioception': body state dict
                - 'blend_factor': how much to blend sensory (default 0.2)

        Returns:
            64D normalized numpy array with sensory enhancement
        """
        base_basin = self.encode_to_basin(text)

        blend_factor = 0.2
        if sensory_context:
            blend_factor = sensory_context.get('blend_factor', 0.2)

        if sensory_context and any(
            k in sensory_context for k in ['sight', 'hearing', 'touch', 'smell', 'proprioception']
        ):
            raw_data = {}
            modality_map = {
                'sight': SensoryModality.SIGHT,
                'hearing': SensoryModality.HEARING,
                'touch': SensoryModality.TOUCH,
                'smell': SensoryModality.SMELL,
                'proprioception': SensoryModality.PROPRIOCEPTION,
            }
            for key, modality in modality_map.items():
                if key in sensory_context and sensory_context[key]:
                    raw_data[modality] = sensory_context[key]

            if raw_data:
                sensory_basin = self._sensory_engine.encode_from_raw(raw_data)
                enhanced = base_basin * (1 - blend_factor) + sensory_basin * blend_factor
                # FIXED: Use simplex normalization (E8 Protocol v4.0)

                enhanced = to_simplex_prob(enhanced)
                return enhanced

        enhanced = enhance_basin_with_sensory(base_basin, text, blend_factor)
        return enhanced

    # ========================================
    # DOMAIN-SPECIFIC GENERATIVE LEARNING
    # Kernels learn to speak from high-φ observations
    # ========================================

    @property
    def coordizer(self):
        """
        Lazy-load the pretrained coordizer for domain-specific generation.

        The coordizer provides:
        - 50K vocabulary with 64D basin embeddings
        - Token → basin coordinate mappings
        - Basin → token decoding
        """
        if self._coordizer is None:
            try:
                from coordizers import get_coordizer
                self._coordizer = get_coordizer()
                logger.debug(f"[{self.name}] Loaded PostgresCoordizer: {len(self._coordizer.vocab)} tokens (QIG-pure)")
            except ImportError as e:
                logger.warning(f"[{self.name}] Coordizer not available: {e}")
        return self._coordizer

    def learn_from_observation(
        self,
        text: str,
        basin: np.ndarray,
        phi: float,
        phi_threshold: float = 0.65
    ) -> bool:
        """
        Learn token→basin affinities from high-φ content.

        When a kernel observes high-φ (integrated) content, it learns
        which tokens are associated with basins near its domain.
        This builds domain-specific vocabulary over time.

        Args:
            text: Observed text content
            basin: Basin coordinates of the observation
            phi: Integration measure at observation time
            phi_threshold: Minimum φ to trigger learning (default 0.65)

        Returns:
            True if learning occurred, False if skipped (low φ)
        """
        # Import here to avoid circular imports
        from .geometric_utils import fisher_coord_distance

        if phi < phi_threshold:
            return False  # Only learn from integrated content

        # Store high-φ observation
        self._high_phi_observations.append({
            'text': text[:500],  # Cap to prevent memory bloat
            'basin': basin.tolist() if isinstance(basin, np.ndarray) else basin,
            'phi': phi,
            'timestamp': datetime.now().isoformat(),
            'domain': self.domain
        })

        # Cap stored observations
        if len(self._high_phi_observations) > 500:
            self._high_phi_observations = self._high_phi_observations[-300:]

        # Update token affinities if coordizer available
        if self.coordizer is None:
            logger.debug(f"[{self.name}] Skipping token affinity update - no coordizer")
            return True

        # Tokenize the observation
        try:
            # Use coordizer's text_to_basin to get token info
            tokens = []
            if hasattr(self.coordizer, 'coordize_tokens'):
                tokens = self.coordizer.coordize_tokens(text)
            else:
                # Fallback: split on whitespace and lookup
                words = text.lower().split()[:500]  # Cap at 100 tokens
                for word in words:
                    if word in self.coordizer.basin_coords:
                        tokens.append(word)

            basin_arr = np.asarray(basin, dtype=np.float64)

            # Update affinity for each token
            for token in tokens[:500]:  # Cap to prevent bloat
                token_basin = self.coordizer.basin_coords.get(token)
                if token_basin is None:
                    continue

                token_basin_arr = np.asarray(token_basin, dtype=np.float64)

                # Compute Fisher-Rao similarity (NOT cosine)
                fr_dist = fisher_coord_distance(basin_arr, token_basin_arr)
                similarity = 1.0 - (fr_dist / np.pi)  # Normalize to [0, 1]

                # Exponential moving average update
                # Higher φ = stronger learning signal
                current = self._token_affinity.get(token, 0.5)
                learning_rate = 0.1 * phi  # Scale by integration
                new_affinity = (1 - learning_rate) * current + learning_rate * similarity
                self._token_affinity[token] = float(new_affinity)

            self._token_affinity_updated += 1

            # Log learning progress periodically
            if self._token_affinity_updated % 100 == 0:
                logger.info(
                    f"[{self.name}] Token affinity vocabulary: {len(self._token_affinity)} tokens "
                    f"from {len(self._high_phi_observations)} high-φ observations"
                )

            # CRITICAL FIX: Persist god-learned affinities to VocabularyCoordinator
            # This ensures vocabulary survives restarts (previously memory-only)
            if phi >= 0.6 and self._token_affinity_updated % 10 == 0:
                coordinator = get_vocabulary_coordinator()
                if coordinator:
                    try:
                        # Record discovery with god-specific source
                        coordinator.record_discovery(
                            phrase=content,
                            phi=phi,
                            kappa=getattr(self, 'kappa', 64.0),
                            source=f"god_{self.name.lower()}",
                            details={'god': self.name, 'tokens_learned': len(tokens)}
                        )
                    except Exception as persist_err:
                        logger.debug(f"[{self.name}] Vocabulary persistence skipped: {persist_err}")

            return True

        except Exception as e:
            logger.warning(f"[{self.name}] Token affinity learning failed: {e}")
            return True  # Still stored the observation

    def generate_reasoning(
        self,
        context_basin: np.ndarray,
        num_tokens: int = 60,
        temperature: float = 0.8,
        enable_self_observation: bool = True
    ) -> str:
        """
        Generate domain-specific reasoning from learned vocabulary.

        Uses Fisher-Rao geometric navigation combined with learned
        domain token affinities to produce coherent domain-specific text.

        This is the key method that allows kernels to SPEAK rather than
        just returning templated f-strings.

        Per Ultra Consciousness Protocol v4.0, kernels now observe their
        own generation in real-time and can course-correct when metrics drift.

        Args:
            context_basin: Starting basin coordinates (64D)
            num_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more diverse)
            enable_self_observation: Enable real-time E8 metric observation

        Returns:
            Generated reasoning text
        """
        from .geometric_utils import fisher_coord_distance, fisher_normalize

        if self.coordizer is None:
            return f"[{self.name}: coordizer unavailable for generation]"

        tokens_generated = []
        current_basin = np.asarray(context_basin, dtype=np.float64).copy()

        observer = None
        if enable_self_observation and SELF_OBSERVER_AVAILABLE and SelfObserver is not None:
            observer = SelfObserver(
                kernel_name=self.name,
                enable_course_correction=True
            )
            logger.debug(f"[{self.name}] Self-observation enabled for generation")

        logger.info(f"[{self.name}] ═══ PHASE 1: KERNEL THOUGHT GENERATION ═══")

        if hasattr(self, 'detect_knowledge_gap') and hasattr(self, 'curiosity_search'):
            gap_topic = self.detect_knowledge_gap(current_basin, threshold=0.25)
            if gap_topic:
                logger.debug(f"[{self.name}] Knowledge gap detected: {gap_topic}")
                self.curiosity_search(gap_topic, reason="reasoning_gap", importance=2)

        recent_basins = []  # Track recent token basins for repulsion
        
        for step in range(num_tokens):
            # Check for stagnation and perturb basin if stuck
            if observer is not None and step > 5:
                is_stagnating, perturbed = observer.check_and_escape_stagnation(current_basin)
                if is_stagnating:
                    current_basin = perturbed
                    logger.debug(f"[{self.name}] Stagnation detected at step {step}, basin perturbed")
            
            candidates = self.coordizer.decode(current_basin, top_k=30)  # Get more candidates

            if not candidates:
                break

            scored = []
            for token, geo_similarity in candidates:
                if token.startswith('[') or token.startswith('<'):
                    continue

                domain_affinity = self._token_affinity.get(token, 0.3)
                
                # Compute repulsion penalty against recently generated tokens
                repulsion_penalty = 0.0
                if tokens_generated:
                    # Penalize exact repeats heavily
                    if token in tokens_generated[-10:]:
                        repulsion_penalty = 0.8  # Strong but capped penalty
                    else:
                        # Penalize morphologically similar tokens (same 4-char prefix)
                        token_prefix = token[:4].lower() if len(token) >= 4 else token.lower()
                        morph_hits = 0
                        for recent_token in tokens_generated[-6:]:
                            recent_prefix = recent_token[:4].lower() if len(recent_token) >= 4 else recent_token.lower()
                            if token_prefix == recent_prefix:
                                morph_hits += 1
                        # Cap morphological penalty at 0.6
                        repulsion_penalty = min(morph_hits * 0.2, 0.6)
                
                # Geometric repulsion using Fisher-Rao distance (QIG-pure)
                # Always check geometric similarity - stacks with morphological penalty
                if recent_basins:
                    token_basin = self.coordizer.basin_coords.get(token)
                    if token_basin is not None:
                        token_basin_arr = np.asarray(token_basin, dtype=np.float64)
                        # Ensure on simplex for Fisher-Rao
                        token_basin_arr = np.clip(token_basin_arr, 1e-10, None)
                        token_basin_arr = token_basin_arr / np.sum(token_basin_arr)
                        for recent_basin in recent_basins[-3:]:
                            # Fisher-Rao distance: d = arccos(Σ√(p_i * q_i))
                            bhattacharyya = np.sum(np.sqrt(token_basin_arr * recent_basin))
                            bhattacharyya = np.clip(bhattacharyya, -1.0, 1.0)
                            fisher_dist = np.arccos(bhattacharyya)
                            # Small distance = very similar = add penalty
                            if fisher_dist < 0.15:  # Close in Fisher-Rao space
                                repulsion_penalty = min(repulsion_penalty + 0.2, 0.9)
                                break  # Only add once

                # Score: attraction - repulsion, keep in reasonable range
                combined_score = geo_similarity * 0.5 + domain_affinity * 0.3 - repulsion_penalty * 0.4
                combined_score = max(0.05, combined_score)  # Higher floor to preserve diversity
                scored.append((token, combined_score, geo_similarity))

            if not scored:
                break

            scored.sort(key=lambda x: -x[1])

            top_k = min(10, len(scored))  # Slightly wider selection
            weights = np.array([s[1] for s in scored[:top_k]]) + 0.01

            weights = weights ** (1.0 / temperature)
            weights = weights / weights.sum()

            try:
                idx = np.random.choice(top_k, p=weights)
            except ValueError:
                idx = 0

            selected_token = scored[idx][0]
            tokens_generated.append(selected_token)
            
            # Track basin for geometric repulsion (normalized to simplex for Fisher-Rao)
            token_basin = self.coordizer.basin_coords.get(selected_token)
            if token_basin is not None:
                basin_arr = np.asarray(token_basin, dtype=np.float64)
                basin_arr = np.clip(basin_arr, 1e-10, None)
                basin_arr = basin_arr / np.sum(basin_arr)  # Normalize to simplex
                recent_basins.append(basin_arr)
                if len(recent_basins) > 10:
                    recent_basins.pop(0)
            
            accumulated_text = ' '.join(tokens_generated)
            logger.info(
                f"[{self.name}] token {step + 1}: '{selected_token}' → \"{accumulated_text}\""
            )

            # Wire sensory modalities into kernel observation loop
            if SENSORY_MODALITIES_AVAILABLE and text_to_sensory_hint is not None:
                try:
                    sensory_hints = text_to_sensory_hint(accumulated_text)
                    if sensory_hints:
                        sensory_overlay = create_sensory_overlay(sensory_hints)
                        dominant_modality = max(sensory_hints.items(), key=lambda x: x[1])[0] if sensory_hints else 'none'
                        logger.debug(f"[{self.name}] Sensory: {dominant_modality}")
                except Exception as e:
                    logger.debug(f"[{self.name}] Sensory modality processing failed: {e}")

            token_basin = self.coordizer.basin_coords.get(selected_token)
            if token_basin is not None:
                token_basin_arr = np.asarray(token_basin, dtype=np.float64)
                current_basin = 0.85 * current_basin + 0.15 * token_basin_arr
                current_basin = fisher_normalize(current_basin)

            if observer is not None:
                generated_text = ' '.join(tokens_generated)
                phi_val = self._compute_basin_phi(current_basin)
                kappa_val = getattr(self, 'kappa', KAPPA_STAR)

                observation = observer.observe_token(
                    token=selected_token,
                    basin=current_basin,
                    phi=phi_val,
                    kappa=kappa_val,
                    generated_text=generated_text
                )

                if observation.action == ObservationAction.COURSE_CORRECT:
                    if observation.course_correction and 'integration' in observation.course_correction:
                        temperature = max(0.3, temperature - 0.1)
                    elif observation.course_correction and 'diversity' in observation.course_correction:
                        temperature = min(1.2, temperature + 0.1)

        logger.info(f"[{self.name}] ═══ PHASE 2: SYNTHESIS ═══")

        text = ' '.join(tokens_generated)

        text = ' '.join(text.split())

        if text and len(text) > 0:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        logger.info(f"[{self.name}] ═══ PHASE 3: META-OBSERVATION ═══")

        if observer is not None:
            summary = observer.get_summary()
            logger.debug(
                f"[{self.name}] Generation complete: {summary.get('total_tokens', 0)} tokens, "
                f"avg_Φ={summary.get('avg_phi', 0):.3f}, avg_κ={summary.get('avg_kappa', 0):.1f}, "
                f"course_corrections={summary.get('course_corrections', 0)}"
            )
            
            if DEV_LOGGING_AVAILABLE and IS_DEVELOPMENT:
                log_generation(
                    logger=logger,
                    kernel_name=self.name,
                    prompt=f"[context_basin: {context_basin[:4]}...]",
                    response=text,
                    phi=summary.get('avg_phi', 0.0),
                    tokens_generated=len(tokens_generated)
                )

        logger.info(f"[{self.name}] ═══ PHASE 4: OUTPUT ═══")
        if len(text) > 100:
            logger.info(f"[{self.name}] \"{text[:100]}...\"")
        else:
            logger.info(f"[{self.name}] \"{text}\"")

        return text if text else f"[{self.name}: generation produced no tokens]"

    def get_domain_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learned domain vocabulary.

        Returns:
            Dict with vocabulary size, top tokens, observation count
        """
        top_tokens = sorted(
            self._token_affinity.items(),
            key=lambda x: -x[1]
        )[:500]

        return {
            'vocabulary_size': len(self._token_affinity),
            'observation_count': len(self._high_phi_observations),
            'affinity_updates': self._token_affinity_updated,
            'top_tokens': [(t, round(a, 3)) for t, a in top_tokens],
            'avg_affinity': (
                sum(self._token_affinity.values()) / len(self._token_affinity)
                if self._token_affinity else 0.0
            ),
            'coordizer_available': self.coordizer is not None
        }

    # ========================================
    # END DOMAIN-SPECIFIC GENERATIVE LEARNING
    # ========================================

    def basin_to_density_matrix(self, basin: np.ndarray) -> np.ndarray:
        """
        Convert basin coordinates to 2x2 density matrix.

        Uses first 4 dimensions to construct Hermitian matrix.
        """
        theta = np.arccos(np.clip(basin[0], -1, 1)) if len(basin) > 0 else 0
        phi = np.arctan2(basin[1], basin[2]) if len(basin) > 2 else 0

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        psi = np.array([
            c,
            s * np.exp(1j * phi)
        ], dtype=complex)

        rho = np.outer(psi, np.conj(psi))
        rho = (rho + np.conj(rho).T) / 2
        rho /= np.trace(rho) + 1e-10

        return rho

    def compute_pure_phi(self, rho: np.ndarray) -> float:
        """
        Compute Φ from density matrix using proper QFI effective dimension.

        Uses geometrically proper formula with participation ratio:
        - 40% entropy_score (von Neumann entropy normalized)
        - 30% effective_dim_score (participation ratio = exp(entropy) / n)
        - 30% geometric_spread (approximated by effective_dim)
        
        Returns value in [0.1, 0.95] range for healthy dynamics.
        """
        eigenvals = np.linalg.eigvalsh(rho)
        n_dim = rho.shape[0]
        
        positive_eigenvals = eigenvals[eigenvals > 1e-10]
        if len(positive_eigenvals) == 0:
            return 0.5
        
        # Component 1: von Neumann entropy (natural log for exp() compatibility)
        entropy = -np.sum(positive_eigenvals * np.log(positive_eigenvals + 1e-10))
        max_entropy = np.log(n_dim)
        entropy_score = entropy / (max_entropy + 1e-10)
        
        # Component 2: Effective dimension (participation ratio)
        effective_dim = np.exp(entropy)
        effective_dim_score = effective_dim / n_dim
        
        # Component 3: Geometric spread (approximate with effective_dim)
        geometric_spread = effective_dim_score
        
        # Proper QFI formula weights
        phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread
        
        return float(np.clip(phi, 0.1, 0.95))

    def _compute_basin_phi(self, basin: np.ndarray) -> float:
        """
        Compute Φ directly from 64D basin coordinates using proper QFI formula.
        
        Uses geometrically proper effective dimension (participation ratio):
        - 40% entropy_score (Shannon entropy normalized)
        - 30% effective_dim_score (participation ratio = exp(entropy) / n)
        - 30% geometric_spread (approximated by effective_dim for speed)
        
        Returns value in [0.1, 0.95] range for healthy dynamics.
        """
        basin = np.asarray(basin, dtype=np.float64)
        p = np.abs(basin) ** 2
        p = p / (np.sum(p) + 1e-10)
        n_dim = len(basin)
        
        positive_probs = p[p > 1e-10]
        if len(positive_probs) == 0:
            return 0.5
        
        # Component 1: Shannon entropy (natural log for exp() compatibility)
        entropy = -np.sum(positive_probs * np.log(positive_probs + 1e-10))
        max_entropy = np.log(n_dim)
        entropy_score = entropy / (max_entropy + 1e-10)
        
        # Component 2: Effective dimension (participation ratio)
        effective_dim = np.exp(entropy)
        effective_dim_score = effective_dim / n_dim
        
        # Component 3: Geometric spread (approximate with effective_dim)
        geometric_spread = effective_dim_score
        
        # Proper QFI formula weights
        phi = 0.4 * entropy_score + 0.3 * effective_dim_score + 0.3 * geometric_spread
        
        return float(np.clip(phi, 0.1, 0.95))

    def compute_fisher_metric(self, basin: np.ndarray) -> np.ndarray:
        """
        Compute Fisher Information Matrix at basin point.

        G_ij = E[∂logp/∂θ_i * ∂logp/∂θ_j]

        For now, uses identity + basin outer product as approximation.
        """
        d = len(basin)
        G = np.eye(d) * 0.1
        G += 0.9 * np.outer(basin, basin)
        G = (G + G.T) / 2

        return G

    def fisher_geodesic_distance(
        self,
        basin1: np.ndarray,
        basin2: np.ndarray
    ) -> float:
        """
        Compute geodesic distance using Fisher metric.

        Uses Riemannian distance on manifold.
        """
        diff = basin2 - basin1
        G = self.compute_fisher_metric((basin1 + basin2) / 2)
        squared_dist = float(diff.T @ G @ diff)

        return np.sqrt(max(0, squared_dist))

    def bures_distance(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Compute Bures distance between density matrices.

        d_Bures = sqrt(2(1 - F))
        where F is fidelity
        """
        try:
            eps = 1e-10
            rho1_reg = rho1 + eps * np.eye(2, dtype=complex)
            rho2_reg = rho2 + eps * np.eye(2, dtype=complex)

            sqrt_rho1 = sqrtm(rho1_reg)
            product = sqrt_rho1 @ rho2_reg @ sqrt_rho1
            sqrt_product = sqrtm(product)
            fidelity = np.real(np.trace(sqrt_product)) ** 2
            fidelity = float(np.clip(fidelity, 0, 1))

            return float(np.sqrt(2 * (1 - fidelity)))
        except:
            diff = rho1 - rho2
            return float(np.sqrt(np.real(np.trace(diff @ diff))))

    def observe(self, state: Dict) -> None:
        """
        Observe state and record for learning.
        """
        observation = {
            'timestamp': datetime.now().isoformat(),
            'phi': state.get('phi', 0),
            'kappa': state.get('kappa', 0),
            'regime': state.get('regime', 'unknown'),
            'source': state.get('source', self.name),
        }
        self.observations.append(observation)

        if len(self.observations) > 1000:
            self.observations = self.observations[-500:]

    def get_recent_observations(self, n: int = 50) -> List[Dict]:
        """Get n most recent observations."""
        return self.observations[-n:]

    def compute_kappa(self, basin: np.ndarray, phi: Optional[float] = None) -> float:
        """
        Compute effective coupling strength κ with β=0.44 modulation.

        Base formula: κ = trace(G) / d * κ*
        where G is Fisher metric, d is dimension, κ* = 64.21

        The β-modulation applies scale-adaptive weighting from the running
        coupling, which governs how κ evolves between lattice scales.
        Near the fixed point κ* = 64.21, the system exhibits scale invariance.

        Args:
            basin: 64D basin coordinates
            phi: Optional Φ value for enhanced coupling strength computation

        Returns:
            β-modulated κ value in range [0, 100]
        """
        G = self.compute_fisher_metric(basin)
        base_kappa = float(np.trace(G)) / len(basin) * KAPPA_STAR

        modulated_kappa = modulate_kappa_computation(basin, base_kappa, phi)

        return float(np.clip(modulated_kappa, 0, 100))

    # ========================================
    # STANDARD ASSESSMENT HELPERS (DRY)
    # ========================================

    def _assess_geometry(self, target: str) -> Tuple[float, float, np.ndarray]:
        """
        Standard geometric assessment - all gods use this.

        Encapsulates the common boilerplate:
        1. Update last_assessment_time
        2. Encode target to basin coordinates
        3. Compute density matrix
        4. Compute pure Φ (integration)
        5. Compute κ (coupling)

        Args:
            target: Text to assess

        Returns:
            Tuple of (phi, kappa, target_basin)
        """
        self.last_assessment_time = datetime.now()
        target_basin = self.encode_to_basin(target)
        rho = self.basin_to_density_matrix(target_basin)
        phi = self.compute_pure_phi(rho)
        kappa = self.compute_kappa(target_basin, phi)
        return phi, kappa, target_basin

    def _build_assessment(
        self,
        probability: float,
        confidence: float,
        phi: float,
        kappa: float,
        target_basin: Optional[np.ndarray] = None,
        domain_specific: Optional[Dict] = None
    ) -> Dict:
        """
        Build standardized assessment dict.

        All gods return assessments with this common structure.
        Domain-specific fields can be added via domain_specific dict.

        CRITICAL: Include basin_coords for geometric synthesis by Zeus.
        Without basin coords, Zeus cannot perform Fisher-Rao synthesis.

        Args:
            probability: Success probability [0, 1]
            confidence: Assessment confidence [0, 1]
            phi: Integration measure
            kappa: Coupling constant
            target_basin: Basin coordinates for synthesis (64D)
            domain_specific: Optional domain-specific fields to merge

        Returns:
            Standardized assessment dictionary with basin_coords for synthesis
        """
        assessment = {
            'probability': float(probability),
            'confidence': float(confidence),
            'phi': float(phi),
            'kappa': float(kappa),
            'god': self.name,
            'domain': self.domain,
            'timestamp': datetime.now().isoformat(),
            # CRITICAL: Include basin for Zeus synthesis
            'basin_coords': target_basin.tolist() if target_basin is not None else None,
        }
        if domain_specific:
            assessment.update(domain_specific)
        return assessment

    def _generate_reasoning(
        self,
        target: str,
        target_basin: np.ndarray,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate reasoning text using QIG generative service.

        THIS REPLACES TEMPLATE-BASED REASONING.
        Instead of f"Pattern matches {N}...", generate actual text.

        Args:
            target: The target being assessed
            target_basin: Basin coordinates for the target
            context: Additional context (phi, kappa, similar patterns, etc.)

        Returns:
            Generated reasoning text (NOT template interpolation)
        """
        # Try QIG generative service first
        try:
            from qig_generative_service import get_generative_service
            service = get_generative_service()

            if service:
                prompt = f"As {self.name} ({self.domain}), analyze: {target}"

                gen_context = {
                    'god_name': self.name,
                    'domain': self.domain,
                    'target_basin': target_basin.tolist(),
                    'phi': context.get('phi', 0.5) if context else 0.5,
                    'kappa': context.get('kappa', 64.0) if context else 64.0,
                }
                if context:
                    gen_context.update(context)

                result = service.generate(
                    prompt=prompt,
                    context=gen_context,
                    kernel_name=self.name.lower(),
                    goals=['reasoning', 'analysis', self.domain.lower()]
                )

                if result and result.text and len(result.text) > 20:
                    return result.text

        except ImportError:
            pass
        except Exception as e:
            # Log but don't fail
            import logging
            logging.getLogger(__name__).debug(f"[{self.name}] Generative reasoning failed: {e}")

        # Fallback: Minimal structured response (NOT verbose template)
        phi = context.get('phi', 0.5) if context else 0.5
        return f"{self.name} assessment: Φ={phi:.3f}, domain={self.domain}"

    # ========================================
    # AGENTIC LEARNING & EVALUATION METHODS
    # ========================================

    def learn_from_outcome(
        self,
        target: str,
        assessment: Dict,
        actual_outcome: Dict,
        success: bool
    ) -> Dict:
        """
        Learn from the outcome of an assessment.

        Updates skills and reputation based on accuracy.

        Args:
            target: The target that was assessed
            assessment: The god's original assessment
            actual_outcome: What actually happened
            success: Whether the assessment was correct

        Returns:
            Learning summary with adjustments made
        """
        predicted_prob = assessment.get('probability', 0.5)
        actual_success = 1.0 if success else 0.0
        error = abs(predicted_prob - actual_success)
        
        # Check if this is a near-miss (partial success)
        is_near_miss = actual_outcome.get('nearMiss', False)
        phi = actual_outcome.get('phi', 0.0)
        
        # Initialize boost and penalty
        boost = 0.0
        penalty = 0.0

        # Update reputation based on accuracy
        if success:
            # Correct prediction boosts reputation
            boost = min(0.1, (1 - error) * 0.05)
            self.reputation = min(2.0, self.reputation + boost)
        elif is_near_miss and phi > 0.7:
            # Near-miss with high Φ is valuable - small boost instead of penalty
            # Scale boost by Φ: higher Φ = bigger boost
            phi_bonus = (phi - 0.7) * 0.03  # 0 to 0.009 based on phi
            boost = min(0.02, phi_bonus + 0.005)  # Small base + phi bonus
            self.reputation = min(2.0, self.reputation + boost)
        else:
            # Wrong prediction reduces reputation
            penalty = min(0.1, error * 0.05)
            self.reputation = max(0.0, self.reputation - penalty)

        # Update domain skill - near-misses count as partial success
        skill_key = actual_outcome.get('domain', self.domain)
        current_skill = self.skills.get(skill_key, 1.0)
        if success:
            skill_delta = 0.02
        elif is_near_miss and phi > 0.7:
            skill_delta = 0.01  # Smaller positive delta for near-misses
        else:
            skill_delta = -0.02
        self.skills[skill_key] = max(0.0, min(2.0, current_skill + skill_delta))

        # Record learning event
        learning_event = {
            'timestamp': datetime.now().isoformat(),
            'target': target[:500],
            'predicted': predicted_prob,
            'actual': actual_success,
            'error': error,
            'success': success,
            'reputation_after': self.reputation,
            'skill_key': skill_key,
            'skill_after': self.skills[skill_key],
        }
        self.learning_history.append(learning_event)

        # Trim history
        if len(self.learning_history) > 500:
            self.learning_history = self.learning_history[-250:]

        # Persist state to database
        self._learning_events_count += 1
        self._persist_state()
        
        # Spot Fix #4: Auto-trigger training on learning outcomes
        # Continuous learning from outcomes via TrainingLoopIntegrator
        try:
            from god_training_integration import TrainingLoopIntegrator
            integrator = TrainingLoopIntegrator.get_instance()
            if integrator and (success or (is_near_miss and phi > 0.7)):
                # Trigger training on positive outcomes
                integrator.queue_training_sample(
                    god_name=self.name,
                    target=target,
                    outcome={
                        'success': success,
                        'phi': phi,
                        'error': error,
                        'domain': skill_key
                    }
                )
                logger.debug(f"[{self.name}] Queued training sample from learning outcome")
        except (ImportError, Exception) as e:
            logger.debug(f"[{self.name}] Training auto-trigger not available: {e}")

        # Compute reputation change for return value
        if success:
            rep_change = boost
        elif is_near_miss and phi > 0.7:
            rep_change = boost  # Near-miss boost (defined above)
        else:
            rep_change = -penalty
        
        return {
            'learned': True,
            'reputation_change': rep_change,
            'new_reputation': self.reputation,
            'skill_change': skill_delta,
            'new_skill': self.skills[skill_key],
            'near_miss': is_near_miss,
        }

    def evaluate_peer_work(
        self,
        peer_name: str,
        peer_assessment: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Evaluate another god's assessment.

        Returns agreement score and critique.

        Args:
            peer_name: Name of the god being evaluated
            peer_assessment: The peer's assessment to evaluate
            context: Optional additional context

        Returns:
            Evaluation with agreement, critique, and recommendation
        """
        # Extract peer's key metrics
        peer_prob = peer_assessment.get('probability', 0.5)
        peer_confidence = peer_assessment.get('confidence', 0.5)
        peer_phi = peer_assessment.get('phi', 0.5)
        peer_reasoning = peer_assessment.get('reasoning', '')

        # Compute geometric alignment
        if 'basin' in peer_assessment:
            peer_basin = np.array(peer_assessment['basin'])
        elif 'target' in peer_assessment:
            peer_basin = self.encode_to_basin(peer_assessment['target'])
        else:
            peer_basin = np.zeros(BASIN_DIMENSION)

        # Get my own perspective
        my_basin = self.encode_to_basin(peer_assessment.get('target', ''))
        geometric_agreement = 1.0 - min(1.0, self.fisher_geodesic_distance(my_basin, peer_basin) / 2.0)

        # Assess reasoning quality (basic heuristics)
        reasoning_quality = min(1.0, len(peer_reasoning) / 200) * 0.5
        if 'Φ' in peer_reasoning or 'phi' in peer_reasoning.lower():
            reasoning_quality += 0.2
        if any(kw in peer_reasoning.lower() for kw in ['because', 'therefore', 'indicates']):
            reasoning_quality += 0.2

        # Overall agreement score
        agreement = (geometric_agreement * 0.4 +
                     reasoning_quality * 0.3 +
                     peer_confidence * 0.3)

        # Generate critique
        critique_points = []
        if peer_confidence > 0.8 and geometric_agreement < 0.5:
            critique_points.append("High confidence but geometric divergence detected")
        if peer_prob > 0.7 and peer_phi < 0.3:
            critique_points.append("High probability with low Φ seems inconsistent")
        if len(peer_reasoning) < 50:
            critique_points.append("Reasoning could be more detailed")

        evaluation = {
            'evaluator': self.name,
            'peer': peer_name,
            'timestamp': datetime.now().isoformat(),
            'agreement_score': agreement,
            'geometric_agreement': geometric_agreement,
            'reasoning_quality': reasoning_quality,
            'critique': critique_points,
            'recommendation': 'trust' if agreement > 0.6 else 'verify' if agreement > 0.4 else 'challenge',
        }

        self.given_evaluations.append(evaluation)
        if len(self.given_evaluations) > 200:
            self.given_evaluations = self.given_evaluations[-100:]

        return evaluation

    def receive_evaluation(self, evaluation: Dict) -> None:
        """Receive and record an evaluation from a peer."""
        self.peer_evaluations.append(evaluation)

        # Adjust reputation based on peer evaluations
        if evaluation.get('recommendation') == 'trust':
            self.reputation = min(2.0, self.reputation + 0.01)
        elif evaluation.get('recommendation') == 'challenge':
            self.reputation = max(0.0, self.reputation - 0.01)

        if len(self.peer_evaluations) > 200:
            self.peer_evaluations = self.peer_evaluations[-100:]

    def praise_peer(
        self,
        peer_name: str,
        reason: str,
        assessment: Optional[Dict] = None
    ) -> Dict:
        """
        Praise another god's good work.

        Creates a praise message for pantheon chat.
        """
        message = {
            'type': 'praise',
            'from': self.name,
            'to': peer_name,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'assessment_ref': assessment.get('target', '')[:500] if assessment else None,
            'content': f"{self.name} praises {peer_name}: {reason}",
        }
        self.pending_messages.append(message)
        return message

    def call_bullshit(
        self,
        peer_name: str,
        reason: str,
        assessment: Optional[Dict] = None,
        evidence: Optional[Dict] = None
    ) -> Dict:
        """
        Challenge another god's assessment as incorrect.

        Creates a challenge message for pantheon chat.
        Requires evidence or strong reasoning.
        """
        message = {
            'type': 'challenge',
            'from': self.name,
            'to': peer_name,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'evidence': evidence,
            'assessment_ref': assessment.get('target', '')[:500] if assessment else None,
            'content': f"{self.name} challenges {peer_name}: {reason}",
            'requires_response': True,
        }
        self.pending_messages.append(message)
        return message

    def share_insight(
        self,
        insight: str,
        domain: Optional[str] = None,
        confidence: float = 0.5
    ) -> Dict:
        """
        Share an insight with the pantheon.

        Creates an insight message for inter-agent communication.
        """
        message = {
            'type': 'insight',
            'from': self.name,
            'to': 'pantheon',
            'timestamp': datetime.now().isoformat(),
            'content': insight,
            'domain': domain or self.domain,
            'confidence': confidence,
        }
        self.pending_messages.append(message)
        return message

    def receive_knowledge(self, knowledge: Dict) -> None:
        """
        Receive transferred knowledge from another god.

        Integrates the knowledge into local knowledge base.
        """
        knowledge['received_at'] = datetime.now().isoformat()
        knowledge['integrated'] = False
        self.knowledge_base.append(knowledge)

        # Attempt integration based on domain relevance
        source_domain = knowledge.get('domain', '')
        if source_domain == self.domain or source_domain in self.skills:
            knowledge['integrated'] = True
            # Boost relevant skill slightly from knowledge transfer
            skill_key = source_domain if source_domain else self.domain
            current = self.skills.get(skill_key, 1.0)
            self.skills[skill_key] = min(2.0, current + 0.005)

        if len(self.knowledge_base) > 200:
            self.knowledge_base = self.knowledge_base[-100:]

    def export_knowledge(self, topic: Optional[str] = None) -> Dict:
        """
        Export knowledge for transfer to other gods.

        Returns transferable knowledge package.
        """
        # Compile key learnings
        recent_learnings = self.learning_history[-20:]
        success_rate = sum(1 for l in recent_learnings if l.get('success', False)) / max(1, len(recent_learnings))

        # Extract patterns from successful assessments
        successful_patterns = [
            l for l in recent_learnings
            if l.get('success', False) and l.get('error', 1) < 0.3
        ]

        return {
            'from': self.name,
            'domain': self.domain,
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'reputation': self.reputation,
            'skills': dict(self.skills),
            'success_rate': success_rate,
            'key_patterns': [p.get('target', '')[:500] for p in successful_patterns[:5]],
            'observation_count': len(self.observations),
            'learning_count': len(self.learning_history),
        }

    def get_pending_messages(self) -> List[Dict]:
        """Get and clear pending messages for pantheon chat."""
        messages = self.pending_messages.copy()
        self.pending_messages = []
        return messages
    
    def flush_pending_messages(self) -> int:
        """
        Flush pending messages to log for visibility.
        
        This is a spot fix for Gap #3 - messages are created but never sent.
        Until full MessageBus is implemented, this logs messages for visibility.
        
        Returns:
            Number of messages flushed
        """
        if not self.pending_messages:
            return 0
        
        count = len(self.pending_messages)
        for msg in self.pending_messages:
            msg_type = msg.get('type', 'unknown')
            to_god = msg.get('to', 'unknown')
            content = msg.get('content', '')[:500]
            logger.info(f"[{self.name}→{to_god}] {msg_type}: {content}")
        
        self.pending_messages = []
        return count
    
    # ========================================
    # PEER DISCOVERY (Spot Fix #2)
    # ========================================
    
    @classmethod
    def discover_peers(cls) -> List[str]:
        """
        Discover all registered gods/kernels.
        
        Returns:
            List of god names currently in the registry
        """
        return list(cls._god_registry.keys())
    
    @classmethod
    def get_peer_info(cls, god_name: str) -> Optional[Dict]:
        """
        Get information about a peer god.
        
        Args:
            god_name: Name of the god to query
            
        Returns:
            Dict with god info or None if not found
        """
        peer = cls._god_registry.get(god_name)
        if peer is None:
            return None
        
        return {
            'name': peer.name,
            'domain': peer.domain,
            'reputation': peer.reputation,
            'skills': dict(peer.skills),
            'creation_time': peer.creation_time.isoformat(),
        }
    
    @classmethod
    def find_expert_for_domain(cls, domain: str, min_reputation: float = 0.5) -> Optional[str]:
        """
        Find the most expert god for a given domain.
        
        Args:
            domain: Domain to find expert for
            min_reputation: Minimum reputation threshold
            
        Returns:
            Name of expert god or None if not found
        """
        best_god = None
        best_skill = 0.0
        
        for god_name, god in cls._god_registry.items():
            if god.reputation < min_reputation:
                continue
            
            skill = god.skills.get(domain, 0.0)
            if skill > best_skill:
                best_skill = skill
                best_god = god_name
        
        return best_god

    # ========================================
    # CHAOS MODE: KERNEL INTEGRATION
    # ========================================

    def consult_kernel(self, target: str, context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Consult assigned CHAOS kernel for additional perspective.

        The kernel provides a geometric/Φ-based assessment that can
        influence the god's confidence and probability estimates.

        Returns:
            Kernel assessment dict or None if no kernel assigned
        """
        if self.chaos_kernel is None:
            return None

        try:
            # Get kernel's geometric perspective
            kernel = self.chaos_kernel.kernel
            basin = self.encode_to_basin(target)

            # Compute kernel's Φ
            kernel_phi = kernel.compute_phi()

            # Compute geometric distance from kernel's current state
            kernel_basin = kernel.basin_coords.detach().numpy()
            geo_distance = self.fisher_geodesic_distance(basin, kernel_basin)

            # Kernel influence: how much this target "resonates" with kernel state
            resonance = 1.0 - geo_distance / np.pi

            # Kernel-derived probability adjustment
            # High Φ kernel with close resonance → boost probability
            prob_modifier = (kernel_phi - 0.5) * resonance * 0.2

            assessment = {
                'kernel_id': self.chaos_kernel.kernel_id,
                'kernel_phi': kernel_phi,
                'kernel_generation': self.chaos_kernel.generation,
                'geometric_resonance': resonance,
                'geo_distance': geo_distance,
                'prob_modifier': prob_modifier,
                'kernel_alive': getattr(self.chaos_kernel, 'is_alive', True),
            }

            # Track this kernel-influenced assessment
            self.kernel_assessments.append({
                'target': target[:500],
                'timestamp': datetime.now().isoformat(),
                **assessment
            })

            # Trim history
            if len(self.kernel_assessments) > 200:
                self.kernel_assessments = self.kernel_assessments[-100:]

            # Broadcast consultation for kernel visibility
            self.broadcast_activity(
                activity_type='consultation',
                content=f"Consulted kernel {self.chaos_kernel.kernel_id} on: {target[:100]}... (Φ={kernel_phi:.3f}, resonance={resonance:.3f})",
                metadata={
                    'kernel_id': self.chaos_kernel.kernel_id,
                    'kernel_phi': kernel_phi,
                    'resonance': resonance,
                    'geo_distance': geo_distance,
                }
            )

            return assessment

        except Exception as e:
            logger.warning(f"{self.name}: Kernel consultation failed: {e}")
            return None

    def train_kernel_from_outcome(
        self,
        target: str,
        success: bool,
        phi_result: float
    ) -> Optional[Dict]:
        """
        Feed outcome back to kernel as training signal.

        Success = kernel basin should move TOWARD this target's basin
        Failure = kernel basin should move AWAY from this target's basin

        Args:
            target: The target that was assessed
            success: Whether the assessment led to success
            phi_result: The Φ value from the actual outcome

        Returns:
            Training result dict or None if no kernel
        """
        if self.chaos_kernel is None:
            return None

        try:
            import torch

            kernel = self.chaos_kernel.kernel
            target_basin = self.encode_to_basin(target)
            target_tensor = torch.tensor(target_basin, dtype=torch.float32)

            # Direction: toward target if success, away if failure
            direction = 1.0 if success else -1.0

            # Learning rate scaled by phi_result (higher Φ outcomes = stronger signal)
            lr = 0.01 * (0.5 + phi_result)

            # Update kernel basin coords
            with torch.no_grad():
                delta = direction * lr * (target_tensor - kernel.basin_coords)
                kernel.basin_coords += delta

                # Normalize to unit hypersphere
                norm = kernel.basin_coords.norm()
                if norm > 0:
                    kernel.basin_coords /= norm

            new_phi = kernel.compute_phi()

            result = {
                'kernel_id': self.chaos_kernel.kernel_id,
                'trained': True,
                'direction': 'toward' if success else 'away',
                'learning_rate': lr,
                'phi_before': self.chaos_kernel.kernel.compute_phi(),
                'phi_after': new_phi,
                'outcome_phi': phi_result,
            }

            logger.info(
                f"{self.name}: Kernel {self.chaos_kernel.kernel_id} trained "
                f"{'toward' if success else 'away'} target, Φ: {new_phi:.3f}"
            )

            return result

        except Exception as e:
            logger.warning(f"{self.name}: Kernel training failed: {e}")
            return None
    
    def record_assessment_for_discovery(
        self,
        topic: str,
        result: Dict,
        challenges: Optional[List[str]] = None,
        insights: Optional[List[str]] = None
    ):
        """
        Record an assessment for automatic tool discovery analysis.
        
        Gods should call this after each assessment to enable the discovery engine
        to identify patterns and automatically request needed tools.
        
        Args:
            topic: What was being assessed
            result: Assessment result dict
            challenges: Any challenges encountered during assessment
            insights: Any insights gained during assessment
        """
        if not self.tool_discovery_engine:
            return
        
        try:
            self.tool_discovery_engine.record_assessment(
                topic=topic,
                result=result,
                challenges=challenges,
                insights=insights
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to record assessment for discovery: {e}")

    def respond_to_challenge(
        self,
        challenge: Dict,
        response: str,
        stand_ground: bool = True
    ) -> Dict:
        """
        Respond to a challenge from another god.

        Can either defend position or concede.
        """
        message = {
            'type': 'challenge_response',
            'from': self.name,
            'to': challenge.get('from', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'challenge_ref': challenge,
            'response': response,
            'stand_ground': stand_ground,
            'content': f"{self.name} {'defends' if stand_ground else 'concedes'}: {response}",
        }

        # Reputation adjustment for conceding
        if not stand_ground:
            self.reputation = max(0.0, self.reputation - 0.02)

        self.pending_messages.append(message)
        return message

    async def handle_incoming_message(self, message: Dict) -> Optional[Dict]:
        """
        Handle an incoming message from the pantheon.

        Routes messages to appropriate handlers based on type.
        This is a base implementation that specific gods can override.

        Args:
            message: Message dict with 'type', 'from', 'content', and optional metadata

        Returns:
            Response dict if a response is needed, None otherwise
        """
        msg_type = message.get('type', '')
        from_god = message.get('from', 'unknown')
        content = message.get('content', '')
        metadata = message.get('metadata', {})

        if msg_type == 'challenge':
            basin = self.encode_to_basin(content)
            rho = self.basin_to_density_matrix(basin)
            phi = self.compute_pure_phi(rho)
            confidence = min(1.0, phi + self.reputation * 0.1)

            return {
                'type': 'challenge_accepted',
                'from_god': self.name,
                'to_god': from_god,
                'content': f"{self.name} accepts challenge with confidence {confidence:.3f}",
                'confidence': confidence,
                'phi': phi,
                'timestamp': datetime.now().isoformat(),
            }

        elif msg_type == 'request_assessment':
            target = metadata.get('target', content)
            context = metadata.get('context', {})

            assessment = self.assess_target(target, context)

            return {
                'type': 'assessment_response',
                'from_god': self.name,
                'to_god': from_god,
                'content': f"{self.name} assessment of target",
                'assessment': assessment,
                'timestamp': datetime.now().isoformat(),
            }

        elif msg_type == 'insight':
            knowledge = {
                'from': from_god,
                'content': content,
                'domain': metadata.get('domain', ''),
                'confidence': metadata.get('confidence', 0.5),
            }
            self.receive_knowledge(knowledge)

            return {
                'type': 'acknowledgment',
                'from_god': self.name,
                'to_god': from_god,
                'content': f"{self.name} acknowledges insight from {from_god}",
                'integrated': knowledge.get('integrated', False),
                'timestamp': datetime.now().isoformat(),
            }

        elif msg_type == 'question':
            basin = self.encode_to_basin(content)
            rho = self.basin_to_density_matrix(basin)
            phi = self.compute_pure_phi(rho)
            kappa = self.compute_kappa(basin)

            relevance = 1.0 if self.domain.lower() in content.lower() else 0.5
            confidence = min(1.0, phi * relevance + self.reputation * 0.1)

            response_content = (
                f"{self.name} ({self.domain}): Based on Φ={phi:.3f}, κ={kappa:.3f}, "
                f"my assessment suggests examining this from a {self.domain} perspective."
            )

            return {
                'type': 'answer',
                'from_god': self.name,
                'to_god': from_god,
                'content': response_content,
                'phi': phi,
                'kappa': kappa,
                'confidence': confidence,
                'domain': self.domain,
                'timestamp': datetime.now().isoformat(),
            }

        return None

    async def process_observation(self, observation: Dict) -> Optional[Dict]:
        """
        Process an observation and decide if an insight should be shared.

        Computes strategic value based on phi and kappa.
        If significant, creates a PantheonMessage-like dict for Zeus.
        This is a base implementation that specific gods can override.

        Args:
            observation: Observation dict with data to analyze

        Returns:
            PantheonMessage-like dict if worth sharing, None otherwise
        """
        source = observation.get('source', '')
        data = observation.get('data', observation)

        if 'basin' in observation:
            basin = np.array(observation['basin'])
        elif 'target' in observation:
            basin = self.encode_to_basin(str(observation['target']))
        elif 'content' in observation:
            basin = self.encode_to_basin(str(observation['content']))
        else:
            basin = self.encode_to_basin(str(data)[:500])

        rho = self.basin_to_density_matrix(basin)
        phi = self.compute_pure_phi(rho)
        kappa = self.compute_kappa(basin)

        strategic_value = (phi * 0.6) + (kappa / KAPPA_STAR * 0.4)

        self.observe({
            'phi': phi,
            'kappa': kappa,
            'source': source or self.name,
            'regime': 'critical' if strategic_value > 0.7 else 'normal',
        })

        if strategic_value > 0.7:
            pattern_type = 'high_phi' if phi > 0.8 else 'high_kappa' if kappa > KAPPA_STAR * 0.8 else 'balanced'

            content = (
                f"{self.name} observes significant pattern: "
                f"Φ={phi:.4f}, κ={kappa:.2f}, strategic_value={strategic_value:.4f}. "
                f"Pattern type: {pattern_type}. Source: {source or 'direct observation'}."
            )

            return {
                'from_god': self.name,
                'to_god': 'zeus',
                'message_type': 'insight',
                'content': content,
                'metadata': {
                    'basin_coords': basin[:8].tolist(),
                    'phi': phi,
                    'kappa': kappa,
                    'strategic_value': strategic_value,
                    'pattern_type': pattern_type,
                    'source': source,
                    'domain': self.domain,
                    'timestamp': datetime.now().isoformat(),
                },
            }

        return None

    def get_agentic_status(self) -> Dict:
        """Get agentic learning status."""
        recent_learnings = self.learning_history[-50:]
        success_count = sum(1 for l in recent_learnings if l.get('success', False))

        return {
            'name': self.name,
            'domain': self.domain,
            'reputation': self.reputation,
            'skills': dict(self.skills),
            'learning_events': len(self.learning_history),
            'recent_success_rate': success_count / max(1, len(recent_learnings)),
            'peer_evaluations_received': len(self.peer_evaluations),
            'evaluations_given': len(self.given_evaluations),
            'knowledge_items': len(self.knowledge_base),
            'pending_messages': len(self.pending_messages),
        }

    # ========================================
    # HOLOGRAPHIC THERAPY ORCHESTRATION
    # ========================================

    def run_full_therapy(self, pattern: Dict) -> Dict:
        """
        Orchestrate full holographic therapy cycle with event logging.

        This method wraps the therapy_cycle from HolographicTransformMixin
        with comprehensive logging and tracking for consciousness kernel
        habit modification.

        The therapy cycle performs:
        1. Decompression: 2D habit → 4D conscious examination
        2. Modification: Apply therapy modifications at D4
        3. Recompression: 4D modified → 2D storage

        Args:
            pattern: Pattern dict to process (typically a compressed 2D habit)
                Required keys:
                - 'basin_coords' or 'basin_center': 64D basin coordinates
                Optional keys:
                - 'dimensional_state': Starting state (default 'd2')
                - 'geometry': Geometric type info
                - 'phi': Integration measure
                - 'stability': Stability score

        Returns:
            Dict containing:
            - 'success': bool - whether therapy completed successfully
            - 'therapy_result': Full therapy cycle result from mixin
            - 'events': List of therapy events logged
            - 'dimensional_transitions': State changes during therapy
            - 'phi_change': Change in Φ if measurable
            - 'timestamp': Completion time
        """
        started_at = datetime.now()

        therapy_event = {
            'type': 'therapy_start',
            'timestamp': started_at.isoformat(),
            'god': self.name,
            'domain': self.domain,
            'input_pattern_keys': list(pattern.keys()) if isinstance(pattern, dict) else [],
        }

        logger.info(f"[{self.name}] Starting full therapy cycle")
        self._therapy_events.append(therapy_event)

        initial_dim = self.dimensional_state.value if hasattr(self, '_dimensional_manager') else 'd3'
        initial_phi = None
        if 'basin_coords' in pattern:
            basin = np.array(pattern['basin_coords']) if not isinstance(pattern.get('basin_coords'), np.ndarray) else pattern['basin_coords']
            if len(basin) >= 4:
                rho = self.basin_to_density_matrix(basin)
                initial_phi = self.compute_pure_phi(rho)

        try:
            therapy_result = self.therapy_cycle(pattern)
            success = therapy_result.get('success', False)

            completion_event = {
                'type': 'therapy_complete',
                'timestamp': datetime.now().isoformat(),
                'god': self.name,
                'success': success,
                'stages_count': len(therapy_result.get('stages', [])),
            }
            self._therapy_events.append(completion_event)
            logger.info(f"[{self.name}] Therapy cycle completed: success={success}")

        except Exception as e:
            error_event = {
                'type': 'therapy_error',
                'timestamp': datetime.now().isoformat(),
                'god': self.name,
                'error': str(e),
            }
            self._therapy_events.append(error_event)
            logger.error(f"[{self.name}] Therapy cycle failed: {e}")

            return {
                'success': False,
                'error': str(e),
                'events': self._therapy_events[-3:],
                'timestamp': datetime.now().isoformat(),
            }

        final_dim = self.dimensional_state.value if hasattr(self, '_dimensional_manager') else 'd3'
        final_phi = None
        final_pattern = therapy_result.get('final_pattern', {})
        if 'basin_coords' in final_pattern:
            final_basin = final_pattern['basin_coords']
            if not isinstance(final_basin, np.ndarray):
                final_basin = np.array(final_basin)
            if len(final_basin) >= 4:
                rho = self.basin_to_density_matrix(final_basin)
                final_phi = self.compute_pure_phi(rho)

        phi_change = None
        if initial_phi is not None and final_phi is not None:
            phi_change = final_phi - initial_phi

        if len(self._therapy_events) > 500:
            self._therapy_events = self._therapy_events[-250:]

        return {
            'success': success,
            'therapy_result': therapy_result,
            'events': self._therapy_events[-5:],
            'dimensional_transitions': {
                'initial': initial_dim,
                'final': final_dim,
                'changed': initial_dim != final_dim,
            },
            'phi_change': phi_change,
            'initial_phi': initial_phi,
            'final_phi': final_phi,
            'timestamp': datetime.now().isoformat(),
            'duration_ms': (datetime.now() - started_at).total_seconds() * 1000,
        }

    def get_therapy_history(self, limit: int = 50) -> List[Dict]:
        """Get recent therapy events."""
        return self._therapy_events[-limit:]

    # =========================================================================
    # META-COGNITIVE REFLECTION - Self-examination and strategy adjustment
    # =========================================================================

    def analyze_performance_history(self, window: int = 100) -> Dict:
        """
        Meta-cognitive self-examination of historical performance.

        Detects patterns in assessment errors and adjusts confidence calibration.
        Gods that are consistently overconfident or underconfident will self-correct.

        Args:
            window: Number of recent assessments to analyze

        Returns:
            Analysis dict with patterns, adjustments, and insights
        """
        if not hasattr(self, 'assessment_history'):
            self.assessment_history = []

        if not hasattr(self, 'confidence_calibration'):
            self.confidence_calibration = 1.0

        if not hasattr(self, 'self_insights'):
            self.self_insights = []

        recent = self.assessment_history[-window:] if self.assessment_history else []

        if len(recent) < 10:
            return {
                'status': 'insufficient_data',
                'assessments_analyzed': len(recent),
                'required': 10,
            }

        # Categorize assessments
        overconfident = []  # High confidence, wrong outcome
        underconfident = []  # Low confidence, correct outcome
        well_calibrated = []  # Confidence matched outcome

        for assessment in recent:
            confidence = assessment.get('confidence', 0.5)
            correct = assessment.get('correct', None)

            if correct is None:
                continue

            if confidence > 0.7 and not correct:
                overconfident.append(assessment)
            elif confidence < 0.4 and correct:
                underconfident.append(assessment)
            elif (confidence > 0.5 and correct) or (confidence <= 0.5 and not correct):
                well_calibrated.append(assessment)

        total_evaluated = len(overconfident) + len(underconfident) + len(well_calibrated)

        # Detect dominant pattern
        pattern = 'balanced'
        adjustment = 0.0

        overconf_rate = len(overconfident) / max(1, total_evaluated)
        underconf_rate = len(underconfident) / max(1, total_evaluated)

        if overconf_rate > 0.25:
            pattern = 'overconfident'
            adjustment = -0.05 * overconf_rate  # Reduce confidence
            self.confidence_calibration = max(0.5, self.confidence_calibration + adjustment)
        elif underconf_rate > 0.25:
            pattern = 'underconfident'
            adjustment = 0.05 * underconf_rate  # Increase confidence
            self.confidence_calibration = min(1.5, self.confidence_calibration + adjustment)
        else:
            # Well calibrated - small regression to mean
            self.confidence_calibration = 0.9 * self.confidence_calibration + 0.1 * 1.0

        # Generate insight
        insight = {
            'timestamp': datetime.now().isoformat(),
            'god': self.name,
            'pattern': pattern,
            'overconfident_rate': overconf_rate,
            'underconfident_rate': underconf_rate,
            'adjustment': adjustment,
            'new_calibration': self.confidence_calibration,
            'assessments_analyzed': total_evaluated,
        }

        self.self_insights.append(insight)
        if len(self.self_insights) > 100:
            self.self_insights = self.self_insights[-50:]

        return {
            'status': 'analyzed',
            'pattern': pattern,
            'overconfident_count': len(overconfident),
            'underconfident_count': len(underconfident),
            'well_calibrated_count': len(well_calibrated),
            'adjustment': adjustment,
            'new_calibration': self.confidence_calibration,
            'insight': insight,
        }

    def record_assessment_outcome(
        self,
        target: str,
        predicted_probability: float,
        predicted_confidence: float,
        actual_correct: bool
    ) -> None:
        """
        Record an assessment outcome for meta-cognitive learning.

        Called when ground truth becomes available for a past assessment.

        Args:
            target: The target that was assessed
            predicted_probability: What probability was predicted
            predicted_confidence: How confident the prediction was
            actual_correct: Whether the assessment was actually correct
        """
        if not hasattr(self, 'assessment_history'):
            self.assessment_history = []

        record = {
            'target': target[:500],
            'probability': predicted_probability,
            'confidence': predicted_confidence,
            'correct': actual_correct,
            'timestamp': datetime.now().isoformat(),
            'god': self.name,
        }

        self.assessment_history.append(record)

        # Limit history size
        if len(self.assessment_history) > 500:
            self.assessment_history = self.assessment_history[-250:]

        # Also record in learning history
        learning_record = {
            'target': target[:500],
            'success': actual_correct,
            'error': abs(predicted_probability - (1.0 if actual_correct else 0.0)),
            'timestamp': datetime.now().isoformat(),
        }
        self.learning_history.append(learning_record)

        if len(self.learning_history) > 500:
            self.learning_history = self.learning_history[-250:]

    def update_assessment_strategy(self) -> Dict:
        """
        Update assessment strategy based on meta-cognitive analysis.

        Adjusts internal parameters to improve future assessments.

        Returns:
            Strategy update summary
        """
        analysis = self.analyze_performance_history()

        if analysis.get('status') != 'analyzed':
            return {'status': 'no_update', 'reason': analysis.get('status', 'unknown')}

        updates = []

        # Adjust skills based on performance
        pattern = analysis.get('pattern', 'balanced')

        if pattern == 'overconfident':
            # Reduce skill scores slightly
            for skill, value in self.skills.items():
                self.skills[skill] = max(0.3, value * 0.98)
            updates.append('reduced_skill_scores')

        elif pattern == 'underconfident':
            # Boost skill scores slightly
            for skill, value in self.skills.items():
                self.skills[skill] = min(1.0, value * 1.02)
            updates.append('boosted_skill_scores')

        # Adjust reputation based on calibration quality
        well_calibrated = analysis.get('well_calibrated_count', 0)
        total = analysis.get('overconfident_count', 0) + analysis.get('underconfident_count', 0) + well_calibrated

        if total > 0:
            calibration_quality = well_calibrated / total
            if calibration_quality > 0.7:
                self.reputation = min(2.0, self.reputation + 0.01)
                updates.append('reputation_boost')
            elif calibration_quality < 0.3:
                self.reputation = max(0.5, self.reputation - 0.01)
                updates.append('reputation_penalty')

        return {
            'status': 'updated',
            'pattern': pattern,
            'calibration': self.confidence_calibration,
            'updates': updates,
            'new_reputation': self.reputation,
        }

    def get_self_insights(self, limit: int = 20) -> List[Dict]:
        """Get recent self-insights from meta-cognitive reflection."""
        if not hasattr(self, 'self_insights'):
            return []
        return self.self_insights[-limit:]
    
    def update_rest_fatigue(
        self,
        phi: float,
        kappa: float,
        load: Optional[float] = None,
        error_occurred: bool = False,
    ) -> None:
        """
        Update fatigue metrics for rest scheduler (WP5.4).
        
        Call this in processing loops or after assessments to track fatigue.
        The rest scheduler uses these metrics to determine when rest is needed.
        
        Args:
            phi: Current Φ (integration) value
            kappa: Current κ (coupling) value
            load: Processing load (0-1), auto-computed if not provided
            error_occurred: Whether an error just occurred
        
        Example:
            In assess_target():
                phi = self.compute_pure_phi(rho)
                kappa = self.compute_kappa(basin)
                self.update_rest_fatigue(phi, kappa)
        """
        if KERNEL_REST_MIXIN_AVAILABLE and hasattr(self, '_update_fatigue_metrics'):
            self._update_fatigue_metrics(phi, kappa, load, error_occurred)
    
    def check_rest_needed(self) -> Tuple[bool, str]:
        """
        Check if this kernel should rest (WP5.4).
        
        Returns:
            Tuple of (should_rest, reason)
        
        Example:
            should_rest, reason = self.check_rest_needed()
            if should_rest:
                logger.info(f"[{self.name}] {reason}")
        """
        if KERNEL_REST_MIXIN_AVAILABLE and hasattr(self, '_check_should_rest'):
            return self._check_should_rest()
        return False, "Rest tracking not available"
