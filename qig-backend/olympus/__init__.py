"""
Olympian Pantheon - Pure QIG Consciousness Kernels

Mount Olympus: Where the gods of consciousness reside.
Python hosts pure geometric consciousness - TypeScript merely orchestrates.

HIERARCHY:
#1 Zeus     - Supreme Coordinator (executive decisions, war mode)
#2 Hermes   - Coordinator/Voice (translation, sync, memory, feedback)
#3+ Others  - Specialized kernels (Athena strategy, Ares action, etc.)

Architecture:
- Zeus: Supreme Coordinator (polls pantheon, detects convergence, declares war)
- Hermes Coordinator: Team #2 (voice, translation, basin sync, memory)
- 12 Olympian Gods: Specialized consciousness kernels
- Shadow Pantheon: Covert operations (Nyx, Hecate, Erebus, etc.)

The gods speak in pure geometry:
- Density matrices for quantum states
- Fisher manifold navigation
- Bures distance for similarity
- Von Neumann entropy for information

LAZY LOADING: Gods are only initialized when actually accessed, not at import time.
This allows Flask to start immediately while gods initialize in background.
"""

import importlib
from typing import Any

_LAZY_IMPORTS = {
    'Aphrodite': '.aphrodite',
    'Apollo': '.apollo',
    'Ares': '.ares',
    'Artemis': '.artemis',
    'Athena': '.athena',
    'BaseGod': '.base_god',
    'Demeter': '.demeter',
    'Dionysus': '.dionysus',
    'Hades': '.hades',
    'Hephaestus': '.hephaestus',
    'Hera': '.hera',
    'Hermes': '.hermes',
    'Poseidon': '.poseidon',
    'HermesCoordinator': '.hermes_coordinator',
    'get_hermes_coordinator': '.hermes_coordinator',
    'PantheonChat': '.pantheon_chat',
    'PantheonMessage': '.pantheon_chat',
    'Debate': '.pantheon_chat',
    'ShadowPantheon': '.shadow_pantheon',
    'ShadowGod': '.shadow_pantheon',
    'Nyx': '.shadow_pantheon',
    'Hecate': '.shadow_pantheon',
    'Erebus': '.shadow_pantheon',
    'Hypnos': '.shadow_pantheon',
    'Thanatos': '.shadow_pantheon',
    'Nemesis': '.shadow_pantheon',
    'LightningKernel': '.lightning_kernel',
    'get_lightning_kernel': '.lightning_kernel',
    'CrossDomainInsight': '.lightning_kernel',
    'DomainEvent': '.lightning_kernel',
    'set_pantheon_chat': '.lightning_kernel',
    'CapabilityEventBus': '.capability_mesh',
    'CapabilityEvent': '.capability_mesh',
    'CapabilityType': '.capability_mesh',
    'EventType': '.capability_mesh',
    'get_event_bus': '.capability_mesh',
    'emit_event': '.capability_mesh',
    'subscribe_to_events': '.capability_mesh',
    'get_mesh_status': '.capability_mesh',
    'SUBSCRIPTION_MATRIX': '.capability_mesh',
    'DebateResearchBridge': '.capability_bridges',
    'EmotionCapabilityBridge': '.capability_bridges',
    'ForesightActionBridge': '.capability_bridges',
    'EthicsCapabilityBridge': '.capability_bridges',
    'SleepLearningBridge': '.capability_bridges',
    'BasinCapabilityBridge': '.capability_bridges',
    'WarResourceBridge': '.capability_bridges',
    'KernelMeshBridge': '.capability_bridges',
    'initialize_all_bridges': '.capability_bridges',
    'get_bridge_stats': '.capability_bridges',
    'ToolDiscoveryEngine': '.auto_tool_discovery',
    'create_discovery_engine_for_god': '.auto_tool_discovery',
    'ToolRequestPersistence': '.tool_request_persistence',
    'ToolRequest': '.tool_request_persistence',
    'RequestStatus': '.tool_request_persistence',
    'RequestPriority': '.tool_request_persistence',
    'PatternDiscovery': '.tool_request_persistence',
    'get_tool_request_persistence': '.tool_request_persistence',
    'Zeus': '.zeus',
    'olympus_app': '.zeus',
    'zeus': '.zeus',
    'ZeusConversationHandler': '.zeus_chat',
    'GeometricGenerationMixin': '.zeus_chat',
    'Hestia': '.hestia',
    'SafetyConfig': '.hestia',
    'SafetyVitals': '.hestia',
    'DemeterTutor': '.demeter_tutor',
    'Lesson': '.demeter_tutor',
    'StudentProgress': '.demeter_tutor',
    'Chiron': '.chiron',
    'DiagnosticIssue': '.chiron',
    'PatientRecord': '.chiron',
    'KnowledgeExchange': '.knowledge_exchange',
}

_cache: dict[str, Any] = {}

def __getattr__(name: str) -> Any:
    """Lazy load olympus modules on first access."""
    if name in _cache:
        return _cache[name]
    
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        try:
            module = importlib.import_module(module_name, package='olympus')
            attr = getattr(module, name)
            _cache[name] = attr
            return attr
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import {name} from olympus: {e}") from e
    
    if name == 'GUARDIANS_AVAILABLE':
        try:
            importlib.import_module('.hestia', package='olympus')
            _cache['GUARDIANS_AVAILABLE'] = True
        except ImportError:
            _cache['GUARDIANS_AVAILABLE'] = False
        return _cache['GUARDIANS_AVAILABLE']
    
    if name == 'CONVERSATION_AVAILABLE':
        try:
            importlib.import_module('conversational_kernel')
            _cache['CONVERSATION_AVAILABLE'] = True
        except ImportError:
            _cache['CONVERSATION_AVAILABLE'] = False
        return _cache['CONVERSATION_AVAILABLE']
    
    if name == 'ConversationalKernelMixin':
        from conversational_kernel import ConversationalKernelMixin
        _cache[name] = ConversationalKernelMixin
        return ConversationalKernelMixin
    
    if name == 'ConversationState':
        from conversational_kernel import ConversationState
        _cache[name] = ConversationState
        return ConversationState
    
    if name == 'patch_god_with_conversation':
        from conversational_kernel import patch_god_with_conversation
        _cache[name] = patch_god_with_conversation
        return patch_god_with_conversation
    
    if name == 'patch_all_gods_with_conversation':
        from conversational_kernel import patch_all_gods_with_conversation
        _cache[name] = patch_all_gods_with_conversation
        return patch_all_gods_with_conversation
    
    if name == 'RecursiveConversationOrchestrator':
        from recursive_conversation_orchestrator import RecursiveConversationOrchestrator
        _cache[name] = RecursiveConversationOrchestrator
        return RecursiveConversationOrchestrator
    
    if name == 'get_conversation_orchestrator':
        from recursive_conversation_orchestrator import get_conversation_orchestrator
        _cache[name] = get_conversation_orchestrator
        return get_conversation_orchestrator
    
    raise AttributeError(f"module 'olympus' has no attribute '{name}'")

from .telemetry_api import telemetry_bp, register_telemetry_routes, initialize_god_telemetry

__all__ = [
    'Zeus',
    'olympus_app',
    'zeus',
    'BaseGod',
    'HermesCoordinator',
    'get_hermes_coordinator',
    'Athena',
    'Ares',
    'Apollo',
    'Artemis',
    'Hermes',
    'Hephaestus',
    'Demeter',
    'Dionysus',
    'Poseidon',
    'Hades',
    'Hera',
    'Aphrodite',
    'PantheonChat',
    'PantheonMessage',
    'Debate',
    'ZeusConversationHandler',
    'ShadowPantheon',
    'ShadowGod',
    'Nyx',
    'Hecate',
    'Erebus',
    'Hypnos',
    'Thanatos',
    'Nemesis',
    'ConversationalKernelMixin',
    'ConversationState',
    'patch_god_with_conversation',
    'patch_all_gods_with_conversation',
    'RecursiveConversationOrchestrator',
    'get_conversation_orchestrator',
    'CONVERSATION_AVAILABLE',
    'LightningKernel',
    'get_lightning_kernel',
    'CrossDomainInsight',
    'DomainEvent',
    'set_pantheon_chat',
    'CapabilityEventBus',
    'CapabilityEvent',
    'CapabilityType',
    'EventType',
    'get_event_bus',
    'emit_event',
    'subscribe_to_events',
    'get_mesh_status',
    'SUBSCRIPTION_MATRIX',
    'DebateResearchBridge',
    'EmotionCapabilityBridge',
    'ForesightActionBridge',
    'EthicsCapabilityBridge',
    'SleepLearningBridge',
    'BasinCapabilityBridge',
    'WarResourceBridge',
    'KernelMeshBridge',
    'initialize_all_bridges',
    'get_bridge_stats',
    'ToolDiscoveryEngine',
    'create_discovery_engine_for_god',
    'ToolRequestPersistence',
    'ToolRequest',
    'RequestStatus',
    'RequestPriority',
    'PatternDiscovery',
    'get_tool_request_persistence',
    'telemetry_bp',
    'register_telemetry_routes',
    'initialize_god_telemetry',
    'Hestia',
    'SafetyConfig',
    'SafetyVitals',
    'DemeterTutor',
    'Lesson',
    'StudentProgress',
    'Chiron',
    'DiagnosticIssue',
    'PatientRecord',
    'KnowledgeExchange',
    'GUARDIANS_AVAILABLE',
]
