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
"""

print("[Olympus/__init__] Starting imports...", flush=True)
print("[Olympus/__init__] Importing Aphrodite...", flush=True)
from .aphrodite import Aphrodite
print("[Olympus/__init__] Importing Apollo...", flush=True)
from .apollo import Apollo
print("[Olympus/__init__] Importing Ares...", flush=True)
from .ares import Ares
print("[Olympus/__init__] Importing Artemis...", flush=True)
from .artemis import Artemis
print("[Olympus/__init__] Imported Aphrodite-Artemis", flush=True)

# Olympian Gods
from .athena import Athena
from .base_god import BaseGod
from .demeter import Demeter
from .dionysus import Dionysus
from .hades import Hades
from .hephaestus import Hephaestus
from .hera import Hera
from .hermes import Hermes
print("[Olympus/__init__] Imported Athena-Hermes", flush=True)

# Team #2 - Coordinator
from .hermes_coordinator import HermesCoordinator, get_hermes_coordinator
print("[Olympus/__init__] Imported HermesCoordinator", flush=True)

# Communication
from .pantheon_chat import Debate, PantheonChat, PantheonMessage
from .poseidon import Poseidon
print("[Olympus/__init__] Imported PantheonChat+Poseidon", flush=True)

# Shadow Pantheon
from .shadow_pantheon import Erebus, Hecate, Hypnos, Nemesis, Nyx, ShadowGod, ShadowPantheon, Thanatos
print("[Olympus/__init__] Imported ShadowPantheon", flush=True)

# Lightning Bolt Insight Kernel (dynamic domains - no hardcoded enums)
from .lightning_kernel import LightningKernel, get_lightning_kernel, CrossDomainInsight, DomainEvent, set_pantheon_chat
print("[Olympus/__init__] Imported LightningKernel", flush=True)

# Universal Capability Mesh
from .capability_mesh import (
    CapabilityEventBus, CapabilityEvent, CapabilityType, EventType,
    get_event_bus, emit_event, subscribe_to_events, get_mesh_status,
    SUBSCRIPTION_MATRIX
)
print("[Olympus/__init__] Imported capability_mesh", flush=True)
from .capability_bridges import (
    DebateResearchBridge, EmotionCapabilityBridge, ForesightActionBridge,
    EthicsCapabilityBridge, SleepLearningBridge, BasinCapabilityBridge,
    WarResourceBridge, KernelMeshBridge, initialize_all_bridges, get_bridge_stats
)
print("[Olympus/__init__] Imported capability_bridges", flush=True)

# Auto Tool Discovery & Persistence
from .auto_tool_discovery import (
    ToolDiscoveryEngine, create_discovery_engine_for_god
)
from .tool_request_persistence import (
    ToolRequestPersistence, ToolRequest, RequestStatus,
    RequestPriority, PatternDiscovery, get_tool_request_persistence
)
print("[Olympus/__init__] Imported ToolDiscovery+Persistence", flush=True)

# Telemetry API
from .telemetry_api import telemetry_bp, register_telemetry_routes, initialize_god_telemetry
print("[Olympus/__init__] Imported telemetry_api", flush=True)

# Ocean+Heart Consensus for autonomic cycle governance
from .ocean_heart_consensus import OceanHeartConsensus, get_ocean_heart_consensus, CycleType, CycleDecision
from .heart_kernel import HeartKernel, get_heart_kernel, HeartState
print("[Olympus/__init__] Imported Ocean+Heart consensus", flush=True)

# Core hierarchy
print("[Olympus/__init__] About to import zeus...", flush=True)
from .zeus import Zeus, olympus_app, zeus
print("[Olympus/__init__] Imported Zeus singleton", flush=True)

# Zeus Chat (voice integration)
from .zeus_chat import ZeusConversationHandler
from .zeus_chat_encoding import GeometricGenerationMixin, ZeusGenerationMixin

# Guardian Gods for kernel development
try:
    from .hestia import Hestia, SafetyConfig, SafetyVitals
    from .demeter_tutor import DemeterTutor, Lesson, StudentProgress
    from .chiron import Chiron, DiagnosticIssue, PatientRecord
    from .knowledge_exchange import KnowledgeExchange
    GUARDIANS_AVAILABLE = True
except ImportError:
    GUARDIANS_AVAILABLE = False

# Conversational Kernel System
try:
    from conversational_kernel import (
        ConversationalKernelMixin,
        ConversationState,
        patch_god_with_conversation,
        patch_all_gods_with_conversation,
    )
    from recursive_conversation_orchestrator import (
        RecursiveConversationOrchestrator,
        get_conversation_orchestrator,
    )
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False

__all__ = [
    # Hierarchy
    'Zeus',
    'olympus_app',
    'zeus',
    'BaseGod',

    # Coordinator (#2)
    'HermesCoordinator',
    'get_hermes_coordinator',

    # Olympians
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

    # Communication
    'PantheonChat',
    'PantheonMessage',
    'Debate',
    'ZeusConversationHandler',

    # Shadow
    'ShadowPantheon',
    'ShadowGod',
    'Nyx',
    'Hecate',
    'Erebus',
    'Hypnos',
    'Thanatos',
    'Nemesis',

    # Conversational Kernel System
    'ConversationalKernelMixin',
    'ConversationState',
    'patch_god_with_conversation',
    'patch_all_gods_with_conversation',
    'RecursiveConversationOrchestrator',
    'get_conversation_orchestrator',
    'CONVERSATION_AVAILABLE',
    
    # Lightning Bolt Insight Kernel (dynamic domains - no hardcoded enums)
    'LightningKernel',
    'get_lightning_kernel',
    'CrossDomainInsight',
    'DomainEvent',
    'set_pantheon_chat',
    
    # Universal Capability Mesh
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
    
    # Auto Tool Discovery & Persistence
    'ToolDiscoveryEngine',
    'create_discovery_engine_for_god',
    'ToolRequestPersistence',
    'ToolRequest',
    'RequestStatus',
    'RequestPriority',
    'PatternDiscovery',
    'get_tool_request_persistence',
    
    # Telemetry API
    'telemetry_bp',
    'register_telemetry_routes',
    'initialize_god_telemetry',
    
    # Guardian Gods
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
    
    # Ocean+Heart Consensus (autonomic cycle governance)
    'OceanHeartConsensus',
    'get_ocean_heart_consensus',
    'CycleType',
    'CycleDecision',
    'HeartKernel',
    'get_heart_kernel',
    'HeartState',
]
