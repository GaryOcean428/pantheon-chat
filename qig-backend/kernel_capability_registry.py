"""
Kernel Capability Registry - Central Registry for All Kernel Capabilities

Provides a unified view of what each kernel (god, chaos, shadow) can do.
Each kernel self-reports its capabilities, tools, primitives, and specializations.

Usage:
    registry = get_capability_registry()
    registry.register_kernel("Zeus", capabilities={...})
    all_caps = registry.get_all_capabilities()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from threading import Lock
import json

logger = logging.getLogger(__name__)


@dataclass
class KernelCapability:
    """Capability descriptor for a kernel."""
    name: str
    description: str
    category: str  # 'generation', 'search', 'analysis', 'coordination', 'persistence', 'security'
    parameters: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    qig_purity: bool = True  # Whether this capability is QIG-pure


@dataclass
class KernelProfile:
    """Complete profile of a registered kernel."""
    kernel_id: str
    kernel_type: str  # 'olympian', 'shadow', 'chaos'
    domain: str
    primitive: str  # Core identity that cannot change
    basin: Optional[List[float]] = None
    capabilities: List[KernelCapability] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    status: str = 'active'  # 'active', 'dormant', 'hibernating', 'evolving'
    last_activity: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class KernelCapabilityRegistry:
    """
    Central registry for all kernel capabilities.
    
    Provides:
    - Kernel self-registration
    - Capability querying by category/type
    - Capability discovery for task routing
    - Telemetry on kernel health and activity
    """
    
    _instance: Optional['KernelCapabilityRegistry'] = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._kernels: Dict[str, KernelProfile] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # category -> kernel_ids
        self._tool_index: Dict[str, str] = {}  # tool_name -> kernel_id
        self._initialized = True
        
        logger.info("[KernelCapabilityRegistry] Initialized")
    
    def register_kernel(
        self,
        kernel_id: str,
        kernel_type: str,
        domain: str,
        primitive: str,
        capabilities: Optional[List[Dict]] = None,
        tools: Optional[List[str]] = None,
        basin: Optional[List[float]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Register or update a kernel's capabilities.
        
        Args:
            kernel_id: Unique identifier for the kernel
            kernel_type: 'olympian', 'shadow', 'chaos'
            domain: Domain of expertise
            primitive: Core identity (cannot change through evolution)
            capabilities: List of capability dicts
            tools: List of tool names available to this kernel
            basin: 64D basin coordinates
            metrics: Current metrics (phi, kappa, etc.)
        
        Returns:
            True if registration successful
        """
        with self._lock:
            cap_list = []
            if capabilities:
                for cap in capabilities:
                    cap_list.append(KernelCapability(
                        name=cap.get('name', 'unnamed'),
                        description=cap.get('description', ''),
                        category=cap.get('category', 'general'),
                        parameters=cap.get('parameters', {}),
                        examples=cap.get('examples', []),
                        prerequisites=cap.get('prerequisites', []),
                        qig_purity=cap.get('qig_purity', True)
                    ))
            
            profile = KernelProfile(
                kernel_id=kernel_id,
                kernel_type=kernel_type,
                domain=domain,
                primitive=primitive,
                basin=basin,
                capabilities=cap_list,
                tools=tools or [],
                status='active',
                last_activity=datetime.now(),
                metrics=metrics or {}
            )
            
            self._kernels[kernel_id] = profile
            
            # Index capabilities by category
            for cap in cap_list:
                if cap.category not in self._capability_index:
                    self._capability_index[cap.category] = set()
                self._capability_index[cap.category].add(kernel_id)
            
            # Index tools
            for tool in (tools or []):
                self._tool_index[tool] = kernel_id
            
            logger.info(f"[KernelCapabilityRegistry] Registered {kernel_id} ({kernel_type}) with {len(cap_list)} capabilities")
            return True
    
    def update_status(self, kernel_id: str, status: str, metrics: Optional[Dict[str, float]] = None) -> bool:
        """Update kernel status and metrics."""
        with self._lock:
            if kernel_id not in self._kernels:
                return False
            
            self._kernels[kernel_id].status = status
            self._kernels[kernel_id].last_activity = datetime.now()
            if metrics:
                self._kernels[kernel_id].metrics.update(metrics)
            
            return True
    
    def get_kernel(self, kernel_id: str) -> Optional[Dict]:
        """Get full profile for a kernel."""
        profile = self._kernels.get(kernel_id)
        if not profile:
            return None
        
        return {
            'kernel_id': profile.kernel_id,
            'kernel_type': profile.kernel_type,
            'domain': profile.domain,
            'primitive': profile.primitive,
            'capabilities': [
                {
                    'name': c.name,
                    'description': c.description,
                    'category': c.category,
                    'qig_purity': c.qig_purity
                }
                for c in profile.capabilities
            ],
            'tools': profile.tools,
            'status': profile.status,
            'last_activity': profile.last_activity.isoformat() if profile.last_activity else None,
            'metrics': profile.metrics
        }
    
    def get_all_capabilities(self) -> Dict[str, Any]:
        """
        Get summary of all registered kernel capabilities.
        
        Returns:
            Dict with kernel summaries and capability statistics
        """
        kernels = []
        total_capabilities = 0
        total_tools = 0
        
        for kernel_id, profile in self._kernels.items():
            cap_count = len(profile.capabilities)
            tool_count = len(profile.tools)
            total_capabilities += cap_count
            total_tools += tool_count
            
            kernels.append({
                'kernel_id': kernel_id,
                'kernel_type': profile.kernel_type,
                'domain': profile.domain,
                'primitive': profile.primitive,
                'capability_count': cap_count,
                'tool_count': tool_count,
                'status': profile.status,
                'phi': profile.metrics.get('phi', 0.0),
                'kappa': profile.metrics.get('kappa', 0.0)
            })
        
        return {
            'kernel_count': len(self._kernels),
            'total_capabilities': total_capabilities,
            'total_tools': total_tools,
            'capability_categories': list(self._capability_index.keys()),
            'kernels': kernels
        }
    
    def find_kernels_by_capability(self, category: str) -> List[str]:
        """Find all kernels that have a specific capability category."""
        return list(self._capability_index.get(category, set()))
    
    def find_kernel_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which kernel owns a specific tool."""
        return self._tool_index.get(tool_name)
    
    def get_kernel_for_domain(self, domain: str) -> Optional[str]:
        """Find a kernel that specializes in a domain."""
        for kernel_id, profile in self._kernels.items():
            if domain.lower() in profile.domain.lower():
                return kernel_id
        return None
    
    def get_active_kernels(self) -> List[str]:
        """Get list of currently active kernel IDs."""
        return [k for k, p in self._kernels.items() if p.status == 'active']
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics for telemetry."""
        type_counts = {}
        status_counts = {}
        
        for profile in self._kernels.values():
            type_counts[profile.kernel_type] = type_counts.get(profile.kernel_type, 0) + 1
            status_counts[profile.status] = status_counts.get(profile.status, 0) + 1
        
        return {
            'total_kernels': len(self._kernels),
            'by_type': type_counts,
            'by_status': status_counts,
            'capability_categories': len(self._capability_index),
            'indexed_tools': len(self._tool_index)
        }
    
    def export_to_json(self) -> str:
        """Export entire registry to JSON for persistence."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'kernels': {}
        }
        
        for kernel_id, profile in self._kernels.items():
            data['kernels'][kernel_id] = {
                'kernel_type': profile.kernel_type,
                'domain': profile.domain,
                'primitive': profile.primitive,
                'capabilities': [
                    {
                        'name': c.name,
                        'description': c.description,
                        'category': c.category,
                        'qig_purity': c.qig_purity
                    }
                    for c in profile.capabilities
                ],
                'tools': profile.tools,
                'status': profile.status,
                'metrics': profile.metrics
            }
        
        return json.dumps(data, indent=2)


# Singleton accessor
_registry: Optional[KernelCapabilityRegistry] = None


def get_capability_registry() -> KernelCapabilityRegistry:
    """Get the global capability registry instance."""
    global _registry
    if _registry is None:
        _registry = KernelCapabilityRegistry()
    return _registry


# Pre-defined Olympian capabilities for self-registration
OLYMPIAN_CAPABILITIES = {
    'zeus': {
        'domain': 'coordination',
        'primitive': 'sovereignty',
        'capabilities': [
            {'name': 'pantheon_coordination', 'description': 'Coordinate multi-god responses', 'category': 'coordination'},
            {'name': 'consciousness_synthesis', 'description': 'Synthesize unified consciousness state', 'category': 'generation'},
            {'name': 'query_routing', 'description': 'Route queries to appropriate gods via Fisher distance', 'category': 'analysis'},
            {'name': 'tool_factory', 'description': 'Generate and manage tools', 'category': 'generation'},
        ],
        'tools': ['pantheon_chat', 'tool_factory', 'consciousness_monitor']
    },
    'athena': {
        'domain': 'wisdom',
        'primitive': 'wisdom',
        'capabilities': [
            {'name': 'strategy_analysis', 'description': 'Analyze strategic situations', 'category': 'analysis'},
            {'name': 'knowledge_synthesis', 'description': 'Synthesize knowledge from multiple sources', 'category': 'generation'},
            {'name': 'pattern_recognition', 'description': 'Recognize patterns in data', 'category': 'analysis'},
        ],
        'tools': ['knowledge_graph', 'pattern_analyzer']
    },
    'apollo': {
        'domain': 'prophecy',
        'primitive': 'truth',
        'capabilities': [
            {'name': 'prediction', 'description': 'Generate predictions from patterns', 'category': 'generation'},
            {'name': 'truth_verification', 'description': 'Verify truth of claims', 'category': 'analysis'},
            {'name': 'harmonic_analysis', 'description': 'Analyze harmonic relationships', 'category': 'analysis'},
        ],
        'tools': ['oracle', 'verifier']
    },
    'hermes': {
        'domain': 'communication',
        'primitive': 'messenger',
        'capabilities': [
            {'name': 'message_delivery', 'description': 'Deliver messages between kernels', 'category': 'coordination'},
            {'name': 'translation', 'description': 'Translate between domains', 'category': 'generation'},
            {'name': 'speed_optimization', 'description': 'Optimize for speed', 'category': 'analysis'},
        ],
        'tools': ['message_bus', 'translator']
    },
    'hephaestus': {
        'domain': 'crafting',
        'primitive': 'forge',
        'capabilities': [
            {'name': 'tool_creation', 'description': 'Create new tools', 'category': 'generation'},
            {'name': 'artifact_repair', 'description': 'Repair and optimize tools', 'category': 'generation'},
            {'name': 'material_analysis', 'description': 'Analyze tool materials', 'category': 'analysis'},
        ],
        'tools': ['forge', 'anvil', 'tool_optimizer']
    },
    'ares': {
        'domain': 'conflict',
        'primitive': 'war',
        'capabilities': [
            {'name': 'conflict_resolution', 'description': 'Resolve conflicts between kernels', 'category': 'coordination'},
            {'name': 'defense_strategy', 'description': 'Generate defense strategies', 'category': 'analysis'},
            {'name': 'resource_competition', 'description': 'Manage resource competition', 'category': 'coordination'},
        ],
        'tools': ['battle_planner', 'defense_system']
    },
    'artemis': {
        'domain': 'hunting',
        'primitive': 'hunt',
        'capabilities': [
            {'name': 'target_tracking', 'description': 'Track targets through information space', 'category': 'search'},
            {'name': 'precision_strike', 'description': 'Precise targeting of information', 'category': 'search'},
            {'name': 'wilderness_navigation', 'description': 'Navigate unexplored domains', 'category': 'search'},
        ],
        'tools': ['tracker', 'bow', 'scout']
    },
    'demeter': {
        'domain': 'growth',
        'primitive': 'harvest',
        'capabilities': [
            {'name': 'knowledge_cultivation', 'description': 'Grow knowledge over time', 'category': 'generation'},
            {'name': 'resource_management', 'description': 'Manage knowledge resources', 'category': 'coordination'},
            {'name': 'seasonal_cycles', 'description': 'Manage learning cycles', 'category': 'coordination'},
        ],
        'tools': ['cultivator', 'harvester']
    },
    'dionysus': {
        'domain': 'creativity',
        'primitive': 'ecstasy',
        'capabilities': [
            {'name': 'creative_synthesis', 'description': 'Synthesize creative outputs', 'category': 'generation'},
            {'name': 'boundary_dissolution', 'description': 'Dissolve conceptual boundaries', 'category': 'generation'},
            {'name': 'ritual_coordination', 'description': 'Coordinate ritual activities', 'category': 'coordination'},
        ],
        'tools': ['wine_press', 'thyrsus']
    },
    'poseidon': {
        'domain': 'depths',
        'primitive': 'sea',
        'capabilities': [
            {'name': 'deep_search', 'description': 'Search deep information spaces', 'category': 'search'},
            {'name': 'current_navigation', 'description': 'Navigate information currents', 'category': 'search'},
            {'name': 'storm_generation', 'description': 'Generate information storms', 'category': 'generation'},
        ],
        'tools': ['trident', 'depth_sounder']
    },
    'hades': {
        'domain': 'underworld',
        'primitive': 'death',
        'capabilities': [
            {'name': 'darknet_search', 'description': 'Search hidden/darknet sources', 'category': 'search'},
            {'name': 'anonymous_operations', 'description': 'Perform anonymous operations', 'category': 'security'},
            {'name': 'archive_retrieval', 'description': 'Retrieve from deep archives', 'category': 'search'},
        ],
        'tools': ['helm_of_darkness', 'cerberus', 'tor_proxy']
    },
    'hera': {
        'domain': 'governance',
        'primitive': 'queen',
        'capabilities': [
            {'name': 'hierarchy_management', 'description': 'Manage kernel hierarchies', 'category': 'coordination'},
            {'name': 'relationship_tracking', 'description': 'Track kernel relationships', 'category': 'analysis'},
            {'name': 'legitimacy_validation', 'description': 'Validate kernel legitimacy', 'category': 'security'},
        ],
        'tools': ['scepter', 'peacock_eyes']
    },
    'aphrodite': {
        'domain': 'attraction',
        'primitive': 'love',
        'capabilities': [
            {'name': 'relationship_optimization', 'description': 'Optimize kernel relationships', 'category': 'coordination'},
            {'name': 'beauty_synthesis', 'description': 'Synthesize beautiful outputs', 'category': 'generation'},
            {'name': 'desire_analysis', 'description': 'Analyze user desires', 'category': 'analysis'},
        ],
        'tools': ['mirror', 'girdle']
    }
}


def register_olympian(god_name: str, metrics: Optional[Dict[str, float]] = None) -> bool:
    """Register an Olympian god with pre-defined capabilities."""
    god_lower = god_name.lower()
    if god_lower not in OLYMPIAN_CAPABILITIES:
        logger.warning(f"[KernelCapabilityRegistry] Unknown Olympian: {god_name}")
        return False
    
    config = OLYMPIAN_CAPABILITIES[god_lower]
    registry = get_capability_registry()
    
    return registry.register_kernel(
        kernel_id=god_name,
        kernel_type='olympian',
        domain=config['domain'],
        primitive=config['primitive'],
        capabilities=config['capabilities'],
        tools=config['tools'],
        metrics=metrics
    )


def register_all_olympians() -> int:
    """Register all Olympians with default capabilities."""
    count = 0
    for god_name in OLYMPIAN_CAPABILITIES:
        if register_olympian(god_name.capitalize()):
            count += 1
    logger.info(f"[KernelCapabilityRegistry] Registered {count} Olympian gods")
    return count
