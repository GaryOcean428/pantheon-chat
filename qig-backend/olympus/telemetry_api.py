"""
Olympus Telemetry API Routes

Exposes capability telemetry data to the frontend dashboard.
Provides fleet-wide and per-kernel capability metrics.

QIG-Pure: No neural networks, pure geometric consciousness tracking.
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from capability_telemetry import CapabilityTelemetryRegistry, create_olympus_capabilities
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    print("[WARNING] capability_telemetry module not available")

telemetry_bp = Blueprint('telemetry', __name__, url_prefix='/api/telemetry')

# Get singleton registry instance
_registry: Optional[CapabilityTelemetryRegistry] = None
_initialized: bool = False

def get_registry() -> Optional[CapabilityTelemetryRegistry]:
    """Get or create the telemetry registry singleton."""
    global _registry
    if _registry is None and TELEMETRY_AVAILABLE:
        _registry = CapabilityTelemetryRegistry()
    return _registry


def initialize_god_telemetry():
    """
    Initialize telemetry for all Olympian gods.
    
    Registers each god with the telemetry system and assigns their capabilities.
    Should be called during application startup after gods are instantiated.
    """
    global _initialized
    
    if _initialized:
        return
    
    if not TELEMETRY_AVAILABLE:
        print("[TelemetryAPI] Capability telemetry not available - skipping initialization")
        return
    
    registry = get_registry()
    if not registry:
        print("[TelemetryAPI] Failed to get telemetry registry")
        return
    
    # Define standard Olympian gods
    OLYMPIAN_GODS = [
        ('zeus', 'Zeus'),
        ('hera', 'Hera'),
        ('poseidon', 'Poseidon'),
        ('athena', 'Athena'),
        ('apollo', 'Apollo'),
        ('artemis', 'Artemis'),
        ('hermes', 'Hermes'),
        ('ares', 'Ares'),
        ('hephaestus', 'Hephaestus'),
        ('aphrodite', 'Aphrodite'),
        ('demeter', 'Demeter'),
        ('dionysus', 'Dionysus'),
    ]
    
    # Define shadow pantheon gods
    SHADOW_GODS = [
        ('hades', 'Hades'),
        ('nyx', 'Nyx'),
        ('hecate', 'Hecate'),
        ('erebus', 'Erebus'),
        ('hypnos', 'Hypnos'),
        ('thanatos', 'Thanatos'),
        ('nemesis', 'Nemesis'),
    ]
    
    try:
        # Get standard capabilities
        standard_caps = create_olympus_capabilities()
        
        # Register all Olympian gods
        for kernel_id, kernel_name in OLYMPIAN_GODS:
            profile = registry.register_kernel(kernel_id, kernel_name)
            for cap in standard_caps:
                profile.register_capability(cap)
        
        # Register shadow gods with same capabilities
        for kernel_id, kernel_name in SHADOW_GODS:
            profile = registry.register_kernel(kernel_id, kernel_name)
            for cap in standard_caps:
                profile.register_capability(cap)
        
        _initialized = True
        print(f"[TelemetryAPI] Initialized telemetry for {len(OLYMPIAN_GODS) + len(SHADOW_GODS)} gods")
        
    except Exception as e:
        print(f"[TelemetryAPI] Error initializing god telemetry: {e}")
        import traceback
        traceback.print_exc()



@telemetry_bp.route('/fleet', methods=['GET'])
def get_fleet_telemetry():
    """
    Get aggregated telemetry across all kernels.
    
    Returns:
        {
            "success": true,
            "data": {
                "kernels": 12,
                "total_capabilities": 120,
                "total_invocations": 450,
                "fleet_success_rate": 0.95,
                "category_distribution": {"research": 30, "communication": 25, ...},
                "kernel_summaries": [...]
            }
        }
    """
    registry = get_registry()
    
    if not registry or not TELEMETRY_AVAILABLE:
        # Graceful degradation - return empty fleet data
        return jsonify({
            'success': True,
            'data': {
                'kernels': 0,
                'total_capabilities': 0,
                'total_invocations': 0,
                'fleet_success_rate': 0.0,
                'category_distribution': {},
                'kernel_summaries': [],
                'message': 'Telemetry system not initialized yet'
            }
        })
    
    try:
        fleet_data = registry.get_fleet_telemetry()
        return jsonify({
            'success': True,
            'data': fleet_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': {
                'kernels': 0,
                'total_capabilities': 0,
                'total_invocations': 0,
                'fleet_success_rate': 0.0,
                'category_distribution': {},
                'kernel_summaries': []
            }
        }), 500


@telemetry_bp.route('/kernel/<kernel_id>/capabilities', methods=['GET'])
def get_kernel_capabilities(kernel_id: str):
    """
    Get detailed capabilities for a specific kernel.
    
    Args:
        kernel_id: The kernel identifier (e.g., 'zeus', 'athena', 'apollo')
    
    Returns:
        {
            "success": true,
            "data": {
                "kernel_id": "zeus",
                "kernel_name": "Zeus",
                "total_capabilities": 10,
                "enabled_capabilities": 9,
                "total_invocations": 150,
                "overall_success_rate": 0.96,
                "capabilities": {...}
            }
        }
    """
    registry = get_registry()
    
    if not registry or not TELEMETRY_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Telemetry system not available',
            'data': None
        }), 503
    
    try:
        profile = registry.get_profile(kernel_id)
        
        if not profile:
            return jsonify({
                'success': False,
                'error': f'Kernel {kernel_id} not found in registry',
                'data': None
            }), 404
        
        introspection = profile.get_introspection()
        return jsonify({
            'success': True,
            'data': introspection
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'data': None
        }), 500


@telemetry_bp.route('/kernel/<kernel_id>/summary', methods=['GET'])
def get_kernel_summary(kernel_id: str):
    """
    Get compact summary for a specific kernel.
    
    Args:
        kernel_id: The kernel identifier
    
    Returns:
        {
            "success": true,
            "data": {
                "kernel_id": "zeus",
                "kernel_name": "Zeus",
                "total_capabilities": 10,
                "enabled": 9,
                "total_invocations": 150,
                "success_rate": 0.96,
                "strongest": "research",
                "weakest": "voting"
            }
        }
    """
    registry = get_registry()
    
    if not registry or not TELEMETRY_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Telemetry system not available'
        }), 503
    
    try:
        profile = registry.get_profile(kernel_id)
        
        if not profile:
            return jsonify({
                'success': False,
                'error': f'Kernel {kernel_id} not found'
            }), 404
        
        summary = profile.get_summary()
        return jsonify({
            'success': True,
            'data': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@telemetry_bp.route('/kernels', methods=['GET'])
def list_registered_kernels():
    """
    Get list of all registered kernels.
    
    Returns:
        {
            "success": true,
            "data": {
                "kernels": ["zeus", "athena", "apollo", ...],
                "count": 12
            }
        }
    """
    registry = get_registry()
    
    if not registry or not TELEMETRY_AVAILABLE:
        return jsonify({
            'success': True,
            'data': {
                'kernels': [],
                'count': 0,
                'message': 'Telemetry system not initialized yet'
            }
        })
    
    try:
        kernel_ids = list(registry.profiles.keys())
        return jsonify({
            'success': True,
            'data': {
                'kernels': kernel_ids,
                'count': len(kernel_ids)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@telemetry_bp.route('/all', methods=['GET'])
def get_all_introspections():
    """
    Get full introspection data for all kernels.
    
    Returns:
        {
            "success": true,
            "data": {
                "zeus": {...},
                "athena": {...},
                ...
            }
        }
    """
    registry = get_registry()
    
    if not registry or not TELEMETRY_AVAILABLE:
        return jsonify({
            'success': True,
            'data': {},
            'message': 'Telemetry system not initialized yet'
        })
    
    try:
        all_data = registry.get_all_introspections()
        return jsonify({
            'success': True,
            'data': all_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@telemetry_bp.route('/health', methods=['GET'])
def telemetry_health():
    """
    Health check for telemetry system.
    
    Returns:
        {
            "success": true,
            "available": true,
            "registered_kernels": 12
        }
    """
    registry = get_registry()
    
    return jsonify({
        'success': True,
        'available': TELEMETRY_AVAILABLE and registry is not None,
        'registered_kernels': len(registry.profiles) if registry else 0
    })


def register_telemetry_routes(app):
    """Register telemetry routes with Flask app."""
    app.register_blueprint(telemetry_bp)
    print("[TelemetryAPI] Routes registered at /api/telemetry")
    
    # Initialize god telemetry on startup
    initialize_god_telemetry()
