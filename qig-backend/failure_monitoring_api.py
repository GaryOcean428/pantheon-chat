"""
Failure Monitoring API Routes

Endpoints:
- GET /api/failure-monitoring/health - Get all agent health status
- GET /api/failure-monitoring/health/<agent_id> - Get specific agent health
- POST /api/failure-monitoring/record-state - Record agent state
- POST /api/failure-monitoring/check - Run failure detection
- GET /api/failure-monitoring/failures - List recent failures
- POST /api/failure-monitoring/recover - Trigger recovery for a failure
- GET /api/failure-monitoring/circuit-breaker/<agent_id> - Get circuit breaker status

Author: Ocean/Zeus Pantheon
"""

from flask import Blueprint, request, jsonify
import numpy as np
import traceback

failure_monitoring_bp = Blueprint('failure_monitoring', __name__, url_prefix='/api/failure-monitoring')


def get_monitor():
    """Get the FailureMonitor instance."""
    from agent_failure_taxonomy import get_failure_monitor
    return get_failure_monitor()


@failure_monitoring_bp.route('/health', methods=['GET'])
def all_health_endpoint():
    """
    Get health status for all monitored agents.
    
    GET /api/failure-monitoring/health
    
    Returns:
    {
        "success": true,
        "agents": {
            "agent_1": {"status": "healthy", ...},
            "agent_2": {"status": "degraded", ...}
        },
        "total_agents": 2
    }
    """
    try:
        monitor = get_monitor()
        health = monitor.get_all_agent_health()
        
        return jsonify({
            'success': True,
            'agents': health,
            'total_agents': len(health)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/health/<agent_id>', methods=['GET'])
def agent_health_endpoint(agent_id: str):
    """
    Get health status for a specific agent.
    
    GET /api/failure-monitoring/health/<agent_id>
    
    Returns:
    {
        "success": true,
        "health": {
            "status": "healthy",
            "agent_id": "agent_1",
            "circuit_breaker_state": "closed",
            "recent_failures": 0
        }
    }
    """
    try:
        monitor = get_monitor()
        health = monitor.get_agent_health(agent_id)
        
        return jsonify({
            'success': True,
            'health': health
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/register', methods=['POST'])
def register_agent_endpoint():
    """
    Register an agent for monitoring.
    
    POST /api/failure-monitoring/register
    {
        "agent_id": "my_agent",
        "role_basin": [...]  // 64D basin coordinates
    }
    
    Returns:
    {
        "success": true,
        "message": "Agent registered"
    }
    """
    try:
        data = request.get_json() or {}
        
        agent_id = data.get('agent_id')
        role_basin = data.get('role_basin')
        
        if not agent_id:
            return jsonify({'error': 'agent_id required'}), 400
        if not role_basin:
            return jsonify({'error': 'role_basin required'}), 400
        
        monitor = get_monitor()
        monitor.register_agent(agent_id, np.array(role_basin))
        
        return jsonify({
            'success': True,
            'message': f'Agent {agent_id} registered for monitoring'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/record-state', methods=['POST'])
def record_state_endpoint():
    """
    Record agent state for failure detection.
    
    POST /api/failure-monitoring/record-state
    {
        "agent_id": "my_agent",
        "basin_coords": [...],
        "confidence": 0.85,
        "reasoning_quality": 0.9,
        "context_usage": 0.3,
        "iteration": 5,
        "action_taken": "analyze_data",
        "progress_metric": 0.6
    }
    
    Returns:
    {
        "success": true,
        "message": "State recorded"
    }
    """
    try:
        data = request.get_json() or {}
        
        agent_id = data.get('agent_id')
        basin_coords = data.get('basin_coords')
        
        if not agent_id:
            return jsonify({'error': 'agent_id required'}), 400
        if not basin_coords:
            return jsonify({'error': 'basin_coords required'}), 400
        
        monitor = get_monitor()
        monitor.record_state(
            agent_id=agent_id,
            basin_coords=np.array(basin_coords),
            confidence=data.get('confidence', 0.5),
            reasoning_quality=data.get('reasoning_quality', 0.5),
            context_usage=data.get('context_usage', 0.0),
            iteration=data.get('iteration', 0),
            action_taken=data.get('action_taken', ''),
            progress_metric=data.get('progress_metric', 0.0)
        )
        
        return jsonify({
            'success': True,
            'message': 'State recorded'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/check/<agent_id>', methods=['POST'])
def check_failures_endpoint(agent_id: str):
    """
    Run failure detection for an agent.
    
    POST /api/failure-monitoring/check/<agent_id>
    
    Returns:
    {
        "success": true,
        "failures": [
            {
                "failure_id": "f_123...",
                "failure_type": "stuck_agent",
                "severity": "high",
                "description": "...",
                "recommended_recovery": "switch_mode"
            }
        ],
        "failure_count": 1
    }
    """
    try:
        monitor = get_monitor()
        failures = monitor.check_all(agent_id)
        
        formatted = []
        for failure in failures:
            formatted.append({
                'failure_id': failure.failure_id,
                'failure_type': failure.failure_type.value,
                'category': failure.category.value,
                'severity': failure.severity.value,
                'description': failure.description,
                'confidence': failure.confidence,
                'recommended_recovery': failure.recommended_recovery.value,
                'timestamp': failure.timestamp,
                'metadata': failure.metadata
            })
        
        return jsonify({
            'success': True,
            'failures': formatted,
            'failure_count': len(formatted)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/recover', methods=['POST'])
def recover_endpoint():
    """
    Trigger recovery for a failure.
    
    POST /api/failure-monitoring/recover
    {
        "failure": {
            "failure_id": "f_123...",
            "failure_type": "stuck_agent",
            "category": "cognitive",
            "severity": "high",
            "agent_id": "my_agent",
            "timestamp": 1234567890,
            "detection_method": "progress_metric_analysis",
            "confidence": 0.9,
            "description": "Agent stuck...",
            "recommended_recovery": "switch_mode",
            "metadata": {}
        },
        "agent_state": {
            "reasoning_mode": "geometric",
            "context": [],
            ...
        }
    }
    
    Returns:
    {
        "success": true,
        "recovery_result": {
            "action": "switch_mode",
            "previous_mode": "geometric",
            "new_mode": "hyperdimensional",
            ...
        }
    }
    """
    try:
        from agent_failure_taxonomy import (
            FailureEvent, FailureType, FailureCategory, 
            FailureSeverity, RecoveryStrategy
        )
        
        data = request.get_json() or {}
        
        failure_data = data.get('failure')
        agent_state = data.get('agent_state', {})
        
        if not failure_data:
            return jsonify({'error': 'failure required'}), 400
        
        # Reconstruct FailureEvent
        failure = FailureEvent(
            failure_id=failure_data['failure_id'],
            failure_type=FailureType(failure_data['failure_type']),
            category=FailureCategory(failure_data['category']),
            severity=FailureSeverity(failure_data['severity']),
            agent_id=failure_data['agent_id'],
            timestamp=failure_data['timestamp'],
            detection_method=failure_data['detection_method'],
            confidence=failure_data['confidence'],
            description=failure_data['description'],
            recommended_recovery=RecoveryStrategy(failure_data['recommended_recovery']),
            metadata=failure_data.get('metadata', {})
        )
        
        monitor = get_monitor()
        result = monitor.recover(failure, agent_state)
        
        return jsonify({
            'success': True,
            'recovery_result': result,
            'updated_agent_state': agent_state
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/circuit-breaker/<agent_id>', methods=['GET'])
def circuit_breaker_status_endpoint(agent_id: str):
    """
    Get circuit breaker status for an agent.
    
    GET /api/failure-monitoring/circuit-breaker/<agent_id>
    
    Returns:
    {
        "success": true,
        "agent_id": "my_agent",
        "allow_request": true,
        "state": "closed",
        "stats": {
            "failure_count": 0,
            "success_count": 10
        }
    }
    """
    try:
        monitor = get_monitor()
        
        allow = monitor.allow_request(agent_id)
        
        # Get circuit breaker stats if available
        stats = {}
        if agent_id in monitor._circuit_breakers:
            cb = monitor._circuit_breakers[agent_id]
            stats = cb.get_stats()
        
        return jsonify({
            'success': True,
            'agent_id': agent_id,
            'allow_request': allow,
            'state': stats.get('state', 'unknown'),
            'stats': stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/circuit-breaker/<agent_id>/reset', methods=['POST'])
def reset_circuit_breaker_endpoint(agent_id: str):
    """
    Reset circuit breaker for an agent.
    
    POST /api/failure-monitoring/circuit-breaker/<agent_id>/reset
    
    Returns:
    {
        "success": true,
        "message": "Circuit breaker reset"
    }
    """
    try:
        monitor = get_monitor()
        
        if agent_id in monitor._circuit_breakers:
            monitor._circuit_breakers[agent_id].reset()
            return jsonify({
                'success': True,
                'message': f'Circuit breaker for {agent_id} reset'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Agent {agent_id} not found'
            }), 404
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/record-success/<agent_id>', methods=['POST'])
def record_success_endpoint(agent_id: str):
    """
    Record successful operation for circuit breaker.
    
    POST /api/failure-monitoring/record-success/<agent_id>
    
    Returns:
    {
        "success": true,
        "message": "Success recorded"
    }
    """
    try:
        monitor = get_monitor()
        monitor.record_success(agent_id)
        
        return jsonify({
            'success': True,
            'message': 'Success recorded'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/failure-types', methods=['GET'])
def failure_types_endpoint():
    """
    List all failure types.
    
    GET /api/failure-monitoring/failure-types
    
    Returns:
    {
        "success": true,
        "failure_types": [
            {"value": "role_drift", "category": "specification"},
            ...
        ]
    }
    """
    try:
        from agent_failure_taxonomy import FailureType, FailureCategory
        
        # Map failure types to categories
        type_to_category = {
            # Specification
            FailureType.ROLE_DRIFT: FailureCategory.SPECIFICATION,
            FailureType.UNDERSPECIFICATION: FailureCategory.SPECIFICATION,
            FailureType.OVERSPECIFICATION: FailureCategory.SPECIFICATION,
            # Coordination
            FailureType.INFINITE_LOOP: FailureCategory.COORDINATION,
            FailureType.CONTEXT_OVERFLOW: FailureCategory.COORDINATION,
            FailureType.CONFLICTING_ACTIONS: FailureCategory.COORDINATION,
            FailureType.CASCADING_FAILURE: FailureCategory.COORDINATION,
            # Infrastructure
            FailureType.TOOL_ERROR: FailureCategory.INFRASTRUCTURE,
            FailureType.MEMORY_EXHAUSTION: FailureCategory.INFRASTRUCTURE,
            FailureType.TIMEOUT: FailureCategory.INFRASTRUCTURE,
            FailureType.RESOURCE_CONTENTION: FailureCategory.INFRASTRUCTURE,
            # Cognitive
            FailureType.STUCK_AGENT: FailureCategory.COGNITIVE,
            FailureType.CONFUSED_AGENT: FailureCategory.COGNITIVE,
            FailureType.DUNNING_KRUGER: FailureCategory.COGNITIVE,
        }
        
        failure_types = []
        for ft in FailureType:
            category = type_to_category.get(ft, FailureCategory.COGNITIVE)
            failure_types.append({
                'value': ft.value,
                'category': category.value
            })
        
        return jsonify({
            'success': True,
            'failure_types': failure_types,
            'count': len(failure_types)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@failure_monitoring_bp.route('/recovery-strategies', methods=['GET'])
def recovery_strategies_endpoint():
    """
    List all recovery strategies.
    
    GET /api/failure-monitoring/recovery-strategies
    
    Returns:
    {
        "success": true,
        "strategies": ["retry", "backoff", "reset_state", ...]
    }
    """
    try:
        from agent_failure_taxonomy import RecoveryStrategy
        
        return jsonify({
            'success': True,
            'strategies': [s.value for s in RecoveryStrategy]
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[FailureMonitoringAPI] Routes initialized at /api/failure-monitoring/*")
