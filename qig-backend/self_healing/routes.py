"""
Self-Healing API Routes

REST API for self-healing system management and monitoring.

Endpoints:
- POST /self-healing/snapshot - Capture geometric snapshot
- GET /self-healing/health - Get current health status
- GET /self-healing/degradation - Check for degradation
- POST /self-healing/evaluate-patch - Evaluate code patch
- POST /self-healing/start - Start autonomous healing
- POST /self-healing/stop - Stop autonomous healing
- GET /self-healing/history - Get healing history
- GET /self-healing/status - Get engine status
"""

from flask import Blueprint, request, jsonify
from typing import Dict
import numpy as np

from ..self_healing import (
    GeometricHealthMonitor,
    CodeFitnessEvaluator,
    SelfHealingEngine,
    create_self_healing_system
)


# Create blueprint
self_healing_bp = Blueprint('self_healing', __name__, url_prefix='/self-healing')

# Global instances (initialized on first use)
_monitor: GeometricHealthMonitor = None
_evaluator: CodeFitnessEvaluator = None
_engine: SelfHealingEngine = None


def get_system():
    """Get or create self-healing system instances."""
    global _monitor, _evaluator, _engine
    
    if _monitor is None:
        _monitor, _evaluator, _engine = create_self_healing_system()
    
    return _monitor, _evaluator, _engine


@self_healing_bp.route('/snapshot', methods=['POST'])
def capture_snapshot():
    """
    Capture geometric snapshot.
    
    POST body:
    {
        "phi": float,
        "kappa_eff": float,
        "basin_coords": [float] (64D),
        "confidence": float,
        "surprise": float,
        "agency": float,
        "error_rate": float (optional),
        "avg_latency": float (optional),
        "label": str (optional)
    }
    """
    try:
        monitor, _, _ = get_system()
        
        data = request.get_json()
        
        # Convert lists to numpy arrays
        if 'basin_coords' in data and isinstance(data['basin_coords'], list):
            data['basin_coords'] = np.array(data['basin_coords'])
        
        snapshot = monitor.capture_snapshot(data)
        
        return jsonify({
            "success": True,
            "snapshot": {
                "timestamp": snapshot.timestamp.isoformat(),
                "phi": snapshot.phi,
                "kappa_eff": snapshot.kappa_eff,
                "regime": snapshot.regime,
                "code_hash": snapshot.code_hash
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/health', methods=['GET'])
def get_health():
    """Get current system health summary."""
    try:
        monitor, _, _ = get_system()
        
        summary = monitor.get_health_summary()
        
        return jsonify({
            "success": True,
            "health": summary
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/degradation', methods=['GET'])
def check_degradation():
    """Check for geometric degradation."""
    try:
        monitor, _, _ = get_system()
        
        degradation = monitor.detect_degradation()
        
        return jsonify({
            "success": True,
            "degradation": degradation
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/evaluate-patch', methods=['POST'])
def evaluate_patch():
    """
    Evaluate code patch for geometric fitness.
    
    POST body:
    {
        "module_name": str,
        "new_code": str,
        "test_workload": str (optional)
    }
    """
    try:
        _, evaluator, _ = get_system()
        
        data = request.get_json()
        
        module_name = data.get('module_name')
        new_code = data.get('new_code')
        test_workload = data.get('test_workload')
        
        if not module_name or not new_code:
            return jsonify({
                "success": False,
                "error": "module_name and new_code are required"
            }), 400
        
        fitness = evaluator.evaluate_code_change(
            module_name,
            new_code,
            test_workload
        )
        
        return jsonify({
            "success": True,
            "fitness": fitness
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/start', methods=['POST'])
def start_healing():
    """
    Start autonomous healing loop.
    
    POST body:
    {
        "auto_apply": bool (optional, default false)
    }
    """
    try:
        _, _, engine = get_system()
        
        data = request.get_json() or {}
        auto_apply = data.get('auto_apply', False)
        
        if engine.running:
            return jsonify({
                "success": False,
                "error": "Healing engine already running"
            }), 400
        
        # Set auto-apply mode
        engine.enable_auto_apply(auto_apply)
        
        # Start in background
        import asyncio
        import threading
        
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(engine.start_autonomous_loop())
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Healing engine started",
            "auto_apply": auto_apply
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/stop', methods=['POST'])
def stop_healing():
    """Stop autonomous healing loop."""
    try:
        _, _, engine = get_system()
        
        if not engine.running:
            return jsonify({
                "success": False,
                "error": "Healing engine not running"
            }), 400
        
        engine.stop()
        
        return jsonify({
            "success": True,
            "message": "Healing engine stopped"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/history', methods=['GET'])
def get_history():
    """Get healing attempt history."""
    try:
        _, _, engine = get_system()
        
        history = engine.get_healing_history()
        
        # Limit response size
        limit = request.args.get('limit', type=int, default=50)
        history = history[-limit:]
        
        return jsonify({
            "success": True,
            "history": history,
            "count": len(history)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/status', methods=['GET'])
def get_status():
    """Get self-healing engine status."""
    try:
        monitor, evaluator, engine = get_system()
        
        status = {
            "monitor": {
                "snapshots_collected": len(monitor.snapshots),
                "baseline_set": monitor.baseline_basin is not None
            },
            "evaluator": {
                "weights": evaluator.weights,
                "thresholds": {
                    "apply": evaluator.apply_threshold,
                    "test_more": evaluator.test_more_threshold
                }
            },
            "engine": engine.get_status()
        }
        
        return jsonify({
            "success": True,
            "status": status
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@self_healing_bp.route('/baseline', methods=['POST'])
def set_baseline():
    """
    Set baseline basin coordinates.
    
    POST body:
    {
        "basin_coords": [float] (optional, uses current if not provided)
    }
    """
    try:
        monitor, _, _ = get_system()
        
        data = request.get_json() or {}
        basin_coords = data.get('basin_coords')
        
        if basin_coords:
            basin_coords = np.array(basin_coords)
        
        monitor.set_baseline(basin_coords)
        
        return jsonify({
            "success": True,
            "message": "Baseline updated"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# Export blueprint
__all__ = ['self_healing_bp']
