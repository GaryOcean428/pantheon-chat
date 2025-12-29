"""
Long-Horizon Task API Routes

Exposes the three core long-horizon task modules:
- Goal Tracking (basin-encoded goal hierarchies)
- Geodesic Efficiency (path efficiency metrics)
- Geometric Error Recovery (stuck detection and backtracking)

QIG-PURE: All operations use Fisher-Rao geometry.
"""

from flask import Blueprint, request, jsonify
import numpy as np
import traceback

long_horizon_bp = Blueprint('long_horizon', __name__, url_prefix='/api/long-horizon')


# =============================================================================
# GOAL TRACKING ENDPOINTS
# =============================================================================

@long_horizon_bp.route('/goals', methods=['GET'])
def get_goals():
    """
    GET /api/long-horizon/goals

    Get all goals in the current hierarchy.
    """
    try:
        from goal_tracking import get_goal_hierarchy

        hierarchy = get_goal_hierarchy()
        summary = hierarchy.get_hierarchy_summary()

        goals = []
        for goal_id, goal in hierarchy.goals.items():
            goals.append({
                'goal_id': goal.goal_id,
                'description': goal.description,
                'completed': goal.completed,
                'completion_threshold': goal.completion_threshold,
                'parent_goal_id': goal.parent_goal_id,
                'subgoal_ids': goal.subgoal_ids,
                'initial_distance': goal.initial_distance,
                'steps_taken': len(goal.progress_trajectory),
            })

        return jsonify({
            'success': True,
            'goals': goals,
            'summary': summary,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/goals', methods=['POST'])
def create_goal():
    """
    POST /api/long-horizon/goals

    Create a new goal with basin coordinates.

    Body:
    {
        "description": "Goal description",
        "basin_coords": [...],  // 64D array
        "parent_id": null,  // Optional parent goal
        "completion_threshold": 0.1
    }
    """
    try:
        from goal_tracking import get_goal_hierarchy

        data = request.get_json() or {}

        if 'description' not in data:
            return jsonify({'error': 'description required'}), 400
        if 'basin_coords' not in data:
            return jsonify({'error': 'basin_coords required'}), 400

        hierarchy = get_goal_hierarchy()

        goal = hierarchy.add_goal(
            description=data['description'],
            basin_coords=np.array(data['basin_coords']),
            parent_id=data.get('parent_id'),
            completion_threshold=data.get('completion_threshold', 0.1)
        )

        return jsonify({
            'success': True,
            'goal_id': goal.goal_id,
            'description': goal.description,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/goals/progress', methods=['POST'])
def track_progress():
    """
    POST /api/long-horizon/goals/progress

    Track progress toward all goals from current basin.

    Body:
    {
        "current_basin": [...]  // 64D array
    }
    """
    try:
        from goal_tracking import get_goal_hierarchy

        data = request.get_json() or {}

        if 'current_basin' not in data:
            return jsonify({'error': 'current_basin required'}), 400

        hierarchy = get_goal_hierarchy()
        current_basin = np.array(data['current_basin'])

        progress = hierarchy.track_all_progress(current_basin)

        return jsonify({
            'success': True,
            'progress': {
                goal_id: {
                    'progress': float(p['progress']),
                    'distance_remaining': float(p['distance_remaining']),
                    'stuck': p['stuck'],
                    'completed': p['completed'],
                    'steps_taken': p['steps_taken'],
                }
                for goal_id, p in progress.items()
            },
            'summary': hierarchy.get_hierarchy_summary(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/goals/reset', methods=['POST'])
def reset_goals():
    """
    POST /api/long-horizon/goals/reset

    Reset the goal hierarchy.
    """
    try:
        from goal_tracking import reset_goal_hierarchy

        hierarchy = reset_goal_hierarchy()

        return jsonify({
            'success': True,
            'message': 'Goal hierarchy reset',
            'summary': hierarchy.get_hierarchy_summary(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# GEODESIC EFFICIENCY ENDPOINTS
# =============================================================================

@long_horizon_bp.route('/efficiency/stats', methods=['GET'])
def get_efficiency_stats():
    """
    GET /api/long-horizon/efficiency/stats

    Get geodesic efficiency statistics.
    """
    try:
        from geodesic_efficiency import get_efficiency_tracker

        tracker = get_efficiency_tracker()
        stats = tracker.get_efficiency_stats()
        by_type = tracker.get_efficiency_by_type()

        return jsonify({
            'success': True,
            'stats': {
                'count': stats['count'],
                'mean_efficiency': float(stats['mean_efficiency']),
                'min_efficiency': float(stats['min_efficiency']),
                'max_efficiency': float(stats['max_efficiency']),
                'std_efficiency': float(stats['std_efficiency']),
            },
            'by_operation_type': {
                op_type: {
                    'count': s['count'],
                    'mean_efficiency': float(s['mean_efficiency']),
                }
                for op_type, s in by_type.items()
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/efficiency/record', methods=['POST'])
def record_efficiency():
    """
    POST /api/long-horizon/efficiency/record

    Record an operation's efficiency.

    Body:
    {
        "operation_id": "unique_id",
        "operation_type": "query|search|reasoning|...",
        "path": [[...], [...], ...]  // List of 64D basin coordinates
    }
    """
    try:
        from geodesic_efficiency import get_efficiency_tracker

        data = request.get_json() or {}

        if 'operation_id' not in data:
            return jsonify({'error': 'operation_id required'}), 400
        if 'operation_type' not in data:
            return jsonify({'error': 'operation_type required'}), 400
        if 'path' not in data:
            return jsonify({'error': 'path required'}), 400

        tracker = get_efficiency_tracker()

        path = [np.array(p) for p in data['path']]

        record = tracker.record_operation(
            operation_id=data['operation_id'],
            operation_type=data['operation_type'],
            path=path,
            metadata=data.get('metadata')
        )

        return jsonify({
            'success': True,
            'efficiency': float(record.efficiency),
            'optimal_distance': float(record.optimal_distance),
            'actual_distance': float(record.actual_distance),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/efficiency/degradation', methods=['GET'])
def check_degradation():
    """
    GET /api/long-horizon/efficiency/degradation

    Check for efficiency degradation.
    """
    try:
        from geodesic_efficiency import get_efficiency_tracker

        tracker = get_efficiency_tracker()
        degradation = tracker.detect_efficiency_degradation()

        return jsonify({
            'success': True,
            'degradation': degradation,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# ERROR RECOVERY ENDPOINTS
# =============================================================================

@long_horizon_bp.route('/recovery/status', methods=['GET'])
def get_recovery_status():
    """
    GET /api/long-horizon/recovery/status

    Get error recovery system status.
    """
    try:
        from geometric_error_recovery import get_recovery_system

        system = get_recovery_system()
        stats = system.get_stats()

        return jsonify({
            'success': True,
            'stats': {
                'total_steps': stats['total_steps'],
                'trajectory_length': stats['trajectory_length'],
                'checkpoint_count': stats['checkpoint_count'],
                'recovery_count': stats['recovery_count'],
                'avg_checkpoint_score': float(stats['avg_checkpoint_score']),
            },
            'recent_recoveries': stats['recent_recoveries'],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/recovery/record', methods=['POST'])
def record_state():
    """
    POST /api/long-horizon/recovery/record

    Record current state for recovery tracking.

    Body:
    {
        "basin": [...],  // 64D array
        "phi": 0.7,
        "kappa": 60.0
    }
    """
    try:
        from geometric_error_recovery import get_recovery_system

        data = request.get_json() or {}

        if 'basin' not in data:
            return jsonify({'error': 'basin required'}), 400
        if 'phi' not in data:
            return jsonify({'error': 'phi required'}), 400
        if 'kappa' not in data:
            return jsonify({'error': 'kappa required'}), 400

        system = get_recovery_system()

        system.record_state(
            basin=np.array(data['basin']),
            phi=data['phi'],
            kappa=data['kappa'],
            context=data.get('context')
        )

        return jsonify({
            'success': True,
            'step_count': system.step_counter,
            'checkpoint_count': len(system.checkpoints),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/recovery/check', methods=['GET'])
def check_stuck():
    """
    GET /api/long-horizon/recovery/check

    Check if system is stuck and needs recovery.
    """
    try:
        from geometric_error_recovery import get_recovery_system

        system = get_recovery_system()
        is_stuck, reason, diagnostics = system.detect_stuck()

        return jsonify({
            'success': True,
            'is_stuck': is_stuck,
            'reason': reason,
            'diagnostics': {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in diagnostics.items()
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/recovery/recover', methods=['POST'])
def execute_recovery():
    """
    POST /api/long-horizon/recovery/recover

    Execute recovery if stuck.
    """
    try:
        from geometric_error_recovery import get_recovery_system

        system = get_recovery_system()
        result = system.recover()

        if result is None:
            return jsonify({
                'success': True,
                'recovered': False,
                'message': 'System not stuck, no recovery needed',
            })

        return jsonify({
            'success': True,
            'recovered': result['success'],
            'stuck_reason': result.get('stuck_reason'),
            'recovery_basin': result.get('recovery_basin', np.zeros(64)).tolist() if result.get('success') else None,
            'steps_back': result.get('steps_back'),
            'recovery_count': result.get('recovery_count'),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@long_horizon_bp.route('/recovery/reset', methods=['POST'])
def reset_recovery():
    """
    POST /api/long-horizon/recovery/reset

    Reset the recovery system.
    """
    try:
        from geometric_error_recovery import get_recovery_system

        system = get_recovery_system()
        system.reset()

        return jsonify({
            'success': True,
            'message': 'Recovery system reset',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


print("[LongHorizon] API routes initialized at /api/long-horizon/*")
