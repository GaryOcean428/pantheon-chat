"""
Activity API Routes

Flask routes for the kernel activity stream.
Exposes the ActivityBroadcaster data to the frontend.
"""

from flask import Blueprint, jsonify, request
from .activity_broadcaster import get_broadcaster, ActivityType

activity_bp = Blueprint('activity', __name__, url_prefix='/api/olympus/pantheon')


@activity_bp.route('/activity', methods=['GET'])
def get_activity():
    """
    Get recent kernel activity.
    
    Query params:
    - limit: Max number of activities (default 50)
    - type: Filter by activity type
    - from_god: Filter by source god
    """
    limit = request.args.get('limit', 50, type=int)
    activity_type = request.args.get('type', None)
    from_god = request.args.get('from_god', None)
    
    broadcaster = get_broadcaster()
    activities = broadcaster.get_recent_activity(
        limit=min(limit, 200),  # Cap at 200
        activity_type=activity_type,
        from_god=from_god
    )
    
    return jsonify({
        'success': True,
        'count': len(activities),
        'activities': activities,
        'filters': {
            'limit': limit,
            'type': activity_type,
            'from_god': from_god
        }
    })


@activity_bp.route('/activity/types', methods=['GET'])
def get_activity_types():
    """Get available activity types."""
    return jsonify({
        'success': True,
        'types': [t.value for t in ActivityType]
    })


@activity_bp.route('/activity/gods', methods=['GET'])
def get_active_gods():
    """
    Get list of gods that have recent activity.
    """
    broadcaster = get_broadcaster()
    activities = broadcaster.get_recent_activity(limit=200)
    
    gods = set()
    for activity in activities:
        if activity.get('from_god'):
            gods.add(activity['from_god'])
        if activity.get('to_god'):
            gods.add(activity['to_god'])
    
    return jsonify({
        'success': True,
        'gods': sorted(list(gods))
    })


@activity_bp.route('/activity/stats', methods=['GET'])
def get_activity_stats():
    """
    Get activity statistics.
    """
    broadcaster = get_broadcaster()
    activities = broadcaster.get_recent_activity(limit=500)
    
    # Count by type
    type_counts = {}
    god_counts = {}
    
    for activity in activities:
        act_type = activity.get('type', 'unknown')
        type_counts[act_type] = type_counts.get(act_type, 0) + 1
        
        from_god = activity.get('from_god', 'unknown')
        god_counts[from_god] = god_counts.get(from_god, 0) + 1
    
    return jsonify({
        'success': True,
        'total_activities': len(activities),
        'by_type': type_counts,
        'by_god': god_counts
    })


def register_activity_routes(app):
    """Register activity routes with Flask app."""
    app.register_blueprint(activity_bp)
    print("[ActivityAPI] Routes registered")
