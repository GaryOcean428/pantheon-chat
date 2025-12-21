"""
Autonomous Curiosity API Routes

Exposes the autonomous learning system to the frontend.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime

curiosity_bp = Blueprint('curiosity', __name__)


def _get_engine():
    """Get the curiosity engine singleton."""
    from autonomous_curiosity import get_curiosity_engine
    return get_curiosity_engine()


@curiosity_bp.route('/status', methods=['GET'])
def get_status():
    """Get autonomous learning system status."""
    engine = _get_engine()
    return jsonify({
        'status': 'running' if engine.running else 'stopped',
        'stats': engine.get_stats(),
        'timestamp': datetime.now().isoformat()
    })


@curiosity_bp.route('/start', methods=['POST'])
def start_learning():
    """Start the autonomous learning loop."""
    engine = _get_engine()
    
    if engine.running:
        return jsonify({
            'success': False,
            'message': 'Autonomous learning already running'
        })
    
    engine.start()
    
    return jsonify({
        'success': True,
        'message': 'Autonomous learning started'
    })


@curiosity_bp.route('/stop', methods=['POST'])
def stop_learning():
    """Stop the autonomous learning loop."""
    engine = _get_engine()
    
    if not engine.running:
        return jsonify({
            'success': False,
            'message': 'Autonomous learning not running'
        })
    
    engine.stop()
    
    return jsonify({
        'success': True,
        'message': 'Autonomous learning stopped'
    })


@curiosity_bp.route('/request', methods=['POST'])
def submit_request():
    """Submit a search request from a kernel."""
    data = request.get_json() or {}
    
    kernel_name = data.get('kernel', 'unknown')
    query = data.get('query', '')
    priority = data.get('priority', 0.5)
    context = data.get('context', {})
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400
    
    engine = _get_engine()
    request_id = engine.request_search(
        kernel_name=kernel_name,
        query=query,
        priority=priority,
        context=context
    )
    
    return jsonify({
        'success': True,
        'request_id': request_id,
        'message': f'Search request submitted for {kernel_name}'
    })


@curiosity_bp.route('/tool-refinement', methods=['POST'])
def request_tool_refinement():
    """Request refinement of an existing tool."""
    data = request.get_json() or {}
    
    kernel_name = data.get('kernel', 'unknown')
    tool_id = data.get('tool_id', '')
    refinement_type = data.get('refinement_type', 'improve')
    details = data.get('details', {})
    
    if not tool_id:
        return jsonify({
            'success': False,
            'error': 'tool_id is required'
        }), 400
    
    engine = _get_engine()
    request_id = engine.request_tool_refinement(
        kernel_name=kernel_name,
        tool_id=tool_id,
        refinement_type=refinement_type,
        details=details
    )
    
    return jsonify({
        'success': True,
        'request_id': request_id,
        'message': f'Tool refinement request submitted'
    })


@curiosity_bp.route('/curriculum/load', methods=['POST'])
def load_curriculum():
    """Load curriculum from a file."""
    data = request.get_json() or {}
    filepath = data.get('filepath', '')
    
    if not filepath:
        return jsonify({
            'success': False,
            'error': 'filepath is required'
        }), 400
    
    engine = _get_engine()
    topics = engine.load_curriculum(filepath)
    
    return jsonify({
        'success': True,
        'topics_loaded': len(topics),
        'topics': [t['title'] for t in topics[:10]]
    })


@curiosity_bp.route('/curriculum/status', methods=['GET'])
def curriculum_status():
    """Get curriculum training status."""
    engine = _get_engine()
    loader = engine.curriculum_loader
    
    return jsonify({
        'total_topics': len(loader.curriculum_topics),
        'completed': len(loader.completed_topics),
        'remaining': len(loader.curriculum_topics) - len(loader.completed_topics),
        'completed_list': list(loader.completed_topics)[:20]
    })


@curiosity_bp.route('/explorations', methods=['GET'])
def get_explorations():
    """Get recent exploration results."""
    engine = _get_engine()
    
    limit = request.args.get('limit', 50, type=int)
    kernel = request.args.get('kernel', None)
    
    results = list(engine.exploration_results)
    
    if kernel:
        results = [r for r in results if r.get('kernel') == kernel]
    
    results = results[-limit:]
    
    return jsonify({
        'explorations': results,
        'total': len(engine.exploration_results),
        'filtered': len(results)
    })


@curiosity_bp.route('/curiosity/<kernel_name>', methods=['GET'])
def get_kernel_curiosity(kernel_name):
    """Get curiosity metrics for a specific kernel."""
    engine = _get_engine()
    
    interests = engine.kernel_interests.get(kernel_name, [])
    
    curiosity_scores = {}
    for topic in interests:
        knowledge = engine._get_current_knowledge(kernel_name, topic)
        score = engine.curiosity_drive.compute_curiosity(topic, knowledge)
        curiosity_scores[topic] = {
            'score': score,
            'knowledge_depth': knowledge.get('depth', 0),
            'recency': knowledge.get('recency', float('inf'))
        }
    
    return jsonify({
        'kernel': kernel_name,
        'interests': interests,
        'curiosity_scores': curiosity_scores
    })


@curiosity_bp.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update curiosity engine configuration."""
    engine = _get_engine()
    
    if request.method == 'GET':
        return jsonify({
            'exploration_interval': engine._exploration_interval,
            'min_curiosity_threshold': engine._min_curiosity_threshold,
            'kernel_interests': engine.kernel_interests
        })
    
    data = request.get_json() or {}
    
    if 'exploration_interval' in data:
        engine._exploration_interval = max(10, int(data['exploration_interval']))
    
    if 'min_curiosity_threshold' in data:
        engine._min_curiosity_threshold = max(0.0, min(1.0, float(data['min_curiosity_threshold'])))
    
    return jsonify({
        'success': True,
        'exploration_interval': engine._exploration_interval,
        'min_curiosity_threshold': engine._min_curiosity_threshold
    })
