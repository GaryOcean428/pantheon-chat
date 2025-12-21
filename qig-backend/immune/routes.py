"""
Immune System API Routes

Provides HTTP endpoints for immune system monitoring and control.
"""

from flask import Blueprint, request, jsonify
from typing import Optional
from . import get_immune_system

immune_bp = Blueprint('immune', __name__, url_prefix='/immune')

_immune_system = None

def get_system():
    """Get lazy-loaded immune system instance."""
    global _immune_system
    if _immune_system is None:
        _immune_system = get_immune_system()
    return _immune_system


@immune_bp.route('/status', methods=['GET'])
def immune_status():
    """Get immune system status and health metrics."""
    try:
        system = get_system()
        return jsonify({
            'active': True,
            'status': system.get_status(),
            'classifier_stats': system.classifier.get_stats(),
            'validator_stats': system.validator.get_stats(),
            'threat_summary': system.response.get_threat_summary()
        })
    except Exception as e:
        return jsonify({'error': str(e), 'active': False}), 500


@immune_bp.route('/health', methods=['GET'])
def immune_health():
    """Get self-healing health status."""
    try:
        system = get_system()
        return jsonify(system.healing.get_health_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/threats', methods=['GET'])
def immune_threats():
    """Get recent threat summary."""
    try:
        system = get_system()
        return jsonify(system.response.get_threat_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/antibodies', methods=['GET'])
def list_antibodies():
    """List active antibodies."""
    try:
        system = get_system()
        antibodies = system.response.antibody_generator.get_active_antibodies()
        return jsonify({
            'count': len(antibodies),
            'antibodies': antibodies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/whitelist', methods=['POST'])
def add_to_whitelist():
    """Add IP to whitelist."""
    try:
        data = request.get_json() or {}
        ip = data.get('ip')
        
        if not ip:
            return jsonify({'error': 'IP address required'}), 400
        
        system = get_system()
        system.classifier.add_to_whitelist(ip)
        
        return jsonify({
            'success': True,
            'ip': ip,
            'action': 'whitelisted'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/blacklist', methods=['POST'])
def add_to_blacklist():
    """Add IP to blacklist."""
    try:
        data = request.get_json() or {}
        ip = data.get('ip')
        reason = data.get('reason', 'manual')
        
        if not ip:
            return jsonify({'error': 'IP address required'}), 400
        
        system = get_system()
        system.classifier.add_to_blacklist(ip, reason)
        
        return jsonify({
            'success': True,
            'ip': ip,
            'action': 'blacklisted',
            'reason': reason
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/blacklist/<ip>', methods=['DELETE'])
def remove_from_blacklist(ip: str):
    """Remove IP from blacklist."""
    try:
        system = get_system()
        system.classifier.remove_from_blacklist(ip)
        
        return jsonify({
            'success': True,
            'ip': ip,
            'action': 'removed_from_blacklist'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/offensive/operations', methods=['GET'])
def list_operations():
    """List active offensive operations."""
    try:
        system = get_system()
        return jsonify({
            'active': system.offensive.get_active_operations(),
            'stats': system.offensive.get_operation_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/offensive/initiate', methods=['POST'])
def initiate_countermeasure():
    """Initiate countermeasure against target."""
    try:
        data = request.get_json() or {}
        signature = data.get('signature', {})
        severity = data.get('severity', 'medium')
        
        if severity not in ['low', 'medium', 'high', 'critical']:
            return jsonify({'error': 'Invalid severity level'}), 400
        
        system = get_system()
        operation = system.offensive.initiate_countermeasure(signature, severity)
        
        return jsonify(operation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/offensive/operations/<operation_id>', methods=['DELETE'])
def cancel_operation(operation_id: str):
    """Cancel an active operation."""
    try:
        system = get_system()
        success = system.offensive.cancel_operation(operation_id)
        
        return jsonify({
            'success': success,
            'operation_id': operation_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available health checkpoints."""
    try:
        system = get_system()
        checkpoints = system.healing.checkpoints[-10:]
        return jsonify({
            'count': len(system.healing.checkpoints),
            'recent': checkpoints
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/checkpoints', methods=['POST'])
def create_checkpoint():
    """Create a new health checkpoint."""
    try:
        data = request.get_json() or {}
        state = data.get('state', {})
        label = data.get('label', '')
        
        system = get_system()
        checkpoint_id = system.healing.create_checkpoint(state, label)
        
        return jsonify({
            'success': True,
            'checkpoint_id': checkpoint_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/checkpoints/<checkpoint_id>/restore', methods=['POST'])
def restore_checkpoint(checkpoint_id: str):
    """Restore system from checkpoint."""
    try:
        system = get_system()
        result = system.healing.restore_from_checkpoint(checkpoint_id)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Checkpoint not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@immune_bp.route('/analyze', methods=['POST'])
def analyze_request():
    """Analyze a request for threat detection (testing endpoint)."""
    try:
        data = request.get_json() or {}
        test_request = {
            'ip': data.get('ip', request.remote_addr),
            'path': data.get('path', '/test'),
            'method': data.get('method', 'GET'),
            'headers': data.get('headers', dict(request.headers)),
            'params': data.get('params', {}),
            'body': data.get('body', {}),
            'geo': data.get('geo', {})
        }
        
        system = get_system()
        signature = system.extractor.extract_request_signature(test_request)
        validation = system.validator.validate(signature)
        decision = system.classifier.classify_request(test_request, signature, validation)
        
        return jsonify({
            'signature': signature,
            'validation': validation,
            'decision': decision
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def register_immune_routes(app):
    """Register immune system routes with Flask app."""
    app.register_blueprint(immune_bp)
    print("[ImmuneSystem] API routes registered at /immune")
