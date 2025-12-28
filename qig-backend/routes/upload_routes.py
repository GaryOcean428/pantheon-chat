"""
Upload Routes for Curriculum and Chat RAG

Flask blueprint providing:
- POST /api/uploads/curriculum - Add file to curriculum for learning
- POST /api/uploads/chat - Upload for immediate chat discussion
"""

import base64
import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

upload_bp = Blueprint('uploads', __name__, url_prefix='/api/uploads')


def get_upload_service():
    """Import upload service lazily to avoid circular imports."""
    from upload_service import get_upload_service as get_svc
    return get_svc()


@upload_bp.route('/curriculum', methods=['POST'])
def upload_curriculum():
    """
    Upload file to curriculum for persistent learning.
    
    Expects JSON body:
    - content: base64-encoded file content
    - filename: original filename
    - metadata: optional metadata dict
    
    Returns upload result with path and learning status.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body provided'}), 400
        
        content_b64 = data.get('content')
        filename = data.get('filename')
        metadata = data.get('metadata')
        
        if not content_b64 or not filename:
            return jsonify({'success': False, 'error': 'Missing content or filename'}), 400
        
        try:
            content = base64.b64decode(content_b64)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid base64 content: {e}'}), 400
        
        service = get_upload_service()
        result = service.upload_curriculum(content, filename, metadata)
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"[UploadRoutes] Curriculum upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@upload_bp.route('/chat', methods=['POST'])
def upload_chat():
    """
    Upload file for immediate chat RAG discussion.
    
    Expects JSON body:
    - content: base64-encoded file content
    - filename: original filename
    - add_to_curriculum: optional bool to also add to curriculum
    - session_id: optional chat session ID
    
    Returns upload result with RAG content for discussion.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON body provided'}), 400
        
        content_b64 = data.get('content')
        filename = data.get('filename')
        add_to_curriculum = data.get('add_to_curriculum', False)
        session_id = data.get('session_id')
        
        if not content_b64 or not filename:
            return jsonify({'success': False, 'error': 'Missing content or filename'}), 400
        
        try:
            content = base64.b64decode(content_b64)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid base64 content: {e}'}), 400
        
        service = get_upload_service()
        result = service.upload_chat(
            content, 
            filename, 
            add_to_curriculum=add_to_curriculum,
            session_id=session_id
        )
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"[UploadRoutes] Chat upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@upload_bp.route('/stats', methods=['GET'])
def upload_stats():
    """Get upload statistics."""
    try:
        service = get_upload_service()
        stats = service.get_upload_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"[UploadRoutes] Stats error: {e}")
        return jsonify({'error': str(e)}), 500


@upload_bp.route('/curriculum/files', methods=['GET'])
def list_curriculum_files():
    """List all curriculum files."""
    try:
        service = get_upload_service()
        files = service.list_curriculum_files()
        return jsonify({'files': files, 'count': len(files)})
    except Exception as e:
        logger.error(f"[UploadRoutes] List files error: {e}")
        return jsonify({'error': str(e)}), 500


def register_upload_routes(app):
    """Register upload blueprint with Flask app."""
    app.register_blueprint(upload_bp)
    logger.info("[UploadRoutes] Registered upload routes at /api/uploads")
    return 1
