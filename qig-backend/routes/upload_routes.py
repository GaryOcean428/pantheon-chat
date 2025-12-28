"""
Upload Routes for Curriculum and Chat RAG

Flask blueprint providing:
- POST /api/uploads/curriculum - Add file to curriculum for learning
- POST /api/uploads/chat - Upload for immediate chat discussion

Accepts both multipart/form-data (from Node.js proxy) and JSON with base64.
"""

import base64
import json
import logging
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

upload_bp = Blueprint('uploads', __name__, url_prefix='/api/uploads')


def get_upload_service():
    """Import upload service lazily to avoid circular imports."""
    from upload_service import get_upload_service as get_svc
    return get_svc()


def extract_file_content(req):
    """
    Extract file content and filename from request.
    Supports both multipart/form-data and JSON with base64.
    
    Returns: (content_bytes, filename, form_data_dict) or raises ValueError
    """
    if req.files and 'file' in req.files:
        file = req.files['file']
        content = file.read()
        filename = file.filename
        form_data = {k: v for k, v in req.form.items()}
        return content, filename, form_data
    
    data = req.get_json(silent=True)
    if data:
        content_b64 = data.get('content')
        filename = data.get('filename')
        if content_b64 and filename:
            try:
                content = base64.b64decode(content_b64)
                return content, filename, data
            except Exception as e:
                raise ValueError(f'Invalid base64 content: {e}')
    
    raise ValueError('No file provided. Send multipart form-data or JSON with base64 content.')


@upload_bp.route('/curriculum', methods=['POST'])
def upload_curriculum():
    """
    Upload file to curriculum for persistent learning.
    
    Accepts:
    - multipart/form-data with 'file' field
    - JSON body with 'content' (base64) and 'filename'
    
    Returns upload result with path and learning status.
    """
    try:
        content, filename, form_data = extract_file_content(request)
        metadata = form_data.get('metadata')
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = None
        
        service = get_upload_service()
        result = service.upload_curriculum(content, filename, metadata)
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"[UploadRoutes] Curriculum upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@upload_bp.route('/chat', methods=['POST'])
def upload_chat():
    """
    Upload file for immediate chat RAG discussion.
    
    Accepts:
    - multipart/form-data with 'file' field
    - JSON body with 'content' (base64), 'filename', 'add_to_curriculum', 'session_id'
    
    Returns upload result with RAG content for discussion.
    """
    try:
        content, filename, form_data = extract_file_content(request)
        
        add_to_curriculum_raw = form_data.get('add_to_curriculum', False)
        if isinstance(add_to_curriculum_raw, str):
            add_to_curriculum = add_to_curriculum_raw.lower() in ('true', '1', 'yes')
        else:
            add_to_curriculum = bool(add_to_curriculum_raw)
        
        session_id = form_data.get('session_id')
        
        service = get_upload_service()
        result = service.upload_chat(
            content, 
            filename, 
            add_to_curriculum=add_to_curriculum,
            session_id=session_id
        )
        
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
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
