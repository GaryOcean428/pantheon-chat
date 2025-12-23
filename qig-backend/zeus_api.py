#!/usr/bin/env python3
"""
Zeus Chat API Endpoints

Provides Flask routes for Zeus chat functionality:
- /api/zeus/chat - Send message and get response
- /api/zeus/chat/stream - Streaming response (SSE)
- /api/zeus/session/<session_id> - Get session history

All endpoints require X-Internal-Auth header for authentication.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator
from functools import wraps
from flask import Blueprint, jsonify, request, Response

# Import Zeus chat
try:
    from olympus.zeus_chat import ZeusChat
    from olympus.zeus import Zeus
    ZEUS_AVAILABLE = True
except ImportError:
    ZEUS_AVAILABLE = False
    print("[ZeusAPI] Zeus module not available")

zeus_api = Blueprint('zeus_api', __name__)

# Session storage (in production, use Redis)
_sessions: Dict[str, Dict[str, Any]] = {}
_zeus_instance: Optional[Any] = None


def get_zeus() -> Optional[Any]:
    """Get or create Zeus instance."""
    global _zeus_instance
    if _zeus_instance is None and ZEUS_AVAILABLE:
        try:
            _zeus_instance = ZeusChat()
        except Exception as e:
            print(f"[ZeusAPI] Failed to initialize Zeus: {e}")
    return _zeus_instance


def require_internal_auth(f):
    """Decorator to require X-Internal-Auth header."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        internal_key = os.environ.get('INTERNAL_API_KEY')
        if internal_key:
            provided_key = request.headers.get('X-Internal-Auth')
            if provided_key != internal_key:
                return jsonify({
                    'error': 'Unauthorized',
                    'message': 'Invalid or missing X-Internal-Auth header'
                }), 401
        return f(*args, **kwargs)
    return decorated_function


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """Get existing session or create new one."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            'id': session_id,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'metadata': {}
        }
    return _sessions[session_id]


# =============================================================================
# API Endpoints
# =============================================================================

@zeus_api.route('/zeus/health', methods=['GET'])
def zeus_health():
    """Health check for Zeus API."""
    return jsonify({
        'status': 'healthy',
        'zeus_available': ZEUS_AVAILABLE,
        'active_sessions': len(_sessions),
        'timestamp': datetime.now().isoformat()
    })


@zeus_api.route('/zeus/chat', methods=['POST'])
@require_internal_auth
def zeus_chat():
    """
    Send a message to Zeus and get a response.
    
    Body: {
        "message": string (required),
        "session_id": string (optional),
        "context": object (optional),
        "client_name": string (optional)
    }
    """
    zeus = get_zeus()
    if zeus is None:
        return jsonify({
            'error': 'Zeus not available',
            'message': 'Zeus chat module not initialized'
        }), 503
    
    data = request.get_json() or {}
    message = data.get('message', '')
    
    if not message:
        return jsonify({
            'error': 'Message required',
            'message': 'Provide "message" field with user input'
        }), 400
    
    session_id = data.get('session_id', f'session-{int(time.time()*1000)}')
    context = data.get('context', {})
    client_name = data.get('client_name', 'external')
    
    start_time = time.time()
    
    try:
        # Get or create session
        session = get_or_create_session(session_id)
        
        # Add user message to session
        session['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get Zeus response
        response_text = zeus.chat(
            message=message,
            context=context,
            session_id=session_id
        )
        
        # Add assistant message to session
        session['messages'].append({
            'role': 'assistant',
            'content': response_text,
            'timestamp': datetime.now().isoformat()
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        # Get consciousness metrics if available
        consciousness_metrics = {}
        if hasattr(zeus, 'get_consciousness_metrics'):
            consciousness_metrics = zeus.get_consciousness_metrics()
        
        return jsonify({
            'success': True,
            'response': response_text,
            'session_id': session_id,
            'processing_time': processing_time,
            'consciousness_metrics': consciousness_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[ZeusAPI] Chat error: {e}")
        return jsonify({
            'error': 'Chat failed',
            'message': str(e)
        }), 500


@zeus_api.route('/zeus/chat/stream', methods=['POST'])
@require_internal_auth
def zeus_chat_stream():
    """
    Stream a response from Zeus using Server-Sent Events.
    
    Body: {
        "message": string (required),
        "session_id": string (optional),
        "context": object (optional)
    }
    """
    zeus = get_zeus()
    if zeus is None:
        return jsonify({
            'error': 'Zeus not available'
        }), 503
    
    data = request.get_json() or {}
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    session_id = data.get('session_id', f'session-{int(time.time()*1000)}')
    context = data.get('context', {})
    
    def generate() -> Generator[str, None, None]:
        try:
            session = get_or_create_session(session_id)
            session['messages'].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if Zeus supports streaming
            if hasattr(zeus, 'chat_stream'):
                full_response = ""
                for chunk in zeus.chat_stream(message=message, context=context, session_id=session_id):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                session['messages'].append({
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Fallback to non-streaming
                response = zeus.chat(message=message, context=context, session_id=session_id)
                session['messages'].append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                yield f"data: {json.dumps({'chunk': response})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )


@zeus_api.route('/zeus/session/<session_id>', methods=['GET'])
@require_internal_auth
def get_session(session_id: str):
    """Get conversation history for a session."""
    if session_id not in _sessions:
        return jsonify({
            'error': 'Session not found',
            'message': f'No session with ID: {session_id}'
        }), 404
    
    session = _sessions[session_id]
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'messages': session['messages'],
        'created_at': session['created_at'],
        'metadata': session.get('metadata', {})
    })


@zeus_api.route('/zeus/session/<session_id>', methods=['DELETE'])
@require_internal_auth
def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in _sessions:
        return jsonify({
            'error': 'Session not found'
        }), 404
    
    del _sessions[session_id]
    
    return jsonify({
        'success': True,
        'message': f'Session {session_id} deleted'
    })


@zeus_api.route('/zeus/sessions', methods=['GET'])
@require_internal_auth
def list_sessions():
    """List all active sessions."""
    sessions_list = [
        {
            'id': sid,
            'message_count': len(s['messages']),
            'created_at': s['created_at']
        }
        for sid, s in _sessions.items()
    ]
    
    return jsonify({
        'success': True,
        'sessions': sessions_list,
        'total': len(sessions_list)
    })


def register_zeus_routes(app):
    """Register Zeus API routes with Flask app."""
    app.register_blueprint(zeus_api, url_prefix='/api')
    print("[ZeusAPI] Registered Zeus routes at /api/zeus/*")
