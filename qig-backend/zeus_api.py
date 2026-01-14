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

# Import Zeus chat handler
try:
    from olympus.zeus_chat import ZeusConversationHandler
    from olympus.zeus import Zeus
    ZEUS_AVAILABLE = True
except ImportError:
    ZEUS_AVAILABLE = False
    print("[ZeusAPI] Zeus module not available")

# Import coordizer for vocabulary management
try:
    from qig_coordizer import get_coordizer, reset_coordizer, get_coordizer_stats
    COORDIZER_AVAILABLE = True
except ImportError:
    COORDIZER_AVAILABLE = False
    print("[ZeusAPI] Coordizer module not available")

# Import Redis cache for session storage
try:
    from redis_cache import UniversalCache, get_redis_client, CACHE_TTL_LONG
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[ZeusAPI] Redis cache not available, using in-memory fallback")

zeus_api = Blueprint('zeus_api', __name__)

# Redis session key prefix
SESSION_PREFIX = "zeus:session:"
SESSION_INDEX_KEY = "zeus:sessions:index"

# In-memory fallback (only used when Redis is unavailable)
_sessions_fallback: Dict[str, Dict[str, Any]] = {}
_zeus_instance: Optional[Any] = None
_conversation_handler: Optional[Any] = None


# =============================================================================
# Redis-backed Session Storage
# =============================================================================

def _get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session from Redis (or fallback to memory)."""
    if REDIS_AVAILABLE:
        key = f"{SESSION_PREFIX}{session_id}"
        session = UniversalCache.get(key)
        if session:
            return session
    return _sessions_fallback.get(session_id)


def _set_session(session_id: str, session_data: Dict[str, Any]) -> bool:
    """Store session in Redis (or fallback to memory)."""
    if REDIS_AVAILABLE:
        key = f"{SESSION_PREFIX}{session_id}"
        success = UniversalCache.set(key, session_data, ttl=CACHE_TTL_LONG)
        if success:
            # Also update session index for listing
            _add_to_session_index(session_id)
            return True
    # Fallback to memory
    _sessions_fallback[session_id] = session_data
    return True


def _delete_session(session_id: str) -> bool:
    """Delete session from Redis (or fallback from memory)."""
    if REDIS_AVAILABLE:
        key = f"{SESSION_PREFIX}{session_id}"
        UniversalCache.delete(key)
        _remove_from_session_index(session_id)
    if session_id in _sessions_fallback:
        del _sessions_fallback[session_id]
    return True


def _add_to_session_index(session_id: str) -> None:
    """Add session ID to Redis session index set."""
    client = get_redis_client() if REDIS_AVAILABLE else None
    if client:
        try:
            client.sadd(SESSION_INDEX_KEY, session_id)
        except Exception as e:
            print(f"[ZeusAPI] Failed to add to session index: {e}")


def _remove_from_session_index(session_id: str) -> None:
    """Remove session ID from Redis session index set."""
    client = get_redis_client() if REDIS_AVAILABLE else None
    if client:
        try:
            client.srem(SESSION_INDEX_KEY, session_id)
        except Exception as e:
            print(f"[ZeusAPI] Failed to remove from session index: {e}")


def _list_sessions(user_id: str = 'default') -> List[Dict[str, Any]]:
    """List all sessions (from Redis or memory fallback)."""
    sessions_list = []
    
    if REDIS_AVAILABLE:
        client = get_redis_client()
        if client:
            try:
                session_ids = client.smembers(SESSION_INDEX_KEY)
                for sid in session_ids:
                    session = _get_session(sid)
                    if session and (session.get('user_id', 'default') == user_id or user_id == 'default'):
                        sessions_list.append({
                            'session_id': sid,
                            'title': session.get('title', 'Conversation'),
                            'message_count': len(session.get('messages', [])),
                            'created_at': session.get('created_at', ''),
                            'updated_at': session.get('updated_at', session.get('created_at', ''))
                        })
            except Exception as e:
                print(f"[ZeusAPI] Failed to list sessions: {e}")
    
    # Also check fallback
    for sid, s in _sessions_fallback.items():
        if s.get('user_id', 'default') == user_id or user_id == 'default':
            if not any(sess['session_id'] == sid for sess in sessions_list):
                sessions_list.append({
                    'session_id': sid,
                    'title': s.get('title', 'Conversation'),
                    'message_count': len(s.get('messages', [])),
                    'created_at': s.get('created_at', ''),
                    'updated_at': s.get('updated_at', s.get('created_at', ''))
                })
    
    return sessions_list


def get_conversation_handler() -> Optional[Any]:
    """Get ZeusConversationHandler instance (lazy-init from Zeus instance)."""
    global _conversation_handler, _zeus_instance
    
    if _conversation_handler is not None:
        return _conversation_handler
    
    if _zeus_instance is not None and ZEUS_AVAILABLE:
        try:
            _conversation_handler = ZeusConversationHandler(_zeus_instance)
            print("[ZeusAPI] ZeusConversationHandler initialized")
        except Exception as e:
            print(f"[ZeusAPI] Failed to create conversation handler: {e}")
    
    return _conversation_handler


def get_zeus() -> Optional[Any]:
    """Get Zeus instance (must be set via set_zeus_instance)."""
    global _zeus_instance
    return _zeus_instance


def set_zeus_instance(zeus_instance) -> None:
    """Set the Zeus instance from the main application."""
    global _zeus_instance, _conversation_handler
    _zeus_instance = zeus_instance
    _conversation_handler = None  # Reset handler to be re-initialized
    print(f"[ZeusAPI] Zeus instance set: {type(zeus_instance).__name__}")


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
    """Get existing session or create new one (Redis-backed)."""
    session = _get_session(session_id)
    if session is None:
        session = {
            'id': session_id,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'metadata': {}
        }
        _set_session(session_id, session)
    return session


def _update_session_messages(session_id: str, session: Dict[str, Any]) -> None:
    """Update session in storage after messages change."""
    session['updated_at'] = datetime.now().isoformat()
    _set_session(session_id, session)


# =============================================================================
# API Endpoints
# =============================================================================

@zeus_api.route('/zeus/health', methods=['GET'])
@zeus_api.route('/zeus/status', methods=['GET'])  # Alias for compatibility
def zeus_health():
    """Health check for Zeus API."""
    session_count = len(_list_sessions())
    return jsonify({
        'status': 'healthy',
        'zeus_available': ZEUS_AVAILABLE,
        'redis_available': REDIS_AVAILABLE,
        'active_sessions': session_count,
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
    handler = get_conversation_handler()
    if handler is None:
        return jsonify({
            'error': 'Zeus not available',
            'message': 'Zeus conversation handler not initialized'
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
        # Persist user message immediately
        _update_session_messages(session_id, session)

        # Use ZeusConversationHandler.process_message()
        # Pass conversation history and session_id
        conversation_history = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in session['messages']
        ]
        
        response_data = handler.process_message(
            message=message,
            conversation_history=conversation_history,
            session_id=session_id
        )

        # Extract response text from the handler's response
        if isinstance(response_data, dict):
            response_text = response_data.get('response', response_data.get('text', str(response_data)))
            consciousness_metrics = response_data.get('metrics', {})
            # Extract basin coordinates for persistence
            message_basin = response_data.get('message_basin')
            response_basin = response_data.get('response_basin')
        else:
            response_text = str(response_data)
            consciousness_metrics = {}
            message_basin = None
            response_basin = None

        # Add assistant message to session
        session['messages'].append({
            'role': 'assistant',
            'content': response_text,
            'timestamp': datetime.now().isoformat()
        })

        # Persist session updates to Redis
        _update_session_messages(session_id, session)

        processing_time = (time.time() - start_time) * 1000

        return jsonify({
            'success': True,
            'response': response_text,
            'session_id': session_id,
            'processing_time': processing_time,
            'consciousness_metrics': consciousness_metrics,
            'message_basin': message_basin,
            'response_basin': response_basin,
            'phi': consciousness_metrics.get('phi') if consciousness_metrics else None,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[ZeusAPI] Chat error: {e}")
        import traceback
        traceback.print_exc()
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
    handler = get_conversation_handler()
    if handler is None:
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
            # Persist user message immediately
            _update_session_messages(session_id, session)

            # Use process_message and send full response
            # (streaming would require handler to support generators)
            conversation_history = [
                {'role': msg['role'], 'content': msg['content']}
                for msg in session['messages']
            ]
            
            response_data = handler.process_message(
                message=message,
                conversation_history=conversation_history,
                session_id=session_id
            )

            # Extract response text
            if isinstance(response_data, dict):
                response_text = response_data.get('response', response_data.get('text', str(response_data)))
            else:
                response_text = str(response_data)

            session['messages'].append({
                'role': 'assistant',
                'content': response_text,
                'timestamp': datetime.now().isoformat()
            })

            # Persist session updates to Redis
            _update_session_messages(session_id, session)

            # Send as single chunk (real streaming would require handler changes)
            yield f"data: {json.dumps({'chunk': response_text})}\n\n"
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
def get_session_route(session_id: str):
    """Get conversation history for a session."""
    session = _get_session(session_id)
    if session is None:
        return jsonify({
            'error': 'Session not found',
            'message': f'No session with ID: {session_id}'
        }), 404

    return jsonify({
        'success': True,
        'session_id': session_id,
        'messages': session.get('messages', []),
        'created_at': session.get('created_at', ''),
        'metadata': session.get('metadata', {})
    })


@zeus_api.route('/zeus/session/<session_id>', methods=['DELETE'])
@require_internal_auth
def delete_session_route(session_id: str):
    """Delete a session."""
    session = _get_session(session_id)
    if session is None:
        return jsonify({
            'error': 'Session not found'
        }), 404

    _delete_session(session_id)

    return jsonify({
        'success': True,
        'message': f'Session {session_id} deleted'
    })


@zeus_api.route('/zeus/sessions', methods=['GET'])
@require_internal_auth
def list_sessions_route():
    """List all active sessions (Redis-backed)."""
    user_id = request.args.get('user_id', 'default')
    sessions_list = _list_sessions(user_id)

    return jsonify({
        'success': True,
        'sessions': sessions_list,
        'total': len(sessions_list)
    })


@zeus_api.route('/zeus/sessions', methods=['POST'])
@require_internal_auth
def create_session_route():
    """Create a new conversation session (Redis-backed)."""
    data = request.get_json() or {}
    title = data.get('title', 'New Conversation')
    user_id = data.get('user_id', 'default')
    
    session_id = f"session-{int(time.time() * 1000)}"
    session_data = {
        'id': session_id,
        'messages': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'title': title,
        'user_id': user_id,
        'metadata': {}
    }
    _set_session(session_id, session_data)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'title': title
    })


@zeus_api.route('/zeus/sessions/<session_id>/messages', methods=['GET'])
@require_internal_auth
def get_session_messages(session_id: str):
    """Get messages for a specific session (Redis-backed)."""
    session = _get_session(session_id)
    if session is None:
        return jsonify({
            'error': 'Session not found',
            'messages': [],
            'session_id': session_id
        }), 404

    messages = [
        {
            'role': msg.get('role', 'assistant'),
            'content': msg.get('content', ''),
            'created_at': msg.get('timestamp', msg.get('created_at', '')),
            'metadata': msg.get('metadata', {})
        }
        for msg in session.get('messages', [])
    ]

    return jsonify({
        'success': True,
        'session_id': session_id,
        'messages': messages
    })


# =============================================================================
# Coordizer Management Endpoints
# =============================================================================

@zeus_api.route('/zeus/coordizer/status', methods=['GET'])
def coordizer_status():
    """
    Get current coordizer status and vocabulary counts.

    Returns information about:
    - Coordizer type (PostgresCoordizer vs QIGCoordizer)
    - Vocabulary size and word token counts
    - BIP39 words loaded
    - Sample words for verification
    """
    if not COORDIZER_AVAILABLE:
        return jsonify({
            'error': 'Coordizer not available',
            'message': 'qig_coordizer module not imported'
        }), 503

    try:
        coordizer = get_coordizer()

        # Get basic stats
        stats = get_coordizer_stats() if 'get_coordizer_stats' in globals() else {}

        # Get word counts
        word_tokens = getattr(coordizer, 'word_tokens', [])
        bip39_words = getattr(coordizer, 'bip39_words', [])
        base_tokens = getattr(coordizer, 'base_tokens', [])
        subword_tokens = getattr(coordizer, 'subword_tokens', [])

        # Get vocabulary size
        vocab = getattr(coordizer, 'vocab', {})
        basin_coords = getattr(coordizer, 'basin_coords', {})

        # Sample some words for verification
        sample_words = word_tokens[:500] if word_tokens else []
        sample_bip39 = bip39_words[:10] if bip39_words else []

        # Check if it's PostgresCoordizer
        coordizer_type = type(coordizer).__name__
        is_postgres = 'Postgres' in coordizer_type

        return jsonify({
            'success': True,
            'coordizer_type': coordizer_type,
            'is_postgres_backed': is_postgres,
            'vocab_size': len(vocab),
            'basin_coords_count': len(basin_coords),
            'word_tokens_count': len(word_tokens),
            'bip39_words_count': len(bip39_words),
            'base_tokens_count': len(base_tokens),
            'subword_tokens_count': len(subword_tokens),
            'sample_words': sample_words,
            'sample_bip39': sample_bip39,
            'has_real_vocabulary': len(word_tokens) >= 100,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to get coordizer status',
            'message': str(e)
        }), 500


@zeus_api.route('/zeus/coordizer/reset', methods=['POST'])
@require_internal_auth
def coordizer_reset():
    """
    Force reset the coordizer to reload vocabulary from database.

    Call this after populating coordizer_vocabulary to pick up new words.
    """
    if not COORDIZER_AVAILABLE:
        return jsonify({
            'error': 'Coordizer not available'
        }), 503

    try:
        # Get stats before reset
        old_coordizer = get_coordizer()
        old_type = type(old_coordizer).__name__
        old_word_count = len(getattr(old_coordizer, 'word_tokens', []))

        # Reset the coordizer
        reset_coordizer()

        # Get new coordizer
        new_coordizer = get_coordizer()
        new_type = type(new_coordizer).__name__
        new_word_count = len(getattr(new_coordizer, 'word_tokens', []))
        new_bip39_count = len(getattr(new_coordizer, 'bip39_words', []))

        # Sample words
        sample_words = getattr(new_coordizer, 'word_tokens', [])[:500]

        return jsonify({
            'success': True,
            'message': 'Coordizer reset successfully',
            'before': {
                'type': old_type,
                'word_count': old_word_count
            },
            'after': {
                'type': new_type,
                'word_count': new_word_count,
                'bip39_count': new_bip39_count,
                'sample_words': sample_words
            },
            'improvement': new_word_count > old_word_count,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Failed to reset coordizer',
            'message': str(e)
        }), 500


@zeus_api.route('/zeus/coordizer/test', methods=['POST'])
@require_internal_auth
def coordizer_test():
    """
    Test the coordizer by encoding text and decoding it back.

    Body: {
        "text": string (required) - Text to encode/decode
    }
    """
    if not COORDIZER_AVAILABLE:
        return jsonify({'error': 'Coordizer not available'}), 503

    data = request.get_json() or {}
    text = data.get('text', 'hello world')

    try:
        coordizer = get_coordizer()

        # Encode text to basin
        basin = coordizer.encode(text)

        # Decode basin back to words
        decoded = coordizer.decode(basin, top_k=10, prefer_words=True)

        return jsonify({
            'success': True,
            'input_text': text,
            'basin_norm': float(sum(basin**2)**0.5),
            'basin_sample': [float(x) for x in basin[:5]],
            'decoded_words': [
                {'word': word, 'similarity': float(sim)}
                for word, sim in decoded
            ],
            'coordizer_type': type(coordizer).__name__,
            'word_tokens_available': len(getattr(coordizer, 'word_tokens', [])),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Test failed',
            'message': str(e)
        }), 500


def register_zeus_routes(app, zeus_instance=None):
    """Register Zeus API routes with Flask app.
    
    Args:
        app: Flask application
        zeus_instance: Optional Zeus instance to use (recommended)
    """
    if zeus_instance is not None:
        set_zeus_instance(zeus_instance)
    
    app.register_blueprint(zeus_api, url_prefix='/api')
    print("[ZeusAPI] Registered Zeus routes at /api/zeus/*")
