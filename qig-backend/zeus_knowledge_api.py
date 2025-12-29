"""
Zeus Knowledge Integration API Routes

Endpoints:
- POST /api/zeus-knowledge/remember - Store conversation in Zettelkasten
- POST /api/zeus-knowledge/retrieve - Retrieve relevant knowledge for a query
- GET /api/zeus-knowledge/stats - Get knowledge memory statistics
- GET /api/zeus-knowledge/conversation/<id> - Get knowledge from specific conversation

Author: Ocean/Zeus Pantheon
"""

from flask import Blueprint, request, jsonify
import traceback

zeus_knowledge_bp = Blueprint('zeus_knowledge', __name__, url_prefix='/api/zeus-knowledge')


def get_knowledge_memory():
    """Get the ZeusKnowledgeMemory instance."""
    from zeus_knowledge_integration import get_zeus_knowledge_memory
    return get_zeus_knowledge_memory()


@zeus_knowledge_bp.route('/remember', methods=['POST'])
def remember_endpoint():
    """
    Store a conversation exchange in Zettelkasten memory.
    
    POST /api/zeus-knowledge/remember
    {
        "user_message": "What is consciousness?",
        "zeus_response": "Consciousness is...",
        "conversation_id": "conv_123",
        "domain_hints": ["philosophy", "neuroscience"],
        "phi": 0.85
    }
    
    Returns:
    {
        "success": true,
        "stored": true,
        "user_zettel_id": "z_123...",
        "response_zettel_id": "z_456...",
        "links_created": 3
    }
    """
    try:
        data = request.get_json() or {}
        
        user_message = data.get('user_message')
        zeus_response = data.get('zeus_response')
        conversation_id = data.get('conversation_id')
        
        if not user_message:
            return jsonify({'error': 'user_message required'}), 400
        if not zeus_response:
            return jsonify({'error': 'zeus_response required'}), 400
        if not conversation_id:
            return jsonify({'error': 'conversation_id required'}), 400
        
        domain_hints = data.get('domain_hints', [])
        phi = data.get('phi', 0.5)
        
        knowledge = get_knowledge_memory()
        result = knowledge.remember_conversation(
            user_message=user_message,
            zeus_response=zeus_response,
            conversation_id=conversation_id,
            domain_hints=domain_hints,
            phi=phi
        )
        
        return jsonify({
            'success': result.get('stored', False),
            **result
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zeus_knowledge_bp.route('/retrieve', methods=['POST'])
def retrieve_endpoint():
    """
    Retrieve relevant knowledge for a query.
    
    POST /api/zeus-knowledge/retrieve
    {
        "query": "What is consciousness?",
        "max_results": 5,
        "include_context": true,
        "format_for_response": true
    }
    
    Returns:
    {
        "success": true,
        "knowledge_items": [...],
        "formatted_context": "Related past knowledge:\n...",
        "count": 5
    }
    """
    try:
        data = request.get_json() or {}
        
        query = data.get('query')
        if not query:
            return jsonify({'error': 'query required'}), 400
        
        max_results = data.get('max_results', 5)
        include_context = data.get('include_context', True)
        format_for_response = data.get('format_for_response', False)
        
        knowledge = get_knowledge_memory()
        items = knowledge.retrieve_knowledge(
            query=query,
            max_results=max_results,
            include_context=include_context
        )
        
        result = {
            'success': True,
            'knowledge_items': items,
            'count': len(items)
        }
        
        if format_for_response:
            result['formatted_context'] = knowledge.format_context_for_response(items)
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zeus_knowledge_bp.route('/stats', methods=['GET'])
def stats_endpoint():
    """
    Get knowledge memory statistics.
    
    GET /api/zeus-knowledge/stats
    
    Returns:
    {
        "success": true,
        "available": true,
        "total_zettels": 100,
        "total_links": 500,
        ...
    }
    """
    try:
        knowledge = get_knowledge_memory()
        stats = knowledge.get_stats()
        
        return jsonify({
            'success': True,
            **stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zeus_knowledge_bp.route('/conversation/<conversation_id>', methods=['GET'])
def conversation_history_endpoint(conversation_id: str):
    """
    Get knowledge items from a specific conversation.
    
    GET /api/zeus-knowledge/conversation/<id>?max_items=20
    
    Returns:
    {
        "success": true,
        "conversation_id": "conv_123",
        "items": [...],
        "count": 10
    }
    """
    try:
        max_items = request.args.get('max_items', 20, type=int)
        
        knowledge = get_knowledge_memory()
        items = knowledge.get_conversation_history(
            conversation_id=conversation_id,
            max_items=max_items
        )
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'items': items,
            'count': len(items)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[ZeusKnowledgeAPI] Routes initialized at /api/zeus-knowledge/*")
