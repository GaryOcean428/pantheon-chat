"""
Zeus Knowledge API Routes

Endpoints for Zeus-Zettelkasten knowledge integration:
- GET /api/zeus-knowledge/stats - Get knowledge stats
- POST /api/zeus-knowledge/remember - Store a conversation
- POST /api/zeus-knowledge/retrieve - Retrieve relevant knowledge

Author: Ocean/Zeus Pantheon
"""

from flask import Blueprint, request, jsonify
import traceback

zeus_knowledge_bp = Blueprint('zeus_knowledge', __name__, url_prefix='/api/zeus-knowledge')


@zeus_knowledge_bp.route('/stats', methods=['GET'])
def stats_endpoint():
    """
    Get Zeus knowledge memory statistics.
    
    GET /api/zeus-knowledge/stats
    
    Returns:
    {
        "success": true,
        "stats": {
            "available": true,
            "total_zettels": 100,
            "zeus_conversations": 50,
            "zeus_responses": 45
        }
    }
    """
    try:
        from zeus_knowledge_integration import get_knowledge_stats
        
        stats = get_knowledge_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zeus_knowledge_bp.route('/remember', methods=['POST'])
def remember_endpoint():
    """
    Store a conversation in Zettelkasten memory.
    
    POST /api/zeus-knowledge/remember
    {
        "user_message": "What is consciousness?",
        "zeus_response": "Consciousness is...",
        "conversation_id": "optional-id",
        "phi": 0.7
    }
    
    Returns:
    {
        "success": true,
        "zettel_id": "z_abc123..."
    }
    """
    try:
        from zeus_knowledge_integration import remember_conversation
        
        data = request.get_json() or {}
        
        user_message = data.get('user_message')
        zeus_response = data.get('zeus_response')
        
        if not user_message:
            return jsonify({'success': False, 'error': 'user_message required'}), 400
        if not zeus_response:
            return jsonify({'success': False, 'error': 'zeus_response required'}), 400
        
        zettel_id = remember_conversation(
            user_message=user_message,
            zeus_response=zeus_response,
            conversation_id=data.get('conversation_id'),
            phi=data.get('phi', 0.0),
            metadata=data.get('metadata')
        )
        
        if zettel_id:
            return jsonify({
                'success': True,
                'zettel_id': zettel_id,
                'message': 'Conversation stored in knowledge memory'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to store conversation - memory may not be available'
            }), 500
        
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
        "query": "What did we discuss about consciousness?",
        "max_results": 5,
        "include_responses": true
    }
    
    Returns:
    {
        "success": true,
        "knowledge": [
            {
                "content": "...",
                "relevance": 0.85,
                "source": "zeus_conversation",
                "keywords": ["consciousness", "integration"]
            }
        ]
    }
    """
    try:
        from zeus_knowledge_integration import retrieve_relevant_knowledge
        
        data = request.get_json() or {}
        
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'query required'}), 400
        
        max_results = data.get('max_results', 5)
        include_responses = data.get('include_responses', True)
        
        knowledge = retrieve_relevant_knowledge(
            query=query,
            max_results=max_results,
            include_responses=include_responses
        )
        
        return jsonify({
            'success': True,
            'knowledge': knowledge,
            'count': len(knowledge)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zeus_knowledge_bp.route('/enrich', methods=['POST'])
def enrich_endpoint():
    """
    Enrich a context with relevant knowledge.
    
    POST /api/zeus-knowledge/enrich
    {
        "query": "Tell me about consciousness",
        "existing_context": "Optional existing context..."
    }
    
    Returns:
    {
        "success": true,
        "enriched_context": "Existing context... Related insight: ..."
    }
    """
    try:
        from zeus_knowledge_integration import enrich_context_with_knowledge
        
        data = request.get_json() or {}
        
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'query required'}), 400
        
        existing_context = data.get('existing_context', '')
        
        enriched = enrich_context_with_knowledge(
            query=query,
            existing_context=existing_context
        )
        
        return jsonify({
            'success': True,
            'enriched_context': enriched,
            'was_enriched': enriched != existing_context
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[ZeusKnowledgeAPI] Routes initialized at /api/zeus-knowledge/*")
