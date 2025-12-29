"""
Zettelkasten Memory API Routes

Endpoints:
- POST /api/zettelkasten/add - Add new memory
- POST /api/zettelkasten/retrieve - Retrieve by query
- GET /api/zettelkasten/stats - Get statistics
- GET /api/zettelkasten/graph - Get visualization data
- GET /api/zettelkasten/zettel/<id> - Get single zettel
- GET /api/zettelkasten/traverse/<id> - Multi-hop traversal
- GET /api/zettelkasten/clusters - Get knowledge clusters
- GET /api/zettelkasten/hubs - Get hub zettels
- GET /api/zettelkasten/path - Find path between zettels

Author: Ocean/Zeus Pantheon
"""

from flask import Blueprint, request, jsonify
import traceback

zettelkasten_bp = Blueprint('zettelkasten', __name__, url_prefix='/api/zettelkasten')


def get_memory():
    """Get the Zettelkasten memory instance."""
    from zettelkasten_memory import get_zettelkasten_memory
    return get_zettelkasten_memory()


@zettelkasten_bp.route('/add', methods=['POST'])
def add_memory_endpoint():
    """
    Add a new memory to the Zettelkasten.
    
    POST /api/zettelkasten/add
    {
        "content": "The atomic idea to remember",
        "source": "optional source info",
        "parent_id": "optional parent zettel id"
    }
    
    Returns:
    {
        "success": true,
        "zettel_id": "z_123...",
        "keywords": ["keyword1", "keyword2"],
        "links_created": 5,
        "evolution_triggered": true
    }
    """
    try:
        data = request.get_json() or {}
        
        content = data.get('content')
        if not content:
            return jsonify({'error': 'content required'}), 400
        
        source = data.get('source', '')
        parent_id = data.get('parent_id')
        
        memory = get_memory()
        zettel = memory.add(
            content=content,
            source=source,
            parent_id=parent_id
        )
        
        return jsonify({
            'success': True,
            'zettel_id': zettel.zettel_id,
            'keywords': zettel.keywords,
            'links_created': len(zettel.links),
            'contextual_description': zettel.contextual_description,
            'evolution_triggered': zettel.evolution_count > 0
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/retrieve', methods=['POST'])
def retrieve_endpoint():
    """
    Retrieve memories relevant to a query.
    
    POST /api/zettelkasten/retrieve
    {
        "query": "search query text",
        "max_results": 10,
        "include_context": true
    }
    
    Returns:
    {
        "success": true,
        "results": [
            {
                "zettel_id": "z_123...",
                "content": "...",
                "relevance": 0.85,
                "keywords": [...],
                "links_count": 5
            }
        ],
        "total_found": 10
    }
    """
    try:
        data = request.get_json() or {}
        
        query = data.get('query')
        if not query:
            return jsonify({'error': 'query required'}), 400
        
        max_results = data.get('max_results', 10)
        include_context = data.get('include_context', True)
        
        memory = get_memory()
        results = memory.retrieve(
            query=query,
            max_results=max_results,
            include_context=include_context
        )
        
        formatted_results = []
        for zettel, relevance in results:
            formatted_results.append({
                'zettel_id': zettel.zettel_id,
                'content': zettel.content,
                'relevance': relevance,
                'keywords': zettel.keywords,
                'contextual_description': zettel.contextual_description,
                'links_count': len(zettel.links),
                'access_count': zettel.access_count,
                'evolution_count': zettel.evolution_count,
                'source': zettel.source
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'total_found': len(formatted_results)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/keyword/<keyword>', methods=['GET'])
def retrieve_by_keyword_endpoint(keyword: str):
    """
    Retrieve memories by keyword.
    
    GET /api/zettelkasten/keyword/<keyword>?max_results=10
    
    Returns:
    {
        "success": true,
        "keyword": "quantum",
        "results": [...],
        "total_found": 5
    }
    """
    try:
        max_results = request.args.get('max_results', 10, type=int)
        
        memory = get_memory()
        results = memory.retrieve_by_keyword(keyword, max_results)
        
        formatted_results = []
        for zettel in results:
            formatted_results.append({
                'zettel_id': zettel.zettel_id,
                'content': zettel.content,
                'keywords': zettel.keywords,
                'links_count': len(zettel.links)
            })
        
        return jsonify({
            'success': True,
            'keyword': keyword,
            'results': formatted_results,
            'total_found': len(formatted_results)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/stats', methods=['GET'])
def stats_endpoint():
    """
    Get Zettelkasten statistics.
    
    GET /api/zettelkasten/stats
    
    Returns:
    {
        "success": true,
        "total_zettels": 100,
        "total_links": 500,
        "total_keywords": 200,
        "total_evolutions": 50,
        "avg_links_per_zettel": 5.0
    }
    """
    try:
        memory = get_memory()
        stats = memory.get_stats()
        
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


@zettelkasten_bp.route('/graph', methods=['GET'])
def graph_endpoint():
    """
    Get graph visualization data.
    
    GET /api/zettelkasten/graph?max_nodes=50
    
    Returns:
    {
        "success": true,
        "nodes": [
            {"id": "z_123", "label": "...", "keywords": [...]}
        ],
        "edges": [
            {"source": "z_123", "target": "z_456", "strength": 0.8}
        ],
        "stats": {...}
    }
    """
    try:
        max_nodes = request.args.get('max_nodes', 50, type=int)
        
        memory = get_memory()
        graph_data = memory.visualize_graph(max_nodes)
        
        return jsonify({
            'success': True,
            **graph_data
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/zettel/<zettel_id>', methods=['GET'])
def get_zettel_endpoint(zettel_id: str):
    """
    Get a single zettel by ID.
    
    GET /api/zettelkasten/zettel/<id>
    
    Returns:
    {
        "success": true,
        "zettel": {
            "zettel_id": "z_123...",
            "content": "...",
            "keywords": [...],
            "links": [...],
            ...
        }
    }
    """
    try:
        memory = get_memory()
        zettel = memory.get(zettel_id)
        
        if not zettel:
            return jsonify({
                'success': False,
                'error': f'Zettel {zettel_id} not found'
            }), 404
        
        # Format links
        links = []
        for link in zettel.links:
            links.append({
                'target_id': link.target_id,
                'link_type': link.link_type.value,
                'strength': link.strength,
                'context': link.context
            })
        
        return jsonify({
            'success': True,
            'zettel': {
                'zettel_id': zettel.zettel_id,
                'content': zettel.content,
                'contextual_description': zettel.contextual_description,
                'keywords': zettel.keywords,
                'links': links,
                'basin_coords': list(zettel.basin_coords),
                'created_at': zettel.created_at,
                'updated_at': zettel.updated_at,
                'access_count': zettel.access_count,
                'evolution_count': zettel.evolution_count,
                'source': zettel.source,
                'parent_id': zettel.parent_id,
                'children_ids': zettel.children_ids
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/linked/<zettel_id>', methods=['GET'])
def get_linked_endpoint(zettel_id: str):
    """
    Get all zettels linked to a given zettel.
    
    GET /api/zettelkasten/linked/<id>
    
    Returns:
    {
        "success": true,
        "zettel_id": "z_123",
        "linked": [...]
    }
    """
    try:
        memory = get_memory()
        linked = memory.get_linked(zettel_id)
        
        formatted = []
        for zettel in linked:
            formatted.append({
                'zettel_id': zettel.zettel_id,
                'content': zettel.content,
                'keywords': zettel.keywords
            })
        
        return jsonify({
            'success': True,
            'zettel_id': zettel_id,
            'linked': formatted,
            'count': len(formatted)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/traverse/<zettel_id>', methods=['GET'])
def traverse_endpoint(zettel_id: str):
    """
    Traverse the knowledge network from a starting zettel.
    
    GET /api/zettelkasten/traverse/<id>?max_depth=3&min_strength=0.3
    
    Returns:
    {
        "success": true,
        "start_id": "z_123",
        "traversal": {
            "0": [...],  // depth 0 (starting zettel)
            "1": [...],  // depth 1
            "2": [...]   // depth 2
        },
        "total_nodes": 15
    }
    """
    try:
        max_depth = request.args.get('max_depth', 3, type=int)
        min_strength = request.args.get('min_strength', 0.3, type=float)
        
        memory = get_memory()
        traversal = memory.traverse(
            start_id=zettel_id,
            max_depth=max_depth,
            min_link_strength=min_strength
        )
        
        if not traversal:
            return jsonify({
                'success': False,
                'error': f'Zettel {zettel_id} not found'
            }), 404
        
        # Format traversal by depth
        formatted = {}
        total_nodes = 0
        
        for depth, zettels in traversal.items():
            formatted[str(depth)] = []
            for zettel in zettels:
                formatted[str(depth)].append({
                    'zettel_id': zettel.zettel_id,
                    'content': zettel.content[:200] + ('...' if len(zettel.content) > 200 else ''),
                    'keywords': zettel.keywords
                })
                total_nodes += 1
        
        return jsonify({
            'success': True,
            'start_id': zettel_id,
            'traversal': formatted,
            'total_nodes': total_nodes,
            'max_depth_reached': max(int(d) for d in formatted.keys()) if formatted else 0
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/path', methods=['GET'])
def find_path_endpoint():
    """
    Find path between two zettels.
    
    GET /api/zettelkasten/path?source=z_123&target=z_456&max_depth=5
    
    Returns:
    {
        "success": true,
        "path_found": true,
        "path": [
            {"zettel_id": "z_123", "content": "..."},
            {"zettel_id": "z_234", "content": "..."},
            {"zettel_id": "z_456", "content": "..."}
        ],
        "path_length": 3
    }
    """
    try:
        source_id = request.args.get('source')
        target_id = request.args.get('target')
        max_depth = request.args.get('max_depth', 5, type=int)
        
        if not source_id or not target_id:
            return jsonify({'error': 'source and target required'}), 400
        
        memory = get_memory()
        path = memory.find_path(
            source_id=source_id,
            target_id=target_id,
            max_depth=max_depth
        )
        
        if path is None:
            return jsonify({
                'success': True,
                'path_found': False,
                'path': [],
                'path_length': 0,
                'message': f'No path found between {source_id} and {target_id}'
            })
        
        formatted_path = []
        for zettel in path:
            formatted_path.append({
                'zettel_id': zettel.zettel_id,
                'content': zettel.content[:200] + ('...' if len(zettel.content) > 200 else ''),
                'keywords': zettel.keywords
            })
        
        return jsonify({
            'success': True,
            'path_found': True,
            'path': formatted_path,
            'path_length': len(formatted_path)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/clusters', methods=['GET'])
def clusters_endpoint():
    """
    Get knowledge clusters.
    
    GET /api/zettelkasten/clusters?min_size=3
    
    Returns:
    {
        "success": true,
        "clusters": [
            {
                "cluster_id": 0,
                "size": 5,
                "zettels": [...]
            }
        ],
        "total_clusters": 3
    }
    """
    try:
        min_size = request.args.get('min_size', 3, type=int)
        
        memory = get_memory()
        clusters = memory.get_clusters(min_cluster_size=min_size)
        
        formatted_clusters = []
        for i, cluster in enumerate(clusters):
            formatted_cluster = {
                'cluster_id': i,
                'size': len(cluster),
                'zettels': []
            }
            
            # Collect all keywords in cluster for summary
            all_keywords = []
            
            for zettel in cluster:
                formatted_cluster['zettels'].append({
                    'zettel_id': zettel.zettel_id,
                    'content': zettel.content[:100] + ('...' if len(zettel.content) > 100 else ''),
                    'keywords': zettel.keywords
                })
                all_keywords.extend(zettel.keywords)
            
            # Find most common keywords in cluster
            keyword_counts = {}
            for kw in all_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            formatted_cluster['top_keywords'] = [kw for kw, _ in top_keywords]
            
            formatted_clusters.append(formatted_cluster)
        
        return jsonify({
            'success': True,
            'clusters': formatted_clusters,
            'total_clusters': len(formatted_clusters)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@zettelkasten_bp.route('/hubs', methods=['GET'])
def hubs_endpoint():
    """
    Get hub zettels (highly connected nodes).
    
    GET /api/zettelkasten/hubs?top_n=10
    
    Returns:
    {
        "success": true,
        "hubs": [
            {
                "zettel_id": "z_123",
                "content": "...",
                "keywords": [...],
                "connection_count": 15
            }
        ]
    }
    """
    try:
        top_n = request.args.get('top_n', 10, type=int)
        
        memory = get_memory()
        hubs = memory.get_hub_zettels(top_n)
        
        formatted_hubs = []
        for zettel in hubs:
            # Count total connections (outgoing + backlinks approximation)
            outgoing = len(zettel.links)
            
            formatted_hubs.append({
                'zettel_id': zettel.zettel_id,
                'content': zettel.content[:200] + ('...' if len(zettel.content) > 200 else ''),
                'keywords': zettel.keywords,
                'outgoing_links': outgoing,
                'access_count': zettel.access_count,
                'evolution_count': zettel.evolution_count
            })
        
        return jsonify({
            'success': True,
            'hubs': formatted_hubs,
            'count': len(formatted_hubs)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


print("[ZettelkastenAPI] Routes initialized at /api/zettelkasten/*")
