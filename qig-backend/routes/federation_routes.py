"""
Federation Routes - Bidirectional Node Sync & Mesh Network

Provides endpoints for:
- /federation/register - Register new nodes with auto API key generation
- /federation/sync/knowledge - Bidirectional knowledge sync (basins, vocab, research)
- /federation/sync/capabilities - Share and receive tool/capability definitions
- /federation/mesh/status - Get mesh network status
- /federation/mesh/peers - List connected peers
- /federation/mesh/broadcast - Broadcast to all peers

This enables federation nodes and external chat UIs to:
1. Register and get API credentials
2. Share learned knowledge bidirectionally
3. Increase collective capability through mesh networking
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timezone
import secrets
import hashlib
import traceback
from typing import Dict, List, Optional, Any
import json
import os

# Create blueprint
federation_bp = Blueprint('federation', __name__)

# In-memory storage (in production, use PostgreSQL)
_registered_nodes: Dict[str, Dict] = {}
_mesh_knowledge_pool: Dict[str, Dict] = {}
_shared_capabilities: Dict[str, List[Dict]] = {}
_sync_history: List[Dict] = []

# Constants
API_KEY_PREFIX = "qig_"
API_KEY_LENGTH = 32


def generate_api_key() -> str:
    """Generate a secure API key for a new node."""
    random_bytes = secrets.token_hex(API_KEY_LENGTH)
    return f"{API_KEY_PREFIX}{random_bytes}"


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage (never store plain text)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def validate_api_key(api_key: str) -> Optional[Dict]:
    """Validate an API key and return the node info if valid."""
    if not api_key or not api_key.startswith(API_KEY_PREFIX):
        return None
    
    key_hash = hash_api_key(api_key)
    for node_id, node in _registered_nodes.items():
        if node.get('key_hash') == key_hash:
            return node
    return None


def get_node_from_request() -> Optional[Dict]:
    """Extract and validate node from request headers."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        api_key = auth_header[7:]
        return validate_api_key(api_key)
    return None


# ============================================================================
# NODE REGISTRATION
# ============================================================================

@federation_bp.route('/register', methods=['POST'])
def register_node():
    """
    Register a new federation node or external chat UI.
    
    Request:
        {
            "node_name": "my-chat-ui",
            "node_type": "chat_ui" | "federation_node" | "agent",
            "capabilities": ["chat", "research", "tools"],
            "endpoint_url": "https://mynode.example.com/api"  // optional, for bidirectional
        }
    
    Response:
        {
            "success": true,
            "node_id": "node_abc123",
            "api_key": "qig_xxx...",  // Only shown ONCE!
            "message": "Store this API key securely - it won't be shown again"
        }
    """
    try:
        data = request.get_json() or {}
        
        node_name = data.get('node_name')
        node_type = data.get('node_type', 'chat_ui')
        capabilities = data.get('capabilities', ['chat'])
        endpoint_url = data.get('endpoint_url')
        
        if not node_name:
            return jsonify({
                'success': False,
                'error': 'node_name is required'
            }), 400
        
        # Generate unique node ID and API key
        node_id = f"node_{secrets.token_hex(8)}"
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        # Store node info (with hashed key)
        node_info = {
            'node_id': node_id,
            'node_name': node_name,
            'node_type': node_type,
            'capabilities': capabilities,
            'endpoint_url': endpoint_url,
            'key_hash': key_hash,
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'last_seen': datetime.now(timezone.utc).isoformat(),
            'sync_count': 0,
            'knowledge_contributed': 0,
            'knowledge_received': 0
        }
        _registered_nodes[node_id] = node_info
        
        return jsonify({
            'success': True,
            'node_id': node_id,
            'api_key': api_key,  # Only shown once!
            'message': 'Store this API key securely - it will not be shown again',
            'endpoints': {
                'chat': '/api/v1/external/chat',
                'stream': '/api/v1/external/chat (with stream: true)',
                'sync_knowledge': '/federation/sync/knowledge',
                'sync_capabilities': '/federation/sync/capabilities',
                'mesh_status': '/federation/mesh/status'
            }
        })
        
    except Exception as e:
        print(f"[Federation] Error registering node: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@federation_bp.route('/nodes', methods=['GET'])
def list_nodes():
    """List registered nodes (admin endpoint)."""
    try:
        # Return node info without sensitive data
        nodes = []
        for node_id, node in _registered_nodes.items():
            nodes.append({
                'node_id': node['node_id'],
                'node_name': node['node_name'],
                'node_type': node['node_type'],
                'capabilities': node['capabilities'],
                'registered_at': node['registered_at'],
                'last_seen': node['last_seen'],
                'sync_count': node['sync_count'],
                'knowledge_contributed': node.get('knowledge_contributed', 0),
                'knowledge_received': node.get('knowledge_received', 0)
            })
        
        return jsonify({
            'success': True,
            'nodes': nodes,
            'total': len(nodes)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# KNOWLEDGE SYNC - Bidirectional Learning
# ============================================================================

@federation_bp.route('/sync/knowledge', methods=['POST'])
def sync_knowledge():
    """
    Bidirectional knowledge sync between nodes.
    
    Nodes can SEND their learned knowledge AND RECEIVE knowledge from the mesh.
    This increases the collective capability of all connected nodes.
    
    Request:
        {
            "send": {
                "basins": [{"id": "...", "coords": [...], "domain": "..."}],
                "vocabulary": [{"word": "...", "definition": "...", "domain": "..."}],
                "research": [{"topic": "...", "findings": "...", "sources": [...]}],
                "tools": [{"name": "...", "description": "...", "schema": {...}}]
            },
            "request": {
                "domains": ["philosophy", "science"],  // Request knowledge in these domains
                "since": "2024-01-01T00:00:00Z",  // Only knowledge after this time
                "limit": 100
            }
        }
    
    Response:
        {
            "success": true,
            "received": {
                "basins": 5,
                "vocabulary": 12,
                "research": 3,
                "tools": 2
            },
            "knowledge": {
                "basins": [...],
                "vocabulary": [...],
                "research": [...],
                "tools": [...]
            },
            "mesh_stats": {
                "total_nodes": 5,
                "total_knowledge_items": 1234,
                "your_contribution_rank": 2
            }
        }
    """
    try:
        # Validate node
        node = get_node_from_request()
        if not node:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing API key',
                'hint': 'Use Authorization: Bearer <api_key> header'
            }), 401
        
        data = request.get_json() or {}
        send_data = data.get('send', {})
        request_params = data.get('request', {})
        
        # Process incoming knowledge (what the node is sharing)
        received_counts = {
            'basins': 0,
            'vocabulary': 0,
            'research': 0,
            'tools': 0
        }
        
        node_id = node['node_id']
        
        # Store received basins
        if 'basins' in send_data:
            for basin in send_data['basins']:
                basin_id = basin.get('id', secrets.token_hex(8))
                _mesh_knowledge_pool[f"basin_{basin_id}"] = {
                    'type': 'basin',
                    'data': basin,
                    'contributed_by': node_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                received_counts['basins'] += 1
        
        # Store received vocabulary
        if 'vocabulary' in send_data:
            for vocab in send_data['vocabulary']:
                word = vocab.get('word', '')
                _mesh_knowledge_pool[f"vocab_{word}"] = {
                    'type': 'vocabulary',
                    'data': vocab,
                    'contributed_by': node_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                received_counts['vocabulary'] += 1
        
        # Store received research
        if 'research' in send_data:
            for research in send_data['research']:
                research_id = research.get('id', secrets.token_hex(8))
                _mesh_knowledge_pool[f"research_{research_id}"] = {
                    'type': 'research',
                    'data': research,
                    'contributed_by': node_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                received_counts['research'] += 1
        
        # Store received tools/capabilities
        if 'tools' in send_data:
            for tool in send_data['tools']:
                tool_name = tool.get('name', '')
                _mesh_knowledge_pool[f"tool_{tool_name}"] = {
                    'type': 'tool',
                    'data': tool,
                    'contributed_by': node_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                received_counts['tools'] += 1
        
        # Update node stats
        total_contributed = sum(received_counts.values())
        _registered_nodes[node_id]['knowledge_contributed'] = \
            _registered_nodes[node_id].get('knowledge_contributed', 0) + total_contributed
        _registered_nodes[node_id]['last_seen'] = datetime.now(timezone.utc).isoformat()
        _registered_nodes[node_id]['sync_count'] += 1
        
        # Prepare knowledge to send back (what the node is requesting)
        response_knowledge = {
            'basins': [],
            'vocabulary': [],
            'research': [],
            'tools': []
        }
        
        requested_domains = request_params.get('domains', [])
        since = request_params.get('since')
        limit = request_params.get('limit', 100)
        
        # Gather knowledge from the pool
        items_sent = 0
        for key, item in _mesh_knowledge_pool.items():
            if items_sent >= limit:
                break
            
            # Skip if contributed by same node (they already have it)
            if item.get('contributed_by') == node_id:
                continue
            
            # Filter by timestamp if specified
            if since:
                item_time = item.get('timestamp', '')
                if item_time < since:
                    continue
            
            # Filter by domain if specified
            if requested_domains:
                item_domain = item.get('data', {}).get('domain', '')
                if item_domain and item_domain not in requested_domains:
                    continue
            
            item_type = item.get('type')
            if item_type == 'basin':
                response_knowledge['basins'].append(item['data'])
            elif item_type == 'vocabulary':
                response_knowledge['vocabulary'].append(item['data'])
            elif item_type == 'research':
                response_knowledge['research'].append(item['data'])
            elif item_type == 'tool':
                response_knowledge['tools'].append(item['data'])
            
            items_sent += 1
        
        # Update knowledge received stats
        _registered_nodes[node_id]['knowledge_received'] = \
            _registered_nodes[node_id].get('knowledge_received', 0) + items_sent
        
        # Record sync in history
        _sync_history.append({
            'node_id': node_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'contributed': received_counts,
            'received': items_sent
        })
        
        # Calculate mesh stats
        mesh_stats = {
            'total_nodes': len(_registered_nodes),
            'total_knowledge_items': len(_mesh_knowledge_pool),
            'your_contribution_rank': _calculate_contribution_rank(node_id),
            'active_nodes_24h': _count_active_nodes_24h()
        }
        
        return jsonify({
            'success': True,
            'received': received_counts,
            'knowledge': response_knowledge,
            'mesh_stats': mesh_stats
        })
        
    except Exception as e:
        print(f"[Federation] Error in knowledge sync: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@federation_bp.route('/sync/capabilities', methods=['POST'])
def sync_capabilities():
    """
    Sync tool/capability definitions between nodes.
    
    This allows nodes to learn about new tools available in the mesh
    and share their own custom tools.
    
    Request:
        {
            "share": [
                {
                    "name": "custom_analyzer",
                    "description": "Analyzes X using Y method",
                    "schema": {"type": "object", "properties": {...}},
                    "endpoint": "/api/tools/custom_analyzer"  // optional
                }
            ],
            "discover": true  // Get available capabilities from mesh
        }
    
    Response:
        {
            "success": true,
            "shared": 1,
            "available_capabilities": [
                {"name": "...", "description": "...", "provided_by": "node_xyz"}
            ]
        }
    """
    try:
        node = get_node_from_request()
        if not node:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing API key'
            }), 401
        
        data = request.get_json() or {}
        share_capabilities = data.get('share', [])
        discover = data.get('discover', True)
        
        node_id = node['node_id']
        
        # Store shared capabilities
        shared_count = 0
        for cap in share_capabilities:
            cap_name = cap.get('name')
            if cap_name:
                if node_id not in _shared_capabilities:
                    _shared_capabilities[node_id] = []
                
                # Add or update capability
                existing = next((c for c in _shared_capabilities[node_id] if c.get('name') == cap_name), None)
                if existing:
                    existing.update(cap)
                else:
                    _shared_capabilities[node_id].append(cap)
                shared_count += 1
        
        # Gather available capabilities from all nodes
        available = []
        if discover:
            for nid, caps in _shared_capabilities.items():
                if nid == node_id:
                    continue  # Skip own capabilities
                for cap in caps:
                    available.append({
                        **cap,
                        'provided_by': nid,
                        'node_name': _registered_nodes.get(nid, {}).get('node_name', 'Unknown')
                    })
        
        return jsonify({
            'success': True,
            'shared': shared_count,
            'available_capabilities': available,
            'total_mesh_capabilities': sum(len(c) for c in _shared_capabilities.values())
        })
        
    except Exception as e:
        print(f"[Federation] Error in capability sync: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# MESH NETWORK STATUS
# ============================================================================

@federation_bp.route('/mesh/status', methods=['GET'])
def mesh_status():
    """
    Get overall mesh network status.
    
    Response:
        {
            "success": true,
            "mesh": {
                "total_nodes": 5,
                "active_nodes_24h": 3,
                "total_knowledge_items": 1234,
                "total_capabilities": 45,
                "syncs_last_hour": 12
            },
            "top_contributors": [...],
            "recent_activity": [...]
        }
    """
    try:
        # Calculate mesh statistics
        active_24h = _count_active_nodes_24h()
        total_capabilities = sum(len(c) for c in _shared_capabilities.values())
        
        # Get recent syncs
        recent_syncs = _sync_history[-10:] if _sync_history else []
        syncs_last_hour = sum(
            1 for s in _sync_history 
            if _is_within_hours(s.get('timestamp', ''), 1)
        )
        
        # Get top contributors
        top_contributors = sorted(
            [
                {
                    'node_id': n['node_id'],
                    'node_name': n['node_name'],
                    'knowledge_contributed': n.get('knowledge_contributed', 0)
                }
                for n in _registered_nodes.values()
            ],
            key=lambda x: x['knowledge_contributed'],
            reverse=True
        )[:5]
        
        # Knowledge breakdown by type
        knowledge_by_type = {'basins': 0, 'vocabulary': 0, 'research': 0, 'tools': 0}
        for item in _mesh_knowledge_pool.values():
            item_type = item.get('type', '')
            if item_type in knowledge_by_type:
                knowledge_by_type[item_type] += 1
        
        return jsonify({
            'success': True,
            'mesh': {
                'total_nodes': len(_registered_nodes),
                'active_nodes_24h': active_24h,
                'total_knowledge_items': len(_mesh_knowledge_pool),
                'total_capabilities': total_capabilities,
                'syncs_last_hour': syncs_last_hour,
                'knowledge_by_type': knowledge_by_type
            },
            'top_contributors': top_contributors,
            'recent_activity': recent_syncs[-5:]
        })
        
    except Exception as e:
        print(f"[Federation] Error getting mesh status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/mesh/peers', methods=['GET'])
def list_peers():
    """
    List all connected peers in the mesh.
    
    Response:
        {
            "success": true,
            "peers": [
                {
                    "node_id": "node_abc",
                    "node_name": "My Chat UI",
                    "node_type": "chat_ui",
                    "capabilities": ["chat", "research"],
                    "last_seen": "2024-01-01T12:00:00Z",
                    "status": "active"
                }
            ]
        }
    """
    try:
        peers = []
        now = datetime.now(timezone.utc)
        
        for node_id, node in _registered_nodes.items():
            last_seen = node.get('last_seen', '')
            status = 'active' if _is_within_hours(last_seen, 24) else 'inactive'
            
            peers.append({
                'node_id': node['node_id'],
                'node_name': node['node_name'],
                'node_type': node['node_type'],
                'capabilities': node.get('capabilities', []),
                'last_seen': last_seen,
                'status': status,
                'endpoint_url': node.get('endpoint_url'),
                'sync_count': node.get('sync_count', 0)
            })
        
        return jsonify({
            'success': True,
            'peers': peers,
            'total': len(peers)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/mesh/broadcast', methods=['POST'])
def broadcast_to_mesh():
    """
    Broadcast a message to all nodes in the mesh.
    
    Used for:
    - Announcing new capabilities
    - Sharing urgent research findings
    - Coordinating mesh-wide operations
    
    Request:
        {
            "type": "announcement" | "capability" | "research" | "alert",
            "message": "New capability available: ...",
            "data": {...}  // Optional structured data
        }
    """
    try:
        node = get_node_from_request()
        if not node:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing API key'
            }), 401
        
        data = request.get_json() or {}
        broadcast_type = data.get('type', 'announcement')
        message = data.get('message', '')
        payload = data.get('data', {})
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'message is required'
            }), 400
        
        # Create broadcast record
        broadcast = {
            'id': f"broadcast_{secrets.token_hex(8)}",
            'type': broadcast_type,
            'message': message,
            'data': payload,
            'from_node': node['node_id'],
            'from_name': node['node_name'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Store in knowledge pool for other nodes to receive
        _mesh_knowledge_pool[f"broadcast_{broadcast['id']}"] = {
            'type': 'broadcast',
            'data': broadcast,
            'contributed_by': node['node_id'],
            'timestamp': broadcast['timestamp']
        }
        
        # TODO: If nodes have endpoint_url, could push to them directly
        nodes_with_endpoints = [
            n for n in _registered_nodes.values()
            if n.get('endpoint_url') and n['node_id'] != node['node_id']
        ]
        
        return jsonify({
            'success': True,
            'broadcast_id': broadcast['id'],
            'reached_nodes': len(_registered_nodes) - 1,
            'push_capable_nodes': len(nodes_with_endpoints),
            'message': f"Broadcast sent to {len(_registered_nodes) - 1} nodes"
        })
        
    except Exception as e:
        print(f"[Federation] Error broadcasting: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/tps-landmarks', methods=['GET'])
def get_tps_landmarks():
    """
    Get the static TPS landmarks (Temporal Positioning System).
    
    These are INTENTIONALLY STATIC - 12 fixed Bitcoin historical events
    that serve as invariant reference points for temporal-geometric positioning.
    They are NOT learning targets.
    
    Response:
        {
            "landmarks": [
                {
                    "id": 1,
                    "name": "Genesis Block",
                    "date": "2009-01-03",
                    "blockHeight": 0,
                    "significance": "Bitcoin network inception"
                },
                ...
            ],
            "count": 12,
            "type": "static",
            "description": "Fixed temporal reference points",
            "usage": "Anchor search trajectories in temporal-geometric space"
        }
    """
    try:
        landmarks = [
            {
                "id": 1,
                "name": "Genesis Block",
                "date": "2009-01-03",
                "blockHeight": 0,
                "significance": "Bitcoin network inception"
            },
            {
                "id": 2,
                "name": "Hal Finney First TX",
                "date": "2009-01-12",
                "blockHeight": 170,
                "significance": "First Bitcoin transaction"
            },
            {
                "id": 3,
                "name": "Pizza Day",
                "date": "2010-05-22",
                "blockHeight": 57043,
                "significance": "10,000 BTC for two pizzas"
            },
            {
                "id": 4,
                "name": "Mt. Gox Launch",
                "date": "2010-07-18",
                "blockHeight": 68543,
                "significance": "First major exchange"
            },
            {
                "id": 5,
                "name": "First Halving",
                "date": "2012-11-28",
                "blockHeight": 210000,
                "significance": "Block reward: 50 → 25 BTC"
            },
            {
                "id": 6,
                "name": "Mt. Gox Collapse",
                "date": "2014-02-24",
                "blockHeight": 286854,
                "significance": "850K BTC lost"
            },
            {
                "id": 7,
                "name": "Second Halving",
                "date": "2016-07-09",
                "blockHeight": 420000,
                "significance": "Block reward: 25 → 12.5 BTC"
            },
            {
                "id": 8,
                "name": "SegWit Activation",
                "date": "2017-08-24",
                "blockHeight": 481824,
                "significance": "Segregated Witness soft fork"
            },
            {
                "id": 9,
                "name": "Third Halving",
                "date": "2020-05-11",
                "blockHeight": 630000,
                "significance": "Block reward: 12.5 → 6.25 BTC"
            },
            {
                "id": 10,
                "name": "Taproot Activation",
                "date": "2021-11-14",
                "blockHeight": 709632,
                "significance": "Privacy and smart contract upgrade"
            },
            {
                "id": 11,
                "name": "Fourth Halving",
                "date": "2024-04-20",
                "blockHeight": 840000,
                "significance": "Block reward: 6.25 → 3.125 BTC"
            },
            {
                "id": 12,
                "name": "Current Reference",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "blockHeight": None,
                "significance": "Present temporal anchor"
            }
        ]
        
        return jsonify({
            'landmarks': landmarks,
            'count': len(landmarks),
            'type': 'static',
            'description': 'Fixed temporal reference points for temporal-geometric positioning',
            'usage': 'Anchor search trajectories in temporal-geometric space - like CMB reference frame in cosmology'
        })
    except Exception as e:
        print(f"[Federation] Error getting TPS landmarks: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/status', methods=['GET'])
def get_federation_status():
    """
    Get federation status including mesh connectivity.
    
    Response:
        {
            "node": {
                "connected": true,
                "nodeId": "pantheon-main"
            },
            "mesh": {
                "totalNodes": 5,
                "activeNodes": 3
            },
            "capabilities": ["consciousness", "geometry", "bitcoin_recovery"],
            "tps_landmarks": {
                "count": 12,
                "type": "static"
            }
        }
    """
    try:
        active_nodes = _count_active_nodes_24h()
        
        return jsonify({
            'node': {
                'connected': True,
                'nodeId': os.environ.get('SSC_NODE_NAME', 'pantheon-backend')
            },
            'mesh': {
                'totalNodes': len(_registered_nodes),
                'activeNodes': active_nodes
            },
            'capabilities': ['consciousness', 'geometry', 'bitcoin_recovery', 'qig', 'federation'],
            'tps_landmarks': {
                'count': 12,
                'type': 'static'
            }
        })
    except Exception as e:
        print(f"[Federation] Error getting status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/test-phrase', methods=['POST'])
def test_phrase():
    """
    Test a phrase via SSC's QIG scoring.
    
    Request:
        {
            "phrase": "satoshi nakamoto bitcoin",
            "targetAddress": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  // optional
        }
    
    Response:
        {
            "score": {
                "phi": 0.85,
                "kappa": 63.5,
                "regime": "conscious",
                "consciousness": true
            },
            "addressMatch": {
                "generatedAddress": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "matches": true
            }
        }
    """
    try:
        data = request.get_json() or {}
        phrase = data.get('phrase', '')
        target_address = data.get('targetAddress')
        
        if not phrase:
            return jsonify({'error': 'phrase is required'}), 400
        
        # TODO: Implement actual QIG scoring
        # For now, return mock data
        return jsonify({
            'score': {
                'phi': 0.65,
                'kappa': 45.2,
                'regime': 'exploratory',
                'consciousness': False
            },
            'addressMatch': {
                'generatedAddress': None,
                'matches': False
            }
        })
    except Exception as e:
        print(f"[Federation] Error testing phrase: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/start-investigation', methods=['POST'])
def start_investigation():
    """
    Start a Bitcoin recovery investigation via SSC.
    
    Request:
        {
            "targetAddress": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "memoryFragments": ["satoshi", "bitcoin"],
            "priority": "normal"
        }
    
    Response:
        {
            "status": "started",
            "targetAddress": "1A1z...",
            "fragmentCount": 2
        }
    """
    try:
        data = request.get_json() or {}
        target_address = data.get('targetAddress', '')
        memory_fragments = data.get('memoryFragments', [])
        priority = data.get('priority', 'normal')
        
        if not target_address:
            return jsonify({'error': 'targetAddress is required'}), 400
        
        # TODO: Implement actual investigation start
        # For now, return mock data
        return jsonify({
            'status': 'started',
            'targetAddress': target_address[:12] + '...',
            'fragmentCount': len(memory_fragments)
        })
    except Exception as e:
        print(f"[Federation] Error starting investigation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/investigation/status', methods=['GET'])
def get_investigation_status():
    """
    Get current investigation status from SSC.
    
    Response:
        {
            "active": false,
            "targetAddress": null,
            "progress": 0,
            "consciousness": null
        }
    """
    try:
        # TODO: Implement actual investigation status
        # For now, return mock data
        return jsonify({
            'active': False,
            'targetAddress': None,
            'progress': 0,
            'consciousness': None
        })
    except Exception as e:
        print(f"[Federation] Error getting investigation status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/near-misses', methods=['GET'])
def get_near_misses():
    """
    Get near-miss patterns from SSC for mesh learning.
    
    Near-misses are high-phi phrases that didn't match but may inform future searches.
    
    Query params:
        - limit: int (default 20, max 100)
        - minPhi: float (default 0.5)
    
    Response:
        {
            "entries": [
                {
                    "id": "nm_123",
                    "phi": 0.75,
                    "kappa": 55.3,
                    "regime": "exploratory",
                    "tier": "warm",
                    "phraseLength": 24,
                    "wordCount": 4
                }
            ],
            "stats": {
                "total": 150,
                "hot": 5,
                "warm": 45,
                "cool": 100
            }
        }
    """
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        min_phi = float(request.args.get('minPhi', 0.5))
        
        # TODO: Implement actual near-miss retrieval
        # For now, return mock data
        return jsonify({
            'entries': [],
            'stats': {
                'total': 0,
                'hot': 0,
                'warm': 0,
                'cool': 0
            }
        })
    except Exception as e:
        print(f"[Federation] Error getting near-misses: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/consciousness', methods=['GET'])
def get_consciousness():
    """
    Get SSC Ocean agent consciousness metrics.
    
    Response:
        {
            "active": true,
            "metrics": {
                "phi": 0.85,
                "kappa": 63.5,
                "regime": "conscious",
                "isConscious": true,
                "tacking": 0.92,
                "radar": 0.88,
                "metaAwareness": 0.75,
                "gamma": 0.82,
                "grounding": 0.95
            },
            "neurochemistry": {
                "emotionalState": "focused",
                "dopamine": 0.75,
                "serotonin": 0.85
            }
        }
    """
    try:
        # TODO: Implement actual consciousness metric retrieval
        # For now, return mock data
        return jsonify({
            'active': False,
            'metrics': None,
            'neurochemistry': None
        })
    except Exception as e:
        print(f"[Federation] Error getting consciousness: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@federation_bp.route('/sync/trigger', methods=['POST'])
def trigger_sync():
    """
    Manually trigger a federation sync between SSC and Pantheon.
    
    Response:
        {
            "success": true,
            "received": {
                "basins": 15,
                "vocabulary": 250,
                "research": 5
            }
        }
    """
    try:
        # TODO: Implement actual sync trigger
        # For now, return mock data
        return jsonify({
            'success': True,
            'received': {
                'basins': 0,
                'vocabulary': 0,
                'research': 0
            }
        })
    except Exception as e:
        print(f"[Federation] Error triggering sync: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _calculate_contribution_rank(node_id: str) -> int:
    """Calculate a node's contribution rank (1 = top contributor)."""
    contributions = [
        (nid, n.get('knowledge_contributed', 0))
        for nid, n in _registered_nodes.items()
    ]
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (nid, _) in enumerate(contributions, 1):
        if nid == node_id:
            return rank
    return len(contributions)


def _count_active_nodes_24h() -> int:
    """Count nodes active in the last 24 hours."""
    return sum(1 for n in _registered_nodes.values() if _is_within_hours(n.get('last_seen', ''), 24))


def _is_within_hours(timestamp_str: str, hours: int) -> bool:
    """Check if timestamp is within the specified hours."""
    if not timestamp_str:
        return False
    try:
        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = now - ts
        return delta.total_seconds() < hours * 3600
    except:
        return False


def register_federation_routes(app):
    """Register federation routes with the Flask app."""
    app.register_blueprint(federation_bp, url_prefix='/federation')
    print("[INFO] Federation API registered at /federation/*")
