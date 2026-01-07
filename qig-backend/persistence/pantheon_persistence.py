"""
Pantheon Persistence
====================

Persists inter-god communication (messages, debates, knowledge transfers)
to PostgreSQL for durability across restarts.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_persistence import BasePersistence


class PantheonPersistence(BasePersistence):
    """Persistence layer for Olympus pantheon chat and debates."""

    def save_message(self, message: Dict) -> bool:
        """Save a message to the database (Railway schema: god_name, role, phi, kappa, regime).
        
        Extra fields (to, read, responded, debate_id) are stored in metadata for backward compatibility.
        Uses JSON merge on conflict to preserve existing metadata fields.
        """
        query = """
            INSERT INTO pantheon_messages 
            (id, god_name, role, content, phi, kappa, regime, session_id, parent_id, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = COALESCE(pantheon_messages.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
        """
        try:
            metadata = message.get('metadata', {}) or {}
            if message.get('to'):
                metadata['to'] = message.get('to')
            if message.get('read'):
                metadata['read'] = message.get('read')
            if message.get('responded'):
                metadata['responded'] = message.get('responded')
            if message.get('debate_id'):
                metadata['debate_id'] = message.get('debate_id')
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            timestamp = message.get('timestamp')
            if isinstance(timestamp, str):
                created_at = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                created_at = datetime.now()
            
            self.execute_query(query, (
                message['id'],
                message.get('from', message.get('god_name', '')),
                message.get('role', message.get('type', 'message')),
                message.get('content', ''),
                message.get('phi'),
                message.get('kappa'),
                message.get('regime'),
                message.get('session_id'),
                message.get('parent_id'),
                metadata_json,
                created_at,
            ), fetch=False)
            return True
        except Exception as e:
            print(f"[PantheonPersistence] Failed to save message: {e}")
            return False

    def load_recent_messages(self, limit: int = 100) -> List[Dict]:
        """Load recent messages from the database in chronological order (oldest first)."""
        query = """
            SELECT id, god_name, role, content, phi, kappa, regime,
                   session_id, parent_id, metadata, created_at
            FROM pantheon_messages
            ORDER BY created_at ASC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        messages = []
        for row in results or []:
            metadata = row['metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            elif metadata is None:
                metadata = {}
            
            msg = {
                'id': row['id'],
                'from': row['god_name'],
                'god_name': row['god_name'],
                'to': metadata.get('to', 'pantheon'),
                'type': row['role'] or 'message',
                'role': row['role'],
                'content': row['content'],
                'phi': row['phi'],
                'kappa': row['kappa'],
                'regime': row['regime'],
                'session_id': row['session_id'],
                'parent_id': row['parent_id'],
                'metadata': metadata,
                'read': metadata.get('read', False),
                'responded': metadata.get('responded', False),
                'debate_id': metadata.get('debate_id'),
                'timestamp': row['created_at'].isoformat() if row['created_at'] else None,
            }
            messages.append(msg)
        return messages

    def mark_message_read(self, message_id: str) -> bool:
        """Mark a message as read (stores in metadata since Railway schema lacks is_read column)."""
        query = """
            UPDATE pantheon_messages 
            SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"read": true}'::jsonb 
            WHERE id = %s
        """
        try:
            self.execute_query(query, (message_id,), fetch=False)
            return True
        except Exception as e:
            print(f"[PantheonPersistence] Failed to mark message read: {e}")
            return False

    def save_debate(self, debate: Dict) -> bool:
        """Save or update a debate."""
        query = """
            INSERT INTO pantheon_debates
            (id, topic, initiator, opponent, context, status, arguments, winner, arbiter, resolution, started_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                arguments = EXCLUDED.arguments,
                winner = EXCLUDED.winner,
                arbiter = EXCLUDED.arbiter,
                resolution = EXCLUDED.resolution,
                resolved_at = EXCLUDED.resolved_at
        """
        try:
            started_at = debate.get('started_at')
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            elif started_at is None:
                started_at = datetime.now()
            
            resolved_at = debate.get('resolved_at')
            if isinstance(resolved_at, str):
                resolved_at = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
            
            self.execute_query(query, (
                debate['id'],
                debate.get('topic', ''),
                debate.get('initiator', ''),
                debate.get('opponent', ''),
                json.dumps(debate.get('context', {})) if debate.get('context') else None,
                debate.get('status', 'active'),
                json.dumps(debate.get('arguments', [])),
                debate.get('winner'),
                debate.get('arbiter'),
                json.dumps(debate.get('resolution', {})) if debate.get('resolution') else None,
                started_at,
                resolved_at,
            ), fetch=False)
            return True
        except Exception as e:
            print(f"[PantheonPersistence] Failed to save debate: {e}")
            return False

    def load_debates(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Load debates from the database."""
        if status:
            query = """
                SELECT * FROM pantheon_debates
                WHERE status = %s
                ORDER BY started_at DESC
                LIMIT %s
            """
            results = self.execute_query(query, (status, limit))
        else:
            query = """
                SELECT * FROM pantheon_debates
                ORDER BY started_at DESC
                LIMIT %s
            """
            results = self.execute_query(query, (limit,))
        
        debates = []
        for row in results or []:
            # Parse JSON fields if they're strings
            context = row['context']
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except:
                    context = {}
            elif context is None:
                context = {}
            
            arguments = row['arguments']
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except:
                    arguments = []
            elif arguments is None:
                arguments = []
            
            resolution = row['resolution']
            if isinstance(resolution, str):
                try:
                    resolution = json.loads(resolution)
                except:
                    resolution = {}
            elif resolution is None:
                resolution = {}
            
            debate = {
                'id': row['id'],
                'topic': row['topic'],
                'initiator': row['initiator'],
                'opponent': row['opponent'],
                'context': context,
                'status': row['status'],
                'arguments': arguments,
                'winner': row['winner'],
                'arbiter': row['arbiter'],
                'resolution': resolution,
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'resolved_at': row['resolved_at'].isoformat() if row['resolved_at'] else None,
            }
            debates.append(debate)
        return debates

    def save_knowledge_transfer(self, transfer: Dict) -> bool:
        """Save a knowledge transfer event."""
        query = """
            INSERT INTO pantheon_knowledge_transfers
            (from_god, to_god, knowledge_type, content, accepted, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            self.execute_query(query, (
                transfer.get('from', ''),
                transfer.get('to', ''),
                transfer.get('type', 'insight'),
                json.dumps(transfer.get('content', {})) if transfer.get('content') else None,
                transfer.get('accepted', False),
                datetime.now(),
            ), fetch=False)
            return True
        except Exception as e:
            print(f"[PantheonPersistence] Failed to save knowledge transfer: {e}")
            return False

    def load_knowledge_transfers(self, limit: int = 100) -> List[Dict]:
        """Load recent knowledge transfers in chronological order."""
        query = """
            SELECT * FROM pantheon_knowledge_transfers
            ORDER BY created_at ASC
            LIMIT %s
        """
        results = self.execute_query(query, (limit,))
        transfers = []
        for row in results or []:
            # Parse content JSON if it's a string
            content = row['content']
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    content = {}
            elif content is None:
                content = {}
            
            transfer = {
                'id': row['id'],
                'from': row['from_god'],
                'to': row['to_god'],
                'type': row['knowledge_type'],
                'content': content,
                'accepted': row['accepted'],
                'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            }
            transfers.append(transfer)
        return transfers


# Singleton instance
_instance: Optional[PantheonPersistence] = None


def get_pantheon_persistence() -> PantheonPersistence:
    """Get the singleton PantheonPersistence instance."""
    global _instance
    if _instance is None:
        _instance = PantheonPersistence()
    return _instance
