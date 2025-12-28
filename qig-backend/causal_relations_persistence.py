"""
Causal Relations Persistence - PostgreSQL + Redis Layer

QIG-Pure persistence for learned causal relations:
- PostgreSQL: Durable storage for all causal relations
- Redis: Hot cache for fast lookups during generation
- Continuous learning: Updates flow through as learning happens

Follows the same patterns as other QIG persistence modules.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Redis cache keys
CAUSAL_CACHE_KEY = "qig:causal_relations"
CAUSAL_SOURCE_PREFIX = "qig:causal:source:"
CAUSAL_STATS_KEY = "qig:causal:stats"
CACHE_TTL = 86400  # 24 hours

# PostgreSQL connection
try:
    import psycopg2
    from psycopg2.extras import execute_batch, RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("[CausalPersistence] psycopg2 not available")

# Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("[CausalPersistence] redis not available")


@dataclass
class CausalRelation:
    """A single causal relation."""
    source_word: str
    target_word: str
    relation_type: str
    occurrence_count: int = 1
    confidence: float = 0.5
    source_curriculum: Optional[str] = None


@dataclass
class CausalStats:
    """Statistics about causal relations."""
    total_relations: int = 0
    unique_sources: int = 0
    unique_targets: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    last_updated: Optional[datetime] = None


class CausalRelationsPersistence:
    """
    PostgreSQL + Redis persistence for causal relations.
    
    QIG-Pure: No geometric operations here, just storage.
    Geometric operations happen in PropositionTrajectoryPlanner.
    """
    
    def __init__(self):
        self._db_conn = None
        self._redis = None
        self._initialized = False
        self._cache: Dict[str, Dict[str, Dict]] = {}  # In-memory fallback
        
        self._init_connections()
    
    def _init_connections(self):
        """Initialize database and Redis connections."""
        # PostgreSQL
        if PSYCOPG2_AVAILABLE:
            try:
                database_url = os.environ.get('DATABASE_URL')
                if database_url:
                    self._db_conn = psycopg2.connect(database_url)
                    self._db_conn.autocommit = False
                    self._ensure_tables()
                    logger.info("[CausalPersistence] PostgreSQL connected")
            except Exception as e:
                logger.error(f"[CausalPersistence] PostgreSQL connection failed: {e}")
                self._db_conn = None
        
        # Redis
        if REDIS_AVAILABLE:
            try:
                redis_url = os.environ.get('REDIS_URL')
                if redis_url:
                    self._redis = redis.from_url(redis_url, decode_responses=True)
                    self._redis.ping()
                    logger.info("[CausalPersistence] Redis connected")
            except Exception as e:
                logger.warning(f"[CausalPersistence] Redis connection failed: {e}")
                self._redis = None
        
        self._initialized = True
    
    def _ensure_tables(self):
        """Ensure causal_relations table exists."""
        if not self._db_conn:
            return
        
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS causal_relations (
                    id SERIAL PRIMARY KEY,
                    source_word VARCHAR(100) NOT NULL,
                    target_word VARCHAR(100) NOT NULL,
                    relation_type VARCHAR(50) NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    confidence FLOAT DEFAULT 0.5,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_curriculum VARCHAR(255),
                    UNIQUE(source_word, target_word, relation_type)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_causal_source ON causal_relations(source_word)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_causal_target ON causal_relations(target_word)")
            self._db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"[CausalPersistence] Table creation failed: {e}")
            self._db_conn.rollback()
    
    def load_all(self) -> Dict[str, Dict[str, Dict]]:
        """
        Load all causal relations.
        
        Returns:
            Dict[source] -> Dict[target] -> {type, count, confidence}
        """
        # Try Redis cache first
        if self._redis:
            try:
                cached = self._redis.get(CAUSAL_CACHE_KEY)
                if cached:
                    data = json.loads(cached)
                    logger.info(f"[CausalPersistence] Loaded {len(data)} sources from Redis cache")
                    return data
            except Exception as e:
                logger.warning(f"[CausalPersistence] Redis read failed: {e}")
        
        # Load from PostgreSQL
        if self._db_conn:
            try:
                cursor = self._db_conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT source_word, target_word, relation_type, 
                           occurrence_count, confidence
                    FROM causal_relations
                    ORDER BY occurrence_count DESC
                """)
                rows = cursor.fetchall()
                cursor.close()
                
                # Build nested dict
                result: Dict[str, Dict[str, Dict]] = {}
                for row in rows:
                    source = row['source_word']
                    target = row['target_word']
                    if source not in result:
                        result[source] = {}
                    result[source][target] = {
                        'type': row['relation_type'],
                        'count': row['occurrence_count'],
                        'confidence': row['confidence']
                    }
                
                # Cache in Redis
                if self._redis and result:
                    try:
                        self._redis.setex(CAUSAL_CACHE_KEY, CACHE_TTL, json.dumps(result))
                    except Exception:
                        pass
                
                total = sum(len(t) for t in result.values())
                logger.info(f"[CausalPersistence] Loaded {total} relations from PostgreSQL")
                return result
                
            except Exception as e:
                logger.error(f"[CausalPersistence] PostgreSQL read failed: {e}")
        
        # Return in-memory cache as fallback
        return self._cache
    
    def save_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        increment: int = 1,
        curriculum: Optional[str] = None
    ) -> bool:
        """
        Save or update a single causal relation.
        
        Uses UPSERT for atomic increment of occurrence_count.
        """
        if not self._db_conn:
            # Fallback to in-memory
            if source not in self._cache:
                self._cache[source] = {}
            if target not in self._cache[source]:
                self._cache[source][target] = {'type': relation_type, 'count': 0, 'confidence': 0.5}
            self._cache[source][target]['count'] += increment
            return True
        
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                INSERT INTO causal_relations (source_word, target_word, relation_type, 
                                              occurrence_count, source_curriculum)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_word, target_word, relation_type) 
                DO UPDATE SET 
                    occurrence_count = causal_relations.occurrence_count + %s,
                    last_updated = CURRENT_TIMESTAMP
            """, (source, target, relation_type, increment, curriculum, increment))
            self._db_conn.commit()
            cursor.close()
            
            # Invalidate Redis cache for this source
            if self._redis:
                try:
                    self._redis.delete(CAUSAL_CACHE_KEY)
                    self._redis.delete(f"{CAUSAL_SOURCE_PREFIX}{source}")
                except Exception:
                    pass
            
            return True
        except Exception as e:
            logger.error(f"[CausalPersistence] Save failed: {e}")
            self._db_conn.rollback()
            return False
    
    def save_batch(
        self,
        relations: List[Tuple[str, str, str, int]],
        curriculum: Optional[str] = None
    ) -> int:
        """
        Save multiple causal relations in a batch.
        
        Args:
            relations: List of (source, target, type, count) tuples
            curriculum: Source curriculum file
            
        Returns:
            Number of relations saved
        """
        if not relations:
            return 0
        
        if not self._db_conn:
            # Fallback to in-memory
            for source, target, rel_type, count in relations:
                if source not in self._cache:
                    self._cache[source] = {}
                if target not in self._cache[source]:
                    self._cache[source][target] = {'type': rel_type, 'count': 0, 'confidence': 0.5}
                self._cache[source][target]['count'] += count
            return len(relations)
        
        try:
            cursor = self._db_conn.cursor()
            
            # Use execute_batch for efficiency
            sql = """
                INSERT INTO causal_relations (source_word, target_word, relation_type, 
                                              occurrence_count, source_curriculum)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_word, target_word, relation_type) 
                DO UPDATE SET 
                    occurrence_count = causal_relations.occurrence_count + EXCLUDED.occurrence_count,
                    last_updated = CURRENT_TIMESTAMP
            """
            
            data = [(s, t, rt, c, curriculum) for s, t, rt, c in relations]
            execute_batch(cursor, sql, data, page_size=100)
            self._db_conn.commit()
            cursor.close()
            
            # Invalidate Redis cache
            if self._redis:
                try:
                    self._redis.delete(CAUSAL_CACHE_KEY)
                except Exception:
                    pass
            
            logger.info(f"[CausalPersistence] Batch saved {len(relations)} relations")
            return len(relations)
            
        except Exception as e:
            logger.error(f"[CausalPersistence] Batch save failed: {e}")
            self._db_conn.rollback()
            return 0
    
    def get_relations_for_source(self, source: str) -> Dict[str, Dict]:
        """
        Get all causal relations for a given source word.
        
        Uses Redis cache for fast lookup.
        """
        cache_key = f"{CAUSAL_SOURCE_PREFIX}{source}"
        
        # Try Redis first
        if self._redis:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        
        # Query PostgreSQL
        if self._db_conn:
            try:
                cursor = self._db_conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT target_word, relation_type, occurrence_count, confidence
                    FROM causal_relations
                    WHERE source_word = %s
                    ORDER BY occurrence_count DESC
                """, (source,))
                rows = cursor.fetchall()
                cursor.close()
                
                result = {}
                for row in rows:
                    result[row['target_word']] = {
                        'type': row['relation_type'],
                        'count': row['occurrence_count'],
                        'confidence': row['confidence']
                    }
                
                # Cache in Redis
                if self._redis and result:
                    try:
                        self._redis.setex(cache_key, CACHE_TTL, json.dumps(result))
                    except Exception:
                        pass
                
                return result
            except Exception as e:
                logger.error(f"[CausalPersistence] Query failed: {e}")
        
        # Fallback to in-memory
        return self._cache.get(source, {})
    
    def get_stats(self) -> CausalStats:
        """Get statistics about causal relations."""
        stats = CausalStats()
        
        if self._db_conn:
            try:
                cursor = self._db_conn.cursor(cursor_factory=RealDictCursor)
                
                # Total count
                cursor.execute("SELECT COUNT(*) as total FROM causal_relations")
                stats.total_relations = cursor.fetchone()['total']
                
                # Unique sources
                cursor.execute("SELECT COUNT(DISTINCT source_word) as cnt FROM causal_relations")
                stats.unique_sources = cursor.fetchone()['cnt']
                
                # Unique targets
                cursor.execute("SELECT COUNT(DISTINCT target_word) as cnt FROM causal_relations")
                stats.unique_targets = cursor.fetchone()['cnt']
                
                # By type
                cursor.execute("""
                    SELECT relation_type, SUM(occurrence_count) as total
                    FROM causal_relations
                    GROUP BY relation_type
                """)
                for row in cursor.fetchall():
                    stats.by_type[row['relation_type']] = row['total']
                
                cursor.close()
            except Exception as e:
                logger.error(f"[CausalPersistence] Stats query failed: {e}")
        
        return stats
    
    def update_confidence(self, source: str, target: str, new_confidence: float) -> bool:
        """Update confidence score for a relation (QIG-pure metric)."""
        if not self._db_conn:
            if source in self._cache and target in self._cache[source]:
                self._cache[source][target]['confidence'] = new_confidence
                return True
            return False
        
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                UPDATE causal_relations 
                SET confidence = %s, last_updated = CURRENT_TIMESTAMP
                WHERE source_word = %s AND target_word = %s
            """, (new_confidence, source, target))
            self._db_conn.commit()
            cursor.close()
            
            # Invalidate cache
            if self._redis:
                self._redis.delete(f"{CAUSAL_SOURCE_PREFIX}{source}")
                self._redis.delete(CAUSAL_CACHE_KEY)
            
            return True
        except Exception as e:
            logger.error(f"[CausalPersistence] Confidence update failed: {e}")
            self._db_conn.rollback()
            return False
    
    def close(self):
        """Close connections."""
        if self._db_conn:
            self._db_conn.close()
        if self._redis:
            self._redis.close()


# Singleton instance
_persistence: Optional[CausalRelationsPersistence] = None


def get_causal_persistence() -> CausalRelationsPersistence:
    """Get or create the singleton persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = CausalRelationsPersistence()
    return _persistence


def migrate_from_json(json_path: str = 'data/learned/word_relationships.json') -> int:
    """
    Migrate causal relations from JSON file to PostgreSQL.
    
    One-time migration helper.
    """
    import json as json_module
    from pathlib import Path
    
    json_file = Path(json_path)
    if not json_file.exists():
        logger.warning(f"[CausalPersistence] JSON file not found: {json_path}")
        return 0
    
    with open(json_file, 'r') as f:
        data = json_module.load(f)
    
    causal_relations = data.get('causal_relations', {})
    if not causal_relations:
        logger.info("[CausalPersistence] No causal relations in JSON file")
        return 0
    
    # Build batch
    relations = []
    for source, targets in causal_relations.items():
        for target, rel_info in targets.items():
            rel_type = rel_info.get('type', 'unknown')
            count = rel_info.get('count', 1)
            relations.append((source, target, rel_type, count))
    
    # Save to DB
    persistence = get_causal_persistence()
    saved = persistence.save_batch(relations, curriculum='json_migration')
    
    logger.info(f"[CausalPersistence] Migrated {saved} relations from JSON to PostgreSQL")
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run migration if executed directly
    count = migrate_from_json()
    print(f"Migrated {count} causal relations")
    
    # Test stats
    persistence = get_causal_persistence()
    stats = persistence.get_stats()
    print(f"Stats: {stats}")
