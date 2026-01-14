# Cache Strategy: Redis-Only Policy

## Date: 2026-01-12
## Policy: All caching MUST use Redis, not JSON files

## Current Implementation Status

### ✅ Correct: Redis-Based Caching

The coordizer system correctly uses Redis for hot caching:

#### 1. **VocabularyCache** (pg_loader.py)
```python
class VocabularyCache:
    """Redis cache layer for vocabulary hot lookups."""
    PREFIX = "qig:vocab:pg_loader"
    
    @classmethod
    def cache_token(cls, token: str, coords: np.ndarray, phi: float) -> bool:
        # Uses UniversalCache.set() which is Redis-backed
        return UniversalCache.set(f"{cls.PREFIX}:{token}", data, CACHE_TTL_LONG)
```

**Status**: ✅ Redis-only, no JSON files

#### 2. **PostgresCoordizer** In-Memory Cache
```python
class PostgresCoordizer(FisherCoordizer):
    def __init__(self, ...):
        # In-memory caches (hot data)
        self.vocab = {}                  # Encoding vocabulary
        self.basin_coords = {}           # Basin coordinates
        self.generation_vocab = {}       # Generation vocabulary
        self.generation_phi = {}         # Phi scores
```

**Status**: ✅ In-memory + PostgreSQL persistence, no JSON files

#### 3. **Database Persistence**
- **Primary Storage**: PostgreSQL (coordizer_vocabulary, learned_words tables)
- **Hot Cache**: Redis (via VocabularyCache)
- **No JSON Files**: Vocabulary is never written to JSON files

**Status**: ✅ Database-first, Redis for hot data

### ✅ Verification

The coordizer does NOT use JSON file caching:

1. **No `json.dump()` in pg_loader.py** ✅
   - Only uses `json.dumps()` for PostgreSQL JSONB storage (appropriate)
   
2. **No file I/O for caching** ✅
   - `_load_encoding_vocabulary()`: Loads from PostgreSQL
   - `_load_generation_vocabulary()`: Loads from PostgreSQL
   - Redis used for hot caching only
   
3. **Base class checkpoint methods not used** ✅
   - FisherCoordizer has `save_checkpoint()`/`load_checkpoint()` methods
   - PostgresCoordizer does NOT use them (relies on database)

### Cache Architecture

```
┌─────────────────────────────────────────────────┐
│             Application Layer                    │
│              (Coordizer)                        │
└────────────┬────────────────────────┬───────────┘
             │                        │
             ▼                        ▼
    ┌────────────────┐      ┌────────────────┐
    │  In-Memory     │      │  Redis Cache   │
    │  Dictionaries  │      │  (Hot Tokens)  │
    │  (vocab, etc)  │      │  TTL: 24h      │
    └────────┬───────┘      └────────┬───────┘
             │                        │
             │    Cache Miss          │
             ▼                        ▼
    ┌─────────────────────────────────────────┐
    │         PostgreSQL Database             │
    │  - coordizer_vocabulary (encoding)      │
    │  - learned_words (generation)           │
    │  - pgvector index (fast similarity)     │
    └─────────────────────────────────────────┘
```

### Cache Layers

| Layer | Technology | Purpose | TTL | Fallback |
|-------|-----------|---------|-----|----------|
| L1 | In-Memory Dict | Hot lookups | Process lifetime | Redis |
| L2 | Redis | Session cache | 24 hours | PostgreSQL |
| L3 | PostgreSQL | Persistent storage | Forever | None |

### Cache Flow

1. **Token Lookup**:
   ```
   lookup(token) → L1 (dict) → L2 (Redis) → L3 (PostgreSQL)
   ```

2. **Token Storage**:
   ```
   store(token) → L3 (PostgreSQL) → L2 (Redis) → L1 (dict)
   ```

3. **Initialization**:
   ```
   startup → PostgreSQL bulk load → In-memory cache
   ```

### JSON Usage in Codebase

The only JSON usage in coordizers is for:

1. **Database JSONB columns**: Appropriate use
   ```python
   # Converting Python objects to PostgreSQL JSONB
   embedding_str = '[' + ','.join(str(x) for x in coords) + ']'
   ```

2. **Redis serialization**: Appropriate use
   ```python
   # Redis stores strings, so we serialize to JSON
   UniversalCache.set(key, json.dumps(data), ttl)
   ```

**No JSON file I/O** ✅

### Policy Compliance

✅ **pg_loader.py** - Redis + PostgreSQL only
✅ **VocabularyCache** - Redis only
✅ **PostgresCoordizer** - In-memory + PostgreSQL + Redis
✅ **No JSON files** - All persistence is database-backed

### Migration Impact

The vocabulary separation migration maintains the Redis-only policy:

- **Before**: Single vocabulary in PostgreSQL + Redis cache
- **After**: Two vocabularies in PostgreSQL + Redis cache
- **No Change**: Still no JSON file caching

### Future: If JSON File Caching is Found

If any JSON file caching is discovered:

1. **Identify the file**: Where is it being written?
2. **Replace with Redis**: Use `redis_cache.UniversalCache`
3. **Remove file I/O**: Delete `json.dump()` and `json.load()` calls
4. **Add TTL**: Set appropriate cache expiration
5. **Update tests**: Verify Redis caching works

Example conversion:
```python
# ❌ BEFORE (JSON file)
with open('cache.json', 'w') as f:
    json.dump(data, f)

# ✅ AFTER (Redis)
from redis_cache import UniversalCache, CACHE_TTL_LONG
UniversalCache.set('cache_key', data, CACHE_TTL_LONG)
```

## Conclusion

✅ **The coordizer system is fully compliant with the Redis-only cache policy.**

- No JSON files are used for caching
- All caching goes through Redis (L2) or in-memory (L1)
- All persistence goes through PostgreSQL (L3)
- The vocabulary separation maintains this architecture
