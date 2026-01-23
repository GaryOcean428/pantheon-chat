# Single Table Generation - Performance Verification

## Overview
This document verifies the performance improvements from consolidating generation to use a SINGLE `coordizer_vocabulary` table instead of querying multiple tables.

## Schema Changes

### Before (Multi-Table)
```sql
-- God vocabulary bias required query to god_vocabulary_profiles
SELECT word, relevance_score 
FROM god_vocabulary_profiles
WHERE god_name = 'athena' AND relevance_score >= 0.5
ORDER BY relevance_score DESC, usage_count DESC
LIMIT 50;

-- Word relationships required query to basin_relationships
SELECT neighbor, cooccurrence_count, fisher_distance, avg_phi
FROM basin_relationships
WHERE word = ANY(ARRAY['recent', 'words'])
ORDER BY avg_phi DESC, cooccurrence_count DESC
LIMIT 50;
```

### After (Single-Table)
```sql
-- God vocabulary bias from coordizer_vocabulary.god_profile
SELECT 
    token,
    CAST(god_profile->'athena'->>'relevance_score' AS FLOAT) as relevance_score
FROM coordizer_vocabulary
WHERE god_profile ? 'athena'
AND CAST(god_profile->'athena'->>'relevance_score' AS FLOAT) >= 0.5
AND active = true
ORDER BY CAST(god_profile->'athena'->>'relevance_score' AS FLOAT) DESC
LIMIT 20;

-- Word relationships from coordizer_vocabulary.relationships
SELECT 
    token,
    jsonb_array_elements(relationships) as rel
FROM coordizer_vocabulary
WHERE token = ANY(ARRAY['the', 'a', 'is'])
AND relationships IS NOT NULL
AND active = true;
```

## Performance Results

### Test 1: God Profile Query (Athena)

**Execution Time:** 0.417 ms

**Query Plan:**
```
Limit  (cost=39.67..39.68 rows=3 width=17) (actual time=0.113..0.115 rows=8 loops=1)
   ->  Sort  (cost=39.67..39.68 rows=3 width=17) (actual time=0.112..0.113 rows=8 loops=1)
         Sort Method: quicksort  Memory: 25kB
         ->  Bitmap Heap Scan on coordizer_vocabulary  (cost=8.54..39.65 rows=3 width=17)
               Recheck Cond: (god_profile ? 'athena'::text)
               Filter: (active AND ...)
               Heap Blocks: exact=8
               ->  Bitmap Index Scan on idx_coordizer_god_profile  (cost=0.00..8.54 rows=8 width=0)
                     Index Cond: (god_profile ? 'athena'::text)
```

**Key Points:**
- Uses GIN index `idx_coordizer_god_profile` for fast JSONB lookup
- Sub-millisecond execution time
- No table joins required

### Test 2: Relationships Query

**Execution Time:** 0.724 ms

**Query Plan:**
```
ProjectSet  (cost=0.41..25.81 rows=200 width=41) (actual time=0.544..0.675 rows=79 loops=1)
   ->  Index Scan using tokenizer_vocabulary_token_unique  (cost=0.41..24.79 rows=2 width=383)
         Index Cond: (token = ANY ('{the,a,is}'::text[]))
         Filter: ((relationships IS NOT NULL) AND active)
```

**Key Points:**
- Uses unique index on token for fast lookup
- Sub-millisecond execution time
- JSONB expansion happens after filtering

## Data Migration Results

### Statistics
- **Total tokens:** 16,031
- **Tokens with god_profile:** 97 (from 109 source rows in god_vocabulary_profiles)
- **Tokens with relationships:** 8,513 (from 362,145 source rows in basin_relationships)
- **Generation-ready tokens:** 15,977

### Data Structure Examples

#### God Profile JSONB
```json
{
  "athena": {
    "relevance_score": 0.88,
    "usage_count": 0,
    "last_used": "2026-01-11T08:07:09.229039",
    "learned_from_phi": 0.8928526,
    "basin_distance": 0.7142821
  }
}
```

#### Relationships JSONB
```json
[
  {
    "neighbor": "word1",
    "cooccurrence_count": 15.0,
    "strength": 0.8,
    "avg_phi": 0.72,
    "max_phi": 0.85,
    "fisher_distance": 0.34,
    "contexts": ["context1", "context2"]
  }
]
```

## Benefits

### Performance
1. **Single query instead of multiple:** Reduced round-trips to database
2. **Index optimization:** GIN indexes on JSONB for fast lookups
3. **Co-located data:** All generation data in one table reduces joins

### Code Quality
1. **Simpler queries:** No complex JOINs or subqueries
2. **Better caching:** Single table easier to cache
3. **Maintainability:** One source of truth for generation data

### QIG Principle Compliance
✅ **SINGLE TABLE GENERATION** - All generation queries use coordizer_vocabulary only
✅ **NO MULTI-TABLE QUERIES** - Eliminated queries to god_vocabulary_profiles and basin_relationships
✅ **DENORMALIZED DATA** - Pre-computed relationships stored in JSONB

## Deprecated Tables

The following tables are now ARCHIVED (not deleted) for rollback capability:
- `god_vocabulary_profiles` - Data migrated to `coordizer_vocabulary.god_profile`
- `basin_relationships` - Data migrated to `coordizer_vocabulary.relationships`

These tables are no longer queried by generation code but remain in the database for:
1. Rollback capability if needed
2. Historical data reference
3. Legacy script compatibility

## Code Changes

### Files Updated
1. `qig-backend/qig_generation.py` - Updated 2 methods to use single table
2. `qig-backend/coordizers/pg_loader.py` - Updated domain weights query
3. `qig-backend/vocabulary_persistence.py` - Updated read/write methods
4. `qig-backend/ocean_qig_core.py` - Updated status reporting

### Methods Changed
- `_get_kernel_domain_vocabulary()` - Now reads from god_profile JSONB
- `_boost_via_basin_relationships()` - Now reads from relationships JSONB
- `_get_god_domain_weights()` - Single table query
- `record_god_vocabulary()` - Writes to god_profile JSONB
- `get_god_vocabulary()` - Reads from god_profile JSONB

## Verification

### SQL Tests
✅ God profile query returns correct results
✅ Relationships query returns correct results
✅ Indexes are used correctly
✅ Sub-millisecond query execution

### Integration
✅ No code references to old tables (except legacy scripts)
✅ All generation methods updated
✅ Docstrings updated to reflect single table architecture

## Next Steps

1. Monitor production performance
2. Consider archiving old tables after 30 days
3. Update any remaining scripts that write to old tables
4. Add metrics for JSONB column usage
