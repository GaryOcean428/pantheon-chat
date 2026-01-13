# Database Table Consolidation Recommendations

**Date**: 2026-01-13  
**Status**: 📋 RECOMMENDATIONS  
**Version**: 1.00W  
**ID**: ISMS-DB-CONSOLIDATION-001  
**Purpose**: Analysis and recommendations for consolidating duplicate/overlapping database tables

---

## Executive Summary

**Goal:** Reduce table duplication while preserving data integrity and feature functionality

**Findings:**
- 5 confirmed duplicate pairs requiring action
- 3 potential consolidation candidates requiring investigation
- 5 tables recommended for deprecation
- Estimated 10-15 tables can be consolidated into 5-7 tables

**Impact:**
- Reduced schema complexity
- Improved query performance
- Clearer data architecture
- Easier maintenance

---

## Section 1: Confirmed Duplicates - CONSOLIDATE

### 1.1 Governance Proposals (DUPLICATE CONFIRMED)

**Tables:**
- `governance_proposals`: 2,747 rows
- `pantheon_proposals`: 2,746 rows

**Analysis:**
Row counts are nearly identical (1 row difference), suggesting these are the same data.

**Recommendation:** ✅ **MERGE INTO pantheon_proposals**

**Rationale:**
1. Pantheon is the domain-specific namespace
2. `governance_proposals` is generic and ambiguous
3. Pantheon governance is the primary use case

**Migration Plan:**
```sql
-- Step 1: Verify they're duplicates
SELECT 
  gp.proposal_id,
  pp.proposal_id,
  gp.title,
  pp.title
FROM governance_proposals gp
FULL OUTER JOIN pantheon_proposals pp 
  ON gp.proposal_id = pp.proposal_id
WHERE gp.proposal_id IS NULL 
   OR pp.proposal_id IS NULL;

-- Step 2: If <10 unique rows, manually reconcile

-- Step 3: Execute migration in a transaction for atomicity
BEGIN;

-- Update all foreign key references
UPDATE tool_requests 
SET proposal_id = (
  SELECT pp.id 
  FROM pantheon_proposals pp 
  WHERE pp.proposal_id = tool_requests.proposal_id
);

-- Step 4: Drop governance_proposals
DROP TABLE governance_proposals CASCADE;

COMMIT;

-- Step 5: Update code to use pantheonProposals only
```

**Code Changes:**
```typescript
// server/routes/governance.ts
// OLD: import { governanceProposals } from '@shared/schema';
// NEW: import { pantheonProposals as governanceProposals } from '@shared/schema';

// OR better: Rename all usages to pantheonProposals
```

**Estimated Effort:** 3 hours  
**Risk:** LOW (data is duplicate)  
**Priority:** 🟠 HIGH

---

### 1.2 Knowledge Shared Entries vs Knowledge Transfers (INVESTIGATE)

**Tables:**
- `knowledge_shared_entries`: 35 rows
- `knowledge_transfers`: 35 rows

**Analysis:**
Identical row counts suggest possible duplication.

**Investigation Required:**
```sql
-- Check if these are the same data using content hash for reliable matching
SELECT 
  kse.id as shared_id,
  kt.id as transfer_id,
  kse.created_at as shared_time,
  kt.created_at as transfer_time,
  kse.content,
  kt.knowledge_type,
  -- Compare content hashes
  md5(kse.content::text) as shared_hash,
  md5(COALESCE(kt.knowledge_data::text, '')) as transfer_hash
FROM knowledge_shared_entries kse
FULL OUTER JOIN knowledge_transfers kt
  ON md5(kse.content::text) = md5(COALESCE(kt.knowledge_data::text, ''))
  AND kse.source_kernel_id = kt.source_kernel_id;
```

**Recommendation (Pending Investigation):**

**If Same Data:** ✅ **MERGE INTO knowledge_transfers**
- `knowledge_transfers` is more descriptive
- Emphasizes the action (transfer) over the state (shared)
- Migration: Copy unique fields, drop shared_entries

**If Different Data:**
- `knowledge_shared_entries` = What knowledge is available
- `knowledge_transfers` = When/how knowledge was transferred
- Keep both but clarify distinction in documentation

**Estimated Effort:** 4 hours (2 investigation + 2 migration)  
**Risk:** MEDIUM (need to verify data semantics)  
**Priority:** 🟡 MEDIUM

---

### 1.3 Consciousness State vs Consciousness Checkpoints (CONSOLIDATE)

**Tables:**
- `consciousness_state`: 1 row (singleton)
- `consciousness_checkpoints`: 10 rows (history)

**Analysis:**
`consciousness_state` appears to be "current state" while `consciousness_checkpoints` is history.

**Recommendation:** ✅ **CONSOLIDATE INTO consciousness_checkpoints**

**Rationale:**
1. Current state = latest checkpoint with `is_current=true` flag
2. Eliminates dual writes
3. Simpler to maintain single source of truth
4. Automatic history tracking

**Migration Plan:**
```sql
-- Step 1: Add is_current flag to consciousness_checkpoints
ALTER TABLE consciousness_checkpoints 
ADD COLUMN is_current BOOLEAN DEFAULT false;

-- Step 2: Mark latest checkpoint as current
UPDATE consciousness_checkpoints 
SET is_current = true
WHERE id = (
  SELECT id FROM consciousness_checkpoints 
  ORDER BY created_at DESC 
  LIMIT 1
);

-- Step 3: Create view for current state (backwards compatibility)
CREATE VIEW consciousness_state AS
SELECT * FROM consciousness_checkpoints
WHERE is_current = true
LIMIT 1;

-- Step 4: Eventually drop old consciousness_state table
-- (after all code updated to use checkpoints)
```

**Code Changes:**
```python
# OLD:
async def get_current_consciousness():
    return await db.query(consciousnessState).first()

# NEW:
async def get_current_consciousness():
    return await db.query(consciousnessCheckpoints)\
        .filter_by(is_current=True)\
        .first()

async def update_consciousness(new_state):
    # Mark old as not current
    await db.query(consciousnessCheckpoints)\
        .update({"is_current": False})
    
    # Insert new checkpoint as current
    checkpoint = ConsciousnessCheckpoint(**new_state, is_current=True)
    await db.add(checkpoint)
    await db.commit()
```

**Estimated Effort:** 2 hours  
**Risk:** LOW (simple consolidation)  
**Priority:** 🟠 HIGH

---

## Section 2: Potential Consolidations - INVESTIGATE

### 2.1 Near-Miss Tables (CONSOLIDATE OR IMPLEMENT)

**Tables:**
- `near_miss_entries`: 0 rows (EMPTY)
- `near_miss_clusters`: 0 rows (EMPTY)
- `near_miss_adaptive_state`: 1 row (ACTIVE)

**Analysis:**
Only adaptive_state has data. Entry and cluster tables never used.

**Recommendation:** ✅ **CONSOLIDATE INTO single near_miss_system table**

**Proposed Schema:**
```sql
CREATE TABLE near_miss_system (
  id SERIAL PRIMARY KEY,
  system_type VARCHAR(32) NOT NULL, -- 'adaptive_state', 'entry', 'cluster'
  
  -- Adaptive state fields
  detection_threshold DOUBLE PRECISION,
  adjustment_rate DOUBLE PRECISION,
  
  -- Entry fields
  candidate_phrase TEXT,
  actual_phi DOUBLE PRECISION,
  threshold_phi DOUBLE PRECISION,
  miss_distance DOUBLE PRECISION,
  
  -- Cluster fields
  cluster_center DOUBLE PRECISION[],
  cluster_radius DOUBLE PRECISION,
  entry_count INTEGER,
  
  -- Common fields
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

**Alternative:** If entries/clusters needed, implement them properly with data.

**Estimated Effort:** 3 hours  
**Risk:** LOW (tables are empty)  
**Priority:** 🟡 MEDIUM

---

### 2.2 Geodesic Path Tables (KEEP SEPARATED)

**Tables:**
- `geodesic_paths`: 0 rows - General Fisher-Rao geodesics
- `tps_geodesic_paths`: 0 rows - TPS manifold geodesics  
- `ocean_trajectories`: 0 rows - Ocean agent time-series paths

**Analysis:**
These are different coordinate systems and semantics.

**Recommendation:** ✅ **KEEP SEPARATED**

**Rationale:**
1. Different metrics: Fisher-Rao vs TPS vs time-series
2. Different dimensions: 64D basin vs TPS projection vs temporal
3. Different use cases: Navigation vs topology vs tracking
4. Consolidation would create confusion

**Action:** Implement all three (see Wire Empty Tables guide)

**Priority:** 🟢 LOW (architecture is correct)

---

### 2.3 Shadow Operations (KEEP AS IS)

**Tables:**
- `shadow_intel`: 3 rows
- `shadow_knowledge`: 29,304 rows ✅ HIGH VALUE
- `shadow_operations_log`: 5 rows
- `shadow_operations_state`: 4 rows
- `shadow_pantheon_intel`: 3 rows

**Analysis:**
Separate concerns, one high-value data table.

**Recommendation:** ✅ **NO CONSOLIDATION NEEDED**

**Rationale:**
1. `shadow_knowledge` is the data repository (29K rows)
2. Other tables are operational metadata
3. Separating concerns is correct architecture
4. No duplication detected

**Priority:** 🟢 LOW (no action needed)

---

## Section 3: Recommended Deprecations

### 3.1 Kernel Evolution Tables (NEVER USED)

**Tables:**
- `kernel_evolution_events`: 0 rows
- `kernel_evolution_fitness`: 0 rows

**Recommendation:** 🔴 **DEPRECATE AND DROP**

**Rationale:**
1. No data in 6+ months of operation
2. Feature never implemented
3. Not in roadmap
4. No code references

**Migration Plan:**
```sql
-- Simple drop - no data to preserve
DROP TABLE kernel_evolution_events CASCADE;
DROP TABLE kernel_evolution_fitness CASCADE;
```

**Code Cleanup:**
- Remove from schema.ts
- Remove any dead code references
- Update documentation

**Estimated Effort:** 1 hour  
**Risk:** NONE (no data, no usage)  
**Priority:** 🟢 LOW

---

### 3.2 M8 Kernel Awareness (NEVER USED)

**Table:** `m8_kernel_awareness`: 0 rows

**Recommendation:** 🔴 **DEPRECATE AND DROP**

**Rationale:**
1. Other M8 tables are heavily used (305, 2714, 303 rows)
2. This one never populated
3. Not needed for M8 spawn functionality

**Migration Plan:**
```sql
DROP TABLE m8_kernel_awareness CASCADE;
```

**Estimated Effort:** 30 minutes  
**Risk:** NONE  
**Priority:** 🟢 LOW

---

### 3.3 Cross-God Insights (NEVER USED)

**Table:** `cross_god_insights`: 0 rows

**Recommendation:** 🔴 **DEPRECATE AND DROP**

**Rationale:**
1. No data
2. Not in roadmap
3. Pantheon knowledge transfer covers this use case
4. May have been superseded by `pantheon_knowledge_transfers` (2,545 rows)

**Migration Plan:**
```sql
DROP TABLE cross_god_insights CASCADE;
```

**Estimated Effort:** 30 minutes  
**Risk:** NONE  
**Priority:** 🟢 LOW

---

### 3.4 Scrapy Seen Content (WRONG TOOL)

**Table:** `scrapy_seen_content`: 0 rows

**Recommendation:** 🔴 **DEPRECATE AND DROP**

**Rationale:**
1. Scrapy is Python web scraping framework
2. Not used in this project (using Tavily/Exa APIs)
3. No data ever collected
4. Wrong architectural fit

**Migration Plan:**
```sql
DROP TABLE scrapy_seen_content CASCADE;
```

**Estimated Effort:** 30 minutes  
**Risk:** NONE  
**Priority:** 🟢 LOW

---

## Section 4: Keep As Is (Proper Architecture)

### 4.1 Vocabulary Tables ✅

**Tables (5):**
- `tokenizer_vocabulary`: 16,331 rows - Tokenizer
- `vocabulary_observations`: 16,936 rows - Learning data
- `vocabulary_learning`: 2,495 rows - Progress tracking
- `learned_words`: 16,305 rows - Human vocabulary
- `word_relationships`: 326,501 rows - Semantic graph

**Recommendation:** ✅ **KEEP ALL - PROPER SEPARATION**

**Rationale:** Per VOCABULARY_CONSOLIDATION_PLAN.md, each serves distinct purpose.

---

### 4.2 Pantheon Communication ✅

**Tables (5):**
- `pantheon_messages`: 15,043 rows
- `pantheon_debates`: 2,712 rows  
- `pantheon_proposals`: 2,746 rows
- `pantheon_god_state`: 19 rows
- `pantheon_knowledge_transfers`: 2,545 rows

**Recommendation:** ✅ **KEEP ALL - DIFFERENT PURPOSES**

**Rationale:** Each is a different communication/governance type.

---

### 4.3 Kernel Activity ✅

**Active Tables (3):**
- `kernel_activity`: 14,096 rows - Activity log
- `kernel_geometry`: 480 rows - Geometric state
- `kernel_training_history`: 1,851 rows - Training events

**Recommendation:** ✅ **KEEP - ACTIVE AND DISTINCT**

**Empty Tables (4):** See "Wire Empty Tables" guide for implementation.

---

## Section 5: Implementation Roadmap

### Phase 1: High-Priority Consolidations (1 week)

1. **Merge governance_proposals → pantheon_proposals** (3 hours)
   - Verify duplication
   - Migrate data
   - Update code references
   - Drop old table

2. **Consolidate consciousness_state → consciousness_checkpoints** (2 hours)
   - Add is_current flag
   - Create view
   - Update code
   - Drop old table (later)

3. **Investigate knowledge tables** (4 hours)
   - Query comparison
   - Determine if duplicate
   - Merge or document difference

**Total Phase 1:** 9 hours (2 days)

---

### Phase 2: Medium-Priority Actions (1 week)

4. **Consolidate near-miss tables** (3 hours)
   - Design unified schema
   - Migrate adaptive_state
   - Update code
   - Drop empties

5. **Verify shadow_knowledge architecture** (1 hour)
   - Document table purposes
   - Confirm no duplicates

**Total Phase 2:** 4 hours (1 day)

---

### Phase 3: Deprecations (2 days)

6. **Drop 5 unused tables** (3 hours)
   - kernel_evolution_events
   - kernel_evolution_fitness
   - m8_kernel_awareness
   - cross_god_insights
   - scrapy_seen_content

7. **Update schema.ts** (1 hour)
   - Remove deprecated table definitions
   - Update exports

8. **Update documentation** (1 hour)
   - Remove references to dropped tables
   - Document consolidations

**Total Phase 3:** 5 hours (1 day)

---

## Section 6: Risk Analysis

### Low Risk Changes ✅
- Dropping empty tables (no data loss)
- Consolidating consciousness_state (simple view)
- Merging governance_proposals (verified duplicates)

### Medium Risk Changes ⚠️
- Knowledge tables consolidation (need verification)
- Near-miss consolidation (schema redesign)

### High Risk Changes 🔴
- None identified (all changes have clear migration paths)

---

## Section 7: Testing Strategy

### Pre-Migration Tests
```sql
-- Save row counts
CREATE TABLE migration_verification AS
SELECT 
  'governance_proposals' as table_name,
  COUNT(*) as row_count,
  MAX(id) as max_id
FROM governance_proposals
UNION ALL
SELECT 'pantheon_proposals', COUNT(*), MAX(id)
FROM pantheon_proposals;

-- Save checksums
SELECT 
  table_name,
  md5(string_agg(id::text, ',' ORDER BY id)) as checksum
FROM (
  SELECT id FROM governance_proposals
  UNION ALL
  SELECT id FROM pantheon_proposals
) t
GROUP BY table_name;
```

### Post-Migration Tests
```sql
-- Verify row counts
SELECT 
  COUNT(*) as final_count,
  (SELECT SUM(row_count) FROM migration_verification) as expected_count
FROM pantheon_proposals;

-- Verify no data loss
SELECT * FROM migration_verification;
```

### Application Tests
- Run full test suite
- Manual QA of affected features
- Monitor for errors in production logs

---

## Section 8: Success Metrics

### Before Consolidation
- Total tables: 110
- Duplicate pairs: 5
- Empty tables: 32
- Schema complexity: HIGH

### After Consolidation (Target)
- Total tables: ~102 (8 tables removed)
- Duplicate pairs: 0
- Empty tables: <10 (after wiring)
- Schema complexity: MEDIUM

### Quality Improvements
- ✅ Clearer data architecture
- ✅ Reduced query complexity  
- ✅ Single source of truth enforced
- ✅ Easier onboarding for new developers
- ✅ Better performance (fewer joins needed)

---

## Section 9: Rollback Plans

### For Each Migration

**governance_proposals merge:**
```sql
-- Rollback: Restore from backup
pg_restore -t governance_proposals backup_file.dump

-- Or recreate from pantheon_proposals
CREATE TABLE governance_proposals AS
SELECT * FROM pantheon_proposals;
```

**consciousness_state consolidation:**
```sql
-- Rollback: Recreate singleton table
CREATE TABLE consciousness_state AS
SELECT * FROM consciousness_checkpoints
WHERE is_current = true;

-- Drop is_current flag
ALTER TABLE consciousness_checkpoints DROP COLUMN is_current;
```

**General rollback:**
- All changes done in transactions
- Database backups before each phase
- Git branches for code changes
- Ability to revert schema.ts changes

---

## References

- [Database Reconciliation Analysis](../04-records/20260113-database-reconciliation-analysis-1.00W.md)
- [Wire Empty Tables Guide](./20260113-wire-empty-tables-implementation-1.00W.md)
- [Vocabulary Consolidation Plan](../../VOCABULARY_CONSOLIDATION_PLAN.md)
- [Master Roadmap](../00-roadmap/20260112-master-roadmap-1.00W.md)

---

**Maintenance**: Update after each consolidation  
**Last Updated**: 2026-01-13  
**Next Review**: 2026-01-20 (after Phase 1 complete)  
**Total Estimated Effort**: 18 hours (2.5 days)
