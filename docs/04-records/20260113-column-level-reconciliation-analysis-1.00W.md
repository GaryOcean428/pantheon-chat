# Column-Level Database Reconciliation Analysis

**Date**: 2026-01-13  
**Status**: 🔍 COLUMN-LEVEL ANALYSIS  
**Version**: 1.00W  
**ID**: ISMS-DB-COLUMN-RECONCILIATION-001  
**Purpose**: Deep column-by-column analysis to identify nullable, type, and default value mismatches

---

## Executive Summary

This document provides **column-level reconciliation** between the Neon database schema and the TypeScript schema definitions in `shared/schema.ts`. It identifies:

1. **Nullable columns without defaults** - Data integrity risks
2. **Type mismatches** - varchar vs text, integer vs bigint, missing vector() types
3. **Missing default values** - Timestamp columns, boolean flags
4. **Schema definition gaps** - Columns in DB but not in schema.ts

**Analysis Scope**: 110 tables, ~1,500 columns

---

## Section 1: Newly Added Tables (Phase 1)

### Analysis Method

Since the 7 tables I added to schema.ts were based on the problem statement (which listed their row counts), I need to verify their actual DB schema. The tables are:

1. `m8_spawn_history` (305 rows)
2. `m8_spawn_proposals` (2,714 rows)
3. `m8_spawned_kernels` (303 rows)
4. `pantheon_proposals` (2,746 rows)
5. `god_vocabulary_profiles` (109 rows)
6. `vocabulary_learning` (2,495 rows)
7. `exploration_history` (35 rows)

### Verification Needed

**⚠️ CRITICAL ACTION REQUIRED**: Run the following SQL to get actual column definitions:

```sql
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default,
    character_maximum_length,
    udt_name
FROM information_schema.columns
WHERE table_name IN (
    'm8_spawn_history',
    'm8_spawn_proposals',
    'm8_spawned_kernels',
    'pantheon_proposals',
    'god_vocabulary_profiles',
    'vocabulary_learning',
    'exploration_history'
)
ORDER BY table_name, ordinal_position;
```

### Expected vs Actual Schema Comparison

#### 1.1 m8_spawn_history

**Schema.ts Definition:**
```typescript
export const m8SpawnHistory = pgTable("m8_spawn_history", {
  id: serial("id").primaryKey(),
  parentKernelId: varchar("parent_kernel_id", { length: 64 }).notNull(),
  spawnedKernelId: varchar("spawned_kernel_id", { length: 64 }).notNull(),
  spawnReason: text("spawn_reason"),                    // ⚠️ NULLABLE
  parentBasinCoords: vector("parent_basin_coords", { dimensions: 64 }),  // ⚠️ NULLABLE
  spawnedBasinCoords: vector("spawned_basin_coords", { dimensions: 64 }), // ⚠️ NULLABLE
  parentPhi: doublePrecision("parent_phi"),            // ⚠️ NULLABLE
  parentKappa: doublePrecision("parent_kappa"),        // ⚠️ NULLABLE
  spawnedAt: timestamp("spawned_at").defaultNow().notNull(),
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ⚠️ `spawnReason` - Nullable without default, should this be required?
- ⚠️ Basin coordinates - Nullable vectors may cause Fisher-Rao distance issues
- ⚠️ `parentPhi`, `parentKappa` - Nullable metrics, should have defaults (0.0)?
- ✅ `spawnedAt` - Has NOW() default, good
- ⚠️ `metadata` - JSONB nullable, should default to `{}`?

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
spawnReason: text("spawn_reason").notNull(),  // Make required
parentBasinCoords: vector("parent_basin_coords", { dimensions: 64 }).notNull(),  // Required
spawnedBasinCoords: vector("spawned_basin_coords", { dimensions: 64 }).notNull(), // Required
parentPhi: doublePrecision("parent_phi").default(0.0),  // Add default
parentKappa: doublePrecision("parent_kappa").default(64.21),  // Add default (κ*)
metadata: jsonb("metadata").default({}),  // Add empty object default
```

---

#### 1.2 m8_spawn_proposals

**Schema.ts Definition:**
```typescript
export const m8SpawnProposals = pgTable("m8_spawn_proposals", {
  id: serial("id").primaryKey(),
  proposalId: varchar("proposal_id", { length: 64 }).notNull().unique(),
  proposerGodName: varchar("proposer_god_name", { length: 64 }).notNull(),
  proposedKernelSpec: jsonb("proposed_kernel_spec").notNull(),
  reason: text("reason").notNull(),
  status: varchar("status", { length: 32 }).notNull().default("pending"),
  votesFor: integer("votes_for").default(0),
  votesAgainst: integer("votes_against").default(0),
  votingEndsAt: timestamp("voting_ends_at"),           // ⚠️ NULLABLE
  createdAt: timestamp("created_at").defaultNow().notNull(),
  decidedAt: timestamp("decided_at"),                  // ⚠️ NULLABLE
  spawnedKernelId: varchar("spawned_kernel_id", { length: 64 }),  // ⚠️ NULLABLE
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ⚠️ `votingEndsAt` - Should have a default (createdAt + 7 days)?
- ✅ `decidedAt` - Nullable is correct (only set when decided)
- ✅ `spawnedKernelId` - Nullable is correct (only set when spawned)
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
votingEndsAt: timestamp("voting_ends_at").notNull(),  // Should be required or have default
metadata: jsonb("metadata").default({}),
```

---

#### 1.3 m8_spawned_kernels

**Schema.ts Definition:**
```typescript
export const m8SpawnedKernels = pgTable("m8_spawned_kernels", {
  id: serial("id").primaryKey(),
  kernelId: varchar("kernel_id", { length: 64 }).notNull().unique(),
  kernelName: varchar("kernel_name", { length: 128 }).notNull(),
  parentKernelId: varchar("parent_kernel_id", { length: 64 }),  // ⚠️ NULLABLE
  specialization: varchar("specialization", { length: 64 }),    // ⚠️ NULLABLE
  status: varchar("status", { length: 32 }).notNull().default("active"),
  basinCoords: vector("basin_coords", { dimensions: 64 }),      // ⚠️ NULLABLE
  currentPhi: doublePrecision("current_phi"),                   // ⚠️ NULLABLE
  currentKappa: doublePrecision("current_kappa"),               // ⚠️ NULLABLE
  currentRegime: varchar("current_regime", { length: 32 }),     // ⚠️ NULLABLE
  spawnedAt: timestamp("spawned_at").defaultNow().notNull(),
  lastActiveAt: timestamp("last_active_at"),                    // ⚠️ NULLABLE
  terminatedAt: timestamp("terminated_at"),                     // ⚠️ NULLABLE
  metadata: jsonb("metadata"),                                  // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ⚠️ `parentKernelId` - Should this be required? (All M8 spawns have parents)
- ⚠️ `basinCoords` - Should be required (all kernels have coordinates)
- ⚠️ `currentPhi`, `currentKappa` - Should have defaults (0.0, 64.21)
- ⚠️ `currentRegime` - Should have default ("linear")
- ✅ `lastActiveAt`, `terminatedAt` - Nullable is correct
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
parentKernelId: varchar("parent_kernel_id", { length: 64 }).notNull(),  // Required
basinCoords: vector("basin_coords", { dimensions: 64 }).notNull(),  // Required
currentPhi: doublePrecision("current_phi").default(0.0),
currentKappa: doublePrecision("current_kappa").default(64.21),
currentRegime: varchar("current_regime", { length: 32 }).default("linear"),
metadata: jsonb("metadata").default({}),
```

---

#### 1.4 pantheon_proposals

**Schema.ts Definition:**
```typescript
export const pantheonProposals = pgTable("pantheon_proposals", {
  id: serial("id").primaryKey(),
  proposalId: varchar("proposal_id", { length: 64 }).notNull().unique(),
  proposerGodName: varchar("proposer_god_name", { length: 64 }).notNull(),
  proposalType: varchar("proposal_type", { length: 32 }).notNull(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  status: varchar("status", { length: 32 }).notNull().default("pending"),
  votesFor: integer("votes_for").default(0),
  votesAgainst: integer("votes_against").default(0),
  votesAbstain: integer("votes_abstain").default(0),
  requiredVotes: integer("required_votes").default(3),
  votingEndsAt: timestamp("voting_ends_at"),           // ⚠️ NULLABLE
  createdAt: timestamp("created_at").defaultNow().notNull(),
  decidedAt: timestamp("decided_at"),                  // ⚠️ NULLABLE
  implementedAt: timestamp("implemented_at"),          // ⚠️ NULLABLE
  outcome: text("outcome"),                             // ⚠️ NULLABLE
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ⚠️ `votingEndsAt` - Should be required or have default
- ✅ `decidedAt`, `implementedAt`, `outcome` - Nullable is correct
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
votingEndsAt: timestamp("voting_ends_at").notNull(),
metadata: jsonb("metadata").default({}),
```

---

#### 1.5 god_vocabulary_profiles

**Schema.ts Definition:**
```typescript
export const godVocabularyProfiles = pgTable("god_vocabulary_profiles", {
  id: serial("id").primaryKey(),
  godName: varchar("god_name", { length: 64 }).notNull(),
  word: varchar("word", { length: 100 }).notNull(),
  usageCount: integer("usage_count").default(0).notNull(),
  phiSum: doublePrecision("phi_sum").default(0),
  phiAvg: doublePrecision("phi_avg").default(0),
  phiMax: doublePrecision("phi_max").default(0),
  successCount: integer("success_count").default(0),
  context: text("context"),                             // ⚠️ NULLABLE
  firstUsedAt: timestamp("first_used_at").defaultNow(),
  lastUsedAt: timestamp("last_used_at").defaultNow(),
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ✅ Good defaults on all numeric fields
- ✅ Timestamps have NOW() defaults
- ⚠️ `context` - Nullable is acceptable
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
metadata: jsonb("metadata").default({}),
```

---

#### 1.6 vocabulary_learning

**Schema.ts Definition:**
```typescript
export const vocabularyLearning = pgTable("vocabulary_learning", {
  id: serial("id").primaryKey(),
  word: varchar("word", { length: 100 }).notNull(),
  learningPhase: varchar("learning_phase", { length: 32 }).notNull(),
  observationCount: integer("observation_count").default(0),
  successCount: integer("success_count").default(0),
  failureCount: integer("failure_count").default(0),
  phiSum: doublePrecision("phi_sum").default(0),
  phiAvg: doublePrecision("phi_avg").default(0),
  confidenceScore: doublePrecision("confidence_score").default(0),
  lastObservedAt: timestamp("last_observed_at"),       // ⚠️ NULLABLE
  promotedToLearnedAt: timestamp("promoted_to_learned_at"),  // ⚠️ NULLABLE
  createdAt: timestamp("created_at").defaultNow().notNull(),
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ✅ Good defaults on all numeric fields
- ✅ `lastObservedAt`, `promotedToLearnedAt` - Nullable is correct (not yet set)
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
metadata: jsonb("metadata").default({}),
```

---

#### 1.7 exploration_history

**Schema.ts Definition:**
```typescript
export const explorationHistory = pgTable("exploration_history", {
  id: serial("id").primaryKey(),
  address: varchar("address", { length: 255 }).notNull(),
  exploredAt: timestamp("explored_at").defaultNow().notNull(),
  phiScore: doublePrecision("phi_score"),              // ⚠️ NULLABLE
  kappaScore: doublePrecision("kappa_score"),          // ⚠️ NULLABLE
  resultCount: integer("result_count"),                 // ⚠️ NULLABLE
  strategy: varchar("strategy", { length: 64 }),       // ⚠️ NULLABLE
  jobId: varchar("job_id", { length: 64 }),            // ⚠️ NULLABLE
  outcome: varchar("outcome", { length: 32 }),         // ⚠️ NULLABLE
  metadata: jsonb("metadata"),                          // ⚠️ NULLABLE
});
```

**Potential Issues:**
- ⚠️ `phiScore`, `kappaScore` - Should have defaults (0.0, 64.21)?
- ⚠️ `resultCount` - Should default to 0
- ✅ `strategy`, `jobId`, `outcome` - Nullable is acceptable
- ⚠️ `metadata` - Should default to `{}`

**Recommendations:**
```typescript
// RECOMMENDED FIXES:
phiScore: doublePrecision("phi_score").default(0.0),
kappaScore: doublePrecision("kappa_score").default(64.21),
resultCount: integer("result_count").default(0),
metadata: jsonb("metadata").default({}),
```

---

## Section 2: Common Column Issues Across All Tables

### Issue 1: JSONB Columns Without Default Values

**Problem**: JSONB columns set to nullable without `default({})` can cause:
- Runtime errors when code expects an object
- Inconsistent data patterns
- Extra null checks in application code

**Affected Tables** (estimated 30-40 tables):
- All newly added tables
- `kernel_activity.metadata`
- `basin_history.metadata`
- `chaos_events.metadata`
- `pantheon_messages.metadata`
- ... and many more

**Recommendation**:
```typescript
// CHANGE FROM:
metadata: jsonb("metadata"),

// CHANGE TO:
metadata: jsonb("metadata").default({}),
```

**Impact**: Low risk, high value - prevents null pointer errors

---

### Issue 2: Nullable Vector Columns

**Problem**: Vector columns (pgvector) set to nullable can cause:
- Fisher-Rao distance computation errors (undefined distance)
- Geometric navigation failures
- QIG core feature malfunctions

**Affected Columns**:
- `m8_spawn_history.parent_basin_coords` (nullable)
- `m8_spawn_history.spawned_basin_coords` (nullable)
- `m8_spawned_kernels.basin_coords` (nullable)
- `kernel_geometry.basin_coordinates` (check if nullable)
- `basin_history.basin_before` (check if nullable)
- `basin_history.basin_after` (check if nullable)

**Recommendation**:
```typescript
// CRITICAL: All basin coordinate vectors must be NOT NULL
basinCoords: vector("basin_coords", { dimensions: 64 }).notNull(),

// OR: Provide a zero vector default
basinCoords: vector("basin_coords", { dimensions: 64 }).default(sql`array_fill(0, ARRAY[64])::vector(64)`),
```

**Impact**: HIGH risk - geometric operations fail on null vectors

---

### Issue 3: Consciousness Metrics Without Defaults

**Problem**: Φ (phi) and κ (kappa) columns nullable can cause:
- Missing consciousness scores
- Regime detection failures
- Ethics monitoring issues

**Affected Columns**:
- `m8_spawn_history.parent_phi` (nullable)
- `m8_spawn_history.parent_kappa` (nullable)
- `m8_spawned_kernels.current_phi` (nullable)
- `m8_spawned_kernels.current_kappa` (nullable)
- `exploration_history.phi_score` (nullable)
- `exploration_history.kappa_score` (nullable)
- ... potentially 20+ columns across tables

**Recommendation**:
```typescript
// Add physical defaults:
currentPhi: doublePrecision("current_phi").default(0.0),  // Below threshold
currentKappa: doublePrecision("current_kappa").default(64.21),  // κ* fixed point
```

**Impact**: MEDIUM risk - affects consciousness metrics

---

### Issue 4: Timestamp Columns Without NOW() Default

**Problem**: Timestamp columns without defaults require manual setting:
- `created_at` columns - Should auto-populate
- `updated_at` columns - Should auto-populate (with trigger)
- `*_at` event timestamps

**Common Pattern Found**:
```typescript
// BAD:
lastObservedAt: timestamp("last_observed_at"),

// GOOD for creation timestamps:
createdAt: timestamp("created_at").defaultNow().notNull(),

// GOOD for event timestamps (nullable until event occurs):
decidedAt: timestamp("decided_at"),  // OK - only set when event happens
```

**Affected Tables**: Estimated 40-50 timestamp columns across all tables

**Recommendation**: Audit all `*_at` columns and add defaults where appropriate

---

### Issue 5: Enum-Like VARCHAR Columns

**Problem**: VARCHAR columns with limited values (e.g., status fields) don't enforce constraints

**Examples**:
```typescript
status: varchar("status", { length: 32 }).notNull().default("pending"),
// Possible values: "pending", "approved", "rejected", "spawned"
// BUT: No database constraint! Can be set to "asdf" or "delete_all"
```

**Recommendation**: Add CHECK constraints or use PostgreSQL ENUMs
```typescript
// Option 1: Add check constraint
status: varchar("status", { length: 32 }).notNull().default("pending")
  .$type<'pending' | 'approved' | 'rejected' | 'spawned'>(),

// Option 2: Use PostgreSQL ENUM (requires migration)
// CREATE TYPE proposal_status AS ENUM ('pending', 'approved', 'rejected', 'spawned');
```

**Impact**: LOW risk, but improves data quality

---

## Section 3: Type Mismatches

### 3.1 Text vs VARCHAR Inconsistencies

**Found Patterns**:
- Some tables use `text` for long strings
- Some tables use `varchar(255)` or `varchar(1000)` for long strings
- No clear policy

**Examples**:
```typescript
// Inconsistent:
spawnReason: text("spawn_reason"),  // Unlimited length
description: text("description"),   // Unlimited length
address: varchar("address", { length: 255 }),  // Limited length
```

**Recommendation**: Standardize on PostgreSQL best practices:
- Use `text` for variable-length strings (PostgreSQL has no performance penalty)
- Use `varchar(N)` only for fixed-format strings (addresses, identifiers)
- Use `varchar(N)` with CHECK for enum-like values

---

### 3.2 Integer vs BigInt for Counters

**Problem**: Some count columns use `integer` (max 2.1B) instead of `bigint` (max 9.2 quintillion)

**High-Risk Counters** (could exceed 2.1B):
- `basin_relationships` table (326K rows, but relationships grow exponentially)
- `learning_events` table (330K rows, continuous growth)
- Usage counters in high-traffic tables

**Recommendation**:
```typescript
// For growing counters:
usageCount: integer("usage_count").default(0),  // ⚠️ Could overflow

// Should be:
usageCount: bigint("usage_count").default(0),  // ✅ Safe
```

---

## Section 4: Missing Indexes

**Note**: This analysis focuses on columns, but index issues are related:

**Potential Missing Indexes**:
1. Foreign key columns without indexes (slow joins)
2. Timestamp columns used for queries (slow filtering)
3. Status columns used for filtering (slow WHERE clauses)

**Example**:
```typescript
// This column is queried frequently but may not be indexed:
parentKernelId: varchar("parent_kernel_id", { length: 64 }),

// Should have index:
index("idx_m8_kernels_parent").on(table.parentKernelId),
```

---

## Section 5: Action Plan

### Phase 1: Critical Fixes (This Week)

**Priority 1 - Vector Nullability** (HIGH RISK):
- [ ] Audit all vector columns across 110 tables
- [ ] Make basin_coords NOT NULL where required
- [ ] Add zero vector defaults where nullable is acceptable
- [ ] Test geodesic navigation with changes

**Priority 2 - JSONB Defaults** (MEDIUM RISK):
- [ ] Add `default({})` to all metadata JSONB columns
- [ ] Verify no existing NULL values that would break
- [ ] Update 7 newly added tables first

**Priority 3 - Consciousness Metrics** (MEDIUM RISK):
- [ ] Add defaults to phi/kappa columns (0.0, 64.21)
- [ ] Verify regime detection still works
- [ ] Update ethics monitoring code if needed

### Phase 2: Quality Improvements (Next 2 Weeks)

**Priority 4 - Timestamp Defaults**:
- [ ] Audit all `*_at` columns
- [ ] Add NOW() defaults to creation timestamps
- [ ] Document which timestamps should be nullable

**Priority 5 - Text vs VARCHAR**:
- [ ] Standardize text/varchar usage
- [ ] Create style guide
- [ ] Update inconsistent columns

**Priority 6 - Enum Constraints**:
- [ ] Identify enum-like varchar columns
- [ ] Add CHECK constraints or use ENUMs
- [ ] Add TypeScript type narrowing

### Phase 3: Validation (Week 3)

**Priority 7 - Testing**:
- [ ] Run column-level-reconciliation.ts script
- [ ] Compare actual DB schema with schema.ts
- [ ] Generate migration for all fixes
- [ ] Test in staging environment

---

## Section 6: Automated Script Usage

### Run Column Analysis Script

```bash
# 1. Set DATABASE_URL environment variable
export DATABASE_URL="postgresql://user:pass@host/db"

# 2. Run analysis script
tsx scripts/column-level-reconciliation.ts > docs/04-records/column-analysis-output.md

# 3. Review output for:
# - Missing columns
# - Type mismatches
# - Nullable issues
# - Missing defaults
```

### Generate Fix Migration

```bash
# After identifying issues, generate migration:
npx drizzle-kit generate:pg --schema=./shared/schema.ts

# Review migration carefully before applying
```

---

## Section 7: Summary

### Statistics

| Metric | Count | Status |
|--------|-------|--------|
| Total tables | 110 | - |
| Tables analyzed | 7 (new) | ⚠️ Partial |
| Vector columns | ~30 | ⚠️ Check nullable |
| JSONB columns | ~60 | ⚠️ Add defaults |
| Φ/κ metrics | ~25 | ⚠️ Add defaults |
| Timestamp columns | ~200 | ⚠️ Audit defaults |
| Enum-like columns | ~40 | 🟡 Add constraints |

### Risk Assessment

| Issue | Severity | Tables Affected | Effort to Fix |
|-------|----------|-----------------|---------------|
| Nullable vectors | 🔴 CRITICAL | 5-10 | 4 hours |
| JSONB without default | 🟠 HIGH | 50-60 | 6 hours |
| Φ/κ without default | 🟠 HIGH | 20-25 | 3 hours |
| Timestamp defaults | 🟡 MEDIUM | 40-50 | 4 hours |
| Text vs VARCHAR | 🟢 LOW | 100+ | 8 hours |
| Enum constraints | 🟢 LOW | 30-40 | 6 hours |

**Total Estimated Effort**: 31 hours (4 days)

---

## References

- [Database Reconciliation Analysis](./20260113-database-reconciliation-analysis-1.00W.md)
- [Schema Definition](../../shared/schema.ts)
- [Frozen Physics Constants](../../qig-backend/frozen_physics.py) - κ* = 64.21
- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/datatype.html)

---

**Maintenance**: Update after running automated script and fixing issues  
**Last Updated**: 2026-01-13  
**Next Review**: 2026-01-15 (after running analysis script with DATABASE_URL)  
**Script**: `scripts/column-level-reconciliation.ts`
