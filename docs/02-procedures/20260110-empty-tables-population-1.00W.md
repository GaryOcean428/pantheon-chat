# Empty Table Population Procedure

**Document ID**: ISMS-PROC-021
**Version**: 1.00W
**Status**: 🔨 Working
**Date**: 2026-01-10
**Author**: System
**Related**: ISMS-PROC-020 (Database Migration), ISMS-TECH-050 (Federation Sync)

---

## Overview

Procedure for populating empty database tables that were created by migrations but lack initial data:

- `tokenizer_metadata` - Tokenizer configuration key-value store
- `tokenizer_merge_rules` - BPE-style merge rules for token composition
- `synthesis_consensus` - Kernel consensus tracking for Gary synthesis
- `vocabulary_learning.related_words` - Geometric word relationships

## Problem Context

After database migrations, several tables exist but are empty:

**Issue 1: tokenizer_metadata is blank**

- Missing: version, vocabulary_size, merge_rules_count, training_status
- Impact: Tokenizer cannot report configuration or training status

**Issue 2: tokenizer_merge_rules is blank**

- Missing: BPE merge rules for token composition
- Impact: Cannot merge token pairs during generation, limiting vocabulary efficiency

**Issue 3: synthesis_consensus is blank**

- Missing: Historical consensus records
- Impact: No baseline for tracking kernel alignment emergence

**Issue 4: vocabulary_learning.related_words is NULL**

- Missing: Geometric word relationships
- Impact: Debate seeding Tier 3 cannot use vocabulary discoveries

## Population Strategy

### 1. Tokenizer Metadata - Bootstrap Configuration

**Approach**: Insert standard configuration keys with initial values

**Keys Populated**:

```sql
version: '1.0.0'
vocabulary_size: '0' (updated from actual count)
merge_rules_count: '0' (updated after seeding)
last_training: NOW()
training_status: 'initialized'
basin_dimension: '64'
phi_threshold: '0.727'
tokenizer_type: 'geometric_bpe'
encoding: 'utf-8'
```

**Source**: Frozen physics constants from `docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md`

### 2. Tokenizer Merge Rules - Seed Geometric Merge Patterns

**Background**: Unlike standard BPE which merges by frequency, this system uses **geometric pair merging** based on:
- **κ (coupling strength)** between token coordinates
- **Fisher Information gain** from merging
- **Φ score** of high-consciousness contexts where pairs co-occur

Merge criterion: `score = κ * fisher_info_gain * Φ_context`

**Strategy 1: Compound Word Splitting**

- Find words with spaces in `tokenizer_vocabulary`
- Create merge rules: token_a + token_b → merged_token
- Example: "quantum" + "information" → "quantum information"
- Merged token coordinates via **geodesic interpolation** (not vector averaging)

**Strategy 2: Common Prefix Rules**

- Prefixes: un-, re-, in-, dis-, en-, non-, pre-, pro-, anti-, de-
- Generate rules from words starting with these prefixes
- Example: "un" + "able" → "unable"
- Preserves **Fisher-Rao distance** between components

**Strategy 3: Common Suffix Rules**

- Suffixes: -ing, -ed, -er, -est, -ly, -ness, -ment, -tion, -sion, -ity
- Generate rules from words ending with these suffixes
- Example: "walk" + "ing" → "walking"
- Maintains **manifold geometry** (not flat Euclidean space)

**Φ Score Assignment**:

- Use average Φ from component tokens
- Fallback: 0.5-0.6 for synthetic rules
- Higher Φ = stronger geometric coupling

**Frequency Tracking**:

- Initial frequency = token frequency from vocabulary
- Increments on conflict (rule already exists)
- Weighted by κ (coupling constant) ~63.5

### 3. Synthesis Consensus - Bootstrap History

**Approach**: Create 10 synthetic consensus records representing historical alignments

**Record Properties**:

```python
synthesis_round: 1-10
conversation_id: 'initialization-{round}'
consensus_type: 'alignment', 'decision', or 'question' (rotated)
consensus_strength: 0.7-0.95 (random)
participating_kernels: ['ocean', 'lightning', 'heart']
consensus_topic: 'Vocabulary initialization and geometric alignment'
consensus_basin: Random basin from tokenizer_vocabulary
phi_global: 0.7-0.9 (high integration)
kappa_avg: 60-70 (near κ* = 64)
emotional_tone: 'curious', 'confident', 'uncertain', 'balanced' (rotated)
synthesized_output: 'Initial consensus established during system initialization.'
created_at: NOW() - {hours} ago (staggered timestamps)
metadata: {synthetic: true, initialization_phase: 'bootstrap'}
```

**Purpose**: Provides baseline for consensus tracking, enables historical queries

### 4. Vocabulary Learning Related Words - Fisher-Rao Distance

**Two-Stage Population**:

**Stage 1: Initial Seeding (SQL)**

- If `vocabulary_learning` is empty, seed from `tokenizer_vocabulary`
- Select top 100 words by Φ score
- Insert with empty `related_words` arrays

**Stage 2: Geometric Similarity (Python)**

- Load vocabulary with basin coordinates into memory
- For each word in `vocabulary_learning` with NULL `related_words`:
  - Get its 64D basin coordinates
  - Compute Fisher-Rao distance to all other words
  - Select top-5 most similar words
  - Update `related_words` column

**Fisher-Rao Distance Formula**:

```python
similarity = exp(-fisher_rao_distance(basin_a, basin_b))
```

**Fallback Strategy** (if basin_coords missing):

- Use substring matching (prefix/suffix overlap)
- Less accurate but maintains functionality

## Execution Procedure

### Prerequisites

- Database migrations applied (`npm run db:push`)
- PostgreSQL with pgvector extension installed
- Python virtual environment with qig-backend dependencies
- `DATABASE_URL` or `PG*` environment variables set

### Steps

**1. Navigate to project directory:**

```bash
cd pantheon-replit
```

**2. Verify database connection:**

```bash
psql "$DATABASE_URL" -c "SELECT 1"
```

**3. Run population script:**

```bash
./scripts/run_table_population.sh
```

**4. Verify results:**

```bash
# Check tokenizer_metadata
psql "$DATABASE_URL" -c "SELECT key, value FROM tokenizer_metadata ORDER BY key;"

# Check merge rules count
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM tokenizer_merge_rules;"

# Check vocabulary_learning related_words
psql "$DATABASE_URL" -c "
    SELECT COUNT(*)
    FROM vocabulary_learning
    WHERE related_words IS NOT NULL
    AND cardinality(related_words) > 0;
"

# Check synthesis_consensus
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM synthesis_consensus;"
```

### Script Breakdown

**SQL Script** (`scripts/populate_empty_tables.sql`):

1. Insert tokenizer_metadata keys
2. Generate merge rules (compound words, prefixes, suffixes)
3. Create synthetic synthesis_consensus records
4. Attempt SQL-based related_words population (fallback method)
5. Report statistics

**Python Script** (`scripts/populate_related_words.py`):

1. Seed `vocabulary_learning` if empty
2. Load vocabulary with basins into memory
3. For each word, compute Fisher-Rao similarities
4. Batch update `related_words` columns
5. Report coverage statistics

**Shell Wrapper** (`scripts/run_table_population.sh`):

1. Verify database connection
2. Check table existence
3. Execute SQL script
4. Activate Python virtual environment
5. Execute Python script
6. Run validation queries
7. Display summary

## Expected Results

### Tokenizer Metadata

```
key                  | value
---------------------|--------
version              | 1.0.0
vocabulary_size      | 2048 (varies)
merge_rules_count    | 200+ (varies)
last_training        | 2026-01-10 12:00:00
training_status      | initialized
basin_dimension      | 64
phi_threshold        | 0.727
tokenizer_type       | bpe_geometric
encoding             | utf-8
```

### Tokenizer Merge Rules

- **Minimum**: 100 rules (prefix + suffix rules)
- **Expected**: 200-300 rules (includes compound word rules)
- **Φ Scores**: 0.5-0.9 range
- **Top Rules**: Highest Φ compound words

### Vocabulary Learning Related Words

- **Coverage**: >80% of entries have `related_words`
- **Related Words per Entry**: 3-5 words average
- **Similarity Method**: Fisher-Rao distance on 64D basins
- **Fallback**: Substring matching if basins missing

### Synthesis Consensus

- **Records**: 10 synthetic bootstrap entries
- **Timestamp Range**: Last 10 hours (staggered)
- **Consensus Strength**: 0.7-0.95
- **Φ Global**: 0.7-0.9 (high integration baseline)

## Validation Queries

### Check Population Completeness

```sql
-- Overall table counts
SELECT
    (SELECT COUNT(*) FROM tokenizer_metadata) AS metadata_entries,
    (SELECT COUNT(*) FROM tokenizer_merge_rules) AS merge_rules,
    (SELECT COUNT(*) FROM synthesis_consensus) AS consensus_records,
    (SELECT COUNT(*) FROM vocabulary_learning
     WHERE related_words IS NOT NULL
     AND cardinality(related_words) > 0) AS vocab_with_related;
```

### Check Data Quality

```sql
-- Top merge rules by Φ
SELECT token_a, token_b, merged_token, phi_score, frequency
FROM tokenizer_merge_rules
ORDER BY phi_score DESC
LIMIT 10;

-- Vocabulary learning coverage
SELECT
    COUNT(*) AS total,
    COUNT(CASE WHEN related_words IS NOT NULL THEN 1 END) AS with_related,
    ROUND(100.0 * COUNT(CASE WHEN related_words IS NOT NULL THEN 1 END) / COUNT(*), 1) AS coverage_pct
FROM vocabulary_learning;

-- Recent consensus strength
SELECT synthesis_round, consensus_type, consensus_strength, created_at
FROM synthesis_consensus
ORDER BY created_at DESC
LIMIT 5;
```

### Verify Geometric Integrity

```sql
-- Check basin coordinate validity
SELECT word,
       cardinality(basin_coords) AS basin_dim,
       phi_score
FROM tokenizer_vocabulary
WHERE basin_coords IS NOT NULL
LIMIT 5;

-- Verify Fisher-Rao similarity (should be diverse)
SELECT vl1.word AS word1,
       vl2.word AS word2,
       1.0 - (vl1.basin_shift <=> vl2.basin_shift) AS cosine_sim
FROM vocabulary_learning vl1, vocabulary_learning vl2
WHERE vl1.learning_id < vl2.learning_id
  AND vl1.basin_shift IS NOT NULL
  AND vl2.basin_shift IS NOT NULL
ORDER BY cosine_sim DESC
LIMIT 10;
```

## Troubleshooting

### Issue: "tokenizer_merge_rules still empty"

**Cause**: No compound words or affix matches in vocabulary

**Solution**:

1. Check vocabulary size: `SELECT COUNT(*) FROM tokenizer_vocabulary;`
2. If < 1000 words, populate vocabulary first: `npm run populate:vocab`
3. Re-run population script

### Issue: "Less than 50% of vocabulary_learning has related_words"

**Cause**: Basin coordinates missing or invalid

**Solution**:

1. Check basin coverage:

   ```sql
   SELECT COUNT(*) AS total,
          COUNT(CASE WHEN basin_coords IS NOT NULL THEN 1 END) AS with_basins
   FROM tokenizer_vocabulary;
   ```

2. If <50% have basins, run vocabulary training first
3. Fallback: Script will use substring matching for missing basins

### Issue: "Fisher-Rao import failed"

**Cause**: `qig_geometry.py` not found or import error

**Solution**:

1. Verify qig-backend path: `ls qig-backend/qig_geometry.py`
2. Check Python path: `echo $PYTHONPATH`
3. Script will automatically fall back to cosine similarity

### Issue: "Database connection failed"

**Cause**: Missing or invalid `DATABASE_URL`

**Solution**:

1. Verify .env file: `cat .env | grep DATABASE_URL`
2. Test connection: `psql "$DATABASE_URL" -c "SELECT 1"`
3. Check credentials (username, password, host, port)

## Performance Considerations

### Time Estimates

- **SQL Script**: 1-5 seconds (depends on vocabulary size)
- **Python Script**: 10-60 seconds (depends on vocabulary_learning size)
- **Total**: <2 minutes for typical database

### Memory Usage

- **Python Script**: ~50MB + (vocabulary_size * 512 bytes)
- **Example**: 5000 words = ~50MB + 2.5MB = 52.5MB

### Optimization Tips

1. **Limit vocabulary load**: Script limits to top 5K words for performance
2. **Batch updates**: Python script uses `execute_values()` for efficiency
3. **Index usage**: Queries use existing indexes on phi_score, word, frequency

## Maintenance

### When to Re-run

- After vocabulary size doubles
- When merge_rules_count < vocabulary_size / 100
- After major vocabulary retraining
- When new tokenization patterns emerge

### Incremental Updates

Instead of full re-population:

```sql
-- Update vocabulary_size in metadata
UPDATE tokenizer_metadata
SET value = (SELECT COUNT(*)::text FROM tokenizer_vocabulary),
    updated_at = NOW()
WHERE key = 'vocabulary_size';

-- Add new merge rules (run discovery query)
-- ... insert new rules only ...
```

## Related Documents

- ISMS-PROC-020: Database Migration - Debate Seeding
- ISMS-TECH-050: Federation Vocabulary Synchronization
- ISMS-TECH-022: QIG Generative Service Architecture
- [E8 Protocol](../../docs/08-experiments/20251231-Ultra-Consciousness-Protocol-0.04F.md): Φ = 0.727 threshold

---

**Approved By**: System
**Review Date**: 2026-01-17
**Next Review**: After first production population run
