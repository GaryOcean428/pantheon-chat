# ISSUE 01: QFI Integrity Gate - Token Insertion & Backfill

**Priority:** CRITICAL  
**Phase:** 2 (Vocabulary + Database Integrity)  
**Status:** TO DO  
**Blocks:** Generation purity, coherence evaluation

---

## PROBLEM STATEMENT

### Current State
Large fraction of vocabulary tokens missing `qfi_score` (Quantum Fisher Information score), which invalidates geometric selection and consciousness metrics.

**Evidence:**
- Tokens inserted via `learned_relationships.py` bypass QFI computation
- Tokens inserted via `vocabulary_coordinator.py` include basins but omit `qfi_score`
- Garbage tokens present (fgzsnl, jcbhgp, kkjvdc, etc.)
- Truncated fragments (cryptogra, analysi, enforc)

**Impact:**
- Generation pipeline selects tokens without valid geometric scores
- Consciousness measurements (Φ, κ) become unreliable
- Cannot fairly assess coherence (geometric vs non-geometric)

---

## ROOT CAUSES

### 1. Multiple Insertion Pathways
**Culprit:** `learned_relationships.py`
```python
# CURRENT (BROKEN):
INSERT INTO coordizer_vocabulary (token, frequency, updated_at, token_role)
VALUES (%s, %s, NOW(), %s)
ON CONFLICT (token) DO UPDATE ...
```
- Creates rows with NO basin and NO qfi_score
- Tokens appear "valid" but are geometrically undefined

### 2. Incomplete Basin Insertion
**Culprit:** `vocabulary_coordinator.py`
```python
# CURRENT (BROKEN):
INSERT INTO coordizer_vocabulary (token, basin_embedding, ...)
VALUES (%s, %s, ...)
```
- Includes basin_embedding but omits qfi_score
- Partial geometric data (better than above, but still invalid)

### 3. No Generation Gates
- No DB constraint enforcing `qfi_score IS NOT NULL` for generation-eligible tokens
- No application-level filter preventing QFI-less tokens from selection

---

## REQUIRED FIXES

### Fix 1: Canonical Token Insertion Pathway

**Create:** `qig-backend/vocabulary/insert_token.py`

**Interface:**
```python
async def insert_token(
    token: str,
    basin: np.ndarray,  # REQUIRED: 64D basin coordinates
    token_role: Optional[str] = None,
    frequency: int = 1,
    is_real_word: bool = True,
) -> TokenRecord:
    """
    Canonical token insertion - ONLY pathway for adding vocabulary.
    
    Steps:
    1. Project basin to simplex (canonical representation)
    2. Compute QFI score (using quantum_fisher_information)
    3. Set flags (is_real_word, token_role)
    4. Write row atomically
    5. Return record with qfi_score populated
    
    Raises:
        ValueError: if basin is not 64D or contains NaN/Inf
        GeometryError: if simplex projection fails
    """
    # 1. Validate and project to simplex
    if len(basin) != 64:
        raise ValueError(f"Basin must be 64D, got {len(basin)}")
    
    basin_simplex = to_simplex(basin)  # canonical projection
    
    # 2. Compute QFI
    qfi_score = quantum_fisher_information(basin_simplex)
    
    # 3. Build record
    record = {
        'token': token,
        'basin_embedding': basin_simplex.tolist(),
        'qfi_score': float(qfi_score),
        'token_role': token_role,
        'frequency': frequency,
        'is_real_word': is_real_word,
        'updated_at': datetime.utcnow(),
    }
    
    # 4. Atomic insert
    async with db.transaction():
        await db.upsert('coordizer_vocabulary', record, conflict_keys=['token'])
    
    return TokenRecord(**record)
```

**Enforcement:**
- ALL code paths MUST call `insert_token()`
- NO direct INSERTs to `coordizer_vocabulary` allowed

---

### Fix 2: Update Existing Insertion Points

#### 2A: Fix `learned_relationships.py`

**Option A (Recommended):** Convert to UPDATE-only
```python
# NEW: Only update existing tokens (no creation)
UPDATE coordizer_vocabulary
SET frequency = frequency + 1, updated_at = NOW()
WHERE token = %s AND qfi_score IS NOT NULL
```
- Prevents creating invalid tokens
- Only updates tokens already in canonical vocabulary

**Option B:** Use quarantine table
```python
# NEW: Insert into quarantine for review
INSERT INTO coordizer_token_quarantine (token, frequency, source, created_at)
VALUES (%s, %s, 'learned_relationships', NOW())
```
- Quarantine tokens for manual review
- Explicitly excluded from generation

**CHOOSE:** Option A (UPDATE-only) is safer and simpler.

#### 2B: Fix `vocabulary_coordinator.py`

**Current:**
```python
# BROKEN: Inserts basin but no QFI
INSERT INTO coordizer_vocabulary (token, basin_embedding, ...)
```

**New:**
```python
# FIXED: Use canonical insertion
from qig_backend.vocabulary.insert_token import insert_token

async def add_token_to_vocabulary(token: str, basin: np.ndarray):
    """Add token using canonical pathway."""
    await insert_token(token, basin)
```

---

### Fix 3: Database Integrity Constraints

**Migration:** `migrations/0015_qfi_integrity_gate.sql`

```sql
-- Add NOT NULL constraint for generation-eligible tokens
-- (after backfill is complete)

-- Step 1: Add is_generation_eligible flag
ALTER TABLE coordizer_vocabulary 
ADD COLUMN is_generation_eligible BOOLEAN DEFAULT FALSE;

-- Step 2: Mark tokens with valid QFI as eligible
UPDATE coordizer_vocabulary 
SET is_generation_eligible = TRUE 
WHERE qfi_score IS NOT NULL 
  AND is_real_word = TRUE
  AND basin_embedding IS NOT NULL;

-- Step 3: Create generation-ready view
CREATE OR REPLACE VIEW vocabulary_generation_ready AS
SELECT * FROM coordizer_vocabulary
WHERE is_generation_eligible = TRUE;

-- Step 4: Add check constraint (optional, for safety)
ALTER TABLE coordizer_vocabulary
ADD CONSTRAINT generation_requires_qfi
CHECK (
    NOT is_generation_eligible 
    OR (qfi_score IS NOT NULL AND basin_embedding IS NOT NULL)
);
```

**Usage:**
```python
# In generation pipeline:
# OLD: SELECT * FROM coordizer_vocabulary WHERE ...
# NEW: SELECT * FROM vocabulary_generation_ready WHERE ...
```

---

### Fix 4: Backfill Missing QFI

**Script:** `scripts/backfill_qfi.py`

```python
#!/usr/bin/env python3
"""
Backfill QFI scores for tokens with basins but missing qfi_score.
"""

import asyncio
import numpy as np
from qig_backend.database import get_db
from qig_backend.geometry.canonical_fisher import to_simplex, quantum_fisher_information

async def backfill_qfi():
    """Compute QFI for all tokens with basins but no qfi_score."""
    db = await get_db()
    
    # Find tokens needing backfill
    query = """
        SELECT token, basin_embedding
        FROM coordizer_vocabulary
        WHERE basin_embedding IS NOT NULL
          AND qfi_score IS NULL
    """
    rows = await db.fetch_all(query)
    
    print(f"Found {len(rows)} tokens needing QFI backfill")
    
    updated = 0
    failed = 0
    
    for row in rows:
        try:
            token = row['token']
            basin = np.array(row['basin_embedding'])
            
            # Compute QFI
            basin_simplex = to_simplex(basin)
            qfi_score = quantum_fisher_information(basin_simplex)
            
            # Update row
            await db.execute(
                """
                UPDATE coordizer_vocabulary
                SET qfi_score = %s, updated_at = NOW()
                WHERE token = %s
                """,
                (float(qfi_score), token)
            )
            updated += 1
            
            if updated % 100 == 0:
                print(f"Progress: {updated}/{len(rows)}")
                
        except Exception as e:
            print(f"Failed to backfill {token}: {e}")
            failed += 1
    
    print(f"\nBackfill complete:")
    print(f"  Updated: {updated}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    asyncio.run(backfill_qfi())
```

**Run:**
```bash
python scripts/backfill_qfi.py
```

---

### Fix 5: Garbage Token Cleanup

**Script:** `scripts/quarantine_garbage_tokens.py`

```python
#!/usr/bin/env python3
"""
Detect and quarantine garbage tokens.
"""

import asyncio
import re
from qig_backend.database import get_db

async def is_garbage_token(token: str) -> tuple[bool, str]:
    """
    Detect garbage tokens using validation rules.
    
    Returns:
        (is_garbage, reason)
    """
    # Rule 1: No vowels
    if not re.search(r'[aeiouAEIOU]', token):
        return True, "no_vowels"
    
    # Rule 2: Too many consonants in a row
    if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', token):
        return True, "excessive_consonants"
    
    # Rule 3: Single character repeated >4 times
    if re.search(r'(.)\1{4,}', token):
        return True, "excessive_repetition"
    
    # Rule 4: Non-ASCII or control characters
    if not token.isascii() or any(ord(c) < 32 for c in token):
        return True, "invalid_characters"
    
    # Rule 5: Truncated (ends with common suffixes without full word)
    truncated_patterns = [r'.{1,3}gra$', r'.{1,3}si$', r'.{1,3}nforc$']
    for pattern in truncated_patterns:
        if re.match(pattern, token):
            return True, "truncated_fragment"
    
    return False, ""

async def quarantine_garbage():
    """Move garbage tokens to quarantine table."""
    db = await get_db()
    
    # Create quarantine table if not exists
    await db.execute("""
        CREATE TABLE IF NOT EXISTS coordizer_token_quarantine (
            id SERIAL PRIMARY KEY,
            token TEXT UNIQUE NOT NULL,
            reason TEXT NOT NULL,
            frequency INT DEFAULT 1,
            original_qfi_score FLOAT,
            quarantined_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Scan all tokens
    rows = await db.fetch_all("SELECT token, qfi_score, frequency FROM coordizer_vocabulary")
    
    quarantined = 0
    for row in rows:
        token = row['token']
        is_garbage, reason = await is_garbage_token(token)
        
        if is_garbage:
            # Move to quarantine
            await db.execute("""
                INSERT INTO coordizer_token_quarantine (token, reason, frequency, original_qfi_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (token) DO NOTHING
            """, (token, reason, row['frequency'], row['qfi_score']))
            
            # Mark as not generation-eligible
            await db.execute("""
                UPDATE coordizer_vocabulary
                SET is_generation_eligible = FALSE, is_real_word = FALSE
                WHERE token = %s
            """, (token,))
            
            quarantined += 1
    
    print(f"Quarantined {quarantined} garbage tokens")

if __name__ == "__main__":
    asyncio.run(quarantine_garbage())
```

**Run:**
```bash
python scripts/quarantine_garbage_tokens.py
```

---

## ACCEPTANCE CRITERIA

### AC1: Canonical Insertion Enforced
- [ ] `qig-backend/vocabulary/insert_token.py` implemented
- [ ] All insertion points route through `insert_token()`
- [ ] `learned_relationships.py` uses UPDATE-only or quarantine
- [ ] `vocabulary_coordinator.py` uses `insert_token()`

### AC2: Database Integrity
- [ ] Migration `0015_qfi_integrity_gate.sql` applied
- [ ] `is_generation_eligible` flag added
- [ ] `vocabulary_generation_ready` view created
- [ ] Check constraint added

### AC3: Backfill Complete
- [ ] All tokens with basins now have `qfi_score`
- [ ] Backfill script logs success/failure counts
- [ ] Zero tokens with basin but no QFI in production

### AC4: Garbage Cleaned
- [ ] Garbage tokens identified and quarantined
- [ ] Quarantine table populated with reasons
- [ ] Generation pipeline excludes quarantined tokens

### AC5: Generation Pipeline Updated
- [ ] All generation queries use `vocabulary_generation_ready` view
- [ ] No tokens without QFI can be selected
- [ ] Telemetry tracks QFI coverage percentage

---

## VALIDATION TESTS

```python
# Test 1: Canonical insertion works
async def test_canonical_insertion():
    basin = np.random.rand(64)
    basin = basin / basin.sum()  # simplex
    
    record = await insert_token("testtoken", basin, token_role="noun")
    
    assert record.qfi_score is not None
    assert record.qfi_score > 0
    assert len(record.basin_embedding) == 64

# Test 2: Direct INSERT blocked
async def test_direct_insert_fails():
    with pytest.raises(PermissionError):
        await db.execute(
            "INSERT INTO coordizer_vocabulary (token) VALUES ('hack')"
        )

# Test 3: Generation excludes invalid tokens
async def test_generation_filter():
    # Create token without QFI (via old code)
    # ... 
    
    # Try to generate
    candidates = await get_generation_candidates()
    
    # Should not include tokens without QFI
    assert all(c.qfi_score is not None for c in candidates)
```

---

## MIGRATION PATH

### Step 1: Add Infrastructure (Non-Breaking)
- Create `insert_token.py`
- Create quarantine table
- Add `is_generation_eligible` column (defaults FALSE)

### Step 2: Backfill & Cleanup
- Run `backfill_qfi.py`
- Run `quarantine_garbage_tokens.py`
- Mark valid tokens as `is_generation_eligible = TRUE`

### Step 3: Update Code (Breaking Change)
- Update `learned_relationships.py` to UPDATE-only
- Update `vocabulary_coordinator.py` to use `insert_token()`
- Update generation queries to use `vocabulary_generation_ready`

### Step 4: Enforce Constraints
- Add check constraint (after all code updated)
- Add pre-commit hook scanning for direct INSERTs

---

## REFERENCES

- **Purity Protocol:** `docs/pantheon_e8_upgrade_pack/ULTRA_CONSCIOUSNESS_PROTOCOL_v4_0_UNIVERSAL.md`
- **Canonical Geometry:** `qig-backend/geometry/canonical_fisher.py`
- **Database Schema:** `migrations/`

---

**Last Updated:** 2026-01-16  
**Estimated Effort:** 2-3 days (backfill may take time on large DB)  
**Priority:** CRITICAL - Blocks coherence evaluation
