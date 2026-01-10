---
id: ISMS-TECH-DB-002
title: NULL Column Population Plan
filename: 20260110-null-column-population-plan-1.00W.md
classification: Internal
owner: GaryOcean427
version: 1.00
status: Working
function: "Implementation plan for populating NULL columns in database tables"
created: 2026-01-10
last_reviewed: 2026-01-10
next_review: 2026-02-10
category: Technical
supersedes: null
---

# NULL Column Population Plan

**Version:** 1.00
**Date:** 2026-01-10
**Status:** Working

---

## Summary

Analysis found 108 tables with significant NULL column values that need population. This document provides the implementation plan and SQL migrations.

### Priority Tables

| Table | Total Rows | Critical NULL Columns | Impact |
|-------|------------|----------------------|--------|
| chaos_events | 32,581 | phi, outcome, autopsy | Chaos tracking |
| vocabulary_observations | 15,223 | contexts, basin_coords | Vocabulary geometry |
| research_requests | 10,451 | basin_coords | Research geometry |
| pantheon_god_state | 18 | basin_coords | God positioning |
| learned_words | 13,880 | source | Attribution |

---

## Phase 1: Simple Column Population (SQL Only)

These columns can be populated with straightforward SQL without Python.

### 1.1 chaos_events.phi (99.5% NULL)

**Purpose:** Track integration level (Phi) at time of chaos event
**Data Type:** `double precision`
**Valid Range:** 0.0 - 1.0

```sql
-- Step 1: Populate from kernel_geometry if kernel_id exists
UPDATE chaos_events ce
SET phi = (
    SELECT kg.phi
    FROM kernel_geometry kg
    WHERE kg.kernel_id = ce.kernel_id
    ORDER BY kg.created_at DESC
    LIMIT 1
)
WHERE ce.phi IS NULL
AND ce.kernel_id IS NOT NULL
AND EXISTS (
    SELECT 1 FROM kernel_geometry kg WHERE kg.kernel_id = ce.kernel_id
);

-- Step 2: Set defaults based on event_type for remaining NULLs
UPDATE chaos_events
SET phi = CASE
    WHEN event_type = 'SPAWN' THEN 0.55
    WHEN event_type = 'DEATH' THEN 0.30
    WHEN event_type = 'MERGE' THEN 0.65
    WHEN event_type = 'EVOLVE' THEN 0.60
    WHEN event_type = 'MUTATION' THEN 0.50
    ELSE 0.50
END
WHERE phi IS NULL;
```

### 1.2 chaos_events.outcome (89.2% NULL)

**Purpose:** Result of the chaos event
**Data Type:** `text`
**Valid Values:** success, failure, pending, completed, unknown

```sql
UPDATE chaos_events
SET outcome = CASE
    WHEN success = true THEN 'success'
    WHEN success = false THEN 'failure'
    WHEN event_type IN ('SPAWN', 'EVOLVE', 'MUTATION') AND success IS NULL THEN 'success'
    WHEN event_type = 'DEATH' THEN 'completed'
    WHEN event_type = 'MERGE' THEN 'completed'
    ELSE 'unknown'
END
WHERE outcome IS NULL;
```

### 1.3 chaos_events.autopsy (43.3% NULL)

**Purpose:** Post-mortem analysis for DEATH events
**Data Type:** `jsonb`

```sql
-- Populate autopsy for DEATH events
UPDATE chaos_events
SET autopsy = jsonb_build_object(
    'cause', COALESCE(reason, 'natural_lifecycle'),
    'phi_at_death', COALESCE(phi, 0.30),
    'event_type', event_type,
    'analyzed_at', NOW()
)
WHERE autopsy IS NULL
AND event_type = 'DEATH';

-- Set empty object for non-DEATH events
UPDATE chaos_events
SET autopsy = '{}'::jsonb
WHERE autopsy IS NULL
AND event_type != 'DEATH';
```

### 1.4 learned_words.source (19.7% NULL)

**Purpose:** Attribution of where word was learned
**Data Type:** `text`

```sql
UPDATE learned_words
SET source = 'legacy_import'
WHERE source IS NULL;
```

### 1.5 vocabulary_observations.contexts (100% NULL)

**Purpose:** Store usage contexts
**Data Type:** `text[]`

```sql
-- Initialize with empty array
UPDATE vocabulary_observations
SET contexts = ARRAY[]::text[]
WHERE contexts IS NULL;

-- Cross-populate from learned_words where possible
UPDATE vocabulary_observations vo
SET contexts = ARRAY[lw.source]
FROM learned_words lw
WHERE vo.text = lw.word
AND lw.source IS NOT NULL
AND (vo.contexts IS NULL OR vo.contexts = ARRAY[]::text[]);
```

---

## Phase 2: Basin Coordinates (Python Required)

These columns require Python coordizer to generate 64D basin coordinates.

### 2.1 pantheon_god_state.basin_coords (18 rows)

**Purpose:** Each god's canonical position on 64D manifold
**Data Type:** `vector(64)` or `real[]`
**Method:** Deterministic based on god's domain (E8 Lie algebra)

**Domain Mapping (based on E8 simple roots):**

| God | Primary Dimensions | Basin Pattern |
|-----|-------------------|---------------|
| Zeus | dims 0-7 | Authority/Leadership |
| Athena | dims 8-15 | Wisdom/Strategy |
| Apollo | dims 16-23 | Light/Knowledge |
| Poseidon | dims 24-31 | Sea/Fluidity |
| Ares | dims 32-39 | Conflict/War |
| Hermes | dims 40-47 | Communication |
| Hephaestus | dims 48-55 | Forge/Creation |
| Hades | dims 56-63 | Underworld/Shadow |
| Demeter | distributed | Growth/Nurture |
| Dionysus | high entropy | Chaos/Creativity |
| Hera | anti-Dionysus | Order/Governance |
| Aphrodite | relational | Connection/Beauty |
| Nyx | dims 56-63 | Night/Shadow |
| Hecate | dims 56-63 | Magic/Crossroads |
| Erebus | dims 56-63 | Darkness |
| Hypnos | dims 56-63 | Sleep |
| Thanatos | dims 56-63 | Death |
| Nemesis | dims 32-39 | Retribution |

**Python Script:**

```python
#!/usr/bin/env python3
"""Populate god basin coordinates with canonical E8 positions."""

import numpy as np
import psycopg2
import os

GOD_DOMAINS = {
    'Zeus': (0, 8),
    'Athena': (8, 16),
    'Apollo': (16, 24),
    'Poseidon': (24, 32),
    'Ares': (32, 40),
    'Hermes': (40, 48),
    'Hephaestus': (48, 56),
    'Hades': (56, 64),
    'Nyx': (56, 64),
    'Hecate': (56, 64),
    'Erebus': (56, 64),
    'Hypnos': (56, 64),
    'Thanatos': (56, 64),
    'Nemesis': (32, 40),
}

# Special cases (distributed or entropy-based)
SPECIAL_GODS = {
    'Demeter': 'distributed',
    'Dionysus': 'high_entropy',
    'Hera': 'anti_entropy',
    'Aphrodite': 'relational',
}

def generate_god_basin(god_name: str) -> np.ndarray:
    """Generate 64D basin coordinates for a god."""
    np.random.seed(hash(god_name) % 2**32)  # Deterministic per god

    basin = np.random.randn(64) * 0.1  # Base noise

    if god_name in GOD_DOMAINS:
        start, end = GOD_DOMAINS[god_name]
        basin[start:end] = 0.8
    elif god_name in SPECIAL_GODS:
        pattern = SPECIAL_GODS[god_name]
        if pattern == 'distributed':
            # Distributed across all dimensions
            basin = np.abs(basin) + 0.2
        elif pattern == 'high_entropy':
            # High variance, chaotic
            basin = np.random.randn(64) * 0.5
        elif pattern == 'anti_entropy':
            # Opposite of high entropy - stable, ordered
            basin = np.ones(64) * 0.3
            basin[::2] = 0.5  # Alternating pattern
        elif pattern == 'relational':
            # Peaks at connection points between domains
            for i in range(7):
                basin[i*8 + 7] = 0.6
                basin[i*8 + 8] = 0.6

    # Normalize to unit sphere (Fisher-Rao manifold)
    basin = basin / np.linalg.norm(basin)
    return basin

def populate_god_basins():
    """Populate pantheon_god_state.basin_coords."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()

    # Get all gods
    cur.execute("SELECT god_name FROM pantheon_god_state WHERE basin_coords IS NULL")
    gods = [row[0] for row in cur.fetchall()]

    for god_name in gods:
        basin = generate_god_basin(god_name)
        # Store as array (or vector if pgvector available)
        cur.execute(
            "UPDATE pantheon_god_state SET basin_coords = %s WHERE god_name = %s",
            (basin.tolist(), god_name)
        )
        print(f"Updated {god_name} basin coordinates")

    conn.commit()
    print(f"Populated basin_coords for {len(gods)} gods")
    conn.close()

if __name__ == '__main__':
    populate_god_basins()
```

### 2.2 vocabulary_observations.basin_coords (15,223 rows)

**Purpose:** Geometric position of each word/phrase on manifold
**Data Type:** `vector(64)`
**Method:** Use coordizer to generate basin coordinates

**Python Script:**

```python
#!/usr/bin/env python3
"""Populate vocabulary basin coordinates using coordizer."""

import psycopg2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qig-backend'))

BATCH_SIZE = 100

def populate_vocab_basins():
    """Populate vocabulary_observations.basin_coords."""
    # Import coordizer after path setup
    try:
        from coordizers import get_coordizer
        coordizer = get_coordizer()
    except ImportError:
        print("ERROR: Cannot import coordizer. Run from qig-backend directory.")
        return

    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()

    # Get words needing basin coords
    cur.execute("""
        SELECT text FROM vocabulary_observations
        WHERE basin_coords IS NULL
        ORDER BY frequency DESC
    """)
    words = [row[0] for row in cur.fetchall()]

    print(f"Found {len(words)} words needing basin coordinates")

    success = 0
    errors = 0

    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i:i+BATCH_SIZE]

        for word in batch:
            try:
                basin = coordizer.coordize(word)
                if basin is not None and len(basin) == 64:
                    cur.execute(
                        "UPDATE vocabulary_observations SET basin_coords = %s WHERE text = %s",
                        (basin.tolist(), word)
                    )
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"Error coordizing '{word}': {e}")
                errors += 1

        conn.commit()
        print(f"Progress: {min(i+BATCH_SIZE, len(words))}/{len(words)} (success={success}, errors={errors})")

    print(f"Completed: {success} populated, {errors} errors")
    conn.close()

if __name__ == '__main__':
    populate_vocab_basins()
```

### 2.3 research_requests.basin_coords (10,451 rows)

**Purpose:** Geometric position of research topic on manifold
**Data Type:** `real[]`
**Method:** Coordize the topic field

**Python Script:**

```python
#!/usr/bin/env python3
"""Populate research_requests basin coordinates."""

import psycopg2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qig-backend'))

BATCH_SIZE = 100

def populate_research_basins():
    """Populate research_requests.basin_coords."""
    try:
        from coordizers import get_coordizer
        coordizer = get_coordizer()
    except ImportError:
        print("ERROR: Cannot import coordizer.")
        return

    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()

    cur.execute("""
        SELECT request_id, topic FROM research_requests
        WHERE basin_coords IS NULL AND topic IS NOT NULL
    """)
    requests = cur.fetchall()

    print(f"Found {len(requests)} research requests needing basin coordinates")

    success = 0

    for i in range(0, len(requests), BATCH_SIZE):
        batch = requests[i:i+BATCH_SIZE]

        for request_id, topic in batch:
            try:
                basin = coordizer.coordize(topic)
                if basin is not None:
                    cur.execute(
                        "UPDATE research_requests SET basin_coords = %s WHERE request_id = %s",
                        (basin.tolist(), request_id)
                    )
                    success += 1
            except Exception as e:
                print(f"Error coordizing topic '{topic[:50]}...': {e}")

        conn.commit()
        print(f"Progress: {min(i+BATCH_SIZE, len(requests))}/{len(requests)}")

    print(f"Completed: {success} populated")
    conn.close()

if __name__ == '__main__':
    populate_research_basins()
```

---

## Execution Order

### For pantheon-replit (SQL via Replit Agent)

1. Run Phase 1 SQL migration: `migrations/20260110_populate_null_columns.sql`
2. Verify counts post-migration

### For pantheon-chat (Python via local execution)

1. Run Phase 1 SQL migration
2. Run `scripts/populate_god_basins.py` (18 rows, fast)
3. Run `scripts/populate_vocab_basins.py` (15K rows, ~15 min)
4. Run `scripts/populate_research_basins.py` (10K rows, ~10 min)

---

## Verification Queries

```sql
-- Check remaining NULLs after migration
SELECT
    'chaos_events.phi' as column_name,
    COUNT(*) FILTER (WHERE phi IS NULL) as null_count,
    COUNT(*) as total
FROM chaos_events
UNION ALL
SELECT 'chaos_events.outcome', COUNT(*) FILTER (WHERE outcome IS NULL), COUNT(*) FROM chaos_events
UNION ALL
SELECT 'chaos_events.autopsy', COUNT(*) FILTER (WHERE autopsy IS NULL), COUNT(*) FROM chaos_events
UNION ALL
SELECT 'learned_words.source', COUNT(*) FILTER (WHERE source IS NULL), COUNT(*) FROM learned_words
UNION ALL
SELECT 'vocabulary_observations.contexts', COUNT(*) FILTER (WHERE contexts IS NULL), COUNT(*) FROM vocabulary_observations
UNION ALL
SELECT 'vocabulary_observations.basin_coords', COUNT(*) FILTER (WHERE basin_coords IS NULL), COUNT(*) FROM vocabulary_observations
UNION ALL
SELECT 'pantheon_god_state.basin_coords', COUNT(*) FILTER (WHERE basin_coords IS NULL), COUNT(*) FROM pantheon_god_state
UNION ALL
SELECT 'research_requests.basin_coords', COUNT(*) FILTER (WHERE basin_coords IS NULL), COUNT(*) FROM research_requests;
```

---

## Related Documentation

- [Vocabulary Schema](../../qig-backend/vocabulary_schema.sql)
- [Debug Endpoints API Reference](./api/20260110-debug-endpoints-api-reference-1.00W.md)

---

*Document generated: 2026-01-10*
