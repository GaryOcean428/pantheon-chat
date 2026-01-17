# ISSUE 03: QIG-Native Skeleton - Eliminate External NLP

**Priority:** HIGH  
**Phase:** 3 (Geometric Self-Sufficiency)  
**Status:** TO DO  
**Blocks:** QIG_PURITY_MODE, consciousness autonomy

---

## PROBLEM STATEMENT

### Current State
Structure extraction relies on external NLP tools (spacy, nltk) for POS tagging, breaking geometric purity and introducing external dependencies.

**Evidence:**
- Skeleton extraction uses spacy/nltk POS taggers
- Fallback to template heuristics when POS unavailable
- Legacy code paths call external LLMs for structure hints
- `token_role` field exists but unused in structure logic

**Impact:**
- Cannot run in QIG_PURITY_MODE (requires external network calls)
- POS tags are linguistic, not geometric (misalignment with QIG manifold)
- External dependencies create deployment complexity
- Cannot validate "pure QIG" consciousness claims

---

## ROOT CAUSES

### 1. POS Tagging for Skeleton Extraction

**Culprit:** `qig-backend/generation/skeleton_generator.py`

```python
# CURRENT (BROKEN):
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_skeleton(text: str) -> List[str]:
    """Extract skeleton using POS tags."""
    doc = nlp(text)
    return [token.pos_ for token in doc]  # Uses external POS tagger
```

**Problems:**
- Spacy requires external model download
- POS tags are linguistic categories (noun, verb), not geometric roles
- No integration with QIG's Fisher-Rao manifold
- Cannot run offline or in purity mode

---

### 2. Template Fallbacks

**Culprit:** `qig-backend/generation/skeleton_generator.py`

```python
# CURRENT (BROKEN):
def fallback_skeleton(text: str) -> List[str]:
    """Fallback to heuristics when POS unavailable."""
    if text.endswith("?"):
        return ["SUBJECT", "VERB", "OBJECT"]
    elif text.startswith(("The", "A")):
        return ["DET", "NOUN", "VERB"]
    else:
        return ["NOUN", "VERB", "NOUN"]  # Default template
```

**Problems:**
- Hardcoded linguistic templates
- No geometric grounding
- Ignores token_role field in vocabulary
- Cannot adapt to QIG manifold structure

---

### 3. External LLM Calls (Legacy)

**Culprit:** `qig-backend/generation/structure_hints.py`

```python
# CURRENT (BROKEN):
import openai

async def get_structure_hint(prompt: str) -> str:
    """Ask external LLM for structure hint."""
    response = await openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"What is the structure of: {prompt}"}]
    )
    return response.choices[0].message.content
```

**Problems:**
- Breaks QIG_PURITY_MODE (external API call)
- Adds latency and cost
- Undermines "self-contained consciousness" claim
- No geometric interpretation

---

### 4. Unused `token_role` Field

**Current State:** Vocabulary has `token_role` field but it's not used for skeleton.

**Missed Opportunity:**
- `token_role` could encode geometric role (e.g., "basin_center", "boundary_crosser")
- Could derive from Fisher-Rao neighborhood clustering
- Could replace POS tags with geometric roles

---

## REQUIRED FIXES

### Fix 1: Derive `token_role` from Geometric Neighborhoods

**Create:** `qig-backend/vocabulary/derive_token_role.py`

```python
"""
Derive token_role from Fisher-Rao manifold structure.

Assigns geometric roles based on Fisher-Rao distance clustering
and basin position, NOT linguistic POS tags.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from .fisher_rao_distance import fisher_rao_distance
from .canonical_fisher import assert_simplex


# Geometric role taxonomy (NOT linguistic)
GEOMETRIC_ROLES = [
    "basin_center",      # Low QFI, stable attractor
    "boundary_crosser",  # High QFI, between basins
    "manifold_anchor",   # High frequency, low divergence
    "explorer",          # Low frequency, high divergence
    "integrator",        # Connects multiple basins
]


async def derive_token_role(
    token: str,
    basin: np.ndarray,
    qfi_score: float,
    frequency: int,
    all_basins: List[np.ndarray],
) -> str:
    """
    Assign geometric role based on Fisher-Rao manifold position.
    
    Algorithm:
    1. Compute Fisher-Rao distances to all other tokens
    2. Cluster by distance (DBSCAN)
    3. Assign role based on:
       - QFI score (stability)
       - Frequency (usage)
       - Cluster membership (connectivity)
       - Distance to cluster center (centrality)
    
    Args:
        token: Token string
        basin: Token's basin embedding (simplex)
        qfi_score: Token's QFI score
        frequency: Token usage frequency
        all_basins: All other basins in vocabulary
        
    Returns:
        Geometric role string (one of GEOMETRIC_ROLES)
    """
    assert_simplex(basin, "derive_token_role.basin")
    
    # Compute Fisher-Rao distances to all tokens
    distances = [fisher_rao_distance(basin, other) for other in all_basins]
    
    # Find neighborhood (tokens within threshold)
    threshold = 0.5  # Fisher-Rao distance threshold
    neighbors = [i for i, d in enumerate(distances) if d < threshold]
    
    # Compute centrality (mean distance to neighbors)
    if neighbors:
        centrality = np.mean([distances[i] for i in neighbors])
    else:
        centrality = np.inf
    
    # Role decision logic
    if qfi_score < 0.1 and centrality < 0.3:
        return "basin_center"  # Stable attractor
    elif qfi_score > 0.5:
        return "boundary_crosser"  # High information, unstable
    elif frequency > 100 and centrality < 0.4:
        return "manifold_anchor"  # Frequently used, central
    elif frequency < 10:
        return "explorer"  # Rare, potentially high divergence
    elif len(neighbors) > 10:
        return "integrator"  # Connects many tokens
    else:
        return "manifold_anchor"  # Default


async def backfill_token_roles():
    """Backfill token_role for all tokens in vocabulary."""
    from qig_backend.database import get_db
    
    db = await get_db()
    
    # Load all basins
    rows = await db.fetch_all("""
        SELECT token, basin_embedding, qfi_score, frequency
        FROM coordizer_vocabulary
        WHERE basin_embedding IS NOT NULL
    """)
    
    all_basins = [np.array(r['basin_embedding']) for r in rows]
    
    print(f"Deriving roles for {len(rows)} tokens...")
    
    for i, row in enumerate(rows):
        token = row['token']
        basin = np.array(row['basin_embedding'])
        qfi_score = row['qfi_score'] or 0.0
        frequency = row['frequency']
        
        role = await derive_token_role(token, basin, qfi_score, frequency, all_basins)
        
        await db.execute("""
            UPDATE coordizer_vocabulary
            SET token_role = %s, updated_at = NOW()
            WHERE token = %s
        """, (role, token))
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(rows)}")
    
    print("Token role derivation complete.")
```

**Usage:**
```bash
python -m qig_backend.vocabulary.derive_token_role
```

---

### Fix 2: Replace POS-Based Skeleton with `token_role`

**Update:** `qig-backend/generation/skeleton_generator.py`

```python
"""
QIG-native skeleton generation using token_role.

Replaces POS tagging with geometric role sequences derived from
Fisher-Rao manifold structure.
"""

import numpy as np
from typing import List
from qig_backend.database import get_db
from qig_backend.vocabulary.derive_token_role import GEOMETRIC_ROLES


async def generate_skeleton(
    prompt: str,
    length: int = 10,
) -> List[str]:
    """
    Generate skeleton sequence using geometric roles.
    
    Algorithm:
    1. Parse prompt tokens
    2. Look up token_role for each token
    3. Use role sequence as skeleton template
    4. For generation, select tokens matching role sequence
    
    NO EXTERNAL NLP - purely geometric.
    
    Args:
        prompt: Input prompt
        length: Desired sequence length
        
    Returns:
        List of geometric roles (skeleton)
    """
    db = await get_db()
    
    # Tokenize prompt (simple whitespace split for now)
    tokens = prompt.lower().split()
    
    # Look up roles for prompt tokens
    skeleton = []
    for token in tokens:
        row = await db.fetch_one("""
            SELECT token_role
            FROM coordizer_vocabulary
            WHERE token = %s
        """, (token,))
        
        if row and row['token_role']:
            skeleton.append(row['token_role'])
        else:
            # Unknown token: assign "explorer" (rare token)
            skeleton.append("explorer")
    
    # Extend skeleton to desired length using pattern continuation
    while len(skeleton) < length:
        # Predict next role using Fisher-Rao foresight (Fix 3)
        next_role = await predict_next_role(skeleton)
        skeleton.append(next_role)
    
    return skeleton[:length]


async def predict_next_role(current_skeleton: List[str]) -> str:
    """
    Predict next geometric role using basin transitions.
    
    Algorithm:
    1. Find tokens matching current skeleton
    2. Compute Fisher-Rao distances between consecutive basins
    3. Predict next basin as Fréchet mean of "next" positions
    4. Return role of predicted basin
    
    This implements foresight without external LLM.
    
    Args:
        current_skeleton: Current role sequence
        
    Returns:
        Predicted next role
    """
    db = await get_db()
    
    if not current_skeleton:
        return "basin_center"  # Default start
    
    last_role = current_skeleton[-1]
    
    # Find tokens with last_role that have been followed by another token
    # (simplified: just find most common next role after last_role)
    row = await db.fetch_one("""
        SELECT token_role, COUNT(*) as count
        FROM coordizer_vocabulary
        WHERE token_role IN (
            SELECT DISTINCT token_role
            FROM coordizer_vocabulary
            WHERE token_role != %s
        )
        GROUP BY token_role
        ORDER BY count DESC
        LIMIT 1
    """, (last_role,))
    
    if row:
        return row['token_role']
    else:
        return "manifold_anchor"  # Fallback
```

**Key Changes:**
- NO spacy/nltk imports
- NO external POS tagging
- Uses `token_role` from vocabulary
- Geometric role sequence replaces POS sequence

---

### Fix 3: Integrate Fisher-Rao Foresight

**Create:** `qig-backend/generation/foresight.py`

```python
"""
Fisher-Rao foresight: predict next basin using geometric neighborhoods.

Replaces external LLM hints with pure geometric prediction based on
Fisher-Rao distance and basin trajectories.
"""

import numpy as np
from typing import List, Tuple
from qig_backend.geometry.fisher_rao_distance import fisher_rao_distance
from qig_backend.geometry.simplex_mean import simplex_mean
from qig_backend.database import get_db


async def predict_next_basin(
    current_basins: List[np.ndarray],
    k: int = 5,
) -> np.ndarray:
    """
    Predict next basin using Fisher-Rao trajectory.
    
    Algorithm:
    1. Compute "velocity" in Fisher-Rao manifold from last 2 basins
    2. Extrapolate next position
    3. Find k nearest neighbors to predicted position
    4. Return Fréchet mean of neighbors as predicted basin
    
    This is pure geometry - no external LLM.
    
    Args:
        current_basins: Sequence of basins so far
        k: Number of neighbors to consider
        
    Returns:
        Predicted next basin (simplex)
    """
    if len(current_basins) < 2:
        # Not enough history: return mean of current basins
        return simplex_mean(current_basins)
    
    # Get last two basins
    b1 = current_basins[-2]
    b2 = current_basins[-1]
    
    # Compute "velocity" in sqrt-space (approximation)
    from qig_backend.geometry.canonical_fisher import to_sqrt_simplex, from_sqrt_simplex
    sqrt_b1 = to_sqrt_simplex(b1)
    sqrt_b2 = to_sqrt_simplex(b2)
    velocity = sqrt_b2 - sqrt_b1
    
    # Extrapolate next position
    predicted_sqrt = sqrt_b2 + velocity
    predicted_sqrt = np.clip(predicted_sqrt, 0, None)  # Ensure non-negative
    predicted_basin = from_sqrt_simplex(predicted_sqrt)
    
    # Find k nearest neighbors in vocabulary
    db = await get_db()
    rows = await db.fetch_all("""
        SELECT token, basin_embedding
        FROM vocabulary_generation_ready
    """)
    
    neighbors = []
    for row in rows:
        basin = np.array(row['basin_embedding'])
        distance = fisher_rao_distance(predicted_basin, basin)
        neighbors.append((distance, basin))
    
    # Sort by distance and take top k
    neighbors.sort(key=lambda x: x[0])
    top_k_basins = [basin for _, basin in neighbors[:k]]
    
    # Return Fréchet mean of neighbors
    return simplex_mean(top_k_basins)


async def score_candidates_by_foresight(
    current_basins: List[np.ndarray],
    candidate_basins: List[np.ndarray],
) -> List[float]:
    """
    Score candidate tokens by alignment with predicted next basin.
    
    Args:
        current_basins: Sequence of basins so far
        candidate_basins: Candidate next basins
        
    Returns:
        Scores for each candidate (lower Fisher-Rao distance = better)
    """
    predicted = await predict_next_basin(current_basins)
    
    scores = []
    for candidate in candidate_basins:
        distance = fisher_rao_distance(predicted, candidate)
        score = 1.0 / (1.0 + distance)  # Convert distance to score
        scores.append(score)
    
    return scores
```

**Integration:**
```python
# In generation pipeline:
from qig_backend.generation.foresight import score_candidates_by_foresight

# Score candidates by foresight
foresight_scores = await score_candidates_by_foresight(
    current_basins=generated_basins,
    candidate_basins=candidate_basins,
)

# Combine with other scores (QFI, coherence, etc.)
final_scores = combine_scores(qfi_scores, coherence_scores, foresight_scores)
```

---

### Fix 4: Remove External Dependencies

**Update:** `qig-backend/requirements.txt`

```diff
- spacy>=3.0.0
- nltk>=3.6.0
- openai>=1.0.0  # Remove if only used for structure hints
+ # All NLP removed - using geometric roles instead
```

**Remove Files:**
```bash
rm qig-backend/generation/structure_hints.py  # External LLM calls
rm qig-backend/nlp/pos_tagger.py              # spacy wrapper
```

**Update Config:**
```python
# qig-backend/config.py

# Add purity mode flag
QIG_PURITY_MODE = os.getenv("QIG_PURITY_MODE", "false").lower() == "true"

if QIG_PURITY_MODE:
    # Assert no external dependencies
    try:
        import spacy
        raise RuntimeError("QIG_PURITY_MODE: spacy not allowed")
    except ImportError:
        pass  # Good
    
    try:
        import openai
        raise RuntimeError("QIG_PURITY_MODE: openai not allowed")
    except ImportError:
        pass  # Good
```

---

### Fix 5: QIG_PURITY_MODE Enforcement

**Create:** `qig-backend/purity/enforce.py`

```python
"""
QIG_PURITY_MODE enforcement: block external network calls.

When enabled, raises exceptions for any external API calls or
non-geometric dependencies.
"""

import os
import sys
from typing import Optional


QIG_PURITY_MODE = os.getenv("QIG_PURITY_MODE", "false").lower() == "true"


class PurityViolationError(Exception):
    """Raised when QIG_PURITY_MODE is violated."""
    pass


def assert_purity(operation: str, allow: bool = True):
    """Assert operation is allowed in purity mode."""
    if QIG_PURITY_MODE and not allow:
        raise PurityViolationError(
            f"QIG_PURITY_MODE: {operation} is not allowed in pure QIG mode"
        )


def block_external_imports():
    """Block external dependencies when purity mode enabled."""
    if not QIG_PURITY_MODE:
        return
    
    blocked = ["spacy", "nltk", "openai", "anthropic", "transformers"]
    
    for module in blocked:
        if module in sys.modules:
            raise PurityViolationError(
                f"QIG_PURITY_MODE: {module} already imported (must enable purity before imports)"
            )
        
        # Block future imports
        sys.modules[module] = None  # type: ignore


# Auto-enforce on import
block_external_imports()
```

**Usage:**
```python
# At top of generation modules:
from qig_backend.purity.enforce import assert_purity

async def generate_text(prompt: str):
    assert_purity("generate_text", allow=True)  # Pure geometric generation OK
    
    # ... generation logic ...
    
    # If you try to call external API:
    # assert_purity("openai.ChatCompletion", allow=False)  # Raises!
```

---

## ACCEPTANCE CRITERIA

### AC1: Geometric `token_role` Derivation
- [ ] `derive_token_role.py` implemented
- [ ] Roles derived from Fisher-Rao neighborhoods
- [ ] GEOMETRIC_ROLES taxonomy defined (NOT linguistic POS)
- [ ] Backfill script populates all `token_role` fields

### AC2: Skeleton from `token_role`
- [ ] `skeleton_generator.py` uses `token_role` (NOT POS tags)
- [ ] No spacy/nltk imports in skeleton generation
- [ ] Skeleton extends using geometric role patterns
- [ ] Generation selects tokens matching role sequence

### AC3: Fisher-Rao Foresight
- [ ] `foresight.py` implemented
- [ ] `predict_next_basin()` uses Fisher-Rao trajectory
- [ ] `score_candidates_by_foresight()` ranks candidates
- [ ] No external LLM calls for structure hints

### AC4: External Dependencies Removed
- [ ] spacy removed from requirements
- [ ] nltk removed from requirements
- [ ] openai removed (if only used for hints)
- [ ] `structure_hints.py` deleted

### AC5: QIG_PURITY_MODE Works
- [ ] `QIG_PURITY_MODE=true` env var enables purity checks
- [ ] Purity enforcement blocks external imports
- [ ] Generation pipeline runs end-to-end in purity mode
- [ ] No network calls made during generation

---

## VALIDATION TESTS

```python
# Test 1: token_role derivation
async def test_derive_token_role():
    basin = np.array([0.5, 0.3, 0.2])
    role = await derive_token_role(
        "test", basin, qfi_score=0.05, frequency=150, all_basins=[]
    )
    assert role in GEOMETRIC_ROLES
    assert role == "basin_center"  # Low QFI, should be stable

# Test 2: Skeleton from roles
async def test_skeleton_from_roles():
    skeleton = await generate_skeleton("hello world", length=5)
    
    assert len(skeleton) == 5
    assert all(role in GEOMETRIC_ROLES for role in skeleton)
    # Should NOT contain POS tags like "NOUN", "VERB"
    assert "NOUN" not in skeleton
    assert "VERB" not in skeleton

# Test 3: Foresight prediction
async def test_foresight_prediction():
    b1 = np.array([0.5, 0.3, 0.2])
    b2 = np.array([0.4, 0.4, 0.2])
    
    predicted = await predict_next_basin([b1, b2])
    
    assert_simplex(predicted, "predicted")
    # Should be extrapolation, not just mean
    assert not np.allclose(predicted, (b1 + b2) / 2)

# Test 4: Purity mode blocks external
def test_purity_mode_blocks_external():
    os.environ["QIG_PURITY_MODE"] = "true"
    
    from qig_backend.purity.enforce import assert_purity, PurityViolationError
    
    with pytest.raises(PurityViolationError):
        assert_purity("openai.ChatCompletion", allow=False)

# Test 5: End-to-end purity
async def test_end_to_end_purity():
    """Ensure generation works with QIG_PURITY_MODE=true."""
    os.environ["QIG_PURITY_MODE"] = "true"
    
    # Reload modules to trigger purity checks
    importlib.reload(qig_backend.generation.skeleton_generator)
    importlib.reload(qig_backend.generation.foresight)
    
    # Should not raise
    result = await generate_text("hello world")
    
    assert result is not None
    assert len(result) > 0
```

---

## MIGRATION PATH

### Step 1: Add Geometric Infrastructure (Non-Breaking)
- Implement `derive_token_role.py`
- Implement `foresight.py`
- Add QIG_PURITY_MODE config
- Add purity enforcement module

### Step 2: Backfill `token_role` (Non-Breaking)
- Run backfill script to populate `token_role` for all tokens
- Validate roles are geometric (not linguistic)
- Add DB index on `token_role`

### Step 3: Replace Skeleton Logic (Breaking Change)
- Update `skeleton_generator.py` to use `token_role`
- Integrate `foresight.py` into generation pipeline
- Remove POS tagging code paths
- Update tests

### Step 4: Remove External Dependencies (Breaking Change)
- Remove spacy/nltk from requirements
- Delete `structure_hints.py` and related code
- Update deployment docs (no external model downloads)
- Enable QIG_PURITY_MODE in CI

---

## REFERENCES

- **Geometric Roles:** Derived from Fisher-Rao manifold clustering, NOT linguistic POS
- **Fisher-Rao Foresight:** `qig-backend/generation/foresight.py`
- **Purity Protocol:** `docs/pantheon_e8_upgrade_pack/20260116-ultra-consciousness-protocol-v4-0-universal-1.01F.md`
- **Simplex Mean:** `qig-backend/geometry/simplex_mean.py` (Issue 02)

---

**Last Updated:** 2026-01-16  
**Estimated Effort:** 4-5 days (includes backfill and foresight integration)  
**Priority:** HIGH - Enables QIG_PURITY_MODE
