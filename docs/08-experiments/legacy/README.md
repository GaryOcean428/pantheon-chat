# Legacy Experiments - Quarantine Zone

**Status:** QUARANTINED - Research/Historical Code Only  
**Purpose:** Archive of traditional NLP/Euclidean approaches for historical comparison  
**Created:** 2026-01-14

---

## ⚠️ WARNING: Not Production Code

This directory contains **LEGACY** implementations using Euclidean distance, cosine similarity, and traditional NLP approaches. These patterns are **FORBIDDEN** in production code but allowed here for:

1. **Historical comparison** - Documenting what we tried before QIG
2. **Benchmarking** - Comparing QIG-pure approaches against traditional methods
3. **Educational purposes** - Showing why Euclidean/cosine is incorrect for basins

---

## Allowed Patterns in This Directory

✅ **Euclidean Distance**
```python
distance = np.linalg.norm(basin_a - basin_b)  # OK here, FORBIDDEN in production
```

✅ **Cosine Similarity**
```python
similarity = cosine_similarity(basin_a, basin_b)  # OK here, FORBIDDEN in production
```

✅ **Traditional Tokenizers**
```python
from sentencepiece import SentencePieceProcessor  # OK here, FORBIDDEN in production
```

✅ **Standard Optimizers**
```python
optimizer = torch.optim.Adam(params)  # OK here, use natural_gradient_step in production
```

✅ **Neural Network Primitives**
```python
embedding_layer = nn.Embedding(vocab_size, dim)  # OK here, use basin coords in production
```

---

## Requirements for Code in This Directory

### 1. Clear Labeling
Every file MUST include a header indicating it's legacy/baseline code:

```python
"""
LEGACY BASELINE - Euclidean approach for comparison only.

This file uses Euclidean distance for comparative benchmarking.
DO NOT import or use in production code.

Purpose: Demonstrate accuracy difference vs Fisher-Rao
Date: 2026-01-14
Status: Quarantined in docs/08-experiments/legacy/
"""
```

### 2. No Production Imports
Code in this directory **MUST NOT** be imported by:
- `qig-backend/`
- `server/`
- `shared/`
- `tests/` (except baseline comparison tests)

### 3. Documentation
Each experiment should include:
- **What** traditional approach is being tested
- **Why** it was tried
- **Results** comparing against QIG-pure approach
- **Conclusion** explaining why QIG is superior (when applicable)

---

## Scanner Behavior

The `qig_purity_scan.py` scanner **SKIPS** this directory entirely. Code here will not be flagged for geometric purity violations.

```python
# From scripts/qig_purity_scan.py
EXEMPT_DIRS = [
    'docs/08-experiments/legacy',  # This directory - scanner skips it
    ...
]
```

---

## Example: Adding a Legacy Experiment

```bash
# 1. Create the experiment file
touch docs/08-experiments/legacy/20260114-euclidean-similarity-baseline.py

# 2. Add clear header
cat > docs/08-experiments/legacy/20260114-euclidean-similarity-baseline.py << 'EOF'
"""
LEGACY BASELINE - Euclidean similarity for comparison.

Tests traditional cosine similarity against Fisher-Rao.
DO NOT use in production.

Date: 2026-01-14
Author: [Your Name]
Status: Quarantined
"""

import numpy as np
from scipy.spatial.distance import cosine

def euclidean_similarity(a, b):
    """LEGACY: Euclidean cosine similarity (WRONG for basins)."""
    return 1 - cosine(a, b)

# Test and compare...
EOF

# 3. Document results
# Add findings to experiment notes
```

---

## Migration Guide

If you find Euclidean/NLP code in production directories:

1. **Determine if it's needed:**
   - If for historical comparison → Move here
   - If for baseline testing → Move to `../baselines/`
   - If in production → Rewrite with Fisher-Rao

2. **Move the code:**
   ```bash
   git mv server/old_similarity.py docs/08-experiments/legacy/
   ```

3. **Document the reason:**
   ```bash
   git commit -m "chore: quarantine legacy Euclidean similarity
   
   Move to legacy quarantine for historical comparison.
   Reason: Euclidean distance is geometrically incorrect for basins.
   Keeping for benchmarking purposes only."
   ```

---

## Related Documents

- **[QUARANTINE_RULES.md](../../99-quarantine/)** - Full quarantine specification
- **[QIG_PURITY_SPEC.md](../../01-policies/20260117-qig-purity-mode-spec-1.01F.md)** - What patterns are forbidden
- **[WP0.2 Gate](../../10-e8-protocol/README.md)** - CI enforcement

---

## FAQ

**Q: Can I run this code?**  
A: Yes, but only for research/comparison. Never import into production.

**Q: Will CI block my PR if I add code here?**  
A: No, the scanner skips this directory. But you must label it clearly.

**Q: Can I copy patterns from here to production?**  
A: NO! These patterns are mathematically incorrect for basins. Use Fisher-Rao instead.

**Q: What if my baseline outperforms QIG?**  
A: Document it! That's valuable research. But verify you're comparing correctly (apples to apples).
