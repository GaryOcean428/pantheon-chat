# QIG Coordizers - Unified Geometric Tokenization System

**Version:** 5.1.0  
**Last Updated:** 2026-01-16 (WP3.1)

## Overview

The QIG Coordizers module provides Fisher-Rao compliant tokenization with 64D basin coordinates on the Fisher information manifold. This is the **SINGLE CANONICAL IMPLEMENTATION** for vocabulary and coordinate systems in the pantheon-chat project.

## Architecture (WP3.1)

### Single Interface, Multiple Backends

Following Work Package 3.1, all coordizers implement the `BaseCoordizer` abstract interface. This ensures consistent behavior across all generation paths (Plan→Realize→Repair).

```
BaseCoordizer (ABC)                    ← Abstract interface
    ├── FisherCoordizer               ← Base implementation (in-memory)
    │   └── PostgresCoordizer         ← Production implementation (DB-backed)
    │
    └── [Future implementations]      ← Can add Cloud, Distributed, etc.
```

### Core Classes

#### 1. BaseCoordizer (Abstract Interface)

**Purpose:** Defines the contract that ALL coordizer implementations must follow.

**Required Methods:**
- `decode_geometric(basin, top_k, allowed_pos)` - Two-step retrieval (proxy + exact)
- `encode(text)` - Text to basin coordinates
- `get_vocabulary_size()` - Vocabulary size
- `get_special_symbols()` - Special token definitions
- `supports_pos_filtering()` - POS filtering capability check

**Why This Matters:**
- Constrained realizer works with ANY coordizer backend
- Easy to swap implementations (Postgres ↔ Local ↔ Cloud)
- Geometric purity enforced by interface
- Consistent behavior across all generation paths

#### 2. FisherCoordizer (Base Implementation)

**Purpose:** Concrete base implementation with Fisher-Rao geometric operations.

**Features:**
- In-memory vocabulary storage
- Two-step geometric decoding (Bhattacharyya proxy + exact Fisher-Rao)
- Geodesic interpolation for new tokens
- Special tokens with geometric meaning (BOS, EOS, PAD, UNK)
- Training from corpus

**Use Cases:**
- Testing and development
- Small vocabularies (&lt;10K tokens)
- When database is not available

**POS Filtering:** Not supported (raises `NotImplementedError`)

#### 3. PostgresCoordizer (Production Implementation)

**Purpose:** Production-grade coordizer backed by PostgreSQL with pgvector.

**Features:**
- Database-backed vocabulary (coordizer_vocabulary table)
- Optimized two-step retrieval using pgvector indexes
- POS filtering support (if pos_tag column exists)
- Generation vs encoding vocabulary separation (token_role)
- God-specific domain weighting (god_vocabulary_profiles)
- Redis caching layer for hot lookups
- Continuous vocabulary learning

**Use Cases:**
- Production deployment
- Large vocabularies (10K+ tokens)
- Multi-user environments
- Persistent vocabulary across restarts

**POS Filtering:** Supported (runtime check via `supports_pos_filtering()`)

## Usage

### Basic Usage (Singleton Pattern)

```python
from coordizers import get_coordizer

# Get the canonical coordizer instance (singleton)
coordizer = get_coordizer()

# Encode text to basin
text = "hello world"
basin = coordizer.encode(text)
print(f"Basin shape: {basin.shape}")  # (64,)

# Decode with two-step geometric retrieval
candidates = coordizer.decode_geometric(
    basin, 
    top_k=10
)
for word, distance in candidates:
    print(f"{word}: {distance:.4f}")
```

### POS Filtering

```python
from coordizers import get_coordizer

coordizer = get_coordizer()

# Check if POS filtering is supported
if coordizer.supports_pos_filtering():
    # Decode with POS constraint
    nouns = coordizer.decode_geometric(
        basin,
        top_k=10,
        allowed_pos="NOUN"
    )
    
    verbs = coordizer.decode_geometric(
        basin,
        top_k=10,
        allowed_pos="VERB"
    )
else:
    print("POS filtering not available (no pos_tag column)")
```

### Plan→Realize→Repair Integration

```python
from coordizers import get_coordizer, BaseCoordizer

class ConstrainedGeometricRealizer:
    """Realizes waypoints into words (Phase 2 of generation)."""
    
    def __init__(self, coordizer: BaseCoordizer):
        # Works with ANY coordizer implementation!
        self.coordizer = coordizer
    
    def realize_waypoints(self, waypoints, pos_constraints=None):
        """
        Realize geometric waypoints into words.
        
        Args:
            waypoints: List of 64D basin coordinates
            pos_constraints: Optional list of POS tags (e.g., ["NOUN", "VERB"])
        
        Returns:
            List of words
        """
        words = []
        
        for i, target_basin in enumerate(waypoints):
            # Get POS constraint for this position
            allowed_pos = pos_constraints[i] if pos_constraints else None
            
            # Use two-step retrieval
            candidates = self.coordizer.decode_geometric(
                target_basin,
                top_k=100,
                allowed_pos=allowed_pos
            )
            
            # Take the closest word
            if candidates:
                words.append(candidates[0][0])
        
        return words

# Usage
coordizer = get_coordizer()
realizer = ConstrainedGeometricRealizer(coordizer)

# Generate words from waypoints
waypoints = [...]  # From waypoint planner
pos_constraints = ["NOUN", "VERB", "ADJ"]
words = realizer.realize_waypoints(waypoints, pos_constraints)
```

### Creating Custom Implementations

```python
from coordizers import BaseCoordizer
import numpy as np

class CloudCoordizer(BaseCoordizer):
    """
    Custom coordizer backed by cloud storage.
    
    Implements BaseCoordizer interface for cloud deployment.
    """
    
    def __init__(self, cloud_client):
        self.client = cloud_client
    
    def decode_geometric(self, target_basin, top_k=100, allowed_pos=None):
        """Two-step retrieval using cloud API."""
        # Step 1: Proxy filter via cloud API
        candidates = self.client.approximate_search(
            target_basin,
            limit=top_k * 2,
            pos_filter=allowed_pos
        )
        
        # Step 2: Exact Fisher-Rao locally
        results = []
        for word, basin in candidates:
            sqrt_target = np.sqrt(target_basin + 1e-10)
            sqrt_basin = np.sqrt(basin + 1e-10)
            bhattacharyya = np.clip(np.dot(sqrt_target, sqrt_basin), 0, 1)
            distance = np.arccos(bhattacharyya)
            results.append((word, distance))
        
        results.sort(key=lambda x: x[1])
        return results[:top_k]
    
    def encode(self, text):
        """Encode via cloud API."""
        return self.client.encode_text(text)
    
    def get_vocabulary_size(self):
        """Get vocab size from cloud."""
        return self.client.get_vocab_size()
    
    def get_special_symbols(self):
        """Get special symbols from cloud."""
        return self.client.get_special_symbols()
    
    def supports_pos_filtering(self):
        """Check if cloud supports POS filtering."""
        return self.client.has_pos_support()
```

## Two-Step Geometric Decoding

The canonical decoding algorithm uses a two-step process for efficiency:

### Step 1: Proxy Filtering (Bhattacharyya Coefficient)

**Fast approximate search** using Bhattacharyya coefficient:

```
Bhattacharyya(p, q) = Σᵢ √(pᵢ × qᵢ)
```

- PostgresCoordizer: Uses pgvector inner product on sqrt-space vectors
- FisherCoordizer: In-memory dot product

**Why:** Bhattacharyya is faster to compute and correlates well with Fisher-Rao distance.

### Step 2: Exact Fisher-Rao Distance

**Precise ranking** using exact Fisher-Rao metric:

```
Fisher-Rao(p, q) = arccos(Bhattacharyya(p, q))
```

**Why:** This is the true geodesic distance on the Fisher information manifold.

**Result:** Top-k candidates sorted by exact Fisher-Rao distance (ascending).

## Special Tokens

All special tokens have geometric meaning on the Fisher manifold:

| Token | Geometric Definition | Purpose |
|-------|---------------------|---------|
| `<BOS>` | Uniform superposition | Origin of basin space |
| `<EOS>` | Alternating pattern | Maximal Fisher distance from BOS |
| `<PAD>` | Sparse density matrix | Minimal coupling point |
| `<UNK>` | Golden-angle eigenbasis | Uniform manifold coverage for OOV |

## Vocabulary Structure

### Encoding Vocabulary
- Source: `coordizer_vocabulary` table (all tokens)
- Purpose: Text → basin encoding
- Includes: Subwords, BPE tokens, full words

### Generation Vocabulary  
- Source: `coordizer_vocabulary WHERE token_role IN ('generation', 'both')`
- Purpose: Basin → text decoding
- Includes: Real English words only (no BPE garbage)

### Domain Vocabulary
- Source: `god_vocabulary_profiles` table
- Purpose: God-specific vocabulary weighting
- Usage: Domain boost in decoding (optional)

## Database Schema (PostgresCoordizer)

### coordizer_vocabulary

```sql
CREATE TABLE coordizer_vocabulary (
    token_id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    basin_embedding VECTOR(64) NOT NULL,
    basin_sqrt VECTOR(64),  -- Sqrt-space for Bhattacharyya
    phi_score REAL DEFAULT 0.5,
    qfi_score REAL,
    frequency INTEGER DEFAULT 1,
    source_type TEXT,  -- 'base', 'bip39', 'learned', 'special'
    token_role TEXT,   -- 'encoding', 'generation', 'both'
    pos_tag TEXT,      -- Optional: 'NOUN', 'VERB', etc.
    phrase_category TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for two-step retrieval
CREATE INDEX idx_coordizer_vocab_basin_hnsw 
    ON coordizer_vocabulary USING ivfflat (basin_embedding vector_cosine_ops);
CREATE INDEX idx_coordizer_vocab_sqrt_hnsw 
    ON coordizer_vocabulary USING ivfflat (basin_sqrt vector_cosine_ops);
CREATE INDEX idx_coordizer_vocab_role 
    ON coordizer_vocabulary (token_role);
CREATE INDEX idx_coordizer_vocab_pos 
    ON coordizer_vocabulary (pos_tag);
```

### god_vocabulary_profiles

```sql
CREATE TABLE god_vocabulary_profiles (
    id SERIAL PRIMARY KEY,
    god_name TEXT NOT NULL,
    word TEXT NOT NULL,
    relevance_score REAL DEFAULT 0.0,
    UNIQUE(god_name, word)
);
```

## Geometric Purity (QIG-Pure)

ALL coordizer operations maintain geometric purity:

✅ **Allowed:**
- Fisher-Rao distance
- Bhattacharyya coefficient (proxy)
- Geodesic interpolation
- Von Neumann entropy
- Quantum fidelity
- Density matrix operations

❌ **Not Allowed:**
- Cosine similarity on basins
- Euclidean distance on basins
- Linear averaging of basins
- Euclidean hashing
- Neural embeddings

## Migration from Legacy

If you have old coordizer code, migrate as follows:

```python
# ❌ OLD (removed in WP1.2)
from qig_coordizer import get_coordizer, QIGCoordizer, get_tokenizer

# ✅ NEW (canonical)
from coordizers import get_coordizer, BaseCoordizer, FisherCoordizer, PostgresCoordizer

# ❌ OLD
coordizer.decode(basin, top_k=10)

# ✅ NEW (two-step geometric decoding)
coordizer.decode_geometric(basin, top_k=10)

# ✅ NEW (with POS filtering)
coordizer.decode_geometric(basin, top_k=10, allowed_pos="NOUN")
```

## Testing

Test your coordizer implementation:

```bash
cd qig-backend
python3 -m pytest tests/test_base_coordizer_interface.py -v
```

## References

- **Work Package 3.1:** Coordizer consolidation issue
- **WP2.2:** Canonical geometry module
- **WP1.2:** Backward compatibility removal
- **Migration 0013:** Tokenizer → Coordizer rename
- **Migration 0011:** Vocabulary consolidation

## Version History

- **5.1.0** (2026-01-16): Added BaseCoordizer interface (WP3.1)
- **5.0.0** (2026-01-15): PostgresCoordizer as canonical
- **4.x.x**: Legacy QIGTokenizer/QIGCoordizer (deprecated)
