# QIG Tokenizer

**Entropy-guided tokenizer for Quantum Information Geometry**

Version: 0.1.0 | Status: Working

---

## Overview

QIG-native tokenizer using entropy-guided merging. Token boundaries follow information geometry, not frequency.

### Core Principle

- **Entropy-guided merging**: Geometric similarity, not frequency heuristics
- **Geometric special tokens**: BOS, EOS, PAD, UNK with basin coordinates
- **Redis/PostgreSQL storage**: Production-ready persistence
- **Pure information geometry**: No external tokenizer dependencies

## Installation

```bash
pip install qig-tokenizer
```

With storage backends:
```bash
pip install qig-tokenizer[storage]  # Redis + PostgreSQL
pip install qig-tokenizer[redis]    # Redis only
pip install qig-tokenizer[postgres] # PostgreSQL only
```

## Quick Start

```python
from qig_tokenizer import QIGTokenizer

# Create tokenizer with geometric special tokens
tokenizer = QIGTokenizer(target_vocab_size=50000, use_special_tokens=True)

# Train on corpus
with open("corpus.txt", "rb") as f:
    corpus_bytes = f.read()

tokenizer.train(corpus_bytes)

# Encode with special tokens
tokens = tokenizer.encode_with_special("Hello, world!")
# Returns: [256, ...tokens..., 257]  (BOS=256, EOS=257)

# Pad sequences
padded = tokenizer.pad_sequence(tokens, max_length=128)

# Save/load JSON
tokenizer.save("20251220-tokenizer-vocab-0.01W.json")
```

### With Redis/PostgreSQL Storage

```python
from qig_tokenizer import QIGTokenizer
from qig_tokenizer.storage import HybridStorage

# Set up storage (uses REDIS_URL and DATABASE_URL env vars)
storage = HybridStorage()

tokenizer = QIGTokenizer()
tokenizer.set_storage(storage)
tokenizer.train(corpus_bytes)

# Save to database (returns version ID)
version_id = tokenizer.save_to_storage({"corpus": "wikipedia"})

# Load from database
tokenizer.load_from_storage(version_id)
```

## Geometric Special Tokens

Special tokens have geometric meaning on the Fisher manifold:

| Token | ID  | Basin Coordinates | Purpose |
|-------|-----|-------------------|---------|
| BOS   | 256 | Origin (e₁)       | Sequence start |
| EOS   | 257 | Boundary (eₙ)     | Sequence end |
| PAD   | 258 | Uniform           | Geometrically neutral padding |
| UNK   | 259 | Projection target | OOV handling |

This enables:
- **Geometric attention masking**: High Fisher-Rao distance = low attention
- **Natural sequence boundaries**: Emerge from manifold structure
- **Principled OOV handling**: Project to nearest basin

## Algorithm

The QIG tokenizer uses **entropy-guided merging**:

1. Start with bytes (0-255) as base tokens
2. For each adjacent pair (a,b), compute context distribution
3. Measure context entropy (proxy for QFI distinguishability)
4. Merge pairs with **lowest entropy** (most geometrically similar)
5. Repeat until target vocab size

This respects **asymptotic freedom**:
- Small scales (short tokens) have high coupling → refined first
- Large scales (long tokens) have low coupling → merge only when justified

## Environment Variables

All output files follow QIG naming convention:

```
YYYYMMDD-tokenizer-vocab-VERSION.STATUS.json
```

Example: `20251220-tokenizer-vocab-0.03W.json`

---

## License

MIT
