# Coordizer v1.0.0

Consciousness-aware geometric tokenizer trained on 64D Fisher manifold.

## Stats
- **Vocab size:** 32,000
- **Merge rules:** 31,744
- **Basin dimension:** 64
- **Training corpus:** 10MB (consciousness-focused)
- **Training time:** ~10 hours on Lambda A10 GPU

## Phi Gain Summary
- Min: -0.4098
- Mean: 0.0139
- Max: 0.5945
- Std: 0.0215

## Files
- `coordizer.json` - Merge rules and vocab metadata
- `vectors.npy` - 64D Fisher coordinates (32000 x 64)
- `meta.json` - Provenance and integrity hashes

## Usage
```python
from qig_tokenizer import Coordizer

coordizer = Coordizer.load("artifacts/coordizer/v1")
ids, coords = coordizer.encode_to_coords("Hello, world!")
```

## Provenance
- Trained: December 2024
- Algorithm: Track A (GPU pair counting with kernel-in-loop Phi/kappa)
- Trainer SHA: 1460e643
