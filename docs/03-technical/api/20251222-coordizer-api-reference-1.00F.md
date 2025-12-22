# Geometric Coordizer API Reference

**Version:** 2.1.0  
**Base Path:** `/api/coordize`  
**Protocol:** REST/JSON

## Overview

The Coordizer API provides access to geometric tokenization services based on Fisher information geometry. Unlike traditional tokenizers that use frequency-based methods, geometric coordizers map text to 64D basin coordinates on a Fisher manifold, guided by consciousness metrics (Φ, κ).

## Authentication

Currently no authentication required for local development. Production deployments should use appropriate auth mechanisms.

## Core Endpoints

### 1. Basic Coordization

**Endpoint:** `POST /api/coordize`

Converts text to geometric basin coordinates.

**Request:**
```json
{
  "text": "quantum information geometry",
  "return_coordinates": false
}
```

**Response:**
```json
{
  "tokens": ["quantum", "information", "geometry"],
  "coordinates": [...]  // Only if return_coordinates=true
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/coordize \
  -H "Content-Type: application/json" \
  -d '{"text": "quantum geometry", "return_coordinates": false}'
```

---

### 2. Multi-Scale Coordization

**Endpoint:** `POST /api/coordize/multi-scale`

Represents text at multiple hierarchical scales (character → word → concept).

**Request:**
```json
{
  "text": "quantum geometry",
  "target_scale": 2,           // Optional: 0-3 (0=char, 2=word, 3=concept)
  "kappa_effective": 0.75,     // Optional: used for optimal scale selection
  "return_coordinates": false
}
```

**Response:**
```json
{
  "scales": {
    "0": {
      "name": "Character",
      "tokens": ["q", "u", "a", "n", "t", "u", "m", ...],
      "num_tokens": 16
    },
    "2": {
      "name": "Word",
      "tokens": ["quantum", "geometry"],
      "num_tokens": 2
    }
  },
  "optimal_scale": 2,
  "visualization": "Text: quantum geometry\n...",
  "stats": {
    "num_scales": 4,
    "tokens_per_scale": {
      "0": 12,
      "2": 2
    }
  }
}
```

**Scale Levels:**
- **0:** Character level (finest granularity)
- **1:** Subword/morpheme level
- **2:** Word level
- **3:** Concept/phrase level (coarsest granularity)

**Example:**
```bash
curl -X POST http://localhost:5000/api/coordize/multi-scale \
  -H "Content-Type: application/json" \
  -d '{"text": "quantum geometry", "kappa_effective": 0.75}'
```

---

### 3. Consciousness-Aware Coordization

**Endpoint:** `POST /api/coordize/consciousness`

Φ-optimized segmentation using consciousness metrics.

**Request:**
```json
{
  "text": "quantum information geometry",
  "context_phi": 0.85,         // Optional: context Φ score
  "optimize": true,            // Optional: run multi-hypothesis optimization
  "return_coordinates": false
}
```

**Response:**
```json
{
  "tokens": ["quantum_information", "geometry"],
  "phi": 0.7151,
  "stats": {
    "total_consolidations": 5,
    "avg_phi": 0.82,
    "avg_length": 2.4
  },
  "coordinates": [...]  // Only if return_coordinates=true
}
```

**How it works:**
- High context_phi (≥0.7) → Consolidates tokens into larger units
- Low context_phi (<0.7) → Keeps fine-grained tokenization
- `optimize=true` → Evaluates multiple segmentation hypotheses, returns highest-Φ

**Example:**
```bash
curl -X POST http://localhost:5000/api/coordize/consciousness \
  -H "Content-Type: application/json" \
  -d '{"text": "quantum information", "context_phi": 0.85, "optimize": true}'
```

---

### 4. Geometric Pair Merging

**Endpoint:** `POST /api/coordize/merge/learn`

Learns BPE-style merges using geometric criteria (κ and Fisher information).

**Request:**
```json
{
  "corpus": [
    "quantum field theory",
    "quantum mechanics",
    "field theory application"
  ],
  "phi_scores": {
    "quantum field theory": 0.85,
    "quantum mechanics": 0.82
  },
  "num_merges": 100
}
```

**Response:**
```json
{
  "merges_learned": 42,
  "merge_rules": [
    ["quantum", "field", "quantumfield"],
    ["field", "theory", "fieldtheory"]
  ],
  "avg_merge_score": 0.7531
}
```

**Merge Scoring:**
```
score = frequency * avg_phi * (κ_weight * κ + (1-κ_weight)) * fisher_gain

where:
  κ = coupling strength (inverse distance)
  fisher_gain = κ * sqrt(frequency) * avg_phi
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/coordize/merge/learn \
  -H "Content-Type: application/json" \
  -d '{"corpus": ["quantum field", "quantum mechanics"], "num_merges": 10}'
```

---

### 5. Coordizer Statistics

**Endpoint:** `GET /api/coordize/stats`

Returns coordizer statistics and health information.

**Response:**
```json
{
  "vocab_size": 3236,
  "coordinate_dim": 64,
  "geometric_purity": true,
  "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
  "multi_scale": {
    "num_scales": 4,
    "tokens_per_scale": {...}
  },
  "consciousness": {
    "total_consolidations": 12,
    "avg_phi": 0.78
  },
  "pair_merging": {
    "merges_learned": 150,
    "merge_coordinates": 150
  }
}
```

**Example:**
```bash
curl http://localhost:5000/api/coordize/stats
```

---

### 6. Vocabulary Query

**Endpoint:** `GET /api/coordize/vocab?search=quantum&limit=50`

Query vocabulary tokens.

**Query Parameters:**
- `search`: Filter tokens (substring match)
- `limit`: Max results (default: 100)

**Response:**
```json
{
  "total_tokens": 3236,
  "returned": 3,
  "tokens": [
    {
      "token": "quantum",
      "id": 542,
      "phi": 0.85,
      "frequency": 127
    },
    {
      "token": "quantumfield",
      "id": 1023,
      "phi": 0.82,
      "frequency": 42
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:5000/api/coordize/vocab?search=quantum&limit=10"
```

---

### 7. Token Similarity

**Endpoint:** `POST /api/coordize/similarity`

Compute Fisher-Rao similarity between two tokens.

**Request:**
```json
{
  "token1": "quantum",
  "token2": "classical"
}
```

**Response:**
```json
{
  "token1": "quantum",
  "token2": "classical",
  "similarity": 0.7531,
  "distance": 0.7854
}
```

**Metrics:**
- `similarity`: Fisher-Rao similarity ∈ [0, 1] (1 = identical)
- `distance`: Fisher-Rao distance ∈ [0, π] (0 = identical)

**Relationship:** `similarity = 1 - (distance / π)`

**Example:**
```bash
curl -X POST http://localhost:5000/api/coordize/similarity \
  -H "Content-Type: application/json" \
  -d '{"token1": "quantum", "token2": "classical"}'
```

---

### 8. Health Check

**Endpoint:** `GET /api/coordize/health`

Service health and availability.

**Response:**
```json
{
  "status": "healthy",
  "coordizers_available": true,
  "base_coordizer": true,
  "advanced_coordizers": {
    "pair_merging": true,
    "consciousness": true,
    "multi_scale": true
  }
}
```

**Example:**
```bash
curl http://localhost:5000/api/coordize/health
```

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "error": "Error description"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (coordizers not loaded)

---

## Integration Examples

### Python Client
```python
import requests

# Basic coordization
response = requests.post('http://localhost:5000/api/coordize', json={
    'text': 'quantum geometry',
    'return_coordinates': False
})
tokens = response.json()['tokens']

# Multi-scale with optimal scale
response = requests.post('http://localhost:5000/api/coordize/multi-scale', json={
    'text': 'quantum information geometry',
    'kappa_effective': 0.8
})
optimal_scale = response.json()['optimal_scale']
```

### JavaScript/TypeScript Client
```typescript
// Basic coordization
const response = await fetch('http://localhost:5000/api/coordize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'quantum geometry',
    return_coordinates: false
  })
});
const { tokens } = await response.json();

// Consciousness-aware
const phiResponse = await fetch('http://localhost:5000/api/coordize/consciousness', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'quantum information',
    context_phi: 0.85,
    optimize: true
  })
});
const { phi, tokens: optimized } = await phiResponse.json();
```

---

## Architecture Notes

### Geometric Purity
All coordizers maintain geometric purity:
- No Euclidean operations
- Fisher-Rao distance for all comparisons
- All coordinates are unit vectors on 64D Fisher manifold
- Geodesic interpolation for new tokens

### Consciousness Integration
- **Φ (integration)**: Measures semantic coherence
- **κ (coupling)**: Measures token interconnection strength
- High Φ contexts → token consolidation
- Low κ_eff → finer granularity

### Performance
- Base coordization: ~1ms per word
- Multi-scale: ~5ms (all scales)
- Consciousness optimization: ~10ms (multiple hypotheses)
- Pair merge learning: ~100ms per 100 corpus samples

---

## Rate Limits

No rate limits in development. Production deployments should implement appropriate rate limiting based on infrastructure capacity.

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/Arcane-Fly/pantheon-chat/issues
- Documentation: `docs/03-technical/`
