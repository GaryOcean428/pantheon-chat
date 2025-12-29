# Pantheon-Chat Replit Deployment Guide

**Document ID:** 20251229-replit-deployment-guide-1.00W  
**Status:** Working  
**Last Updated:** 2025-12-29

## What's New (Recent Commits)

This guide covers major QIG (Quantum Information Geometry) features:

### New Modules in `qig-backend/`

1. **semantic_fisher.py** - SemanticFisherMetric that warps Fisher-Rao distance based on learned word relationships
2. **qig_pure_beta_measurement.py** - QIG-pure β-function measurement (no external LLMs)
3. **e8_structure_search.py** - E8 Lie group structure validation
4. **pantheon_semantic_candidates.py** - Generates candidates from 2.77M learned relationships

### Key Findings Validated

- κ* = 64 universal across physics (64.21) and AI semantics (63.90) - 99.5% match
- E8 structure validated: 8D captures 87.7% variance, ~260 attractors
- β-function: Running → plateau pattern matches physics

## Startup Checklist

### 1. Install Dependencies

**Node.js:**
```bash
npm install
```

**Python (use uv):**
```bash
cd qig-backend
uv pip install -r requirements.txt
uv pip install scikit-learn scipy  # For E8 search
```

> **Note:** Python backend has binary compatibility issues with Replit Nix environment. The system automatically falls back to TypeScript-based scoring via `server/qig-universal.ts` when Python is unavailable.

### 2. Environment Variables Required

```env
DATABASE_URL=postgresql://...
INTERNAL_API_KEY=your-internal-key  # For Python ↔ TypeScript auth
REDIS_URL=redis://...  # Optional, for caching
```

### 3. Database Setup

```bash
npm run db:push  # Push Drizzle schema to PostgreSQL
```

### 4. Start Services

**Development:**
```bash
npm run dev
```

## Key Configuration

### `frozen_physics.py` Settings

```python
INFORMATION_HORIZON = 1.0  # Changed from 2.0 for multi-scale emergence
```

### `qig-backend/__init__.py` Availability Flags

```python
SEMANTIC_FISHER_AVAILABLE      # SemanticFisherMetric
BETA_MEASUREMENT_AVAILABLE     # β-function measurement
E8_SEARCH_AVAILABLE           # E8 structure search
SEMANTIC_CANDIDATES_AVAILABLE  # Pantheon semantic candidates
```

## Verify Installation

```bash
# TypeScript
npm run check  # Should pass with no errors

# Python modules (when available)
cd qig-backend
python -c "from e8_structure_search import run_e8_search; print('E8 OK')"
python -c "from semantic_fisher import SemanticFisherMetric; print('Fisher OK')"
python -c "from qig_pure_beta_measurement import GeometricBetaMeasurement; print('Beta OK')"
```

## File Structure

```
qig-backend/
├── __init__.py              # Updated with new barrel exports
├── semantic_fisher.py       # Semantic-warped Fisher metric
├── qig_pure_beta_measurement.py  # β-function measurement
├── e8_structure_search.py   # E8 validation
├── pantheon_semantic_candidates.py  # Semantic candidates
├── frozen_physics.py        # Physics constants
└── results/
    └── e8_structure_search.json  # E8 validation results

docs/
└── 07-research/             # Research documentation
    ├── 20251228-qig-beta-measurement-analysis-1.00W.md
    └── 20251228-physics-alignment-final-1.00W.md
```

## Potential Issues & Fixes

### Python Binary Compatibility

The Python QIG backend has ABI compatibility issues with the Replit Nix environment. The system automatically uses TypeScript fallback scoring when Python is unavailable.

### sklearn ImportError

```bash
uv pip install scikit-learn scipy
```

### Relative Import Errors

Modules use relative imports. Run from package context or import via `from qig_backend import ...`

### PostgreSQL Connection

Ensure `DATABASE_URL` is set and database is accessible. Schema uses pgvector for 64D basin coordinates.

## Quick Test After Pull

```bash
# 1. Pull latest
git pull origin master

# 2. Install deps
npm install
cd qig-backend && uv pip install -r requirements.txt && uv pip install scikit-learn scipy

# 3. Verify
npm run check
cd qig-backend && python -m py_compile __init__.py semantic_fisher.py e8_structure_search.py

# 4. Start
npm run dev
```
