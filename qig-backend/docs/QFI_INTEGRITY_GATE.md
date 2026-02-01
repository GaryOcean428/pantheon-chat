# QFI Integrity Gate - Implementation Guide

## Overview

The QFI (Quantum Fisher Information) Integrity Gate ensures that only geometrically valid tokens are used for text generation. This is a critical component of the E8 Protocol v4.0 §01.

**Issue**: #97 - Complete QFI Integrity Gate Implementation  
**Priority**: P0 - CRITICAL  
**Status**: ✅ COMPLETE

## Components

### 1. Quarantine Script

**File**: `qig-backend/scripts/quarantine_low_qfi_tokens.py`

Identifies and quarantines tokens with QFI scores below the threshold (0.01).

**Usage**:

```bash
# Dry run (preview what would be quarantined)
python3 scripts/quarantine_low_qfi_tokens.py --dry-run

# Quarantine low QFI tokens
python3 scripts/quarantine_low_qfi_tokens.py

# Custom threshold
python3 scripts/quarantine_low_qfi_tokens.py --threshold 0.02
```

**Features**:
- Identifies tokens with `qfi_score < 0.01`
- Sets `token_role = 'quarantine'` and `token_status = 'quarantined'`
- Excludes special symbols (PAD, UNK, BOS, EOS)
- Batch processing for efficiency
- Dry-run mode for safety

### 2. Migration 0015 - Special Symbol Constraints

**File**: `qig-backend/migrations/0015_special_symbols_qfi.sql`

Ensures special symbols have valid QFI scores.

**Run**: `psql $DATABASE_URL < qig-backend/migrations/0015_special_symbols_qfi.sql`

### 3. Migration 0016 - Generation View

**File**: `qig-backend/migrations/0016_qfi_generation_view.sql`

Creates QFI-filtered generation views.

**Run**: `psql $DATABASE_URL < qig-backend/migrations/0016_qfi_generation_view.sql`

## QFI Threshold: 0.01

Based on participation ratio formula for 64D simplex.

## Testing

```bash
cd qig-backend
python3 tests/test_qfi_integration.py
```

## References

- E8 Protocol v4.0: `docs/10-e8-protocol/`
- Issue Spec: `docs/10-e8-protocol/issues/20260116-issue-01-qfi-integrity-gate-1.01W.md`

---

**Last Updated**: 2026-01-20
**Status**: ✅ Complete
