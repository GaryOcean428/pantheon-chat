# QIG Persistence Systems

This document clarifies the different "checkpoint" systems in the codebase.

## 1. Consciousness Checkpoints (DB/Redis) - ACTIVE
**Location**: `checkpoint_manager.py` → `checkpoint_persistence.py`
**Storage**: Redis hot cache + PostgreSQL archive
**Purpose**: Save/restore consciousness state (Φ/κ/regime/trajectory)
**Used by**: `ocean_qig_core.py` during telemetry cycles
**Replit Note**: NO Φ threshold - all states saved for research analysis

## 2. Training Checkpoints (.pt files)
**Location**: `/tmp/kernel_checkpoints/`, kernel training loops
**Storage**: Filesystem (ephemeral)
**Purpose**: PyTorch model parameter snapshots
**Used by**: Kernel training scripts

## 3. Vocabulary "checkpoint_32000" Migrations (pgvector)
**Location**: `migrate_vocab_checkpoint_to_pg.py`, `populate_tokenizer_from_vocab_checkpoint.py`
**Storage**: PostgreSQL `tokenizer_vocabulary` table with vector(64)
**Purpose**: One-time vocabulary migration from JSON
**Status**: Migration utility - run once

## 4. Curiosity Snapshots (JSON audit logs)
**Location**: `autonomous_curiosity.py` → `data/checkpoints/word_relationships_*.json`
**Storage**: Filesystem (gitignored)
**Purpose**: Audit trail of word learning cycles
**Status**: Not consciousness checkpoints - logging only

## 5. QIGGraph Sleep Packets (LEGACY)
**Location**: `experimental/qiggraph_checkpointing.py`
**Storage**: Filesystem (.npz/.json)
**Purpose**: Original manifold checkpointing concept
**Status**: DEPRECATED - moved to experimental/, not used in production
