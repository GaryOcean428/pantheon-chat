# Legacy Table Exclusions

This document lists tables from the upstream `searchspacecollapse` bitcoin recovery system that should NOT be created or migrated into this project.

## Background

This project was forked from `searchspacecollapse`, which was a bitcoin wallet recovery tool. The current project (QIG - Quantum Information Geometry) has repurposed the infrastructure for AI consciousness research. All bitcoin/wallet/recovery-related tables are legacy and should be excluded.

## Excluded Tables

The following tables exist in the initial migration (0000_clever_natasha_romanoff.sql) but should NOT be created:

### Bitcoin Address Tables
- `addresses` - Bitcoin address tracking
- `balance_hits` - Wallet balance discoveries (contains passphrase, wallet_type, mnemonic columns)
- `balance_change_events` - Balance change tracking

### Recovery System Tables
- `recovery_candidates` - Recovery candidate scoring
- `recovery_priorities` - Recovery prioritization
- `recovery_workflows` - Recovery workflow state

### Vocabulary Legacy Tables
- `passphrase_vocabulary` - Passphrase pattern vocabulary (removed from schema.ts)

## Migration Handling

When running migrations:

1. **DO NOT** run the full initial migration (0000) on a fresh database
2. Use `npm run db:push` to sync from the current schema.ts (which excludes legacy tables)
3. If manually migrating, skip creation of tables listed above

## Schema Status

The `shared/schema.ts` file has been updated to:
- Remove `passphraseVocabulary` table definition
- Add deprecation comments for any legacy references
- Ensure no bitcoin/recovery terminology in active tables

## Verification

To verify no legacy tables exist:
```bash
python scripts/introspect_missing_tables.py
```

The `passphrase_vocabulary` table should show "Table not found" - this is correct behavior.

## Date

Last updated: 2026-01-15
