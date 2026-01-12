# Migration Scripts

**ARCHIVED** - Historical migration scripts from 2025-2026

These scripts were used for one-time data migrations and are preserved
for reference. Most are no longer needed for new deployments.

## Scripts

- `migrate_checkpoint_to_pg.py` - Moved checkpoints to PostgreSQL (2025-12)
- `migrate_vocab_checkpoint_to_pg.py` - Moved vocabulary to PostgreSQL (2025-12)
- `migrate_olympus_schema.py` - Olympus schema migration (2026-01)
- `fast_migrate_checkpoint.py` - Fast checkpoint migration utility (2025-12)
- `fast_migrate_vocab_checkpoint.py` - Fast vocabulary migration utility (2025-12)
- `populate_tokenizer_vocabulary.py` - Initial vocabulary population (2025-12)

## Status

These scripts are **archived** and should not be needed for new deployments.

For new deployments, use the standard setup process in the main README.md.

If you need to run a migration script, make sure to:
1. Review the script code first
2. Backup your database
3. Test on a development environment
4. Check for any deprecation warnings

## Historical Context

These migrations were part of the transition from file-based persistence to PostgreSQL-backed storage during December 2025 - January 2026.
