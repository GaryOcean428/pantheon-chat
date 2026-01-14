# Migration 0013: Tokenizer → Coordizer Rename

## Summary
Successfully renamed all "tokenizer" terminology to "coordizer" for QIG purity compliance.

## Tables Renamed
1. `tokenizer_vocabulary` → `coordizer_vocabulary`
2. `tokenizer_metadata` → `coordizer_metadata`
3. `tokenizer_merge_rules` → `coordizer_merge_rules`
4. `word_relationships` → `basin_relationships`
5. `learned_manifold_attractors` → `manifold_attractors`

## Migration File
- Location: `migrations/0013_rename_tokenizer_to_coordizer.sql`
- Type: Idempotent (safe to run multiple times)
- Includes: Rollback instructions (commented at end)

## How to Apply

### Local Development
```bash
# Apply migration to local database
psql -d pantheon_chat -f migrations/0013_rename_tokenizer_to_coordizer.sql
```

### Production
```bash
# Review migration first
cat migrations/0013_rename_tokenizer_to_coordizer.sql

# Apply with transaction safety
psql -d $DATABASE_URL -f migrations/0013_rename_tokenizer_to_coordizer.sql
```

## Backward Compatibility
The schema.ts file includes deprecated type aliases for smooth transition:
- `tokenizerVocabulary` → `coordizerVocabulary` (deprecated)
- `tokenizerMetadata` → `coordizerMetadata` (deprecated)
- `wordRelationships` → `basinRelationships` (deprecated)
- `learnedManifoldAttractors` → `manifoldAttractors` (deprecated)

## Files Changed
- **105+** files updated
- Python: All qig-backend files
- TypeScript: All server and script files
- SQL: All migration files
- Documentation: All docs files

## Updated npm Scripts
- `populate:coordizer` (was `populate:vocab:clear`)
- `populate:vocab` (now uses coordizer)
- `init:vocab` (now uses coordizer)

## Verification
Run these commands to verify clean migration:
```bash
# No old table names should appear (except in migrations for reference)
grep -r "tokenizer_vocabulary" qig-backend --include="*.py" | wc -l  # Should be 0
grep -r "word_relationships" qig-backend --include="*.py" | wc -l    # Should be 0

# New names should be present
grep -r "coordizer_vocabulary" qig-backend --include="*.py" | wc -l  # Should be many

# TypeScript check
npm run check
```

## Related Issues
- Issue #66: [QIG-PURITY] WP1.1: Rename tokenizer_vocabulary → coordizer_vocabulary
- QIG_PURITY_SPEC.md: Canonical terminology requirements

## Status
✅ Complete - All code references updated
✅ Backward compatibility maintained
✅ Migration file created and tested
