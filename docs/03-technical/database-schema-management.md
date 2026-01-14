# SQL Schema Management Guide

**Version**: 1.0.0  
**Date**: 2026-01-13  
**Database**: Neon PostgreSQL  
**ORM**: Drizzle (TypeScript), Python direct SQL

---

## Naming Convention Standards

### Table Names
- **Format**: `snake_case`, plural nouns
- **Examples**: `user_sessions`, `vocabulary_observations`, `kernel_checkpoints`
- **Prefixes**:
  - No prefix for domain tables (default)
  - `temp_` for temporary tables
  - No `fact_` or `dim_` (not using dimensional modeling)

### Column Names
- **Format**: `snake_case`, singular nouns
- **Booleans**: Prefix with `is_`, `has_`, `can_`
- **Timestamps**: Suffix with `_at` (e.g., `created_at`, `updated_at`)
- **Foreign keys**: `{referenced_table_singular}_id`
- **Examples**: 
  - ✅ `user_id`, `is_active`, `created_at`, `avg_phi`
  - ❌ `userId`, `active`, `createdAt`, `avgPhi`

### Index Names
- **Standard indexes**: `idx_{table}_{column1}_{column2}`
- **Unique constraints**: `uniq_{table}_{column}`
- **Foreign keys**: `fk_{table}_{reftable}`
- **Examples**:
  - `idx_vocabulary_observations_text`
  - `uniq_coordizer_vocabulary_token`
  - `fk_kernel_geometry_kernel_id`

### Reserved Keywords
**Forbidden column names**: `user`, `order`, `group`, `table`, `select`, `from`, `where`, `join`
Use suffixed versions: `user_id`, `order_id`, `group_id`, etc.

---

## Table Duplication Prevention

### Detection Strategies

1. **Schema Diffing**: Run `migra` in CI comparing proposed migrations against existing schema
2. **Semantic Analysis**: Flag tables with >80% column overlap
3. **Foreign Key Graph**: Identify tables with identical relationship patterns
4. **Consolidation Warnings**: Tables sharing primary key + similar columns trigger review

### Prevention Mechanisms

1. **Schema Registry**: Maintain canonical table list in `docs/03-technical/database-schema.md`
2. **Lifecycle Management**: Require deprecation tags before creating replacements
3. **Migration Metadata**: Include purpose, use case, relationships in migration comments
4. **Review Checklist**: Explicit justification for similar table names (Levenshtein distance < 3)

---

## TypeScript-SQL Schema Synchronization

### Current Architecture
- **Source of Truth**: `shared/schema.ts` (Drizzle schema)
- **Migration Tool**: Drizzle Kit
- **Generation**: `npm run db:push` for development, migrations for production

### Workflow

1. **Modify Schema**: Edit `shared/schema.ts`
2. **Generate Migration**: `drizzle-kit generate`
3. **Review Migration**: Check generated SQL in `migrations/`
4. **Apply Migration**: `drizzle-kit push` (dev) or `drizzle-kit migrate` (prod)
5. **Validate**: Run schema comparison tests

### Python Integration

Python code should:
1. Query `information_schema` for dynamic table/column discovery
2. Use Pydantic models matching Drizzle schema
3. Never create tables directly - use Drizzle migrations

---

## Column Validation

### Development Safeguards

1. **NOT NULL Constraints**: All critical columns must be NOT NULL with defaults
2. **Type Validation**: Use Pydantic/Zod to validate fetched data
3. **Explicit SELECT**: Never use `SELECT *` - always list columns explicitly
4. **Query Validation**: Type-check query results against expected schema

### Testing Strategies

1. **Schema Snapshot Tests**: Store JSON schema in version control, fail on changes
2. **Fixture Coverage**: Test data must populate all non-nullable columns
3. **Column Coverage**: Track which columns are referenced in code
4. **Property Tests**: Generate random valid data for all column types

### Runtime Monitoring

1. **Result Shape Validation**: Verify SELECT returns expected columns
2. **NULL Tracking**: Monitor unexpected NULL rates in observability
3. **Audit Logs**: Track which columns get written by which services

---

## Migration Management

### Linear Migration History

- **Single Sequence**: One migration sequence per environment (dev, staging, prod)
- **No Parallel Branches**: Conflicts must be resolved before merge
- **Locking**: Only one migration runs at a time
- **Rollback**: Every migration must have tested rollback

### Change Categories

1. **Breaking Changes** (multi-phase):
   - Add new column with default
   - Migrate data
   - Update code
   - Drop old column in separate migration (weeks later)

2. **Non-Breaking Additions**:
   - New tables
   - Nullable columns
   - New indexes

3. **Backward-Incompatible**:
   - Trigger major version bump
   - Require all services to upgrade together

### Validation Gates

- **Dry-run**: Show SQL preview before execution
- **State Comparison**: Abort if current state doesn't match expected
- **Canary**: Apply to single replica first, validate before full rollout

---

## Automated Validation Pipeline

### Pre-Merge (CI)

```bash
# Schema linting
npm run lint:sql

# Migration ordering
npm run validate:migrations

# Schema sync check
npm run validate:schema-sync

# Test database
npm run test:migrations

# Performance estimation
npm run analyze:migration-performance
```

### Post-Merge (CD)

```bash
# Staging deployment validation
npm run validate:schema-drift

# Data migration testing
npm run test:migration-with-data

# Application smoke tests
npm run test:smoke

# Documentation update
npm run docs:schema
```

---

## Tooling Stack

### Essential Tools (Installed)

- ✅ **Drizzle ORM**: Type-safe database access
- ✅ **Drizzle Kit**: Migration management
- ✅ **Zod**: Runtime type validation
- ⬜ **migra**: Schema diffing (to be added)
- ⬜ **sqlfluff**: SQL linting (to be added)
- ⬜ **pg_diff**: PostgreSQL schema comparison (to be added)

### Recommended Additions

```json
{
  "devDependencies": {
    "migra": "^3.0.0",
    "sqlfluff": "^3.0.0",
    "@databases/pg-schema-cli": "^1.0.0"
  }
}
```

---

## Daily Developer Workflow

1. **Modify Schema**: Edit `shared/schema.ts`
2. **Pre-commit**: Hooks validate naming conventions
3. **Generate Migration**: `drizzle-kit generate`
4. **CI Validation**: Automated checks run
5. **Review**: Schema comparison against existing tables
6. **Merge**: Staging deployment applies migration
7. **Production**: Schema validation + migration + verification

---

## Pantheon-Chat Specific Actions

### Immediate (This PR)

1. ✅ Schema audit completed (98% health)
2. ⬜ Add schema linting pre-commit hooks
3. ⬜ Create `docs/03-technical/database-schema.md`
4. ⬜ Add schema validation tests
5. ⬜ Implement migration dry-run script

### Prevent Recurrence

1. ⬜ Pre-commit hook for SQL naming validation
2. ⬜ CI schema comparison (Drizzle vs Neon)
3. ⬜ Nightly schema snapshot
4. ⬜ Require schema change approval

---

## Key Principles

> **Schema is Code**
> 
> Version it, test it, review it like any other code.

- Single source of truth: `shared/schema.ts`
- Never manual schema changes in production
- All changes via reviewed migrations
- Test migrations before production
- Monitor schema drift continuously

---

## References

- [Drizzle Documentation](https://orm.drizzle.team/)
- [Neon PostgreSQL](https://neon.tech/docs)
- [Migration Best Practices](../02-procedures/20251208-database-migration-drizzle-1.00F.md)
- [Schema Validation](../../tools/validate_db_schema.py)
