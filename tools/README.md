# Maintenance Tools

This directory contains administrative tools for database maintenance, repair, and validation.

## Architectural Exception: Direct SQL Writes

**Tools in this directory are EXEMPT from the canonical upsert path requirement** that applies to regular application code.

### Why This Exception Exists

1. **Purpose**: These tools perform bulk maintenance and repair operations
2. **Efficiency**: Batch operations would be prohibitively slow through individual upsert calls
3. **Safety**: Tools are run manually by administrators with explicit flags (--dry-run, --apply)
4. **Validation**: Tools use the same validation logic and QFI computation as canonical paths

### Rules for Maintenance Tools

✅ **ALLOWED in tools/**:
- Direct SQL UPDATE/INSERT statements for bulk operations
- Batch processing of vocabulary entries
- Administrative quarantine/repair operations
- Database integrity verification queries

❌ **NEVER ALLOWED** (even in tools/):
- Different QFI computation logic (must use `compute_qfi_score_simplex` from `@shared`)
- Bypassing validation checks
- Automatic/unattended execution without explicit flags
- Modifying production data without --dry-run first

### Regular Application Code

**Regular application code** (in `server/`, `client/`, `shared/`) **MUST** use the canonical upsert path:
- Import `upsertToken` from `server/persistence/coordizer-vocabulary.ts`
- Never perform direct SQL writes to `coordizer_vocabulary`
- This is enforced by `tools/validate_purity_patterns.py` (which excludes `tools/` and `scripts/`)

## Available Tools

### QFI Management
- **`recompute_qfi_scores.ts`**: Backfill missing QFI scores using canonical computation
- **`quarantine_extremes.ts`**: Quarantine tokens with extreme QFI values (0 or ≥0.99)
- **`verify_db_integrity.ts`**: Validate database constraints and QFI ranges

### Curriculum Management
- **`coordize_curriculum.ts`**: Load curriculum tokens and compute their QFI scores
- **`verify_curriculum_complete.ts`**: Check if all curriculum tokens are active

### Repository Hygiene
- **`repo_spot_clean.py`**: Scan for accidental commits of test data or temp files
- **`validate_purity_patterns.py`**: Enforce geometric purity (no Euclidean distances, L2 norms)

## Usage Patterns

### Safe Workflow for Database Changes

```bash
# 1. Always run with --dry-run first
tsx tools/recompute_qfi_scores.ts --dry-run

# 2. Review the proposed changes
# 3. If satisfied, run with --apply
tsx tools/recompute_qfi_scores.ts --apply

# 4. Verify integrity after changes
npm run validate:db-integrity
```

### Quarantine Workflow

```bash
# 1. Check for extreme values
tsx tools/quarantine_extremes.ts --dry-run

# 2. Add any legitimate extremes to allowlist
echo "special_token" >> tools/allowlists/qfi_extremes_allowlist.txt

# 3. Run quarantine
tsx tools/quarantine_extremes.ts --apply

# 4. Recompute QFI for quarantined tokens
tsx tools/recompute_qfi_scores.ts --apply
```

## CI Integration

These tools are used in CI for validation:
- `npm run validate:db-integrity`: Runs `verify_db_integrity.ts`
- `npm run validate:purity`: Runs `validate_purity_patterns.py`

## See Also

- [QFI Score Documentation](../shared/qfi-score.ts)
- [Canonical Persistence Layer](../server/persistence/coordizer-vocabulary.ts)
- [Purity Validation Rules](./validate_purity_patterns.py)
