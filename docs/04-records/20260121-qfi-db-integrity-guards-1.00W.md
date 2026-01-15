# QFI Integrity Guards (2026-01-21)

## Summary
- QFI scores are now constrained to [0, 1] at the database level.
- Active tokens require non-null QFI and basin embeddings.
- Validation tooling is available to confirm invariants in CI or ops workflows.

## Operational Steps
1. Run the migration `0014_qfi_constraints.sql` to apply constraints.
2. Use `npm run validate:db-integrity` to confirm invariants post-migration.
3. Use `tsx tools/recompute_qfi_scores.ts --apply` to backfill QFI values when needed.

## Runtime Guarantees
- `coordizer_vocabulary.qfi_score` is always between 0 and 1 when present.
- `token_status='active'` implies valid QFI and basin embedding.
