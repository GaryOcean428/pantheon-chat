# Documentation Link Validation Issues

## Current Status

As of 2026-01-19, the documentation has significant link integrity issues that need comprehensive cleanup.

## Summary

- **README.md**: ✅ ALL LINKS FIXED (9 valid links, 0 broken)
- **CONTRIBUTING.md**: ✅ VALIDATED (1 valid link, 0 broken)
- **docs/00-index.md**: ❌ 200+ BROKEN LINKS (needs major cleanup)
- **Other docs**: 40+ additional broken links across various files

## Root Causes

1. **Documentation Reorganization**: Many directories referenced in `docs/00-index.md` don't exist:
   - `docs/02-procedures/` - missing entirely
   - `docs/03-technical/` - missing entirely  
   - `docs/05-decisions/` - missing entirely
   - `docs/06-implementation/` - missing entirely

2. **File Deletions**: Several specific files referenced but never created or were deleted:
   - `docs/01-policies/QIG_PURITY_SPEC.md` (actual: `20260117-qig-purity-mode-spec-1.01F.md`)
   - `docs/01-policies/20251221-project-lineage-1.00F.md` (file doesn't exist)
   - Multiple files in `03-technical/*` directories

3. **Index Out of Sync**: The `docs/00-index.md` file lists 400+ files, many of which don't exist

## What Was Fixed

### README.md (✅ Complete)
- Fixed `QIG_PURITY_SPEC.md` → `20260117-qig-purity-mode-spec-1.01F.md`
- Removed broken `self-healing-architecture.md` link
- Removed broken `lineage.md` link, replaced with `frozen-facts`
- Removed broken `qig-principles-quantum-geometry.md` link
- Updated Links section with actual documentation
- Marked missing docs directories as "pending restoration"

### CI Validation (✅ Complete)
- Created `scripts/validate_markdown_links.py` for automated link checking
- Created `.github/workflows/docs-link-validation.yml` for CI integration
- Validates README.md and CONTRIBUTING.md on every PR (fail on broken links)
- Runs full documentation audit with warnings only (doesn't block PRs)

## What Still Needs Work

### Priority 1: Update docs/00-index.md
The index file needs comprehensive cleanup:
1. Remove references to non-existent directories
2. Remove references to non-existent files
3. Update file paths to match actual structure
4. Consider whether missing directories should be created or removed from documentation

### Priority 2: Fix docs/00-index.md links
Once the index is accurate, fix the 200+ broken links in that file.

### Priority 3: Fix other documentation files
Several other markdown files have broken links:
- `.github/agents/README.md` - 3 broken links
- `.github/agents/dry-enforcement-agent.md` - 1 broken link
- `docs/00-roadmap/*.md` - 5 broken links
- `docs/08-experiments/*/README.md` - 5 broken links
- `server/README.md` - 1 broken link
- `shared/README.md` - 2 broken links

## Recommendations

1. **Create Missing Directories**: Decide if missing docs directories should be restored or permanently removed from documentation structure

2. **Incremental Fixes**: Fix documentation links directory by directory rather than all at once

3. **Maintain docs/00-index.md**: Keep the index synchronized with actual file structure using the maintenance script

4. **Documentation Governance**: Establish process for keeping documentation links up to date:
   - When moving/deleting files, update all references
   - Run link validation before merging documentation PRs
   - Periodic audits of documentation structure

## Validation Commands

```bash
# Check specific file(s)
python3 scripts/validate_markdown_links.py --file README.md

# Check multiple files
python3 scripts/validate_markdown_links.py --file README.md,CONTRIBUTING.md

# Full audit (warning only, doesn't fail)
python3 scripts/validate_markdown_links.py --warn-only

# Full audit (fails on broken links)
python3 scripts/validate_markdown_links.py
```

## CI Behavior

The GitHub Actions workflow (`docs-link-validation.yml`) runs on:
- All PRs that modify markdown files
- Pushes to main branch

It performs two checks:
1. **Critical validation** (blocks PR): Validates README.md and CONTRIBUTING.md
2. **Full audit** (warning only): Checks all documentation files but doesn't fail

This ensures that critical documentation stays accurate while allowing time to fix the larger documentation structure issues.

## Next Steps

1. Create follow-up issue for docs/00-index.md comprehensive cleanup
2. Create follow-up issue for missing directory structure restoration/removal
3. Document which directories are "pending restoration" vs "permanently removed"
4. Update documentation governance procedures

---

**Related Issues:**
- Original Issue: #[issue-number] - WP6.1: Fix Broken Documentation Links  
- Follow-up needed: Comprehensive docs/00-index.md cleanup
- Follow-up needed: Documentation structure restoration/removal decision
